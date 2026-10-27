"""
tests/test_llm_client.py
========================
Request shaping and config for pipeline/llm_client.py. HTTP and auth are faked; no network.
"""
import pytest

from pipeline.llm_client import (
    FakeClient, LLMError, VertexOpenAICompatClient, client_from_env, cost_usd, price_for,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.posts = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


OK = FakeResponse(200, {"model": "google/gemma-4-26b-a4b-it-maas",
                        "choices": [{"message": {"content": '{"ok": true}'}}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 5}})
MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def make(session, location="global", model="google/gemma-4-26b-a4b-it-maas"):
    return VertexOpenAICompatClient("my-proj", location, model, token_provider=lambda: "tok", session=session)


def test_request_shape_global():
    session = FakeSession([OK])
    resp = make(session).complete(MSGS, max_tokens=50)
    post = session.posts[0]
    assert post["url"] == ("https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global"
                           "/endpoints/openapi/chat/completions")
    assert post["headers"]["Authorization"] == "Bearer tok"
    body = post["json"]
    assert body["model"] == "google/gemma-4-26b-a4b-it-maas" and body["temperature"] == 0
    assert body["max_tokens"] == 50 and body["response_format"] == {"type": "json_object"}
    # Gemma: system prompt folded into the first user turn
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"].startswith("Be brief.")
    assert (resp.text, resp.input_tokens, resp.output_tokens) == ('{"ok": true}', 12, 5)


def test_regional_host_and_system_role_kept_for_non_gemma():
    session = FakeSession([OK])
    make(session, location="asia-south1", model="meta/llama-x").complete(MSGS)
    assert session.posts[0]["url"].startswith("https://asia-south1-aiplatform.googleapis.com/v1/projects/my-proj/"
                                              "locations/asia-south1/")
    assert [m["role"] for m in session.posts[0]["json"]["messages"]] == ["system", "user"]


def test_retries_without_json_mode_when_rejected():
    session = FakeSession([FakeResponse(400, text="response_format is not supported"), OK])
    make(session).complete(MSGS)
    assert "response_format" in session.posts[0]["json"]
    assert "response_format" not in session.posts[1]["json"]


def test_http_error_raises():
    with pytest.raises(LLMError, match="HTTP 403"):
        make(FakeSession([FakeResponse(403, text="denied")])).complete(MSGS)


def test_missing_project_raises():
    with pytest.raises(LLMError, match="GOOGLE_CLOUD_PROJECT"):
        VertexOpenAICompatClient("", token_provider=lambda: "t", session=FakeSession([]))


def test_prices():
    assert price_for("google/gemma-4-26b-a4b-it-maas") == (0.15, 0.60)
    assert cost_usd("gemma-4-26b-a4b-it-maas", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert price_for("some/unknown-model") is None and cost_usd("unknown", 10, 10) is None


def test_price_env_override(monkeypatch):
    monkeypatch.setenv("LCA_LLM_PRICE_IN", "1")
    monkeypatch.setenv("LCA_LLM_PRICE_OUT", "2")
    assert price_for("anything") == (1.0, 2.0)


def test_client_from_env(monkeypatch):
    monkeypatch.setenv("LCA_LLM_BACKEND", "fake")
    assert isinstance(client_from_env(), FakeClient)
    monkeypatch.setenv("LCA_LLM_BACKEND", "nope")
    with pytest.raises(LLMError, match="Unknown"):
        client_from_env()


def test_fake_scripted_runs_out():
    fake = FakeClient(responses=["a"])
    assert fake.complete(MSGS).text == "a"
    with pytest.raises(LLMError):
        fake.complete(MSGS)


def test_genai_client_maps_roles_and_usage():
    pytest.importorskip("google.genai")
    from unittest.mock import MagicMock
    from pipeline.llm_client import GenAIClient

    sdk = MagicMock()
    sdk.models.generate_content.return_value = MagicMock(
        text='{"ok": true}', usage_metadata=MagicMock(prompt_token_count=7, candidates_token_count=3))
    msgs = MSGS + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "again"}]
    resp = GenAIClient("my-proj", model="gemini-x", client=sdk).complete(msgs)
    kw = sdk.models.generate_content.call_args.kwargs
    assert kw["model"] == "gemini-x"
    assert [c.role for c in kw["contents"]] == ["user", "model", "user"]
    assert kw["config"].system_instruction == "Be brief."
    assert kw["config"].response_mime_type == "application/json"
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
