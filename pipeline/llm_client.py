"""
pipeline/llm_client.py
======================
Thin LLM clients for the v2 generator (pipeline/llm_generator.py), all with one method:

    client.complete(messages, json_mode=True, max_tokens=1024) -> LLMResponse

Backends (pick with LCA_LLM_BACKEND):
    vertex_openai  (default) Vertex AI's OpenAI-compatible chat endpoint. Serves the managed
                   Gemma 4 model (google/gemma-4-26b-a4b-it-maas, listed in the `global` location).
    genai          google-genai SDK against Vertex AI, for a Gemini comparison arm.
    fake           offline, deterministic; quotes the first retrieved section. For dry runs/tests.

Configuration (environment):
    GOOGLE_CLOUD_PROJECT   GCP project id (required for vertex_openai / genai)
    LCA_LLM_LOCATION       Vertex location, default "global"
    LCA_LLM_MODEL          model id, default per backend (DEFAULT_MODELS)
    LCA_LLM_PRICE_IN/OUT   USD per 1M input/output tokens, overrides PRICES_PER_M
    LCA_LLM_TIMEOUT        request timeout in seconds, default 60

Auth is Application Default Credentials on the machine running this; no keys live in the repo:
    gcloud auth application-default login

Check the endpoint and model id with one tiny call before a full run:
    python -m pipeline.llm_client --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MODELS = {
    "vertex_openai": "google/gemma-4-26b-a4b-it-maas",
    "genai": "gemini-3.5-flash",
    "fake": "fake-echo",
}

# USD per 1M tokens (input, output), from the Vertex AI pricing page (checked Oct 2026).
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gemma-4-26b-a4b-it-maas": (0.15, 0.60),
    "gemini-3.5-flash": (1.50, 9.00),
    "fake-echo": (0.0, 0.0),
}

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)


def price_for(model: str) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for a model id, env override first. None if unknown."""
    env_in, env_out = os.environ.get("LCA_LLM_PRICE_IN"), os.environ.get("LCA_LLM_PRICE_OUT")
    if env_in is not None and env_out is not None:
        return float(env_in), float(env_out)
    short = model.split("/")[-1]
    return PRICES_PER_M.get(short)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _fold_system(messages: list[dict]) -> list[dict]:
    """Merge system messages into the first user turn (for models without a system role)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    return rest


def _adc_token() -> str:
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=_SCOPES)
    creds.refresh(Request())
    return creds.token


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class VertexOpenAICompatClient:
    """Vertex AI OpenAI-compatible chat completions (managed open models such as Gemma 4)."""

    backend = "vertex_openai"

    def __init__(self, project: str, location: str = "global", model: str = DEFAULT_MODELS["vertex_openai"],
                 timeout: float = 60.0, token_provider: Callable[[], str] | None = None, session=None):
        if not project:
            raise LLMError("GOOGLE_CLOUD_PROJECT is not set")
        self.project, self.location, self.model, self.timeout = project, location, model, timeout
        self._token = token_provider or _adc_token
        if session is None:
            import requests
            session = requests.Session()
        self._session = session

    @property
    def url(self) -> str:
        host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{self.project}/locations/{self.location}"
                f"/endpoints/openapi/chat/completions")

    def complete(self, messages: list[dict], json_mode: bool = True, max_tokens: int = 1024) -> LLMResponse:
        # Timeouts, connection errors, auth refresh and SDK errors all become LLMError, so the
        # generator records a refusal for this clause instead of aborting a whole scoring run.
        try:
            return self._complete(messages, json_mode, max_tokens)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.backend} request failed: {type(exc).__name__}: {exc}") from exc

    def _complete(self, messages: list[dict], json_mode: bool, max_tokens: int) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": _fold_system(messages) if "gemma" in self.model.lower() else messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}
        resp = self._session.post(self.url, json=body, headers=headers, timeout=self.timeout)
        if resp.status_code == 400 and json_mode and "response_format" in resp.text:
            # Some served models reject JSON mode; the prompt still demands JSON and the
            # generator validates it, so retry once without it.
            body = {k: v for k, v in body.items() if k != "response_format"}
            resp = self._session.post(self.url, json=body, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise LLMError(f"Vertex returned HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:500]}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(text=text, input_tokens=int(usage.get("prompt_tokens", 0)),
                           output_tokens=int(usage.get("completion_tokens", 0)),
                           model=data.get("model", self.model), raw=data)


class GenAIClient:
    """google-genai SDK on Vertex AI (Gemini models)."""

    backend = "genai"

    def __init__(self, project: str, location: str = "global", model: str = DEFAULT_MODELS["genai"], client=None):
        if client is None:
            if not project:
                raise LLMError("GOOGLE_CLOUD_PROJECT is not set")
            from google import genai
            client = genai.Client(vertexai=True, project=project, location=location)
        self.model, self._client = model, client

    def complete(self, messages: list[dict], json_mode: bool = True, max_tokens: int = 1024) -> LLMResponse:
        try:
            return self._complete(messages, json_mode, max_tokens)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.backend} request failed: {type(exc).__name__}: {exc}") from exc

    def _complete(self, messages: list[dict], json_mode: bool, max_tokens: int) -> LLMResponse:
        from google.genai import types

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0, max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=resp.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            model=self.model,
        )


_SECTION_BLOCK_RE = re.compile(r"^\[([A-Z]+_S\w+)\][^\n]*\n(.+?)(?=\n\[[A-Z]+_S\w+\]|\Z)", re.MULTILINE | re.DOTALL)


class FakeClient:
    """
    Offline client. With `responses`, returns them in order (tests). Without, answers every
    prompt by quoting the first retrieved section, so a full scoring run works with no creds.
    """

    backend = "fake"

    def __init__(self, responses: list[str] | None = None, model: str = DEFAULT_MODELS["fake"]):
        self.model = model
        self._responses = list(responses) if responses is not None else None
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], json_mode: bool = True, max_tokens: int = 1024) -> LLMResponse:
        self.calls.append(messages)
        if self._responses is not None:
            if not self._responses:
                raise LLMError("FakeClient ran out of scripted responses")
            text = self._responses.pop(0)
        else:
            text = self._echo(messages[-1]["content"])
        n_in = sum(len(m["content"].split()) for m in messages)
        return LLMResponse(text=text, input_tokens=n_in, output_tokens=len(text.split()), model=self.model)

    @staticmethod
    def _echo(prompt: str) -> str:
        m = _SECTION_BLOCK_RE.search(prompt)
        if not m:
            return json.dumps({"status": "refused", "reason": "no sections provided"})
        sid, body = m.group(1), m.group(2)
        # Section text is "Section N. Title. body"; quote the start of the body.
        words = re.sub(r"^Section \S+\. .*?\. ", "", " ".join(body.split())).split()[:12]
        return json.dumps({
            "status": "answered",
            "risk_level": "medium",
            "enforceability": "unclear",
            "risks": [{"statement": f"This clause engages {sid}.", "citations": [sid],
                       "evidence_quote": " ".join(words)}],
            "recommended_action": "Have counsel review this clause against the cited section.",
        })


def client_from_env(backend: str | None = None, model: str | None = None):
    backend = backend or os.environ.get("LCA_LLM_BACKEND", "vertex_openai")
    model = model or os.environ.get("LCA_LLM_MODEL") or DEFAULT_MODELS.get(backend)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("LCA_LLM_LOCATION", "global")
    if backend == "vertex_openai":
        return VertexOpenAICompatClient(project, location, model, timeout=float(os.environ.get("LCA_LLM_TIMEOUT", 60)))
    if backend == "genai":
        return GenAIClient(project, location, model)
    if backend == "fake":
        return FakeClient(model=model)
    raise LLMError(f"Unknown LCA_LLM_BACKEND {backend!r} (use vertex_openai, genai or fake)")


def _smoke(backend: str | None, model: str | None) -> int:
    try:
        client = client_from_env(backend, model)
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    messages = [
        {"role": "system", "content": "Reply with a single JSON object and nothing else."},
        {"role": "user", "content": 'Return {"ok": true}.'},
    ]
    t0 = time.perf_counter()
    try:
        resp = client.complete(messages, max_tokens=20)
    except Exception as exc:  # report, don't trace back: this is a config check
        print(f"FAILED ({client.backend}, {client.model}): {exc}", file=sys.stderr)
        return 1
    ms = (time.perf_counter() - t0) * 1000
    cost = cost_usd(client.model, resp.input_tokens, resp.output_tokens)
    print(f"backend : {client.backend}\nmodel   : {resp.model}\nreply   : {resp.text.strip()[:200]}")
    print(f"tokens  : {resp.input_tokens} in / {resp.output_tokens} out\nlatency : {ms:.0f} ms")
    print(f"cost    : {'unknown (set LCA_LLM_PRICE_IN/OUT)' if cost is None else f'${cost:.6f}'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM client config check")
    ap.add_argument("--smoke", action="store_true", help="send one tiny request and print the result")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    if not a.smoke:
        ap.print_help()
        sys.exit(0)
    sys.exit(_smoke(a.backend, a.model))
