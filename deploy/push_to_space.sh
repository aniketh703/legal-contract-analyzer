#!/usr/bin/env bash
# Push the current master to a Hugging Face Space (Docker SDK).
#
# One-time setup (do this yourself first):
#   1. Create the Space at https://huggingface.co/new-space
#      - SDK: Docker, Visibility: Public
#   2. git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
#
# Usage:
#   ./deploy/push_to_space.sh
#
# This builds a throwaway local branch where README.md is the Space's
# frontmatter (deploy/space_readme_header.md) + the project's real README,
# so the GitHub repo's own README.md is never touched, then pushes that
# branch to the `space` remote's `main` branch and cleans up.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! git remote get-url space >/dev/null 2>&1; then
  echo "No 'space' remote configured. Run:" >&2
  echo "  git remote add space https://huggingface.co/spaces/<your-username>/<space-name>" >&2
  exit 1
fi

DEPLOY_BRANCH="deploy/space-push-tmp"
git branch -D "$DEPLOY_BRANCH" >/dev/null 2>&1 || true
git checkout -b "$DEPLOY_BRANCH" master

cat deploy/space_readme_header.md README.md > README.md.space
mv README.md.space README.md
git add README.md
git commit -m "Space config: add HF Spaces frontmatter to README" --quiet

echo "Pushing to Hugging Face Space..."
if ! git push space "$DEPLOY_BRANCH:main" --force; then
  echo "git push failed (likely LFS size limits on tracked binaries)." >&2
  echo "Falling back to: huggingface-cli upload <space-id> . --repo-type space" >&2
  echo "(pip install -U huggingface_hub first if that command isn't found)" >&2
  git checkout master
  git branch -D "$DEPLOY_BRANCH"
  exit 1
fi

git checkout master
git branch -D "$DEPLOY_BRANCH"
echo "Done. Check the Space's build log on huggingface.co for progress."
