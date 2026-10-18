#!/usr/bin/env bash
# Start the Hive on a Hugging Face Space (deploy/huggingface). Space secrets:
#   MNX_TOKEN   your access token (stays the same across restarts)
#   MNX_MODELS  models to install, e.g. "mnx-1.5-max-coder=<Drive folder link>" (space-separated for more)
# A free Space forgets its files when it restarts, so the models install again in the background each start.
set -u
cd "$(dirname "$0")/.."
mkdir -p data
if [ -n "${MNX_MODELS:-}" ]; then
  # shellcheck disable=SC2086  # several id=link pairs, split on spaces
  ( python -m hive.admin install-models $MNX_MODELS > data/models-install.log 2>&1 ) &
fi
exec uvicorn gateway.api:app --host 0.0.0.0 --port "${MNX_PORT:-7860}"
