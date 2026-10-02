#!/usr/bin/env bash
# Update Mnx Hive to the latest code and restart it.
# Training runs, Cells and running models keep going and are picked up again after the restart.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"
git pull --ff-only
"$APP_DIR/.venv/bin/pip" install -q -r requirements.txt
sudo systemctl restart mnx-hive
sleep 2
systemctl --no-pager --lines=5 status mnx-hive
