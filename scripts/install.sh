#!/usr/bin/env bash
# One-time setup of Mnx Hive on a fresh Ubuntu server (Oracle Cloud free ARM works well).
# Run from the repo folder:  bash scripts/install.sh
# Installs Docker, a Python venv with the gateway's packages, Chromium for the Browser Bee,
# and a systemd service that keeps the Hive running 24/7 and after reboots.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"
echo "==> Installing Mnx Hive from $APP_DIR for user $USER_NAME"

sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip git curl tar openssh-client

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker"
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER_NAME"

if command -v nvidia-smi >/dev/null 2>&1 && ! docker info 2>/dev/null | grep -q nvidia; then
  echo "==> NVIDIA GPU found. To let training containers use it, install the NVIDIA Container Toolkit:"
  echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

echo "==> Python packages"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> Chromium for the Browser Bee"
sudo "$APP_DIR/.venv/bin/python" -m playwright install-deps chromium
"$APP_DIR/.venv/bin/python" -m playwright install chromium

echo "==> Base image for Cells and training (first download takes a minute)"
sudo docker pull -q python:3.12-slim >/dev/null

echo "==> systemd service"
sed -e "s|^User=.*|User=$USER_NAME|" -e "s|/home/ubuntu/Mnx|$APP_DIR|g" "$APP_DIR/services/mnx-hive.service" \
  | sudo tee /etc/systemd/system/mnx-hive.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now mnx-hive
sleep 3

TOKEN="$(cat "$APP_DIR/data/token.txt" 2>/dev/null || true)"
IP="$(curl -s -m 5 https://api.ipify.org || hostname -I | awk '{print $1}')"
cat <<MSG

Mnx Hive is running.
  Open:          http://$IP:8000
  Access token:  ${TOKEN:-see: journalctl -u mnx-hive | grep token}
  Logs:          journalctl -u mnx-hive -f
  Update later:  bash scripts/update.sh

If the page doesn't open, allow TCP port 8000:
  - Oracle Cloud: add an ingress rule for 8000 in the subnet's security list, then on the server:
      sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT && sudo netfilter-persistent save
  - Better for the internet: put it behind HTTPS (e.g. Caddy or a Cloudflare Tunnel).
MSG
