#!/usr/bin/env bash
# Sets up a rented Linux GPU machine (RunPod, Lambda, Vast.ai...) for training,
# then makes the data. Run it from the repo:
#
#   bash training/gpu_setup.sh
#
# Then: python3 training/train.py --preset smoke   (see "Train on a rented GPU" in the README)
set -euo pipefail
cd "$(dirname "$0")/.."
sudo=""; [ "$(id -u)" -ne 0 ] && sudo="sudo"

nvidia-smi --query-gpu=name,memory.total --format=csv
command -v nvcc >/dev/null || echo "Note: no nvcc, so llama.cpp builds without CUDA and the eval runs on the CPU (slow). Pick a CUDA 'devel' image to avoid this."

if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null || ! command -v curl >/dev/null; then
  $sudo apt-get update -qq && $sudo apt-get install -y -qq build-essential cmake curl git > /dev/null
fi
if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 18 ]; then
  echo "==> Installing Node.js 22"
  curl -fsSL https://deb.nodesource.com/setup_22.x | $sudo bash - > /dev/null
  $sudo apt-get install -y -qq nodejs > /dev/null
fi
echo "==> Installing Unsloth"
python3 -m pip install -q unsloth huggingface_hub

echo "==> Making the training data (a couple of minutes)"
npm install --omit=optional --silent
npm run -s train:data
echo "Ready. Next: python3 training/train.py --preset smoke"
