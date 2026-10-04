#!/usr/bin/env bash
# One-time setup for running Mnx on an Android phone with Termux.
# Usage (inside the Mnx folder):  bash scripts/termux-setup.sh
set -e
cd "$(dirname "$0")/.."

echo "==> Installing Node.js and llama.cpp"
pkg update -y
pkg install -y nodejs-lts git
if ! pkg install -y llama-cpp || ! command -v llama-server >/dev/null; then
  echo "==> llama-cpp package not available; building llama-server from source (takes a while)"
  pkg install -y cmake clang make
  [ -d "$HOME/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
  cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DLLAMA_CURL=OFF
  cmake --build "$HOME/llama.cpp/build" --config Release -j "$(nproc)" --target llama-server
  mkdir -p "$PREFIX/bin"
  ln -sf "$HOME/llama.cpp/build/bin/llama-server" "$PREFIX/bin/llama-server"
fi

echo "==> Installing Mnx (skipping the desktop-only llama binding)"
npm install --omit=optional

echo "==> Looking for your model"
mkdir -p models
if ls models/*.gguf >/dev/null 2>&1; then
  echo "    Found: $(ls models/*.gguf | head -1)"
else
  [ -d "$HOME/storage" ] || termux-setup-storage || true
  sleep 2
  found=""
  for f in "$HOME"/storage/downloads/mnx*.gguf "$HOME"/storage/shared/Download/mnx*.gguf "$HOME"/storage/downloads/*.gguf; do
    [ -f "$f" ] && found="$f" && break
  done
  if [ -n "$found" ]; then
    # Link instead of copying, so the ~2 GB file isn't stored twice.
    ln -sf "$found" "models/$(basename "$found")"
    echo "    Linked $found"
  elif [ -n "$HF_TOKEN" ]; then
    npm run get-model
  else
    echo "    No model yet. Either download mnx-q4_k_m.gguf to your phone's Downloads folder and re-run this script,"
    echo "    or run:  HF_TOKEN=hf_your_token npm run get-model"
  fi
fi

echo
echo "All set. Start Mnx with:   npm start"
echo "Then open http://localhost:3000 in your phone's browser."
echo "Tip: run 'termux-wake-lock' so Android doesn't pause Mnx in the background."
