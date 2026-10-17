#!/data/data/com.termux/files/usr/bin/bash
# Start the Mnx Hive on a phone (Termux) and keep it running: pulls updates, installs what's missing, keeps the
# phone awake, and restarts the server if it stops. Run:   bash scripts/termux-start.sh
cd "$(dirname "$0")/.." || exit 1
termux-wake-lock 2>/dev/null || true
git pull --ff-only -q 2>/dev/null && echo "Updated to the latest Hive." || true
python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null || pip install fastapi "uvicorn[standard]" httpx
python3 -c "import websockets" 2>/dev/null || pip install websockets   # live terminals and the phone drone
python3 -c "import gdown" 2>/dev/null || pip install gdown
command -v llama-server >/dev/null || pkg install -y llama-cpp || true
command -v node >/dev/null || pkg install -y nodejs || true                      # Chroma: the sites Bees build
command -v clang >/dev/null && command -v make >/dev/null || pkg install -y clang make || true
[ -x data/expertstream/es_serve ] || python3 -m hive.admin install-expertstream || true  # Expert Stream: the faster engine
mkdir -p data
echo "The Hive is starting. Open http://localhost:8000 in your browser; the access token is printed below."
while true; do
  python3 -m uvicorn gateway.api:app --host 0.0.0.0 --port 8000 2>&1 | tee -a data/server.log
  echo "The Hive stopped. Restarting in 3 seconds (press Ctrl+C twice to quit)…"
  sleep 3
done
