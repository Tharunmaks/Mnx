#!/usr/bin/env bash
# Makes a faster copy of your model for phones: models/mnx-q4_0.gguf.
#
# Phone CPUs (ARM) have special fast paths for the Q4_0 format, so it reads
# your messages several times faster than Q4_K_M and writes answers a bit
# faster too. Answers stay nearly the same. Mnx uses the Q4_0 copy
# automatically on Android once it exists.
#
#   bash scripts/make-fast-model.sh            (uses models/mnx*.gguf)
#   bash scripts/make-fast-model.sh path/to/model.gguf
set -euo pipefail
cd "$(dirname "$0")/.."

src="${1:-}"
if [ -z "$src" ]; then
  for f in models/mnx*.gguf; do
    case "$f" in *q4_0*) continue ;; esac
    [ -f "$f" ] && src="$f" && break
  done
fi
if [ -z "$src" ] || [ ! -f "$src" ]; then
  echo "No model found. Put mnx-q4_k_m.gguf in models/ (or pass its path) first." >&2
  exit 1
fi

bin="$(command -v llama-quantize || true)"
if [ -z "$bin" ]; then
  if command -v pkg >/dev/null; then pkg install -y llama-cpp >/dev/null && bin="$(command -v llama-quantize || true)"; fi
fi
if [ -z "$bin" ]; then
  echo "llama-quantize wasn't found. On Termux run: pkg install llama-cpp" >&2
  exit 1
fi

out="models/mnx-q4_0.gguf"
echo "==> Making $out from $src (takes a minute or two)"
"$bin" --allow-requantize "$src" "$out.part" Q4_0 >/dev/null 2>&1 || {
  rm -f "$out.part"
  echo "Converting failed. Run: $bin --allow-requantize \"$src\" $out Q4_0   to see why." >&2
  exit 1
}
mv "$out.part" "$out"
echo "Done: $out ($(du -h "$out" | cut -f1)). Restart Mnx (Ctrl+C, then npm start) to use it."
