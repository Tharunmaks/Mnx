#!/usr/bin/env bash
# Turns a merged 16-bit model folder into mnx-q4_k_m.gguf and mnx-q4_0.gguf
# with llama.cpp. Builds llama.cpp once (with CUDA when nvcc is there, so the
# same build runs the eval on the GPU).
#
#   bash training/export_gguf.sh <merged-model-dir> <out-dir> [q4_k_m,q4_0]
#
# LLAMA_CPP=path/to/llama.cpp uses an existing checkout or build.
set -euo pipefail
src="$1"; out="$2"; quants="${3:-q4_k_m,q4_0}"
llama="${LLAMA_CPP:-llama.cpp}"
mkdir -p "$out"

if [ ! -f "$llama/convert_hf_to_gguf.py" ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$llama"
fi
if [ ! -x "$llama/build/bin/llama-quantize" ] || [ ! -x "$llama/build/bin/llama-server" ]; then
  echo "==> Building llama.cpp (once, a few minutes)"
  cuda=OFF; command -v nvcc >/dev/null && cuda=ON
  cmake -S "$llama" -B "$llama/build" -DGGML_CUDA=$cuda -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$llama/build" --config Release -j "$(nproc)" --target llama-quantize llama-server > /dev/null
fi
# The converter needs only these; installing llama.cpp's requirements file would replace Colab's torch.
python3 -c "import sentencepiece, transformers" 2>/dev/null || python3 -m pip install -q sentencepiece transformers

f16="$out/mnx-f16.gguf"
echo "==> Converting to GGUF"
python3 "$llama/convert_hf_to_gguf.py" "$src" --outfile "$f16" --outtype f16 > "$out/convert.log" 2>&1 || {
  tail -20 "$out/convert.log"; exit 1; }
for q in ${quants//,/ }; do
  echo "==> Quantizing to ${q^^}"
  "$llama/build/bin/llama-quantize" "$f16" "$out/mnx-$q.gguf" "${q^^}" > "$out/quantize.log" 2>&1 || {
    tail -20 "$out/quantize.log"; exit 1; }
done
rm -f "$f16"
ls -lh "$out"/mnx-*.gguf
