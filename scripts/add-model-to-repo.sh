#!/usr/bin/env bash
# Adds your GGUF model to this Git repo with Git LFS and pushes it.
# Usage:  bash scripts/add-model-to-repo.sh [path/to/mnx-q4_k_m.gguf]
# Run it on a computer or phone that has the model file and push access to
# the repo. GitHub stores files over 100 MB only through Git LFS (max 2 GB
# per file on free plans), and LFS storage/bandwidth count against your quota.
set -e
cd "$(dirname "$0")/.."

src="${1:-}"
if [ -z "$src" ]; then
  for f in models/mnx*.gguf "$HOME"/storage/downloads/mnx*.gguf "$HOME"/Downloads/mnx*.gguf; do
    [ -f "$f" ] && src="$f" && break
  done
fi
[ -n "$src" ] && [ -f "$src" ] || { echo "Model file not found. Pass its path: bash scripts/add-model-to-repo.sh /path/to/mnx-q4_k_m.gguf"; exit 1; }

if ! git lfs version >/dev/null 2>&1; then
  echo "==> Installing Git LFS"
  if command -v pkg >/dev/null; then pkg install -y git-lfs
  elif command -v apt-get >/dev/null; then sudo apt-get install -y git-lfs
  elif command -v brew >/dev/null; then brew install git-lfs
  else echo "Install Git LFS from https://git-lfs.com and re-run."; exit 1; fi
fi
git lfs install --local

dest="models/$(basename "$src")"
src_real="$(readlink -f "$src")"
if [ "$src_real" != "$(readlink -f "$dest" 2>/dev/null || true)" ] || [ -L "$dest" ]; then
  echo "==> Copying model into $dest"
  rm -f "$dest"
  cp "$src_real" "$dest"
fi

echo "==> Committing $dest with Git LFS"
git add .gitattributes "$dest"
git lfs ls-files | grep -q "$(basename "$dest")" || { echo "Git LFS isn't tracking $dest; aborting."; exit 1; }
git commit -m "Add Mnx model ($(basename "$dest")) via Git LFS"

echo "==> Pushing (uploads ~$(du -h "$dest" | cut -f1); this can take a while)"
git push origin "$(git rev-parse --abbrev-ref HEAD)"
echo "Done. Anyone who clones the repo gets the model with: git lfs pull"
