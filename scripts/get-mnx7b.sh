#!/usr/bin/env bash
# Downloads YOUR Mnx 7B model from your Google Drive into models/mnx7b/,
# checks every part's md5, joins them, and finds the tokenizer.
#
#   bash scripts/get-mnx7b.sh
#
# Needs: rclone signed in to your Google account (one time: `rclone config`,
# add a "drive" remote called gdrive), python3 + numpy, and ~15 GB free storage.
# Resumable: if it stops, run it again and it continues from the last part.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${MNX_RCLONE_REMOTE:-gdrive}"
FOLDER_ID="${MNX7B_FOLDER_ID:-10v4qpNEUg4PrM71kNAMrZqC7SW8VhaLm}"   # "Mnx 7B" in your Drive
RUN="${MNX7B_RUN:-Mnx 7B (Oct 16 2026)}"
DEST="models/mnx7b"
PARTS="$DEST/.parts"
mkdir -p "$PARTS"

need() { command -v "$1" >/dev/null || { echo "Missing $1. $2" >&2; exit 1; }; }
need rclone "Install it: pkg install rclone (Termux) or https://rclone.org/install/"
need python3 "Install it: pkg install python python-numpy"
python3 -c "import numpy" 2>/dev/null || { echo "numpy is missing: pkg install python-numpy (Termux) or pip install numpy" >&2; exit 1; }
if ! rclone listremotes | grep -qx "$REMOTE:"; then
  cat >&2 <<EOF
rclone isn't connected to Google Drive yet. One-time setup:
  rclone config
    n  (new remote)   name: $REMOTE   storage: drive   (leave client id/secret empty)
    scope: 1 (full access)   then accept the defaults and sign in with tharunmakes@gmail.com
Then run this script again.
EOF
  exit 1
fi

R=(rclone --drive-root-folder-id "$FOLDER_ID")
SRC="$REMOTE:$RUN"

echo "==> Getting the part lists"
"${R[@]}" copy "$SRC/checkpoint" "$PARTS" --include "manifest_core_only.pt.txt" --include "manifest_table.bin.txt"

# One part at a time: download (~95 MB), check its md5, append, delete.
# Needs only the final size (~14.3 GB) plus one part, and resumes where it stopped.
PART_SIZE="${MNX7B_PART_SIZE:-95000000}"
join_file() {  # name  manifest  drive-folder
  local name=$1 manifest=$2 folder=$3 out="$DEST/$1"
  local total; total=$(awk '{s += $2} END {print s}' "$manifest")
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$total" ]; then echo "    $name: already done"; return; fi
  local n; n=$(wc -l < "$manifest")
  [ -f "$out.part" ] || : > "$out.part"
  local have; have=$(stat -c %s "$out.part")
  local done=$((have / PART_SIZE))
  truncate -s $((done * PART_SIZE)) "$out.part"   # drop a half-appended part
  echo "==> $name: $n parts, $done already done"
  local i=0
  while read -r md5 size part; do
    if [ "$i" -lt "$done" ]; then i=$((i + 1)); continue; fi
    printf '    %s (%d/%d)\n' "$part" $((i + 1)) "$n"
    for try in 1 2 3; do
      rm -f "$PARTS/$part"
      "${R[@]}" copyto "$SRC/checkpoint/$folder/$part" "$PARTS/$part" --retries 5 --low-level-retries 20 || true
      [ -f "$PARTS/$part" ] && [ "$(md5sum "$PARTS/$part" | cut -d' ' -f1)" = "$md5" ] && break
      echo "    md5 mismatch or download failed, retrying ($try/3)" >&2
      [ "$try" = 3 ] && { echo "Couldn't download $part. Run the script again to resume." >&2; exit 1; }
    done
    cat "$PARTS/$part" >> "$out.part"
    rm -f "$PARTS/$part"
    i=$((i + 1))
  done < "$manifest"
  [ "$(stat -c %s "$out.part")" = "$total" ] || { echo "$name has the wrong size after joining; delete $out.part and run again." >&2; exit 1; }
  mv "$out.part" "$out"
}
join_file core_only.pt "$PARTS/manifest_core_only.pt.txt" core_only.pt_parts
join_file table.bin "$PARTS/manifest_table.bin.txt" table.bin_parts

if [ ! -f "$DEST/tok16k.json" ]; then
  echo "==> Looking for the tokenizer (tok16k.json) in your dataset archive (35 MB)"
  "${R[@]}" copy "$SRC/dataset/dataset_small_files.tar.gz" "$PARTS" -P
  tar -tzf "$PARTS/dataset_small_files.tar.gz" > "$PARTS/list.txt"
  if tok=$(grep -m1 -E '(^|/)tok16k\.json$' "$PARTS/list.txt"); then
    tar -xzf "$PARTS/dataset_small_files.tar.gz" -C "$PARTS" "$tok"
    mv "$PARTS/$tok" "$DEST/tok16k.json"
  else
    cat >&2 <<EOF

The model is downloaded, but tok16k.json (its tokenizer) isn't in your Drive.
It was in /workspace/mnx/ on the machine you trained on. Copy it to:
  $DEST/tok16k.json
and Mnx will use your model.
EOF
    exit 1
  fi
fi
rm -rf "$PARTS"
echo
echo "Done. Your Mnx 7B is in $DEST ($(du -sh "$DEST" | cut -f1)). Start Mnx (npm start): it shows up as \"Mnx 7B (your model)\"."
echo "Quick test in the terminal:  python3 mnx7b/run.py -m $DEST --prompt \"Hello!\""
