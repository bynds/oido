#!/usr/bin/env bash
# model-manifest.sh: sizes, SHA-256 and TNM1 header fields of the English models this port uses, written to
# manifests/models.tsv. Fails on a missing file, a Git LFS pointer or a wrong magic number, so a download error
# cannot pass for a model. The header layout is the one tasr_nemo_load() reads (esp32/components/tinyasr/tasr_nemo.c):
# "TNM1" then 15 little-endian uint32 values: [1]=d [2]=heads [3]=ff [4]=conv kernel [5]=layers
# [6]=subsampling channels [7]=vocabulary [8]=weight bits [10]=flags [11]=stream chunk [12]=stream left.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
mkdir -p "$ROOT/manifests"
OUT=$ROOT/manifests/models.tsv
printf 'file\tbytes\tsha256\tmagic\td\theads\tff\tconv_k\tlayers\tsub_ch\tvocab\tbits\tflags\tstream_chunk\tstream_left\n' > "$OUT"
for f in models/nemo8.tnm models/oido_stream.tnm models/nemo_lm.tlm; do
  p=$ROOT/$f
  [ -f "$p" ] || { echo "missing $f" >&2; exit 1; }
  head -c 64 "$p" | grep -q "git-lfs" && { echo "$f is a Git LFS pointer" >&2; exit 1; }
  magic=$(head -c 4 "$p")
  size=$(stat -c %s "$p")
  sum=$(sha256sum "$p" | cut -d' ' -f1)
  if [ "$magic" = TNM1 ]; then
    read -r -a h <<<"$(od -A n -t u4 -j 4 -N 60 --endian=little "$p" | tr -s ' \n' ' ')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$f" "$size" "$sum" "$magic" \
      "${h[1]}" "${h[2]}" "${h[3]}" "${h[4]}" "${h[5]}" "${h[6]}" "${h[7]}" "${h[8]}" "${h[10]}" "${h[11]}" "${h[12]}" >> "$OUT"
  else
    case "$f" in *.tnm) echo "$f: magic '$magic' is not TNM1" >&2; exit 1 ;; esac
    printf '%s\t%s\t%s\t%s\n' "$f" "$size" "$sum" "$magic" >> "$OUT"
  fi
done
cat "$OUT"
