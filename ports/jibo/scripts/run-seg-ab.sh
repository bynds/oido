#!/usr/bin/env bash
# run-seg-ab.sh LABEL BINARY MODEL [LABELS [STREAM_DIR]]: oido_seg_ab over every stream (all arms), JOBS processes in
# parallel over streams, into build/runs/LABEL/ab.tsv. Defaults: build/ab/labels.tsv and build/ab/streams
# (scripts/make-ab-streams.py). RUNNER as in run-baseline.sh. Then: scripts/score-seg-ab.py build/runs/LABEL/ab.tsv
set -euo pipefail
LABEL=${1:?usage: run-seg-ab.sh LABEL BINARY MODEL [LABELS [STREAM_DIR]]}
BIN=${2:?}; MODEL=${3:?}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LABELS=${4:-$ROOT/build/ab/labels.tsv}; DIR=${5:-$ROOT/build/ab/streams}
ARMS=${ARMS:-A,B,C,B25,C25}
OUT=$ROOT/build/runs/$LABEL
JOBS=${JOBS:-1}
rm -rf "$OUT"; mkdir -p "$OUT"
mapfile -t STREAMS < <(tail -n +2 "$LABELS" | cut -f1 | awk '!seen[$0]++')
pids=()
for ((j = 0; j < JOBS; j++)); do
  { head -1 "$LABELS"; for ((k = j; k < ${#STREAMS[@]}; k += JOBS)); do
      awk -F'\t' -v s="${STREAMS[k]}" 'NR>1 && $1==s' "$LABELS"; done; } > "$OUT/labels$j.tsv"
  # shellcheck disable=SC2086
  ${RUNNER:-} "$BIN" --arms "$ARMS" "$MODEL" "$OUT/labels$j.tsv" "$DIR" > "$OUT/part$j.tsv" 2> "$OUT/part$j.log" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
{ head -1 "$OUT/part0.tsv"; for ((j = 0; j < JOBS; j++)); do tail -n +2 "$OUT/part$j.tsv"; done; } > "$OUT/ab.tsv"
cat "$OUT"/part*.log > "$OUT/run.log"; rm -f "$OUT"/part* "$OUT"/labels*.tsv
{ echo "label: $LABEL"; echo "binary: $BIN ($(sha256sum "$BIN" | cut -d' ' -f1))"
  echo "model: $MODEL ($(sha256sum "$MODEL" | cut -d' ' -f1))"; echo "labels: $LABELS ($(sha256sum "$LABELS" | cut -d' ' -f1))"
  echo "arms: $ARMS"; echo "runner: ${RUNNER:-native}"; echo "date: $(date -u +%FT%TZ)"; echo "exit: $status"; } > "$OUT/RUN-INFO.txt"
echo "$OUT/ab.tsv (exit $status)"
exit $status
