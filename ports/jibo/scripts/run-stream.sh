#!/usr/bin/env bash
# run-stream.sh LABEL BINARY [MODEL]: oido_stream_replay over the fixture pack (every schedule per file; first
# schedule's logits kept) into build/runs/LABEL/. RUNNER and JOBS as in run-baseline.sh. MODEL defaults to
# models/oido_stream.tnm with its own chunk/context defaults.
set -euo pipefail
LABEL=${1:?usage: run-stream.sh LABEL BINARY [MODEL]}
BIN=${2:?}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL=${3:-$ROOT/models/oido_stream.tnm}
OUT=$ROOT/build/runs/$LABEL
JOBS=${JOBS:-1}
rm -rf "$OUT"; mkdir -p "$OUT/logits"
mapfile -t FILES < <(ls "$ROOT"/build/fixtures/*.wav | grep -v long_concat)
status=0
pids=()
for ((j = 0; j < JOBS; j++)); do
  part=()
  for ((k = j; k < ${#FILES[@]}; k += JOBS)); do part+=("${FILES[k]}"); done
  # shellcheck disable=SC2086
  ${RUNNER:-} "$BIN" --logits "$OUT/logits" "$MODEL" "${part[@]}" > "$OUT/part$j.tsv" 2> "$OUT/part$j.log" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || status=1; done
{ head -1 "$OUT/part0.tsv"; for ((j = 0; j < JOBS; j++)); do tail -n +2 "$OUT/part$j.tsv"; done | sort -t$'\t' -k1,1 -s; } > "$OUT/replay.tsv"
# compare-runs.py input: the 320-sample schedule's record per file, in oido_cli's column layout
awk -F'\t' 'BEGIN{OFS="\t"; print "file","samples","audio_s","frames","wall_ms","cpu_ms","wall_rtf","cpu_rtf","truncated","text"}
            NR>1 && $2=="320" {print $1,$3,$3/16000,$4,$13,$14,$15,"",$18,$19}' "$OUT/replay.tsv" > "$OUT/transcripts.tsv"
cat "$OUT"/part*.log > "$OUT/run.log"
rm -f "$OUT"/part*
grep -q "SCHEDULES DISAGREE" "$OUT/replay.tsv" && { echo "block schedules disagree" | tee -a "$OUT/run.log"; status=1; }
{
  echo "label: $LABEL"; echo "binary: $BIN ($(sha256sum "$BIN" | cut -d' ' -f1))"
  echo "model: $MODEL ($(sha256sum "$MODEL" | cut -d' ' -f1))"; echo "runner: ${RUNNER:-native}"; echo "jobs: $JOBS"
  echo "host: $(uname -srm); $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')"
  echo "date: $(date -u +%FT%TZ)"; echo "exit: $status"
} > "$OUT/RUN-INFO.txt"
echo "$OUT (exit $status); invariant records: $(grep -c 'same text, frames and logits' "$OUT/replay.tsv")"
exit $status
