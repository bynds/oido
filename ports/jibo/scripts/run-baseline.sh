#!/usr/bin/env bash
# run-baseline.sh LABEL BINARY [MODEL]: run oido_cli over the fixture pack and keep transcripts, timings, logits
# and logs under build/runs/LABEL/. For an ARM binary on an x86 host set RUNNER="qemu-arm -L <sysroot>" (an
# emulator: correctness only, its timings are not a Jibo measurement). JOBS splits the fixtures over that many
# processes (default 1; the timing columns are only meaningful with 1 on an otherwise idle machine).
set -euo pipefail
LABEL=${1:?usage: run-baseline.sh LABEL BINARY [MODEL]}
BIN=${2:?}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL=${3:-$ROOT/models/nemo8.tnm}
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
{ head -1 "$OUT/part0.tsv"; for ((j = 0; j < JOBS; j++)); do tail -n +2 "$OUT/part$j.tsv"; done | sort; } > "$OUT/transcripts.tsv"
cat "$OUT"/part*.log > "$OUT/run.log"
rm -f "$OUT"/part*
# The over-length input must be refused, with a nonzero exit.
if ${RUNNER:-} "$BIN" "$MODEL" "$ROOT/build/fixtures/long_concat.wav" > /dev/null 2>> "$OUT/run.log"; then
  echo "long_concat.wav was accepted despite the 20 s bound" | tee -a "$OUT/run.log"; status=1
fi
{
  echo "label: $LABEL"; echo "binary: $BIN ($(sha256sum "$BIN" | cut -d' ' -f1))"
  echo "model: $MODEL ($(sha256sum "$MODEL" | cut -d' ' -f1))"; echo "runner: ${RUNNER:-native}"; echo "jobs: $JOBS"
  echo "host: $(uname -srm); $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')"
  echo "date: $(date -u +%FT%TZ)"; echo "exit: $status"
} > "$OUT/RUN-INFO.txt"
echo "$OUT (exit $status)"
exit $status
