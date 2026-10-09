#!/usr/bin/env bash
# bench.sh LABEL [CLIP]: build bench_engine for both ARMv7 variants (plain VFPv3-D16, as shipped, and NEON) from this
# tree into a private build directory, run them in perfvm, and write results/LABEL.jsonl: one line per variant and
# workload (utterance, stream_setup, stream) with exact user-space instruction counts, in total and per engine stage
# ([instructions, calls]), plus "other" (glue between stages). After bynds/needle-rs perfvm/bench.sh (dcb8f79).
#   CLIP          default build/fixtures/real_22.wav (5.76 s of speech; scripts/make-fixtures.sh)
#   JIBO_SYSROOT  the stand-in sysroot
# The build happens first and the VM runs copies, so the tree may be edited once "built" is printed.
set -euo pipefail
LABEL=${1:?usage: bench.sh LABEL [CLIP]}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CLIP=${2:-$ROOT/build/fixtures/real_22.wav}
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${PERFVM_DIR:-${TMPDIR:-/tmp}/perfvm}
B=$W/bench-$LABEL
rm -rf "$B"; mkdir -p "$B"
BUILD_ROOT=$B/build "$ROOT/ports/jibo/build-jibo.sh" perf-plain perf-neon > "$B/build.log"
for v in plain neon; do cp "$B/build/perf-$v/bench_engine" "$B/bench-$v"; done
cp "$B/build/perf-plain/selftest" "$B/selftest"
cp "$B/build/perf-plain/BUILD-INFO.txt" "$B/BUILD-INFO-plain.txt"; cp "$B/build/perf-neon/BUILD-INFO.txt" "$B/BUILD-INFO-neon.txt"
echo "built (tree free to edit)"
J=$B/jobs
{ printf '/bin/selftest\n'
  for v in plain neon; do printf '/bin/bench-%s\t/work/nemo8.tnm\t/work/oido_stream.tnm\t/work/clip.wav\n' "$v"; done; } > "$J"
PERFVM_DIR=$W "$HERE/run.sh" "$J" bin/selftest="$B/selftest" bin/bench-plain="$B/bench-plain" bin/bench-neon="$B/bench-neon" \
  work/nemo8.tnm="$ROOT/models/nemo8.tnm" work/oido_stream.tnm="$ROOT/models/oido_stream.tnm" work/clip.wav="$CLIP" \
  > "$B/console.txt"
mkdir -p "$HERE/results"
python3 - "$B/console.txt" "$HERE/results/$LABEL.jsonl" "$LABEL" "$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" status --porcelain -- esp32 ports | grep -q . && echo +)" "$(basename "$CLIP")" <<'PY'
import json, sys
variant, rows, selftest = None, [], None
for line in open(sys.argv[1]):
    if line.startswith('perfvm: job /bin/'):
        variant = line.split('/bin/')[1].split()[0].replace('bench-', '')
    elif line.startswith('selftest:'):
        selftest = line.strip()
    elif line.startswith('{'):
        v = json.loads(line)
        v.update(variant=variant, label=sys.argv[3], source=sys.argv[4], clip=sys.argv[5])
        rows.append(v)
if not selftest or '6000056' not in selftest:
    sys.exit(f'counter selftest unexpected: {selftest}')
if len(rows) != 6 or any(r['clock'] != 'instructions' for r in rows):
    sys.exit(f'expected 6 instruction-count results, got {len(rows)}: see console.txt')
open(sys.argv[2], 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(f'{len(rows)} results -> {sys.argv[2]} ({selftest})')
for r in rows:
    print(f"{r['variant']:6} {r['workload']:13} {r['total']/1e9:8.4f} G instructions  {r['text']}")
PY
