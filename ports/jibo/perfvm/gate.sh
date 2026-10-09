#!/usr/bin/env bash
# gate.sh golden|check: bit-parity gate for optimisation work. Nothing here runs on the robot.
#
#   golden  from this tree, built for x86: dump the reference trace into $GATE_DIR/golden (do this on the
#           pre-change tree, once per series of changes) and print its SHA-256 manifest
#   check   from this tree: build x86, ARM plain (VFPv3-D16) and ARM NEON into a private directory, produce the same
#           trace with each, and require every file to be byte-identical to the golden one; then run the repo's
#           suites (tests/run-tests.sh quick: sanitizers, loader, kernels incl. NEON under qemu, stream lifecycle,
#           service). Exit 1 on any difference or failure.
#
# The trace, on fixtures covering clean, noisy, reverberant, spontaneous, very short and noise-only input:
#   utt8/   nemo8 utterance mode (tasr_nemo_transcribe): transcripts, frame counts, every raw logit
#   utt4/   nemo4 (the int4 kernels): the same
#   stream/ oido_stream through the stream API (int8 K/V caches, chunked attention), 320-sample and seeded
#           irregular blocks: transcripts, frames, logits, and the schedule-invariance check
#   lm/     nemo8 with the GRU LM and beam search: transcripts
# ARM binaries run under qemu-user against Ubuntu's armhf glibc 2.39 (/usr/arm-linux-gnueabihf), whose libm matches
# the x86 host's; with Jibo's glibc 2.21 libm, logf differs by an ulp on 1.85% of inputs and so do the logits
# (README: Libm). Timing columns are dropped before comparing; everything else must match byte for byte.
#   JIBO_SYSROOT (build), GATE_DIR (default $TMPDIR/oido-gate), GATE_SKIP_SUITES=1 skips run-tests.sh
set -euo pipefail
MODE=${1:?usage: gate.sh golden|check}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
G=${GATE_DIR:-${TMPDIR:-/tmp}/oido-gate}
FX=$ROOT/build/fixtures
FILES=(real_20 real_22 real_25 rooms_06 rooms_14 edge_cut_321 edge_cut_16000 edge_noise_3s)
WAVS=(); for f in "${FILES[@]}"; do WAVS+=("$FX/$f.wav"); done
ARMLIB=/usr/arm-linux-gnueabihf

# trace BIN_DIR OUT RUNNER...: the four parts of the trace with one build
trace() {
  local bin=$1 out=$2; shift 2
  local run=("$@")
  rm -rf "$out"; mkdir -p "$out"/{utt8,utt4,stream,lm}/logits
  # columns kept: file, samples, frames, truncated, text (timings dropped)
  "${run[@]}" "$bin/oido_cli" --logits "$out/utt8/logits" "$ROOT/models/nemo8.tnm" "${WAVS[@]}" 2>/dev/null \
    | cut -f1,2,4,9,10 > "$out/utt8/transcripts.tsv"
  "${run[@]}" "$bin/oido_cli" --logits "$out/utt4/logits" "$ROOT/models/nemo4.tnm" "${WAVS[@]}" 2>/dev/null \
    | cut -f1,2,4,9,10 > "$out/utt4/transcripts.tsv"
  "${run[@]}" "$bin/oido_stream_replay" --schedules 320,irregular --logits "$out/stream/logits" "$ROOT/models/oido_stream.tnm" \
    "${WAVS[@]}" 2>/dev/null | cut -f1-5,7,18,19 > "$out/stream/replay.tsv"
  "${run[@]}" "$bin/oido_cli" --lm "$ROOT/models/nemo_lm.tlm" "$ROOT/models/nemo8.tnm" "${WAVS[@]}" 2>/dev/null \
    | cut -f1,2,4,9,10 > "$out/lm/transcripts.tsv"
  (cd "$out" && find . -type f | sort | xargs sha256sum) > "$out.sha256"
}

case "$MODE" in
  golden)
    # GOLDEN_BIN: binaries built elsewhere (e.g. from a pristine checkout of the pre-change commit) instead
    if [ -z "${GOLDEN_BIN:-}" ]; then BUILD_ROOT=$G/golden-build "$ROOT/ports/jibo/build-jibo.sh" host > /dev/null; fi
    trace "${GOLDEN_BIN:-$G/golden-build/host}" "$G/golden"
    echo "golden trace: $G/golden ($(wc -l < "$G/golden.sha256") files) from $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" status --porcelain -- esp32 ports | grep -q . && echo ' + uncommitted changes')"
    ;;
  check)
    [ -f "$G/golden.sha256" ] || { echo "no golden trace: run gate.sh golden on the pre-change tree" >&2; exit 2; }
    : "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
    C=$G/check
    rm -rf "$C"; mkdir -p "$C"
    BUILD_ROOT=$C/build "$ROOT/ports/jibo/build-jibo.sh" host jibo jibo-neon > "$C/build.log"
    echo "built (tree free to edit)"
    trace "$C/build/host" "$C/x86" &
    trace "$C/build/jibo-scalar" "$C/arm-plain" qemu-arm -L "$ARMLIB" &
    trace "$C/build/jibo-neon" "$C/arm-neon" qemu-arm -L "$ARMLIB" &
    wait
    status=0
    for v in x86 arm-plain arm-neon; do
      if cmp -s "$G/golden.sha256" "$C/$v.sha256"; then
        echo "parity $v: identical ($(wc -l < "$C/$v.sha256") files)"
      else
        echo "parity $v: DIFFERS"; diff <(cut -c67- "$G/golden.sha256" | paste - <(cut -c1-64 "$G/golden.sha256")) \
          <(cut -c67- "$C/$v.sha256" | paste - <(cut -c1-64 "$C/$v.sha256")) | head -20; status=1
      fi
    done
    grep -q "SCHEDULES DISAGREE" "$C"/*/stream/replay.tsv && { echo "stream schedules disagree"; status=1; }
    if [ -z "${GATE_SKIP_SUITES:-}" ]; then
      if BUILD_ROOT=$C/build "$ROOT/ports/jibo/tests/run-tests.sh" quick > "$C/suites.log" 2>&1; then
        echo "suites: ok ($(grep -c '^ok' "$C/suites.log") ok lines)"
      else
        echo "suites: FAILED (see $C/suites.log)"; tail -5 "$C/suites.log"; status=1
      fi
    fi
    echo "gate: $([ $status = 0 ] && echo PASS || echo FAIL)"
    exit $status
    ;;
  *) echo "usage: gate.sh golden|check" >&2; exit 2 ;;
esac
