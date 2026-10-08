#!/usr/bin/env bash
# robot-run.sh STEP: the on-Jibo baseline steps, for the owner to run. Not run by anyone so far.
#
# Same conventions as the Strands Decider port's ports/jibo/scripts/robot-run.sh (bynds/strands-decider at
# 29e78ff): every step is a deployment step, to run only within the agreed scope (which robot, which
# directory, which user). Nothing here flashes, overwrites system files, stops or restarts services, changes
# clocks or thermal settings, adds swap or commands motors. It copies into one directory, runs one process at
# a time with normal services left running, and `cleanup` removes that directory.
#
# Environment (all required; nothing is guessed):
#   JIBO_SSH    the SSH destination, e.g. jibo-skill@<robot> (prefer the jibo-skill user); "local" runs every step
#               in JIBO_DIR on this machine instead (a dry run of this script, e.g. with host builds)
#   JIBO_DIR    an isolated directory on a filesystem with room for ~50 MB (check `df` first; /tmp may be RAM)
# Optional: NEON_BUILD, SCALAR_BUILD (default build/jibo-neon, build/jibo-scalar).
#
# Steps:
#   deploy          copy build/jibo-scalar/oido_cli (as oido_cli_scalar), build/jibo-neon/{oido_cli,
#                   oido_stream_replay,oido_service,oido_feed,test_kernels,bench_kernels}, libm_fingerprint,
#                   models/{nemo8.tnm,oido_stream.tnm,nemo_lm.tlm} and build/fixtures/*.wav (about 45 MB); record hashes
#   status          thermal zones, MemAvailable, load (read-only files)
#   kernels         test_kernels (NEON vs C kernels, exact) and bench_kernels (their speed on the hot shapes);
#                   results/jibo/kernels-<time>.{txt,jsonl}
#   run [VARIANT]   libm_fingerprint (which emulated reference applies), then oido_cli over every fixture,
#                   sequentially (peak RSS from the program's own report), with status before and after;
#                   VARIANT neon (default),
#                   neon-forced-scalar (same binary, OIDO_KERNELS=scalar) or scalar (the milestone-1 binary);
#                   results/jibo/run-<VARIANT>-<time>/
#   stream          oido_stream_replay (oido_stream.tnm) over the fixtures: block-schedule invariance, chunk compute
#                   p50/p95/p99, modeled backlog and end-to-final latency on the robot; results/jibo/stream-<time>/
#   service         oido_feed --realtime | oido_service on four fixtures (stream model, then nemo8): live-paced partial
#                   and final events with endpoint_to_final_ms, peak RSS; results/jibo/service-<time>.jsonl
#   lm              oido_cli with nemo_lm.tlm (beam 4) over the fixtures: the LM's cost and transcripts on the robot
#   cleanup         remove JIBO_DIR
set -euo pipefail
STEP=${1:?usage: robot-run.sh deploy|status|kernels|run [neon|neon-forced-scalar|scalar]|stream|service|lm|cleanup}
: "${JIBO_SSH:?set JIBO_SSH}" "${JIBO_DIR:?set JIBO_DIR}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=$ROOT/results/jibo
mkdir -p "$RESULTS"
NEON_BUILD=${NEON_BUILD:-$ROOT/build/jibo-neon}
SCALAR_BUILD=${SCALAR_BUILD:-$ROOT/build/jibo-scalar}
if [ "$JIBO_SSH" = local ]; then
  remote() { bash -c "$*"; }
  scp() {  # scp -q [-r] SRC... DST with "local:" standing for this machine
    local args=() a
    for a in "$@"; do case "$a" in -q) ;; *) args+=("${a#local:}") ;; esac; done
    cp "${args[@]}"
  }
else
  remote() { ssh -o BatchMode=yes "$JIBO_SSH" "$@"; }
fi
status() {
  remote 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done;
          grep -E "MemAvailable|MemTotal" /proc/meminfo; cat /proc/loadavg; date -u +%FT%TZ'
}

case "$STEP" in
  deploy)
    remote "mkdir -p '$JIBO_DIR/fixtures' && df -k '$JIBO_DIR'"
    scp -q "$SCALAR_BUILD/oido_cli" "$JIBO_SSH:$JIBO_DIR/oido_cli_scalar"
    N=$NEON_BUILD
    scp -q "$N/oido_cli" "$N/oido_stream_replay" "$N/oido_service" "$N/oido_feed" "$N/test_kernels" "$N/bench_kernels" \
      "$SCALAR_BUILD/libm_fingerprint" "$ROOT/models/nemo8.tnm" "$ROOT/models/oido_stream.tnm" \
      "$ROOT/models/nemo_lm.tlm" "$JIBO_SSH:$JIBO_DIR/"
    scp -q "$ROOT"/build/fixtures/*.wav "$JIBO_SSH:$JIBO_DIR/fixtures/"
    remote "cd '$JIBO_DIR' || exit 1; ls -la && sha256sum oido_* test_kernels bench_kernels libm_fingerprint *.tnm *.tlm fixtures/*.wav" \
      | tee "$RESULTS/deploy-$STAMP.txt"
    ;;
  status)
    status | tee "$RESULTS/status-$STAMP.txt"
    ;;
  kernels)
    remote "cd '$JIBO_DIR' || exit 1; ./test_kernels" | tee "$RESULTS/kernels-$STAMP.txt"
    remote "cd '$JIBO_DIR' || exit 1; ./bench_kernels 20" | tee "$RESULTS/kernels-$STAMP.jsonl"
    ;;
  run)
    variant=${2:-neon}
    case "$variant" in
      neon) bin=oido_cli; envp="" ;;
      neon-forced-scalar) bin=oido_cli; envp="OIDO_KERNELS=scalar" ;;
      scalar) bin=oido_cli_scalar; envp="" ;;
      *) echo "unknown variant $variant" >&2; exit 2 ;;
    esac
    out=$RESULTS/run-$variant-$STAMP
    mkdir -p "$out"
    status > "$out/status-before.txt"
    remote "cd '$JIBO_DIR' || exit 1; ./libm_fingerprint" | tee "$out/libm-fingerprint.txt"
    # One process over all fixtures but the over-length one. Its own report (ru_maxrss) gives the peak resident set:
    # sampling /proc from the shell would need the right PID, which a backgrounded list does not give.
    remote "cd '$JIBO_DIR' || exit 1; rm -rf logits && mkdir logits || exit 1; \
            $envp ./$bin --logits logits nemo8.tnm \$(ls fixtures/*.wav | grep -v long_concat) > transcripts.tsv 2> run.log; \
            echo \"exit \$?\" >> run.log; uname -a >> run.log"
    status > "$out/status-after.txt"
    scp -q "$JIBO_SSH:$JIBO_DIR/transcripts.tsv" "$JIBO_SSH:$JIBO_DIR/run.log" "$out/"
    scp -q -r "$JIBO_SSH:$JIBO_DIR/logits" "$out/"
    # the reference is the emulated run with Jibo's glibc 2.21 (regenerate it with run-baseline.sh, see README)
    echo "compare with: ports/jibo/scripts/compare-runs.py build/runs/qemu-armv7-neon/{transcripts.tsv,logits} $out/{transcripts.tsv,logits}"
    tail -3 "$out/run.log"
    ;;
  stream)
    out=$RESULTS/stream-$STAMP
    mkdir -p "$out"
    status > "$out/status-before.txt"
    remote "cd '$JIBO_DIR' || exit 1; ./oido_stream_replay oido_stream.tnm \$(ls fixtures/*.wav | grep -v long_concat) \
            > replay.tsv 2> replay.log; echo exit \$? >> replay.log"
    status > "$out/status-after.txt"
    scp -q "$JIBO_SSH:$JIBO_DIR/replay.tsv" "$JIBO_SSH:$JIBO_DIR/replay.log" "$out/"
    grep -c "same text, frames and logits" "$out/replay.tsv" | sed 's/$/ files block-invariant/'
    ;;
  service)
    # the service's status events carry its own resident set and peak (rss_kb, rss_peak_kb)
    out=$RESULTS/service-$STAMP.jsonl
    for model in oido_stream.tnm nemo8.tnm; do
      remote "cd '$JIBO_DIR' || exit 1; ./oido_feed --realtime --status fixtures/real_22.wav fixtures/real_24.wav \
              fixtures/rooms_00.wav fixtures/edge_silence_3s.wav | ./oido_service $model" | tee -a "$out"
    done
    ;;
  lm)
    out=$RESULTS/lm-$STAMP
    mkdir -p "$out"
    remote "cd '$JIBO_DIR' || exit 1; ./oido_cli --lm nemo_lm.tlm nemo8.tnm \$(ls fixtures/*.wav | grep -v long_concat) \
            > lm.tsv 2> lm.log; echo exit \$? >> lm.log"
    scp -q "$JIBO_SSH:$JIBO_DIR/lm.tsv" "$JIBO_SSH:$JIBO_DIR/lm.log" "$out/"
    tail -2 "$out/lm.log"
    ;;
  cleanup)
    remote "rm -rf '$JIBO_DIR'" && echo "removed $JIBO_DIR on $JIBO_SSH"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
