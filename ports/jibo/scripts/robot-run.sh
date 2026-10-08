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
#   JIBO_SSH    the SSH destination, e.g. jibo-skill@<robot> (prefer the jibo-skill user)
#   JIBO_DIR    an isolated directory on a filesystem with room for ~25 MB (check `df` first; /tmp may be RAM)
#
# Steps:
#   deploy      copy build/jibo-scalar/{oido_cli,libm_fingerprint}, models/nemo8.tnm and build/fixtures/*.wav; record hashes
#   status      thermal zones, MemAvailable, load (read-only files)
#   run         libm_fingerprint (which emulated reference applies), then oido_cli over every fixture, sequentially, sampling VmHWM/VmRSS from /proc and status
#               before and after; results/jibo/run-<time>/
#   cleanup     remove JIBO_DIR
set -euo pipefail
STEP=${1:?usage: robot-run.sh deploy|status|run|cleanup}
: "${JIBO_SSH:?set JIBO_SSH}" "${JIBO_DIR:?set JIBO_DIR}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=$ROOT/results/jibo
mkdir -p "$RESULTS"
remote() { ssh -o BatchMode=yes "$JIBO_SSH" "$@"; }
status() {
  remote 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done;
          grep -E "MemAvailable|MemTotal" /proc/meminfo; cat /proc/loadavg; date -u +%FT%TZ'
}

case "$STEP" in
  deploy)
    remote "mkdir -p '$JIBO_DIR/fixtures' && df -k '$JIBO_DIR'"
    scp -q "$ROOT/build/jibo-scalar/oido_cli" "$ROOT/build/jibo-scalar/libm_fingerprint" "$ROOT/models/nemo8.tnm" "$JIBO_SSH:$JIBO_DIR/"
    scp -q "$ROOT"/build/fixtures/*.wav "$JIBO_SSH:$JIBO_DIR/fixtures/"
    remote "cd '$JIBO_DIR' && ls -la && sha256sum oido_cli libm_fingerprint nemo8.tnm fixtures/*.wav" | tee "$RESULTS/deploy-$STAMP.txt"
    ;;
  status)
    status | tee "$RESULTS/status-$STAMP.txt"
    ;;
  run)
    out=$RESULTS/run-$STAMP
    mkdir -p "$out"
    status > "$out/status-before.txt"
    remote "cd '$JIBO_DIR' && ./libm_fingerprint" | tee "$out/libm-fingerprint.txt"
    # One process over all fixtures but the over-length one; the peak resident set comes from /proc.
    remote "cd '$JIBO_DIR' && mkdir -p logits && \
            ./oido_cli --logits logits nemo8.tnm \$(ls fixtures/*.wav | grep -v long_concat) > transcripts.tsv 2> run.log & pid=\$!; \
            hwm=0; while kill -0 \$pid 2>/dev/null; do r=\$(awk '/VmHWM/{print \$2}' /proc/\$pid/status 2>/dev/null); \
            [ -n \"\$r\" ] && [ \"\$r\" -gt \"\$hwm\" ] && hwm=\$r; sleep 0.5; done; wait \$pid; rc=\$?; \
            echo \"exit \$rc; VmHWM \$hwm kB\" >> run.log; uname -a >> run.log"
    status > "$out/status-after.txt"
    scp -q "$JIBO_SSH:$JIBO_DIR/transcripts.tsv" "$JIBO_SSH:$JIBO_DIR/run.log" "$out/"
    scp -q -r "$JIBO_SSH:$JIBO_DIR/logits" "$out/"
    echo "compare with: ports/jibo/scripts/compare-runs.py build/runs/host-scalar/transcripts.tsv build/runs/host-scalar/logits $out/transcripts.tsv $out/logits"
    tail -3 "$out/run.log"
    ;;
  cleanup)
    remote "rm -rf '$JIBO_DIR'" && echo "removed $JIBO_DIR on $JIBO_SSH"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
