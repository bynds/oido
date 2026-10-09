#!/usr/bin/env bash
# run-tests.sh [quick]: build and run the port's host tests under AddressSanitizer and UBSan.
#   test_model_load   loader and stream constructor against truncation, header mutation, misalignment and
#                     allocation failure, on every shipped .tnm (several minutes; `quick` runs nemo8 only)
#   test_kernels      dispatch layer vs C kernels (on this host both are C: checks the test and the layer)
#   test_stream_lifecycle  stream reset/reuse, leakage, finish/feed-after-finish, short inputs, chunk boundaries,
#                     truncation flag, frame counts, interleaved streams (needs build/fixtures: make-fixtures.sh)
#   test_service.py   oido_service end to end against build/host (needs build-jibo.sh host and the fixtures)
#   ARM, if build/jibo-neon/test_kernels exists, qemu-arm is installed and JIBO_SYSROOT is set: the NEON kernels
#                     against the C kernels, exactly, under emulation
# Nothing here needs the robot.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
E=$ROOT/esp32/components/tinyasr
B=${BUILD_ROOT:-$ROOT/build}   # binaries from build-jibo.sh (fixtures always come from build/fixtures)
OUT=$B/tests
mkdir -p "$OUT"
CC=${HOST_CC:-cc}
SAN=(-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined)
FLAGS=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -ffp-contract=off -DTASR_NO_SIMD -Wall -Wno-unused-function
       -I"$E/include" -I"$E")
ENGINE=("$E/tinyasr.c" "$E/kernels.c" "$E/tinyasr_lm.c" "$E/tasr_nemo.c" "$E/tasr_seg.c")

"$CC" "${SAN[@]}" "${FLAGS[@]}" "$ROOT/ports/jibo/tests/test_model_load.c" "${ENGINE[@]}" -lm -o "$OUT/test_model_load"
if [ "${1:-}" = quick ]; then MODELS=("$ROOT/models/nemo8.tnm"); else MODELS=("$ROOT"/models/*.tnm); fi
"$OUT/test_model_load" "${MODELS[@]}"

"$CC" "${SAN[@]}" "${FLAGS[@]}" -DTASR_KERNEL_DISPATCH -I"$ROOT/ports/jibo" "$ROOT/ports/jibo/tests/test_kernels.c" \
  "$E/kernels.c" "$ROOT/ports/jibo/kernels_neon.c" -lm -o "$OUT/test_kernels"
"$OUT/test_kernels" | tail -2

if [ -f "$ROOT/build/fixtures/real_23.wav" ]; then
  "$CC" "${SAN[@]}" "${FLAGS[@]}" -I"$ROOT/ports/jibo" "$ROOT/ports/jibo/tests/test_stream_lifecycle.c" \
    "$ROOT/ports/jibo/port_util.c" "${ENGINE[@]}" -lm -o "$OUT/test_stream_lifecycle"
  ASAN_OPTIONS=detect_leaks=0 "$OUT/test_stream_lifecycle" "$ROOT/models/oido_stream.tnm" "$ROOT/build/fixtures/real_23.wav" | tail -1
  if [ -x "$B/host/oido_service" ]; then
    python3 "$ROOT/ports/jibo/tests/test_service.py" "$B/host" | tail -1
  else
    echo "test_service.py skipped (needs build-jibo.sh host)"
  fi
else
  echo "stream and service tests skipped (needs scripts/make-fixtures.sh)"
fi

if [ -x "$B/jibo-neon/test_kernels" ] && command -v qemu-arm >/dev/null && [ -n "${JIBO_SYSROOT:-}" ]; then
  echo "ARM (qemu-arm, $JIBO_SYSROOT):"
  qemu-arm -L "$JIBO_SYSROOT" "$B/jibo-neon/test_kernels" | tail -3
else
  echo "ARM kernel test skipped (needs build/jibo-neon, qemu-arm and JIBO_SYSROOT)"
fi
