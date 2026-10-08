#!/usr/bin/env bash
# run-tests.sh [quick]: build and run the port's host tests under AddressSanitizer and UBSan.
#   test_model_load   loader and stream constructor against truncation, header mutation, misalignment and
#                     allocation failure, on every shipped .tnm (several minutes; `quick` runs nemo8 only)
# Nothing here needs the robot. ARM-specific tests run under qemu from their own scripts.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
E=$ROOT/esp32/components/tinyasr
OUT=$ROOT/build/tests
mkdir -p "$OUT"
CC=${HOST_CC:-cc}
SAN=(-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined)
FLAGS=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -ffp-contract=off -DTASR_NO_SIMD -Wall -Wno-unused-function
       -I"$E/include" -I"$E")
ENGINE=("$E/tinyasr.c" "$E/kernels.c" "$E/tinyasr_lm.c" "$E/tasr_nemo.c" "$E/tasr_seg.c")

"$CC" "${SAN[@]}" "${FLAGS[@]}" "$ROOT/ports/jibo/tests/test_model_load.c" "${ENGINE[@]}" -lm -o "$OUT/test_model_load"
if [ "${1:-}" = quick ]; then MODELS=("$ROOT/models/nemo8.tnm"); else MODELS=("$ROOT"/models/*.tnm); fi
"$OUT/test_model_load" "${MODELS[@]}"
