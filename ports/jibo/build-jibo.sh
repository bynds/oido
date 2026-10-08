#!/usr/bin/env bash
# build-jibo.sh [host|jibo|all]: scalar baseline builds of the Oído NeMo engine for the Jibo port.
#
#   host   build/host/{tasr_cli,oido_cli,libm_fingerprint}         this machine's compiler (HOST_CC, default cc)
#   jibo   build/jibo-scalar/{tasr_cli,oido_cli,libm_fingerprint}  ARMv7 hard float, checked by check-jibo-abi.sh
#
# The Jibo compiler comes from the environment, as in the owner's Strands Decider port
# (bynds/strands-decider, ports/jibo/scripts/build-jibo.sh at 29e78ff), whose two routes this keeps:
#
#   JIBO_CC=<the owner's jibo-armcc wrapper>   intended: Linaro GCC 4.8.4 with Jibo's target libraries
#   JIBO_SYSROOT=<dir>                        stand-in: arm-linux-gnueabihf-gcc against a glibc <= 2.21
#                                             armhf sysroot (see README.md for the one used so far)
#
# Both builds are the same scalar program: no ESP32 SIMD (TASR_NO_SIMD, and ESP_PLATFORM is never
# defined), no auto-vectorisation, no fast math, no FP contraction. -mfpu=neon only selects the
# target; no NEON code is written yet. Nothing here runs on, or copies to, the robot.
set -euo pipefail
WHAT=${1:-all}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENGINE=$ROOT/esp32/components/tinyasr
ENGINE_SRC=("$ENGINE/tinyasr.c" "$ENGINE/kernels.c" "$ENGINE/tinyasr_lm.c" "$ENGINE/tasr_nemo.c" "$ENGINE/tasr_seg.c")
FLAGS=(-O2 -std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
       -ffp-contract=off -fno-fast-math -fno-tree-vectorize -DTASR_NO_SIMD
       -I"$ENGINE/include" -I"$ENGINE" -Wall -Wno-unused-function)
ARCH=(-march=armv7-a -mfpu=neon -mfloat-abi=hard)

# build CC OUT_DIR [extra flags...] -- [link flags...]
build() {
  local cc=$1 out=$2; shift 2
  local cflags=() ldflags=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do cflags+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  ldflags=("$@")
  mkdir -p "$out"
  rm -f "$out/tasr_cli" "$out/oido_cli" "$out/libm_fingerprint"
  "$cc" "${FLAGS[@]}" "${cflags[@]}" "$ROOT/esp32/host/tasr_cli.c" "${ENGINE_SRC[@]}" "${ldflags[@]}" -lm -o "$out/tasr_cli"
  "$cc" "${FLAGS[@]}" "${cflags[@]}" "$ROOT/ports/jibo/oido_cli.c" "${ENGINE_SRC[@]}" "${ldflags[@]}" -lm -o "$out/oido_cli"
  "$cc" "${FLAGS[@]}" "${cflags[@]}" "$ROOT/ports/jibo/tests/libm_fingerprint.c" "${ldflags[@]}" -lm -o "$out/libm_fingerprint"
  {
    echo "compiler: $cc"
    "$cc" --version | head -1
    echo "flags: ${FLAGS[*]} ${cflags[*]}"
    echo "link: ${ldflags[*]} -lm"
    echo "source: $(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" diff --quiet HEAD -- esp32 ports || echo ' (dirty)')"
  } > "$out/BUILD-INFO.txt"
}

if [ "$WHAT" = host ] || [ "$WHAT" = all ]; then
  build "${HOST_CC:-cc}" "$ROOT/build/host"
  echo "host: $ROOT/build/host"
fi

if [ "$WHAT" = jibo ] || [ "$WHAT" = all ]; then
  OUT=$ROOT/build/jibo-scalar
  if [ -n "${JIBO_CC:-}" ]; then
    build "$JIBO_CC" "$OUT" "${ARCH[@]}"
  elif [ -n "${JIBO_SYSROOT:-}" ]; then
    R=$(cd "$JIBO_SYSROOT" && pwd)
    CC=${JIBO_STANDIN_CC:-arm-linux-gnueabihf-gcc}
    M=arm-linux-gnueabihf
    # The host's cross gcc searches its own library directories before any --sysroot, so name the
    # sysroot's headers, start files and libraries explicitly (as the Decider port does).
    build "$CC" "$OUT" "${ARCH[@]}" -nostdinc -isystem "$("$CC" -print-file-name=include)" \
      -isystem "$R/usr/include/$M" -isystem "$R/usr/include" -- \
      -B"$R/usr/lib/$M/" -L"$R/usr/lib/$M" -L"$R/lib/$M" -Wl,--sysroot="$R" -Wl,-rpath-link,"$R/lib/$M" -static-libgcc
    echo "sysroot: $R" >> "$OUT/BUILD-INFO.txt"
  else
    echo "set JIBO_CC (the jibo-armcc wrapper) or JIBO_SYSROOT (a glibc <= 2.21 armhf sysroot)" >&2
    exit 2
  fi
  "$ROOT/ports/jibo/check-jibo-abi.sh" "$OUT/tasr_cli" "$OUT/oido_cli" "$OUT/libm_fingerprint" | tee "$OUT/ABI-CHECK.txt"
  echo "jibo: $OUT"
fi
