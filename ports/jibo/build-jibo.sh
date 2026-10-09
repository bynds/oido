#!/usr/bin/env bash
# build-jibo.sh [TARGET...]: builds of the Oído NeMo engine for the Jibo port. Targets (default: host jibo jibo-neon),
# under $BUILD_ROOT (default build/; benches and gates set their own so they never share a build directory).
# EXTRA_CFLAGS adds flags to every target (e.g. -DTASR_KERNEL_CHECKS: abort on a kernels.h contract violation):
#
#   host          host/          this machine's compiler (HOST_CC, default cc), portable C kernels
#   host-profile  host-profile/  as host, plus TASR_PROFILE stage times and TASR_KERNEL_STATS shape counts
#   jibo          jibo-scalar/   ARMv7 "plain": -mfpu=vfpv3-d16, hard float, no NEON anywhere; int8 kernels in
#                                ARMv6 SIMD32 (sxtb16/smlad, kernels_neon.c); OIDO_KERNELS=scalar gives the C ones
#   jibo-neon     jibo-neon/     ARMv7 with -mfpu=neon and the NEON kernels (ports/jibo/kernels_neon.c);
#                                OIDO_KERNELS=scalar at run time switches the same binary to the C kernels
#   jibo-profile  jibo-profile/  jibo-neon plus TASR_PROFILE and TASR_KERNEL_STATS
#   perf-plain    perf-plain/    perfvm/bench_engine for the plain variant, with TASR_PROFILE spans
#   perf-neon     perf-neon/     perfvm/bench_engine for the NEON variant, with TASR_PROFILE spans
#
# Each target builds tasr_cli (upstream CLI; not in the dispatch targets), oido_cli, oido_stream_replay, oido_service,
# oido_feed, oido_seg_ab,
# libm_fingerprint and, in the
# dispatch targets, test_kernels (NEON vs C kernels, exact) and bench_kernels (their speed on the hot shapes), and
# records
# compiler, flags and source in BUILD-INFO.txt. Jibo targets are checked by check-jibo-abi.sh (ABI-CHECK.txt).
#
# The Jibo compiler comes from the environment, as in the owner's Strands Decider port (bynds/strands-decider,
# ports/jibo/scripts/build-jibo.sh at 29e78ff), whose two routes this keeps:
#
#   JIBO_CC=<the owner's jibo-armcc wrapper>   intended: Linaro GCC 4.8.4 with Jibo's target libraries
#   JIBO_SYSROOT=<dir>                        stand-in: arm-linux-gnueabihf-gcc against a glibc <= 2.21
#                                             armhf sysroot (see manifests/toolchain.txt for the one used so far)
#
# All targets: no ESP32 SIMD (TASR_NO_SIMD; ESP_PLATFORM is never defined), no auto-vectorisation, no fast math, no
# FP contraction, so the float arithmetic is the same program everywhere. NEON is used only by the explicit int8
# kernels, whose results are exact. Nothing here runs on, or copies to, the robot.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENGINE=$ROOT/esp32/components/tinyasr
ENGINE_SRC=("$ENGINE/tinyasr.c" "$ENGINE/kernels.c" "$ENGINE/tinyasr_lm.c" "$ENGINE/tasr_nemo.c" "$ENGINE/tasr_seg.c")
FLAGS=(-O2 -std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
       -ffp-contract=off -fno-fast-math -fno-tree-vectorize -DTASR_NO_SIMD
       -I"$ENGINE/include" -I"$ENGINE" -I"$ROOT/ports/jibo" -Wall -Wno-unused-function ${EXTRA_CFLAGS:-})
ARCH_PLAIN=(-march=armv7-a -mfpu=vfpv3-d16 -mfloat-abi=hard)
ARCH_NEON=(-march=armv7-a -mfpu=neon -mfloat-abi=hard)
B=${BUILD_ROOT:-$ROOT/build}
DISPATCH=(-DTASR_KERNEL_DISPATCH)
PROFILE=(-DTASR_PROFILE -DTASR_KERNEL_STATS)

# build OUT_DIR CC "cflags..." "ldflags..." [dispatch] [bench]
build() {
  local out=$1 cc=$2 cflags=$3 ldflags=$4 dispatch=${5:-} bench=${6:-}
  local -a cf lf src=("${ENGINE_SRC[@]}")
  read -r -a cf <<<"$cflags"
  read -r -a lf <<<"$ldflags"
  [ -n "$dispatch" ] && src+=("$ROOT/ports/jibo/kernels_neon.c")
  mkdir -p "$out"
  if [ -n "$bench" ]; then   # perfvm programs only
    rm -f "$out/bench_engine" "$out/selftest"
    "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/perfvm/bench_engine.c" "$ROOT/ports/jibo/port_util.c" "${src[@]}" \
      "${lf[@]}" -lm -o "$out/bench_engine"
    "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/perfvm/selftest.c" "${lf[@]}" -o "$out/selftest"
    { echo "compiler: $cc"; "$cc" --version | head -1; echo "flags: ${FLAGS[*]} ${cf[*]}"; echo "link: ${lf[*]} -lm"
      echo "source: $(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" status --porcelain -- esp32 ports | grep -q . && echo ' (uncommitted changes)')"
    } > "$out/BUILD-INFO.txt"
    return
  fi
  rm -f "$out/tasr_cli" "$out/oido_cli" "$out/oido_stream_replay" "$out/oido_service" "$out/oido_feed" "$out/oido_seg_ab" "$out/libm_fingerprint"
  [ -z "$dispatch" ] && "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/esp32/host/tasr_cli.c" "${src[@]}" "${lf[@]}" -lm -o "$out/tasr_cli"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/oido_cli.c" "$ROOT/ports/jibo/port_util.c" "${src[@]}" "${lf[@]}" -lm \
    -o "$out/oido_cli"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/oido_stream_replay.c" "$ROOT/ports/jibo/port_util.c" "${src[@]}" \
    "${lf[@]}" -lm -o "$out/oido_stream_replay"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/oido_service.c" "$ROOT/ports/jibo/port_util.c" "$ROOT/ports/jibo/sha256.c" \
    "${src[@]}" "${lf[@]}" -lm -lpthread -o "$out/oido_service"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/oido_feed.c" "$ROOT/ports/jibo/port_util.c" "${src[@]}" "${lf[@]}" -lm \
    -o "$out/oido_feed"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/oido_seg_ab.c" "$ROOT/ports/jibo/port_util.c" "${src[@]}" "${lf[@]}" -lm \
    -o "$out/oido_seg_ab"
  "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/tests/libm_fingerprint.c" "${lf[@]}" -lm -o "$out/libm_fingerprint"
  rm -f "$out/test_kernels"
  [ -n "$dispatch" ] && "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/tests/test_kernels.c" "$ENGINE/kernels.c" \
    "$ROOT/ports/jibo/kernels_neon.c" "${lf[@]}" -lm -o "$out/test_kernels"
  rm -f "$out/bench_kernels"
  [ -n "$dispatch" ] && "$cc" "${FLAGS[@]}" "${cf[@]}" "$ROOT/ports/jibo/tests/bench_kernels.c" "$ENGINE/kernels.c" \
    "$ROOT/ports/jibo/kernels_neon.c" "${lf[@]}" -lm -o "$out/bench_kernels"
  {
    echo "compiler: $cc"
    "$cc" --version | head -1
    echo "flags: ${FLAGS[*]} ${cf[*]}"
    echo "link: ${lf[*]} -lm"
    echo "sources: ${src[*]#"$ROOT/"}"
    echo "source: $(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" status --porcelain -- esp32 ports | grep -q . && echo ' (uncommitted changes)')"
  } > "$out/BUILD-INFO.txt"
}

JIBO_CF="" JIBO_LF="" JIBO_COMPILER=""
jibo_toolchain() {
  [ -n "$JIBO_COMPILER" ] && return
  if [ -n "${JIBO_CC:-}" ]; then
    JIBO_COMPILER=$JIBO_CC
    JIBO_CF=""
  elif [ -n "${JIBO_SYSROOT:-}" ]; then
    local R M=arm-linux-gnueabihf
    R=$(cd "$JIBO_SYSROOT" && pwd)
    JIBO_COMPILER=${JIBO_STANDIN_CC:-arm-linux-gnueabihf-gcc}
    # The host's cross gcc searches its own library directories before any --sysroot, so name the sysroot's
    # headers, start files and libraries explicitly (as the Decider port does).
    JIBO_CF="-nostdinc -isystem $("$JIBO_COMPILER" -print-file-name=include) -isystem $R/usr/include/$M -isystem $R/usr/include"
    JIBO_LF="-B$R/usr/lib/$M/ -L$R/usr/lib/$M -L$R/lib/$M -Wl,--sysroot=$R -Wl,-rpath-link,$R/lib/$M -static-libgcc"
  else
    echo "set JIBO_CC (the jibo-armcc wrapper) or JIBO_SYSROOT (a glibc <= 2.21 armhf sysroot)" >&2
    exit 2
  fi
}
jibo_build() {  # OUT extra-cflags (arch first) dispatch [bench]
  jibo_toolchain
  build "$1" "$JIBO_COMPILER" "$2 $JIBO_CF" "$JIBO_LF" "$3" "${4:-}"
  [ -n "${JIBO_SYSROOT:-}" ] && echo "sysroot: $(cd "$JIBO_SYSROOT" && pwd)" >> "$1/BUILD-INFO.txt"
  if [ -n "${4:-}" ]; then "$ROOT/ports/jibo/check-jibo-abi.sh" "$1/bench_engine" "$1/selftest" | tee "$1/ABI-CHECK.txt"; return; fi
  local bins=("$1/oido_cli" "$1/oido_stream_replay" "$1/oido_service" "$1/oido_feed" "$1/oido_seg_ab" "$1/libm_fingerprint")
  [ -f "$1/tasr_cli" ] && bins+=("$1/tasr_cli")
  [ -f "$1/test_kernels" ] && bins+=("$1/test_kernels" "$1/bench_kernels")
  "$ROOT/ports/jibo/check-jibo-abi.sh" "${bins[@]}" | tee "$1/ABI-CHECK.txt"
}

[ $# -gt 0 ] || set -- host jibo jibo-neon
for target in "$@"; do
  case "$target" in
    host)         build "$B/host" "${HOST_CC:-cc}" "" "" ;;
    host-profile) build "$B/host-profile" "${HOST_CC:-cc}" "${DISPATCH[*]} ${PROFILE[*]}" "" dispatch ;;
    jibo)         jibo_build "$B/jibo-scalar" "${ARCH_PLAIN[*]} ${DISPATCH[*]} -DTASR_SIMD32" dispatch ;;
    jibo-neon)    jibo_build "$B/jibo-neon" "${ARCH_NEON[*]} ${DISPATCH[*]} -DTASR_NEON" dispatch ;;
    jibo-profile) jibo_build "$B/jibo-profile" "${ARCH_NEON[*]} ${DISPATCH[*]} -DTASR_NEON ${PROFILE[*]}" dispatch ;;
    perf-plain)   jibo_build "$B/perf-plain" "${ARCH_PLAIN[*]} ${DISPATCH[*]} -DTASR_SIMD32 -DTASR_PROFILE" dispatch bench ;;
    perf-neon)    jibo_build "$B/perf-neon" "${ARCH_NEON[*]} ${DISPATCH[*]} -DTASR_NEON -DTASR_PROFILE" dispatch bench ;;
    all)          "$0" host host-profile jibo jibo-neon jibo-profile perf-plain perf-neon ;;
    *) echo "unknown target $target" >&2; exit 2 ;;
  esac
  echo "built $target"
done
