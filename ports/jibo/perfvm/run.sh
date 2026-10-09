#!/usr/bin/env bash
# run.sh JOBS [DEST=SRC ...]: run ARMv7 programs in full-system qemu with exact instruction counting, and print
# what they print. After bynds/needle-rs ports/jibo/perfvm/run.sh (dcb8f79); the sysroot library list is this
# port's (its binaries link libgcc statically).
#
# The guest is an arm64 Ubuntu kernel (fetch-kernel.sh, pinned by SHA-256) running the armhf programs in AArch32
# EL0 with the glibc 2.21 stand-in sysroot's loader and libraries. qemu runs with -icount shift=0, so the emulated
# PMU's instructions-retired event is exact and repeatable; programs read it with perf_event_open (icount.h).
# Only instruction counts mean anything here: qemu's time does not model a Cortex-A9/A15.
#
# JOBS: one command per line, arguments separated by tabs, absolute paths inside the guest. Each DEST=SRC puts a
# file into the guest (programs under bin/, data under work/, which is the jobs' working directory).
#   JIBO_SYSROOT   the stand-in sysroot (manifests/toolchain.txt)
#   PERFVM_DIR     kernel, init and initramfs (default: $TMPDIR/perfvm); PERFVM_MEM guest MB (default 1024)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
JOBS=${1:?usage: run.sh JOBS [DEST=SRC ...]}
shift
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${PERFVM_DIR:-${TMPDIR:-/tmp}/perfvm}
mkdir -p "$W"
KERNEL=$W/vmlinuz
[ -f "$KERNEL" ] || "$HERE/fetch-kernel.sh" "$W" >/dev/null
[ -f "$W/init" ] && [ "$W/init" -nt "$HERE/init.c" ] ||
  arm-linux-gnueabihf-gcc -static -O2 -Wall -o "$W/init" "$HERE/init.c"
L=$JIBO_SYSROOT/lib/arm-linux-gnueabihf
libs=()
for f in ld-linux-armhf.so.3 libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1; do
  libs+=("lib/arm-linux-gnueabihf/$f=$L/$f")
done
python3 "$HERE/mkinitramfs.py" "$W/initramfs.gz" init="$W/init" jobs="$JOBS" work/ \
  lib/ld-linux-armhf.so.3="$L/ld-linux-armhf.so.3" "${libs[@]}" "$@"
qemu-system-aarch64 -M virt -cpu cortex-a57 -m "${PERFVM_MEM:-1024}" -nographic -no-reboot \
  -icount shift=0 -kernel "$KERNEL" -initrd "$W/initramfs.gz" \
  -append "console=ttyAMA0 rdinit=/init quiet loglevel=1" </dev/null |
  tr -d '\r' | grep -v -e '^\[' -e '^EFI' || true
