#!/usr/bin/env bash
# fetch-kernel.sh DIR: the pinned arm64 Ubuntu kernel perfvm boots (its PMU driver gives guests
# Copied from bynds/needle-rs (branch claude/port-creation-y2ehfd, commit dcb8f79), ports/jibo/perfvm.
# perf_event_open; qemu's -icount makes the instruction count exact). Writes DIR/vmlinuz.
set -euo pipefail
DIR=${1:?usage: fetch-kernel.sh DIR}
DEB=linux-image-unsigned-6.8.0-146-generic_6.8.0-146.146_arm64.deb
SHA=d5ef7b0684f397b12621f250be6d3e5b10d2cbcee2656b58c35b8a0d2b50a7b7
mkdir -p "$DIR"
cd "$DIR"
[ -f "$DEB" ] || curl -fsSO "http://ports.ubuntu.com/ubuntu-ports/pool/main/l/linux/$DEB"
echo "$SHA  $DEB" | sha256sum -c - >/dev/null
rm -rf x && dpkg-deb -x "$DEB" x
cp x/boot/vmlinuz-6.8.0-146-generic vmlinuz
rm -rf x
echo "$DIR/vmlinuz"
