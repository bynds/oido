#!/usr/bin/env python3
"""mkinitramfs.py OUT.cpio.gz ENTRY...: a newc initramfs for perfvm, without root or cpio(1).

Copied from bynds/needle-rs (branch claude/port-creation-y2ehfd, commit dcb8f79), ports/jibo/perfvm.

ENTRY is DEST=SRC (a file; mode kept), DEST/ (a directory) or DEST=@dev:c:MAJOR:MINOR. Symlinks
in SRC are followed. /dev/console is always added, so /init has a console."""
import gzip, os, sys

def header(name, mode, size, ino, rdev=(0, 0)):
    f = [ino, mode, 0, 0, 1, 0, size, 0, 0, rdev[0], rdev[1], len(name) + 1, 0]
    h = b'070701' + b''.join(b'%08x' % v for v in f) + name.encode() + b'\0'
    return h + b'\0' * ((4 - len(h) % 4) % 4)

def main():
    out, entries = sys.argv[1], sys.argv[2:] + ['dev/', 'dev/console=@dev:c:5:1']
    ino, buf, dirs = 1, [], set()
    def add(name, mode, data=b'', rdev=(0, 0)):
        nonlocal ino
        buf.append(header(name, mode, len(data), ino, rdev) + data + b'\0' * ((4 - len(data) % 4) % 4))
        ino += 1
    def mkdirs(path):
        parts = path.strip('/').split('/')
        for i in range(1, len(parts) + 1):
            d = '/'.join(parts[:i])
            if d and d not in dirs:
                dirs.add(d)
                add(d, 0o040755)
    for e in entries:
        if e.endswith('/') and '=' not in e:
            mkdirs(e)
            continue
        dest, src = e.split('=', 1)
        dest = dest.strip('/')
        mkdirs(os.path.dirname(dest))
        if src.startswith('@dev:'):
            _, kind, ma, mi = src.split(':')
            add(dest, (0o020000 if kind == 'c' else 0o060000) | 0o600, rdev=(int(ma), int(mi)))
            continue
        data = open(src, 'rb').read()
        add(dest, 0o100000 | (os.stat(src).st_mode & 0o777), data)
    add('TRAILER!!!', 0)
    with gzip.open(out, 'wb', compresslevel=1) as f:
        f.write(b''.join(buf))

main()
