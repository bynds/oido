#!/usr/bin/env python3
"""compare-runs.py REF_TSV REF_LOGITS_DIR TEST_TSV TEST_LOGITS_DIR: same-model, same-mode comparison of two oido_cli runs.

Reports, per file: transcript match, encoder frame match, and over the raw CTC logits the max absolute and max relative
difference, the number of frames whose argmax differs, and the first (frame, index) where the logits differ at all.
Exit status 1 if any transcript or frame count differs or a file is missing. Logit differences are reported, not
failed: across compilers/libm they are expected to be small and nonzero, and are explained, not hidden.
Standard library only (host analysis tool; nothing here runs on the robot).
"""
import array, os, sys

V1 = 1025


def read_tsv(path):
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    head = rows[0]
    return {r[0]: dict(zip(head, r + [""] * (len(head) - len(r)))) for r in rows[1:]}


def read_logits(d, name):
    p = os.path.join(d, name + ".f32")
    a = array.array("f")
    with open(p, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main(ref_tsv, ref_dir, test_tsv, test_dir):
    ref, test = read_tsv(ref_tsv), read_tsv(test_tsv)
    bad = 0
    print("file\ttext\tframes\tmax_abs\tmax_rel\targmax_diff_frames\tfirst_diff(frame,idx)")
    for name in sorted(ref):
        if name not in test:
            print(f"{name}\tMISSING"); bad += 1; continue
        r, t = ref[name], test[name]
        text_ok = r["text"] == t["text"]
        frames_ok = r["frames"] == t["frames"]
        bad += (not text_ok) + (not frames_ok)
        la, lb = read_logits(ref_dir, name), read_logits(test_dir, name)
        n = min(len(la), len(lb))
        max_abs = max_rel = 0.0
        first = "-"
        amax_diff = 0
        for fr in range(n // V1):
            a, b = la[fr * V1:(fr + 1) * V1], lb[fr * V1:(fr + 1) * V1]
            if a != b and first == "-":
                first = next(f"{fr},{i}" for i in range(V1) if a[i] != b[i])
            for x, y in zip(a, b):
                d = abs(x - y)
                if d > max_abs: max_abs = d
                if d and d / max(abs(x), 1e-6) > max_rel: max_rel = d / max(abs(x), 1e-6)
            if max(range(V1), key=a.__getitem__) != max(range(V1), key=b.__getitem__):
                amax_diff += 1
        if len(la) != len(lb):
            first += f" (logit lengths {len(la)} vs {len(lb)})"; bad += 1
        print(f"{name}\t{'same' if text_ok else 'DIFF'}\t{'same' if frames_ok else 'DIFF'}\t{max_abs:.3g}\t{max_rel:.3g}\t{amax_diff}\t{first}")
        if not text_ok:
            print(f"  ref:  {r['text']}\n  test: {t['text']}")
    print(f"{len(ref)} files, {bad} transcript/frame/length mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
