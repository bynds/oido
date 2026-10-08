#!/usr/bin/env python3
"""score.py LABEL=TRANSCRIPTS.tsv ...: word error rate of runs against upstream's reference transcripts.

References: docs/samples/results.json ("ref" per clip; 27 clips: LibriSpeech in 20 room/noise conditions, Common
Voice, VoxPopuli, AMI). Text normalization is upstream's (eval/wer_utils.py: lower case, keep [a-z0-9'], drop
fillers); WER is word-level Levenshtein, computed here without jiwer. Also reports text produced on the negative
fixtures (digital silence, white noise), where the right output is nothing.

These are 27 published demo clips decoded from 48 kbps MP3, not a held-out robot set: a smoke test of recognition
quality and of the relative effect of a decoder change, not a WER claim. Standard library only.
"""
import json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FILLERS = {"uh", "um", "umm", "uhm", "mm", "mmm", "hmm", "hm", "ah", "er", "eh", "erm", "inaudible", "crosstalk",
           "laughter", "noise", "unintelligible"}
_num_re = re.compile(r"[^a-z0-9' ]+")


def normalize(t):
    t = t.lower().replace("’", "'")
    t = _num_re.sub(" ", t)
    return " ".join("okay" if w == "ok" else w for w in t.split() if w not in FILLERS)


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def main(args):
    refs = {e["mp3"].split("/")[-1][:-4]: e for e in json.load(open(os.path.join(ROOT, "docs/samples/results.json")))}
    print("run\tclips\twords\terrors\tWER%\tWER%_clean_rooms\tWER%_noisy_rooms\tWER%_real\tnegatives_with_text")
    for a in args:
        label, path = a.split("=", 1)
        rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
        hyp = {r[0][:-4]: (r[9] if len(r) > 9 else "") for r in rows}
        groups = {"all": [0, 0, 0], "clean": [0, 0, 0], "noisy": [0, 0, 0], "real": [0, 0, 0]}
        for k, e in refs.items():
            if k not in hyp:
                continue
            r, h = normalize(e["ref"]).split(), normalize(hyp[k]).split()
            n, err = len(r), edits(r, h)
            g = "real" if e["group"] == "real" else ("clean" if e["cond"] == "clean" else "noisy")
            for gg in ("all", g):
                groups[gg][0] += 1; groups[gg][1] += n; groups[gg][2] += err
        neg = [k for k in ("edge_silence_3s", "edge_noise_3s") if hyp.get(k)]
        w = lambda g: f"{100.0 * groups[g][2] / groups[g][1]:.1f}" if groups[g][1] else "-"
        print(f"{label}\t{groups['all'][0]}\t{groups['all'][1]}\t{groups['all'][2]}\t{w('all')}\t{w('clean')}\t"
              f"{w('noisy')}\t{w('real')}\t{','.join(neg) or '-'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
