#!/usr/bin/env python3
"""score-seg-ab.py [--labels LABELS.tsv] AB.tsv [AB.tsv ...]: summarize oido_seg_ab runs, per arm and condition.

For each arm and condition: WER of the concatenated segment transcripts against the concatenated references
(upstream's normalization, as score.py: so words cut at an onset, lost to a missed utterance or added by a false
trigger all count); utterances with no overlapping segment (missed); streams segmented into a different number of
pieces than the true utterance count; false triggers (segments, and segments with text, on streams with no speech);
mean onset and end offsets of segments relative to the labelled spans (for the demo clips the labelled span is the
whole clip, including the clip's own leading/trailing silence, so these are not speech-onset measurements); AGC
samples driven to full scale. Missed utterances need the label spans (--labels, default build/ab/labels.tsv).
Standard library only.
"""
import collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
score = import_module("score")


def main(argv):
    labels_path = os.path.join(score.ROOT, "build", "ab", "labels.tsv")
    if argv[:1] == ["--labels"]:
        labels_path, argv = argv[1], argv[2:]
    spans_of = collections.defaultdict(list)
    for l in list(open(labels_path))[1:]:
        f = l.rstrip("\n").split("\t")
        if int(f[2]) >= 0:
            spans_of[f[0]].append((int(f[2]), int(f[3])))
    for path in argv:
        rows = [l.rstrip("\n").split("\t") for l in open(path)]
        head, rows = rows[0], rows[1:]
        ix = {h: i for i, h in enumerate(head)}
        agg = collections.OrderedDict()
        for r in rows:
            r += [""] * (len(head) - len(r))
            cond = r[ix["condition"]]
            group = "negative" if cond.endswith("negative") else cond
            for key in ((r[ix["arm"]], group), (r[ix["arm"]], "ALL speech" if group != "negative" else "ALL negative")):
                a = agg.setdefault(key, collections.Counter())
                exp, seg = int(r[ix["expected"]]), int(r[ix["segments"]])
                hyp = score.normalize(r[ix["hypothesis"]].replace("|", " ")).split()
                ref = score.normalize(r[ix["reference"]].replace("|", " ")).split()
                a["streams"] += 1
                a["words"] += len(ref)
                a["errors"] += score.edits(ref, hyp)
                a["wrong_count"] += seg != exp
                a["sat"] += int(r[ix["saturated"]] or 0)
                if exp == 0:
                    a["false_seg"] += seg
                    a["false_text"] += bool(hyp)
                if r[ix["onset_s"]]:
                    a["on"] += float(r[ix["onset_s"]]); a["off"] += float(r[ix["end_delay_s"]]); a["timed"] += 1
                spans = [tuple(map(int, s.split("-"))) for s in r[ix["spans"]].split(";") if s]
                a["missed"] += sum(not any(s0 < e and s1 > s for s0, s1 in spans) for s, e in spans_of[r[ix["stream"]]])
        print(f"# {path}")
        print("arm\tcondition\tstreams\tWER%\tmissed_utts\twrong_segment_count\tfalse_segments\tfalse_text"
              "\tmean_onset_s\tmean_end_offset_s\tsaturated_samples")
        for (arm, cond), a in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            wer = f"{100.0 * a['errors'] / a['words']:.1f}" if a["words"] else "-"
            t = a["timed"]
            print(f"{arm}\t{cond}\t{a['streams']}\t{wer}\t{a['missed']}\t{a['wrong_count']}\t{a['false_seg']}"
                  f"\t{a['false_text']}\t{a['on'] / t if t else 0:.2f}\t{a['off'] / t if t else 0:.2f}\t{a['sat']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
