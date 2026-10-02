"""Spanish LM text for the on-device GRU language model (all sources CC0 / CC-BY-4.0 / public domain):
  - MLS Spanish LM corpus (public-domain books; MLS excludes its dev/test books)       -> --books
  - Common Voice 17 es validated sentences (CC0), minus every dev/test sentence          -> --cv_sentences
  - training transcripts (VoxPopuli, MLS train)                                         -> --manifests
Writes cleaned lines (data.clean_text_es) to <out>_books.txt.gz and <out>_asr.txt.gz for lm.py prep.

python prep_lm_es.py --books lm/mls_lm_spanish --cv_sentences raw/transcript/es/validated_sentences.tsv \
    --exclude manifests/dev_*.jsonl manifests/test_*.jsonl --manifests manifests/train_vox.jsonl manifests/train_mls.jsonl \
    --out lm/es
"""
import argparse, csv, glob, gzip, json, os, random, sys
from data import clean_text_es

csv.field_size_limit(sys.maxsize)
ap = argparse.ArgumentParser()
ap.add_argument("--books", required=True, help="MLS LM corpus text (mls_lm_spanish/data.txt)")
ap.add_argument("--books_lines", type=int, default=4_000_000)
ap.add_argument("--cv_sentences", required=True)
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--manifests", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()

held = set()
for p in a.exclude:
    for l in open(p):
        t = clean_text_es(json.loads(l)["text"])
        if t:
            held.add(t)
print("held-out sentences excluded:", len(held), flush=True)

asr = []
with open(a.cv_sentences, newline="") as f:
    for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        t = clean_text_es(r["sentence"])
        if t and t not in held:
            asr.append(t)
n_cv = len(asr)
for p in a.manifests:
    for l in open(p):
        t = clean_text_es(json.loads(l)["text"])
        if t and t not in held:
            asr.append(t)
print(f"asr/cv lines: {n_cv} CV sentences + {len(asr) - n_cv} transcripts", flush=True)
with gzip.open(a.out + "_asr.txt.gz", "wt") as f:
    f.write("\n".join(asr) + "\n")

files = [a.books]
print("book text files:", len(files), flush=True)
rng = random.Random(0)
books = []
for p in files:
    for l in open(p, errors="ignore"):
        if rng.random() < 0.5:  # thin out evenly, then cap
            t = clean_text_es(l)
            if t and len(t.split()) >= 3 and t not in held:
                books.append(t)
rng.shuffle(books)
books = books[: a.books_lines]
with gzip.open(a.out + "_books.txt.gz", "wt") as f:
    f.write("\n".join(books) + "\n")
print("book lines:", len(books), flush=True)
