"""Train a 1024-piece SentencePiece unigram tokenizer (same settings as NVIDIA's English one) on cleaned transcripts.

python make_tok.py --lang es --out es1024 train_cv.jsonl train_mls.jsonl ...
"""
import argparse, json, random, sentencepiece as spm
from data import CLEANERS

ap = argparse.ArgumentParser()
ap.add_argument("manifests", nargs="+")
ap.add_argument("--lang", default="es")
ap.add_argument("--out", required=True)
ap.add_argument("--max_twer", type=float, default=1e9)
a = ap.parse_args()
clean = CLEANERS[a.lang]
lines = []
for p in a.manifests:
    for l in open(p):
        d = json.loads(l)
        if d.get("twer", 0) <= a.max_twer:
            t = clean(d["text"])
            if t:
                lines.append(t)
random.Random(0).shuffle(lines)
open(a.out + ".txt", "w").write("\n".join(lines[:2_000_000]) + "\n")
spm.SentencePieceTrainer.train(input=a.out + ".txt", model_prefix=a.out, vocab_size=1024, model_type="unigram",
                               character_coverage=1.0, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
                               input_sentence_size=2_000_000, shuffle_input_sentence=True, num_threads=8)
sp = spm.SentencePieceProcessor(model_file=a.out + ".model")
print("pieces", sp.get_piece_size(), [sp.id_to_piece(i) for i in range(1, 30)])
print(sp.encode("¿qué tal estás? me llamo españa y el pingüino".replace("¿", "").replace("?", ""), out_type=str))
