"""Score every utterance of a manifest with a large teacher model (e.g. NVIDIA stt_es_conformer_ctc_large, CC-BY-4.0)
and write the manifest back with the teacher transcript ("thyp") and its WER against the reference ("twer").
Training then drops utterances whose reference text is probably wrong (OCR errors in MLS books, misaligned clips).
On test sets the aggregate WER is the large model's reference number.

python filter_teacher.py --teacher /opt/dlami/nvme/es_large --lang es --out_dir scored a.jsonl b.jsonl
"""
import argparse, json, os, sys, time
import jiwer, sentencepiece as spm, torch
from torch.utils.data import DataLoader

from data import AudioDataset, BucketBatchSampler, collate, load_manifest, CLEANERS
from nemo_small import NemoSmall
from train_nemo_stream import BatchPre, SModel, ctc_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_sec", type=float, default=600)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    sd = torch.load(f"{a.teacher}/state_dict_plain.pt", map_location="cpu")
    m = NemoSmall(sd)
    net = SModel(m).to(dev).eval()
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"]).to(dev)
    sp = spm.SentencePieceProcessor(model_file=f"{a.teacher}/tokenizer.model")
    clean = CLEANERS[a.lang]
    print(f"teacher d={m.d} layers={len(m.layers)} vocab={m.vocab}", flush=True)
    for man in a.manifests:
        items = load_manifest([man], min_dur=0.2, max_dur=40.0, lang=a.lang)
        hyp = [""] * len(items)
        t0 = time.time()
        dl = DataLoader(AudioDataset(items), batch_sampler=BucketBatchSampler(items, a.batch_sec, seed=0),
                        collate_fn=collate, num_workers=a.workers, pin_memory=True)
        with torch.no_grad():
            for wav, lens, _, _, idx in dl:
                wav, lens = wav.to(dev).float() / 32768.0, lens.to(dev)
                f, fl = pre(wav, lens)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, ol = net(f, fl)
                lg = lg.float()
                for j, i in enumerate(idx.tolist()):
                    hyp[i] = clean(sp.decode(ctc_greedy(lg[j], int(ol[j]), m.vocab)))
        out = os.path.join(a.out_dir, os.path.basename(man))
        refs, hyps = [], []
        with open(out, "w") as f:
            for it, h in zip(items, hyp):
                w = jiwer.wer(it["text"], h) if it["text"] else 1.0
                refs.append(it["text"]); hyps.append(h)
                f.write(json.dumps({"wav": it["wav"], "text": it["text"], "duration": it["dur"], "source": it["src"],
                                    "twer": round(100 * w, 1), "thyp": h}, ensure_ascii=False) + "\n")
        hrs = sum(it["dur"] for it in items) / 3600
        print(f"{os.path.basename(man)}: {len(items)} utts {hrs:.1f} h, teacher WER {100*jiwer.wer(refs, hyps):.2f}, "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
