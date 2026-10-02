"""Manifest dataset + duration-bucketed batch sampler. Audio returned as int16 numpy; features computed on GPU."""
import json, random, re
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

OK_TEXT = re.compile(r"^[a-z' ]+$")
_ES_MAP = str.maketrans({"à": "a", "â": "a", "ä": "a", "ã": "a", "è": "e", "ê": "e", "ë": "e", "ì": "i", "î": "i",
                         "ï": "i", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ù": "u", "û": "u", "ç": "c",
                         "’": " ", "'": " ", "`": " ", "-": " "})


def clean_text(t):
    t = t.lower().replace("’", "'").replace("`", "'")
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_text_es(t):
    """Spanish: lowercase, keep a-z and á é í ó ú ü ñ, drop punctuation. Text with digits is unusable for training a
    spelled-out recognizer, so it returns "" (filtered)."""
    import unicodedata
    t = unicodedata.normalize("NFC", t.lower()).translate(_ES_MAP)
    if re.search(r"[0-9]", t):
        return ""
    t = re.sub(r"[^a-záéíóúüñ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


CLEANERS = {"en": clean_text, "es": clean_text_es}


def load_manifest(paths, min_dur=0.3, max_dur=20.0, sp=None, max_per_source=None, lang="en"):
    items = []
    per_src = {}
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            dur = d.get("duration", 0)
            if not (min_dur <= dur <= max_dur):
                continue
            txt = CLEANERS[lang](d["text"])
            if not txt:
                continue
            src = d.get("source", "?")
            if max_per_source and per_src.get(src, 0) >= max_per_source.get(src, 1 << 60):
                continue
            per_src[src] = per_src.get(src, 0) + 1
            items.append({"wav": d["wav"], "text": txt, "dur": dur, "src": src, **({"twer": d["twer"]} if "twer" in d else {})})
    if sp is not None:
        keep = []
        for it in items:
            ids = sp.encode(it["text"])
            # CTC feasibility at 25 Hz output (need >= len + repeats frames); be conservative
            if len(ids) + 2 < it["dur"] * 25 * 0.9:
                it["ids"] = ids
                keep.append(it)
        items = keep
    return items


class AudioDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            a, sr = sf.read(it["wav"], dtype="int16", always_2d=False)
            if a.ndim > 1:
                a = a[:, 0]
            if sr != 16000:  # e.g. Ogg/Opus decoded at 48 kHz
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(16000, sr)
                a = np.clip(resample_poly(a.astype(np.float32), 16000 // g, sr // g), -32768, 32767).astype(np.int16)
        except Exception as e:  # corrupt file -> tiny silence, filtered by loss mask
            print("bad audio", it["wav"], e, flush=True)
            a = np.zeros(1600, dtype=np.int16)
        return a, it.get("ids", []), i


def collate(batch):
    lens = torch.tensor([len(b[0]) for b in batch])
    wav = torch.zeros(len(batch), int(lens.max()), dtype=torch.int16)
    for j, b in enumerate(batch):
        wav[j, : len(b[0])] = torch.from_numpy(b[0])
    tl = torch.tensor([len(b[1]) for b in batch])
    tok = torch.zeros(len(batch), max(1, int(tl.max())), dtype=torch.long)
    for j, b in enumerate(batch):
        if len(b[1]):
            tok[j, : len(b[1])] = torch.tensor(b[1])
    return wav, lens, tok, tl, torch.tensor([b[2] for b in batch])


class BucketBatchSampler(Sampler):
    """Batches of similar duration; each batch's padded duration <= max_batch_sec. Shards across DDP ranks."""

    def __init__(self, items, max_batch_sec, seed=0, rank=0, world=1, n_buckets=40, max_batch_size=512):
        self.durs = np.array([it["dur"] for it in items])
        self.max_sec, self.seed, self.rank, self.world = max_batch_sec, seed, rank, world
        self.n_buckets, self.max_bs = n_buckets, max_batch_size
        self.epoch = 0
        self._batches = self._make()

    def set_epoch(self, e):
        self.epoch = e
        self._batches = self._make()

    def _make(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.argsort(self.durs + rng.uniform(0, 0.3, len(self.durs)))
        batches, cur, cur_max = [], [], 0.0
        for i in order:
            d = self.durs[i]
            m = max(cur_max, d)
            if cur and (m * (len(cur) + 1) > self.max_sec or len(cur) >= self.max_bs):
                batches.append(cur)
                cur, m = [], d
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        n = len(batches) // self.world * self.world
        return batches[self.rank:n:self.world]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)
