"""Waveform augmentation for ASR training: additive noise / music / babble from MUSAN (CC-BY-4.0) at random SNR and
simulated room reverberation (pyroomacoustics RIRs), applied on the GPU to a padded batch.

Build the banks once:
  python augment.py bank --musan /opt/dlami/nvme/aug/musan --out /opt/dlami/nvme/aug     # bank.i16 + bank.json
  python augment.py rirs --out /opt/dlami/nvme/aug --n 400                                 # rirs.npy

Use in training: wrap the dataset so each worker augments its own utterances:  aug = NpAug("/opt/dlami/nvme/aug"); a = aug(a)  # int16
The evaluation benchmark (eval/make_robust.py) uses DEMAND noise and its own room, which are not in MUSAN or this RIR set.
"""
import argparse, glob, json, os, random
import numpy as np


def build_bank(musan, out, hours=(("noise", 6.0), ("music", 10.0), ("speech", 14.0))):
    import soundfile as sf
    rng = random.Random(0)
    index, pos = [], 0
    with open(os.path.join(out, "bank.i16"), "wb") as f:
        for kind, h in hours:
            files = sorted(glob.glob(os.path.join(musan, kind, "**", "*.wav"), recursive=True))
            rng.shuffle(files)
            tot = 0.0
            for p in files:
                if tot >= h * 3600:
                    break
                x, sr = sf.read(p, dtype="int16")
                if x.ndim > 1:
                    x = x[:, 0]
                if sr != 16000 or len(x) < 16000:
                    continue
                f.write(x.tobytes())
                index.append([kind, pos, len(x)])
                pos += len(x)
                tot += len(x) / 16000
            print(kind, round(tot / 3600, 1), "h", flush=True)
    json.dump(index, open(os.path.join(out, "bank.json"), "w"))


def build_rirs(out, n=400, length=16000, seed=0):
    import pyroomacoustics as pra
    rng = np.random.default_rng(seed)
    rirs = np.zeros((n, length), dtype=np.float32)
    i = 0
    while i < n:
        dims = [rng.uniform(3, 12), rng.uniform(3, 10), rng.uniform(2.4, 4.0)]
        rt60 = rng.uniform(0.15, 1.0)
        try:
            e_abs, order = pra.inverse_sabine(rt60, dims)
            room = pra.ShoeBox(dims, fs=16000, materials=pra.Material(e_abs), max_order=min(order, 17))
            src = [rng.uniform(0.5, d - 0.5) for d in dims[:2]] + [rng.uniform(1.0, 1.9)]
            for _ in range(20):  # microphone 0.5-4 m from the talker
                mic = [rng.uniform(0.5, d - 0.5) for d in dims[:2]] + [rng.uniform(0.8, 1.8)]
                if 0.5 <= np.linalg.norm(np.array(mic) - np.array(src)) <= 4.0:
                    break
            room.add_source(src)
            room.add_microphone(np.array(mic)[:, None])
            room.compute_rir()
            h = np.asarray(room.rir[0][0], dtype=np.float32)[:length]
        except Exception:
            continue
        if len(h) < 100 or not np.isfinite(h).all():
            continue
        rirs[i, : len(h)] = h / (np.abs(h).max() + 1e-9)
        i += 1
    np.save(os.path.join(out, "rirs.npy"), rirs)
    print("rirs", rirs.shape)


class NpAug:
    """Per-utterance augmentation in numpy (runs inside the DataLoader workers): int16 in, int16 out."""

    def __init__(self, d, p_noise=0.6, p_rir=0.3, snr=(0.0, 20.0)):
        self.d, self.p_noise, self.p_rir, self.snr = d, p_noise, p_rir, snr
        self.bank = None

    def _init(self):  # lazily, in the worker process
        d = self.d
        self.bank = np.memmap(os.path.join(d, "bank.i16"), dtype=np.int16, mode="r")
        idx = json.load(open(os.path.join(d, "bank.json")))
        self.by = {k: [(s, n) for kk, s, n in idx if kk == k] for k in ("noise", "music", "speech")}
        self.rirs = np.load(os.path.join(d, "rirs.npy"))
        self.rng = random.Random(os.getpid() * 7919 + 1)

    def _clip(self, kind, T):
        s, n = self.rng.choice(self.by[kind])
        if n >= T:
            o = s + self.rng.randrange(n - T + 1)
            return np.asarray(self.bank[o:o + T], dtype=np.float32) / 32768.0
        x = np.asarray(self.bank[s:s + n], dtype=np.float32) / 32768.0
        return np.tile(x, T // n + 1)[:T]

    def __call__(self, a):
        if self.bank is None:
            self._init()
        from scipy.signal import fftconvolve
        L = len(a)
        x = a.astype(np.float32) / 32768.0
        if self.rng.random() < self.p_rir:  # reverberation, direct path kept aligned, level kept
            h = self.rirs[self.rng.randrange(len(self.rirs))]
            pk = int(np.abs(h).argmax())
            y = fftconvolve(x, h)[pk:pk + L].astype(np.float32)
            x = y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-8))
        if self.rng.random() < self.p_noise:
            r = self.rng.random()
            if r < 0.45:
                nz = self._clip("noise", L)
            elif r < 0.65:
                nz = self._clip("music", L)
            else:  # babble: 2-4 overlapping speakers
                nz = sum(self._clip("speech", L) for _ in range(self.rng.randint(2, 4)))
            snr = self.rng.uniform(*self.snr)
            ps, pn = (x ** 2).mean() + 1e-10, (nz ** 2).mean() + 1e-10
            x = x + nz * np.sqrt(ps / (pn * 10 ** (snr / 10)))
        return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "rirs"])
    ap.add_argument("--musan", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    build_bank(a.musan, a.out) if a.cmd == "bank" else build_rirs(a.out, a.n)
