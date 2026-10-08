#!/usr/bin/env python3
"""make-ab-streams.py: simulated continuous audio for the segmenter A/B (oido_seg_ab), in build/ab/.

Each stream is what a live feed would deliver around an utterance: continuous background noise with speech inside,
plus a labels file giving the true utterance spans. This is a stand-in until real recordings of Jibo's processed
audio exist; those plug into the same harness with hand-made labels in the same format.

  speech     the 27 published demo clips (build/fixtures, from make-fixtures.sh) at 0 dB and -20 dB (a far talker)
  background seeded pink noise (1/f power, high-passed at 80 Hz as a microphone/voice chain would), continuous under
             the whole stream, at -60 dBFS RMS ("quiet") and -40 dBFS ("noisy");
             1.5 s before and after the speech
  negatives  6 s of background alone, 3 seeds per level (any transcript is a false trigger)
  pairs      6 pairs of clips back to back with 0.5 s and 1.2 s of background between them (quiet, 0 dB): below and
             above the segmenter's 0.8 s hang, so one or two utterances are the right segmentation respectively

Output: build/ab/streams/*.wav (16 kHz mono PCM16) and build/ab/labels.tsv with one row per true utterance:
  stream  condition  start_sample  end_sample  reference_text      (negatives: one row with start = end = -1)
Deterministic (fixed seeds). Host tool; needs numpy.
"""
import json, os, wave
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIX = os.path.join(ROOT, "build", "fixtures")
OUT = os.path.join(ROOT, "build", "ab")
SR = 16000


def read(name):
    with wave.open(os.path.join(FIX, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def write(name, x):
    y = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with wave.open(os.path.join(OUT, "streams", name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.tobytes())


def pink(n, dbfs, seed):
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64) * SR / n   # Hz
    shape = np.where(f >= 80.0, 1.0 / np.sqrt(np.maximum(f, 1.0)), 0.0)   # 1/f power above 80 Hz: microphones and
    x = np.fft.irfft(spec * shape, n)                                      # voice processing high-pass below that
    x *= 32768.0 * 10 ** (dbfs / 20) / np.sqrt(np.mean(x ** 2))
    return x


def main():
    os.makedirs(os.path.join(OUT, "streams"), exist_ok=True)
    refs = {e["mp3"].split("/")[-1][:-4]: e["ref"] for e in json.load(open(os.path.join(ROOT, "docs/samples/results.json")))}
    clips = sorted(refs)
    lead = tail = int(1.5 * SR)
    rows = []
    seed = 1
    for bg_name, bg_db in (("quiet", -60), ("noisy", -40)):
        for lvl in (0, -20):
            cond = f"{bg_name}_{'0dB' if lvl == 0 else 'm20dB'}"
            for c in clips:
                s = read(c + ".wav") * 10 ** (lvl / 20)
                n = lead + len(s) + tail
                x = pink(n, bg_db, seed); seed += 1
                x[lead:lead + len(s)] += s
                name = f"{cond}__{c}.wav"
                write(name, x)
                rows.append((name, cond, lead, lead + len(s), refs[c]))
        for k in range(3):
            name = f"{bg_name}_negative__noise{k}.wav"
            write(name, pink(6 * SR, bg_db, 1000 + seed + k))
            rows.append((name, f"{bg_name}_negative", -1, -1, ""))
        seed += 10
    pairs = [("real_20", "real_21"), ("real_22", "real_23"), ("real_24", "real_25"), ("real_26", "rooms_00"),
             ("rooms_02", "rooms_10"), ("rooms_12", "real_22")]
    for gap in (0.5, 1.2):
        g = int(gap * SR)
        for a, b in pairs:
            sa, sb = read(a + ".wav"), read(b + ".wav")
            n = lead + len(sa) + g + len(sb) + tail
            x = pink(n, -60, seed); seed += 1
            x[lead:lead + len(sa)] += sa
            o = lead + len(sa) + g
            x[o:o + len(sb)] += sb
            cond = f"pair_gap{int(gap * 1000)}ms"
            name = f"{cond}__{a}+{b}.wav"
            write(name, x)
            rows.append((name, cond, lead, lead + len(sa), refs[a]))
            rows.append((name, cond, o, o + len(sb), refs[b]))
    with open(os.path.join(OUT, "labels.tsv"), "w") as f:
        f.write("stream\tcondition\tstart_sample\tend_sample\treference\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")
    print(f"{len({r[0] for r in rows})} streams, {sum(r[2] >= 0 for r in rows)} utterances -> {OUT}")


if __name__ == "__main__":
    main()
