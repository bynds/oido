"""Quantization-aware fine-tuning of the NeMo conformer-ctc-small port for the ESP32 engine.

Weights: int4 per-channel for all conformer-layer linears (except linear_pos), int8 front end / head; activations
int8 per row; attention int8 (same arithmetic as tasr_nemo.c). Batched, padded, full-context.

python train_nemo_qat.py --train ../train_v2a.jsonl --nemo ../nemo_small --out ../exp/nemo_qat --steps 8000
"""
import argparse, copy, math, os, random, sys, time
import numpy as np, sentencepiece as spm, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from data import AudioDataset, BucketBatchSampler, collate, load_manifest
from nemo_small import NemoSmall, rel_pos_emb
from quant_ops import fq_act, fq_weight, fq_rows


class BatchPre(nn.Module):
    """Batched NeMo preprocessing with per-utterance per-feature normalization."""
    def __init__(self, window, fb):
        super().__init__()
        self.register_buffer("window", window.float())
        self.register_buffer("fb", fb.float())  # (80, 257)

    def forward(self, wav, lens):
        x = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], 1)
        spec = torch.stft(x, 512, 160, 400, window=self.window, center=True, pad_mode="constant", return_complex=True)
        mel = torch.log(torch.matmul(self.fb, spec.real.square() + spec.imag.square()) + 2.0 ** -24)  # (B, 80, T)
        n = lens // 160
        T = mel.shape[-1]
        valid = (torch.arange(T, device=wav.device)[None] < n[:, None]).float()[:, None]
        mean = (mel * valid).sum(-1, keepdim=True) / n[:, None, None]
        std = torch.sqrt(((mel - mean) ** 2 * valid).sum(-1, keepdim=True) / (n[:, None, None] - 1).clamp(min=1)) + 1e-5
        out = ((mel - mean) / std) * valid
        return out.transpose(1, 2), n + 1  # (B, T, 80); encoder sees n+1 frames (last is zero)


class QModel(nn.Module):
    def __init__(self, m: NemoSmall, bits=4, quant=True):
        super().__init__()
        self.m, self.bits, self.quant = m, bits, quant

    def ql(self, lin, x, bits):
        if not self.quant:
            return F.linear(x, lin.weight, lin.bias)
        return F.linear(fq_act(x), fq_weight(lin.weight, bits), lin.bias)

    def fr(self, x):
        return fq_rows(x) if self.quant else x

    def layer(self, L, x, mask_k, pad, pe):
        b = self.bits
        B, T, d = x.shape
        h = L.norm_feed_forward1(x)
        x = x + 0.5 * self.ql(L.ff1_l2, F.silu(self.ql(L.ff1_l1, h, b)), b)
        h = L.norm_self_att(x)
        q = self.ql(L.linear_q, h, b).view(B, T, L.h, L.dk)
        k = self.fr(self.ql(L.linear_k, h, b).view(B, T, L.h, L.dk).transpose(1, 2))
        v = self.ql(L.linear_v, h, b).view(B, T, L.h, L.dk).transpose(1, 2)
        p = self.fr(self.ql(L.linear_pos, pe, 8).view(2 * T - 1, L.h, L.dk).transpose(0, 1))  # (H, 2T-1, dk)
        qu = self.fr((q + L.pos_bias_u).transpose(1, 2))
        qv = self.fr((q + L.pos_bias_v).transpose(1, 2))
        ac = qu @ k.transpose(-1, -2)
        bd = qv @ p.transpose(-1, -2)[None]
        idx = (T - 1) - torch.arange(T, device=x.device)[:, None] + torch.arange(T, device=x.device)[None, :]
        bd = torch.gather(bd, 3, idx[None, None].expand(B, L.h, T, T))
        s = (ac + bd) / math.sqrt(L.dk)
        a = torch.softmax(s.masked_fill(~mask_k[:, None, None, :], -1e4), -1)
        if self.quant:
            sv = v.detach().abs().amax(-1, keepdim=True).clamp(min=1e-30) / 127.0
            vq = torch.clamp(torch.round(v / sv), -127, 127) + (v / sv - (v / sv).detach())  # STE
            o = fq_rows(a * sv.transpose(-1, -2)) @ vq
        else:
            o = a @ v
        x = x + self.ql(L.linear_out, o.transpose(1, 2).reshape(B, T, d), b)
        h = L.norm_conv(x)
        g = F.glu(self.ql(L.pw1, h, b), dim=-1).masked_fill(pad[..., None], 0.0)
        c = F.silu(L.dw(g.transpose(1, 2)).transpose(1, 2))
        x = x + self.ql(L.pw2, c, b)
        h = L.norm_feed_forward2(x)
        x = x + 0.5 * self.ql(L.ff2_l2, F.silu(self.ql(L.ff2_l1, h, b)), b)
        return L.norm_out(x)

    def forward(self, feats, flen):
        m = self.m
        x = F.relu(m.conv0(feats[:, None]))
        B, C, T1, F1 = x.shape
        cols = F.unfold(x, 3, padding=1, stride=2)  # (B, C*9, L)
        T2, F2 = (T1 + 2 - 3) // 2 + 1, (F1 + 2 - 3) // 2 + 1
        if self.quant:
            y = F.linear(fq_act(cols.transpose(1, 2)), fq_weight(m.conv2.weight.flatten(1), 8), m.conv2.bias)  # (B, L, C)
        else:
            y = F.linear(cols.transpose(1, 2), m.conv2.weight.flatten(1), m.conv2.bias)
        y = F.relu(y).view(B, T2, F2, C).permute(0, 1, 3, 2).reshape(B, T2, C * F2)
        x = self.ql(m.sub_out, y, 8) * math.sqrt(176.0)
        lens = ((flen + 2 - 3) // 2 + 1 + 2 - 3) // 2 + 1
        mask_k = torch.arange(T2, device=x.device)[None] < lens[:, None]
        pe = rel_pos_emb(T2, 176).to(x.device)
        for L in m.layers:
            x = self.layer(L, x, mask_k, ~mask_k, pe)
        return self.ql(m.head, x, 8), lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch_sec", type=float, default=500)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--init", default="", help="start from a NemoSmall state dict checkpoint (e.g. a fine-tune)")
    ap.add_argument("--teacher", default="", help="'base' or a checkpoint: frame-level KD from its fp32 posteriors")
    ap.add_argument("--kd", type=float, default=0.0, help="KD loss weight")
    ap.add_argument("--ctc", type=float, default=1.0, help="CTC loss weight")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    sd = torch.load(f"{a.nemo}/state_dict_plain.pt", map_location="cpu")
    base = NemoSmall(sd)
    if a.init:
        base.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
        print("init from", a.init, flush=True)
    teacher = None
    if a.teacher:
        tb = NemoSmall(sd)
        if a.teacher != "base":
            tb.load_state_dict(torch.load(a.teacher, map_location="cpu")["model"])
        teacher = QModel(tb, a.bits, quant=False).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("KD teacher:", a.teacher, "weight", a.kd, flush=True)
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"].reshape(80, 257)).to(dev)
    model = QModel(base, a.bits).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    sp = spm.SentencePieceProcessor(model_file=f"{a.nemo}/tokenizer.model")
    items = load_manifest([a.train], max_dur=18.0, sp=None)
    for it in items:
        it["ids"] = sp.encode(it["text"])
    items = [it for it in items if 0 < len(it["ids"]) < it["dur"] * 25 * 0.8]
    val = []
    for v in [x for x in a.val.split(",") if x]:
        val += load_manifest([v], max_dur=35.0)[:300]
    print(f"train utts {len(items)}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sampler = BucketBatchSampler(items, a.batch_sec, seed=1)
    step, t0, ag = 0, time.time(), 0.0

    def evaluate(net):
        import jiwer
        net.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for it in val:
                import soundfile as sf
                w, _ = sf.read(it["wav"], dtype="float32")
                wav = torch.from_numpy(w)[None].to(dev)
                f, fl = pre(wav, torch.tensor([len(w)], device=dev))
                lg, _ = net(f, fl)
                ids = lg[0].argmax(-1).tolist()
                out, prev = [], -1
                for i in ids:
                    if i != prev and i != 1024:
                        out.append(i)
                    prev = i
                hyps.append(sp.decode(out)); refs.append(it["text"])
        model.train()
        return 100 * jiwer.wer(refs, hyps)

    if val:
        print(f"step 0 int{a.bits} val WER {evaluate(model):.2f}", flush=True)
    model.train()
    ep = 0
    while step < a.steps:
        sampler.set_epoch(ep)
        dl = DataLoader(AudioDataset(items), batch_sampler=sampler, collate_fn=collate, num_workers=a.workers,
                        pin_memory=True, prefetch_factor=4)
        for wav, lens, tok, tl, _ in dl:
            wav = wav.to(dev).float() / 32768.0
            lens, tok, tl = lens.to(dev), tok.to(dev), tl.to(dev)
            with torch.no_grad():
                f, fl = pre(wav, lens)
                # light SpecAugment
                B, T, _ = f.shape
                for _ in range(2):
                    w = torch.randint(0, 16, (B, 1), device=dev); s0 = (torch.rand(B, 1, device=dev) * (80 - w)).long()
                    fr = torch.arange(80, device=dev)[None]
                    f = f.masked_fill(((fr >= s0) & (fr < s0 + w))[:, None, :], 0.0)
                for _ in range(max(1, T // 200)):
                    w = torch.randint(0, 25, (B, 1), device=dev); s0 = (torch.rand(B, 1, device=dev) * (fl[:, None] - w).clamp(min=1)).long()
                    tr = torch.arange(T, device=dev)[None]
                    f = f.masked_fill(((tr >= s0) & (tr < s0 + w))[:, :, None], 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, ol = model(f, fl)
            lp = lg.float().log_softmax(-1)
            loss = a.ctc * F.ctc_loss(lp.transpose(0, 1), tok, ol, tl, blank=1024, reduction="sum", zero_infinity=True) / tl.sum()
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    tg, _ = teacher(f, fl)
                tp = tg.float().log_softmax(-1)
                valid = (torch.arange(lp.shape[1], device=dev)[None] < ol[:, None]).float()
                kl = (tp.exp() * (tp - lp)).sum(-1)  # KL(teacher || student) per frame
                loss = loss + a.kd * (kl * valid).sum() / valid.sum()
            lr = a.lr * min(1.0, (step + 1) / 300) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe_, pm in zip(ema.parameters(), model.parameters()):
                    pe_.mul_(0.999).add_(pm.detach(), alpha=0.001)
            step += 1
            ag += loss.item()
            if step % 100 == 0:
                print(f"step {step} loss {ag/100:.4f} lr {lr:.2e} {time.time()-t0:.0f}s", flush=True)
                ag = 0.0
            if step % a.val_every == 0 or step == a.steps:
                if val:
                    print(f"step {step} EMA int{a.bits} val WER {evaluate(ema):.2f}", flush=True)
                torch.save({"model": ema.m.state_dict(), "bits": a.bits, "step": step}, f"{a.out}/last.pt")
            if step >= a.steps:
                break
        ep += 1


if __name__ == "__main__":
    main()
