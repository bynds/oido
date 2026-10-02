"""Export the NeMo conformer-ctc-small port to the TNM1 blob for the C engine (tasr_nemo.c).

python export_nemo.py ../models/nemo_small out.tnm [bits] [checkpoint] [tokenizer.model]

Header word 10 holds flags: bit 0 = fixed feature normalization (80 means + 80 stds appended after the tokens),
bit 1 = trained for streaming (chunked attention, chunk-limited depthwise conv); words 11/12 = default streaming
chunk and left context in encoder frames.
"""
import os, struct, sys
import numpy as np, torch, sentencepiece as spm
from export import W
from nemo_small import NemoSmall


def export(d, out, bits=8, ckpt=None, tokenizer=None, chunk=32, left=128):
    # CTC model: state_dict_plain.pt (fetch_nemo_small.py); transducer: model_weights.ckpt unpacked from the .nemo
    path = f"{d}/state_dict_plain.pt" if os.path.exists(f"{d}/state_dict_plain.pt") else f"{d}/model_weights.ckpt"
    sd = dict(torch.load(path, map_location="cpu"))
    rnnt = "decoder.prediction.embed.weight" in sd
    if rnnt:  # NemoSmall builds a CTC head; the transducer has none
        sd["decoder.decoder_layers.0.weight"] = torch.zeros(1025, 176, 1)
        sd["decoder.decoder_layers.0.bias"] = torch.zeros(1025)
    m = NemoSmall(sd).eval()
    st = {}
    if ckpt:  # fine-tuned weights (NemoSmall state dict; train_nemo_qat.py or train_nemo_stream.py)
        st = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(st["model"])
        print("loaded fine-tuned weights from", ckpt, "step", st.get("step"))
    assert not st.get("causal"), "causal-conv checkpoints are not supported by the engine"
    fixed = st.get("norm_mean") is not None
    flags = (1 if fixed else 0) | (2 if st.get("stream") else 0)
    if tokenizer is None:
        tok = "tokenizer.model" if os.path.exists(f"{d}/tokenizer.model") else [f for f in os.listdir(d) if f.endswith("tokenizer.model")][0]
        tokenizer = f"{d}/{tok}"
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    V = sp.get_piece_size()
    w = W()
    w.buf += b"TNM1" + struct.pack("<15I", 1, 176, 4, 704, 31, 16, 176, V, bits, int(rnnt), flags,
                                   chunk if flags & 2 else 0, left if flags & 2 else 0, 0, 0)
    w.f32(m.pre.window)
    w.f32(m.pre.fb.t().contiguous().reshape(-1))            # (257, 80) row-major, like TASR
    w.f32(m.conv0.weight.reshape(-1)); w.f32(m.conv0.bias)
    w.qlin(m.conv2.weight.flatten(1), m.conv2.bias, 8)       # im2col K = 176*9 in (c, kh, kw) order
    w.qlin(m.sub_out.weight, m.sub_out.bias, 8)
    for L in m.layers:
        def ln(n):
            w.f32(n.weight); w.f32(n.bias)
        ln(L.norm_feed_forward1)
        w.qlin(L.ff1_l1.weight, L.ff1_l1.bias, bits); w.qlin(L.ff1_l2.weight, L.ff1_l2.bias, bits)
        ln(L.norm_self_att)
        w.qlin(torch.cat([L.linear_q.weight, L.linear_k.weight, L.linear_v.weight]),
               torch.cat([L.linear_q.bias, L.linear_k.bias, L.linear_v.bias]), bits)
        w.qlin(L.linear_out.weight, L.linear_out.bias, bits)
        w.qlin(L.linear_pos.weight, None, 8)
        w.f32(L.pos_bias_u.reshape(-1)); w.f32(L.pos_bias_v.reshape(-1))
        ln(L.norm_conv)
        w.qlin(L.pw1.weight, L.pw1.bias, bits)
        w.f32(L.dw.weight.reshape(-1)); w.f32(L.dw.bias)
        w.qlin(L.pw2.weight, L.pw2.bias, bits)
        ln(L.norm_feed_forward2)
        w.qlin(L.ff2_l1.weight, L.ff2_l1.bias, bits); w.qlin(L.ff2_l2.weight, L.ff2_l2.bias, bits)
        ln(L.norm_out)
    if rnnt:  # RNN-T prediction network (embedding + 1-layer LSTM 320) and joint network, all int8
        g = lambda k: sd[k].float()
        w.qlin(g("decoder.prediction.embed.weight"), torch.zeros(V + 1), 8)
        w.qlin(g("decoder.prediction.dec_rnn.lstm.weight_ih_l0"), g("decoder.prediction.dec_rnn.lstm.bias_ih_l0"), 8)
        w.qlin(g("decoder.prediction.dec_rnn.lstm.weight_hh_l0"), g("decoder.prediction.dec_rnn.lstm.bias_hh_l0"), 8)
        w.qlin(g("joint.enc.weight"), g("joint.enc.bias"), 8)
        w.qlin(g("joint.pred.weight"), g("joint.pred.bias"), 8)
        w.qlin(g("joint.joint_net.2.weight"), g("joint.joint_net.2.bias"), 8)  # 1025 outputs, blank = 1024
    else:
        w.qlin(m.head.weight, m.head.bias, 8)                # 1025 outputs, blank = index 1024
    w.align()
    for i in range(V):
        b = sp.id_to_piece(i).replace("▁", " ").encode()
        if sp.is_unknown(i) or sp.is_control(i):
            b = b""
        w.buf += bytes([len(b)]) + b
    if fixed:
        w.f32(st["norm_mean"].float()); w.f32(st["norm_std"].float())
    open(out, "wb").write(w.buf)
    print(f"wrote {out}: {len(w.buf)/1e6:.3f} MB (bits={bits}, vocab={V}, flags={flags})")


if __name__ == "__main__":
    a = sys.argv[1:]
    export(a[0], a[1], int(a[2]) if len(a) > 2 else 8, a[3] if len(a) > 3 and a[3] != "-" else None,
           a[4] if len(a) > 4 else None)
