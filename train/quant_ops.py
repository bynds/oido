"""Fake-quantization ops that mirror the ESP32 engine's integer arithmetic (straight-through gradients).

fq_weight: per-output-row symmetric int8/int4 weights; fq_act / fq_rows: dynamic per-row int8 activations
(the C code computes inv = 127 / max, q = rint(x * inv), dequantizes with max / 127).
Used by train_nemo_qat.py and train_nemo_stream.py.
"""
import torch


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def rnd(x):
    return _RoundSTE.apply(x)


def fq_weight(w, bits):
    """w: (N, K). symmetric per-row."""
    qmax = 2 ** (bits - 1) - 1
    s = w.detach().abs().amax(1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.clamp(rnd(w / s), -qmax, qmax) * s


def fq_act(x):
    m = x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return torch.clamp(rnd(x * (127.0 / m)), -127, 127) * (m / 127.0)


def fq_rows(x, qmax=127):
    m = x.detach().abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return torch.clamp(rnd(x * (qmax / m)), -qmax, qmax) * (m / qmax)
