# perfvm: exact ARMv7 instruction counts for the Oído engine

Engine cost on Jibo's instruction set, measured without a robot. Adapted from bynds/needle-rs
(`ports/jibo/perfvm`, branch `claude/port-creation-y2ehfd`, commit `dcb8f79`); file headers say what was copied.

`qemu-user` has no PMU: a guest's `perf_event_open` would count the emulator's own instructions. Full-system qemu
does. With `-icount shift=0`, TCG retires instructions deterministically and the emulated PMU's
instructions-retired event is exact and repeatable. The guest is a pinned arm64 Ubuntu kernel
(`fetch-kernel.sh`, SHA-256 checked) on `qemu-system-aarch64 -M virt -cpu cortex-a57`. It runs the unmodified
armhf binaries in AArch32 EL0 against the glibc 2.21 stand-in sysroot, from an initramfs that `mkinitramfs.py`
writes without root (newc cpio in pure Python). `init.c` is a tiny static armhf `/init`: it runs a tab-separated
jobs file and powers off. `selftest.c` checks the counter itself: the 1M- and 2M-iteration loops must cost
6,000,056 and 12,000,056 instructions on every run (bench.sh refuses a run otherwise).

```sh
JIBO_SYSROOT=... perfvm/bench.sh LABEL              # build both variants, run, write results/LABEL.jsonl
perfvm/compare.py results/LABEL.jsonl [results/BASE.jsonl]   # per-stage table, change vs baseline
perfvm/gate.sh golden                               # once, on the pre-change tree: reference trace (x86)
JIBO_SYSROOT=... perfvm/gate.sh check               # after every change: byte-for-byte parity + suites
perfvm/run.sh JOBS DEST=SRC...                      # any ARMv7 programs, counted
```

**Variants.** *plain*: `-march=armv7-a -mfpu=vfpv3-d16 -mfloat-abi=hard`, as shipped. *neon*: `-mfpu=neon`
with the NEON int8 kernels (`ports/jibo/kernels_neon.c`).

**Workloads** (`bench_engine.c`; this engine has no prefill/decode, so these are its own):
- **utterance**: `nemo8.tnm`, full-context transcription of `real_22.wav` (5.76 s of speech, 145 encoder frames);
- **stream**: `oido_stream.tnm`, the same clip through the stream API in 320-sample blocks, then finish. The
  stream object is built beforehand and not counted;
- **stream_setup**: building that stream object (per-object positional projections).

**Spans** (`tasr_nemo.c`, `-DTASR_PROFILE`, compiled out by default; the clock is pluggable and here is the PMU):
features, conv0, im2col, gemm_fe (front-end GEMMs, K > 1024), gemm_k704 (256 < K ≤ 1024), gemm (K ≤ 256),
quant, layernorm, act, qkv_int8, pos, attention, dwconv, head+dec. Each span counts its own instructions,
exclusive of spans nested in it, plus calls. `other` is the total minus all spans (glue between stages).

**What the counts are not**: time. Instructions on a Cortex-A9/A15 differ in cost (a NEON multiply, a cache miss)
and qemu models none of that. They are the right measure for comparing two implementations of the same work,
which is what the optimisation loop needs. Absolute latency needs the robot.

**Parity gate** (`gate.sh`). The golden trace (28 files) comes from the pre-change tree built for x86:
- `nemo8` utterance and `nemo4` (int4) transcripts, frame counts and every raw logit on 8 fixtures (clean, noisy,
  reverberant, spontaneous, 321 samples, 1 s, noise only);
- `oido_stream` through the stream API (int8 K/V caches) in 320-sample and irregular blocks;
- `nemo8` with LM beam search.

`check` builds x86, ARM plain and ARM NEON from the tree into a private directory and requires every file to be
byte-identical. ARM runs under qemu-user against Ubuntu's armhf glibc 2.39, whose libm matches the host's (with
Jibo's 2.21, `logf` differs by an ulp on 1.85% of inputs; see the port README). It then runs
`tests/run-tests.sh quick`: the sanitizer, loader, kernel (NEON under qemu), stream-lifecycle and service suites.
One gate takes about 15 minutes. Benches and gates build into their own directories, so the tree can be edited
once they print "built".

## Rounds

Instructions for the 5.76 s clip (millions). Every round passed the gate before it was committed.

| Round | Change | plain utterance | plain stream | neon utterance | neon stream |
|---|---|---|---|---|---|
| r00 | baseline | 15,384.6 | 14,203.0 | 2,129.3 | 1,968.4 |
| r01 | plain: ARMv6 SIMD32 int8 kernels (`sxtb16` + `smlad`, 4-row register blocking) | 6,495.9 (−57.8%) | 6,001.6 (−57.7%) | 2,129.3 | 1,968.4 |
