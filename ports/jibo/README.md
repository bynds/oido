# Oído on the original Jibo: port status

Native, local English transcription on Jibo's existing firmware (Tegra K1, 32-bit ARMv7 hard float, glibc 2.21),
with Oído's own engine and models. Port code lives here. Upstream sources changed: the NeMo engine's loader and
robustness code and its Linux profiler (`esp32/components/tinyasr/tasr_nemo.{c,h}`), and an opt-in renaming hook
in the kernels (`kernels.{c,h}`); see [Engine changes](#engine-changes). Every model output is bit-identical to the
unmodified engine's.

**Status (8 October 2026): everything that can be done without the robot is done for milestones 1–6, on the host
and in an ARM emulator. Nothing has run on a Jibo.** Every result below names where it ran.

| Milestone | State |
|---|---|
| 1. Baseline, builds, ABI | done: host and ARMv7 builds, ABI-clean (libc/libm/loader, `GLIBC_2.17` at most); emulated ARM bit-identical to host given the same libm |
| 2. Validation, lifecycle | done: loader validation, allocation-failure cleanup, NaN-safe lookups; fuzz-style test under ASan/UBSan |
| 3. NEON int8 kernels | done in emulation: exact against the C kernels (11,185 kernel cases; all 35 fixtures bit-identical end to end); **speed unmeasured** (needs the robot: `robot-run.sh kernels`) |
| 4. Streaming replay | done: block-size invariance on 35/35 fixtures (host, ARM scalar, ARM NEON); lifecycle test |
| 5. Service | done except live audio: persistent process, bounded queue, overrun reporting, JSON events, stdin or 0600 socket; tested under TSan and ASan |
| 6. Language model | host comparison done; robot cost pending (`robot-run.sh lm`) |
| 7. GPU | not started, by design: only if robot profiling shows the CPU cannot meet the budget |
| Physical Jibo | **pending**: no route to the robot from here; `scripts/robot-run.sh` is ready and dry-run tested |
| Owner's `jibo-armcc` (Linaro GCC 4.8.4) | **not built**: the wrapper is not in this environment |
| Live audio endpoint | **unknown**: must be recovered on the robot (format, channel, rate, access) |

## Layout

```text
ports/jibo/
  README.md                 this file
  build-jibo.sh             builds: host, host-profile, jibo (scalar), jibo-neon, jibo-profile
  check-jibo-abi.sh         ELF/ABI gate, copied from the Decider port (provenance in its header)
  oido_cli.c                utterance transcription: bounded WAV input, wall + CPU time, allocations, logits,
                            optional LM beam search, profile output, strict exit status
  oido_stream_replay.c      streaming replay: block schedules, invariance check, chunk timing, modeled latency
  oido_service.c            persistent recognizer: queue, partial/final JSON events, stdin or Unix socket
  oido_feed.c               WAV -> service protocol (optionally paced at real time), for tests and the robot
  oido_seg_ab.c             A/B of upstream's VAD/AGC segmenter (tasr_seg.c) in front of the recognizer
  kernels_neon.c            kernel dispatch: ARMv7 NEON int8 kernels, force-scalar switch, shape statistics
  kernels_dispatch.h        statistics interface of the dispatch layer
  port_util.{c,h}           WAV reader, timing, allocation accounting shared by the programs
  sha256.{c,h}              model hash reported by the service
  tests/
    run-tests.sh            all host tests under sanitizers, plus the NEON kernel test under qemu if available
    test_model_load.c       loader/stream constructor vs truncation, bad headers, misalignment, failing allocations
    test_kernels.c          dispatched (NEON) vs C kernels, exact, every branch and edge case
    bench_kernels.c         kernel speed on the model's hot shapes, scalar vs NEON (meaningful on the robot only)
    test_stream_lifecycle.c reset/reuse, leakage, finish semantics, chunk boundaries, truncation, two streams
    test_service.py         oido_service end to end (host; standard library Python)
    libm_fingerprint.c      hashes of logf/expf results: tells which libm (and so which reference) applies
  scripts/
    model-manifest.sh       model sizes, SHA-256, TNM1 header fields -> manifests/models.tsv
    make-fixtures.sh        fixture pack -> build/fixtures, hashes -> manifests/
    run-baseline.sh         oido_cli over the fixtures -> build/runs/LABEL (native, or RUNNER=qemu-arm ...)
    run-stream.sh           oido_stream_replay over the fixtures -> build/runs/LABEL
    compare-runs.py         same-model comparison of two runs: transcripts, frames, logit error, first divergence
    score.py                WER against upstream's reference transcripts of the demo clips
    make-ab-streams.py      simulated continuous audio with labelled utterances for the segmenter A/B
    run-seg-ab.sh           oido_seg_ab over labelled streams (simulated, or recordings from the robot)
    score-seg-ab.py         A/B summary per arm and condition
    robot-run.sh            owner-run steps on the robot (JIBO_SSH=local dry-runs them on this machine)
manifests/                  upstream commit, model and fixture hashes, toolchain provenance
results/jibo/               recorded results (transcripts, logs, comparisons, profile); logits are regenerated
```

`build/` is ignored by git.

## Reproduce

```bash
# inputs
ports/jibo/scripts/model-manifest.sh          # fails on a missing model, an LFS pointer or a wrong magic
ports/jibo/scripts/make-fixtures.sh           # needs ffmpeg; compare with manifests/fixtures-sha256.txt

# builds (JIBO_CC=<jibo-armcc> is the intended toolchain; JIBO_SYSROOT is the stand-in, manifests/toolchain.txt)
JIBO_SYSROOT=/path/to/glibc-2.21-armhf ports/jibo/build-jibo.sh all

# tests (sanitizers on the host; the NEON kernel test under qemu when JIBO_SYSROOT and qemu-arm are present)
JIBO_SYSROOT=... ports/jibo/tests/run-tests.sh          # `quick` checks one model in the loader test

# runs and comparisons
ports/jibo/scripts/run-baseline.sh host-scalar build/host/oido_cli
RUNNER="qemu-arm -L $JIBO_SYSROOT" JOBS=4 ports/jibo/scripts/run-baseline.sh qemu-armv7-neon build/jibo-neon/oido_cli
ports/jibo/scripts/compare-runs.py build/runs/A/{transcripts.tsv,logits} build/runs/B/{transcripts.tsv,logits}
ports/jibo/scripts/run-stream.sh stream-host build/host/oido_stream_replay
build/host-profile/oido_cli --profile profile.tsv models/nemo8.tnm build/fixtures/real_2*.wav

# the service
build/host/oido_feed --realtime build/fixtures/real_22.wav | build/host/oido_service models/oido_stream.tnm

# on the robot, within the agreed scope only (owner); JIBO_SSH=local JIBO_DIR=/tmp/x dry-runs every step here
JIBO_SSH=jibo-skill@<robot> JIBO_DIR=<isolated dir> ports/jibo/scripts/robot-run.sh \
  deploy | status | kernels | run [neon|neon-forced-scalar|scalar] | stream | service | lm | cleanup
```

All builds compile the same float program: `-O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-tree-vectorize
-DTASR_NO_SIMD`; ARM adds `-march=armv7-a -mfpu=neon -mfloat-abi=hard`. `ESP_PLATFORM` is never defined, so the
ESP32-S3 kernels are not compiled. NEON is used only by the explicit int8 kernels (`jibo-neon`), whose sums are exact,
and `OIDO_KERNELS=scalar` switches the same binary back to the C kernels. Each build records compiler, flags and
source in `BUILD-INFO.txt` and its ABI check in `ABI-CHECK.txt`.

## Inputs

**Models** (`manifests/models.tsv`): `models/nemo8.tnm` (13,982,621 bytes, SHA-256 `452e332e…`, CC-BY-4.0),
`models/oido_stream.tnm` (13,983,264 bytes, `d815ad97…`, CC-BY-SA-4.0) and `models/nemo_lm.tlm` (1,288,852 bytes,
`00f174f8…`, CC-BY-4.0) are real files, not LFS pointers. Headers match the published architecture: d 176, 4 heads,
ff 704, kernel 31, 16 layers, vocabulary 1024, 8-bit. `oido_stream` has flags 3, stream chunk 32 and left context
128 (1.28 s and 5.12 s), not the header comment's fallback of 16. Sizes equal the Hugging Face files of
`lokutor-ai/oido-ctc-small-int8` and `-stream-int8`; their SHA-256 and repository revisions could **not** be read
(huggingface.co is blocked by this environment's network policy).

**Fixtures** (`scripts/make-fixtures.sh`): the 27 clips upstream publishes in `docs/samples` (LibriSpeech
CC-BY-4.0 with DEMAND noise CC-BY-4.0 and simulated rooms, Common Voice 17 CC0, VoxPopuli CC0, AMI CC-BY-4.0;
credits in `docs/index.html`), decoded from 48 kbps MP3 (already 16 kHz mono: no resampling, no downmix); digital
silence and seeded white noise (3 s each); one clip cut to 100, 319, 320, 321, 641 and 16,000 samples; and a 22.6 s
concatenation that must be refused by the 20 s bound. These are not robot recordings and not a held-out set.
Upstream's `host_hyp` strings are LM beam-4 decodes of the pre-MP3 audio (`eval/make_demo_page.py:128`), so they are
not a reference for greedy output here; the reference is this repository's own host build on these exact files.

## Programs

**`oido_cli`** (utterance mode). The upstream `tasr_cli` still builds and agrees with it, but exits 0 after skipped
files, times with `clock()`, accepts truncated WAV data and has no memory figures. `oido_cli` takes TNM1 models;
accepts only RIFF/WAVE PCM 16-bit mono 16 kHz, rejecting truncated chunks, odd lengths and anything above
`--max-seconds` (default 20); exits 1 on any failure; reports wall (`CLOCK_MONOTONIC`) and process CPU time as
`wall_rtf` and `cpu_rtf`; counts engine allocations through the `tasr_alloc` hook (keeping its zeroed, 16-byte
aligned contract) and fails a file that leaves any outstanding; prints peak RSS; writes raw CTC logits
(`--logits DIR`); flags a transcript near its buffer's end as possibly truncated; optionally decodes with the LM
(`--lm`, `--beam`, `--lm-weight`, `--token-bonus`, logging the values in use); writes stage times and kernel shapes
in profile builds (`--profile FILE`); and aborts with a message if an engine allocation fails mid-inference, since
`tasr_nemo_transcribe` does not check its working buffers.

**`oido_stream_replay`** (milestone 4). `tasr_cli` and `oido_cli` always run full context, even for
`oido_stream.tnm`. This program feeds each file through one persistent stream object (`tasr_nemo_stream_*`) in
several block schedules (320, 160, 1600 samples and seeded irregular blocks of 1–4000) and fails unless the final
text, the frame count and every logit agree across schedules. It reports partial-transcript changes, the audio time of
the first partial, per-chunk compute p50/p95/p99, finish time, and a modeled real-time timeline (audio arriving at
16 kHz, each block processed when it has arrived and the previous one is done): maximum backlog and end-of-audio to
final latency. The timeline is computed from measured compute, without sleeping; on an emulator it means nothing.

**`oido_service`** (milestone 5). One process, one model loaded once (path fixed at startup), one recognition stream,
one inference worker. A reader thread moves requests into a queue bounded to `--queue-seconds` of audio (default 4),
so reading never waits for inference; audio that does not fit is dropped, the drop is recorded at its place in the
queue, the utterance is marked `audio_discontinuity` and an `overrun` event gives the samples lost: audio is never
spliced silently. Requests are text lines (`audio N` + N int16 samples, `end`, `reset`, `status`, `quit`, at most
64 bytes, N ≤ 16000); anything else is a protocol error that closes the connection. Events are JSON lines: `ready`,
`partial` (provisional; streaming models only), `final`, `overrun`, `reset`, `status`, `error`, `shutdown`. A final
carries the text, endpoint (`end`, `max_duration`, `eof`), model id and SHA-256 (computed in-process), sample count,
discontinuity and truncation flags, `decoder: ctc_greedy`, `confidence: null` (greedy CTC gives no calibrated
probability), and measured timing: compute, finalization, endpoint-to-final and maximum queue wait. `status` adds
resident set and its peak from `/proc`. Utterances reaching `--max-seconds` (default 20) are finalized there.
Non-streaming models (`nemo8.tnm`) buffer the utterance and transcribe at the endpoint. Transport is stdin/stdout or
`--socket PATH` (mode 0600, refused if PATH exists and is not a socket, one client at a time, model kept loaded).
Audio and transcripts are never logged. JSON strings are fully escaped, with invalid UTF-8 replaced.

Not done, and needing the robot: connecting it to Jibo's processed audio. The endpoint, sample format, rate,
channel selection and permissions of Jibo's audio service are not known here, and the service must not take
exclusive ownership of the microphones or stack its own AGC/VAD on Jibo's processing without an A/B test. Endpointing
stays the client's decision (`end`). Partial events must not trigger actions; only finals should reach the controller.

## Engine changes

All changed upstream files carry dated modification notices. None changes arithmetic: after each change the host
run is bit-identical to the unmodified engine on all 35 fixtures, and the ARM runs match.

- **Loader header ranges** checked first: version 1; d even, 2–1024, divisible by heads (1–16), head dimension ≤ 64
  (fixed scratch arrays); ff 1–8192; odd kernel 1–63; 1–64 layers; 1–1024 channels; vocabulary 1–8192; 4 or 8 bits;
  known flag bits; stream chunk/left ≤ 4096.
- **Bounds-checked cursor**: no pointer is formed past the blob and no table (filterbank, depthwise weights, token
  strings) is read before its bytes are known present. The original code walked token strings and the filterbank
  before its single end-of-load check.
- **Exact size**: the layout must end exactly at `size` (all four shipped `.tnm` files do; the ESP32 firmware passes
  the exact size). Every dimension shapes the layout, so this refuses most wrong headers.
- **16-byte blob alignment** is required: weights sit at 16-byte boundaries relative to an aligned base, so a
  misaligned blob (an 8-byte-aligned 32-bit `malloc`) was silently misread.
- **Allocation failures** in `tasr_nemo_load` and `tasr_nemo_stream_new` free everything and return NULL (the stream
  constructor checked 7 of its 36 allocations).
- **NaN-safe table lookups**: a header that passes every check can still mislabel the data (`ff` 703 instead of 704
  consumes the same length), giving non-finite activations; the sigmoid and softmax-exponent lookups then indexed
  their tables with INT_MIN. NaN is now out of range; finite inputs take the same path as before.
- **`tasr_nemo_stream_truncated()`** reports text dropped by the stream's 2 KiB buffer or a short `maxlen`.
- **Profiler**: 64-bit nanosecond timestamps off the ESP32 (32 bits wrapped within 4.29 s) and
  `tasr_nemo_profile_reset()`.
- **`TASR_KERNEL_DISPATCH`** (opt-in): the portable kernels keep their code under `*_scalar` names so that
  `kernels_neon.c` can provide the public names. Builds without the macro are unchanged.

`tests/test_model_load.c` (ASan + UBSan, exact-size allocations) checks on every shipped model: it loads and stream
support matches flag bit 1; ~73,000 truncations are refused; each header word set to 0, 1, ±1, 0x7fffffff and
0xffffffff is refused, or loads and then transcribes and streams 0.5 s cleanly; a misaligned blob is refused; the Nth
allocation failing, for every N reached (21 in load, 41 in stream creation), returns NULL without leaks.

## Results

Environments: **host** = x86-64 Xeon 2.1 GHz, GCC 13.3, glibc 2.39; **qemu** = `qemu-arm` 8.2 running the ARMv7
binaries with the sysroot's glibc 2.21 (Jibo's version) unless stated. Emulator timings are not performance figures.

### Correctness chain

| Comparison | Result |
|---|---|
| ARMv7 scalar (qemu, glibc 2.39) vs host | logits bit-identical, 35/35 |
| ARMv7 scalar (qemu, glibc 2.21) vs host | 28/35 transcripts same; all differences from 1-ulp `logf` (below) |
| ARMv7 NEON vs ARMv7 scalar (qemu, glibc 2.21), utterance mode | logits bit-identical, 35/35 |
| ARMv7 NEON vs ARMv7 scalar (qemu, glibc 2.21), streaming | logits bit-identical, 35/35 |
| Streaming block schedules (320/160/1600/irregular): host, ARM scalar, ARM NEON | identical text, frames, logits, 35/35 each |
| NEON kernels vs C kernels (`test_kernels`, qemu) | 11,185 cases exact (all -128/127 extremes, tails, strides, every `tasr_qlin_range` branch); a deliberately broken NEON reduction is caught |
| Engine after milestone-2/3 changes vs before (host, and ARM scalar) | bit-identical, 35/35 |

**Libm.** With Jibo's glibc 2.21, the first stage to differ from the host is the log-mel output, whose only libm
call is `logf` (`TASR_DEBUG_SUMS`, `debug-sums-real_23.txt`); FFT twiddles and positional encodings agree.
glibc 2.21 and 2.39 differ by exactly 1 ulp on 155,536 of 8,388,606 `logf` inputs over the log-mel range (1.85%) and
on 202 of 327,681 `expf` inputs (`libm-fingerprints.txt`); the differences change dynamic int8 quantization
decisions and propagate. The seven changed transcripts are all on degraded audio (`course of peel` → `course of
peal`). A physical-Jibo run should therefore match the **glibc 2.21 emulated runs** bit for bit, if the robot's
`libm_fingerprint` prints the sysroot's hashes (`95cbba12` / `372829a9`); `robot-run.sh run` records it first.

### Profile (host, scalar, `TASR_PROFILE` + `TASR_KERNEL_STATS`; `profile-host-nemo8.tsv`)

The three int8 kernels take ~95% of inference time: projections with K = 176 via `tasr_gemm_s8_xr` (40%), the
front-end projections over long rows via `tasr_dot_rows_s8` (28%), the K = 704 feed-forward outputs (17%), attention
(10%). Everything else (features, layer norm, activations, quantization, convolution) is under 5% together. The NEON
kernels cover exactly these three functions; the int4 tile kernel (`nemo4.tnm` only) is not vectorized. The share on
a Cortex-A15 may differ; `robot-run.sh kernels` measures scalar vs NEON per hot shape there.

### Memory (host)

Model file 13.98 MB, read into one aligned buffer and used in place (13.89 MB of weights); 0.37 MB of engine state
after load; utterance working set 1.6 MB at 7–10 s; stream object 3.27 MB (fixed, whatever the utterance length;
no allocation while streaming); process peak RSS 16–22 MiB (CLI, replay, service). Well inside the proposed 100 MiB.

### Speed (host only; not a Jibo figure)

Utterance mode wall RTF 0.20–0.30 (scalar). Streaming (320-sample blocks, speech clips): median chunk compute
164–296 ms per 1.28 s chunk; end of audio to final 64–267 ms; first partial once the first chunk has run (1.3 s of
audio). Paced at real time through the service: endpoint-to-final ~170 ms on a 3.8 s clip.

### Recognition (host; `score-host.tsv`)

WER on the 27 published demo clips (518 words; MP3-decoded; not a held-out or robot set, so only a smoke test):

| Model, decoder, mode | WER | negatives (silence, noise) with text |
|---|---|---|
| `nemo8`, greedy, utterance | 7.9% | silence → `i` |
| `nemo8`, LM beam 4 (weight 0.3, bonus 0.5), utterance | 7.3% | silence → `i` |
| `oido_stream`, greedy, full context | 6.0% | none |
| `oido_stream`, greedy, streaming 32/128 | 11.4% | none |

The LM helps a little here, at a cost not yet measured cleanly; it is not enabled anywhere by default. Streaming
costs accuracy on these clips, as expected from its limited context; the service therefore also supports `nemo8` in
utterance mode, where the final comes after the endpoint.

### Segmenter A/B (host; simulated audio; `seg-ab/`)

Question: should upstream's segmenter (`tasr_seg.c`: energy VAD against a tracked noise floor, slow AGC towards
-20 dBFS, 0.2 s pre-roll, utterance closed after 0.8 s of non-speech) sit between Jibo's audio and the recognizer, and
if so with or without its AGC? `oido_seg_ab` runs, on continuous audio with labelled utterance spans:
**A** oracle spans, raw audio; **B** `tasr_seg` boundaries with its AGC'd audio (upstream's behaviour); **C** `tasr_seg`
boundaries with raw audio (endpointer only); **B25/C25** the same with a 0.5 s hang.

**This is not the robot A/B.** No recording of Jibo's processed audio exists yet. The streams here
(`scripts/make-ab-streams.py`) are the 27 demo clips inside continuous seeded pink noise (high-passed at 80 Hz) at
-60 and -40 dBFS, at speech levels 0 and -20 dB, plus noise-only negatives and clip pairs 0.5 s and 1.2 s apart:
126 streams, 132 utterances. They test the segmenter's logic on stationary noise; they say nothing about Jibo's
microphones, its echo cancellation, its own gain stage, motor noise or the robot's own speech.

| | `nemo8` WER | `oido_stream` (streaming) WER |
|---|---|---|
| A oracle | 26.6% (11.0%¹) | 30.9% |
| B tasr_seg + AGC | 29.0% (12.3%¹) | 32.2% |
| **C tasr_seg, no AGC** | **28.1% (11.0%¹)** | **31.2%** |
| B25 | 29.2% | 32.5% |
| C25 | 28.4% | 31.3% |

¹ excluding the -20 dB speech in -40 dBFS noise condition, where even the oracle fails (90% WER) and the segmenter
misses 24 of 27 utterances; that condition is beyond the recognizer either way.

Findings:
- **The AGC hurts, the segmentation does not.** Per stream (excluding that condition), dropping the AGC (C vs B) gave
  fewer word errors on 18 streams and more on 8 with `nemo8` (188 vs 209 errors), and 27 vs 15 with `oido_stream`
  (269 vs 292); with the 0.5 s hang, 18 vs 6 and 29 vs 15 (sign test p = 0.02–0.09 individually, all four in the
  same direction). The AGC starts at +12 dB and drives 46,000 samples to full scale (clipping) across the speech
  streams. `nemo8` normalizes each utterance's level away anyway, and `oido_stream` did not benefit either.
  Segmentation without AGC matched the oracle spans (`nemo8`: 188 vs 188 errors; 20 streams better, 18 worse).
- **Hang 0.5 s vs 0.8 s**: same accuracy (C25 vs C: 191 vs 188 errors, 10 vs 9 streams), utterance end detected
  ~0.55 s sooner on average, but more utterances split at internal pauses (30 vs 28 streams with the wrong segment
  count), while two utterances 0.5 s apart are separated only with the shorter hang.
- **No false triggers** on noise-only streams (stationary noise; real rooms are not).
- **Misses**: outside the extreme condition, one quiet AMI clip (`real_26`) at -20 dB speech or -40 dBFS noise.

Recommendation, pending robot data: if Jibo's audio service gives no usable utterance boundary, use `tasr_seg` as an
endpointer only (arm C: its decisions, raw audio to the recognizer), not its AGC; start with the 0.8 s hang and tune
it on robot recordings. Nothing is wired into `oido_service` yet; it still takes `end` from its client.

**The robot A/B** uses the same tools, and the recognition can run on any host because the question is the
segmenter, not the CPU: record continuous processed audio from Jibo's audio path (with the owner's agreement and
the speakers' consent, in the scenarios that matter: quiet room, TV, motors moving, Jibo speaking, far talker),
write a labels file in the same format (`stream condition start_sample end_sample reference`, start = end = -1 for
no-speech recordings), and run `scripts/run-seg-ab.sh LABEL build/host/oido_seg_ab models/nemo8.tnm labels.tsv
recordings/` then `scripts/score-seg-ab.py --labels labels.tsv build/runs/LABEL/ab.tsv`. Compare against Jibo's own
boundary signal, if its audio service has one, as a further arm.

## Open items

1. **Robot runs** (owner, within the agreed scope; dry-run tested locally): `robot-run.sh deploy`, `status`,
   `kernels` (exactness and NEON speed on the Cortex-A15), `run neon` and `run neon-forced-scalar` (compare both
   with `build/runs/qemu-armv7-neon` via `compare-runs.py`; first real RTF, CPU time, peak RSS and thermals),
   `stream`, `service`, `lm`, then `cleanup`. Then the same with the Decider active (G6).
2. **Owner's toolchain**: rebuild with `JIBO_CC=<jibo-armcc>` (GCC 4.8.4), re-run the ABI check and the qemu
   comparisons; GCC 4.8's NEON intrinsics and code generation differ from 13.3.
3. **Live audio**: recover Jibo's processed-audio endpoint, format and channel; write the capture client for
   `oido_service`; run the segmenter A/B on robot recordings (tools ready; simulated result: endpointer yes, AGC
   no); measure with playback (self-speech) and Jibo's echo cancellation.
4. **Recognition quality (G5)**: a held-out 200–300 utterance robot-relevant set (numbers, names, negation, accents,
   noise, self-speech), recorded with consent through the real microphone path.
5. **Model SHA-256 against Hugging Face**: sizes match; hashes and revisions unread (network policy).
6. `tasr_nemo_transcribe`'s working allocations are unchecked (the port's allocator aborts with a message instead);
   the LM loader (`tasr_lm_load`) is not hardened like the model loader; use only the pinned LM. The engine is
   non-reentrant (static tables, profiler state): one inference thread per process, as the service does.
7. Utterance mode can transcribe featureless input: `nemo8` normalizes each mel band by the utterance's own mean
   and deviation, so digital silence becomes all-zero features, and some lengths come out as text (`i` at 2.5–3.25 s,
   `okay` at 1.5 s; 5 of 37 lengths tested). `oido_stream` (fixed normalization) and any input with real variance
   (dither, DC, speech) do not. Only speech-bearing segments should reach `nemo8`: an endpointer (above) helps.
8. Libm sensitivity: one-ulp `logf` differences change 20% of transcripts on degraded audio. A libm-independent
   log-mel would make results reproducible across machines but changes the numerical contract; not done.
9. GPU (milestone 7): only if the robot's measurements show the NEON CPU path misses the budget.
