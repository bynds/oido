// oido_cli: utterance-mode transcription of 16 kHz mono PCM16 WAV files with the Oído NeMo engine, for the Jibo port.
//
// Differs from esp32/host/tasr_cli.c in what a port baseline needs:
//   - NeMo (TNM1) models only; greedy CTC by default, or CTC prefix beam search with the GRU language model
//     (--lm FILE [--beam N] [--lm-weight W] [--token-bonus B]; unset weights come from the LM file, else upstream's
//     defaults 0.3 and 0.5, as in tasr_cli); the settings in use are printed
//   - every failure (unreadable or malformed WAV, wrong format, oversize input, engine failure, transcript that
//     filled the buffer) makes the process exit nonzero, after the remaining files have been tried
//   - wall time (CLOCK_MONOTONIC) and process CPU time (CLOCK_PROCESS_CPUTIME_ID), both as 64-bit nanoseconds
//   - engine allocation accounting (current/peak bytes through the tasr_alloc hook) and the process peak RSS
//   - optional raw logits per file (--logits DIR) for numerical comparison between builds
//   - in builds with the kernel dispatch layer (kernels_neon.c): OIDO_KERNELS=scalar forces the C kernels, and the
//     backend in use is printed; --profile FILE writes per-stage times (TASR_PROFILE builds) and kernel shapes
//     (TASR_KERNEL_STATS builds) summed over all files
//
// usage: oido_cli [--max-seconds S] [--logits DIR] [--profile FILE] [--lm FILE [--beam N] [--lm-weight W]
//                 [--token-bonus B]] model.tnm file.wav [file.wav ...]
// stdout: one tab-separated record per file (header first); stderr: load and memory figures.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include "port_util.h"
#include "tasr_nemo.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"
#ifdef TASR_KERNEL_DISPATCH
#include "kernels_dispatch.h"
#endif

#define TEXT_MAX 16384
#define TEXT_MARGIN 256   // longer than any token string

int main(int argc, char **argv)
{
    double max_seconds = 20.0;
    const char *logit_dir = NULL, *profile_path = NULL, *lm_path = NULL;
    int beam = 4;
    float lm_weight = -1.f, token_bonus = -1.f;
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--max-seconds") && i + 1 < argc) max_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logit_dir = argv[++i];
        else if (!strcmp(argv[i], "--profile") && i + 1 < argc) profile_path = argv[++i];
        else if (!strcmp(argv[i], "--lm") && i + 1 < argc) lm_path = argv[++i];
        else if (!strcmp(argv[i], "--beam") && i + 1 < argc) beam = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lm-weight") && i + 1 < argc) lm_weight = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--token-bonus") && i + 1 < argc) token_bonus = (float)atof(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (argc - i < 2 || !(max_seconds > 0 && max_seconds <= 600) || beam < 1 || beam > 64) {
        fprintf(stderr, "usage: %s [--max-seconds S (default 20, max 600)] [--logits DIR] [--profile FILE] model.tnm file.wav ...\n", argv[0]);
        return 2;
    }
    const long max_samples = (long)(max_seconds * 16000);
    pu_alloc_install();
    if (pu_kernels_init()) return 2;

    const uint64_t w0 = pu_now_ns(CLOCK_MONOTONIC);
    size_t msz;
    uint8_t *mb = pu_read_file(argv[i], (size_t)256 << 20, &msz);
    if (!mb) return 1;
    if (msz < 4 || memcmp(mb, "TNM1", 4)) { fprintf(stderr, "%s: not a TNM1 model\n", argv[i]); return 1; }
    const uint64_t w1 = pu_now_ns(CLOCK_MONOTONIC);
    tasr_nemo_t *nm = tasr_nemo_load(mb, msz);
    const uint64_t w2 = pu_now_ns(CLOCK_MONOTONIC);
    if (!nm) { fprintf(stderr, "%s: model load failed\n", argv[i]); return 1; }
    fprintf(stderr, "model %s: %zu bytes, read %.1f ms, load %.1f ms, weights %zu bytes, engine allocations %zu bytes\n",
            argv[i], msz, (w1 - w0) / 1e6, (w2 - w1) / 1e6, tasr_nemo_weight_bytes(nm), pu_alloc_cur);
    tasr_lm_t *lm = NULL;
    tasr_decoder_t *dec = NULL;
    uint8_t *lb = NULL;
    if (lm_path) {
        size_t lsz;
        lb = pu_read_file(lm_path, (size_t)64 << 20, &lsz);
        lm = lb ? tasr_lm_load(lb, lsz) : NULL;
        if (!lm) { fprintf(stderr, "%s: LM load failed\n", lm_path); return 1; }
        const float w = lm_weight >= 0 ? lm_weight : tasr_lm_weight(lm, 0.3f);
        const float b = token_bonus >= 0 ? token_bonus : tasr_lm_bonus(lm, 0.5f);
        dec = tasr_decoder_create(lm, 1025, beam, 6, w, b);
        if (!dec) { fprintf(stderr, "decoder creation failed\n"); return 1; }
        fprintf(stderr, "decoder: ctc_prefix_beam beam %d topk 6 lm_weight %.3f%s token_bonus %.3f%s lm %s\n", beam, w,
                lm_weight >= 0 ? " (set)" : " (LM file/default)", b, token_bonus >= 0 ? " (set)" : " (LM file/default)",
                lm_path);
    } else {
        fprintf(stderr, "decoder: ctc_greedy\n");
    }
    const size_t alloc_after_load = pu_alloc_cur;

    static char text[TEXT_MAX];
    int failures = 0;
    printf("file\tsamples\taudio_s\tframes\twall_ms\tcpu_ms\twall_rtf\tcpu_rtf\ttruncated\ttext\n");
    for (int f = i + 1; f < argc; f++) {
        int ns = 0;
        int16_t *pcm = pu_read_wav(argv[f], max_samples, &ns);
        if (!pcm) { failures++; continue; }
        const int max_frames = ns / 640 + 4;   // 10 ms mel hop, 4x subsampling, plus padding
        float *logits = NULL;
        if (logit_dir && !(logits = malloc(sizeof(float) * 1025 * (size_t)max_frames))) {
            fprintf(stderr, "%s: out of memory for logits\n", argv[f]);
            failures++;
            free(pcm);
            continue;
        }
        int nf = 0;
        text[0] = 0;
        pu_alloc_peak = pu_alloc_cur;
        const uint64_t c0 = pu_now_ns(CLOCK_PROCESS_CPUTIME_ID), t0 = pu_now_ns(CLOCK_MONOTONIC);
        pu_alloc_fatal = 1;
        const int frames = tasr_nemo_transcribe(nm, pcm, ns, dec, text, TEXT_MAX, logits, max_frames, &nf);
        pu_alloc_fatal = 0;
        const uint64_t t1 = pu_now_ns(CLOCK_MONOTONIC), c1 = pu_now_ns(CLOCK_PROCESS_CPUTIME_ID);
        const double audio_s = ns / 16000.0, wall = (t1 - t0) / 1e9, cpu = (c1 - c0) / 1e9;
        // The engine silently drops a token that would not fit, so treat a nearly full buffer as possibly truncated.
        const int truncated = strlen(text) >= TEXT_MAX - TEXT_MARGIN;
        if (frames < 0 || truncated) failures++;
        if (pu_alloc_cur != alloc_after_load) {
            fprintf(stderr, "%s: engine allocations not released (%zu bytes outstanding)\n", argv[f], pu_alloc_cur - alloc_after_load);
            failures++;
        }
        printf("%s\t%d\t%.3f\t%d\t%.1f\t%.1f\t%.4f\t%.4f\t%d\t%s\n", pu_base_name(argv[f]), ns, audio_s, frames, wall * 1e3,
               cpu * 1e3, audio_s > 0 ? wall / audio_s : 0, audio_s > 0 ? cpu / audio_s : 0, truncated, text);
        fflush(stdout);
        fprintf(stderr, "%s: engine peak working allocations %zu bytes\n", pu_base_name(argv[f]), pu_alloc_peak - alloc_after_load);
        if (logits) {
            const int keep = nf < max_frames ? nf : max_frames;
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s.f32", logit_dir, pu_base_name(argv[f]));
            FILE *o = fopen(path, "wb");
            if (!o || fwrite(logits, sizeof(float) * 1025, (size_t)keep, o) != (size_t)keep) {
                fprintf(stderr, "%s: cannot write logits\n", path);
                failures++;
            }
            if (o) fclose(o);
            if (nf != frames) fprintf(stderr, "%s: logit frames %d != encoder frames %d\n", argv[f], nf, frames);
            free(logits);
        }
        free(pcm);
    }
    if (profile_path) {
        FILE *pf = fopen(profile_path, "w");
        if (!pf) { fprintf(stderr, "%s: cannot write profile\n", profile_path); failures++; }
        else {
            fprintf(pf, "stage\tns\n");
            for (int k = 0; tasr_nemo_profile_name(k); k++)
                fprintf(pf, "%s\t%llu\n", tasr_nemo_profile_name(k), (unsigned long long)tasr_nemo_profile_value(k));
            fprintf(pf, "\n");
#ifdef TASR_KERNEL_DISPATCH
            tasr_kernel_stats_dump(pf);
#endif
            fclose(pf);
        }
    }
    tasr_decoder_free(dec);
    tasr_lm_free(lm);
    free(lb);
    tasr_nemo_free(nm);
    free(mb);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "peak RSS %ld KiB; engine allocations %zu calls, %zu bytes outstanding after free; %d failure(s)\n",
            ru.ru_maxrss, pu_alloc_calls, pu_alloc_cur, failures);
    return failures ? 1 : 0;
}
