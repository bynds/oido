// oido_cli: utterance-mode transcription of 16 kHz mono PCM16 WAV files with the Oído NeMo engine, for the Jibo port.
//
// Differs from esp32/host/tasr_cli.c in what a port baseline needs:
//   - NeMo (TNM1) models only; greedy CTC only (no language model in the baseline)
//   - every failure (unreadable or malformed WAV, wrong format, oversize input, engine failure, transcript that
//     filled the buffer) makes the process exit nonzero, after the remaining files have been tried
//   - wall time (CLOCK_MONOTONIC) and process CPU time (CLOCK_PROCESS_CPUTIME_ID), both as 64-bit nanoseconds
//   - engine allocation accounting (current/peak bytes through the tasr_alloc hook) and the process peak RSS
//   - optional raw logits per file (--logits DIR) for numerical comparison between builds
//
// usage: oido_cli [--max-seconds S] [--logits DIR] model.tnm file.wav [file.wav ...]
// stdout: one tab-separated record per file (header first); stderr: load and memory figures.
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include "tasr_nemo.h"
#include "tinyasr.h"

#define TEXT_MAX 16384
#define TEXT_MARGIN 256   // longer than any token string

static uint64_t now_ns(clockid_t id)
{
    struct timespec t;
    clock_gettime(id, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// ---- allocation accounting. Keeps the default allocator's contract: zeroed, 16-byte aligned.
static size_t alloc_cur, alloc_peak, alloc_count;
static void *counting_alloc(size_t n, int kind)
{
    (void)kind;
    void *p = NULL;
    if (n > SIZE_MAX - 16 || posix_memalign(&p, 16, n + 16)) return NULL;
    memset(p, 0, n + 16);
    memcpy(p, &n, sizeof(n));
    alloc_cur += n;
    alloc_count++;
    if (alloc_cur > alloc_peak) alloc_peak = alloc_cur;
    return (uint8_t *)p + 16;
}
static void counting_free(void *q)
{
    if (!q) return;
    uint8_t *p = (uint8_t *)q - 16;
    size_t n;
    memcpy(&n, p, sizeof(n));
    alloc_cur -= n;
    free(p);
}

// Whole file into a 16-byte-aligned buffer; fails on any short read or a file above max_bytes.
static uint8_t *read_file(const char *path, size_t max_bytes, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "%s: %s\n", path, strerror(errno)); return NULL; }
    uint8_t *buf = NULL;
    long sz = -1;
    if (fseek(f, 0, SEEK_END) == 0) sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fprintf(stderr, "%s: cannot size file\n", path); goto fail; }
    if ((unsigned long)sz > max_bytes) { fprintf(stderr, "%s: %ld bytes exceeds limit %zu\n", path, sz, max_bytes); goto fail; }
    if (posix_memalign((void **)&buf, 16, (size_t)sz + 16)) { buf = NULL; fprintf(stderr, "%s: out of memory\n", path); goto fail; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "%s: short read\n", path); goto fail; }
    fclose(f);
    *size = (size_t)sz;
    return buf;
fail:
    free(buf);
    fclose(f);
    return NULL;
}

static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

// RIFF/WAVE, PCM (format 1) 16-bit mono 16 kHz, exactly one well-formed data chunk after fmt. Returns samples.
static int16_t *read_wav(const char *path, long max_samples, int *ns)
{
    size_t n;
    uint8_t *b = read_file(path, (size_t)max_samples * 2 + 4096, &n);
    if (!b) return NULL;
    int16_t *pcm = NULL;
    int have_fmt = 0;
    if (n < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { fprintf(stderr, "%s: not RIFF/WAVE\n", path); goto out; }
    for (size_t p = 12; p + 8 <= n;) {
        const uint32_t len = le32(b + p + 4);
        if (len > n - p - 8) { fprintf(stderr, "%s: truncated chunk\n", path); goto out; }
        if (!memcmp(b + p, "fmt ", 4)) {
            if (len < 16) { fprintf(stderr, "%s: short fmt chunk\n", path); goto out; }
            const uint16_t fmt = le16(b + p + 8), ch = le16(b + p + 10), bits = le16(b + p + 22);
            const uint32_t rate = le32(b + p + 12);
            if (fmt != 1 || ch != 1 || rate != 16000 || bits != 16) {
                fprintf(stderr, "%s: need PCM 16 kHz mono 16-bit (got format %u, %u ch, %u Hz, %u bit)\n", path, fmt, ch,
                        rate, bits);
                goto out;
            }
            have_fmt = 1;
        } else if (!memcmp(b + p, "data", 4)) {
            if (!have_fmt) { fprintf(stderr, "%s: data before fmt\n", path); goto out; }
            if (len & 1) { fprintf(stderr, "%s: odd data length\n", path); goto out; }
            if (len / 2 > (uint32_t)max_samples) { fprintf(stderr, "%s: %u samples exceeds limit %ld\n", path, len / 2, max_samples); goto out; }
            pcm = malloc(len ? len : 2);
            if (!pcm) { fprintf(stderr, "%s: out of memory\n", path); goto out; }
            for (uint32_t i = 0; i < len / 2; i++) pcm[i] = (int16_t)le16(b + p + 8 + 2 * i);
            *ns = (int)(len / 2);
            goto out;
        }
        p += 8 + (size_t)len + (len & 1);
    }
    fprintf(stderr, "%s: no data chunk\n", path);
out:
    free(b);
    return pcm;
}

static const char *base_name(const char *p)
{
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

int main(int argc, char **argv)
{
    double max_seconds = 20.0;
    const char *logit_dir = NULL;
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--max-seconds") && i + 1 < argc) max_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logit_dir = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (argc - i < 2 || !(max_seconds > 0 && max_seconds <= 600)) {
        fprintf(stderr, "usage: %s [--max-seconds S (default 20, max 600)] [--logits DIR] model.tnm file.wav ...\n", argv[0]);
        return 2;
    }
    const long max_samples = (long)(max_seconds * 16000);
    tasr_alloc = counting_alloc;
    tasr_free = counting_free;

    const uint64_t w0 = now_ns(CLOCK_MONOTONIC);
    size_t msz;
    uint8_t *mb = read_file(argv[i], (size_t)256 << 20, &msz);
    if (!mb) return 1;
    if (msz < 4 || memcmp(mb, "TNM1", 4)) { fprintf(stderr, "%s: not a TNM1 model\n", argv[i]); return 1; }
    const uint64_t w1 = now_ns(CLOCK_MONOTONIC);
    tasr_nemo_t *nm = tasr_nemo_load(mb, msz);
    const uint64_t w2 = now_ns(CLOCK_MONOTONIC);
    if (!nm) { fprintf(stderr, "%s: model load failed\n", argv[i]); return 1; }
    fprintf(stderr, "model %s: %zu bytes, read %.1f ms, load %.1f ms, weights %zu bytes, engine allocations %zu bytes\n",
            argv[i], msz, (w1 - w0) / 1e6, (w2 - w1) / 1e6, tasr_nemo_weight_bytes(nm), alloc_cur);
    const size_t alloc_after_load = alloc_cur;

    static char text[TEXT_MAX];
    int failures = 0;
    printf("file\tsamples\taudio_s\tframes\twall_ms\tcpu_ms\twall_rtf\tcpu_rtf\ttruncated\ttext\n");
    for (int f = i + 1; f < argc; f++) {
        int ns = 0;
        int16_t *pcm = read_wav(argv[f], max_samples, &ns);
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
        alloc_peak = alloc_cur;
        const uint64_t c0 = now_ns(CLOCK_PROCESS_CPUTIME_ID), t0 = now_ns(CLOCK_MONOTONIC);
        const int frames = tasr_nemo_transcribe(nm, pcm, ns, NULL, text, TEXT_MAX, logits, max_frames, &nf);
        const uint64_t t1 = now_ns(CLOCK_MONOTONIC), c1 = now_ns(CLOCK_PROCESS_CPUTIME_ID);
        const double audio_s = ns / 16000.0, wall = (t1 - t0) / 1e9, cpu = (c1 - c0) / 1e9;
        // The engine silently drops a token that would not fit, so treat a nearly full buffer as possibly truncated.
        const int truncated = strlen(text) >= TEXT_MAX - TEXT_MARGIN;
        if (frames < 0 || truncated) failures++;
        if (alloc_cur != alloc_after_load) {
            fprintf(stderr, "%s: engine allocations not released (%zu bytes outstanding)\n", argv[f], alloc_cur - alloc_after_load);
            failures++;
        }
        printf("%s\t%d\t%.3f\t%d\t%.1f\t%.1f\t%.4f\t%.4f\t%d\t%s\n", base_name(argv[f]), ns, audio_s, frames, wall * 1e3,
               cpu * 1e3, audio_s > 0 ? wall / audio_s : 0, audio_s > 0 ? cpu / audio_s : 0, truncated, text);
        fflush(stdout);
        fprintf(stderr, "%s: engine peak working allocations %zu bytes\n", base_name(argv[f]), alloc_peak - alloc_after_load);
        if (logits) {
            const int keep = nf < max_frames ? nf : max_frames;
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s.f32", logit_dir, base_name(argv[f]));
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
    tasr_nemo_free(nm);
    free(mb);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "peak RSS %ld KiB; engine allocations %zu calls, %zu bytes outstanding after free; %d failure(s)\n",
            ru.ru_maxrss, alloc_count, alloc_cur, failures);
    return failures ? 1 : 0;
}
