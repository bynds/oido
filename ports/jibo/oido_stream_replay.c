// oido_stream_replay: the NeMo streaming path (tasr_nemo_stream_*), replayed from WAV files.
//
// tasr_cli and oido_cli always call tasr_nemo_transcribe, i.e. full context even for oido_stream.tnm. This program
// feeds the same audio through one persistent stream object in blocks, as live capture would, and checks that the
// result does not depend on how the audio was split:
//
//   - every file is replayed with each block schedule (default: 320, 160, 1600 samples and seeded irregular blocks of
//     1..4000 samples); the final transcript, encoder frame count and every logit must be identical across schedules
//   - the partial transcript is read after every block (tasr_nemo_stream_text); reported: how many times it changed
//     and the audio time at which it first became non-empty
//   - per block: compute wall time; blocks during which the encoder ran a chunk are "chunk blocks", whose times give
//     p50/p95/p99 chunk compute; finish (the last partial chunk) is timed separately
//   - a modeled real-time timeline (no sleeping): audio arrives at 16 kHz in the block schedule, each block is
//     processed when it has arrived and the previous one is done; reported: maximum backlog (how far processing fell
//     behind arrival) and end-of-audio to final transcript. On the robot these model the latency of a live feed with
//     this machine's measured compute; on an emulator they mean nothing
//   - the stream object is created once per model and reset between replays; its creation time and allocations are
//     reported separately from steady-state feeding
//
// usage: oido_stream_replay [--schedules 320,160,1600,irregular] [--seed N] [--chunk C] [--left L] [--logits DIR]
//                           [--max-seconds S] model.tnm file.wav ...
//   --chunk/--left: encoder frames per chunk and left context (0 = the model's defaults, which is the evaluated
//   configuration; anything else is an accuracy/latency experiment). --logits DIR writes the first schedule's logits.
// stdout: one tab-separated record per file and schedule, then one "invariant" record per file; exit 1 on any
// failure or any schedule disagreement.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "port_util.h"
#include "tasr_nemo.h"

#define TEXT_MAX 4096
#define MAX_SCHED 8
#define IRREGULAR (-1)

static uint32_t rng;
static uint32_t xs32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static double pct(const uint64_t *v, int n, double p)  // nearest-rank percentile of a sorted array, in ms
{
    if (!n) return 0;
    int k = (int)(p / 100.0 * n + 0.999999) - 1;
    if (k < 0) k = 0;
    if (k >= n) k = n - 1;
    return v[k] / 1e6;
}

typedef struct {
    int frames, truncated, partials, n_chunks;
    double first_partial_audio_s, compute_ms, cpu_ms, finish_ms, backlog_max_ms, final_latency_ms;
    double chunk_p50, chunk_p95, chunk_p99, chunk_max;
    char text[TEXT_MAX];
} replay_t;

// One replay of pcm[0..ns) through s with block size `block` (IRREGULAR: seeded random sizes).
static void replay(tasr_nemo_stream_t *s, const int16_t *pcm, int ns, int block, uint32_t seed, float *sink,
                   int sink_max, replay_t *r)
{
    memset(r, 0, sizeof(*r));
    r->first_partial_audio_s = -1;
    tasr_nemo_stream_reset(s);
    tasr_nemo_stream_set_sink(s, sink, sink_max);
    rng = seed ? seed : 1;
    const int max_blocks = ns + 1;
    uint64_t *chunk_ns = malloc(sizeof(uint64_t) * (size_t)(max_blocks > 0 ? max_blocks : 1));
    char partial[TEXT_MAX], last[TEXT_MAX] = "";
    int prev_frames = 0;
    double proc_done = 0;   // modeled timeline, seconds since audio start
    const uint64_t c0 = pu_now_ns(CLOCK_PROCESS_CPUTIME_ID);
    for (int o = 0; o < ns;) {
        int k = block == IRREGULAR ? 1 + (int)(xs32() % 4000) : block;
        if (k > ns - o) k = ns - o;
        const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
        const int frames = tasr_nemo_stream_feed(s, pcm + o, k);
        tasr_nemo_stream_text(s, partial, sizeof(partial));
        const uint64_t dt = pu_now_ns(CLOCK_MONOTONIC) - t0;
        o += k;
        r->compute_ms += dt / 1e6;
        const double arrive = o / 16000.0, start = arrive > proc_done ? arrive : proc_done;
        proc_done = start + dt / 1e9;
        if (proc_done - arrive > r->backlog_max_ms / 1e3) r->backlog_max_ms = (proc_done - arrive) * 1e3;
        if (frames != prev_frames) { chunk_ns[r->n_chunks++] = dt; prev_frames = frames; }
        if (partial[0] && r->first_partial_audio_s < 0) r->first_partial_audio_s = arrive;
        if (strcmp(partial, last)) { r->partials++; memcpy(last, partial, sizeof(last)); }
    }
    const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
    r->frames = tasr_nemo_stream_finish(s, r->text, sizeof(r->text));
    const uint64_t dt = pu_now_ns(CLOCK_MONOTONIC) - t0;
    r->cpu_ms = (pu_now_ns(CLOCK_PROCESS_CPUTIME_ID) - c0) / 1e6;
    r->finish_ms = dt / 1e6;
    r->compute_ms += dt / 1e6;
    const double end = ns / 16000.0, fin_start = end > proc_done ? end : proc_done;
    r->final_latency_ms = (fin_start + dt / 1e9 - end) * 1e3;
    r->truncated = tasr_nemo_stream_truncated(s);
    qsort(chunk_ns, r->n_chunks, sizeof(uint64_t), cmp_u64);
    r->chunk_p50 = pct(chunk_ns, r->n_chunks, 50);
    r->chunk_p95 = pct(chunk_ns, r->n_chunks, 95);
    r->chunk_p99 = pct(chunk_ns, r->n_chunks, 99);
    r->chunk_max = r->n_chunks ? chunk_ns[r->n_chunks - 1] / 1e6 : 0;
    free(chunk_ns);
}

int main(int argc, char **argv)
{
    int sched[MAX_SCHED] = {320, 160, 1600, IRREGULAR}, n_sched = 4, chunk = 0, left = 0;
    uint32_t seed = 12345;
    double max_seconds = 20.0;
    const char *logit_dir = NULL;
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--schedules") && i + 1 < argc) {
            n_sched = 0;
            for (char *tok = strtok(argv[++i], ","); tok && n_sched < MAX_SCHED; tok = strtok(NULL, ",")) {
                const int v = !strcmp(tok, "irregular") ? IRREGULAR : atoi(tok);
                if (v == 0 || v < IRREGULAR) { fprintf(stderr, "bad schedule %s\n", tok); return 2; }
                sched[n_sched++] = v;
            }
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--left") && i + 1 < argc) left = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) logit_dir = argv[++i];
        else if (!strcmp(argv[i], "--max-seconds") && i + 1 < argc) max_seconds = atof(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (argc - i < 2 || !n_sched || !(max_seconds > 0 && max_seconds <= 600) || chunk < 0 || left < 0) {
        fprintf(stderr, "usage: %s [--schedules 320,160,1600,irregular] [--seed N] [--chunk C] [--left L] [--logits DIR] "
                        "[--max-seconds S] model.tnm file.wav ...\n", argv[0]);
        return 2;
    }
    pu_alloc_install();
    if (pu_kernels_init()) return 2;
    const long max_samples = (long)(max_seconds * 16000);
    size_t msz;
    uint8_t *mb = pu_read_file(argv[i], (size_t)256 << 20, &msz);
    if (!mb) return 1;
    tasr_nemo_t *m = tasr_nemo_load(mb, msz);
    if (!m) { fprintf(stderr, "%s: model load failed\n", argv[i]); return 1; }
    if (!tasr_nemo_stream_supported(m)) { fprintf(stderr, "%s: not a streaming model\n", argv[i]); return 1; }
    const size_t a0 = pu_alloc_cur;
    const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
    tasr_nemo_stream_t *s = tasr_nemo_stream_new(m, chunk, left, NULL);
    const uint64_t t1 = pu_now_ns(CLOCK_MONOTONIC);
    if (!s) { fprintf(stderr, "stream creation failed\n"); return 1; }
    fprintf(stderr, "stream: chunk %d, left %d (0 = model default); created in %.1f ms, %zu bytes of state\n", chunk, left,
            (t1 - t0) / 1e6, pu_alloc_cur - a0);
    const size_t a_stream = pu_alloc_cur;

    int failures = 0;
    printf("file\tschedule\tsamples\tframes\tpartials\tfirst_partial_audio_s\tchunks\tchunk_p50_ms\tchunk_p95_ms"
           "\tchunk_p99_ms\tchunk_max_ms\tfinish_ms\tcompute_ms\tcpu_ms\twall_rtf\tbacklog_max_ms\tend_to_final_ms"
           "\ttruncated\ttext\n");
    static replay_t r[MAX_SCHED];
    for (int f = i + 1; f < argc; f++) {
        int ns = 0;
        int16_t *pcm = pu_read_wav(argv[f], max_samples, &ns);
        if (!pcm) { failures++; continue; }
        const int sink_max = ns / 640 + 8;
        float *sinks[MAX_SCHED];
        int ok = 1;
        for (int k = 0; k < n_sched; k++) {
            sinks[k] = calloc((size_t)sink_max * 1025, sizeof(float));
            if (!sinks[k]) { fprintf(stderr, "out of memory\n"); return 1; }
            replay(s, pcm, ns, sched[k], seed, sinks[k], sink_max, &r[k]);
            char name[16];
            if (sched[k] == IRREGULAR) snprintf(name, sizeof(name), "irregular");
            else snprintf(name, sizeof(name), "%d", sched[k]);
            printf("%s\t%s\t%d\t%d\t%d\t%.2f\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.1f\t%.1f\t%.4f\t%.1f\t%.1f\t%d\t%s\n",
                   pu_base_name(argv[f]), name, ns, r[k].frames, r[k].partials, r[k].first_partial_audio_s, r[k].n_chunks,
                   r[k].chunk_p50, r[k].chunk_p95, r[k].chunk_p99, r[k].chunk_max, r[k].finish_ms, r[k].compute_ms,
                   r[k].cpu_ms, ns ? r[k].compute_ms / 1e3 / (ns / 16000.0) : 0, r[k].backlog_max_ms,
                   r[k].final_latency_ms, r[k].truncated, r[k].text);
            if (r[k].truncated) failures++;
            if (pu_alloc_cur != a_stream) {
                fprintf(stderr, "%s: allocations changed while streaming (%zu -> %zu bytes)\n", argv[f], a_stream, pu_alloc_cur);
                failures++;
            }
            if (k && (strcmp(r[k].text, r[0].text) || r[k].frames != r[0].frames ||
                      memcmp(sinks[k], sinks[0], sizeof(float) * 1025 * (size_t)(r[0].frames < sink_max ? r[0].frames : sink_max))))
                ok = 0;
        }
        printf("%s\tinvariant\t%d\t%d\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%s\n", pu_base_name(argv[f]), ns, r[0].frames,
               ok ? "same text, frames and logits for every schedule" : "SCHEDULES DISAGREE");
        if (!ok) failures++;
        if (logit_dir) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s.f32", logit_dir, pu_base_name(argv[f]));
            FILE *o = fopen(path, "wb");
            const int keep = r[0].frames < sink_max ? r[0].frames : sink_max;
            if (!o || fwrite(sinks[0], sizeof(float) * 1025, (size_t)keep, o) != (size_t)keep) {
                fprintf(stderr, "%s: cannot write logits\n", path);
                failures++;
            }
            if (o) fclose(o);
        }
        for (int k = 0; k < n_sched; k++) free(sinks[k]);
        fflush(stdout);
        free(pcm);
    }
    tasr_nemo_stream_free(s);
    tasr_nemo_free(m);
    free(mb);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "peak RSS %ld KiB; %zu bytes outstanding after free; %d failure(s)\n", ru.ru_maxrss, pu_alloc_cur, failures);
    return failures ? 1 : 0;
}
