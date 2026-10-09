// bench_engine: fixed Oído workloads with per-stage costs, for perfvm (bench.sh). Build with -DTASR_PROFILE.
//
// Workloads (one JSON line each):
//   utterance     nemo8.tnm, tasr_nemo_transcribe of one clip (full context; the port's utterance mode)
//   stream        oido_stream.tnm, the same clip fed in 320-sample blocks through tasr_nemo_stream_feed, then finish
//                 (the live mode; the stream object is created beforehand and not counted)
//   stream_setup  oido_stream.tnm, tasr_nemo_stream_new: the per-object precomputation (positional projections)
// Each line: {"workload", "clock" ("instructions" from the PMU via perf_event_open, else "ns"), "total" (the
// workload's cost), "ops" {stage: [cost, calls]} with stages exclusive of each other, "other" (total minus the
// stages: glue between them), "frames", "text"}. Model loading is never counted.
//
// usage: bench_engine nemo8.tnm oido_stream.tnm clip.wav
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "icount.h"
#include "port_util.h"
#include "tasr_nemo.h"

static int use_icount;
static uint64_t clk(void) { return use_icount ? icount_read() : pu_now_ns(CLOCK_MONOTONIC); }

static void json_str(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') printf("\\%c", *s);
        else if ((unsigned char)*s < 0x20) printf("\\u%04x", *s);
        else putchar(*s);
    }
    putchar('"');
}

static void report(const char *workload, uint64_t total, int frames, const char *text)
{
    uint64_t staged = 0;
    printf("{\"workload\":\"%s\",\"clock\":\"%s\",\"total\":%llu,\"ops\":{", workload, use_icount ? "instructions" : "ns",
           (unsigned long long)total);
    for (int i = 0; tasr_nemo_profile_name(i); i++) {
        printf("%s\"%s\":[%llu,%llu]", i ? "," : "", tasr_nemo_profile_name(i), (unsigned long long)tasr_nemo_profile_value(i),
               (unsigned long long)tasr_nemo_profile_calls(i));
        staged += tasr_nemo_profile_value(i);
    }
    printf("},\"other\":%lld,\"frames\":%d,\"text\":", (long long)(total - staged), frames);
    json_str(text);
    printf("}\n");
    fflush(stdout);
}

static tasr_nemo_t *load(const char *path, uint8_t **blob)
{
    size_t n;
    *blob = pu_read_file(path, (size_t)256 << 20, &n);
    tasr_nemo_t *m = *blob ? tasr_nemo_load(*blob, n) : NULL;
    if (!m) { fprintf(stderr, "%s: load failed\n", path); exit(1); }
    return m;
}

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: %s nemo8.tnm oido_stream.tnm clip.wav\n", argv[0]); return 2; }
    use_icount = icount_open();
    tasr_nemo_profile_set_clock(clk);
    int ns = 0;
    int16_t *pcm = pu_read_wav(argv[3], 600L * 16000, &ns);
    if (!pcm) return 1;
    static char text[4096];
    uint8_t *b1, *b2;

    tasr_nemo_t *m = load(argv[1], &b1);
    tasr_nemo_profile_reset();
    uint64_t t0 = clk();
    int frames = tasr_nemo_transcribe(m, pcm, ns, NULL, text, sizeof(text), NULL, 0, NULL);
    report("utterance", clk() - t0, frames, text);
    tasr_nemo_free(m);

    tasr_nemo_t *sm = load(argv[2], &b2);
    tasr_nemo_profile_reset();
    t0 = clk();
    tasr_nemo_stream_t *s = tasr_nemo_stream_new(sm, 0, 0, NULL);
    const uint64_t setup = clk() - t0;
    if (!s) return 1;
    report("stream_setup", setup, 0, "");
    tasr_nemo_profile_reset();
    t0 = clk();
    tasr_nemo_stream_reset(s);
    for (int o = 0; o < ns; o += 320) tasr_nemo_stream_feed(s, pcm + o, ns - o < 320 ? ns - o : 320);
    frames = tasr_nemo_stream_finish(s, text, sizeof(text));
    report("stream", clk() - t0, frames, text);
    tasr_nemo_stream_free(s);
    tasr_nemo_free(sm);
    free(b1); free(b2); free(pcm);
    return 0;
}
