// oido_seg_ab: A/B of upstream's utterance segmenter (tasr_seg.c: energy VAD against a noise floor, slow AGC,
// 0.2 s pre-roll, hang-over end of utterance) in front of the recognizer, on continuous audio with labelled
// utterance spans.
//
// Arms (--arms, default A,B,C,B25):
//   A    oracle: the labelled spans, raw audio (what a perfect endpointer would hand the recognizer)
//   B    tasr_seg boundaries and its AGC'd audio, hang 0.8 s (upstream's firmware/live demo behaviour)
//   C    tasr_seg boundaries, raw audio (the segmenter as an endpointer only, no gain stage stacked on the input)
//   B25  as B with a 0.5 s hang (upstream's suggestion for voice agents)
//   C25  as C with a 0.5 s hang
// The segmenter runs over the whole stream in 320-sample blocks, keeping its state (noise floor, gain) across
// utterances as it would live; a segment still open at the end of the stream is closed there.
//
// usage: oido_seg_ab [--arms A,B,C,B25,C25] model.tnm labels.tsv stream_dir
//   labels.tsv: header, then  stream  condition  start_sample  end_sample  reference  (start = end = -1: a stream with
//   no speech). Streaming models (oido_stream.tnm) decode each segment through the stream API; others transcribe it.
// stdout, one row per stream and arm:
//   stream condition arm expected segments spans onset_s end_delay_s saturated compute_ms hypothesis reference
//   spans: start-end sample pairs; onset_s: segment start minus utterance start for the first segment overlapping
//   each utterance (negative = pre-roll kept, positive = speech onset cut), averaged; end_delay_s: segment end minus
//   utterance end, averaged; saturated: AGC output samples at full scale; hypothesis/reference: per-segment and
//   per-utterance texts joined with " | ".
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port_util.h"
#include "tasr_nemo.h"
#include "tasr_seg.h"

#define MAX_LAB 16
#define MAX_SEG 64
#define CAP (20 * 16000)
#define TEXT 4096

typedef struct {
    char stream[256], cond[64];
    int n;
    long s[MAX_LAB], e[MAX_LAB];
    char ref[MAX_LAB][1024];
} item_t;

static tasr_nemo_t *M;
static tasr_nemo_stream_t *S;
static uint64_t compute_ns;

static void recognize(const int16_t *pcm, int n, char *text, int cap)
{
    const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
    text[0] = 0;
    if (S) {
        tasr_nemo_stream_reset(S);
        tasr_nemo_stream_feed(S, pcm, n);
        tasr_nemo_stream_finish(S, text, cap);
    } else {
        pu_alloc_fatal = 1;
        tasr_nemo_transcribe(M, pcm, n, NULL, text, cap, NULL, 0, NULL);
        pu_alloc_fatal = 0;
    }
    compute_ns += pu_now_ns(CLOCK_MONOTONIC) - t0;
}

static void append(char *dst, size_t cap, const char *s)
{
    size_t l = strlen(dst);
    snprintf(dst + l, cap - l, "%s%s", l ? " | " : "", s);
}

static void run_arm(const item_t *it, const int16_t *pcm, int ns, const char *arm)
{
    long ss[MAX_SEG], se[MAX_SEG];
    int nseg = 0;
    long saturated = 0;
    static char hyp[TEXT * 4], text[TEXT];
    hyp[0] = 0;
    compute_ns = 0;
    if (!strcmp(arm, "A")) {
        for (int k = 0; k < it->n; k++) {
            if (it->s[k] < 0) continue;
            ss[nseg] = it->s[k]; se[nseg] = it->e[k];
            recognize(pcm + it->s[k], (int)(it->e[k] - it->s[k]), text, sizeof(text));
            append(hyp, sizeof(hyp), text);
            nseg++;
        }
    } else {
        const int agc = arm[0] == 'B', hang = strstr(arm, "25") ? 25 : 0;
        static int16_t buf[CAP];
        tasr_seg_t seg;
        tasr_seg_init(&seg, buf, CAP);
        seg.hang = hang;
        long pos = 0;
        for (int o = 0; nseg < MAX_SEG; o += 320) {   // the last pass (k == 0) closes a segment still open
            const int k = o >= ns ? 0 : (ns - o < 320 ? ns - o : 320);
            int n = 0;
            long end = pos;
            if (k > 0) {
                const int prev = seg.n;
                n = tasr_seg_feed(&seg, pcm + o, k);
                pos += k;
                if (n) end = pos - k + (n - prev);
            } else if (seg.speech && seg.voiced >= TASR_SEG_MIN_VOICED) {   // end of stream: close an open segment
                n = seg.n;
                end = pos;
            }
            if (!n) { if (k == 0) break; continue; }
            ss[nseg] = end - n; se[nseg] = end;
            if (agc)
                for (int i = 0; i < n; i++) saturated += buf[i] == 32767 || buf[i] == -32768;
            recognize(agc ? buf : pcm + (end - n), n, text, sizeof(text));
            append(hyp, sizeof(hyp), text);
            nseg++;
            tasr_seg_next(&seg);
            if (k == 0) break;
        }
    }
    // onset and end-of-utterance timing, per labelled utterance, from the first overlapping segment
    double on = 0, off = 0;
    int matched = 0, expected = 0;
    static char ref[TEXT * 4];
    ref[0] = 0;
    for (int k = 0; k < it->n; k++) {
        if (it->s[k] < 0) continue;
        expected++;
        append(ref, sizeof(ref), it->ref[k]);
        for (int j = 0; j < nseg; j++)
            if (ss[j] < it->e[k] && se[j] > it->s[k]) {
                on += (ss[j] - it->s[k]) / 16000.0;
                off += (se[j] - it->e[k]) / 16000.0;
                matched++;
                break;
            }
    }
    printf("%s\t%s\t%s\t%d\t%d\t", it->stream, it->cond, arm, expected, nseg);
    for (int j = 0; j < nseg; j++) printf("%s%ld-%ld", j ? ";" : "", ss[j], se[j]);
    char ons[32] = "", offs[32] = "";   // empty when no segment overlaps any utterance
    if (matched) { snprintf(ons, sizeof(ons), "%.3f", on / matched); snprintf(offs, sizeof(offs), "%.3f", off / matched); }
    printf("\t%s\t%s\t%ld\t%.1f\t%s\t%s\n", ons, offs, saturated, compute_ns / 1e6, hyp, ref);
}

int main(int argc, char **argv)
{
    char arms_buf[128] = "A,B,C,B25";
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--arms") && i + 1 < argc) snprintf(arms_buf, sizeof(arms_buf), "%s", argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (argc - i != 3) { fprintf(stderr, "usage: %s [--arms A,B,C,B25,C25] model.tnm labels.tsv stream_dir\n", argv[0]); return 2; }
    pu_alloc_install();
    size_t msz;
    uint8_t *mb = pu_read_file(argv[i], (size_t)256 << 20, &msz);
    if (!mb || !(M = tasr_nemo_load(mb, msz))) { fprintf(stderr, "%s: model load failed\n", argv[i]); return 1; }
    if (tasr_nemo_stream_supported(M) && !(S = tasr_nemo_stream_new(M, 0, 0, NULL))) return 1;
    FILE *lf = fopen(argv[i + 1], "r");
    if (!lf) { perror(argv[i + 1]); return 1; }
    char *arms[8];
    int narm = 0;
    for (char *t = strtok(arms_buf, ","); t && narm < 8; t = strtok(NULL, ",")) {
        if (strcmp(t, "A") && strcmp(t, "B") && strcmp(t, "C") && strcmp(t, "B25") && strcmp(t, "C25")) {
            fprintf(stderr, "unknown arm %s\n", t);
            return 2;
        }
        arms[narm++] = t;
    }
    printf("stream\tcondition\tarm\texpected\tsegments\tspans\tonset_s\tend_delay_s\tsaturated\tcompute_ms\thypothesis\treference\n");
    static char line[8192];
    item_t it;
    memset(&it, 0, sizeof(it));
    int have = 0, failures = 0;
    if (!fgets(line, sizeof(line), lf)) return 1;   // header
    for (;;) {
        const int got = fgets(line, sizeof(line), lf) != NULL;
        char st[256] = "", cond[64] = "", ref[1024] = "";
        long s = 0, e = 0;
        if (got) {
            line[strcspn(line, "\n")] = 0;
            char *f[5] = {0};
            int nf = 0;
            for (char *p = line; nf < 5; nf++) { f[nf] = p; char *t = strchr(p, '\t'); if (!t) { nf++; break; } *t = 0; p = t + 1; }
            if (nf < 4) continue;
            snprintf(st, sizeof(st), "%s", f[0]); snprintf(cond, sizeof(cond), "%s", f[1]);
            s = atol(f[2]); e = atol(f[3]);
            if (nf > 4 && f[4]) snprintf(ref, sizeof(ref), "%s", f[4]);
        }
        if (have && (!got || strcmp(st, it.stream))) {   // a stream's rows are complete: run it
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s", argv[i + 2], it.stream);
            int ns = 0;
            int16_t *pcm = pu_read_wav(path, 600L * 16000, &ns);
            if (!pcm) failures++;
            else {
                for (int a = 0; a < narm; a++) run_arm(&it, pcm, ns, arms[a]);
                fflush(stdout);
                free(pcm);
            }
            memset(&it, 0, sizeof(it));
            have = 0;
        }
        if (!got) break;
        if (!have) { snprintf(it.stream, sizeof(it.stream), "%s", st); snprintf(it.cond, sizeof(it.cond), "%s", cond); have = 1; }
        if (it.n < MAX_LAB) { it.s[it.n] = s; it.e[it.n] = e; snprintf(it.ref[it.n], sizeof(it.ref[0]), "%s", ref); it.n++; }
    }
    fclose(lf);
    tasr_nemo_stream_free(S);
    tasr_nemo_free(M);
    free(mb);
    return failures ? 1 : 0;
}
