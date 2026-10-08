// test_stream_lifecycle: state handling of the NeMo stream API, on a streaming model and one speech WAV.
//   1. reset and reuse: replaying the same audio gives identical text, frames and logits
//   2. no leakage: after a different input (noise), reset, then the reference audio: identical again
//   3. finish twice: same text and frames, nothing recomputed
//   4. feed after finish: ignored (frame count and text unchanged)
//   5. short inputs: 0, 1, 100, 319 samples give 0 frames and no text; 320 and 321 give 1 frame (as transcribe)
//   6. chunk boundaries: inputs ending just before, at and after chunk ends (multiples of chunk * 640 samples, +/- 1
//      and +/- one encoder frame) give the same result fed in 320-sample blocks as in one block
//   7. truncation: a 4-byte text buffer sets tasr_nemo_stream_truncated; reset clears it
//   8. frame count equals tasr_nemo_transcribe's for the same audio (texts differ: full vs limited context)
//   9. two stream objects on one model, fed alternately with different audio, match sequential replays
// usage: test_stream_lifecycle stream_model.tnm speech.wav   (exit 1 on any failure)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "port_util.h"
#include "tasr_nemo.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL " __VA_ARGS__); printf("\n"); } else if (verbose) { printf("ok   " __VA_ARGS__); printf("\n"); } } while (0)
static int verbose = 1;

typedef struct {
    int frames;
    char text[2048];
    float *logits;
    int nlog;
} result_t;

static void run(tasr_nemo_stream_t *s, const int16_t *pcm, int n, int block, result_t *r, int max_frames)
{
    tasr_nemo_stream_reset(s);
    tasr_nemo_stream_set_sink(s, r->logits, max_frames);
    for (int o = 0; o < n; o += block) tasr_nemo_stream_feed(s, pcm + o, n - o < block ? n - o : block);
    r->frames = tasr_nemo_stream_finish(s, r->text, sizeof(r->text));
    r->nlog = r->frames < max_frames ? r->frames : max_frames;
}
static int same(const result_t *a, const result_t *b)
{
    return a->frames == b->frames && !strcmp(a->text, b->text) &&
           !memcmp(a->logits, b->logits, sizeof(float) * 1025 * (size_t)a->nlog);
}
static result_t *mk(int max_frames)
{
    result_t *r = calloc(1, sizeof(result_t));
    r->logits = calloc((size_t)max_frames * 1025, sizeof(float));
    return r;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s stream_model.tnm speech.wav\n", argv[0]); return 2; }
    pu_alloc_install();
    size_t msz;
    uint8_t *mb = pu_read_file(argv[1], (size_t)256 << 20, &msz);
    tasr_nemo_t *m = mb ? tasr_nemo_load(mb, msz) : NULL;
    if (!m || !tasr_nemo_stream_supported(m)) { printf("FAIL cannot load a streaming model from %s\n", argv[1]); return 1; }
    int n = 0;
    int16_t *pcm = pu_read_wav(argv[2], 20 * 16000, &n);
    if (!pcm) return 1;
    uint32_t hd[15];
    memcpy(hd, mb + 4, sizeof(hd));
    const int C = hd[11] ? (int)hd[11] : 16, chunk_samples = C * 640;
    // a noise input for the leakage check
    int16_t *noise = malloc(sizeof(int16_t) * n);
    uint32_t x = 99;
    for (int i = 0; i < n; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; noise[i] = (int16_t)((int)(x % 4001) - 2000); }

    const int MF = n / 640 + 8;
    tasr_nemo_stream_t *s = tasr_nemo_stream_new(m, 0, 0, NULL);
    CHECK(s != NULL, "stream created");
    result_t *ref = mk(MF), *r = mk(MF), *r2 = mk(MF);
    run(s, pcm, n, 320, ref, MF);
    printf("reference: %d samples, %d frames, chunk %d frames: \"%s\"\n", n, ref->frames, C, ref->text);

    // 1, 2
    run(s, pcm, n, 320, r, MF);
    CHECK(same(ref, r), "1 reset and reuse: identical");
    run(s, noise, n, 320, r, MF);
    run(s, pcm, n, 320, r, MF);
    CHECK(same(ref, r), "2 after a noise input and reset: identical");

    // 3, 4
    char t2[2048], t3[2048];
    const int f2 = tasr_nemo_stream_finish(s, t2, sizeof(t2));
    CHECK(f2 == r->frames && !strcmp(t2, r->text), "3 second finish: same frames (%d) and text", f2);
    const int f3 = tasr_nemo_stream_feed(s, pcm, 3200);
    const int f4 = tasr_nemo_stream_finish(s, t3, sizeof(t3));
    CHECK(f3 == r->frames && f4 == r->frames && !strcmp(t3, r->text), "4 feed after finish ignored");

    // 5
    const int shorts[] = {0, 1, 100, 319, 320, 321};
    for (size_t k = 0; k < sizeof(shorts) / sizeof(shorts[0]); k++) {
        run(s, pcm, shorts[k], 320, r, MF);
        char tt[64];
        const int tf = tasr_nemo_transcribe(m, pcm, shorts[k], NULL, tt, sizeof(tt), NULL, 0, NULL);
        CHECK(r->frames == tf && r->frames == (shorts[k] < 320 ? 0 : 1) && (shorts[k] >= 320 || !r->text[0]),
              "5 %d samples: %d frames (transcribe %d)", shorts[k], r->frames, tf);
    }

    // 6
    int boundary_cases = 0, boundary_ok = 0;
    for (int k = 1; k * chunk_samples + 641 <= n; k++) {
        const int deltas[] = {-641, -640, -1, 0, 1, 640, 641};
        for (size_t dd = 0; dd < sizeof(deltas) / sizeof(deltas[0]); dd++) {
            const int len = k * chunk_samples + deltas[dd];
            run(s, pcm, len, 320, r, MF);
            run(s, pcm, len, len, r2, MF);
            boundary_cases++;
            if (same(r, r2)) boundary_ok++;
            else printf("     boundary mismatch at %d samples (%d frames vs %d)\n", len, r->frames, r2->frames);
        }
    }
    CHECK(boundary_cases > 0 && boundary_ok == boundary_cases, "6 chunk boundaries: %d of %d lengths block-invariant",
          boundary_ok, boundary_cases);

    // 7
    tasr_nemo_stream_reset(s);
    tasr_nemo_stream_feed(s, pcm, n);
    char tiny[4];
    tasr_nemo_stream_finish(s, tiny, sizeof(tiny));
    CHECK(tasr_nemo_stream_truncated(s) && strlen(tiny) <= 3, "7 4-byte buffer: truncation reported");
    tasr_nemo_stream_reset(s);
    CHECK(!tasr_nemo_stream_truncated(s), "7 reset clears the truncation flag");

    // 8
    char full[2048];
    int nf = 0;
    const int tf = tasr_nemo_transcribe(m, pcm, n, NULL, full, sizeof(full), NULL, 0, &nf);
    CHECK(tf == ref->frames, "8 frames: stream %d, transcribe %d (full-context text: \"%s\")", ref->frames, tf, full);

    // 9
    tasr_nemo_stream_t *s2 = tasr_nemo_stream_new(m, 0, 0, NULL);
    result_t *noise_ref = mk(MF), *a = mk(MF), *b = mk(MF);
    run(s, noise, n, 320, noise_ref, MF);
    tasr_nemo_stream_reset(s); tasr_nemo_stream_reset(s2);
    tasr_nemo_stream_set_sink(s, a->logits, MF); tasr_nemo_stream_set_sink(s2, b->logits, MF);
    for (int o = 0; o < n; o += 320) {
        const int k = n - o < 320 ? n - o : 320;
        tasr_nemo_stream_feed(s, pcm + o, k);
        tasr_nemo_stream_feed(s2, noise + o, k);
    }
    a->frames = tasr_nemo_stream_finish(s, a->text, sizeof(a->text)); a->nlog = a->frames;
    b->frames = tasr_nemo_stream_finish(s2, b->text, sizeof(b->text)); b->nlog = b->frames;
    CHECK(same(ref, a) && same(noise_ref, b), "9 two interleaved streams match sequential replays");
    tasr_nemo_stream_free(s2);

    tasr_nemo_stream_free(s);
    tasr_nemo_free(m);
    free(mb);
    CHECK(pu_alloc_cur == 0, "all engine allocations released (%zu bytes outstanding)", pu_alloc_cur);
    printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
