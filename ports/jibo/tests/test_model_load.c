// test_model_load: the TNM1 loader and stream constructor against malformed input and failing allocations.
// Build with -fsanitize=address,undefined: every model copy is an exact-size allocation, so a read past the end of a
// truncated blob is reported. Checks:
//   - every shipped .tnm given on the command line loads, and stream support matches header flag bit 1
//   - truncations (every size up to 64 KiB, then a stride across the file, then the last 4 KiB) are refused
//   - each header word set to 0, 1, its value +/- 1, 0x7fffffff and 0xffffffff: refused, or loaded and then able to
//     transcribe and stream 0.5 s without a sanitizer report
//   - a misaligned blob is refused
//   - the Nth allocation failing, for every N reached, during load and stream creation: NULL and nothing leaked
// usage: test_model_load model.tnm [model.tnm ...]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tasr_nemo.h"
#include "tinyasr.h"

static long live, calls, fail_at;  // fail_at: the 1-based allocation call to fail (0 = never)
static void *t_alloc(size_t n, int kind)
{
    (void)kind;
    if (fail_at && ++calls == fail_at) return NULL;
    void *p = NULL;
    if (posix_memalign(&p, 16, n ? n : 16)) return NULL;
    memset(p, 0, n);
    live++;
    return p;
}
static void t_free(void *p)
{
    if (p) { live--; free(p); }
}

static uint8_t *copy_exact(const uint8_t *src, size_t n, size_t offset)
{
    uint8_t *b = NULL;
    if (posix_memalign((void **)&b, 16, n + offset ? n + offset : 1)) abort();
    memcpy(b + offset, src, n);
    return b;
}

#define PCM_N 8000
static int16_t pcm[PCM_N];   // 0.5 s of a deterministic chirp-like signal

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; printf("FAIL " __VA_ARGS__); printf("\n"); } } while (0)

static void test_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL cannot open %s\n", path); failures++; return; }
    fseek(f, 0, SEEK_END);
    const size_t size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *orig = malloc(size);
    if (fread(orig, 1, size, f) != size) abort();
    fclose(f);
    uint32_t flags;
    memcpy(&flags, orig + 4 + 10 * 4, 4);

    // whole file
    uint8_t *b = copy_exact(orig, size, 0);
    tasr_nemo_t *m = tasr_nemo_load(b, size);
    CHECK(m, "%s: does not load", path);
    if (m) {
        CHECK(!!tasr_nemo_stream_supported(m) == !!(flags & 2), "%s: stream support %d, flags %u", path,
              tasr_nemo_stream_supported(m), (unsigned)flags);
        CHECK(tasr_nemo_blob_bytes(m) <= size, "%s: blob bytes beyond file", path);
        tasr_nemo_free(m);
    }
    free(b);
    CHECK(live == 0, "%s: %ld allocations leaked after free", path, live);

    // truncations
    int tried = 0;
    const size_t tail = size > 4096 ? size - 4096 : 0;
    for (size_t n = 0; n < size; n = n < 65536 || n >= tail ? n + 1 : (n + 4093 < tail ? n + 4093 : tail)) {
        b = copy_exact(orig, n, 0);
        m = tasr_nemo_load(b, n);
        CHECK(!m, "%s: truncated to %zu bytes but loaded", path, n);
        tasr_nemo_free(m);
        free(b);
        tried++;
    }
    CHECK(live == 0, "%s: %ld allocations leaked after truncations", path, live);

    // header words
    int loaded = 0, refused = 0;
    for (int w = 0; w < 15; w++) {
        uint32_t v0;
        memcpy(&v0, orig + 4 + 4 * w, 4);
        const uint32_t vals[] = {0, 1, v0 + 1, v0 - 1, 0x7fffffffu, 0xffffffffu};
        for (size_t k = 0; k < sizeof(vals) / sizeof(vals[0]); k++) {
            b = copy_exact(orig, size, 0);
            memcpy(b + 4 + 4 * w, &vals[k], 4);
            m = tasr_nemo_load(b, size);
            if (getenv("TML_VERBOSE")) printf("word %d value %u -> %s\n", w, (unsigned)vals[k], m ? "loaded" : "refused");
            if (m) {   // a mutation the header check lets through must still run: transcribe and stream 0.5 s
                loaded++;
                char text[256];
                tasr_nemo_transcribe(m, pcm, PCM_N, NULL, text, sizeof(text), NULL, 0, NULL);
                tasr_nemo_stream_t *s = tasr_nemo_stream_supported(m) ? tasr_nemo_stream_new(m, 0, 0, NULL) : NULL;
                if (s) {
                    tasr_nemo_stream_feed(s, pcm, PCM_N);
                    tasr_nemo_stream_finish(s, text, sizeof(text));
                }
                tasr_nemo_stream_free(s);
                tasr_nemo_free(m);
            } else {
                refused++;
            }
            free(b);
        }
    }
    CHECK(live == 0, "%s: %ld allocations leaked after header mutations", path, live);

    // misaligned blob
    b = copy_exact(orig, size, 4);
    m = tasr_nemo_load(b + 4, size);
    CHECK(!m, "%s: misaligned blob loaded", path);
    tasr_nemo_free(m);
    free(b);

    // allocation failures in load, then in stream creation
    b = copy_exact(orig, size, 0);
    long n_load = 0, n_stream = 0;
    for (fail_at = 1;; fail_at++) {
        calls = 0;
        m = tasr_nemo_load(b, size);
        if (m) { tasr_nemo_free(m); break; }   // fail_at is past the last allocation
        CHECK(live == 0, "%s: load with allocation %ld failing leaked %ld", path, fail_at, live);
        n_load++;
    }
    if (flags & 2) {
        fail_at = 0;
        m = tasr_nemo_load(b, size);
        for (fail_at = 1;; fail_at++) {
            calls = 0;
            const long before = live;
            tasr_nemo_stream_t *s = tasr_nemo_stream_new(m, 0, 0, NULL);
            if (s) { tasr_nemo_stream_free(s); break; }
            CHECK(live == before, "%s: stream_new with allocation %ld failing leaked %ld", path, fail_at, live - before);
            n_stream++;
        }
        fail_at = 0;
        tasr_nemo_free(m);
    }
    fail_at = 0;
    free(b);
    CHECK(live == 0, "%s: %ld allocations leaked at the end", path, live);
    printf("%s: %zu bytes; %d truncations refused; header mutations %d loaded, %d refused; "
           "%ld load and %ld stream allocation failures handled\n",
           path, size, tried, loaded, refused, n_load, n_stream);
    free(orig);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s model.tnm ...\n", argv[0]); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int i = 0; i < PCM_N; i++) pcm[i] = (int16_t)(8000 * ((i * i / 37 + i) % 200 - 100) / 100);
    tasr_alloc = t_alloc;
    tasr_free = t_free;
    // a blob shorter than the header, and a NULL blob
    CHECK(!tasr_nemo_load((const uint8_t *)"TNM1", 4), "4-byte blob loaded");
    CHECK(!tasr_nemo_load(NULL, 100), "NULL blob loaded");
    for (int i = 1; i < argc; i++) test_model(argv[i]);
    printf("%s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
