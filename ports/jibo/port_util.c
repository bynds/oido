// port_util: see port_util.h.
#define _GNU_SOURCE
#include "port_util.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinyasr.h"
#ifdef TASR_KERNEL_DISPATCH
#include "kernels.h"
#endif

uint64_t pu_now_ns(clockid_t id)
{
    struct timespec t;
    clock_gettime(id, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

size_t pu_alloc_cur, pu_alloc_peak, pu_alloc_calls;
int pu_alloc_fatal;
static void *counting_alloc(size_t n, int kind)
{
    (void)kind;
    void *p = NULL;
    if (n > SIZE_MAX - 16 || posix_memalign(&p, 16, n + 16)) {
        if (pu_alloc_fatal) { fprintf(stderr, "fatal: engine allocation of %zu bytes failed during inference\n", n); abort(); }
        return NULL;
    }
    memset(p, 0, n + 16);
    memcpy(p, &n, sizeof(n));
    pu_alloc_cur += n;
    pu_alloc_calls++;
    if (pu_alloc_cur > pu_alloc_peak) pu_alloc_peak = pu_alloc_cur;
    return (uint8_t *)p + 16;
}
static void counting_free(void *q)
{
    if (!q) return;
    uint8_t *p = (uint8_t *)q - 16;
    size_t n;
    memcpy(&n, p, sizeof(n));
    pu_alloc_cur -= n;
    free(p);
}
void pu_alloc_install(void)
{
    tasr_alloc = counting_alloc;
    tasr_free = counting_free;
}

uint8_t *pu_read_file(const char *path, size_t max_bytes, size_t *size)
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

int16_t *pu_read_wav(const char *path, long max_samples, int *ns)
{
    size_t n;
    uint8_t *b = pu_read_file(path, (size_t)max_samples * 2 + 4096, &n);
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

const char *pu_base_name(const char *p)
{
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

int pu_kernels_init(void)
{
#ifdef TASR_KERNEL_DISPATCH
    const char *kenv = getenv("OIDO_KERNELS");
    if (kenv && !strcmp(kenv, "scalar")) tasr_kernel_force_scalar = 1;
    else if (kenv && *kenv && strcmp(kenv, "auto")) { fprintf(stderr, "OIDO_KERNELS must be scalar or auto\n"); return -1; }
#endif
    fprintf(stderr, "kernels: %s\n", pu_kernels_name());
    return 0;
}
const char *pu_kernels_name(void)
{
#ifdef TASR_KERNEL_DISPATCH
    return tasr_kernel_backend();
#else
    return "scalar (no dispatch layer)";
#endif
}
