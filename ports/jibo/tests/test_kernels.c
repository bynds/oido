// test_kernels: the dispatched int8 kernels (NEON in a -DTASR_NEON build) against the portable C kernels, which must
// agree exactly (int32), on:
//   - data: uniform random int8; all -128; all 127; -128 against 127; alternating extremes; zeros; random sparse
//   - shapes: kp 16..1024 in steps of 16, the model's real ones (48, 176, 704, 1584, 3520, the long-row split's
//     chunks) and kp not a multiple of 16 (the NEON tails); T 1..67 (remainders of the 4-row and 2-row unrolls);
//     nb 1..16; row strides equal to kp, padded, odd, and unaligned base pointers
//   - the public tasr_qlin_range on every branch (x-stationary kp <= 256, row layout, long rows > 1024, int4 blocked
//     and row) with tasr_kernel_force_scalar off and on: identical float outputs
// Deterministic (fixed-seed xorshift). Prints the backend and the number of comparisons; exit 1 on any mismatch.
// Build with -DTASR_KERNEL_DISPATCH (and -DTASR_NEON for ARM), linking kernels.c and ports/jibo/kernels_neon.c.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"

static uint32_t rng = 2463534242u;
static uint32_t xs32(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

enum { D_RANDOM, D_MIN, D_MAX, D_MINMAX, D_ALT, D_ZERO, D_SPARSE, D_N };
static void fill(int8_t *p, size_t n, int mode, int side)
{
    for (size_t i = 0; i < n; i++) {
        switch (mode) {
        case D_RANDOM: p[i] = (int8_t)xs32(); break;
        case D_MIN: p[i] = -128; break;
        case D_MAX: p[i] = 127; break;
        case D_MINMAX: p[i] = side ? 127 : -128; break;
        case D_ALT: p[i] = (i + side) & 1 ? 127 : -128; break;
        case D_ZERO: p[i] = 0; break;
        default: p[i] = xs32() % 7 == 0 ? (int8_t)xs32() : 0; break;
        }
    }
}

static long checks, bad;
static void cmp(const char *what, const int32_t *a, const int32_t *b, int n, int kp, int T, int nb, int ld, int mode)
{
    checks++;
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) {
            if (bad++ < 20)
                printf("MISMATCH %s kp %d T %d nb %d ld %d data %d at %d: %d vs %d\n", what, kp, T, nb, ld, mode, i,
                       (int)a[i], (int)b[i]);
            return;
        }
}

#define MAXK 3600
#define MAXT 67
static int8_t xbuf[(MAXT + 1) * (MAXK + 64) + 64], wbuf[16 * (MAXK + 64) + 64];
static int32_t o1[MAXT * 16], o2[MAXT * 16];

static void test_shape(int kp, int T, int nb, int ldpad, int offset, int mode)
{
    const int ldq = kp + ldpad;
    int8_t *x = xbuf + offset, *w = wbuf + offset;
    fill(x, (size_t)T * ldq, mode, 0);
    fill(w, (size_t)16 * kp + 48, mode, 1);
    tasr_dot_rows_s8(w, x, ldq, T, kp, o1);
    tasr_dot_rows_s8_scalar(w, x, ldq, T, kp, o2);
    cmp("dot_rows_s8", o1, o2, T, kp, T, 1, ldq, mode);
    if (kp <= 1024) {
        tasr_gemm_s8_xr(w, kp, nb, x, ldq, T, o1);
        tasr_gemm_s8_xr_scalar(w, kp, nb, x, ldq, T, o2);
        cmp("gemm_s8_xr", o1, o2, T * nb, kp, T, nb, ldq, mode);
    }
    if (kp >= 48) {
        tasr_dot48_rows(w, x, ldq, T, o1);
        tasr_dot48_rows_scalar(w, x, ldq, T, o2);
        cmp("dot48_rows", o1, o2, T, 48, T, 1, ldq, mode);
    }
    int32_t d1 = 0;
    if (kp % 16 == 0 && offset == 0) {
        d1 = tasr_dot_s8(x, w, kp);
        int32_t d2;
        tasr_dot_rows_s8_scalar(w, x, kp, 1, kp, &d2);
        cmp("dot_s8", &d1, &d2, 1, kp, 1, 1, kp, mode);
    }
}

// tasr_qlin_range on a synthetic layer, NEON vs forced scalar: the float outputs must be bit-identical
static void test_qlin(int n, int k, int bits, int T)
{
    tasr_qlin_t L;
    memset(&L, 0, sizeof(L));
    L.blocked = bits == 4 && n % 16 == 0;
    const int blk = (bits == 8 || L.blocked) ? 16 : 32;
    L.n = n; L.k = k; L.bits = bits; L.kp = (k + blk - 1) / blk * blk;
    const size_t wb = (size_t)n * (bits == 8 ? L.kp : L.kp / 2);
    int8_t *w = aligned_alloc(16, (wb + 15) & ~(size_t)15);
    float *s = malloc(sizeof(float) * n), *b = malloc(sizeof(float) * n);
    for (size_t i = 0; i < wb; i++) w[i] = (int8_t)xs32();
    for (int i = 0; i < n; i++) { s[i] = (xs32() % 1000 + 1) * 1e-5f; b[i] = ((int)(xs32() % 2001) - 1000) * 1e-3f; }
    L.w = w; L.s = s; L.b = b;
    const int ldq = L.kp;
    int8_t *xq = aligned_alloc(16, (size_t)T * ldq);
    float *xsc = malloc(sizeof(float) * T);
    for (size_t i = 0; i < (size_t)T * ldq; i++) xq[i] = (int8_t)(xs32() % 255 - 127);
    for (int t = 0; t < T; t++) xsc[t] = (xs32() % 1000 + 1) * 1e-4f;
    float *y1 = malloc(sizeof(float) * T * n), *y2 = malloc(sizeof(float) * T * n);
    int8_t *wtmp = aligned_alloc(16, 16 * (size_t)L.kp + 16);
    int32_t *acc = malloc(sizeof(int32_t) * 64 * 16);
    tasr_kernel_force_scalar = 0;
    tasr_qlin_range(&L, xq, xsc, T, ldq, y1, n, wtmp, acc, 0, n);
    tasr_kernel_force_scalar = 1;
    tasr_qlin_range(&L, xq, xsc, T, ldq, y2, n, wtmp, acc, 0, n);
    tasr_kernel_force_scalar = 0;
    checks++;
    if (memcmp(y1, y2, sizeof(float) * T * n)) {
        if (bad++ < 20) printf("MISMATCH qlin_range n %d k %d bits %d T %d\n", n, k, bits, T);
    }
    free(w); free(s); free(b); free(xq); free(xsc); free(y1); free(y2); free(wtmp); free(acc);
}

int main(void)
{
    printf("backend: %s\n", tasr_kernel_backend());
    // every kp up to 1024 (multiples of 16), every data mode, T and nb covering the unroll remainders
    for (int kp = 16; kp <= 1024; kp += 16)
        for (int mode = 0; mode < D_N; mode++) {
            const int T = 1 + (int)(xs32() % MAXT), nb = 1 + (int)(xs32() % 16);
            test_shape(kp, T, nb, 0, 0, mode);
        }
    // the model's shapes, exhaustively over T and nb, plus padded / odd strides and unaligned bases
    const int real_kp[] = {48, 176, 704, 1584, 3520, 528, 1184, 1760};
    for (size_t r = 0; r < sizeof(real_kp) / sizeof(real_kp[0]); r++)
        for (int T = 1; T <= MAXT; T++)
            for (int mode = 0; mode < (T % 7 == 0 ? D_N : 1); mode++) {   // every data mode on every 7th T
                const int nb = 1 + (T - 1) % 16;
                test_shape(real_kp[r], T, nb, 0, 0, mode);
                test_shape(real_kp[r], T, nb, 16, 0, D_RANDOM);
                test_shape(real_kp[r], T, nb, 3, 1, D_RANDOM);
            }
    // kp not a multiple of 16 (outside the kernels' contract; the NEON versions handle the tail exactly)
    for (int kp = 1; kp < 80; kp++) test_shape(kp, 1 + kp % 9, 1 + kp % 16, kp % 5, kp % 3, D_RANDOM);
    // extremes at the longest rows: 3520 * 16384 is far below 2^31
    for (int mode = D_MIN; mode <= D_ALT; mode++) test_shape(3520, MAXT, 16, 0, 0, mode);
    // every tasr_qlin_range branch
    test_qlin(176, 176, 8, 50);    // x-stationary (kp <= 256): attention, ff in, conv, head
    test_qlin(1025, 176, 8, 64);   // CTC head shape, nb remainder 1
    test_qlin(176, 704, 8, 65);    // row layout (256 < kp <= 1024): ff out
    test_qlin(176, 3520, 8, 33);   // long rows (kp > 1024): front-end projection
    test_qlin(176, 1584, 8, 20);   // long rows: conv2 as GEMM
    test_qlin(704, 176, 4, 40);    // int4 blocked (n % 16 == 0)
    test_qlin(170, 176, 4, 40);    // int4 row layout
    test_qlin(176, 704, 4, 64);    // int4 row layout, longer rows
    printf("%ld comparisons, %ld mismatches\n%s\n", checks, bad, bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
