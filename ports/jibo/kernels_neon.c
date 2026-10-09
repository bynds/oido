// kernels_neon.c: the public int8 kernel names for Linux builds of the Oído engine, built with -DTASR_KERNEL_DISPATCH.
//
//   -DTASR_NEON          ARMv7 NEON versions of the int8 dot-product kernels (needs -mfpu=neon)
//   -DTASR_SIMD32        ARMv6/v7 SIMD32 versions (sxtb16 + smlad on core registers; no NEON, no VFP): for the
//                        plain VFPv3-D16 build
//   neither              every call goes to the portable C version in kernels.c (named *_scalar)
//   -DTASR_KERNEL_STATS  count calls and multiply-accumulates per kernel and shape; tasr_kernel_stats_dump()
//
// Arithmetic: the NEON kernels compute the same int32 sums as the C ones, exactly. Each pair of int8 products is
// formed in int16 by vmull_s8 (|-128 * -128| = 16384 fits) and widened into int32 lanes by vpadalq_s16 before any
// further addition, so no int16 sum is ever formed. A lane collects kp/4 products and the four lanes are added at the
// end, so the result equals the scalar int32 sum whenever that sum itself cannot overflow: kp * 16384 < 2^31, i.e.
// kp < 131072. The model's longest row is 3520 (front-end projection). Only ARMv7 instructions are used (no SDOT,
// no AArch64 across-vector adds). Unaligned rows are fine: vld1q_s8 has no alignment requirement on ARMv7.
//
// The int4 tile kernel (tasr_gemm_blk16, nemo4.tnm only) has no NEON version yet and always runs the C code.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "kernels_dispatch.h"

#ifndef TASR_KERNEL_DISPATCH
#error "kernels_neon.c needs -DTASR_KERNEL_DISPATCH (kernels.c then provides the *_scalar kernels)"
#endif
#ifdef TASR_NEON
#ifndef __ARM_NEON
#error "-DTASR_NEON needs a NEON target (-mfpu=neon)"
#endif
#include <arm_neon.h>
#elif defined(TASR_SIMD32)
#if !defined(__arm__) || !defined(__ARM_ARCH) || __ARM_ARCH < 6
#error "-DTASR_SIMD32 needs an ARMv6 or later 32-bit ARM target"
#endif
#endif

int tasr_kernel_force_scalar;

const char *tasr_kernel_backend(void)
{
#ifdef TASR_NEON
    if (!tasr_kernel_force_scalar) return "neon";
#elif defined(TASR_SIMD32)
    if (!tasr_kernel_force_scalar) return "simd32";
#endif
    return "scalar";
}

// ------------------------------------------------------------------------------------------------ statistics
#ifdef TASR_KERNEL_STATS
typedef struct {
    int fn, kp, T, nb, ld;
    unsigned long long calls, macs;
} kstat_t;
#define KSTAT_N 1024
static kstat_t kstat[KSTAT_N];
static int kstat_overflow;
static const char *kstat_fn[] = {"dot_rows_s8", "gemm_s8_xr", "dot48_rows", "gemm_blk16"};
static void kstat_add(int fn, int kp, int T, int nb, int ld, unsigned long long macs)
{
    unsigned h = ((unsigned)fn * 2654435761u) ^ ((unsigned)kp * 40503u) ^ ((unsigned)T * 9973u) ^ ((unsigned)nb * 131u) ^
                 (unsigned)ld;
    for (int probe = 0; probe < KSTAT_N; probe++) {
        kstat_t *e = &kstat[(h + probe) % KSTAT_N];
        if (!e->calls) { e->fn = fn; e->kp = kp; e->T = T; e->nb = nb; e->ld = ld; }
        if (e->fn == fn && e->kp == kp && e->T == T && e->nb == nb && e->ld == ld) {
            e->calls++;
            e->macs += macs;
            return;
        }
    }
    kstat_overflow = 1;
}
static int kstat_cmp(const void *a, const void *b)
{
    const kstat_t *x = a, *y = b;
    return x->macs < y->macs ? 1 : x->macs > y->macs ? -1 : 0;
}
void tasr_kernel_stats_dump(FILE *f)
{
    kstat_t tmp[KSTAT_N];
    int n = 0;
    unsigned long long total = 0;
    for (int i = 0; i < KSTAT_N; i++)
        if (kstat[i].calls) { tmp[n++] = kstat[i]; total += kstat[i].macs; }
    qsort(tmp, n, sizeof(tmp[0]), kstat_cmp);
    unsigned long long by_fn[4] = {0};
    for (int i = 0; i < n; i++) by_fn[tmp[i].fn] += tmp[i].macs;
    fprintf(f, "kernel\tshare_of_macs\tmacs\n");
    for (int i = 0; i < 4; i++)
        fprintf(f, "%s\t%.4f\t%llu\n", kstat_fn[i], total ? (double)by_fn[i] / total : 0.0, by_fn[i]);
    fprintf(f, "\nkernel\tkp\tT\tnb\tld\tcalls\tmacs\tshare\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%s\t%d\t%d\t%d\t%d\t%llu\t%llu\t%.4f\n", kstat_fn[tmp[i].fn], tmp[i].kp, tmp[i].T, tmp[i].nb, tmp[i].ld,
                tmp[i].calls, tmp[i].macs, total ? (double)tmp[i].macs / total : 0.0);
    if (kstat_overflow) fprintf(f, "# table full: some shapes not recorded\n");
}
void tasr_kernel_stats_reset(void) { memset(kstat, 0, sizeof(kstat)); kstat_overflow = 0; }
#define KSTAT(...) kstat_add(__VA_ARGS__)
#else
void tasr_kernel_stats_dump(FILE *f) { fprintf(f, "# kernel statistics not compiled in (-DTASR_KERNEL_STATS)\n"); }
void tasr_kernel_stats_reset(void) {}
#define KSTAT(...) ((void)0)
#endif

// ------------------------------------------------------------------------------------------------ NEON kernels
#ifdef TASR_NEON
// acc += a . b over 16 int8 lanes, exactly (int16 products, widened pairwise into int32 lanes)
static inline int32x4_t mac16(int32x4_t acc, int8x16_t a, int8x16_t b)
{
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b)));
    return vpadalq_s16(acc, vmull_s8(vget_high_s8(a), vget_high_s8(b)));
}
static inline int32_t hsum(int32x4_t v)
{
    int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(s, s), 0);
}
// remainder of a row that is not a multiple of 16 (the kernels' contract says it is; this keeps them total)
static inline int32_t tail_dot(const int8_t *a, const int8_t *b, int from, int to)
{
    int32_t s = 0;
    for (int i = from; i < to; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

// Four dot products against one shared vector, 16 bytes per step, in hand-written asm: GCC compiled the intrinsics
// version with base+offset addressing (five address adds per step: 32 instructions per 64 multiply-accumulates);
// post-increment loads make it 23. Same arithmetic (vmull.s8 then vpadal.s16 into int32 lanes), so the same sums.
// Clobbers only q0-q3 and q8-q15 (q4-q7 are callee-saved). n16 >= 1 blocks of 16 bytes. out4[i] = dot(s, p_i).
static inline void dot1x4_neon(const int8_t *sv, const int8_t *p0, const int8_t *p1, const int8_t *p2, const int8_t *p3,
                               int n16, int32_t *out4)
{
    __asm__ volatile(
        "vmov.i32   q8, #0\n\t"
        "vmov.i32   q9, #0\n\t"
        "vmov.i32   q10, #0\n\t"
        "vmov.i32   q11, #0\n\t"
        "1:\n\t"
        "vld1.8     {d0-d1}, [%[s]]!\n\t"
        "vld1.8     {d2-d3}, [%[p0]]!\n\t"
        "vld1.8     {d4-d5}, [%[p1]]!\n\t"
        "vmull.s8   q12, d0, d2\n\t"
        "vmull.s8   q13, d1, d3\n\t"
        "vld1.8     {d6-d7}, [%[p2]]!\n\t"
        "vpadal.s16 q8, q12\n\t"
        "vpadal.s16 q8, q13\n\t"
        "vmull.s8   q14, d0, d4\n\t"
        "vmull.s8   q15, d1, d5\n\t"
        "vld1.8     {d2-d3}, [%[p3]]!\n\t"
        "vpadal.s16 q9, q14\n\t"
        "vpadal.s16 q9, q15\n\t"
        "vmull.s8   q12, d0, d6\n\t"
        "vmull.s8   q13, d1, d7\n\t"
        "vpadal.s16 q10, q12\n\t"
        "vpadal.s16 q10, q13\n\t"
        "vmull.s8   q14, d0, d2\n\t"
        "vmull.s8   q15, d1, d3\n\t"
        "vpadal.s16 q11, q14\n\t"
        "vpadal.s16 q11, q15\n\t"
        "subs       %[n], %[n], #1\n\t"
        "bne        1b\n\t"
        "vpadd.i32  d16, d16, d17\n\t"   // lane sums: [s0 s0'] ... then pairwise across accumulators
        "vpadd.i32  d17, d18, d19\n\t"
        "vpadd.i32  d18, d20, d21\n\t"
        "vpadd.i32  d19, d22, d23\n\t"
        "vpadd.i32  d16, d16, d17\n\t"   // [dot0 dot1]
        "vpadd.i32  d17, d18, d19\n\t"   // [dot2 dot3]
        "vst1.32    {d16-d17}, [%[o]]\n\t"
        : [s] "+r"(sv), [p0] "+r"(p0), [p1] "+r"(p1), [p2] "+r"(p2), [p3] "+r"(p3), [n] "+r"(n16)
        : [o] "r"(out4)
        : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
          "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31", "memory", "cc");
}

// out[t] = dot(w, x + t*ldq): four rows share each weight vector
static void dot_rows_s8_neon(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    const int k16 = kp & ~15;
    int t = 0;
    for (; k16 && t + 4 <= T; t += 4) {
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        dot1x4_neon(w, a0, a1, a2, a3, k16 >> 4, out + t);
        if (k16 < kp) {
            out[t] += tail_dot(a0, w, k16, kp);
            out[t + 1] += tail_dot(a1, w, k16, kp);
            out[t + 2] += tail_dot(a2, w, k16, kp);
            out[t + 3] += tail_dot(a3, w, k16, kp);
        }
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32x4_t s = vdupq_n_s32(0);
        for (int i = 0; i < k16; i += 16) s = mac16(s, vld1q_s8(a + i), vld1q_s8(w + i));
        out[t] = hsum(s) + tail_dot(a, w, k16, kp);
    }
}

// acc[t*nb + j] = dot(x + t*ldq, W + j*kp): each activation vector is shared by four weight rows
static void gemm_s8_xr_neon(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    const int k16 = kp & ~15;
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t *o = acc + (size_t)t * nb;
        int j = 0;
        for (; k16 && j + 4 <= nb; j += 4) {
            const int8_t *w0 = W + (size_t)j * kp, *w1 = w0 + kp, *w2 = w1 + kp, *w3 = w2 + kp;
            dot1x4_neon(a, w0, w1, w2, w3, k16 >> 4, o + j);
            if (k16 < kp) {
                o[j] += tail_dot(a, w0, k16, kp);
                o[j + 1] += tail_dot(a, w1, k16, kp);
                o[j + 2] += tail_dot(a, w2, k16, kp);
                o[j + 3] += tail_dot(a, w3, k16, kp);
            }
        }
        for (; j < nb; j++) {
            const int8_t *w = W + (size_t)j * kp;
            int32x4_t s = vdupq_n_s32(0);
            for (int i = 0; i < k16; i += 16) s = mac16(s, vld1q_s8(a + i), vld1q_s8(w + i));
            o[j] = hsum(s) + tail_dot(a, w, k16, kp);
        }
    }
}

// out[t] = dot48(q, x + t*ldx): the 48-byte query stays in three registers, two rows per iteration
static void dot48_rows_neon(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    const int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16), q2 = vld1q_s8(q + 32);
    int t = 0;
    for (; t + 2 <= T; t += 2) {
        const int8_t *a = x + (size_t)t * ldx, *b = a + ldx;
        int32x4_t sa = vdupq_n_s32(0), sb = sa;
        sa = mac16(sa, vld1q_s8(a), q0);
        sb = mac16(sb, vld1q_s8(b), q0);
        sa = mac16(sa, vld1q_s8(a + 16), q1);
        sb = mac16(sb, vld1q_s8(b + 16), q1);
        sa = mac16(sa, vld1q_s8(a + 32), q2);
        sb = mac16(sb, vld1q_s8(b + 32), q2);
        out[t] = hsum(sa);
        out[t + 1] = hsum(sb);
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32x4_t s = vdupq_n_s32(0);
        s = mac16(s, vld1q_s8(a), q0);
        s = mac16(s, vld1q_s8(a + 16), q1);
        s = mac16(s, vld1q_s8(a + 32), q2);
        out[t] = hsum(s);
    }
}
#define USE_NEON (!tasr_kernel_force_scalar)
#else
#define USE_NEON 0
#endif

// ------------------------------------------------------------------------------------------------ SIMD32 kernels
#if defined(TASR_SIMD32) && !defined(TASR_NEON)
// One 32-bit load brings four int8 values; sxtb16 sign-extends bytes 0 and 2 (and, rotated by 8, bytes 1 and 3) into
// two 16-bit lanes, and smlad adds both 16 x 16 products to a 32-bit accumulator. All integer, so the sums are the C
// kernels' sums exactly (smlad's Q flag only reports overflow, which the |sum| <= kp * 16384 bound rules out). The
// accumulators are named scalars, not arrays, so they stay in registers; each loaded and widened word is shared by
// four rows (register blocking).
static inline uint32_t ld32(const int8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint32_t sx02(uint32_t v) { uint32_t r; __asm__("sxtb16 %0, %1" : "=r"(r) : "r"(v)); return r; }
static inline uint32_t sx13(uint32_t v) { uint32_t r; __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(v)); return r; }
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm__("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
static inline int32_t tail_dot32(const int8_t *a, const int8_t *b, int from, int to)
{
    int32_t s = 0;
    for (int i = from; i < to; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}
// acc + dot4(word a, word b)
#define MAC4(acc, a, b) acc = smlad(sx13(a), sx13(b), smlad(sx02(a), sx02(b), acc))

static void dot_rows_s8_simd32(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    const int k4 = kp & ~3;
    int t = 0;
    for (; t + 4 <= T; t += 4) {   // four rows share each widened weight word
        const int8_t *a0 = x + (size_t)t * ldq, *a1 = a0 + ldq, *a2 = a1 + ldq, *a3 = a2 + ldq;
        int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < k4; i += 4) {
            const uint32_t wv = ld32(w + i), w02 = sx02(wv), w13 = sx13(wv);
            uint32_t v;
            v = ld32(a0 + i); s0 = smlad(sx13(v), w13, smlad(sx02(v), w02, s0));
            v = ld32(a1 + i); s1 = smlad(sx13(v), w13, smlad(sx02(v), w02, s1));
            v = ld32(a2 + i); s2 = smlad(sx13(v), w13, smlad(sx02(v), w02, s2));
            v = ld32(a3 + i); s3 = smlad(sx13(v), w13, smlad(sx02(v), w02, s3));
        }
        out[t] = s0 + tail_dot32(a0, w, k4, kp);
        out[t + 1] = s1 + tail_dot32(a1, w, k4, kp);
        out[t + 2] = s2 + tail_dot32(a2, w, k4, kp);
        out[t + 3] = s3 + tail_dot32(a3, w, k4, kp);
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t s = 0;
        for (int i = 0; i < k4; i += 4) MAC4(s, ld32(a + i), ld32(w + i));
        out[t] = s + tail_dot32(a, w, k4, kp);
    }
}

static void gemm_s8_xr_simd32(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    const int k4 = kp & ~3;
    for (int t = 0; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldq;
        int32_t *o = acc + (size_t)t * nb;
        int j = 0;
        for (; j + 4 <= nb; j += 4) {   // four weight rows share each widened activation word
            const int8_t *w0 = W + (size_t)j * kp, *w1 = w0 + kp, *w2 = w1 + kp, *w3 = w2 + kp;
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < k4; i += 4) {
                const uint32_t av = ld32(a + i), a02 = sx02(av), a13 = sx13(av);
                uint32_t v;
                v = ld32(w0 + i); s0 = smlad(sx13(v), a13, smlad(sx02(v), a02, s0));
                v = ld32(w1 + i); s1 = smlad(sx13(v), a13, smlad(sx02(v), a02, s1));
                v = ld32(w2 + i); s2 = smlad(sx13(v), a13, smlad(sx02(v), a02, s2));
                v = ld32(w3 + i); s3 = smlad(sx13(v), a13, smlad(sx02(v), a02, s3));
            }
            o[j] = s0 + tail_dot32(a, w0, k4, kp);
            o[j + 1] = s1 + tail_dot32(a, w1, k4, kp);
            o[j + 2] = s2 + tail_dot32(a, w2, k4, kp);
            o[j + 3] = s3 + tail_dot32(a, w3, k4, kp);
        }
        for (; j < nb; j++) {
            const int8_t *w = W + (size_t)j * kp;
            int32_t s = 0;
            for (int i = 0; i < k4; i += 4) MAC4(s, ld32(a + i), ld32(w + i));
            o[j] = s + tail_dot32(a, w, k4, kp);
        }
    }
}

static void dot48_rows_simd32(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    int t = 0;
    for (; t + 2 <= T; t += 2) {   // two rows share each widened query word
        const int8_t *a = x + (size_t)t * ldx, *b = a + ldx;
        int32_t sa = 0, sb = 0;
        for (int i = 0; i < 48; i += 4) {
            const uint32_t qv = ld32(q + i), q02 = sx02(qv), q13 = sx13(qv);
            uint32_t v;
            v = ld32(a + i); sa = smlad(sx13(v), q13, smlad(sx02(v), q02, sa));
            v = ld32(b + i); sb = smlad(sx13(v), q13, smlad(sx02(v), q02, sb));
        }
        out[t] = sa;
        out[t + 1] = sb;
    }
    for (; t < T; t++) {
        const int8_t *a = x + (size_t)t * ldx;
        int32_t s = 0;
        for (int i = 0; i < 48; i += 4) MAC4(s, ld32(a + i), ld32(q + i));
        out[t] = s;
    }
}
#define USE_SIMD32 (!tasr_kernel_force_scalar)
#endif

// ------------------------------------------------------------------------------------------------ public names
void tasr_dot_rows_s8(const int8_t *w, const int8_t *x, int ldq, int T, int kp, int32_t *out)
{
    KSTAT(0, kp, T, 1, ldq, (unsigned long long)T * kp);
#ifdef TASR_NEON
    if (USE_NEON) { dot_rows_s8_neon(w, x, ldq, T, kp, out); return; }
#elif defined(TASR_SIMD32)
    if (USE_SIMD32) { dot_rows_s8_simd32(w, x, ldq, T, kp, out); return; }
#endif
    tasr_dot_rows_s8_scalar(w, x, ldq, T, kp, out);
}

void tasr_gemm_s8_xr(const int8_t *W, int kp, int nb, const int8_t *x, int ldq, int T, int32_t *acc)
{
    KSTAT(1, kp, T, nb, ldq, (unsigned long long)T * kp * nb);
#ifdef TASR_NEON
    if (USE_NEON) { gemm_s8_xr_neon(W, kp, nb, x, ldq, T, acc); return; }
#elif defined(TASR_SIMD32)
    if (USE_SIMD32) { gemm_s8_xr_simd32(W, kp, nb, x, ldq, T, acc); return; }
#endif
    tasr_gemm_s8_xr_scalar(W, kp, nb, x, ldq, T, acc);
}

void tasr_dot48_rows(const int8_t *q, const int8_t *x, int ldx, int T, int32_t *out)
{
    KSTAT(2, 48, T, 1, ldx, (unsigned long long)T * 48);
#ifdef TASR_NEON
    if (USE_NEON) { dot48_rows_neon(q, x, ldx, T, out); return; }
#elif defined(TASR_SIMD32)
    if (USE_SIMD32) { dot48_rows_simd32(q, x, ldx, T, out); return; }
#endif
    tasr_dot48_rows_scalar(q, x, ldx, T, out);
}

void tasr_gemm_blk16(const int8_t *tile, int kp, const int8_t *x, int ldq, int T, int32_t *out)
{
    KSTAT(3, kp, T, 16, ldq, (unsigned long long)T * kp * 16);
    tasr_gemm_blk16_scalar(tile, kp, x, ldq, T, out);
}
