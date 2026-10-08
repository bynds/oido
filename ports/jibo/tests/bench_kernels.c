// bench_kernels: wall time of the int8 kernels on the model's hottest shapes, C kernels vs the dispatched ones (NEON
// in a -DTASR_NEON build), and a check that both give the same sums. For the robot: emulator timings mean nothing.
// Shapes and call counts come from a TASR_KERNEL_STATS profile of nemo8.tnm (results/jibo/profile-host-nemo8.tsv).
// Output: one JSON line per shape. usage: bench_kernels [repeats (default 20)]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernels.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    const char *name;
    int fn, kp, T, nb, ld;   // fn: 0 dot_rows_s8, 1 gemm_s8_xr, 2 dot48_rows
} shape_t;

static const shape_t shapes[] = {
    {"proj_k176 (gemm_s8_xr)", 1, 176, 64, 16, 704},
    {"conv2_chunk (dot_rows_s8)", 0, 416, 60, 1, 1584},
    {"ff_out_k704 (dot_rows_s8)", 0, 704, 64, 1, 704},
    {"sub_chunk (dot_rows_s8)", 0, 512, 64, 1, 3520},
    {"att_scores (dot48_rows)", 2, 48, 175, 1, 48},
    {"att_values (dot_rows_s8)", 0, 176, 44, 1, 176},
};

static int8_t *rand_buf(size_t n)
{
    int8_t *p = aligned_alloc(16, (n + 15) & ~(size_t)15);
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)(rand() % 255 - 127);
    return p;
}

static void run(const shape_t *s, const int8_t *w, const int8_t *x, int32_t *out)
{
    switch (s->fn) {
    case 0: tasr_dot_rows_s8(w, x, s->ld, s->T, s->kp, out); break;
    case 1: tasr_gemm_s8_xr(w, s->kp, s->nb, x, s->ld, s->T, out); break;
    default: tasr_dot48_rows(w, x, s->ld, s->T, out); break;
    }
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    srand(1);
    int mismatches = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const shape_t *s = &shapes[i];
        int8_t *w = rand_buf((size_t)s->nb * s->kp + 64), *x = rand_buf((size_t)s->T * s->ld + 64);
        int32_t *o1 = malloc(sizeof(int32_t) * s->T * s->nb), *o2 = malloc(sizeof(int32_t) * s->T * s->nb);
        // inner repeat count chosen so one timed block is ~10-50 ms of scalar work on a 2 GHz core
        const long macs = (long)s->T * s->kp * s->nb;
        const int inner = (int)(20000000L / macs) + 1;
        double best[2] = {1e9, 1e9};
        for (int b = 0; b < 2; b++) {
            tasr_kernel_force_scalar = b == 0;
            run(s, w, x, b ? o2 : o1);   // warm-up, and the result to compare
            for (int r = 0; r < reps; r++) {
                const double t0 = now();
                for (int k = 0; k < inner; k++) run(s, w, x, o2);
                const double dt = (now() - t0) / inner;
                if (dt < best[b]) best[b] = dt;
            }
        }
        tasr_kernel_force_scalar = 1;
        run(s, w, x, o1);
        tasr_kernel_force_scalar = 0;
        run(s, w, x, o2);
        const int same = !memcmp(o1, o2, sizeof(int32_t) * s->T * s->nb);
        mismatches += !same;
        printf("{\"shape\":\"%s\",\"kp\":%d,\"T\":%d,\"nb\":%d,\"ld\":%d,\"macs\":%ld,\"scalar_us\":%.2f,"
               "\"dispatch_us\":%.2f,\"backend\":\"%s\",\"speedup\":%.2f,\"scalar_gmacs\":%.3f,\"dispatch_gmacs\":%.3f,"
               "\"same\":%s}\n",
               s->name, s->kp, s->T, s->nb, s->ld, macs, best[0] * 1e6, best[1] * 1e6, tasr_kernel_backend(),
               best[0] / best[1], macs / best[0] * 1e-9, macs / best[1] * 1e-9, same ? "true" : "false");
        free(w); free(x); free(o1); free(o2);
    }
    return mismatches ? 1 : 0;
}
