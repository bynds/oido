// libm_fingerprint: hashes of the float results of the libm calls the NeMo engine makes at run time, over fixed
// input sweeps. Two C libraries that print the same line give the engine identical inputs from these calls.
//
// Why: the engine's log-mel front end calls logf (tasr_nemo.c), and its outputs are requantized to int8, so a
// one-ulp difference can change a transcript. glibc 2.21 (Jibo's) and glibc 2.39 differ by one ulp on about 1.9%
// of the logf sweep below and on 202 of the expf inputs; the ARMv7 build is otherwise bit-identical to the host.
// Run it on the robot to learn which emulated reference Jibo should match.
//
// usage: libm_fingerprint [-v]     (-v also prints every input and result, for diffing two libraries)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static float from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t fnv(uint32_t h, uint32_t v) { return (h ^ v) * 16777619u; }

int main(int argc, char **argv)
{
    const int verbose = argc > 1 && !strcmp(argv[1], "-v");
    // logf over the log-mel argument range (power + 2^-24): every 64th float in [2^-24, ~1.1e12)
    uint32_t hl = 2166136261u, nl = 0;
    for (uint32_t b = bits(5.9604644775390625e-08f); b < bits(1.0995e12f); b += 64, nl++) {
        const uint32_t r = bits(logf(from_bits(b)));
        hl = fnv(hl, r);
        if (verbose) printf("logf %08x %08x\n", b, r);
    }
    // expf over the softmax / beam-decoder range [-80, 0] in steps of 1/4096
    uint32_t he = 2166136261u, ne = 0;
    for (int i = -80 * 4096; i <= 0; i++, ne++) {
        const float x = i / 4096.0f;
        const uint32_t r = bits(expf(x));
        he = fnv(he, r);
        if (verbose) printf("expf %08x %08x\n", bits(x), r);
    }
    printf("logf %u inputs %08x; expf %u inputs %08x\n", (unsigned)nl, (unsigned)hl, (unsigned)ne, (unsigned)he);
    return 0;
}
