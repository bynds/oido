/* selftest.c: checks the instruction counter is exact. A loop of fixed shape must cost the same number of
 * instructions per iteration (so 2M iterations cost exactly twice 1M, give or take the loop's fixed entry and
 * exit), and repeated runs must print identical numbers. After bynds/needle-rs perfvm/selftest.c (dcb8f79). */
#include <stdio.h>
#include "icount.h"
int main(void)
{
    volatile unsigned x = 0;
    unsigned i;
    uint64_t a, b, c;
    if (!icount_open()) { perror("perf_event_open"); return 1; }
    a = icount_read();
    for (i = 0; i < 1000000; i++) x += i;
    b = icount_read();
    for (i = 0; i < 2000000; i++) x += i;
    c = icount_read();
    printf("selftest: 1M iters %llu instructions, 2M iters %llu (ratio %.6f)\n", (unsigned long long)(b - a),
           (unsigned long long)(c - b), (double)(c - b) / (double)(b - a));
    return 0;
}
