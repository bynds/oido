/* icount.h: user-space instructions retired by this process, from the PMU through perf_event_open
 * (PERF_TYPE_HARDWARE / PERF_COUNT_HW_INSTRUCTIONS, kernel excluded). Under perfvm (full-system qemu with
 * -icount shift=0) the count is exact and repeatable; where no PMU is available (qemu-user, most containers)
 * icount_open fails and callers fall back to time. Header-only. After bynds/needle-rs nd_icount.h (dcb8f79). */
#ifndef OIDO_ICOUNT_H
#define OIDO_ICOUNT_H
#include <linux/perf_event.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

long syscall(long number, ...);  /* not declared under strict -D_POSIX_C_SOURCE */

static int icount_fd = -1;

static int icount_open(void)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.type = PERF_TYPE_HARDWARE;
    a.size = sizeof a;
    a.config = PERF_COUNT_HW_INSTRUCTIONS;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    icount_fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
    return icount_fd >= 0;
}

static uint64_t icount_read(void)
{
    uint64_t v = 0;
    if (icount_fd < 0 || read(icount_fd, &v, sizeof v) != (ssize_t)sizeof v) return 0;
    return v;
}
#endif
