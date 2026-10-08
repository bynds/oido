// Port-side controls of the kernel dispatch layer (kernels_neon.c). The switch itself, tasr_kernel_force_scalar, and
// tasr_kernel_backend() are declared in kernels.h under TASR_KERNEL_DISPATCH.
#pragma once
#include <stdio.h>

// per kernel and shape: calls and multiply-accumulates, sorted by share (needs -DTASR_KERNEL_STATS, else a note)
void tasr_kernel_stats_dump(FILE *f);
void tasr_kernel_stats_reset(void);
