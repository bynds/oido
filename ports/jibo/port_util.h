// port_util: shared pieces of the Jibo port's programs (oido_cli, oido_stream_replay, oido_service, tests).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <time.h>

uint64_t pu_now_ns(clockid_t id);   // CLOCK_MONOTONIC (wall) or CLOCK_PROCESS_CPUTIME_ID, 64-bit nanoseconds

// Engine allocation accounting, installed as tasr_alloc/tasr_free. Keeps the default allocator's contract (zeroed,
// 16-byte aligned). tasr_nemo_load and tasr_nemo_stream_new handle a failed allocation; tasr_nemo_transcribe does not
// check its working buffers, so a caller sets pu_alloc_fatal around it: a failure then ends the process with a
// message instead of writing through NULL.
void pu_alloc_install(void);
extern size_t pu_alloc_cur, pu_alloc_peak, pu_alloc_calls;
extern int pu_alloc_fatal;

// Whole file into a 16-byte-aligned buffer (free()); NULL with a message on any error or a file above max_bytes.
uint8_t *pu_read_file(const char *path, size_t max_bytes, size_t *size);
// RIFF/WAVE, PCM 16-bit mono 16 kHz, fmt before one well-formed data chunk, at most max_samples samples.
// NULL with a message on anything else (truncated chunks and odd data lengths included). free() the result.
int16_t *pu_read_wav(const char *path, long max_samples, int *ns);
const char *pu_base_name(const char *path);
// Kernel backend from OIDO_KERNELS (scalar|auto) in dispatch builds; prints "kernels: ..." to stderr. 0 on success.
int pu_kernels_init(void);
const char *pu_kernels_name(void);
