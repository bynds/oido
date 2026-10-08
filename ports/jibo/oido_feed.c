// oido_feed: write WAV files to stdout in oido_service's request protocol, for tests and on-robot measurement
// without Python: `oido_feed --realtime a.wav b.wav | oido_service model.tnm`.
//
// usage: oido_feed [--block N (default 320)] [--realtime] [--gap-ms M (default 500)] [--status] file.wav ...
//   each file is one utterance: "audio N" blocks then "end"; --realtime paces blocks at 16 kHz (as live capture
//   would) and waits --gap-ms between utterances; --status asks for a status event after each utterance.
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "port_util.h"

static int write_all(const void *p, size_t n)
{
    const char *c = p;
    while (n) {
        const ssize_t k = write(1, c, n);
        if (k > 0) { c += k; n -= (size_t)k; }
        else if (k < 0 && errno == EINTR) continue;
        else return -1;
    }
    return 0;
}
static void sleep_until(uint64_t t_ns)
{
    const uint64_t now = pu_now_ns(CLOCK_MONOTONIC);
    if (t_ns <= now) return;
    struct timespec ts = {(time_t)((t_ns - now) / 1000000000ull), (long)((t_ns - now) % 1000000000ull)};
    while (nanosleep(&ts, &ts) && errno == EINTR) {}
}

int main(int argc, char **argv)
{
    int block = 320, realtime = 0, gap_ms = 500, status = 0, i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc) block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--realtime")) realtime = 1;
        else if (!strcmp(argv[i], "--gap-ms") && i + 1 < argc) gap_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--status")) status = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (i >= argc || block < 1 || block > 16000 || gap_ms < 0) {
        fprintf(stderr, "usage: %s [--block N] [--realtime] [--gap-ms M] [--status] file.wav ...\n", argv[0]);
        return 2;
    }
    for (; i < argc; i++) {
        int ns = 0;
        int16_t *pcm = pu_read_wav(argv[i], 600L * 16000, &ns);
        if (!pcm) return 1;
        const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
        for (int o = 0; o < ns; o += block) {
            const int k = ns - o < block ? ns - o : block;
            if (realtime) sleep_until(t0 + (uint64_t)(o + k) * 1000000000ull / 16000);   // block complete at 16 kHz
            char hdr[32];
            const int hl = snprintf(hdr, sizeof(hdr), "audio %d\n", k);
            uint8_t raw[2 * 16000];
            for (int j = 0; j < k; j++) { raw[2 * j] = (uint8_t)pcm[o + j]; raw[2 * j + 1] = (uint8_t)((uint16_t)pcm[o + j] >> 8); }
            if (write_all(hdr, (size_t)hl) || write_all(raw, 2 * (size_t)k)) { free(pcm); return 1; }
        }
        if (write_all("end\n", 4) || (status && write_all("status\n", 7))) { free(pcm); return 1; }
        free(pcm);
        if (realtime && gap_ms) sleep_until(pu_now_ns(CLOCK_MONOTONIC) + (uint64_t)gap_ms * 1000000ull);
    }
    return 0;
}
