// oido_service: a persistent Oído recognizer process for the Jibo port. One model, one recognition stream, one
// inference worker; audio in, JSON-line transcript events out. Not yet connected to Jibo's audio service: the audio
// endpoint, format and channel are still to be recovered on the robot, so input is plain 16 kHz mono PCM16.
//
// usage: oido_service [--socket PATH] [--max-seconds S] [--queue-seconds Q] [--model-id ID] model.tnm
//
// Transport: stdin/stdout by default; with --socket, a Unix-domain stream socket created mode 0600 (refused if PATH
// exists and is not a socket), one client at a time, the model staying loaded between clients. The model path is
// fixed at startup; nothing a client sends can name a file.
//
// Requests, one text line each (at most 64 bytes), audio followed by its payload:
//   audio N        then N little-endian int16 samples (1 <= N <= 16000), 16 kHz mono
//   end            end of utterance (the endpoint is the client's decision): the final transcript follows
//   reset          discard the current utterance without a transcript
//   status         health and counters
//   quit           close (stdin mode: exit; socket mode: next client)
// Anything else, or a malformed header, is a protocol error: an "error" event, then the connection closes. End of
// input finalizes a pending utterance (endpoint "eof").
//
// Events, one JSON object per line: ready, partial (streaming models: provisional text, never to trigger actions),
// final, overrun, reset, status, error, shutdown. A final carries only measured fields:
//   {"event":"final","utterance_id":"u0001","text":"...","endpoint":"end","model_id":"...","model_sha256":"...",
//    "sample_rate":16000,"audio_samples":N,"audio_discontinuity":false,"text_truncated":false,"decoder":"ctc_greedy",
//    "confidence":null,"mode":"stream","timing":{"compute_wall_ms":..,"finalization_ms":..,"endpoint_to_final_ms":..,
//    "queue_wait_max_ms":..}}
// endpoint_to_final_ms runs from the endpoint request reaching the service to the final being written: the
// end-of-speech-to-transcript latency the service adds, given the client's endpoint decision.
// confidence is null on purpose: greedy CTC gives no calibrated probability of being right.
//
// Bounds: a reader thread moves requests into a queue holding at most --queue-seconds of audio (default 4), so
// reading never waits for inference. Audio that does not fit is dropped and the drop is recorded at its place in
// the queue; the utterance it falls in is marked audio_discontinuity and an "overrun" event says how many samples
// were lost. Audio is never silently spliced. An utterance reaching --max-seconds (default 20) is finalized there
// (endpoint "max_duration") and the rest starts a new one.
//
// Models: a streaming model (oido_stream.tnm) is decoded as audio arrives, with partial events after each encoder
// chunk; any other TNM1 model (nemo8.tnm) buffers the utterance and transcribes it at the endpoint (final only).
// Audio and transcripts are never logged; stderr gets startup and connection lines only.
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include "port_util.h"
#include "sha256.h"
#include "tasr_nemo.h"

#define MAX_BLOCK 16000
#define MAX_CTRL 256
#define TEXT_MAX 4096
#define HEADER_MAX 64

// ------------------------------------------------------------------------------------------------ request queue
enum { M_AUDIO, M_GAP, M_END, M_RESET, M_STATUS, M_QUIT, M_EOF, M_FATAL };
typedef struct msg {
    int type, n;           // n: samples (audio) or dropped samples (gap)
    uint64_t t_enq;        // enqueue time, ns
    char err[96];
    struct msg *next;
    int16_t pcm[];
} msg_t;
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    msg_t *head, *tail;
    long samples, cap;     // queued audio samples, capacity
    int ctrl;              // queued non-audio messages
    int fd;
} queue_t;

static void q_push_locked(queue_t *q, msg_t *m)
{
    m->next = NULL;
    m->t_enq = pu_now_ns(CLOCK_MONOTONIC);
    if (q->tail) q->tail->next = m; else q->head = m;
    q->tail = m;
    pthread_cond_signal(&q->cv);
}
// control message (anything but audio); a gap merges into a gap already at the tail. 0 = queued, -1 = too many
static int q_push_ctrl(queue_t *q, int type, int n, const char *err)
{
    pthread_mutex_lock(&q->mu);
    if (type == M_GAP && q->tail && q->tail->type == M_GAP) {
        q->tail->n += n;
        pthread_mutex_unlock(&q->mu);
        return 0;
    }
    if (q->ctrl >= MAX_CTRL && type != M_FATAL && type != M_EOF) { pthread_mutex_unlock(&q->mu); return -1; }
    msg_t *m = calloc(1, sizeof(msg_t));
    if (!m) { pthread_mutex_unlock(&q->mu); return -1; }
    m->type = type;
    m->n = n;
    if (err) snprintf(m->err, sizeof(m->err), "%s", err);
    q->ctrl++;
    q_push_locked(q, m);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

// ------------------------------------------------------------------------------------------------ reader thread
typedef struct {
    int fd;
    uint8_t buf[65536];
    size_t pos, len;
} rd_t;
static int rd_fill(rd_t *r)
{
    if (r->pos < r->len) return 1;
    for (;;) {
        const ssize_t k = read(r->fd, r->buf, sizeof(r->buf));
        if (k > 0) { r->pos = 0; r->len = (size_t)k; return 1; }
        if (k == 0) return 0;
        if (errno != EINTR) return -1;
    }
}
static int rd_bytes(rd_t *r, uint8_t *dst, size_t n)  // 1 ok, 0 eof, -1 error
{
    while (n) {
        const int f = rd_fill(r);
        if (f <= 0) return f;
        size_t k = r->len - r->pos < n ? r->len - r->pos : n;
        memcpy(dst, r->buf + r->pos, k);
        r->pos += k; dst += k; n -= k;
    }
    return 1;
}
static int rd_line(rd_t *r, char *line, size_t cap)  // 1 ok, 0 eof at a line start, -1 error/overlong/eof mid-line
{
    size_t n = 0;
    for (;;) {
        const int f = rd_fill(r);
        if (f < 0) return -1;
        if (f == 0) return n ? -1 : 0;
        const char c = (char)r->buf[r->pos++];
        if (c == '\n') { line[n] = 0; return 1; }
        if (n + 1 >= cap) return -1;
        line[n++] = c;
    }
}

static void *reader_main(void *arg)
{
    queue_t *q = arg;
    rd_t *r = calloc(1, sizeof(rd_t));
    if (!r) { q_push_ctrl(q, M_FATAL, 0, "out of memory"); return NULL; }
    pthread_cleanup_push(free, r);   // the worker cancels this thread while it waits in read()
    r->fd = q->fd;
    char line[HEADER_MAX];
    for (;;) {
        const int f = rd_line(r, line, sizeof(line));
        if (f == 0) { q_push_ctrl(q, M_EOF, 0, NULL); break; }
        if (f < 0) { q_push_ctrl(q, M_FATAL, 0, "malformed or overlong request line"); break; }
        if (!strncmp(line, "audio ", 6)) {
            char *end;
            const long n = strtol(line + 6, &end, 10);
            if (*end || n < 1 || n > MAX_BLOCK) { q_push_ctrl(q, M_FATAL, 0, "audio needs 1..16000 samples"); break; }
            msg_t *m = malloc(sizeof(msg_t) + sizeof(int16_t) * (size_t)n);
            if (!m) { q_push_ctrl(q, M_FATAL, 0, "out of memory"); break; }
            uint8_t raw[2 * MAX_BLOCK];
            if (rd_bytes(r, raw, 2 * (size_t)n) != 1) { free(m); q_push_ctrl(q, M_FATAL, 0, "audio payload cut short"); break; }
            for (long i = 0; i < n; i++) m->pcm[i] = (int16_t)(raw[2 * i] | raw[2 * i + 1] << 8);
            m->type = M_AUDIO;
            m->n = (int)n;
            pthread_mutex_lock(&q->mu);
            if (q->samples + n > q->cap) {   // full: drop, and record the gap where the audio would have been
                pthread_mutex_unlock(&q->mu);
                free(m);
                if (q_push_ctrl(q, M_GAP, (int)n, NULL)) { q_push_ctrl(q, M_FATAL, 0, "request queue overflow"); break; }
                continue;
            }
            q->samples += n;
            q_push_locked(q, m);
            pthread_mutex_unlock(&q->mu);
        } else {
            int type = -1;
            if (!strcmp(line, "end")) type = M_END;
            else if (!strcmp(line, "reset")) type = M_RESET;
            else if (!strcmp(line, "status")) type = M_STATUS;
            else if (!strcmp(line, "quit")) type = M_QUIT;
            if (type < 0) { q_push_ctrl(q, M_FATAL, 0, "unknown request"); break; }
            if (q_push_ctrl(q, type, 0, NULL)) { q_push_ctrl(q, M_FATAL, 0, "request queue overflow"); break; }
            if (type == M_QUIT) break;
        }
    }
    pthread_cleanup_pop(1);
    return NULL;
}

// ------------------------------------------------------------------------------------------------ output
static int out_fd = 1, out_broken;
static void out_write(const char *s, size_t n)
{
    while (n && !out_broken) {
        const ssize_t k = write(out_fd, s, n);
        if (k > 0) { s += k; n -= (size_t)k; }
        else if (k < 0 && errno == EINTR) continue;
        else out_broken = 1;
    }
}
typedef struct {
    char *p;
    size_t n, cap;
} sb_t;
static void sb_put(sb_t *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n + 1) c *= 2;
        char *q = realloc(b->p, c);
        if (!q) abort();
        b->p = q;
        b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}
static void sb_printf(sb_t *b, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    const int k = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sb_put(b, tmp, k < (int)sizeof(tmp) ? (size_t)k : sizeof(tmp) - 1);
}
// JSON string: quotes, backslash and control characters escaped; invalid UTF-8 bytes become U+FFFD
static void sb_json_str(sb_t *b, const char *s)
{
    sb_put(b, "\"", 1);
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        const unsigned c = *p;
        if (c == '"' || c == '\\') { char e[2] = {'\\', (char)c}; sb_put(b, e, 2); p++; continue; }
        if (c < 0x20) { sb_printf(b, "\\u%04x", c); p++; continue; }
        if (c < 0x80) { sb_put(b, (const char *)p, 1); p++; continue; }
        int len = c >= 0xf0 && c <= 0xf4 ? 4 : c >= 0xe0 ? 3 : c >= 0xc2 && c <= 0xdf ? 2 : 0;
        int ok = len > 0;
        for (int i = 1; ok && i < len; i++) ok = (p[i] & 0xc0) == 0x80;
        if (ok && len == 3) ok = !(c == 0xe0 && p[1] < 0xa0) && !(c == 0xed && p[1] >= 0xa0);
        if (ok && len == 4) ok = !(c == 0xf0 && p[1] < 0x90) && !(c == 0xf4 && p[1] >= 0x90);
        if (ok) { sb_put(b, (const char *)p, (size_t)len); p += len; }
        else { sb_put(b, "\\ufffd", 6); p++; }
    }
    sb_put(b, "\"", 1);
}
static void emit(sb_t *b)
{
    sb_put(b, "\n", 1);
    out_write(b->p, b->n);
    b->n = 0;
}

// ------------------------------------------------------------------------------------------------ recognizer
typedef struct {
    const char *model_id;
    char sha[65];
    tasr_nemo_t *m;
    tasr_nemo_stream_t *s;   // streaming models
    int16_t *buf;            // utterance models: buffered audio
    long max_samples, queue_cap;
    // utterance
    int active, discontinuity, last_frames;
    long samples, utt_count, overruns, dropped;
    uint64_t compute_ns, wait_max_ns, t_start;
    char partial[TEXT_MAX], text[TEXT_MAX];
} rec_t;

static void utt_begin(rec_t *r)
{
    r->active = 1;
    r->discontinuity = 0;
    r->samples = 0;
    r->compute_ns = r->wait_max_ns = 0;
    r->last_frames = 0;
    r->partial[0] = 0;
    r->utt_count++;
    if (r->s) tasr_nemo_stream_reset(r->s);
}
static void utt_id(const rec_t *r, char *id, size_t cap) { snprintf(id, cap, "u%04ld", r->utt_count); }

// endpoint_wait_ns: how long the endpoint request waited in the queue (0 when the service set the endpoint itself)
static void finalize(rec_t *r, sb_t *b, const char *endpoint, uint64_t endpoint_wait_ns)
{
    char id[32];
    if (!r->active) { r->utt_count++; r->samples = 0; r->compute_ns = r->wait_max_ns = 0; r->discontinuity = 0; }
    utt_id(r, id, sizeof(id));
    const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
    int truncated = 0;
    r->text[0] = 0;
    if (r->s) {
        if (r->active) tasr_nemo_stream_finish(r->s, r->text, sizeof(r->text));
        truncated = r->active && tasr_nemo_stream_truncated(r->s);
    } else if (r->samples) {
        pu_alloc_fatal = 1;
        tasr_nemo_transcribe(r->m, r->buf, (int)r->samples, NULL, r->text, sizeof(r->text), NULL, 0, NULL);
        pu_alloc_fatal = 0;
        truncated = strlen(r->text) >= sizeof(r->text) - 256;
    }
    const uint64_t fin = pu_now_ns(CLOCK_MONOTONIC) - t0;
    r->compute_ns += fin;
    sb_printf(b, "{\"event\":\"final\",\"utterance_id\":\"%s\",\"text\":", id);
    sb_json_str(b, r->text);
    sb_printf(b, ",\"endpoint\":\"%s\",\"model_id\":", endpoint);
    sb_json_str(b, r->model_id);
    sb_printf(b, ",\"model_sha256\":\"%s\",\"sample_rate\":16000,\"audio_samples\":%ld,\"audio_discontinuity\":%s,"
                 "\"text_truncated\":%s,\"decoder\":\"ctc_greedy\",\"confidence\":null,\"mode\":\"%s\","
                 "\"timing\":{\"compute_wall_ms\":%.1f,\"finalization_ms\":%.1f,\"endpoint_to_final_ms\":%.1f,"
                 "\"queue_wait_max_ms\":%.1f}}",
              r->sha, r->samples, r->discontinuity ? "true" : "false", truncated ? "true" : "false",
              r->s ? "stream" : "utterance", r->compute_ns / 1e6, fin / 1e6, (endpoint_wait_ns + fin) / 1e6,
              r->wait_max_ns / 1e6);
    emit(b);
    r->active = 0;
}

static void on_audio(rec_t *r, sb_t *b, const int16_t *pcm, int n)
{
    while (n > 0) {
        if (!r->active) utt_begin(r);
        long room = r->max_samples - r->samples;
        const int k = n < room ? n : (int)room;
        const uint64_t t0 = pu_now_ns(CLOCK_MONOTONIC);
        if (r->s) {
            const int frames = tasr_nemo_stream_feed(r->s, pcm, k);
            if (frames != r->last_frames) {
                r->last_frames = frames;
                char now[TEXT_MAX];
                tasr_nemo_stream_text(r->s, now, sizeof(now));
                if (strcmp(now, r->partial)) {
                    memcpy(r->partial, now, sizeof(now));
                    char id[32];
                    utt_id(r, id, sizeof(id));
                    sb_printf(b, "{\"event\":\"partial\",\"utterance_id\":\"%s\",\"text\":", id);
                    sb_json_str(b, now);
                    sb_printf(b, ",\"provisional\":true,\"audio_samples\":%ld,\"frames\":%d}", r->samples + k, frames);
                    emit(b);
                }
            }
        } else {
            memcpy(r->buf + r->samples, pcm, sizeof(int16_t) * (size_t)k);
        }
        r->compute_ns += pu_now_ns(CLOCK_MONOTONIC) - t0;
        r->samples += k;
        pcm += k;
        n -= k;
        if (r->samples >= r->max_samples) finalize(r, b, "max_duration", 0);
    }
}

static void status(rec_t *r, sb_t *b, const queue_t *q, uint64_t t_start)
{
    long rss_kb = -1, hwm_kb = -1;   // resident set now and its peak, from /proc (-1 if unavailable)
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "VmRSS:", 6)) rss_kb = strtol(line + 6, NULL, 10);
            if (!strncmp(line, "VmHWM:", 6)) hwm_kb = strtol(line + 6, NULL, 10);
        }
        fclose(f);
    }
    char id[32];
    utt_id(r, id, sizeof(id));
    sb_printf(b, "{\"event\":\"status\",\"state\":\"%s\",\"utterance_id\":\"%s\",\"utterance_samples\":%ld,"
                 "\"utterances\":%ld,\"overruns\":%ld,\"dropped_samples\":%ld,\"queued_samples\":%ld,"
                 "\"queue_capacity\":%ld,\"max_utterance_samples\":%ld,\"mode\":\"%s\",\"kernels\":\"%s\","
                 "\"rss_kb\":%ld,\"rss_peak_kb\":%ld,\"engine_bytes\":%zu,\"uptime_s\":%.1f,\"model_id\":",
              r->active ? "listening" : "idle", r->active ? id : "", r->active ? r->samples : 0, r->utt_count,
              r->overruns, r->dropped, q->samples, q->cap, r->max_samples, r->s ? "stream" : "utterance",
              pu_kernels_name(), rss_kb, hwm_kb, pu_alloc_cur, (pu_now_ns(CLOCK_MONOTONIC) - t_start) / 1e9);
    sb_json_str(b, r->model_id);
    sb_printf(b, ",\"model_sha256\":\"%s\"}", r->sha);
    emit(b);
}

static volatile sig_atomic_t stop_flag;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

// Serve one connection (stdin/stdout or one socket client). Returns 0 on a clean end, 1 after a protocol error.
static int serve(rec_t *r, int in, int out, long queue_cap, uint64_t t_start)
{
    queue_t q;
    memset(&q, 0, sizeof(q));
    pthread_mutex_init(&q.mu, NULL);
    pthread_cond_init(&q.cv, NULL);
    q.cap = queue_cap;
    q.fd = in;
    out_fd = out;
    out_broken = 0;
    pthread_t th;
    if (pthread_create(&th, NULL, reader_main, &q)) { fprintf(stderr, "cannot start reader thread\n"); return 1; }
    sb_t b = {0};
    sb_printf(&b, "{\"event\":\"ready\",\"model_id\":");
    sb_json_str(&b, r->model_id);
    sb_printf(&b, ",\"model_sha256\":\"%s\",\"mode\":\"%s\",\"sample_rate\":16000,\"max_utterance_samples\":%ld,"
                  "\"queue_capacity\":%ld,\"max_block\":%d,\"kernels\":\"%s\"}",
              r->sha, r->s ? "stream" : "utterance", r->max_samples, queue_cap, MAX_BLOCK, pu_kernels_name());
    emit(&b);
    int rc = 0, done = 0;
    while (!done) {
        pthread_mutex_lock(&q.mu);
        while (!q.head && !stop_flag) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100000000;
            if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
            pthread_cond_timedwait(&q.cv, &q.mu, &ts);
        }
        if (stop_flag) { pthread_mutex_unlock(&q.mu); break; }
        msg_t *m = q.head;
        q.head = m->next;
        if (!q.head) q.tail = NULL;
        if (m->type == M_AUDIO) q.samples -= m->n; else q.ctrl--;
        pthread_mutex_unlock(&q.mu);
        const uint64_t wait = pu_now_ns(CLOCK_MONOTONIC) - m->t_enq;
        char id[32];
        switch (m->type) {
        case M_AUDIO:
            if (!r->active) utt_begin(r);
            if (wait > r->wait_max_ns) r->wait_max_ns = wait;
            on_audio(r, &b, m->pcm, m->n);
            break;
        case M_GAP:
            r->overruns++;
            r->dropped += m->n;
            if (!r->active) utt_begin(r);
            r->discontinuity = 1;
            utt_id(r, id, sizeof(id));
            sb_printf(&b, "{\"event\":\"overrun\",\"utterance_id\":\"%s\",\"dropped_samples\":%d}", id, m->n);
            emit(&b);
            break;
        case M_END: finalize(r, &b, "end", wait); break;
        case M_RESET:
            utt_id(r, id, sizeof(id));
            sb_printf(&b, "{\"event\":\"reset\",\"utterance_id\":\"%s\",\"discarded_samples\":%ld}", id,
                      r->active ? r->samples : 0);
            emit(&b);
            r->active = 0;
            break;
        case M_STATUS: status(r, &b, &q, t_start); break;
        case M_EOF:
            if (r->active) finalize(r, &b, "eof", wait);
            done = 1;
            break;
        case M_QUIT: done = 1; break;
        default:
            sb_printf(&b, "{\"event\":\"error\",\"message\":");
            sb_json_str(&b, m->err);
            sb_put(&b, "}", 1);
            emit(&b);
            r->active = 0;
            rc = 1;
            done = 1;
            break;
        }
        free(m);
        if (out_broken) done = 1;
    }
    if (stop_flag) { sb_printf(&b, "{\"event\":\"shutdown\"}"); emit(&b); }
    // the reader ends on its own at quit/eof/error; otherwise (signal, broken output) stop it by closing input
    if (!done || out_broken || stop_flag) shutdown(in, SHUT_RD);
    pthread_cancel(th);
    pthread_join(th, NULL);
    for (msg_t *m = q.head; m;) { msg_t *n = m->next; free(m); m = n; }
    r->active = 0;
    free(b.p);
    return rc;
}

int main(int argc, char **argv)
{
    const char *sock_path = NULL, *model_id = NULL;
    double max_seconds = 20.0, queue_seconds = 4.0;
    int i = 1;
    for (; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--socket") && i + 1 < argc) sock_path = argv[++i];
        else if (!strcmp(argv[i], "--max-seconds") && i + 1 < argc) max_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--queue-seconds") && i + 1 < argc) queue_seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--model-id") && i + 1 < argc) model_id = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (argc - i != 1 || !(max_seconds >= 1 && max_seconds <= 120) || !(queue_seconds >= 1 && queue_seconds <= 60)) {
        fprintf(stderr, "usage: %s [--socket PATH] [--max-seconds 1..120] [--queue-seconds 1..60] [--model-id ID] model.tnm\n",
                argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    pu_alloc_install();
    if (pu_kernels_init()) return 2;

    rec_t r;
    memset(&r, 0, sizeof(r));
    size_t msz;
    uint8_t *mb = pu_read_file(argv[i], (size_t)256 << 20, &msz);
    if (!mb) return 1;
    r.m = tasr_nemo_load(mb, msz);
    if (!r.m) { fprintf(stderr, "%s: model load failed\n", argv[i]); return 1; }
    sha256_hex(mb, msz, r.sha);
    static char idbuf[128];
    if (!model_id) {
        snprintf(idbuf, sizeof(idbuf), "%s", pu_base_name(argv[i]));
        char *dot = strrchr(idbuf, '.');
        if (dot) *dot = 0;
        model_id = idbuf;
    }
    r.model_id = model_id;
    r.max_samples = (long)(max_seconds * 16000);
    if (tasr_nemo_stream_supported(r.m)) {
        r.s = tasr_nemo_stream_new(r.m, 0, 0, NULL);
        if (!r.s) { fprintf(stderr, "stream creation failed\n"); return 1; }
    } else {
        r.buf = malloc(sizeof(int16_t) * (size_t)r.max_samples);
        if (!r.buf) { fprintf(stderr, "out of memory\n"); return 1; }
    }
    const long queue_cap = (long)(queue_seconds * 16000);
    const uint64_t t_start = pu_now_ns(CLOCK_MONOTONIC);
    fprintf(stderr, "oido_service: model %s (%s), %s mode, %zu bytes of engine state\n", model_id, r.sha,
            r.s ? "stream" : "utterance", pu_alloc_cur);

    int rc = 0;
    if (!sock_path) {
        rc = serve(&r, 0, 1, queue_cap, t_start);
    } else {
        struct stat st;
        if (lstat(sock_path, &st) == 0) {
            if (!S_ISSOCK(st.st_mode)) { fprintf(stderr, "%s exists and is not a socket\n", sock_path); return 1; }
            unlink(sock_path);
        }
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        if (strlen(sock_path) >= sizeof(addr.sun_path)) { fprintf(stderr, "socket path too long\n"); return 1; }
        strcpy(addr.sun_path, sock_path);
        const int ls = socket(AF_UNIX, SOCK_STREAM, 0);
        const mode_t old = umask(077);
        const int bound = ls >= 0 && bind(ls, (struct sockaddr *)&addr, sizeof(addr)) == 0;
        umask(old);
        if (!bound || chmod(sock_path, 0600) || listen(ls, 1)) { fprintf(stderr, "%s: %s\n", sock_path, strerror(errno)); return 1; }
        fprintf(stderr, "oido_service: listening on %s (mode 0600)\n", sock_path);
        while (!stop_flag) {
            const int c = accept(ls, NULL, NULL);
            if (c < 0) { if (errno == EINTR) continue; fprintf(stderr, "accept: %s\n", strerror(errno)); rc = 1; break; }
            fprintf(stderr, "oido_service: client connected\n");
            serve(&r, c, c, queue_cap, t_start);
            close(c);
            fprintf(stderr, "oido_service: client closed\n");
        }
        close(ls);
        unlink(sock_path);
    }
    tasr_nemo_stream_free(r.s);
    free(r.buf);
    tasr_nemo_free(r.m);
    free(mb);
    return rc;
}
