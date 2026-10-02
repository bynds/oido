// Utterance-level engine for NVIDIA NeMo Conformer-CTC small (13 M params, int8) — higher-accuracy mode.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tasr_nemo tasr_nemo_t;
struct tasr_decoder;

tasr_nemo_t *tasr_nemo_load(const uint8_t *blob, size_t size);
void tasr_nemo_free(tasr_nemo_t *m);
size_t tasr_nemo_weight_bytes(const tasr_nemo_t *m);
size_t tasr_nemo_blob_bytes(const tasr_nemo_t *m);
// relocate weight matrices to tasr_alloc(kind 0) memory up to budget bytes (front end first)
size_t tasr_nemo_place_weights(tasr_nemo_t *m, size_t budget);

// Transcribe one utterance (16 kHz PCM). dec: optional CTC beam decoder (LM over the model's 1024 tokens), or NULL
// for greedy. logit_sink (optional): raw logits [frames][1025] (blank last). Returns encoder frames.
int tasr_nemo_transcribe(const tasr_nemo_t *m, const int16_t *pcm, int n, struct tasr_decoder *dec, char *text,
                         int maxlen, float *logit_sink, int max_frames, int *n_frames);

// Streaming (models exported from train_nemo_stream.py --stream): feed audio as it arrives; the encoder runs every
// `chunk` encoder frames (40 ms each) with `left` frames of attention context (0 = the model's defaults, 16 and 128).
// tasr_nemo_stream_text gives the transcript so far; tasr_nemo_stream_finish runs the last partial chunk and returns
// the final text. NULL for models without streaming support. dec: optional CTC beam decoder, as for transcribe.
typedef struct tasr_nemo_stream tasr_nemo_stream_t;
tasr_nemo_stream_t *tasr_nemo_stream_new(const tasr_nemo_t *m, int chunk, int left, struct tasr_decoder *dec);
void tasr_nemo_stream_reset(tasr_nemo_stream_t *s);
int tasr_nemo_stream_feed(tasr_nemo_stream_t *s, const int16_t *pcm, int n);   // returns encoder frames decoded so far
int tasr_nemo_stream_text(tasr_nemo_stream_t *s, char *text, int maxlen);      // partial transcript
int tasr_nemo_stream_finish(tasr_nemo_stream_t *s, char *text, int maxlen);    // end of utterance: final transcript
void tasr_nemo_stream_set_sink(tasr_nemo_stream_t *s, float *logits, int max_frames);  // optional raw logits
void tasr_nemo_stream_free(tasr_nemo_stream_t *s);
int tasr_nemo_stream_supported(const tasr_nemo_t *m);

// profiling (TASR_PROFILE builds): per-stage cycle totals
const char *tasr_nemo_profile_name(int i);
uint64_t tasr_nemo_profile_value(int i);

#ifdef __cplusplus
}
#endif
