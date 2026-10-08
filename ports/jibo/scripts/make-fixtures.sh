#!/usr/bin/env bash
# make-fixtures.sh: the immediate recorded-audio fixture pack, as 16 kHz mono PCM16 WAV in build/fixtures/.
#
# Speech comes from the clips upstream already publishes in docs/samples (with reference transcripts in
# docs/samples/results.json and credits in docs/index.html: LibriSpeech CC-BY-4.0, Common Voice 17 CC0,
# VoxPopuli CC0, AMI CC-BY-4.0, DEMAND noise CC-BY-4.0). Those are 48 kbps MP3s, already 16 kHz mono, so
# decoding is the only step (no resampling or downmixing); the decoded audio is lossy and is NOT the audio
# upstream scored, so upstream's transcripts are a sanity check, not a golden reference. The golden
# reference is this repository's own host build on these exact WAV files (hashes in manifests/).
#
# Synthetic negatives and edge cases are generated deterministically: digital silence, seeded white
# noise, inputs shorter than the engine's minimum, and boundary-length clips.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$ROOT/build/fixtures
mkdir -p "$OUT"
rm -f "$OUT"/*.wav
FF=(ffmpeg -nostdin -hide_banner -loglevel error -y)

for mp3 in "$ROOT"/docs/samples/*.mp3; do
  b=$(basename "$mp3" .mp3)
  "${FF[@]}" -i "$mp3" -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact "$OUT/$b.wav"
done

# Edge cases: silence, seeded noise (ffmpeg's anoisesrc with a fixed seed), and a speech clip cut to lengths
# around the engine's 320-sample minimum and to a few short durations.
gen() { "${FF[@]}" -f lavfi -i "$1" -t "$2" -ac 1 -ar 16000 -c:a pcm_s16le -fflags +bitexact "$OUT/$3.wav"; }
gen "anullsrc=r=16000:cl=mono" 3 edge_silence_3s
gen "anoisesrc=r=16000:c=white:a=0.05:seed=1234" 3 edge_noise_3s
for n in 100 319 320 321 641 16000; do
  "${FF[@]}" -i "$OUT/rooms_00.wav" -af "atrim=end_sample=$n" -c:a pcm_s16le -fflags +bitexact "$OUT/edge_cut_${n}.wav"
done
# a long input: three clips back to back (~21 s), above the 20 s default bound on purpose
"${FF[@]}" -i "$OUT/real_24.wav" -i "$OUT/rooms_00.wav" -i "$OUT/real_21.wav" \
  -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" -c:a pcm_s16le -fflags +bitexact "$OUT/long_concat.wav"

mkdir -p "$ROOT/manifests"
(cd "$OUT" && sha256sum *.wav) > "$ROOT/manifests/fixtures-sha256.txt"
(cd "$ROOT" && sha256sum docs/samples/*.mp3) > "$ROOT/manifests/fixture-sources-sha256.txt"
echo "$(ls "$OUT"/*.wav | wc -l) fixtures in $OUT; $(ffmpeg -version | head -1)"
