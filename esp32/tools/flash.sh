#!/bin/bash
# Build + flash firmware, model and LM to an ESP32-S3 N16R8 (16 MB flash, 8 MB octal PSRAM).
# usage: tools/flash.sh <port> <model.tnm|model.tasr> [lm.tlm] [clip.wav "reference" ...]
#   TASR_MODE=mic  (default) live INMP441 microphone: speak, the transcript prints after each pause
#   TASR_MODE=file transcribe the given clips from flash and print WER and the measured real-time factor
#   TASR_OLED=1    also drive an SSD1306 128x64 I2C OLED (SDA GPIO8, SCL GPIO9) with the live transcript
set -e
PORT=$1; MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); LM=${3:-}; shift 2; [ $# -gt 0 ] && shift
[ -n "$LM" ] && LM=$(cd "$(dirname "$LM")" && pwd)/$(basename "$LM")
WAVS=(); REFS=()
while [ $# -gt 1 ]; do WAVS+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); REFS+=("$2"); shift 2; done
MODE=${TASR_MODE:-mic}
PY=${PYTHON:-$(command -v python3)}   # needs numpy + soundfile (ESP-IDF's own python does not have them)
cd "$(dirname "$0")/.."
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null
case "$MODEL" in
  *rnnt*.tnm) LAYOUT=rnnt16; CFG="sdkconfig.defaults;sdkconfig.rnnt16"; LM="" ;;   # transducer: no external LM
  *.tnm)      LAYOUT=nemo16lm; CFG="sdkconfig.defaults;sdkconfig.nemo16lm" ;;
  *)          LAYOUT=tinyasr16; CFG="sdkconfig.defaults" ;;
esac
[ "$MODE" = mic ] && CFG="$CFG;sdkconfig.mic"
SUF=""; [ -n "$TASR_OLED" ] && CFG="$CFG;sdkconfig.oled" && SUF=_oled
B=build_${LAYOUT}_${MODE}${SUF}
(cd firmware && idf.py -B $B -D SDKCONFIG=sdkconfig_${LAYOUT}_${MODE}${SUF} -D SDKCONFIG_DEFAULTS="$CFG" build)
OUT=build_images/${LAYOUT}_${MODE}; mkdir -p $OUT
LMARG=""; [ -n "$LM" ] && LMARG="--lm $LM"
"$PY" tools/mkimages.py "$MODEL" $OUT --layout $LAYOUT --wavs "${WAVS[@]}" --refs "${REFS[@]}" $LMARG --merge firmware/$B --flash_mode keep
esptool.py --chip esp32s3 -p "$PORT" -b 921600 write_flash 0x0 $OUT/flash.bin
echo "flashed ($LAYOUT, $MODE). monitor: (cd firmware && idf.py -B $B -p $PORT monitor)"
