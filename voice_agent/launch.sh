#!/usr/bin/env bash
# My usual setup: Mac mic/speaker + robot motion only (hybrid), wake word with
# custom verifier, wake clips recorded for retraining, web UI.
# Extra flags are passed through, e.g.:  ./launch.sh --output-device 1
# (--output-device 1 = Mac mini Speakers, if playback ever freezes the Reachy
#  USB audio device that the mic also records from.)
cd "$(dirname "$0")"
exec .venv_supertonic/bin/python agent.py \
    --wake wake/models/billou.onnx \
    --wake-verifier wake/models/billou_verifier.joblib \
    --record-wake ~/reachy_wake_data \
    --webui --no-robot --motion "$@"
