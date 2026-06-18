"""Long-lived STT worker. Runs inside .venv_whisper.

Protocol (one JSON object per line, both directions):

    out:  {"ready": true}                    (once, after model loads)

    in:   {"audio_path": "/tmp/x.wav"}
    out:  {"text": "...", "error": null}

On failure: {"text": null, "error": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np

from audio_stt import WhisperSTT


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[stt_server] {msg}", file=sys.stderr, flush=True)


def main():
    stt = WhisperSTT()
    log("loading model...")
    stt._ensure_loaded()
    log("warming up (fetches + compiles the model)...")
    # One throwaway transcription on 1s of silence so the FIRST real turn
    # doesn't pay the download/compile cost.
    try:
        stt.transcribe(np.zeros(16000, dtype=np.float32))
    except Exception as e:
        log(f"warmup failed (non-fatal): {e}")
    log("ready.")
    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            text = stt.transcribe(job["audio_path"])
            emit({"text": text, "error": None})
        except Exception as e:
            tb = traceback.format_exc()
            log(tb)
            emit({"text": None, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
