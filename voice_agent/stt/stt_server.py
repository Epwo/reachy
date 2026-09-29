"""Long-lived STT worker. Runs inside .venv_stt.

    python stt_server.py [--engine whisper|parakeet]

Protocol (one JSON object per line, both directions):

    out:  {"ready": true}                    (once, after model loads)

    in:   {"audio_path": "/tmp/x.wav"}
    out:  {"text": "...", "error": null}

On failure: {"text": null, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from audio_stt import ENGINES, make_stt


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[stt_server] {msg}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=list(ENGINES), default="whisper")
    args = parser.parse_args()

    stt = make_stt(args.engine)
    log(f"engine = {args.engine}")
    log("loading model...")
    stt._ensure_loaded()
    log("warming up (fetches + compiles the model)...")
    # One throwaway transcription so the FIRST real turn doesn't pay the
    # download/compile cost. Low noise at 44.1 kHz so the resampling path
    # (lazy scipy import, ~0.5 s) is warmed up too.
    try:
        rng = np.random.default_rng(0)
        noise = (0.01 * rng.standard_normal(88200)).astype(np.float32)
        stt.transcribe(noise, sample_rate=44100)
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
