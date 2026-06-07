"""Long-lived TTS worker.

Engine is selected at startup via `--engine`:
    --engine supertonic   → Supertonic 3 (ONNX, runs in .venv_supertonic)
    --engine kokoro       → Kokoro-82M  (MLX, runs in .venv_kokoro)
    --engine kyutai       → Kyutai-1.6B (MLX, runs in .venv_tts)

The wrapper module for each engine lives next to this file; we lazy-import
only the one we need so this script works in any of the three venvs.

Protocol (one JSON object per line):

    out:  {"ready": true}
    in:   {"text": "...", "voice": "..." (optional)}
    out:  {"audio_path": "/tmp/x.wav", "sr": ..., "error": null}
          {"audio_path": null, "error": "..."}

Audio file is written to a temp WAV and orchestrator deletes it after play.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback

import soundfile as sf


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[tts_server] {msg}", file=sys.stderr, flush=True)


def make_engine(engine: str):
    """Lazy-import only the engine we'll actually use."""
    if engine == "supertonic":
        from tts_supertonic import SupertonicTTS, DEFAULT_FR_VOICE
        return SupertonicTTS(), DEFAULT_FR_VOICE
    if engine == "kokoro":
        from tts_kokoro import KokoroTTS, DEFAULT_FR_VOICE
        return KokoroTTS(), DEFAULT_FR_VOICE
    if engine == "kyutai":
        from tts_kyutai import KyutaiTTS, DEFAULT_FR_VOICE
        return KyutaiTTS(quantize=8), DEFAULT_FR_VOICE
    raise ValueError(f"Unknown engine: {engine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True,
                        choices=["supertonic", "kokoro", "kyutai"])
    args = parser.parse_args()

    log(f"engine = {args.engine}")
    tts, default_voice = make_engine(args.engine)

    log("loading model...")
    tts._ensure_loaded()
    log("warming up...")
    _ = tts.synthesize("Test de chauffe.")
    log("ready.")
    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            text = job["text"]
            voice = job.get("voice", default_voice)
            audio, sr = tts.synthesize(text, voice=voice)

            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, audio, sr, subtype="PCM_16")
            emit({"audio_path": path, "sr": sr, "error": None})
        except Exception as e:
            tb = traceback.format_exc()
            log(tb)
            emit({"audio_path": None, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
