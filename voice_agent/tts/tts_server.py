"""Long-lived TTS worker (Supertonic 3, ONNX). Runs inside .venv_supertonic.

Stage 3 of the cascade pipeline (Whisper STT → chat LM → this TTS).

Protocol (one JSON object per line):

    out:  {"ready": true}
    in:   {"text": "...", "voice": "..." (optional)}
    out:  {"audio_path": "/tmp/x.wav", "sr": ..., "error": null}
          {"audio_path": null, "error": "..."}

Audio file is written to a temp WAV and orchestrator deletes it after play.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

import soundfile as sf

from tts_supertonic import DEFAULT_FR_VOICE, SupertonicTTS


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[tts_server] {msg}", file=sys.stderr, flush=True)


def main():
    tts = SupertonicTTS()
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
            voice = job.get("voice", DEFAULT_FR_VOICE)
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
