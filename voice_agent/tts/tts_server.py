"""Long-lived TTS worker. Runs inside .venv_kokoro.

Default engine: Kokoro-82M via mlx-audio. Tiny, fast (RTF ~0.1 on M4 base),
multilingual. To swap back to Kyutai-1.6B, change the import + venv below.

Protocol (one JSON object per line):

    out:  {"ready": true}
    in:   {"text": "...", "voice": "..." (optional)}
    out:  {"audio_path": "/tmp/x.wav", "sr": 24000, "error": null}
          {"audio_path": null, "error": "..."}

We write the WAV to a temp file and hand back the path. Orchestrator is
responsible for deleting it after playback.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

import soundfile as sf

from tts_kokoro import KokoroTTS as TTSEngine, DEFAULT_FR_VOICE


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[tts_server] {msg}", file=sys.stderr, flush=True)


def main():
    tts = TTSEngine()
    log("loading model...")
    tts._ensure_loaded()
    log("warming up MLX kernels...")
    _ = tts.synthesize("Test.")
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
