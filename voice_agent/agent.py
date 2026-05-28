"""Voice agent orchestrator.

Spawns the LM worker (in .venv_lm) and the TTS worker (in .venv_kokoro)
as long-lived subprocesses, then shuttles audio between them. Each model
loads once at startup; subsequent turns avoid the multi-second warmup.

Audio flow:
    Reachy or laptop mic
        → VADCapture (utterances chunked by silence)
        → temp WAV (16 kHz mono)
        → lm_server  (Gemma 4)   → text
        → tts_server (Kokoro)    → temp WAV (24 kHz mono)
        → speaker (Reachy or laptop), with VAD muted during playback

Run from any venv that has numpy + soundfile + sounddevice (e.g. .venv_kokoro):
    cd voice_agent
    source .venv_kokoro/bin/activate
    python agent.py --no-robot         # uses laptop mic + speaker
    python agent.py                    # uses Reachy mic + speaker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from audio_io import LaptopBackend, ReachyBackend, VADCapture


HERE = Path(__file__).resolve().parent
# The lm worker now parses [heard]/[reply] itself and forwards them as JSON
# fields — no extra splitting needed in the orchestrator.


# ---------------------------------------------------------------------------
# Long-lived subprocess wrapper
# ---------------------------------------------------------------------------

class Worker:
    """A JSON-over-pipes wrapper around a model subprocess."""

    def __init__(self, venv_dir: Path, script_path: Path, label: str):
        py = venv_dir / "bin" / "python"
        if not py.exists():
            raise RuntimeError(
                f"Python not found at {py}. "
                f"Did you create the venv? See voice_agent/README.md."
            )
        if not script_path.exists():
            raise RuntimeError(f"Script not found at {script_path}.")
        self.label = label
        print(f"[{label}] spawning subprocess...")
        # cwd = the script's own folder so it can import sibling modules
        # (e.g. lm_server.py imports audio_lm.py from the same lm/ dir).
        self.proc = subprocess.Popen(
            [str(py), script_path.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            cwd=str(script_path.parent),
        )
        self._await_ready()

    def _await_ready(self) -> None:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.label} subprocess died before ready")
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                print(f"[{self.label}] ready.")
                return

    def call(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label} subprocess closed stdout")
        return json.loads(line.strip())

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(use_robot: bool):
    # Backend
    mini = mini_ctx = None
    if use_robot:
        from reachy_mini import ReachyMini
        mini_ctx = ReachyMini()
        mini = mini_ctx.__enter__()
        mini.wake_up()
        backend = ReachyBackend(mini)
    else:
        backend = LaptopBackend()

    # Workers — these take 10-30 s to start as each loads + warms MLX kernels.
    lm = Worker(HERE / ".venv_lm",     HERE / "lm"  / "lm_server.py",  "lm")
    tts = Worker(HERE / ".venv_kokoro", HERE / "tts" / "tts_server.py", "tts")

    capture = VADCapture(backend)

    print("\nReady. Speak in French. Ctrl-C to quit.\n")
    try:
        for utterance in capture.utterances():
            # Save the captured audio so the LM subprocess can read it.
            fd, wav_in = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(wav_in, utterance, 16000, subtype="PCM_16")
            wav_out: Optional[str] = None

            try:
                # Step 1: audio → text  (history is maintained inside lm_server)
                t0 = time.monotonic()
                resp = lm.call({"audio_path": wav_in})
                t_lm = time.monotonic() - t0
                if resp.get("error"):
                    print(f"[lm error] {resp['error']}")
                    continue
                heard = resp.get("heard") or ""
                reply_text = resp.get("reply") or resp.get("raw") or ""
                if heard:
                    print(f"\n[heard] {heard}")
                print(f"[reply {t_lm:.2f}s] {reply_text}")

                # Step 2: text → audio
                t0 = time.monotonic()
                resp = tts.call({"text": reply_text})
                t_tts = time.monotonic() - t0
                if resp.get("error"):
                    print(f"[tts error] {resp['error']}")
                    continue
                wav_out = resp["audio_path"]

                # Step 3: play through speaker, with VAD gated to avoid echo
                audio, sr = sf.read(wav_out, dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                print(f"[tts {t_tts:.2f}s, {len(audio)/sr:.2f}s audio]")
                capture.muted = True
                try:
                    backend.play(audio, sr)
                finally:
                    time.sleep(0.3)
                    capture._drain()
                    capture.muted = False
            finally:
                for p in (wav_in, wav_out):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        lm.close()
        tts.close()
        if mini_ctx is not None:
            try:
                mini.goto_sleep()
            except Exception:
                pass
            mini_ctx.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-robot", action="store_true",
                        help="Use laptop mic + speaker instead of Reachy.")
    args = parser.parse_args()
    run(use_robot=not args.no_robot)


if __name__ == "__main__":
    main()
