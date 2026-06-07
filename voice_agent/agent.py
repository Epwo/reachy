"""Voice agent orchestrator.

Spawns the LM worker (in .venv_lm) and the TTS worker (in .venv_supertonic)
as long-lived subprocesses, then shuttles audio between them. Each model
loads once at startup; subsequent turns avoid the multi-second warmup.

Audio flow:
    Reachy or laptop mic
        → VADCapture (utterances chunked by silence)
        → temp WAV (16 kHz mono)
        → lm_server  (Gemma 4)    → text
        → tts_server (Supertonic) → temp WAV (44.1 kHz mono)
        → speaker (Reachy or laptop), with VAD muted during playback

Run from any venv that has numpy + soundfile + sounddevice
(.venv_supertonic works since it has all three):
    cd voice_agent
    source .venv_supertonic/bin/activate
    python agent.py --no-robot         # uses laptop mic + speaker
    python agent.py                    # uses Reachy mic + speaker
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from audio_io import LaptopBackend, ReachyBackend, VADCapture, WakeGatedCapture


HERE = Path(__file__).resolve().parent
# The lm worker now parses [heard]/[reply] itself and forwards them as JSON
# fields — no extra splitting needed in the orchestrator.


# ---------------------------------------------------------------------------
# Robot wake/sleep animations
# ---------------------------------------------------------------------------

def _robot_wake(mini, silent: bool = False) -> None:
    """Raise the head + neutral antennas. `silent` skips the wake sound so it
    doesn't fight the Mac mic over CoreAudio in hybrid mode."""
    if silent:
        import numpy as np
        from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS
        mini.goto_target(np.eye(4), antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                         duration=0.8, method="minjerk")
    else:
        mini.wake_up()


def _robot_sleep(mini, silent: bool = False) -> None:
    """Tuck the head down + fold antennas. `silent` skips the sleep sound."""
    if silent:
        from reachy_mini.reachy_mini import (
            SLEEP_HEAD_POSE, SLEEP_ANTENNAS_JOINT_POSITIONS,
        )
        mini.goto_target(SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS,
                         duration=1.0, method="minjerk")
    else:
        mini.goto_sleep()


# ---------------------------------------------------------------------------
# Long-lived subprocess wrapper
# ---------------------------------------------------------------------------

class Worker:
    """A JSON-over-pipes wrapper around a model subprocess."""

    def __init__(self, venv_dir: Path, script_path: Path, label: str,
                 script_args: Optional[list[str]] = None):
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
        cmd = [str(py), script_path.name] + list(script_args or [])
        self.proc = subprocess.Popen(
            cmd,
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

TTS_ENGINES = {
    # name        : venv folder name
    "supertonic" : ".venv_supertonic",
    "kokoro"     : ".venv_kokoro",
    "kyutai"     : ".venv_tts",
}


def run(use_robot: bool, tts_engine: str, webui: bool, webui_port: int,
        save_audio_dir: Optional[Path] = None,
        wake_model: Optional[str] = None, wake_threshold: float = 0.5,
        conversation_timeout: float = 6.0, motion: bool = False,
        vad_threshold: float = 0.02):
    # We connect to the robot if we use it for audio (ReachyBackend) OR just
    # for movement/animation (--motion with laptop audio = hybrid mode).
    connect_robot = use_robot or motion

    # Hybrid (motion but laptop audio): connect Reachy WITHOUT grabbing the
    # audio hardware, so its sounds don't fight the Mac mic over CoreAudio
    # (the -10863 "cannot do in current context" glitch). Animations are then
    # movement-only (silent). Full-robot mode keeps default media + sounds.
    hybrid = motion and not use_robot

    mini = mini_ctx = None
    if connect_robot:
        from reachy_mini import ReachyMini
        mini_ctx = ReachyMini(media_backend="no_media") if hybrid else ReachyMini()
        mini = mini_ctx.__enter__()
        if wake_model:
            _robot_sleep(mini, silent=hybrid)
        else:
            _robot_wake(mini, silent=hybrid)

    # Audio backend: Reachy mic/speaker, or the Mac's default devices.
    backend = ReachyBackend(mini) if use_robot else LaptopBackend()

    # Workers — these take 10-30 s to start as each loads + warms its model.
    lm = Worker(HERE / ".venv_lm", HERE / "lm" / "lm_server.py", "lm")
    tts_venv = HERE / TTS_ENGINES[tts_engine]
    tts = Worker(
        tts_venv, HERE / "tts" / "tts_server.py", "tts",
        script_args=["--engine", tts_engine],
    )

    # Optional web UI — starts a FastAPI server in a daemon thread that
    # shares state with this main loop via `state`.
    state = None
    say_queue: "queue.Queue[str]" = queue.Queue()
    if webui:
        try:
            from webui import AgentState, start_in_background
        except ModuleNotFoundError as e:
            print(f"[webui] missing dep ({e}). To enable the web UI, run:")
            print(f"        uv pip install -r requirements_webui.txt")
            print(f"        (in whichever venv you launched agent.py from)")
            print(f"[webui] continuing without the web UI...")
        else:
            state = AgentState()
            state.set_tts_engine(tts_engine)
            state.bind_handlers(
                on_clear_history=lambda: (
                    lm.call({"command": "clear_history"}),
                    state.clear_turns(),
                ),
                on_say=lambda text: say_queue.put_nowait(text),
                on_switch_mic=lambda idx: (
                    isinstance(backend, LaptopBackend)
                    and backend.switch_device(idx)
                ),
            )
            start_in_background(state, port=webui_port)

    if wake_model:
        from wake_detector import WakeDetector
        print(f"[wake] loading wake model: {wake_model} (threshold {wake_threshold})")
        detector = WakeDetector(wake_model, threshold=wake_threshold)

        def _on_wake():
            print("🔔 RÉVEILLÉ")
            if mini is not None:
                try: _robot_wake(mini, silent=hybrid)
                except Exception as e: print(f"[wake] wake anim failed: {e}")

        def _on_sleep():
            print("😴 Reachy se rendort")
            if mini is not None:
                try: _robot_sleep(mini, silent=hybrid)
                except Exception as e: print(f"[wake] sleep anim failed: {e}")

        capture = WakeGatedCapture(
            backend, detector,
            threshold=vad_threshold,
            conversation_timeout=conversation_timeout,
            on_rms=(state.set_mic_rms if state else None),
            on_wake=_on_wake,
            on_sleep=_on_sleep,
            on_state=((lambda s: state.set_state(s)) if state else None),
            verbose=True,
        )
        print(f"[wake] Reachy dort. Dis le mot de réveil pour lui parler "
              f"(conversation continue pendant {conversation_timeout:.0f}s).")
    else:
        capture = VADCapture(
            backend,
            on_rms=(state.set_mic_rms if state else None),
        )

    def _play_text(text: str) -> tuple[float, float]:
        """Synthesize + play `text`. Returns (tts_ms, audio_s)."""
        nonlocal_wav: Optional[str] = None
        try:
            if state: state.set_state("speaking")
            t0 = time.monotonic()
            resp = tts.call({"text": text})
            t_tts = time.monotonic() - t0
            if resp.get("error"):
                print(f"[tts error] {resp['error']}")
                return t_tts * 1000, 0.0
            nonlocal_wav = resp["audio_path"]
            audio, sr = sf.read(nonlocal_wav, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio_s = len(audio) / sr
            print(f"[tts {t_tts:.2f}s, {audio_s:.2f}s audio]")
            capture.muted = True
            try:
                backend.play(audio, sr)
            finally:
                time.sleep(0.3)
                capture._drain()
                capture.muted = False
            return t_tts * 1000, audio_s
        finally:
            if nonlocal_wav and os.path.exists(nonlocal_wav):
                try: os.remove(nonlocal_wav)
                except OSError: pass
            if state: state.set_state("idle")

    print("\nReady. Speak in French. Ctrl-C to quit.\n")
    try:
        for utterance in capture.utterances():
            # Drain any pending /say requests first (so the UI feels snappy).
            while not say_queue.empty():
                try:
                    msg = say_queue.get_nowait()
                except queue.Empty:
                    break
                _play_text(msg)

            # Save the captured audio so the LM subprocess can read it.
            fd, wav_in = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(wav_in, utterance, 16000, subtype="PCM_16")
            wav_out: Optional[str] = None
            # Permanent copy for later review, if --save-audio is on.
            saved_path: Optional[Path] = None
            if save_audio_dir is not None:
                ts = time.strftime("%Y%m%d-%H%M%S")
                saved_path = save_audio_dir / f"{ts}_{int(time.time()*1000)%1000:03d}.wav"
                sf.write(str(saved_path), utterance, 16000, subtype="PCM_16")

            try:
                # Step 1: audio → text
                if state: state.set_state("thinking")
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

                # Write the sidecar JSON next to the saved WAV so we can
                # diff the model's transcript against ground truth later.
                if saved_path is not None:
                    meta_path = saved_path.with_suffix(".json")
                    try:
                        meta_path.write_text(json.dumps({
                            "wav": saved_path.name,
                            "duration_s": round(len(utterance) / 16000, 3),
                            "samplerate": 16000,
                            "gemma_heard": heard,
                            "gemma_reply": reply_text,
                            "lm_ms": round(t_lm * 1000, 1),
                            "timestamp": time.time(),
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                    except OSError as e:
                        print(f"[save_audio] failed to write {meta_path}: {e}")

                # Step 2 + 3: synthesize + play
                t_tts_ms, audio_s = _play_text(reply_text)

                if state:
                    from webui import Turn
                    state.add_turn(Turn(
                        heard=heard, reply=reply_text,
                        lm_ms=t_lm * 1000, tts_ms=t_tts_ms, audio_s=audio_s,
                    ))
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
    parser.add_argument("--tts", choices=list(TTS_ENGINES.keys()),
                        default="supertonic",
                        help="Which TTS engine to spawn. The corresponding "
                             "venv must already exist (see README).")
    parser.add_argument("--webui", action="store_true",
                        help="Open a small web UI showing mic level, "
                             "transcripts, controls, and a type-to-speak box.")
    parser.add_argument("--webui-port", type=int, default=8765,
                        help="Port for the web UI (default 8765).")
    parser.add_argument("--save-audio", metavar="DIR",
                        help="Save every captured utterance to DIR as WAV+JSON "
                             "for later review with other STT models.")
    parser.add_argument("--wake", metavar="MODEL",
                        help="Enable wake-word gating. MODEL is a pretrained "
                             "name (hey_jarvis, ...) or a path to a custom "
                             ".onnx (e.g. wake/models/billou.onnx). Requires "
                             "openwakeword in the agent's venv.")
    parser.add_argument("--wake-threshold", type=float, default=0.5,
                        help="Wake-word detection threshold 0–1 (default 0.5).")
    parser.add_argument("--conversation-timeout", type=float, default=6.0,
                        help="Seconds to keep listening for a follow-up (no "
                             "wake word needed) before going back to sleep "
                             "(default 6).")
    parser.add_argument("--motion", action="store_true",
                        help="Connect to Reachy for movement/animations even "
                             "with laptop audio (--no-robot). Hybrid mode: "
                             "Mac mic+speaker, robot still wakes/sleeps/moves.")
    parser.add_argument("--vad-threshold", type=float, default=0.02,
                        help="Energy threshold for detecting speech after the "
                             "wake word (default 0.02). RAISE it if Reachy "
                             "never goes back to sleep (mic noise floor too "
                             "high); lower it if it misses quiet speech.")
    args = parser.parse_args()

    save_dir = None
    if args.save_audio:
        save_dir = Path(args.save_audio).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save_audio] writing utterances to {save_dir}/")

    run(use_robot=not args.no_robot, tts_engine=args.tts,
        webui=args.webui, webui_port=args.webui_port,
        save_audio_dir=save_dir,
        wake_model=args.wake, wake_threshold=args.wake_threshold,
        conversation_timeout=args.conversation_timeout,
        motion=args.motion, vad_threshold=args.vad_threshold)


if __name__ == "__main__":
    main()
