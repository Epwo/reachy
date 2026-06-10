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
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from audio_io import LaptopBackend, ReachyBackend, VADCapture, WakeGatedCapture
import tools as toolkit


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


_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff←-⇿⌀-⏿]+"
)


def _clean_for_tts(text: str) -> str:
    """Strip emojis/pictographs the TTS would mispronounce, tidy whitespace."""
    if not text:
        return text
    text = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


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
        wake_verifier: Optional[str] = None,
        conversation_timeout: float = 6.0, motion: bool = False,
        vad_threshold: float = 0.02, use_tools: bool = True,
        barge_enabled: bool = True, barge_threshold: float = 0.05,
        barge_echo_gain: float = 0.6, barge_min_ms: float = 250.0,
        record_wake_dir: Optional[Path] = None):
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
        detector = WakeDetector(wake_model, threshold=wake_threshold,
                                verifier=wake_verifier)

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
            record_dir=record_wake_dir,
        )
        # The webui "missed wake word" button needs the capture to exist.
        if state is not None:
            state.set_wake_miss_handler(capture.flag_miss)
        if record_wake_dir is not None:
            print(f"[wake-rec] enregistrement des détections → {record_wake_dir}/")
        print(f"[wake] Reachy dort. Dis le mot de réveil pour lui parler "
              f"(conversation continue pendant {conversation_timeout:.0f}s).")
    else:
        capture = VADCapture(
            backend,
            on_rms=(state.set_mic_rms if state else None),
        )

    # One lock serializes all speaking (main turns AND background timer
    # announcements) so they never interleave on the TTS pipe or the speaker.
    speak_lock = threading.Lock()
    active_timers: list = []

    def _announce(text: str) -> None:
        """Speak `text` from any thread (e.g. a timer firing). Mutes the mic
        during playback so the robot doesn't hear itself, then restores."""
        text = _clean_for_tts(text)
        if not text:
            return
        with speak_lock:
            was_muted = capture.muted
            capture.muted = True
            wav = None
            try:
                resp = tts.call({"text": text})
                if resp.get("error"):
                    print(f"[timer] tts error: {resp['error']}")
                    return
                wav = resp["audio_path"]
                audio, sr = sf.read(wav, dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                backend.play(audio, sr)
            finally:
                if wav and os.path.exists(wav):
                    try: os.remove(wav)
                    except OSError: pass
                time.sleep(0.2)
                capture._drain()
                capture.muted = was_muted

    def _schedule_timer(seconds: float, label: str) -> None:
        def _fire():
            lbl = f" de {label}" if label else ""
            print(f"[timer] ⏰ minuteur{lbl} terminé")
            _announce(f"Ding ding ! Ton minuteur{lbl} est terminé !")
        t = threading.Timer(seconds, _fire)
        t.daemon = True
        t.start()
        active_timers.append(t)
        print(f"[timer] ⏱  programmé : {seconds:.0f}s ({label})")

    def _monitored_play(audio: np.ndarray, sr: int) -> bool:
        """Play `audio` while listening for the user barging in. Returns True
        if interrupted. Uses echo-compensated energy: mic level minus a
        fraction of the clip's own level at the current playback position, so
        the robot's own voice doesn't trigger a false interruption."""
        win = max(1, int(sr * 0.03))                 # 30 ms RMS window on the clip
        backend.play_async(audio, sr)
        t_start = time.monotonic()
        sustained = 0.0
        # Mic is NOT muted here — we need to hear a possible interruption.
        while backend.playback_active():
            chunk = backend.read_chunk()
            if chunk is None:
                time.sleep(0.005)
                continue
            mic_rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))
            # Concurrent level of the clip itself (echo we expect to hear).
            pos = int((time.monotonic() - t_start) * sr)
            seg = audio[max(0, pos - win):pos]
            clip_rms = float(np.sqrt(np.mean(seg * seg) + 1e-9)) if len(seg) else 0.0
            effective = mic_rms - barge_echo_gain * clip_rms
            if effective > barge_threshold:
                sustained += len(chunk) * 1000 / sr
                if sustained >= barge_min_ms:
                    backend.stop_playback()
                    return True
            else:
                sustained = 0.0
        return False

    def _play_text(text: str, allow_barge: bool = True) -> tuple[float, float]:
        """Synthesize + play `text`. Returns (tts_ms, audio_s).
        If barge-in is enabled and the user interrupts, playback stops early
        and barge_state['interrupted'] is set."""
        nonlocal_wav: Optional[str] = None
        try:
          with speak_lock:   # serialize vs. background timer announcements
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

            do_barge = allow_barge and barge_enabled
            if do_barge:
                interrupted = _monitored_play(audio, sr)
                if interrupted:
                    print("[barge] ✋ interrompu — j'écoute")
                    # Don't drain: keep the user's ongoing speech for capture.
                else:
                    time.sleep(0.3)
                    capture._drain()
            else:
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

    # Tools: docs appended to the system prompt + a context dict tools receive.
    tools_extra = toolkit.tools_system_prompt() if use_tools else ""
    tool_ctx = {"mini": mini, "capture": capture, "hybrid": hybrid}
    if use_tools:
        names = ", ".join(toolkit.TOOLS.keys())
        print(f"[tools] activés: {names}")
        if os.environ.get("BRAVE_API_KEY"):
            print("[tools] recherche web: Brave (BRAVE_API_KEY)")
        else:
            try:
                import ddgs  # noqa: F401
                print("[tools] recherche web: DuckDuckGo (gratuit, sans clé)")
            except ImportError:
                print("[tools] recherche web: indisponible (pip install ddgs)")

    if barge_enabled:
        print(f"[barge] interruption activée (seuil {barge_threshold:.3f}, "
              f"écho-gain {barge_echo_gain:.2f}) — parle pour l'interrompre. "
              f"Idéal avec le micro de Reachy (AEC) ou un casque.")

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
                resp = lm.call({"audio_path": wav_in, "extra_system": tools_extra})
                t_lm = time.monotonic() - t0
                if resp.get("error"):
                    print(f"[lm error] {resp['error']}")
                    continue
                heard = resp.get("heard") or ""
                raw_reply = resp.get("reply") or resp.get("raw") or ""
                if heard:
                    print(f"\n[heard] {heard}")

                # Step 1b: parse + run any tool calls the model emitted.
                tool_calls, spoken = toolkit.parse_tool_calls(raw_reply)
                terminal = False
                followups: list[str] = []
                for name, args in tool_calls:
                    r = toolkit.dispatch(name, args, tool_ctx)
                    print(f"[tool] {name} {args!r} → {r.note or r.followup or 'ok'}")
                    terminal = terminal or r.terminal
                    if r.followup:
                        followups.append(r.followup)
                    if r.timer_seconds:
                        _schedule_timer(r.timer_seconds, r.timer_label)

                # Fallback: the model sometimes puts its spoken goodbye as the
                # argument of an argless tool, e.g. `[tool:sleep] À bientôt !`.
                # If we have no separate spoken text but a terminal tool carried
                # sentence-like args, speak those so it says goodbye before
                # going to sleep.
                if not spoken and terminal:
                    for name, args in tool_calls:
                        if name == "sleep" and len(args.split()) >= 2:
                            spoken = args
                            break

                spoken = _clean_for_tts(spoken)
                print(f"[reply {t_lm:.2f}s] {spoken}")

                if saved_path is not None:
                    meta_path = saved_path.with_suffix(".json")
                    try:
                        meta_path.write_text(json.dumps({
                            "wav": saved_path.name,
                            "duration_s": round(len(utterance) / 16000, 3),
                            "gemma_heard": heard,
                            "gemma_reply": raw_reply,
                            "tools": [n for n, _ in tool_calls],
                            "lm_ms": round(t_lm * 1000, 1),
                            "timestamp": time.time(),
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                    except OSError as e:
                        print(f"[save_audio] failed to write {meta_path}: {e}")

                # Step 2: decide what to say.
                if followups:
                    # Quick feedback while we fetch + reason (search latency).
                    if spoken:
                        _play_text(spoken)
                    if state: state.set_state("thinking")
                    # No tools_extra here: the 2nd pass is a pure summary task,
                    # it must NOT emit new [tool:...] calls.
                    fr = lm.call({"command": "respond_text",
                                  "text": "\n".join(followups)})
                    answer = (fr.get("reply") if not fr.get("error") else None) \
                             or "Désolé, je n'ai rien trouvé."
                    print(f"[reply+] {answer}")
                    t_tts_ms, audio_s = _play_text(answer)
                    final_reply = answer
                else:
                    final_reply = spoken
                    t_tts_ms, audio_s = _play_text(spoken) if spoken else (0.0, 0.0)

                if state:
                    from webui import Turn
                    state.add_turn(Turn(
                        heard=heard, reply=final_reply,
                        lm_ms=t_lm * 1000, tts_ms=t_tts_ms, audio_s=audio_s,
                    ))

                # Step 3: sleep tool → end the conversation now.
                if terminal:
                    if hasattr(capture, "end_conversation"):
                        capture.end_conversation()
                    elif mini is not None:
                        try: _robot_sleep(mini, silent=hybrid)
                        except Exception: pass
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
        for t in active_timers:
            try: t.cancel()
            except Exception: pass
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
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool calling (sleep, web search, emotes). "
                             "Web search needs BRAVE_API_KEY in the env.")
    parser.add_argument("--no-barge", action="store_true",
                        help="Disable barge-in (interrupting Reachy while he "
                             "speaks). With it on, talk over him to stop him.")
    parser.add_argument("--barge-threshold", type=float, default=0.05,
                        help="Echo-compensated energy needed to interrupt "
                             "(default 0.05). Lower = easier to interrupt but "
                             "more false stops; raise it if his own voice "
                             "keeps stopping him (laptop mic near speaker).")
    parser.add_argument("--barge-echo-gain", type=float, default=0.6,
                        help="How much of his own voice to subtract from the "
                             "mic when checking for interruption (0–1, default "
                             "0.6). Higher if the speaker bleeds into the mic.")
    parser.add_argument("--record-wake", metavar="DIR",
                        help="Record wake-word events (detections, near-misses, "
                             "and web-UI-flagged misses) to DIR as WAV+JSON, for "
                             "labeling + retraining. See wake/label_recordings.py.")
    parser.add_argument("--wake-verifier", metavar="JOBLIB",
                        help="Custom verifier model (from wake/train_verifier.py) "
                             "to filter out false detections using your own "
                             "voice + false-trigger clips.")
    args = parser.parse_args()

    save_dir = None
    if args.save_audio:
        save_dir = Path(args.save_audio).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save_audio] writing utterances to {save_dir}/")

    record_wake_dir = None
    if args.record_wake:
        record_wake_dir = Path(args.record_wake).expanduser().resolve()
        record_wake_dir.mkdir(parents=True, exist_ok=True)

    run(use_robot=not args.no_robot, tts_engine=args.tts,
        webui=args.webui, webui_port=args.webui_port,
        save_audio_dir=save_dir,
        wake_model=args.wake, wake_threshold=args.wake_threshold,
        wake_verifier=args.wake_verifier,
        conversation_timeout=args.conversation_timeout,
        motion=args.motion, vad_threshold=args.vad_threshold,
        use_tools=not args.no_tools,
        barge_enabled=not args.no_barge,
        barge_threshold=args.barge_threshold,
        barge_echo_gain=args.barge_echo_gain,
        record_wake_dir=record_wake_dir)


if __name__ == "__main__":
    main()
