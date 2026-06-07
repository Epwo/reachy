"""Audio capture / playback abstraction.

Works with two backends:
  * Reachy: uses `mini.media.*`
  * Laptop: uses sounddevice (default mic / speaker)

VAD endpointing is energy-based RMS — cheap, no extra deps.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000        # what Whisper expects


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

class AudioBackend:
    """Interface used by VADCapture and the TTS player."""

    def start_recording(self) -> None: ...
    def stop_recording(self) -> None: ...
    def read_chunk(self) -> np.ndarray | None: ...
    def input_samplerate(self) -> int: ...
    def play(self, audio: np.ndarray, sr: int) -> None: ...


class ReachyBackend(AudioBackend):
    """Reachy mic + speaker."""

    def __init__(self, mini):
        self.mini = mini

    def start_recording(self):
        self.mini.media.start_recording()

    def stop_recording(self):
        self.mini.media.stop_recording()

    def read_chunk(self):
        samples = self.mini.media.get_audio_sample()
        if samples is None or len(samples) == 0:
            return None
        mono = samples.mean(axis=1) if samples.ndim == 2 else samples
        return mono.astype(np.float32)

    def input_samplerate(self) -> int:
        return self.mini.media.get_input_audio_samplerate()

    def play(self, audio: np.ndarray, sr: int):
        try:
            sr_out = self.mini.media.get_output_audio_samplerate()
        except Exception:
            sr_out = 16000
        if sr != sr_out:
            audio = _resample_mono(audio, sr, sr_out)
        self.mini.media.start_playing()
        self.mini.media.push_audio_sample(audio.reshape(-1, 1).astype(np.float32))
        time.sleep(len(audio) / sr_out + 0.1)


class LaptopBackend(AudioBackend):
    """sounddevice-based fallback that uses the Mac's default mic + speaker.

    `device` is a sounddevice input index (see `sd.query_devices()`); None
    means "system default". Call `switch_device(idx)` while recording to
    hot-swap inputs — used by the webui.
    """

    def __init__(self, samplerate: int = 16000, chunk_ms: int = 100,
                 device: Optional[int] = None):
        import sounddevice as sd
        import queue
        self.sd = sd
        self._queue: queue.Queue = queue.Queue()
        self.samplerate = samplerate
        self.chunk_frames = int(samplerate * chunk_ms / 1000)
        self._stream = None
        self._device = device

    def _callback(self, indata, frames, time_, status):
        if status:
            print(f"[audio in] {status}")
        self._queue.put_nowait(indata.copy().reshape(-1))

    def start_recording(self):
        self._stream = self.sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=self.chunk_frames, callback=self._callback,
            device=self._device,
        )
        self._stream.start()

    def stop_recording(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def switch_device(self, device: Optional[int]) -> None:
        """Tear down the current input stream and restart on `device`.
        Safe to call from a different thread."""
        self._device = device
        if self._stream is not None:
            self.stop_recording()
            self.start_recording()

    def read_chunk(self):
        try:
            return self._queue.get(timeout=0.05)
        except Exception:
            return None

    def input_samplerate(self) -> int:
        return self.samplerate

    def play(self, audio: np.ndarray, sr: int):
        self.sd.play(audio, samplerate=sr, blocking=True)


class VADCapture:
    """Yields utterances bracketed by silence.

    Set `.muted = True` to make the capture loop drop any audio and reset
    its speech state — used to gate the mic while the robot is talking
    (otherwise it transcribes its own TTS in a feedback loop).
    """

    def __init__(self, backend: AudioBackend,
                 threshold: float = 0.012,
                 silence_ms: int = 700,
                 min_utterance_ms: int = 250,
                 on_rms=None):
        self.backend = backend
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_utterance_ms = min_utterance_ms
        self.muted = False
        # Optional callback(rms: float) — called for every chunk while the
        # capture is active. Lets a webui or VU meter show live mic level
        # without modifying the VAD logic.
        self.on_rms = on_rms

    def _drain(self) -> None:
        """Throw away any audio currently buffered by the backend."""
        while self.backend.read_chunk() is not None:
            pass

    def utterances(self) -> Iterator[np.ndarray]:
        self.backend.start_recording()
        sr = self.backend.input_samplerate()
        try:
            buffer: list[np.ndarray] = []
            in_speech = False
            silent_ms = 0.0

            while True:
                if self.muted:
                    # Drop everything that arrived while we were speaking, and
                    # forget any in-progress utterance — the tail of it is
                    # almost certainly the robot's own voice.
                    self._drain()
                    in_speech = False
                    buffer = []
                    silent_ms = 0.0
                    time.sleep(0.02)
                    continue

                chunk = self.backend.read_chunk()
                if chunk is None:
                    time.sleep(0.005)
                    continue

                rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))
                if self.on_rms is not None:
                    try:
                        self.on_rms(rms)
                    except Exception:
                        pass
                if rms > self.threshold:
                    if not in_speech:
                        in_speech = True
                        buffer = []
                    buffer.append(chunk)
                    silent_ms = 0.0
                elif in_speech:
                    buffer.append(chunk)
                    silent_ms += len(chunk) * 1000 / sr
                    if silent_ms > self.silence_ms:
                        utt = np.concatenate(buffer)
                        in_speech = False
                        buffer = []
                        silent_ms = 0.0
                        if len(utt) * 1000 / sr >= self.min_utterance_ms:
                            yield _resample_mono(utt, sr, SAMPLE_RATE)
        finally:
            self.backend.stop_recording()


class WakeGatedCapture:
    """Like VADCapture, but gated by a wake word + a conversation window.

    Flow:
      1. ASLEEP — feed mic chunks to the wake detector (cheap).
      2. On wake → fire `on_wake` (e.g. play Reachy's wake animation).
      3. AWAKE — capture the phrase that follows, yield it. After the agent
         finishes responding (generator resumes), keep listening for a
         follow-up WITHOUT requiring the wake word again.
      4. If no follow-up within `conversation_timeout` seconds → fire
         `on_sleep` (e.g. tuck Reachy back down) and return to step 1.

    Drop-in replacement for VADCapture: same `utterances()` generator and
    the same `.muted` / `.on_rms` hooks.
    """

    def __init__(self, backend: AudioBackend, wake_detector,
                 threshold: float = 0.02,
                 silence_ms: int = 700,
                 min_utterance_ms: int = 250,
                 max_wait_s: float = 2.5,
                 conversation_timeout: float = 6.0,
                 max_phrase_s: float = 8.0,
                 on_rms=None,
                 on_wake=None,
                 on_sleep=None,
                 on_state=None,
                 verbose: bool = False):
        self.backend = backend
        self.wake = wake_detector
        self.threshold = threshold
        self.verbose = verbose
        self.silence_ms = silence_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_wait_s = max_wait_s
        self.conversation_timeout = conversation_timeout
        self.max_phrase_s = max_phrase_s
        self.muted = False
        self.on_rms = on_rms        # called with mic rms every chunk
        self.on_wake = on_wake      # called when wake fires (asleep→awake)
        self.on_sleep = on_sleep    # called when conversation ends (awake→asleep)
        self.on_state = on_state    # called with "asleep"/"awake" on transitions

    def _drain(self) -> None:
        while self.backend.read_chunk() is not None:
            pass

    def utterances(self) -> Iterator[np.ndarray]:
        self.backend.start_recording()
        sr = self.backend.input_samplerate()
        if self.on_state:
            self.on_state("asleep")
        try:
            while True:
                # ---- Phase 1: ASLEEP — wait for the wake word --------------
                if self.verbose:
                    print("[wake] 😴 en attente du mot de réveil…")
                self.wake.reset()
                triggered = False
                last_audio_t = time.monotonic()    # watchdog reference
                while not triggered:
                    if self.muted:
                        self._drain()
                        time.sleep(0.02)
                        last_audio_t = time.monotonic()
                        continue
                    chunk = self.backend.read_chunk()
                    if chunk is None:
                        time.sleep(0.005)
                        # Watchdog: if the input stream stops delivering audio
                        # for a few seconds (e.g. a CoreAudio -10863 glitch
                        # after the robot played a sound), restart it so wake
                        # detection keeps working.
                        if time.monotonic() - last_audio_t > 3.0:
                            if self.verbose:
                                print("[wake] ⚠ flux micro bloqué → redémarrage")
                            try:
                                self.backend.stop_recording()
                                time.sleep(0.15)
                                self.backend.start_recording()
                            except Exception as e:
                                print(f"[wake] échec redémarrage micro: {e}")
                            last_audio_t = time.monotonic()
                        continue
                    last_audio_t = time.monotonic()
                    rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))
                    if self.on_rms:
                        try: self.on_rms(rms)
                        except Exception: pass
                    wk = _resample_mono(chunk, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else chunk
                    if self.wake.triggered(wk):
                        triggered = True

                # Wake animation (may play a sound) — fire it, then drain the
                # buffer so we don't capture the animation noise.
                if self.on_wake:
                    try: self.on_wake()
                    except Exception: pass
                self._drain()
                if self.on_state:
                    self.on_state("awake")

                # ---- Phase 2: AWAKE — conversation loop -------------------
                first = True
                while True:
                    wait = self.max_wait_s if first else self.conversation_timeout
                    first = False
                    if self.verbose:
                        print(f"[wake] 👂 écoute (max {wait:.1f}s, seuil {self.threshold:.3f})…")
                    phrase = self._capture_phrase(sr, max_wait_s=wait)
                    if phrase is None:
                        if self.verbose:
                            print("[wake] ⏱  rien entendu → rendormissement")
                        break          # no (further) speech → end conversation
                    dur = len(phrase) / SAMPLE_RATE
                    if self.verbose:
                        print(f"[wake] 🎙  phrase captée : {dur:.2f}s")
                    if len(phrase) * 1000 / SAMPLE_RATE >= self.min_utterance_ms:
                        yield phrase
                        # On resume, the agent has finished LM+TTS+playback and
                        # cleared `muted`; we loop to listen for a follow-up.

                # ---- Back to sleep ----------------------------------------
                if self.on_sleep:
                    try: self.on_sleep()
                    except Exception: pass
                self._drain()
                if self.on_state:
                    self.on_state("asleep")
        finally:
            self.backend.stop_recording()

    def _capture_phrase(self, sr: int, max_wait_s: float) -> Optional[np.ndarray]:
        frames: list[np.ndarray] = []
        started = False
        silent_ms = 0.0
        t_start = time.monotonic()

        while True:
            if self.muted:
                # Robot is talking — drop audio and don't count the wait.
                self._drain()
                time.sleep(0.02)
                t_start = time.monotonic()
                continue

            chunk = self.backend.read_chunk()
            if chunk is None:
                time.sleep(0.005)
                if not started and time.monotonic() - t_start > max_wait_s:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))
            if self.on_rms:
                try: self.on_rms(rms)
                except Exception: pass

            if not started:
                if rms > self.threshold:
                    started = True
                    frames.append(chunk)
                elif time.monotonic() - t_start > max_wait_s:
                    return None     # nobody spoke within the wait window
            else:
                frames.append(chunk)
                if rms > self.threshold:
                    silent_ms = 0.0
                else:
                    silent_ms += len(chunk) * 1000 / sr
                    if silent_ms > self.silence_ms:
                        break
                if (sum(len(f) for f in frames) / sr) > self.max_phrase_s:
                    break

        if not frames:
            return None
        utt = np.concatenate(frames)
        return _resample_mono(utt, sr, SAMPLE_RATE)


def _resample_mono(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


