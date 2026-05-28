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
    """sounddevice-based fallback that uses the Mac's default mic + speaker."""

    def __init__(self, samplerate: int = 16000, chunk_ms: int = 100):
        import sounddevice as sd
        import queue
        self.sd = sd
        self._queue: queue.Queue = queue.Queue()
        self.samplerate = samplerate
        self.chunk_frames = int(samplerate * chunk_ms / 1000)
        self._stream = None

    def _callback(self, indata, frames, time_, status):
        if status:
            print(f"[audio in] {status}")
        self._queue.put_nowait(indata.copy().reshape(-1))

    def start_recording(self):
        self._stream = self.sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=self.chunk_frames, callback=self._callback,
        )
        self._stream.start()

    def stop_recording(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
                 min_utterance_ms: int = 250):
        self.backend = backend
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_utterance_ms = min_utterance_ms
        self.muted = False

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


def _resample_mono(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


