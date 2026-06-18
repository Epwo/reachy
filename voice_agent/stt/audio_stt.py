"""Speech-to-text on MLX, via Whisper.

Wraps `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`) so the robot can
turn a recorded utterance into French text quickly on Apple Silicon. This is
the first stage of the cascade pipeline (STT → LM → TTS): a purpose-built ASR
transcribes the audio, then a text-only LM only has to *write* a reply rather
than also *listen*. That keeps comprehension high while letting the LM stay
small and fast.

Importable on its own so you can A/B test ASR models:

    from audio_stt import WhisperSTT
    stt = WhisperSTT()
    text = stt.transcribe("/tmp/recording.wav")
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# large-v3-turbo: strong multilingual ASR (good French), ~809M params, fast on
# an M4. Alternatives: "mlx-community/whisper-large-v3-mlx" (slower, a touch
# more accurate) or a distil model (faster, weaker French).


class WhisperSTT:
    """Audio → text on MLX. Lazy-loads on first use."""

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL,
        language: str = "fr",
    ):
        self.model_repo = model_repo
        self.language = language
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Importing mlx_whisper is cheap; the model itself is fetched/compiled
        # on the first transcribe() call. We trigger that explicitly in the
        # server's warmup so the first real turn doesn't pay for it.
        import mlx_whisper  # noqa: F401
        self._loaded = True

    def transcribe(self, audio: "str | np.ndarray", sample_rate: int = 16000) -> str:
        """Transcribe a WAV path or a float32 mono array. Returns the text.

        We always hand mlx_whisper a numpy array (never a path): given a path it
        shells out to `ffmpeg` to decode, which we don't want as a system
        dependency. We decode with soundfile ourselves and resample to 16 kHz.
        """
        self._ensure_loaded()
        import mlx_whisper

        samples = _load_mono_16k(audio, sample_rate)
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model_repo,
            language=self.language,
            # Greedy + no temperature fallback = lowest latency. Short desktop
            # utterances don't need the fallback search.
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()


WHISPER_SR = 16000


def _load_mono_16k(audio, sample_rate: int) -> np.ndarray:
    """Return a float32 mono array at 16 kHz (what Whisper expects), decoding a
    path with soundfile if needed — no ffmpeg involved."""
    if isinstance(audio, str):
        if not os.path.exists(audio):
            raise FileNotFoundError(audio)
        arr, sample_rate = sf.read(audio, dtype="float32", always_2d=False)
    elif isinstance(audio, np.ndarray):
        arr = audio.astype(np.float32)
    else:
        raise TypeError(f"audio must be a path or numpy array, got {type(audio)}")

    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sample_rate != WHISPER_SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sample_rate), WHISPER_SR)
        arr = resample_poly(arr, WHISPER_SR // g, int(sample_rate) // g)
    return arr.astype(np.float32)
