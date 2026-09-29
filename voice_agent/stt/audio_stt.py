"""Speech-to-text on MLX: Whisper (default) or Parakeet.

First stage of the cascade pipeline (STT → LM → TTS): a purpose-built ASR
transcribes the audio, so the LM only has to *write* a reply.

Two engines, same `transcribe()` interface (pick with `make_stt(name)`):
    whisper   mlx-community/whisper-large-v3-turbo   ~0.8 s / phrase, 4.3 % WER*
    parakeet  mlx-community/parakeet-tdt-0.6b-v3     ~0.14 s / phrase, 5.2 % WER*
* on 12 synthetic noisy French clips — compare on your real voice with the
  agent's `--stt` flag + stats.py before switching the default.

    from audio_stt import make_stt
    stt = make_stt("parakeet")
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


PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
# NVIDIA Parakeet TDT v3 (0.6B), 25 European languages incl. French, language
# auto-detected. Non-autoregressive decoder → ~6x faster than Whisper turbo.


class ParakeetSTT:
    """Audio → text with parakeet-mlx. Loads the model on first use."""

    def __init__(self, model_repo: str = PARAKEET_MODEL):
        self.model_repo = model_repo
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from parakeet_mlx import from_pretrained
            self._model = from_pretrained(self.model_repo)

    def transcribe(self, audio: "str | np.ndarray", sample_rate: int = 16000) -> str:
        """Same contract as WhisperSTT.transcribe. We skip parakeet's own file
        loader (it needs ffmpeg) and feed it a 16 kHz array directly."""
        self._ensure_loaded()
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        samples = _load_mono_16k(audio, sample_rate)
        mel = get_logmel(mx.array(samples), self._model.preprocessor_config)
        return self._model.generate(mel)[0].text.strip()


ENGINES = {"whisper": WhisperSTT, "parakeet": ParakeetSTT}


def make_stt(engine: str = "whisper"):
    return ENGINES[engine]()


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
