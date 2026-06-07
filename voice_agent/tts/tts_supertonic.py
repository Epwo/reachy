"""Supertonic 3 TTS via the official `supertonic` PyPI package.

Released April 2026, 99M params on ONNX Runtime. Supports 31 languages,
including French. Voices are speaker-style identifiers (M1-M5 male,
F3-F5 female) that work across every supported language.

Same interface as tts_kyutai.KyutaiTTS and tts_kokoro.KokoroTTS so the
rest of the pipeline can swap without changes.

    from supertonic_tts import SupertonicTTS
    tts = SupertonicTTS(voice="F4", lang="fr")
    audio, sr = tts.synthesize("Bonjour, je suis Reachy.")
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


# Built-in speaker styles (10 voices shipped with the open weights).
#   M1, M2, M3, M4, M5  male variants
#   F1, F2, F3, F4, F5  female variants
# Custom voices (e.g. "alex") are downloaded as separate embedding files from
# https://supertonic.supertone.ai/voice-builder — pass the file path as `voice`
# and the wrapper will detect that and call `get_voice_style_from_path` instead.
DEFAULT_FR_VOICE = "F1"
DEFAULT_LANG = "fr"
DEFAULT_SAMPLE_RATE = 44100  # Supertonic 3 emits 44.1 kHz


class SupertonicTTS:
    """Lazy-loading wrapper around `supertonic.TTS`."""

    def __init__(
        self,
        voice: str = DEFAULT_FR_VOICE,
        lang: str = DEFAULT_LANG,
        quantize: Optional[int] = None,  # ignored — Supertonic is ONNX
        speed: float = 1,
        total_steps: int = 5,  # 5–12, default 8 (higher = better quality)
    ):
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.total_steps = total_steps
        self._tts = None
        self._style = None
        self._style_voice = None  # name of the currently-cached style

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        from supertonic import TTS

        print("[tts] loading Supertonic 3 — first call only...")
        self._tts = TTS(auto_download=True)
        self._style = self._resolve_style(self.voice)
        self._style_voice = self.voice
        print("[tts] ready.")

    def _resolve_style(self, voice: str):
        """`voice` may be a built-in name ("M1", "F4") or a path to a custom
        voice embedding (e.g. one downloaded from the Voice Builder).
        Detect which one it is and dispatch to the right Supertonic API.
        """
        looks_like_path = (
            os.sep in voice
            or voice.endswith((".json", ".npy", ".npz", ".pt", ".bin", ".safetensors"))
            or os.path.exists(voice)
        )
        if looks_like_path:
            return self._tts.get_voice_style_from_path(voice)
        return self._tts.get_voice_style(voice_name=voice)

    def _style_for(self, voice: str):
        """Get (and cache) the voice style object for `voice`."""
        if voice != self._style_voice:
            self._style = self._resolve_style(voice)
            self._style_voice = voice
        return self._style

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        v = voice or self.voice
        lc = lang or self.lang

        wav, _duration = self._tts.synthesize(
            text=text,
            lang=lc,
            voice_style=self._style_for(v),
            total_steps=self.total_steps,
            speed=self.speed,
        )

        # Supertonic returns shape (channels, samples). Collapse to mono
        # along the channel axis, not the samples axis.
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
        elif audio.ndim > 2:
            audio = audio.reshape(-1)

        sr = int(getattr(self._tts, "sample_rate", DEFAULT_SAMPLE_RATE))
        return audio, sr
