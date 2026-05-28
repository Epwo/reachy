"""Step 2 alternative: Kokoro-82M via mlx-audio.

A small, fast multilingual TTS (82 M params) compared to Kyutai's 1.8 B.
Trade-off: less expressive prosody, but ~10–20× faster on M-series Macs.

Same interface as `tts_kyutai.KyutaiTTS` so the rest of the pipeline
(test_tts.py, tts_server.py, agent.py) can swap without changes.

    from tts_kokoro import KokoroTTS
    tts = KokoroTTS(voice="ff_siwis", lang_code="f")
    audio, sr = tts.synthesize("Bonjour, je suis Reachy.")
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


# Phonemizer logs INFO/WARN lines whenever a French sentence contains
# English tokens like "Reachy Mini". They're cosmetic — the "remove-flags"
# policy handles them correctly. Mute everything below ERROR.
logging.getLogger("phonemizer").setLevel(logging.ERROR)


def _register_espeak() -> None:
    """Point phonemizer at the bundled espeak-ng .dylib.

    Without this, Kokoro's pipeline fails to find espeak unless it's
    installed system-wide (e.g. via `brew install espeak-ng`). The
    `espeakng-loader` pip package ships the library, so we just wire it up.
    Safe to call multiple times.
    """
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    except Exception:
        # If either package is missing or the wrapper API changes, fall back
        # to whatever espeak resolution phonemizer does by default.
        pass


_register_espeak()


# Some commonly-installed Kokoro voices.  `kokoro --list-voices` after
# install will print the full set.
#   ff_siwis       French female (Swiss French corpus)
#   af_heart       American English female
#   am_adam        American English male
#   bf_emma        British English female
DEFAULT_FR_VOICE = "ff_siwis"
DEFAULT_LANG = "f"            # Kokoro lang codes: a=American, b=British, f=French...


def _repo_for_quant(quantize: Optional[int]) -> str:
    if quantize == 4:
        return "mlx-community/Kokoro-82M-4bit"
    if quantize == 8:
        return "mlx-community/Kokoro-82M-8bit"
    return "mlx-community/Kokoro-82M-bf16"


class KokoroTTS:
    """Kokoro 82M on MLX. Lazy-loads the model on first call."""

    def __init__(
        self,
        voice: str = DEFAULT_FR_VOICE,
        quantize: Optional[int] = None,
        lang_code: str = DEFAULT_LANG,
        speed: float = 1.0,
    ):
        self.voice = voice
        self.quantize = quantize
        self.lang_code = lang_code
        self.speed = speed
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model
        repo = _repo_for_quant(self.quantize)
        print(f"[tts] loading Kokoro ({repo}) — first call only...")
        self._model = load_model(repo)
        print("[tts] ready.")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        lang_code: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        v = voice or self.voice
        lc = lang_code or self.lang_code

        chunks: list[np.ndarray] = []
        sample_rate = 24000  # Kokoro default

        for result in self._model.generate(
            text=text, voice=v, speed=self.speed, lang_code=lc,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            chunks.append(audio)
            sr = getattr(result, "sample_rate", None) or getattr(result, "sr", None)
            if sr:
                sample_rate = int(sr)

        if not chunks:
            return np.zeros(0, dtype=np.float32), sample_rate
        return np.concatenate(chunks, axis=-1), sample_rate
