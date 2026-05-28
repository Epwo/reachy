"""
LEGACY: This module is a wrapper around the Kyutai TTS model from moshi_mlx, which was way too slow ( ~1.5 s latency on GPU) for real-time
"""

"""Step 2 of the voice agent pipeline: Kyutai TTS on MLX.

Wraps the `kyutai/tts-1.6b-en_fr` model via the `moshi_mlx` package so the
robot can speak French (or English) with low-latency neural TTS.

Designed to be importable on its own:

    from tts_kyutai import KyutaiTTS
    tts = KyutaiTTS(voice="fr/siwis/...")    # or any voice from kyutai/tts-voices
    audio, sr = tts.synthesize("Bonjour, je suis Reachy.")

`audio` is a float32 mono numpy array at `sr` (24 kHz from Mimi codec).
Play it through whichever audio backend you like.
"""

from __future__ import annotations

import json
import queue
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# Default voice paths inside the `kyutai/tts-voices` repo. The model is
# bilingual, so the voice mostly affects timbre/accent; pick one that
# matches the target language for the most natural result.
DEFAULT_FR_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# ^ Default English voice from the Kyutai sample script. The model still
# pronounces French text correctly with it, but with an English accent.
# Browse https://huggingface.co/kyutai/tts-voices/tree/main for French
# voices (the "siwis" subdirectory has French speakers) and pass it as
# `KyutaiTTS(voice="siwis/...wav")` or `synthesize(..., voice=...)`.


class KyutaiTTS:
    """Lazy-loading wrapper around moshi_mlx's TTSModel."""

    def __init__(
        self,
        voice: str = DEFAULT_FR_VOICE,
        quantize: Optional[int] = None,  # 4 / 8 / None (bf16, default)
        temp: float = 0.6,
    ):
        self.voice = voice
        self.quantize = quantize
        self.temp = temp
        self._tts_model = None

    def _ensure_loaded(self) -> None:
        if self._tts_model is not None:
            return

        # Imports are local so importing this module is cheap.
        from moshi_mlx import models
        from moshi_mlx.models.tts import (
            DEFAULT_DSM_TTS_REPO,
            DEFAULT_DSM_TTS_VOICE_REPO,
            TTSModel,
        )
        from moshi_mlx.utils.loaders import hf_get
        import sentencepiece

        print(f"[tts] loading Kyutai TTS ({DEFAULT_DSM_TTS_REPO}) — first call only...")

        raw_config_path = hf_get("config.json", DEFAULT_DSM_TTS_REPO)
        with open(raw_config_path, "r") as f:
            raw_config = json.load(f)

        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        mimi_weights = hf_get(raw_config["mimi_name"], DEFAULT_DSM_TTS_REPO)
        moshi_weights = hf_get(moshi_name, DEFAULT_DSM_TTS_REPO)
        tokenizer_path = hf_get(raw_config["tokenizer_name"], DEFAULT_DSM_TTS_REPO)
        # Kyutai TTS config exposes total / depformer codebook counts as `n_q`
        # and `dep_q` (both 32 for tts-1.6b-en_fr). Mimi takes the depformer
        # output count.
        generated_codebooks = (
            raw_config.get("dep_q")
            or raw_config.get("n_q")
            or raw_config.get("audio_codebooks", 8)
        )

        lm_config = models.LmConfig.from_config_dict(raw_config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)

        # Load PyTorch weights into the bf16 model FIRST. If we quantize
        # before loading, the model's parameter names change to .biases /
        # .scales which the PyTorch checkpoint doesn't have → load fails.
        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if self.quantize:
            # Two constraints on what we can quantize in moshi_mlx 0.3.0:
            #   1. Small projection layers (last dim < group_size) fail with
            #      "matrix needs to be divisible by group size".
            #   2. Cross-attention layers use a hand-rolled fused-QKV slice
            #      (`qkv_w[:d_model].T`) that breaks under quantization —
            #      see modules/transformer.py:92. Skip those.
            #
            # Result: most of the LM gets quantized, but a small slice
            # (cross-attention + small embeddings) stays in bf16. If you
            # still hit shape errors during generate(), it's a deeper
            # incompatibility; fall back to `--quantize 0` (bf16, ~3.6 GB).
            group_size = 64

            def _quantize_predicate(path, module):
                if "cross_attention" in path:
                    return False
                if not isinstance(module, nn.Linear):
                    return False
                return (
                    getattr(module, "weight", None) is not None
                    and module.weight.shape[-1] % group_size == 0
                )

            nn.quantize(
                model,
                bits=self.quantize,
                group_size=group_size,
                class_predicate=_quantize_predicate,
            )

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        self._tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
            temp=self.temp,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=raw_config,
        )
        print(f"[tts] ready. sample rate = {self._tts_model.mimi.sample_rate} Hz")

    # ------------------------------------------------------------------

    def synthesize(
        self, text: str, voice: Optional[str] = None
    ) -> tuple[np.ndarray, int]:
        """Generate audio for `text`. Returns (audio float32 mono, sample_rate)."""
        self._ensure_loaded()
        tts = self._tts_model
        v = voice or self.voice

        if tts.multi_speaker:
            voices = [tts.get_voice_path(v)]
        else:
            voices = []

        # API surface verified against moshi_mlx 0.3.0:
        #   prepare_script(script, padding_between=0)
        #   make_condition_attributes(voices, cfg_coef=None)
        #   generate(all_entries, attributes, ..., cfg_is_no_prefix=True,
        #            cfg_is_no_text=True, on_frame=...)
        all_entries = [tts.prepare_script([text], padding_between=1)]
        all_attributes = [tts.make_condition_attributes(voices, cfg_coef=1.0)]

        wav_frames: queue.Queue = queue.Queue()

        def _on_frame(frame):
            if (frame == -1).any():
                return
            pcm = tts.mimi.decode_step(frame[:, :, None])
            pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
            wav_frames.put_nowait(pcm)

        tts.generate(
            all_entries,
            all_attributes,
            cfg_is_no_prefix=True,
            cfg_is_no_text=True,
            on_frame=_on_frame,
        )

        # Drain queue
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(wav_frames.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            return np.zeros(0, dtype=np.float32), tts.mimi.sample_rate

        audio = np.concatenate(chunks, axis=-1).astype(np.float32)
        return audio, tts.mimi.sample_rate
