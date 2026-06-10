"""Wake-word detector wrapper, shared by the agent.

Wraps openWakeWord and adapts it to the agent's audio plumbing:
- accepts mono float32 16 kHz audio of ANY chunk size
- internally rebuffers into the 1280-sample (80 ms) int16 frames openWakeWord
  expects
- returns the peak wake score seen across the complete frames in each feed

Requires `openwakeword` + `onnxruntime` in whatever venv runs the agent.
Both are pure-ONNX so they coexist with the Supertonic / Kokoro TTS venvs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


FRAME = 1280   # 80 ms @ 16 kHz — openWakeWord's unit

# Names that ship pre-trained (so `--wake hey_jarvis` works for demos).
PRETRAINED = {"alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy",
              "timer", "weather"}


def _resolve(model_arg: str) -> str:
    if model_arg in PRETRAINED:
        return model_arg
    p = Path(model_arg).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Wake model not found: {model_arg} "
            f"(use a pretrained name {sorted(PRETRAINED)} or a path to a .onnx)"
        )
    return str(p)


class WakeDetector:
    def __init__(self, model: str, threshold: float = 0.5,
                 verifier: str = None, verifier_threshold: float = 0.3):
        from openwakeword.model import Model
        self.threshold = threshold

        model_path = _resolve(model)
        kwargs = dict(wakeword_models=[model_path], inference_framework="onnx")
        if verifier:
            # The verifier dict is keyed by the base model NAME (the key
            # openWakeWord's predict() uses), i.e. the .onnx basename.
            from pathlib import Path
            name = Path(model_path).stem
            kwargs["custom_verifier_models"] = {name: str(Path(verifier).expanduser())}
            kwargs["custom_verifier_threshold"] = verifier_threshold
            print(f"[wake] verifier custom chargé pour '{name}': {verifier}")

        self._model = Model(**kwargs)
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def feed(self, audio_f32: np.ndarray) -> float:
        """Feed mono float32 16 kHz audio. Returns the peak wake score over
        the complete 1280-frames processed this call (0 if none completed)."""
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)
        pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._buf = np.concatenate([self._buf, pcm])

        best = 0.0
        while len(self._buf) >= FRAME:
            frame = self._buf[:FRAME]
            self._buf = self._buf[FRAME:]
            scores = self._model.predict(frame)
            if scores:
                best = max(best, max(scores.values()))
        self.last_score = best
        return best

    def triggered(self, audio_f32: np.ndarray) -> bool:
        return self.feed(audio_f32) >= self.threshold

    def reset(self) -> None:
        """Clear internal state after a detection so we don't re-trigger on
        the tail of the same utterance."""
        self._model.reset()
        self._buf = np.zeros(0, dtype=np.int16)
