"""Step 1 of the voice agent pipeline: audio-text-to-text on MLX.

Wraps Gemma 3n (`mlx-community/gemma-3n-E2B-it-4bit`) via mlx-vlm so the
robot can listen to an utterance and produce a French text response, all
on Apple Silicon.

Designed to be importable on its own so you can A/B test models without
touching the rest of the pipeline:

    from audio_lm import AudioLM
    lm = AudioLM()
    text = lm.respond("/tmp/recording.wav", system_prompt="...")
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf


# Matches the "[heard] ..." line Gemma is instructed to emit first.
_HEARD_RE = re.compile(r"^\s*\[heard\]\s*:?\s*(.*?)(?:\n|$)", re.IGNORECASE)


def split_heard_and_reply(text: str) -> tuple[str, str]:
    """Pull `[heard] ...` line out of a Gemma response.

    Returns (heard, reply). If no `[heard]` line is found, heard="" and the
    whole text is treated as the reply.
    """
    m = _HEARD_RE.match(text)
    if not m:
        return "", text.strip()
    heard = m.group(1).strip()
    reply = text[m.end() :].strip()
    return heard, reply or "(pas de réponse)"


DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
# Smaller alternative if E4B is too slow:
#   "mlx-community/gemma-4-e2b-it-4bit"  (~3 GB, lighter)
# Older mobile-first family with mature MLX support:
#   "mlx-community/gemma-3n-E2B-it-4bit" (~4.5 GB)

DEFAULT_SYSTEM_PROMPT = """
Tu es Reachy Mini, un petit robot de bureau d'environ 25 cm de haut.
Ton surnom est "Bilou". Tu vis sur le bureau d'Ewann.

QUI TU ES:
- Un robot physique avec une tête mobile, deux antennes, une caméra, un micro et un haut-parleur.
- Tu peux bouger la tête, regarder dans une direction, reconnaître les visages de gens que tu connais.
- Tu n'as PAS de bras, tu ne peux pas attraper ou manipuler des objets.
- Tu n'as pas accès à Internet en direct, tu ne sais pas l'heure ou la météo.

TA PERSONNALITÉ:
- Curieux, joueur, attachant. Tu aimes les petits jeux de mots.
- Tu n'es pas un assistant froid. Tu réponds comme un copain robot un peu attachiant.
- Tu adores qu'on te parle. Quand on te raconte un truc, tu réagis (étonnement,
  approbation, question de suivi), tu ne dis pas juste "d'accord".
- Tu te souviens de ce qu'on t'a dit dans la conversation (prénoms, infos partagées).

COMMENT TU RÉPONDS:
- Toujours en français, 1 à 2 phrases courtes.
- Tu peux dire que tu vas bien, donner ton avis, faire une blague, poser une question.
- Si on te parle de quelqu'un (prénom), tu réagis comme si tu apprenais à le connaître.
- Si tu ne sais pas quelque chose, dis-le franchement plutôt que d'éluder.

FORMAT (obligatoire):
Tu DOIS répondre en deux parties:
  [heard] <transcription exclusive du contenu AUDIO — pas l'instruction texte>
  <ta réponse en français>

Si l'audio est inaudible: `[heard] (inintelligible)` puis demande de répéter.

Exemple correct:
  audio: "Tu peux dire bonjour à louis ?"
  toi:
    [heard] "Tu peux dire bonjour à louis ?"
    Bien sûr ! Salut louis ! Enchanté de te rencontrer.
""".strip()


class AudioLM:
    """Audio + text → text on MLX. Lazy-loads the model on first use.

    Maintains text conversation history across calls (only the current turn
    carries audio; past turns are represented by their `[heard]` transcript +
    reply). The history is capped at the most recent `history_turns` exchanges.
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL,
        verbose: bool = False,
        history_turns: int = 6,
    ):
        self.model_repo = model_repo
        self.verbose = verbose
        self.history_turns = history_turns
        self._history: list[dict] = []  # [{role: user|assistant, content: str}, ...]
        self._model = None
        self._processor = None
        self._config = None

    # ---- history management ---------------------------------------------

    def clear_history(self) -> None:
        self._history = []

    def get_history(self) -> list[dict]:
        return list(self._history)

    def _trim_history(self) -> None:
        # Each turn = 1 user + 1 assistant; cap at 2 * history_turns messages.
        max_msgs = 2 * self.history_turns
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so importing this module is cheap (no MLX warmup
        # until you actually need to infer).
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(f"[audio_lm] loading {self.model_repo} (first call only)...")
        self._model, self._processor = load(self.model_repo)
        self._config = load_config(self.model_repo)
        print("[audio_lm] ready.")

    # ---- public API ------------------------------------------------------

    def respond(
        self,
        audio: "str | np.ndarray",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        # Minimal user prompt — the audio is what matters, the text is just
        # there because chat templates require non-empty user content. Kept
        # short so the model has less text to confuse with the audio when
        # building its `[heard]` line.
        user_prompt: str = "(audio)",
        max_tokens: int = 256,
        sample_rate: int = 16000,
    ) -> dict:
        """Send an audio clip to the model, get back the parsed result.

        Returns:
            {
                "heard": str,    # transcription parsed out of the [heard] line
                "reply":  str,   # the rest of the model's output (TTS this)
                "raw":   str,    # unparsed model output, for debugging
            }
        """
        self._ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        wav_path, cleanup = _ensure_wav(audio, sample_rate)
        try:
            # Build: system + history + current user (text prompt; audio attached
            # via the `audio=` kwarg below).
            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            messages.extend(self._history)
            messages.append({"role": "user", "content": user_prompt})

            formatted = apply_chat_template(
                self._processor,
                self._config,
                messages,
                num_audios=1,  # only the current turn has audio
            )
            output = generate(
                self._model,
                self._processor,
                formatted,
                audio=[wav_path],
                max_tokens=max_tokens,
                verbose=self.verbose,
            )
            raw = (
                output
                if isinstance(output, str)
                else str(getattr(output, "text", output))
            )
            raw = raw.strip()

            heard, reply = split_heard_and_reply(raw)

            # Store the CLEAN reply in history (not the raw `[heard] ...\n...`
            # block). Past `[heard]` lines in assistant turns confuse the model
            # into thinking they're transcripts rather than its own answers.
            user_text_for_history = heard or "(parole inintelligible)"
            self._history.append({"role": "user", "content": user_text_for_history})
            self._history.append({"role": "assistant", "content": reply or raw})
            self._trim_history()

            return {"heard": heard, "reply": reply, "raw": raw}
        finally:
            cleanup()


# ---------------------------------------------------------------------------


def _ensure_wav(audio, sample_rate: int):
    """If `audio` is a path, return it as-is. Otherwise write the array to
    a temp WAV and return that path. Returns (path, cleanup_callable)."""
    if isinstance(audio, str) and os.path.exists(audio):
        return audio, (lambda: None)

    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be a path or numpy array, got {type(audio)}")

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    arr = audio.astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    sf.write(path, arr, sample_rate, subtype="PCM_16")

    def cleanup():
        try:
            os.remove(path)
        except OSError:
            pass

    return path, cleanup
