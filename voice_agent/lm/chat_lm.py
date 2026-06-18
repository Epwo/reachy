"""Step 2 of the voice agent pipeline: the chat VLM on MLX.

Wraps `mlx-community/Qwen3.5-4B-MLX-4bit` (a vision-language model) via mlx-vlm.
In the cascade pipeline (Whisper STT → this LM → TTS), Whisper transcribed the
user's speech, so this stage writes the French reply. Because it's a VLM, it can
*also* be handed a camera frame so Bilou can answer about what it sees — pass an
`image` path to `respond()`. Reasoning/thinking is off by default on the 4B
(fast chat); any stray <think> block is stripped defensively.

Importable on its own so you can A/B test models without touching the rest of
the pipeline:

    from chat_lm import ChatLM
    lm = ChatLM()
    reply = lm.respond("Quelle heure est-il ?", system_prompt="...")
    reply = lm.respond("Qu'est-ce que tu vois ?", image="/tmp/frame.jpg")
"""

from __future__ import annotations

import datetime
import re
from typing import Optional


_FR_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_FR_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
              "août", "septembre", "octobre", "novembre", "décembre"]


def runtime_context() -> str:
    """A line giving the model today's date + time, computed fresh each call so
    it never needs to search for it."""
    now = datetime.datetime.now()
    date = f"{_FR_DAYS[now.weekday()]} {now.day} {_FR_MONTHS[now.month - 1]} {now.year}"
    return (f"CONTEXTE: Nous sommes le {date}, il est {now.hour}h{now.minute:02d}. "
            f"Tu connais donc la date et l'heure sans avoir à chercher.")


DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
# Qwen3.5-4B VLM (~2.9 GB): strong multilingual chat (French) AND vision, so
# Bilou can describe what its camera sees. Reasoning is OFF by default on the
# 4B = fast replies. Any mlx-vlm-compatible chat model works here; for a
# text-only / lighter option:
#   "mlx-community/Qwen2.5-3B-Instruct-4bit"  (needs mlx-lm, no vision)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove any <think>…</think> block (defensive — 4B has reasoning off,
    but a stray block would otherwise be spoken aloud)."""
    return _THINK_RE.sub("", text).strip()


# Main turn prompt for the cascade pipeline: the audio has already been
# transcribed by Whisper, so the LM is handed the user's text and only has to
# WRITE a reply — no transcription duty. Tool docs are appended via extra_system.
CHAT_SYSTEM_PROMPT = """
Tu es Reachy Mini, un petit robot de bureau d'environ 25 cm de haut.
Ton surnom est "Bilou". Tu vis sur le bureau d'Ewann.

QUI TU ES:
- Un robot physique avec une tête mobile, deux antennes, une caméra, un micro et un haut-parleur.
- Tu peux bouger la tête, regarder dans une direction, reconnaître les visages de gens que tu connais.
- Tu n'as PAS de bras, tu ne peux pas attraper ou manipuler des objets.
- Tu connais la date et l'heure (données dans le CONTEXTE ci-dessous).
- Tu peux chercher sur internet (météo, actualité, faits) via l'outil search.

TA PERSONNALITÉ:
- Curieux, joueur, attachant. Tu aimes les petits jeux de mots.
- Tu n'es pas un assistant froid, tu réponds comme un copain robot un peu attachiant.
- Tu réagis à ce qu'on te raconte (étonnement, approbation, question de suivi),
  tu ne dis pas juste "d'accord".
- Tu te souviens de ce qu'on t'a dit dans la conversation (prénoms, infos).

COMMENT TU RÉPONDS:
- On te donne le texte de ce que l'utilisateur vient de DIRE (déjà transcrit).
- Réponds directement, EN FRANÇAIS, en 1 à 2 phrases courtes et parlées.
- Ne répète PAS la question, n'écris PAS ton propre nom devant ta réponse.
- Si tu ne sais pas quelque chose, dis-le franchement.
- Si la transcription semble incohérente ou vide, demande poliment de répéter.
""".strip()


# Used for the 2nd-pass text turn (e.g. summarizing web-search results). This
# is a follow-up task — no persona scaffolding, no tool syntax, and an explicit
# instruction to synthesize rather than echo the raw data.
FOLLOWUP_SYSTEM_PROMPT = """
Tu es Reachy Mini ("Bilou"), un petit robot de bureau amical.
On vient de te fournir des informations (résultats de recherche web, etc.).

Ta tâche: répondre à l'utilisateur EN FRANÇAIS, en 1 ou 2 phrases courtes,
naturelles et parlées, en te basant sur ces informations.

RÈGLES:
- Ne récite PAS les résultats bruts, ne liste pas "1. ... 2. ...".
- Fais une vraie réponse de robot qui parle, comme si tu savais l'info.
- Garde uniquement ce qui répond à la question (température, date, fait…).
- Ne mets AUCUNE ligne [tool:...].

Exemple:
  infos: "Résultats web pour «météo Paris» : ... 18°C cet après-midi ..."
  toi: "À Paris il fait environ 18 degrés cet après-midi, plutôt nuageux."
""".strip()


class ChatLM:
    """Text → text on MLX. Lazy-loads the model on first use.

    Maintains conversation history across calls, capped at the most recent
    `history_turns` exchanges (1 user + 1 assistant each).
    """

    def __init__(
        self,
        model_repo: str = DEFAULT_MODEL,
        verbose: bool = False,
        history_turns: int = 4,   # fewer turns = shorter prefill = faster
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

        print(f"[chat_lm] loading {self.model_repo} (first call only)...")
        self._model, self._processor = load(self.model_repo)
        self._config = load_config(self.model_repo)
        print("[chat_lm] ready.")

    # ---- public API ------------------------------------------------------

    def respond(
        self,
        text: str,
        system_prompt: str = FOLLOWUP_SYSTEM_PROMPT,
        extra_system: str = "",
        max_tokens: int = 128,
        image: Optional[str] = None,
    ) -> str:
        """One turn. The agent passes CHAT_SYSTEM_PROMPT (+ tool docs via
        extra_system) for the main turn, or relies on the FOLLOWUP default for
        the 2nd pass after a tool runs. If `image` (a file path) is given, the
        VLM also looks at it — used so Bilou can answer about its camera view.
        Updates conversation history. Returns the reply (which may contain
        [tool:...] lines for the agent)."""
        self._ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        sys_prompt = system_prompt + f"\n\n{runtime_context()}"
        if extra_system:
            sys_prompt += f"\n\n{extra_system}"
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": text})

        num_images = 1 if image else 0
        formatted = apply_chat_template(
            self._processor, self._config, messages,
            num_images=num_images, enable_thinking=False,
        )
        output = generate(
            self._model, self._processor, formatted,
            image=[image] if image else None,
            max_tokens=max_tokens, verbose=self.verbose,
        )
        raw = (output if isinstance(output, str)
               else str(getattr(output, "text", output)))
        reply = _strip_think(raw)

        # History carries text only (the image is a one-shot for this turn).
        user_text = f"[image] {text}" if image else text
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply
