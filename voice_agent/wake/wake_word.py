"""Prototype wake-word — détecte un mot de réveil ("Hey Jarvis" en démo,
"Bilou"/"Reachy" une fois entraînés) avec openWakeWord.

Isolé de la pipeline LLM pour l'instant. Deux modes gérés en même temps :

  - Mot SEUL  : tu dis "Bilou", tu fais une pause, puis tu parles.
  - Mot+phrase: tu dis "Bilou, quelle heure il est ?" d'un coup.

Le script écoute en continu (faible CPU), et quand le score dépasse le
seuil il affiche "🔔 RÉVEILLÉ", puis capture la phrase qui suit (avec un
petit délai d'attente pour le cas "mot seul"). Optionnellement il sauve
le WAV capturé pour écoute.

Usage:
    # Démo immédiate avec un modèle pré-entraîné (anglais) :
    python wake_word.py --model hey_jarvis

    # Avec ton modèle custom une fois entraîné :
    python wake_word.py --model models/bilou.onnx --threshold 0.5

    # Sauver les phrases captées après réveil :
    python wake_word.py --model hey_jarvis --save-dir /tmp/wake_captures

Entraîner "Bilou"/"Reachy" : voir wake/README.md (notebook Colab, ~1h).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from openwakeword.model import Model


SR = 16000
CHUNK = 1280            # 80 ms à 16 kHz — l'unité attendue par openWakeWord


# ---------------------------------------------------------------------------
# Capture de la phrase qui suit le réveil (VAD énergie simple)
# ---------------------------------------------------------------------------

def capture_phrase(
    stream,
    prebuffer: np.ndarray,
    energy_threshold: float = 0.015,
    max_wait_s: float = 1.5,
    silence_ms: float = 700,
    max_phrase_s: float = 8.0,
) -> Optional[np.ndarray]:
    """Capture la parole après le wake word.

    `prebuffer` = audio déjà capté juste après la détection (cas mot+phrase
    enchaîné). On attend jusqu'à `max_wait_s` qu'une parole démarre (cas mot
    seul avec pause), puis on enregistre jusqu'à `silence_ms` de silence.
    Retourne un float32 mono 16 kHz, ou None si rien n'a été dit.
    """
    frames: list[np.ndarray] = []
    started = False
    silent_ms = 0.0
    waited_s = 0.0

    # Le prébuffer peut déjà contenir de la parole (mot+phrase d'un coup).
    if prebuffer is not None and len(prebuffer):
        rms = float(np.sqrt(np.mean(prebuffer ** 2) + 1e-9))
        if rms > energy_threshold:
            started = True
            frames.append(prebuffer)

    t_start = time.monotonic()
    while True:
        audio, _ = stream.read(CHUNK)
        chunk = audio.flatten().astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        if not started:
            waited_s = time.monotonic() - t_start
            if rms > energy_threshold:
                started = True
                frames.append(chunk)
            elif waited_s > max_wait_s:
                return None     # personne n'a parlé après le réveil
        else:
            frames.append(chunk)
            if rms > energy_threshold:
                silent_ms = 0.0
            else:
                silent_ms += 80
                if silent_ms > silence_ms:
                    break
            if (len(frames) * CHUNK) / SR > max_phrase_s:
                break           # garde-fou longueur

    if not frames:
        return None
    return np.concatenate(frames)


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

# Modèles pré-entraînés livrés avec openWakeWord (pour la démo).
PRETRAINED = {"alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy",
              "timer", "weather"}


def resolve_model_arg(model_arg: str) -> str:
    """Un nom pré-entraîné est passé tel quel ; un chemin est résolu absolu."""
    if model_arg in PRETRAINED:
        return model_arg
    p = Path(model_arg).expanduser().resolve()
    if not p.exists():
        raise SystemExit(
            f"Modèle introuvable : {model_arg}\n"
            f"Soit un nom pré-entraîné {sorted(PRETRAINED)},\n"
            f"soit un chemin vers un .onnx custom (voir wake/README.md)."
        )
    return str(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="hey_jarvis",
                        help="Nom pré-entraîné (hey_jarvis, alexa, ...) ou "
                             "chemin vers un .onnx custom.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil de détection 0–1 (défaut 0.5). Plus haut "
                             "= moins de faux positifs mais plus de ratés.")
    parser.add_argument("--cooldown", type=float, default=2.0,
                        help="Secondes de pause après une détection avant de "
                             "pouvoir re-déclencher.")
    parser.add_argument("--no-phrase", action="store_true",
                        help="Ne capture PAS la phrase après le réveil "
                             "(juste détecter le mot).")
    parser.add_argument("--save-dir",
                        help="Sauver les phrases captées (WAV) dans ce dossier.")
    parser.add_argument("--meter", action="store_true",
                        help="Afficher le score du wake word en continu.")
    args = parser.parse_args()

    model_path = resolve_model_arg(args.model)
    print(f"Chargement du modèle wake-word : {args.model}")
    oww = Model(wakeword_models=[model_path], inference_framework="onnx")

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Captures → {save_dir}/")

    # Ring buffer de ~400 ms pour ne pas perdre le début de la phrase quand
    # elle suit le wake word immédiatement.
    ring: deque = deque(maxlen=5)   # 5 × 80 ms = 400 ms

    cooldown_until = 0.0
    print(f"\nEn écoute… dis « {args.model} ». Ctrl-C pour quitter.\n")

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=CHUNK) as stream:
        try:
            while True:
                audio, _ = stream.read(CHUNK)
                frame = audio.flatten()
                ring.append(frame.astype(np.float32) / 32768.0)

                scores = oww.predict(frame)
                score = max(scores.values()) if scores else 0.0

                if args.meter:
                    bar = "█" * int(score * 30)
                    print(f"\r  score {score:0.2f} |{bar:<30}|", end="", flush=True)

                now = time.monotonic()
                if score >= args.threshold and now >= cooldown_until:
                    if args.meter:
                        print()   # newline après la barre
                    print(f"🔔 RÉVEILLÉ  (score={score:.2f})")
                    cooldown_until = now + args.cooldown
                    oww.reset()   # vide l'état interne pour ne pas re-trigger

                    if not args.no_phrase:
                        prebuf = np.concatenate(list(ring)) if ring else None
                        ring.clear()
                        phrase = capture_phrase(stream, prebuf)
                        if phrase is None:
                            print("   (rien entendu après le réveil)\n")
                        else:
                            dur = len(phrase) / SR
                            print(f"   📝 phrase captée : {dur:.2f}s")
                            if save_dir is not None:
                                ts = time.strftime("%Y%m%d-%H%M%S")
                                path = save_dir / f"{ts}.wav"
                                sf.write(str(path), phrase, SR, subtype="PCM_16")
                                print(f"   💾 {path}")
                            print()
                    else:
                        print()
        except KeyboardInterrupt:
            print("\nArrêt.")


if __name__ == "__main__":
    main()
