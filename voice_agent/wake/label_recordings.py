"""Label recorded wake-word clips for retraining.

The agent (run with `--record-wake DIR`) saves clips into:
    DIR/detections/   — the wake word fired (verify: real or false positive?)
    DIR/nearmiss/     — score approached the threshold but didn't fire
    DIR/misses/       — you flagged "I said it but it didn't hear" in the webui

This tool plays each unlabeled clip and asks you to label it, writing the
label back into the clip's sidecar .json. At the end it can export two flat
folders ready for openWakeWord retraining:
    DIR/_train_positive/   — clips that ARE the wake word
    DIR/_train_negative/   — clips that are NOT (false positives, noise)

Usage:
    python wake/label_recordings.py /path/to/record_dir          # interactive
    python wake/label_recordings.py /path/to/record_dir --export # build _train_* sets
    python wake/label_recordings.py /path/to/record_dir --stats  # just counts
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _play(wav: Path) -> None:
    """Play a wav via sounddevice if available, else macOS afplay."""
    try:
        import soundfile as sf
        import sounddevice as sd
        audio, sr = sf.read(str(wav), dtype="float32")
        sd.play(audio, sr)
        sd.wait()
    except Exception:
        import subprocess
        subprocess.run(["afplay", str(wav)], check=False)


def _iter_clips(root: Path):
    for sub in ("detections", "nearmiss", "misses"):
        d = root / sub
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            meta = wav.with_suffix(".json")
            yield sub, wav, meta


def cmd_stats(root: Path) -> None:
    counts: dict[str, dict] = {}
    for sub, wav, meta in _iter_clips(root):
        c = counts.setdefault(sub, {"total": 0, "positive": 0, "negative": 0,
                                    "unlabeled": 0})
        c["total"] += 1
        label = None
        if meta.exists():
            try:
                label = json.loads(meta.read_text()).get("label")
            except json.JSONDecodeError:
                pass
        if label == "positive":
            c["positive"] += 1
        elif label == "negative":
            c["negative"] += 1
        else:
            c["unlabeled"] += 1
    if not counts:
        print("(aucun clip trouvé)")
        return
    print(f"{'dossier':12s} {'total':>6} {'pos':>5} {'neg':>5} {'à faire':>8}")
    for sub, c in counts.items():
        print(f"{sub:12s} {c['total']:>6} {c['positive']:>5} "
              f"{c['negative']:>5} {c['unlabeled']:>8}")


def cmd_label(root: Path) -> None:
    clips = [(sub, wav, meta) for sub, wav, meta in _iter_clips(root)]
    todo = []
    for sub, wav, meta in clips:
        label = None
        if meta.exists():
            try:
                label = json.loads(meta.read_text()).get("label")
            except json.JSONDecodeError:
                pass
        if label not in ("positive", "negative", "skip"):
            todo.append((sub, wav, meta))

    if not todo:
        print("Tout est déjà labellisé. (--stats pour les compteurs)")
        return

    print(f"{len(todo)} clip(s) à labelliser.")
    print("Commandes : [p]ositif (c'est le mot)  [n]égatif (faux/bruit)  "
          "[r]ejouer  [s]kip  [q]uitter\n")

    for i, (sub, wav, meta) in enumerate(todo, 1):
        score = "?"
        if meta.exists():
            try:
                score = json.loads(meta.read_text()).get("score", "?")
            except json.JSONDecodeError:
                pass
        print(f"[{i}/{len(todo)}] {sub}/{wav.name}  (score={score})")
        _play(wav)
        while True:
            try:
                key = input("  p/n/r/s/q > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if key == "r":
                _play(wav)
                continue
            if key in ("p", "n", "s", "q"):
                break
            print("  touche inconnue")

        if key == "q":
            print("Arrêt.")
            return
        label = {"p": "positive", "n": "negative", "s": "skip"}[key]
        data = {}
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
            except json.JSONDecodeError:
                pass
        data["label"] = label
        meta.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\nTerminé. `--export` pour construire les jeux d'entraînement.")


def cmd_export(root: Path) -> None:
    pos_dir = root / "_train_positive"
    neg_dir = root / "_train_negative"
    pos_dir.mkdir(exist_ok=True)
    neg_dir.mkdir(exist_ok=True)
    npos = nneg = 0
    for sub, wav, meta in _iter_clips(root):
        if not meta.exists():
            continue
        try:
            label = json.loads(meta.read_text()).get("label")
        except json.JSONDecodeError:
            continue
        if label == "positive":
            shutil.copy(wav, pos_dir / f"{sub}_{wav.name}")
            npos += 1
        elif label == "negative":
            shutil.copy(wav, neg_dir / f"{sub}_{wav.name}")
            nneg += 1
    print(f"Exporté {npos} positifs → {pos_dir}")
    print(f"Exporté {nneg} négatifs → {neg_dir}")
    print("\nUtilise ces dossiers dans le notebook openWakeWord :")
    print("  - positifs = exemples réels de « billou » (ta voix !)")
    print("  - négatifs = faux déclenchements à NE PLUS détecter")
    print("Ré-entraîne en partant du modèle existant pour l'améliorer.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_dir", help="Dossier passé à --record-wake.")
    parser.add_argument("--stats", action="store_true", help="Juste les compteurs.")
    parser.add_argument("--export", action="store_true",
                        help="Construire _train_positive/ et _train_negative/.")
    args = parser.parse_args()

    root = Path(args.record_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"dossier introuvable : {root}")

    if args.stats:
        cmd_stats(root)
    elif args.export:
        cmd_export(root)
    else:
        cmd_label(root)
        print()
        cmd_stats(root)


if __name__ == "__main__":
    main()
