"""Train a custom verifier from your recorded + labeled wake-word clips.

openWakeWord supports a tiny second-stage "verifier" model: when the main
model fires, the verifier checks the clip against YOUR voice profile (positive
clips) and YOUR false-activation profile (negative clips). It's a logistic
regression on the audio embeddings — trains in seconds, needs only
scikit-learn (no GPU, no torch), and both cuts false positives AND improves
detection of your specific voice.

Workflow:
    1. Run the agent with  --record-wake DIR  and use it for a while.
       Click "raté" in the web UI whenever it misses your wake word.
    2. Label the clips:   python wake/label_recordings.py DIR
       (positive = real wake word, negative = false trigger / noise)
    3. Export:            python wake/label_recordings.py DIR --export
    4. Train:             python wake/train_verifier.py DIR
    5. Use it:            python ../agent.py --wake wake/models/billou.onnx \
                                   --wake-verifier wake/models/billou_verifier.joblib

Run this in .venv_wake (it has openwakeword + scikit-learn).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_dir",
                        help="Folder passed to --record-wake (must contain "
                             "_train_positive/ and _train_negative/ from "
                             "`label_recordings.py --export`).")
    parser.add_argument("--model", default="models/billou.onnx",
                        help="Base wake model the verifier sits on top of "
                             "(default models/billou.onnx, relative to wake/).")
    parser.add_argument("--out",
                        help="Output .joblib path (default: <model>_verifier.joblib).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(args.record_dir).expanduser().resolve()
    pos = root / "_train_positive"
    neg = root / "_train_negative"

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (here / model_path).resolve()
    out = Path(args.out) if args.out else model_path.with_name(
        model_path.stem + "_verifier.joblib")

    # Sanity checks
    if not pos.is_dir() or not neg.is_dir():
        sys.exit(f"Manque {pos} ou {neg}.\n"
                 f"Lance d'abord : python wake/label_recordings.py {root} --export")
    n_pos = len(list(pos.glob("*.wav")))
    n_neg = len(list(neg.glob("*.wav")))
    print(f"positifs (ta voix disant le mot) : {n_pos}")
    print(f"négatifs (faux déclenchements)   : {n_neg}")
    if n_pos < 3 or n_neg < 3:
        print("\n⚠ Très peu de clips. Le verifier marchera mieux avec ~10+ de "
              "chaque. Continue à collecter avec --record-wake, mais on tente "
              "quand même.")
    if not model_path.exists():
        sys.exit(f"Modèle de base introuvable : {model_path}")

    from openwakeword.custom_verifier_model import train_custom_verifier

    # NB: despite the docstring saying "path to a directory",
    # train_custom_verifier iterates its argument directly — it needs a LIST
    # of WAV file paths, not a directory string.
    pos_clips = [str(p) for p in sorted(pos.glob("*.wav"))]
    neg_clips = [str(p) for p in sorted(neg.glob("*.wav"))]

    print(f"\nEntraînement du verifier sur {model_path.name}…")
    train_custom_verifier(
        positive_reference_clips=pos_clips,
        negative_reference_clips=neg_clips,
        output_path=str(out),
        model_name=str(model_path),
    )
    print(f"\n✅ Verifier sauvegardé : {out}")
    print("\nUtilise-le :")
    print(f"  python ../agent.py --wake {model_path} \\")
    print(f"      --wake-verifier {out} --webui")


if __name__ == "__main__":
    main()
