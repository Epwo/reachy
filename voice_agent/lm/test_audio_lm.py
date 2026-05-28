"""Standalone tester for step 1 (audio → text via Gemma 3n on MLX).

Two modes:

    # Pass an audio file you already have:
    python test_audio_lm.py --audio /path/to/something.wav

    # Live: capture from laptop mic with VAD, send to the model, print reply:
    python test_audio_lm.py --live
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make the shared `audio_io.py` (one level up) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_lm import AudioLM, DEFAULT_SYSTEM_PROMPT
from audio_io import LaptopBackend, VADCapture


def run_file(path: str, system_prompt: str) -> None:
    lm = AudioLM(verbose=False)
    print(f"\nLoading and inferring on {path}...")
    t0 = time.monotonic()
    reply = lm.respond(path, system_prompt=system_prompt)
    dt = time.monotonic() - t0
    print(f"\n[reply] {reply}")
    print(f"[timing] {dt:.2f} s\n")


def run_live(system_prompt: str) -> None:
    lm = AudioLM(verbose=False)
    print("\nPre-warming the model on a silent clip (first inference is slow)...")
    import numpy as np
    _ = lm.respond(np.zeros(16000, dtype="float32"),
                   system_prompt="Reply with the single word 'pret'.",
                   user_prompt="Es-tu prêt ?", max_tokens=4)

    backend = LaptopBackend()
    capture = VADCapture(backend)
    print("\nLive mic. Speak in French. Ctrl-C to quit.\n")

    try:
        for utterance in capture.utterances():
            t0 = time.monotonic()
            reply = lm.respond(utterance, system_prompt=system_prompt)
            dt = time.monotonic() - t0
            print(f"\n[reply] {reply}")
            print(f"[timing] {dt:.2f} s\n")
    except KeyboardInterrupt:
        print("\nStopping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="Path to a WAV/MP3 to send to the model.")
    src.add_argument("--live", action="store_true",
                     help="Capture from the laptop mic with VAD.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT,
                        help="Override the system prompt.")
    args = parser.parse_args()

    if args.audio:
        run_file(args.audio, args.system)
    else:
        run_live(args.system)


if __name__ == "__main__":
    main()
