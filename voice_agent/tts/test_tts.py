"""Standalone tester for Kyutai TTS on MLX.

Two modes:

  ── One-shot (default if you pass text on the CLI):
       python test_tts.py "Bonjour, je suis Reachy."

  ── Interactive (no text on the CLI):
       python test_tts.py

In interactive mode you get a prompt where:
  - typing anything synthesizes + plays it (timing is printed)
  - `:voice <path>`       switches to a voice from kyutai/tts-voices
  - `:voices`             lists voices already downloaded locally
  - `:save <path.wav>`    next utterance is saved instead of played
  - `:reachy on / off`    toggles between laptop and Reachy speakers
  - `:quit`               (or Ctrl-D)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make the shared `audio_io.py` (one level up) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

# Engine modules are imported lazily inside `make_tts` so this file can be
# run from either .venv_tts (kyutai) or .venv_kokoro (kokoro) without
# requiring the other engine's deps to be installed.


# ---------------------------------------------------------------------------
# Voice discovery
# ---------------------------------------------------------------------------

def list_local_voices() -> list[str]:
    """Return voice relative paths that are already in the HF cache."""
    cache = Path.home() / ".cache/huggingface/hub/models--kyutai--tts-voices/snapshots"
    if not cache.exists():
        return []
    voices: set[str] = set()
    for snap_dir in cache.iterdir():
        if not snap_dir.is_dir():
            continue
        for root, _dirs, files in os.walk(snap_dir):
            root_path = Path(root)
            for f in files:
                if f.endswith(".wav"):
                    rel = (root_path / f).relative_to(snap_dir)
                    voices.add(str(rel))
    return sorted(voices)


# A short curated list. The repo is bigger — see the HF page for the full set.
SUGGESTED_VOICES = [
    ("expresso/ex03-ex01_happy_001_channel1_334s.wav",      "English, happy male"),
    ("expresso/ex04-ex01_default_001_channel1_334s.wav",    "English, neutral male"),
    ("siwis/...",                                           "French (browse the repo)"),
]


# ---------------------------------------------------------------------------
# Generation + playback helpers
# ---------------------------------------------------------------------------

def play_or_save(audio: np.ndarray, sr: int, save_to: Optional[str], use_reachy: bool):
    if save_to:
        sf.write(save_to, audio, sr, subtype="PCM_16")
        print(f"  wrote → {save_to}")
        return
    if use_reachy:
        from reachy_mini import ReachyMini
        from audio_io import ReachyBackend
        with ReachyMini() as mini:
            mini.wake_up()
            try:
                ReachyBackend(mini).play(audio, sr)
            finally:
                mini.goto_sleep()
    else:
        from audio_io import LaptopBackend
        LaptopBackend().play(audio, sr)


def synthesize_and_report(tts: KyutaiTTS, text: str, voice: str) -> tuple[np.ndarray, int, float]:
    t0 = time.monotonic()
    audio, sr = tts.synthesize(text, voice=voice)
    gen_dt = time.monotonic() - t0
    duration = len(audio) / sr if sr else 0
    rtf = gen_dt / max(duration, 1e-6)
    print(f"  [gen {gen_dt:.2f}s | audio {duration:.2f}s | RTF {rtf:.2f}]")
    return audio, sr, gen_dt


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_oneshot(tts: KyutaiTTS, text: str, voice: str,
                save_to: Optional[str], use_reachy: bool) -> None:
    print(f"\nSynthesizing: {text!r}")
    audio, sr, _ = synthesize_and_report(tts, text, voice)
    play_or_save(audio, sr, save_to, use_reachy)


def run_interactive(tts: KyutaiTTS, voice: str, use_reachy: bool) -> None:
    save_next: Optional[str] = None
    local_voices = list_local_voices()

    print()
    print("Interactive TTS. Type text to speak it. Commands:")
    print("  :voice <path>      change voice (use :voices to list local ones)")
    print("  :voices            list voices already in the HF cache")
    print("  :save <file.wav>   write next utterance to file instead of playing")
    print("  :reachy on|off     toggle Reachy speaker playback")
    print("  :quit              exit (or Ctrl-D / Ctrl-C)")
    print(f"\ncurrent voice : {voice}")
    print(f"play target   : {'Reachy speaker' if use_reachy else 'laptop default'}")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        # Commands -----------------------------------------------------------
        if line in (":quit", ":q", ":exit"):
            return

        if line == ":voices":
            print(f"\n{len(local_voices)} voice(s) downloaded:")
            for v in local_voices[:40]:
                print(f"  {v}")
            if len(local_voices) > 40:
                print(f"  ... and {len(local_voices) - 40} more")
            print("\nSuggested starting points:")
            for path, desc in SUGGESTED_VOICES:
                print(f"  {path}    ({desc})")
            print("Full list: https://huggingface.co/kyutai/tts-voices/tree/main")
            continue

        if line.startswith(":voice "):
            new = line.split(" ", 1)[1].strip()
            voice = new
            print(f"voice set: {voice}")
            continue

        if line.startswith(":save "):
            save_next = line.split(" ", 1)[1].strip()
            print(f"next utterance will be saved to: {save_next}")
            continue

        if line.startswith(":reachy"):
            arg = line.split()[-1] if " " in line else "on"
            use_reachy = (arg == "on")
            print(f"play target: {'Reachy speaker' if use_reachy else 'laptop default'}")
            continue

        # Otherwise: synthesize the text -------------------------------------
        try:
            audio, sr, _ = synthesize_and_report(tts, line, voice)
            play_or_save(audio, sr, save_next, use_reachy)
            save_next = None
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_tts(engine: str, voice: Optional[str], quantize: int):
    """Instantiate the chosen TTS engine. Voice defaults are engine-specific.

    Engines are imported lazily so each venv only needs its own engine's deps:
      - kyutai → activate .venv_tts   (needs moshi_mlx)
      - kokoro → activate .venv_kokoro (needs mlx-audio)
    """
    q = quantize or None
    if engine == "kokoro":
        from tts_kokoro import KokoroTTS, DEFAULT_FR_VOICE as DEFAULT_VOICE
        return KokoroTTS(voice=voice or DEFAULT_VOICE, quantize=q)
    from tts_kyutai import KyutaiTTS, DEFAULT_FR_VOICE as DEFAULT_VOICE
    return KyutaiTTS(voice=voice or DEFAULT_VOICE, quantize=q)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?",
                        help="Text to speak. Omit for interactive mode.")
    parser.add_argument("--engine", choices=["kyutai", "kokoro"], default="kyutai",
                        help="Which TTS to use. Kokoro is much smaller / faster, "
                             "Kyutai sounds more expressive.")
    parser.add_argument("--voice",
                        help="Voice id. Engine-specific: a path inside "
                             "kyutai/tts-voices for kyutai, or a name like "
                             "'ff_siwis' for kokoro.")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="Bits per weight (0 = bf16). For Kyutai, "
                             "use 8 to halve RTF on M4 base; 4 is smallest.")
    parser.add_argument("--out", help="One-shot: write to WAV instead of playing.")
    parser.add_argument("--reachy", action="store_true",
                        help="Play through Reachy's speaker.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup (first generation includes MLX kernel JIT).")
    args = parser.parse_args()

    tts = make_tts(args.engine, args.voice, args.quantize)
    args.voice = tts.voice   # so the interactive prompt prints the resolved default

    if not args.no_warmup:
        print("\n[warmup] compiling MLX kernels (~10-15 s, one-time)...")
        t0 = time.monotonic()
        _ = tts.synthesize("Test de chauffe.")
        print(f"[warmup] done in {time.monotonic() - t0:.2f} s")

    if args.text:
        run_oneshot(tts, args.text, args.voice, args.out, args.reachy)
    else:
        run_interactive(tts, args.voice, args.reachy)


if __name__ == "__main__":
    main()
