"""Compare Gemma's transcripts against Whisper's, across a saved-audio dir.

Usage:
    # Capture some conversation with the agent:
    python ../agent.py --no-robot --save-audio /tmp/reachy_audio

    # Then in this venv (.venv_lm, has mlx-whisper):
    python compare_transcripts.py /tmp/reachy_audio

For each audio file with a matching .json sidecar, runs Whisper, prints
both transcripts, and a quick "match-ish?" heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def normalize(s: str) -> str:
    """Cheap normalization for fuzzy comparison."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def loose_match(a: str, b: str) -> str:
    """Return a short qualitative tag."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return "empty"
    if na == nb:
        return "exact"
    if na in nb or nb in na:
        return "substring"
    # Word overlap ratio
    sa, sb = set(na.split()), set(nb.split())
    if not sa or not sb:
        return "empty"
    overlap = len(sa & sb) / max(len(sa), len(sb))
    if overlap > 0.7:
        return f"close ({overlap*100:.0f}%)"
    if overlap > 0.4:
        return f"partial ({overlap*100:.0f}%)"
    return f"different ({overlap*100:.0f}%)"


def iter_jobs(audio_dir: Path) -> Iterable[tuple[Path, Path, dict]]:
    """Yield (wav_path, json_path, meta) tuples for each pair found."""
    for wav in sorted(audio_dir.glob("*.wav")):
        meta_path = wav.with_suffix(".json")
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[skip] {wav.name}: bad json — {e}")
            continue
        yield wav, meta_path, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_dir", help="Directory written by agent.py --save-audio")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-turbo",
                        help="Whisper model to compare against (mlx-whisper repo).")
    parser.add_argument("--language", default="fr",
                        help="Whisper language hint (default fr).")
    parser.add_argument("--out", help="Optional JSONL output with full results.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        sys.exit(f"not a directory: {audio_dir}")

    # Lazy import so this script can be inspected without mlx-whisper installed
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx-whisper not installed. Run in the LM venv (.venv_lm).")

    print(f"comparing transcripts in {audio_dir} with {args.model}\n")
    rows = []
    for wav, meta_path, meta in iter_jobs(audio_dir):
        gemma = meta.get("gemma_heard", "")
        print(f"── {wav.name}  ({meta.get('duration_s', '?')}s)")
        print(f"   gemma  : {gemma}")
        try:
            result = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=args.model,
                language=args.language,
                fp16=True,
                condition_on_previous_text=False,
            )
            whisper_text = (result.get("text") or "").strip()
        except Exception as e:
            whisper_text = ""
            print(f"   whisper: [error {type(e).__name__}: {e}]")
        else:
            print(f"   whisper: {whisper_text}")
        tag = loose_match(gemma, whisper_text)
        print(f"   verdict: {tag}\n")
        rows.append({
            "wav": wav.name,
            "duration_s": meta.get("duration_s"),
            "gemma_heard": gemma,
            "whisper_heard": whisper_text,
            "verdict": tag,
        })

    if not rows:
        print("(no audio + json pairs found)")
        return

    # Quick summary
    counts: dict[str, int] = {}
    for r in rows:
        key = r["verdict"].split()[0]   # exact / close / partial / different / empty
        counts[key] = counts.get(key, 0) + 1
    print("─── summary ───")
    for k in ("exact", "substring", "close", "partial", "different", "empty"):
        if k in counts:
            print(f"  {k:10s} {counts[k]:3d}")
    print(f"  {'total':10s} {len(rows):3d}")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                            encoding="utf-8")
        print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
