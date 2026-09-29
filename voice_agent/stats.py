"""Summarize the per-turn metrics log written by agent.py (--metrics-log).

Reads a JSONL file (one turn per line) and prints aggregate timing stats so you
can see how fast/slow the pipeline is: time to understand (STT), to think (LM),
to speak (TTS), and total response latency — plus tool usage.

Usage:
    python stats.py                       # reads logs/metrics.jsonl
    python stats.py path/to/metrics.jsonl
    python stats.py --last 50             # only the most recent 50 turns
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (json key, label, unit) for the timing fields we summarize.
FIELDS = [
    ("stt_ms", "Compréhension (STT)", "ms"),
    ("lm_ms", "Génération (LM)", "ms"),
    ("lm2_ms", "2e passage (vision/recherche)", "ms"),
    ("tts_ms", "Synthèse (TTS)", "ms"),
    ("response_ms", "Latence totale avant parole", "ms"),
    ("audio_s", "Durée parlée", "s"),
    ("utterance_s", "Durée parole utilisateur", "s"),
]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=str(HERE / "logs" / "metrics.jsonl"),
                        help="JSONL metrics file (default logs/metrics.jsonl).")
    parser.add_argument("--last", type=int, default=None,
                        help="Only summarize the most recent N turns.")
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    if not path.exists():
        raise SystemExit(f"Fichier introuvable : {path}")

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if args.last:
        rows = rows[-args.last:]
    if not rows:
        raise SystemExit("Aucun tour à analyser.")

    print(f"\n{len(rows)} tours — {path}\n")
    header = f"{'métrique':32s} {'moy':>8} {'méd':>8} {'p90':>8} {'min':>8} {'max':>8}"
    print(header)
    print("-" * len(header))
    for key, label, unit in FIELDS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        print(f"{label:32s} {statistics.mean(vals):>8.0f} "
              f"{statistics.median(vals):>8.0f} {_pct(vals, 0.9):>8.0f} "
              f"{min(vals):>8.0f} {max(vals):>8.0f}  {unit}")

    # Tool usage + mode counts.
    tool_counts: dict[str, int] = {}
    n_vision = n_search = 0
    for r in rows:
        for t in r.get("tools", []) or []:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        n_vision += bool(r.get("vision"))
        n_search += bool(r.get("search"))
    print(f"\nVision : {n_vision} tours   |   Recherche web : {n_search} tours")
    if tool_counts:
        print("Outils : " + ", ".join(f"{k}×{v}" for k, v in
                                       sorted(tool_counts.items(), key=lambda x: -x[1])))
    print()


if __name__ == "__main__":
    main()
