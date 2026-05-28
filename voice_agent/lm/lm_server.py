"""Long-lived LM worker. Runs inside .venv_lm.

Protocol (one JSON object per line, both directions):

    out:  {"ready": true}                   (once, after model loads)

    in:   {"audio_path": "/tmp/x.wav",
           "system_prompt": "..."}          (system_prompt optional)
    out:  {"heard": "...", "reply": "...", "raw": "...", "error": null}

    in:   {"command": "clear_history"}      (resets the conversation)
    out:  {"ok": true}

    in:   {"command": "get_history"}
    out:  {"history": [...]}

On failure: {"heard": null, "reply": null, "error": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback

from audio_lm import AudioLM, DEFAULT_SYSTEM_PROMPT


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[lm_server] {msg}", file=sys.stderr, flush=True)


def main():
    lm = AudioLM()
    log("loading model...")
    lm._ensure_loaded()
    log("model loaded.")
    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)

            if job.get("command") == "clear_history":
                lm.clear_history()
                emit({"ok": True})
                continue
            if job.get("command") == "get_history":
                emit({"history": lm.get_history()})
                continue

            result = lm.respond(
                job["audio_path"],
                system_prompt=job.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                user_prompt=job.get("user_prompt", "Écoute cet audio et réponds."),
            )
            emit({**result, "error": None})
        except Exception as e:
            tb = traceback.format_exc()
            log(tb)
            emit({"heard": None, "reply": None, "raw": None,
                  "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
