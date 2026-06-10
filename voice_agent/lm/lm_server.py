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
import os
import sys
import traceback

# Skip huggingface_hub's slow online file-existence checks IF the model is
# already cached. Opt-in via HF_HUB_OFFLINE=1 (don't force it, so switching to
# a not-yet-downloaded model still works). The warmup below is what actually
# removes the first-turn delay.
#   export HF_HUB_OFFLINE=1   # for an extra speedup once everything is cached

import numpy as np

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
    log("warming up (compiles MLX kernels)...")
    # Run one throwaway inference on 1s of silence so the FIRST real turn
    # doesn't pay the kernel-compilation + HF-check cost. Then wipe history.
    try:
        lm.respond(np.zeros(16000, dtype=np.float32), max_tokens=4)
    except Exception as e:
        log(f"warmup failed (non-fatal): {e}")
    lm.clear_history()
    log("ready.")
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
            if job.get("command") == "respond_text":
                # 2nd-pass text turn (e.g. after a web search).
                reply = lm.respond_text(
                    job["text"],
                    extra_system=job.get("extra_system", ""),
                )
                emit({"reply": reply, "error": None})
                continue

            result = lm.respond(
                job["audio_path"],
                extra_system=job.get("extra_system", ""),
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
