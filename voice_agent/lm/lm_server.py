"""Long-lived chat-LM worker. Runs inside .venv_lm.

Stage 2 of the cascade pipeline (Whisper STT → this LM → TTS): given the
already-transcribed user text, it writes a French reply. It never touches
audio (that's stt/).

Protocol (one JSON object per line, both directions):

    out:  {"ready": true}                    (once, after model loads)

    in:   {"command": "respond_chat",         (main turn)
           "text": "...", "extra_system": "..."}
    out:  {"reply": "...", "error": null}

    in:   {"command": "respond_text",         (2nd pass, e.g. search summary)
           "text": "...", "extra_system": "..."}
    out:  {"reply": "...", "error": null}

    in:   {"command": "clear_history"}        (resets the conversation)
    out:  {"ok": true}

    in:   {"command": "get_history"}
    out:  {"history": [...]}

On failure: {"reply": null, "error": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback

# Skip huggingface_hub's slow online file-existence checks IF the model is
# already cached. Opt-in via HF_HUB_OFFLINE=1 (don't force it, so switching to
# a not-yet-downloaded model still works). The warmup below is what actually
# removes the first-turn delay.
#   export HF_HUB_OFFLINE=1   # for an extra speedup once everything is cached

from chat_lm import ChatLM, CHAT_SYSTEM_PROMPT


def emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    print(f"[lm_server] {msg}", file=sys.stderr, flush=True)


def main():
    lm = ChatLM()
    log("loading model...")
    lm._ensure_loaded()
    log("warming up (compiles MLX kernels)...")
    # Run one throwaway inference so the FIRST real turn doesn't pay the
    # kernel-compilation cost.
    try:
        lm.respond("Bonjour", system_prompt=CHAT_SYSTEM_PROMPT, max_tokens=4)
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
            cmd = job.get("command")

            if cmd == "clear_history":
                lm.clear_history()
                emit({"ok": True})
            elif cmd == "get_history":
                emit({"history": lm.get_history()})
            elif cmd == "respond_chat":
                # Main turn: write a reply from the user's transcribed text
                # (persona + tool docs). Tool lines are parsed by the agent.
                # Optional image_path = a camera frame for Bilou to look at.
                reply = lm.respond(
                    job["text"],
                    system_prompt=CHAT_SYSTEM_PROMPT,
                    extra_system=job.get("extra_system", ""),
                    image=job.get("image_path"),
                )
                emit({"reply": reply, "error": None})
            elif cmd == "respond_text":
                # 2nd pass (e.g. after a web search): summarize via the default
                # FOLLOWUP prompt (no persona scaffolding).
                reply = lm.respond(
                    job["text"],
                    extra_system=job.get("extra_system", ""),
                )
                emit({"reply": reply, "error": None})
            else:
                emit({"reply": None, "error": f"unknown command: {cmd!r}"})
        except Exception as e:
            tb = traceback.format_exc()
            log(tb)
            emit({"reply": None, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
