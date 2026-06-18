"""Standalone tester for step 2 (text → French reply via Gemma on MLX).

    # One-shot:
    python test_chat_lm.py --text "Quelle heure est-il ?"

    # Interactive REPL (keeps conversation history):
    python test_chat_lm.py
"""

from __future__ import annotations

import argparse
import time

from chat_lm import ChatLM, CHAT_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="A single message to send, then exit.")
    parser.add_argument("--system", default=CHAT_SYSTEM_PROMPT,
                        help="Override the system prompt.")
    args = parser.parse_args()

    lm = ChatLM(verbose=False)
    print("Loading + warming up the model...")
    lm.respond("Bonjour", system_prompt=args.system, max_tokens=4)
    lm.clear_history()

    def say(text: str) -> None:
        t0 = time.monotonic()
        reply = lm.respond(text, system_prompt=args.system)
        print(f"\n[reply] {reply}\n[timing] {time.monotonic() - t0:.2f} s\n")

    if args.text:
        say(args.text)
        return

    print("\nInteractive. Type a message in French. Ctrl-C / empty line to quit.\n")
    try:
        while True:
            text = input("> ").strip()
            if not text:
                break
            say(text)
    except (KeyboardInterrupt, EOFError):
        print("\nStopping.")


if __name__ == "__main__":
    main()
