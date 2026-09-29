# Reachy voice agent — local French assistant on Apple Silicon

```
"Billou" ─► openWakeWord ─► Whisper large-v3-turbo ─► Qwen3.5-4B (VLM) ─► Supertonic 3 ─► 🔊
 wake word    + verifier       speech → French text    reply + [tool:…]     text → speech
                                  .venv_whisper           .venv_lm         .venv_supertonic
```

Everything runs on-device on a Mac mini M4 (12 GB). `agent.py` keeps the three
models loaded as long-lived subprocesses (one venv each, because their MLX pins
conflict) and talks to them with JSON over pipes. Why these models and not
others: see [MODELES_TESTES.md](MODELES_TESTES.md).

What Bilou can do:

- **Wake word** "Billou", with a custom verifier trained on your own voice → [wake/README.md](wake/README.md)
- **Conversation mode**: stays awake for follow-ups (no wake word) until 6 s of silence
- **Barge-in**: talk over him and he stops to listen
- **Tools** the LM calls with `[tool:name] args` lines:

  | Tool | What happens |
  |---|---|
  | `sleep` | goes back to sleep, ends the conversation |
  | `search` | web search (free DuckDuckGo, or Brave if `BRAVE_API_KEY` is set), then a 2nd LM pass speaks the answer |
  | `vision` | grabs a camera frame and re-asks the VLM with the image |
  | `timer` | background timer, announced out loud when it fires |
  | `emote`, `look`, `nod`, `shake`, `dance` | head/antenna body language, played *while* he speaks ([animations.py](animations.py)) |

- **Web UI** (`--webui`): mic meter, transcripts, mic picker, type-to-speak, "missed wake word" button
- **Metrics**: one JSON line per turn in `logs/metrics.jsonl`, summarized by `stats.py`

## Setup

```bash
cd voice_agent

# Main venv: agent.py + TTS worker + wake word
uv venv .venv_supertonic --python 3.12
uv pip install --python .venv_supertonic/bin/python -r requirements.txt
.venv_supertonic/bin/python -c "import openwakeword; openwakeword.utils.download_models()"

# STT worker (Whisper)
uv venv .venv_whisper --python 3.12
uv pip install --python .venv_whisper/bin/python -r stt/requirements.txt

# Chat LM worker (Qwen3.5-4B VLM)
uv venv .venv_lm --python 3.12
uv pip install --python .venv_lm/bin/python -r lm/requirements.txt

# Optional — wake-word tooling (threshold meter, labeling, verifier training)
uv venv .venv_wake --python 3.13
uv pip install --python .venv_wake/bin/python -r wake/requirements.txt
```

Model weights download from Hugging Face on first launch (~1.5 GB Whisper,
~2.9 GB Qwen3.5-4B, plus Supertonic). The first start takes a while; after
that each worker loads and warms up in ~10–30 s.

## Run

```bash
./launch.sh                      # my usual setup (see below), extra flags pass through
./launch.sh --output-device 1
```

or by hand:

```bash
.venv_supertonic/bin/python agent.py --wake wake/models/billou.onnx \
    --wake-verifier wake/models/billou_verifier.joblib --webui [mode flags]
```

### Audio modes

| Mode | Flags | Mic / speaker | Robot | Camera for `vision` |
|---|---|---|---|---|
| Full robot | *(none)* | Reachy's (hardware echo cancel — best barge-in) | wakes/sleeps with sound, moves | yes, via `mini.media` |
| Hybrid | `--no-robot --motion` | Mac's | moves, silent animations | only with `--camera-index N` |
| Laptop only | `--no-robot` | Mac's | not connected | only with `--camera-index N` |

`launch.sh` uses the hybrid mode.

### Useful flags

| Flag | Default | Use it when… |
|---|---|---|
| `--wake-threshold` | 0.5 | it wakes on noise (raise) or misses you (lower) |
| `--vad-threshold` | 0.02 | he never goes back to sleep (raise) or misses quiet speech (lower) |
| `--conversation-timeout` | 6 | you want a longer/shorter follow-up window |
| `--barge-threshold` / `--barge-echo-gain` | 0.05 / 0.6 | his own voice interrupts him (raise both), or he's hard to interrupt |
| `--no-barge` | | you'd rather he never gets interrupted |
| `--output-device N` | system default | playback freezes or errors on the Reachy USB audio device (use the Mac speakers' index) |
| `--camera-index N` | | vision in hybrid/laptop mode (Reachy camera as a USB webcam; needs macOS camera permission for your terminal) |
| `--record-wake DIR` | | collecting clips to retrain the wake-word verifier |
| `--save-audio DIR` | | keeping every utterance as WAV + JSON |
| `--metrics-log FILE` / `--no-metrics` | `logs/metrics.jsonl` | |
| `--no-tools` | | plain chat, no actions |

List audio devices: `.venv_supertonic/bin/python -c "import sounddevice as sd; print(sd.query_devices())"`

## Metrics

```bash
.venv_supertonic/bin/python stats.py             # all turns
.venv_supertonic/bin/python stats.py --last 50
```

Prints mean / median / p90 / min / max for time to understand (`stt_ms`), to
think (`lm_ms`, plus `lm2_ms` for the vision/search 2nd pass), to synthesize
(`tts_ms`), the total latency before he starts talking (`response_ms`), and tool
usage. The raw JSONL is easy to load in pandas for anything else.

## Test a single stage

```bash
.venv_lm/bin/python lm/test_chat_lm.py                        # chat REPL with the LM
.venv_lm/bin/python lm/test_chat_lm.py --text "Salut Bilou !"
.venv_wake/bin/python wake/wake_word.py --model wake/models/billou.onnx --meter   # live wake score
```

## Memory budget (12 GB)

| Component | Approx. RAM |
|---|---|
| Qwen3.5-4B 4-bit (`.venv_lm`) | ~3 GB |
| Whisper large-v3-turbo (`.venv_whisper`) | ~1.6 GB |
| Supertonic 3 + wake word (`.venv_supertonic`) | ~0.6 GB |
| Python/MLX runtime, 3 processes | ~1.5 GB |
| macOS + background apps | ~3–4 GB |

## Troubleshooting

- **Freezes right after he speaks** — mic and speaker on the same Reachy USB
  audio device; the agent prints a ⚠ at startup when that's the case. Use
  `--output-device` with another device, or headphones.
- **"Camera is not initialized"** — you're in hybrid/laptop mode (Reachy media
  off). Use full-robot mode or `--camera-index 0`.
- **LM worker fails to import `transformers`** — `huggingface-hub` got
  downgraded by another install:
  `uv pip install --python .venv_lm/bin/python -U "huggingface-hub>=1.5.0,<2.0"`.
- **Barge-in on the Mac mic is approximate** — there's no real echo
  cancellation, only "mic level minus a fraction of his own level". Tune the
  barge flags, use Reachy's mic, or headphones.

## Files

```
voice_agent/
├── agent.py            orchestrator: workers, turn loop, tools, barge-in, metrics
├── audio_io.py         Reachy/laptop audio backends, VAD + wake-gated capture
├── wake_detector.py    openWakeWord wrapper (+ custom verifier)
├── tools.py            tool registry + implementations
├── animations.py       head/antenna motions for the body-language tools
├── webui.py            FastAPI web UI (+ webui_index.html)
├── stats.py            metrics summary
├── launch.sh           my usual launch command
├── requirements.txt    main venv (.venv_supertonic)
├── stt/                Whisper worker (.venv_whisper)
├── lm/                 chat-LM worker + REPL tester (.venv_lm)
├── tts/                Supertonic worker (.venv_supertonic)
├── wake/               wake-word models, verifier training, labeling tools
└── logs/               metrics.jsonl (gitignored)
```
