# Reachy voice agent — local pipeline on Apple Silicon

```
mic → VAD → Gemma 4 (audio→text) → Kokoro TTS (text→audio) → speaker
       └─── .venv_lm ─────────┘   └────── .venv_kokoro ─────┘
```

Two MLX models in separate venvs (their `mlx` pins conflict; can't share
an environment). The orchestrator (`agent.py`) keeps both as long-lived
subprocesses with models loaded — first turn includes the LM/TTS warmup
(~30 s total), subsequent turns are real-time (sub-2 s on M4 base).

The pipeline ships with Kokoro-82M (RTF ~0.1) by default. An optional
Kyutai-1.6B engine is kept around for quality comparison via `test_tts.py
--engine kyutai`; it lives in `.venv_tts` and isn't wired into the
default pipeline because it runs slower than realtime on M4 base.

## Files

```
voice_agent/
├── README.md
├── agent.py                  orchestrator — spawns both workers, shuttles audio
├── audio_io.py               shared: VAD capture, Reachy/Laptop backends
│
├── lm/                       step 1 — audio → text
│   ├── requirements.txt        deps for .venv_lm (mlx-vlm + Gemma 4)
│   ├── audio_lm.py             Gemma 4 wrapper, used by lm_server
│   ├── lm_server.py            long-lived worker, runs in .venv_lm
│   └── test_audio_lm.py        standalone tester
│
├── tts/                      step 2 — text → audio
│   ├── requirements_kokoro.txt    deps for .venv_kokoro (default TTS, fast)
│   ├── requirements_kyutai.txt    deps for .venv_tts (alternative, expressive)
│   ├── tts_kokoro.py              Kokoro wrapper (active engine)
│   ├── tts_kyutai.py              Kyutai wrapper (for A/B comparison)
│   ├── tts_server.py              long-lived worker, runs in .venv_kokoro
│   └── test_tts.py                standalone tester (interactive, --engine flag)
│
└── (venvs live at this root: .venv_lm/ .venv_kokoro/ .venv_tts/)
```

## Setup

```bash
cd voice_agent

# .venv_lm: audio → text (Gemma 4)
uv venv .venv_lm
source .venv_lm/bin/activate
uv pip install -r lm/requirements.txt
huggingface-cli download mlx-community/gemma-4-e2b-it-4bit    # ~3 GB
deactivate

# .venv_kokoro: text → audio (default TTS)
uv venv .venv_kokoro
source .venv_kokoro/bin/activate
uv pip install -r tts/requirements_kokoro.txt
huggingface-cli download mlx-community/Kokoro-82M-bf16        # ~330 MB
deactivate

# .venv_tts: alternative TTS for comparison (Kyutai 1.6B) — OPTIONAL
uv venv .venv_tts
source .venv_tts/bin/activate
uv pip install -r tts/requirements_kyutai.txt
huggingface-cli download kyutai/tts-1.6b-en_fr                # ~4 GB
huggingface-cli download kyutai/tts-voices
deactivate
```

Total disk: ~3.5 GB for the default pipeline (Gemma + Kokoro), plus ~4 GB
extra if you also want to A/B test Kyutai.

## Test each step on its own

### Step 1 — audio → text (Gemma 4)

```bash
source .venv_lm/bin/activate
python lm/test_audio_lm.py --live              # mic → text
python lm/test_audio_lm.py --audio file.wav    # file → text
deactivate
```

### Step 2 — text → audio (Kokoro / Kyutai)

Two ways to test. **Kokoro** (default, fast):

```bash
source .venv_kokoro/bin/activate
python tts/test_tts.py "Bonjour, je suis Reachy Mini." --engine kokoro
python tts/test_tts.py "Bonjour" --out /tmp/hello.wav --engine kokoro
python tts/test_tts.py "Bonjour" --reachy --engine kokoro
deactivate
```

**Kyutai** (more expressive, slower, optional):

```bash
source .venv_tts/bin/activate
python tts/test_tts.py "Bonjour" --engine kyutai --quantize 8
deactivate
```

**Interactive**, when you want to iterate on phrases / voices and watch
the timings:

```bash
source .venv_kokoro/bin/activate
python tts/test_tts.py --engine kokoro
```

Inside the prompt:

```
> Bonjour, comment vas-tu ?
  [gen 1.82 s | audio 2.04 s | RTF 0.89]

> :voice expresso/ex04-ex01_default_001_channel1_334s.wav
voice set: expresso/ex04-ex01_default_001_channel1_334s.wav

> :voices
12 voice(s) downloaded:
  expresso/ex01-ex01_default_001_channel1_334s.wav
  expresso/ex03-ex01_happy_001_channel1_334s.wav
  siwis/...
  ...
Suggested starting points:
  expresso/ex03-...    (English, happy male)
  ...

> :save /tmp/sample.wav
next utterance will be saved to: /tmp/sample.wav

> Salut.
  [gen 0.91 s | audio 0.83 s | RTF 1.10]
  wrote → /tmp/sample.wav

> :quit
```

The first generation in any process pays a ~10–15 s MLX kernel warmup —
the script does one warmup call automatically before the prompt opens so
the first timing you see is the real steady-state.

## Run the full pipeline

```bash
cd voice_agent
source .venv_kokoro/bin/activate
python agent.py --no-robot              # laptop mic + speaker
# or
python agent.py                         # Reachy mic + speaker
```

On startup the orchestrator spawns both workers in their respective venvs
and waits for both to report ready (each loads + warms its model — about
30 s total on M4 base). Then it listens; speak in French and the robot
replies in French.

## Tools (function calling)

The model can trigger actions by emitting `[tool:name] args` lines in its
reply. The agent parses them, runs the tool, and (for tools that fetch
information, like web search) does a 2nd LM pass to speak a natural answer.
Tools are on by default; disable with `--no-tools`.

Implemented:

- **`sleep`** — "va dormir / tais-toi" → robot goes back to sleep (ends the
  wake-conversation).
- **`search`** — web search. Default backend is **DuckDuckGo** via the free
  `ddgs` package (no API key, no signup):
  ```bash
  source .venv_supertonic/bin/activate
  uv pip install -r requirements_tools.txt   # installs ddgs
  python agent.py --wake wake/models/billou.onnx
  ```
  Optionally, set `BRAVE_API_KEY` to use the Brave Search API instead (higher
  quality, but the free tier needs a verified account). DuckDuckGo needs
  nothing.

Placeholders (acknowledged but not yet wired to motion — edit `tools.py`):
`emote`, `look`, `nod`, `shake`, `dance`.

Add a tool = one `@tool(...)` function in [`tools.py`](tools.py); its
description is shown to the model automatically.

## Memory budget on a 12 GB Mac

| Component                         | Approx RAM   |
| --------------------------------- | ------------ |
| Gemma 4 E2B (bf16, in .venv_lm)   | ~3 GB        |
| Kokoro-82M bf16 (in .venv_kokoro) | ~0.5 GB      |
| Python + MLX runtime per process  | ~0.5 GB each |
| macOS + light background apps     | ~3-4 GB      |
| **Total**                         | **~7-8 GB**  |

Plenty of headroom on 12 GB now. The Kokoro switch is the single
biggest win for this hardware — it removed ~3 GB of resident weights
and the ~1.5 RTF that came with Kyutai.

## Barge-in (interrupting while it speaks)

On by default: talk over Reachy and he stops to listen. Detection is
energy-based with a "poor man's AEC" — the mic level minus a fraction of the
clip's own level at the current playback position, so his own voice doesn't
trigger a false stop.

- **Best with Reachy's mic** (`python agent.py` full robot): the ReSpeaker does
  hardware echo cancellation, so only *your* voice reaches the detector.
- **Laptop mic near the speaker**: his voice can be louder than yours — tune
  `--barge-threshold` up and `--barge-echo-gain` toward your real echo level,
  or wear headphones. Or disable with `--no-barge`.

```bash
python agent.py --wake wake/models/billou.onnx              # barge-in on
python agent.py --wake ... --barge-threshold 0.08           # harder to trigger
python agent.py --wake ... --no-barge                       # off
```

## Known caveats

- **Barge-in on laptop mic is approximate.** Without true AEC, a speaker close
  to the mic can self-trigger or be hard to interrupt; see tuning above. Real
  WebRTC AEC would fix it but adds a dependency.
- **Kokoro default voice** is `ff_siwis` (Swiss French female). For other
  voices browse `mlx-community/Kokoro-82M-*` voice files or use
  `:voice <name>` inside `test_tts.py --engine kokoro` to try them.
- **Kyutai still around** for A/B testing via `test_tts.py --engine kyutai`.
  Quantizing it is broken in moshi_mlx 0.3.0 (matmul shape mismatch in
  cross-attention); stay on `--quantize 0` for bf16 if you want to compare.
