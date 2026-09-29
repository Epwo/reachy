# Reachy Mini — vision & voice

Turning a [Reachy Mini](https://huggingface.co/docs/reachy_mini) into an
interactive desktop robot: it follows faces, recognizes who it's looking at,
and holds a spoken conversation in French — all running locally on a Mac mini
M4 (12 GB), no cloud.

## `face_track/` — follows you with its head

The head smoothly tracks the most relevant face and recognizes enrolled people
(InsightFace embeddings).

```bash
cd face_track
python enroll.py Ewann          # register a face
python main.py                  # follow + recognize on the robot
python experiment.py --webcam 0 # tune tracking live
```

Details: [`face_track/readme.md`](face_track/readme.md).

## `voice_agent/` — "Billou", the voice assistant

```
"Billou" → openWakeWord → Whisper or Parakeet → Qwen3.5-4B (VLM) → Supertonic 3 → 🔊
```

```bash
cd voice_agent
./launch.sh
```

- **Wake word** "Billou" (openWakeWord + a verifier trained on my voice)
- **Conversation mode** — follow-ups without repeating the wake word
- **Barge-in** — talk over him to interrupt
- **Tools** — web search, camera vision, timers, sleep, and body language
  (nod, shake, look, emotes, dance)
- **Web UI** and per-turn latency **metrics** (`stats.py`)

Details: [`voice_agent/README.md`](voice_agent/README.md) ·
wake word training: [`voice_agent/wake/README.md`](voice_agent/wake/README.md) ·
why these models: [`voice_agent/MODELES_TESTES.md`](voice_agent/MODELES_TESTES.md).

## Layout

```
reachy/
├── face_track/          face tracking + recognition   (.venv_reachy)
└── voice_agent/         voice assistant
    ├── stt/             speech → text, Whisper/Parakeet (.venv_stt)
    ├── lm/              text/image → reply, Qwen3.5   (.venv_lm)
    ├── tts/             text → speech, Supertonic     (.venv_supertonic, also runs agent.py)
    └── wake/            wake-word models + training   (.venv_wake)
```

Each venv has its own `requirements.txt` next to the code it runs (separate
venvs because the MLX builds they pin conflict). Venvs, model weights, known
faces and logs are gitignored.
