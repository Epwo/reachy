# Reachy Mini — vision & voice experiments

Personal experiments turning a [Reachy Mini](https://huggingface.co/docs/reachy_mini)
into an interactive desktop robot: it follows faces, recognizes who it's
looking at, and holds a spoken conversation in French — all running locally
on an Apple Silicon Mac (M4).

## Two subsystems

### `face_track/` — face tracking & recognition
The head smoothly follows the largest/known face, recognizes enrolled people
(InsightFace embeddings), and can publish an auto-framed feed as a virtual
webcam.

```bash
cd face_track
python enroll.py Alice          # register a face
python main.py                  # track + recognize on the robot
python experiment.py --webcam 0 # tune tracking params live
python virtual_camera.py        # auto-framing virtual webcam
```

### `voice_agent/` — local voice assistant
Wake word → speech understanding → French reply → speech, fully on-device:

```
"Billou" → openWakeWord → Gemma 4 (audio→text) → Supertonic (text→audio) → 🔊
```

```bash
cd voice_agent
# See voice_agent/README.md for the full multi-venv setup.
python agent.py --no-robot --motion --wake wake/models/billou.onnx --webui
```

Highlights:
- **Wake word** ("Billou") trained with openWakeWord, so it only listens when called.
- **Conversation mode**: stays awake for follow-ups without re-saying the wake word.
- **Robot animations**: wakes up / goes to sleep with head + antenna motion.
- **Web UI** (`--webui`): live mic meter, transcripts, device picker, type-to-speak.
- Pluggable TTS (`--tts supertonic|kokoro|kyutai`), each in its own venv.

Full docs: [`voice_agent/README.md`](voice_agent/README.md).
Model exploration history & rationale: [`voice_agent/MODELES_TESTES.md`](voice_agent/MODELES_TESTES.md).

## Hardware / constraints

- Mac mini **M4 base, 12 GB** unified memory
- Everything local (no cloud) — model choices are driven by what fits & runs
  in real time on this hardware. `MODELES_TESTES.md` records what was tried
  and why each option was kept or dropped.

## Layout

```
reachy/
├── face_track/      face tracking + recognition + virtual camera
├── voice_agent/     wake word + speech LM + TTS pipeline
│   ├── lm/          audio→text (Gemma 4, mlx-vlm)
│   ├── tts/         text→audio (Supertonic / Kokoro / Kyutai)
│   ├── wake/        wake-word model + trainer notes
│   └── *.py         orchestrator, audio I/O, web UI
└── Readme.md
```

Virtualenvs (`.venv_*`) and model weights are gitignored; see each subsystem's
README for how to recreate them.
