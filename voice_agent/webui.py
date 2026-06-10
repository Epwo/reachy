"""Optional web UI for the voice agent.

Run agent.py with --webui to start a small FastAPI server alongside the
audio pipeline. The page at http://localhost:<port>/ shows:

  - Live mic VU meter
  - Conversation history (heard + reply per turn, with timings)
  - Indicator: listening / thinking / speaking
  - Audio input device selector
  - Type-to-speak box (pushes text directly to the TTS worker)
  - Clear-history button

State is shared with agent.py through an `AgentState` instance — the agent
publishes events into it; the UI reads them and pushes them to the browser
over a WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Shared agent state
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One conversational exchange."""
    heard: str = ""
    reply: str = ""
    lm_ms: float = 0.0
    tts_ms: float = 0.0
    audio_s: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "heard": self.heard,
            "reply": self.reply,
            "lm_ms": round(self.lm_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "audio_s": round(self.audio_s, 2),
            "ts": self.ts,
        }


class AgentState:
    """Thread-safe state shared between the agent and the webui.

    Producer-side (agent.py) calls `set_*` / `add_turn` to push updates.
    Consumer-side (webui) reads snapshots via `snapshot()` or subscribes via
    `subscribe()` to get a queue that receives change notifications.
    """

    def __init__(self, history_size: int = 50):
        self._lock = threading.Lock()
        self._mic_rms: float = 0.0
        self._state: str = "idle"               # idle | listening | thinking | speaking
        self._turns: Deque[Turn] = deque(maxlen=history_size)
        self._tts_engine: str = ""
        self._mic_device: Optional[int] = None
        self._subscribers: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Hooks that the agent registers for the webui to drive control.
        self._on_clear_history: Optional[Callable[[], None]] = None
        self._on_say: Optional[Callable[[str], None]] = None
        self._on_switch_mic: Optional[Callable[[int], None]] = None
        self._on_wake_miss: Optional[Callable[[], None]] = None

    # ---- producer side (called from agent thread) -----------------------

    def set_mic_rms(self, rms: float) -> None:
        with self._lock:
            self._mic_rms = rms
        self._notify({"type": "mic_rms", "rms": rms})

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state
        self._notify({"type": "state", "state": state})

    def set_tts_engine(self, engine: str) -> None:
        with self._lock:
            self._tts_engine = engine
        self._notify({"type": "engine", "engine": engine})

    def set_mic_device(self, device: Optional[int]) -> None:
        with self._lock:
            self._mic_device = device
        self._notify({"type": "mic_device", "device": device})

    def add_turn(self, turn: Turn) -> None:
        with self._lock:
            self._turns.append(turn)
        self._notify({"type": "turn", "turn": turn.to_dict()})

    def clear_turns(self) -> None:
        with self._lock:
            self._turns.clear()
        self._notify({"type": "history_cleared"})

    # ---- consumer side --------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mic_rms": self._mic_rms,
                "state": self._state,
                "engine": self._tts_engine,
                "mic_device": self._mic_device,
                "turns": [t.to_dict() for t in self._turns],
            }

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Tell AgentState which event loop to schedule subscriber notifies on.
        Called once by the webui after its uvicorn loop starts."""
        self._loop = loop

    def _notify(self, msg: dict) -> None:
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except RuntimeError:
                # loop closed
                pass

    # ---- control hooks (set by agent, invoked from webui handlers) -----

    def bind_handlers(
        self,
        on_clear_history: Optional[Callable[[], None]] = None,
        on_say: Optional[Callable[[str], None]] = None,
        on_switch_mic: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._on_clear_history = on_clear_history
        self._on_say = on_say
        self._on_switch_mic = on_switch_mic

    def clear_history(self) -> None:
        if self._on_clear_history:
            self._on_clear_history()

    def say(self, text: str) -> None:
        if self._on_say:
            self._on_say(text)

    def switch_mic(self, device: int) -> None:
        if self._on_switch_mic:
            self._on_switch_mic(device)

    def set_wake_miss_handler(self, fn: Callable[[], None]) -> None:
        """Bound separately because the capture object doesn't exist yet when
        bind_handlers() is first called."""
        self._on_wake_miss = fn

    def wake_miss(self) -> None:
        if self._on_wake_miss:
            self._on_wake_miss()


# ---------------------------------------------------------------------------
# Audio device enumeration
# ---------------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Return [{index, name, channels, default}] for every audio input."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = -1
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "default": i == default_idx,
                "samplerate": int(d.get("default_samplerate", 16000)),
            })
    return devices


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(state: AgentState) -> FastAPI:
    app = FastAPI(title="Reachy Voice Agent")

    @app.on_event("startup")
    async def _bind_loop():
        state.bind_loop(asyncio.get_running_loop())

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        path = HERE / "webui_index.html"
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/devices")
    async def get_devices() -> JSONResponse:
        return JSONResponse({"devices": list_input_devices()})

    @app.post("/clear")
    async def post_clear() -> JSONResponse:
        state.clear_history()
        return JSONResponse({"ok": True})

    @app.post("/say")
    async def post_say(payload: dict) -> JSONResponse:
        text = (payload or {}).get("text", "").strip()
        if not text:
            return JSONResponse({"error": "empty text"}, status_code=400)
        state.say(text)
        return JSONResponse({"ok": True})

    @app.post("/mic")
    async def post_mic(payload: dict) -> JSONResponse:
        try:
            idx = int((payload or {}).get("device"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "bad device index"}, status_code=400)
        state.switch_mic(idx)
        return JSONResponse({"ok": True})

    @app.post("/wake_miss")
    async def post_wake_miss() -> JSONResponse:
        # "I said the wake word but Reachy didn't hear it" — saves the recent
        # audio buffer as a training-positive (false negative).
        state.wake_miss()
        return JSONResponse({"ok": True})

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        q = state.subscribe()
        try:
            # Initial snapshot
            await ws.send_text(json.dumps({"type": "snapshot",
                                           "snapshot": state.snapshot()}))
            while True:
                msg = await q.get()
                await ws.send_text(json.dumps(msg))
        except WebSocketDisconnect:
            pass
        finally:
            state.unsubscribe(q)

    return app


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def start_in_background(state: AgentState, host: str = "127.0.0.1",
                        port: int = 8765) -> threading.Thread:
    """Run uvicorn on a daemon thread so the agent's main loop stays in
    charge of the process. Returns the thread (already started)."""
    import uvicorn

    app = build_app(state)
    config = uvicorn.Config(app, host=host, port=port,
                            log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)

    def _run():
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="webui")
    t.start()
    print(f"[webui] http://{host}:{port}/")
    return t
