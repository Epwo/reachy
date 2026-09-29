"""Expressive head/antenna motions for the motion tools (emote, look, nod,
shake, dance).

Movement only — no sounds — so they're safe in every mode, including the
hybrid `--motion` mode where Reachy's media stack is disabled. Each animation
ends back at the neutral pose. They run in a background thread (see
`play_async`) so Bilou moves *while* he talks instead of before.

Axis conventions (reachy_mini.utils.create_head_pose, x forward / z up):
    +yaw   = turn left      +pitch = look down      +roll = tilt right
Antennas are [right, left] in radians; INIT ≈ upright, larger |angle| = droop.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE
from reachy_mini.utils import create_head_pose

# One motion at a time: a second tool call in the same reply waits its turn
# instead of fighting the first one for the head.
_motion_lock = threading.Lock()

ANT_UP = INIT_ANTENNAS_JOINT_POSITIONS  # [-0.17, 0.17]


def _pose(roll: float = 0, pitch: float = 0, yaw: float = 0):
    return create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)


def _go(mini, head=None, antennas=None, duration: float = 0.3) -> None:
    # body_yaw=None keeps the base where it is (the default 0.0 would snap it).
    mini.goto_target(
        head=head, antennas=antennas, duration=duration, method="minjerk", body_yaw=None
    )


def _neutral(mini, duration: float = 0.5) -> None:
    _go(mini, head=INIT_HEAD_POSE, antennas=ANT_UP, duration=duration)


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------


def nod(mini) -> None:
    for pitch in (15, -5, 12, 0):
        _go(mini, head=_pose(pitch=pitch), duration=0.25)


def shake(mini) -> None:
    for yaw in (20, -20, 15, 0):
        _go(mini, head=_pose(yaw=yaw), duration=0.25)


_LOOK = {
    "left": dict(yaw=30),
    "right": dict(yaw=-30),
    "up": dict(pitch=-20),
    "down": dict(pitch=20),
}


def look(mini, direction: str) -> None:
    target = _LOOK.get(direction)
    if target is None:
        return
    _go(mini, head=_pose(**target), duration=0.6)
    _go(mini, head=_pose(**target), duration=1.5)  # hold
    _neutral(mini, duration=0.6)


def dance(mini) -> None:
    moves = [
        (dict(roll=15, yaw=10), [-0.9, 0.2]),
        (dict(roll=-15, yaw=-10), [-0.2, 0.9]),
        (dict(roll=15, yaw=10), [-0.9, 0.2]),
        (dict(roll=-15, yaw=-10), [-0.2, 0.9]),
        (dict(pitch=-10), [-0.6, 0.6]),
    ]
    for head, ant in moves:
        _go(mini, head=_pose(**head), antennas=ant, duration=0.35)
    _neutral(mini)


def _happy(mini) -> None:
    _go(mini, head=_pose(pitch=-8), antennas=[-0.6, 0.6], duration=0.2)
    for ant in ([0.1, -0.1], [-0.6, 0.6], [0.1, -0.1]):
        _go(mini, antennas=ant, duration=0.18)
    _neutral(mini, duration=0.4)


def _sad(mini) -> None:
    _go(mini, head=_pose(pitch=15), antennas=[-1.3, 1.3], duration=0.8)
    _go(mini, head=_pose(pitch=15), antennas=[-1.3, 1.3], duration=1.2)  # hold
    _neutral(mini, duration=0.8)


def _surprised(mini) -> None:
    _go(mini, head=_pose(pitch=-12), antennas=[0.0, 0.0], duration=0.15)
    _go(mini, head=_pose(pitch=-12), antennas=[0.0, 0.0], duration=0.8)  # hold
    _neutral(mini, duration=0.5)


def _curious(mini) -> None:
    _go(mini, head=_pose(roll=15), antennas=[-0.17, 0.8], duration=0.5)
    _go(mini, head=_pose(roll=15), antennas=[-0.17, 0.8], duration=1.0)  # hold
    _neutral(mini)


def _thinking(mini) -> None:
    _go(mini, head=_pose(pitch=-12, yaw=15), antennas=[-0.5, 0.2], duration=0.6)
    _go(mini, head=_pose(pitch=-12, yaw=15), antennas=[-0.5, 0.2], duration=1.2)
    _neutral(mini)


_EMOTES: dict[str, Callable] = {
    "happy": _happy,
    "sad": _sad,
    "surprised": _surprised,
    "curious": _curious,
    "thinking": _thinking,
}

# The LM sometimes answers in French.
_ALIASES = {
    "content": "happy", "heureux": "happy", "joyeux": "happy", "joie": "happy",
    "triste": "sad",
    "surpris": "surprised", "étonné": "surprised",
    "curieux": "curious",
    "pensif": "thinking", "réfléchi": "thinking", "réflexion": "thinking",
    "gauche": "left", "droite": "right", "haut": "up", "bas": "down",
}


def _norm(arg: str) -> str:
    arg = arg.strip().strip(".!\"'«»").lower()
    return _ALIASES.get(arg, arg)


def resolve(name: str) -> Optional[Callable]:
    """Map a ToolResult.animation string ("nod", "look:left", "emote:happy")
    to a callable(mini), or None if unknown. "sleep" is handled by the wake
    capture, not here."""
    kind, _, arg = name.partition(":")
    if kind == "nod":
        return nod
    if kind == "shake":
        return shake
    if kind == "dance":
        return dance
    if kind == "look":
        direction = _norm(arg)
        return (lambda mini: look(mini, direction)) if direction in _LOOK else None
    if kind == "emote":
        return _EMOTES.get(_norm(arg))
    return None


def play_async(mini, name: str) -> bool:
    """Run animation `name` in a background thread. Returns False if the name
    is unknown (nothing played)."""
    fn = resolve(name)
    if fn is None or mini is None:
        return False

    def _run():
        with _motion_lock:
            try:
                fn(mini)
            except Exception as e:
                print(f"[anim] {name} a échoué: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return True
