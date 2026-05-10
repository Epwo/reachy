"""Experiment harness for the face-tracking framework.

Use this to test the tracker on static images and short video clips, and to
tune parameters live with sliders. The robot can be either:

  - Real:  --robot         (connects, wakes up, follows the input)
  - None:  --no-robot      (detection + visualization only, no commands sent)

Inputs:
  python experiment.py --image  path/to/face.jpg
  python experiment.py --video  path/to/clip.mp4
  python experiment.py --webcam 0

Press Q to quit. Use the trackbars to tune parameters in real time.
"""

import argparse
import time

import cv2
import numpy as np

from src.face_tracker import FaceTracker, TrackingParams, TrackingState


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

def iter_image(path: str, fps: int = 25):
    """Yield the same image forever at a fixed FPS."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    interval = 1.0 / fps
    while True:
        yield img.copy()
        time.sleep(interval)


def iter_video(path: str, loop: bool = True):
    """Yield frames from a video file, optionally looping forever."""
    while True:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = 1.0 / fps
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            time.sleep(interval)
        cap.release()
        if not loop:
            return


def iter_webcam(index: int = 0):
    """Yield frames from a local webcam."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {index}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def fit_to_camera(frame: np.ndarray, mini) -> np.ndarray:
    """Resize an arbitrary frame to the resolution Reachy expects.

    The robot's `look_at_image` validates pixel coords against the camera
    resolution, so feeding a foreign image needs a resize first.
    """
    if mini is None or mini.media.camera is None:
        return frame
    target_w, target_h = mini.media.camera.resolution
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h))


# ---------------------------------------------------------------------------
# Stub robot (for --no-robot mode)
# ---------------------------------------------------------------------------

class _StubCamera:
    def __init__(self, w=640, h=480):
        self.resolution = (w, h)


class _StubMedia:
    def __init__(self):
        self.camera = _StubCamera()

    def get_frame(self):
        return None


class StubMini:
    """A no-op stand-in for ReachyMini. `look_at_image` returns identity poses,
    `set_target_*` calls are recorded but not sent anywhere."""

    def __init__(self):
        self.media = _StubMedia()
        self.head_calls: list = []
        self.antenna_calls: list = []

    def look_at_image(self, u, v, duration=1.0, perform_movement=True):
        # Fake: synthesize a small rotation proportional to (u - cx, v - cy).
        # Good enough to keep the pose-space EMA exercising real numbers.
        cx, cy = self.media.camera.resolution[0] / 2, self.media.camera.resolution[1] / 2
        yaw = (u - cx) / cx * 0.3   # ±0.3 rad mapped across width
        pitch = (v - cy) / cy * 0.2
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        cp_, sp_ = np.cos(pitch), np.sin(pitch)
        pose = np.eye(4)
        pose[:3, :3] = np.array([
            [cy_,        0.0,  sy_],
            [sp_*sy_,    cp_,  -sp_*cy_],
            [-cp_*sy_,   sp_,  cp_*cy_],
        ])
        return pose

    def get_current_head_pose(self):
        # Pretend we instantly reach whatever was last set.
        if self.head_calls:
            last = self.head_calls[-1]
            if isinstance(last, np.ndarray):
                return last.copy()
        return np.eye(4)

    def set_target_head_pose(self, pose):
        self.head_calls.append(pose)

    def set_target_antenna_joint_positions(self, pos):
        self.antenna_calls.append(list(pos))

    def goto_target(self, head=None, antennas=None, duration=1.0, method="minjerk", **kw):
        # Stub: just record the request, don't actually wait.
        self.head_calls.append(("goto", head))

    def wake_up(self): pass
    def goto_sleep(self): pass

    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Tunable trackbar window
# ---------------------------------------------------------------------------

WINDOW = "Face Tracker Experiment"


# Each entry: (label, attr, scale, max_int, min_int)
# `attr` is the TrackingParams field. Slider value / scale = field value.
_TRACKBARS = [
    # label,                   attr,                  scale, max, min
    ("dead_zone %",            "dead_zone",            100,  60,  0),
    ("recenter_zone %",        "recenter_zone",        100,  40,  0),
    ("tracking_speed /1000",   "tracking_speed",       1000, 500, 1),
    ("max_step_deg /10",       "max_step_deg",         10,   100, 1),
    ("detection_smooth /100",  "detection_smoothing",  100,  100, 1),
    ("flip_u",                 "flip_u",               1,    1,   0),
]
# fps_limit is not in TrackingParams (it's a loop knob), handled separately.


def _setup_trackbars(params: TrackingParams, fps_limit_init: int):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for label, attr, scale, mx, _mn in _TRACKBARS:
        val = getattr(params, attr)
        if isinstance(val, bool):
            init = int(val)
        else:
            init = max(0, min(mx, int(round(val * scale))))
        cv2.createTrackbar(label, WINDOW, init, mx, lambda v: None)
    cv2.createTrackbar("fps_limit", WINDOW, fps_limit_init, 60, lambda v: None)


def _read_trackbars(params: TrackingParams) -> int:
    """Push every slider value into `params`. Returns the current fps_limit."""
    for label, attr, scale, _mx, mn in _TRACKBARS:
        raw = cv2.getTrackbarPos(label, WINDOW)
        clamped_int = max(mn, raw)
        if attr == "flip_u":
            setattr(params, attr, bool(clamped_int))
        else:
            setattr(params, attr, clamped_int / scale)
    return max(1, cv2.getTrackbarPos("fps_limit", WINDOW))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_experiment(frame_iter, mini, params: TrackingParams,
                   send_commands: bool, fps_limit: int = 25,
                   model_name: str = "buffalo_l", det_size: int = 640):
    tracker = FaceTracker(det_size=(det_size, det_size), model_name=model_name)
    state = TrackingState()
    _setup_trackbars(params, fps_limit)

    print("Experiment running. Q=quit, R=reset position, S=save params.")
    save_flash_until = 0.0
    try:
        for frame in frame_iter:
            t0 = time.monotonic()
            fps_limit = _read_trackbars(params)
            frame = fit_to_camera(frame, mini)

            res = tracker.step(frame, mini, state, params, send_commands=send_commands)

            # HUD with all current values
            hud_lines = [
                f"dz={params.dead_zone:.2f}  rc={params.recenter_zone:.2f}  speed={params.tracking_speed:.3f}",
                f"step={params.max_step_deg:.1f}deg  det_smooth={params.detection_smoothing:.2f}  flip_u={int(params.flip_u)}",
                f"fps={fps_limit}   Q=quit  R=reset  S=save",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(res.display, line,
                            (10, res.display.shape[0] - 12 - 22 * (len(hud_lines) - 1 - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # "Saved!" flash
            if t0 < save_flash_until:
                cv2.putText(res.display, "SAVED", (res.display.shape[1] - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW, res.display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                if send_commands:
                    tracker.reset_position(mini, state)
                else:
                    state.smooth_u = state.smooth_v = None
                    state.streamed_pose = None
                    state.is_tracking = False
                    print("Reset (stub): cleared state.")
            elif key == ord("s"):
                path = params.save()
                save_flash_until = t0 + 1.5
                print(f"Saved tracking params → {path}")

            # Honour fps_limit from the slider
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, 1.0 / fps_limit - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a static image (loops at 25 fps)")
    src.add_argument("--video", help="Path to a video file (loops by default)")
    src.add_argument("--webcam", type=int, nargs="?", const=0,
                     help="Use a local webcam (default index 0)")

    parser.add_argument("--no-loop", action="store_true",
                        help="Don't loop the video — exit when it ends")
    parser.add_argument("--no-robot", action="store_true",
                        help="Don't connect to Reachy — detection only")

    # Initial parameter values (sliders override them live)
    parser.add_argument("--load", action="store_true",
                        help="Start from the saved tracking_params.json instead of CLI defaults")
    parser.add_argument("--dead-zone", type=float, default=0.20)
    parser.add_argument("--recenter-zone", type=float, default=0.05)
    parser.add_argument("--tracking-speed", type=float, default=0.10)
    parser.add_argument("--max-step-deg", type=float, default=1.5)
    parser.add_argument("--detection-smoothing", type=float, default=0.4)
    parser.add_argument("--flip-u", action="store_true")
    parser.add_argument("--fps-limit", type=int, default=25)
    parser.add_argument("--model", default="buffalo_l",
                        choices=["buffalo_sc", "buffalo_s", "buffalo_l"])
    parser.add_argument("--det-size", type=int, default=640)

    args = parser.parse_args()

    if args.load:
        params = TrackingParams.load()
        print(f"Loaded saved params from tracking_params.json")
    else:
        params = TrackingParams(
            dead_zone=args.dead_zone,
            recenter_zone=args.recenter_zone,
            tracking_speed=args.tracking_speed,
            max_step_deg=args.max_step_deg,
            detection_smoothing=args.detection_smoothing,
            flip_u=args.flip_u,
        )

    if args.image:
        frame_iter = iter_image(args.image)
    elif args.video:
        frame_iter = iter_video(args.video, loop=not args.no_loop)
    else:
        frame_iter = iter_webcam(args.webcam)

    kwargs = dict(fps_limit=args.fps_limit, model_name=args.model,
                  det_size=args.det_size)
    if args.no_robot:
        mini = StubMini()
        run_experiment(frame_iter, mini, params, send_commands=False, **kwargs)
    else:
        from reachy_mini import ReachyMini
        with ReachyMini() as mini:
            mini.wake_up()
            try:
                run_experiment(frame_iter, mini, params, send_commands=True, **kwargs)
            finally:
                mini.goto_sleep()


if __name__ == "__main__":
    main()
