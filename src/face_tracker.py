"""
Face tracking and recognition for Reachy Mini.

Usage:
    tracker = FaceTracker()
    tracker.enroll("Alice", frame)          # register a person
    result = tracker.process(frame)         # detect + recognize
    mini.look_at_image(result.u, result.v)  # track with the robot

Known faces are persisted to ./known_faces/ as .npy embedding files.
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    SLEEP_ANTENNAS_JOINT_POSITIONS,
)


KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
PARAMS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "tracking_params.json")
RECOGNITION_THRESHOLD = 0.5  # cosine distance; lower = stricter


@dataclass
class DetectedFace:
    name: str               # identity label, or "unknown"
    confidence: float       # recognition score (1.0 = perfect match)
    u: int                  # horizontal pixel center
    v: int                  # vertical pixel center
    bbox: tuple             # (x1, y1, x2, y2)
    embedding: np.ndarray = field(repr=False)


@dataclass
class TrackingParams:
    """All tunable knobs for the tracking pipeline. Pass one to `step()`."""
    dead_zone: float = 0.20          # fraction of frame to START moving
    recenter_zone: float = 0.05      # fraction of frame to STOP moving (must be < dead_zone)
    tracking_speed: float = 0.10     # EMA alpha on streamed head pose
    max_step_deg: float = 1.5        # max rotation per frame (degrees)
    detection_smoothing: float = 0.4 # light EMA on raw detection (jitter)
    flip_u: bool = False             # mirror horizontal axis

    def save(self, path: str = PARAMS_FILE) -> str:
        """Persist these parameters to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = PARAMS_FILE) -> "TrackingParams":
        """Load parameters from a JSON file. Returns defaults if missing."""
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        valid = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class TrackingState:
    """Mutable per-frame state. Pass one in, get an updated one back."""
    smooth_u: Optional[float] = None             # EMA-smoothed face position
    smooth_v: Optional[float] = None
    streamed_pose: Optional[np.ndarray] = None   # Last 4x4 pose streamed to motor
    is_tracking: bool = False                    # hysteresis flag (see step())
    last_seen_t: float = field(default_factory=time.monotonic)
    # Multi-face selection: stickiness so we don't ping-pong between faces.
    locked_name: Optional[str] = None            # name of currently-followed face
    locked_bbox: Optional[tuple] = None          # last bbox (used for unknowns)


@dataclass
class StepResult:
    """Everything the loop needs after one `step()`."""
    face: Optional[DetectedFace]
    display: np.ndarray
    target_pose: Optional[np.ndarray] = None  # 4x4, what was streamed (if any)


class FaceTracker:
    """Detect, recognize, and track faces using InsightFace + Reachy Mini."""

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        model_name: str = "buffalo_l",
        load_extras: bool = False,
    ):
        """Create the face tracker.

        Args:
            det_size: Detection input resolution. Smaller = less memory + CPU.
                      (640, 640) is the default and detects small/far faces.
                      Drop to (320, 320) if you only need close-range tracking.
            model_name: InsightFace model pack:
                - "buffalo_sc": ~13 MB, detection only (NO recognition).
                - "buffalo_s":  ~25 MB, detection + recognition. Lightweight.
                - "buffalo_l": ~280 MB, best accuracy. Default.
            load_extras: If False (default), skip the landmark and gender/age
                         models we don't need. Roughly halves RAM use without
                         changing tracking or recognition behavior.
        """
        kwargs = dict(name=model_name, providers=["CPUExecutionProvider"])
        if not load_extras:
            kwargs["allowed_modules"] = ["detection", "recognition"]
        self._app = FaceAnalysis(**kwargs)
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._known: dict[str, np.ndarray] = {}  # name → mean embedding
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        self._load_known_faces()

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, name: str, frame: np.ndarray, n_samples: int = 1) -> bool:
        """Register a face from a single BGR frame.

        If called multiple times for the same name, embeddings are averaged
        so that multiple angles improve recognition accuracy.

        Returns True if a face was successfully enrolled.
        """
        faces = self._app.get(frame)
        if not faces:
            return False

        # Use the largest detected face
        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        new_emb = _normalize(face.embedding)

        if name in self._known:
            # Average with existing embedding (equal weight per call)
            self._known[name] = _normalize(self._known[name] + new_emb)
        else:
            self._known[name] = new_emb

        np.save(os.path.join(KNOWN_FACES_DIR, f"{name}.npy"), self._known[name])
        return True

    def enroll_from_webcam(self, name: str, n_samples: int = 30, camera_index: int = 0) -> bool:
        """Interactively capture frames from a local webcam to enroll a face.

        Press SPACE to capture a sample, Q to finish early.
        Returns True if at least one sample was captured.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {camera_index}")

        captured = 0
        print(f"Enrolling '{name}': press SPACE to capture ({n_samples} needed), Q to finish.")
        try:
            while captured < n_samples:
                ret, frame = cap.read()
                if not ret:
                    break

                preview = frame.copy()
                faces = self._app.get(preview)
                for f in faces:
                    x1, y1, x2, y2 = map(int, f.bbox)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"Samples: {captured}/{n_samples}  [SPACE=capture  Q=done]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Enroll", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if self.enroll(name, frame):
                        captured += 1
                        print(f"  captured {captured}/{n_samples}")
                    else:
                        print("  no face detected, try again")
                elif key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return captured > 0

    def forget(self, name: str) -> bool:
        """Remove a registered person."""
        path = os.path.join(KNOWN_FACES_DIR, f"{name}.npy")
        removed = self._known.pop(name, None) is not None
        if os.path.exists(path):
            os.remove(path)
        return removed

    @property
    def known_names(self) -> list[str]:
        return list(self._known.keys())

    # ------------------------------------------------------------------
    # Detection + recognition
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Detect and recognize the largest face in a BGR frame.

        Returns a DetectedFace or None if no face is found.
        """
        faces = self._app.get(frame)
        if not faces:
            return None

        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        x1, y1, x2, y2 = map(int, face.bbox)
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2
        emb = _normalize(face.embedding)

        name, confidence = self._match(emb)
        return DetectedFace(
            name=name,
            confidence=confidence,
            u=u,
            v=v,
            bbox=(x1, y1, x2, y2),
            embedding=emb,
        )

    def process_all(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect and recognize every face in a BGR frame."""
        results = []
        for face in self._app.get(frame):
            x1, y1, x2, y2 = map(int, face.bbox)
            emb = _normalize(face.embedding)
            name, confidence = self._match(emb)
            results.append(DetectedFace(
                name=name,
                confidence=confidence,
                u=(x1 + x2) // 2,
                v=(y1 + y2) // 2,
                bbox=(x1, y1, x2, y2),
                embedding=emb,
            ))
        return results

    def _select_face(self, faces: list[DetectedFace],
                     state: TrackingState) -> Optional[DetectedFace]:
        """Pick which face to follow, with stickiness.

        Priority:
          1. The face we're already locked onto, if it's still visible
             (matched by name for known faces, by bbox proximity for unknowns).
          2. Any KNOWN (recognized) face — break ties by largest bbox.
          3. The face whose bbox is closest to the last locked bbox.
          4. Just the largest face in the frame.

        This keeps the head from oscillating between two equally-valid faces.
        """
        if not faces:
            return None

        # 1. Try to keep the lock
        if state.locked_name is not None:
            if state.locked_name != "unknown":
                # Known face: match by name
                same = [f for f in faces if f.name == state.locked_name]
                if same:
                    return max(same, key=lambda f: _bbox_area(f.bbox))
            elif state.locked_bbox is not None:
                # Unknown face: re-identify by bbox proximity
                lcx = (state.locked_bbox[0] + state.locked_bbox[2]) / 2
                lcy = (state.locked_bbox[1] + state.locked_bbox[3]) / 2
                near = [
                    (((f.bbox[0]+f.bbox[2])/2 - lcx)**2
                     + ((f.bbox[1]+f.bbox[3])/2 - lcy)**2,
                     f) for f in faces
                ]
                near.sort(key=lambda t: t[0])
                # Only re-acquire if reasonably close (within ~120 px center distance)
                if near[0][0] < 120 * 120:
                    return near[0][1]

        # 2. Prefer a known face (by largest bbox among knowns)
        knowns = [f for f in faces if f.name != "unknown"]
        if knowns:
            return max(knowns, key=lambda f: _bbox_area(f.bbox))

        # 3. No lock candidate matched and no known face: pick largest unknown
        return max(faces, key=lambda f: _bbox_area(f.bbox))

    # ------------------------------------------------------------------
    # Reachy Mini integration
    # ------------------------------------------------------------------

    def step(
        self,
        frame: np.ndarray,
        mini,
        state: TrackingState,
        params: TrackingParams,
        send_commands: bool = True,
    ) -> StepResult:
        """One iteration of the tracking pipeline.

        Pure-ish: takes a frame + state + params, mutates `state` in place,
        and (if send_commands) sends head/antenna commands to `mini`. Returns
        a StepResult with the detected face (if any), the rendered display,
        and the target pose that was streamed.

        Use this directly to drive the robot from any frame source — live
        camera, image, video, anything — without going through the full
        `run_tracking_loop`.
        """
        h, w = frame.shape[:2]
        dz_hw = w * params.dead_zone / 2
        dz_hh = h * params.dead_zone / 2
        rc_hw = w * params.recenter_zone / 2
        rc_hh = h * params.recenter_zone / 2
        cx, cy = w / 2, h / 2

        all_faces = self.process_all(frame)
        result = self._select_face(all_faces, state)
        target_pose = None
        now = time.monotonic()

        if result is not None:
            state.last_seen_t = now
            state.locked_name = result.name
            state.locked_bbox = result.bbox

            raw_u = float(w - result.u) if params.flip_u else float(result.u)
            raw_v = float(result.v)

            # Light EMA on raw detection (jitter suppression only)
            if state.smooth_u is None:
                state.smooth_u, state.smooth_v = raw_u, raw_v
            else:
                state.smooth_u += params.detection_smoothing * (raw_u - state.smooth_u)
                state.smooth_v += params.detection_smoothing * (raw_v - state.smooth_v)

            # Hysteresis: face must EXIT the dead zone to start tracking, and
            # ENTER the (smaller) recenter zone to stop. Without this, the head
            # halts the moment the face crosses the dead-zone boundary, leaving
            # it parked at the edge instead of near the center.
            offset_u = abs(state.smooth_u - cx)
            offset_v = abs(state.smooth_v - cy)
            if not state.is_tracking and (offset_u > dz_hw or offset_v > dz_hh):
                state.is_tracking = True
            elif state.is_tracking and offset_u < rc_hw and offset_v < rc_hh:
                state.is_tracking = False

            if state.is_tracking:
                # `look_at_image` returns the absolute world pose to look at
                # the 3D point that pixel (smooth_u, smooth_v) maps to from
                # the CURRENT head pose. As the head rotates, the face moves
                # toward image center, so the target pose stays close to a
                # fixed 3D anchor (the face) and convergence is clean.
                face_target = mini.look_at_image(
                    int(np.clip(state.smooth_u, 1, w - 1)),
                    int(np.clip(state.smooth_v, 1, h - 1)),
                    perform_movement=False,
                )

                # Init streamed pose at the actual current head pose so the
                # first emit doesn't jump.
                if state.streamed_pose is None:
                    state.streamed_pose = mini.get_current_head_pose()

                # Pose-space EMA toward face target, with angular cap per frame.
                max_step_rad = np.deg2rad(params.max_step_deg)
                state.streamed_pose = _lerp_pose_capped(
                    state.streamed_pose, face_target,
                    params.tracking_speed, max_step_rad,
                )
                target_pose = state.streamed_pose
                if send_commands:
                    mini.set_target_head_pose(target_pose)
        else:
            # Face lost: hold streamed_pose where it is, stop tracking.
            state.smooth_u = state.smooth_v = None
            state.is_tracking = False

        display = self.draw(
            frame,
            all_faces,
            locked=result,
            dead_zone_box=(
                int(cx - dz_hw), int(cy - dz_hh),
                int(cx + dz_hw), int(cy + dz_hh),
            ),
            recenter_box=(
                int(cx - rc_hw), int(cy - rc_hh),
                int(cx + rc_hw), int(cy + rc_hh),
            ),
            is_tracking=state.is_tracking,
            smooth_pos=(
                (int(state.smooth_u), int(state.smooth_v))
                if state.smooth_u is not None else None
            ),
            target_pos=None,
        )

        return StepResult(face=result, display=display, target_pose=target_pose)

    def reset_position(self, mini, state: TrackingState) -> None:
        """Send the head back to the forward-facing rest pose and clear state.

        Triggered by pressing R in the tracking window. Use this when the
        head has wandered or you want a clean starting point for tuning.
        """
        # INIT_HEAD_POSE in the SDK is just the identity matrix (forward-facing).
        init_pose = np.eye(4)
        mini.goto_target(init_pose, antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                         duration=0.8, method="minjerk")
        state.smooth_u = state.smooth_v = None
        state.streamed_pose = None
        state.is_tracking = False
        state.locked_name = None
        state.locked_bbox = None
        state.last_seen_t = time.monotonic()
        print("Reset to home position.")

    def run_tracking_loop(
        self,
        mini,
        on_recognized=None,
        fps_limit: int = 25,
        params: Optional[TrackingParams] = None,
        visualize: bool = True,
        window_name: str = "Reachy Face Tracker",
        # Convenience overrides (override params fields if given)
        dead_zone: Optional[float] = None,
        tracking_speed: Optional[float] = None,
        max_step_deg: Optional[float] = None,
        flip_u: Optional[bool] = None,
    ):
        """Continuously track and recognize faces from Reachy Mini's camera.

        Internally calls `step()` once per frame. For experimentation with
        videos or static images, call `step()` directly.

        Press Ctrl-C or Q (in the window) to stop.
        """
        params = params or TrackingParams()
        if dead_zone is not None: params.dead_zone = dead_zone
        if tracking_speed is not None: params.tracking_speed = tracking_speed
        if max_step_deg is not None: params.max_step_deg = max_step_deg
        if flip_u is not None: params.flip_u = flip_u

        interval = 1.0 / fps_limit
        state = TrackingState()

        print("Face tracking started. Press Ctrl-C to stop. Press R to reset position.")
        try:
            while True:
                t0 = time.monotonic()
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(interval)
                    continue

                result = self.step(frame, mini, state, params)

                if result.face is not None and on_recognized is not None:
                    on_recognized(result.face)

                if visualize:
                    cv2.imshow(window_name, result.display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        self.reset_position(mini, state)

                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            if visualize:
                cv2.destroyAllWindows()
            print("Returning to sleep position...")
            mini.goto_sleep()

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        faces: list[DetectedFace],
        locked: Optional[DetectedFace] = None,
        dead_zone_box: Optional[tuple] = None,
        recenter_box: Optional[tuple] = None,
        is_tracking: bool = False,
        smooth_pos: Optional[tuple] = None,
        target_pos: Optional[tuple] = None,
    ) -> np.ndarray:
        """Draw detections, dead/recenter zones, smoothed face, and head target.

        `locked` is the face the tracker has chosen to follow; it gets a thicker
        outline and a "★" prefix in the label.
        """
        out = frame.copy()

        # Dead zone box (outer): yellow when idle, orange when tracking
        if dead_zone_box is not None:
            x1, y1, x2, y2 = dead_zone_box
            dz_color = (0, 140, 255) if is_tracking else (0, 220, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), dz_color, 1)
            cv2.putText(out, "dead zone", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dz_color, 1)

        # Recenter zone box (inner): green target where the face should land
        if recenter_box is not None:
            x1, y1, x2, y2 = recenter_box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 1)
            cv2.putText(out, "recenter", (x1 + 4, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

        # Face bounding boxes + labels. Locked face gets thicker outline + star.
        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox
            is_locked = (locked is not None and face.bbox == locked.bbox)
            color = (0, 200, 0) if face.name != "unknown" else (0, 60, 220)
            thickness = 3 if is_locked else 1
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), color, thickness)
            star = "* " if is_locked else "  "
            label = f"{star}{face.name} ({face.confidence:.2f})"
            cv2.putText(out, label, (fx1, fy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 if is_locked else 0.5,
                        color, 2 if is_locked else 1)

        # Smoothed face position (yellow cross)
        if smooth_pos is not None:
            cv2.drawMarker(out, smooth_pos, (0, 220, 220),
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # Streamed head target (cyan circle) + line to face
        if target_pos is not None:
            cv2.circle(out, target_pos, 8, (255, 200, 0), 2, cv2.LINE_AA)
            if smooth_pos is not None:
                cv2.line(out, target_pos, smooth_pos, (255, 200, 0), 1, cv2.LINE_AA)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_known_faces(self):
        for fname in os.listdir(KNOWN_FACES_DIR):
            if fname.endswith(".npy"):
                name = fname[:-4]
                self._known[name] = np.load(os.path.join(KNOWN_FACES_DIR, fname))
        if self._known:
            print(f"Loaded {len(self._known)} known face(s): {', '.join(self._known)}")

    def _match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return (name, confidence) for the closest known face."""
        if not self._known:
            return "unknown", 0.0

        best_name, best_score = "unknown", 0.0
        for name, known_emb in self._known.items():
            score = float(np.dot(embedding, known_emb))  # cosine similarity (both normalized)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < (1.0 - RECOGNITION_THRESHOLD):
            return "unknown", best_score
        return best_name, best_score


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _angular_distance(R0: np.ndarray, R1: np.ndarray) -> float:
    """Angle in radians between two 3x3 rotation matrices."""
    R_rel = R1 @ R0.T
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _lerp_pose(P0: np.ndarray, P1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two 4x4 SE(3) poses.

    Translation is lerped; rotation is lerped element-wise then projected back
    to SO(3) via SVD. For small alpha this matches slerp closely and is much
    cheaper than scipy.Rotation.Slerp.
    """
    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * P0[:3, 3] + alpha * P1[:3, 3]
    M = (1.0 - alpha) * P0[:3, :3] + alpha * P1[:3, :3]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = U @ Vt
    out[:3, :3] = R
    return out


def _lerp_pose_capped(P0: np.ndarray, P1: np.ndarray, alpha: float,
                      max_angle_rad: float) -> np.ndarray:
    """Like _lerp_pose, but additionally clamps the rotation magnitude per
    call so a far-away target never causes a fast jerk."""
    full_angle = _angular_distance(P0[:3, :3], P1[:3, :3])
    eff_alpha = alpha
    if full_angle > 1e-9 and full_angle * alpha > max_angle_rad:
        eff_alpha = max_angle_rad / full_angle
    return _lerp_pose(P0, P1, eff_alpha)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
