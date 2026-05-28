"""Auto-framing virtual webcam.

Captures from a real camera (or Reachy), runs face detection, crops and
zooms around your face with smooth EMA tracking, and publishes the result
as a virtual webcam. Discord, Zoom, Meet, browsers — anything that asks
for a camera can pick this as the source.

On macOS this uses the OBS Virtual Camera driver as its sink, so:
    1. Install OBS once (https://obsproject.com).
       Open it once so it registers the virtual camera driver, then quit.
    2. pip install pyvirtualcam
    3. python virtual_camera.py
    4. In Discord / browser / app, pick "OBS Virtual Camera".

Usage:
    python virtual_camera.py                  # default webcam (index 0)
    python virtual_camera.py --camera 2       # specific webcam
    python virtual_camera.py --reachy         # use Reachy's camera instead

Tip: turn OFF Center Stage in macOS Control Center before running, otherwise
you're tracking inside an already-cropped feed.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import pyvirtualcam

from src.face_tracker import FaceTracker, _bbox_area


def _ema(prev, new, alpha):
    return new if prev is None else prev + alpha * (new - prev)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0). Ignored if --reachy.")
    parser.add_argument("--reachy", action="store_true",
                        help="Use Reachy's camera as the source.")
    parser.add_argument("--out-width", type=int, default=1280)
    parser.add_argument("--out-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target-face-frac", type=float, default=0.30,
                        help="Desired face width as a fraction of output width. "
                             "Smaller = more 'room around you', bigger = closer crop.")
    parser.add_argument("--smooth", type=float, default=0.08,
                        help="EMA factor for crop motion. Lower = smoother / slower.")
    parser.add_argument("--model", default="buffalo_l",
                        choices=["buffalo_sc", "buffalo_s", "buffalo_l"])
    args = parser.parse_args()

    tracker = FaceTracker(det_size=(640, 640), model_name=args.model)

    # --- Source ----------------------------------------------------------
    if args.reachy:
        from reachy_mini import ReachyMini
        mini_ctx = ReachyMini()
        mini = mini_ctx.__enter__()
        mini.wake_up()
        def grab():
            return mini.media.get_frame()
        def cleanup():
            mini.goto_sleep()
            mini_ctx.__exit__(None, None, None)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {args.camera}")
        def grab():
            ok, frame = cap.read()
            return frame if ok else None
        def cleanup():
            cap.release()

    # --- Smoothed framing state -----------------------------------------
    cx = cy = scale = None
    target_aspect = args.out_width / args.out_height

    # --- Run -------------------------------------------------------------
    with pyvirtualcam.Camera(args.out_width, args.out_height, args.fps) as vcam:
        print(f"Publishing to virtual camera: {vcam.device}")
        print(f"Size: {args.out_width}x{args.out_height} @ {args.fps} fps")
        print("In Discord, pick this camera. Press Ctrl-C to stop.\n")
        try:
            while True:
                frame = grab()
                if frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                faces = tracker.process_all(frame)

                if faces:
                    face = max(faces, key=lambda f: _bbox_area(f.bbox))
                    fx1, fy1, fx2, fy2 = face.bbox
                    target_cx = (fx1 + fx2) / 2
                    target_cy = (fy1 + fy2) / 2
                    face_w = fx2 - fx1
                    # Crop width so the face occupies `target_face_frac` of output
                    target_scale = face_w / args.target_face_frac
                else:
                    # No face — drift back to whole frame, centered
                    target_cx = w / 2
                    target_cy = h / 2
                    target_scale = min(w, h * target_aspect)

                cx    = _ema(cx,    target_cx,    args.smooth)
                cy    = _ema(cy,    target_cy,    args.smooth)
                scale = _ema(scale, target_scale, args.smooth)

                # Compute crop window matching the output aspect ratio
                crop_w = min(scale, w)
                crop_h = min(crop_w / target_aspect, h)
                # If crop_h hit the cap, reduce crop_w so aspect stays right
                if crop_h * target_aspect > w:
                    crop_w = h * target_aspect
                    crop_h = h
                else:
                    crop_w = crop_h * target_aspect

                x1 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
                y1 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))
                x2 = int(x1 + crop_w)
                y2 = int(y1 + crop_h)

                crop = frame[y1:y2, x1:x2]
                output = cv2.resize(crop, (args.out_width, args.out_height),
                                    interpolation=cv2.INTER_LINEAR)
                # pyvirtualcam wants RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                vcam.send(output_rgb)
                vcam.sleep_until_next_frame()
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            cleanup()


if __name__ == "__main__":
    main()
