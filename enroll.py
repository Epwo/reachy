"""Enroll a face using the Reachy Mini camera.

Press SPACE to capture a sample, Q to finish.
Move/turn your head slightly between samples for better recognition.

Usage:
    python enroll.py Alice
    python enroll.py Alice --samples 30
"""

import argparse
import cv2

from reachy_mini import ReachyMini
from src.face_tracker import FaceTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name to register the face under")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of samples to capture (default: 20)")
    parser.add_argument("--model", default="buffalo_l",
                        choices=["buffalo_sc", "buffalo_s", "buffalo_l"])
    parser.add_argument("--det-size", type=int, default=640)
    args = parser.parse_args()

    tracker = FaceTracker(det_size=(args.det_size, args.det_size),
                          model_name=args.model)
    print(f"Already known: {tracker.known_names}")

    with ReachyMini() as mini:
        mini.wake_up()
        print(f"\nEnrolling '{args.name}': SPACE = capture, Q = done")
        captured = 0

        try:
            while captured < args.samples:
                frame = mini.media.get_frame()
                if frame is None:
                    continue

                preview = frame.copy()
                faces = tracker._app.get(preview)
                for f in faces:
                    x1, y1, x2, y2 = map(int, f.bbox)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"{args.name}: {captured}/{args.samples}  [SPACE=capture  Q=done]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("Reachy Enrollment", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if tracker.enroll(args.name, frame):
                        captured += 1
                        print(f"  captured {captured}/{args.samples}")
                    else:
                        print("  no face detected, try again")
                elif key == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            mini.goto_sleep()

        if captured > 0:
            print(f"\n✓ Enrolled '{args.name}' with {captured} samples.")
            print(f"   Known faces: {tracker.known_names}")
        else:
            print(f"\n✗ No samples captured for '{args.name}'.")


if __name__ == "__main__":
    main()
