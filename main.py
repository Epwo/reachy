import argparse

from reachy_mini import ReachyMini
from src.face_tracker import FaceTracker, TrackingParams


def on_face(face):
    if face.name != "unknown":
        print(
            f"Recognized: {face.name} (score={face.confidence:.2f})  @ ({face.u}, {face.v})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="buffalo_l",
                        choices=["buffalo_sc", "buffalo_s", "buffalo_l"],
                        help="InsightFace model pack (default: buffalo_l, best quality)")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size (square). 640 is default.")
    args = parser.parse_args()

    tracker = FaceTracker(det_size=(args.det_size, args.det_size),
                          model_name=args.model)

    # Auto-load tuning saved by experiment.py (S key). Falls back to defaults
    # if tracking_params.json doesn't exist.
    params = TrackingParams.load()
    print(f"Tracking params: {params}")
    print(f"Known faces: {tracker.known_names}")

    with ReachyMini() as mini:
        mini.wake_up()
        tracker.run_tracking_loop(mini, params=params, on_recognized=on_face)


if __name__ == "__main__":
    main()
