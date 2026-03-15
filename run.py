"""
run.py — CLI entry point for real-time face recognition
---------------------------------------------------------
Usage:
    python run.py                          # webcam
    python run.py --source video.mp4       # video file
    python run.py --source image.jpg       # single image
    python run.py --register "Alice" photo.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from core.engine import FaceEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("facevision.run")


class FPSCounter:
    def __init__(self, window: int = 30):
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


class FrameBuffer:
    """
    Accumulate frames and flush every N frames (batch_size).
    On CPU this gives no speed benefit; on GPU it enables
    batched inference — just swap engine.process_frame with
    engine.process_batch and the rest is unchanged.
    """
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self._buf: list = []

    def push(self, frame) -> list | None:
        self._buf.append(frame)
        if len(self._buf) >= self.batch_size:
            batch = list(self._buf)
            self._buf.clear()
            return batch
        return None


def draw_hud(frame, detections, fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    known   = sum(1 for d in detections if d.identity != "unknown")
    unknown = len(detections) - known
    hud = (f"FaceVision  |  Faces: {len(detections)}"
           f"  Known: {known}  Unknown: {unknown}"
           f"  FPS: {fps:.1f}")
    cv2.putText(frame, hud, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 235), 1, cv2.LINE_AA)
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 28), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame,
                "Q / ESC — quit   S — snapshot   L — landmarks   + / - — threshold",
                (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 110), 1, cv2.LINE_AA)


def run_video(engine: FaceEngine, source, batch_size: int = 1):
    is_image = isinstance(source, str) and \
               Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"[ERROR] Cannot open image: {source}")
        annotated, detections = engine.process_frame(frame)
        draw_hud(annotated, detections, fps=0.0)
        print(f"Detections: {[(d.identity, f'{d.similarity:.2f}') for d in detections]}")
        cv2.imshow("FaceVision", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter  = FPSCounter()
    frame_buffer = FrameBuffer(batch_size)
    last_dets    = []
    last_frame   = None
    process_every = max(1, batch_size)
    frame_n = 0

    print("\n── Running — controls shown in window ───────────────\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        if frame_n % process_every == 0:
            elapsed_ms, last_dets = engine.process_frame(frame)
            last_frame = engine._draw(frame.copy(), last_dets)
        else:
            last_frame = engine._draw(frame.copy(), last_dets)

        fps = fps_counter.tick()

        draw_hud(last_frame, last_dets, fps)
        cv2.imshow("FaceVision", last_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            snap = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap, last_frame)
            print(f"[SNAP] Saved {snap}")
        elif key == ord('l'):
            print("[INFO] Landmark toggle — restart with --landmarks flag")
        elif key == ord('+'):
            engine.threshold = min(1.0, engine.threshold + 0.02)
            print(f"[INFO] Threshold → {engine.threshold:.2f}")
        elif key == ord('-'):
            engine.threshold = max(0.1, engine.threshold - 0.02)
            print(f"[INFO] Threshold → {engine.threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()

    s = engine.stats
    print(f"\n── Session stats")
    print(f"  Frames processed : {s['frames']}")
    print(f"  Total detections : {s['detections']}")
    print(f"  Matches          : {s['matches']}")
    print(f"  Avg inference ms : {s['avg_ms']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="FaceVision — Real-Time Multi-Face Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              # webcam
  python run.py --source video.mp4           # video
  python run.py --source photo.jpg           # image
  python run.py --register "Alice" alice.jpg # register person
  python run.py --backend insightface        # use ArcFace
        """
    )
    parser.add_argument("--source",     default="0",
                        help="0=webcam, or path to video/image")
    parser.add_argument("--register",   nargs=2, metavar=("NAME", "IMAGE"),
                        help="Register a person: --register 'Alice' photo.jpg")
    parser.add_argument("--backend",    default="deepface",
                        choices=["deepface", "insightface"])
    parser.add_argument("--threshold",  type=float, default=0.55,
                        help="Cosine similarity threshold (0–1)")
    parser.add_argument("--db",         default="data/face_db.pkl",
                        help="Path to face database file")
    parser.add_argument("--batch",      type=int, default=1,
                        help="Batch size (>1 speeds up GPU inference)")
    parser.add_argument("--skip",       type=int, default=1,
                        help="Run detection every N frames (1=every frame)")
    parser.add_argument("--landmarks",  action="store_true")
    args = parser.parse_args()

    engine = FaceEngine(
        backend=args.backend,
        threshold=args.threshold,
        db_path=args.db,
    )
    if args.register:
        name, img_path = args.register
        import cv2 as _cv2
        img = _cv2.imread(img_path)
        if img is None:
            sys.exit(f"[ERROR] Cannot read image: {img_path}")
        ok = engine.register(name, img)
        print(f"{'[OK]' if ok else '[FAIL]'} Registered '{name}'")
        return

    source = int(args.source) if args.source.isdigit() else args.source
    run_video(engine, source, batch_size=args.skip)


if __name__ == "__main__":
    main()