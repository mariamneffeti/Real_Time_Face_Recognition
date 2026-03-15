"""
api.py - FastAPI Backend
Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import asyncio
import base64
import io
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.engine import FaceEngine



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("facevision.api")

DB_PATH = "data/attendance.db"
Model_DB = "data/face_db.pkl"
Backend = "deepface"  # or "insightface"
THRESHOLD = 0.55

Path("data").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)


def init_db() -> sqlite3.Connection:
    """Initialize SQLite database for attendance logging."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            similarity REAL,
            source TEXT DEFAULT 'camera'
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON attendance (date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_person ON attendance (person_id)")
    conn.commit()
    return conn



def log_attendance(conn: sqlite3.Connection, person_id: str, name: str, similarity: float, source: str = "camera") -> None:
    """Log attendance record to database (max once per 60 seconds per person)."""
    now = datetime.now() 
    recent = conn.execute("SELECT * FROM attendance WHERE person_id = ? AND timestamp > datetime('now','-60 seconds')",
                          (person_id,)).fetchone()
    if recent:
        return  # Already logged within 60 seconds
    conn.execute("INSERT INTO attendance (person_id, name, timestamp, date, similarity, source) VALUES (?, ?, ?, ?, ?, ?)",
                 (person_id, name, now.isoformat(), now.date().isoformat(), similarity, source))
    conn.commit()
    logger.info(f"Attendance logged for {name} ({person_id}) with similarity {similarity:.2f} from {source}")


engine: Optional[FaceEngine] = None
db_conn: Optional[sqlite3.Connection] = None
_cap: Optional[cv2.VideoCapture] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for FastAPI app."""
    global engine, db_conn, _cap
    logger.info("Starting FaceVision API...")
    engine = FaceEngine(backend=Backend, threshold=THRESHOLD, db_path=Model_DB) 
    db_conn = init_db()
    _cap = cv2.VideoCapture(0)
    yield
    if _cap:
        _cap.release()
    logger.info("Shutting down FaceVision API...")
app = FastAPI(title="FaceVision API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class DetectionRequest(BaseModel):
    """Request body for detection (from client)."""
    identity: str
    similarity: float
    bbox: list[int]


class DetectionResult(BaseModel):
    """Result for a single detected face."""
    identity: str
    similarity: float
    bbox: list[int]


class RecognitionRequest(BaseModel):
    """Request body for recognition."""
    faces: list[DetectionRequest]
    count: int
    processing_ms: float


class RecognitionResponse(BaseModel):
    """Response body for recognition endpoint."""
    faces: list[DetectionResult]
    count: int
    processing_ms: float


class AttendanceRecord(BaseModel):
    id : int
    person_id: str
    name: str
    timestamp: datetime
    date: date
    similarity: float
    source: str
class PersonInfo(BaseModel):
    person_id: str
    name: str
    registered_at: Optional[str] = None
    last_seen: Optional[str] = None
    total_detections: int = 0
    average_similarity : Optional[float] = None


async def decode_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img
@app.post("/register", summary="Register a new person")
async def register(
    name: str = Form(...),
    person_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """Register a new person by uploading their face image."""
    img = await decode_upload(image)
    ok = engine.register(name, img, person_id)
    if not ok:
        raise HTTPException(422, "No face detected in the provided image")
    return {
        "status": "ok",
        "name": name,
        "person_id": person_id or name.lower().replace(" ", "_"),
        "db_size": len(engine.database)
    }


@app.post("/recognize", response_model=RecognitionResponse, summary="Recognize faces in image")
async def recognize(image: UploadFile = File(...), log: bool = True):
    """Recognize faces in uploaded image and optionally log attendance."""
    img = await decode_upload(image)
    elapsed_ms, detections = engine.process_frame(img)  # FIX: correct unpacking (elapsed_ms, detections)

    results = []
    for det in detections:
        results.append(DetectionResult(
            identity=det.identity,
            similarity=round(det.similarity, 4),
            bbox=list(det.bbox),
        ))
        if log and det.identity != "unknown":  # FIX: "Unknown" → "unknown" (lowercase)
            pid = next(
                (engine.database.ids[i] for i, n in enumerate(engine.database.names)
                 if n == det.identity), det.identity
            )
            log_attendance(db_conn, pid, det.identity, det.similarity)

    return RecognitionResponse(faces=results, count=len(results), processing_ms=round(elapsed_ms, 2))


@app.get("/attendance", response_model=list[AttendanceRecord], summary="Query attendance log")
def get_attendance(
    date_filter: Optional[str] = None,
    person_id: Optional[str] = None,
    limit: int = 200,
):
    q = "SELECT * FROM attendance WHERE 1=1"
    params: list = []
    if date_filter:
        q += " AND date=?"
        params.append(date_filter)
    if person_id:
        q += " AND person_id=?"
        params.append(person_id)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = db_conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/people", response_model=list[PersonInfo], summary="List registered people")
def list_people():
    db = engine.database
    seen = {}
    for pid, name in zip(db.ids, db.names):
        if pid not in seen:
            seen[pid] = name
    return [{"person_id": pid, "name": name} for pid, name in seen.items()]


@app.delete("/people/{person_id}", summary="Remove a person")
def delete_person(person_id: str):
    """Delete a registered person and all their embeddings from database."""
    db = engine.database
    indices = [i for i, pid in enumerate(db.ids) if pid == person_id]
    if not indices:
        raise HTTPException(404, f"person_id '{person_id}' not found")
    keep = [i for i in range(len(db.names)) if i not in set(indices)]
    db.embeddings = db.embeddings[keep] if keep else np.empty((0, 512), dtype=np.float32)
    db.names = [db.names[i] for i in keep]
    db.ids = [db.ids[i] for i in keep]
    db.save(Model_DB)  # FIX: MODEL_DB → Model_DB
    return {"status": "deleted", "person_id": person_id, "db_size": len(db)}


@app.get("/stats", summary="Engine and database statistics")
def get_stats():
    s = engine.stats
    return {
        "engine": s,
        "database": {"people": len(set(engine.database.ids)),
                     "reference_images": len(engine.database)},
    }
def _gen_frames():
    """Generator for MJPEG stream frames."""
    while True:
        if _cap is None or not _cap.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = _cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        elapsed_ms, detections = engine.process_frame(frame)  # FIX: correctly unpack (elapsed_ms, detections)
        annotated = engine._draw(frame, detections)  # Draw annotations on frame

        # Log attendance from stream
        for det in detections:
            if det.identity != "unknown":  # FIX: "Unknown" → "unknown"
                pid = next(
                    (engine.database.ids[i] for i, n in enumerate(engine.database.names)
                     if n == det.identity), det.identity
                )
                log_attendance(db_conn, pid, det.identity, det.similarity, source="stream")

        # HUD overlay with stats
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 36), (10, 10, 20), -1)
        hud = f"FaceVision  |  Faces: {len(detections)} | {elapsed_ms:.1f}ms"
        cv2.putText(annotated, hud, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.get("/stream", summary="MJPEG live stream")
def video_stream():
    return StreamingResponse(_gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time face stream with JSON updates."""
    await websocket.accept()
    error_count = 0
    try:
        frame_count = 0
        while True:
            try:
                if _cap is None or not _cap.isOpened():
                    await asyncio.sleep(0.1)
                    continue
                ret, frame = _cap.read()
                if not ret:
                    await asyncio.sleep(0.05)
                    continue

                elapsed_ms, detections = engine.process_frame(frame)
                annotated = engine._draw(frame, detections)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf).decode()

                payload = {
                    "frame": b64,
                    "detections": [
                        {"identity": d.identity, "similarity": round(d.similarity, 3),
                         "bbox": list(d.bbox)}
                        for d in detections
                    ],
                    "elapsed_ms": round(elapsed_ms, 2),
                    "ts": time.time(),
                }
                await websocket.send_json(payload)
                error_count = 0
                await asyncio.sleep(0.03)   # ~30 fps
            except Exception as e:
                error_count += 1
                logger.warning(f"WebSocket frame error ({error_count}): {e}")
                if error_count > 10:
                    break
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve dashboard HTML or redirect to API docs."""
    index = Path("static/index.html")
    if index.exists():
        return index.read_text()
    return "<h2>FaceVision API — visit <a href='/docs'>/docs</a></h2>"