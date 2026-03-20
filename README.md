# FaceVision — Real-Time Face Recognition System
> v1.0 · Improved Edition

![CI](https://github.com/mariamneffeti/Real_Time_Face_Recognition/actions/workflows/ci.yml/badge.svg)

A modern real-time face recognition and attendance tracking system built with **DeepFace** / **InsightFace**, **FastAPI**, and a responsive web dashboard.

---

## 🎯 Features

### Core
- Real-time face detection & recognition from webcam or video files
- **Anti-spoofing / liveness detection** — rejects photos and screens
- Automatic attendance logging with timestamps and confidence scores
- Face registration with single or multiple reference images
- Configurable detection threshold for sensitivity control
- Performance metrics: FPS, inference time, detection stats

### Web Dashboard
- Live camera feed with face detection overlay
- Real-time detections panel via WebSocket (updates instantly)
- Attendance log with date/person filtering
- People management — register, view, and delete personnel
- Statistics dashboard with system metrics
- Responsive dark-themed UI

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register` | Register new person with face image |
| `POST` | `/recognize` | Recognize faces in uploaded image |
| `GET` | `/stream` | MJPEG live stream from camera |
| `GET` | `/ws/stream` | WebSocket for real-time detections |
| `GET` | `/attendance` | Query attendance records with filters |
| `GET` | `/people` | List all registered people with stats |
| `DELETE` | `/people/{person_id}` | Remove person from database |
| `GET` | `/stats` | Get engine and database statistics |

---

## 🧠 How It Works

### Tech Stack
| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI | Async, fast, auto-generates API docs |
| Face Recognition | DeepFace (FaceNet512) | Best accuracy on diverse faces, easy setup |
| Anti-Spoofing | ONNX (MiniFASNet) | Lightweight liveness detection on CPU |
| Database | SQLite | Lightweight, zero-config, perfect for local deployment |
| Frontend | Vanilla JS + HTML | No build step, fast to load |
| Streaming | MJPEG + WebSocket | Low latency for real-time camera feed |

### Recognition Pipeline
1. **Frame capture** — OpenCV reads from webcam
2. **Face detection** — DeepFace locates faces in the frame
3. **Liveness check** — MiniFASNet verifies the face is real, not a photo
4. **Embedding extraction** — Each face is converted to a 512-dimension vector
5. **Similarity search** — Vector compared to registered faces using cosine similarity
6. **Attendance logging** — Matches above threshold saved to SQLite with timestamp

---

## 📋 Installation

### Requirements
- Python 3.9+
- Webcam (for live detection)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/mariamneffeti/Real_Time_Face_Recognition.git
cd Real_Time_Face_Recognition

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env

# 5. Create data directory
mkdir -p data

# 6. Download liveness model
curl -L "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx" -o models/liveness.onnx

# 7. Verify setup
python verify_setup.py
```

---

## 🚀 Usage

### Web Dashboard (Recommended)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Dashboard:  http://localhost:8000/
# API docs:   http://localhost:8000/docs
```

### CLI Interface
```bash
# Webcam real-time recognition
python run.py

# Register a new face
python run.py --register "Alice" photo.jpg

# Process a video file
python run.py --source video.mp4

# Adjust sensitivity
python run.py --threshold 0.50

# Skip frames for speed
python run.py --skip 2
```

**Keyboard controls:**
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `+` / `-` | Adjust threshold |
| `L` | Toggle landmarks |

---

## 🐳 Docker

```bash
docker compose up --build
# Visit http://localhost:8000
```

> On Windows, ensure Docker Desktop is running and WSL 2 is configured first.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and adjust as needed:

```env
THRESHOLD=0.55
BACKEND=deepface
DB_PATH=data/attendance.db
MODEL_DB=data/face_db.pkl
```

### CLI Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | `0` = webcam, or path to video/image |
| `--backend` | `deepface` | Detection backend |
| `--threshold` | `0.55` | Similarity threshold (0–1) |
| `--db` | `data/face_db.pkl` | Face database path |
| `--batch` | `1` | Batch size (GPU optimisation) |
| `--skip` | `1` | Process every Nth frame |
| `--landmarks` | — | Show facial landmarks |
| `--register` | — | `--register "Name" image.jpg` |

---

## 🔧 Backend Comparison

| | DeepFace ✅ Recommended | InsightFace |
|---|---|---|
| Setup | Easy (auto model download) | Manual model install required |
| Accuracy | Better on diverse faces | Good |
| Speed | Slightly slower | Faster inference |
| Model | FaceNet512 | ArcFace (buffalo_l) |

---

## 🛡️ Liveness Detection (Anti-Spoofing)

Prevents attackers from fooling the system with a printed photo or screen.

```
❌ Hold up a photo of Alice  →  "Spoof detected"
✅ Real Alice in front of camera  →  "Hello Alice (92%)"
```

Uses MiniFASNetV2 — a lightweight ONNX model that runs entirely on CPU with no extra GPU requirements.

---

## 📊 Database Schema

### `attendance` table
```sql
CREATE TABLE attendance (
    id         INTEGER PRIMARY KEY,
    person_id  TEXT,     -- Unique person identifier
    name       TEXT,     -- Person's name
    timestamp  TEXT,     -- ISO format datetime
    date       TEXT,     -- YYYY-MM-DD
    similarity REAL,     -- Confidence score (0–1)
    source     TEXT      -- 'camera' or 'upload'
)
```

### `face_db.pkl`
- `embeddings` — NumPy array `(N, 512)` of face vectors
- `names` — list of person names
- `ids` — unique person IDs

---

## 📦 Project Structure

```
Real_Time_Face_Recognition/
├── run.py                 # CLI entry point
├── api.py                 # FastAPI server
├── verify_setup.py        # Setup verification
├── requirements.txt
├── .env.example           # Environment config template
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI
├── core/
│   ├── __init__.py
│   ├── engine.py          # Face recognition engine
│   └── liveness.py        # Anti-spoofing module
├── models/
│   └── liveness.onnx      # Liveness model (download separately)
├── static/
│   ├── index.html
│   └── dashboard.html
└── data/                  # Auto-created at runtime
    ├── face_db.pkl
    └── attendance.db
```

---

## 📚 API Examples

```bash
# Register a person
curl -X POST http://localhost:8000/register \
  -F "name=John Doe" \
  -F "image=@photo.jpg"

# Recognize faces
curl -X POST http://localhost:8000/recognize \
  -F "image=@test.jpg"

# Get today's attendance
curl "http://localhost:8000/attendance?date_filter=2026-03-20"

# List registered people
curl http://localhost:8000/people
```

---

## 🎯 Performance Tips

1. **Frame skipping** for real-time: `python run.py --skip 2`
2. **Batch processing** on GPU: `python run.py --batch 4`
3. Use **DeepFace** for better CPU performance
4. Lower image resolution on weak hardware

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No faces detected" | Improve lighting; move closer (30cm–1m); try `--threshold 0.45` |
| Camera not opening | Check permissions; try `--source 1` |
| DeepFace model fails | Check internet; models download to `~/.deepface/weights/` |
| Poor accuracy | Register 2–3 photos per person; ensure good lighting |
| Liveness model missing | Run the curl download command in setup step 6 |

---

## 🔒 Security & Privacy

- Do **not** publicly expose the API without authentication
- Use **HTTPS** in production
- The attendance database contains personal data — secure it accordingly
- Back up `face_db.pkl` and `attendance.db` regularly
- `data/` and `models/` are excluded from version control via `.gitignore`

---

## 📝 Changelog

### Bug Fixes
- Fixed FPS counter double-call bug
- Fixed WebSocket URL construction in frontend
- Fixed known/unknown classification (case sensitivity)
- Fixed DeepFace model name compatibility
- Fixed `db_conn` PRAGMA calls outside function scope

### Features Added
- Liveness / anti-spoofing detection
- `.env` based configuration
- GitHub Actions CI pipeline
- Docker + Docker Compose support
- WebSocket real-time detections panel
- Auto-refreshing attendance table

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

**Status:** Production Ready · **Last Updated:** March 2026