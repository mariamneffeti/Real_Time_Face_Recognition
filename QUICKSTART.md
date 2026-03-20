# FaceVision — Quick Start Guide

## First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env

# 3. Download liveness model
curl -L "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx" -o models/liveness.onnx

# 4. Verify everything is ready
python verify_setup.py
```

---

## Mode 1: Web Dashboard

```bash
uvicorn api:app --reload
```

Then open **http://localhost:8000** in your browser.

| Tab | What it does |
|-----|-------------|
| Dashboard | Live camera feed + real-time detections |
| Attendance Log | Search and filter attendance records |
| Registered People | View, manage, delete registered faces |
| Register New Face | Add a new person via drag-and-drop |

---

## Mode 2: Command Line

```bash
# Start webcam recognition
python run.py

# Register a new person
python run.py --register "Alice Smith" photo.jpg

# Process a video file
python run.py --source video.mp4

# See all options
python run.py --help
```

**Controls while running:**
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save snapshot |
| `+` / `-` | Adjust threshold |
| `L` | Toggle landmarks |

---

## Mode 3: Docker

```bash
docker compose up --build
# Visit http://localhost:8000
```

---

## Common Workflows

### Register & Recognize
```bash
# Register someone
python run.py --register "Bob" bob_photo.jpg

# Start recognition
python run.py
# Bob's face will be recognized when visible
```

### Attendance Tracking
```bash
# Register your people
python run.py --register "Employee 1" photo1.jpg
python run.py --register "Employee 2" photo2.jpg

# Start web server and point camera at entrance
uvicorn api:app --reload

# View daily attendance at http://localhost:8000
```

---

## Tuning Detection

| Scenario | Command |
|----------|---------|
| High security (fewer false positives) | `python run.py --threshold 0.65` |
| High sensitivity (fewer misses) | `python run.py --threshold 0.45` |
| Low-power device | `python run.py --skip 3` |
| Fast processing | `python run.py --skip 2 --batch 4` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No camera found | Try `--source 1` or `--source 2` |
| Low FPS | Add `--skip 2` |
| Not recognizing faces | Lower threshold: `--threshold 0.45` |
| Too many false positives | Raise threshold: `--threshold 0.65` |
| Slow first startup | Normal — DeepFace downloads models once |
| Liveness model missing | Run the curl command in setup step 3 |

---

## File Locations

| File | Purpose |
|------|---------|
| `data/face_db.pkl` | Registered face embeddings |
| `data/attendance.db` | Attendance records |
| `models/liveness.onnx` | Anti-spoofing model |
| `.env` | Configuration (thresholds, paths) |

---

## Performance

- **CPU inference:** ~50–150ms per frame
- **With skip=2:** ~20–30 FPS effective
- **GPU (if available):** 2–3× faster

---

**Need more help?** See `README.md` or visit `http://localhost:8000/docs` for full API documentation.