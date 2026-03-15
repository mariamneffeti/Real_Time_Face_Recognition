# Quick Start Guide for FaceVision

## Installation (One-time setup)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify everything is ready
python verify_setup.py
```

---

## Usage Mode 1: Command-Line (run.py)

### Start with webcam
```bash
python run.py
```

**Controls:**
- `Q` or `ESC` - Quit
- `S` - Save snapshot
- `+` / `-` - Adjust detection sensitivity

### Register a new person
```bash
python run.py --register "Alice Smith" photo.jpg
```

### Process a video file
```bash
python run.py --source video.mp4
```

### Process a single image
```bash
python run.py --source photo.jpg
```

### Advanced options
```bash
python run.py --help  # See all options
python run.py --threshold 0.50  # Adjust sensitivity
python run.py --skip 2  # Process every 2nd frame (faster)
```

---

## Usage Mode 2: Web Dashboard (api.py)

### Start the server
```bash
uvicorn api:app --reload
```

### Open in browser
- Dashboard: `http://localhost:8000/`
- API Docs: `http://localhost:8000/docs`

### Features
- 📹 **Live camera feed** with real-time detections
- 📊 **Statistics** - people registered, FPS, accuracy
- 📅 **Attendance log** - search and filter records
- 👥 **People manager** - add, view, delete faces
- 📤 **Upload images** for batch recognition

---

## Example Workflows

### Workflow 1: Register and Recognize (CLI)
```bash
# 1. Register a person
python run.py --register "Bob" bob_photo.jpg

# 2. Start recognition with webcam
python run.py

# 3. Bob's face will be detected when visible
```

### Workflow 2: Batch Recognition (API)
```bash
# 1. Start server
uvicorn api:app --reload

# 2. Open dashboard
# http://localhost:8000/

# 3. Register people via "Register New Face" tab

# 4. Upload images for recognition via API

# 5. View attendance log with timestamps
```

### Workflow 3: Attendance Tracking (Live)
```bash
# 1. Register employees
python run.py --register "Employee 1" photo1.jpg
python run.py --register "Employee 2" photo2.jpg

# 2. Start API server
uvicorn api:app --reload

# 3. Point camera at entrance

# 4. Attendance automatically logged to database

# 5. View daily records in dashboard
```

---

## Common Commands

| Task | Command |
|------|---------|
| **Quick demo** | `python run.py` |
| **Register person** | `python run.py --register "Name" photo.jpg` |
| **Lower detection threshold** | `python run.py --threshold 0.45` |
| **Skip frames for speed** | `python run.py --skip 2` |
| **See all options** | `python run.py --help` |
| **Start web server** | `uvicorn api:app --reload` |
| **API documentation** | Visit `http://localhost:8000/docs` |
| **Verify setup** | `python verify_setup.py` |

---

## Typical Settings

### For High Security (Low False Positives)
```bash
python run.py --threshold 0.60
```

### For High Sensitivity (No Misses)
```bash
python run.py --threshold 0.45
```

### For Low-Power Devices
```bash
python run.py --skip 3 --threshold 0.50
```

### For Batch Processing
```bash
python run.py --source video.mp4 --skip 2
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No camera** | `python run.py --source 1` (try device 1, 2, etc) |
| **Low FPS** | Use `--skip 2` or `--skip 3` |
| **Not recognizing faces** | Lower threshold: `--threshold 0.45` |
| **False detections** | Raise threshold: `--threshold 0.65` |
| **API won't start** | Run `pip install fastapi uvicorn` |
| **Slow startup** | Already optimized, first run downloads models |

---

## File Locations

- **Face Database:** `data/face_db.pkl`
- **Attendance Log:** `data/attendance.db`
- **Configuration:** `api.py` (top variables)

---

## Key Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `ESC` | Quit |
| `S` | Save snapshot |
| `+` | Increase threshold |
| `-` | Decrease threshold |

---

## Performance

- **Average inference time:** 50-150ms per frame (CPU)
- **FPS with skip:** 20-30 FPS (skip=1), 5-10 FPS (skip=1)
- **GPU mode:** 2-3x faster (if available)

---

## What Gets Logged

- **Face embeddings** → `data/face_db.pkl`
- **Attendance records** → `data/attendance.db`
- **Console output** → Real-time status & FPS
- **Snapshots** → Current directory with timestamp

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Run `python run.py --help` for all CLI options
3. Visit `http://localhost:8000/docs` for API documentation
4. Run `python verify_setup.py` to diagnose issues

---

**That's it! You're ready to use FaceVision.** 🎉
