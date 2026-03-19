# FaceVision - Face Recognition System
## v1.0 - Improved Edition
![CI](https://github.com/mariamneffeti/Real_Time_Face_Recognition/actions/workflows/ci.yml/badge.svg)
A modern real-time face recognition and attendance tracking system using DeepFace/InsightFace.

---

## 🎯 Features

### Core Functionality
- **Real-time face detection & recognition** from webcam or video files
- **Automatic attendance logging** with timestamp and confidence scores  
- **Face registration** with single or multiple reference images
- **Configurable detection threshold** for sensitivity control
- **Performance metrics** (FPS, inference time, detection stats)

### Web Dashboard
- **Live camera feed** with face detection overlay
- **Current detections** display with confidence scores
- **Attendance log** with date/person filtering
- **People management** - register, view, and delete personnel
- **Statistics dashboard** with system metrics
- **Responsive dark-themed UI** for modern appearance

### API Endpoints
- `POST /register` - Register new person with face image
- `POST /recognize` - Recognize faces in uploaded image
- `GET /stream` - MJPEG live stream from camera
- `GET /ws/stream` - WebSocket for real-time detections
- `GET /attendance` - Query attendance records with filters
- `GET /people` - List all registered people with stats
- `DELETE /people/{person_id}` - Remove person from database
- `GET /stats` - Get engine and database statistics

## Liveness Detection
Download the model before running:
curl -L "https://github.com/minivision-ai/..." -o models/liveness.onnx
---

## 📋 Installation

### Requirements
- Python 3.9+
- OpenCV (opencv-python)
- FastAPI & Uvicorn
- DeepFace or InsightFace
- SQLite3 (built-in)

### Setup

```bash
# 1. Clone/download project
cd Face_recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create data directory
mkdir -p data
```

---

## 🚀 Usage

### CLI Interface (run.py)

```bash
# Webcam real-time recognition
python run.py

# Process video file
python run.py --source video.mp4

# Process single image
python run.py --source photo.jpg

# Register new face
python run.py --register "Alice" photo.jpg

# Use DeepFace backend (recommended)
python run.py --backend deepface

# Adjust detection sensitivity
python run.py --threshold 0.50
```

**Controls (while running):**
- `Q` / `ESC` - Quit
- `S` - Save snapshot
- `+` / `-` - Adjust threshold
- `L` - Toggle landmarks

### Web API (api.py)

```bash
# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Access dashboard at http://localhost:8000/
# API documentation at http://localhost:8000/docs
```

---

## 🎨 Web Dashboard Features

### Dashboard Tab
- Live camera feed with real-time detections
- Current face detections panel
- System statistics (people, detections, FPS)
- One-click snapshot capture

### Attendance Log
- View all attendance records
- Filter by date and person
- See confidence scores
- Track detection source (camera/upload)

### Registered People
- Grid view of all registered individuals
- Detection count and average confidence
- Last seen timestamp
- Quick delete option

### Register New Face
- Drag-and-drop image upload
- Face detection validation
- Automatic ID generation
- Image preview before registration

---

## ⚙️ Configuration

### api.py Constants
```python
DB_PATH = "data/attendance.db"        # Attendance database
Model_DB = "data/face_db.pkl"         # Face embeddings database
Backend = "deepface"                   # or "insightface"
THRESHOLD = 0.55                       # Detection confidence threshold (0-1)
```

### run.py Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | 0=webcam, path to video/image |
| `--backend` | `deepface` | Detection backend |
| `--threshold` | `0.55` | Similarity threshold (0-1) |
| `--db` | `data/face_db.pkl` | Face database path |
| `--batch` | `1` | Batch size (GPU optimization) |
| `--skip` | `1` | Process every Nth frame |
| `--landmarks` | - | Show facial landmarks |
| `--register` | - | Register: `--register "Name" image.jpg` |

---

## 🔧 Backend Comparison

### DeepFace (Recommended)
- ✅ Easier setup (auto model download)
- ✅ Multiple model options
- ✅ Better accuracy on diverse faces
- ❌ Slightly slower inference
- Model: FaceNet512

### InsightFace
- ✅ Faster inference
- ✅ Lighter weight
- ❌ Manual model installation required
- ❌ More complex setup
- Model: ArcFace (buffalo_l)

---

## 📊 Database Schema

### attendance table
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY,
    person_id TEXT,              -- Unique person identifier
    name TEXT,                   -- Person's name
    timestamp TEXT,              -- ISO format datetime
    date TEXT,                   -- YYYY-MM-DD
    similarity REAL,            -- Confidence score (0-1)
    source TEXT                 -- 'camera' or 'upload'
)
```

### face_db.pkl
- Pickled Python dictionary with:
  - `embeddings`: NumPy array (N, 512) of face vectors
  - `names`: List of person names
  - `ids`: Unique person IDs

---

## 🐛 Troubleshooting

### "No faces detected in frame"
- Improve lighting conditions
- Get face closer to camera (30cm-1m)
- Lower threshold: `--threshold 0.45`

### Camera not opening
- Check camera permissions
- Try different source: `python run.py --source 1`
- Verify no other app is using camera

### DeepFace model download fails
- Check internet connection
- Try: `python -c "from deepface import DeepFace; DeepFace.extract_faces(np.zeros((100,100,3)))"` 
- Models download to: `~/.deepface/weights/`

### InsightFace requires model download
- Run: `python download_models.py` (if provided)
- Or use DeepFace instead: `--backend deepface`

### Poor recognition accuracy
- Register with 2-3 different photos per person
- Ensure good lighting in both registration and detection
- Increase threshold if too many false positives
- Decrease threshold if missing detections

---

## 🎯 Performance Tips

1. **Use frame skipping** for real-time streams:
   ```bash
   python run.py --skip 2  # Process every 2nd frame
   ```

2. **Enable batch processing** on GPU:
   ```bash
   python run.py --batch 4  # Process 4 frames at once
   ```

3. **Use DeepFace** - better CPU performance

4. **Lower image resolution** if running on weak hardware

5. **Adjust JPEG quality** in `api.py` `_gen_frames()`:
   ```python
   cv2.IMWRITE_JPEG_QUALITY = 60  # Default 80
   ```

---

## 🔒 Security & Privacy

### Recommendations
- **Do not publicly expose** the API without authentication
- **Secure the attendance database** - contains personal data
- **Use HTTPS** in production
- **Implement API authentication** (JWT, API keys)
- **Regular backups** of face_db.pkl and attendance.db
- **Clean old attendance records** periodically

### Example: Add API authentication
```python
from fastapi.security import HTTPBearer, HTTPAuthCredentials
security = HTTPBearer()

@app.post("/recognize")
async def recognize(credentials: HTTPAuthCredentials, ...):
    if credentials.credentials != "secret-key":
        raise HTTPException(401, "Unauthorized")
    ...
```

---

## 📝 Improvements Made

### Bug Fixes
✅ Fixed FPS counter double-call bug  
✅ Fixed return value unpacking (`process_frame()`)  
✅ Fixed known/unknown classification (case sensitivity)  
✅ Fixed DeepFace model name compatibility  
✅ Changed backend default to `deepface` (more reliable)

### Error Handling
✅ Input validation on all endpoints  
✅ File size and format checks  
✅ Graceful error messages to users  
✅ Error logging throughout  
✅ WebSocket error recovery

### UI/UX Improvements
✅ Modern dark-themed dashboard  
✅ Real-time statistics  
✅ Attendance history with filtering  
✅ People management (CRUD)  
✅ Live camera stream  
✅ Drag-and-drop file upload  
✅ Responsive design

### Code Quality
✅ Better error messages  
✅ Input sanitization  
✅ Comprehensive logging  
✅ Type hints  
✅ Documentation

---

## 📚 API Usage Examples

### Register a Person
```bash
curl -X POST http://localhost:8000/register \
  -F "name=John Doe" \
  -F "image=@photo.jpg"
```

### Recognize Faces
```bash
curl -X POST http://localhost:8000/recognize \
  -F "image=@test.jpg"
```

### Get Attendance (Today)
```bash
curl "http://localhost:8000/attendance?date_filter=2026-03-14"
```

### List Registered People
```bash
curl http://localhost:8000/people
```

---

## 📦 Project Structure

```
Face_recognition/
├── run.py                 # CLI entry point
├── api.py                 # FastAPI server
├── core/
│   ├── __init__.py
│   └── engine.py         # Face recognition engine
├── static/
│   ├── index.html        # Redirect to dashboard
│   └── dashboard.html    # Web UI
├── data/
│   ├── face_db.pkl       # Face embeddings (auto-created)
│   └── attendance.db     # Attendance log (auto-created)
└── requirements.txt      # Python dependencies
```

---

## 🤝 Contributing

Suggestions for further improvements:
- Face tracking across frames
- Multiple detection models (YOLO, etc.)
- Docker containerization
- Docker Compose for easy deployment
- REST API authentication
- Database backup/restore utilities
- Facial landmarks visualization
- Emotion/age detection
- Batch attendance import/export

## 🐳 Docker
A Dockerfile and docker-compose.yml are included for containerized deployment.
Tested on Linux. On Windows, ensure WSL 2 is properly configured before running
`docker compose up --build`.
---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 💡 Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in console output
3. Verify all dependencies are installed
4. Test with the example image/video files

---

**Last Updated:** March 14, 2026  
**Status:** Production Ready
