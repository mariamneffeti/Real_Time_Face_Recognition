""" engine.py - Core face recognition engine for FaceVision
Detection : OpenCV DNN
Embedding : InsightFace (ArcFace) or DeepFace (FaceNet512)
Matching : Cosine similarity with configurable threshold
Database : Simple pickle-based storage for embeddings and metadata
"""

#imports
from __future__ import annotations
import logging
import pickle
import time
from dataclasses import dataclass,field
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

#loggings for easier debugging and monitoring
logger = logging.getLogger("facevision.engine")

#dataclass to initialize the __init__ method of the FaceDetection and FaceDatabase classes, which will hold the detected face information and the database of known faces respectively.
@dataclass
class FaceDetection:
    """Represents a detected face with embedding and identity information.
    
    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) - top-left and bottom-right corners
        embedding: 512D face embedding vector from model
        identity: Recognized person's name ("unknown" if not recognized)
        similarity: Confidence score (0 to 1) for the match
        landmarks: Optional facial landmarks (eyes, nose, mouth, etc.)
    """
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    identity: str= "unknown"
    similarity: float = 0.0
    landmarks: Optional[np.ndarray] = None

@dataclass
class FaceDatabase:
    """In-memory database of registered faces with persistence support.
    
    Stores face embeddings and metadata (names, IDs) that can be saved/loaded
    from disk. New embeddings are appended to the database via add_face().
    All three lists (embeddings, names, ids) maintain the same order.
    """
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 512), dtype=np.float32))
    # Shape: (N, 512) where N = number of registered faces. Grows as faces are added.
    names: list[str] = field(default_factory=list)
    # Person names in same row order as embeddings array
    ids: list[str] = field(default_factory=list)
    # Unique person IDs in same row order as embeddings array
    def add_face(self, embedding: np.ndarray, name: str, person_id: Optional[str] = None)-> None:
        """Add a new face embedding to the database.
        
        Args:
            embedding: 512D face vector
            name: Person's name
            person_id: Optional unique ID (auto-generated from name if None)
        """
        emb = _unit(embedding).reshape(1, -1)
        self.embeddings = np.vstack([self.embeddings, emb]) if len(self.embeddings) else emb
        self.names.append(name)
        self.ids.append(person_id or name.lower().replace(" ", "_"))
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "names": self.names, "ids": self.ids}, f)
        logger.info(f"Saved {len(self.names)} faces to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FaceDatabase":
        """Load database from pickle file on disk.
        
        Args:
            path: Path to pickle file containing saved embeddings/names/ids
        
        Returns:
            FaceDatabase instance with all data restored from disk
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        db = cls()
        db.embeddings = data["embeddings"]
        db.names = data["names"]
        db.ids = data["ids"]
        logger.info(f"Loaded {len(db.names)} faces from {path}")
        return db
    def __len__(self) -> int:
        return len(self.names)
def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length (L2 normalization).
    This makes cosine similarity equal to dot product.
    1e-9 added to avoid division by zero."""
    n = np.linalg.norm(v)
    return v / (n + 1e-9)
def cosine_similarity_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query embedding and all database embeddings.
    Returns array of similarity scores (0 to 1) for each face in database.
    Works because embeddings are normalized to unit length."""
    return db @ query 


class _InsightFaceBackend:
    def __init__(self, det_size=(640,640)):
        """Initialize InsightFace model.
        
        Args:
            det_size: Input resolution for detection (default 640x640)
        """
        import insightface
        import os
        
        # Create model directory
        model_dir = os.path.expanduser("~/.insightface/models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Set environment variable for model location
        os.environ['INSIGHTFACE_HOME'] = os.path.expanduser("~/.insightface")
        
        try:
            # Try to initialize InsightFace
            self.app = insightface.app.FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=-1, det_size=det_size)
            logger.info("InsightFace (ArcFace) backend ready")
        except Exception as e:
            logger.error("InsightFace initialization failed: %s", e)
            logger.info("Falling back to DeepFace backend...")
            # Fallback to DeepFace if InsightFace fails
            raise RuntimeError(
                "InsightFace models not properly installed. "
                "Run: python download_models.py\n"
                "Or use --backend deepface instead."
            ) from e
    def get_faces(self,bgr_frame : np.ndarray) -> list[dict]:
        faces = self.app.get(bgr_frame)
        results = []
        for face in faces:
            results.append({
                "bbox": tuple(map(int, face.bbox)),
                "embedding": face.normed_embedding.astype(np.float32),
                "landmarks": face.kps,
            })
        return results
class _DeepFaceBackend:
    def __init__(self,model_name="Facenet512",detector = "opencv"):
        from deepface import DeepFace
        self._df = DeepFace
        self._model = model_name
        self._detector = detector
        # Note: Models are lazy-loaded on first use (in get_faces method)
        # Skip dummy run to avoid unnecessary downloads during init
        logger.info("DeepFace (%s) backend ready (models will load on first use)",model_name)
    def get_faces(self,bgr_frame : np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try :
            results= self._df.represent(rgb, model_name=self._model, detector_backend=self._detector, enforce_detection=False)
        except Exception as e:
            logger.error("DeepFace error: %s",e)
            return []
        faces = []
        for res in results:
            r = res.get("facial_area",{})
            x = r.get("x",0)
            y = r.get("y",0)
            w = r.get("w",0)
            h = r.get("h",0)
            
            faces.append({
                "bbox": (x, y, x + w, y + h),
                "embedding": _unit(np.array(res["embedding"], dtype=np.float32)),
                "landmarks": None,
            })
        return faces
class FaceEngine:
    """Main face recognition engine. Handles detection, matching, and database management.
    
    Workflow:
    1. Initialize with a detection backend (InsightFace or DeepFace)
    2. Load existing database if available
    3. Process frames to detect and match faces against database
    4. Register new faces with register()
    5. Save database on updates
    
    Can be used for:
    - Real-time video processing (webcam, video file)
    - Single image processing
    - Batch processing of multiple frames
    """
    def __init__(self, backend: str = "insightface", threshold: float = 0.55, db_path: str = "face_db.pkl", batch_size: int = 1):
        """Initialize FaceEngine.
        
        Args:
            backend: Detection backend - 'insightface' (faster, better) or 'deepface' (flexible models)
            threshold: Minimum similarity score (0-1) to recognize a face as known person.
                      Lower = more lenient matching. Range: 0.4-0.7 typical.
            db_path: Path to save/load face database pickle file
            batch_size: Number of frames to process in parallel (not yet implemented)
        """
        self.threshold = threshold
        self.db_path = db_path
        self.batch_size = batch_size
        self._db: FaceDatabase = FaceDatabase()
        # Track performance metrics
        self._stats = {"frames": 0, "detections": 0, "matches": 0, "total_ms": 0.0}
        
        # Load existing database if available
        if Path(db_path).exists():
            self._db = FaceDatabase.load(db_path)
            logger.info(f"Loaded existing database from {db_path}")
        
        # Initialize face detection backend
        if backend == "insightface":
            self._backend = _InsightFaceBackend()
        else:
            self._backend = _DeepFaceBackend()
        logger.info("FaceEngine initialized with backend: %s, threshold: %.2f, database has %d people", backend, threshold, len(self._db))
    
    def register(self, name: str, image: np.ndarray, person_id: Optional[str] = None) -> bool:
        """Register a new face to the database from an image.
        Takes the largest face detected (usually the main subject).
        
        Args:
            name: Person's name
            image: Image containing the face (BGR format, OpenCV)
            person_id: Optional unique ID (auto-generated from name if None)
        
        Returns:
            True if successful, False if no face detected
        """
        faces = self._backend.get_faces(image)
        if not faces:
            logger.warning("No face detected for registration: %s", name)
            return False
        # Sort by face size (area) and take the largest
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        # BUG FIX: embedding first, then name
        self._db.add_face(faces[0]["embedding"], name, person_id)
        self._db.save(self.db_path)
        logger.info("Registered new face: %s (db_size: %d)", name, len(self._db))
        return True


    def process_frame(self, bgr_frame: np.ndarray) -> tuple[float, list[FaceDetection]]:
        """Detect faces in frame and match against database.
        
        Args:
            bgr_frame: Input frame in BGR format (OpenCV standard)
        
        Returns:
            (elapsed_ms, list of FaceDetection objects with identity info)
        """
        t0 = time.perf_counter()
        raw_faces = self._backend.get_faces(bgr_frame)
        detections: list[FaceDetection] = []
        
        # For each detected face, find matching identity in database
        for f in raw_faces:
            identity, similarity = self._match(f["embedding"])
            det = FaceDetection(bbox=f["bbox"], embedding=f["embedding"], identity=identity, similarity=similarity, landmarks=f.get("landmarks"))
            detections.append(det)
        
        # BUG FIX: use consistent variable name elapsed_ms
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stats["frames"] += 1
        self._stats["detections"] += len(detections)
        self._stats["matches"] += sum(1 for d in detections if d.identity != "unknown")
        self._stats["total_ms"] += elapsed_ms
        return elapsed_ms, detections
    def process_batch(self, frames: list[np.ndarray]) -> list[tuple[float, list[FaceDetection]]]:
        """Process multiple frames.
        
        Args:
            frames: List of frames in BGR format
        
        Returns:
            List of (elapsed_ms, detections) tuples
        """
        return [self.process_frame(f) for f in frames]
    @property
    def database(self) -> FaceDatabase:
        return self._db
    @property
    def stats(self) -> dict:
        s= dict(self._stats)
        s["avg_ms"] = s["total_ms"] / s["frames"] if s["frames"] else 0.0
        return s
    def _match(self,embedding: np.ndarray) -> tuple[str,float]:
        """Find best matching face in database based on embedding similarity.
        Returns (name, similarity_score). If best match below threshold, returns (unknown, score)."""
        if len(self._db) == 0:
            return "unknown", 0.0
        # Compute similarity to all database faces
        sims = cosine_similarity_batch(_unit(embedding), self._db.embeddings)
        best_idx = int(np.argmax(sims))  # Get index of highest similarity
        best_sim = float(sims[best_idx])
        # Only return a name if similarity exceeds threshold
        if best_sim >= self.threshold:
            return self._db.names[best_idx], best_sim
        else:
            return "unknown",best_sim
    def _draw(self, frame: np.ndarray, detections: list[FaceDetection]) -> np.ndarray:
        """Draw bounding boxes, labels, and landmarks on frame.
        
        Visualization:
        - Bright cyan (0, 229, 180): Known faces (identity != "unknown")
        - Red (70, 70, 255): Unknown faces
        
        Args:
            frame: Input frame to annotate (will be modified in-place)
            detections: List of FaceDetection objects with boxes and IDs
        
        Returns:
            Annotated frame with boxes, labels, and landmarks drawn
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            known = det.identity != "unknown"
            # Color: cyan for known, red for unknown
            color = (0, 229, 180) if known else (70, 70, 255)
            
            # Draw main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw corner brackets (looks better than full box)
            cl = 14  # corner line length
            for cx, cy, sx, xy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, 1)]:
                cv2.line(frame, (cx, cy), (cx + cl * sx, cy), color, 3)  # horizontal line
                cv2.line(frame, (cx, cy), (cx, cy + cl * xy), color, 3)  # vertical line
            
            # Draw label with name and confidence score
            label = f"{det.identity} ({det.similarity*100:.0f}%)" if known else "unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Draw dark background for text readability
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 12, y1), (12, 12, 22), -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            # Draw facial landmarks if available (eyes, nose, mouth, etc.)
            if det.landmarks is not None:
                for pt in det.landmarks.astype(int):
                    cv2.circle(frame, tuple(pt), 2, (0, 229, 180), -1)
        
        return frame