"""
core/liveness.py - Anti-spoofing / Liveness Detection
Uses a lightweight ONNX model to detect if a face is real or fake (photo/screen).
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("facevision.liveness")

class LivenessDetector:
    def __init__(self, model_path: str = "models/liveness.onnx", threshold: float = 0.6):
        """
        Initialize liveness detector.
        
        Args:
            model_path: Path to the ONNX liveness model
            threshold: Confidence threshold (0-1). Higher = stricter.
        """
        self.threshold = threshold
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the ONNX model."""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.model.get_inputs()[0].name
            logger.info(f"Liveness model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Liveness model not found at {model_path}. Liveness check disabled.")
        except Exception as e:
            logger.warning(f"Failed to load liveness model: {e}. Liveness check disabled.")

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face crop for the model."""
        img = cv2.resize(face_img, (80, 80))
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def is_live(self, frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
        """
        Check if the face in the given bounding box is real (live) or fake.

        Args:
            frame: Full BGR frame from the camera
            bbox:  (x1, y1, x2, y2) bounding box of the face

        Returns:
            (is_live: bool, confidence: float)
            is_live=True means real face, False means spoof detected
        """
        # If model failed to load, skip check and assume live
        if self.model is None:
            return True, 1.0

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Add padding around face for better accuracy
        h, w = frame.shape[:2]
        pad = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return True, 1.0

        try:
            input_tensor = self._preprocess(face_crop)
            outputs = self.model.run(None, {self.input_name: input_tensor})
            scores = outputs[0][0]  # shape: (2,) — [spoof_score, live_score]

            # Apply softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            live_score = float(probs[1])  # index 1 = live
            is_live = live_score >= self.threshold

            return is_live, live_score

        except Exception as e:
            logger.warning(f"Liveness check error: {e}")
            return True, 1.0  # Fail open — don't block on errors


_detector: Optional[LivenessDetector] = None

def get_detector(model_path: str = "models/liveness.onnx") -> LivenessDetector:
    """Get or create a singleton liveness detector."""
    global _detector
    if _detector is None:
        _detector = LivenessDetector(model_path=model_path)
    return _detector