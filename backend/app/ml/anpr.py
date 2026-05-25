"""
app/ml/anpr.py
Automatic Number Plate Recognition using PaddleOCR.
Two-stage pipeline:
  1. Plate localisation  — custom YOLO model OR contour-based fallback
  2. Text recognition    — PaddleOCR
"""
from __future__ import annotations
import re
import base64
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger
from app.core.config import settings

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not installed — ANPR will run in MOCK mode")

# Regex patterns for Indian number plates  (e.g. MH12AB1234)
INDIAN_PLATE_RE = re.compile(
    r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$",
    re.IGNORECASE,
)

# Common OCR noise fixes
OCR_FIXES = {
    "0": "O", "O": "0",   # handled contextually below
    "I": "1", "l": "1",
    "S": "5", "5": "S",
}


@dataclass
class PlateResult:
    plate_text:   str
    confidence:   float
    plate_image:  Optional[np.ndarray] = None   # cropped plate region
    state_code:   str = ""
    is_valid:     bool = False


class ANPREngine:
    """Plate localisation + OCR."""

    def __init__(self):
        self.ocr = None
        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ANPR_LANGUAGE,
                show_log=False,
                use_gpu=settings.GPU_ENABLED,
            )
            logger.info("PaddleOCR initialised")

    # ── Public API ───────────────────────────────────────────────

    def read_plate(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[PlateResult]:
        """
        Given a full frame and the vehicle bounding box (x1,y1,x2,y2),
        locate the plate region and OCR it.
        Returns PlateResult or None if no plate found.
        """
        vehicle_crop = self._crop(frame, bbox, pad=10)
        if vehicle_crop is None:
            return None

        plate_crop = self._localise_plate(vehicle_crop)
        if plate_crop is None:
            return None

        text, conf = self._ocr_crop(plate_crop)
        if not text:
            return None

        text = self._clean_text(text)
        return PlateResult(
            plate_text=text,
            confidence=conf,
            plate_image=plate_crop,
            state_code=text[:2].upper() if len(text) >= 2 else "",
            is_valid=bool(INDIAN_PLATE_RE.match(text)),
        )

    def plate_to_b64(self, plate_img: np.ndarray) -> str:
        """Encode plate crop to base64 JPEG for DB/WS storage."""
        _, buf = cv2.imencode(".jpg", plate_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()

    # ── Plate localisation ───────────────────────────────────────

    def _localise_plate(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the licence plate rectangle inside the vehicle crop.
        Uses Canny + contour analysis as a lightweight fallback
        (replace with a dedicated LP detector YOLO for production).
        """
        h, w = vehicle_crop.shape[:2]
        # Focus on lower 60% of the vehicle crop (where plates usually are)
        roi = vehicle_crop[int(h * 0.4):, :]

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = cw / max(ch, 1)
            area  = cw * ch
            # Plate aspect ratio roughly 2:1 to 5:1
            if 2.0 <= ratio <= 5.5 and 1500 <= area <= int(w * h * 0.25):
                candidates.append((x, y, cw, ch, area))

        if not candidates:
            # fallback: use bottom-centre strip of vehicle crop
            strip = vehicle_crop[int(h * 0.6):int(h * 0.9), int(w * 0.2):int(w * 0.8)]
            return strip if strip.size > 0 else None

        # Pick largest candidate
        candidates.sort(key=lambda c: c[4], reverse=True)
        x, y, cw, ch, _ = candidates[0]
        y_offset = int(h * 0.4)
        crop = vehicle_crop[y_offset + y: y_offset + y + ch, x: x + cw]
        return crop if crop.size > 0 else None

    # ── OCR ──────────────────────────────────────────────────────

    def _ocr_crop(self, img: np.ndarray) -> Tuple[str, float]:
        """Run PaddleOCR on a plate crop. Returns (text, confidence)."""
        if self.ocr is None:
            return self._mock_ocr()

        # Upscale tiny crops for better OCR accuracy
        if img.shape[1] < 200:
            scale = 200 / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        results = self.ocr.ocr(img, cls=True)
        if not results or not results[0]:
            return "", 0.0

        texts, confs = [], []
        for line in results[0]:
            if line and len(line) >= 2:
                text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                    texts.append(str(text_conf[0]))
                    confs.append(float(text_conf[1]))

        if not texts:
            return "", 0.0

        combined = "".join(texts).replace(" ", "").upper()
        avg_conf  = sum(confs) / len(confs)
        return combined, round(avg_conf, 3)

    def _clean_text(self, text: str) -> str:
        """Normalise OCR output to match Indian plate format."""
        text = text.upper().replace(" ", "").replace("-", "").replace(".", "")
        # Simple contextual fix: digits in positions 2-3 and 6-9, letters elsewhere
        cleaned = []
        for i, ch in enumerate(text):
            if i < 2 or (4 <= i <= 5):       # state code / letter sections
                cleaned.append(ch if ch.isalpha() else OCR_FIXES.get(ch, ch))
            elif 2 <= i <= 3 or i >= 6:       # digit sections
                cleaned.append(ch if ch.isdigit() else OCR_FIXES.get(ch, ch))
            else:
                cleaned.append(ch)
        return "".join(cleaned)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _crop(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        pad: int = 0,
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _mock_ocr(self) -> Tuple[str, float]:
        import random
        states = ["MH", "KA", "DL", "GJ", "UP", "TN", "RJ"]
        st  = random.choice(states)
        num = random.randint(10, 99)
        lt  = "".join(random.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=2))
        dn  = random.randint(1000, 9999)
        return f"{st}{num:02d}{lt}{dn}", round(random.uniform(0.82, 0.98), 3)


# Singleton
_engine: ANPREngine | None = None


def get_anpr_engine() -> ANPREngine:
    global _engine
    if _engine is None:
        _engine = ANPREngine()
    return _engine
