"""
app/ml/detector.py
YOLOv8-based vehicle detector.
Downloads yolov8n.pt automatically on first run (ultralytics does this).
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from loguru import logger

# Lazy import — allows the rest of the app to start even without GPU libs
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — detector will run in MOCK mode")

from app.core.config import settings

# COCO classes we care about
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
}


@dataclass
class Detection:
    track_id:    int
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        Tuple[float, float, float, float]   # x1 y1 x2 y2 (pixels)
    speed_kmh:   float = 0.0
    lane:        int   = 0
    direction:   str   = "unknown"
    plate_text:  str   = ""
    plate_conf:  float = 0.0


class VehicleDetector:
    """Wraps YOLOv8 + DeepSORT tracker."""

    def __init__(self):
        self.model = None
        self.tracker = None
        self._frame_count = 0
        self._prev_positions: dict = {}   # track_id → (cx, cy, timestamp)
        self._load_model()
        self._load_tracker()

    # ── Model loading ────────────────────────────────────────────

    def _load_model(self):
        if not YOLO_AVAILABLE:
            logger.warning("YOLO unavailable — mock detections will be returned")
            return
        model_path = settings.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info(f"Model not found at {model_path} — downloading yolov8n.pt …")
            model_path = "yolov8n.pt"   # ultralytics auto-downloads
        self.model = YOLO(model_path)
        device = "cuda" if settings.GPU_ENABLED else "cpu"
        self.model.to(device)
        logger.info(f"YOLOv8 loaded on {device}")

    def _load_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=settings.MAX_TRACK_AGE,
                n_init=3,
                nms_max_overlap=0.6,
                max_cosine_distance=0.3,
            )
            logger.info("DeepSORT tracker loaded")
        except ImportError:
            logger.warning("deep-sort-realtime not installed — tracking disabled")

    # ── Frame processing ─────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, camera_id: str) -> List[Detection]:
        """
        Run detection + tracking on one frame.
        Returns a list of Detection objects.
        """
        self._frame_count += 1

        # Skip frames to reduce CPU load
        if self._frame_count % settings.FRAME_SKIP != 0:
            return []

        if self.model is None:
            return self._mock_detections(frame)

        # --- YOLOv8 inference ---
        results = self.model(
            frame,
            conf=settings.YOLO_CONFIDENCE,
            iou=settings.YOLO_IOU_THRESHOLD,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        raw_dets = []   # for DeepSORT: [[x1,y1,x2,y2], conf, class_id]
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                raw_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        # --- DeepSORT tracking ---
        detections: List[Detection] = []
        if self.tracker and raw_dets:
            tracks = self.tracker.update_tracks(raw_dets, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = ltrb
                cls_id = track.det_class if hasattr(track, "det_class") else 2
                conf   = track.det_conf  if hasattr(track, "det_conf")  else 0.5
                speed  = self._estimate_speed(tid, x1, y1, x2, y2)
                lane   = self._estimate_lane(y1, y2, frame.shape[0])
                det = Detection(
                    track_id=tid,
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES.get(cls_id, "vehicle"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    speed_kmh=speed,
                    lane=lane,
                    direction=self._estimate_direction(tid, x1, x2),
                )
                detections.append(det)
        elif raw_dets:
            # No tracker — return raw detections with fake track_ids
            for i, (bbox_xywh, conf, cls_id) in enumerate(raw_dets):
                x, y, w, h = bbox_xywh
                detections.append(Detection(
                    track_id=i,
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES.get(cls_id, "vehicle"),
                    confidence=conf,
                    bbox=(x, y, x + w, y + h),
                ))

        return detections

    # ── Speed & lane helpers ─────────────────────────────────────

    def _estimate_speed(self, tid: int, x1: float, y1: float, x2: float, y2: float) -> float:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        now = time.time()
        speed = 0.0
        if tid in self._prev_positions:
            px, py, pt = self._prev_positions[tid]
            dt = now - pt
            if dt > 0:
                dist_px = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                dist_m  = dist_px / settings.PIXELS_PER_METER
                speed   = (dist_m / dt) * 3.6   # m/s → km/h
        self._prev_positions[tid] = (cx, cy, now)
        # Clamp to realistic range
        return round(min(max(speed, 0.0), 200.0), 1)

    def _estimate_lane(self, y1: float, y2: float, frame_h: int) -> int:
        cy = (y1 + y2) / 2
        ratio = cy / frame_h
        if ratio < 0.33:
            return 1
        elif ratio < 0.66:
            return 2
        return 3

    def _estimate_direction(self, tid: int, x1: float, x2: float) -> str:
        if tid not in self._prev_positions:
            return "unknown"
        px, _, _ = self._prev_positions[tid]
        cx = (x1 + x2) / 2
        return "right" if cx > px else "left"

    # ── Mock detections (when YOLO not installed) ────────────────

    def _mock_detections(self, frame: np.ndarray) -> List[Detection]:
        import random
        n = random.randint(2, 6)
        h, w = frame.shape[:2]
        return [
            Detection(
                track_id=random.randint(1, 99),
                class_id=2,
                class_name=random.choice(["car", "truck", "bus", "motorcycle"]),
                confidence=round(random.uniform(0.60, 0.98), 2),
                bbox=(
                    random.uniform(0, w * 0.7),
                    random.uniform(0, h * 0.7),
                    random.uniform(w * 0.3, w),
                    random.uniform(h * 0.3, h),
                ),
                speed_kmh=round(random.uniform(20, 110), 1),
                lane=random.randint(1, 3),
            )
            for _ in range(n)
        ]


# Singleton — shared across the app
_detector: VehicleDetector | None = None


def get_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector
