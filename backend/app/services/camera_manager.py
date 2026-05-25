"""
app/services/camera_manager.py
Manages multiple camera feeds.
Each camera runs in its own asyncio task:
  frame → YOLOv8 detector → DeepSORT tracker → ANPR → Redis publish → DB write
"""
from __future__ import annotations
import asyncio
import json
import time
import cv2
import numpy as np
from typing import Dict, Optional
from loguru import logger
from app.core.config import settings
from app.ml.detector import get_detector, Detection
from app.ml.anpr import get_anpr_engine
# self.watchlist = WatchlistService()
# from app.services.alert_service import AlertService
# from app.services.redis_service import get_redis


class CameraWorker:
    """Handles one RTSP stream in a background asyncio task."""

    def __init__(self, camera_id: str, source: str | int, config: dict):
        self.camera_id = camera_id
        self.source    = source
        self.config    = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.running   = False
        self._frame_no = 0
        self._last_frame_time = 0.0
        self._frame_interval  = 1.0 / settings.STREAM_FPS

    async def start(self):
        self.running = True
        logger.info(f"[{self.camera_id}] Starting stream from {self.source}")
        await asyncio.get_event_loop().run_in_executor(None, self._open_capture)
        asyncio.create_task(self._run_loop())

    async def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"[{self.camera_id}] Cannot open stream: {self.source}")

    async def _run_loop(self):
        detector    = get_detector()
        anpr_engine = get_anpr_engine()
        watchlist   = WatchlistService()
        alert_svc   = AlertService()
        redis       = await get_redis()

        loop = asyncio.get_event_loop()

        while self.running:
            now = time.time()
            if now - self._last_frame_time < self._frame_interval:
                await asyncio.sleep(0.01)
                continue

            # Read frame in thread pool (blocking I/O)
            ret, frame = await loop.run_in_executor(None, self.cap.read)
            if not ret:
                logger.warning(f"[{self.camera_id}] Stream ended — reconnecting in 3s")
                await asyncio.sleep(3)
                await loop.run_in_executor(None, self._open_capture)
                continue

            self._frame_no      += 1
            self._last_frame_time = now

            # --- AI processing in thread pool ---
            detections: list[Detection] = await loop.run_in_executor(
                None, detector.process_frame, frame, self.camera_id
            )

            enriched = []
            for det in detections:
                # ANPR on each detected vehicle
                plate_result = await loop.run_in_executor(
                    None, anpr_engine.read_plate, frame, det.bbox
                )
                if plate_result:
                    det.plate_text = plate_result.plate_text
                    det.plate_conf = plate_result.confidence

                    # Watchlist check
                    if watchlist.is_watchlisted(det.plate_text):
                        await alert_svc.create_alert(
                            camera_id=self.camera_id,
                            alert_type="WATCHLIST",
                            severity="danger",
                            message=f"Watchlisted vehicle detected: {det.plate_text}",
                            plate_number=det.plate_text,
                        )

                # Speed violation check
                speed_limit = self.config.get("speed_limit", settings.SPEED_LIMIT_KMH)
                if det.speed_kmh and det.speed_kmh > speed_limit:
                    await alert_svc.create_alert(
                        camera_id=self.camera_id,
                        alert_type="SPEEDING",
                        severity="warning",
                        message=(
                            f"Speed violation: {det.speed_kmh:.0f} km/h "
                            f"(limit {speed_limit:.0f} km/h)"
                        ),
                        plate_number=det.plate_text,
                        speed_kmh=det.speed_kmh,
                    )

                enriched.append(det)

            # --- Encode frame as JPEG for streaming ---
            _, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY],
            )
            frame_b64 = buf.tobytes().hex()   # hex string for Redis

            # --- Publish to Redis channel ---
            payload = json.dumps({
                "camera_id": self.camera_id,
                "frame_no":  self._frame_no,
                "ts":        now,
                "detections": [
                    {
                        "track_id":   d.track_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox":       d.bbox,
                        "speed_kmh":  d.speed_kmh,
                        "lane":       d.lane,
                        "plate":      d.plate_text,
                        "plate_conf": d.plate_conf,
                    }
                    for d in enriched
                ],
                "frame_hex": frame_b64,
            })
            await redis.publish(f"camera:{self.camera_id}", payload)


class CameraManager:
    """Singleton that owns all CameraWorker instances."""

    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}

    async def start_all(self):
        """Boot workers for each configured camera."""
        if settings.DEMO_MODE:
            sources = [settings.DEMO_VIDEO_PATH or 0]
            cam_ids = ["CAM-DEMO"]
        else:
            sources = settings.camera_url_list
            cam_ids = [f"CAM-{i+1:02d}" for i in range(len(sources))]

        for cam_id, src in zip(cam_ids, sources):
            worker = CameraWorker(
                camera_id=cam_id,
                source=src,
                config={"speed_limit": settings.SPEED_LIMIT_KMH},
            )
            self._workers[cam_id] = worker
            await worker.start()
        logger.info(f"Started {len(self._workers)} camera worker(s)")

    async def stop_all(self):
        for worker in self._workers.values():
            await worker.stop()

    def get_camera_ids(self) -> list[str]:
        return list(self._workers.keys())

    async def add_camera(self, cam_id: str, rtsp_url: str, config: dict):
        if cam_id in self._workers:
            await self._workers[cam_id].stop()
        worker = CameraWorker(cam_id, rtsp_url, config)
        self._workers[cam_id] = worker
        await worker.start()

    async def remove_camera(self, cam_id: str):
        if cam_id in self._workers:
            await self._workers[cam_id].stop()
            del self._workers[cam_id]


_manager: CameraManager | None = None


def get_camera_manager() -> CameraManager:
    global _manager
    if _manager is None:
        _manager = CameraManager()
    return _manager
