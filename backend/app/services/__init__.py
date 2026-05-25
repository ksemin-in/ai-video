"""
app/services/redis_service.py
app/services/alert_service.py
app/services/watchlist_service.py
"""

# ── redis_service.py ─────────────────────────────────────────────────────────
import redis.asyncio as aioredis
from app.core.config import settings
from loguru import logger

_redis_client = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
        )
        logger.info("Redis connected")
    return _redis_client


# ── watchlist_service.py ─────────────────────────────────────────────────────
import os


class WatchlistService:
    """Loads plate watchlist from a flat text file. Reloads on every call (small file)."""

    def __init__(self, path: str | None = None):
        self.path = path or settings.WATCHLIST_FILE

    def load(self) -> set[str]:
        if not os.path.exists(self.path):
            return set()
        with open(self.path) as f:
            return {ln.strip().upper() for ln in f if ln.strip()}

    def is_watchlisted(self, plate: str) -> bool:
        return plate.upper() in self.load()

    def add(self, plate: str):
        plates = self.load()
        plates.add(plate.upper())
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            f.write("\n".join(sorted(plates)))

    def remove(self, plate: str):
        plates = self.load()
        plates.discard(plate.upper())
        with open(self.path, "w") as f:
            f.write("\n".join(sorted(plates)))


# ── alert_service.py ─────────────────────────────────────────────────────────
import json
from typing import Optional
from app.services.notification_service import NotificationService


class AlertService:
    """Creates alerts: writes to Redis pub/sub AND triggers notifications."""

    def __init__(self):
        self.notifier = NotificationService()

    async def create_alert(
        self,
        camera_id: str,
        alert_type: str,
        severity: str,
        message: str,
        plate_number: Optional[str] = None,
        speed_kmh: Optional[float] = None,
    ):
        import time
        redis = await get_redis()
        payload = json.dumps({
            "type":         alert_type,
            "severity":     severity,
            "camera_id":    camera_id,
            "message":      message,
            "plate_number": plate_number,
            "speed_kmh":    speed_kmh,
            "ts":           time.time(),
        })
        await redis.publish("alerts", payload)
        logger.info(f"[ALERT] {severity.upper()} | {camera_id} | {message}")

        # Send SMS + email for danger-level alerts
        if severity == "danger":
            await self.notifier.send_sms(f"[ALERT] {message}")
            await self.notifier.send_email(
                subject=f"Traffic Alert — {alert_type}",
                body=message,
            )
