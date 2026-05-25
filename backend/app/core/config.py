"""
app/core/config.py
Central configuration loaded from .env
"""
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_NAME: str = "AI Traffic Surveillance System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "changeme"
    ALLOWED_HOSTS: List[str] = ["*"]
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://traffic_user:traffic_pass@localhost:5432/traffic_db"
    DATABASE_URL_SYNC: str = "postgresql://traffic_user:traffic_pass@localhost:5432/traffic_db"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Cameras
    CAMERA_URLS: str = ""
    DEMO_MODE: bool = True
    DEMO_VIDEO_PATH: str = "./data/demo_traffic.mp4"

    # AI / Models
    YOLO_MODEL_PATH: str = "./models/yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.45
    YOLO_IOU_THRESHOLD: float = 0.45
    ANPR_LANGUAGE: str = "en"
    FRAME_SKIP: int = 2
    MAX_TRACK_AGE: int = 30
    GPU_ENABLED: bool = False

    # Speed
    SPEED_LIMIT_KMH: float = 60.0
    PIXELS_PER_METER: float = 8.5

    # Watchlist
    WATCHLIST_FILE: str = "./data/watchlist.txt"

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""
    TWILIO_ALERT_NUMBER: str = ""

    # SendGrid
    SENDGRID_API_KEY: str = ""
    ALERT_EMAIL_FROM: str = ""
    ALERT_EMAIL_TO: str = ""

    # Auth / JWT
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    DEFAULT_ADMIN_EMAIL: str = "admin@traffic.local"
    DEFAULT_ADMIN_PASSWORD: str = "ChangeMe123!"

    # Streaming
    STREAM_FPS: int = 10
    JPEG_QUALITY: int = 70

    @property
    def camera_url_list(self) -> List[str]:
        if not self.CAMERA_URLS:
            return []
        return [u.strip() for u in self.CAMERA_URLS.split(",") if u.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
