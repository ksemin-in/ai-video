"""
app/main.py  — FastAPI application (updated with video analytics)
"""
from contextlib import asynccontextmanager
from sys import prefix
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.database import init_db
from app.services.camera_manager import get_camera_manager

from app.api.routes import (
    auth, cameras, detections, anpr,
    alerts, analytics
)
from app.api.routes.video_analytics import router as video_analytics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== AI Traffic Surveillance System starting ===")

    # Database disabled for demo
    # await init_db()

    # Camera manager disabled for demo
    # manager = get_camera_manager()
    # await manager.start_all()

    logger.info("All systems operational")

    yield

    logger.info("Shutting down ...")

    # await manager.stop_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description=(
            "AI-powered traffic surveillance system.\n\n"
            "Features: YOLOv8 detection, DeepSORT tracking, PaddleOCR ANPR, "
            "speed estimation, video file analytics, WebSocket live feeds, "
            "SMS/email alerts."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.API_V1_PREFIX
    app.include_router(auth.router,            prefix=prefix, tags=["Auth"])
    app.include_router(cameras.router,         prefix=prefix, tags=["Cameras"])
    app.include_router(detections.router,      prefix=prefix, tags=["Detections"])
    app.include_router(anpr.router,            prefix=prefix, tags=["ANPR"])
    app.include_router(alerts.router,          prefix=prefix, tags=["Alerts"])
    app.include_router(analytics.router,       prefix=prefix, tags=["Analytics"])

    # Disabled optional modules
    # app.include_router(watchlist.router, prefix=prefix, tags=["Watchlist"])
    # app.include_router(stream.router, tags=["WebSocket Stream"])

    app.include_router(video_analytics_router, prefix=prefix, tags=["Video Analytics"])

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}

    return app


app = create_app()
