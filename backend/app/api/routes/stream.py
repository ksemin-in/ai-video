"""
app/api/routes/stream.py — WebSocket live feed (camera frames + detections)
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from app.services.redis_service import get_redis

router = APIRouter()


@router.websocket("/ws/stream/{camera_id}")
async def stream_camera(websocket: WebSocket, camera_id: str):
    """
    Subscribe to live camera feed.
    Sends JSON messages with detections + base64 JPEG frame.
    Connect from browser: new WebSocket("ws://localhost:8000/ws/stream/CAM-01")
    """
    await websocket.accept()
    redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"camera:{camera_id}")
    logger.info(f"WS client subscribed to camera:{camera_id}")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())
    except WebSocketDisconnect:
        logger.info(f"WS client disconnected from {camera_id}")
    finally:
        await pubsub.unsubscribe(f"camera:{camera_id}")


@router.websocket("/ws/alerts")
async def stream_alerts(websocket: WebSocket):
    """Subscribe to real-time alert feed."""
    await websocket.accept()
    redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe("alerts")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe("alerts")
