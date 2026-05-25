"""
app/api/routes/watchlist.py
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.security import get_current_user
from app.services import WatchlistService

router = APIRouter()


class PlateBody(BaseModel):
    plate: str


@router.get("/watchlist")
async def get_watchlist(user=Depends(get_current_user)):
    svc = WatchlistService()
    return {"plates": sorted(svc.load())}


@router.post("/watchlist", status_code=201)
async def add_plate(body: PlateBody, user=Depends(get_current_user)):
    WatchlistService().add(body.plate)
    return {"message": f"{body.plate} added to watchlist"}


@router.delete("/watchlist/{plate}")
async def remove_plate(plate: str, user=Depends(get_current_user)):
    WatchlistService().remove(plate)
    return {"message": f"{plate} removed from watchlist"}


"""
app/api/routes/detections.py
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import VehicleDetection

router = APIRouter()


@router.get("/detections")
async def list_detections(
    camera_id: str | None = Query(None),
    limit: int = Query(100, le=1000),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    q = select(VehicleDetection).order_by(desc(VehicleDetection.timestamp)).limit(limit)
    if camera_id:
        q = q.where(VehicleDetection.camera_id == camera_id)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {
        "total": len(rows),
        "detections": [
            {
                "id":           r.id,
                "camera_id":    r.camera_id,
                "track_id":     r.track_id,
                "vehicle_type": r.vehicle_type,
                "confidence":   r.confidence,
                "speed_kmh":    r.speed_kmh,
                "lane":         r.lane,
                "timestamp":    r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ],
    }
