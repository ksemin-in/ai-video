"""
app/api/routes/analytics.py — Traffic analytics & statistics
"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import VehicleDetection, ANPRRead, Alert

router = APIRouter()


@router.get("/analytics/summary")
async def summary(db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    """Overall traffic summary for today."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    total_vehicles = await db.scalar(
        select(func.count()).select_from(VehicleDetection).where(VehicleDetection.timestamp >= today)
    )
    total_anpr = await db.scalar(
        select(func.count()).select_from(ANPRRead).where(ANPRRead.timestamp >= today)
    )
    total_alerts = await db.scalar(
        select(func.count()).select_from(Alert).where(Alert.timestamp >= today)
    )
    avg_speed = await db.scalar(
        select(func.avg(VehicleDetection.speed_kmh)).where(
            VehicleDetection.timestamp >= today,
            VehicleDetection.speed_kmh.isnot(None),
        )
    )
    watchlist_hits = await db.scalar(
        select(func.count()).select_from(ANPRRead).where(
            ANPRRead.timestamp >= today,
            ANPRRead.is_watchlisted == True,
        )
    )
    return {
        "date":            today.date().isoformat(),
        "total_vehicles":  total_vehicles or 0,
        "total_anpr_reads": total_anpr or 0,
        "total_alerts":    total_alerts or 0,
        "avg_speed_kmh":   round(float(avg_speed or 0), 1),
        "watchlist_hits":  watchlist_hits or 0,
    }


@router.get("/analytics/hourly")
async def hourly_volume(
    hours: int = Query(24, le=168),
    camera_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Vehicle count grouped by hour for the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = select(
        func.date_trunc("hour", VehicleDetection.timestamp).label("hour"),
        func.count().label("count"),
    ).where(VehicleDetection.timestamp >= since)
    if camera_id:
        q = q.where(VehicleDetection.camera_id == camera_id)
    q = q.group_by("hour").order_by("hour")
    result = await db.execute(q)
    rows = result.all()
    return {
        "hours": hours,
        "data": [{"hour": str(r.hour), "count": r.count} for r in rows],
    }


@router.get("/analytics/vehicle-types")
async def vehicle_types(db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    q = select(
        VehicleDetection.vehicle_type,
        func.count().label("count"),
    ).where(VehicleDetection.timestamp >= today).group_by(VehicleDetection.vehicle_type)
    result = await db.execute(q)
    rows = result.all()
    return {"types": [{"type": r.vehicle_type, "count": r.count} for r in rows]}


@router.get("/analytics/speed-distribution")
async def speed_distribution(db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    bands = [(0, 30), (30, 60), (60, 80), (80, 100), (100, 999)]
    result = []
    for lo, hi in bands:
        count = await db.scalar(
            select(func.count()).select_from(VehicleDetection).where(
                VehicleDetection.timestamp >= today,
                VehicleDetection.speed_kmh >= lo,
                VehicleDetection.speed_kmh < hi,
            )
        )
        result.append({"range": f"{lo}-{hi}", "count": count or 0})
    return {"bands": result}
