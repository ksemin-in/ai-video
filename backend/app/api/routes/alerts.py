"""
app/api/routes/alerts.py
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Alert

router = APIRouter()


@router.get("/alerts")
async def list_alerts(
    limit: int = Query(50, le=500),
    severity: str | None = None,
    resolved: bool = False,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    q = select(Alert).order_by(desc(Alert.timestamp)).limit(limit)
    if severity:
        q = q.where(Alert.severity == severity)
    if not resolved:
        q = q.where(Alert.resolved == False)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {
        "total": len(rows),
        "alerts": [
            {
                "id":           r.id,
                "type":         r.alert_type,
                "severity":     r.severity,
                "camera_id":    r.camera_id,
                "message":      r.message,
                "plate_number": r.plate_number,
                "speed_kmh":    r.speed_kmh,
                "resolved":     r.resolved,
                "timestamp":    r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ],
    }


@router.patch("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    await db.execute(update(Alert).where(Alert.id == alert_id).values(resolved=True))
    return {"message": "Alert resolved"}
