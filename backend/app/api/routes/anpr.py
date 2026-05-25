"""
app/api/routes/anpr.py — ANPR read history
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import ANPRRead

router = APIRouter()


@router.get("/anpr")
async def list_anpr_reads(
    camera_id: str | None = Query(None),
    limit: int = Query(50, le=500),
    watchlisted_only: bool = False,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    q = select(ANPRRead).order_by(desc(ANPRRead.timestamp)).limit(limit)
    if camera_id:
        q = q.where(ANPRRead.camera_id == camera_id)
    if watchlisted_only:
        q = q.where(ANPRRead.is_watchlisted == True)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {
        "total": len(rows),
        "reads": [
            {
                "id":             r.id,
                "plate":          r.plate_number,
                "camera_id":      r.camera_id,
                "confidence":     r.confidence,
                "is_watchlisted": r.is_watchlisted,
                "state_code":     r.state_code,
                "timestamp":      r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ],
    }


@router.get("/anpr/search/{plate}")
async def search_plate(plate: str, db: AsyncSession = Depends(get_db), user=Depends(get_current_user)):
    q = select(ANPRRead).where(ANPRRead.plate_number.ilike(f"%{plate}%")).order_by(desc(ANPRRead.timestamp)).limit(100)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {"plate": plate, "count": len(rows), "history": [r.timestamp.isoformat() for r in rows if r.timestamp]}
