"""
app/api/routes/cameras.py
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.security import get_current_user
from app.services.camera_manager import get_camera_manager

router = APIRouter()


class CameraAdd(BaseModel):
    camera_id: str
    rtsp_url: str
    name: str = ""
    speed_limit: float = 60.0


@router.get("/cameras")
async def list_cameras(user=Depends(get_current_user)):
    mgr = get_camera_manager()
    return {"cameras": mgr.get_camera_ids()}


@router.post("/cameras", status_code=201)
async def add_camera(data: CameraAdd, user=Depends(get_current_user)):
    mgr = get_camera_manager()
    await mgr.add_camera(data.camera_id, data.rtsp_url, {"speed_limit": data.speed_limit})
    return {"message": f"Camera {data.camera_id} added"}


@router.delete("/cameras/{camera_id}")
async def remove_camera(camera_id: str, user=Depends(get_current_user)):
    mgr = get_camera_manager()
    await mgr.remove_camera(camera_id)
    return {"message": f"Camera {camera_id} removed"}
