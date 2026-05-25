"""
app/models/vehicle.py  — Vehicle detection & tracking records
app/models/anpr.py     — ANPR plate reads
app/models/alert.py    — System alerts
app/models/user.py     — Dashboard users
"""

# ── vehicle.py ───────────────────────────────────────────────────────────────
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id            = Column(Integer, primary_key=True, index=True)
    camera_id     = Column(String(32), nullable=False, index=True)
    track_id      = Column(Integer, nullable=False, index=True)
    vehicle_type  = Column(String(32))           # car, truck, bus, motorcycle, ...
    confidence    = Column(Float)
    bbox_x        = Column(Float)
    bbox_y        = Column(Float)
    bbox_w        = Column(Float)
    bbox_h        = Column(Float)
    speed_kmh     = Column(Float, nullable=True)
    direction     = Column(String(16), nullable=True)  # north/south/east/west
    lane          = Column(Integer, nullable=True)
    frame_number  = Column(Integer)
    timestamp     = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    extra         = Column(JSON, default={})

    anpr_reads    = relationship("ANPRRead", back_populates="detection")


class ANPRRead(Base):
    __tablename__ = "anpr_reads"

    id             = Column(Integer, primary_key=True, index=True)
    detection_id   = Column(Integer, ForeignKey("vehicle_detections.id"), nullable=True)
    camera_id      = Column(String(32), nullable=False, index=True)
    plate_number   = Column(String(20), nullable=False, index=True)
    confidence     = Column(Float)
    is_watchlisted = Column(Boolean, default=False, index=True)
    state_code     = Column(String(4), nullable=True)
    vehicle_class  = Column(String(32), nullable=True)
    plate_image_b64= Column(String, nullable=True)  # base64 cropped plate image
    timestamp      = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    detection      = relationship("VehicleDetection", back_populates="anpr_reads")
    alert          = relationship("Alert", back_populates="anpr_read", uselist=False)


class Alert(Base):
    __tablename__ = "alerts"

    id           = Column(Integer, primary_key=True, index=True)
    anpr_read_id = Column(Integer, ForeignKey("anpr_reads.id"), nullable=True)
    camera_id    = Column(String(32), nullable=False)
    alert_type   = Column(String(32))   # WATCHLIST | SPEEDING | WRONG_LANE | CONGESTION
    severity     = Column(String(16))   # info | warning | danger
    message      = Column(String(512))
    plate_number = Column(String(20), nullable=True)
    speed_kmh    = Column(Float, nullable=True)
    resolved     = Column(Boolean, default=False)
    notified     = Column(Boolean, default=False)
    timestamp    = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    anpr_read    = relationship("ANPRRead", back_populates="alert")


class User(Base):
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String(255), unique=True, nullable=False, index=True)
    full_name    = Column(String(128))
    hashed_pw    = Column(String(255), nullable=False)
    role         = Column(String(16), default="viewer")  # admin | operator | viewer
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())


class CameraConfig(Base):
    __tablename__ = "camera_configs"

    id           = Column(Integer, primary_key=True, index=True)
    camera_id    = Column(String(32), unique=True, nullable=False)
    name         = Column(String(128))
    location     = Column(String(256))
    rtsp_url     = Column(String(512))
    latitude     = Column(Float, nullable=True)
    longitude    = Column(Float, nullable=True)
    speed_limit  = Column(Float, default=60.0)
    is_active    = Column(Boolean, default=True)
    pixels_per_m = Column(Float, default=8.5)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
