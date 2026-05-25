"""
app/api/routes/video_analytics.py

Upload any traffic video → AI processes every frame →
Returns:
  • Annotated video with bounding boxes, track IDs, speed labels, ANPR overlays
  • Full JSON analytics report (vehicle counts, types, speeds, ANPR reads, violations)
  • Per-second timeline
  • Heatmap of vehicle positions
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import get_current_user
from app.ml.detector import get_detector, Detection
from app.ml.anpr import get_anpr_engine

router = APIRouter()

# Where processed videos are temporarily stored
OUTPUT_DIR = Path("./data/video_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job store: job_id → status dict (in-memory; replace with Redis for production)
_jobs: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class JobStatus(BaseModel):
    job_id:      str
    status:      str          # queued | processing | done | error
    progress:    float = 0.0  # 0-100
    message:     str  = ""
    result_url:  Optional[str] = None
    report_url:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Colour palette for 20 vehicle track IDs
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = [
    (0, 214, 143), (0, 200, 224), (168, 85, 247), (240, 165, 0),
    (74, 158, 255), (255, 123, 71), (255, 68, 68), (0, 255, 127),
    (255, 215, 0),  (100, 149, 237), (255, 99, 132), (54, 162, 235),
    (255, 206, 86), (75, 192, 192), (153, 102, 255), (255, 159, 64),
    (199, 199, 199),(83, 102, 255), (40, 159, 110), (210, 105, 30),
]


def track_color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# Core processing function (runs in a thread-pool executor)
# ══════════════════════════════════════════════════════════════════════════════

def process_video_sync(
    input_path: str,
    output_video_path: str,
    output_report_path: str,
    job_id: str,
    run_anpr: bool = True,
    speed_limit: float = 60.0,
) -> dict:
    """
    Synchronous video processing (called via run_in_executor).
    Reads every frame, runs YOLOv8 + ANPR, draws annotations,
    writes annotated video, returns full analytics report dict.
    """
    detector    = get_detector()
    anpr_engine = get_anpr_engine()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps

    logger.info(
        f"[{job_id}] Video: {total_frames} frames, {fps:.1f} fps, "
        f"{width}x{height}, {duration_s:.1f}s"
    )

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Analytics accumulators
    report = {
        "job_id":          job_id,
        "input_file":      Path(input_path).name,
        "duration_s":      round(duration_s, 2),
        "fps":             round(fps, 2),
        "resolution":      f"{width}x{height}",
        "total_frames":    total_frames,
        "speed_limit_kmh": speed_limit,
        # populated below
        "total_vehicles":  0,
        "unique_tracks":   0,
        "vehicle_types":   defaultdict(int),
        "anpr_reads":      [],
        "violations":      {"speeding": [], "watchlist": []},
        "avg_speed_kmh":   0.0,
        "max_speed_kmh":   0.0,
        "timeline":        [],          # per-second bucket
        "heatmap_data":    [],          # (cx, cy) list for heatmap generation
    }

    seen_tracks:  set[int]           = set()
    speed_samples: list[float]       = []
    anpr_seen:    set[str]           = set()
    per_second:   dict[int, dict]    = defaultdict(lambda: {
        "count": 0, "speeds": [], "plates": []
    })

    # Track trajectory history for drawing tails
    trajectories: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # Watchlist (loaded once)
    watchlist: set[str] = set()
    wl_path = settings.WATCHLIST_FILE
    if os.path.exists(wl_path):
        with open(wl_path) as f:
            watchlist = {ln.strip().upper() for ln in f if ln.strip()}

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        second_bucket = int((frame_no - 1) / fps)

        # Update job progress
        if frame_no % 30 == 0:
            pct = round((frame_no / max(total_frames, 1)) * 100, 1)
            _jobs[job_id]["progress"] = pct
            _jobs[job_id]["message"]  = f"Processing frame {frame_no}/{total_frames}"

        # ── AI Detection ──────────────────────────────────────────
        detections: list[Detection] = detector.process_frame(frame, "VIDEO")

        bucket = per_second[second_bucket]
        bucket["count"] += len(detections)

        for det in detections:
            seen_tracks.add(det.track_id)
            color = track_color(det.track_id)

            # Trajectory
            cx = int((det.bbox[0] + det.bbox[2]) / 2)
            cy = int((det.bbox[1] + det.bbox[3]) / 2)
            trajectories[det.track_id].append((cx, cy))
            if len(trajectories[det.track_id]) > 40:
                trajectories[det.track_id].pop(0)

            report["heatmap_data"].append([cx, cy])

            # Speed
            if det.speed_kmh:
                speed_samples.append(det.speed_kmh)
                bucket["speeds"].append(det.speed_kmh)
                report["max_speed_kmh"] = max(report["max_speed_kmh"], det.speed_kmh)
                report["vehicle_types"][det.class_name] += 1

                # Speeding violation
                if det.speed_kmh > speed_limit:
                    entry = {
                        "track_id":  det.track_id,
                        "speed_kmh": round(det.speed_kmh, 1),
                        "frame":     frame_no,
                        "time_s":    round(frame_no / fps, 2),
                        "plate":     det.plate_text,
                    }
                    if entry not in report["violations"]["speeding"][-10:]:
                        report["violations"]["speeding"].append(entry)

            # ── ANPR ─────────────────────────────────────────────
            plate_text = ""
            plate_conf = 0.0
            if run_anpr:
                pr = anpr_engine.read_plate(frame, det.bbox)
                if pr and pr.plate_text:
                    plate_text = pr.plate_text
                    plate_conf = pr.confidence
                    det.plate_text = plate_text
                    det.plate_conf = plate_conf

                    if plate_text not in anpr_seen:
                        anpr_seen.add(plate_text)
                        report["anpr_reads"].append({
                            "plate":      plate_text,
                            "confidence": round(plate_conf, 3),
                            "frame":      frame_no,
                            "time_s":     round(frame_no / fps, 2),
                            "watchlisted": plate_text in watchlist,
                        })
                        bucket["plates"].append(plate_text)

                    if plate_text in watchlist:
                        report["violations"]["watchlist"].append({
                            "plate":  plate_text,
                            "frame":  frame_no,
                            "time_s": round(frame_no / fps, 2),
                        })

            # ── Draw annotations on frame ─────────────────────────
            _draw_detection(frame, det, color, plate_text, plate_conf, speed_limit)

        # Draw trajectories
        for tid, pts in trajectories.items():
            col = track_color(tid)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in col)
                cv2.line(frame, pts[i-1], pts[i], c, 1, cv2.LINE_AA)

        # Dashboard overlay on frame
        _draw_dashboard_overlay(frame, frame_no, fps, len(seen_tracks), len(report["anpr_reads"]))

        writer.write(frame)

    cap.release()
    writer.release()

    # ── Finalise report ───────────────────────────────────────────
    report["total_vehicles"]  = frame_no  # approximate (sums over frames; use unique below)
    report["unique_tracks"]   = len(seen_tracks)
    report["avg_speed_kmh"]   = round(sum(speed_samples) / max(len(speed_samples), 1), 1)
    report["vehicle_types"]   = dict(report["vehicle_types"])
    report["processed_frames"] = frame_no

    # Per-second timeline
    report["timeline"] = [
        {
            "second":     s,
            "count":      d["count"],
            "avg_speed":  round(sum(d["speeds"]) / max(len(d["speeds"]), 1), 1),
            "plates":     d["plates"],
        }
        for s, d in sorted(per_second.items())
    ]

    # Heatmap PNG embedded as base64
    report["heatmap_b64"] = _generate_heatmap_b64(report["heatmap_data"], width, height)

    # Save report JSON
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"[{job_id}] Done — {len(seen_tracks)} unique vehicles, "
        f"{len(report['anpr_reads'])} plates read"
    )
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_detection(
    frame: np.ndarray,
    det: Detection,
    color: tuple,
    plate_text: str,
    plate_conf: float,
    speed_limit: float,
):
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    spd = det.speed_kmh or 0

    # Bounding box — thicker + brighter for speeding
    thickness = 2 if spd <= speed_limit else 3
    box_color = color if spd <= speed_limit else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    # Corner accents
    L = 12
    for (px, py), (dx, dy) in [
        ((x1, y1), (L, L)), ((x2, y1), (-L, L)),
        ((x1, y2), (L, -L)), ((x2, y2), (-L, -L))
    ]:
        cv2.line(frame, (px, py), (px + dx, py), box_color, 2)
        cv2.line(frame, (px, py), (px, py + dy), box_color, 2)

    # Label background
    label_parts = [f"#{det.track_id} {det.class_name}"]
    if spd > 0:
        spd_tag = f"{spd:.0f}km/h"
        if spd > speed_limit:
            spd_tag += " !"
        label_parts.append(spd_tag)
    if plate_text:
        label_parts.append(plate_text)

    label     = "  ".join(label_parts)
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness_txt = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_txt)

    lx, ly = x1, max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 6, ly + 2),
                  box_color, cv2.FILLED)
    cv2.putText(frame, label, (lx + 3, ly - 2), font, font_scale,
                (0, 0, 0), thickness_txt, cv2.LINE_AA)

    # Confidence bar under bbox
    bar_w = x2 - x1
    filled = int(bar_w * det.confidence)
    cv2.rectangle(frame, (x1, y2 + 1), (x2, y2 + 4), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(frame, (x1, y2 + 1), (x1 + filled, y2 + 4), color, cv2.FILLED)

    # ANPR plate box (bottom of vehicle)
    if plate_text:
        pt_x, pt_y = x1, y2 + 8
        (pw, ph), _ = cv2.getTextSize(plate_text, font, 0.5, 1)
        cv2.rectangle(frame, (pt_x, pt_y), (pt_x + pw + 6, pt_y + ph + 4),
                      (30, 100, 30), cv2.FILLED)
        cv2.rectangle(frame, (pt_x, pt_y), (pt_x + pw + 6, pt_y + ph + 4),
                      (0, 220, 100), 1)
        cv2.putText(frame, plate_text, (pt_x + 3, pt_y + ph + 1),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_dashboard_overlay(
    frame: np.ndarray,
    frame_no: int,
    fps: float,
    unique_vehicles: int,
    anpr_count: int,
):
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (10, 20, 35), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ts   = time.strftime("%H:%M:%S")
    sec  = frame_no / fps

    texts = [
        f"AI TRAFFIC ANALYTICS",
        f"Frame: {frame_no}",
        f"Time: {sec:.1f}s",
        f"Vehicles: {unique_vehicles}",
        f"ANPR: {anpr_count}",
        ts,
    ]
    x = 8
    for i, t in enumerate(texts):
        color = (0, 214, 143) if i == 0 else (180, 220, 200)
        cv2.putText(frame, t, (x, 18), font, 0.42, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(t, font, 0.42, 1)
        x += tw + 18


def _generate_heatmap_b64(points: list, width: int, height: int) -> str:
    """Generate a vehicle density heatmap and return as base64 PNG."""
    try:
        hm = np.zeros((height, width), dtype=np.float32)
        for cx, cy in points:
            cx, cy = int(cx), int(cy)
            if 0 <= cy < height and 0 <= cx < width:
                hm[cy, cx] += 1.0

        # Gaussian blur to spread heat
        hm = cv2.GaussianBlur(hm, (51, 51), 0)

        # Normalise and apply colormap
        if hm.max() > 0:
            hm = (hm / hm.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # Resize for thumbnail
        thumb = cv2.resize(colored, (320, int(height * 320 / width)))
        _, buf = cv2.imencode(".png", thumb)
        return base64.b64encode(buf).decode()
    except Exception as e:
        logger.warning(f"Heatmap generation failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Background job runner
# ══════════════════════════════════════════════════════════════════════════════

async def _run_job(job_id: str, input_path: str, run_anpr: bool, speed_limit: float):
    out_video  = str(OUTPUT_DIR / f"{job_id}_annotated.mp4")
    out_report = str(OUTPUT_DIR / f"{job_id}_report.json")
    _jobs[job_id]["status"]  = "processing"
    _jobs[job_id]["message"] = "Starting AI pipeline …"
    try:
        loop   = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None,
            process_video_sync,
            input_path, out_video, out_report, job_id, run_anpr, speed_limit,
        )
        _jobs[job_id]["status"]     = "done"
        _jobs[job_id]["progress"]   = 100.0
        _jobs[job_id]["message"]    = "Processing complete"
        _jobs[job_id]["result_url"] = f"/api/v1/video-analytics/{job_id}/video"
        _jobs[job_id]["report_url"] = f"/api/v1/video-analytics/{job_id}/report"
        _jobs[job_id]["summary"]    = {
            "unique_vehicles":  report["unique_tracks"],
            "anpr_reads":       len(report["anpr_reads"]),
            "avg_speed_kmh":    report["avg_speed_kmh"],
            "max_speed_kmh":    report["max_speed_kmh"],
            "speeding_events":  len(report["violations"]["speeding"]),
            "watchlist_hits":   len(report["violations"]["watchlist"]),
            "duration_s":       report["duration_s"],
        }
    except Exception as e:
        logger.exception(f"[{job_id}] Processing failed: {e}")
        _jobs[job_id]["status"]  = "error"
        _jobs[job_id]["message"] = str(e)
    finally:
        # Clean up temp input
        try:
            os.unlink(input_path)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/video-analytics/upload",
    summary="Upload a video for AI analysis",
    description=(
        "Upload any MP4/AVI/MOV traffic video. "
        "Returns a job_id — poll /status for progress, "
        "then download the annotated video and JSON report."
    ),
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file:        UploadFile  = File(..., description="Traffic video file (MP4/AVI/MOV)"),
    run_anpr:    bool        = Form(True,  description="Run ANPR plate recognition"),
    speed_limit: float       = Form(60.0, description="Speed limit in km/h for violation detection"),
    user=Depends(get_current_user),
):
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}
    suffix  = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    # Save upload to temp file
    job_id    = str(uuid.uuid4())[:8]
    tmp_dir   = Path(tempfile.gettempdir())
    tmp_path  = str(tmp_dir / f"{job_id}_input{suffix}")

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    file_size_mb = len(content) / 1_048_576
    logger.info(f"[{job_id}] Received {file.filename} ({file_size_mb:.1f} MB)")

    # Register job
    _jobs[job_id] = {
        "job_id":      job_id,
        "status":      "queued",
        "progress":    0.0,
        "message":     "Job queued",
        "filename":    file.filename,
        "size_mb":     round(file_size_mb, 2),
        "run_anpr":    run_anpr,
        "speed_limit": speed_limit,
        "result_url":  None,
        "report_url":  None,
    }

    # Launch background processing
    background_tasks.add_task(_run_job, job_id, tmp_path, run_anpr, speed_limit)

    return {
        "job_id":   job_id,
        "message":  "Video uploaded and queued for processing",
        "poll_url": f"/api/v1/video-analytics/{job_id}/status",
    }


@router.get(
    "/video-analytics/{job_id}/status",
    response_model=JobStatus,
    summary="Poll processing status",
)
async def job_status(job_id: str, user=Depends(get_current_user)):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    j = _jobs[job_id]
    return JobStatus(
        job_id=j["job_id"],
        status=j["status"],
        progress=j.get("progress", 0.0),
        message=j.get("message", ""),
        result_url=j.get("result_url"),
        report_url=j.get("report_url"),
    )


@router.get(
    "/video-analytics/{job_id}/report",
    summary="Download full JSON analytics report",
)
async def download_report(job_id: str, user=Depends(get_current_user)):
    if job_id not in _jobs or _jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Report not ready")
    path = OUTPUT_DIR / f"{job_id}_report.json"
    if not path.exists():
        raise HTTPException(404, "Report file missing")
    with open(path) as f:
        return JSONResponse(content=json.load(f))


@router.get(
    "/video-analytics/{job_id}/video",
    summary="Download annotated video",
)
async def download_video(job_id: str, user=Depends(get_current_user)):
    if job_id not in _jobs or _jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Video not ready")
    path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=f"traffic_annotated_{job_id}.mp4",
    )


@router.get(
    "/video-analytics/{job_id}/summary",
    summary="Get quick summary stats (no auth needed for demo)",
)
async def job_summary(job_id: str, user=Depends(get_current_user)):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    j = _jobs[job_id]
    return {
        "job_id":   job_id,
        "status":   j["status"],
        "filename": j.get("filename"),
        "summary":  j.get("summary", {}),
    }


@router.get("/video-analytics", summary="List all video analysis jobs")
async def list_jobs(user=Depends(get_current_user)):
    return {
        "jobs": [
            {
                "job_id":   jid,
                "status":   j["status"],
                "filename": j.get("filename"),
                "progress": j.get("progress", 0),
                "size_mb":  j.get("size_mb"),
            }
            for jid, j in _jobs.items()
        ]
    }
