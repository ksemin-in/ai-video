# 🚦 AI-Powered Intelligent Traffic Surveillance & Vehicle Analytics System

> **Real-time vehicle detection · Multi-object tracking · ANPR · Speed estimation · Video analytics**
> Built with Python · FastAPI · YOLOv8 · PaddleOCR · DeepSORT · PostgreSQL · Redis

---

## 📁 Project Structure

```
traffic-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── auth.py              # JWT login / register
│   │   │   ├── cameras.py           # Camera CRUD
│   │   │   ├── detections.py        # Detection history
│   │   │   ├── anpr.py              # ANPR reads & search
│   │   │   ├── alerts.py            # Alert management
│   │   │   ├── analytics.py         # Traffic stats & reports
│   │   │   ├── stream.py            # WebSocket live feed
│   │   │   ├── watchlist.py         # Watchlist management
│   │   │   └── video_analytics.py   # ★ Upload video → AI analysis
│   │   ├── core/
│   │   │   ├── config.py            # Settings from .env
│   │   │   ├── database.py          # Async SQLAlchemy
│   │   │   └── security.py          # JWT auth
│   │   ├── ml/
│   │   │   ├── detector.py          # YOLOv8 + DeepSORT
│   │   │   └── anpr.py              # PaddleOCR plate recognition
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── camera_manager.py    # RTSP feed manager
│   │   │   ├── notification_service.py  # SMS + email
│   │   │   └── __init__.py          # Redis, watchlist, alert services
│   │   └── main.py                  # FastAPI app
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── nginx/nginx.conf
├── scripts/
│   ├── setup_local.sh               # One-shot local setup
│   └── analyze_video.py             # CLI video analyzer (no server needed)
└── docker-compose.yml
```

---

## ⚡ Quick Start — Local Laptop (No Docker)

### Step 1 — Prerequisites

```bash
# Python 3.10 or 3.11
python3 --version   # must be 3.10+

# Redis (pick one):
# macOS:
brew install redis && brew services start redis
# Ubuntu / Debian:
sudo apt install redis-server && sudo systemctl start redis

# PostgreSQL (or skip and use Docker just for DB):
# macOS:
brew install postgresql && brew services start postgresql
# Ubuntu:
sudo apt install postgresql postgresql-contrib
```

### Step 2 — Clone & Setup

```bash
git clone https://github.com/yourname/traffic-ai.git
cd traffic-ai

# Run automated setup (installs packages, downloads YOLOv8, creates .env)
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
```

### Step 3 — Configure .env

```bash
nano backend/.env
```

**Required fields to fill in:**

| Key | Description | Where to get it |
|-----|-------------|-----------------|
| `SECRET_KEY` | Random string | `openssl rand -hex 32` |
| `DATABASE_URL` | PostgreSQL URL | Your DB credentials |
| `DEMO_MODE` | `true` = no real camera needed | Keep `true` to start |
| `TWILIO_*` | SMS alerts (optional) | [twilio.com/try-twilio](https://twilio.com/try-twilio) — free trial |
| `SENDGRID_API_KEY` | Email alerts (optional) | [app.sendgrid.com](https://app.sendgrid.com) — free tier |

### Step 4 — Create Database

```bash
# Create the PostgreSQL database
psql -U postgres << SQL
CREATE USER traffic_user WITH PASSWORD 'traffic_pass';
CREATE DATABASE traffic_db OWNER traffic_user;
SQL
```

### Step 5 — Run

```bash
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** — full Swagger UI with all endpoints.

---

## 🎬 Video Analytics — Analyze Any Traffic Video

### Via API (server running)

```bash
# 1. Login and get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin@traffic.local&password=ChangeMe123!" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Upload video for analysis
curl -X POST http://localhost:8000/api/v1/video-analytics/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/traffic_video.mp4" \
  -F "run_anpr=true" \
  -F "speed_limit=60"

# Returns: {"job_id": "abc12345", "poll_url": "/api/v1/video-analytics/abc12345/status"}

# 3. Poll for status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/video-analytics/abc12345/status

# 4. Download annotated video
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/video-analytics/abc12345/video \
  -o traffic_annotated.mp4

# 5. Get full JSON report
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/video-analytics/abc12345/report
```

### Via CLI (no server needed — fastest way to test)

```bash
source venv/bin/activate
cd traffic-ai

# Basic usage
python scripts/analyze_video.py --input traffic.mp4

# With options
python scripts/analyze_video.py \
  --input  traffic.mp4 \
  --output annotated_output.mp4 \
  --report analytics_report.json \
  --speed-limit 80 \
  --skip-frames 2   # process every 2nd frame (faster)

# Skip ANPR for speed
python scripts/analyze_video.py --input traffic.mp4 --no-anpr
```

**CLI Output example:**
```
Analysis Results
─────────────────────────────────
Processing time    │ 42.3s
Unique vehicles    │ 34
ANPR reads         │ 28
Avg speed          │ 52.4 km/h
Max speed          │ 94.7 km/h
Speeding events    │ 3
Watchlist hits     │ 1
─────────────────────────────────
✓ Annotated video: traffic_annotated.mp4
✓ Full JSON report: traffic_report.json
```

**What you get in the output:**
- `*_annotated.mp4` — video with coloured bounding boxes, track IDs, speed labels, ANPR overlays, trajectory tails, dashboard HUD
- `*_report.json` — complete analytics: per-vehicle data, ANPR reads, violations, per-second timeline, heatmap PNG (base64)

---

## 🐳 Docker Compose (Recommended)

```bash
# Copy env file
cp backend/.env.example backend/.env
# Edit credentials
nano backend/.env

# Build and start all services
docker compose up --build

# Services started:
#   DB        → localhost:5432
#   Redis     → localhost:6379
#   Backend   → localhost:8000
#   Nginx     → localhost:80
```

---

## ☁️ Cloud Deployment

### AWS EC2 (Ubuntu 22.04)

```bash
# 1. Launch EC2 — recommended: t3.medium (2vCPU, 4GB RAM) minimum
#    For GPU: g4dn.xlarge (NVIDIA T4)

# 2. SSH in and install Docker
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# 3. Clone your repo
git clone https://github.com/yourname/traffic-ai.git
cd traffic-ai

# 4. Configure .env
cp backend/.env.example backend/.env
nano backend/.env
# Set: APP_ENV=production, DEBUG=false, SECRET_KEY=<strong-key>

# 5. Start
docker compose up -d

# 6. Open port 80 in EC2 Security Group → your server is live
```

### Google Cloud Run (serverless)

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/traffic-backend ./backend

# Deploy
gcloud run deploy traffic-backend \
  --image gcr.io/YOUR_PROJECT/traffic-backend \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars DATABASE_URL=...,REDIS_URL=...
```

### Railway (easiest — push to deploy)

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up
# Set env vars in Railway dashboard
```

---

## 🔑 API Keys Required

| Service | Purpose | Free Tier | Sign Up |
|---------|---------|-----------|---------|
| **None** | Core AI (YOLOv8, PaddleOCR) | ✅ Fully free | Auto-downloads |
| Twilio | SMS alerts on violations | ✅ $15 trial credit | [twilio.com/try-twilio](https://www.twilio.com/try-twilio) |
| SendGrid | Email alert reports | ✅ 100 emails/day free | [sendgrid.com](https://sendgrid.com) |
| Roboflow | Custom model training (optional) | ✅ Free tier | [roboflow.com](https://roboflow.com) |

> **All AI runs locally on your machine — no external API key needed for detection, tracking, or ANPR.**

---

## 📡 API Reference (key endpoints)

```
POST   /api/v1/auth/login                    Login → get JWT token
POST   /api/v1/auth/register                 Create user

GET    /api/v1/cameras                       List active cameras
POST   /api/v1/cameras                       Add RTSP camera

POST   /api/v1/video-analytics/upload        ★ Upload video for AI analysis
GET    /api/v1/video-analytics/{id}/status   Poll job progress
GET    /api/v1/video-analytics/{id}/video    Download annotated video
GET    /api/v1/video-analytics/{id}/report   Download JSON report
GET    /api/v1/video-analytics              List all jobs

GET    /api/v1/anpr                          ANPR read history
GET    /api/v1/anpr/search/{plate}           Search a plate number
GET    /api/v1/alerts                        Alert history
PATCH  /api/v1/alerts/{id}/resolve           Resolve an alert
GET    /api/v1/watchlist                     View watchlist
POST   /api/v1/watchlist                     Add plate to watchlist

GET    /api/v1/analytics/summary             Today's traffic summary
GET    /api/v1/analytics/hourly              Hourly vehicle counts
GET    /api/v1/analytics/vehicle-types       Vehicle type breakdown
GET    /api/v1/analytics/speed-distribution  Speed band breakdown

WS     /ws/stream/{camera_id}               Live camera frames + detections
WS     /ws/alerts                           Real-time alert stream
```

---

## 🖥️ Hardware Requirements

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Demo / testing | 4GB RAM, 2-core CPU | 8GB RAM, 4-core |
| 1–2 live cameras | 8GB RAM, 4-core | 16GB RAM, 6-core |
| 4–6 live cameras | 16GB RAM + GPU | 32GB RAM + NVIDIA GPU |
| Video file analysis | 4GB RAM | 8GB RAM |

**GPU acceleration:** Set `GPU_ENABLED=true` in `.env` — requires CUDA 11.8+ and matching PyTorch.
