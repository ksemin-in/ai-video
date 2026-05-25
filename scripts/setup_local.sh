#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
#  AI Traffic Surveillance System — Local Setup Script
#  Run:  chmod +x scripts/setup_local.sh && ./scripts/setup_local.sh
# ════════════════════════════════════════════════════════════════
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  AI Traffic Surveillance System — Local Setup   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── 0. Prerequisites check ────────────────────────────────────────
info "Checking prerequisites …"
command -v python3 >/dev/null || error "Python 3.9+ required. Install from https://python.org"
command -v pip3    >/dev/null || error "pip3 required"
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER detected"

# ── 1. Virtual environment ─────────────────────────────────────────
cd "$(dirname "$0")/.."
info "Creating virtual environment …"
python3 -m venv venv
source venv/bin/activate
success "venv activated"

# ── 2. Install Python dependencies ────────────────────────────────
info "Installing Python packages (this may take 5–10 minutes) …"
pip install --upgrade pip setuptools wheel -q
pip install -r backend/requirements.txt -q
success "Python packages installed"

# ── 3. Download YOLOv8 model ──────────────────────────────────────
info "Downloading YOLOv8n model …"
mkdir -p backend/models
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")        # auto-downloads ~6 MB
import shutil, os
shutil.copy("yolov8n.pt", "backend/models/yolov8n.pt")
if os.path.exists("yolov8n.pt"):
    os.remove("yolov8n.pt")
print("  Model saved to backend/models/yolov8n.pt")
PY
success "YOLOv8n ready"

# ── 4. Copy .env ───────────────────────────────────────────────────
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    warn "Created backend/.env from template — edit it before running!"
else
    info "backend/.env already exists — skipping"
fi

# ── 5. Create data directories ─────────────────────────────────────
mkdir -p backend/data/video_outputs
touch backend/data/watchlist.txt
success "Data directories ready"

# ── 6. Check Redis ────────────────────────────────────────────────
info "Checking Redis …"
if command -v redis-cli >/dev/null && redis-cli ping >/dev/null 2>&1; then
    success "Redis is running"
else
    warn "Redis not running. Start it:"
    warn "  macOS:  brew install redis && brew services start redis"
    warn "  Ubuntu: sudo apt install redis-server && sudo systemctl start redis"
    warn "  OR use Docker: docker run -d -p 6379:6379 redis:7-alpine"
fi

# ── 7. Check PostgreSQL ───────────────────────────────────────────
info "Checking PostgreSQL …"
if command -v psql >/dev/null; then
    success "PostgreSQL client found"
    warn "Make sure a DB named 'traffic_db' exists with user 'traffic_user'"
    warn "  psql -U postgres -c \"CREATE USER traffic_user WITH PASSWORD 'traffic_pass';\""
    warn "  psql -U postgres -c \"CREATE DATABASE traffic_db OWNER traffic_user;\""
else
    warn "psql not found. Use Docker DB or install PostgreSQL:"
    warn "  docker run -d -p 5432:5432 -e POSTGRES_USER=traffic_user \\"
    warn "    -e POSTGRES_PASSWORD=traffic_pass -e POSTGRES_DB=traffic_db \\"
    warn "    timescale/timescaledb:latest-pg15"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!  Next steps:${NC}"
echo ""
echo -e "  1. Edit  ${YELLOW}backend/.env${NC}  (DB creds, API keys, camera URLs)"
echo ""
echo -e "  2. Start the server:"
echo -e "     ${CYAN}source venv/bin/activate${NC}"
echo -e "     ${CYAN}cd backend && uvicorn app.main:app --reload${NC}"
echo ""
echo -e "  3. Open API docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo -e "  4. Upload a video for analysis:"
echo -e "     ${CYAN}POST http://localhost:8000/api/v1/video-analytics/upload${NC}"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
