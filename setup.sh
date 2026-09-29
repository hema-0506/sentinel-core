#!/usr/bin/env bash
# =============================================================
# defense setup script — CPU-only (i3 / no NVIDIA toolkit)
# Run this once after cloning the repo.
# =============================================================
set -e

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

echo -e "${BOLD}defense Defense Lab — Setup${RESET}"
echo "====================================="

# ── 1. Check Docker ──────────────────────────────────────────
command -v docker  >/dev/null || error "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v docker  >/dev/null && docker compose version >/dev/null 2>&1 || \
  command -v docker-compose >/dev/null || error "Docker Compose not found."
info "Docker found: $(docker --version)"

# ── 2. Model placement ───────────────────────────────────────
MODEL_DIR="./python_service/models"
MODEL_FILE="$MODEL_DIR/defense_final.pth"
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  info "Model already present at $MODEL_FILE"
else
  warn "Model not found at $MODEL_FILE"
  echo ""
  echo -e "${BOLD}How to get your model from Kaggle:${RESET}"
  echo ""
  echo "  Option A — Kaggle CLI (recommended):"
  echo "    1. pip install kaggle"
  echo "    2. Get your API token from https://www.kaggle.com/settings → API → Create New Token"
  echo "    3. Place kaggle.json in ~/.kaggle/"
  echo "    4. Run:"
  echo "         kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_NAME -p $MODEL_DIR"
  echo "    5. Rename the downloaded file:"
  echo "         mv $MODEL_DIR/defense_ep100.pth $MODEL_FILE"
  echo "         # (or whatever your final checkpoint is named)"
  echo ""
  echo "  Option B — Manual download:"
  echo "    1. Open your Kaggle notebook → Output tab"
  echo "    2. Find defense_ep100.pth (or defense_final.pth)"
  echo "    3. Download it and place it at:"
  echo "         $MODEL_FILE"
  echo ""
  echo "  Option C — Run without model (demo mode, untrained weights):"
  echo "    The service starts fine without a checkpoint."
  echo "    Stego quality will be random — only for testing the plumbing."
  echo ""
  read -p "Continue setup without model? [y/N] " yn
  case "$yn" in
    [Yy]) warn "Continuing without model — demo mode" ;;
    *)    info "Re-run this script after placing your model file."; exit 0 ;;
  esac
fi

# ── 3. Which training variant? ───────────────────────────────
echo ""
echo -e "${BOLD}Which training code did you use on Kaggle?${RESET}"
echo "  1) Aligned (affine ACB + permutation block) — most recent version"
echo "  2) Original (additive ACB, first version)"
read -p "Enter 1 or 2 [default: 1]: " variant_choice
case "$variant_choice" in
  2) VARIANT="original" ;;
  *) VARIANT="aligned"  ;;
esac
info "Using MODEL_VARIANT=$VARIANT"

# Patch docker-compose with chosen variant
if command -v sed >/dev/null; then
  sed -i "s/MODEL_VARIANT: .*/MODEL_VARIANT: $VARIANT/" docker-compose.yml
  info "Updated docker-compose.yml with MODEL_VARIANT=$VARIANT"
fi

# ── 4. Secrets warning ───────────────────────────────────────
echo ""
warn "SECURITY: Before production use, change these in docker-compose.yml:"
echo "  JWT_SECRET           (currently: 'change-me-to-32-random-chars-now')"
echo "  KEY_VAULT_MASTER_KEY (currently: '00112233445566778899aabbccddeeff')"
echo "  POSTGRES_PASSWORD    (currently: 'defense')"
echo ""
echo "  Generate random values with:"
echo "    openssl rand -hex 16   # for KEY_VAULT_MASTER_KEY (32 hex chars = 16 bytes)"
echo "    openssl rand -base64 32 # for JWT_SECRET"
echo ""

# ── 5. Build & start ─────────────────────────────────────────
info "Building Docker images (first build downloads ~800 MB CPU PyTorch)…"
docker compose build

info "Starting services…"
docker compose up -d

echo ""
info "Waiting for services to be ready…"
sleep 5

# ── 6. Health check ──────────────────────────────────────────
for i in $(seq 1 10); do
  if curl -sf http://localhost:8080/api/health >/dev/null 2>&1; then
    info "Go API is up"
    break
  fi
  sleep 3
  echo "  Waiting ($i/10)…"
done

for i in $(seq 1 10); do
  if curl -sf http://localhost:5001/health >/dev/null 2>&1; then
    info "Python service is up"
    break
  fi
  sleep 3
  echo "  Waiting ($i/10)…"
done

# ── 7. Create admin user ─────────────────────────────────────
echo ""
echo -e "${BOLD}Create first admin user${RESET}"
read -p "  Username: " ADMIN_USER
read -p "  Email:    " ADMIN_EMAIL
read -s -p "  Password (min 12 chars): " ADMIN_PASS
echo ""

REG_RESULT=$(curl -sf -X POST http://localhost:8080/api/auth/register \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"$ADMIN_USER\",\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASS\"}" 2>&1)

if echo "$REG_RESULT" | grep -q '"id"'; then
  info "User created. Promoting to admin…"
  # Promote via postgres directly
  docker compose exec postgres psql -U defense -d defense \
    -c "UPDATE users SET role='admin' WHERE username='$ADMIN_USER';" 2>/dev/null && \
    info "Promoted $ADMIN_USER to admin" || \
    warn "Could not auto-promote. Run manually: UPDATE users SET role='admin' WHERE username='$ADMIN_USER';"
else
  warn "Registration response: $REG_RESULT"
fi

# ── 8. Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo "  Application: http://localhost:8080"
echo "  Python API:  http://localhost:5001/health"
echo ""
echo "  Logs:        docker compose logs -f"
echo "  Stop:        docker compose down"
echo "  Restart:     docker compose restart"
echo ""
echo "  To load a new model checkpoint without rebuilding:"
echo "    cp /path/to/new.pth ./python_service/models/defense_final.pth"
echo "    docker compose restart python"
echo ""