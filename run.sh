#!/usr/bin/env bash
# Starts both the FastAPI backend and the React+Electron frontend for development.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load environment variables if .env exists
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

cleanup() {
  echo "Shutting down..."
  kill $BACKEND_PID 2>/dev/null || true
  kill $FRONTEND_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting FastAPI backend on http://localhost:8000 ..."
cd "$ROOT_DIR"
/opt/anaconda3/bin/python3 -m uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend (requires Node 22)
echo "Starting React frontend on http://localhost:5173 ..."
cd "$ROOT_DIR/frontend"
source ~/.nvm/nvm.sh && nvm use 22 > /dev/null 2>&1
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both."
echo ""

wait
