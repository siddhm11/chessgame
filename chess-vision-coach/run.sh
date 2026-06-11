#!/usr/bin/env bash
# One-command start for Chess Vision Coach.
# Checks Python, sets up a venv, installs deps, verifies Stockfish, and
# starts uvicorn on port 8000.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

# --- 1. Python version --------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not found. Install Python 3.10 or newer." >&2
  exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
  echo "ERROR: Python 3.10+ required (found $PYVER)." >&2
  exit 1
fi
echo "Python $PYVER OK"

# --- 2. virtualenv + dependencies --------------------------------------
if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv)..."
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Installing dependencies..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -r requirements.txt

# --- 3. Stockfish ------------------------------------------------------
STOCKFISH_FOUND=""
if [ -n "${STOCKFISH_PATH:-}" ] && [ -x "${STOCKFISH_PATH:-}" ]; then
  STOCKFISH_FOUND="$STOCKFISH_PATH"
elif command -v stockfish >/dev/null 2>&1; then
  STOCKFISH_FOUND="$(command -v stockfish)"
elif [ -x /usr/games/stockfish ]; then
  STOCKFISH_FOUND="/usr/games/stockfish"
fi

if [ -n "$STOCKFISH_FOUND" ]; then
  echo "Stockfish OK ($STOCKFISH_FOUND)"
else
  echo "WARNING: Stockfish not found. The app will still start, but"
  echo "'Get best move' will not work until you install it:"
  case "$(uname -s)" in
    Darwin) echo "  brew install stockfish" ;;
    Linux)  echo "  sudo apt-get install stockfish" ;;
    *)      echo "  Download from https://stockfishchess.org/download/" ;;
  esac
fi

# --- 4. start the server -----------------------------------------------
echo ""
echo "Starting Chess Vision Coach -> http://localhost:8000"
echo "(Ctrl+C to stop)"
# Default to the fine-tuned YOLO backend (98% per-square accuracy on real
# photos). Override with `CVC_BACKEND=classical ./run.sh` for rendered
# diagrams / synthetic boards.
export CVC_BACKEND="${CVC_BACKEND:-model}"
echo "Vision backend: $CVC_BACKEND"
# Single worker: session state is in-process memory (see app/session.py).
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
