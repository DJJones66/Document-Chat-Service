#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"

API_HOST="${API_HOST:-127.0.0.1}"
PORT_START="${PORT_START:-18081}"
PORT_COUNT="${PORT_COUNT:-10}"
VENV_PATH="${VENV_PATH:-.venv}"
OLLAMA_REMOTE_HOST="${OLLAMA_REMOTE_HOST:-10.1.2.149}"
OLLAMA_REMOTE_PORT="${OLLAMA_REMOTE_PORT:-11434}"
OLLAMA_REMOTE_BASE="http://${OLLAMA_REMOTE_HOST}:${OLLAMA_REMOTE_PORT}"
OLLAMA_LLM_BASE_URL="${OLLAMA_LLM_BASE_URL:-$OLLAMA_REMOTE_BASE}"
OLLAMA_EMBEDDING_BASE_URL="${OLLAMA_EMBEDDING_BASE_URL:-$OLLAMA_REMOTE_BASE}"
OLLAMA_CONTEXTUAL_LLM_BASE_URL="${OLLAMA_CONTEXTUAL_LLM_BASE_URL:-$OLLAMA_REMOTE_BASE}"
OLLAMA_BASE_URL="${OLLAMA_LLM_BASE_URL:-$OLLAMA_EMBEDDING_BASE_URL}"
REQUIRE_OLLAMA="${REQUIRE_OLLAMA:-0}"

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return 0
  fi
  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_CMD="$(pick_python)" || {
  echo "No python found. Set PYTHON_BIN to a Python 3.11-3.13 executable." >&2
  exit 1
}
export PYTHON_BIN="$PYTHON_CMD"
export VENV_PATH="$VENV_PATH"
export OLLAMA_LLM_BASE_URL="$OLLAMA_LLM_BASE_URL"
export OLLAMA_EMBEDDING_BASE_URL="$OLLAMA_EMBEDDING_BASE_URL"
export OLLAMA_CONTEXTUAL_LLM_BASE_URL="$OLLAMA_CONTEXTUAL_LLM_BASE_URL"

if [[ "$VENV_PATH" = /* ]]; then
  VENV_DIR="$VENV_PATH"
else
  VENV_DIR="$REPO_ROOT/$VENV_PATH"
fi
VENV_PY="$VENV_DIR/bin/python"

find_free_port() {
  "$PYTHON_CMD" - "$PORT_START" "$PORT_COUNT" <<'PY'
import socket
import sys

start = int(sys.argv[1])
count = int(sys.argv[2])
for i in range(count):
    port = start + i
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    sys.exit(0)
sys.exit(1)
PY
}

wait_for_health() {
  local url="$1"
  local retries="${2:-12}"
  local delay="${3:-3}"
  for ((i = 0; i < retries; i++)); do
    if curl --fail --silent --show-error "$url" >/dev/null; then
      return 0
    fi
    sleep "$delay"
  done
  echo "Health check failed at $url" >&2
  return 1
}

is_local_ollama() {
  case "$OLLAMA_BASE_URL" in
    *localhost*|*127.0.0.1*|*0.0.0.0*) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_ollama_or_skip() {
  if [[ "${OLLAMA_HEALTH_SKIP:-0}" == "1" ]]; then
    echo "OLLAMA_HEALTH_SKIP=1 set; skipping Ollama health checks."
    return 0
  fi

  if is_local_ollama && ! command -v ollama >/dev/null 2>&1; then
    if [[ "$REQUIRE_OLLAMA" == "1" ]]; then
      echo "Ollama not found. Install/start Ollama or unset REQUIRE_OLLAMA=1." >&2
      return 1
    fi
    echo "Ollama not found; setting OLLAMA_HEALTH_SKIP=1 for this run."
    export OLLAMA_HEALTH_SKIP=1
    return 0
  fi

  if ! curl --fail --silent --show-error "$OLLAMA_BASE_URL/api/tags" >/dev/null; then
    if [[ "$REQUIRE_OLLAMA" == "1" ]]; then
      echo "Ollama not reachable at $OLLAMA_BASE_URL. Start Ollama or unset REQUIRE_OLLAMA=1." >&2
      return 1
    fi
    echo "Ollama not reachable at $OLLAMA_BASE_URL; setting OLLAMA_HEALTH_SKIP=1 for this run."
    export OLLAMA_HEALTH_SKIP=1
  fi
}

cleanup() {
  local code=$?
  set +e
  if [[ -x "$VENV_PY" ]]; then
    "$VENV_PY" "$REPO_ROOT/service_scripts/shutdown_with_venv.py" >/dev/null 2>&1 || true
  fi
  if [[ -n "${START_PID:-}" ]] && kill -0 "$START_PID" 2>/dev/null; then
    kill "$START_PID" 2>/dev/null
  fi
  if [[ -n "${RESTART_PID:-}" ]] && kill -0 "$RESTART_PID" 2>/dev/null; then
    kill "$RESTART_PID" 2>/dev/null
  fi
  if [[ -n "${START_PID:-}" ]]; then
    wait "$START_PID" 2>/dev/null || true
  fi
  if [[ -n "${RESTART_PID:-}" ]]; then
    wait "$RESTART_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT

PORT="$(find_free_port)" || {
  echo "No free port found in range $PORT_START-$((PORT_START + PORT_COUNT - 1))." >&2
  exit 1
}
export API_HOST="$API_HOST"
export API_PORT="$PORT"

cd "$REPO_ROOT"

echo "Using python: $PYTHON_CMD"
echo "Using port: $API_HOST:$API_PORT"
echo "Using Ollama base URL: $OLLAMA_BASE_URL"

ensure_ollama_or_skip

"$PYTHON_CMD" "$REPO_ROOT/service_scripts/create_venv.py"
"$PYTHON_CMD" "$REPO_ROOT/service_scripts/install_with_venv.py"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Venv python not found at $VENV_PY" >&2
  exit 1
fi

START_STDOUT="$DATA_DIR/test_start_stdout.log"
START_STDERR="$DATA_DIR/test_start_stderr.log"
RESTART_STDOUT="$DATA_DIR/test_restart_stdout.log"
RESTART_STDERR="$DATA_DIR/test_restart_stderr.log"

"$VENV_PY" "$REPO_ROOT/service_scripts/start_with_venv.py" >"$START_STDOUT" 2>"$START_STDERR" &
START_PID=$!

HEALTH_URL="http://$API_HOST:$API_PORT/health"
wait_for_health "$HEALTH_URL"

"$VENV_PY" "$REPO_ROOT/service_scripts/restart_with_venv.py" >"$RESTART_STDOUT" 2>"$RESTART_STDERR" &
RESTART_PID=$!

wait_for_health "$HEALTH_URL"

"$VENV_PY" "$REPO_ROOT/service_scripts/shutdown_with_venv.py"
sleep 2

echo "System test OK on $HEALTH_URL"
echo "Logs: $START_STDERR, $RESTART_STDERR"
