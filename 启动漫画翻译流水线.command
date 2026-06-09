#!/bin/zsh
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
PID_FILE="$ROOT_DIR/.local/server.pid"
PORT=8765

clear_pid_file() {
  if [[ -f "$PID_FILE" ]]; then
    rm "$PID_FILE"
  fi
}

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" >/dev/null 2>&1; then
    :
  else
    clear_pid_file
  fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "尚未完成首次安装，请先双击“首次安装.command”。"
  read "?按回车键退出..." || true
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR/vendor/manga-image-translator:$ROOT_DIR"

if ! python scripts/check_environment.py >/dev/null 2>&1; then
  echo "图像处理模型尚未准备完成，请重新运行“首次安装.command”。"
  read "?按回车键退出..." || true
  exit 1
fi

OLLAMA_BASE_URL="$(python -c 'from manga_pipeline.secret_store import SecretStore; print(SecretStore().get()["ollama_base_url"].rstrip("/"))')"
if ! curl -fsS --max-time 2 "$OLLAMA_BASE_URL/api/version" >/dev/null 2>&1; then
  echo "提示：当前未连接到 Ollama。网页仍会启动，但使用 Ollama 前请先启动 Ollama。"
fi

if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "漫画翻译流水线已经运行，正在打开网页。"
  open "http://127.0.0.1:$PORT"
  exit 0
fi

echo "正在启动漫画翻译流水线：http://127.0.0.1:$PORT"
python -m uvicorn manga_pipeline.main:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"
trap 'kill "$SERVER_PID" 2>/dev/null || true; clear_pid_file' INT TERM EXIT

for _ in {1..40}; do
  if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    open "http://127.0.0.1:$PORT"
    break
  fi
  sleep 0.25
done

wait "$SERVER_PID"
