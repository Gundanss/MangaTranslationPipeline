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

wait_for_health_down() {
  for _ in {1..120}; do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

if curl -fsS --max-time 2 -X POST "http://127.0.0.1:$PORT/api/system/shutdown" >/dev/null 2>&1; then
  echo "已发送优雅停机请求，正在等待当前图片处理完成后关闭服务..."
  if wait_for_health_down; then
    if [[ -f "$PID_FILE" ]]; then
      SERVER_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
      if [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        clear_pid_file
      fi
    fi
    echo "漫画翻译流水线已停止。"
    exit 0
  fi
  echo "服务仍在运行，尝试使用 PID 兜底停止。"
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "当前没有检测到正在运行的漫画翻译流水线。"
  exit 0
fi

SERVER_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$SERVER_PID" ]]; then
  clear_pid_file
  echo "PID 文件为空，已清理。当前没有检测到正在运行的服务。"
  exit 0
fi

if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
  clear_pid_file
  echo "服务进程已不存在，已清理陈旧 PID 文件。"
  exit 0
fi

COMMAND_LINE="$(ps -p "$SERVER_PID" -o command= 2>/dev/null || true)"
if [[ "$COMMAND_LINE" != *"uvicorn manga_pipeline.main:app"* ]]; then
  echo "PID $SERVER_PID 不是当前项目的漫画翻译流水线进程，已停止兜底终止。"
  exit 1
fi

kill -TERM "$SERVER_PID"
echo "已发送 TERM 信号，等待服务退出..."
for _ in {1..60}; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    clear_pid_file
    echo "漫画翻译流水线已停止。"
    exit 0
  fi
  sleep 0.5
done

echo "服务仍未退出，请稍后再试。"
exit 1
