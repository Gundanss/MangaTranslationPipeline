#!/bin/zsh
set -e

installation_failed() {
  local code=$?
  echo
  echo "安装未完成（退出码 $code）。请检查上方错误后再次运行，本脚本会继续已完成的步骤。"
  read "?按回车键退出..." || true
  exit "$code"
}
trap installation_failed ERR

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "========================================"
echo " 漫画翻译流水线 - 首次安装"
echo "========================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 Python 3，请先安装 Python 3.11。"
  read "?按回车键退出..." || true
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
  echo "当前 Python 版本为 $PYTHON_VERSION，本项目需要 Python 3.11。"
  read "?按回车键退出..." || true
  exit 1
fi

echo "[1/4] 初始化固定版本的漫画处理核心..."
git submodule update --init --depth 1 vendor/manga-image-translator

echo "[2/4] 创建 Python 虚拟环境..."
if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
PIP_SOURCE_ARGS=(--index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org)
python -m pip install "${PIP_SOURCE_ARGS[@]}" --upgrade pip "setuptools<82" wheel

echo "[3/4] 安装应用与漫画处理依赖，这一步可能需要较长时间..."
python -m pip install "${PIP_SOURCE_ARGS[@]}" -e ".[dev]"
python -m pip install "${PIP_SOURCE_ARGS[@]}" -r vendor/manga-image-translator/requirements.txt

echo "[4/4] 下载约 504 MB 的文字检测、OCR 与去字模型..."
python scripts/download_models.py

echo
echo "安装完成。系统没有下载、删除或修改任何 Ollama 模型。"
echo "现在可以双击“启动漫画翻译流水线.command”。"
read "?按回车键退出..." || true
