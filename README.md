# 漫画翻译流水线

“漫画翻译流水线”是面向 macOS Apple Silicon 的本地 Web 应用。它把漫画图片中的日语或英语识别出来，翻译为指定语言，自动去除原文并把译文嵌回图片；处理完成后，还可以在网页中修改 OCR、译文和基础排版并重新嵌字。

## 主要功能

- 每个任务手动选择一种源语言：日语或英语。
- 支持单张图片和文件夹批量导入，支持 JPG、JPEG、PNG、WebP。
- 支持 Ollama 本地模型、Google Cloud Translation、Microsoft/Bing Translator。
- Ollama 模型下拉框实时读取本机 `/api/tags`，系统不会自动下载、删除或修改 Ollama 模型。
- 在线机器翻译可选用 Ollama 进行译文润色，默认关闭。
- 实时显示真实处理阶段、总体进度、OCR 原文、译文与错误日志。
- 新建任务时可选择初始文字方向、对齐方式和固定字号，也可以保持自动排版。
- 支持修改 OCR、译文、字号、方向、对齐、文字色和描边色后重新嵌字。
- 批量结果保存在 `output/<时间戳-任务名>/`，并保留原文件夹相对结构。

## 技术路线

应用使用 FastAPI、原生 HTML/CSS/JavaScript 和 SQLite。图像流水线固定复用 GPLv3 项目 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 的提交 `d5a3eee4a7b7b7754b71baa2ee82309dfff468bc`：

```text
CTD 漫画文字检测
  → 48px 多语言漫画 OCR
  → Ollama / Google / Microsoft 翻译
  → LaMa Large 去字
  → 自适应排版与嵌字
```

日语任务默认使用从右到左的阅读顺序，英语任务默认使用从左到右的阅读顺序。任务以单工作线程串行执行，避免多个深度学习模型同时占满内存。

## 首次安装

1. 确保系统已安装 Python 3.11、Git 和 Ollama。
2. 双击 `首次安装.command`。
3. 安装完成后，双击 `启动漫画翻译流水线.command`。

首次安装会初始化固定版本的图像处理核心、创建 `.venv`、安装 Python 依赖，并下载以下模型：

| 模型 | 用途 | 约下载量 |
| --- | --- | ---: |
| `comictextdetector.pt.onnx` | CTD 漫画文字检测 | 94.7 MB |
| `ocr_ar_48px.ckpt` 与字典 | 日英漫画 OCR | 204.5 MB |
| `lama_large_512px.ckpt` | LaMa Large 漫画去字 | 约 204.5 MB |

图像模型总下载量约 504 MB。Python、PyTorch、OpenCV 等运行库会额外占用磁盘空间。安装脚本不会下载 Ollama 模型；网页只会显示本机已经存在的模型。

## 使用 Ollama

先启动 Ollama，并自行准备希望使用的模型。例如当前机器已经安装的模型会直接出现在下拉框中。应用每次创建任务前都会确认所选模型仍然存在；如果 Ollama 未启动或模型已被移除，任务不会开始。

推荐优先使用专用翻译模型，例如 `demonbyron/HY-MT1.5-7B:latest`。通用模型也可以选择，但翻译质量取决于模型对目标语言和编号格式的遵循程度。

## 配置 Google 与 Microsoft/Bing

点击网页右上角的“API 设置”：

- Google：填写 Google Cloud Translation API Key。
- Microsoft/Bing：填写 Translator API Key 和 Azure 区域，例如 `eastasia`。

密钥保存在本机 `.local/settings.json`，文件权限设置为仅当前用户可读写，并已被 Git 忽略。只有 OCR 文本会发送给在线翻译服务，原始漫画图片不会发送给 Google 或 Microsoft。

## 输出与本地数据

- `output/`：翻译成品和用于重新嵌字的本地上下文。
- `uploads/`：用户导入图片的本地副本。
- `data/pipeline.db`：任务、进度和结构化日志。
- `models/`：文字检测、OCR 和去字模型。
- `.local/settings.json`：API 密钥与本地设置。

项目不会自动清理或批量删除以上数据。

## 开发运行

完成首次安装后：

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/vendor/manga-image-translator:$PWD"
python -m uvicorn manga_pipeline.main:app --host 127.0.0.1 --port 8765
```

运行测试：

```bash
source .venv/bin/activate
pytest
```

## 已知限制

- 首版针对当前 macOS Apple M3 Pro 环境，优先使用 MPS，核心不支持时会回退到 CPU。
- 自动去字和嵌字效果受气泡形状、文字颜色、字体风格和画面背景影响；复杂区域可在网页中人工校正。
- 每个任务只处理一种源语言，不支持同一任务中的日英混合识别。
- Google 与 Microsoft/Bing 使用官方 API，需要用户自行开通服务并承担相应费用。

## 许可证

本项目采用 GNU GPLv3，第三方核心及其模型许可证以各自项目说明为准。
