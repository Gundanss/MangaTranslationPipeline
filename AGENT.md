# 漫画翻译流水线项目笔记

## 1. 文档目的

这份 `AGENT.md` 是我通读当前仓库后整理出的项目认知笔记，目标是帮助后续接手的人或智能代理快速理解：

- 这个项目现在是做什么的；
- 主要代码分别在哪些文件里；
- 一条图片翻译任务是怎么流转的；
- 当前分支已经实现了哪些关键能力和边界行为；
- 修改时哪些地方最容易踩坑。


## 2. 项目定位

这是一个面向 **macOS Apple Silicon** 的本地 Web 应用，项目名为 **漫画翻译流水线**。

核心目标：

- 输入日语或英语漫画/图像；
- 识别图片中的文本；
- 翻译为目标语言；
- 自动去除原文并把译文重新嵌回图片；
- 在网页中继续人工校正 OCR、译文、框位置和排版，直到结果满意。

当前设计约束：

- **每个任务只允许一种源语言**：`ja` 或 `en`；
- 不支持同一任务里日英混合识别；
- 支持单图和文件夹批量处理；
- 批量结果保存到 `output/<时间戳-任务名>/`，并保留原始相对目录结构。


## 3. 技术栈与运行环境

### 3.1 后端

- Python `3.11`
- FastAPI
- SQLite
- Pillow / OpenCV
- httpx

### 3.2 前端

- 原生 HTML / CSS / JavaScript
- 无前端框架

### 3.3 图像翻译核心

固定复用 vendored 上游项目：

- `vendor/manga-image-translator`
- 锁定提交：`d5a3eee4a7b7b7754b71baa2ee82309dfff468bc`

处理链路固定为：

1. CTD 漫画文字检测
2. 48px 多语言漫画 OCR
3. 翻译（Ollama / Google / Microsoft）
4. LaMa Large 去字
5. 自动排版与嵌字

### 3.4 许可证

- 本项目：`GPL-3.0-only`
- 上游核心和模型许可证需继续参考各自说明


## 4. 仓库结构总览

### 4.1 关键源码目录

- `manga_pipeline/`
  - 后端主逻辑
- `static/`
  - 前端页面、样式、交互脚本
- `scripts/`
  - 环境检查、模型下载脚本
- `tests/`
  - pytest 测试
- `vendor/manga-image-translator/`
  - vendored 图像翻译核心

### 4.2 关键运行目录

- `uploads/`
  - 用户上传图片的本地副本
- `output/`
  - 输出图片、区域 JSON、重嵌字上下文
- `data/pipeline.db`
  - SQLite 数据库
- `models/`
  - OCR / 检测 / 去字模型
- `.local/settings.json`
  - API 设置与上次选中的 Ollama 模型


## 5. 关键文件职责

### 5.1 后端

#### `manga_pipeline/main.py`

FastAPI 入口，负责：

- 初始化 `Database`、`SecretStore`、`TaskManager`
- 挂载 `static/`
- 暴露 API 路由
- 处理任务创建、任务查询、SSE 事件流、图片文件访问、区域读取、重新 OCR、重新嵌字等接口

主要公开接口包括：

- `GET /api/health`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/ollama/models`
- `POST /api/tasks`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/events`
- `GET /api/images/{image_id}/regions`
- `POST /api/images/{image_id}/rerender`
- `POST /api/images/{image_id}/reprocess-regions`

#### `manga_pipeline/tasks.py`

任务调度器，负责：

- 维护任务队列；
- 启动单个后台 worker；
- 按图片依次处理整个任务；
- 调用 `CoreEngine` 完成单图处理；
- 管理进度回调、日志记录和任务状态。

当前并发模型很重要：

- 整体仍是 **单 worker 串行任务**；
- 重模型执行通过 `model_lock` 串行保护；
- 同一张图片还会额外使用 `image_lock`，避免人工编辑与后台写同一图片时互相覆盖；
- 这样做的结果是：
  - 不会让多套模型同时冲击内存；
  - 但已经完成的图片可以在批量任务继续跑后续页面时被人工编辑。

#### `manga_pipeline/engine.py`

这是项目最核心的文件，职责包括：

- 包装并调用 `manga-image-translator`；
- 构造检测、OCR、去字、渲染配置；
- 处理 MPS / CPU 回退；
- 序列化和反序列化区域数据；
- 保存上下文，支持后续重新 OCR 和重新嵌字；
- 为人工 OCR 框、人工译文框提供二次处理能力。

当前这条分支上的关键行为：

- **无文本图片不报错**，直接按成功处理，输出原图并保存空区域列表；
- **重新嵌字** 时使用保存下来的干净底图，而不是在旧译文图片上继续叠字；
- **人工 OCR 框重处理** 时，不再把大框硬当作单行文本：
  - 会先按用户 `ocr_bbox` 裁剪原图；
  - 在框内重新跑一次文字检测；
  - 对检测到的子文字线做 OCR；
  - 再按源语言阅读顺序合并回一个 UI 区域；
  - 如果框内没检测到子文字线，再回退为单框 OCR；
- 渲染前会补齐 `target_lang`、方向、对齐、颜色等字段，兼容历史上下文。

#### `manga_pipeline/providers.py`

翻译提供方抽象层，支持：

- Ollama
- Google Cloud Translation
- Microsoft / Bing Translator

这里有几个很关键的细节：

- Ollama 采用带编号的 `<|1|>...` 格式做多区域翻译；
- 如果模型批量返回格式不稳定，会自动回退为逐区域翻译；
- 已对异常模型输出做清洗，能去掉类似：
  - `<|1|>`
  - `</|2|>`
  - `<|/3|>`
- 也会去掉 `译文:` / `translation:` 这类前缀；
- 内置重试与重试日志。

#### `manga_pipeline/db.py`

SQLite 持久化层，负责：

- 任务表 `tasks`
- 图片表 `images`
- 日志表 `logs`

特点：

- 使用 WAL；
- 打开外键；
- 写入使用线程锁串行化；
- 日志是结构化的，支持 `details_json`。

#### `manga_pipeline/secret_store.py`

负责本地设置与密钥存储：

- Ollama 地址
- Google API Key
- Microsoft API Key / Region / Endpoint
- 上次使用的 Ollama 模型

默认保存在 `.local/settings.json`，并设置权限为 `600`。

#### `manga_pipeline/config.py`

负责：

- 根目录与运行目录定义；
- 运行目录初始化；
- 文件名清洗；
- 相对路径安全校验；
- 限制支持的图片扩展名：
  - `.jpg`
  - `.jpeg`
  - `.png`
  - `.webp`

#### `manga_pipeline/schemas.py`

Pydantic 模型定义，主要包括：

- `TaskConfig`
- `SettingsUpdate`
- `RegionUpdate`
- `RerenderRequest`
- `ReprocessRegionsRequest`

当前区域模型里很重要的字段：

- `bbox`
- `ocr_bbox`
- `render_bbox`
- `enabled`
- `text`
- `translation`
- `font_size`
- `direction`
- `alignment`
- `foreground`
- `outline`

### 5.2 前端

#### `static/index.html`

页面主结构，包含：

- 顶部状态栏与 API 设置按钮
- 新建任务表单
- 最近任务列表
- 当前任务状态与进度条
- 图片条带
- “校正与重嵌”标签页
- “运行日志”标签页

#### `static/app.js`

前端主控制器，负责：

- 初始化健康状态、设置、Ollama 模型列表；
- 创建任务；
- 通过 `EventSource` 订阅任务 SSE；
- 维护任务、图片、区域、日志等前端状态；
- 驱动 OCR 框 / 译文框 的拖拽、缩放、输入同步；
- 发起“重新识别并处理 OCR 框”和“保存校正并重新嵌字”。

当前前端状态管理里几个重要结论：

- 任务运行时，**已完成图片仍可点击进入编辑**；
- 正在处理或尚未处理的图片保持只读；
- SSE 刷新会更新当前图片状态，但会尽量避免把用户正在输入的内容覆盖掉；
- 日志窗口有“自动跟随”逻辑：
  - 用户停留在底部时自动滚动；
  - 用户手动上滚后不强制拉回底部；
  - 回到底部后恢复自动跟随。

#### `static/styles.css`

原生样式文件，整体是暖色、卡片式界面，没有依赖外部 UI 框架。


## 6. 一条任务的完整流转

### 6.1 创建任务

1. 前端收集：
   - 任务名
   - 源语言
   - 目标语言
   - 翻译提供方
   - Ollama 模型
   - 可选润色模型
   - 初始排版参数
   - 图片文件和相对路径
2. `POST /api/tasks`
3. 后端校验配置：
   - 提供方配置是否齐全
   - Ollama 是否可连通
   - 指定模型是否真实存在
4. 图片复制到 `uploads/<task_id>/...`
5. 输出目录创建为 `output/<时间戳-任务名>/...`
6. 为每张图片预留：
   - 输出路径
   - 上下文 `context.pkl`
   - 区域 JSON
7. 数据写入 SQLite
8. 任务入队

### 6.2 后台处理单张图片

对每张图，后台会：

1. OCR / 检测 / 翻译 / 去字 / 嵌字
2. 输出成品图
3. 保存区域 JSON
4. 保存私有上下文，用于后续重处理
5. 记录结构化日志
6. 更新任务和图片进度

### 6.3 人工编辑

当前已经支持两类人工操作：

#### A. 重新嵌字

用户可修改：

- 译文
- 字号
- 方向
- 对齐
- 文字色
- 描边色
- 译文框位置与大小

然后只重做“排版与嵌字”，不会重新跑 OCR。

#### B. 重新识别并处理 OCR 框

用户可修改：

- OCR 框位置
- OCR 框大小
- 新增 OCR 框
- 禁用误识别区域

之后后台会重新：

1. 在该 OCR 框内检测文本；
2. OCR；
3. 翻译；
4. 重新生成相关区域；
5. 去字并重新嵌字。


## 7. 当前编辑器能力

在当前分支里，网页编辑器已经具备：

- 查看翻译结果图；
- 切换 `OCR框` / `译文框` 两种编辑模式；
- 查看并调整单个区域的：
  - 坐标
  - OCR 原文
  - 译文
  - 字号
  - 方向
  - 对齐
  - 文字色
  - 描边色
- 拖拽和缩放 OCR 框；
- 拖拽和缩放译文框；
- 新增 OCR 框；
- 禁用 / 恢复某个区域；
- 重新识别并处理 OCR 框；
- 保存校正并重新嵌字。


## 8. 本地数据格式与持久化

### 8.1 SQLite

`data/pipeline.db` 中的核心实体：

- `tasks`
  - 任务级别状态、配置、进度
- `images`
  - 每张图片的输入、输出、上下文和状态
- `logs`
  - 结构化运行日志

### 8.2 区域 JSON

区域 JSON 是前后端共享的重要结构。当前版本应优先保留这些字段：

- `index`
- `bbox`
- `ocr_bbox`
- `render_bbox`
- `enabled`
- `text`
- `translation`
- `font_size`
- `direction`
- `alignment`
- `foreground`
- `outline`

兼容逻辑已经做过：

- 老数据如果只有 `bbox`，会尽量补齐 `ocr_bbox` / `render_bbox`；
- 翻译文本会在读取时做一次清洗。

### 8.3 私有上下文

每张图会在输出目录下的 `.pipeline/` 中保存私有上下文，供后续：

- 重新 OCR
- 重新翻译
- 重新去字
- 重新嵌字

这个上下文是编辑器可工作的关键依赖，不应随意改变结构。


## 9. 翻译与模型相关结论

### 9.1 Ollama

- 模型列表来自本机 `/api/tags`
- 应用不会自动下载、删除或修改 Ollama 模型
- 任务启动前会验证所选模型是否存在
- 如果 Ollama 未启动或模型不存在，任务会在创建前直接报错

### 9.2 在线翻译

支持：

- Google Cloud Translation
- Microsoft / Bing Translator

可选：

- 使用 Ollama 对在线译文做二次润色

### 9.3 首次安装下载的不是翻译模型

首次安装下载的是图像处理模型，不包括 Ollama 模型，主要是：

- CTD 检测模型
- 48px OCR 模型与字典
- LaMa Large 去字模型


## 10. 已知约束与容易踩坑的点

### 10.1 内存与并发

项目明显在有意识地控制内存：

- 后台只有一个 worker；
- 重模型执行受 `model_lock` 保护；
- 任务处理中会周期性做内存清理；
- 这是为了避免一次批量翻译时同时跑多套深度学习模型导致内存暴涨。

### 10.2 不能把新译文叠在旧译文图上

重新嵌字必须基于保存的干净底图，否则会出现文字重叠。当前实现已经按这个方向修过，后续改动不要把它退化回去。

### 10.3 人工 OCR 不应简单假设“一个框就是一条文字线”

很多气泡里是多行、多列、竖排或不规则分布。当前更合理的做法是：

- 用户先给一个粗框；
- 框内重新检测子文字线；
- 再 OCR；
- 最后按阅读顺序合并。

后续如果继续优化 OCR，应该沿着这个方向做，而不是回退成单框暴力识别。

### 10.4 无文本图片应视为成功

如果图片本来就没有可翻译文字：

- 不应把任务标成失败；
- 应直接输出原图；
- 区域列表为空即可。

### 10.5 前端不要把 SSE 刷新变成“覆盖用户输入”

图片状态刷新和表单编辑是两条状态流。后续修改前端时，要特别注意：

- 任务状态可以更新；
- 当前选中图片的元数据可以更新；
- 但不要把用户正在编辑的 OCR / 译文 / 坐标内容意外写回旧值。


## 11. 测试与常用命令

### 11.1 启动开发服务

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/vendor/manga-image-translator:$PWD"
python -m uvicorn manga_pipeline.main:app --host 127.0.0.1 --port 8765
```

### 11.2 常用检查

```bash
.venv/bin/python -m pytest -q
node --check static/app.js
.venv/bin/python -m compileall -q manga_pipeline scripts
zsh -n 首次安装.command
zsh -n 启动漫画翻译流水线.command
```

### 11.3 现有测试覆盖重点

测试已经覆盖到的重点包括：

- 配置与路径安全
- 数据库读写
- 翻译响应清洗与编号解析
- 无文本图片成功处理
- 重新嵌字使用干净底图
- 区域序列化字段完整性
- OCR 重处理后返回新 OCR / 新译文
- 人工 OCR 框内多子文本检测与合并
- 批量处理中已完成图片可插队人工编辑


## 12. 对后续维护者的建议

如果后续继续开发，这几个原则值得保持：

1. **优先兼容已有上下文和区域 JSON**
   - 因为重嵌字、重 OCR 都依赖它们。
2. **不要破坏单源语言任务的假设**
   - 前后端、提示词、阅读顺序都建立在这个约束上。
3. **不要让前端状态刷新覆盖人工正在编辑的内容**
   - 这是当前交互体验最敏感的点之一。
4. **不要让人工操作重新拉起整套并发重模型**
   - 现有锁模型是在效果和内存之间的折中。
5. **对翻译文本的清洗逻辑要谨慎**
   - 这里已经专门修过模型多余标签污染的问题。
6. **输出目录和相对路径结构尽量保持稳定**
   - 批量任务依赖它来维持页面与文件夹的一致性。


## 13. 一句话总结

这不是一个“纯 OCR 工具”，也不是一个“纯翻译工具”，而是一个围绕 **漫画图像翻译 + 去字嵌字 + 人工校正闭环** 搭起来的本地工作台。后续改动时，最好始终把这三个阶段当成一个整体来维护：**识别、翻译、重绘**。
