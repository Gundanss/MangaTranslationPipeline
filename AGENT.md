# 漫画翻译流水线 AGENT 交接笔记

## 1. 这份文档是干什么的

这份 `AGENT.md` 面向后续接手项目的人类开发者和智能体，目标不是重复 README，而是把当前仓库里已经落地的：

- 架构分层
- 任务调用链
- 数据落盘方式
- 编辑器能力
- 翻译提供方策略
- 已修复过、不能回退的行为

系统地整理出来，方便继续开发、排查问题和交接。


## 2. 项目当前定位

**漫画翻译流水线** 是一个面向 **macOS Apple Silicon** 的本地 Web 应用。它把漫画或图像里的日语/英语文本识别出来，翻译成目标语言，自动去除原文并重嵌译文；然后允许用户在网页里继续人工修正 OCR、译文和文本框，直到结果满意。

当前产品边界：

- 每个任务只允许一种源语言：`ja` 或 `en`
- 不支持同一任务里混合日英 OCR/翻译
- 支持单图和文件夹批量处理
- 支持网页内二次编辑、重新 OCR、重新嵌字
- 批量输出保留输入文件夹的相对目录结构


## 3. 技术路线总览

### 3.1 总体栈

- 后端：Python 3.11 + FastAPI + SQLite + httpx + Pillow + OpenCV
- 前端：原生 HTML / CSS / JavaScript
- 图像核心：vendored `manga-image-translator`
- 任务执行：单 worker 串行 + 模型锁 + 单图写锁

### 3.2 固定图像处理链路

项目固定复用 `vendor/manga-image-translator` 的图像流水线，核心路线是：

```text
CTD 漫画文字检测
  -> 48px 多语言漫画 OCR
  -> 翻译
  -> LaMa Large 去字
  -> 自动排版与嵌字
```

其中：

- 日语任务默认按右到左阅读逻辑渲染
- 英语任务默认按左到右阅读逻辑渲染
- 中文目标语言默认关闭英文断词，优先适合气泡内排版

### 3.3 模型与依赖下载策略

首次安装脚本只会下载图像处理相关模型，不会下载任何 Ollama 模型。

首次安装固定准备的模型：

- `comictextdetector.pt.onnx`：CTD 检测
- `ocr_ar_48px.ckpt` + `alphabet-all-v7.txt`：48px OCR
- `lama_large_512px.ckpt`：LaMa 去字

总图像模型下载量约 **504 MB**。Ollama 模型必须由用户自行安装，本项目只读取，不自动下载、删除或修改。


## 4. 仓库结构和职责分层

### 4.1 关键目录

- `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline`
  - 后端业务逻辑
- `/Users/leijh/Documents/MangaTranslationPipeline/static`
  - 前端页面、样式和交互逻辑
- `/Users/leijh/Documents/MangaTranslationPipeline/scripts`
  - 安装与环境检查辅助脚本
- `/Users/leijh/Documents/MangaTranslationPipeline/tests`
  - pytest 测试
- `/Users/leijh/Documents/MangaTranslationPipeline/vendor/manga-image-translator`
  - vendored 上游图像处理核心

### 4.2 关键运行目录

- `/Users/leijh/Documents/MangaTranslationPipeline/uploads`
  - 上传文件副本
- `/Users/leijh/Documents/MangaTranslationPipeline/output`
  - 成品图与私有上下文
- `/Users/leijh/Documents/MangaTranslationPipeline/data/pipeline.db`
  - SQLite
- `/Users/leijh/Documents/MangaTranslationPipeline/models`
  - 图像模型
- `/Users/leijh/Documents/MangaTranslationPipeline/.local/settings.json`
  - 本地设置和 API key


## 5. 关键源码文件说明

### 5.1 API 和入口层

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/main.py`

FastAPI 入口，职责：

- 初始化 `Database`、`SecretStore`、`TaskManager`
- 暴露 REST API 和 SSE
- 处理文件上传、任务创建、图片读取、区域读取、重 OCR、重嵌字

核心接口：

- `GET /`
- `GET /api/health`
- `GET /api/settings`
- `PUT /api/settings`
- `GET /api/ollama/models`
- `POST /api/tasks`
- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/tasks/{task_id}/events`
- `GET /api/images/{image_id}/file/{kind}`
- `GET /api/images/{image_id}/regions`
- `POST /api/images/{image_id}/rerender`
- `POST /api/images/{image_id}/reprocess-regions`

这里还有一个很重要的兼容函数 `_normalize_region_json()`，会在读取旧区域 JSON 时补齐：

- `ocr_bbox`
- `render_bbox`
- `enabled`
- 清洗后的 `translation`

### 5.2 调度层

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/tasks.py`

`TaskManager` 是整个任务调度中枢。

它负责：

- 维护 `asyncio.Queue`
- 启动单个后台 worker
- 对任务里的图片逐张处理
- 在人工编辑时复用同一套模型执行通道

这里有两个非常重要的锁：

- `model_lock`
  - 保证重模型推理串行，避免内存爆炸
- `image_locks[image_id]`
  - 保证同一张图在“后台批量处理”和“人工重 OCR / 重嵌字”之间不会同时写上下文和输出文件

当前调度策略不是“整本任务大锁”，而是：

- 整个批量任务继续串行处理后续图片
- 已完成图片允许插队进行人工编辑
- 但人工操作会等待当前正在执行的模型操作结束，再进入模型执行区

这套设计是为了在“可编辑性”和“内存控制”之间取平衡。

### 5.3 核心图像流水线

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/engine.py`

这是最核心的文件，基本控制了：

- 上游核心加载
- 配置构建
- OCR / 检测 / 去字 / 嵌字执行
- 人工 OCR 框重处理
- 重新嵌字
- 上下文序列化
- 运行内存回收

关键点：

#### 1. 核心导入与模型目录接管

`_import_core()` 会把 vendored core 注入 `sys.path`，并把上游 `ModelWrapper._MODEL_DIR` 指向本仓库的 `/models`。

同时当前项目强制 CTD 走 CPU 映射，避免 ONNX 检测在这里走不稳定路径。

#### 2. MPS 到 CPU 回退

`process()` 和 `reprocess_regions()` 都是“两段式”：

- 先尝试 MPS
- 如果异常文本包含 `mps` / `metal` / `bfloat16` / `not implemented` 等标记
- 自动清缓存后改用 CPU 重跑

#### 3. 手动 OCR 的策略

用户拖一个 OCR 框后，系统不会简单把整个框当成一条文字线硬识别。

当前逻辑是：

1. 按用户 `ocr_bbox` 从原图裁剪子图
2. 在子图内重新跑文字检测
3. 对检测出的子文字线做 OCR
4. 再按源语言规则合并成一个区域文本
5. 如果框内完全没检测到子文字线，再回退为“单框 OCR”

这个行为是为了处理：

- 多行文字
- 多列竖排
- 不规则排布
- 大气泡内多个子文本块

后续优化 OCR 时不要退化回“一个框永远只做一行 OCR”。

#### 4. 重新嵌字必须使用干净底图

这是项目里一个非常重要、也修过 bug 的点。

当前实现会把“去字后但尚未嵌字”的干净底图作为重嵌字基底保存下来：

- pickle 上下文：`<image_id>.pkl`
- sidecar 干净图：`<image_id>.clean.png`

`rerender()` 会优先使用这张干净底图，而不是拿已经嵌过旧译文的结果图继续写字。否则就会出现文字叠字。

如果老上下文没有 sidecar，但有足够的原始上下文，系统会尝试重新生成干净底图。

#### 5. 无文本图片视为成功

当前实现已经修好：

- 没有文字的图片不会报错
- 任务不会因此失败
- 直接输出原图
- `regions` 返回空数组

#### 6. 序列化与兼容

区域数据的统一出口是 `serialize_regions()`，它会输出：

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

重处理和重嵌字都依赖这套字段，后续不要轻易改结构。

### 5.4 翻译提供方层

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/providers.py`

这是 OCR 后文本翻译的抽象层，当前支持三种逻辑入口：

- `ollama`
- `google`
- `microsoft`

但 `google` 和 `microsoft` 都已经演化成“官方 API + 免费网页通道”双路径策略。

#### 1. Ollama

- 通过 `<|1|>...` 这种编号格式做多区域翻译
- 如果模型不按格式返回，会自动回退为逐区域翻译
- 支持对在线翻译结果再做一轮 Ollama 润色

#### 2. Google

- 如果配置了 `google_api_key`，走官方 Google Translation API
- 如果没配 key，默认走免费网页接口：
  - `https://translate.googleapis.com/translate_a/single`

#### 3. Microsoft / Bing

这里的 `provider="microsoft"` 当前有两条路径：

- 配置了 `microsoft_api_key + microsoft_region`
  - 走 Microsoft 官方 Translator API
- 没有配置官方 key
  - 自动走免费 Bing 网页翻译

免费 Bing 路径的真实实现方式：

1. 先打开 `https://cn.bing.com/translator`
2. 跟随到最终页面
3. 从页面源码里提取：
   - `IG`
   - `IID`
   - `token`
   - `key`
   - TTL
4. 使用接近浏览器真实 XHR 的请求头向 `/ttranslatev3` 发请求
5. 会话参数按 TTL 缓存
6. 失效或 400 时自动刷新页面会话后重试

这里已经验证过真实请求可用，并实测翻译过：

- `Hello world -> 你好，世界`
- `最近アルバイトをしていると思ったら母親にプレゼントを買うためだったなんて…`

需要注意：

- 免费 Bing 走的是网页私有接口，不是官方公开 API
- 它将来有被网页改版打断的风险
- 这块如果出问题，优先先检查页面提取参数和请求头

#### 4. 译文清洗

`sanitize_translation_text()` 很关键，当前会清掉：

- `<|1|>`
- `</|2|>`
- `<|/3|>`
- `译文:`
- `translation:`
- 代码块包裹

这是为了避免模型或网页翻译返回污染文本，后续不要轻易删掉。

### 5.5 数据与设置层

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/db.py`

SQLite 存储层，3 张核心表：

- `tasks`
- `images`
- `logs`

特点：

- WAL 模式
- 开启外键
- 写入串行锁
- 日志 `details_json` 为结构化 JSON

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/secret_store.py`

本地设置存储层。

保存内容包括：

- `ollama_base_url`
- `google_api_key`
- `microsoft_api_key`
- `microsoft_region`
- `microsoft_endpoint`
- `last_ollama_model`

`public()` 只返回前端需要的公开字段，其中：

- `google_configured`
  - 表示是否填了官方 Google key
- `microsoft_configured`
  - 表示是否填了官方 Microsoft 配置
  - **不表示 provider 是否可用**，因为未配置时也能走免费 Bing

### 5.6 配置与 schema

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/config.py`

负责：

- 目录常量
- 运行目录创建
- 安全文件名清洗
- 相对路径安全限制
- 图片扩展名白名单

当前支持的输入后缀：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

#### `/Users/leijh/Documents/MangaTranslationPipeline/manga_pipeline/schemas.py`

Pydantic 模型定义，核心模型：

- `TaskConfig`
- `SettingsUpdate`
- `RegionUpdate`
- `RerenderRequest`
- `ReprocessRegionsRequest`

注意：

- `provider` 仍固定是 `Literal["ollama", "google", "microsoft"]`
- 没有新增 `bing-free` 这种枚举，历史兼容成本更低


## 6. 前端架构与编辑器行为

### 6.1 页面结构

#### `/Users/leijh/Documents/MangaTranslationPipeline/static/index.html`

页面由几块组成：

- 新建任务表单
- 任务历史
- 当前任务状态卡片
- 进度条
- 图片条带
- 编辑器 tab
- 日志 tab
- 设置弹窗

翻译提供方文案当前应理解为：

- `Ollama 本地模型`
- `Google 网页翻译（免费）`
- `Bing 网页翻译（免费） / Microsoft 官方 API`

### 6.2 前端状态机

#### `/Users/leijh/Documents/MangaTranslationPipeline/static/app.js`

前端全局状态 `state` 包括：

- 当前任务
- 当前图片
- 当前区域列表
- 当前激活区域
- OCR / 译文框编辑模式
- 脏 OCR 区域索引集合
- 拖拽状态
- 日志自动跟随状态

关键行为：

#### 1. 任务 SSE

前端通过 `EventSource(/api/tasks/{task_id}/events)` 拉：

- 最新任务状态
- 最新图片状态
- 新增日志

#### 2. 批量处理中仍可编辑已完成图片

图片条带中：

- `status === completed` 的图片可点击进入编辑
- `running/queued/failed` 图片不进入可编辑态

这是当前一个很重要的交互特性，不要回退成“任务 running 时整本都锁死”。

#### 3. 两种编辑模式

- `OCR框`
  - 调整 OCR 框
  - 新增 OCR 框
  - 触发重新识别并处理 OCR 框
- `译文框`
  - 调整最终嵌字区域
  - 只重做嵌字

#### 4. 表单同步策略

前端有意避免 SSE 或视图切换覆盖用户正在编辑的内容。
如果后续继续改前端，最容易踩坑的就是：

- 刷新图片状态时
- 重选区域时
- 切换 OCR/译文框模式时

把旧值写回当前表单，导致用户改过的 OCR 或译文被覆盖。

#### 5. 日志窗口滚动

日志窗口当前有“智能自动跟随”：

- 用户在底部时，新日志会跟随
- 用户手动上滚后，不强制拉回底部
- 回到底部后重新开启跟随


## 7. 一条任务的完整调用链

### 7.1 新建任务

1. 前端收集任务参数和文件
2. `POST /api/tasks`
3. `main.py` 调用 `_validate_config()`
4. `create_provider()` 校验 provider 可用性
5. 上传文件写入 `uploads/<task_id>/...`
6. 输出目录创建为 `output/<timestamp-task-name>/...`
7. 每张图预生成：
   - `output_path`
   - `context_path`
   - `regions_path`
8. `Database.create_task()` 写 `tasks/images`
9. `TaskManager.enqueue(task_id)`

### 7.2 后台 worker 处理单张图

`TaskManager._process_task()` 对每张图：

1. 生成 `CoreEngine`
2. 进入 `model_lock`
3. 进入该图自己的 `image_lock`
4. 调用 `CoreEngine.process()`
5. `engine.py` 内调用上游 translator
6. 输出成品图
7. 输出区域 JSON
8. 输出 pickle 上下文和干净底图 sidecar
9. 更新 DB 中的 image/task 状态
10. 写结构化日志

### 7.3 编辑器打开图片

1. 前端点击条带中的某张已完成图片
2. `GET /api/images/{image_id}/regions`
3. 后端读取区域 JSON，并做兼容补齐
4. 前端显示图片、框、右侧表单

### 7.4 人工“重新识别并处理 OCR 框”

1. 前端提交 `POST /api/images/{image_id}/reprocess-regions`
2. `TaskManager.reprocess_image()`
3. 进入 `model_lock + image_lock`
4. `CoreEngine.reprocess_regions()`
5. 对变更框重新检测、OCR、翻译、去字、嵌字
6. 更新：
   - 输出图
   - 区域 JSON
   - 上下文
7. 前端拿到最新 `regions`，刷新右侧 OCR 和译文

### 7.5 人工“保存校正并重新嵌字”

1. 前端提交 `POST /api/images/{image_id}/rerender`
2. `TaskManager.rerender_image()`
3. 进入 `model_lock + image_lock`
4. `engine.rerender()`
5. 使用干净底图 + 最新 `render_bbox/translation`
6. 只重做排版与嵌字，不重跑 OCR
7. 更新输出图、区域 JSON 和上下文


## 8. 上下文、区域 JSON 和输出目录

### 8.1 输出目录结构

每个任务输出目录大致是：

```text
output/<时间戳-任务名>/
  <与输入一致的相对路径图片>
  .pipeline/
    contexts/<image_id>.pkl
    contexts/<image_id>.clean.png
    regions/<image_id>.json
```

说明：

- 成品图走用户原始相对路径
- 私有上下文放在 `.pipeline/`
- `.clean.png` 是重新嵌字必须依赖的干净底图

### 8.2 区域 JSON 是前后端契约

当前区域 JSON 是前后端共同依赖的数据契约。
后续任何改动都必须优先兼容以下字段：

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

### 8.3 pickle 上下文的作用

pickle 上下文不是临时缓存，而是重处理能力的基础设施。它支撑：

- 重新嵌字
- 人工 OCR 框重处理
- 无需重新走整张图全流程

所以：

- 不要随便删字段
- 不要轻易切换格式
- 如果必须调整，优先加兼容读取逻辑


## 9. 运行脚本与本地启动

### 9.1 首次安装脚本

#### `/Users/leijh/Documents/MangaTranslationPipeline/首次安装.command`

职责：

1. 校验 Python 3.11
2. 初始化 `vendor/manga-image-translator` 子模块
3. 创建 `.venv`
4. 安装主项目和上游依赖
5. 运行 `scripts/download_models.py`

特点：

- 失败时会保留已完成步骤
- 会明确提示“不会下载任何 Ollama 模型”

### 9.2 启动脚本

#### `/Users/leijh/Documents/MangaTranslationPipeline/启动漫画翻译流水线.command`

职责：

- 检查 `.venv`
- 设置 `PYTHONPATH`
- 检查核心和模型是否齐全
- 探测 Ollama 状态
- 启动 uvicorn
- 自动打开浏览器到 `http://127.0.0.1:8765`


## 10. 已知关键约束和不可回退行为

### 10.1 内存控制优先

当前项目明确选择了“控制内存占用”优先于“并行吞吐最大化”：

- 单 worker
- 模型串行
- 周期性 `trim_runtime_memory()`

这是故意的，不是偶然写成的。

### 10.2 不允许重新嵌字叠在旧译文上

任何改动只要让重嵌字退回到“基于旧成品图再写字”，都属于回归 bug。

### 10.3 无文本图片不能报错

“没有可翻译文本”是正常场景，不应变成失败。

### 10.4 人工框编辑要优先保留用户输入

无论 SSE 刷新、重选区域还是模式切换，都不要把用户当前输入意外覆盖掉。

### 10.5 provider 兼容优先

现在 `provider="microsoft"` 已经同时承担：

- 免费 Bing
- 官方 Microsoft

所以不要轻易改 provider 枚举，除非愿意处理：

- 历史任务配置兼容
- 前端下拉值兼容
- 数据库里旧任务配置兼容

### 10.6 路径和输出结构要稳定

批量任务结果目录保留相对路径，这对用户找图和后续脚本都很重要。不要随便打乱输出结构。


## 11. 测试与常用命令

### 11.1 开发启动

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
git diff --check
```

### 11.3 当前测试重点

测试当前已覆盖这些重点：

- 配置与路径安全
- SQLite 读写
- 译文清洗与编号解析
- Google 免费翻译回退
- Bing 免费翻译 bootstrap 和 provider 选择
- 无文本图片成功处理
- 干净底图重嵌字
- OCR 重处理返回最新 OCR / 译文
- OCR 框内多子文本检测与合并
- 批量处理中已完成图片插队人工编辑


## 12. 后续维护建议

如果后续继续开发，优先遵守这些原则：

1. 先兼容已有 `regions.json` 和 `context.pkl`
2. 不破坏“单任务单源语言”假设
3. 不让前端刷新覆盖正在编辑的内容
4. 不让重模型并发失控导致内存飙升
5. 不让重嵌字退化为叠字
6. 对免费 Bing 这类网页私有接口保持警惕，改动时优先先做真实联机 smoke test


## 13. 一句话理解这个项目

这不是单纯的 OCR 工具，也不是单纯的机翻工具。它本质上是一个围绕 **漫画图像识别 -> 翻译 -> 去字嵌字 -> 人工校正闭环** 组织起来的本地工作台。理解和维护它时，最好始终把这四段当作一个整体系统来看。
