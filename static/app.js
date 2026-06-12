const $ = (id) => document.getElementById(id);

// 单页控制器状态。真正的任务事实仍以 SQLite 为准，这里只保存当前 UI
// 选择项、尚未提交的区域编辑内容，以及按钮的临时状态。
const state = {
  files: [],
  relativePaths: [],
  models: [],
  settings: null,
  activeTask: null,
  eventSource: null,
  logs: new Map(),
  activeImage: null,
  regions: [],
  activeRegion: null,
  editorMode: "ocr",
  dirtyOcrIndices: new Set(),
  dirtyMaskIndices: new Set(),
  dragState: null,
  logsAutoFollow: true,
  serverStopping: false,
  serverStopped: false,
  shutdownPoll: null,
  resumeImageId: null,
};

const stageNames = {
  queued: "排队中", starting: "准备中", running_pre_translation_hooks: "初始化",
  "mps-fallback": "MPS 失败，切换 CPU",
  detection: "文字检测", ocr: "OCR 识别", textline_merge: "合并文本区域",
  translating: "翻译中", "after-translating": "翻译校验", "mask-generation": "生成去字掩膜",
  inpainting: "去除原文", rendering: "自动嵌字", saved: "保存结果",
  finished: "已完成", error: "处理失败", rerender: "重新嵌字", retry: "翻译重试",
  "manual-ocr": "人工 OCR 框", shutdown: "停止中", stopped: "已停止",
};
const statusNames = {
  queued: "排队中", running: "处理中", completed: "已完成",
  completed_with_errors: "部分失败", failed: "失败", stopped: "已停止", idle: "空闲",
};
const ROTATION_SNAP_TARGETS = [0, 90, -90, 180];
const ROTATION_SNAP_THRESHOLD = 6;

// API 与格式化辅助函数
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = response.headers.get("content-type")?.includes("application/json")
    ? await response.json() : null;
  if (!response.ok) throw new Error(body?.detail || `请求失败：${response.status}`);
  return body;
}

function formatBytes(bytes) {
  if (!bytes) return "未知大小";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes, index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function setHealthChip(text, color) {
  $("healthChip").textContent = text;
  $("healthChip").style.color = color;
}

function setServiceBanner(message = "", tone = "warning") {
  const banner = $("serviceBanner");
  banner.textContent = message;
  banner.classList.toggle("hidden", !message);
  banner.classList.toggle("error", tone === "error");
}

function setMutationControlsDisabled(disabled) {
  // 进入优雅停机后，所有会改后端状态的控件都要禁用。
  [
    "startButton",
    "refreshModels",
    "addRegionButton",
    "toggleRegionButton",
    "reprocessButton",
    "rerenderButton",
    "machineTranslateButton",
    "ollamaTranslateButton",
    "saveSettings",
  ].forEach((id) => {
    const element = $(id);
    if (element) element.disabled = disabled;
  });
}

function markServiceStopping(message) {
  state.serverStopping = true;
  state.serverStopped = false;
  $("stopServiceButton").disabled = true;
  $("stopServiceButton").textContent = "正在停止...";
  setMutationControlsDisabled(true);
  setHealthChip("服务正在停止", "#ffd08e");
  setServiceBanner(message, "warning");
}

function markServiceStopped(message = "服务已停止，请重新双击启动脚本后刷新页面。") {
  // 主动关闭 SSE，避免本地进程退出后前端还在不断重连。
  state.serverStopping = false;
  state.serverStopped = true;
  if (state.shutdownPoll) {
    clearTimeout(state.shutdownPoll);
    state.shutdownPoll = null;
  }
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  $("stopServiceButton").disabled = true;
  $("stopServiceButton").textContent = "服务已停止";
  setMutationControlsDisabled(true);
  setHealthChip("服务已停止", "#ffb0b0");
  setServiceBanner(message, "error");
  $("formMessage").textContent = message;
  $("editorMessage").textContent = message;
}

async function loadHealth() {
  // health 同时驱动头部状态提示，也用来感知停机后的真正退出时刻。
  try {
    const health = await api("/api/health");
    if (health.shutting_down) {
      markServiceStopping("服务正在停止，会在当前这张图片处理完成后关闭。");
      return;
    }
    const ready = health.core_present && Object.values(health.models).every(Boolean);
    setHealthChip(ready ? "图像模型已就绪" : "需要首次安装", ready ? "#a9f0cf" : "#ffd08e");
  } catch {
    if (state.serverStopping || state.serverStopped) {
      markServiceStopped();
      return;
    }
    setHealthChip("环境检查失败", "#ffb0b0");
  }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  $("ollamaBaseUrl").value = state.settings.ollama_base_url;
  $("microsoftRegion").value = state.settings.microsoft_region || "";
  $("microsoftEndpoint").value = state.settings.microsoft_endpoint || "";
  $("googleState").textContent = state.settings.google_configured
    ? "已保存可选官方密钥；不填也会走免费网页翻译"
    : "默认使用免费网页翻译，无需密钥";
  $("microsoftState").textContent = state.settings.microsoft_configured
    ? "已保存官方 Microsoft 配置；当前也可继续使用免费 Bing 网页翻译"
    : "默认使用免费 Bing 网页翻译，无需密钥";
}

async function loadModels() {
  // Ollama 模型只从本机服务读取，不会触发任何下载动作。
  $("modelMeta").textContent = "正在读取本机 Ollama 模型...";
  try {
    const data = await api("/api/ollama/models");
    state.models = data.models;
    for (const select of [$("ollamaModel"), $("polishModel")]) {
      const previous = select.value;
      select.innerHTML = "";
      for (const model of state.models) {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = `${model.name} · ${model.parameter_size || "?"} · ${formatBytes(model.size)}`;
        select.append(option);
      }
      const installed = new Set(state.models.map((item) => item.name));
      const preferred = [
        previous,
        state.settings?.last_ollama_model,
        "demonbyron/HY-MT1.5-7B:latest",
        state.models[0]?.name,
      ].find((name) => name && installed.has(name));
      if (preferred) select.value = preferred;
    }
    updateModelMeta();
  } catch (error) {
    state.models = [];
    $("ollamaModel").innerHTML = "<option value=''>Ollama 不可用</option>";
    $("polishModel").innerHTML = "<option value=''>Ollama 不可用</option>";
    $("modelMeta").textContent = error.message;
  }
}

function updateModelMeta() {
  const model = state.models.find((item) => item.name === $("ollamaModel").value);
  $("modelMeta").textContent = model
    ? `本机模型 · ${model.family || "未知架构"} · ${model.quantization || "未知量化"} · ${formatBytes(model.size)}`
    : "请选择已下载的本地模型";
}

function updateProviderFields() {
  // 不同提供方的字段只是隐藏，不销毁，这样切换时表单值还能保留。
  const provider = $("provider").value;
  $("ollamaField").classList.toggle("hidden", provider !== "ollama");
  $("polishField").classList.toggle("hidden", provider === "ollama");
  $("polishModelField").classList.toggle("hidden", provider === "ollama" || !$("polishWithOllama").checked);
}

function setFiles(fileList) {
  // 文件夹上传会带 webkitRelativePath，保留下来才能复原输出目录结构。
  const allowed = /\.(jpe?g|png|webp)$/i;
  state.files = [...fileList].filter((file) => allowed.test(file.name));
  state.relativePaths = state.files.map((file) => file.webkitRelativePath || file.name);
  const total = state.files.reduce((sum, file) => sum + file.size, 0);
  $("fileSummary").textContent = state.files.length
    ? `已选择 ${state.files.length} 张图片，共 ${formatBytes(total)}`
    : "没有找到支持的 JPG、PNG 或 WebP 图片";
}

async function createTask() {
  // 任务会携带多张图片，因此这里用 multipart/form-data 提交。
  $("formMessage").textContent = "";
  if (state.serverStopping || state.serverStopped) {
    $("formMessage").textContent = "服务已进入停止流程，请重新启动后再创建任务";
    return;
  }
  if (!state.files.length) {
    $("formMessage").textContent = "请先选择单张图片或图片文件夹";
    return;
  }
  const provider = $("provider").value;
  const config = {
    name: $("taskName").value,
    source_language: $("sourceLanguage").value,
    target_language: $("targetLanguage").value,
    provider,
    ollama_model: provider === "ollama" ? $("ollamaModel").value : null,
    polish_with_ollama: provider !== "ollama" && $("polishWithOllama").checked,
    polish_model: provider !== "ollama" && $("polishWithOllama").checked ? $("polishModel").value : null,
    render_direction: $("initialDirection").value,
    render_alignment: $("initialAlignment").value,
    font_size: $("initialFontSize").value ? Number($("initialFontSize").value) : null,
    mask_dilation_offset: updateMaskDilationLabel("globalMaskDilationOffset", "globalMaskDilationValue"),
  };
  const form = new FormData();
  form.append("config_json", JSON.stringify(config));
  state.files.forEach((file, index) => {
    form.append("files", file, file.name);
    form.append("relative_paths", state.relativePaths[index]);
  });
  $("startButton").disabled = true;
  $("startButton").textContent = "正在创建任务...";
  try {
    const task = await api("/api/tasks", { method: "POST", body: form });
    activateTask(task.id);
    loadTasks();
  } catch (error) {
    $("formMessage").textContent = error.message;
  } finally {
    $("startButton").disabled = state.serverStopping || state.serverStopped;
    $("startButton").textContent = "开始翻译";
  }
}

async function loadTasks() {
  // 侧边栏只显示任务摘要；活动任务详情由下面的 SSE 实时刷新。
  const data = await api("/api/tasks");
  const container = $("taskHistory");
  if (!data.tasks.length) {
    container.className = "task-history empty-state";
    container.textContent = "暂无任务";
    return;
  }
  container.className = "task-history";
  container.innerHTML = "";
  data.tasks.forEach((task) => {
    const button = document.createElement("button");
    button.className = "task-row";
    button.innerHTML = `<strong>${escapeHtml(task.name)}</strong><span>${statusNames[task.status] || task.status}</span>
      <small>${task.completed_files}/${task.total_files} · ${new Date(task.created_at).toLocaleString()}</small>`;
    button.onclick = () => activateTask(task.id);
    container.append(button);
  });
}

function activateTask(taskId) {
  // 当前活动任务独占一个 SSE 流，用来接收快照和增量日志。
  if (state.serverStopped) return;
  state.logs.clear();
  state.logsAutoFollow = true;
  $("logWindow").innerHTML = "";
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/tasks/${taskId}/events`);
  state.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    state.activeTask = payload.task;
    payload.logs.forEach((log) => state.logs.set(log.id, log));
    renderTask();
    renderLogs();
  };
  state.eventSource.onerror = () => {
    if (state.serverStopping) {
      markServiceStopped();
    }
  };
}

function renderTask() {
  // 这里每次都按最新任务快照整体重绘，逻辑最直观也足够轻量。
  const task = state.activeTask;
  if (!task) return;
  syncActiveImageFromTask();
  $("activeTaskName").textContent = task.name;
  $("activeTaskMeta").textContent =
    `${task.config.source_language === "ja" ? "日语" : "英语"} → ${task.config.target_language} · ${task.config.provider} · 输出：${task.output_dir}`;
  $("activeStatus").textContent = statusNames[task.status] || task.status;
  $("activeStatus").className = `status-badge ${task.status}`;
  const percent = Math.round(task.progress * 100);
  $("progressBar").style.width = `${percent}%`;
  $("progressPercent").textContent = `${percent}%`;
  $("progressStage").textContent = stageNames[task.current_stage] || task.current_stage;
  renderImageStrip(task.images || []);
}

function isImageEditable(image) {
  return image?.status === "completed";
}

function canResumeImage(image) {
  // “从此页续跑”只针对失败、停止或尚未处理的页，已完成页不重跑。
  const taskStatus = state.activeTask?.status;
  return ["failed", "completed_with_errors", "stopped"].includes(taskStatus)
    && ["failed", "stopped", "queued"].includes(image?.status);
}

function syncActiveImageFromTask() {
  // 后台 worker 更新图片状态时，编辑器里选中的图片也要同步最新状态。
  if (!state.activeTask || !state.activeImage) return;
  const latest = (state.activeTask.images || []).find((image) => image.id === state.activeImage.id);
  if (latest) {
    state.activeImage = { ...state.activeImage, ...latest };
  }
}

function renderImageStrip(images) {
  // 已完成卡片进入编辑器；可续跑失败卡片显示明确操作按钮。
  const strip = $("imageStrip");
  if (!images.length) {
    strip.className = "image-strip empty-state";
    strip.textContent = "任务图片会显示在这里";
    return;
  }
  strip.className = "image-strip";
  strip.innerHTML = "";
  images.forEach((image) => {
    const card = document.createElement("article");
    card.className = `image-card ${state.activeImage?.id === image.id ? "active" : ""}`;
    const editable = isImageEditable(image);
    const resumable = canResumeImage(image);
    const statusText = editable
      ? "可校正"
      : `${stageNames[image.stage] || image.stage} ${Math.round(image.progress * 100)}%`;
    card.innerHTML = `<strong>${escapeHtml(image.relative_path)}</strong>
      <small>${statusText}</small>`;
    if (editable) {
      const action = document.createElement("button");
      action.className = "button secondary mini";
      action.type = "button";
      action.textContent = "打开校正";
      action.disabled = state.serverStopping || state.serverStopped;
      action.onclick = () => openImage(image);
      card.append(action);
    } else if (resumable) {
      const action = document.createElement("button");
      action.className = "button primary mini";
      action.type = "button";
      action.textContent = state.resumeImageId === image.id ? "续跑中..." : "从此页续跑";
      action.disabled = !!state.resumeImageId || state.serverStopping || state.serverStopped;
      action.onclick = () => resumeTaskFromImage(image);
      card.append(action);
    }
    strip.append(card);
  });
}

function normalizeOptionalFontSize(value) {
  // 空字号表示自动适配；非法值也按自动处理。
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 6) return null;
  return Math.round(parsed);
}

function normalizeMaskDilationOffset(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 20;
  return Math.max(0, Math.min(40, Math.round(parsed)));
}

function maskDilationStrengthText(value) {
  if (value <= 8) return "轻";
  if (value <= 16) return "偏轻";
  if (value <= 24) return "标准";
  if (value <= 32) return "偏强";
  return "强";
}

function formatMaskDilationLabel(value) {
  return `${value} / 40 · ${maskDilationStrengthText(value)}`;
}

function updateMaskDilationLabel(inputId, labelId) {
  const value = normalizeMaskDilationOffset($(inputId).value);
  $(inputId).value = value;
  $(labelId).textContent = formatMaskDilationLabel(value);
  return value;
}

function normalizeRegionAlignment(value) {
  return ["left", "center", "right"].includes(value) ? value : "left";
}

function normalizeRegionAngle(value) {
  let angle = Number(value);
  if (!Number.isFinite(angle)) angle = 0;
  while (angle > 180) angle -= 360;
  while (angle < -180) angle += 360;
  return Math.round(angle * 10) / 10;
}

function snapRegionAngle(value) {
  const angle = normalizeRegionAngle(value);
  let snapped = angle;
  let minDistance = ROTATION_SNAP_THRESHOLD + 1;
  ROTATION_SNAP_TARGETS.forEach((target) => {
    const distance = Math.abs(normalizeRegionAngle(angle - target));
    if (distance <= ROTATION_SNAP_THRESHOLD && distance < minDistance) {
      snapped = target;
      minDistance = distance;
    }
  });
  return normalizeRegionAngle(snapped);
}

function ensureRegionShape(region, index) {
  // 旧版 region JSON 可能缺少新字段，先补齐后再进入编辑流程。
  const fallback = region.render_bbox || region.ocr_bbox || region.bbox || [0, 0, 80, 80];
  region.index = index;
  region.ocr_bbox = region.ocr_bbox || fallback;
  region.render_bbox = region.render_bbox || fallback;
  region.bbox = region.render_bbox;
  region.enabled = region.enabled !== false;
  region.font_size = normalizeOptionalFontSize(region.font_size);
  region.direction = region.direction || "auto";
  region.alignment = normalizeRegionAlignment(region.alignment);
  region.foreground = region.foreground || "#000000";
  region.outline = region.outline || "#FFFFFF";
  region.text = region.text || "";
  region.translation = region.translation || "";
  region.angle = normalizeRegionAngle(region.angle);
  region.mask_dilation_offset = normalizeMaskDilationOffset(region.mask_dilation_offset);
  return region;
}

function bboxField() {
  // OCR 模式编辑识别框；译文框模式编辑嵌字位置。
  return state.editorMode === "ocr" ? "ocr_bbox" : "render_bbox";
}

function getRegionBBox(region) {
  return [...(region[bboxField()] || region.bbox || [0, 0, 80, 80])];
}

function getMaskPreviewBBox(region) {
  // 去字强度作用在 OCR 框对应的掩膜外扩上，预览也固定按 OCR 框计算。
  const source = [...(region.ocr_bbox || region.bbox || region.render_bbox || [0, 0, 80, 80])];
  const padding = normalizeMaskDilationOffset(region.mask_dilation_offset);
  return clampBBox([
    source[0] - padding,
    source[1] - padding,
    source[2] + padding,
    source[3] + padding,
  ]);
}

function imagePixelSize() {
  const image = $("resultImage");
  return { width: image.naturalWidth || 1, height: image.naturalHeight || 1 };
}

function clampBBox(bbox) {
  // 所有框坐标都以原图像素保存，而不是按当前屏幕缩放尺寸保存。
  const { width, height } = imagePixelSize();
  let [x1, y1, x2, y2] = bbox.map((value) => Math.round(Number(value) || 0));
  x1 = Math.max(0, Math.min(width, x1));
  x2 = Math.max(0, Math.min(width, x2));
  y1 = Math.max(0, Math.min(height, y1));
  y2 = Math.max(0, Math.min(height, y2));
  [x1, x2] = x1 <= x2 ? [x1, x2] : [x2, x1];
  [y1, y2] = y1 <= y2 ? [y1, y2] : [y2, y1];
  if (x2 - x1 < 2) x2 = Math.min(width, x1 + 2);
  if (y2 - y1 < 2) y2 = Math.min(height, y1 + 2);
  return [x1, y1, x2, y2];
}

function markOcrDirty(index) {
  state.dirtyOcrIndices.add(index);
}

function markMaskDirty(index) {
  state.dirtyMaskIndices.add(index);
}

function setRegionBBox(region, bbox) {
  // OCR 框一旦移动，就必须重新 OCR 才能信任后续渲染结果。
  const next = clampBBox(bbox);
  const current = region[bboxField()] || region.bbox || [];
  const changed = current.length !== 4 || current.some((value, index) => value !== next[index]);
  if (state.editorMode === "ocr") {
    region.ocr_bbox = next;
    if (changed) markOcrDirty(region.index);
  } else {
    region.render_bbox = next;
    region.bbox = next;
  }
  updateBBoxInputs(region);
}

function setRegionAngle(region, angle) {
  region.angle = normalizeRegionAngle(angle);
}

function editorImageUrl() {
  // 重嵌字或重处理后给图片 URL 加时间戳，避免浏览器继续用旧缓存。
  if (!state.activeImage) return "";
  const url = state.editorMode === "ocr"
    ? state.activeImage.original_url
    : state.activeImage.result_url;
  return `${url}?t=${Date.now()}`;
}

function loadEditorImage() {
  const image = $("resultImage");
  image.onload = renderOverlays;
  image.src = editorImageUrl();
  $("canvasModeLabel").textContent = state.editorMode === "ocr"
    ? "原图 OCR 识别框"
    : "翻译结果译文框";
  $("ocrModeButton").classList.toggle("active", state.editorMode === "ocr");
  $("renderModeButton").classList.toggle("active", state.editorMode === "render");
}

function editorPlaceholder() {
  return document.querySelector(".editor-placeholder");
}

function updateEditorPlaceholder() {
  const placeholder = editorPlaceholder();
  if (!placeholder) return;
  placeholder.textContent = state.regions.length
    ? "点击图片中的文本区域进行编辑"
    : "这张图片没有可编辑文本区域";
}

function setEditorMode(mode) {
  // 切换 OCR/译文框模式前，先把表单改动同步回内存状态。
  syncActiveRegion();
  const active = state.activeRegion;
  state.activeRegion = null;
  state.editorMode = mode;
  loadEditorImage();
  if (active !== null) selectRegion(active, { sync: false });
}

async function openImage(image) {
  // 只有已完成图片才有可编辑的 region JSON 和干净底图。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能继续校正";
    return;
  }
  if (!isImageEditable(image)) {
    $("editorMessage").textContent = "这张图片还在处理中，完成后即可校正";
    return;
  }
  state.activeImage = image;
  state.activeRegion = null;
  state.dirtyOcrIndices.clear();
  state.dirtyMaskIndices.clear();
  try {
    const data = await api(image.regions_url);
    state.regions = data.regions.map(ensureRegionShape);
    $("openOriginal").href = image.original_url;
    $("canvasWrap").classList.add("ready");
    $("editorFields").classList.add("hidden");
    updateEditorPlaceholder();
    editorPlaceholder().classList.remove("hidden");
    $("editorMessage").textContent = "";
    loadEditorImage();
    renderImageStrip(state.activeTask.images);
  } catch (error) {
    $("editorMessage").textContent = error.message;
  }
}

function renderOverlays() {
  // 覆盖层按当前显示尺寸绘制，但内部坐标始终保持原图像素。
  const image = $("resultImage");
  const overlay = $("regionOverlay");
  if (!image.naturalWidth || !state.activeImage) return;
  const rect = image.getBoundingClientRect();
  const wrapRect = $("canvasWrap").getBoundingClientRect();
  const scaleX = rect.width / image.naturalWidth;
  const scaleY = rect.height / image.naturalHeight;
  overlay.style.left = `${rect.left - wrapRect.left + $("canvasWrap").scrollLeft}px`;
  overlay.style.top = `${rect.top - wrapRect.top + $("canvasWrap").scrollTop}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.innerHTML = "";
  state.regions.forEach((region, index) => {
    ensureRegionShape(region, index);
    if (state.activeRegion === region.index && region.enabled) {
      // 预览层只做前端提示，不影响真实保存值和后端去字逻辑。
      const [px1, py1, px2, py2] = getMaskPreviewBBox(region);
      const preview = document.createElement("div");
      preview.className = "mask-preview";
      preview.style.left = `${px1 * scaleX}px`;
      preview.style.top = `${py1 * scaleY}px`;
      preview.style.width = `${Math.max(12, (px2 - px1) * scaleX)}px`;
      preview.style.height = `${Math.max(12, (py2 - py1) * scaleY)}px`;
      const label = document.createElement("span");
      label.className = "mask-preview-label";
      label.textContent = "预估涂抹范围";
      preview.append(label);
      overlay.append(preview);
    }
    const [x1, y1, x2, y2] = getRegionBBox(region);
    const width = Math.max(12, (x2 - x1) * scaleX);
    const height = Math.max(12, (y2 - y1) * scaleY);
    const box = document.createElement("button");
    const classes = [
      "region-box",
      state.editorMode === "render" ? "render" : "",
      state.activeRegion === region.index ? "active" : "",
      region.enabled ? "" : "disabled",
    ].filter(Boolean);
    box.className = classes.join(" ");
    box.style.left = `${x1 * scaleX}px`;
    box.style.top = `${y1 * scaleY}px`;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    box.style.transform = state.editorMode === "render" ? `rotate(${region.angle}deg)` : "";
    box.textContent = region.index + 1;
    box.onpointerdown = (event) => startBoxDrag(event, region.index, null);
    if (state.activeRegion === region.index) {
      ["nw", "n", "ne", "e", "se", "s", "sw", "w"].forEach((handle) => {
        const grip = document.createElement("span");
        grip.className = `resize-handle ${handle}`;
        grip.onpointerdown = (event) => startBoxDrag(event, region.index, handle);
        box.append(grip);
      });
    }
    overlay.append(box);
    if (state.editorMode === "render" && state.activeRegion === region.index) {
      const rotate = document.createElement("span");
      const angleRad = region.angle * Math.PI / 180;
      const centerX = x1 * scaleX + width / 2;
      const centerY = y1 * scaleY + height / 2;
      const offset = Math.max(height / 2 + 28, 38);
      rotate.className = "rotate-handle";
      rotate.title = "拖拽旋转译文框";
      rotate.style.left = `${centerX + Math.sin(angleRad) * offset}px`;
      rotate.style.top = `${centerY - Math.cos(angleRad) * offset}px`;
      rotate.style.transform = `translate(-50%, -50%) rotate(${region.angle}deg)`;
      rotate.onpointerdown = (event) => startBoxDrag(event, region.index, "rotate");
      overlay.append(rotate);
    }
  });
}

function pointerToPixel(event) {
  // 把浏览器指针坐标反算成原图中的像素坐标。
  const image = $("resultImage");
  const rect = image.getBoundingClientRect();
  const scaleX = image.naturalWidth / rect.width;
  const scaleY = image.naturalHeight / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function startBoxDrag(event, index, handle) {
  // 同一套拖拽状态同时支持移动、缩放和译文框旋转。
  event.preventDefault();
  event.stopPropagation();
  selectRegion(index);
  state.dragState = {
    index,
    handle,
    startPoint: pointerToPixel(event),
    startBBox: getRegionBBox(state.regions[index]),
  };
  window.addEventListener("pointermove", moveBoxDrag);
  window.addEventListener("pointerup", endBoxDrag, { once: true });
}

function moveBoxDrag(event) {
  if (!state.dragState) return;
  const region = state.regions[state.dragState.index];
  const point = pointerToPixel(event);
  const dx = point.x - state.dragState.startPoint.x;
  const dy = point.y - state.dragState.startPoint.y;
  let [x1, y1, x2, y2] = state.dragState.startBBox;
  const handle = state.dragState.handle;
  if (handle === "rotate") {
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;
    const rawAngle = Math.atan2(point.y - centerY, point.x - centerX) * 180 / Math.PI + 90;
    setRegionAngle(region, snapRegionAngle(rawAngle));
  } else if (!handle) {
    x1 += dx; x2 += dx; y1 += dy; y2 += dy;
  } else {
    if (handle.includes("w")) x1 += dx;
    if (handle.includes("e")) x2 += dx;
    if (handle.includes("n")) y1 += dy;
    if (handle.includes("s")) y2 += dy;
  }
  setRegionBBox(region, [x1, y1, x2, y2]);
  renderOverlays();
}

function endBoxDrag() {
  window.removeEventListener("pointermove", moveBoxDrag);
  state.dragState = null;
}

function updateBBoxInputs(region) {
  const [x1, y1, x2, y2] = getRegionBBox(region);
  $("regionX").value = x1;
  $("regionY").value = y1;
  $("regionW").value = x2 - x1;
  $("regionH").value = y2 - y1;
  $("regionBox").textContent = `${x1}, ${y1}, ${x2}, ${y2}`;
}

function syncBoxInputs() {
  if (state.activeRegion === null) return;
  const region = state.regions[state.activeRegion];
  const x = Number($("regionX").value);
  const y = Number($("regionY").value);
  const width = Number($("regionW").value);
  const height = Number($("regionH").value);
  setRegionBBox(region, [x, y, x + width, y + height]);
}

function populateRegionForm(index) {
  // 侧边栏一次只编辑一个区域；切换前靠 syncActiveRegion 写回。
  state.activeRegion = index;
  const region = state.regions[index];
  if (!region) return;
  editorPlaceholder().classList.add("hidden");
  $("editorFields").classList.remove("hidden");
  $("regionNumber").textContent = `区域 #${index + 1}`;
  $("regionText").value = region.text;
  $("regionTranslation").value = region.translation;
  $("regionFontSize").value = region.font_size ?? "";
  $("regionDirection").value = ["auto", "horizontal", "vertical"].includes(region.direction) ? region.direction : "auto";
  $("regionAlignment").value = normalizeRegionAlignment(region.alignment);
  $("regionForeground").value = region.foreground;
  $("regionOutline").value = region.outline;
  $("regionMaskDilationOffset").value = normalizeMaskDilationOffset(region.mask_dilation_offset);
  updateMaskDilationLabel("regionMaskDilationOffset", "regionMaskDilationValue");
  $("toggleRegionButton").textContent = region.enabled ? "禁用区域" : "恢复区域";
  updateBBoxInputs(region);
  renderOverlays();
}

function selectRegion(index, options = {}) {
  if (options.sync !== false) syncActiveRegion();
  populateRegionForm(index);
}

function syncActiveRegion() {
  // 在保存、切模式、切区域之前，把表单值写回 state。
  if (state.activeRegion === null) return;
  const region = state.regions[state.activeRegion];
  if (!region) return;
  region.text = $("regionText").value;
  region.translation = $("regionTranslation").value;
  region.font_size = normalizeOptionalFontSize($("regionFontSize").value);
  region.direction = $("regionDirection").value;
  region.alignment = $("regionAlignment").value;
  region.foreground = $("regionForeground").value;
  region.outline = $("regionOutline").value;
  region.mask_dilation_offset = updateMaskDilationLabel("regionMaskDilationOffset", "regionMaskDilationValue");
  syncBoxInputs();
}

function addRegion() {
  // 新增人工 OCR 框默认居中，并立即标记为需要 OCR 和重做掩膜。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能新增 OCR 框";
    return;
  }
  if (!state.activeImage || !$("resultImage").naturalWidth) return;
  if (!isImageEditable(state.activeImage)) {
    $("editorMessage").textContent = "这张图片还在处理中，完成后即可新增 OCR 框";
    return;
  }
  syncActiveRegion();
  state.editorMode = "ocr";
  const { width, height } = imagePixelSize();
  const boxWidth = Math.max(80, Math.round(width * 0.12));
  const boxHeight = Math.max(80, Math.round(height * 0.12));
  const x1 = Math.round((width - boxWidth) / 2);
  const y1 = Math.round((height - boxHeight) / 2);
  const bbox = [x1, y1, x1 + boxWidth, y1 + boxHeight];
  const index = state.regions.length;
  state.regions.push(ensureRegionShape({
    index,
    bbox,
    ocr_bbox: bbox,
    render_bbox: bbox,
    enabled: true,
    text: "",
    translation: "",
    angle: 0,
    font_size: null,
    direction: "auto",
    alignment: "left",
    foreground: "#000000",
    outline: "#FFFFFF",
    mask_dilation_offset: updateMaskDilationLabel("globalMaskDilationOffset", "globalMaskDilationValue"),
  }, index));
  markOcrDirty(index);
  markMaskDirty(index);
  loadEditorImage();
  selectRegion(index);
}

function toggleRegion() {
  // 禁用区域仍保留在 JSON 里，但不会参与去字和嵌字。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能修改区域";
    return;
  }
  if (state.activeRegion === null) return;
  const region = state.regions[state.activeRegion];
  region.enabled = !region.enabled;
  markOcrDirty(region.index);
  markMaskDirty(region.index);
  $("toggleRegionButton").textContent = region.enabled ? "禁用区域" : "恢复区域";
  renderOverlays();
}

async function reprocessRegions() {
  // 文字重识别默认复用干净底图；涂抹变化才会重新去字。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能重新识别";
    return;
  }
  if (!state.activeImage) return;
  if (!isImageEditable(state.activeImage)) {
    $("editorMessage").textContent = "这张图片还在处理中，完成后即可重新识别";
    return;
  }
  syncActiveRegion();
  const changed = [...state.dirtyOcrIndices].filter((index) => state.regions[index]);
  const maskChanged = [...state.dirtyMaskIndices].filter((index) => state.regions[index]);
  if (!changed.length && !maskChanged.length) {
    $("editorMessage").textContent = "没有需要重新识别文字或重新去字的 OCR 框";
    return;
  }
  $("reprocessButton").disabled = true;
  $("rerenderButton").disabled = true;
  const willRepaint = maskChanged.length > 0;
  let message = "正在重新识别文字、翻译并嵌字...";
  if (changed.length && willRepaint) {
    message = "正在重新识别文字，并按新的涂抹强度重新去字后嵌字...";
  } else if (willRepaint) {
    message = "正在按新的涂抹强度重新去字，并嵌入译文...";
  }
  $("editorMessage").textContent = message;
  try {
    const data = await api(`/api/images/${state.activeImage.id}/reprocess-regions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        regions: state.regions,
        changed_indices: changed,
        mask_changed_indices: maskChanged,
      }),
    });
    state.regions = data.regions.map(ensureRegionShape);
    state.activeImage = {
      ...state.activeImage,
      status: "completed",
      stage: "saved",
      progress: 1,
      result_url: data.result_url || state.activeImage.result_url,
    };
    state.dirtyOcrIndices.clear();
    state.dirtyMaskIndices.clear();
    if (state.activeRegion !== null && state.activeRegion >= state.regions.length) {
      state.activeRegion = state.regions.length ? state.regions.length - 1 : null;
    }
    updateEditorPlaceholder();
    loadEditorImage();
    if (state.activeRegion !== null) {
      selectRegion(state.activeRegion, { sync: false });
    } else {
      $("editorFields").classList.add("hidden");
      editorPlaceholder().classList.remove("hidden");
    }
    $("editorMessage").textContent = "OCR 框重处理完成";
  } catch (error) {
    $("editorMessage").textContent = error.message;
  } finally {
    $("reprocessButton").disabled = state.serverStopping || state.serverStopped;
    $("rerenderButton").disabled = state.serverStopping || state.serverStopped;
  }
}

async function rerenderImage() {
  // rerender 直接信任现有 OCR/译文，只重画改过的文本和排版。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能重新嵌字";
    return;
  }
  if (!state.activeImage) return;
  if (!isImageEditable(state.activeImage)) {
    $("editorMessage").textContent = "这张图片还在处理中，完成后即可重新嵌字";
    return;
  }
  syncActiveRegion();
  $("rerenderButton").disabled = true;
  $("editorMessage").textContent = "正在重新嵌字...";
  try {
    const data = await api(`/api/images/${state.activeImage.id}/rerender`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regions: state.regions }),
    });
    state.regions = data.regions.map(ensureRegionShape);
    state.activeImage = {
      ...state.activeImage,
      status: "completed",
      stage: "saved",
      progress: 1,
      result_url: data.result_url || state.activeImage.result_url,
    };
    updateEditorPlaceholder();
    loadEditorImage();
    if (state.activeRegion !== null) {
      selectRegion(state.activeRegion, { sync: false });
    } else {
      $("editorFields").classList.add("hidden");
      editorPlaceholder().classList.remove("hidden");
    }
    $("editorMessage").textContent = "重新嵌字完成";
  } catch (error) {
    $("editorMessage").textContent = error.message;
  } finally {
    $("rerenderButton").disabled = state.serverStopping || state.serverStopped;
  }
}

async function resumeTaskFromImage(image) {
  // 后端决定从这里往后哪些页面需要重新入队，前端只负责选起点。
  if (state.serverStopping || state.serverStopped) {
    $("formMessage").textContent = "服务已停止，重新启动后才能续跑失败页面";
    return;
  }
  if (!state.activeTask || !canResumeImage(image) || state.resumeImageId) return;
  state.resumeImageId = image.id;
  $("formMessage").textContent = "";
  renderImageStrip(state.activeTask.images || []);
  try {
    const task = await api(`/api/tasks/${state.activeTask.id}/resume-from-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: image.id }),
    });
    state.activeTask = task;
    if (state.activeTask?.id !== task.id || !state.eventSource) {
      activateTask(task.id);
    } else {
      renderTask();
    }
    await loadTasks();
  } catch (error) {
    $("formMessage").textContent = error.message;
  } finally {
    state.resumeImageId = null;
    if (state.activeTask) renderImageStrip(state.activeTask.images || []);
  }
}

function setRegionTranslateButtonsDisabled(disabled) {
  $("machineTranslateButton").disabled = disabled;
  $("ollamaTranslateButton").disabled = disabled;
}

async function translateActiveRegion(mode) {
  // 单区域翻译只回填文本框，是否保存仍由用户决定。
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能重新翻译";
    return;
  }
  if (!state.activeImage || state.activeRegion === null) return;
  if (!isImageEditable(state.activeImage)) {
    $("editorMessage").textContent = "这张图片还在处理中，完成后即可重新翻译";
    return;
  }
  syncActiveRegion();
  const region = state.regions[state.activeRegion];
  const text = (region.text || "").trim();
  if (!text) {
    $("editorMessage").textContent = "OCR 原文为空，无法翻译";
    return;
  }
  setRegionTranslateButtonsDisabled(true);
  $("editorMessage").textContent = mode === "machine" ? "正在调用机器翻译..." : "正在调用 Ollama 翻译...";
  try {
    const data = await api(`/api/images/${state.activeImage.id}/translate-region`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, text }),
    });
    region.translation = data.translation || "";
    $("regionTranslation").value = region.translation;
    $("editorMessage").textContent = "译文已更新，确认后可保存校正并重新嵌字";
  } catch (error) {
    $("editorMessage").textContent = error.message;
  } finally {
    setRegionTranslateButtonsDisabled(state.serverStopping || state.serverStopped);
  }
}

function renderLogs() {
  // 限制日志 DOM 长度，避免越跑越卡，同时保留自动跟随到底部的体验。
  const windowEl = $("logWindow");
  const logs = [...state.logs.values()].slice(-500);
  const wasNearBottom =
    windowEl.scrollHeight - windowEl.scrollTop - windowEl.clientHeight <= 32;
  if (!logs.length) {
    windowEl.innerHTML = '<div class="empty-state">运行日志会显示在这里</div>';
    return;
  }
  windowEl.innerHTML = "";
  logs.forEach((log) => {
    const entry = document.createElement("div");
    entry.className = `log-entry ${log.level.toLowerCase()}`;
    const time = new Date(log.created_at).toLocaleTimeString();
    entry.innerHTML = `<span class="time">${time}</span> <span class="stage">[${stageNames[log.stage] || log.stage}]</span> ${escapeHtml(log.message)}`;
    if (log.details?.pairs) {
      log.details.pairs.forEach((pair) => {
        const line = document.createElement("div");
        line.className = "log-pair";
        line.textContent = `OCR: ${pair.source}\n译文: ${pair.translation}`;
        entry.append(line);
      });
    } else if (log.details?.texts) {
      const line = document.createElement("div");
      line.className = "log-pair";
      line.textContent = log.details.texts.map((text, i) => `#${i + 1} ${text}`).join("\n");
      entry.append(line);
    } else if (log.details?.traceback || log.details?.error) {
      const line = document.createElement("div");
      line.className = "log-pair";
      line.textContent = log.details.traceback || log.details.error;
      entry.append(line);
    }
    windowEl.append(entry);
  });
  if (state.logsAutoFollow || wasNearBottom) {
    windowEl.scrollTop = windowEl.scrollHeight;
    state.logsAutoFollow = true;
  }
}

async function waitForServiceStop(attempt = 0) {
  // 优雅停机后轮询 health，直到本地进程真正退出为止。
  try {
    const health = await api("/api/health");
    if (!health.shutting_down && attempt > 0) {
      setServiceBanner("停止请求已发送，正在等待服务退出...", "warning");
    }
  } catch {
    markServiceStopped();
    return;
  }
  if (attempt >= 240) {
    setServiceBanner("停止请求已发送，但服务仍未完全退出。可以稍候片刻，或双击停止脚本。", "warning");
    $("stopServiceButton").disabled = false;
    $("stopServiceButton").textContent = "再次检查停止状态";
    return;
  }
  state.shutdownPoll = setTimeout(() => {
    waitForServiceStop(attempt + 1);
  }, 500);
}

async function stopService() {
  // 服务端限制为本机停机，并且会优先处理完当前图片。
  if (state.serverStopped) return;
  if (state.serverStopping) {
    await waitForServiceStop();
    return;
  }
  const confirmed = window.confirm("停止服务后，当前这张图片会处理完成，随后整个本地系统关闭。确定继续吗？");
  if (!confirmed) return;
  markServiceStopping("停止请求已发送，服务会在当前这张图片处理完成后关闭。");
  try {
    const result = await api("/api/system/shutdown", { method: "POST" });
    setServiceBanner(result.message || "停止请求已受理，正在关闭服务。", "warning");
    await waitForServiceStop();
  } catch (error) {
    state.serverStopping = false;
    $("stopServiceButton").disabled = false;
    $("stopServiceButton").textContent = "停止服务";
    setMutationControlsDisabled(false);
    setServiceBanner(error.message, "error");
    await loadHealth();
  }
}

async function saveSettings() {
  // 密码框留空表示保留已有密钥，而不是把密钥清掉。
  if (state.serverStopping || state.serverStopped) {
    $("settingsMessage").textContent = "服务已停止，重新启动后才能保存设置";
    return;
  }
  const payload = {
    ollama_base_url: $("ollamaBaseUrl").value,
    microsoft_region: $("microsoftRegion").value,
    microsoft_endpoint: $("microsoftEndpoint").value,
  };
  if ($("googleApiKey").value) payload.google_api_key = $("googleApiKey").value;
  if ($("microsoftApiKey").value) payload.microsoft_api_key = $("microsoftApiKey").value;
  try {
    state.settings = await api("/api/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    $("settingsMessage").textContent = "设置已保存到本机私有配置文件";
    $("googleApiKey").value = "";
    $("microsoftApiKey").value = "";
    await loadSettings();
    await loadModels();
  } catch (error) { $("settingsMessage").textContent = error.message; }
}

function escapeHtml(value) {
  // 所有通过 innerHTML 插入的任务名和路径都先走这里转义。
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}

function bindEvents() {
  // 所有事件绑定集中在这里，后续排查 DOM 交互会更清晰。
  $("singleButton").onclick = () => $("singleInput").click();
  $("folderButton").onclick = () => $("folderInput").click();
  $("singleInput").onchange = (event) => setFiles(event.target.files);
  $("folderInput").onchange = (event) => setFiles(event.target.files);
  $("provider").onchange = updateProviderFields;
  $("polishWithOllama").onchange = updateProviderFields;
  $("ollamaModel").onchange = updateModelMeta;
  $("refreshModels").onclick = loadModels;
  $("startButton").onclick = createTask;
  $("refreshTasks").onclick = loadTasks;
  $("stopServiceButton").onclick = stopService;
  $("ocrModeButton").onclick = () => setEditorMode("ocr");
  $("renderModeButton").onclick = () => setEditorMode("render");
  $("addRegionButton").onclick = addRegion;
  $("toggleRegionButton").onclick = toggleRegion;
  $("reprocessButton").onclick = reprocessRegions;
  $("rerenderButton").onclick = rerenderImage;
  $("machineTranslateButton").onclick = () => translateActiveRegion("machine");
  $("ollamaTranslateButton").onclick = () => translateActiveRegion("ollama");
  $("globalMaskDilationOffset").oninput = () => {
    updateMaskDilationLabel("globalMaskDilationOffset", "globalMaskDilationValue");
  };
  $("regionMaskDilationOffset").oninput = () => {
    const value = updateMaskDilationLabel("regionMaskDilationOffset", "regionMaskDilationValue");
    if (state.activeRegion === null) return;
    const region = state.regions[state.activeRegion];
    if (!region) return;
    if (region.mask_dilation_offset !== value) {
      region.mask_dilation_offset = value;
      markMaskDirty(region.index);
    }
    renderOverlays();
  };
  $("clearLogView").onclick = () => {
    state.logs.clear();
    state.logsAutoFollow = true;
    $("logWindow").innerHTML = '<div class="empty-state">日志显示已清空</div>';
  };
  $("settingsButton").onclick = () => $("settingsDialog").showModal();
  $("saveSettings").onclick = saveSettings;
  ["regionX", "regionY", "regionW", "regionH"].forEach((id) => {
    $(id).onchange = () => {
      syncBoxInputs();
      renderOverlays();
    };
  });
  window.addEventListener("resize", renderOverlays);
  $("canvasWrap").addEventListener("scroll", renderOverlays);
  $("logWindow").addEventListener("scroll", (event) => {
    const target = event.currentTarget;
    state.logsAutoFollow =
      target.scrollHeight - target.scrollTop - target.clientHeight <= 32;
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
      $("editorTab").classList.toggle("active", tab.dataset.tab === "editor");
      $("logsTab").classList.toggle("active", tab.dataset.tab === "logs");
    };
  });
}

async function init() {
  // 初次加载时，这几类服务状态彼此独立，可以并行拉取。
  bindEvents();
  updateMaskDilationLabel("globalMaskDilationOffset", "globalMaskDilationValue");
  updateMaskDilationLabel("regionMaskDilationOffset", "regionMaskDilationValue");
  updateEditorPlaceholder();
  updateProviderFields();
  await Promise.all([loadHealth(), loadSettings(), loadTasks()]);
  if (!state.serverStopping && !state.serverStopped) {
    await loadModels();
  }
}

init();
