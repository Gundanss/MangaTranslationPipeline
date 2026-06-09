const $ = (id) => document.getElementById(id);

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
  dragState: null,
  logsAutoFollow: true,
  serverStopping: false,
  serverStopped: false,
  shutdownPoll: null,
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
  [
    "startButton",
    "refreshModels",
    "addRegionButton",
    "toggleRegionButton",
    "reprocessButton",
    "rerenderButton",
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
  const provider = $("provider").value;
  $("ollamaField").classList.toggle("hidden", provider !== "ollama");
  $("polishField").classList.toggle("hidden", provider === "ollama");
  $("polishModelField").classList.toggle("hidden", provider === "ollama" || !$("polishWithOllama").checked);
}

function setFiles(fileList) {
  const allowed = /\.(jpe?g|png|webp)$/i;
  state.files = [...fileList].filter((file) => allowed.test(file.name));
  state.relativePaths = state.files.map((file) => file.webkitRelativePath || file.name);
  const total = state.files.reduce((sum, file) => sum + file.size, 0);
  $("fileSummary").textContent = state.files.length
    ? `已选择 ${state.files.length} 张图片，共 ${formatBytes(total)}`
    : "没有找到支持的 JPG、PNG 或 WebP 图片";
}

async function createTask() {
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

function syncActiveImageFromTask() {
  if (!state.activeTask || !state.activeImage) return;
  const latest = (state.activeTask.images || []).find((image) => image.id === state.activeImage.id);
  if (latest) {
    state.activeImage = { ...state.activeImage, ...latest };
  }
}

function renderImageStrip(images) {
  const strip = $("imageStrip");
  if (!images.length) {
    strip.className = "image-strip empty-state";
    strip.textContent = "完成的图片会显示在这里";
    return;
  }
  strip.className = "image-strip";
  strip.innerHTML = "";
  images.forEach((image) => {
    const button = document.createElement("button");
    button.className = `image-card ${state.activeImage?.id === image.id ? "active" : ""}`;
    const editable = isImageEditable(image);
    button.innerHTML = `<strong>${escapeHtml(image.relative_path)}</strong>
      <small>${editable ? "可校正" : `${stageNames[image.stage] || image.stage} ${Math.round(image.progress * 100)}%`}</small>`;
    button.disabled = !editable;
    button.onclick = () => openImage(image);
    strip.append(button);
  });
}

function normalizeOptionalFontSize(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 6) return null;
  return Math.round(parsed);
}

function ensureRegionShape(region, index) {
  const fallback = region.render_bbox || region.ocr_bbox || region.bbox || [0, 0, 80, 80];
  region.index = index;
  region.ocr_bbox = region.ocr_bbox || fallback;
  region.render_bbox = region.render_bbox || fallback;
  region.bbox = region.render_bbox;
  region.enabled = region.enabled !== false;
  region.font_size = normalizeOptionalFontSize(region.font_size);
  region.direction = region.direction || "auto";
  region.alignment = region.alignment || "left";
  region.foreground = region.foreground || "#000000";
  region.outline = region.outline || "#FFFFFF";
  region.text = region.text || "";
  region.translation = region.translation || "";
  return region;
}

function bboxField() {
  return state.editorMode === "ocr" ? "ocr_bbox" : "render_bbox";
}

function getRegionBBox(region) {
  return [...(region[bboxField()] || region.bbox || [0, 0, 80, 80])];
}

function imagePixelSize() {
  const image = $("resultImage");
  return { width: image.naturalWidth || 1, height: image.naturalHeight || 1 };
}

function clampBBox(bbox) {
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

function setRegionBBox(region, bbox) {
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

function editorImageUrl() {
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
  syncActiveRegion();
  const active = state.activeRegion;
  state.activeRegion = null;
  state.editorMode = mode;
  loadEditorImage();
  if (active !== null) selectRegion(active, { sync: false });
}

async function openImage(image) {
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
    const [x1, y1, x2, y2] = getRegionBBox(region);
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
    box.style.width = `${Math.max(12, (x2 - x1) * scaleX)}px`;
    box.style.height = `${Math.max(12, (y2 - y1) * scaleY)}px`;
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
  });
}

function pointerToPixel(event) {
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
  if (!handle) {
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
  $("regionAlignment").value = ["auto", "left", "center", "right"].includes(region.alignment) ? region.alignment : "left";
  $("regionForeground").value = region.foreground;
  $("regionOutline").value = region.outline;
  $("toggleRegionButton").textContent = region.enabled ? "禁用区域" : "恢复区域";
  updateBBoxInputs(region);
  renderOverlays();
}

function selectRegion(index, options = {}) {
  if (options.sync !== false) syncActiveRegion();
  populateRegionForm(index);
}

function syncActiveRegion() {
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
  syncBoxInputs();
}

function addRegion() {
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
    font_size: null,
    direction: "auto",
    alignment: "left",
    foreground: "#000000",
    outline: "#FFFFFF",
  }, index));
  markOcrDirty(index);
  loadEditorImage();
  selectRegion(index);
}

function toggleRegion() {
  if (state.serverStopping || state.serverStopped) {
    $("editorMessage").textContent = "服务已停止，重新启动后才能修改区域";
    return;
  }
  if (state.activeRegion === null) return;
  const region = state.regions[state.activeRegion];
  region.enabled = !region.enabled;
  markOcrDirty(region.index);
  $("toggleRegionButton").textContent = region.enabled ? "禁用区域" : "恢复区域";
  renderOverlays();
}

async function reprocessRegions() {
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
  if (!changed.length) {
    $("editorMessage").textContent = "没有需要重新识别的 OCR 框";
    return;
  }
  $("reprocessButton").disabled = true;
  $("rerenderButton").disabled = true;
  $("editorMessage").textContent = "正在重新识别、翻译、去字并嵌字...";
  try {
    const data = await api(`/api/images/${state.activeImage.id}/reprocess-regions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regions: state.regions, changed_indices: changed }),
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

function renderLogs() {
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
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}

function bindEvents() {
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
  bindEvents();
  updateEditorPlaceholder();
  updateProviderFields();
  await Promise.all([loadHealth(), loadSettings(), loadTasks()]);
  if (!state.serverStopping && !state.serverStopped) {
    await loadModels();
  }
}

init();
