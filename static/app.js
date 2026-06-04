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
};

const stageNames = {
  queued: "排队中", starting: "准备中", running_pre_translation_hooks: "初始化",
  "mps-fallback": "MPS 失败，切换 CPU",
  detection: "文字检测", ocr: "OCR 识别", textline_merge: "合并文本区域",
  translating: "翻译中", "after-translating": "翻译校验", "mask-generation": "生成去字掩膜",
  inpainting: "去除原文", rendering: "自动嵌字", saved: "保存结果",
  finished: "已完成", error: "处理失败", rerender: "重新嵌字", retry: "翻译重试",
};
const statusNames = {
  queued: "排队中", running: "处理中", completed: "已完成",
  completed_with_errors: "部分失败", failed: "失败", idle: "空闲",
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

async function loadHealth() {
  try {
    const health = await api("/api/health");
    const ready = health.core_present && Object.values(health.models).every(Boolean);
    $("healthChip").textContent = ready ? "图像模型已就绪" : "需要首次安装";
    $("healthChip").style.color = ready ? "#a9f0cf" : "#ffd08e";
  } catch { $("healthChip").textContent = "环境检查失败"; }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  $("ollamaBaseUrl").value = state.settings.ollama_base_url;
  $("microsoftRegion").value = state.settings.microsoft_region || "";
  $("microsoftEndpoint").value = state.settings.microsoft_endpoint || "";
  $("googleState").textContent = state.settings.google_configured ? "已保存密钥" : "尚未配置";
  $("microsoftState").textContent = state.settings.microsoft_configured ? "已保存密钥和区域" : "尚未完整配置";
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
    $("startButton").disabled = false;
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
  state.logs.clear();
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
  state.eventSource.onerror = () => {};
}

function renderTask() {
  const task = state.activeTask;
  if (!task) return;
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
    button.innerHTML = `<strong>${escapeHtml(image.relative_path)}</strong>
      <small>${image.status === "completed" ? "可校正" : `${stageNames[image.stage] || image.stage} ${Math.round(image.progress * 100)}%`}</small>`;
    button.disabled = image.status !== "completed";
    button.onclick = () => openImage(image);
    strip.append(button);
  });
}

async function openImage(image) {
  state.activeImage = image;
  state.activeRegion = null;
  try {
    const data = await api(image.regions_url);
    state.regions = data.regions;
    const resultImage = $("resultImage");
    resultImage.onload = renderOverlays;
    resultImage.src = `${image.result_url}?t=${Date.now()}`;
    $("openOriginal").href = image.original_url;
    $("canvasWrap").classList.add("ready");
    $("editorFields").classList.add("hidden");
    document.querySelector(".editor-placeholder").classList.remove("hidden");
    $("editorMessage").textContent = "";
    renderImageStrip(state.activeTask.images);
  } catch (error) {
    $("editorMessage").textContent = error.message;
  }
}

function renderOverlays() {
  const image = $("resultImage");
  const overlay = $("regionOverlay");
  const rect = image.getBoundingClientRect();
  const wrapRect = $("canvasWrap").getBoundingClientRect();
  const scaleX = rect.width / image.naturalWidth;
  const scaleY = rect.height / image.naturalHeight;
  overlay.style.left = `${rect.left - wrapRect.left + $("canvasWrap").scrollLeft}px`;
  overlay.style.top = `${rect.top - wrapRect.top + $("canvasWrap").scrollTop}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.innerHTML = "";
  state.regions.forEach((region) => {
    const [x1, y1, x2, y2] = region.bbox;
    const box = document.createElement("button");
    box.className = `region-box ${state.activeRegion === region.index ? "active" : ""}`;
    box.style.left = `${x1 * scaleX}px`;
    box.style.top = `${y1 * scaleY}px`;
    box.style.width = `${Math.max(12, (x2 - x1) * scaleX)}px`;
    box.style.height = `${Math.max(12, (y2 - y1) * scaleY)}px`;
    box.textContent = region.index + 1;
    box.onclick = () => selectRegion(region.index);
    overlay.append(box);
  });
}

function selectRegion(index) {
  syncActiveRegion();
  state.activeRegion = index;
  const region = state.regions[index];
  document.querySelector(".editor-placeholder").classList.add("hidden");
  $("editorFields").classList.remove("hidden");
  $("regionNumber").textContent = `区域 #${index + 1}`;
  $("regionBox").textContent = region.bbox.join(", ");
  $("regionText").value = region.text;
  $("regionTranslation").value = region.translation;
  $("regionFontSize").value = region.font_size;
  $("regionDirection").value = ["auto", "horizontal", "vertical"].includes(region.direction) ? region.direction : "auto";
  $("regionAlignment").value = ["auto", "left", "center", "right"].includes(region.alignment) ? region.alignment : "auto";
  $("regionForeground").value = region.foreground;
  $("regionOutline").value = region.outline;
  renderOverlays();
}

function syncActiveRegion() {
  if (state.activeRegion === null) return;
  const region = state.regions[state.activeRegion];
  region.text = $("regionText").value;
  region.translation = $("regionTranslation").value;
  region.font_size = Number($("regionFontSize").value);
  region.direction = $("regionDirection").value;
  region.alignment = $("regionAlignment").value;
  region.foreground = $("regionForeground").value;
  region.outline = $("regionOutline").value;
}

async function rerenderImage() {
  if (!state.activeImage) return;
  syncActiveRegion();
  $("rerenderButton").disabled = true;
  $("editorMessage").textContent = "正在重新嵌字...";
  try {
    const data = await api(`/api/images/${state.activeImage.id}/rerender`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regions: state.regions }),
    });
    state.regions = data.regions;
    $("resultImage").src = `${data.result_url}?t=${Date.now()}`;
    $("editorMessage").textContent = "重新嵌字完成";
  } catch (error) {
    $("editorMessage").textContent = error.message;
  } finally {
    $("rerenderButton").disabled = false;
  }
}

function renderLogs() {
  const windowEl = $("logWindow");
  const logs = [...state.logs.values()].slice(-500);
  if (!logs.length) return;
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
  windowEl.scrollTop = windowEl.scrollHeight;
}

async function saveSettings() {
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
  $("rerenderButton").onclick = rerenderImage;
  $("clearLogView").onclick = () => { state.logs.clear(); $("logWindow").innerHTML = '<div class="empty-state">日志显示已清空</div>'; };
  $("settingsButton").onclick = () => $("settingsDialog").showModal();
  $("saveSettings").onclick = saveSettings;
  window.addEventListener("resize", renderOverlays);
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
  updateProviderFields();
  await Promise.all([loadHealth(), loadSettings(), loadTasks()]);
  await loadModels();
}

init();
