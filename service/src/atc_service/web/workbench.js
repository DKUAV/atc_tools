import { inputSettings, mountDataset, renderInputSettings } from "./datasets.js";

const $ = (id) => document.getElementById(id);
const apiRoot = "/web/api";
const connectionError = "无法连接服务，请检查网络后重试。";
const stateText = {
  idle: "未开始", uploading: "上传中", queued: "排队中", running: "进行中",
  succeeded: "已完成", failed: "失败", cancelled: "已取消", pending: "未开始", skipped: "已跳过",
};
const operationText = { inspect_onnx: "模型检查", compile_om: "编译" };
const presetText = { "310b4-simulation": "310B4 · 模拟转换", "310b4-float": "310B4 · 浮点编译" };
const precisionText = { origin: "原始类型", mixed_float16: "Float16 · 自动混合", fp16: "Float16 · 优先 FP16", int8: "INT8 · 默认量化" };
const stageText = { prepare_calibration: "准备校准数据", quantize: "AMCT INT8 量化", compile: "编译模型",
  prepare_validation: "准备验证数据", compare: "精度对比", profile: "msprof 性能采集" };
const terminal = new Set(["succeeded", "failed", "cancelled"]);
let session;
let model;
let busy = false;
let attemptKey;
let viewing;
let timer;
const dataset = {
  calibration: mountDataset($("calibration-set"), "校准集", changed),
  validation: mountDataset($("validation-set"), "验证集", changed),
};

function message(value) {
  $("notice").textContent = value;
  $("notice").hidden = !value;
}

async function api(path, options = {}) {
  const activeSession = session;
  const headers = new Headers(options.headers);
  if (activeSession && options.method === "POST") headers.set("X-ATC-CSRF", activeSession.csrf);
  let response;
  try {
    response = await fetch(apiRoot + path, { ...options, headers, cache: "no-store" });
  } catch (error) {
    if (error instanceof TypeError) throw new Error(connectionError, { cause: error });
    throw error;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 401 && path !== "/login" && activeSession && session === activeSession) {
      signInView("会话已结束，请重新登录。");
    }
    throw new Error(typeof body.detail === "string" ? body.detail : `请求失败 (${response.status})`);
  }
  if (response.status === 204) return undefined;
  return response.json();
}

function showStage(stage, status) {
  const icon = $(`${stage}-icon`);
  icon.className = `stage-icon ${status === "uploading" ? "running" : status}`;
  icon.textContent = { queued: "·", succeeded: "✓", failed: "!", cancelled: "×" }[status] || "";
  const label = $(`${stage}-state`);
  label.dataset.state = status;
  if (label.textContent !== stateText[status]) label.textContent = stateText[status];
}

function selectedPreset() {
  return session?.workflow_presets?.find((preset) => preset.id === $("preset").value);
}

function choices() {
  const preset = selectedPreset();
  const compare = Boolean(preset?.checks && $("compare").checked);
  const profile = Boolean(preset?.checks && $("profile").checked);
  return { preset, calibration: preset?.precision === "int8", validation: compare || profile, compare, profile };
}

function currentPlan(options) {
  const ordered = options.calibration ? ["prepare_calibration", "quantize", "compile"] : ["compile"];
  if (options.validation) ordered.push("prepare_validation");
  if (options.compare) ordered.push("compare");
  if (options.profile) ordered.push("profile");
  return Object.fromEntries(ordered.map((stage) => [stage, "pending"]));
}

function renderStages(stages) {
  const names = Object.keys(stageText).filter((stage) => stage in stages);
  const list = $("stages");
  if (list.dataset.names !== names.join(",")) {
    while (list.children.length > 1) list.lastElementChild.remove();
    for (const stage of names) {
      const row = document.createElement("li");
      const icon = document.createElement("span");
      icon.id = `${stage}-icon`;
      icon.setAttribute("aria-hidden", "true");
      const title = document.createElement("span");
      title.className = "stage-title";
      title.textContent = stageText[stage];
      const state = document.createElement("span");
      state.id = `${stage}-state`;
      state.className = "status-pill";
      state.setAttribute("aria-live", "polite");
      row.append(icon, title, state);
      list.append(row);
    }
    list.dataset.names = names.join(",");
  }
  for (const stage of names) showStage(stage, stages[stage]);
}

function configuration() {
  const options = choices();
  const inputs = model.inputs;
  const limit = session.max_upload_bytes;
  const calibration = options.calibration ? dataset.calibration.samples(inputs.length, limit) : [];
  const validation = options.validation ? dataset.validation.samples(inputs.length, limit) : [];
  const raw = options.calibration && dataset.calibration.hasBinary() || options.validation && dataset.validation.hasBinary();
  return { calibration, validation, compare: options.compare, profile: options.profile,
    ...inputSettings(inputs, options.calibration || options.validation, raw) };
}

function changed() {
  attemptKey = undefined;
  refreshStart();
}

function refreshStart() {
  const options = choices();
  const inputs = model?.inputs || [];
  const offline = !navigator.onLine;
  $("model").disabled = busy || offline;
  $("new-model").disabled = busy;
  $("target").disabled = busy || !$("target").options.length;
  $("preset").disabled = busy || !$("preset").options.length;
  for (const field of $("compile-form").querySelectorAll("fieldset")) field.disabled = busy;
  $("capability-note").hidden = !options.preset?.simulated;
  $("model-loading").hidden = !model || model.ready || model.failed;
  $("checks").hidden = !options.preset?.checks;
  dataset.calibration.show(options.calibration, inputs);
  dataset.validation.show(options.validation, inputs);
  $("preparation").hidden = !options.calibration && !options.validation;
  $("padding").hidden = !["pad", "letterbox"].includes($("geometry").value);
  $("custom").hidden = $("values").value !== "custom";
  $("raw-bin").hidden = !(options.calibration && dataset.calibration.hasBinary() || options.validation && dataset.validation.hasBinary());
  if (!viewing || viewing.reuseInspection || viewing.inspection) renderStages(currentPlan(options));
  let hint = "";
  if (offline) hint = "网络恢复后可提交任务。";
  else if (busy && model?.ready) hint = "正在上传数据并提交任务…";
  else if (model?.submitted) hint = "已提交。可新建转换以使用另一份模型。";
  else if (!model) hint = "请选择本地 ONNX 文件。";
  else if (model.failed) hint = "模型检查失败，请更换文件后再试。";
  else if (!model.ready) hint = "等待模型检查通过。";
  else if (!options.preset) hint = "当前部署没有可用的编译预设。";
  else {
    try { configuration(); }
    catch (error) { hint = error.message; }
  }
  $("start").disabled = busy || Boolean(hint);
  $("start-hint").textContent = hint;
}

function stopWatching() {
  clearTimeout(timer);
  viewing = undefined;
}

function resetForm(clearFile = true) {
  stopWatching();
  model = undefined;
  attemptKey = undefined;
  if (clearFile) $("model").value = "";
  $("model-inputs").replaceChildren();
  $("model-inputs").hidden = true;
  $("model-name").hidden = true;
  $("task-detail").hidden = true;
  dataset.calibration.clear();
  dataset.validation.clear();
  $("compare").checked = false;
  $("profile").checked = false;
  $("geometry").value = "keep";
  $("values").value = "raw";
  $("layout").value = "auto";
  $("fill").value = "0";
  $("mean").value = "";
  $("std").value = "";
  $("basis").value = "raw";
  $("raw-bin").open = false;
  renderInputSettings([]);
  showStage("inspect", "idle");
  refreshStart();
}

function signInView(notice = "") {
  session = undefined;
  busy = false;
  resetForm();
  $("history").replaceChildren();
  $("history-loading").hidden = true;
  $("empty-history").hidden = true;
  $("refresh").disabled = false;
  $("session-loading").hidden = true;
  $("account").hidden = true;
  $("workbench").hidden = true;
  $("sign-in").hidden = false;
  $("login-form").reset();
  message(notice);
  $("login-form").elements.username.focus();
}

function loadPresets() {
  const target = $("target").value;
  const presets = session.workflow_presets.filter((preset) => preset.target === target);
  $("preset").replaceChildren(...presets.map((preset) => new Option(precisionText[preset.precision], preset.id)));
  changed();
}

async function enter(info) {
  session = info;
  $("session-loading").hidden = true;
  $("account-name").textContent = info.username;
  $("account").hidden = false;
  $("workbench").hidden = false;
  $("sign-in").hidden = true;
  const targets = new Set(info.workflow_presets.map((preset) => preset.target));
  $("target").replaceChildren(...Array.from(targets, (target) => new Option(target, target)));
  $("file-help").textContent = `支持内嵌权重的 .onnx 文件，大小上限 ${Math.floor(info.max_upload_bytes / 1048576)} MiB。`;
  loadPresets();
  await loadHistory();
}

async function loadHistory() {
  const activeSession = session;
  $("refresh").disabled = true;
  if (!$("history").children.length) {
    $("history-loading").hidden = false;
    $("empty-history").hidden = true;
  }
  let tasks;
  try {
    tasks = await api("/tasks");
    if (session !== activeSession) return;
  } finally {
    if (session === activeSession) {
      $("history-loading").hidden = true;
      $("refresh").disabled = false;
    }
  }
  $("empty-history").hidden = tasks.length > 0;
  const items = tasks.map((task) => {
    const row = document.createElement("li");
    const button = document.createElement("button");
    const info = document.createElement("span");
    const name = document.createElement("span");
    const summary = document.createElement("span");
    const status = document.createElement("span");
    button.type = "button";
    name.textContent = task.filename;
    name.className = "history-name";
    summary.className = "help history-caption";
    status.className = "status-pill";
    status.dataset.state = task.state;
    const [target, precision] = (task.preset || "").split("-");
    const description = precisionText[precision] ? `${target.toUpperCase()} · ${precisionText[precision]}` : presetText[task.preset] || operationText[task.operation];
    summary.textContent = description;
    status.textContent = stateText[task.state];
    info.append(name, summary);
    button.append(info, status);
    row.append(button);
    button.addEventListener("click", () => {
      if (busy) return;
      message("");
      resetForm();
      watchTask(task.id, operationText[task.operation], task.filename, task.operation === "inspect_onnx");
    });
    return row;
  });
  $("history").replaceChildren(...items);
}

function showOutputs(job) {
  $("results").hidden = !job.outputs.length;
  const items = job.outputs.map((output) => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `${apiRoot}/files/${output.id}`;
    link.textContent = output.filename;
    item.append(link);
    return item;
  });
  $("downloads").replaceChildren(...items);
}

async function inspectResult(job, view) {
  const selected = model;
  if (!selected || job.input_id !== selected.id || job.operation !== "inspect_onnx") return;
  selected.failed = job.state === "failed" || job.state === "cancelled";
  if (job.state === "succeeded" && !selected.ready) {
    const report = job.outputs.find((output) => output.kind === "inspection");
    if (report) {
      const metadata = await api(`/files/${report.id}`);
      if (viewing !== view || model !== selected) return;
      const rows = metadata.inputs.map(({ name, shape, type }) => {
        const row = document.createElement("li");
        const dimensions = Array.isArray(shape) ? shape.map((dim) => dim ?? "?").join(" × ") : "?";
        row.textContent = `${name} · ${type ?? "未知类型"} · [${dimensions}]`;
        return row;
      });
      $("model-inputs").replaceChildren(...rows);
      $("model-inputs").hidden = !rows.length;
      selected.inputs = metadata.inputs;
      renderInputSettings(metadata.inputs);
      selected.ready = true;
    }
  }
  refreshStart();
}

async function refreshTask(view) {
  try {
    const job = await api(`/tasks/${view.id}`);
    if (viewing !== view) return;
    if (view.reuseInspection && !model && job.outputs.some((output) => output.kind === "inspection")) {
      model = { id: job.input_id, filename: view.filename };
      $("model-name").textContent = view.filename;
      $("model-name").hidden = false;
    }
    if (job.operation === "inspect_onnx") {
      view.inspection = true;
      showStage("inspect", job.state);
      renderStages(currentPlan(choices()));
    } else {
      showStage("inspect", "succeeded");
      renderStages(Object.keys(job.stages).length ? job.stages : { compile: job.state });
    }
    $("cancel").hidden = terminal.has(job.state) || job.cancel_requested;
    const taskMessage = job.error || (job.cancel_requested && !terminal.has(job.state) ? "正在取消…" : "") ||
      (job.state === "succeeded" && Object.values(job.stages).includes("failed") ? "编译已完成。部分验证阶段失败，详情见日志。" : "") ||
      (job.state === "succeeded" && !job.outputs.length ? "该任务的文件已不在当前会话中，日志仍可查看。" : "") ||
      (job.state === "succeeded" && job.operation === "compile_om" ? "请在退出登录或会话到期前下载所需文件。" : "");
    $("task-message").textContent = taskMessage;
    $("task-message").hidden = !taskMessage;
    const outputIds = job.outputs.map(({ id }) => id).join(",");
    if (view.outputs !== outputIds) showOutputs(job);
    view.outputs = outputIds;
    await inspectResult(job, view);
    if (viewing !== view) return;

    if (view.attempt !== job.attempt_id) {
      view.attempt = job.attempt_id;
      view.offset = 0;
      view.log = "";
    }
    const part = await api(`/tasks/${view.id}/logs?offset=${view.offset}`);
    if (viewing !== view) return;
    if (part.attempt_id !== view.attempt) {
      timer = setTimeout(() => refreshTask(view), 250);
      return;
    }
    const additionalLog = part.next_offset > view.offset;
    view.offset = part.next_offset;
    if (additionalLog) view.log += part.text;
    const log = $("logs");
    const nextText = view.log || (terminal.has(job.state) ? "暂无日志。" : "等待日志…");
    if (log.textContent !== nextText) log.textContent = nextText;
    if (!terminal.has(job.state) || additionalLog) {
      timer = setTimeout(() => refreshTask(view), additionalLog ? 150 : 900);
    } else {
      await loadHistory();
    }
  } catch (error) {
    if (viewing !== view) return;
    message(error.message);
    timer = setTimeout(() => refreshTask(view), 2500);
  }
}

function watchTask(id, label, filename, reuseInspection = false) {
  stopWatching();
  viewing = { id, filename, reuseInspection, offset: 0, attempt: null, log: "" };
  $("task-title").textContent = `${label} · ${filename}`;
  $("task-detail").hidden = false;
  $("task-message").hidden = true;
  $("results").hidden = true;
  $("cancel").hidden = true;
  $("logs").textContent = "等待日志…";
  void refreshTask(viewing);
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  message("");
  const form = event.currentTarget;
  const button = form.querySelector("button");
  if (button.disabled) return;
  button.disabled = true;
  try {
    await api("/login", {
      method: "POST", headers: { "Content-Type": "application/json", "X-ATC-Login": "1" },
      body: JSON.stringify({ username: form.elements.username.value.trim(), password: form.elements.password.value }),
    });
    form.reset();
    await enter(await api("/session"));
  } catch (error) {
    message(error.message);
  } finally {
    form.elements.password.value = "";
    button.disabled = false;
  }
});

$("logout").addEventListener("click", async () => {
  if (!window.confirm("退出后，本次会话的文件将无法下载。正在执行的任务会继续，日志仍可查看。确定退出？")) return;
  try {
    await api("/logout", { method: "POST" });
    signInView();
  } catch (error) {
    message(error.message);
  }
});

$("target").addEventListener("change", loadPresets);
$("compile-form").addEventListener("input", (event) => {
  if (!["model", "target"].includes(event.target.id) && event.target.type !== "file") changed();
});
$("new-model").addEventListener("click", () => {
  resetForm();
  message("");
  $("model").focus();
});
$("refresh").addEventListener("click", () => loadHistory().catch((error) => message(error.message)));

function showConnection() {
  $("connection").hidden = navigator.onLine;
  if (navigator.onLine && $("notice").textContent === connectionError) message("");
  refreshStart();
}

window.addEventListener("online", showConnection);
window.addEventListener("offline", showConnection);
showConnection();

$("model").addEventListener("change", async (event) => {
  const file = event.currentTarget.files[0];
  if (!file) return;
  const activeSession = session;
  resetForm(false);
  message("");
  if (!file.name.toLowerCase().endsWith(".onnx") || !file.size || file.size > session.max_upload_bytes) {
    $("model").value = "";
    message("请选择大小在限制内的非空 .onnx 文件。");
    return;
  }
  busy = true;
  model = { filename: file.name };
  showStage("inspect", "uploading");
  refreshStart();
  try {
    const job = await api(`/models?${new URLSearchParams({ filename: file.name })}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    if (session !== activeSession) return;
    model.id = job.input_id;
    showStage("inspect", job.state);
    watchTask(job.id, "模型检查", file.name);
  } catch (error) {
    if (session !== activeSession) return;
    model.failed = true;
    $("model").value = "";
    showStage("inspect", "failed");
    message(error.message);
  } finally {
    if (session === activeSession) {
      busy = false;
      refreshStart();
    }
  }
});

$("compile-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("start").disabled) return;
  const activeSession = session;
  const preset = $("preset").value;
  attemptKey ??= Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");
  busy = true;
  const selected = model;
  refreshStart();
  message("");
  try {
    const payload = configuration();
    const uploaded = selected.uploaded ??= new Map();
    for (const role of ["calibration", "validation"]) {
      const ids = [];
      for (const sample of payload[role]) {
        const row = [];
        for (const file of sample) {
          if (session !== activeSession) return;
          if (!uploaded.has(file)) {
            const asset = await api(`/dataset-files?${new URLSearchParams({ filename: file.name })}`, {
              method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
            });
            uploaded.set(file, asset.id);
          }
          row.push(uploaded.get(file));
        }
        ids.push(row);
      }
      payload[role] = ids;
    }
    if (session !== activeSession) return;
    const job = await api("/conversions", {
      method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": attemptKey },
      body: JSON.stringify({ model_id: selected.id, preset, ...payload }),
    });
    if (session !== activeSession) return;
    model.submitted = true;
    showStage("compile", job.state);
    watchTask(job.id, "编译", model.filename);
  } catch (error) {
    if (session === activeSession) message(error.message);
  } finally {
    if (session === activeSession) {
      busy = false;
      refreshStart();
    }
  }
});

$("cancel").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const view = viewing;
  if (!view || button.disabled || !window.confirm("确定取消该任务？")) return;
  button.disabled = true;
  try {
    await api(`/tasks/${view.id}/cancel`, { method: "POST" });
    if (viewing === view) $("cancel").hidden = true;
  } catch (error) {
    message(error.message);
  } finally {
    button.disabled = false;
  }
});

try {
  await enter(await api("/session"));
} catch (error) {
  if (!session) signInView(error.message === "Sign in to continue." ? "" : error.message);
  else message(error.message);
}
