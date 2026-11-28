import { bindDrop, droppedFiles, pickedFiles } from "./uploads.js";

const $ = (id) => document.getElementById(id);
const supported = /\.(jpe?g|png|npy|bin)$/i;
const modelDtypes = { FLOAT: "float32", FLOAT16: "float16", DOUBLE: "float64" };
const rawTypes = ["float32", "float16", "float64", "uint8", "int8", "int16", "int32", "int64", "bool"];

function field(tag, title, value = "") {
  const label = document.createElement("label");
  const control = document.createElement(tag);
  label.append(title, control);
  if (tag === "input") control.value = value;
  return { label, control };
}

function numbers(value, integer = false) {
  if (!value.trim()) return [];
  const result = value.trim().split(/[,\s×，]+/u).map(Number);
  if (result.some((number) => !Number.isFinite(number) || integer && (!Number.isInteger(number) || number < 1 || number > 65536))) {
    throw new Error(integer ? "尺寸需要是 1 到 65536 的整数。" : "均值和标准差需要是有限数值。");
  }
  return result;
}

export function mountDataset(container, title, changed) {
  container.append($("dataset-picker").content.cloneNode(true));
  const zone = container.querySelector(".upload-zone");
  const help = container.querySelector("[data-upload-help]");
  const actions = container.querySelector(".picker-actions");
  const count = container.querySelector("[data-count]");
  const guidance = container.querySelector("[data-mapping]");
  container.querySelector("[data-upload-title]").textContent = title;
  zone.setAttribute("aria-label", title);
  actions.id = `${container.id}-actions`;
  help.id = `${container.id}-help`;
  zone.setAttribute("aria-controls", actions.id);
  zone.setAttribute("aria-describedby", help.id);
  let files = [];
  let reading = false;
  let selection = 0;

  function showActions(open) {
    actions.hidden = !open;
    zone.setAttribute("aria-expanded", String(open));
  }

  function showCount(text, error = false) {
    count.hidden = !text;
    count.textContent = text;
    count.classList.toggle("selection-error", error);
  }

  function selectFiles(picked) {
    reading = false;
    files = picked.filter(({ file }) => supported.test(file.name));
    const skipped = picked.length - files.length;
    showCount(files.length ? `已添加 ${files.length} 个文件${skipped ? `，跳过 ${skipped} 个不支持的文件` : ""}。` : "没有可用的图片、.npy 或 .bin 文件。", !files.length);
    changed();
  }

  zone.addEventListener("click", () => showActions(actions.hidden));
  container.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !actions.hidden) {
      showActions(false);
      zone.focus();
    }
  });
  for (const input of container.querySelectorAll("input")) {
    container.querySelector(`[data-picker=${input.dataset.mode}]`).addEventListener("click", () => {
      showActions(false);
      zone.focus({ preventScroll: true });
      input.click();
    });
    input.addEventListener("change", () => {
      const picked = pickedFiles(input.files);
      input.value = "";
      if (!picked.length) return;
      selection++;
      selectFiles(picked);
    });
  }
  bindDrop(zone, async (transfer) => {
    const current = ++selection;
    reading = true;
    showActions(false);
    showCount("正在读取所拖入的文件…");
    changed();
    try {
      const picked = await droppedFiles(transfer);
      if (current === selection) selectFiles(picked);
    } catch {
      if (current !== selection) return;
      reading = false;
      showCount(files.length ? `读取失败，仍保留先前添加的 ${files.length} 个文件。` : "无法读取拖入的文件，请重试。", true);
      changed();
    }
  });
  return {
    clear() {
      selection++;
      reading = false;
      files = [];
      showCount("");
      showActions(false);
    },
    show(visible, inputs) {
      container.hidden = !visible;
      if (!visible) showActions(false);
      guidance.hidden = inputs.length < 2;
      guidance.textContent = inputs.length > 1
        ? `按输入分目录 input_0、input_1 等，同一样本的文件名主体相同。多选文件时用 sample.input_0.npy 这类命名。${inputs.map(({ name }, index) => `input_${index} = ${name}`).join("，")}`
        : "";
    },
    hasBinary() { return files.some(({ file }) => /\.bin$/i.test(file.name)); },
    samples(inputCount, limit) {
      if (reading) throw new Error(`正在读取${title}。`);
      if (!files.length) throw new Error(`请添加${title}。`);
      if (!inputCount || inputCount > 16) throw new Error("数据集工作流支持 1 到 16 个模型输入。");
      if (files.some(({ file }) => !file.size || file.size > limit || file.name.length > 200)) {
        throw new Error(`${title}包含空文件、超出大小限制的文件或过长文件名。`);
      }
      if (files.length > inputCount * 128) throw new Error(`${title}最多支持 128 个样本。`);
      const sorted = [...files].sort((a, b) => a.path.localeCompare(b.path));
      if (inputCount === 1) return sorted.map(({ file }) => [file]);
      const groups = new Map();
      for (const { file, path: relativePath } of sorted) {
        const basename = file.name.replace(supported, "");
        const path = relativePath.split("/");
        const directory = /^input_(\d+)$/.exec(path.at(-2) || "");
        const suffix = /^(.*)\.input_(\d+)$/.exec(basename);
        const index = directory ? Number(directory[1]) : Number(suffix?.[2]);
        const sample = directory ? [...path.slice(0, -2), basename].join("/") : suffix?.[1];
        if (!sample || !Number.isInteger(index) || index >= inputCount) throw new Error(`${title}的文件命名与模型输入不匹配。`);
        if (!groups.has(sample)) groups.set(sample, Array(inputCount).fill(null));
        const fields = groups.get(sample);
        if (fields[index]) throw new Error(`${title}包含重复的样本输入。`);
        fields[index] = file;
      }
      const result = [...groups.values()];
      if (result.some((sample) => sample.includes(null))) throw new Error(`${title}中的每个样本都需要包含所有模型输入。`);
      return result;
    },
  };
}

export function renderInputSettings(inputs) {
  $("shape-inputs").replaceChildren();
  $("raw-inputs").replaceChildren();
  let dynamic = false;
  for (const [index, { name, shape, type }] of inputs.entries()) {
    if (!shape || shape.some((dimension) => !Number.isInteger(dimension) || dimension < 1)) {
      const { label, control } = field("input", name);
      control.placeholder = shape?.map((dimension) => Number.isInteger(dimension) && dimension > 0 ? dimension : "?").join(", ") || "用逗号分隔各维度";
      control.dataset.shapeIndex = index;
      $("shape-inputs").append(label);
      dynamic = true;
    }
    const row = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = name;
    const raw = field("input", "源尺寸");
    raw.control.placeholder = "留空使用模型输入尺寸";
    raw.control.dataset.rawShape = index;
    const dtype = field("select", "源类型");
    dtype.control.dataset.rawType = index;
    const modelType = modelDtypes[type] || type?.toLowerCase();
    dtype.control.append(new Option(`模型类型（${modelType || "未知"}）`, ""), ...rawTypes.map((name) => new Option(name, name)));
    row.append(legend, raw.label, dtype.label);
    $("raw-inputs").append(row);
  }
  $("shapes").hidden = !dynamic;
}

export function inputSettings(inputs, showPreparation, showRaw) {
  const shapes = {};
  for (const field of $("shape-inputs").querySelectorAll("input")) {
    const index = Number(field.dataset.shapeIndex);
    const shape = numbers(field.value, true);
    const declared = inputs[index].shape;
    if (!shape.length) throw new Error(`请填写 ${inputs[index].name} 的固定输入尺寸。`);
    if (shape.length > 16 || declared && (shape.length !== declared.length || declared.some((dim, axis) => Number.isInteger(dim) && dim > 0 && dim !== shape[axis]))) {
      throw new Error(`${inputs[index].name} 的尺寸需要保留 ONNX 的固定维度。`);
    }
    shapes[index] = shape;
  }
  if (!showPreparation) return { shapes };
  const values = $("values").value;
  const mean = values === "custom" ? numbers($("mean").value) : [];
  const std = values === "custom" ? numbers($("std").value) : [];
  if (values === "custom" && (!mean.length || mean.length > 16 || mean.length !== std.length || std.some((value) => value <= 0))) {
    throw new Error("请为各通道填写对应的均值和正标准差。");
  }
  const geometry = $("geometry").value;
  const usesPadding = ["pad", "letterbox"].includes(geometry);
  const fill = usesPadding ? Number($("fill").value) : 0;
  if (usesPadding && (!$("fill").value.trim() || !Number.isFinite(fill))) throw new Error("请填写有效的补边值。");
  const raw_inputs = {};
  if (showRaw) {
    for (const field of $("raw-inputs").querySelectorAll("input")) {
      const index = Number(field.dataset.rawShape);
      const selected = $("raw-inputs").querySelector(`[data-raw-type="${index}"]`).value;
      if (!field.value.trim() && !selected) continue;
      const shape = field.value.trim() ? numbers(field.value, true) : shapes[index] || inputs[index].shape;
      const dtype = selected || modelDtypes[inputs[index].type] || inputs[index].type?.toLowerCase();
      if (!shape || shape.length > 16 || !rawTypes.includes(dtype)) throw new Error("请为 .bin 指定有效的源尺寸和类型。");
      raw_inputs[index] = { shape, dtype };
    }
  }
  return { shapes, preparation: { geometry, layout: $("layout").value, fill, values, mean, std,
    unit_first: values === "custom" && $("basis").value === "unit", raw_inputs } };
}
