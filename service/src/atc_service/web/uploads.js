function isFileDrag(event) {
  return event.dataTransfer?.types.includes("Files");
}

export function preventFileNavigation() {
  for (const name of ["dragover", "drop"]) {
    window.addEventListener(name, (event) => {
      if (!event.defaultPrevented && isFileDrag(event)) {
        event.preventDefault();
        event.dataTransfer.dropEffect = "none";
      }
    });
  }
}

export function bindDrop(zone, receive) {
  zone.addEventListener("dragover", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    const enabled = !zone.matches(":disabled");
    event.dataTransfer.dropEffect = enabled ? "copy" : "none";
    zone.classList.toggle("drag-over", enabled);
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (event) => {
    if (!isFileDrag(event)) return;
    event.preventDefault();
    zone.classList.remove("drag-over");
    if (!zone.matches(":disabled")) void receive(event.dataTransfer);
  });
}

export function pickedFiles(files) {
  return Array.from(files, (file) => ({ file, path: file.webkitRelativePath || file.name }));
}

async function appendEntry(entry, path, result) {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    result.push({ file, path });
    return;
  }
  const reader = entry.createReader();
  // A directory reader can return multiple batches before returning an empty one.
  for (;;) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) return;
    for (const child of batch) await appendEntry(child, `${path}/${child.name}`, result);
  }
}

export async function droppedFiles(transfer, allowDirectories = true) {
  // Capture the entries while the drop event's data store is readable.
  const items = Array.from(transfer.items).filter((item) => item.kind === "file");
  const sources = items.map((item) => {
    const getEntry = item.getAsEntry || item.webkitGetAsEntry;
    const entry = getEntry?.call(item);
    return entry?.isDirectory ? entry : item.getAsFile() || entry;
  });
  if (!allowDirectories && sources.some((source) => source?.isDirectory)) {
    throw new Error("请拖入一个 .onnx 文件。");
  }
  if (!sources.some(Boolean)) return pickedFiles(transfer.files);
  const result = [];
  for (const source of sources) {
    if (source instanceof File) result.push({ file: source, path: source.name });
    else if (source) await appendEntry(source, source.name, result);
  }
  return result;
}
