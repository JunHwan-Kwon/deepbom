import { sha1Hex, sha256Hex } from "./hash.js";

const MAX_FILE_COUNT = 1_024;
const MAX_FILE_BYTES = 536_870_912;
const MAX_AGGREGATE_BYTES = 1_073_741_824;

export async function prepareExternalDataFiles(fileList, { onProgress = null, label = "External data" } = {}) {
  const files = [...(fileList || [])];
  if (files.length > MAX_FILE_COUNT) throw new Error(`${label} selection exceeds ${MAX_FILE_COUNT} files.`);
  const selectedDirectoryRoot = commonSelectedDirectoryRoot(files);
  const paths = new Set();
  const records = [];
  let aggregateBytes = 0;
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const rawPath = normalizeBrowserFilePath(file?.webkitRelativePath || file?.name || "");
    const path = selectedDirectoryRoot && rawPath.startsWith(`${selectedDirectoryRoot}/`)
      ? rawPath.slice(selectedDirectoryRoot.length + 1) : rawPath;
    if (!path) throw new Error(`A ${label.toLowerCase()} file has no name.`);
    if (!isSafeRelativePath(path)) throw new Error(`Unsafe ${label.toLowerCase()} file path: ${path}.`);
    if (paths.has(path)) throw new Error(`Duplicate ${label.toLowerCase()} file path: ${path}.`);
    paths.add(path);
    const size = Number(file?.size || 0);
    if (!Number.isSafeInteger(size) || size < 0 || size > MAX_FILE_BYTES) throw new Error(`${label} file ${path} exceeds ${MAX_FILE_BYTES} bytes.`);
    aggregateBytes += size;
    if (!Number.isSafeInteger(aggregateBytes) || aggregateBytes > MAX_AGGREGATE_BYTES) throw new Error(`${label} selection exceeds ${MAX_AGGREGATE_BYTES} bytes.`);
    onProgress?.({ index, count: files.length, path, phase: "reading" });
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (bytes.byteLength !== size) throw new Error(`${label} file ${path} changed while it was being read.`);
    onProgress?.({ index, count: files.length, path, phase: "hashing" });
    const [sha256, sha1] = await Promise.all([sha256Hex(bytes), sha1Hex(bytes)]);
    records.push({ name: path, path, bytes, sha256, sha1 });
  }
  return records;
}

export function prepareOnnxExternalDataFiles(fileList, options = {}) {
  return prepareExternalDataFiles(fileList, { label: "ONNX external data", ...options });
}

function normalizeBrowserFilePath(path) {
  return String(path || "").replace(/\\/g, "/").replace(/^(?:\.\/)+/, "");
}

function commonSelectedDirectoryRoot(files) {
  if (!files.length || files.some((file) => !file?.webkitRelativePath)) return "";
  const roots = new Set(files.map((file) => normalizeBrowserFilePath(file.webkitRelativePath).split("/")[0]));
  return roots.size === 1 ? [...roots][0] : "";
}

function isSafeRelativePath(path) {
  if (!path || path.includes("\0") || /^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("/")) return false;
  return path.split("/").every((part) => part && part !== "." && part !== "..");
}
