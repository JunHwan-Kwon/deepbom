import { lstat, open, readdir } from "node:fs/promises";
import path from "node:path";

import { sha256FileHex, sha256Hex } from "../web/lib/hash.js";
import { prepareExternalDataFiles, prepareOnnxExternalDataFiles } from "../web/lib/onnx-external-data.js";

const MAX_BUNDLE_FILES = 20_000;

export async function loadCliInput(inputPath) {
  const metadata = await statNoLinks(inputPath, "artifact input");
  if (metadata.isFile()) {
    const file = new DiskFile(inputPath, path.basename(inputPath), metadata);
    const prefix = new Uint8Array(await file.slice(0, Math.min(file.size, 4096)).arrayBuffer());
    return { kind: "file", path: inputPath, filename: file.name, file, prefix };
  }
  if (!metadata.isDirectory()) throw new Error("Artifact input must be a regular file or directory.");
  return {
    kind: "bundle",
    path: inputPath,
    filename: path.basename(inputPath),
    files: await collectDirectoryFiles(inputPath),
  };
}

export async function loadCliBundleMembers(members, rootName = "remote-artifact") {
  if (!Array.isArray(members) || !members.length || members.length > MAX_BUNDLE_FILES) {
    throw new Error("Remote artifact bundle member count is invalid.");
  }
  const safeRoot = safePathComponent(rootName, "remote artifact root");
  const paths = new Set();
  const files = [];
  for (const member of members) {
    const relative = safeRelativePath(member?.path, "remote artifact member");
    if (paths.has(relative)) throw new Error(`Remote artifact bundle repeats ${relative}.`);
    paths.add(relative);
    const absolute = path.resolve(String(member?.resolved_path || ""));
    const metadata = await statNoLinks(absolute, `remote artifact member ${relative}`);
    if (!metadata.isFile()) throw new Error(`Remote artifact member ${relative} is not a regular file.`);
    files.push(new DiskFile(absolute, `${safeRoot}/${relative}`, metadata));
  }
  files.sort((left, right) => left.webkitRelativePath.localeCompare(right.webkitRelativePath));
  return { kind: "bundle", path: null, filename: safeRoot, files, virtual: true };
}

export async function readCliFileBytes(input) {
  if (input?.kind !== "file") throw new Error("A single-file CLI input is required.");
  if (input.bytes) return input.bytes;
  await input.file.assertUnchanged();
  const bytes = new Uint8Array(await input.file.arrayBuffer());
  await input.file.assertUnchanged();
  input.bytes = bytes;
  return bytes;
}

export async function identifyCliFile(input) {
  if (input?.kind !== "file") throw new Error("A single-file CLI input is required.");
  await input.file.assertUnchanged();
  const sha256 = input.bytes ? await sha256Hex(input.bytes) : await sha256FileHex(input.file);
  await input.file.assertUnchanged();
  return { filename: input.filename, size: input.file.size, sha256 };
}

export async function loadOnnxExternalData(modelPath, analysis, externalDataRoot = "") {
  const locations = [...new Set((analysis?.onnx_external_data?.tensors || [])
    .map((row) => String(row.normalized_location || ""))
    .filter(Boolean))].sort();
  if (!locations.length) return [];
  const root = path.resolve(externalDataRoot || path.dirname(modelPath));
  const rootMetadata = await statNoLinks(root, "ONNX external-data root");
  if (!rootMetadata.isDirectory()) throw new Error("ONNX external-data root must be a directory.");
  const selectedRoot = path.basename(root);
  const files = [];
  for (const location of locations) {
    const candidate = resolveContainedPath(root, location, "ONNX external_data location");
    let metadata;
    try { metadata = await statNoLinks(candidate, `ONNX external data ${location}`); }
    catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (!metadata.isFile()) throw new Error(`ONNX external data ${location} is not a regular file.`);
    files.push(new DiskFile(candidate, `${selectedRoot}/${location}`, metadata));
  }
  return prepareOnnxExternalDataFiles(files);
}

export async function loadOnnxExternalDataMembers(analysis, members) {
  const locations = [...new Set((analysis?.onnx_external_data?.tensors || [])
    .map((row) => String(row.normalized_location || "")).filter(Boolean))].sort();
  if (!locations.length) return [];
  const byLocation = new Map();
  for (const member of members || []) {
    const location = safeRelativePath(member?.model_relative_path, "ONNX external_data member location");
    if (byLocation.has(location)) throw new Error(`Remote ONNX closure repeats external_data location ${location}.`);
    byLocation.set(location, member);
  }
  const files = [];
  for (const location of locations) {
    const member = byLocation.get(location);
    if (!member) throw new Error(`Remote ONNX closure is missing external_data location ${location}.`);
    const absolute = path.resolve(String(member.resolved_path || ""));
    const metadata = await statNoLinks(absolute, `ONNX external data ${location}`);
    if (!metadata.isFile()) throw new Error(`ONNX external data ${location} is not a regular file.`);
    files.push(new DiskFile(absolute, `remote-onnx/${location}`, metadata));
  }
  return prepareOnnxExternalDataFiles(files);
}

export async function loadExecuTorchExternalData(programPath, externalDataRoot = "") {
  const root = path.resolve(externalDataRoot || path.dirname(programPath));
  const rootMetadata = await statNoLinks(root, "ExecuTorch external-data root");
  if (!rootMetadata.isDirectory()) throw new Error("ExecuTorch external-data root must be a directory.");
  const files = (await collectDirectoryFiles(root)).filter((file) => /\.ptd$/i.test(file.name));
  return prepareExternalDataFiles(files, { label: "ExecuTorch PTD sidecar" });
}

export async function verifyBundleSnapshot(files, records) {
  const expected = new Map((records || []).map((row) => [String(row.path), row]));
  if (expected.size !== files.length) throw new Error("Bundle evidence does not cover every selected file.");
  for (const file of files) {
    const row = expected.get(file.webkitRelativePath);
    if (!row || Number(row.byte_length) !== file.size) throw new Error(`Bundle file ledger changed for ${file.webkitRelativePath}.`);
    await file.assertUnchanged();
    const observed = await sha256FileHex(file);
    await file.assertUnchanged();
    if (observed !== row.sha256) throw new Error(`Bundle file ${file.webkitRelativePath} changed during analysis.`);
  }
}

async function collectDirectoryFiles(root) {
  const rootName = path.basename(root);
  const rows = [];
  await walk(root, "");
  if (!rows.length) throw new Error("Artifact directory contains no files.");
  return rows.sort((a, b) => a.webkitRelativePath.localeCompare(b.webkitRelativePath));

  async function walk(directory, relativeDirectory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const relative = relativeDirectory ? `${relativeDirectory}/${entry.name}` : entry.name;
      const absolute = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`Artifact directory contains a symbolic link: ${relative}`);
      if (entry.isDirectory()) await walk(absolute, relative);
      else if (entry.isFile()) {
        if (rows.length >= MAX_BUNDLE_FILES) throw new Error(`Artifact directory exceeds ${MAX_BUNDLE_FILES} files.`);
        rows.push(new DiskFile(absolute, `${rootName}/${relative}`, await statNoLinks(absolute, relative)));
      } else throw new Error(`Artifact directory contains an unsupported filesystem entry: ${relative}`);
    }
  }
}

class DiskFile {
  constructor(absolutePath, relativePath, metadata) {
    this.absolutePath = absolutePath;
    this.webkitRelativePath = relativePath.replaceAll("\\", "/");
    this.name = path.basename(absolutePath);
    this.size = Number(metadata.size);
    this.type = "application/octet-stream";
    this.snapshot = snapshot(metadata);
  }

  async arrayBuffer() { return this.slice(0, this.size).arrayBuffer(); }

  slice(start = 0, end = this.size) {
    const begin = boundedOffset(start, this.size);
    const finish = Math.max(begin, boundedOffset(end, this.size));
    const file = this;
    return {
      size: finish - begin,
      async arrayBuffer() {
        await file.assertUnchanged();
        const handle = await open(file.absolutePath, "r");
        try {
          const buffer = Buffer.allocUnsafe(finish - begin);
          const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, begin);
          if (bytesRead !== buffer.byteLength) throw new Error(`Bundle file ${file.webkitRelativePath} changed while being read.`);
          await file.assertUnchanged();
          return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        } finally { await handle.close(); }
      },
    };
  }

  assertUnchanged() {
    return statNoLinks(this.absolutePath, this.webkitRelativePath).then((metadata) => {
      if (snapshot(metadata) !== this.snapshot) throw new Error(`Bundle file ${this.webkitRelativePath} changed during analysis.`);
    });
  }
}

async function statNoLinks(candidate, label) {
  const metadata = await lstat(candidate, { bigint: true });
  if (metadata.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link.`);
  return metadata;
}

function snapshot(metadata) {
  return [metadata.dev, metadata.ino, metadata.size, metadata.mtimeNs, metadata.ctimeNs].map(String).join(":");
}

function resolveContainedPath(root, relative, label) {
  const normalized = String(relative).replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)
    || normalized.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error(`${label} is not a safe relative path: ${relative}`);
  }
  const candidate = path.resolve(root, ...normalized.split("/"));
  const prefix = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
  if (!candidate.startsWith(prefix)) throw new Error(`${label} escapes its declared root.`);
  return candidate;
}

function safeRelativePath(value, label) {
  const normalized = String(value || "").replaceAll("\\", "/");
  if (!normalized || normalized.length > 2048 || normalized.startsWith("/") || /^[A-Za-z]:/.test(normalized)
    || normalized.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error(`${label} is not a safe relative path.`);
  }
  return normalized;
}

function safePathComponent(value, label) {
  const normalized = String(value || "").trim();
  if (!normalized || normalized.length > 255 || /[\\/:*?"<>|]/.test(normalized) || normalized === "." || normalized === "..") {
    throw new Error(`${label} is invalid.`);
  }
  return normalized;
}

function boundedOffset(value, size) {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new Error("File slice offset must be a safe integer.");
  return Math.max(0, Math.min(size, number));
}
