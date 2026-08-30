import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { basename } from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

import { finalizeNvidiaAcceleratorProfile, NVIDIA_ACCELERATOR_PROFILE_SCHEMA } from "../web/lib/nvidia-accelerator-profile.js";

const QUERY_FIELDS = Object.freeze([
  "index", "name", "uuid", "pci.bus_id", "driver_version", "memory.total", "compute_cap", "clocks.max.sm", "clocks.max.memory",
]);

export async function collectNvidiaAcceleratorProfile({
  collectorVersion,
  deviceIndex = null,
  includeDeviceIdentifiers = false,
  collectedAt = new Date().toISOString(),
  commandRunner = runCommand,
  executableResolver = resolveExecutable,
  fileHasher = sha256File,
} = {}) {
  const smiPath = executableResolver("nvidia-smi");
  if (!smiPath) throw new Error("nvidia-smi was not found on PATH; an observed NVIDIA profile cannot be produced.");
  const query = commandRunner(smiPath, [`--query-gpu=${QUERY_FIELDS.join(",")}`, "--format=csv,noheader,nounits"]);
  if (query.status !== 0) throw new Error(`nvidia-smi inventory query failed: ${bounded(query.stderr || query.stdout, 600)}`);
  const devices = parseDeviceRows(query.stdout, includeDeviceIdentifiers)
    .filter((device) => deviceIndex == null || device.index === deviceIndex);
  if (!devices.length) throw new Error(deviceIndex == null ? "nvidia-smi returned no GPU rows." : `nvidia-smi returned no GPU at index ${deviceIndex}.`);
  const overview = commandRunner(smiPath, []);
  const cudaDriverApi = /CUDA Version:\s*([0-9.]+)/i.exec(`${overview.stdout}\n${overview.stderr}`)?.[1] || null;
  const tools = [await observedTool("nvidia_smi", smiPath, `${query.stdout}\n${overview.stdout}\n${overview.stderr}`, fileHasher)];
  const nvcc = await optionalTool("nvcc", ["--version"], "cuda_compiler", commandRunner, executableResolver, fileHasher);
  const trtexec = await optionalTool("trtexec", ["--version"], "tensorrt_exec", commandRunner, executableResolver, fileHasher);
  tools.push(nvcc.tool, trtexec.tool);
  const cudaToolkitVersion = /release\s+([0-9.]+)/i.exec(nvcc.output)?.[1] || null;
  const tensorRtVersion = /TensorRT[^0-9]*([0-9]+(?:\.[0-9]+){1,3})/i.exec(trtexec.output)?.[1]
    || /([0-9]+(?:\.[0-9]+){2,3})/.exec(trtexec.output)?.[1] || null;
  return finalizeNvidiaAcceleratorProfile({
    schema: NVIDIA_ACCELERATOR_PROFILE_SCHEMA,
    evidence_class: "OBSERVED_HOST_TOOLING",
    collection: {
      collected_at: collectedAt,
      platform: process.platform,
      architecture: process.arch,
      collector: "deepbom-cli:nvidia",
      collector_version: String(collectorVersion || "development"),
      tools,
    },
    devices,
    software: {
      nvidia_driver: { status: "observed", version: devices[0].driver_version },
      cuda_driver_api: cudaDriverApi ? { status: "observed", version: cudaDriverApi } : { status: "not_exposed", version: null },
      cuda_toolkit: cudaToolkitVersion ? { status: "observed", version: cudaToolkitVersion } : { status: nvcc.tool.status === "not_installed" ? "not_installed" : "not_exposed", version: null },
      tensorrt: tensorRtVersion ? { status: "observed", version: tensorRtVersion } : { status: trtexec.tool.status === "not_installed" ? "not_installed" : "not_exposed", version: null },
    },
    roofline_contract: {
      status: "not_assessable_missing_exact_hardware_contract",
      theoretical_compute_ceiling: null,
      theoretical_memory_bandwidth: null,
      missing_fields: ["cores_per_sm_by_precision", "memory_bus_width_bits", "sm_count"],
      reason: "nvidia-smi does not expose the complete immutable architecture and memory-interface contract required for an exact theoretical roofline.",
    },
    interpretation_boundary: "The profile records host-tool observations and binary identities. Device presence, driver version, memory capacity, compute capability, and reported maximum clocks do not establish ORT/TensorRT build inclusion, kernel availability, original-op assignment, engine tactics, sustained bandwidth, occupancy, latency, correctness, or fit after backend-private allocation.",
  });
}

export function parseDeviceRows(source, includeDeviceIdentifiers = false) {
  const rows = String(source || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  return rows.map((line) => {
    const values = parseCsvLine(line);
    if (values.length !== QUERY_FIELDS.length) throw new Error(`nvidia-smi returned ${values.length} fields; expected ${QUERY_FIELDS.length}.`);
    const index = integer(values[0], "GPU index");
    const memoryMib = integer(values[5], "GPU memory.total");
    const uuid = required(values[2], "GPU UUID");
    const pci = required(values[3], "GPU PCI bus ID");
    const memoryBytes = BigInt(memoryMib) * 1024n * 1024n;
    return {
      index,
      name: required(values[1], "GPU name"),
      uuid_sha256: sha256Text(`nvidia-uuid\0${uuid}`),
      pci_bus_id_sha256: sha256Text(`nvidia-pci\0${pci}`),
      ...(includeDeviceIdentifiers ? { uuid, pci_bus_id: pci } : { uuid: null, pci_bus_id: null }),
      driver_version: required(values[4], "NVIDIA driver version"),
      memory_total_bytes: exact(memoryBytes),
      compute_capability: required(values[6], "GPU compute capability"),
      maximum_sm_clock_mhz: optionalPositiveInteger(values[7], "maximum SM clock"),
      maximum_memory_clock_mhz: optionalPositiveInteger(values[8], "maximum memory clock"),
      sm_count: null,
      memory_bus_width_bits: null,
      unexposed_fields: [
        "memory_bus_width_bits",
        "sm_count",
        ...(/^n\/a$/i.test(values[7]) ? ["maximum_sm_clock_mhz"] : []),
        ...(/^n\/a$/i.test(values[8]) ? ["maximum_memory_clock_mhz"] : []),
      ],
    };
  }).sort((left, right) => left.index - right.index);
}

async function optionalTool(command, args, role, commandRunner, executableResolver, fileHasher) {
  const executable = executableResolver(command);
  if (!executable) return { tool: { role, status: "not_installed", executable_name: null, executable_sha256: null, observation_sha256: null }, output: "" };
  const result = commandRunner(executable, args);
  const output = `${result.stdout || ""}\n${result.stderr || ""}`.trim();
  return { tool: await observedTool(role, executable, output, fileHasher, result.status === 0 ? "observed" : "execution_failed"), output: result.status === 0 ? output : "" };
}

async function observedTool(role, executable, observation, fileHasher, status = "observed") {
  return {
    role,
    status,
    executable_name: basename(executable),
    executable_sha256: await fileHasher(executable),
    observation_sha256: sha256Text(observation),
  };
}

function runCommand(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8", windowsHide: true, maxBuffer: 8 * 1024 * 1024 });
  return { status: result.status ?? -1, stdout: result.stdout || "", stderr: result.stderr || result.error?.message || "" };
}

function resolveExecutable(command) {
  const resolver = process.platform === "win32" ? "where.exe" : "which";
  const result = spawnSync(resolver, [command], { encoding: "utf8", windowsHide: true });
  if (result.status !== 0) return null;
  return String(result.stdout || "").split(/\r?\n/).map((line) => line.trim()).find(Boolean) || null;
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

function parseCsvLine(line) {
  const values = [];
  let value = "", quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') { value += '"'; index += 1; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) { values.push(value.trim()); value = ""; }
    else value += char;
  }
  if (quoted) throw new Error("nvidia-smi returned malformed quoted CSV.");
  values.push(value.trim());
  return values;
}

function integer(value, label) {
  const number = Number(String(value).trim());
  if (!Number.isSafeInteger(number) || number < 0) throw new Error(`nvidia-smi ${label} is invalid.`);
  return number;
}

function optionalPositiveInteger(value, label) {
  if (/^n\/a$/i.test(String(value).trim())) return null;
  const number = integer(value, label);
  if (number <= 0) throw new Error(`nvidia-smi ${label} is invalid.`);
  return number;
}

function required(value, label) {
  const text = String(value || "").trim();
  if (!text || /^n\/a$/i.test(text)) throw new Error(`nvidia-smi ${label} is unavailable.`);
  return text;
}

function exact(value) {
  return { decimal: value.toString(), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null };
}

function sha256Text(value) { return createHash("sha256").update(String(value), "utf8").digest("hex"); }
function bounded(value, maximum) { return String(value || "").trim().slice(0, maximum); }
