import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const NVIDIA_ACCELERATOR_PROFILE_SCHEMA = "deepbom.accelerator_profile.v1";
const SHA256 = /^[a-f0-9]{64}$/;
const COMPUTE_CAPABILITY = /^\d+\.\d+$/;

export function finalizeNvidiaAcceleratorProfile(document) {
  const body = structuredCopy(document);
  delete body.profile_sha256;
  validateProfileBody(body);
  return Object.freeze({ ...body, profile_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateNvidiaAcceleratorProfile(document) {
  if (!document || document.schema !== NVIDIA_ACCELERATOR_PROFILE_SCHEMA) throw new Error("NVIDIA accelerator profile schema is invalid.");
  const body = structuredCopy(document);
  const declared = String(body.profile_sha256 || "").toLowerCase();
  delete body.profile_sha256;
  validateProfileBody(body);
  if (!SHA256.test(declared) || declared !== sha256TextHex(canonicalJson(body))) throw new Error("NVIDIA accelerator profile SHA-256 is invalid.");
  return Object.freeze({ ...body, profile_sha256: declared });
}

function validateProfileBody(value) {
  if (value.schema !== NVIDIA_ACCELERATOR_PROFILE_SCHEMA || value.evidence_class !== "OBSERVED_HOST_TOOLING") {
    throw new Error("NVIDIA accelerator profile identity is invalid.");
  }
  if (!Number.isFinite(Date.parse(value.collection?.collected_at)) || !text(value.collection?.platform, 80)
    || !text(value.collection?.architecture, 80) || !text(value.collection?.collector, 120)
    || !text(value.collection?.collector_version, 80)) {
    throw new Error("NVIDIA accelerator profile collection identity is invalid.");
  }
  if (!Array.isArray(value.devices) || !value.devices.length || value.devices.length > 64) throw new Error("NVIDIA accelerator device inventory is invalid.");
  const indices = new Set();
  for (const device of value.devices) {
    if (!Number.isSafeInteger(device.index) || device.index < 0 || indices.has(device.index) || !text(device.name, 300)
      || !COMPUTE_CAPABILITY.test(String(device.compute_capability || "")) || !text(device.driver_version, 80)
      || !SHA256.test(String(device.uuid_sha256 || "")) || !SHA256.test(String(device.pci_bus_id_sha256 || ""))) {
      throw new Error("NVIDIA accelerator device identity is invalid.");
    }
    indices.add(device.index);
    validateExactBytes(device.memory_total_bytes, "device memory");
    for (const [field, number, key] of [["maximum SM clock", device.maximum_sm_clock_mhz, "maximum_sm_clock_mhz"], ["maximum memory clock", device.maximum_memory_clock_mhz, "maximum_memory_clock_mhz"]]) {
      if (number != null && (!Number.isSafeInteger(number) || number <= 0)) throw new Error(`NVIDIA accelerator ${field} is invalid.`);
      if (number == null && !device.unexposed_fields?.includes(key)) throw new Error(`NVIDIA accelerator ${field} absence is not declared.`);
    }
    if (device.uuid != null && !text(device.uuid, 160)) throw new Error("NVIDIA accelerator raw UUID is invalid.");
    if (device.pci_bus_id != null && !text(device.pci_bus_id, 80)) throw new Error("NVIDIA accelerator raw PCI bus ID is invalid.");
    if (device.sm_count != null || device.memory_bus_width_bits != null) throw new Error("NVIDIA accelerator profile must not infer fields not exposed by the collector contract.");
  }
  if (JSON.stringify([...value.devices].sort((left, right) => left.index - right.index)) !== JSON.stringify(value.devices)) {
    throw new Error("NVIDIA accelerator devices must be sorted by index.");
  }
  validateSoftware(value.software?.nvidia_driver, "NVIDIA driver", true);
  validateSoftware(value.software?.cuda_driver_api, "CUDA driver API", false);
  validateSoftware(value.software?.cuda_toolkit, "CUDA toolkit", false);
  validateSoftware(value.software?.tensorrt, "TensorRT", false);
  if (value.devices.some((device) => device.driver_version !== value.software.nvidia_driver.version)) {
    throw new Error("NVIDIA accelerator device and software driver versions differ.");
  }
  const tools = value.collection?.tools;
  if (!Array.isArray(tools) || !tools.length || tools.length > 16 || !tools.some((tool) => tool.role === "nvidia_smi" && tool.status === "observed")) {
    throw new Error("NVIDIA accelerator collector tool inventory is invalid.");
  }
  for (const tool of tools) {
    if (!text(tool.role, 80) || !["observed", "execution_failed", "not_installed"].includes(tool.status)) throw new Error("NVIDIA accelerator tool status is invalid.");
    if (tool.status === "observed" || tool.status === "execution_failed") {
      if (!text(tool.executable_name, 200) || !SHA256.test(String(tool.executable_sha256 || ""))
        || !SHA256.test(String(tool.observation_sha256 || ""))) throw new Error("NVIDIA accelerator observed tool identity is invalid.");
    } else if (tool.executable_name != null || tool.executable_sha256 != null || tool.observation_sha256 != null) {
      throw new Error("NVIDIA accelerator absent tool must not carry invented identity.");
    }
  }
  const roofline = value.roofline_contract;
  if (roofline?.status !== "not_assessable_missing_exact_hardware_contract" || roofline.theoretical_compute_ceiling != null
    || roofline.theoretical_memory_bandwidth != null || !Array.isArray(roofline.missing_fields)
    || !roofline.missing_fields.includes("sm_count") || !roofline.missing_fields.includes("memory_bus_width_bits")) {
    throw new Error("NVIDIA accelerator roofline boundary is invalid.");
  }
  if (!text(value.interpretation_boundary, 1200)) throw new Error("NVIDIA accelerator profile interpretation boundary is invalid.");
}

function validateSoftware(value, label, required) {
  if (!value || !["observed", "not_installed", "not_exposed"].includes(value.status)) throw new Error(`${label} status is invalid.`);
  if (required && value.status !== "observed") throw new Error(`${label} must be observed.`);
  if (value.status === "observed" && !text(value.version, 120)) throw new Error(`${label} version is invalid.`);
  if (value.status !== "observed" && value.version != null) throw new Error(`${label} absent status must not carry a version.`);
}

function validateExactBytes(value, label) {
  const decimal = String(value?.decimal || "");
  if (!/^\d+$/.test(decimal) || BigInt(decimal) <= 0n) throw new Error(`NVIDIA accelerator ${label} exact value is invalid.`);
  const exact = BigInt(decimal);
  const expected = exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null;
  if ((value.number == null ? null : Number(value.number)) !== expected) throw new Error(`NVIDIA accelerator ${label} number does not match its exact decimal value.`);
}

function text(value, maximum) {
  const normalized = String(value || "").trim();
  return normalized && normalized.length <= maximum;
}

function structuredCopy(value) {
  return JSON.parse(JSON.stringify(value));
}
