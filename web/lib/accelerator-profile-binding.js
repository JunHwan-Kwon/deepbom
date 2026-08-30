import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { validateNvidiaAcceleratorProfile } from "./nvidia-accelerator-profile.js";

export const ACCELERATOR_PROFILE_BINDING_SCHEMA = "deepbom.accelerator_profile_binding.v1";
const SHA256 = /^[a-f0-9]{64}$/;

export function buildNvidiaAcceleratorProfileBinding(analysis, sidecar, { deviceIndex = null } = {}) {
  if (!validSha(sidecar?.sha256)) throw new Error("NVIDIA accelerator profile sidecar SHA-256 is required.");
  const profile = validateNvidiaAcceleratorProfile(sidecar?.document);
  const selected = selectDevice(profile.devices, deviceIndex);
  const body = {
    schema: ACCELERATOR_PROFILE_BINDING_SCHEMA,
    status: "observed_host_profile_bound_selected_build_unbound",
    evidence_class: "OBSERVED_HOST_TOOLING/DERIVED_CONDITIONAL_CAPACITY",
    source: sidecar?.path || null,
    source_sha256: sidecar.sha256,
    profile_sha256: profile.profile_sha256,
    selected_device: {
      index: selected.index,
      name: selected.name,
      compute_capability: selected.compute_capability,
      driver_version: selected.driver_version,
      memory_total_bytes: normalizeExact(selected.memory_total_bytes),
      uuid_sha256: selected.uuid_sha256,
      pci_bus_id_sha256: selected.pci_bus_id_sha256,
    },
    software: profile.software,
    selected_build: {
      status: "not_bound",
      evidence_class: "NOT_ASSESSABLE",
      required_evidence: [
        "runtime_binary_and_library_hashes",
        "compiled_provider_or_backend_registry",
        "precision_and_builder_flags",
      ],
    },
    roofline: profile.roofline_contract,
    llm_accelerator_residency: buildLlmAcceleratorResidency(analysis, selected.memory_total_bytes),
    interpretation_boundary: "This binding proves the selected host-observed NVIDIA device, driver-facing software inventory, and exact physical VRAM reported by the collector. It does not prove that a CUDA, TensorRT, ORT, or other backend was compiled into the selected application; it does not prove provider assignment, engine selection, runtime allocation, latency, throughput, or fit. LLM capacity rows compare static serialized-storage plus logical-state lower bounds with physical VRAM under their stated simultaneous-residency assumption only.",
  };
  return Object.freeze({ ...body, binding_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateNvidiaAcceleratorProfileBinding(value) {
  const body = JSON.parse(JSON.stringify(value || null));
  const declared = String(body?.binding_sha256 || "").toLowerCase();
  if (body) delete body.binding_sha256;
  if (!body || body.schema !== ACCELERATOR_PROFILE_BINDING_SCHEMA
    || body.status !== "observed_host_profile_bound_selected_build_unbound"
    || !validSha(body.profile_sha256) || !validSha(body.source_sha256)
    || body.selected_build?.status !== "not_bound" || body.selected_build?.evidence_class !== "NOT_ASSESSABLE"
    || body.selected_device?.uuid != null || body.selected_device?.pci_bus_id != null) {
    throw new Error("NVIDIA accelerator profile binding structure is invalid.");
  }
  normalizeExact(body.selected_device?.memory_total_bytes);
  for (const scenario of body.llm_accelerator_residency?.scenarios || []) {
    if (scenario.fit_claim !== "not_emitted" || !String(scenario.status || "").includes("lower_bound")) {
      throw new Error("NVIDIA accelerator LLM residency claim boundary is invalid.");
    }
  }
  if (!validSha(declared) || declared !== sha256TextHex(canonicalJson(body))) {
    throw new Error("NVIDIA accelerator profile binding SHA-256 is invalid.");
  }
  return Object.freeze({ ...body, binding_sha256: declared });
}

function selectDevice(devices, deviceIndex) {
  if (deviceIndex == null) {
    if (devices.length !== 1) throw new Error("The accelerator profile contains multiple devices; --accelerator-device is required.");
    return devices[0];
  }
  const selected = devices.find((row) => row.index === deviceIndex);
  if (!selected) throw new Error(`NVIDIA accelerator device index ${deviceIndex} is not present in the profile.`);
  return selected;
}

function buildLlmAcceleratorResidency(analysis, capacityValue) {
  const contract = analysis?.on_device_llm;
  if (!contract) return {
    status: "not_applicable_non_llm_artifact",
    evidence_class: "NOT_APPLICABLE",
    scenarios: [],
    fit_claim: "not_emitted",
  };
  const capacity = exactFrom(capacityValue);
  const feasibility = contract.memory_feasibility;
  const scenarios = [];
  const cliScenario = analysis?.cli_context_scenario;
  if (exactFrom(cliScenario?.memory_feasibility?.static_lower_bound_bytes) != null
    && exactFrom(cliScenario?.memory_feasibility?.logical_kv_state_bytes) != null) {
    scenarios.push({
      scenario_source: "cli_declared",
      state_kind: "transformer_kv",
      context_length: cliScenario.context_length,
      batch_size: cliScenario.batch_size,
      storage_bits: cliScenario.state_storage_bits,
      logical_state_bytes: cliScenario.memory_feasibility.logical_kv_state_bytes,
      static_lower_bound_bytes: cliScenario.memory_feasibility.static_lower_bound_bytes,
    });
  }
  for (const scenario of Array.isArray(feasibility?.static_scenarios) ? feasibility.static_scenarios : []) {
    scenarios.push({ scenario_source: "artifact_registered_matrix", ...scenario });
  }
  if (capacity == null || !scenarios.length) return {
    status: "not_assessable_static_lower_bound_unavailable",
    evidence_class: "NOT_ASSESSABLE",
    scenarios: [],
    fit_claim: "not_emitted",
  };
  const layerStorage = contract.storage?.layer_storage;
  const exactLayers = layerStorage?.status === "assessed_exact_serialized_layer_storage"
    && layerStorage?.conservation?.status === "pass" && Array.isArray(layerStorage.layers)
    && layerStorage.layers.every((row, index) => row.layer_index === index && exactFrom(row.serialized_bytes) != null);
  const rows = scenarios.map((scenario) => {
    const required = exactFrom(scenario.static_lower_bound_bytes);
    if (required == null) return null;
    const exceeds = required > capacity;
    const row = {
      scenario_source: scenario.scenario_source,
      state_kind: scenario.state_kind,
      context_length: scenario.context_length,
      batch_size: scenario.batch_size,
      storage_bits: scenario.storage_bits,
      physical_vram_bytes: exact(capacity),
      simultaneous_all_accelerator_lower_bound_bytes: exact(required),
      status: exceeds
        ? "simultaneous_residency_lower_bound_exceeds_physical_vram"
        : "simultaneous_residency_lower_bound_at_or_below_physical_vram_fit_unresolved",
      deficit_bytes: exceeds ? exact(required - capacity) : null,
      headroom_after_lower_bound_bytes: exceeds ? null : exact(capacity - required),
      fit_claim: "not_emitted",
    };
    if (exactLayers) row.serialized_layer_offload = layerOffloadProjection(layerStorage, scenario.logical_state_bytes, capacity);
    return row;
  }).filter(Boolean);
  return {
    status: rows.some((row) => row.status === "simultaneous_residency_lower_bound_exceeds_physical_vram")
      ? "assessed_with_insufficient_scenarios"
      : "assessed_lower_bounds_do_not_prove_fit",
    evidence_class: "OBSERVED_SERIALIZED_STORAGE/OBSERVED_HOST_VRAM/DERIVED_CONDITIONAL_SCENARIO",
    capacity_scope: "selected_nvidia_device_physical_vram",
    residency_assumption: feasibility?.residency_assumption || null,
    scenarios: rows,
    fit_claim: "not_emitted",
    boundary: "A row exceeding physical VRAM proves that the stated simultaneous all-accelerator lower-bound scenario is insufficient. A row at or below VRAM does not prove fit. Serialized layer offload rows count serialized bytes only and exclude backend repacking, workspace, allocator overhead, replicas, application allocations, and runtime placement.",
  };
}

function layerOffloadProjection(layerStorage, stateValue, capacity) {
  const state = exactFrom(stateValue);
  const nonLayer = exactFrom(layerStorage.non_layer_bytes);
  if (state == null || nonLayer == null) return null;
  const ordered = [...layerStorage.layers].sort((left, right) => right.layer_index - left.layer_index);
  let used = state + nonLayer;
  let count = 0;
  const indices = [];
  if (used <= capacity) {
    for (const layer of ordered) {
      const bytes = exactFrom(layer.serialized_bytes);
      if (bytes == null || used + bytes > capacity) break;
      used += bytes;
      count += 1;
      indices.push(layer.layer_index);
    }
  }
  return {
    policy: "highest_index_first_serialized_bytes",
    non_layer_and_state_lower_bound_bytes: exact(state + nonLayer),
    maximum_layer_count_not_exceeding_physical_vram_lower_bound: count,
    selected_layer_indices: indices.sort((left, right) => left - right),
    accounted_accelerator_lower_bound_bytes: exact(used),
    status: state + nonLayer > capacity
      ? "non_layer_and_state_lower_bound_exceeds_physical_vram"
      : "serialized_lower_bound_not_exceeding_physical_vram_fit_unresolved",
    fit_claim: "not_emitted",
  };
}

function exactFrom(value) {
  if (value && typeof value === "object" && /^(?:0|[1-9]\d*)$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^(?:0|[1-9]\d*)$/.test(value)) return BigInt(value);
  if (typeof value === "bigint" && value >= 0n) return value;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function normalizeExact(value) {
  const parsed = exactFrom(value);
  if (parsed == null) throw new Error("NVIDIA accelerator memory value is invalid.");
  return exact(parsed);
}

function exact(value) {
  return { decimal: String(value), number: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null };
}

function validSha(value) {
  return typeof value === "string" && SHA256.test(value);
}
