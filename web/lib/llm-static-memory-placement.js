import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const LLM_STATIC_MEMORY_PROFILE_SCHEMA = "deepbom.llm_static_memory_profile.v1";
export const LLM_STATIC_MEMORY_PLACEMENT_SCHEMA = "deepbom.llm_static_memory_placement.v1";

const POOLS = new Set(["cpu", "accelerator"]);
const ORDERS = new Set(["highest_index_first", "lowest_index_first"]);
const MAX_LAYER_CANDIDATES = 4096;

export function buildLlmStaticMemoryPlacement(contract, analysis, sidecar) {
  if (!sidecar?.document) return unbound();
  const profile = normalizeProfile(sidecar.document, contract, analysis);
  const layerStorage = contract?.storage?.layer_storage;
  if (layerStorage?.status !== "assessed_exact_serialized_layer_storage"
    || layerStorage?.conservation?.status !== "pass") {
    throw new Error("LLM static memory placement requires an exact conserved layer-storage ledger.");
  }
  const layers = layerStorage.layers || [];
  if (!layers.length || layers.length > MAX_LAYER_CANDIDATES
    || layers.some((row, index) => row.layer_index !== index || exactFrom(row.serialized_bytes) == null)) {
    throw new Error(`LLM static memory placement requires 1-${MAX_LAYER_CANDIDATES} contiguous exact layer rows.`);
  }

  const state = logicalStateBytes(contract, profile.policy);
  const stateBytes = requiredExact(state.bytes, "logical state bytes");
  const nonLayer = requiredExact(layerStorage.non_layer_bytes, "non-layer serialized bytes");
  const layerBytes = layers.map((row) => requiredExact(row.serialized_bytes, `layer ${row.layer_index} serialized bytes`));
  const totalLayerBytes = layerBytes.reduce((sum, value) => sum + value, 0n);
  const serializedTotal = requiredExact(layerStorage.serialized_tensor_bytes, "serialized tensor bytes");
  if (totalLayerBytes + nonLayer !== serializedTotal) throw new Error("LLM layer and non-layer bytes do not conserve against serialized tensor bytes.");

  const order = profile.policy.layer_order === "highest_index_first"
    ? layers.map((row) => row.layer_index).reverse()
    : layers.map((row) => row.layer_index);
  const acceleratorSet = new Set();
  let acceleratorLayerBytes = 0n;
  const candidates = [];
  for (let count = 0; count <= layers.length; count += 1) {
    if (count > 0) {
      const index = order[count - 1];
      acceleratorSet.add(index);
      acceleratorLayerBytes += layerBytes[index];
    }
    const cpuLayerBytes = totalLayerBytes - acceleratorLayerBytes;
    const cpuRequired = cpuLayerBytes
      + (profile.policy.non_layer_pool === "cpu" ? nonLayer : 0n)
      + (profile.policy.state_pool === "cpu" ? stateBytes : 0n);
    const acceleratorRequired = acceleratorLayerBytes
      + (profile.policy.non_layer_pool === "accelerator" ? nonLayer : 0n)
      + (profile.policy.state_pool === "accelerator" ? stateBytes : 0n);
    const cpu = comparePool(cpuRequired, profile.capacities.cpu_bytes, profile.reserves.cpu_bytes);
    const accelerator = comparePool(acceleratorRequired, profile.capacities.accelerator_bytes, profile.reserves.accelerator_bytes);
    candidates.push({
      accelerator_layer_count: count,
      accelerator_layer_indices: [...acceleratorSet].sort((left, right) => left - right),
      cpu_layer_count: layers.length - count,
      cpu_layer_serialized_bytes: exact(cpuLayerBytes),
      accelerator_layer_serialized_bytes: exact(acceleratorLayerBytes),
      cpu_accounted_lower_bound_bytes: exact(cpuRequired),
      accelerator_accounted_lower_bound_bytes: exact(acceleratorRequired),
      cpu_capacity_assessment: cpu,
      accelerator_capacity_assessment: accelerator,
      status: cpu.exceeds || accelerator.exceeds
        ? "accounted_lower_bound_exceeds_at_least_one_pool"
        : "accounted_lower_bound_not_exceeding_pools_fit_unresolved",
    });
  }
  const notDisproven = candidates.filter((row) => row.status === "accounted_lower_bound_not_exceeding_pools_fit_unresolved");
  const normalizedProfile = {
    schema: LLM_STATIC_MEMORY_PROFILE_SCHEMA,
    artifact: profile.artifact,
    capacities: mapExact(profile.capacities),
    reserves: mapExact(profile.reserves),
    policy: profile.policy,
  };
  return {
    schema: LLM_STATIC_MEMORY_PLACEMENT_SCHEMA,
    status: notDisproven.length ? "assessed_lower_bound_candidates" : "assessed_all_candidates_exceed_lower_bound",
    evidence_class: "OBSERVED_SERIALIZED_STORAGE/DERIVED_CONDITIONAL_STATIC_LOWER_BOUND",
    source: sidecar.path || null,
    source_sha256: validSha(sidecar.sha256) ? sidecar.sha256 : null,
    normalized_profile: normalizedProfile,
    normalized_profile_sha256: sha256TextHex(canonicalJson(normalizedProfile)),
    layer_count: layers.length,
    serialized_tensor_bytes: exact(serializedTotal),
    layer_serialized_bytes: exact(totalLayerBytes),
    non_layer_serialized_bytes: exact(nonLayer),
    logical_state: state,
    candidate_count: candidates.length,
    candidates,
    lower_bound_not_exceeding_candidate_count: notDisproven.length,
    minimum_accelerator_layer_count_not_disproven: notDisproven.length ? notDisproven[0].accelerator_layer_count : null,
    maximum_accelerator_layer_count_not_disproven: notDisproven.length ? notDisproven.at(-1).accelerator_layer_count : null,
    fit_claim: "not_emitted",
    conservation: {
      status: "pass",
      equation: "layer_serialized_bytes + non_layer_serialized_bytes = serialized_tensor_bytes; every candidate assigns each serialized byte and logical-state byte to exactly one declared pool",
    },
    boundary: "Candidates are exact only for serialized tensor bytes and the logical KV/SSM state under the declared profile. Exceeding either effective pool capacity proves that candidate insufficient under these assumptions. Not exceeding both capacities does not prove fit because runtime-expanded or repacked weights, replicas, workspace, allocator overhead, backend-private allocations, application memory, operating-system reserve beyond the declared reserve, and physical placement remain unbound.",
  };
}

export function validateLlmStaticMemoryPlacement(value, contract, analysis) {
  if (value?.status === "not_bound") return value.schema === LLM_STATIC_MEMORY_PLACEMENT_SCHEMA ? [] : ["llm_static_memory_placement_schema_invalid"];
  if (!value?.normalized_profile) return ["llm_static_memory_profile_missing"];
  try {
    const expected = buildLlmStaticMemoryPlacement(contract, analysis, {
      document: value.normalized_profile,
      path: value.source,
      sha256: value.source_sha256,
    });
    return canonicalJson(value) === canonicalJson(expected) ? [] : ["llm_static_memory_placement_recomputation_mismatch"];
  } catch (error) {
    return [`llm_static_memory_placement_invalid:${error?.message || error}`];
  }
}

function normalizeProfile(document, contract, analysis) {
  if (!document || typeof document !== "object" || Array.isArray(document)
    || document.schema !== LLM_STATIC_MEMORY_PROFILE_SCHEMA) {
    throw new Error(`LLM static memory profile must use ${LLM_STATIC_MEMORY_PROFILE_SCHEMA}.`);
  }
  const artifactSha = requireSha(document.artifact?.sha256, "profile artifact SHA-256");
  const activeSha = requireSha(analysis?.artifact_bundle?.model_source_sha256 || analysis?.model_sha256, "active model-source SHA-256");
  const format = String(document.artifact?.format || "").toLowerCase();
  if (artifactSha !== activeSha || format !== String(contract?.format || "").toLowerCase()) {
    throw new Error("LLM static memory profile is not bound to the active artifact identity.");
  }
  const capacities = {
    cpu_bytes: requiredPositive(document.capacities?.cpu_bytes, "CPU capacity"),
    accelerator_bytes: requiredPositive(document.capacities?.accelerator_bytes, "accelerator capacity"),
  };
  const reserves = {
    cpu_bytes: requiredNonNegative(document.reserves?.cpu_bytes, "CPU reserve"),
    accelerator_bytes: requiredNonNegative(document.reserves?.accelerator_bytes, "accelerator reserve"),
  };
  if (reserves.cpu_bytes >= capacities.cpu_bytes || reserves.accelerator_bytes >= capacities.accelerator_bytes) {
    throw new Error("Each LLM memory reserve must be smaller than its declared pool capacity.");
  }
  const layerOrder = String(document.policy?.layer_order || "");
  const nonLayerPool = String(document.policy?.non_layer_pool || "");
  const statePool = String(document.policy?.state_pool || "");
  if (!ORDERS.has(layerOrder) || !POOLS.has(nonLayerPool) || !POOLS.has(statePool)) {
    throw new Error("LLM memory policy must bind layer order, non-layer pool, and state pool.");
  }
  const batch = positiveInteger(document.policy?.batch_size, "batch size");
  const bits = positiveInteger(document.policy?.state_storage_bits, "state storage bits");
  if (bits % 8 !== 0) throw new Error("LLM state storage bits must be byte-aligned.");
  const context = document.policy?.context_length == null ? null : positiveInteger(document.policy.context_length, "context length");
  return {
    artifact: { format, sha256: artifactSha },
    capacities,
    reserves,
    policy: {
      layer_order: layerOrder,
      non_layer_pool: nonLayerPool,
      state_pool: statePool,
      context_length: context,
      batch_size: batch,
      state_storage_bits: bits,
    },
  };
}

function logicalStateBytes(contract, policy) {
  const bytesPerElement = BigInt(policy.state_storage_bits / 8);
  const kv = exactFrom(contract?.state?.kv_projection?.elements_per_token_per_batch);
  const recurrent = exactFrom(contract?.state?.recurrent_projection?.recurrent_state_elements_all_layers_per_batch);
  if (kv != null && recurrent != null) {
    if (policy.context_length == null) throw new Error("Hybrid KV/SSM placement requires a declared context length.");
    const kvElements = kv * BigInt(policy.context_length);
    const elements = (kvElements + recurrent) * BigInt(policy.batch_size);
    return {
      kind: "hybrid_kv_ssm", context_length: policy.context_length, batch_size: policy.batch_size,
      storage_bits: policy.state_storage_bits, kv_elements_per_batch: exact(kvElements), recurrent_elements_per_batch: exact(recurrent), bytes: exact(elements * bytesPerElement),
    };
  }
  if (kv != null) {
    if (policy.context_length == null) throw new Error("Transformer KV placement requires a declared context length.");
    const bytes = kv * BigInt(policy.context_length) * BigInt(policy.batch_size) * bytesPerElement;
    return { kind: "transformer_kv", context_length: policy.context_length, batch_size: policy.batch_size, storage_bits: policy.state_storage_bits, bytes: exact(bytes) };
  }
  if (recurrent != null) {
    if (policy.context_length != null) throw new Error("SSM recurrent-state placement must not declare a context length.");
    const bytes = recurrent * BigInt(policy.batch_size) * bytesPerElement;
    return { kind: "ssm_recurrent", context_length: null, batch_size: policy.batch_size, storage_bits: policy.state_storage_bits, bytes: exact(bytes) };
  }
  throw new Error("LLM static memory placement requires a complete KV or recurrent-state contract.");
}

function comparePool(required, capacity, reserve) {
  const effective = capacity - reserve;
  const exceeds = required > effective;
  return {
    declared_capacity_bytes: exact(capacity),
    declared_reserve_bytes: exact(reserve),
    effective_capacity_bytes: exact(effective),
    accounted_lower_bound_bytes: exact(required),
    status: exceeds ? "accounted_lower_bound_exceeds_effective_capacity" : "accounted_lower_bound_at_or_below_effective_capacity_fit_unresolved",
    exceeds,
    deficit_bytes: exceeds ? exact(required - effective) : null,
    headroom_after_accounted_lower_bound_bytes: exceeds ? null : exact(effective - required),
    fit_claim: "not_emitted",
  };
}

function unbound() {
  return {
    schema: LLM_STATIC_MEMORY_PLACEMENT_SCHEMA,
    status: "not_bound",
    evidence_class: "NOT_ASSESSABLE",
    required_profile: ["artifact_identity", "cpu_and_accelerator_capacities", "per_pool_reserves", "layer_order", "non_layer_pool", "state_pool", "context_batch_and_state_storage_width"],
    candidates: [],
    fit_claim: "not_emitted",
    boundary: "No per-pool placement result is emitted until an artifact-bound static memory profile declares both capacities, reserves, and residency policy.",
  };
}

function exactFrom(value) {
  if (value && typeof value === "object" && /^(?:0|[1-9]\d*)$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^(?:0|[1-9]\d*)$/.test(value)) return BigInt(value);
  if (typeof value === "bigint" && value >= 0n) return value;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function requiredExact(value, label) {
  const result = exactFrom(value);
  if (result == null) throw new Error(`${label} must be an exact non-negative integer.`);
  return result;
}

function requiredPositive(value, label) {
  const result = requiredExact(value, label);
  if (result <= 0n) throw new Error(`${label} must be positive.`);
  return result;
}

function requiredNonNegative(value, label) {
  return requiredExact(value, label);
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive safe integer.`);
  return value;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

function mapExact(values) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, exact(value)]));
}

function validSha(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function requireSha(value, label) {
  if (!validSha(value)) throw new Error(`${label} must be a lowercase SHA-256 digest.`);
  return value;
}
