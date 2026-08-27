export const LLM_MEMORY_FEASIBILITY_SCHEMA = "deepbom.llm_memory_feasibility.v1.1";
export const LLM_STATIC_RESIDENCY_ASSUMPTION = "All serialized tensor bytes and the logical state bytes are simultaneously resident in one aggregate primary-memory budget; storage-backed weight paging, remote execution, and capacity split across independently constrained memory pools are excluded.";

const MIB = 1024n ** 2n;
const GIB = 1024n ** 3n;

export const LLM_MEMORY_CAPACITY_TIERS = Object.freeze([
  ["512 MiB", 512n * MIB],
  ["1 GiB", GIB],
  ["2 GiB", 2n * GIB],
  ["4 GiB", 4n * GIB],
  ["8 GiB", 8n * GIB],
  ["16 GiB", 16n * GIB],
  ["24 GiB", 24n * GIB],
  ["32 GiB", 32n * GIB],
  ["48 GiB", 48n * GIB],
  ["64 GiB", 64n * GIB],
  ["96 GiB", 96n * GIB],
  ["128 GiB", 128n * GIB],
].map(([label, bytes]) => Object.freeze({ label, bytes: String(bytes) })));

function exactFrom(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  if (typeof value === "bigint" && value >= 0n) return value;
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

export function compareLlmMemoryCapacity(requiredValue, capacityValue) {
  const requiredBytes = exactFrom(requiredValue);
  const capacityBytes = exactFrom(capacityValue);
  if (requiredBytes == null) return {
    status: "not_assessable_lower_bound_unbound",
    deficit_bytes: null,
    headroom_after_lower_bound_bytes: null,
    fit_claim: "not_emitted",
  };
  if (capacityBytes == null) return {
    status: "lower_bound_derived_capacity_unbound",
    deficit_bytes: null,
    headroom_after_lower_bound_bytes: null,
    fit_claim: "not_emitted",
  };
  return {
    status: requiredBytes > capacityBytes
      ? "lower_bound_exceeds_capacity"
      : "lower_bound_at_or_below_capacity_fit_unresolved",
    deficit_bytes: requiredBytes > capacityBytes ? exact(requiredBytes - capacityBytes) : null,
    headroom_after_lower_bound_bytes: requiredBytes <= capacityBytes ? exact(capacityBytes - requiredBytes) : null,
    fit_claim: "not_emitted",
  };
}

function capacityAssessment(requiredBytes, tiers) {
  let firstCapacityNotExceeded = null;
  let lowerBoundExceededCapacityCount = 0;
  for (const tier of tiers) {
    const capacity = BigInt(tier.bytes);
    if (requiredBytes > capacity) lowerBoundExceededCapacityCount += 1;
    else if (firstCapacityNotExceeded == null) firstCapacityNotExceeded = tier.label;
  }
  return {
    first_capacity_not_exceeded: firstCapacityNotExceeded || `>${tiers.at(-1)?.label || "largest reference tier"}`,
    lower_bound_exceeded_capacity_count: lowerBoundExceededCapacityCount,
  };
}

function scenarioRows(contract, weightFloor, tiers) {
  const rows = Array.isArray(contract?.state?.scenario_matrix) ? contract.state.scenario_matrix : [];
  if (!rows.length) {
    const capacity = capacityAssessment(weightFloor, tiers);
    return [{
      state_kind: "state_contract_unbound",
      context_length: null,
      batch_size: null,
      storage_bits: null,
      serialized_weight_floor_bytes: exact(weightFloor),
      logical_state_bytes: null,
      static_lower_bound_bytes: exact(weightFloor),
      evidence_class: "OBSERVED_SERIALIZED_STORAGE",
      ...capacity,
    }];
  }
  return rows.map((row) => {
    const stateBytes = exactFrom(row.logical_bytes);
    if (stateBytes == null) return null;
    const lowerBound = weightFloor + stateBytes;
    return {
      state_kind: row.state_kind,
      context_length: row.context_length ?? null,
      batch_size: row.batch_size,
      storage_bits: row.storage_bits,
      serialized_weight_floor_bytes: exact(weightFloor),
      logical_state_bytes: exact(stateBytes),
      static_lower_bound_bytes: exact(lowerBound),
      evidence_class: "OBSERVED_STORAGE_PLUS_DERIVED_CONDITIONAL_STATE",
      ...capacityAssessment(lowerBound, tiers),
    };
  }).filter(Boolean);
}

function runtimePrimaryResidency(contract, tiers) {
  const runtime = contract?.runtime_contract;
  if (!runtime || !["artifact_bound_declared_runtime", "artifact_bound_observed_runtime"].includes(runtime.status)) return null;
  const cpu = exactFrom(runtime.weight_residency?.cpu_bytes);
  const accelerator = exactFrom(runtime.weight_residency?.accelerator_bytes);
  const stateResident = exactFrom(runtime.state_cache?.resident_bytes);
  const stateAllocated = exactFrom(runtime.state_cache?.allocated_bytes);
  if ([cpu, accelerator, stateResident, stateAllocated].some((value) => value == null)) return null;
  const resident = cpu + accelerator + stateResident;
  const working = exactFrom(runtime.working_memory?.accounted_nonweight_runtime_bytes);
  const allocatedLowerBound = cpu + accelerator + stateAllocated;
  const allocatedAccounted = working == null ? null : allocatedLowerBound + working;
  return {
    status: allocatedAccounted != null
      ? runtime.status === "artifact_bound_observed_runtime" ? "assessed_observed_accounted_primary_allocation" : "assessed_declared_accounted_primary_allocation"
      : runtime.status === "artifact_bound_observed_runtime" ? "assessed_observed_primary_residency_lower_bound" : "assessed_declared_primary_residency_lower_bound",
    evidence_class: runtime.evidence_class,
    context_length: runtime.deployment?.context_length ?? null,
    batch_size: runtime.deployment?.batch_size ?? null,
    state_storage_bits: runtime.deployment?.state_storage_bits ?? null,
    exclusive_weight_resident_bytes: exact(cpu + accelerator),
    state_resident_bytes: exact(stateResident),
    state_allocated_bytes: exact(stateAllocated),
    primary_resident_lower_bound_bytes: exact(resident),
    primary_allocated_lower_bound_bytes: exact(allocatedLowerBound),
    working_memory_accounted_bytes: working == null ? null : exact(working),
    primary_allocated_accounted_bytes: allocatedAccounted == null ? null : exact(allocatedAccounted),
    allocation_accounting_status: allocatedAccounted == null ? "working_memory_categories_unbound" : runtime.working_memory.coverage,
    capacity_scope: "Aggregate CPU plus accelerator primary residency. Per-pool capacity constraints are not assessed.",
    resident_capacity_assessment: capacityAssessment(resident, tiers),
    allocated_capacity_assessment: capacityAssessment(allocatedAccounted ?? allocatedLowerBound, tiers),
    fit_claim: "not_emitted",
    boundary: allocatedAccounted == null
      ? "Exclusive primary weight residency plus state-cache residency/allocation from the bound manifest. Working-memory categories are unbound, so the allocated value remains a lower bound. CPU and accelerator bytes are aggregated and do not establish fit within either pool."
      : "Exclusive primary weight residency, state allocation, and six explicitly conserved working-memory categories from the bound manifest. The accounted value aggregates CPU and accelerator allocations and is not process RSS or a per-pool fit claim; application memory and operating-system reserve remain external.",
  };
}

export function buildLlmMemoryFeasibility(contract, { capacityTiers = LLM_MEMORY_CAPACITY_TIERS } = {}) {
  const tiers = capacityTiers.map((row) => ({ label: String(row.label), bytes: String(row.bytes) }));
  if (!tiers.length || tiers.some((row, index) => !row.label || !/^\d+$/.test(row.bytes)
    || BigInt(row.bytes) <= 0n || (index > 0 && BigInt(row.bytes) <= BigInt(tiers[index - 1].bytes)))) {
    throw new Error("LLM memory capacity tiers must be non-empty, positive, and strictly increasing");
  }
  const weightFloor = exactFrom(contract?.storage?.serialized_tensor_bytes_decimal);
  if (weightFloor == null) return {
    schema: LLM_MEMORY_FEASIBILITY_SCHEMA,
    status: "not_assessable_serialized_weight_bytes_unbound",
    evidence_class: "NOT_ASSESSABLE",
    capacity_scope: "single_aggregate_primary_memory_budget",
    residency_assumption: LLM_STATIC_RESIDENCY_ASSUMPTION,
    reference_capacity_tiers: tiers.map((row) => ({ label: row.label, bytes: exact(BigInt(row.bytes)) })),
    static_scenarios: [],
    runtime_primary_residency: null,
    fit_claim: "not_emitted",
    boundary: "No fit claim is emitted because exact serialized tensor bytes are unavailable.",
  };
  const staticScenarios = scenarioRows(contract, weightFloor, tiers);
  const lowerBounds = staticScenarios.map((row) => exactFrom(row.static_lower_bound_bytes)).filter((value) => value != null);
  return {
    schema: LLM_MEMORY_FEASIBILITY_SCHEMA,
    status: staticScenarios.some((row) => row.logical_state_bytes == null)
      ? "partial_serialized_weight_floor_only"
      : "assessed_static_lower_bound_scenarios",
    evidence_class: staticScenarios.some((row) => row.logical_state_bytes == null)
      ? "OBSERVED"
      : "OBSERVED/DERIVED_CONDITIONAL_SCENARIO",
    capacity_scope: "single_aggregate_primary_memory_budget",
    residency_assumption: LLM_STATIC_RESIDENCY_ASSUMPTION,
    serialized_weight_floor_bytes: exact(weightFloor),
    minimum_static_lower_bound_bytes: lowerBounds.length ? exact(lowerBounds.reduce((left, right) => left < right ? left : right)) : null,
    maximum_static_lower_bound_bytes: lowerBounds.length ? exact(lowerBounds.reduce((left, right) => left > right ? left : right)) : null,
    reference_capacity_tiers: tiers.map((row) => ({ label: row.label, bytes: exact(BigInt(row.bytes)) })),
    static_scenarios: staticScenarios,
    runtime_primary_residency: runtimePrimaryResidency(contract, tiers),
    fit_claim: "not_emitted",
    boundary: "Under the explicit simultaneous-residency assumption, static rows add exact serialized tensor bytes to logical KV/SSM state bytes. A smaller aggregate capacity is insufficient only under that assumption. A capacity at or above the lower bound is not proven sufficient. A v2 artifact-bound runtime manifest may separately account for working-memory categories, but per-pool capacity, application memory, and operating-system reserve remain external.",
  };
}

export function validateLlmMemoryFeasibility(contract) {
  const observed = contract?.memory_feasibility;
  if (!observed || observed.schema !== LLM_MEMORY_FEASIBILITY_SCHEMA) return ["llm_memory_feasibility_schema_invalid"];
  const expected = buildLlmMemoryFeasibility({ ...contract, memory_feasibility: undefined });
  return JSON.stringify(observed) === JSON.stringify(expected) ? [] : ["llm_memory_feasibility_recomputation_mismatch"];
}
