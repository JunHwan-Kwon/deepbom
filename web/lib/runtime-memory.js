import { sha256TextHex } from "./sha256-sync.js";

const RUNTIME_MEMORY_SCHEMA = "deepbom.runtime_memory.v1";
const RECONCILIATION_SCHEMA = "deepbom.arena_runtime_reconciliation.v1";
const TENSORFLOW_SOURCE_COMMIT = "87bbf65b8d23d3f06912b1b2183587e1884bc45c";
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const ARENAS = new Set(["kTfLiteArenaRw", "kTfLiteArenaRwPersistent"]);

function integer(value, field, { positive = false } = {}) {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < (positive ? 1 : 0)) {
    throw new Error(`Runtime memory ${field} must be a ${positive ? "positive" : "non-negative"} safe integer.`);
  }
  return value;
}

function checkedAdd(left, right, field) {
  const value = left + right;
  if (!Number.isSafeInteger(value)) throw new Error(`Runtime memory ${field} exceeds the safe integer range.`);
  return value;
}

function intervalsOverlap(left, right) {
  const leftLast = left.last_node == null ? Number.MAX_SAFE_INTEGER : left.last_node;
  const rightLast = right.last_node == null ? Number.MAX_SAFE_INTEGER : right.last_node;
  return left.first_node <= rightLast && right.first_node <= leftLast;
}

function byteRangesOverlap(left, right) {
  return left.offset_bytes < right.offset_bytes + right.size_bytes
    && right.offset_bytes < left.offset_bytes + left.size_bytes;
}

function canonicalRows(values, key, field) {
  const seen = new Set();
  let previous = -1;
  for (const value of values) {
    const current = key(value);
    if (seen.has(current) || current <= previous) throw new Error(`Runtime memory ${field} must be unique and canonically sorted.`);
    seen.add(current);
    previous = current;
  }
}

function validateSnapshot(source, expectedId, artifactTensorCount) {
  const snapshotId = integer(source?.memory_snapshot_id, `snapshots[${expectedId}].memory_snapshot_id`);
  if (snapshotId !== expectedId) throw new Error("Runtime memory snapshot IDs must be contiguous from zero.");
  const nonPersistent = integer(source?.non_persistent_arena_bytes, `snapshots[${expectedId}].non_persistent_arena_bytes`);
  const persistent = integer(source?.persistent_arena_bytes, `snapshots[${expectedId}].persistent_arena_bytes`);
  const combined = integer(source?.combined_arena_bytes, `snapshots[${expectedId}].combined_arena_bytes`);
  if (combined !== checkedAdd(nonPersistent, persistent, `snapshots[${expectedId}].combined_arena_bytes`)) {
    throw new Error(`Runtime memory snapshot ${expectedId} combined arena bytes do not conserve both arenas.`);
  }
  const tensorCount = integer(source?.tensor_count, `snapshots[${expectedId}].tensor_count`, { positive: true });
  const executionNodeCount = integer(source?.execution_node_count, `snapshots[${expectedId}].execution_node_count`, { positive: true });
  if (tensorCount < artifactTensorCount) throw new Error(`Runtime memory snapshot ${expectedId} omits artifact tensors.`);
  const sourceAllocations = source?.allocations;
  const sourceAliases = source?.aliases;
  if (!Array.isArray(sourceAllocations) || sourceAllocations.length > 100000
    || !Array.isArray(sourceAliases) || sourceAliases.length > 100000) {
    throw new Error(`Runtime memory snapshot ${expectedId} allocation or alias inventory is invalid.`);
  }
  const allocations = sourceAllocations.map((item, index) => {
    const tensorIndex = integer(item?.tensor_index, `snapshots[${expectedId}].allocations[${index}].tensor_index`);
    const arena = String(item?.arena || "");
    if (tensorIndex >= tensorCount || !ARENAS.has(arena)) throw new Error(`Runtime memory snapshot ${expectedId} allocation ${index} has an invalid tensor or arena.`);
    const offsetBytes = integer(item?.offset_bytes, `snapshots[${expectedId}].allocations[${index}].offset_bytes`);
    const sizeBytes = integer(item?.size_bytes, `snapshots[${expectedId}].allocations[${index}].size_bytes`, { positive: true });
    const firstNode = integer(item?.first_node, `snapshots[${expectedId}].allocations[${index}].first_node`);
    const lastNode = item?.last_node == null ? null : integer(item.last_node, `snapshots[${expectedId}].allocations[${index}].last_node`);
    if (firstNode >= executionNodeCount || (lastNode != null && (lastNode < firstNode || lastNode >= executionNodeCount))) {
      throw new Error(`Runtime memory snapshot ${expectedId} allocation T${tensorIndex} has an invalid lifetime.`);
    }
    const end = checkedAdd(offsetBytes, sizeBytes, `snapshots[${expectedId}].allocations[${index}] byte end`);
    const limit = arena === "kTfLiteArenaRw" ? nonPersistent : persistent;
    if (end > limit) throw new Error(`Runtime memory snapshot ${expectedId} allocation T${tensorIndex} exceeds ${arena}.`);
    return { tensor_index: tensorIndex, arena, offset_bytes: offsetBytes, size_bytes: sizeBytes, first_node: firstNode, last_node: lastNode };
  });
  canonicalRows(allocations, (item) => item.tensor_index, `snapshot ${expectedId} allocations`);
  const allocationByTensor = new Map(allocations.map((item) => [item.tensor_index, item]));
  for (let leftIndex = 0; leftIndex < allocations.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < allocations.length; rightIndex += 1) {
      const left = allocations[leftIndex];
      const right = allocations[rightIndex];
      if (left.arena === right.arena && intervalsOverlap(left, right) && byteRangesOverlap(left, right)) {
        throw new Error(`Runtime memory snapshot ${expectedId} has overlapping live allocations T${left.tensor_index} and T${right.tensor_index}.`);
      }
    }
  }
  const aliases = sourceAliases.map((item, index) => {
    const tensorIndex = integer(item?.tensor_index, `snapshots[${expectedId}].aliases[${index}].tensor_index`);
    const root = integer(item?.shared_with_tensor_index, `snapshots[${expectedId}].aliases[${index}].shared_with_tensor_index`);
    if (tensorIndex >= tensorCount || root >= tensorCount || tensorIndex === root || !allocationByTensor.has(root)) {
      throw new Error(`Runtime memory snapshot ${expectedId} alias T${tensorIndex} is not rooted in an owning allocation.`);
    }
    return { tensor_index: tensorIndex, shared_with_tensor_index: root };
  });
  canonicalRows(aliases, (item) => item.tensor_index, `snapshot ${expectedId} aliases`);
  const aliasSet = new Set(aliases.map((item) => item.tensor_index));
  if (allocations.some((item) => aliasSet.has(item.tensor_index)) || aliases.some((item) => aliasSet.has(item.shared_with_tensor_index))) {
    throw new Error(`Runtime memory snapshot ${expectedId} has an owning/alias conflict or alias chain.`);
  }
  const allocationCount = integer(source?.allocation_count, `snapshots[${expectedId}].allocation_count`);
  const aliasCount = integer(source?.alias_count, `snapshots[${expectedId}].alias_count`);
  if (allocationCount !== allocations.length || aliasCount !== aliases.length) throw new Error(`Runtime memory snapshot ${expectedId} row counts do not conserve their inventories.`);
  const allocatedIntervalBytes = allocations.reduce((sum, item) => checkedAdd(sum, item.size_bytes, `snapshot ${expectedId} allocated interval bytes`), 0);
  if (integer(source?.allocated_interval_bytes, `snapshots[${expectedId}].allocated_interval_bytes`) !== allocatedIntervalBytes) {
    throw new Error(`Runtime memory snapshot ${expectedId} allocated interval bytes do not conserve allocation rows.`);
  }
  return {
    memory_snapshot_id: snapshotId,
    non_persistent_arena_bytes: nonPersistent,
    persistent_arena_bytes: persistent,
    combined_arena_bytes: combined,
    tensor_count: tensorCount,
    execution_node_count: executionNodeCount,
    allocation_count: allocationCount,
    alias_count: aliasCount,
    allocated_interval_bytes: allocatedIntervalBytes,
    allocations,
    aliases,
  };
}

export function validateRuntimeMemoryEvidence(source, analysis, { sourceSchema, collector } = {}) {
  if (source == null) return null;
  if (!["deepbom.runtime_assignment.v1.9", "deepbom.runtime_assignment.v1.10"].includes(sourceSchema)) {
    throw new Error("Runtime memory evidence requires runtime assignment schema v1.9 or v1.10.");
  }
  if (collector?.schema !== "deepbom.native_runtime_collector.v1.1" || collector?.instrumentation?.arena_allocations !== true) {
    throw new Error("Runtime memory evidence requires declared native arena allocation instrumentation.");
  }
  if (source.schema !== RUNTIME_MEMORY_SCHEMA || source.status !== "assessed" || source.evidence_class !== "OBSERVED_RUNTIME") {
    throw new Error("Runtime memory schema, status, or evidence class is invalid.");
  }
  if (source.tensorflow_source_commit !== TENSORFLOW_SOURCE_COMMIT) throw new Error("Runtime memory evidence is not bound to the pinned TensorFlow source commit.");
  if (!SHA256_PATTERN.test(String(source.allocation_ledger_sha256 || ""))) throw new Error("Runtime memory allocation ledger SHA-256 is invalid.");
  if (!Array.isArray(source.snapshots) || source.snapshots.length < 1 || source.snapshots.length > 4096) throw new Error("Runtime memory snapshots must be a bounded non-empty array.");
  const snapshotCount = integer(source.snapshot_count, "snapshot_count", { positive: true });
  if (snapshotCount !== source.snapshots.length) throw new Error("Runtime memory snapshot_count does not match the snapshot inventory.");
  const artifactTensorCount = Array.isArray(analysis?.tensors) ? analysis.tensors.length : 0;
  const snapshots = source.snapshots.map((snapshot, index) => validateSnapshot(snapshot, index, artifactTensorCount));
  const canonicalLedgerSha256 = sha256TextHex(JSON.stringify(snapshots));
  if (canonicalLedgerSha256 !== source.allocation_ledger_sha256) {
    const sourceJson = JSON.stringify(source.snapshots);
    const canonicalJson = JSON.stringify(snapshots);
    let mismatch = 0;
    while (mismatch < sourceJson.length && mismatch < canonicalJson.length && sourceJson[mismatch] === canonicalJson[mismatch]) mismatch += 1;
    throw new Error(`Runtime memory allocation ledger SHA-256 does not match the canonical snapshot inventory (expected ${canonicalLedgerSha256}, received ${source.allocation_ledger_sha256}; first canonicalization difference at byte ${mismatch}).`);
  }
  const peakNonPersistent = Math.max(...snapshots.map((item) => item.non_persistent_arena_bytes));
  const peakPersistent = Math.max(...snapshots.map((item) => item.persistent_arena_bytes));
  const peakCombined = Math.max(...snapshots.map((item) => item.combined_arena_bytes));
  const finalSnapshot = snapshots.at(-1);
  for (const [field, expected] of [
    ["peak_non_persistent_arena_bytes", peakNonPersistent],
    ["peak_persistent_arena_bytes", peakPersistent],
    ["peak_combined_arena_bytes", peakCombined],
    ["final_non_persistent_arena_bytes", finalSnapshot.non_persistent_arena_bytes],
    ["final_persistent_arena_bytes", finalSnapshot.persistent_arena_bytes],
    ["final_combined_arena_bytes", finalSnapshot.combined_arena_bytes],
  ]) {
    if (integer(source[field], field) !== expected) throw new Error(`Runtime memory ${field} does not conserve the snapshot inventory.`);
  }
  return {
    schema: RUNTIME_MEMORY_SCHEMA,
    status: "assessed",
    evidence_class: "OBSERVED_RUNTIME",
    tensorflow_source_commit: TENSORFLOW_SOURCE_COMMIT,
    snapshot_count: snapshotCount,
    peak_non_persistent_arena_bytes: peakNonPersistent,
    peak_persistent_arena_bytes: peakPersistent,
    peak_combined_arena_bytes: peakCombined,
    final_non_persistent_arena_bytes: finalSnapshot.non_persistent_arena_bytes,
    final_persistent_arena_bytes: finalSnapshot.persistent_arena_bytes,
    final_combined_arena_bytes: finalSnapshot.combined_arena_bytes,
    allocation_ledger_sha256: source.allocation_ledger_sha256,
    snapshots,
    method: String(source.method || ""),
    interpretation_boundary: String(source.interpretation_boundary || ""),
  };
}

function nullableDelta(observed, projected) {
  return observed == null || projected == null ? null : observed - projected;
}

export function deriveArenaRuntimeReconciliation(analysis, runtimeMemory) {
  if (!runtimeMemory) return null;
  const plan = analysis?.tensor_arena_plan || null;
  const snapshot = runtimeMemory.snapshots.at(-1);
  const artifactTensorCount = Array.isArray(analysis?.tensors) ? analysis.tensors.length : 0;
  const projectedAllocations = new Map((plan?.allocations || [])
    .filter((item) => item?.allocation_status === "allocated")
    .map((item) => [Number(item.tensor_index), item]));
  const observedAllocations = new Map(snapshot.allocations.map((item) => [item.tensor_index, item]));
  const projectedAliases = new Map((plan?.aliases || []).map((item) => [Number(item.tensor_index), Number(item.shared_with_tensor_index)]));
  const observedAliases = new Map(snapshot.aliases.map((item) => [item.tensor_index, item.shared_with_tensor_index]));
  const allocationRows = [...new Set([...projectedAllocations.keys(), ...observedAllocations.keys()])]
    .sort((left, right) => left - right)
    .map((tensorIndex) => {
      const projected = projectedAllocations.get(tensorIndex) || null;
      const observed = observedAllocations.get(tensorIndex) || null;
      const artifactTensor = tensorIndex < artifactTensorCount;
      return {
        tensor_index: tensorIndex,
        tensor_name: analysis?.tensors?.[tensorIndex]?.name || (artifactTensor ? `tensor_${tensorIndex}` : `runtime_temporary_${tensorIndex}`),
        artifact_tensor: artifactTensor,
        projected_present: projected != null,
        observed_present: observed != null,
        projected_arena: projected?.arena || null,
        observed_arena: observed?.arena || null,
        projected_size_bytes: projected?.size_bytes ?? null,
        observed_size_bytes: observed?.size_bytes ?? null,
        size_delta_bytes: nullableDelta(observed?.size_bytes, projected?.size_bytes),
        projected_offset_bytes: projected?.offset_bytes ?? null,
        observed_offset_bytes: observed?.offset_bytes ?? null,
        offset_delta_bytes: nullableDelta(observed?.offset_bytes, projected?.offset_bytes),
        size_match: projected != null && observed != null ? Number(projected.size_bytes) === observed.size_bytes : null,
        arena_match: projected != null && observed != null ? projected.arena === observed.arena : null,
        offset_match: projected != null && observed != null ? Number(projected.offset_bytes) === observed.offset_bytes : null,
      };
    });
  const aliasRows = [...new Set([...projectedAliases.keys(), ...observedAliases.keys()])]
    .sort((left, right) => left - right)
    .map((tensorIndex) => ({
      tensor_index: tensorIndex,
      projected_root_tensor_index: projectedAliases.get(tensorIndex) ?? null,
      observed_root_tensor_index: observedAliases.get(tensorIndex) ?? null,
      root_match: projectedAliases.has(tensorIndex) && observedAliases.has(tensorIndex)
        ? projectedAliases.get(tensorIndex) === observedAliases.get(tensorIndex)
        : null,
    }));
  const projectedCombined = plan?.combined_arena_bytes == null ? null : Number(plan.combined_arena_bytes);
  const runtimeOnlyRows = allocationRows.filter((item) => item.observed_present && !item.projected_present);
  const missingObservedRows = allocationRows.filter((item) => item.projected_present && !item.observed_present);
  const sizeMismatchRows = allocationRows.filter((item) => item.size_match === false);
  const offsetMismatchRows = allocationRows.filter((item) => item.offset_match === false);
  const aliasMismatchRows = aliasRows.filter((item) => item.root_match === false || item.projected_root_tensor_index == null || item.observed_root_tensor_index == null);
  const runtimeTemporaryRows = runtimeOnlyRows.filter((item) => !item.artifact_tensor);
  const runtimeTemporaryBytes = runtimeTemporaryRows.reduce((sum, item) => sum + Number(item.observed_size_bytes || 0), 0);
  return {
    schema: RECONCILIATION_SCHEMA,
    status: plan?.status === "assessed" ? "assessed" : "partial_static_projection_unavailable",
    evidence_class: "DERIVED_FROM_OBSERVED_RUNTIME",
    static_projection_schema: plan?.schema || null,
    runtime_memory_schema: runtimeMemory.schema,
    tensorflow_source_commit: runtimeMemory.tensorflow_source_commit,
    runtime_snapshot_id: snapshot.memory_snapshot_id,
    projected_combined_arena_bytes: projectedCombined,
    observed_peak_combined_arena_bytes: runtimeMemory.peak_combined_arena_bytes,
    observed_final_combined_arena_bytes: runtimeMemory.final_combined_arena_bytes,
    peak_delta_bytes: nullableDelta(runtimeMemory.peak_combined_arena_bytes, projectedCombined),
    peak_to_projection_ratio: projectedCombined > 0 ? runtimeMemory.peak_combined_arena_bytes / projectedCombined : null,
    projected_root_allocation_count: projectedAllocations.size,
    observed_root_allocation_count: snapshot.allocations.length,
    matched_allocation_count: allocationRows.filter((item) => item.projected_present && item.observed_present).length,
    runtime_only_allocation_count: runtimeOnlyRows.length,
    missing_observed_allocation_count: missingObservedRows.length,
    size_mismatch_count: sizeMismatchRows.length,
    offset_mismatch_count: offsetMismatchRows.length,
    projected_alias_count: projectedAliases.size,
    observed_alias_count: observedAliases.size,
    alias_mismatch_count: aliasMismatchRows.length,
    runtime_temporary_allocation_count: runtimeTemporaryRows.length,
    runtime_temporary_interval_bytes: runtimeTemporaryBytes,
    allocation_rows: allocationRows,
    alias_rows: aliasRows,
    method: "Exact tensor-index join between the pinned declared-shape ArenaPlanner projection and the final validated post-commit runtime allocation snapshot; observed peak bytes remain a separate snapshot aggregate.",
    interpretation_boundary: "A difference is evidence about this build and invocation, not automatically a defect. Delegation, Prepare-time resize/temporaries, allocation-type changes, and execution-plan rewrites can legitimately alter offsets, lifetimes, and arena high-water. Non-arena delegate and kernel allocations remain excluded.",
  };
}

export const RUNTIME_MEMORY_EVIDENCE_SCHEMA = RUNTIME_MEMORY_SCHEMA;
export const ARENA_RUNTIME_RECONCILIATION_SCHEMA = RECONCILIATION_SCHEMA;
