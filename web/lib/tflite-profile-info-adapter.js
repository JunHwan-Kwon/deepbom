import { ProtoReader, toProtoBytes } from "./tflite-runtime-info-adapter.js";

const PROFILE_ADAPTER_SCHEMA = "deepbom.tflite_benchmark_profile_adapter.v1";
const COMBINED_ADAPTER_SCHEMA = "deepbom.tflite_runtime_info_adapter.v2";
const RUNTIME_ASSIGNMENT_SCHEMA = "deepbom.runtime_assignment.v1.10";
const MAX_PROTO_BYTES = 32 * 1024 * 1024;
const MAX_ROWS = 1_000_000;

export const TFLITE_PROFILE_INFO_SOURCE = Object.freeze({
  source_commit: "tensorflow/tensorflow@87bbf65b8d23d3f06912b1b2183587e1884bc45c",
  proto_file: "tensorflow/lite/profiling/proto/profiling_info.proto",
  proto_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/profiling/proto/profiling_info.proto",
  proto_sha256: "dd286650b64ae913cd9db25baa87a1527835fce4201ad407716b4af4f7dafc69",
  listener_file: "tensorflow/lite/tools/benchmark/profiling_listener.cc",
  listener_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/tools/benchmark/profiling_listener.cc",
  listener_sha256: "7890e11f356c4a0c92ac19f2c5f1c3187df077d49b0d31f52b2d43c2b4cd5a7a",
  summarizer_file: "tensorflow/lite/profiling/profile_summarizer.cc",
  summarizer_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/profiling/profile_summarizer.cc",
  summarizer_sha256: "de7010e97438b7e233e8ec80c8bb19fcbe435475d4f2cca5ccb8bbc03e4d0f91",
  formatter_file: "tensorflow/lite/profiling/profile_summary_formatter.cc",
  formatter_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/profiling/profile_summary_formatter.cc",
  formatter_sha256: "cda02354a7a4c3f1c87263206fd13a64f3c3aae56270e0339a59e4aeae82491f",
  profiler_api_file: "tensorflow/lite/core/api/profiler.h",
  profiler_api_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/core/api/profiler.h",
  profiler_api_sha256: "87f644f353c917464ce6eb82e8c6f893d88d04b7d2311e0f2fbe83a7b2df37db",
  subgraph_file: "tensorflow/lite/core/subgraph.cc",
  subgraph_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/core/subgraph.cc",
  subgraph_sha256: "f1820803ba838bf4261c1cf679ae664a9b29ea3661760d8b3aaae031a135516f",
  xnnpack_delegate_file: "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc",
  xnnpack_delegate_ref: "https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc",
  xnnpack_delegate_sha256: "d9a1b2b5b28dd67c2c81e73671470da0c720b1774d48d41832ed7550083f93ad",
  export_flag: "benchmark_model --enable_op_profiling=true --op_profiling_output_mode=proto --op_profiling_output_file=<path>",
});

export function parseTfliteBenchmarkProfileSource(input) {
  const bytes = toProtoBytes(input, "TFLite BenchmarkProfilingData evidence");
  if (!bytes.length) throw new Error("TFLite BenchmarkProfilingData protobuf is empty.");
  if (bytes.length > MAX_PROTO_BYTES) throw new Error("TFLite BenchmarkProfilingData protobuf must be 32 MiB or smaller.");
  const benchmark = parseBenchmark(new ProtoReader(bytes, "BenchmarkProfilingData"));
  if (!benchmark.runtime_profile) throw new Error("TFLite BenchmarkProfilingData has no runtime_profile.");
  return {
    kind: "tflite_benchmark_profile",
    schema: PROFILE_ADAPTER_SCHEMA,
    source_byte_length: bytes.length,
    model_name: benchmark.model_name,
    init_profile: summarizeModelProfile(benchmark.init_profile),
    runtime_profile: benchmark.runtime_profile,
    source_basis: TFLITE_PROFILE_INFO_SOURCE,
  };
}

export function previewTfliteBenchmarkProfileMapping(profile, runtimeEvidence, analysis) {
  return bindProfile(profile, runtimeEvidence, analysis).summary;
}

export function buildTfliteProfiledAssignmentDocument(profile, runtimeEvidence, analysis, metadata = {}) {
  const binding = bindProfile(profile, runtimeEvidence, analysis);
  const profileSha256 = requiredSha(metadata.profileSha256, "profileSha256");
  const captureId = requiredText(metadata.captureId, "captureId", 160);
  const collectedAt = requiredText(metadata.collectedAt, "collectedAt", 64);
  if (captureId !== runtimeEvidence?.source?.capture_id) {
    throw new Error("TFLite profiling capture ID does not match the imported ModelRuntimeDetails capture ID.");
  }
  const baseAdapter = { ...(runtimeEvidence.source.adapter || {}) };
  delete baseAdapter.timing_profile;
  return {
    schema: RUNTIME_ASSIGNMENT_SCHEMA,
    artifact_sha256: runtimeEvidence.artifact_sha256,
    target_profile_id: runtimeEvidence.target_profile_id,
    target_profile_sha256: runtimeEvidence.target_profile_sha256,
    runtime: { ...runtimeEvidence.runtime },
    source: {
      ...runtimeEvidence.source,
      kind: "tflite_model_runtime_info_and_benchmark_profile_proto_adapter",
      duration_semantics: "per_execution_plan_node_exclusive",
      duration_statistic: "execution_node_mean_per_run_from_one_event_per_primary_run; ancillary_sum_divided_by_common_primary_run_count",
      adapter: {
        ...baseAdapter,
        schema: COMBINED_ADAPTER_SCHEMA,
        timing_profile: {
          schema: PROFILE_ADAPTER_SCHEMA,
          ...TFLITE_PROFILE_INFO_SOURCE,
          profile_sha256: profileSha256,
          source_byte_length: profile.source_byte_length,
          source_model_name: profile.model_name,
          collected_at: collectedAt,
          capture_id: captureId,
          capture_binding_semantics: "DECLARED_SAME_BENCHMARK_INVOCATION",
          ...binding.timing,
          interpretation_boundary: "Primary-subgraph execution-node rows are mapped only by the formatter-emitted node-ID suffix and the imported ModelRuntimeDetails execution plan. Primary execution nodes derive run count only when the formatter reports one call per run. The formatter stores times_called using integer division, so delegate-internal and other events never invert that field; their per-run values use a common primary execution-node run count or remain withheld. Delegate-profiled rows prefixed Delegate/ use delegate-internal IDs and are never assigned to original ops or partitions. A graph total is emitted only when every execution-plan node has an exclusive row with one common derived run count.",
        },
        interpretation_boundary: "ModelRuntimeDetails observes placement and partition replacement maps. BenchmarkProfilingData observes timing statistics but embeds neither artifact SHA-256 nor capture identity; the shared capture ID, runtime identity, build, and collection context are declared. Executed microkernel symbols and tensor-copy materialization remain unobserved.",
      },
    },
    assignments: binding.assignments,
  };
}

function bindProfile(profile, runtimeEvidence, analysis) {
  if (profile?.kind !== "tflite_benchmark_profile") throw new Error("Expected parsed TFLite BenchmarkProfilingData evidence.");
  if (String(analysis?.format || "").toLowerCase() !== "tflite") throw new Error("TFLite profiling evidence requires an active TFLite artifact.");
  if (!runtimeEvidence || !["deepbom.tflite_runtime_info_adapter.v1", COMBINED_ADAPTER_SCHEMA].includes(runtimeEvidence.source?.adapter?.schema)) {
    throw new Error("Import matching TFLite ModelRuntimeDetails evidence before importing profiling timing.");
  }
  if (runtimeEvidence.artifact_sha256 !== String(analysis?.model_sha256 || "").toLowerCase()) throw new Error("TFLite runtime evidence no longer matches the active artifact.");
  if (!runtimeEvidence.source?.capture_id) throw new Error("The imported TFLite runtime plan has no capture ID; re-import it before attaching timing evidence.");
  if (profile.model_name && runtimeEvidence.source.adapter.source_model_name && profile.model_name !== runtimeEvidence.source.adapter.source_model_name) {
    throw new Error("TFLite profiling model_name does not match the imported runtime plan.");
  }
  const primary = profile.runtime_profile.subgraph_profiles.find((item) => item.subgraph_index === 0);
  if (!primary) throw new Error("TFLite runtime_profile has no primary subgraph index 0.");
  const execution = new Map();
  const originalByNode = new Map();
  for (const row of runtimeEvidence.assignments || []) {
    if (row.delegated === false) {
      execution.set(row.runtime_node_index, { kind: "original_op", assignment: row });
      originalByNode.set(row.runtime_node_index, row);
    } else if (row.delegated === true) {
      originalByNode.set(row.op_index, row);
    }
  }
  for (const partition of runtimeEvidence.source.adapter.partitions || []) {
    execution.set(partition.delegate_node_id, { kind: "delegate_partition", partition });
  }
  if (execution.size !== Number(runtimeEvidence.source.adapter.execution_plan_node_count)) {
    throw new Error("TFLite runtime-plan execution-node inventory is inconsistent.");
  }
  const mapped = [];
  const internal = [];
  const other = [];
  const seenNodeIds = new Set();
  for (const row of primary.per_op_profiles) {
    if (row.name.startsWith("Delegate/")) {
      internal.push(timingRow(row, "primary_subgraph_delegate_profiled_event"));
      continue;
    }
    const nodeId = nodeIdSuffix(row.name);
    const owner = nodeId == null ? null : execution.get(nodeId);
    if (!owner) {
      if (nodeId != null && originalByNode.get(nodeId)?.delegated === true) {
        throw new Error(`TFLite profile contains delegated original node ${nodeId} outside the observed execution plan.`);
      }
      other.push(timingRow(row, "unmapped_primary_subgraph_event"));
      continue;
    }
    if (seenNodeIds.has(nodeId)) throw new Error(`TFLite profile contains duplicate execution-node timing for node ${nodeId}.`);
    validateNodeType(row.node_type, owner, nodeId);
    seenNodeIds.add(nodeId);
    mapped.push({
      ...timingRow(row, owner.kind),
      runtime_node_index: nodeId,
      op_index: owner.kind === "original_op" ? owner.assignment.op_index : null,
      partition_id: owner.kind === "delegate_partition" ? owner.partition.partition_id : null,
      provider: owner.kind === "delegate_partition" ? owner.partition.delegate_name : owner.assignment.provider,
    });
  }
  for (const delegate of profile.runtime_profile.delegate_profiles) {
    for (const row of delegate.per_op_profiles) {
      internal.push({ ...timingRow(row, "delegate_internal_section_event"), delegate_name: delegate.delegate_name });
    }
  }
  const commonRunCount = commonTimingRunCount(mapped);
  const normalizedInternal = internal.map((row) => attachCommonRunCount(row, commonRunCount));
  const normalizedOther = other.map((row) => attachCommonRunCount(row, commonRunCount));
  const timingByNode = new Map(mapped.map((row) => [row.runtime_node_index, row]));
  const assignments = runtimeEvidence.assignments.map((row) => {
    const timing = row.delegated === false ? timingByNode.get(row.runtime_node_index) : null;
    return {
      ...row,
      duration_us: timing?.run_count == null ? null : timing.mean_per_run_us,
      duration_sum_us: timing?.run_count == null ? null : timing.sum_us,
      sample_count: timing?.run_count ?? null,
    };
  });
  const complete = mapped.length === execution.size && commonRunCount != null;
  const originalRows = mapped.filter((row) => row.node_kind === "original_op");
  const partitionRows = mapped.filter((row) => row.node_kind === "delegate_partition");
  const primaryDelegateRows = normalizedInternal.filter((row) => row.node_kind === "primary_subgraph_delegate_profiled_event");
  const nestedDelegateRows = normalizedInternal.filter((row) => row.node_kind === "delegate_internal_section_event");
  const timing = {
    primary_subgraph_index: 0,
    runtime_subgraph_count: profile.runtime_profile.subgraph_profiles.length,
    unassessed_nonprimary_subgraph_count: profile.runtime_profile.subgraph_profiles.filter((item) => item.subgraph_index !== 0).length,
    execution_plan_node_count: execution.size,
    mapped_execution_node_count: mapped.length,
    execution_plan_coverage_ratio: mapped.length / Math.max(1, execution.size),
    original_execution_node_timing_count: originalRows.length,
    delegate_partition_timing_count: partitionRows.length,
    delegate_internal_event_count: normalizedInternal.length,
    other_primary_event_count: other.length,
    common_run_count: commonRunCount,
    execution_node_total_us: complete ? sum(mapped.map((row) => row.mean_per_run_us)) : null,
    mapped_execution_node_subtotal_us: commonTimingRunCount(mapped) == null ? null : sum(mapped.map((row) => row.mean_per_run_us)),
    cpu_execution_node_subtotal_us: commonTimingRunCount(originalRows) == null ? null : sum(originalRows.map((row) => row.mean_per_run_us)),
    delegate_partition_subtotal_us: commonTimingRunCount(partitionRows) == null ? null : sum(partitionRows.map((row) => row.mean_per_run_us)),
    primary_delegate_profiled_subtotal_us: commonTimingRunCount(primaryDelegateRows) == null ? null : sum(primaryDelegateRows.map((row) => row.mean_per_run_us)),
    delegate_internal_section_subtotal_us: commonTimingRunCount(nestedDelegateRows) == null ? null : sum(nestedDelegateRows.map((row) => row.mean_per_run_us)),
    delegate_internal_profiled_subtotal_us: commonTimingRunCount(normalizedInternal) == null ? null : sum(normalizedInternal.map((row) => row.mean_per_run_us)),
    total_status: complete ? "assessed_complete_execution_plan" : mapped.length ? "partial_execution_plan_coverage" : "not_assessed_no_execution_node_rows",
    execution_nodes: mapped.sort((a, b) => a.run_order - b.run_order),
    delegate_internal_events: normalizedInternal.sort((a, b) => a.run_order - b.run_order),
    other_primary_events: normalizedOther.sort((a, b) => a.run_order - b.run_order),
  };
  return {
    assignments,
    timing,
    summary: {
      execution_plan_node_count: execution.size,
      mapped_execution_node_count: mapped.length,
      execution_plan_coverage_ratio: timing.execution_plan_coverage_ratio,
      original_execution_node_timing_count: originalRows.length,
      delegate_partition_timing_count: partitionRows.length,
      delegate_internal_event_count: normalizedInternal.length,
      other_primary_event_count: other.length,
      graph_total_available: complete,
      graph_total_us: timing.execution_node_total_us,
      primary_delegate_profiled_subtotal_us: timing.primary_delegate_profiled_subtotal_us,
      delegate_internal_section_subtotal_us: timing.delegate_internal_section_subtotal_us,
      delegate_internal_profiled_subtotal_us: timing.delegate_internal_profiled_subtotal_us,
    },
  };
}

function timingRow(row, nodeKind) {
  const stat = row.inference_microseconds;
  const executionNode = nodeKind === "original_op" || nodeKind === "delegate_partition";
  const runCount = executionNode && row.times_called === 1 ? stat.count : null;
  return {
    node_kind: nodeKind,
    node_type: row.node_type,
    name: row.name,
    run_order: row.run_order,
    formatter_times_called_integer_average: row.times_called,
    event_sample_count: stat.count,
    run_count: runCount,
    run_count_derivation_status: runCount == null
      ? executionNode ? "not_derivable_execution_node_calls_per_run_not_one" : "pending_common_primary_execution_run_count"
      : "derived_primary_execution_node_one_event_per_run",
    first_us: stat.first,
    last_us: stat.last,
    min_us: stat.min,
    max_us: stat.max,
    sum_us: stat.sum,
    mean_per_event_us: stat.sum / stat.count,
    mean_per_run_us: runCount == null ? null : stat.sum / runCount,
    stddev_us: stat.stddev,
    variance_us2: stat.variance,
  };
}

function attachCommonRunCount(row, runCount) {
  if (runCount == null) {
    return {
      ...row,
      run_count: null,
      run_count_derivation_status: "not_derivable_no_common_primary_execution_run_count",
      mean_per_run_us: null,
    };
  }
  if (Math.floor(row.event_sample_count / runCount) !== row.formatter_times_called_integer_average) {
    throw new Error(`TFLite profile event ${row.name} has times_called inconsistent with the common primary execution run count.`);
  }
  return {
    ...row,
    run_count: runCount,
    run_count_derivation_status: "derived_from_common_primary_execution_run_count",
    mean_per_run_us: row.sum_us / runCount,
  };
}

function validateNodeType(nodeType, owner, nodeId) {
  const expected = owner.kind === "original_op" ? owner.assignment.op_name : owner.partition.runtime_node_name;
  if (nodeType !== expected && !nodeType.startsWith(`${expected}/`)) {
    throw new Error(`TFLite profile node type ${nodeType} does not match runtime node ${nodeId} (${expected}).`);
  }
}

function nodeIdSuffix(name) {
  const match = /:([0-9]+)$/.exec(name);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isSafeInteger(value) ? value : null;
}

function commonTimingRunCount(rows) {
  if (!rows.length || rows.some((row) => row.run_count == null)) return null;
  const counts = new Set(rows.map((row) => row.run_count));
  return counts.size === 1 ? rows[0].run_count : null;
}

function summarizeModelProfile(profile) {
  if (!profile) return null;
  return {
    subgraph_count: profile.subgraph_profiles.length,
    delegate_profile_count: profile.delegate_profiles.length,
    op_profile_count: profile.subgraph_profiles.reduce((sumValue, item) => sumValue + item.per_op_profiles.length, 0),
    delegate_event_count: profile.delegate_profiles.reduce((sumValue, item) => sumValue + item.per_op_profiles.length, 0),
  };
}

function parseBenchmark(reader) {
  const result = { model_name: null, init_profile: null, runtime_profile: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.model_name = reader.stringField(wire, singular, field, "model_name");
    else if (field === 2 || field === 3) {
      if (singular.has(field)) throw new Error(`BenchmarkProfilingData contains duplicate singular field ${field === 2 ? "init_profile" : "runtime_profile"}.`);
      singular.add(field);
      result[field === 2 ? "init_profile" : "runtime_profile"] = parseModelProfile(reader.message(wire, field === 2 ? "init_profile" : "runtime_profile"));
    } else reader.skip(wire);
  }
  return result;
}

function parseModelProfile(reader) {
  const result = { subgraph_profiles: [], delegate_profiles: [] };
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.subgraph_profiles.push(parseSubgraphProfile(reader.message(wire, "SubGraphProfilingData")));
    else if (field === 2) result.delegate_profiles.push(parseDelegateProfile(reader.message(wire, "DelegateProfilingData")));
    else reader.skip(wire);
    guardRows(result.subgraph_profiles, "subgraph profiles");
    guardRows(result.delegate_profiles, "delegate profiles");
  }
  const ids = result.subgraph_profiles.map((item) => item.subgraph_index);
  if (new Set(ids).size !== ids.length) throw new Error("TFLite profiling data contains duplicate subgraph indices.");
  return result;
}

function parseSubgraphProfile(reader) {
  const result = { subgraph_name: null, subgraph_index: null, per_op_profiles: [] };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.subgraph_name = reader.stringField(wire, singular, field, "subgraph_name");
    else if (field === 2) result.subgraph_index = reader.intField(wire, singular, field, "subgraph_index");
    else if (field === 3) result.per_op_profiles.push(parseOpProfile(reader.message(wire, "OpProfileData")));
    else reader.skip(wire);
    guardRows(result.per_op_profiles, "per-op profiles");
  }
  if (result.subgraph_index == null) throw new Error("TFLite SubGraphProfilingData is missing subgraph_index.");
  validateProfileRows(result.per_op_profiles, `subgraph ${result.subgraph_index}`);
  return result;
}

function parseDelegateProfile(reader) {
  const result = { delegate_name: null, per_op_profiles: [] };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1) result.delegate_name = reader.stringField(wire, singular, field, "delegate_name");
    else if (field === 2) result.per_op_profiles.push(parseOpProfile(reader.message(wire, "OpProfileData")));
    else reader.skip(wire);
    guardRows(result.per_op_profiles, "delegate per-op profiles");
  }
  validateProfileRows(result.per_op_profiles, "delegate profile");
  return result;
}

function parseOpProfile(reader) {
  const result = { node_type: null, inference_microseconds: null, mem_kb: null, times_called: null, name: null, run_order: null };
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if (field === 1 || field === 5) result[field === 1 ? "node_type" : "name"] = reader.stringField(wire, singular, field, field === 1 ? "node_type" : "name");
    else if (field === 2 || field === 3) {
      if (singular.has(field)) throw new Error(`OpProfileData contains duplicate singular field ${field === 2 ? "inference_microseconds" : "mem_kb"}.`);
      singular.add(field);
      result[field === 2 ? "inference_microseconds" : "mem_kb"] = parseStat(reader.message(wire, field === 2 ? "inference_microseconds" : "mem_kb"));
    } else if (field === 4 || field === 6) result[field === 4 ? "times_called" : "run_order"] = reader.int64Field(wire, singular, field, field === 4 ? "times_called" : "run_order");
    else reader.skip(wire);
  }
  if (!result.node_type || !result.name || !result.inference_microseconds || result.times_called == null || result.run_order == null) {
    throw new Error("TFLite OpProfileData is missing node_type, name, timing, times_called, or run_order.");
  }
  if (!Number.isSafeInteger(result.times_called) || result.times_called < 0 || !Number.isSafeInteger(result.run_order) || result.run_order < 0) {
    throw new Error("TFLite OpProfileData times_called and run_order must be non-negative integers.");
  }
  return result;
}

function parseStat(reader) {
  const names = [null, "first", "last", "avg", "stddev", "variance", "min", "max", "sum", "count"];
  const result = {};
  const singular = new Set();
  while (!reader.done) {
    const { field, wire } = reader.key();
    if ([1, 2, 3, 6, 7, 8, 9].includes(field)) result[names[field]] = reader.int64Field(wire, singular, field, names[field]);
    else if (field === 4 || field === 5) result[names[field]] = reader.floatField(wire, singular, field, names[field]);
    else reader.skip(wire);
  }
  if (names.slice(1).some((name) => result[name] == null)) throw new Error("TFLite OpProfilingStat is incomplete.");
  for (const name of ["first", "last", "avg", "stddev", "variance", "min", "max", "sum"]) {
    if (result[name] < 0) throw new Error(`TFLite OpProfilingStat ${name} must be non-negative.`);
  }
  if (!Number.isSafeInteger(result.count) || result.count <= 0 || result.min > result.max
    || result.first < result.min || result.first > result.max || result.last < result.min || result.last > result.max
    || result.sum < result.min * result.count || result.sum > result.max * result.count
    || result.avg !== Math.trunc(result.sum / result.count)) {
    throw new Error("TFLite OpProfilingStat fields are arithmetically inconsistent.");
  }
  return result;
}

function validateProfileRows(rows, label) {
  const names = new Set();
  const orders = new Set();
  for (const row of rows) {
    if (names.has(row.name)) throw new Error(`TFLite ${label} contains duplicate profile name ${row.name}.`);
    if (orders.has(row.run_order)) throw new Error(`TFLite ${label} contains duplicate run_order ${row.run_order}.`);
    names.add(row.name);
    orders.add(row.run_order);
  }
}

function guardRows(rows, label) {
  if (rows.length > MAX_ROWS) throw new Error(`TFLite profiling ${label} exceeds ${MAX_ROWS.toLocaleString("en-US")} rows.`);
}

function sum(values) { return values.reduce((total, value) => total + Number(value || 0), 0); }

function requiredText(value, field, maxLength) {
  const text = String(value || "").trim();
  if (!text) throw new Error(`TFLite profiling metadata ${field} is required.`);
  if (text.length > maxLength) throw new Error(`TFLite profiling metadata ${field} exceeds ${maxLength} characters.`);
  return text;
}

function requiredSha(value, field) {
  const text = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(text)) throw new Error(`TFLite profiling metadata ${field} must be SHA-256.`);
  return text;
}
