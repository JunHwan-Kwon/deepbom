import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const RUNTIME_DATA_MOVEMENT_EVIDENCE_SCHEMA = "deepbom.runtime_data_movement_evidence.v1";

const COPY_DIRECTIONS = Object.freeze(new Map([
  ["memcpyfromhost", "host_to_accelerator"],
  ["memcpytohost", "accelerator_to_host"],
  ["memcpy", "runtime_defined"],
]));

export function buildRuntimeDataMovementEvidence(runtimeEvidence) {
  const adapter = runtimeEvidence?.source?.adapter || null;
  const capture = adapter?.native_capture || null;
  const graph = capture?.paired_profile_runtime_graph || null;
  if (!capture || !String(adapter?.schema || "").startsWith("deepbom.ort_profile_adapter.v2.")) return null;
  const production = (graph?.profiles || []).find((profile) => profile.role === "production") || null;
  const body = {
    schema: RUNTIME_DATA_MOVEMENT_EVIDENCE_SCHEMA,
    evidence_class: "OBSERVED_RUNTIME_PROFILE",
    artifact_sha256: validSha(runtimeEvidence?.artifact_sha256) ? runtimeEvidence.artifact_sha256 : null,
    runtime_binary_sha256: validSha(runtimeEvidence?.runtime?.binary_sha256) ? runtimeEvidence.runtime.binary_sha256 : null,
    capture_id: capture.capture_id || null,
    profile_schema: graph?.schema || null,
    profile_sha256: production?.profile_sha256 || null,
    status: "not_exposed_by_capture_schema",
    invocation_run_count: null,
    observed_copy_node_count: 0,
    observed_copy_event_count: 0,
    observed_copy_event_payload_bytes: null,
    observed_copy_payload_per_invocation_bytes: null,
    copy_nodes: [],
    physical_transfer_bytes: null,
    physical_transfer_status: "not_exposed_by_ort_profile",
    interpretation_boundary: "Source-backed backend eligibility and logical graph-boundary payload are separate from this runtime ledger. ORT copy-node output_size is an observed logical payload for an executed copy node; it is not physical bus traffic, zero-copy behavior, synchronization cost, memory residency, or latency.",
  };
  if (graph?.schema !== "deepbom.ort_paired_runtime_graph.v1.1" || !production) {
    return seal(body);
  }
  const rows = production.nodes.filter((node) => COPY_DIRECTIONS.has(normalizeOp(node.op_name))).map((node) => ({
    runtime_node_index: node.runtime_node_index,
    runtime_node_name: node.runtime_node_name,
    op_name: node.op_name,
    provider: node.provider,
    direction: COPY_DIRECTIONS.get(normalizeOp(node.op_name)),
    sample_count: node.sample_count,
    output_payload_bytes: exactOrNull(node.output_size_bytes_decimal),
  }));
  const completePayload = rows.every((row) => row.output_payload_bytes != null);
  const eventTotal = completePayload
    ? rows.reduce((sum, row) => sum + BigInt(row.output_payload_bytes.decimal) * BigInt(row.sample_count), 0n)
    : null;
  const runCount = production.invocation_run_count;
  const perInvocationComplete = completePayload && rows.every((row) => row.sample_count === runCount);
  const perInvocation = perInvocationComplete
    ? rows.reduce((sum, row) => sum + BigInt(row.output_payload_bytes.decimal), 0n)
    : null;
  return seal({
    ...body,
    status: rows.length ? completePayload ? "observed_copy_node_payload" : "partial_copy_node_payload"
      : "observed_no_profiled_copy_nodes_for_captured_configuration",
    invocation_run_count: runCount,
    observed_copy_node_count: rows.length,
    observed_copy_event_count: rows.reduce((sum, row) => sum + row.sample_count, 0),
    observed_copy_event_payload_bytes: eventTotal == null ? null : exact(eventTotal),
    observed_copy_payload_per_invocation_bytes: perInvocation == null ? null : exact(perInvocation),
    copy_nodes: rows,
  });
}

export function validateRuntimeDataMovementEvidence(value, runtimeEvidence) {
  const expected = buildRuntimeDataMovementEvidence(runtimeEvidence);
  if (canonicalJson(value) !== canonicalJson(expected)) throw new Error("Runtime data-movement evidence does not reconstruct from the imported ORT capture.");
  return true;
}

function normalizeOp(value) { return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }
function validSha(value) { return /^[a-f0-9]{64}$/.test(String(value || "")); }
function exactOrNull(value) { return /^\d+$/.test(String(value ?? "")) ? exact(BigInt(value)) : null; }
function exact(value) { return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) }; }
function seal(body) { return Object.freeze({ ...body, ledger_sha256: sha256TextHex(canonicalJson(body)) }); }
