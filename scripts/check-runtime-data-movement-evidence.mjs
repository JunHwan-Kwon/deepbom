import assert from "node:assert/strict";

import {
  buildRuntimeDataMovementEvidence,
  validateRuntimeDataMovementEvidence,
} from "../web/lib/runtime-data-movement-evidence.js";

const profile = (nodes, invocationRunCount = 2) => ({
  role: "production",
  profile_sha256: "3".repeat(64),
  invocation_run_count: invocationRunCount,
  nodes,
});
const runtime = (nodes) => ({
  artifact_sha256: "1".repeat(64),
  runtime: { binary_sha256: "2".repeat(64) },
  source: {
    adapter: {
      schema: "deepbom.ort_profile_adapter.v2.2",
      native_capture: {
        capture_id: "capture-copy-1",
        paired_profile_runtime_graph: {
          schema: "deepbom.ort_paired_runtime_graph.v1.1",
          profiles: [profile(nodes)],
        },
      },
    },
  },
});

const evidence = buildRuntimeDataMovementEvidence(runtime([
  { runtime_node_index: 7, runtime_node_name: "copy_in", op_name: "MemcpyFromHost", provider: "CUDAExecutionProvider", sample_count: 2, output_size_bytes_decimal: "4096" },
  { runtime_node_index: 8, runtime_node_name: "copy_out", op_name: "MemcpyToHost", provider: "CUDAExecutionProvider", sample_count: 2, output_size_bytes_decimal: "1024" },
]));
assert.equal(evidence.status, "observed_copy_node_payload");
assert.equal(evidence.observed_copy_node_count, 2);
assert.equal(evidence.observed_copy_event_count, 4);
assert.equal(evidence.observed_copy_event_payload_bytes.decimal, "10240");
assert.equal(evidence.observed_copy_payload_per_invocation_bytes.decimal, "5120");
assert.equal(evidence.physical_transfer_bytes, null);
assert.equal(evidence.physical_transfer_status, "not_exposed_by_ort_profile");
assert.equal(validateRuntimeDataMovementEvidence(evidence, runtime([
  { runtime_node_index: 7, runtime_node_name: "copy_in", op_name: "MemcpyFromHost", provider: "CUDAExecutionProvider", sample_count: 2, output_size_bytes_decimal: "4096" },
  { runtime_node_index: 8, runtime_node_name: "copy_out", op_name: "MemcpyToHost", provider: "CUDAExecutionProvider", sample_count: 2, output_size_bytes_decimal: "1024" },
])), true);

const missingPayload = buildRuntimeDataMovementEvidence(runtime([
  { runtime_node_index: 7, runtime_node_name: "copy_in", op_name: "MemcpyFromHost", provider: "CUDAExecutionProvider", sample_count: 2, output_size_bytes_decimal: null },
]));
assert.equal(missingPayload.status, "partial_copy_node_payload");
assert.equal(missingPayload.observed_copy_event_payload_bytes, null);

const noCopy = buildRuntimeDataMovementEvidence(runtime([
  { runtime_node_index: 0, runtime_node_name: "matmul", op_name: "MatMul", provider: "CPUExecutionProvider", sample_count: 2, output_size_bytes_decimal: "128" },
]));
assert.equal(noCopy.status, "observed_no_profiled_copy_nodes_for_captured_configuration");
assert.equal(noCopy.observed_copy_event_payload_bytes.decimal, "0");
assert.equal(noCopy.physical_transfer_bytes, null, "No copy-node event must not be promoted to zero physical transfer bytes.");

const legacy = structuredClone(runtime([]));
legacy.source.adapter.native_capture.paired_profile_runtime_graph.schema = "deepbom.ort_paired_runtime_graph.v1";
assert.equal(buildRuntimeDataMovementEvidence(legacy).status, "not_exposed_by_capture_schema");

console.log("Runtime data-movement evidence passed (copy-node payload conservation and physical-transfer fail-closed boundary)." );
