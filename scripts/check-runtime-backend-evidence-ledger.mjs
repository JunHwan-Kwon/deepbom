import assert from "node:assert/strict";

import {
  buildRuntimeBackendEvidenceLedger,
  validateRuntimeBackendEvidenceLedger,
} from "../web/lib/runtime-backend-evidence-ledger.js";

const selected = {
  provider_inventory_status: "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS",
  supported_backends_sha256: "a".repeat(64),
  bindings: [
    { backend_name: "cuda", bundled: true, source_profile: "cuda", binding_status: "BUNDLED_BACKEND_WITH_PINNED_SOURCE_PROFILE" },
    { backend_name: "tensorrt", bundled: false, source_profile: null, binding_status: "DISCOVERED_BACKEND_WITHOUT_STATIC_SOURCE_PROFILE" },
    { backend_name: "qnn", bundled: true, source_profile: "qnn", binding_status: "BUNDLED_BACKEND_WITH_PINNED_SOURCE_PROFILE" },
    { backend_name: "webgpu", bundled: false, source_profile: "webgpu", binding_status: "DISCOVERED_BACKEND_WITH_PINNED_SOURCE_PROFILE_NOT_BUNDLED" },
  ],
  source_profiles_not_listed_by_selected_build: ["nnapi", "coreml", "webnn", "directml", "xnnpack", "wasm_cpu"],
};
const runtime = {
  artifact_sha256: "b".repeat(64),
  runtime: { binary_sha256: "c".repeat(64) },
  source: { kind: "onnxruntime_profile_json_adapter", adapter: { schema: "deepbom.ort_profile_adapter.v2.1", native_capture: { selected_build_provider_binding: selected } } },
  assignments: [
    { op_index: 0, provider: "QNNExecutionProvider", sample_count: 2, duration_sum_us: 10 },
    { op_index: 1, provider: "QNNExecutionProvider", sample_count: 2, duration_sum_us: 14 },
    { op_index: 2, provider: "CPUExecutionProvider", sample_count: 2, duration_sum_us: 4 },
    { op_index: 3, provider: "CUDAExecutionProvider", sample_count: 3, duration_sum_us: 6 },
  ],
};
const ledger = buildRuntimeBackendEvidenceLedger(runtime);
assert.equal(ledger.provider_count, 8);
assert.equal(ledger.configured_inclusion_assessed_count, 8);
assert.equal(ledger.capability_acceptance_observed_count, 2);
assert.equal(ledger.assigned_provider_count, 2);
assert.equal(ledger.executed_provider_count, 2);
const qnn = ledger.providers.find((row) => row.provider_id === "qnn");
assert.equal(qnn.configured_inclusion.status, "observed_bundled");
assert.equal(qnn.capability_acceptance.status, "accepted_by_observed_assignment");
assert.equal(qnn.assignment.assigned_original_op_count, 2);
assert.equal(qnn.execution.event_sample_count, 4);
assert.equal(qnn.execution.duration_sum_us, 24);
const cuda = ledger.providers.find((row) => row.provider_id === "cuda");
assert.equal(cuda.configured_inclusion.status, "observed_bundled");
assert.equal(cuda.assignment.assigned_original_op_count, 1);
assert.equal(cuda.execution.event_sample_count, 3);
const tensorRt = ledger.providers.find((row) => row.provider_id === "tensorrt");
assert.equal(tensorRt.source_eligibility.status, "separate_static_preflight_required");
assert.equal(tensorRt.source_eligibility.evidence_class, "NOT_ASSESSED_IN_ORT_SOURCE_RULEPACK");
assert.equal(tensorRt.source_eligibility.separate_static_contract, "deepbom.tensorrt_static_preflight.v1");
assert.equal(tensorRt.configured_inclusion.status, "observed_available_not_bundled");
const directMl = ledger.providers.find((row) => row.provider_id === "directml");
assert.equal(directMl.source_eligibility.status, "source_profile_present");
assert.equal(directMl.configured_inclusion.status, "observed_not_listed");
const nnapi = ledger.providers.find((row) => row.provider_id === "nnapi");
assert.equal(nnapi.configured_inclusion.status, "observed_not_listed");
assert.equal(nnapi.capability_acceptance.status, "not_assessed");
assert.equal(nnapi.assignment.status, "not_observed");
assert.equal(nnapi.execution.status, "not_observed");
validateRuntimeBackendEvidenceLedger(ledger, runtime);
assert.throws(() => validateRuntimeBackendEvidenceLedger({ ...ledger, assigned_provider_count: 3 }, runtime), /does not reconstruct/);

const fusedRuntime = {
  ...runtime,
  source: {
    kind: "onnxruntime_profile_json_adapter",
    adapter: {
      schema: "deepbom.ort_profile_adapter.v2.2",
      native_capture: {
        profile_role: "identity",
        selected_build_provider_binding: {
          ...selected,
          bindings: [{ backend_name: "dml", bundled: true, source_profile: "directml", binding_status: "BUNDLED_BACKEND_WITH_PINNED_SOURCE_PROFILE" }],
        },
        paired_profile_runtime_graph: {
          profiles: [{ role: "identity", nodes: [{ provider: "DmlExecutionProvider", sample_count: 1, duration_sum_us: 1099 }] }],
        },
      },
    },
  },
  assignments: [],
};
const fusedLedger = buildRuntimeBackendEvidenceLedger(fusedRuntime);
const fusedDirectMl = fusedLedger.providers.find((row) => row.provider_id === "directml");
assert.equal(fusedDirectMl.capability_acceptance.status, "accepted_by_observed_runtime_graph");
assert.equal(fusedDirectMl.capability_acceptance.accepted_original_op_count, 0);
assert.equal(fusedDirectMl.capability_acceptance.accepted_runtime_node_count, 1);
assert.equal(fusedDirectMl.assignment.status, "not_observed");
assert.equal(fusedDirectMl.execution.executed_original_op_count, 0);
assert.equal(fusedDirectMl.execution.executed_runtime_node_count, 1);
assert.equal(fusedDirectMl.execution.duration_sum_us, 1099);
assert.equal(fusedLedger.executed_provider_count, 1);
assert.equal(buildRuntimeBackendEvidenceLedger({ schema: "deepbom.coreml_compute_plan.v1" }), null);
console.log("Runtime backend evidence ledger passed (eight accelerator providers, five independent evidence layers, fused-node preservation, digest, and tamper rejection). ");
