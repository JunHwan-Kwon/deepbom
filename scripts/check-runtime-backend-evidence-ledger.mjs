import assert from "node:assert/strict";

import {
  buildRuntimeBackendEvidenceLedger,
  validateRuntimeBackendEvidenceLedger,
} from "../web/lib/runtime-backend-evidence-ledger.js";

const selected = {
  provider_inventory_status: "OBSERVED_FROM_ORT_LIST_SUPPORTED_BACKENDS",
  supported_backends_sha256: "a".repeat(64),
  bindings: [
    { backend_name: "qnn", bundled: true, source_profile: "qnn", binding_status: "BUNDLED_BACKEND_WITH_PINNED_SOURCE_PROFILE" },
    { backend_name: "webgpu", bundled: false, source_profile: "webgpu", binding_status: "DISCOVERED_BACKEND_WITH_PINNED_SOURCE_PROFILE_NOT_BUNDLED" },
  ],
  source_profiles_not_listed_by_selected_build: ["nnapi", "coreml", "webnn"],
};
const runtime = {
  artifact_sha256: "b".repeat(64),
  runtime: { binary_sha256: "c".repeat(64) },
  source: { kind: "onnxruntime_profile_json_adapter", adapter: { schema: "deepbom.ort_profile_adapter.v2.1", native_capture: { selected_build_provider_binding: selected } } },
  assignments: [
    { op_index: 0, provider: "QNNExecutionProvider", sample_count: 2, duration_sum_us: 10 },
    { op_index: 1, provider: "QNNExecutionProvider", sample_count: 2, duration_sum_us: 14 },
    { op_index: 2, provider: "CPUExecutionProvider", sample_count: 2, duration_sum_us: 4 },
  ],
};
const ledger = buildRuntimeBackendEvidenceLedger(runtime);
assert.equal(ledger.provider_count, 5);
assert.equal(ledger.configured_inclusion_assessed_count, 5);
assert.equal(ledger.capability_acceptance_observed_count, 1);
assert.equal(ledger.assigned_provider_count, 1);
assert.equal(ledger.executed_provider_count, 1);
const qnn = ledger.providers.find((row) => row.provider_id === "qnn");
assert.equal(qnn.configured_inclusion.status, "observed_bundled");
assert.equal(qnn.capability_acceptance.status, "accepted_by_observed_assignment");
assert.equal(qnn.assignment.assigned_original_op_count, 2);
assert.equal(qnn.execution.event_sample_count, 4);
assert.equal(qnn.execution.duration_sum_us, 24);
const nnapi = ledger.providers.find((row) => row.provider_id === "nnapi");
assert.equal(nnapi.configured_inclusion.status, "observed_not_listed");
assert.equal(nnapi.capability_acceptance.status, "not_assessed");
assert.equal(nnapi.assignment.status, "not_observed");
assert.equal(nnapi.execution.status, "not_observed");
validateRuntimeBackendEvidenceLedger(ledger, runtime);
assert.throws(() => validateRuntimeBackendEvidenceLedger({ ...ledger, assigned_provider_count: 2 }, runtime), /does not reconstruct/);
assert.equal(buildRuntimeBackendEvidenceLedger({ schema: "deepbom.coreml_compute_plan.v1" }), null);
console.log("Runtime backend evidence ledger passed (five providers, four independent evidence layers, digest, and tamper rejection). ");
