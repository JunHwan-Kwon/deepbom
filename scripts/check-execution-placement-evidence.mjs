import assert from "node:assert/strict";
import {
  EXECUTION_PLACEMENT_EVIDENCE_SCHEMA,
  buildExecutionPlacementEvidence,
  validateExecutionPlacementEvidence,
} from "../web/lib/execution-placement-evidence.js";
import { formatEvidenceScope } from "../web/lib/format-evidence-scope.js";

const sha = "a".repeat(64);
const ops = (names) => names.map((name, index) => ({ index, name }));
const conserves = (evidence) => {
  const flow = evidence.flow;
  assert.equal(flow.rendered_item_count, flow.segments.reduce((sum, row) => sum + row.item_count, 0));
  if (flow.segments.length) {
    assert.equal(flow.segments[0].start_position, 0);
    assert.equal(flow.segments.at(-1).end_position + 1, flow.scope_item_count);
    flow.segments.forEach((row, index) => {
      assert.equal(row.item_count, row.end_position - row.start_position + 1);
      if (index) assert.equal(row.start_position, flow.segments[index - 1].end_position + 1);
    });
  }
  assert.equal(validateExecutionPlacementEvidence(evidence).valid, true);
};

const tflite = {
  format: "tflite",
  model_sha256: sha,
  ops: ops(["CONV_2D", "RELU", "AVERAGE_POOL_2D", "CONV_2D", "ADD"]).map((op, index) => ({
    ...op,
    xnnpack_chain_id: [0, 0, -1, 1, 1][index],
    xnnpack_break_class: index === 2 ? "high-adjacent-mac-exposure" : null,
  })),
  delegation_repair: {
    runtime_build_risks: [{
      baseline_conditionally_delegatable_op_count: 4,
      required_build_configuration: "XNNPACK enabled",
    }],
  },
  tflite_delegate_compatibility_evidence: {
    profiles: [
      { id: "gpu", label: "GPU delegate", source_candidate_after_artifact_precheck_count: 3 },
      { id: "nnapi", label: "NNAPI", source_candidate_after_artifact_precheck_count: 2 },
    ],
  },
};
const tfliteStatic = buildExecutionPlacementEvidence(tflite);
assert.equal(tfliteStatic.schema, EXECUTION_PLACEMENT_EVIDENCE_SCHEMA);
assert.equal(tfliteStatic.state, "SOURCE-PINNED PREDICTION");
assert.equal(tfliteStatic.flow.scope_item_count, 5);
assert.equal(tfliteStatic.flow.covered_item_count, 5);
assert.deepEqual(tfliteStatic.flow.segments.map((row) => row.item_count), [2, 1, 2]);
assert.deepEqual(tfliteStatic.portfolios.map((row) => [row.id, row.candidate_count, row.total_count]), [
  ["xnnpack", 4, 5], ["gpu", 3, 5], ["nnapi", 2, 5],
]);
conserves(tfliteStatic);

const partialAssignments = {
  artifact_sha256: sha,
  assignments: [
    { op_index: 0, provider: "XNNPACK" },
    { op_index: 1, provider: "XNNPACK" },
  ],
  runtime_identity_status: "bound",
};
const tflitePartial = buildExecutionPlacementEvidence(tflite, partialAssignments);
assert.equal(tflitePartial.state, "PARTIAL RUNTIME OBSERVATION");
assert.equal(tflitePartial.levels[3].state, "PARTIAL OBSERVED");
assert.equal(tflitePartial.flow.covered_item_count, 2);
assert.equal(tflitePartial.flow.rendered_item_count, 5);
assert.equal(tflitePartial.runtime_observation.rejected_row_count, 0);
assert.equal(formatEvidenceScope("tflite", { analysis: tflite, runtimeEvidence: partialAssignments }).runtimePartiallyObserved, true);
assert.match(formatEvidenceScope("tflite", { analysis: tflite, runtimeEvidence: partialAssignments }).runtimeStatus, /2\/5 items/);
conserves(tflitePartial);
assert.throws(() => buildExecutionPlacementEvidence(tflite, {
  artifact_sha256: sha,
  assignments: [{ op_index: 0, provider: "CPU" }, { op_index: 0, provider: "XNNPACK" }],
}), /duplicates original op/);
assert.throws(() => buildExecutionPlacementEvidence(tflite, {
  artifact_sha256: sha,
  assignments: [{ op_index: 99, provider: "CPU" }],
}), /out-of-scope/);
assert.throws(() => buildExecutionPlacementEvidence(tflite, {
  artifact_sha256: "b".repeat(64),
  assignments: [{ op_index: 0, provider: "CPU" }],
}), /active artifact SHA-256/);
assert.throws(() => buildExecutionPlacementEvidence({
  ...tflite,
  delegation_repair: { runtime_build_risks: [{ baseline_conditionally_delegatable_op_count: 3 }] },
}), /baseline differs/);

const onnx = {
  format: "onnx",
  model_sha256: sha,
  ops: ops(["Conv", "Relu", "Add"]),
  ort_compatibility_evidence: {
    execution_providers: [
      { execution_provider: "CPUExecutionProvider", label: "CPU EP", source_candidate_after_artifact_precheck_count: 3 },
      { execution_provider: "XnnpackExecutionProvider", label: "XNNPACK EP", source_candidate_after_artifact_precheck_count: 2 },
    ],
  },
};
const onnxStatic = buildExecutionPlacementEvidence(onnx);
assert.equal(onnxStatic.flow.segments.length, 0);
assert.equal(onnxStatic.flow.scope_item_count, 0);
assert.equal(onnxStatic.portfolios.length, 2);
conserves(onnxStatic);
const onnxPartial = buildExecutionPlacementEvidence(onnx, {
  artifact_sha256: sha,
  assignments: [{ op_index: 0, observed_provider: "XnnpackExecutionProvider" }],
});
assert.equal(onnxPartial.state, "PARTIAL RUNTIME OBSERVATION");
assert.equal(onnxPartial.levels[2].state, "BUILD BOUND");
assert.equal(onnxPartial.flow.scope_item_count, 3);
conserves(onnxPartial);

const coreml = {
  format: "coreml",
  model_sha256: sha,
  ops: ops(["conv", "relu", "add"]),
  coreml: { deployment_floor: { status: "assessed" } },
};
const corePlan = {
  schema: "deepbom.coreml_compute_plan.v1",
  artifact: { sha256: sha },
  configuration: { compute_units: "ALL" },
  structure: { rows: [
    { op_index: 0, preferred_compute_device: "NEURAL_ENGINE" },
    { op_index: 1, preferred_compute_device: "GPU" },
    { op_index: 2, preferred_compute_device: "GPU" },
  ] },
  summary: { preferred_compute_device_counts: { NEURAL_ENGINE: 1, GPU: 2 } },
  boundary: "Anticipated plan only.",
};
const coreEvidence = buildExecutionPlacementEvidence(coreml, corePlan);
assert.equal(coreEvidence.state, "ANTICIPATED - NOT EXECUTED");
assert.equal(coreEvidence.levels[3].state, "NOT OBSERVED");
assert.deepEqual(coreEvidence.flow.segments.map((row) => row.item_count), [1, 2]);
conserves(coreEvidence);
assert.throws(() => buildExecutionPlacementEvidence(coreml, {
  ...corePlan,
  structure: { rows: corePlan.structure.rows.map((row, index) => index ? row : { ...row, op_index: 2 }) },
}), /operation order differs/);
assert.throws(() => buildExecutionPlacementEvidence(coreml, {
  ...corePlan,
  structure: { rows: corePlan.structure.rows.slice(0, 2) },
}), /operation count differs/);

const gguf = {
  format: "gguf",
  model_sha256: sha,
  tensor_count: 4,
  gguf: { backend_compatibility: { status: "source_candidate" } },
};
const ggufManifest = {
  schema: "deepbom.gguf_runtime_environment.v2",
  artifact: { sha256: sha },
  selection: { requested_backend_label: "CUDA", gpu_layers: 8, context_size: 2048, batch_size: 128 },
  compute_graph: {
    graph_count: 1,
    split_count: 2,
    successful_dispatch_count: 2,
    dispatch_count: 2,
    interpretation_boundary: "Captured scheduler graph only.",
    graphs: [{ scheduled_nodes: [
      { scheduled_index: 0, backend: "CPU" },
      { scheduled_index: 1, backend: "CUDA" },
      { scheduled_index: 2, backend: "CUDA" },
    ] }],
  },
};
const ggufEvidence = buildExecutionPlacementEvidence(gguf, ggufManifest);
assert.equal(ggufEvidence.state, "RUNTIME OBSERVED");
assert.deepEqual(ggufEvidence.flow.segments.map((row) => row.item_count), [1, 2]);
conserves(ggufEvidence);
assert.throws(() => buildExecutionPlacementEvidence(gguf, {
  ...ggufManifest,
  compute_graph: { ...ggufManifest.compute_graph, graphs: [{ scheduled_nodes: [{ scheduled_index: 0 }] }] },
}), /lacks a backend/);

const safetensors = buildExecutionPlacementEvidence({
  format: "safetensors",
  model_sha256: sha,
  tensor_count: 7,
});
assert.equal(safetensors.state, "NOT ASSESSABLE FROM CONTAINER");
assert.equal(safetensors.flow.scope_item_count, 0);
assert.equal(safetensors.portfolios.length, 0);
conserves(safetensors);

const executorch = {
  format: "executorch",
  executorch_container: "pte",
  model_sha256: sha,
  ops: [
    { index: 0, name: "aten::add.out", instruction_kind: "KernelCall" },
    { index: 1, name: "QnnBackend", instruction_kind: "DelegateCall" },
    { index: 2, name: "aten::relu.out", instruction_kind: "KernelCall" },
  ],
};
const execuTorchEvidence = buildExecutionPlacementEvidence(executorch);
assert.equal(execuTorchEvidence.state, "SERIALIZED DELEGATE CALLS");
assert.equal(execuTorchEvidence.levels[2].state, "UNBOUND");
assert.deepEqual(execuTorchEvidence.flow.segments.map((row) => row.item_count), [1, 1, 1]);
assert.match(execuTorchEvidence.interpretation_boundary, /not an execution trace/i);
conserves(execuTorchEvidence);
const execuTorchPtdEvidence = buildExecutionPlacementEvidence({
  format: "executorch",
  executorch_container: "ptd",
  model_sha256: sha,
  tensor_count: 2,
  ops: [],
});
assert.equal(execuTorchPtdEvidence.state, "NOT APPLICABLE TO PTD");
assert.equal(execuTorchPtdEvidence.flow.scope_item_count, 0);
conserves(execuTorchPtdEvidence);
assert.throws(() => buildExecutionPlacementEvidence({ format: "unknown", model_sha256: sha }), /does not support format/);

const tampered = structuredClone(tfliteStatic);
tampered.flow.segments[0].item_count += 1;
assert.equal(validateExecutionPlacementEvidence(tampered).valid, false);

console.log("Execution placement evidence passed (6 formats, source/config/runtime boundaries, partial coverage, segment conservation, and fail-closed mutations).");
