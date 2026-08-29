import assert from "node:assert/strict";

import { deriveCurrentArtifactCapabilityRow, FORMAT_CAPABILITY_MATRIX } from "../web/lib/format-capability-view.js";
import { buildPublicVerificationManifest, validatePublicVerificationManifest } from "../web/lib/public-verification-manifest.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { buildRuntimeCapturePlan, runtimeCapturePlanAvailability } from "../web/lib/runtime-evidence-closure.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const SHA = "a".repeat(64);
const TARGET_SHA = "b".repeat(64);

checkCapabilityMatrix();
checkRuntimeCapturePlans();
checkPublicVerificationManifest();

console.log("Product guidance contracts passed (format scope, runtime capture plan, detached report manifest).");

function checkCapabilityMatrix() {
  assert.deepEqual(Object.keys(FORMAT_CAPABILITY_MATRIX), ["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"]);
  assert.equal(FORMAT_CAPABILITY_MATRIX.tflite.cells.length, 5);
  assert.equal(FORMAT_CAPABILITY_MATRIX.onnx.cells[3], "source");
  assert.equal(FORMAT_CAPABILITY_MATRIX.gguf.cells[1], "generated_graph");
  assert.equal(FORMAT_CAPABILITY_MATRIX.gguf.cells[2], "payload");
  assert.equal(FORMAT_CAPABILITY_MATRIX.gguf.cells[3], "source");
  assert.equal(FORMAT_CAPABILITY_MATRIX.gguf.cells[4], "import");
  assert.equal(FORMAT_CAPABILITY_MATRIX.safetensors.cells[1], "canonical_scenario");
  assert.equal(FORMAT_CAPABILITY_MATRIX.safetensors.cells[2], "payload");
  assert.equal(FORMAT_CAPABILITY_MATRIX.coreml.cells[1], "dag");
  assert.equal(FORMAT_CAPABILITY_MATRIX.coreml.cells[3], "floor");
  assert.equal(FORMAT_CAPABILITY_MATRIX.coreml.cells[4], "import");
  assert.deepEqual(FORMAT_CAPABILITY_MATRIX.executorch.cells, ["full", "dag", "tensor_contract", "serialized_delegate", "external"]);

  const gguf = deriveCurrentArtifactCapabilityRow("gguf", {
    format: "gguf", filename: "model.gguf",
    tensor_inventory: { status: "partial" },
    tensor_numerical_integrity: { status: "assessed" },
  });
  assert.deepEqual(gguf.cells.map((entry) => entry.id), ["partial", "na", "full_payload", "unbound", "import_ready"]);
  const ggufSourceCandidate = deriveCurrentArtifactCapabilityRow("gguf", {
    format: "gguf", filename: "registered.gguf",
    tensor_inventory: { status: "assessed" },
    tensor_numerical_integrity: { status: "assessed" },
    gguf: { backend_compatibility: { status: "source_candidate" } },
  });
  assert.deepEqual(ggufSourceCandidate.cells.map((entry) => entry.id), ["assessed", "na", "full_payload", "source", "import_ready"]);

  const safeTensorsScenario = deriveCurrentArtifactCapabilityRow("safetensors", {
    format: "safetensors", filename: "model.safetensors",
    tensor_inventory: { status: "assessed" },
    tensor_numerical_integrity: { status: "assessed" },
    safetensors: { hf_architecture_contract: {
      status: "assessed", model_type: "llama", tensor_layout_id: "llama_separate_qkv_gated_mlp",
      tensor_contract: { canonical_tensor_check_count: 4, canonical_tensor_shape_match_count: 4 },
    } },
  });
  assert.deepEqual(safeTensorsScenario.cells.map((entry) => entry.id), ["assessed", "canonical_scenario", "full_payload", "unbound", "import_ready"]);
  assert.match(safeTensorsScenario.cells[1].title, /4\/4 canonical tensor-shape matches/);
  assert.match(safeTensorsScenario.cells[4].title, /deepbom\.runtime\.json/);

  const onnx = deriveCurrentArtifactCapabilityRow("onnx", {
    format: "onnx", filename: "model.onnx", ops: [{ index: 0 }],
    onnx_shape_inference: { status: "partial" },
    weight_integrity: { status: "assessed" },
    ort_compatibility_assessment_status: "not_loaded",
  });
  assert.deepEqual(onnx.cells.map((entry) => entry.id), ["assessed", "partial", "assessed", "not_loaded", "import_ready"]);
  assert.doesNotMatch(onnx.cells[1].title, /0\/0 shape-rule nodes|0\/0 MAC-bearing ops/);
  assert.match(onnx.cells[1].title, /Inspect the unresolved-node ledger|MAC coverage is not closed/);
  const onnxSource = deriveCurrentArtifactCapabilityRow("onnx", {
    format: "onnx", filename: "source.onnx", ops: [{ index: 0 }],
    onnx_shape_inference: { status: "assessed" },
    weight_integrity: { status: "assessed" },
    ort_compatibility_assessment_status: "complete",
  });
  assert.deepEqual(onnxSource.cells.map((entry) => entry.id), ["assessed", "decoded", "assessed", "source", "import_ready"]);
  const onnxBuildBound = deriveCurrentArtifactCapabilityRow("onnx", {
    format: "onnx", filename: "build.onnx", ops: [{ index: 0 }],
    onnx_shape_inference: { status: "assessed" },
    weight_integrity: { status: "assessed" },
    ort_compatibility_assessment_status: "complete",
  }, { source: { adapter: { native_capture: { selected_build_provider_binding: { bindings: [{ backend_name: "cpu", bundled: true }] } } } } });
  assert.equal(onnxBuildBound.cells[3].id, "build_bound");

  const coreml = deriveCurrentArtifactCapabilityRow("coreml", {
    format: "coreml", filename: "model.mlmodel", ops: [],
    weight_integrity: { status: "partial_package_blob_binding_required" },
  });
  assert.deepEqual(coreml.cells.map((entry) => entry.id), ["assessed", "not_decoded", "partial", "runtime_needed", "external"]);
  const coremlWithFloor = deriveCurrentArtifactCapabilityRow("coreml", {
    format: "coreml", filename: "model.mlmodel", ops: [{ index: 0 }],
    weight_integrity: { status: "assessed" },
    coreml: { neural_network: {}, deployment_floor: { status: "assessed" } },
  });
  assert.deepEqual(coremlWithFloor.cells.map((entry) => entry.id), ["assessed", "decoded", "assessed", "floor", "import_ready"]);

  const executorch = deriveCurrentArtifactCapabilityRow("executorch", {
    format: "executorch", filename: "model.pte", executorch_container: "pte",
    subgraphs: 1, operator_count: 2, tensor_count: 3,
    executorch_program: { delegate_instruction_count: 1, delegates: [{ backend_id: "XnnpackBackend" }] },
    weight_integrity: { assessed_tensors: 2 },
  });
  assert.deepEqual(executorch.cells.map((entry) => entry.id), ["assessed", "decoded", "partial", "serialized", "import_ready"]);
  assert.match(executorch.cells[3].title, /1 delegate call/);
  assert.match(executorch.cells[4].title, /deepbom\.executorch-build\.json/);
  assert.match(executorch.cells[4].title, /Execution remains a separate native evidence layer/);
}

function checkRuntimeCapturePlans() {
  const tflite = buildRuntimeCapturePlan({
    format: "tflite",
    filename: "model.tflite",
    model_sha256: SHA,
    target_profile: { id: "rpi5_a76", profile_sha256: TARGET_SHA },
    inputs: [{ index: 3, name: "serving_default_image:0", dtype: "UINT8", shape: [1, 224, 224, 3] }],
  });
  assert.match(tflite.command, /capture:pinned-runtime/);
  assert.match(tflite.command, /--input 3:serving_default_image-0:1x224x224x3/);
  assert.equal(tflite.target_profile.sha256, TARGET_SHA);
  assertSelfHash(tflite, "plan_sha256");

  const onnx = buildRuntimeCapturePlan({
    format: "onnx",
    filename: "model.onnx",
    model_sha256: SHA,
    inputs: [{ index: 0, name: "image", dtype: "FLOAT32", shape: [1, 3, 224, 224] }],
  });
  assert.match(onnx.command, /capture:pinned-ort/);
  assert.match(onnx.command, /--shape=image=1,3,224,224/);
  assertSelfHash(onnx, "plan_sha256");

  const gguf = buildRuntimeCapturePlan({ format: "gguf", filename: "model.gguf", model_sha256: SHA });
  assert.match(gguf.command, /capture:gguf-runtime/);
  assert.equal(gguf.import_contract.schema, "deepbom.gguf_runtime_environment.v2");
  assertSelfHash(gguf, "plan_sha256");
  assert.equal(runtimeCapturePlanAvailability({ format: "safetensors", model_sha256: SHA }).available, false);
  const coreml = buildRuntimeCapturePlan({ format: "coreml", filename: "model.mlmodel", model_sha256: SHA, ops: [{ index: 0 }], coreml: { model_type: "neuralNetwork", neural_network: {} } });
  assert.match(coreml.command, /capture:coreml-plan/);
  assert.equal(coreml.import_contract.schema, "deepbom.coreml_compute_plan.v1");
  assertSelfHash(coreml, "plan_sha256");

  assert.throws(() => buildRuntimeCapturePlan({
    format: "onnx", model_sha256: SHA, inputs: [{ name: "dynamic", shape_signature: [-1, 3, 224, 224] }],
  }), /statically bound input shape/);
  assert.equal(runtimeCapturePlanAvailability({
    format: "onnx", model_sha256: SHA, inputs: [{ name: "dynamic", shape_signature: [-1, 3, 224, 224] }],
  }).available, false);
}

function checkPublicVerificationManifest() {
  const analysis = {
    filename: "model.tflite",
    format: "tflite",
    model_sha256: SHA,
    file_size_bytes: 16,
    operator_count: 1,
    tensor_count: 2,
    total_macs: 4,
    mac_assessment: { status: "assessed", compute_ops: 1, assessed_compute_ops: 1 },
    target_profile: { id: "rpi5_a76", profile_sha256: TARGET_SHA },
    inputs: [{ name: "image", dtype: "UINT8", shape: [1, 1, 1, 1], scale_sample: [0.5], zero_point_sample: [128] }],
    outputs: [{ name: "score", dtype: "UINT8", shape: [1, 1], scale_sample: [0.25], zero_point_sample: [0] }],
    quantization_status: { label: "full integer", summary: "assessed", detail: "test" },
    findings: [],
  };
  const scope = {
    label: "TFLite",
    completion: "TFLite static deployment audit run complete",
    evidenceClass: "STATIC ARTIFACT EVIDENCE",
    depth: "Deep graph and deployment-model audit",
    assessed: "Artifact graph and contracts",
    runtimeStatus: "Runtime execution not observed in this run",
    runtimeBoundary: "Runtime assignment requires imported evidence",
    releaseStatus: "Release readiness not assessed",
  };
  const manifest = buildPublicVerificationManifest({
    analysis,
    context: { generatedAt: "2026-08-11T00:00:00.000Z" },
    scope,
    origin: "https://deepbom.org",
  });
  assert(validatePublicVerificationManifest(manifest));
  assert.equal(manifest.artifact.sha256, SHA);
  assert.equal(manifest.target_profile.sha256, TARGET_SHA);
  assert.equal(manifest.runtime_evidence.binding_status, "not_imported");
  assertSelfHash(manifest, "manifest_sha256");
  assert.equal(validatePublicVerificationManifest({ ...manifest, origin: "https://tampered.invalid" }), false);
}

function assertSelfHash(value, field) {
  const body = { ...value };
  const digest = body[field];
  delete body[field];
  assert.equal(digest, sha256TextHex(canonicalJson(body)));
}
