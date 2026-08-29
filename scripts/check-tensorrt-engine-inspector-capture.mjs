import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { canonicalJson } from "../web/lib/report-utils.js";
import { buildTensorRtStaticPreflight, createTensorRtBuildProfile } from "../web/lib/tensorrt-static-preflight.js";
import { captureTensorRtEngineInspector } from "./tensorrt-engine-inspector-capture-lib.mjs";

const root = await mkdtemp(path.join(os.tmpdir(), "deepbom-trt-inspector-check-"));
try {
  const paths = Object.fromEntries(["model.onnx", "engine.plan", "layer-info.json", "trtexec", "evidence.json"]
    .map((name) => [name, path.join(root, name)]));
  const engineInformation = {
    Layers: [{
      Name: "gemm [ONNX Layer: MatMul]", LayerType: "MatrixMultiply",
      Inputs: [{ Name: "x", Dimensions: [1, 16], "Format/Datatype": "Row major FP16" }],
      Outputs: [{ Name: "y", Dimensions: [1, 16], "Format/Datatype": "Row major FP16" }],
      TacticName: "fixture_tactic",
    }],
    Bindings: ["x", "y"],
  };
  await Promise.all([
    writeFile(paths["model.onnx"], Buffer.from("onnx-fixture")),
    writeFile(paths["engine.plan"], Buffer.from("plan-fixture")),
    writeFile(paths["layer-info.json"], `${canonicalJson(engineInformation)}\n`, "utf8"),
    writeFile(paths.trtexec, Buffer.from("tool-fixture")),
  ]);
  const config = {
    execution_path: "native_tensorrt", expected_tensorrt_version: "10.14.1", expected_cuda_version: "13.0",
    device_id: 0, device_compute_capability: "8.7",
    precision: { tf32: true, fp16: true, bf16: false, int8: false, fp8: false },
    workspace_limit_bytes: 1_073_741_824, builder_optimization_level: 3,
    dla_core: null, allow_gpu_fallback: false, calibration_cache_sha256: null, plugins: [], optimization_profiles: [],
  };
  const profile = createTensorRtBuildProfile(config);
  const evidence = await captureTensorRtEngineInspector({
    modelPath: paths["model.onnx"], profile, enginePath: paths["engine.plan"], inspectorPath: paths["layer-info.json"],
    outputPath: paths["evidence.json"], tensorrtVersion: "10.14.1", cudaVersion: "13.0", deviceId: 0,
    deviceComputeCapability: "8.7", deviceIdentity: "fixture GPU / CC 8.7", toolBinaryPath: paths.trtexec,
    invocation: "trtexec --onnx=model.onnx --saveEngine=engine.plan --exportLayerInfo=layer-info.json --profilingVerbosity=detailed",
  });
  assert.equal(evidence.build_capture.evidence_class, "DECLARED_BUILD_CAPTURE");
  assert.equal(evidence.inspector.schema_generation, "tensorrt_10x");
  assert.equal(evidence.engine.byte_length, Buffer.byteLength("plan-fixture"));
  assert.deepEqual(JSON.parse(await readFile(paths["evidence.json"], "utf8")), evidence);
  const analysis = {
    format: "onnx", model_sha256: evidence.artifact_sha256,
    inputs: [{ index: 0, name: "x", dtype: "FLOAT16", shape: [1, 16] }],
    tensors: [{ index: 0, name: "x", dtype: "FLOAT16", shape: [1, 16] }, { index: 1, name: "y", dtype: "FLOAT16", shape: [1, 16] }],
    ops: [{ index: 0, name: "MatMul", inputs: [0], outputs: [1], macs: 256, macs_status: "assessed", estimated_bytes: 64, estimated_bytes_status: "assessed" }],
    quantized_tensors: 0,
  };
  const preflight = buildTensorRtStaticPreflight(analysis, config, null, evidence);
  assert.equal(preflight.status, "engine_inspected_parser_observation_unbound");
  assert.equal(preflight.engine_inspector_evidence.engine_layer_count, 1);

  const duplicateKeyPath = path.join(root, "duplicate.json");
  await writeFile(duplicateKeyPath, '{"Layers":[],"Layers":[],"Bindings":[]}\n', "utf8");
  await assert.rejects(() => captureTensorRtEngineInspector({
    modelPath: paths["model.onnx"], profile, enginePath: paths["engine.plan"], inspectorPath: duplicateKeyPath,
    tensorrtVersion: "10.14.1", cudaVersion: "13.0", deviceId: 0, deviceIdentity: "fixture",
    toolBinaryPath: paths.trtexec, invocation: "fixture",
  }), /duplicate JSON key/);
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("TensorRT engine-inspector capture passed (strict JSON, file digests, declared provenance boundary, atomic export, and validator round-trip).");
