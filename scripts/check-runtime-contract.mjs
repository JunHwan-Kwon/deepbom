import {
  availableBrowserBackends,
  backendCandidates,
  backendProfileText,
  benchmarkErrorStatus,
  createBenchmarkInputSpec,
  createBenchmarkOutputContract,
  createFakeInputData,
  createFakeInputSpec,
  measureBrowserBenchmarkPhases,
  resolveFakeInputShape,
  runtimeDtype,
  runtimeGuardMessage,
  runtimeGuardTitle,
  runtimeInputMatchesPrepared,
  selectWasmCalibrationResult,
  RUNTIME_CLOCK_INVALID,
  RUNTIME_EXPIRED,
  BROWSER_BENCHMARK_TIMING_METHOD,
  BROWSER_BENCHMARK_STATISTICS_METHOD,
  BROWSER_BENCHMARK_NOISE_METHOD,
} from "../web/lib/runtime.js";
import { benchmarkOnnxModel, createOnnxBenchmarkInputSpec } from "../web/lib/onnx-benchmark.js";
import { benchmarkTfliteModel } from "../web/lib/tflite-benchmark.js";
import { benchmarkNoise } from "../web/lib/format.js";
import { sha256Hex } from "../web/lib/hash.js";
import { readReportHistorySettings, writeReportHistorySettings } from "../web/lib/report-store.js";
import {
  createResearchInputData,
  perturbationOptions,
  perturbTypedArray,
  summarizeOutputDriftProjectionEnsemble,
  summarizeProjectedHessians,
} from "../web/lib/research.js";
import { createCheck } from "./check-assert.mjs";

const {
  done,
  expect,
  expectDeepEqual,
  expectEqual,
  expectThrows,
  expectTypedArray,
} = createCheck("Runtime contract check");

const noAccelerators = {};
const webGpuOnly = { gpu: {} };
const webNnAndGpu = { ml: {}, gpu: {} };

expectDeepRows([
  [availableBrowserBackends(noAccelerators), ["WASM"], "backend list: wasm"],
  [availableBrowserBackends(webGpuOnly), ["WebGPU", "WASM"], "backend list: webgpu first"],
  [availableBrowserBackends(webNnAndGpu), ["WebNN", "WebGPU", "WASM"], "backend list: webnn first"],
]);

expectRows([
  [backendProfileText("onnx", "auto", webNnAndGpu), "ONNX Runtime Web", "profile: onnx auto"],
  [backendProfileText("tflite", "auto", webNnAndGpu), "Auto: WebNN / WebGPU / WASM", "profile: tflite auto"],
  [backendProfileText("tflite", "webgpu", noAccelerators), "WebGPU", "profile: explicit webgpu"],
]);

expectDeepRows([
  [backendCandidates("auto", "onnx", webNnAndGpu), ["wasm"], "candidates: onnx auto"],
  [backendCandidates("auto", "tflite", webNnAndGpu), ["webnn", "webgpu", "wasm"], "candidates: tflite auto"],
  [backendCandidates("webgpu", "tflite", noAccelerators), ["webgpu"], "candidates: explicit webgpu"],
]);

const calibrationCandidate = { ok: true, backend: "wasm", stats: { p50: 4.2 } };
expectEqual(
  selectWasmCalibrationResult([{ ok: true, backend: "webgpu" }, calibrationCandidate]),
  calibrationCandidate,
  "runtime calibration should consume the measured WASM backend result.",
);
expectEqual(selectWasmCalibrationResult([{ ok: false, backend: "wasm" }]), null, "failed WASM results must not calibrate the static estimate.");

for (const [dtype, expected] of [
  ["FLOAT32", "float32"],
  ["INT8", "int8"],
  ["UINT8", "uint8"],
  ["INT32", "int32"],
]) {
  expectEqual(runtimeDtype(dtype), expected, `${dtype} should map to runtime ${expected}.`);
}

for (const [dtype, ctor] of [["float32", Float32Array], ["int32", Int32Array]]) {
  expectTypedArray(createFakeInputData(dtype, 3), ctor, `${dtype} fake input should use ${ctor.name}.`);
}
expectDeepRows([
  [[...createFakeInputData("uint8", 3, { zero_point_sample: [300] })], [255, 255, 255], "fake input: uint8 clamp"],
  [[...createFakeInputData("int8", 3, { zero_point_sample: [-200] })], [-128, -128, -128], "fake input: int8 clamp"],
]);
expectThrows(() => createFakeInputData("bool", 1, null, "image"), "Input image dtype bool is not supported", "unsupported fake input dtype should explain input name.");

for (const [runtimeShape, tensor, expected, label] of [
  [[-1, 144, 240, 3], { shape: [1, 144, 240, 3] }, [1, 144, 240, 3], "shape: static batch"],
  [[-1, 144, 240, 3], { shape: [-1, 144, 240, 3] }, [1, 144, 240, 3], "shape: default batch"],
]) {
  expectDeepEqual(resolveFakeInputShape(runtimeShape, tensor), expected, label);
}
expectThrows(
  () => resolveFakeInputShape([1, -1, 240, 3], { shape: [1, -1, 240, 3] }),
  "Cannot resolve dynamic input dimension at axis 1",
  "non-batch dynamic dimension should be rejected.",
);

expect(runtimeInputMatchesPrepared({ dtype: "float32", shape: [-1, 2] }, { shape: [1, 2] }, { dtype: "float32", shape: [1, 2] }), "prepared input should match resolved runtime shape and dtype.");
expect(!runtimeInputMatchesPrepared({ dtype: "float32", shape: [-1, 2] }, { shape: [1, 2] }, { dtype: "int8", shape: [1, 2] }), "prepared input should reject dtype mismatch.");
expect(runtimeInputMatchesPrepared({ dtype: "float32", shape: [-1, -1] }, { shape: [-1, -1] }, { dtype: "float32", shape: [1, 7] }), "prepared input should explicitly bind otherwise unresolved dynamic dimensions.");
expect(runtimeInputMatchesPrepared({ dtype: "float32", shape: [-1, 2] }, { shape: [1, 2], shape_signature: [-1, 2] }, { dtype: "float32", shape: [2, 2] }), "artifact shape signatures should permit prepared values on dynamic axes even when the stored default shape is concrete.");
expect(!runtimeInputMatchesPrepared({ dtype: "float32", shape: [2, 2] }, { shape: [1, 2], shape_signature: [1, 2] }, { dtype: "float32", shape: [2, 2] }), "conflicting known runtime and artifact dimensions must fail closed.");

const preparedSpec = createBenchmarkInputSpec(
  { dtype: "float32", shape: [-1, 2], name: "input" },
  { shape: [1, 2] },
  { dtype: "float32", shape: [1, 2], data: new Float32Array([3, 4]) },
  0,
);
expectEqual(preparedSpec.basis, "prepared tensor", "benchmark input spec should prefer a matching prepared input for input 0.");
expectDeepEqual([...preparedSpec.data], [3, 4], "benchmark input spec should clone prepared data.");
const preparedDynamicSpec = createBenchmarkInputSpec(
  { dtype: "float32", shape: [-1, -1], name: "dynamic" },
  { shape: [-1, -1] },
  { dtype: "float32", shape: [1, 7], data: new Float32Array(7) },
  0,
);
expectDeepEqual(preparedDynamicSpec.shape, [1, 7], "A prepared TFLite tensor should bind an unresolved non-batch dimension.");
expectThrows(() => createBenchmarkInputSpec(
  { dtype: "float32", shape: [1, 2], name: "input" },
  { shape: [1, 2] },
  { dtype: "int8", shape: [1, 2], data: new Int8Array(2) },
  0,
), "does not match runtime dtype", "A mismatched prepared TFLite tensor must fail closed instead of silently switching to synthetic input.");
expectThrows(() => createBenchmarkInputSpec(
  { dtype: "float32", shape: [1, 2], name: "input" },
  { shape: [1, 2] },
  { dtype: "float32", shape: [1, 2], data: new Float32Array(1) },
  0,
), "does not match shape element count", "Prepared TFLite input data length must conserve the executed shape.");

const fakeSpec = createFakeInputSpec({ dtype: "uint8", shape: [-1, 2], name: "image" }, { shape: [1, 2], zero_point_sample: [7] });
expectEqual(fakeSpec.basis, "synthetic tensor", "fake input spec should mark synthetic tensor basis.");
expectDeepEqual(fakeSpec.shape, [1, 2], "fake input spec should resolve dynamic batch.");
expectDeepEqual([...fakeSpec.data], [7, 7], "fake input spec should use dtype-aware fake input data.");
expectThrows(() => createFakeInputSpec({ dtype: "float32", shape: [10_001, 10_001], name: "huge" }, { shape: [10_001, 10_001] }), "browser benchmark limit", "Excessive TFLite browser input allocation must fail closed.");
const outputContract = createBenchmarkOutputContract({ index: 0, name: "scores", artifactDtype: "FLOAT32", runtimeDtype: "float32", declaredShape: [1, 2], artifactShapeSignature: [-1, 2], runtimeDeclaredShape: [1, 2], executedShape: [1, 2], data: new Float32Array(2) });
expectEqual(outputContract.element_count, 2, "Observed output contracts should conserve executed shape and element count.");
expectThrows(() => createBenchmarkOutputContract({ name: "scores", artifactDtype: "FLOAT32", runtimeDtype: "int32", declaredShape: [1], executedShape: [1], data: new Int32Array(1) }), "conflicts with artifact dtype", "Observed output dtype disagreement must fail closed.");
expectThrows(() => createBenchmarkOutputContract({ name: "scores", artifactDtype: "FLOAT32", runtimeDtype: "float32", declaredShape: [1, 2], executedShape: [1, 3], data: new Float32Array(3) }), "conflicts with artifact", "Observed output shape disagreement must fail closed.");
expectThrows(() => createBenchmarkOutputContract({ name: "scores", artifactDtype: "FLOAT32", runtimeDtype: "float32", declaredShape: [1, 2], executedShape: [1, 2], data: new Float32Array(1) }), "does not match shape element count", "Observed output data length must conserve its executed shape.");

const onnxWide = createOnnxBenchmarkInputSpec({ name: "wide", dtype: "FLOAT32", shape: [1, 5000], shape_declared: true });
expectDeepEqual(onnxWide.shape, [1, 5000], "ONNX benchmark input must preserve declared dimensions above 4096 without silent clamping.");
expectEqual(onnxWide.contract.element_count, 5000, "ONNX benchmark input element count should bind the executed shape.");
expectEqual(onnxWide.contract.runtime_declared_shape, null, "ONNX static parser input contracts must not pretend that session runtime metadata was separately observed.");
expectEqual(onnxWide.contract.artifact_shape_signature, null, "ONNX input contracts must not invent a TFLite shape signature.");
const onnxRuntimeBound = createOnnxBenchmarkInputSpec(
  { name: "runtime_bound", dtype: "FLOAT32", shape: [-1, 5000], shape_declared: true }, null, 0,
  { name: "runtime_bound", isTensor: true, type: "float32", shape: [2, 5000] },
);
expectDeepEqual(onnxRuntimeBound.shape, [2, 5000], "ONNX runtime metadata should bind a positive dynamic dimension for execution.");
expectDeepEqual(onnxRuntimeBound.contract.runtime_declared_shape, [2, 5000], "ONNX benchmark evidence should preserve session-observed input shape metadata.");
expectThrows(() => createOnnxBenchmarkInputSpec(
  { name: "dtype_conflict", dtype: "FLOAT32", shape: [1], shape_declared: true }, null, 0,
  { name: "dtype_conflict", isTensor: true, type: "int32", shape: [1] },
), "conflicts with artifact dtype", "ONNX runtime and artifact dtype disagreement must fail closed.");
expectThrows(() => createOnnxBenchmarkInputSpec(
  { name: "shape_conflict", dtype: "FLOAT32", shape: [1, 4], shape_declared: true }, null, 0,
  { name: "shape_conflict", isTensor: true, type: "float32", shape: [2, 4] },
), "conflicts with artifact", "ONNX runtime and artifact shape disagreement must fail closed.");
expectThrows(() => createOnnxBenchmarkInputSpec(
  { name: "artifact_input", dtype: "FLOAT32", shape: [1], shape_declared: true }, null, 0,
  { name: "runtime_input", isTensor: true, type: "float32", shape: [1] },
), "metadata name", "ONNX runtime metadata must not bind a differently named artifact input by position.");
const onnxBatchDynamic = createOnnxBenchmarkInputSpec({ name: "batch", dtype: "UINT8", shape: [-1, 3], shape_declared: true, zero_point_sample: [129] });
expectDeepEqual(onnxBatchDynamic.shape, [1, 3], "ONNX synthetic benchmark may bind only an unresolved batch dimension to one.");
expectDeepEqual([...onnxBatchDynamic.data], [129, 129, 129], "ONNX integer synthetic input should represent real zero with the embedded zero point.");
const onnxPrepared = createOnnxBenchmarkInputSpec(
  { name: "dynamic", dtype: "FLOAT32", shape: [1, -1], shape_declared: true },
  { dtype: "float32", shape: [1, 7], data: new Float32Array(7).fill(2) },
);
expectDeepEqual(onnxPrepared.shape, [1, 7], "A prepared ONNX tensor should bind an otherwise unresolved non-batch dimension.");
expectEqual(onnxPrepared.contract.basis, "prepared_tensor", "Prepared ONNX input basis should be explicit.");
const onnxScalar = createOnnxBenchmarkInputSpec({ name: "scalar", dtype: "INT32", shape: [], shape_declared: true });
expectEqual(onnxScalar.contract.element_count, 1, "Declared rank-zero ONNX input should allocate one scalar element.");
expectThrows(() => createOnnxBenchmarkInputSpec({ name: "dynamic", dtype: "FLOAT32", shape: [1, -1], shape_declared: true }), "Cannot resolve dynamic ONNX input", "Unbound non-batch dynamic ONNX input must fail closed.");
expectThrows(() => createOnnxBenchmarkInputSpec({ name: "missing", dtype: "FLOAT32", shape: [], shape_declared: false }), "has no declared shape", "Missing ONNX input shape must not be substituted with [1].");
expectThrows(() => createOnnxBenchmarkInputSpec({ name: "huge", dtype: "FLOAT32", shape: [10_001, 10_001], shape_declared: true }), "browser benchmark limit", "Excessive ONNX browser input allocation must fail closed.");
expectThrows(() => createOnnxBenchmarkInputSpec({ name: "bf16", dtype: "BFLOAT16", shape: [1], shape_declared: true }), "cannot be relabeled as FLOAT16", "BFLOAT16 input must not execute through a mislabeled FLOAT16 tensor.");

let fakeOrtBackend = "";
let fakeOrtRunCount = 0;
let fakeOrtInputDisposals = 0;
let fakeOrtOutputDisposals = 0;
let fakeOrtSessionReleases = 0;
let fakeOrtExternalData = null;
const fakeOrtExternalBytes = new Uint8Array([1, 3, 3, 7]);
const fakeOrtExternalSha256 = await sha256Hex(fakeOrtExternalBytes);
const fakeOrtSession = {
  inputNames: ["input"],
  inputMetadata: [{ name: "input", isTensor: true, type: "float32", shape: [1, 2] }],
  outputNames: ["scores"],
  outputMetadata: [{ name: "scores", isTensor: true, type: "float32", shape: [1, 1] }],
  run: async (feeds) => {
    fakeOrtRunCount += 1;
    expectEqual(feeds.input.type, "float32", "Extracted ONNX benchmark should bind the artifact dtype to the runtime tensor.");
    expectDeepEqual(feeds.input.dims, [1, 2], "Extracted ONNX benchmark should bind the artifact shape to the runtime tensor.");
    return {
      scores: {
        type: "float32",
        dims: [1, 1],
        data: new Float32Array([fakeOrtRunCount]),
        dispose: () => { fakeOrtOutputDisposals += 1; },
      },
    };
  },
  release: async () => { fakeOrtSessionReleases += 1; },
};
const fakeOrtResult = await benchmarkOnnxModel({
  modelBytes: new Uint8Array([8, 9]),
  analysis: {
    inputs: [{ name: "input", dtype: "FLOAT32", shape: [1, 2], shape_declared: true }],
    outputs: [{ name: "scores", dtype: "FLOAT32", shape: [1, 1], shape_declared: true }],
    onnx_external_data: {
      tensor_count: 1,
      verified_payload_count: 1,
      payload_verification_failed_count: 0,
      tensors: [{ sidecar_path: "weights.bin", sidecar_bytes: 4, sidecar_sha256: fakeOrtExternalSha256 }],
    },
  },
  backend: "wasm",
  warmup: 1,
  runs: 2,
  externalDataFiles: [{ path: "weights.bin", bytes: fakeOrtExternalBytes, sha256: fakeOrtExternalSha256 }],
}, {
  createSession: async (_bytes, options) => {
    fakeOrtBackend = options.executionProviders[0];
    fakeOrtExternalData = options.externalData;
    return fakeOrtSession;
  },
  createTensor: (type, data, dims) => ({
    type,
    data,
    dims,
    dispose: () => { fakeOrtInputDisposals += 1; },
  }),
});
expectEqual(fakeOrtBackend, "wasm", "Extracted ONNX benchmark should create the session with the requested execution provider.");
expectEqual(fakeOrtExternalData?.[0]?.path, "weights.bin", "Extracted ONNX benchmark should mount the statically verified model-relative sidecar path.");
expectEqual(fakeOrtExternalData?.[0]?.data, fakeOrtExternalBytes, "Extracted ONNX benchmark should mount the exact verified sidecar bytes without copying or substituting another payload.");
expectEqual(fakeOrtRunCount, 4, "Extracted ONNX benchmark should execute one cold, one warmup, and two measured runs.");
expectEqual(fakeOrtInputDisposals, 1, "Extracted ONNX benchmark should release every feed tensor after the session finishes.");
expectEqual(fakeOrtOutputDisposals, 4, "Extracted ONNX benchmark should release outputs from every execution phase.");
expectEqual(fakeOrtSessionReleases, 1, "Extracted ONNX benchmark should release the runtime session exactly once.");
expectDeepEqual(fakeOrtResult.phaseCounts, { cold_first_runs: 1, warmup_runs: 1, measured_runs: 2 }, "Extracted ONNX benchmark should preserve exact phase counts.");
expectEqual(fakeOrtResult.inputContracts[0].element_count, 2, "Extracted ONNX benchmark should preserve its executed input contract.");
expectEqual(fakeOrtResult.outputContracts[0].element_count, 1, "Extracted ONNX benchmark should preserve its observed output contract.");
expectEqual(fakeOrtResult.externalDataContract.status, "bound_verified_external_data", "Successful external-data execution should expose its exact runtime binding contract.");
expectEqual(fakeOrtResult.externalDataContract.files[0].sha256, fakeOrtExternalSha256, "Runtime external-data evidence should retain the immediately rechecked full-file SHA-256.");
expect(/^[0-9a-f]{64}$/.test(fakeOrtResult.outputDigest), "Extracted ONNX benchmark should hash the final observed output.");

let changedExternalDataMessage = "";
try {
  await benchmarkOnnxModel({
    modelBytes: new Uint8Array([8, 9]),
    analysis: {
      inputs: [], outputs: [],
      onnx_external_data: {
        tensor_count: 1, verified_payload_count: 1, payload_verification_failed_count: 0,
        tensors: [{ sidecar_path: "weights.bin", sidecar_bytes: 4, sidecar_sha256: fakeOrtExternalSha256 }],
      },
    },
    backend: "wasm", warmup: 0, runs: 1,
    externalDataFiles: [{ path: "weights.bin", bytes: new Uint8Array([1, 3, 3, 8]), sha256: fakeOrtExternalSha256 }],
  }, { createSession: async () => { throw new Error("session must not be created"); } });
} catch (error) {
  changedExternalDataMessage = String(error?.message || error);
}
expect(changedExternalDataMessage.includes("changed after static verification"), "ONNX benchmark should rehash and reject sidecar bytes changed after static verification before creating a runtime session.");

let failedOrtInputDisposals = 0;
let failedOrtSessionReleases = 0;
let failedOrtBenchmarkMessage = "";
try {
  await benchmarkOnnxModel({
    modelBytes: new Uint8Array([1]),
    analysis: {
      inputs: [{ name: "input", dtype: "FLOAT32", shape: [1], shape_declared: true }],
      outputs: [{ name: "output", dtype: "FLOAT32", shape: [1], shape_declared: true }],
    },
    backend: "wasm",
    warmup: 0,
    runs: 1,
  }, {
    createSession: async () => ({
      inputNames: ["input"],
      inputMetadata: [{ name: "input", isTensor: true, type: "float32", shape: [1] }],
      outputNames: ["output"],
      outputMetadata: [{ name: "output", isTensor: true, type: "float32", shape: [1] }],
      run: async () => { throw new Error("synthetic ORT runtime failure"); },
      release: async () => { failedOrtSessionReleases += 1; },
    }),
    createTensor: () => ({ dispose: () => { failedOrtInputDisposals += 1; } }),
  });
} catch (error) {
  failedOrtBenchmarkMessage = String(error?.message || error);
}
expect(failedOrtBenchmarkMessage.includes("synthetic ORT runtime failure"), "Extracted ONNX benchmark should propagate runtime failure evidence.");
expectEqual(failedOrtInputDisposals, 1, "Extracted ONNX benchmark should release inputs after a runtime failure.");
expectEqual(failedOrtSessionReleases, 1, "Extracted ONNX benchmark should release the session after a runtime failure.");

const phaseEvents = [];
let phaseRun = 0;
const phases = await measureBrowserBenchmarkPhases({
  execute: async () => { phaseRun += 1; phaseEvents.push(`run:${phaseRun}`); return { phaseRun }; },
  dispose: async (output) => phaseEvents.push(`dispose:${output.phaseRun}`),
  observeFinal: async (output) => { phaseEvents.push(`observe:${output.phaseRun}`); return { final: output.phaseRun }; },
  warmupRuns: 2,
  measuredRuns: 3,
});
expectEqual(phases.timingMethod, BROWSER_BENCHMARK_TIMING_METHOD, "Browser benchmark timing method must be versioned.");
expectEqual(phases.statisticsMethod, BROWSER_BENCHMARK_STATISTICS_METHOD, "Browser benchmark statistics method must be versioned.");
expectEqual(phases.noiseMethod, BROWSER_BENCHMARK_NOISE_METHOD, "Browser benchmark noise method must be versioned.");
expectDeepEqual(phases.noiseDiagnostics, benchmarkNoise(phases.timings), "Browser benchmark noise diagnostics must derive from the exact measured samples.");
expectDeepEqual(phases.phaseCounts, { cold_first_runs: 1, warmup_runs: 2, measured_runs: 3 }, "Browser benchmark phases should conserve cold, warmup, and measured executions.");
expectEqual(phases.timings.length, 3, "Only post-warmup measured executions belong in latency statistics.");
expectEqual(phases.finalObservation.final, 6, "Final observation should bind the last measured execution.");
expectDeepEqual(phaseEvents, ["run:1", "dispose:1", "run:2", "dispose:2", "run:3", "dispose:3", "run:4", "dispose:4", "run:5", "dispose:5", "run:6", "observe:6", "dispose:6"], "Benchmark execution order must keep the cold run outside warmup and measured samples.");
expectDeepEqual(phases.steadyStats, phases.stats, "Every measured sample occurs after the declared warmup phase.");

let fakeRuntimeBackend = "";
let fakeCompileBackend = "";
let fakeRunCount = 0;
let fakeInputDeletes = 0;
let fakeOutputDeletes = 0;
let fakeModelDeletes = 0;
const fakeTfliteResult = await benchmarkTfliteModel({
  modelBytes: new Uint8Array([1, 2, 3]),
  analysis: {
    inputs: [{ name: "input", dtype: "FLOAT32", shape: [1, 2], shape_signature: [1, 2] }],
    outputs: [{ name: "scores", dtype: "FLOAT32", shape: [1, 1], shape_signature: [1, 1] }],
  },
  backend: "wasm",
  warmup: 1,
  runs: 2,
  ensureRuntime: async (backend) => { fakeRuntimeBackend = backend; },
}, {
  loadAndCompile: async (_bytes, options) => {
    fakeCompileBackend = options.accelerator;
    return {
      getInputDetails: () => [{ name: "input", dtype: "float32", shape: new Int32Array([1, 2]) }],
      getOutputDetails: () => [{ name: "scores", dtype: "float32", shape: new Int32Array([1, 1]) }],
      run: async () => {
        fakeRunCount += 1;
        return [{
          type: { dtype: "float32", layout: { dimensions: new Int32Array([1, 1]) } },
          data: async () => new Float32Array([fakeRunCount]),
          delete: () => { fakeOutputDeletes += 1; },
        }];
      },
      delete: () => { fakeModelDeletes += 1; },
    };
  },
  tensorFromTypedArray: (data, shape) => ({ data, shape, delete: () => { fakeInputDeletes += 1; } }),
});
expectEqual(fakeRuntimeBackend, "wasm", "Extracted TFLite benchmark should prepare the requested runtime backend.");
expectEqual(fakeCompileBackend, "wasm", "Extracted TFLite benchmark should compile for the requested accelerator.");
expectEqual(fakeRunCount, 4, "Extracted TFLite benchmark should execute cold, warmup, and measured phases exactly once each.");
expectEqual(fakeInputDeletes, 1, "Extracted TFLite benchmark should release every input tensor.");
expectEqual(fakeOutputDeletes, 4, "Extracted TFLite benchmark should release every output tensor.");
expectEqual(fakeModelDeletes, 1, "Extracted TFLite benchmark should release the compiled model.");
expectDeepEqual(fakeTfliteResult.phaseCounts, { cold_first_runs: 1, warmup_runs: 1, measured_runs: 2 }, "Extracted TFLite benchmark should preserve exact phase counts.");
expectEqual(fakeTfliteResult.inputContracts[0].element_count, 2, "Extracted TFLite benchmark should preserve its executed input contract.");
expectEqual(fakeTfliteResult.outputContracts[0].element_count, 1, "Extracted TFLite benchmark should preserve its observed output contract.");
expect(/^[0-9a-f]{64}$/.test(fakeTfliteResult.outputDigest), "Extracted TFLite benchmark should hash the final observed output.");
let failedInputDeletes = 0;
let failedModelDeletes = 0;
let failedBenchmarkMessage = "";
try {
  await benchmarkTfliteModel({
    modelBytes: new Uint8Array([1]),
    analysis: { inputs: [{ name: "input", dtype: "FLOAT32", shape: [1] }], outputs: [{ name: "output", dtype: "FLOAT32", shape: [1] }] },
    backend: "wasm",
    warmup: 0,
    runs: 1,
  }, {
    loadAndCompile: async () => ({
      getInputDetails: () => [{ name: "input", dtype: "float32", shape: new Int32Array([1]) }],
      run: async () => { throw new Error("synthetic runtime failure"); },
      delete: () => { failedModelDeletes += 1; },
    }),
    tensorFromTypedArray: () => ({ delete: () => { failedInputDeletes += 1; } }),
  });
} catch (error) {
  failedBenchmarkMessage = String(error?.message || error);
}
expect(failedBenchmarkMessage.includes("synthetic runtime failure"), "Extracted TFLite benchmark should propagate runtime failure evidence.");
expectEqual(failedInputDeletes, 1, "Extracted TFLite benchmark should release inputs after a runtime failure.");
expectEqual(failedModelDeletes, 1, "Extracted TFLite benchmark should release the model after a runtime failure.");

for (const [guard, expected, label] of [
  [RUNTIME_EXPIRED, "Build expired", "guard: expired title"],
  [RUNTIME_CLOCK_INVALID, "Clock check failed", "guard: clock title"],
]) {
  expectEqual(runtimeGuardTitle(guard), expected, label);
}
expect(runtimeGuardMessage(RUNTIME_EXPIRED).includes("latest hosted deployment"), "expired runtime guard should guide users to latest hosted deployment.");

expect(
  benchmarkErrorStatus(new Error("Asyncify is not defined"), "webgpu").includes("Asyncify/JSPI"),
  "WebGPU Asyncify failure should include actionable JSPI wording.",
);
expect(
  benchmarkErrorStatus(new Error("Element type INT8 is not supported"), "wasm").startsWith("Runtime support gap:"),
  "unsupported runtime errors should be framed as support gaps.",
);

function perturbedInt8Values(mode) {
  const tensor = { zero_point_sample: [0] };
  const data = createResearchInputData("int8", 8, tensor);
  perturbTypedArray(data, "int8", tensor, perturbationOptions(mode));
  return [...data];
}

const curvePlus = perturbedInt8Values("curve_plus");
const curveMinus = perturbedInt8Values("curve_minus");
const curveWide = perturbedInt8Values("curve_wide");

expectDeepEqual(curvePlus, [1, -1, 1, -1, 1, -1, 1, -1], "curve_plus should perturb alternating +1/-1 LSB.");
expectDeepEqual(curveMinus, [-1, 1, -1, 1, -1, 1, -1, 1], "curve_minus should perturb the inverse alternating direction.");
expectDeepEqual(curveWide, [2, 2, 2, 2, 2, 2, 2, 2], "curve_wide should use a distinct +2 LSB probe, not reuse plus/minus.");

expectDeepEqual(
  summarizeProjectedHessians([]),
  { assessedCount: 0, lambdaMean: null, lambdaStd: null, directionalLambdaMaxCv: null },
  "degenerate projection ensemble should remain explicitly unassessed.",
);
expectDeepEqual(
  summarizeProjectedHessians([{ lambdaMax: 2 }, { lambdaMax: 4 }, null]),
  { assessedCount: 2, lambdaMean: 3, lambdaStd: 1, directionalLambdaMaxCv: 1 / 3 },
  "projected-curvature summary should use only finite assessed projections.",
);
const ensemble = summarizeOutputDriftProjectionEnsemble({
  axes: [-1, 0, 1],
  grids: [
    [2, 1, 2, 1, 0, 1, 2, 1, 2],
    [4, 2, 4, 2, 0, 2, 4, 2, 4],
  ],
}, { numProjections: 2, gridSize: 3, radius: 1 });
expectEqual(ensemble.meanGrid[0][0], 3, "projection ensemble mean grid should conserve finite slices");
expectEqual(ensemble.varGrid[0][0], 1, "projection ensemble spread should use population standard deviation");
expectEqual(ensemble.hessianAssessedCount, 2, "projection ensemble should retain the assessed finite-difference denominator");
expectEqual(ensemble.directionalLambdaMaxCv, 1 / 3, "projection ensemble should expose raw directional lambda CV");
expect(ensemble.protocol.interpretationBoundary.includes("not training loss"), "projection protocol should disclose its scientific interpretation boundary");

const historyStorage = {
  value: "",
  getItem() { return this.value || null; },
  setItem(_key, value) { this.value = value; },
};
expectDeepEqual(
  readReportHistorySettings(historyStorage),
  { enabled: true, retentionDays: 90, maxSnapshots: 50 },
  "local report history should default to an explicit finite retention policy",
);
writeReportHistorySettings({ enabled: false, retentionDays: 365 }, historyStorage);
expectDeepEqual(
  readReportHistorySettings(historyStorage),
  { enabled: false, retentionDays: 365, maxSnapshots: 50 },
  "local report history opt-out and retention should round-trip deterministically",
);

done("Runtime contract passed (backend candidates, fake inputs, dynamic shapes, and guard messages).");

function expectRows(rows) {
  for (const [actual, expected, label] of rows) {
    expectEqual(actual, expected, label);
  }
}

function expectDeepRows(rows) {
  for (const [actual, expected, label] of rows) {
    expectDeepEqual(actual, expected, label);
  }
}
