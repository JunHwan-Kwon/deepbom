import { benchmarkNoise, clampInteger, cloneTypedArray, latencyStats, shortError } from "./format.js";

export const RUNTIME_OK = 0;
export const RUNTIME_EXPIRED = 1;
export const RUNTIME_CLOCK_INVALID = 2;
export const BROWSER_BENCHMARK_TIMING_METHOD = "deepbom.browser_benchmark_timing.v2";
export const BROWSER_BENCHMARK_STATISTICS_METHOD = "deepbom.browser_latency_statistics.nearest_rank_population.v1";
export const BROWSER_BENCHMARK_NOISE_METHOD = "deepbom.browser_latency_noise.population_2p5sd_ols_trim10.v1";
const MAX_BROWSER_INPUT_ELEMENTS = 100_000_000;

export function runtimeGuardTitle(code) {
  if (code === RUNTIME_EXPIRED) return "Build expired";
  if (code === RUNTIME_CLOCK_INVALID) return "Clock check failed";
  return "Runtime locked";
}

export function runtimeGuardMessage(code) {
  if (code === RUNTIME_EXPIRED) {
    return "This static build is outside its allowed execution window. Open the latest hosted deployment to continue.";
  }
  if (code === RUNTIME_CLOCK_INVALID) {
    return "The device clock is outside this build's allowed range. Correct the system clock or open the latest hosted deployment.";
  }
  return "The local runtime guard blocked execution for this build.";
}

export function browserBucket(userAgent = globalThis.navigator?.userAgent || "") {
  const ua = String(userAgent || "");
  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /Chrome\//.test(ua)
      ? "Chrome"
      : /Firefox\//.test(ua)
        ? "Firefox"
        : /Safari\//.test(ua)
          ? "Safari"
          : "Browser";
  const os = /Windows/i.test(ua)
    ? "Windows"
    : /Android/i.test(ua)
      ? "Android"
      : /Mac OS X/i.test(ua)
        ? "macOS"
        : /Linux/i.test(ua)
          ? "Linux"
          : "OS";
  return `${browser} / ${os}`;
}

export function availableBrowserBackends(navigatorLike = globalThis.navigator) {
  const nav = navigatorLike || {};
  const names = [];
  if ("ml" in nav) names.push("WebNN");
  if ("gpu" in nav) names.push("WebGPU");
  names.push("WASM");
  return names;
}

export function backendProfileText(format, selected, navigatorLike = globalThis.navigator) {
  if (format === "onnx") return "ONNX Runtime Web";
  if (selected === "auto") {
    const labels = availableBrowserBackends(navigatorLike);
    return labels.length > 1 ? `Auto: ${labels.join(" / ")}` : "Auto: WASM only";
  }
  const labels = { wasm: "WASM", webgpu: "WebGPU", webnn: "WebNN" };
  return labels[selected] || selected;
}

export function backendCandidates(selected, format = "", navigatorLike = globalThis.navigator) {
  if (selected !== "auto") return [selected];
  if (format === "onnx") return ["wasm"];
  const nav = navigatorLike || {};
  const candidates = [];
  if ("ml" in nav) candidates.push("webnn");
  if ("gpu" in nav) candidates.push("webgpu");
  candidates.push("wasm");
  return candidates;
}

export function selectWasmCalibrationResult(results) {
  return (Array.isArray(results) ? results : []).find((result) => result?.ok && result.backend === "wasm") || null;
}

export function benchmarkErrorStatus(error, backend) {
  const message = error?.message || String(error);
  if (backend === "webgpu" && message.includes("Asyncify is not defined")) {
    return "WebGPU runtime path needs Asyncify/JSPI support in this browser build; WASM result is still valid.";
  }
  if (backend === "webgpu" && /webgpu|gpu|adapter|device/i.test(message)) {
    return `WebGPU runtime/device path failed: ${shortError(error)}`;
  }
  if (/not supported|unsupported/i.test(message)) {
    return `Runtime support gap: ${shortError(error)}`;
  }
  return shortError(error);
}

export function deleteTensors(value) {
  if (!value) return;
  if (Array.isArray(value)) {
    value.forEach((tensor) => tensor?.delete?.());
    return;
  }
  if (typeof value === "object") {
    Object.values(value).forEach((tensor) => tensor?.delete?.());
  }
}

export function runtimeDtype(dtype) {
  if (dtype === "FLOAT32") return "float32";
  if (dtype === "INT8") return "int8";
  if (dtype === "UINT8") return "uint8";
  if (dtype === "INT32") return "int32";
  return String(dtype || "").toLowerCase();
}

export function createFakeInputData(dtype, size, staticTensor, inputName = "input") {
  if (dtype === "float32") return new Float32Array(size);
  if (dtype === "int32") return new Int32Array(size);
  if (dtype === "uint8") {
    const data = new Uint8Array(size);
    data.fill(clampInteger(staticTensor?.zero_point_sample?.[0] ?? 0, 0, 255));
    return data;
  }
  if (dtype === "int8") {
    const data = new Int8Array(size);
    data.fill(clampInteger(staticTensor?.zero_point_sample?.[0] ?? 0, -128, 127));
    return data;
  }
  throw new Error(`Input ${inputName} dtype ${dtype} is not supported by this fake-input runner`);
}

export function createBenchmarkInputSpec(details, staticTensor, preparedInput = null, index = 0) {
  if (index === 0 && preparedInput) {
    const mismatch = preparedInputMismatchReason(details, staticTensor, preparedInput);
    if (mismatch) throw new Error(`Prepared input ${details?.name || index} ${mismatch}`);
    const shape = preparedInput.shape.map(Number);
    const elementCount = checkedRuntimeInputElementCount(details?.name || `input_${index}`, shape);
    if (!ArrayBuffer.isView(preparedInput.data) || preparedInput.data.length !== elementCount) {
      throw new Error(`Prepared input ${details?.name || index} data length ${preparedInput.data?.length ?? "unknown"} does not match shape element count ${elementCount}`);
    }
    return {
      data: cloneTypedArray(preparedInput.data),
      shape,
      basis: "prepared tensor",
    };
  }
  return createFakeInputSpec(details, staticTensor);
}

export function createFakeInputSpec(details, staticTensor) {
  const originalShape = Array.from(details.shape);
  const shape = resolveFakeInputShape(originalShape, staticTensor);
  const size = checkedRuntimeInputElementCount(details.name || "input", shape);
  return {
    data: createFakeInputData(details.dtype, size, staticTensor, details.name),
    shape,
    basis: "synthetic tensor",
  };
}

export function resolveFakeInputShape(runtimeShape, staticTensor) {
  const staticShape = Array.isArray(staticTensor?.shape) ? staticTensor.shape : [];
  return runtimeShape.map((dim, index) => {
    if (dim > 0) return dim;
    const staticDim = staticShape[index];
    if (staticDim > 0) return staticDim;
    if (index === 0) return 1;
    throw new Error(
      `Cannot resolve dynamic input dimension at axis ${index}; runtime=${runtimeShape.join("x")} static=${staticShape.join("x") || "unknown"}`,
    );
  });
}

export function runtimeInputMatchesPrepared(details, staticTensor, preparedInput) {
  return Boolean(preparedInput) && preparedInputMismatchReason(details, staticTensor, preparedInput) === "";
}

export function createBenchmarkOutputContract({ index = 0, name, artifactDtype, runtimeDtype: observedDtype, declaredShape = null, artifactShapeSignature = null, runtimeDeclaredShape = null, executedShape, data } = {}) {
  const artifactType = String(artifactDtype || "UNKNOWN").toLowerCase();
  const observedType = String(observedDtype || "").toLowerCase();
  if (!observedType || artifactType === "unknown" || artifactType !== observedType) throw new Error(`Runtime output ${name || index} dtype ${observedType || "UNKNOWN"} conflicts with artifact dtype ${artifactType}`);
  const executed = Array.from(executedShape || [], Number);
  const elements = checkedRuntimeInputElementCount(name || `output_${index}`, executed);
  if (!ArrayBuffer.isView(data) || data.length !== elements) throw new Error(`Runtime output ${name || index} data length ${data?.length ?? "unknown"} does not match shape element count ${elements}`);
  const declared = Array.isArray(declaredShape) ? declaredShape.map(Number) : null;
  const signature = Array.isArray(artifactShapeSignature) ? artifactShapeSignature.map(Number) : null;
  const runtimeShape = Array.isArray(runtimeDeclaredShape) ? runtimeDeclaredShape.map(Number) : null;
  for (const [label, shape] of [["artifact", signature || declared], ["runtime", runtimeShape]]) {
    if (!shape) continue;
    if (shape.length !== executed.length) throw new Error(`Runtime output ${name || index} ${label} rank ${shape.length} conflicts with executed rank ${executed.length}`);
    shape.forEach((dim, axis) => { if (dim > 0 && dim !== executed[axis]) throw new Error(`Runtime output ${name || index} executed dimension ${executed[axis]} at axis ${axis} conflicts with ${label} ${dim}`); });
  }
  return { output_index: index, output_name: name || `output_${index}`, artifact_dtype: artifactDtype || "UNKNOWN", runtime_dtype: observedDtype, declared_shape: declared, artifact_shape_signature: signature, runtime_declared_shape: runtimeShape, executed_shape: executed, element_count: elements, basis: "observed_runtime_output" };
}

function preparedInputMismatchReason(details, staticTensor, preparedInput) {
  if (details?.dtype !== preparedInput?.dtype) return `dtype ${preparedInput?.dtype || "UNKNOWN"} does not match runtime dtype ${details?.dtype || "UNKNOWN"}`;
  const runtimeShape = Array.from(details?.shape || []);
  const preparedShape = Array.isArray(preparedInput?.shape) ? preparedInput.shape.map(Number) : [];
  const staticShape = Array.isArray(staticTensor?.shape) ? staticTensor.shape.map(Number) : [];
  const signature = Array.isArray(staticTensor?.shape_signature) && staticTensor.shape_signature.length === runtimeShape.length
    ? staticTensor.shape_signature.map(Number) : null;
  const artifactShape = signature || staticShape;
  if (preparedShape.length !== runtimeShape.length) return `rank ${preparedShape.length} does not match runtime rank ${runtimeShape.length}`;
  for (let axis = 0; axis < preparedShape.length; axis += 1) {
    const actual = preparedShape[axis];
    if (!Number.isSafeInteger(actual) || actual <= 0) return `has invalid dimension ${actual} at axis ${axis}`;
    const runtimeRequired = runtimeShape[axis] > 0 ? runtimeShape[axis] : null;
    const artifactRequired = artifactShape[axis] > 0 ? artifactShape[axis] : null;
    if (runtimeRequired != null && artifactRequired != null && runtimeRequired !== artifactRequired) {
      return `runtime dimension ${runtimeRequired} conflicts with artifact contract ${artifactRequired} at axis ${axis}`;
    }
    if (runtimeRequired != null && actual !== runtimeRequired) return `dimension ${actual} at axis ${axis} does not match runtime ${runtimeRequired}`;
    if (artifactRequired != null && actual !== artifactRequired) return `dimension ${actual} at axis ${axis} does not match artifact contract ${artifactRequired}`;
  }
  return "";
}

function checkedRuntimeInputElementCount(name, shape) {
  let total = 1;
  for (const [axis, dim] of shape.entries()) {
    if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`Input ${name} has invalid dimension ${dim} at axis ${axis}`);
    total *= dim;
    if (!Number.isSafeInteger(total) || total > MAX_BROWSER_INPUT_ELEMENTS) {
      throw new Error(`Input ${name} requires ${total} elements; browser benchmark limit is ${MAX_BROWSER_INPUT_ELEMENTS}`);
    }
  }
  return total;
}

export async function measureBrowserBenchmarkPhases({
  execute,
  dispose = () => {},
  observeFinal = null,
  warmupRuns = 0,
  measuredRuns = 1,
  yieldEvery = 0,
  yieldControl = async () => {},
} = {}) {
  if (typeof execute !== "function") throw new Error("Benchmark execute callback is required");
  if (!Number.isInteger(warmupRuns) || warmupRuns < 0) throw new Error("Benchmark warmupRuns must be a non-negative integer");
  if (!Number.isInteger(measuredRuns) || measuredRuns < 1) throw new Error("Benchmark measuredRuns must be a positive integer");

  const runAndDispose = async (observe = null) => {
    let output = null;
    const started = performance.now();
    try {
      output = await execute();
      const durationMs = performance.now() - started;
      const observation = observe ? await observe(output) : null;
      return { durationMs, observation };
    } finally {
      await dispose(output);
    }
  };

  const cold = await runAndDispose();
  for (let index = 0; index < warmupRuns; index += 1) await runAndDispose();

  const timings = [];
  let finalObservation = null;
  for (let index = 0; index < measuredRuns; index += 1) {
    if (yieldEvery > 0 && index > 0 && index % yieldEvery === 0) await yieldControl();
    const measured = await runAndDispose(index === measuredRuns - 1 ? observeFinal : null);
    timings.push(measured.durationMs);
    if (index === measuredRuns - 1) finalObservation = measured.observation;
  }
  const stats = latencyStats(timings);
  return {
    timingMethod: BROWSER_BENCHMARK_TIMING_METHOD,
    statisticsMethod: BROWSER_BENCHMARK_STATISTICS_METHOD,
    noiseMethod: BROWSER_BENCHMARK_NOISE_METHOD,
    phaseCounts: { cold_first_runs: 1, warmup_runs: warmupRuns, measured_runs: measuredRuns },
    firstRunMs: cold.durationMs,
    timings,
    stats,
    steadyStats: { ...stats },
    noiseDiagnostics: benchmarkNoise(timings),
    finalObservation,
  };
}
