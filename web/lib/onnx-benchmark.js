import * as ort from "onnxruntime-web/wasm";
import { sha256Hex, sha256TypedArrayListHex } from "./hash.js";
import { createBenchmarkOutputContract, measureBrowserBenchmarkPhases } from "./runtime.js";

const MAX_ONNX_BROWSER_INPUT_ELEMENTS = 100_000_000;

const ARTIFACT_TO_ORT_DTYPE = new Map([
  ["FLOAT32", "float32"],
  ["UINT8", "uint8"],
  ["INT8", "int8"],
  ["UINT16", "uint16"],
  ["INT16", "int16"],
  ["INT32", "int32"],
  ["INT64", "int64"],
  ["BOOL", "bool"],
  ["FLOAT16", "float16"],
  ["FLOAT64", "float64"],
  ["UINT32", "uint32"],
  ["UINT64", "uint64"],
]);

let onnxRuntimeConfigured = false;

export function configureOnnxRuntime(wasmPath) {
  if (onnxRuntimeConfigured) return;
  ort.env.logLevel = "warning";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = wasmPath;
  onnxRuntimeConfigured = true;
}

export async function benchmarkOnnxModel({
  modelBytes,
  analysis,
  backend,
  warmup,
  runs,
  preparedInput = null,
  externalDataFiles = [],
}, {
  createSession = (bytes, options) => ort.InferenceSession.create(bytes, options),
  createTensor = (dtype, data, shape) => new ort.Tensor(dtype, data, shape),
} = {}) {
  const externalDataContract = await buildOnnxExternalRuntimeContract(analysis, externalDataFiles);
  const sessionOptions = { executionProviders: [backend] };
  if (externalDataContract.files.length) {
    sessionOptions.externalData = externalDataContract.files.map((file) => ({ path: file.path, data: file.bytes }));
  }
  const compileStarted = performance.now();
  const session = await createSession(modelBytes, sessionOptions);
  const compileMs = performance.now() - compileStarted;
  const feeds = {};

  try {
    const inputContracts = [];
    for (const [index, name] of session.inputNames.entries()) {
      const staticTensor = findStaticTensor(analysis?.inputs, name, index, "input");
      const inputSpec = createOnnxBenchmarkInputSpec(staticTensor, preparedInput, index, session.inputMetadata?.[index] || null);
      feeds[name] = createTensor(inputSpec.runtime_dtype, inputSpec.data, inputSpec.shape);
      inputContracts.push(inputSpec.contract);
    }

    const phases = await measureBrowserBenchmarkPhases({
      execute: () => session.run(feeds),
      dispose: disposeOrtOutputs,
      warmupRuns: warmup,
      measuredRuns: runs,
      observeFinal: async (outputs) => observeOnnxOutputs(session, analysis, outputs),
    });

    return {
      compileMs,
      firstRunMs: phases.firstRunMs,
      timings: phases.timings,
      outputCount: session.outputNames.length,
      outputDigest: phases.finalObservation?.output_digest || "",
      outputContracts: phases.finalObservation?.output_contracts || [],
      stats: phases.stats,
      steadyStats: phases.steadyStats,
      timingMethod: phases.timingMethod,
      statisticsMethod: phases.statisticsMethod,
      noiseMethod: phases.noiseMethod,
      noiseDiagnostics: phases.noiseDiagnostics,
      phaseCounts: phases.phaseCounts,
      inputContracts,
      externalDataContract: publicExternalDataRuntimeContract(externalDataContract),
    };
  } finally {
    disposeOrtInputs(feeds);
    await session.release();
  }
}

export async function buildOnnxExternalRuntimeContract(analysis, suppliedFiles = []) {
  const evidence = analysis?.onnx_external_data || {};
  const tensorCount = Number(evidence.tensor_count || 0);
  if (!tensorCount) {
    return {
      schema: "deepbom.onnx_external_runtime_binding.v1",
      status: "not_applicable",
      evidence_class: "OBSERVED",
      tensor_count: 0,
      file_count: 0,
      total_file_bytes: 0,
      files: [],
      binding_method: "No TensorProto external_data references are present.",
    };
  }
  if (Number(evidence.verified_payload_count || 0) !== tensorCount || Number(evidence.payload_verification_failed_count || 0) > 0) {
    throw new Error(`ONNX external data runtime binding requires all ${tensorCount} tensor range(s) to pass static sidecar verification`);
  }
  const suppliedByPath = new Map((suppliedFiles || []).map((file) => [String(file?.path || ""), file]));
  const requiredRows = [...new Map((evidence.tensors || []).map((row) => [row.sidecar_path, row])).values()];
  const files = [];
  for (const row of requiredRows) {
    const file = suppliedByPath.get(String(row.sidecar_path || ""));
    if (!file || !(file.bytes instanceof Uint8Array)) throw new Error(`Verified ONNX external data file ${row.sidecar_path || "(missing)"} is unavailable for runtime session creation`);
    if (file.bytes.byteLength !== Number(row.sidecar_bytes || 0)) throw new Error(`ONNX external data file ${row.sidecar_path} changed size after static verification`);
    const sha256 = await sha256Hex(file.bytes);
    if (sha256 !== row.sidecar_sha256 || sha256 !== file.sha256) throw new Error(`ONNX external data file ${row.sidecar_path} changed after static verification`);
    files.push({ path: row.sidecar_path, bytes: file.bytes, byte_length: file.bytes.byteLength, sha256 });
  }
  return {
    schema: "deepbom.onnx_external_runtime_binding.v1",
    status: "bound_verified_external_data",
    evidence_class: "OBSERVED/DERIVED",
    tensor_count: tensorCount,
    file_count: files.length,
    total_file_bytes: files.reduce((total, file) => total + file.byte_length, 0),
    files,
    binding_method: "Every ORT externalData entry reuses the statically verified model-relative path and exact full sidecar bytes after an immediate SHA-256 recheck.",
  };
}

function publicExternalDataRuntimeContract(contract) {
  return {
    ...contract,
    files: contract.files.map(({ path, byte_length, sha256 }) => ({ path, byte_length, sha256 })),
  };
}

async function observeOnnxOutputs(session, analysis, outputs) {
  const arrays = [];
  const contracts = session.outputNames.map((name, index) => {
    const value = outputs?.[name];
    const artifact = findStaticTensor(analysis?.outputs, name, index, "output");
    const metadata = session.outputMetadata?.[index] || null;
    if (!value || !ArrayBuffer.isView(value.data)) throw new Error(`ONNX runtime output ${name} has no tensor data`);
    if (metadata?.name && metadata.name !== name) throw new Error(`ONNX runtime output metadata name ${metadata.name} conflicts with session output ${name}`);
    if (metadata && (metadata.isTensor !== true || normalizeOrtDtype(metadata.type) !== normalizeOrtDtype(value.type))) {
      throw new Error(`ONNX runtime output ${name} metadata conflicts with executed tensor dtype`);
    }
    const normalizedShape = metadata ? normalizeOnnxRuntimeShape(metadata.shape) : null;
    const runtimeShape = normalizedShape && (normalizedShape.length || !artifact.shape?.length) ? normalizedShape : null;
    arrays.push(value.data);
    return createBenchmarkOutputContract({
      index,
      name,
      artifactDtype: artifact.dtype,
      runtimeDtype: value.type,
      declaredShape: artifact.shape_declared ? artifact.shape : null,
      runtimeDeclaredShape: runtimeShape,
      executedShape: value.dims,
      data: value.data,
    });
  });
  if (arrays.length !== contracts.length) {
    throw new Error(`ONNX runtime returned ${arrays.length} tensor outputs for ${contracts.length} declared outputs`);
  }
  return {
    output_count: contracts.length,
    output_digest: await sha256TypedArrayListHex(arrays),
    output_contracts: contracts,
  };
}

function findStaticTensor(tensors, runtimeName, index, role) {
  const exact = tensors?.find((item) => item.name === runtimeName) || null;
  if (exact) return exact;
  const positional = tensors?.[index] || null;
  if (!positional) throw new Error(`ONNX runtime ${role} ${runtimeName || index} has no bound static ${role} contract`);
  if (runtimeName && positional.name && positional.name !== runtimeName) {
    throw new Error(`ONNX runtime ${role} ${runtimeName} conflicts with artifact ${role} ${positional.name} at index ${index}`);
  }
  return positional;
}

export function createOnnxBenchmarkInputSpec(staticTensor, preparedInput = null, index = 0, runtimeMetadata = null) {
  if (!staticTensor || !staticTensor.name) throw new Error(`ONNX input ${index} has no parsed tensor identity`);
  if (staticTensor.value_kind && staticTensor.value_kind !== "tensor") {
    throw new Error(`ONNX input ${staticTensor.name} declares ${staticTensor.value_kind}; the browser benchmark accepts dense tensor inputs only`);
  }
  const artifactRuntimeDtype = onnxToOrtType(staticTensor.dtype || "UNKNOWN");
  if (runtimeMetadata?.name && runtimeMetadata.name !== staticTensor.name) {
    throw new Error(`ONNX runtime input metadata name ${runtimeMetadata.name} conflicts with artifact input ${staticTensor.name}`);
  }
  if (runtimeMetadata && runtimeMetadata.isTensor !== true) throw new Error(`ONNX runtime input ${staticTensor.name} is not a tensor`);
  const runtimeDtype = runtimeMetadata ? normalizeOrtDtype(runtimeMetadata.type) : artifactRuntimeDtype;
  if (runtimeDtype !== artifactRuntimeDtype) {
    throw new Error(`ONNX runtime input ${staticTensor.name} dtype ${runtimeDtype} conflicts with artifact dtype ${artifactRuntimeDtype}`);
  }
  const declaredShape = Array.isArray(staticTensor.shape) ? staticTensor.shape.map(Number) : null;
  if (!declaredShape || (!declaredShape.length && staticTensor.shape_declared !== true)) {
    throw new Error(`ONNX input ${staticTensor.name} has no declared shape; bind an explicit input contract before benchmarking`);
  }
  const normalizedRuntimeShape = runtimeMetadata ? normalizeOnnxRuntimeShape(runtimeMetadata.shape) : null;
  const runtimeDeclaredShape = normalizedRuntimeShape && (normalizedRuntimeShape.length || !declaredShape.length) ? normalizedRuntimeShape : null;
  validateOnnxShapeContracts(staticTensor.name, declaredShape, runtimeDeclaredShape);

  let shape;
  let data;
  let basis;
  if (index === 0 && preparedInput) {
    const preparedDtype = normalizeOrtDtype(preparedInput.dtype);
    if (preparedDtype !== runtimeDtype) {
      throw new Error(`Prepared ONNX input ${staticTensor.name} dtype ${preparedInput.dtype || "UNKNOWN"} does not match ${runtimeDtype}`);
    }
    shape = Array.isArray(preparedInput.shape) ? preparedInput.shape.map(Number) : [];
    validatePreparedOnnxShape(staticTensor.name, declaredShape, runtimeDeclaredShape, shape);
    data = cloneTypedArray(preparedInput.data);
    basis = "prepared_tensor";
  } else {
    shape = declaredShape.map((dim, axis) => {
      if (runtimeDeclaredShape?.[axis] > 0) return runtimeDeclaredShape[axis];
      if (Number.isSafeInteger(dim) && dim > 0) return dim;
      if (axis === 0) return 1;
      throw new Error(`Cannot resolve dynamic ONNX input ${staticTensor.name} dimension at axis ${axis}; declared=${declaredShape.join("x") || "scalar"}`);
    });
    const elementCount = checkedOnnxInputElementCount(staticTensor.name, shape);
    data = createOnnxInputData(runtimeDtype, elementCount, syntheticOnnxFillValue(staticTensor, runtimeDtype), staticTensor.name);
    basis = "synthetic_zero_or_zero_point_tensor";
  }

  const elementCount = checkedOnnxInputElementCount(staticTensor.name, shape);
  if (!ArrayBuffer.isView(data) || data.length !== elementCount) {
    throw new Error(`ONNX input ${staticTensor.name} data length ${data?.length ?? "unknown"} does not match shape element count ${elementCount}`);
  }
  const fillValue = basis.startsWith("synthetic") && data.length ? Number(data[0]) : null;
  return {
    runtime_dtype: runtimeDtype,
    shape,
    data,
    contract: {
      input_index: index,
      input_name: staticTensor.name,
      artifact_dtype: staticTensor.dtype,
      runtime_dtype: runtimeDtype,
      declared_shape: declaredShape,
      artifact_shape_signature: null,
      runtime_declared_shape: runtimeDeclaredShape,
      executed_shape: shape,
      element_count: elementCount,
      basis,
      synthetic_fill_value: fillValue,
    },
  };
}

function validatePreparedOnnxShape(name, declaredShape, runtimeShape, preparedShape) {
  if (preparedShape.length !== declaredShape.length) {
    throw new Error(`Prepared ONNX input ${name} rank ${preparedShape.length} does not match declared rank ${declaredShape.length}`);
  }
  preparedShape.forEach((dim, axis) => {
    if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`Prepared ONNX input ${name} has invalid dimension ${dim} at axis ${axis}`);
    if (declaredShape[axis] > 0 && dim !== declaredShape[axis]) {
      throw new Error(`Prepared ONNX input ${name} dimension ${dim} at axis ${axis} does not match declared ${declaredShape[axis]}`);
    }
    if (runtimeShape?.[axis] > 0 && dim !== runtimeShape[axis]) {
      throw new Error(`Prepared ONNX input ${name} dimension ${dim} at axis ${axis} does not match runtime ${runtimeShape[axis]}`);
    }
  });
}

function normalizeOnnxRuntimeShape(shape) {
  return Array.isArray(shape) ? shape.map((dim) => Number.isSafeInteger(dim) && dim > 0 ? dim : -1) : [];
}

function validateOnnxShapeContracts(name, artifactShape, runtimeShape) {
  if (!runtimeShape) return;
  if (runtimeShape.length !== artifactShape.length) {
    throw new Error(`ONNX runtime input ${name} rank ${runtimeShape.length} conflicts with artifact rank ${artifactShape.length}`);
  }
  runtimeShape.forEach((dim, axis) => {
    if (dim > 0 && artifactShape[axis] > 0 && dim !== artifactShape[axis]) {
      throw new Error(`ONNX runtime input ${name} dimension ${dim} at axis ${axis} conflicts with artifact ${artifactShape[axis]}`);
    }
  });
}

function checkedOnnxInputElementCount(name, shape) {
  let total = 1;
  for (const [axis, dim] of shape.entries()) {
    if (!Number.isSafeInteger(dim) || dim <= 0) throw new Error(`ONNX input ${name} has invalid dimension ${dim} at axis ${axis}`);
    total *= dim;
    if (!Number.isSafeInteger(total) || total > MAX_ONNX_BROWSER_INPUT_ELEMENTS) {
      throw new Error(`ONNX input ${name} requires ${total} elements; browser benchmark limit is ${MAX_ONNX_BROWSER_INPUT_ELEMENTS}`);
    }
  }
  return total;
}

function normalizeOrtDtype(dtype) {
  const normalized = String(dtype || "").toLowerCase();
  return ARTIFACT_TO_ORT_DTYPE.get(normalized.toUpperCase()) || normalized;
}

function syntheticOnnxFillValue(staticTensor, runtimeDtype) {
  if (!["uint8", "int8", "uint16", "int16", "int32", "uint32", "int64", "uint64"].includes(runtimeDtype)) return 0;
  const value = Number(staticTensor?.zero_point_sample?.[0] ?? 0);
  return Number.isFinite(value) && Number.isInteger(value) ? value : 0;
}

function createOnnxInputData(dtype, size, fillValue, name) {
  const constructors = {
    float32: Float32Array,
    float16: Uint16Array,
    uint8: Uint8Array,
    int8: Int8Array,
    uint16: Uint16Array,
    int16: Int16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    bool: Uint8Array,
    float64: Float64Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
  };
  const Constructor = constructors[dtype];
  if (!Constructor) throw new Error(`ONNX input ${name} runtime dtype ${dtype} is not supported by the synthetic-input runner`);
  const data = new Constructor(size);
  if (fillValue !== 0) data.fill(dtype === "int64" || dtype === "uint64" ? BigInt(fillValue) : fillValue);
  return data;
}

function disposeOrtInputs(feeds) {
  for (const value of Object.values(feeds)) value?.dispose?.();
}

function disposeOrtOutputs(outputs) {
  for (const value of Object.values(outputs || {})) value?.dispose?.();
}

function onnxToOrtType(dtype) {
  if (dtype === "BFLOAT16") {
    throw new Error("Unsupported ONNX tensor dtype for browser benchmark input: BFLOAT16 cannot be relabeled as FLOAT16");
  }
  const found = ARTIFACT_TO_ORT_DTYPE.get(dtype);
  if (!found) throw new Error(`Unsupported ONNX tensor dtype for fake input: ${dtype}`);
  return found;
}

function cloneTypedArray(value) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView) throw new Error("Prepared ONNX input data must be a typed array");
  return new value.constructor(value);
}
