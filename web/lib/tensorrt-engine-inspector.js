import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { TENSORRT_SOURCE_METADATA } from "./tensorrt-source-metadata.js";

export const TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA = "deepbom.tensorrt_engine_inspector_evidence.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const GENERATIONS = new Set(["tensorrt_10x", "tensorrt_11x"]);
const VERBOSITIES = new Set(["none", "layer_names_only", "detailed"]);
const SOURCES = new Set(["trtexec_exportLayerInfo", "IEngineInspector_getEngineInformation_kJSON"]);
const CAPTURE_CLASSES = new Set(["DECLARED_BUILD_CAPTURE", "COLLECTOR_OBSERVED_BUILD_CAPTURE"]);
const MAX_LAYERS = 1_000_000;

export function tensorRtParserObservationIdentity(value) {
  if (!value) return null;
  const normalized = structuredCloneSafe(value);
  delete normalized.observed_node_count;
  delete normalized.unobserved_node_count;
  delete normalized.coverage_status;
  return sha256TextHex(canonicalJson(normalized));
}

export function validateTensorRtEngineInspectorEvidence(analysis, buildProfile, parserObservation, value) {
  if (String(analysis?.format || "").toLowerCase() !== "onnx" || !buildProfile?.profile_sha256) {
    throw new Error("TensorRT engine-inspector evidence requires an ONNX artifact and bound build profile.");
  }
  if (!value || value.schema !== TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA
    || !SHA256.test(String(value.artifact_sha256 || ""))
    || value.artifact_sha256 !== String(analysis.model_sha256 || "").toLowerCase()
    || value.build_profile_sha256 !== buildProfile.profile_sha256
    || !SHA256.test(String(value.engine?.sha256 || ""))
    || !Number.isSafeInteger(value.engine?.byte_length) || value.engine.byte_length <= 0
    || !String(value.runtime?.tensorrt_version || "").trim()
    || !String(value.runtime?.cuda_version || "").trim()
    || Number(value.runtime?.device_id) !== Number(buildProfile.device_id)
    || !String(value.runtime?.device_identity || "").trim()
    || !VERBOSITIES.has(value.inspector?.profiling_verbosity)
    || !SOURCES.has(value.inspector?.source)
    || !GENERATIONS.has(value.inspector?.schema_generation)
    || typeof value.inspector?.execution_context_bound !== "boolean"
    || !SHA256.test(String(value.inspector?.canonical_json_sha256 || ""))) {
    throw new Error("TensorRT engine-inspector identity or schema is invalid.");
  }
  if (!versionMatches(value.runtime.tensorrt_version, buildProfile.expected_tensorrt_version)
    || !versionMatches(value.runtime.cuda_version, buildProfile.expected_cuda_version)
    || (buildProfile.device_compute_capability != null
      && String(value.runtime.device_compute_capability || "") !== String(buildProfile.device_compute_capability))) {
    throw new Error("TensorRT engine-inspector runtime or device differs from the bound build profile.");
  }
  const expectedParserIdentity = tensorRtParserObservationIdentity(parserObservation);
  if ((value.parser_observation_sha256 ?? null) !== expectedParserIdentity) {
    throw new Error("TensorRT engine-inspector parser-observation binding is invalid.");
  }
  validateBuildCapture(value.build_capture, value);

  const engineInformation = value.inspector.engine_information;
  if (!engineInformation || typeof engineInformation !== "object" || Array.isArray(engineInformation)
    || sha256TextHex(canonicalJson(engineInformation)) !== value.inspector.canonical_json_sha256) {
    throw new Error("TensorRT engine-inspector canonical JSON digest does not reproduce.");
  }
  const normalized = normalizeEngineInformation(engineInformation, value.inspector);
  return {
    schema: TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
    status: normalized.layer_detail_status === "detailed"
      ? "engine_inspector_observed_detailed" : normalized.layer_detail_status === "names_only"
        ? "engine_inspector_observed_names_only" : "engine_inspector_observed_no_layer_detail",
    evidence_class: value.build_capture.evidence_class === "COLLECTOR_OBSERVED_BUILD_CAPTURE"
      ? "ENGINE_BUILD_AND_INSPECTOR_OBSERVED" : "DECLARED_BUILD_PROVENANCE/ENGINE_INSPECTOR_OBSERVED",
    artifact_sha256: value.artifact_sha256,
    build_profile_sha256: value.build_profile_sha256,
    parser_observation_sha256: value.parser_observation_sha256 ?? null,
    engine: { sha256: value.engine.sha256, byte_length: value.engine.byte_length },
    runtime: {
      tensorrt_version: String(value.runtime.tensorrt_version),
      cuda_version: String(value.runtime.cuda_version),
      device_id: Number(value.runtime.device_id),
      device_compute_capability: nullableText(value.runtime.device_compute_capability, 80),
      device_identity: requiredText(value.runtime.device_identity, 500, "device identity"),
    },
    build_capture: normalizeBuildCapture(value.build_capture),
    inspector: {
      source: value.inspector.source,
      profiling_verbosity: value.inspector.profiling_verbosity,
      schema_generation: value.inspector.schema_generation,
      execution_context_bound: value.inspector.execution_context_bound,
      source_file_sha256: nullableSha(value.inspector.source_file_sha256, "inspector source file SHA-256"),
      source_file_byte_length: nullablePositiveInteger(value.inspector.source_file_byte_length, "inspector source byte length"),
      canonical_json_sha256: value.inspector.canonical_json_sha256,
    },
    ...normalized,
    source_basis: {
      schema: TENSORRT_SOURCE_METADATA.schema,
      source_commit: TENSORRT_SOURCE_METADATA.tensorrt.source_commit,
      runtime_header_sha256: TENSORRT_SOURCE_METADATA.tensorrt.files
        .find((row) => row.path === "include/NvInferRuntime.h")?.sha256 || null,
      documentation: [
        "https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html",
        "https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-IEngineInspector.html",
      ],
    },
    artifact_engine_relation: value.build_capture.evidence_class === "COLLECTOR_OBSERVED_BUILD_CAPTURE"
      ? "collector_observed_same_process_binding" : "declared_build_capture_binding",
    interpretation_boundary: "The identity-bound inspector describes the optimized serialized engine after build, including emitted engine-layer names, tensor contracts, formats, precisions, and selected tactic identifiers when detailed profiling metadata exists. It is not original-ONNX-op assignment, parser-to-engine provenance beyond the stated capture class, tactic timing, kernel execution, physical transfer, memory allocation, latency, accuracy, or device fit. TensorRT may omit next-generation optimizer subgraphs from inspector output.",
  };
}

function normalizeEngineInformation(value, inspector) {
  const layers = value.Layers;
  if (!Array.isArray(layers) || layers.length > MAX_LAYERS) {
    throw new Error("TensorRT engine information must contain a bounded Layers array.");
  }
  const hasBindings = Array.isArray(value.Bindings);
  const hasIoTensors = Array.isArray(value["I/O Tensors"]);
  const detectedGeneration = hasIoTensors && !hasBindings ? "tensorrt_11x"
    : hasBindings && !hasIoTensors ? "tensorrt_10x" : null;
  if (!detectedGeneration || detectedGeneration !== inspector.schema_generation) {
    throw new Error("TensorRT engine-inspector 10.x/11.x schema generation is contradictory.");
  }
  const objectCount = layers.filter((row) => row && typeof row === "object" && !Array.isArray(row)).length;
  const stringCount = layers.filter((row) => typeof row === "string").length;
  if (objectCount + stringCount !== layers.length || (objectCount && stringCount)) {
    throw new Error("TensorRT engine layer rows must be uniformly strings or objects.");
  }
  const expectedDetail = inspector.profiling_verbosity === "detailed" ? "detailed"
    : inspector.profiling_verbosity === "layer_names_only" ? "names_only" : "none";
  const observedDetail = objectCount ? "detailed" : stringCount ? "names_only" : "none";
  if (observedDetail !== expectedDetail) {
    throw new Error("TensorRT engine layer detail contradicts the declared profiling verbosity.");
  }
  const normalizedLayers = objectCount ? layers.map(normalizeLayer) : layers.map((name, index) => ({
    layer_index: index,
    name: requiredText(name, 131072, `layer #${index} name`),
    layer_type: null,
    inputs: [],
    outputs: [],
    tactic_name: null,
    tactic_value: null,
    stream_id: null,
    source_node_name_tokens: sourceNameTokens(name),
    canonical_sha256: sha256TextHex(canonicalJson(name)),
  }));
  const ioTensors = hasIoTensors
    ? value["I/O Tensors"].map((row, index) => normalizeIoTensor(row, index))
    : value.Bindings.map((name, index) => ({
      tensor_index: index,
      name: requiredText(name, 16384, `binding #${index}`),
      io_mode: null,
      data_type: null,
      dimensions: null,
      location: null,
      is_shape_inference_io: null,
      profile_info: [],
    }));
  const layerTypes = countBy(normalizedLayers.map((row) => row.layer_type).filter(Boolean));
  const dataTypes = countBy(normalizedLayers.flatMap((row) => [...row.inputs, ...row.outputs]
    .map((tensor) => tensor.data_type).filter(Boolean)));
  const multiSource = normalizedLayers.filter((row) => row.source_node_name_tokens.length > 1).length;
  const tacticCount = normalizedLayers.filter((row) => row.tactic_name != null || row.tactic_value != null).length;
  const dynamicTensorCount = normalizedLayers.flatMap((row) => [...row.inputs, ...row.outputs])
    .filter((tensor) => tensor.dimensions?.includes(-1)).length;
  return {
    layer_detail_status: observedDetail,
    engine_layer_count: normalizedLayers.length,
    io_tensor_count: ioTensors.length,
    tactic_annotated_layer_count: tacticCount,
    multi_source_metadata_layer_count: multiSource,
    dynamic_dimension_tensor_count: dynamicTensorCount,
    layer_type_inventory: layerTypes,
    data_type_inventory: dataTypes,
    io_tensors: ioTensors,
    layers: normalizedLayers,
    source_mapping_status: multiSource || normalizedLayers.some((row) => row.source_node_name_tokens.length)
      ? "source_name_tokens_observed_original_op_identity_not_established" : "not_exposed",
    conservation: {
      status: normalizedLayers.length === layers.length ? "pass" : "fail",
      source_layer_count: layers.length,
      normalized_layer_count: normalizedLayers.length,
    },
  };
}

function normalizeLayer(row, index) {
  const name = requiredText(row.Name, 131072, `layer #${index} name`);
  const metadata = row.Metadata == null ? "" : Array.isArray(row.Metadata)
    ? row.Metadata.map((item) => String(item)).join("") : String(row.Metadata);
  return {
    layer_index: index,
    name,
    layer_type: nullableText(row.LayerType, 4096),
    parameter_type: nullableText(row.ParameterType, 4096),
    inputs: normalizeLayerTensors(row.Inputs, `layer #${index} input`),
    outputs: normalizeLayerTensors(row.Outputs, `layer #${index} output`),
    tactic_name: nullableText(row.TacticName, 131072),
    tactic_value: nullableText(row.TacticValue, 4096),
    stream_id: row.StreamId == null ? null : safeInteger(row.StreamId, `layer #${index} stream ID`),
    source_node_name_tokens: sourceNameTokens(`${name}${metadata}`),
    canonical_sha256: sha256TextHex(canonicalJson(row)),
  };
}

function normalizeLayerTensors(value, label) {
  if (value == null) return [];
  if (!Array.isArray(value) || value.length > 1_000_000) throw new Error(`${label} ledger is invalid.`);
  return value.map((row, index) => {
    if (!row || typeof row !== "object" || Array.isArray(row)) throw new Error(`${label} #${index} is invalid.`);
    const separateType = row.Datatype ?? row.DataType ?? null;
    return {
      name: requiredText(row.Name, 16384, `${label} #${index} name`),
      dimensions: normalizeDimensions(row.Dimensions, `${label} #${index}`),
      data_type: nullableText(separateType, 256),
      data_type_source: separateType == null ? "not_separately_exposed" : "separate_inspector_field",
      format: nullableText(row.Format, 16384),
      combined_format_datatype: nullableText(row["Format/Datatype"], 16384),
    };
  });
}

function normalizeIoTensor(row, index) {
  if (!row || typeof row !== "object" || Array.isArray(row)) throw new Error(`I/O tensor #${index} is invalid.`);
  const profiles = row.ProfileInfo == null ? [] : row.ProfileInfo;
  if (!Array.isArray(profiles) || profiles.length > 4096) throw new Error(`I/O tensor #${index} profile ledger is invalid.`);
  return {
    tensor_index: index,
    name: requiredText(row.Name, 16384, `I/O tensor #${index} name`),
    io_mode: nullableText(row.IOMode, 80),
    data_type: nullableText(row.Datatype ?? row.DataType, 256),
    dimensions: normalizeDimensions(row.Dimensions, `I/O tensor #${index}`),
    location: nullableText(row.Location, 256),
    is_shape_inference_io: typeof row.IsShapeInferenceIO === "boolean" ? row.IsShapeInferenceIO : null,
    profile_info: profiles.map((profile, profileIndex) => ({
      profile_index: profileIndex,
      min_shape: normalizeDimensions(profile?.MinShape, `I/O tensor #${index} min shape`, true),
      opt_shape: normalizeDimensions(profile?.OptShape, `I/O tensor #${index} opt shape`, true),
      max_shape: normalizeDimensions(profile?.MaxShape, `I/O tensor #${index} max shape`, true),
      format: nullableText(profile?.Format, 16384),
    })),
  };
}

function validateBuildCapture(capture, value) {
  if (!capture || !CAPTURE_CLASSES.has(capture.evidence_class)
    || !String(capture.binding_method || "").trim()
    || !SHA256.test(String(capture.tool_binary_sha256 || ""))
    || !SHA256.test(String(capture.invocation_sha256 || ""))) {
    throw new Error("TensorRT engine build-capture identity is invalid.");
  }
  if (capture.evidence_class === "COLLECTOR_OBSERVED_BUILD_CAPTURE"
    && (capture.model_input_sha256 !== value.artifact_sha256
      || capture.serialized_engine_sha256 !== value.engine.sha256
      || !SHA256.test(String(capture.collector_source_set_sha256 || "")))) {
    throw new Error("TensorRT observed engine build capture does not bind its model, engine, and collector source.");
  }
}

function normalizeBuildCapture(value) {
  return {
    evidence_class: value.evidence_class,
    binding_method: String(value.binding_method),
    tool_name: nullableText(value.tool_name, 256),
    tool_binary_sha256: value.tool_binary_sha256,
    invocation_sha256: value.invocation_sha256,
    collector_source_set_sha256: nullableSha(value.collector_source_set_sha256, "collector source-set SHA-256"),
    model_input_sha256: nullableSha(value.model_input_sha256, "build model SHA-256"),
    serialized_engine_sha256: nullableSha(value.serialized_engine_sha256, "build engine SHA-256"),
  };
}

function sourceNameTokens(value) {
  const rows = [];
  for (const match of String(value || "").matchAll(/\[ONNX Layer:\s*([^\]]+)\]/g)) {
    const name = match[1].trim();
    if (name && !rows.includes(name)) rows.push(name);
  }
  return rows;
}

function normalizeDimensions(value, label, nullable = false) {
  if (value == null && nullable) return null;
  if (!Array.isArray(value) || value.length > 32
    || value.some((item) => !Number.isSafeInteger(item) || item < -1)) {
    throw new Error(`${label} dimensions are invalid.`);
  }
  return value.map(Number);
}

function countBy(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  return [...counts].sort(([left], [right]) => left.localeCompare(right)).map(([name, count]) => ({ name, count }));
}

function versionMatches(observed, expected) {
  if (!expected) return true;
  const wanted = String(expected).trim();
  const actual = String(observed || "").trim();
  return actual === wanted || (actual.match(/\d+(?:\.\d+){1,3}/g) || []).includes(wanted);
}

function requiredText(value, maximum, label) {
  const output = String(value || "").trim();
  if (!output || output.length > maximum) throw new Error(`TensorRT ${label} is invalid.`);
  return output;
}

function nullableText(value, maximum) {
  if (value == null) return null;
  const output = String(value).trim();
  if (!output || output.length > maximum) throw new Error("TensorRT inspector text field is invalid.");
  return output;
}

function nullableSha(value, label) {
  if (value == null) return null;
  if (!SHA256.test(String(value))) throw new Error(`TensorRT ${label} is invalid.`);
  return String(value);
}

function nullablePositiveInteger(value, label) {
  if (value == null) return null;
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`TensorRT ${label} is invalid.`);
  return value;
}

function safeInteger(value, label) {
  if (!Number.isSafeInteger(value)) throw new Error(`TensorRT ${label} is invalid.`);
  return value;
}

function structuredCloneSafe(value) {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}
