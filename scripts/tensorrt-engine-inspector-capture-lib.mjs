import { createHash } from "node:crypto";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { canonicalJson } from "../web/lib/report-utils.js";
import { parseStrictJson } from "../web/lib/strict-json.js";
import {
  TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
  tensorRtParserObservationIdentity,
} from "../web/lib/tensorrt-engine-inspector.js";
import { createTensorRtBuildProfile } from "../web/lib/tensorrt-static-preflight.js";

const SOURCES = new Set(["trtexec_exportLayerInfo", "IEngineInspector_getEngineInformation_kJSON"]);
const VERBOSITIES = new Set(["none", "layer_names_only", "detailed"]);

export async function captureTensorRtEngineInspector({
  modelPath,
  profile,
  enginePath,
  inspectorPath,
  parserObservation = null,
  outputPath = null,
  tensorrtVersion,
  cudaVersion,
  deviceId,
  deviceComputeCapability = null,
  deviceIdentity,
  source = "trtexec_exportLayerInfo",
  profilingVerbosity = "detailed",
  executionContextBound = false,
  toolBinaryPath,
  invocation,
} = {}) {
  const model = path.resolve(String(modelPath || ""));
  const engine = path.resolve(String(enginePath || ""));
  const inspectorFile = path.resolve(String(inspectorPath || ""));
  const tool = path.resolve(String(toolBinaryPath || ""));
  const normalizedProfile = createTensorRtBuildProfile(profile);
  if (!SOURCES.has(source) || !VERBOSITIES.has(profilingVerbosity)) {
    throw new Error("TensorRT inspector source or profiling verbosity is invalid.");
  }
  if (!String(tensorrtVersion || "").trim() || !String(cudaVersion || "").trim()
    || !String(deviceIdentity || "").trim() || !String(invocation || "").trim()) {
    throw new Error("TensorRT runtime, CUDA, device identity, and exact inspector invocation are required.");
  }
  if (Number(deviceId) !== normalizedProfile.device_id) {
    throw new Error("TensorRT inspector device ID differs from the build profile.");
  }
  const [modelInfo, engineInfo, inspectorInfo, toolInfo] = await Promise.all(
    [model, engine, inspectorFile, tool].map(async (file) => {
      const info = await stat(file);
      if (!info.isFile()) throw new Error(`TensorRT capture input is not a file: ${file}`);
      return info;
    }),
  );
  if (path.extname(model).toLowerCase() !== ".onnx") {
    throw new Error("TensorRT engine-inspector capture requires an ONNX artifact.");
  }
  const engineInformation = parseStrictJson(await readFile(inspectorFile, "utf8"), "TensorRT engine inspector JSON");
  if (!engineInformation || typeof engineInformation !== "object" || Array.isArray(engineInformation)) {
    throw new Error("TensorRT engine inspector output must be one JSON object.");
  }
  const hasBindings = Array.isArray(engineInformation.Bindings);
  const hasIoTensors = Array.isArray(engineInformation["I/O Tensors"]);
  const schemaGeneration = hasBindings && !hasIoTensors ? "tensorrt_10x"
    : hasIoTensors && !hasBindings ? "tensorrt_11x" : null;
  if (!schemaGeneration) throw new Error("TensorRT inspector output does not have one unambiguous 10.x/11.x schema generation.");
  const artifactSha256 = await sha256File(model);
  const engineSha256 = await sha256File(engine);
  const toolSha256 = await sha256File(tool);
  const sourceFileSha256 = await sha256File(inspectorFile);
  const value = {
    schema: TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
    artifact_sha256: artifactSha256,
    build_profile_sha256: normalizedProfile.profile_sha256,
    parser_observation_sha256: tensorRtParserObservationIdentity(parserObservation),
    engine: { sha256: engineSha256, byte_length: engineInfo.size },
    runtime: {
      tensorrt_version: String(tensorrtVersion),
      cuda_version: String(cudaVersion),
      device_id: Number(deviceId),
      device_compute_capability: deviceComputeCapability == null ? null : String(deviceComputeCapability),
      device_identity: String(deviceIdentity),
    },
    build_capture: {
      evidence_class: "DECLARED_BUILD_CAPTURE",
      binding_method: "user_supplied_model_profile_engine_and_inspector_files_with_independent_sha256",
      tool_name: path.basename(tool),
      tool_binary_sha256: toolSha256,
      invocation_sha256: sha256Text(String(invocation)),
      collector_source_set_sha256: null,
      model_input_sha256: artifactSha256,
      serialized_engine_sha256: engineSha256,
    },
    inspector: {
      source,
      profiling_verbosity: profilingVerbosity,
      schema_generation: schemaGeneration,
      execution_context_bound: Boolean(executionContextBound),
      source_file_sha256: sourceFileSha256,
      source_file_byte_length: inspectorInfo.size,
      canonical_json_sha256: sha256Text(canonicalJson(engineInformation)),
      engine_information: engineInformation,
    },
  };
  if (outputPath) await atomicWriteJson(path.resolve(outputPath), value);
  return value;
}

async function sha256File(filePath) {
  return createHash("sha256").update(await readFile(filePath)).digest("hex");
}

function sha256Text(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

async function atomicWriteJson(outputPath, value) {
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${canonicalJson(value)}\n`, "utf8");
  await rename(temporaryPath, outputPath);
}
