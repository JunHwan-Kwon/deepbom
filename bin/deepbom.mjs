#!/usr/bin/env node

import { createHash } from "node:crypto";
import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { detectModelFormat } from "../web/lib/model-file.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildTensorRtStaticPreflight } from "../web/lib/tensorrt-static-preflight.js";
import { buildOnDeviceLlmContract } from "../web/lib/on-device-llm-contract.js";
import { buildLlmStaticMemoryPlacement } from "../web/lib/llm-static-memory-placement.js";
import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { identifyCliFile, loadCliInput, loadExecuTorchExternalData, loadOnnxExternalData, readCliFileBytes, verifyBundleSnapshot } from "./deepbom-input.mjs";
import {
  buildCanonicalGatedDecoderProjection,
  buildKvStateProjection,
} from "../web/lib/transformer-architecture-projection.js";
import {
  compareLlmMemoryCapacity,
  LLM_STATIC_RESIDENCY_ASSUMPTION,
} from "../web/lib/llm-memory-feasibility.js";
import {
  analyze_tflite_for_target,
  initSync as initTfliteWasm,
} from "../pkg/tflite_wasm_audit.js";

const DEFAULT_TARGET = "android_mid_a55";
const VERSION = typeof __DEEPBOM_RELEASE_VERSION__ === "string" ? __DEEPBOM_RELEASE_VERSION__ : "1.94.2";
const EXPECTED_TFLITE_WASM_SHA256 = typeof __DEEPBOM_TFLITE_WASM_SHA256__ === "string" ? __DEEPBOM_TFLITE_WASM_SHA256__ : "";

async function main(argv) {
  const parsed = parseArguments(argv);
  if (parsed.help) return printHelp();
  if (parsed.version) return process.stdout.write(`${VERSION}\n`);
  if (!parsed.input) throw new Error("An artifact path is required.");

  const inputPath = path.resolve(parsed.input);
  const input = await loadCliInput(inputPath);
  const filename = input.filename;
  const detectedFormat = input.kind === "file" ? detectModelFormat(filename, input.prefix) : "package";
  if (parsed.externalDataRoot && input.kind !== "file") throw new Error("--external-data-dir applies only to an ONNX or ExecuTorch PTE file.");
  if (input.kind === "file" && ["unsupported", "pytorch_pickle"].includes(detectedFormat)) {
    throw new Error(`Unsupported artifact format: ${detectedFormat}`);
  }
  let analysis = await analyzeArtifact({ input, filename, format: detectedFormat, target: parsed.target, externalDataRoot: parsed.externalDataRoot });
  const format = String(analysis.format || detectedFormat).toLowerCase();
  if (["unsupported", "pytorch_pickle"].includes(format)) {
    throw new Error(`Unsupported artifact format: ${format}`);
  }
  if (parsed.externalDataRoot && !["onnx", "executorch"].includes(format)) {
    throw new Error("--external-data-dir applies only to an ONNX or ExecuTorch PTE file.");
  }
  if (parsed.command === "gguf" && format !== "gguf") {
    throw new Error(`The gguf command requires a GGUF artifact, received ${format}.`);
  }
  if ((parsed.context || parsed.batch !== 1 || parsed.stateBits !== 16 || parsed.memoryMib) && format !== "gguf") {
    throw new Error("--context, --batch, --state-bits, and --memory-mib are valid only for GGUF artifacts.");
  }
  if (!parsed.context && (parsed.batch !== 1 || parsed.stateBits !== 16 || parsed.memoryMib)) {
    throw new Error("--batch, --state-bits, and --memory-mib require --context.");
  }

  if (parsed.context) analysis.cli_context_scenario = buildGgufContextScenario(analysis, parsed.context, {
    batchSize: parsed.batch,
    stateBits: parsed.stateBits,
    memoryMib: parsed.memoryMib,
  });
  const artifact = input.kind === "file"
    ? { ...(await identifyCliFile(input)), format }
    : packageIdentity(analysis, format, filename);
  enforceArtifactIdentity(analysis, artifact);
  const artifactSha256 = artifact.sha256;
  if (["onnx", "tflite"].includes(format)) analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
  if (parsed.llmMemoryProfile && !["gguf", "safetensors"].includes(format)) {
    throw new Error("--llm-memory-profile applies only to GGUF or SafeTensors artifacts with an exact layer-storage contract.");
  }
  if ((parsed.tensorrtLlmConfig || parsed.tensorrtLlmBinding) && format !== "safetensors") {
    throw new Error("--tensorrt-llm-config and --tensorrt-llm-binding apply only to a SafeTensors artifact in this CLI surface.");
  }
  if (parsed.tensorrtLlmConfig) {
    const sidecars = { tensorrt_llm_engine_config: await readJsonSidecar(parsed.tensorrtLlmConfig, "tensorrt_llm_engine_config") };
    if (parsed.tensorrtLlmBinding) sidecars.tensorrt_llm_binding = await readJsonSidecar(parsed.tensorrtLlmBinding, "tensorrt_llm_binding");
    analysis.on_device_llm = buildOnDeviceLlmContract(analysis, { sidecars });
  } else if (parsed.tensorrtLlmBinding) {
    throw new Error("--tensorrt-llm-binding requires --tensorrt-llm-config.");
  }
  if (parsed.llmMemoryProfile) {
    if (!analysis.on_device_llm) analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
    const sidecar = await readJsonSidecar(parsed.llmMemoryProfile, "llm_static_memory_profile");
    analysis.on_device_llm.static_memory_placement = buildLlmStaticMemoryPlacement(analysis.on_device_llm, analysis, sidecar);
  }
  if ((parsed.tensorrtProfile || parsed.tensorrtParserEvidence) && format !== "onnx") {
    throw new Error("--tensorrt-profile and --tensorrt-parser-evidence apply only to ONNX artifacts.");
  }
  if (format === "onnx") {
    const parserEvidence = parsed.tensorrtParserEvidence ? await readJsonDocument(parsed.tensorrtParserEvidence) : null;
    const buildProfile = parsed.tensorrtProfile
      ? await readJsonDocument(parsed.tensorrtProfile)
      : parserEvidence?.build_profile || null;
    analysis.tensorrt_static_preflight = buildTensorRtStaticPreflight(analysis, buildProfile, parserEvidence);
  }

  const document = parsed.outputFormat === "cyclonedx"
    ? buildMlBomDocument(analysis, {
        hash: artifactSha256,
        fileSizeBytes: artifact.size,
        timestamp: parsed.timestamp || new Date().toISOString(),
        ...(analysis.target_profile ? {
          target: analysis.target_profile,
          targetId: analysis.target_profile.id,
        } : {}),
      })
    : analysis;
  const text = `${JSON.stringify(document, bigintReplacer, parsed.compact ? 0 : 2)}\n`;
  if (parsed.output) await writeFile(path.resolve(parsed.output), text, "utf8");
  else process.stdout.write(text);
}

async function analyzeArtifact({ input, filename, format, target, externalDataRoot }) {
  if (input.kind === "bundle") {
    const parsed = await readArtifactBundle(input.files);
    await verifyBundleSnapshot(input.files, parsed.analysis.artifact_bundle?.files);
    return parsed.analysis;
  }
  if (format === "tflite") {
    const bytes = await readCliFileBytes(input);
    const wasm = await readFile(await resolveRuntimeAsset("tflite_wasm_audit_bg.wasm"));
    if (EXPECTED_TFLITE_WASM_SHA256 && createHash("sha256").update(wasm).digest("hex") !== EXPECTED_TFLITE_WASM_SHA256) {
      throw new Error("Packaged TFLite WASM failed its release SHA-256 check.");
    }
    initTfliteWasm({ module: wasm });
    return analyze_tflite_for_target(bytes, filename, target || DEFAULT_TARGET);
  }
  if (format === "onnx") {
    const bytes = await readCliFileBytes(input);
    const structural = analyzeOnnxModel(bytes, filename);
    const externalDataFiles = await loadOnnxExternalData(input.path, structural, externalDataRoot);
    return externalDataFiles.length ? analyzeOnnxModel(bytes, filename, null, { externalDataFiles }) : structural;
  }

  if (format === "executorch") {
    const bytes = await readCliFileBytes(input);
    const structural = analyzeExecuTorchModel(bytes, filename);
    if (structural.executorch_container !== "pte") {
      if (externalDataRoot) throw new Error("--external-data-dir is not applicable to a standalone ExecuTorch PTD file.");
      return structural;
    }
    if (!structural.executorch_program?.external_tensor_data?.required_name_count) return structural;
    const externalDataFiles = await loadExecuTorchExternalData(input.path, externalDataRoot);
    return externalDataFiles.length ? analyzeExecuTorchModel(bytes, filename, { externalDataFiles }) : structural;
  }

  if (format === "gguf" || format === "safetensors") {
    return (await readMetadataModelFile(input.file, format)).analysis;
  }
  if (format === "coreml") return (await readCoreMlModelFile(input.file)).analysis;
  throw new Error(`No analyzer is registered for ${format}.`);
}

function packageIdentity(analysis, format, filename) {
  const sha256 = String(analysis?.model_sha256 || "").toLowerCase();
  const size = Number(analysis?.file_size_bytes);
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error("Package analyzer did not emit a canonical SHA-256 identity.");
  if (!Number.isSafeInteger(size) || size <= 0) throw new Error("Package analyzer did not emit a valid aggregate byte size.");
  return { filename, format, size, sha256 };
}

async function resolveRuntimeAsset(filename) {
  const roots = [
    process.env.DEEPBOM_RUNTIME_ASSET_DIR,
    process.argv[1] ? path.resolve(path.dirname(process.argv[1]), "../pkg") : "",
    path.resolve(path.dirname(process.execPath), "pkg"),
    path.resolve(path.dirname(process.execPath), "deepbom-assets"),
  ].filter(Boolean);
  const candidates = [...new Set(roots.map((root) => path.join(root, filename)))];
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Continue through the deterministic asset search order.
    }
  }
  throw new Error(`Required runtime asset ${filename} was not found. Checked: ${candidates.join(", ")}`);
}

function enforceArtifactIdentity(analysis, artifact) {
  if (!analysis || typeof analysis !== "object") throw new Error("Analyzer returned no document.");
  const observedFormat = String(analysis.format || "").toLowerCase();
  if (observedFormat && observedFormat !== artifact.format) {
    throw new Error(`Analyzer format mismatch: detected ${artifact.format}, returned ${observedFormat}.`);
  }
  const observedSha = String(analysis.model_sha256 || "").toLowerCase();
  if (observedSha && observedSha !== artifact.sha256) {
    throw new Error("Analyzer artifact SHA-256 differs from the CLI input SHA-256.");
  }
  analysis.format = artifact.format;
  analysis.filename = analysis.filename || artifact.filename;
  analysis.file_size_bytes = analysis.file_size_bytes ?? artifact.size;
  analysis.model_sha256 = artifact.sha256;
}

function buildGgufContextScenario(analysis, contextLength, { batchSize = 1, stateBits = 16, memoryMib = null } = {}) {
  const contract = analysis.gguf?.semantic_contract;
  if (!contract || typeof contract !== "object") {
    return {
      schema: "deepbom.gguf_cli_context_scenario.v1",
      status: "NOT_ASSESSABLE",
      evidence_class: "DECLARED/DERIVED_SCENARIO",
      context_length: contextLength,
      reason: "GGUF semantic contract is unavailable.",
    };
  }
  const keyHeadWidth = contract.attention_key_length || contract.derived_attention_head_width;
  const valueHeadWidth = contract.attention_value_length || contract.derived_attention_head_width;
  const kvReady = [contract.block_count, contract.attention_head_count_kv, keyHeadWidth, valueHeadWidth]
    .every(isPositiveSafeInteger);
  const decoderReady = kvReady && [
    contract.tokenizer?.vocabulary_count,
    contract.embedding_length,
    contract.feed_forward_length,
    contract.attention_head_count,
  ].every(isPositiveSafeInteger) && keyHeadWidth === valueHeadWidth;
  const kvStateProjection = kvReady ? buildKvStateProjection({
    layerCount: contract.block_count,
    kvHeadCount: contract.attention_head_count_kv,
    keyHeadWidth,
    valueHeadWidth,
    contextLength,
  }) : null;
  const stateElements = exactInteger(kvStateProjection?.elements_at_context_batch_one);
  const serializedWeightBytes = exactInteger(analysis?.tensor_storage_summary?.byte_length_decimal);
  const stateBytes = stateElements == null ? null : stateElements * BigInt(batchSize) * BigInt(stateBits / 8);
  const staticLowerBound = serializedWeightBytes == null || stateBytes == null ? null : serializedWeightBytes + stateBytes;
  const capacityBytes = memoryMib == null ? null : BigInt(memoryMib) * 1024n * 1024n;
  const capacityComparison = compareLlmMemoryCapacity(staticLowerBound, capacityBytes);
  return {
    schema: "deepbom.gguf_cli_context_scenario.v1",
    status: kvReady ? "ASSESSED" : "NOT_ASSESSABLE",
    evidence_class: "DECLARED/DERIVED_SCENARIO",
    context_length: contextLength,
    batch_size: batchSize,
    state_storage_bits: stateBits,
    context_source: "cli_argument",
    serialized_context_length: contract.context_length ?? null,
    kv_state_projection: kvStateProjection,
    compute_projection: decoderReady ? buildCanonicalGatedDecoderProjection({
      vocabularySize: contract.tokenizer.vocabulary_count,
      hiddenSize: contract.embedding_length,
      intermediateSize: contract.feed_forward_length,
      layerCount: contract.block_count,
      attentionHeadCount: contract.attention_head_count,
      kvHeadCount: contract.attention_head_count_kv,
      headWidth: keyHeadWidth,
      contextLength,
    }) : null,
    compute_projection_status: decoderReady ? "assessed_registered_canonical_decoder_scenario"
      : "not_assessable_incomplete_or_nonuniform_attention_metadata",
    memory_feasibility: {
      schema: "deepbom.gguf_cli_memory_scenario.v1",
      status: capacityComparison.status,
      evidence_class: staticLowerBound == null ? "NOT_ASSESSABLE" : "OBSERVED_STORAGE_PLUS_DERIVED_CONDITIONAL_STATE",
      capacity_scope: "single_aggregate_primary_memory_budget",
      residency_assumption: LLM_STATIC_RESIDENCY_ASSUMPTION,
      serialized_weight_floor_bytes: serializedWeightBytes == null ? null : exact(serializedWeightBytes),
      logical_kv_state_bytes: stateBytes == null ? null : exact(stateBytes),
      static_lower_bound_bytes: staticLowerBound == null ? null : exact(staticLowerBound),
      declared_capacity_bytes: capacityBytes == null ? null : exact(capacityBytes),
      deficit_bytes: capacityComparison.deficit_bytes,
      headroom_after_lower_bound_bytes: capacityComparison.headroom_after_lower_bound_bytes,
      fit_claim: capacityComparison.fit_claim,
      boundary: "Under the emitted simultaneous-residency assumption, exact serialized tensor bytes plus logical KV-state bytes form a conditional resident-set lower bound. A declared capacity at or above that bound does not establish fit because weight paging/offload policy, runtime-expanded weights, packing, replicas, graph/workspace memory, allocator alignment, backend-private allocations, application memory, and operating-system reserve are unbound.",
    },
    boundary: "The CLI context is a user-declared scenario. It does not replace the serialized GGUF context contract or establish runtime allocation, backend lowering, kernel choice, latency, or device residency.",
  };
}

function exactInteger(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

function isPositiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function parseArguments(argv) {
  const values = [...argv];
  const first = values[0] || "";
  if (["-h", "--help", "help"].includes(first)) return { help: true };
  if (["-v", "--version", "version"].includes(first)) return { version: true };
  const command = ["audit", "gguf"].includes(first) ? values.shift() : "audit";
  const parsed = {
    command,
    input: "",
    target: DEFAULT_TARGET,
    outputFormat: "analysis",
    output: "",
    timestamp: "",
    context: null,
    batch: 1,
    stateBits: 16,
    memoryMib: null,
    tensorrtProfile: "",
    tensorrtParserEvidence: "",
    tensorrtLlmConfig: "",
    tensorrtLlmBinding: "",
    llmMemoryProfile: "",
    externalDataRoot: "",
    compact: false,
  };
  while (values.length) {
    const token = values.shift();
    if (token === "--target") parsed.target = requiredValue(values, token);
    else if (token === "--context") parsed.context = positiveInteger(requiredValue(values, token), token);
    else if (token === "--batch") parsed.batch = positiveInteger(requiredValue(values, token), token);
    else if (token === "--state-bits") parsed.stateBits = stateBits(requiredValue(values, token));
    else if (token === "--memory-mib") parsed.memoryMib = positiveInteger(requiredValue(values, token), token);
    else if (token === "--tensorrt-profile") parsed.tensorrtProfile = requiredValue(values, token);
    else if (token === "--tensorrt-parser-evidence") parsed.tensorrtParserEvidence = requiredValue(values, token);
    else if (token === "--tensorrt-llm-config") parsed.tensorrtLlmConfig = requiredValue(values, token);
    else if (token === "--tensorrt-llm-binding") parsed.tensorrtLlmBinding = requiredValue(values, token);
    else if (token === "--llm-memory-profile") parsed.llmMemoryProfile = requiredValue(values, token);
    else if (token === "--external-data-dir") parsed.externalDataRoot = requiredValue(values, token);
    else if (token === "--format") parsed.outputFormat = requiredValue(values, token).toLowerCase();
    else if (token === "--output" || token === "-o") parsed.output = requiredValue(values, token);
    else if (token === "--timestamp") parsed.timestamp = normalizeTimestamp(requiredValue(values, token));
    else if (token === "--compact") parsed.compact = true;
    else if (token === "--help" || token === "-h") parsed.help = true;
    else if (token === "--version" || token === "-v") parsed.version = true;
    else if (token.startsWith("-")) throw new Error(`Unknown option: ${token}`);
    else if (!parsed.input) parsed.input = token;
    else throw new Error(`Unexpected positional argument: ${token}`);
  }
  if (!new Set(["analysis", "cyclonedx"]).has(parsed.outputFormat)) {
    throw new Error("--format must be analysis or cyclonedx.");
  }
  return parsed;
}

function requiredValue(values, option) {
  const value = values.shift();
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value.`);
  return value;
}

function normalizeTimestamp(value) {
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds)) throw new Error("--timestamp must be an ISO-8601 timestamp.");
  return new Date(milliseconds).toISOString();
}

function positiveInteger(value, option) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${option} must be a positive safe integer.`);
  return parsed;
}

function stateBits(value) {
  const parsed = positiveInteger(value, "--state-bits");
  if (![8, 16, 32].includes(parsed)) throw new Error("--state-bits must be 8, 16, or 32.");
  return parsed;
}

function bigintReplacer(_key, value) {
  return typeof value === "bigint" ? value.toString() : value;
}

async function readJsonDocument(filePath) {
  const resolved = path.resolve(filePath);
  try {
    return JSON.parse(await readFile(resolved, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read JSON document ${resolved}: ${error?.message || error}`);
  }
}

async function readJsonSidecar(filePath, role) {
  const resolved = path.resolve(filePath);
  try {
    const bytes = await readFile(resolved);
    return {
      role,
      path: path.basename(resolved),
      byte_length: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      document: JSON.parse(bytes.toString("utf8")),
    };
  } catch (error) {
    throw new Error(`Cannot read ${role} JSON ${resolved}: ${error?.message || error}`);
  }
}

function printHelp() {
  process.stdout.write(`DEEPBOM ${VERSION}\n\nUsage:\n  deepbom audit <artifact-or-package> [options]\n  deepbom gguf <artifact.gguf> [options]\n\nSupported inputs:\n  .tflite, .onnx, .gguf, .safetensors, .mlmodel, .pte, .ptd\n  .mlpackage directories and sharded SafeTensors repository directories\n\nOptions:\n  --target <id>          TFLite target profile (default: ${DEFAULT_TARGET})\n  --external-data-dir <directory>\n                          Resolve ONNX external_data or ExecuTorch PTD sidecars from this directory\n  --context <tokens>     GGUF context-length scenario\n  --batch <count>        GGUF scenario batch size (default: 1)\n  --state-bits <bits>    GGUF KV-state width: 8, 16, or 32 (default: 16)\n  --memory-mib <MiB>     Compare the static lower bound with a declared capacity\n  --tensorrt-profile <json>\n                          Bind an ONNX TensorRT native/ORT EP build profile\n  --tensorrt-parser-evidence <json>\n                          Import identity-bound TensorRT parser/build evidence\n  --tensorrt-llm-config <json>\n                          Assess a TensorRT-LLM engine config with SafeTensors\n  --tensorrt-llm-binding <json>\n                          Bind that config to model-source/component digests\n  --llm-memory-profile <json>\n                          Evaluate serialized layer/state lower bounds against declared CPU and accelerator pools\n  --format <kind>        analysis or cyclonedx (default: analysis)\n  --timestamp <iso>      Fixed CycloneDX generation timestamp\n  --output, -o <path>    Write JSON to a file\n  --compact              Emit compact JSON\n  --version              Print version\n  --help                 Show this help\n`);
}

main(process.argv.slice(2)).catch((error) => {
  process.stderr.write(`deepbom: ${error?.message || String(error)}\n`);
  process.exitCode = 1;
});
