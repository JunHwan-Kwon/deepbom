import { downloadText } from "./download.js";
import { shortError } from "./format.js";
import { sha256Hex } from "./hash.js";
import {
  buildRuntimeAssignmentTemplate,
  parseRuntimeAssignmentDocument,
} from "./kernel-inspector.js";
import { jsonForDownload } from "./report-utils.js";
import {
  buildGgufRuntimeEnvironmentTemplate,
  isGgufRuntimeEnvironmentDocument,
  parseGgufRuntimeEnvironmentDocument,
} from "./gguf-runtime-environment.js";
import {
  buildCoreMlComputePlanTemplate,
  isCoreMlComputePlanDocument,
  parseCoreMlComputePlanDocument,
} from "./coreml-compute-plan.js";
import { runtimeCapturePlanAvailability } from "./runtime-evidence-closure.js";
import {
  buildOrtRuntimeAssignmentDocument,
  parseRuntimeProfileSource,
  verifyOrtNativeCaptureProfile,
} from "./runtime-profile-adapter.js";
import {
  buildTfliteRuntimeAssignmentDocument,
  parseTfliteRuntimeInfoSource,
} from "./tflite-runtime-info-adapter.js";
import {
  buildTfliteProfiledAssignmentDocument,
  parseTfliteBenchmarkProfileSource,
} from "./tflite-profile-info-adapter.js";
import {
  buildTensorRtStaticPreflight,
  TENSORRT_PARSER_OBSERVATION_SCHEMA,
} from "./tensorrt-static-preflight.js";

export function installRuntimeEvidenceController({
  elements,
  modal,
  getAnalysis,
  getEvidence,
  setEvidence,
  getPending,
  setPending,
  ensureArtifactHash,
  artifactFilename,
  onChanged,
  setStatus,
}) {
  const {
    input,
    form,
    collectedAt,
    version,
    backend,
    build,
    binarySha,
    optimization,
    executionMode,
    capture,
    formStatus,
    templateButton,
    capturePlanButton,
    clearButton,
  } = elements;

  input?.addEventListener("change", async () => {
    const file = input.files?.[0];
    input.value = "";
    const analysis = getAnalysis();
    if (!file || !analysis) return;
    try {
      if (file.size > 32 * 1024 * 1024) throw new Error("Runtime evidence must be 32 MiB or smaller.");
      await ensureArtifactHash();
      const bytes = new Uint8Array(await file.arrayBuffer());
      const sourceFileSha256 = await sha256Hex(bytes);
      const jsonText = looksLikeJson(bytes) || file.name.toLowerCase().endsWith(".json")
        ? new TextDecoder().decode(bytes)
        : null;
      const jsonValue = jsonText == null ? null : parseJsonForSchema(jsonText);
      if (jsonValue?.schema === TENSORRT_PARSER_OBSERVATION_SCHEMA) {
        const buildProfile = jsonValue.build_profile || analysis.tensorrt_static_preflight?.build_profile;
        if (!buildProfile) throw new Error("TensorRT parser evidence requires its exact build_profile object.");
        analysis.tensorrt_static_preflight = buildTensorRtStaticPreflight(analysis, buildProfile, jsonValue);
        onChanged();
        setStatus("TensorRT parser/configuration evidence imported", "ok");
        return;
      }
      if (isCoreMlComputePlanDocument(jsonValue)) {
        setEvidence(parseCoreMlComputePlanDocument(jsonValue, analysis, { fileSha256: sourceFileSha256 }));
        onChanged();
        setStatus("Core ML compute-plan evidence imported", "ok");
        return;
      }
      if (isGgufRuntimeEnvironmentDocument(jsonValue)) {
        setEvidence(parseGgufRuntimeEnvironmentDocument(jsonValue, analysis, { fileSha256: sourceFileSha256 }));
        onChanged();
        setStatus("GGUF runtime environment evidence imported", "ok");
        return;
      }
      const parsedSource = jsonText != null
        ? parseRuntimeProfileSource(jsonText, analysis)
        : parseTfliteBinaryRuntimeSource(bytes, analysis);
      if (parsedSource.kind === "canonical") {
        setEvidence(parseRuntimeAssignmentDocument(JSON.stringify(parsedSource.source), analysis, { fileSha256: sourceFileSha256 }));
        onChanged();
        setStatus("Runtime assignment evidence imported", "ok");
        return;
      }
      const verifiedNativeCapture = parsedSource.kind === "onnxruntime_profile"
        ? await verifyOrtNativeCaptureProfile(parsedSource, analysis)
        : null;
      setPending({
        profile: parsedSource,
        profileSha256: verifiedNativeCapture?.profileSha256 || sourceFileSha256,
        importFileSha256: sourceFileSha256,
        verifiedMetadata: verifiedNativeCapture?.metadata || null,
        nativeCaptureEvidence: verifiedNativeCapture?.evidence || null,
      });
      modal.open();
    } catch (error) {
      setEvidence(null);
      onChanged();
      setStatus(shortError(error), "error");
    }
  });

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const pending = getPending();
    const analysis = getAnalysis();
    if (!pending || !analysis) return;
    try {
      const tflitePlan = pending.profile.kind === "tflite_model_runtime_info";
      const tfliteTiming = pending.profile.kind === "tflite_benchmark_profile";
      const timestamp = new Date(collectedAt.value);
      if (!Number.isFinite(timestamp.getTime())) throw new Error("Runtime evidence collection time is required.");
      const metadata = {
        runtimeVersion: version.value,
        backend: backend.value,
        runtimeBuild: build.value,
        binarySha256: binarySha.value.trim().toLowerCase() || null,
        graphOptimizationLevel: optimization.value,
        executionMode: executionMode.value,
        collectedAt: timestamp.toISOString(),
        profileSha256: pending.profileSha256,
        captureId: capture?.value.trim() || null,
        nativeCaptureEvidence: pending.nativeCaptureEvidence || null,
      };
      const document = tfliteTiming
        ? buildTfliteProfiledAssignmentDocument(pending.profile, getEvidence(), analysis, metadata)
        : tflitePlan
          ? buildTfliteRuntimeAssignmentDocument(pending.profile, analysis, metadata)
          : buildOrtRuntimeAssignmentDocument(pending.profile, analysis, metadata);
      setEvidence(parseRuntimeAssignmentDocument(JSON.stringify(document), analysis, { fileSha256: pending.importFileSha256 }));
      modal.close();
      onChanged();
      setStatus(tfliteTiming
        ? "TFLite execution timing evidence imported"
        : tflitePlan ? "TFLite runtime plan evidence imported" : "ONNX Runtime profile evidence imported", "ok");
    } catch (error) {
      formStatus.textContent = shortError(error);
      formStatus.dataset.tone = "error";
    }
  });

  templateButton?.addEventListener("click", async () => {
    const analysis = getAnalysis();
    if (!analysis) return;
    await ensureArtifactHash();
    const gguf = analysis.format === "gguf";
    const coreml = analysis.format === "coreml";
    const template = gguf ? buildGgufRuntimeEnvironmentTemplate(analysis)
      : coreml ? buildCoreMlComputePlanTemplate(analysis) : buildRuntimeAssignmentTemplate(analysis);
    const suffix = gguf ? "gguf_runtime_environment.template.json" : coreml ? "coreml_compute_plan.template.json" : "runtime_assignment.template.json";
    downloadText(artifactFilename(suffix), jsonForDownload(template), "application/json");
    setStatus(gguf ? "GGUF runtime environment template downloaded" : coreml ? "Core ML compute-plan template downloaded" : "Runtime assignment template downloaded", "ok");
  });

  capturePlanButton?.addEventListener("click", async () => {
    const analysis = getAnalysis();
    if (!analysis) return;
    try {
      await ensureArtifactHash();
      const availability = runtimeCapturePlanAvailability(analysis);
      if (!availability.available) throw new Error(availability.reason);
      downloadText(artifactFilename("runtime_capture_plan.json"), jsonForDownload(availability.plan), "application/json");
      setStatus("Identity-bound runtime capture plan downloaded", "ok");
    } catch (error) {
      setStatus(shortError(error), "error");
    }
  });

  clearButton?.addEventListener("click", () => {
    setEvidence(null);
    onChanged();
    setStatus("Runtime evidence cleared", "ok");
  });
}

function parseJsonForSchema(text) {
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("Runtime evidence JSON is invalid.");
  }
}

function looksLikeJson(bytes) {
  let index = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0;
  while (index < bytes.length && [0x09, 0x0a, 0x0d, 0x20].includes(bytes[index])) index += 1;
  return bytes[index] === 0x7b || bytes[index] === 0x5b;
}

function parseTfliteBinaryRuntimeSource(bytes, analysis) {
  try {
    return parseTfliteRuntimeInfoSource(bytes, analysis);
  } catch (runtimeInfoError) {
    try {
      return parseTfliteBenchmarkProfileSource(bytes);
    } catch (profileError) {
      throw new Error(`Unsupported TFLite runtime protobuf. ModelRuntimeDetails: ${shortError(runtimeInfoError)} BenchmarkProfilingData: ${shortError(profileError)}`);
    }
  }
}
