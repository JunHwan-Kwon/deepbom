#!/usr/bin/env node

import { createHash } from "node:crypto";
import { access, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { detectModelFormat } from "../web/lib/model-file.js";
import { analyzeOnnxModel, MAX_ONNX_DECODED_ELEMENTS } from "../web/onnx.js";
import { attachOnnxContractConflictCapsule } from "../web/lib/onnx-contract-conflict.js";
import { analyzeExecuTorchModel } from "../web/executorch.js";
import { EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA } from "../web/lib/executorch-build-binding.js";
import { parseStrictJson, readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { validateCustomTargetSpec } from "../web/lib/custom-targets.js";
import { buildInterfaceQuantizationContractLedger } from "../web/lib/quantization-contract-summary.js";
import { compareInterfaceContracts } from "../web/lib/interface-contract.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { buildMlBomDocument } from "../web/lib/report-mlbom.js";
import { buildArtifactEvidenceEnvelope, validateArtifactEvidenceEnvelope } from "../web/lib/artifact-evidence-envelope.js";
import { buildTensorRtStaticPreflight } from "../web/lib/tensorrt-static-preflight.js";
import { buildOnDeviceLlmContract } from "../web/lib/on-device-llm-contract.js";
import { buildLlmStaticMemoryPlacement } from "../web/lib/llm-static-memory-placement.js";
import { readArtifactBundle } from "../web/lib/artifact-bundle.js";
import { identifyCliFile, loadCliBundleMembers, loadCliInput, loadExecuTorchExternalData, loadOnnxExternalData, loadOnnxExternalDataMembers, readCliFileBytes, verifyBundleSnapshot } from "./deepbom-input.mjs";
import { collectNvidiaAcceleratorProfile } from "./nvidia-accelerator-collector.mjs";
import { resolveArtifactSource } from "./remote-artifact-resolver.mjs";
import { resolveHuggingFaceOnnxExternalDataClosure, resolveHuggingFaceSafeTensorsClosure } from "./remote-artifact-closure.mjs";
import { buildSingleFileArtifactSet, finalizeArtifactSet } from "../web/lib/artifact-set.js";
import { getArtifactIrContext } from "../web/lib/artifact-ir-context.js";
import { exportGraphVisualization } from "../web/lib/graph-export.js";
import { exportGraphPng } from "./graph-png-export.mjs";
import { buildNvidiaAcceleratorProfileBinding } from "../web/lib/accelerator-profile-binding.js";
import {
  buildCoreMlAcceleratorBinding,
  buildEdgeTpuAcceleratorBinding,
  buildLiteRtQualcommAcceleratorBinding,
  buildNvidiaHostAcceleratorBinding,
  buildTensorRtAcceleratorBinding,
  buildTfliteAcceleratorBindings,
  buildTfliteAdditionalAcceleratorBindings,
  mergeAcceleratorBindings,
} from "../web/lib/accelerator-binding.js";
import { parseCoreMlComputePlanDocument } from "../web/lib/coreml-compute-plan.js";
import { parseEdgeTpuCompilerEvidence } from "../web/lib/edgetpu-compiler-evidence.js";
import { parseLiteRtQualcommEvidence } from "../web/lib/litert-qualcomm-evidence.js";
import { buildExecutionPlacementEvidence } from "../web/lib/execution-placement-evidence.js";
import { buildPlacementComparison } from "../web/lib/placement-comparison.js";
import { buildCpuCostTargetBinding } from "../web/lib/cpu-target-binding.js";
import { evaluateReviewPolicy, validateReviewPolicy } from "../web/lib/review-policy.js";
import { buildOnnxExternalDataStructureBinding, onnxExternalInitializerElementCount } from "../web/lib/onnx-external-data-structure-binding.js";
import {
  buildCanonicalGatedDecoderProjection,
} from "../web/lib/transformer-architecture-projection.js";
import { buildLlmTokenBudgetScenario } from "../web/lib/llm-token-budget-scenario.js";
import {
  buildCliCapabilities,
  buildSarifDocument,
  evaluateFindingPolicy,
  normalizeFailOn,
  outputExists,
  renderCliError,
  resolveGenerationTimestamp,
  writeOutputAtomically,
} from "./deepbom-automation.mjs";
import {
  analyze_tflite_for_target,
  compute_deployment_delta,
  explore_tflite_redesign_pareto,
  initSync as initTfliteWasm,
} from "../pkg/tflite_wasm_audit.js";

const DEFAULT_TARGET = "android_mid_a55";
const DEFAULT_DELTA_TARGETS = Object.freeze([DEFAULT_TARGET, "rpi4_a72", "x86_avx2", "wasm_simd"]);
const MAX_TARGET_PROFILE_BYTES = 16_384;
const MAX_JSON_SIDECAR_BYTES = 16 * 1024 * 1024;
const MAX_IN_MEMORY_EXECUTABLE_ARTIFACT_BYTES = 1024 * 1024 * 1024;
const METADATA_STRUCTURE_DEFAULT_BYTES = 10 * 1024 * 1024 * 1024;
const METADATA_INTEGRITY_DEFAULT_BYTES = 2 * 1024 * 1024 * 1024;
const VERSION = typeof __DEEPBOM_RELEASE_VERSION__ === "string" ? __DEEPBOM_RELEASE_VERSION__ : "1.96.3";
const EXPECTED_TFLITE_WASM_SHA256 = typeof __DEEPBOM_TFLITE_WASM_SHA256__ === "string" ? __DEEPBOM_TFLITE_WASM_SHA256__ : "";

async function main(argv) {
  const parsed = parseArguments(argv);
  if (parsed.help) return printHelp();
  if (parsed.version) return process.stdout.write(`${VERSION}\n`);
  if (parsed.command === "capabilities") {
    validateCapabilitiesInvocation(parsed);
    await preflightOutputDestinations(parsed);
    const capabilities = buildCliCapabilities(VERSION, { defaultTarget: DEFAULT_TARGET, deltaTargets: DEFAULT_DELTA_TARGETS });
    return emitDocument(parsed, capabilities, () => buildCapabilitiesSummary(capabilities));
  }
  if (parsed.command === "accelerator") {
    validateAcceleratorInvocation(parsed);
    await preflightOutputDestinations(parsed);
    const profile = await collectNvidiaAcceleratorProfile({
      collectorVersion: VERSION,
      deviceIndex: parsed.deviceIndex,
      includeDeviceIdentifiers: parsed.includeDeviceIdentifiers,
    });
    return emitDocument(parsed, profile, () => buildAcceleratorSummary(profile));
  }
  if (!parsed.input) throw new Error("An artifact path is required.");
  validateInvocation(parsed);
  await preflightOutputDestinations(parsed);
  const targetBinding = await resolveTargetBinding(parsed);
  if (parsed.command === "diff") return runDiffCommand(parsed, targetBinding);

  let resolvedSource = await resolveCliArtifactSource(parsed.input, parsed);
  resolvedSource = await resolveHuggingFaceSafeTensorsClosure(parsed.input, resolvedSource, remoteResolveOptions(parsed));
  const inputPath = resolvedSource.path;
  const input = resolvedSource.virtual_bundle_members
    ? await loadCliBundleMembers(resolvedSource.virtual_bundle_members, "remote-model")
    : await loadCliInput(inputPath);
  const filename = input.filename;
  const detectedFormat = input.kind === "file" ? detectModelFormat(filename, input.prefix) : "package";
  let scanPolicy = resolveScanPolicy(parsed.scan, detectedFormat, input);
  if ((parsed.targetProfile || parsed.targetExplicit) && input.kind === "file" && detectedFormat !== "tflite") {
    throw new Error(`${parsed.targetProfile ? "--target-profile" : "--target"} applies only to TFLite artifacts, received ${detectedFormat}.`);
  }
  if (parsed.externalDataRoot && input.kind !== "file") throw new Error("--external-data-dir applies only to an ONNX or ExecuTorch PTE file.");
  if (parsed.executorchBuild && (input.kind !== "file" || detectedFormat !== "executorch")) {
    throw new Error("--executorch-build applies only to an ExecuTorch PTE artifact.");
  }
  if (input.kind === "file" && ["unsupported", "pytorch_pickle"].includes(detectedFormat)) {
    throw new Error(`Unsupported artifact format: ${detectedFormat}`);
  }
  const execuTorchBuild = parsed.executorchBuild
    ? await readJsonSidecar(parsed.executorchBuild, "executorch_selected_build_attestation") : null;
  const acceleratorProfile = parsed.acceleratorProfile
    ? await readJsonSidecar(parsed.acceleratorProfile, "nvidia_accelerator_profile") : null;
  const reviewPolicyInput = parsed.reviewPolicy
    ? await readJsonSidecar(parsed.reviewPolicy, "review_policy") : null;
  const coreMlComputePlanInput = parsed.coreMlComputePlan
    ? await readJsonSidecar(parsed.coreMlComputePlan, "coreml_compute_plan") : null;
  const edgeTpuCompilerInput = parsed.edgeTpuCompilerEvidence
    ? await readJsonSidecar(parsed.edgeTpuCompilerEvidence, "edgetpu_compiler_evidence") : null;
  const liteRtQualcommInput = parsed.liteRtQualcommEvidence
    ? await readJsonSidecar(parsed.liteRtQualcommEvidence, "litert_qualcomm_compiler_dispatch_evidence") : null;
  let preanalyzedOnnx = null;
  if (input.kind === "file" && detectedFormat === "onnx" && resolvedSource.acquisition?.source?.kind === "huggingface" && !parsed.externalDataRoot) {
    const bytes = await readCliFileBytes(input);
    preanalyzedOnnx = analyzeOnnxModel(bytes, filename);
    const locations = [...new Set((preanalyzedOnnx?.onnx_external_data?.tensors || [])
      .map((row) => String(row.normalized_location || "")).filter(Boolean))].sort();
    if (locations.length) resolvedSource = await resolveHuggingFaceOnnxExternalDataClosure(
      parsed.input, resolvedSource, locations, remoteResolveOptions(parsed),
    );
    if (resolvedSource.external_data_members) {
      scanPolicy = resolveOnnxExternalAutoScanPolicy(scanPolicy, parsed.scan, preanalyzedOnnx, resolvedSource.closure);
    }
  }
  let analysis = await analyzeArtifact({
    input,
    filename,
    format: detectedFormat,
    target: targetBinding.value,
    externalDataRoot: parsed.externalDataRoot || resolvedSource.closure?.model_root || "",
    execuTorchBuild,
    scanPolicy,
    preanalyzedOnnx,
    externalDataMembers: resolvedSource.external_data_members,
  });
  const format = String(analysis.format || detectedFormat).toLowerCase();
  if (["unsupported", "pytorch_pickle"].includes(format)) {
    throw new Error(`Unsupported artifact format: ${format}`);
  }
  if (parsed.targetProfile && format !== "tflite") {
    throw new Error(`--target-profile applies only to TFLite artifacts, received ${format}.`);
  }
  if (parsed.targetExplicit && format !== "tflite") {
    throw new Error(`--target applies only to TFLite artifacts, received ${format}.`);
  }
  if (parsed.contract && parsed.command !== "verify") throw new Error("--contract is valid only with the verify command.");
  if (parsed.request && parsed.command !== "explore") throw new Error("--request is valid only with the explore command.");
  if (parsed.externalDataRoot && !["onnx", "executorch"].includes(format)) {
    throw new Error("--external-data-dir applies only to an ONNX or ExecuTorch PTE file.");
  }
  if (parsed.executorchBuild && (format !== "executorch" || analysis.executorch_container !== "pte")) {
    throw new Error("--executorch-build applies only to an ExecuTorch PTE artifact.");
  }
  if (parsed.command === "gguf" && format !== "gguf") {
    throw new Error(`The gguf command requires a GGUF artifact, received ${format}.`);
  }
  const llmScenarioRequested = parsed.context || parsed.batch !== 1 || parsed.stateBits !== 16
    || parsed.memoryMib || parsed.images || parsed.tokensPerImage;
  if (llmScenarioRequested && !["tflite", "onnx", "gguf", "safetensors"].includes(format)) {
    throw new Error("LLM token-budget options require a TFLite, ONNX, GGUF, or SafeTensors artifact with a statically derived KV-state contract.");
  }
  if (!parsed.context && (parsed.batch !== 1 || parsed.stateBits !== 16 || parsed.memoryMib || parsed.images || parsed.tokensPerImage)) {
    throw new Error("--batch, --state-bits, --memory-mib, --images, and --tokens-per-image require --context.");
  }
  if ((parsed.images > 0) !== (parsed.tokensPerImage != null)) {
    throw new Error("--images and --tokens-per-image must be provided together.");
  }
  const artifact = input.kind === "file"
    ? { ...(await identifyCliFile(input)), format }
    : packageIdentity(analysis, format, filename);
  if (parsed.expectedSha256 && artifact.sha256 !== parsed.expectedSha256) {
    throw new Error(`Artifact SHA-256 mismatch: expected ${parsed.expectedSha256}, observed ${artifact.sha256}.`);
  }
  if (resolvedSource.acquisition && resolvedSource.closure?.kind !== "huggingface_safetensors_shards"
      && artifact.sha256 !== resolvedSource.acquisition.file.sha256) {
    throw new Error("Analyzed artifact SHA-256 differs from the remote acquisition receipt.");
  }
  enforceArtifactIdentity(analysis, artifact);
  analysis.cli_scan_policy = scanPolicy;
  analysis.artifact_set = buildCliArtifactSet(artifact, resolvedSource.acquisition, analysis, resolvedSource.closure);
  if (format === "onnx") attachOnnxContractConflictCapsule(analysis);
  analysis.accelerator_bindings = [];
  if (format === "tflite") {
    analysis.cpu_cost_target_binding = buildCpuCostTargetBinding(analysis.target_profile, {
      bindingSource: targetBinding.bindingSource,
      sourceInput: targetBinding.evidence,
    });
    if (targetBinding.evidence) {
      analysis.cli_target_profile_input = {
        ...targetBinding.evidence,
        resolved_target_id: analysis.target_profile?.id || null,
        resolved_target_profile_sha256: analysis.target_profile?.profile_sha256 || null,
      };
    }
  }
  const artifactSha256 = artifact.sha256;
  if (coreMlComputePlanInput) {
    if (format !== "coreml") throw new Error("--coreml-compute-plan applies only to a Core ML artifact or package.");
    analysis.coreml_compute_plan = parseCoreMlComputePlanDocument(coreMlComputePlanInput.document, analysis, {
      fileSha256: coreMlComputePlanInput.sha256,
    });
    analysis.accelerator_bindings.push(buildCoreMlAcceleratorBinding(analysis, analysis.coreml_compute_plan));
  }
  if (edgeTpuCompilerInput) {
    if (format !== "tflite") throw new Error("--edgetpu-compiler-evidence applies only to a TFLite artifact.");
    analysis.edgetpu_compiler_evidence = parseEdgeTpuCompilerEvidence(edgeTpuCompilerInput.document, analysis, {
      fileSha256: edgeTpuCompilerInput.sha256,
    });
    analysis.accelerator_bindings.push(buildEdgeTpuAcceleratorBinding(analysis, analysis.edgetpu_compiler_evidence));
  }
  if (liteRtQualcommInput) {
    if (format !== "tflite") throw new Error("--litert-qualcomm-evidence applies only to a TFLite artifact.");
    analysis.litert_qualcomm_evidence = parseLiteRtQualcommEvidence(liteRtQualcommInput.document, analysis, {
      fileSha256: liteRtQualcommInput.sha256,
    });
    analysis.accelerator_bindings.push(buildLiteRtQualcommAcceleratorBinding(analysis, analysis.litert_qualcomm_evidence));
  }
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
  if (parsed.context) {
    if (!analysis.on_device_llm) analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
    const scenario = buildCliLlmTokenBudgetScenario(analysis, parsed.context, {
      imageCount: parsed.images,
      tokensPerImage: parsed.tokensPerImage,
      batchSize: parsed.batch,
      stateBits: parsed.stateBits,
      memoryMib: parsed.memoryMib,
    });
    analysis.llm_token_budget_scenario = scenario;
    analysis.cli_context_scenario = scenario;
  }
  if ((parsed.tensorrtProfile || parsed.tensorrtParserEvidence || parsed.tensorrtEngineInspector) && format !== "onnx") {
    throw new Error("--tensorrt-profile, --tensorrt-parser-evidence, and --tensorrt-engine-inspector apply only to ONNX artifacts.");
  }
  if (format === "onnx") {
    const parserEvidence = parsed.tensorrtParserEvidence ? await readJsonDocument(parsed.tensorrtParserEvidence) : null;
    const engineInspectorEvidence = parsed.tensorrtEngineInspector ? await readJsonDocument(parsed.tensorrtEngineInspector) : null;
    const buildProfile = parsed.tensorrtProfile
      ? await readJsonDocument(parsed.tensorrtProfile)
      : parserEvidence?.build_profile || null;
    analysis.tensorrt_static_preflight = buildTensorRtStaticPreflight(analysis, buildProfile, parserEvidence, engineInspectorEvidence);
    const tensorRtBinding = buildTensorRtAcceleratorBinding(analysis, analysis.tensorrt_static_preflight);
    if (tensorRtBinding) analysis.accelerator_bindings.push(tensorRtBinding);
  }
  if (acceleratorProfile) {
    analysis.accelerator_profile_binding = buildNvidiaAcceleratorProfileBinding(analysis, acceleratorProfile, {
      deviceIndex: parsed.acceleratorDeviceIndex,
    });
    analysis.accelerator_bindings.push(buildNvidiaHostAcceleratorBinding(analysis, analysis.accelerator_profile_binding));
  } else if (parsed.acceleratorDeviceIndex != null) {
    throw new Error("--accelerator-device requires --accelerator-profile.");
  }
  analysis.accelerator_bindings = mergeAcceleratorBindings(
    analysis.accelerator_bindings,
    buildTfliteAcceleratorBindings(analysis),
    buildTfliteAdditionalAcceleratorBindings(analysis),
  );
  const reviewPolicy = reviewPolicyInput ? validateReviewPolicy(reviewPolicyInput.document) : null;
  if (reviewPolicy) {
    analysis.policy_identity = {
      schema: reviewPolicy.schema,
      policy_sha256: reviewPolicy.policy_sha256,
      source_file_sha256: reviewPolicyInput.sha256,
      mode: reviewPolicy.mode,
    };
  }
  const artifactIrContext = requiresArtifactIrContext(parsed)
    ? getArtifactIrContext(analysis, {
        ...artifact,
        artifact_set_sha256: analysis.artifact_set?.artifact_set_sha256 || null,
      })
    : null;
  if (requiresArtifactIrContext(parsed) && !artifactIrContext) throw new Error("Canonical Artifact Evidence IR could not be constructed for the analyzed artifact.");
  const analysisView = artifactIrContext?.primary_view || analysis;

  if (parsed.command === "graph") return runGraphCommand(parsed, artifactIrContext);
  if (parsed.command === "placement") return runPlacementCommand(parsed, analysisView, artifact);

  if (parsed.command === "verify") return runVerifyCommand(parsed, analysisView, artifact);
  if (parsed.command === "explore") return runExploreCommand(parsed, analysisView, artifact, input, targetBinding.value);

  const generatedAt = resolveGenerationTimestamp(parsed.timestamp);
  const envelope = parsed.outputFormat === "envelope" || parsed.outputFormat === "sarif" || parsed.failOn !== "none" || reviewPolicy
    ? buildArtifactEvidenceEnvelope(analysisView, {
        hash: artifactSha256,
        fileSizeBytes: artifact.size,
        filename: artifact.filename,
        generatedAt,
        provenance: {
          analyzer: "DEEPBOM",
          version: VERSION,
          command: parsed.command,
          target_profile_id: analysis.target_profile?.id || null,
        },
      })
    : null;
  if (envelope) {
    const validation = validateArtifactEvidenceEnvelope(envelope);
    if (!validation.valid) throw new Error(`Canonical evidence envelope validation failed: ${validation.errors.join(", ")}`);
  }
  const policyResult = reviewPolicy
    ? evaluateReviewPolicy(envelope, reviewPolicy, {
        analyzerVersion: VERSION,
        rulepackVersion: analysis.rulepack_version || null,
        evaluatedAt: generatedAt || new Date().toISOString(),
        sourceFileSha256: reviewPolicyInput.sha256,
      })
    : envelope ? evaluateFindingPolicy(envelope, parsed.failOn) : null;
  const document = parsed.outputFormat === "cyclonedx"
    ? buildMlBomDocument(analysisView, {
        hash: artifactSha256,
        fileSizeBytes: artifact.size,
        timestamp: generatedAt || new Date().toISOString(),
        ...(analysis.target_profile ? {
          target: analysis.target_profile,
          targetId: analysis.target_profile.id,
        } : {}),
        artifactIr: artifactIrContext.artifact_ir,
      })
    : parsed.outputFormat === "envelope"
      ? envelope
      : parsed.outputFormat === "sarif"
        ? buildSarifDocument(envelope, { version: VERSION, policyResult })
        : analysis;
  await emitDocument(parsed, document, () => buildHumanSummary(analysisView, artifact));
  if (parsed.policyOutput) {
    await writeOutputAtomically(parsed.policyOutput, `${JSON.stringify(policyResult, null, parsed.compact ? 0 : 2)}\n`, { noClobber: parsed.noClobber });
  }
  if (policyResult?.status === "block") {
    const blockingCount = policyResult.blocking_finding_count ?? policyResult.finding_policy?.blocking_finding_count ?? 0;
    const threshold = policyResult.fail_on ?? policyResult.finding_policy?.fail_on ?? "configured policy";
    process.stderr.write(`deepbom: review policy blocked (${blockingCount} finding(s) at or above ${threshold}; coverage ${policyResult.coverage_status || "not evaluated"}).\n`);
    process.exitCode = 2;
  }
}

async function resolveTargetBinding(parsed) {
  if (!parsed.targetProfile) return {
    value: parsed.target || DEFAULT_TARGET,
    evidence: null,
    bindingSource: parsed.targetExplicit ? "explicit_id" : "default_assumption",
  };
  if (parsed.targetExplicit) throw new Error("--target and --target-profile are mutually exclusive.");
  const sidecar = await readJsonSidecar(parsed.targetProfile, "target_profile", MAX_TARGET_PROFILE_BYTES);
  const validated = validateCustomTargetSpec(sidecar.document);
  const normalized = canonicalJson(validated);
  return {
    value: JSON.stringify(validated),
    bindingSource: "profile_file",
    evidence: {
      schema: "deepbom.cli_target_profile_input.v1",
      filename: sidecar.path,
      byte_length: sidecar.byte_length,
      source_sha256: sidecar.sha256,
      normalized_profile_sha256: createHash("sha256").update(normalized, "utf8").digest("hex"),
      duplicate_key_validation: "complete",
    },
  };
}

function requiresArtifactIrContext(parsed) {
  const humanAnalysisOutput = parsed.outputFormat === "analysis" && !parsed.json && !parsed.compact && !parsed.output;
  return ["graph", "placement"].includes(parsed.command)
    || parsed.outputFormat !== "analysis"
    || parsed.failOn !== "none"
    || Boolean(parsed.reviewPolicy)
    || humanAnalysisOutput;
}

function resolveCliArtifactSource(spec, parsed) {
  return resolveArtifactSource(spec, {
    cacheDir: parsed.cacheDir || undefined,
    expectedSha256: parsed.expectedSha256,
    offline: parsed.offline,
    maximumBytes: BigInt(parsed.maxDownloadGib) * 1024n * 1024n * 1024n,
  });
}

function remoteResolveOptions(parsed) {
  return {
    cacheDir: parsed.cacheDir || undefined,
    expectedSha256: parsed.expectedSha256,
    offline: parsed.offline,
    maximumBytes: BigInt(parsed.maxDownloadGib) * 1024n * 1024n * 1024n,
  };
}

function validateInvocation(parsed) {
  if (parsed.deviceIndex != null || parsed.includeDeviceIdentifiers) {
    throw new Error("--device and --include-device-identifiers apply only to accelerator collect nvidia.");
  }
  if (parsed.targetProfile && parsed.targetExplicit) throw new Error("--target and --target-profile are mutually exclusive.");
  if (parsed.contract && parsed.command !== "verify") throw new Error("--contract is valid only with the verify command.");
  if (parsed.request && parsed.command !== "explore") throw new Error("--request is valid only with the explore command.");
  if (parsed.view && parsed.command !== "graph") throw new Error("--view is valid only with the graph command.");
  if (parsed.command === "verify" && !parsed.contract) throw new Error("The verify command requires --contract <json>.");
  if (parsed.command === "diff" && !parsed.candidate) throw new Error("The diff command requires baseline and candidate TFLite artifacts.");
  if (parsed.executorchBuild && parsed.command !== "audit") throw new Error("--executorch-build is valid only with the audit command.");
  if (parsed.scanExplicit && ["verify", "diff", "explore"].includes(parsed.command)) {
    throw new Error(`--scan is not accepted by the ${parsed.command} command.`);
  }
  if (parsed.acceleratorProfile && !["audit", "gguf", "graph"].includes(parsed.command)) {
    throw new Error("--accelerator-profile is valid only with audit, gguf, or graph.");
  }
  if ((parsed.coreMlComputePlan || parsed.edgeTpuCompilerEvidence || parsed.liteRtQualcommEvidence)
      && !["audit", "graph", "placement"].includes(parsed.command)) {
    throw new Error("Compiled accelerator evidence is valid only with audit, graph, or placement.");
  }
  if (parsed.reviewPolicy && !["audit", "gguf"].includes(parsed.command)) {
    throw new Error("--review-policy is valid only with audit or gguf.");
  }
  if (parsed.acceleratorDeviceIndex != null && !parsed.acceleratorProfile) {
    throw new Error("--accelerator-device requires --accelerator-profile.");
  }
  if (parsed.reviewPolicy && parsed.failOnExplicit) throw new Error("--review-policy and --fail-on are mutually exclusive sources of finding policy.");
  if (parsed.policyOutput && parsed.failOn === "none" && !parsed.reviewPolicy) throw new Error("--policy-output requires --fail-on or --review-policy.");
  if (parsed.noClobber && !parsed.output && !parsed.policyOutput) throw new Error("--no-clobber requires --output or --policy-output.");
  if (parsed.noClobber && parsed.output === "-" && !parsed.policyOutput) {
    throw new Error("--no-clobber cannot protect stdout; provide a file path with --output.");
  }
  if (parsed.policyOutput === "-") throw new Error("--policy-output requires a file path; use --output - for the primary document.");
  if (parsed.timestamp && parsed.outputFormat === "analysis" && parsed.failOn === "none") {
    throw new Error("--timestamp applies only to envelope, CycloneDX, SARIF, or finding-policy evidence.");
  }
  if (["verify", "diff", "explore"].includes(parsed.command) && (parsed.failOn !== "none" || parsed.policyOutput)) {
    throw new Error(`The ${parsed.command} command does not accept finding-policy options.`);
  }
  if (["verify", "diff", "explore"].includes(parsed.command) && parsed.timestamp) {
    throw new Error(`The ${parsed.command} command does not accept --timestamp.`);
  }
  if (parsed.output && parsed.output !== "-" && parsed.policyOutput
      && path.resolve(parsed.output) === path.resolve(parsed.policyOutput)) {
    throw new Error("--output and --policy-output must identify different files.");
  }
  if (["verify", "diff", "explore"].includes(parsed.command)) assertCommandOutputFormat(parsed, parsed.command);
  if (parsed.command === "graph") {
    if (parsed.failOn !== "none" || parsed.policyOutput) throw new Error("The graph command does not accept finding-policy options.");
    if (parsed.timestamp) throw new Error("The graph command is deterministic and does not accept --timestamp.");
    if (parsed.contract || parsed.request) throw new Error("The graph command does not accept verify or redesign sidecars.");
  }
  if (parsed.command === "placement") {
    if (parsed.failOn !== "none" || parsed.policyOutput || parsed.reviewPolicy) throw new Error("The placement command does not accept finding-policy options.");
    if (parsed.timestamp) throw new Error("The placement command is deterministic and does not accept --timestamp.");
    if (parsed.contract || parsed.request) throw new Error("The placement command does not accept verify or redesign sidecars.");
    if (parsed.formatExplicit) throw new Error("The placement command emits deepbom.placement_comparison.v1; --format is not supported.");
  } else if (parsed.placementProfilesExplicit) {
    throw new Error("--profiles is valid only with the placement command.");
  }
}

async function runPlacementCommand(parsed, analysis, artifact) {
  const runtimeEvidence = analysis.coreml_compute_plan || null;
  const execution = buildExecutionPlacementEvidence(analysis, runtimeEvidence);
  const document = parsed.placementProfiles == null
    ? execution.placement_comparison
    : buildPlacementComparison(analysis, execution.static_profiles, { selectedProfileIds: parsed.placementProfiles });
  await emitDocument(parsed, document, () => buildPlacementSummary(document, artifact));
}

async function preflightOutputDestinations(parsed) {
  if (!parsed.noClobber) return;
  const paths = [parsed.output === "-" ? "" : parsed.output, parsed.policyOutput].filter(Boolean);
  for (const outputPath of paths) {
    if (await outputExists(outputPath)) throw new Error(`Output already exists: ${path.resolve(outputPath)}`);
  }
}

function validateCapabilitiesInvocation(parsed) {
  if (parsed.input) throw new Error("The capabilities command does not accept an artifact path.");
  assertNoOptions(parsed, [
    "targetProfile", "contract", "request", "context", "images", "tokensPerImage", "batch", "stateBits", "memoryMib", "tensorrtProfile",
    "tensorrtParserEvidence", "tensorrtEngineInspector", "tensorrtLlmConfig", "tensorrtLlmBinding",
    "llmMemoryProfile", "acceleratorProfile", "coreMlComputePlan", "edgeTpuCompilerEvidence", "liteRtQualcommEvidence", "externalDataRoot", "executorchBuild", "policyOutput", "cacheDir", "expectedSha256", "offline",
  ], "capabilities");
  if (parsed.targetExplicit || parsed.failOn !== "none") throw new Error("The capabilities command does not accept target or finding-policy options.");
  if (parsed.deviceIndex != null || parsed.includeDeviceIdentifiers) throw new Error("The capabilities command does not accept NVIDIA collector options.");
  if (parsed.acceleratorDeviceIndex != null) throw new Error("The capabilities command does not accept --accelerator-device.");
  if (parsed.scanExplicit) throw new Error("The capabilities command does not accept --scan.");
  if (parsed.placementProfilesExplicit) throw new Error("The capabilities command does not accept --profiles.");
  if (parsed.maxDownloadExplicit) throw new Error("The capabilities command does not accept --max-download-gib.");
  if (parsed.timestamp) throw new Error("The capabilities command does not accept --timestamp.");
  if (parsed.noClobber && !parsed.output) throw new Error("--no-clobber requires --output.");
  if (parsed.noClobber && parsed.output === "-") throw new Error("--no-clobber cannot protect stdout; provide a file path with --output.");
  if (parsed.formatExplicit) throw new Error("The capabilities command emits deepbom.cli_capabilities.v1; --format is not supported.");
}

function validateAcceleratorInvocation(parsed) {
  if (parsed.input || parsed.candidate) throw new Error("The accelerator collect nvidia command does not accept an artifact path.");
  if (parsed.acceleratorAction !== "collect" || parsed.acceleratorProvider !== "nvidia") {
    throw new Error("The accelerator command requires: deepbom accelerator collect nvidia.");
  }
  assertNoOptions(parsed, [
    "targetProfile", "contract", "request", "context", "images", "tokensPerImage", "batch", "stateBits", "memoryMib", "tensorrtProfile",
    "tensorrtParserEvidence", "tensorrtEngineInspector", "tensorrtLlmConfig", "tensorrtLlmBinding",
    "llmMemoryProfile", "acceleratorProfile", "coreMlComputePlan", "edgeTpuCompilerEvidence", "liteRtQualcommEvidence", "externalDataRoot", "executorchBuild", "policyOutput", "cacheDir", "expectedSha256", "offline",
  ], "accelerator collect nvidia");
  if (parsed.acceleratorDeviceIndex != null) throw new Error("The collector uses --device, not --accelerator-device.");
  if (parsed.scanExplicit) throw new Error("The accelerator collect nvidia command does not accept --scan.");
  if (parsed.placementProfilesExplicit) throw new Error("The accelerator collect nvidia command does not accept --profiles.");
  if (parsed.targetExplicit || parsed.failOn !== "none") throw new Error("The accelerator collect nvidia command does not accept target or finding-policy options.");
  if (parsed.maxDownloadExplicit) throw new Error("The accelerator collect nvidia command does not accept --max-download-gib.");
  if (parsed.timestamp) throw new Error("The accelerator collect nvidia command records its observation time and does not accept --timestamp.");
  if (parsed.noClobber && !parsed.output) throw new Error("--no-clobber requires --output.");
  if (parsed.noClobber && parsed.output === "-") throw new Error("--no-clobber cannot protect stdout; provide a file path with --output.");
  if (parsed.formatExplicit) throw new Error("The accelerator command emits deepbom.accelerator_profile.v1; --format is not supported.");
}

async function runDiffCommand(parsed, targetBinding) {
  assertCommandOutputFormat(parsed, "diff");
  assertNoOptions(parsed, [
    "contract", "request", "context", "images", "tokensPerImage", "batch", "stateBits", "memoryMib", "tensorrtProfile", "tensorrtParserEvidence", "tensorrtEngineInspector",
    "tensorrtLlmConfig", "tensorrtLlmBinding", "llmMemoryProfile", "externalDataRoot", "executorchBuild",
  ], "diff");
  const baselineSource = await resolveCliArtifactSource(parsed.input, parsed);
  const candidateSource = await resolveArtifactSource(parsed.candidate, {
    cacheDir: parsed.cacheDir || undefined,
    offline: parsed.offline,
    maximumBytes: BigInt(parsed.maxDownloadGib) * 1024n * 1024n * 1024n,
  });
  const baselineInput = await loadCliInput(baselineSource.path);
  const candidateInput = await loadCliInput(candidateSource.path);
  if (baselineInput.kind !== "file" || candidateInput.kind !== "file") {
    throw new Error("The diff command requires two standalone TFLite files.");
  }
  const baselineFormat = detectModelFormat(baselineInput.filename, baselineInput.prefix);
  const candidateFormat = detectModelFormat(candidateInput.filename, candidateInput.prefix);
  if (baselineFormat !== "tflite" || candidateFormat !== "tflite") {
    throw new Error(`The diff command supports TFLite artifacts only, received ${baselineFormat} and ${candidateFormat}.`);
  }
  await initializeTfliteWasm();
  const baselineBytes = await readCliFileBytes(baselineInput);
  const candidateBytes = await readCliFileBytes(candidateInput);
  const targetIds = parsed.targetProfile
    ? [targetBinding.value, ...DEFAULT_DELTA_TARGETS]
    : parsed.targetExplicit
      ? [targetBinding.value, ...DEFAULT_DELTA_TARGETS.filter((id) => id !== parsed.target)]
      : [...DEFAULT_DELTA_TARGETS];
  const delta = compute_deployment_delta(
    baselineBytes,
    baselineInput.filename,
    candidateBytes,
    candidateInput.filename,
    JSON.stringify(targetIds),
  );
  if (baselineSource.acquisition || candidateSource.acquisition) {
    delta.cli_artifact_sources = {
      baseline: baselineSource.acquisition,
      candidate: candidateSource.acquisition,
    };
  }
  if (targetBinding.evidence) delta.cli_target_profile_input = targetBinding.evidence;
  return emitDocument(parsed, delta, () => buildDiffSummary(delta));
}

async function runVerifyCommand(parsed, analysis, artifact) {
  assertCommandOutputFormat(parsed, "verify");
  const ledger = buildInterfaceQuantizationContractLedger(analysis);
  if (!Array.isArray(ledger.parameters) || ledger.parameters.length === 0) {
    throw new Error(`${artifact.format} does not expose a serialized external interface contract for verification.`);
  }
  const sidecar = await readJsonSidecar(parsed.contract, "production_interface_contract", MAX_JSON_SIDECAR_BYTES);
  const comparison = compareInterfaceContracts(ledger, sidecar.document, artifact.sha256);
  const document = {
    schema: "deepbom.cli_interface_contract_verification.v1",
    command: "verify",
    artifact,
    target_profile: analysis.target_profile || null,
    declaration: {
      filename: sidecar.path,
      byte_length: sidecar.byte_length,
      sha256: sidecar.sha256,
      duplicate_key_validation: "complete",
    },
    interface_contract_ledger: ledger,
    comparison,
    interpretation_boundary: "This command compares serialized external tensor ABI facts with one supplied declaration. It does not establish preprocessing semantics that are absent from both sources, runtime assignment, inference correctness, task accuracy, clinical validity, or release readiness.",
  };
  await emitDocument(parsed, document, () => buildVerifySummary(document));
  process.exitCode = comparison.gate_result === "pass" ? 0 : comparison.gate_result === "block" ? 2 : 3;
}

async function runExploreCommand(parsed, analysis, artifact, input, target) {
  assertCommandOutputFormat(parsed, "explore");
  if (artifact.format !== "tflite" || input.kind !== "file") {
    throw new Error(`The explore command supports standalone TFLite artifacts only, received ${artifact.format}.`);
  }
  const requestSidecar = parsed.request
    ? await readJsonSidecar(parsed.request, "redesign_request", MAX_JSON_SIDECAR_BYTES)
    : null;
  const request = requestSidecar?.document || {
    schema: "deepbom.redesign_request.v1",
    source_sha256: artifact.sha256,
    input_height: null,
    input_width: null,
    width_multiplier: 1,
    activation_dtype: "source",
    block_edits: [],
  };
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new Error("The redesign request must be a JSON object.");
  }
  const boundRequest = { ...request, source_sha256: request.source_sha256 || artifact.sha256 };
  const bytes = await readCliFileBytes(input);
  const pareto = explore_tflite_redesign_pareto(bytes, artifact.filename, target, boundRequest);
  pareto.artifact = artifact;
  pareto.target_profile = analysis.target_profile || null;
  if (requestSidecar) {
    pareto.request_input = {
      filename: requestSidecar.path,
      byte_length: requestSidecar.byte_length,
      sha256: requestSidecar.sha256,
      duplicate_key_validation: "complete",
    };
  }
  if (analysis.cli_target_profile_input) pareto.cli_target_profile_input = analysis.cli_target_profile_input;
  return emitDocument(parsed, pareto, () => buildExploreSummary(pareto));
}

async function runGraphCommand(parsed, artifactIrContext) {
  const artifactIr = artifactIrContext.artifact_ir;
  const graph = artifactIrContext.graph_ir;
  if (parsed.outputFormat === "png") {
    const exported = exportGraphPng(graph, { view: parsed.view });
    if (parsed.output && parsed.output !== "-") await writeOutputAtomically(parsed.output, exported.bytes, { noClobber: parsed.noClobber });
    else process.stdout.write(exported.bytes);
    return;
  }
  const exported = exportGraphVisualization(graph, { view: parsed.view, format: parsed.outputFormat, compact: parsed.compact, artifactIr });
  if (parsed.output && parsed.output !== "-") await writeOutputAtomically(parsed.output, exported.text, { noClobber: parsed.noClobber });
  else process.stdout.write(exported.text);
}

async function initializeTfliteWasm() {
  const wasm = await readFile(await resolveRuntimeAsset("tflite_wasm_audit_bg.wasm"));
  if (EXPECTED_TFLITE_WASM_SHA256 && createHash("sha256").update(wasm).digest("hex") !== EXPECTED_TFLITE_WASM_SHA256) {
    throw new Error("Packaged TFLite WASM failed its release SHA-256 check.");
  }
  initTfliteWasm({ module: wasm });
}

async function emitDocument(parsed, document, humanBuilder) {
  const machineReadable = parsed.json || parsed.compact || Boolean(parsed.output) || parsed.outputFormat !== "analysis";
  const text = machineReadable
    ? `${JSON.stringify(document, bigintReplacer, parsed.compact ? 0 : 2)}\n`
    : humanBuilder();
  if (parsed.output && parsed.output !== "-") await writeOutputAtomically(parsed.output, text, { noClobber: parsed.noClobber });
  else process.stdout.write(text);
}

function assertCommandOutputFormat(parsed, command) {
  if (parsed.outputFormat !== "analysis") {
    throw new Error(`The ${command} command emits its own evidence schema; --format ${parsed.outputFormat} is not supported.`);
  }
}

function assertNoOptions(parsed, names, command) {
  const used = names.filter((name) => {
    const value = parsed[name];
    return Boolean(value) && value !== 1 && value !== 16;
  });
  if (used.length) throw new Error(`The ${command} command does not accept: ${used.map(optionName).join(", ")}.`);
}

function optionName(name) {
  if (name === "externalDataRoot") return "--external-data-dir";
  return `--${name.replace(/[A-Z]/g, (character) => `-${character.toLowerCase()}`)}`;
}

async function analyzeArtifact({ input, filename, format, target, externalDataRoot, execuTorchBuild, scanPolicy, preanalyzedOnnx = null, externalDataMembers = null }) {
  if (input.kind === "bundle") {
    const parsed = await readArtifactBundle(input.files, {
      scanMode: scanPolicy.effective_mode,
      onProgress: buildBundleScanProgressReporter(input.files),
    });
    await verifyBundleSnapshot(input.files, parsed.analysis.artifact_bundle?.files);
    return parsed.analysis;
  }
  if (format === "tflite") {
    const bytes = await readCliFileBytes(input);
    await initializeTfliteWasm();
    return analyze_tflite_for_target(bytes, filename, target || DEFAULT_TARGET);
  }
  if (format === "onnx") {
    const bytes = await readCliFileBytes(input);
    const structural = preanalyzedOnnx || analyzeOnnxModel(bytes, filename);
    if (externalDataMembers && scanPolicy.effective_mode === "structure") {
      structural.onnx_external_data_structure_binding = buildOnnxExternalDataStructureBinding(structural, externalDataMembers);
      return structural;
    }
    const externalDataFiles = externalDataMembers
      ? await loadOnnxExternalDataMembers(structural, externalDataMembers)
      : await loadOnnxExternalData(input.path, structural, externalDataRoot);
    return externalDataFiles.length ? analyzeOnnxModel(bytes, filename, null, { externalDataFiles }) : structural;
  }

  if (format === "executorch") {
    const bytes = await readCliFileBytes(input);
    const buildOptions = execuTorchBuild ? {
      selectedBuildAttestation: execuTorchBuild.document,
      selectedBuildInput: {
        schema: EXECUTORCH_SELECTED_BUILD_INPUT_SCHEMA,
        path: execuTorchBuild.path,
        byte_length: execuTorchBuild.byte_length,
        file_sha256: execuTorchBuild.sha256,
        attestation_sha256: execuTorchBuild.document?.attestation_sha256,
        duplicate_key_validation: "complete",
      },
    } : {};
    const structural = analyzeExecuTorchModel(bytes, filename, buildOptions);
    if (structural.executorch_container !== "pte") {
      if (externalDataRoot || execuTorchBuild) throw new Error("ExecuTorch PTD files do not accept external-data or selected-build sidecars.");
      return structural;
    }
    if (!structural.executorch_program?.external_tensor_data?.required_name_count) return structural;
    const externalDataFiles = await loadExecuTorchExternalData(input.path, externalDataRoot);
    return externalDataFiles.length ? analyzeExecuTorchModel(bytes, filename, { ...buildOptions, externalDataFiles }) : structural;
  }

  if (format === "gguf" || format === "safetensors") {
    return (await readMetadataModelFile(input.file, format, {
      scanMode: scanPolicy.effective_mode,
      onProgress: buildTensorScanProgressReporter(input.file.size),
    })).analysis;
  }
  if (format === "coreml") return (await readCoreMlModelFile(input.file)).analysis;
  throw new Error(`No analyzer is registered for ${format}.`);
}

function resolveScanPolicy(requested, format, input) {
  const mode = String(requested || "auto").toLowerCase();
  if (!["auto", "structure", "integrity", "full"].includes(mode)) {
    throw new Error("--scan must be auto, structure, integrity, or full.");
  }
  if (input.kind === "bundle") {
    const size = input.files.reduce((sum, file) => sum + file.size, 0);
    const safeTensors = input.files.some((file) => /\.safetensors(?:\.index\.json)?$/i.test(file.name));
    if (safeTensors) {
      const effective = mode === "auto"
        ? size > METADATA_STRUCTURE_DEFAULT_BYTES ? "structure" : size > METADATA_INTEGRITY_DEFAULT_BYTES ? "integrity" : "full"
        : mode;
      return createScanPolicy(mode, effective, effective === "structure"
        ? "manifest_bound_shard_headers_and_tensor_directories_payload_numerical_scan_skipped"
        : "manifest_bound_shards_streamed_with_payload_integrity", size);
    }
    if (!new Set(["auto", "full"]).has(mode)) throw new Error("Core ML package inputs currently support --scan full only.");
    return createScanPolicy(mode, "full", "coreml_package_analyzer_requires_complete_manifest_selected_file_set", size);
  }
  const size = input.file.size;
  if (["gguf", "safetensors"].includes(format)) {
    const effective = mode === "auto"
      ? size > METADATA_STRUCTURE_DEFAULT_BYTES ? "structure" : size > METADATA_INTEGRITY_DEFAULT_BYTES ? "integrity" : "full"
      : mode;
    return createScanPolicy(mode, effective, effective === "structure"
      ? "range_parsed_header_and_tensor_directory_payload_numerical_scan_skipped"
      : "range_parsed_header_and_streamed_full_declared_tensor_payloads", size);
  }
  if (mode !== "auto" && mode !== "full") {
    throw new Error(`${format} does not provide a truthful ${mode} scan path; use --scan full.`);
  }
  if (["tflite", "onnx", "executorch"].includes(format) && size > MAX_IN_MEMORY_EXECUTABLE_ARTIFACT_BYTES) {
    throw new Error(`${format} artifact is ${size} bytes and exceeds the ${MAX_IN_MEMORY_EXECUTABLE_ARTIFACT_BYTES}-byte fail-closed monolithic parser limit. Use an external-data or sharded representation where the format supports it; this CLI will not risk an unbounded full-file allocation.`);
  }
  return createScanPolicy(mode, "full", format === "coreml"
    ? "range_parsed_coreml_protobuf_and_payload_contract"
    : "complete_serialized_executable_artifact_parse", size);
}

function resolveOnnxExternalAutoScanPolicy(current, requested, analysis, closure) {
  if (String(requested || "auto").toLowerCase() !== "auto") return current;
  const elements = onnxExternalInitializerElementCount(analysis);
  if (elements == null || elements <= BigInt(MAX_ONNX_DECODED_ELEMENTS)) return current;
  const totalBytes = (closure?.members || []).reduce((sum, member) => {
    const value = member?.byte_length?.number ?? Number(member?.byte_length?.decimal);
    return Number.isSafeInteger(value) && value >= 0 ? sum + value : sum;
  }, 0);
  return createScanPolicy(
    "auto",
    "structure",
    `onnx_external_initializer_elements_${elements}_exceed_decoder_safety_limit_${MAX_ONNX_DECODED_ELEMENTS}`,
    totalBytes,
  );
}

function createScanPolicy(requested, effective, reason, byteLength) {
  return {
    schema: "deepbom.cli_scan_policy.v1",
    requested_mode: requested,
    effective_mode: effective,
    artifact_byte_length: { decimal: String(byteLength), number: Number.isSafeInteger(byteLength) ? byteLength : null },
    reason,
    boundary: effective === "structure"
      ? "Header, metadata, tensor descriptors, serialized ranges, and storage cardinality are assessed. Tensor payload numerical integrity is explicitly not assessed."
      : "Full declared tensor payload ranges are streamed for container integrity where supported. Executable graph formats use their complete parser contract; runtime execution remains outside scope.",
  };
}

function buildTensorScanProgressReporter(totalBytes) {
  if (!process.stderr.isTTY && process.env.DEEPBOM_PROGRESS !== "1") return undefined;
  let last = 0;
  return (event) => {
    const now = Date.now();
    if (now - last < 1000 && event.index + 1 < event.count) return;
    last = now;
    process.stderr.write(`deepbom: PAYLOAD ${event.index + 1}/${event.count} ${event.tensor} ${event.bytes_read}/${event.tensor_bytes} bytes (artifact ${totalBytes} bytes)\n`);
  };
}

function buildBundleScanProgressReporter(files) {
  if (!process.stderr.isTTY && process.env.DEEPBOM_PROGRESS !== "1") return undefined;
  const totalBytes = files.reduce((sum, file) => sum + Number(file.size || 0), 0);
  let last = 0;
  return (event = {}) => {
    const now = Date.now();
    if (now - last < 1000 && event.phase !== "complete") return;
    last = now;
    const index = Number.isSafeInteger(event.index) ? event.index + 1 : null;
    const count = Number.isSafeInteger(event.count) ? event.count : null;
    const item = index != null && count != null ? ` ${index}/${count}` : "";
    const pathLabel = event.path ? ` ${String(event.path).slice(0, 120)}` : "";
    process.stderr.write(`[${String(event.phase || "ANALYZE").toUpperCase()}]${item}${pathLabel} · package ${formatProgressBytes(totalBytes)}\n`);
  };
}

function formatProgressBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes / 1024;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1; }
  return `${amount.toFixed(amount >= 100 ? 0 : amount >= 10 ? 1 : 2)} ${units[index]}`;
}

function packageIdentity(analysis, format, filename) {
  const sha256 = String(analysis?.model_sha256 || "").toLowerCase();
  const size = Number(analysis?.file_size_bytes);
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error("Package analyzer did not emit a canonical SHA-256 identity.");
  if (!Number.isSafeInteger(size) || size <= 0) throw new Error("Package analyzer did not emit a valid aggregate byte size.");
  return { filename, format, size, sha256 };
}

function buildCliArtifactSet(artifact, acquisition, analysis, closure) {
  const bundleFiles = Array.isArray(analysis?.artifact_bundle?.files) ? analysis.artifact_bundle.files : [];
  const closureFiles = Array.isArray(closure?.members) ? closure.members : [];
  const sourceFiles = closureFiles.length ? closureFiles : bundleFiles;
  if (!acquisition && !sourceFiles.length) {
    return buildSingleFileArtifactSet({
      filename: artifact.filename,
      format: artifact.format,
      sha256: artifact.sha256,
      byteLength: artifact.size,
    });
  }
  const primaryIndex = sourceFiles.length ? selectArtifactSetPrimary(sourceFiles) : 0;
  const files = sourceFiles.length ? sourceFiles.map((row, index) => ({
    role: index === primaryIndex ? "primary" : artifactSetRole(row.role),
    path: String(row.path),
    sha256: String(row.sha256),
    byte_length: normalizeExactBytes(row.byte_length),
  })) : [{
    role: "primary",
    path: artifact.filename,
    sha256: artifact.sha256,
    byte_length: { decimal: String(artifact.size), number: artifact.size },
  }];
  return finalizeArtifactSet({
    schema: "deepbom.artifact_set.v1",
    evidence_class: "OBSERVED_ACQUISITION",
    source: acquisition?.source || {
      kind: "local",
      canonical_locator: `local:${artifact.filename}`,
      immutability: { kind: "sha256", value: artifact.sha256 },
    },
    subject: {
      filename: artifact.filename,
      format: artifact.format,
      sha256: artifact.sha256,
      byte_length: { decimal: String(artifact.size), number: artifact.size },
      identity_basis: sourceFiles.length ? "canonical_manifest_bound_file_set" : "artifact_file_bytes",
    },
    files,
    trust: {
      remote_code_execution: "forbidden",
      pickle_execution: "forbidden",
      model_code_execution: "forbidden",
    },
  });
}

function selectArtifactSetPrimary(files) {
  const preferred = ["shard_index", "package_manifest", "root_model"];
  for (const role of preferred) {
    const index = files.findIndex((row) => row.role === role);
    if (index >= 0) return index;
  }
  return 0;
}

function artifactSetRole(role) {
  if (["tensor_shard", "weights"].includes(role)) return "shard";
  if (["architecture_config", "quantization_config", "configuration"].includes(role)) return "configuration";
  if (["shard_index", "package_manifest"].includes(role)) return "manifest";
  return "sidecar";
}

function normalizeExactBytes(value) {
  const decimal = String(value?.decimal ?? value);
  const exact = BigInt(decimal);
  return { decimal, number: exact <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(exact) : null };
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

function buildCliLlmTokenBudgetScenario(analysis, textTokens, {
  imageCount = 0, tokensPerImage = null, batchSize = 1, stateBits = 16, memoryMib = null,
} = {}) {
  const capacityBytes = memoryMib == null ? null : BigInt(memoryMib) * 1024n * 1024n;
  const scenario = buildLlmTokenBudgetScenario(analysis, {
    textTokens,
    imageCount,
    tokensPerImage,
    batchSize,
    stateStorageBits: stateBits,
    memoryCapacityBytes: capacityBytes,
    source: "cli_argument",
  });
  const totalContext = scenario.token_budget.total_context_tokens.value;
  const contract = analysis.gguf?.semantic_contract;
  const keyHeadWidth = contract?.attention_key_length || contract?.derived_attention_head_width;
  const valueHeadWidth = contract?.attention_value_length || contract?.derived_attention_head_width;
  const decoderReady = totalContext != null && [
    contract?.block_count,
    contract?.attention_head_count_kv,
    keyHeadWidth,
    valueHeadWidth,
    contract?.tokenizer?.vocabulary_count,
    contract?.embedding_length,
    contract?.feed_forward_length,
    contract?.attention_head_count,
  ].every(isPositiveSafeInteger) && keyHeadWidth === valueHeadWidth;
  const { scenario_sha256: _baseScenarioSha256, ...scenarioBody } = scenario;
  const body = {
    ...scenarioBody,
    context_length: totalContext,
    text_context_length: textTokens,
    image_count: imageCount,
    tokens_per_image: tokensPerImage,
    image_token_count: scenario.token_budget.image_tokens.value,
    batch_size: batchSize,
    state_storage_bits: stateBits,
    context_source: "cli_argument",
    serialized_context_length: scenario.serialized_context_contract.context_length,
    kv_state_projection: scenario.state_projection,
    compute_projection: decoderReady ? buildCanonicalGatedDecoderProjection({
      vocabularySize: contract.tokenizer.vocabulary_count,
      hiddenSize: contract.embedding_length,
      intermediateSize: contract.feed_forward_length,
      layerCount: contract.block_count,
      attentionHeadCount: contract.attention_head_count,
      kvHeadCount: contract.attention_head_count_kv,
      headWidth: keyHeadWidth,
      contextLength: totalContext,
    }) : null,
    compute_projection_status: decoderReady ? "assessed_registered_canonical_decoder_scenario"
      : "not_emitted_without_registered_gguf_canonical_decoder_contract",
    memory_feasibility: {
      ...scenario.memory_feasibility,
      logical_kv_state_bytes: scenario.state_projection.state_kind === "transformer_kv"
        ? scenario.memory_feasibility.logical_state_bytes : null,
    },
  };
  return { ...body, scenario_sha256: createHash("sha256").update(canonicalJson(body)).digest("hex") };
}

function isPositiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function buildHumanSummary(analysis, artifact) {
  const format = String(analysis?.format || artifact?.format || "unknown").toUpperCase();
  const operators = nonNegativeInteger(analysis?.operator_count);
  const tensors = nonNegativeInteger(analysis?.tensor_count);
  const macs = exactDecimal(analysis?.total_macs_decimal ?? analysis?.total_macs);
  const quantized = nonNegativeInteger(analysis?.quantized_tensors);
  const quantization = analysis?.quantization_status;
  const findings = Array.isArray(analysis?.findings) ? analysis.findings : [];
  const target = analysis?.target_profile;
  const segments = Array.isArray(analysis?.xnnpack_chains) ? analysis.xnnpack_chains.length : null;
  const breaks = nonNegativeInteger(analysis?.xnnpack_effective_chain_breaks);
  const lines = [
    `DEEPBOM ${VERSION} deployment-artifact audit`,
    `Artifact: ${artifact.filename}`,
    `Identity: sha256:${artifact.sha256}`,
    `Format: ${format} | Size: ${formatBytes(artifact.size)}`,
  ];

  if (operators !== null || tensors !== null || macs !== null) {
    const graph = [];
    if (operators !== null) graph.push(`${formatInteger(operators)} operators`);
    if (tensors !== null) graph.push(`${formatInteger(tensors)} tensors`);
    graph.push(macs === null ? "MACs not assessable" : `${formatInteger(macs)} MACs`);
    lines.push(`Graph: ${graph.join(" | ")}`);
  }
  if (quantization?.classification || quantized !== null) {
    const detail = [];
    if (quantization?.label || quantization?.classification) detail.push(String(quantization.label || quantization.classification));
    if (quantized !== null && tensors !== null) detail.push(`${formatInteger(quantized)}/${formatInteger(tensors)} tensors quantized`);
    if (Number.isFinite(Number(quantization?.quantized_compute_mac_percent))) {
      detail.push(`${formatPercent(quantization.quantized_compute_mac_percent)} MAC-bearing compute coverage`);
    }
    lines.push(`Quantization: ${detail.join(" | ")}`);
  }
  if (target?.id || target?.label) lines.push(`Target: ${target.label || target.id} (${target.id || "profile-bound"})`);
  if (format === "TFLITE" && (segments !== null || breaks !== null || Number.isFinite(Number(analysis?.delegated_mac_percent)))) {
    const placement = [];
    if (segments !== null) placement.push(`${segments} predicted XNNPACK segment${segments === 1 ? "" : "s"}`);
    if (breaks !== null) placement.push(`${breaks} predicted break${breaks === 1 ? "" : "s"}`);
    if (Number.isFinite(Number(analysis?.delegated_mac_percent))) {
      placement.push(`${formatPercent(analysis.delegated_mac_percent)} of MACs conditionally delegatable`);
    }
    lines.push(`Placement: ${placement.join(" | ")} under the stated rulepack and build conditions`);
  }
  appendTokenBudgetScenario(lines, analysis?.llm_token_budget_scenario || analysis?.cli_context_scenario);
  if (findings.length) {
    lines.push(`Findings: ${findings.length}`);
    for (const finding of findings.slice(0, 5)) lines.push(`  - ${finding.title || finding.finding_id || finding.id || "Unnamed finding"}`);
    if (findings.length > 5) lines.push(`  - ${findings.length - 5} more; use --json for the complete evidence ledger`);
  } else {
    lines.push("Findings: none emitted by the applicable static checks");
  }
  lines.push(`Evidence boundary: ${evidenceBoundary(format)}`);
  lines.push("Machine output: rerun with --json, --compact, --format cyclonedx, or --output <path>.");
  return `${lines.join("\n")}\n`;
}

function buildVerifySummary(document) {
  const comparison = document.comparison;
  const lines = [
    `DEEPBOM ${VERSION} external-interface contract verification`,
    `Artifact: ${document.artifact.filename}`,
    `Identity: sha256:${document.artifact.sha256}`,
    `Declaration: ${document.declaration.filename} | sha256:${document.declaration.sha256}`,
    `Result: ${comparison.gate_result.toUpperCase()} | ${comparison.status}`,
    `Parameters: ${comparison.declared_parameter_count}/${comparison.expected_parameter_count} declared/expected`,
    `Mismatches: ${comparison.mismatch_count}`,
  ];
  for (const mismatch of comparison.mismatches.slice(0, 8)) {
    lines.push(`  - ${mismatch.parameter_id || "document"}: ${mismatch.field} expected=${displayValue(mismatch.expected)} declared=${displayValue(mismatch.declared)}`);
  }
  if (comparison.mismatches.length > 8) lines.push(`  - ${comparison.mismatches.length - 8} more; use --json for the complete ledger`);
  lines.push(`Evidence boundary: ${document.interpretation_boundary}`);
  lines.push("Exit codes: 0 exact pass, 2 contradiction/invalid declaration, 3 incomplete release binding.");
  return `${lines.join("\n")}\n`;
}

function buildDiffSummary(delta) {
  const alignment = delta.alignment || {};
  const worst = delta.target_deltas?.find((row) => row.target_id === delta.worst_relative_delta_target_id) || null;
  const lines = [
    `DEEPBOM ${VERSION} deterministic TFLite deployment delta`,
    `Baseline: ${delta.baseline.filename} | sha256:${delta.baseline.sha256}`,
    `Candidate: ${delta.candidate.filename} | sha256:${delta.candidate.sha256}`,
    `Relation: ${alignment.artifact_relation || "not assessed"}`,
    `Alignment: ${alignment.matched_op_count ?? 0} matched | ${alignment.added_op_count ?? 0} added | ${alignment.removed_op_count ?? 0} removed`,
    `Targets: ${delta.target_count}`,
  ];
  for (const target of delta.target_deltas || []) {
    lines.push(`  - ${target.target_id}: ${formatSigned(target.signed_delta_us, " us")} | ${formatSignedPercent(target.relative_delta)}`);
  }
  if (worst) lines.push(`Largest relative modeled delta: ${worst.target_id} (${formatSignedPercent(worst.relative_delta)})`);
  lines.push(`Evidence boundary: ${delta.interpretation_boundary}`);
  lines.push("The alignment is a deterministic comparison coordinate and does not establish model lineage or semantic layer identity.");
  return `${lines.join("\n")}\n`;
}

function buildExploreSummary(pareto) {
  const frontier = (pareto.candidates || []).filter((candidate) => candidate.pareto_optimal);
  const lines = [
    `DEEPBOM ${VERSION} deterministic TFLite redesign exploration`,
    `Artifact: ${pareto.artifact.filename} | sha256:${pareto.source_sha256}`,
    `Target: ${pareto.target_profile?.label || pareto.target_id} (${pareto.target_id})`,
    `Candidates: ${pareto.evaluated_candidate_count} evaluated | ${pareto.accepted_candidate_count} accepted | ${pareto.rejected_candidate_count} rejected`,
    `Pareto frontier: ${pareto.frontier_candidate_count}`,
  ];
  for (const candidate of frontier.slice(0, 8)) {
    const request = candidate.request || {};
    lines.push(`  - ${candidate.candidate_id}: ${request.input_height}x${request.input_width}, width ${formatDecimal(request.width_multiplier, 3)} | ${formatDecimal(candidate.modeled_latency_ms, 3)} ms modeled | ${formatInteger(candidate.parameter_elements)} parameters`);
  }
  if (frontier.length > 8) lines.push(`  - ${frontier.length - 8} more; use --json for the complete candidate ledger`);
  lines.push(`Evidence boundary: ${pareto.interpretation_boundary}`);
  return `${lines.join("\n")}\n`;
}

function buildPlacementSummary(comparison, artifact) {
  const lines = [
    `DEEPBOM ${VERSION} N-way placement comparison`,
    `Artifact: ${artifact.filename}`,
    `Identity: sha256:${artifact.sha256}`,
    `Profiles: ${comparison.rows.length}/${comparison.available_profile_ids.length} selected/available`,
  ];
  if (!comparison.rows.length) lines.push("Placement: not applicable; no static execution profile exists for this artifact class.");
  for (const row of comparison.rows) {
    lines.push(`  - ${row.profile_id}: ${row.conditionally_eligible_ops}/${row.op_count} conditionally eligible | ${row.definite_exclusion_ops} excluded | ${row.unresolved_ops} unresolved | ${row.boundary_edge_count} logical boundary edges`);
  }
  lines.push(`Evidence boundary: ${comparison.interpretation_boundary}`);
  lines.push("Machine output: rerun with --json or --compact for source identities, exact state counts, and boundary payloads.");
  return `${lines.join("\n")}\n`;
}

function buildCapabilitiesSummary(capabilities) {
  const commands = capabilities.commands.map((row) => row.name).join(", ");
  const formats = capabilities.inputs.standalone_extensions.join(", ");
  return [
    `DEEPBOM ${capabilities.cli_version} CLI capabilities`,
    `Commands: ${commands}`,
    `Standalone inputs: ${formats}`,
    "Machine discovery: rerun with --json or --compact.",
    "Canonical automation outputs: envelope, CycloneDX 1.7, and SARIF 2.1.0.",
    "Exit codes: 0 pass, 1 invocation/analysis failure, 2 policy or verification block, 3 incomplete verification binding.",
    "",
  ].join("\n");
}

function buildAcceleratorSummary(profile) {
  const lines = [
    `DEEPBOM ${VERSION} NVIDIA accelerator observation`,
    `Profile: sha256:${profile.profile_sha256}`,
    `Devices: ${profile.devices.length}`,
  ];
  for (const device of profile.devices) {
    lines.push(`  - GPU ${device.index}: ${device.name} | CC ${device.compute_capability} | ${formatBytes(device.memory_total_bytes.decimal)}`);
  }
  lines.push(`Driver: ${profile.software.nvidia_driver.version}`);
  lines.push(`CUDA toolkit: ${profile.software.cuda_toolkit.version || profile.software.cuda_toolkit.status}`);
  lines.push(`TensorRT: ${profile.software.tensorrt.version || profile.software.tensorrt.status}`);
  lines.push("GPU roofline: not assessable without exact SM and memory-interface contracts.");
  lines.push(`Evidence boundary: ${profile.interpretation_boundary}`);
  return `${lines.join("\n")}\n`;
}

function displayValue(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, bigintReplacer);
}

function formatSigned(value, suffix = "") {
  const number = Number(value);
  if (!Number.isFinite(number)) return "not assessable";
  return `${number > 0 ? "+" : ""}${formatDecimal(number, 3)}${suffix}`;
}

function formatSignedPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "not assessable";
  const percent = number * 100;
  return `${percent > 0 ? "+" : ""}${formatDecimal(percent, 1)}%`;
}

function formatDecimal(value, maximumFractionDigits = 3) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(Number(value));
}

function appendMemoryScenario(lines, memory) {
  if (!memory || typeof memory !== "object") return;
  const lowerBound = exactDecimal(memory.static_lower_bound_bytes?.decimal ?? memory.static_lower_bound_bytes?.value);
  const capacity = exactDecimal(memory.declared_capacity_bytes?.decimal ?? memory.declared_capacity_bytes?.value);
  if (lowerBound === null) return;
  const parts = [`${formatBytes(lowerBound)} static resident-set lower bound`];
  if (capacity !== null) parts.push(`${formatBytes(capacity)} declared capacity`);
  if (memory.status) parts.push(String(memory.status).replaceAll("_", " "));
  lines.push(`Memory scenario: ${parts.join(" | ")}`);
}

function appendTokenBudgetScenario(lines, scenario) {
  if (!scenario || typeof scenario !== "object") return;
  const budget = scenario.token_budget || {};
  const total = exactDecimal(budget.total_context_tokens?.decimal ?? scenario.context_length);
  if (total !== null) {
    const parts = [`${formatInteger(total)} total context tokens`, `${formatInteger(budget.text_tokens ?? scenario.text_context_length ?? total)} text`];
    if (Number(budget.image_count ?? scenario.image_count) > 0) {
      parts.push(`${formatInteger(budget.image_count ?? scenario.image_count)} image(s)`);
      parts.push(`${formatInteger(budget.tokens_per_image ?? scenario.tokens_per_image)} tokens/image`);
    }
    parts.push(scenario.serialized_context_contract?.assessment || "serialized context not assessed");
    lines.push(`Token scenario: ${parts.join(" | ")}`);
  }
  appendMemoryScenario(lines, scenario.memory_feasibility);
}

function evidenceBoundary(format) {
  if (format === "GGUF" || format === "SAFETENSORS") {
    return "the container does not serialize an executable DAG; runtime lowering, placement, latency, and task quality require separate evidence.";
  }
  if (format === "COREML") {
    return "serialized program evidence does not establish observed device placement, latency, task quality, or release readiness.";
  }
  return "static compatibility and cost results do not establish observed runtime assignment, latency, task quality, clinical validity, or release readiness.";
}

function nonNegativeInteger(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function exactDecimal(value) {
  if (typeof value === "bigint") return value >= 0n ? value.toString() : null;
  if (typeof value === "string" && /^\d+$/.test(value)) return value;
  return nonNegativeInteger(value);
}

function formatInteger(value) {
  if (typeof value === "string") return value.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "not assessable";
  const percent = Math.abs(number) <= 1 ? number * 100 : number;
  return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(percent)}%`;
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "not assessable";
  if (bytes < 1024) return `${formatInteger(Math.round(bytes))} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let scaled = bytes;
  let unit = "B";
  for (const candidate of units) {
    scaled /= 1024;
    unit = candidate;
    if (scaled < 1024) break;
  }
  return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: scaled < 10 ? 2 : 1 }).format(scaled)} ${unit}`;
}

function parseArguments(argv) {
  const values = [...argv];
  const first = values[0] || "";
  if (["-h", "--help", "help"].includes(first)) return { help: true };
  if (["-v", "--version", "version"].includes(first)) return { version: true };
  const command = ["audit", "gguf", "verify", "diff", "explore", "graph", "placement", "capabilities", "accelerator"].includes(first) ? values.shift() : "audit";
  const acceleratorAction = command === "accelerator" ? values.shift() || "" : "";
  const acceleratorProvider = command === "accelerator" ? values.shift() || "" : "";
  const parsed = {
    command,
    acceleratorAction,
    acceleratorProvider,
    input: "",
    candidate: "",
    target: DEFAULT_TARGET,
    targetExplicit: false,
    targetProfile: "",
    contract: "",
    request: "",
    view: command === "graph" ? "structure" : "",
    outputFormat: command === "graph" ? "svg" : "analysis",
    output: "",
    policyOutput: "",
    reviewPolicy: "",
    failOn: "none",
    failOnExplicit: false,
    noClobber: false,
    errorFormat: "text",
    timestamp: "",
    context: null,
    images: 0,
    tokensPerImage: null,
    batch: 1,
    stateBits: 16,
    memoryMib: null,
    tensorrtProfile: "",
    tensorrtParserEvidence: "",
    tensorrtEngineInspector: "",
    tensorrtLlmConfig: "",
    tensorrtLlmBinding: "",
    llmMemoryProfile: "",
    scan: "auto",
    scanExplicit: false,
    acceleratorProfile: "",
    acceleratorDeviceIndex: null,
    coreMlComputePlan: "",
    edgeTpuCompilerEvidence: "",
    liteRtQualcommEvidence: "",
    placementProfiles: null,
    placementProfilesExplicit: false,
    externalDataRoot: "",
    executorchBuild: "",
    cacheDir: "",
    expectedSha256: "",
    offline: false,
    maxDownloadGib: 50,
    maxDownloadExplicit: false,
    deviceIndex: null,
    includeDeviceIdentifiers: false,
    json: false,
    compact: false,
    formatExplicit: false,
  };
  while (values.length) {
    const token = values.shift();
    if (token === "--target") {
      parsed.target = requiredValue(values, token);
      parsed.targetExplicit = true;
    }
    else if (token === "--target-profile") parsed.targetProfile = requiredValue(values, token);
    else if (token === "--contract") parsed.contract = requiredValue(values, token);
    else if (token === "--request") parsed.request = requiredValue(values, token);
    else if (token === "--view") parsed.view = requiredValue(values, token).toLowerCase();
    else if (token === "--context") parsed.context = positiveInteger(requiredValue(values, token), token);
    else if (token === "--images") parsed.images = positiveInteger(requiredValue(values, token), token);
    else if (token === "--tokens-per-image") parsed.tokensPerImage = positiveInteger(requiredValue(values, token), token);
    else if (token === "--batch") parsed.batch = positiveInteger(requiredValue(values, token), token);
    else if (token === "--state-bits") parsed.stateBits = stateBits(requiredValue(values, token));
    else if (token === "--memory-mib") parsed.memoryMib = positiveInteger(requiredValue(values, token), token);
    else if (token === "--tensorrt-profile") parsed.tensorrtProfile = requiredValue(values, token);
    else if (token === "--tensorrt-parser-evidence") parsed.tensorrtParserEvidence = requiredValue(values, token);
    else if (token === "--tensorrt-engine-inspector") parsed.tensorrtEngineInspector = requiredValue(values, token);
    else if (token === "--tensorrt-llm-config") parsed.tensorrtLlmConfig = requiredValue(values, token);
    else if (token === "--tensorrt-llm-binding") parsed.tensorrtLlmBinding = requiredValue(values, token);
    else if (token === "--llm-memory-profile") parsed.llmMemoryProfile = requiredValue(values, token);
    else if (token === "--scan") {
      parsed.scan = requiredValue(values, token).toLowerCase();
      parsed.scanExplicit = true;
    }
    else if (token === "--accelerator-profile") parsed.acceleratorProfile = requiredValue(values, token);
    else if (token === "--accelerator-device") parsed.acceleratorDeviceIndex = parseNonNegativeInteger(requiredValue(values, token), token);
    else if (token === "--coreml-compute-plan") parsed.coreMlComputePlan = requiredValue(values, token);
    else if (token === "--edgetpu-compiler-evidence") parsed.edgeTpuCompilerEvidence = requiredValue(values, token);
    else if (token === "--litert-qualcomm-evidence") parsed.liteRtQualcommEvidence = requiredValue(values, token);
    else if (token === "--profiles") {
      parsed.placementProfiles = parsePlacementProfiles(requiredValue(values, token));
      parsed.placementProfilesExplicit = true;
    }
    else if (token === "--external-data-dir") parsed.externalDataRoot = requiredValue(values, token);
    else if (token === "--executorch-build") parsed.executorchBuild = requiredValue(values, token);
    else if (token === "--cache-dir") parsed.cacheDir = requiredValue(values, token);
    else if (token === "--expected-sha256") parsed.expectedSha256 = requiredValue(values, token).toLowerCase();
    else if (token === "--offline") parsed.offline = true;
    else if (token === "--max-download-gib") {
      parsed.maxDownloadGib = boundedPositiveInteger(requiredValue(values, token), token, 1024);
      parsed.maxDownloadExplicit = true;
    }
    else if (token === "--device") parsed.deviceIndex = parseNonNegativeInteger(requiredValue(values, token), token);
    else if (token === "--include-device-identifiers") parsed.includeDeviceIdentifiers = true;
    else if (token === "--format" || token === "--output-format") {
      parsed.outputFormat = requiredValue(values, token).toLowerCase();
      parsed.formatExplicit = true;
    }
    else if (token === "--output" || token === "-o") parsed.output = requiredValue(values, token);
    else if (token === "--policy-output") parsed.policyOutput = requiredValue(values, token);
    else if (token === "--review-policy") parsed.reviewPolicy = requiredValue(values, token);
    else if (token === "--fail-on") {
      parsed.failOn = normalizeFailOn(requiredValue(values, token));
      parsed.failOnExplicit = true;
    }
    else if (token === "--no-clobber") parsed.noClobber = true;
    else if (token === "--error-format") parsed.errorFormat = requiredValue(values, token).toLowerCase();
    else if (token === "--timestamp") parsed.timestamp = normalizeTimestamp(requiredValue(values, token));
    else if (token === "--json") parsed.json = true;
    else if (token === "--compact") parsed.compact = true;
    else if (token === "--help" || token === "-h") parsed.help = true;
    else if (token === "--version" || token === "-v") parsed.version = true;
    else if (token.startsWith("-")) throw new Error(`Unknown option: ${token}`);
    else if (!parsed.input) parsed.input = token;
    else if (parsed.command === "diff" && !parsed.candidate) parsed.candidate = token;
    else throw new Error(`Unexpected positional argument: ${token}`);
  }
  if (parsed.command === "graph" && (parsed.json || parsed.compact)) {
    if (parsed.formatExplicit && parsed.outputFormat !== "json") throw new Error("--json or --compact conflicts with an explicit non-JSON graph --format.");
    parsed.outputFormat = "json";
  }
  const outputFormats = parsed.command === "graph"
    ? new Set(["svg", "png", "html", "mermaid", "dot", "json"])
    : new Set(["analysis", "envelope", "cyclonedx", "sarif"]);
  if (!outputFormats.has(parsed.outputFormat)) {
    throw new Error(parsed.command === "graph" ? "--format must be svg, png, html, mermaid, dot, or json for graph." : "--format must be analysis, envelope, cyclonedx, or sarif.");
  }
  if (parsed.command === "graph" && !new Set(["structure", "placement", "quantization", "architecture"]).has(parsed.view)) {
    throw new Error("--view must be structure, placement, quantization, or architecture.");
  }
  if (!new Set(["text", "json"]).has(parsed.errorFormat)) throw new Error("--error-format must be text or json.");
  if (parsed.json && parsed.compact) throw new Error("--json and --compact are mutually exclusive.");
  return parsed;
}

function requiredValue(values, option) {
  const value = values.shift();
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value.`);
  return value;
}

function parsePlacementProfiles(value) {
  if (String(value).trim().toLowerCase() === "all") return null;
  const profiles = String(value).split(",").map((item) => item.trim()).filter(Boolean);
  if (!profiles.length || profiles.some((item) => !/^[a-z0-9][a-z0-9_.-]*$/i.test(item))) {
    throw new Error("--profiles must be all or a comma-separated list of placement profile IDs.");
  }
  if (new Set(profiles).size !== profiles.length) throw new Error("--profiles must not contain duplicate profile IDs.");
  return profiles;
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

function boundedPositiveInteger(value, option, maximum) {
  const parsed = positiveInteger(value, option);
  if (parsed > maximum) throw new Error(`${option} must not exceed ${maximum}.`);
  return parsed;
}

function parseNonNegativeInteger(value, option) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(`${option} must be a non-negative safe integer.`);
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
  return (await readJsonSidecar(filePath, "JSON document", MAX_JSON_SIDECAR_BYTES)).document;
}

async function readJsonSidecar(filePath, role, maximumBytes = MAX_JSON_SIDECAR_BYTES) {
  const resolved = path.resolve(filePath);
  try {
    const bytes = await readFile(resolved);
    if (bytes.byteLength > maximumBytes) throw new Error(`byte length ${bytes.byteLength} exceeds the ${maximumBytes}-byte limit`);
    const source = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return {
      role,
      path: path.basename(resolved),
      byte_length: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      document: parseStrictJson(source, role),
    };
  } catch (error) {
    throw new Error(`Cannot read ${role} JSON ${resolved}: ${error?.message || error}`);
  }
}

function printHelp() {
  process.stdout.write(`DEEPBOM ${VERSION}\n\nUsage:\n  deepbom audit <artifact-or-package> [options]\n  deepbom gguf <artifact.gguf> [options]\n  deepbom verify <artifact> --contract <json> [options]\n  deepbom diff <baseline.tflite> <candidate.tflite> [options]\n  deepbom explore <artifact.tflite> [options]\n  deepbom graph <artifact> [options]\n  deepbom accelerator collect nvidia [options]\n  deepbom capabilities [--json|--compact]\n\nSupported inputs:\n  .tflite, .onnx, .gguf, .safetensors, .mlmodel, .pte, .ptd\n  .mlpackage directories and sharded SafeTensors repository directories\n\nOptions:\n  --target <id>          TFLite target profile (default: ${DEFAULT_TARGET})\n  --target-profile <json>\n                          Bind a strict custom TFLite target profile (mutually exclusive with --target)\n  --contract <json>      Production external-interface contract for verify\n  --request <json>       Bound redesign request for explore\n  --external-data-dir <directory>\n                          Resolve ONNX external_data or ExecuTorch PTD sidecars from this directory\n  --context <tokens>     Declared text-token scenario for a statically derived LLM KV contract\n  --images <count>       Declared image count; requires --tokens-per-image\n  --tokens-per-image <count>\n                          Declared projector output tokens per image; never inferred\n  --batch <count>        LLM scenario batch size (default: 1)\n  --state-bits <bits>    LLM state width: 8, 16, or 32 (default: 16)\n  --memory-mib <MiB>     Compare the conditional lower bound with a declared capacity\n  --tensorrt-profile <json>\n                          Bind an ONNX TensorRT native/ORT EP build profile\n  --tensorrt-parser-evidence <json>\n                          Import identity-bound TensorRT parser/build evidence\n  --tensorrt-llm-config <json>\n                          Assess a TensorRT-LLM engine config with SafeTensors\n  --tensorrt-llm-binding <json>\n                          Bind that config to model-source/component digests\n  --llm-memory-profile <json>\n                          Evaluate serialized layer/state lower bounds against declared CPU and accelerator pools\n  --format <kind>        analysis, envelope, cyclonedx, or sarif (audit/gguf only)\n  --timestamp <iso>      Fixed generation timestamp; SOURCE_DATE_EPOCH is also honored\n  --fail-on <severity>   Exit 2 for findings at/above informational, low, medium, or high\n  --policy-output <path> Write the deterministic finding-gate decision JSON\n  --output, -o <path>    Atomically write the complete document; use - for stdout\n  --no-clobber           Refuse to replace an existing output or policy file\n  --error-format <kind>  text or json structured stderr (default: text)\n  --json                 Emit the complete formatted evidence JSON\n  --compact              Emit the complete compact evidence JSON\n  --version              Print version\n  --help                 Show this help\n\nExit codes:\n  0 pass; 1 invocation/input/analysis/output failure; 2 policy or verification block; 3 incomplete verification binding\n`);
  process.stdout.write("\nNVIDIA accelerator binding:\n  --accelerator-profile <json>\n                          Bind an observed NVIDIA host profile without inferring selected-build or runtime assignment\n  --accelerator-device <index>\n                          Select one device when the bound profile contains multiple NVIDIA devices\n");
  process.stdout.write("\nN-way placement comparison:\n  deepbom placement <artifact> [--profiles <id,id|all>] [--json|--compact]\n  --profiles <ids|all>   Compare selected independent profiles (default: all available profiles)\n");
  process.stdout.write("\nCompiled accelerator evidence:\n  --coreml-compute-plan <json>\n                          Import an artifact- and compiled-model-bound MLComputePlan estimate; not executed placement\n  --edgetpu-compiler-evidence <json>\n                          Import an artifact/compiler/invocation/compiled-artifact-bound Edge TPU operation ledger\n  --litert-qualcomm-evidence <json>\n                          Import an artifact/source/compiler/QNN-plan-bound operation ledger\n");
  process.stdout.write("\nNVIDIA accelerator observation:\n  deepbom accelerator collect nvidia [--device <index>] [--json|--compact]\n  --device <index>        Collect one NVIDIA device index (default: all devices)\n  --include-device-identifiers\n                          Include raw GPU UUID and PCI bus ID; hashes are always emitted\n");
  process.stdout.write("\nRemote immutable artifact input:\n  hf://owner/repo@<40-hex-commit>/path\n  gs://bucket/object#generation=<generation>\n  https://host/path#sha256=<64-hex>\n  --expected-sha256 <hex> Add an independent content digest requirement\n  --cache-dir <directory> Use a content-addressed cache directory\n  --offline               Refuse network access and require a verified cache receipt\n  --max-download-gib <n>  Bound one remote download (default: 50, maximum: 1024)\n");
  process.stdout.write("\nBounded scan policy:\n  --scan <mode>           auto, structure, integrity, or full\n  GGUF/SafeTensors use range reads; auto selects structure above 10 GiB and streamed integrity above 2 GiB.\n  Monolithic TFLite/ONNX/ExecuTorch files above 1 GiB fail before full-file allocation.\n");
  process.stdout.write("\nRepeat-review policy:\n  --review-policy <json> Bind required analysis coverage, finding threshold, and identity-scoped expiring exceptions\n  --policy-output <path> Write the deterministic policy decision JSON\n  --review-policy and --fail-on are mutually exclusive policy sources.\n");
  process.stdout.write("\nOutput format selection:\n  --output-format <kind>  Preferred unambiguous spelling for --format\n  --format <kind>         Backward-compatible alias; retained for existing automation\n");
  process.stdout.write("\nDeterministic graph export:\n  deepbom graph <artifact> --view structure --output-format svg -o graph.svg\n  --view <kind>           structure, placement, quantization, or architecture\n  --output-format <kind>  svg, png, html, mermaid, dot, or json for graph\n");
  process.stdout.write("\nTensorRT optimized-engine option:\n  --tensorrt-engine-inspector <json>\n                          Import identity-bound TensorRT optimized-engine inspector evidence\n");
  process.stdout.write("\nExecuTorch selected-build option:\n  --executorch-build <json>\n                          Bind backend/operator inventories and runtime binary digests to a PTE audit\n");
}

main(process.argv.slice(2)).catch((error) => {
  process.stderr.write(renderCliError(error));
  process.exitCode = 1;
});
