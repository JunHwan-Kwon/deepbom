import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { readCoreMlModelFile } from "../web/lib/coreml-metadata-adapter.js";
import { readMetadataModelFile } from "../web/lib/metadata-model-adapters.js";
import {
  PUBLIC_SAMPLE_MANIFEST_SCHEMA,
  PUBLIC_SAMPLE_MODELS,
  publicSampleManifestDocument,
  publicSampleModel,
} from "../web/lib/sample-models.js";
import { collectPublicSampleObservedEvidence, comparePublicSampleEvidence } from "../web/lib/sample-verification.js";
import { buildDeploymentContractDocuments } from "../web/lib/report-export-contracts.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildOnDeviceLlmContract } from "../web/lib/on-device-llm-contract.js";
import { buildTensorRtStaticPreflight } from "../web/lib/tensorrt-static-preflight.js";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";

function expect(value, message) {
  if (!value) throw new Error(message);
}

expect(PUBLIC_SAMPLE_MODELS.length === 11, "The public Try menu must contain five artifact formats plus three accelerator/LLM probes.");
expect(new Set(PUBLIC_SAMPLE_MODELS.map((sample) => sample.id)).size === PUBLIC_SAMPLE_MODELS.length, "Public sample ids must be unique.");
expect(new Set(PUBLIC_SAMPLE_MODELS.map((sample) => sample.path)).size === PUBLIC_SAMPLE_MODELS.length, "Public sample paths must be unique.");
expect(publicSampleModel("missing") === null, "Unknown public sample ids must fail closed.");
const manifest = publicSampleManifestDocument();
expect(manifest.schema === PUBLIC_SAMPLE_MANIFEST_SCHEMA, "Public sample expected-evidence schema changed unexpectedly.");
expect(manifest.examples.length === PUBLIC_SAMPLE_MODELS.length, "Public sample manifest must cover every selectable example exactly once.");
expect(/^[a-f0-9]{64}$/.test(manifest.analyzer_rulepack_sha256), "Public sample manifest must bind the analyzer rulepack SHA-256.");

function verifyExpectedEvidence(sample, analysis) {
  const expected = sample.expectedEvidence;
  const actual = collectPublicSampleObservedEvidence(sample.format, analysis);
  for (const [key, value] of Object.entries(expected)) {
    if (typeof value !== "number" && value !== null) continue;
    expect(Object.hasOwn(actual, key), `${sample.id} expected-evidence field ${key} has no independent analysis binding.`);
    expect(Object.is(actual[key], value), `${sample.id} expected ${key}=${value}; recalculated ${actual[key]}.`);
  }
}

initSync({ module: await readFile("pkg/tflite_wasm_audit_bg.wasm") });
for (const sample of PUBLIC_SAMPLE_MODELS) {
  expect(typeof sample.focus === "string" && sample.focus.length >= 24, `${sample.id} must state the analysis evidence it demonstrates.`);
  expect(typeof sample.purpose === "string" && sample.purpose.length >= 40, `${sample.id} must explain what the example teaches.`);
  expect(typeof sample.analysisDepth === "string" && sample.analysisDepth.length >= 12, `${sample.id} must declare its analysis depth.`);
  expect(sample.expectedEvidence.runtimeAssignment === "not_observed", `${sample.id} must not imply observed runtime assignment.`);
  expect(sample.expectedEvidence.applicableMetricFamilies.length > 2 && sample.expectedEvidence.notApplicableMetricFamilies.length > 2, `${sample.id} must disclose both assessed and outside-scope metric families.`);
  expect(!sample.expectedEvidence.applicableMetricFamilies.some((name) => sample.expectedEvidence.notApplicableMetricFamilies.includes(name)), `${sample.id} has an overlapping applicable/not-applicable family.`);
  const manifestEntry = manifest.examples.find((entry) => entry.example_id === sample.id);
  expect(manifestEntry?.artifact.sha256 === sample.sha256 && manifestEntry?.artifact.byte_length === sample.byteLength, `${sample.id} manifest identity diverged from the browser verifier.`);
  expect(Array.isArray(manifestEntry?.companions) && manifestEntry.companions.length === sample.companions.length, `${sample.id} manifest companion inventory diverged from the browser verifier.`);
  let bytes;
  if (sample.delivery === "remote") {
    const artifactUrl = new URL(sample.path);
    expect(artifactUrl.protocol === "https:" && artifactUrl.hostname === "storage.googleapis.com", `${sample.id} must use an official HTTPS Google Storage artifact URL.`);
    expect(/^\d+$/.test(artifactUrl.searchParams.get("generation") || ""), `${sample.id} must pin an immutable GCS generation.`);
    expect(/^https:\/\/developers\.google\.com\//.test(sample.source || ""), `${sample.id} must link its official model documentation.`);
    expect(/^[a-f0-9]{64}$/.test(sample.sha256) && sample.byteLength > 0, `${sample.id} must pin byte length and SHA-256.`);
    expect(sample.license === "Apache-2.0" || sample.license === "Not embedded (remote official asset)", `${sample.id} must report only source-backed license evidence.`);
    if (!process.argv.includes("--remote")) continue;
    const response = await fetch(sample.path);
    expect(response.ok, `${sample.id} remote artifact fetch failed (${response.status}).`);
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    const filePath = path.join("web", sample.path);
    bytes = new Uint8Array(await readFile(filePath));
  }
  expect(bytes.byteLength === sample.byteLength, `${sample.id} byte length does not match its declaration.`);
  expect(createHash("sha256").update(bytes).digest("hex") === sample.sha256, `${sample.id} SHA-256 does not match its declaration.`);
  expect(["Apache-2.0", "MIT", "Not embedded (remote official asset)"].includes(sample.license), `${sample.id} must declare its source-backed license state.`);
  let analysis;
  if (sample.format === "tflite") {
    analysis = analyze_tflite_for_target(bytes, sample.filename, "android_mid_a55");
    const arena = analysis.tensor_arena_plan;
    expect(Number(arena?.planned_tensor_count) === Number(arena?.root_allocation_count) + Number(arena?.shared_tensor_count), "Public TFLite arena root/alias ledger does not conserve planned tensors.");
    const arenaHighWater = (name) => Math.max(0, ...arena.allocations
      .filter((allocation) => allocation.arena === name && allocation.allocation_status === "allocated")
      .map((allocation) => Number(allocation.offset_bytes) + Number(allocation.size_bytes)));
    expect(arenaHighWater("kTfLiteArenaRw") === Number(arena.non_persistent_arena_bytes), "Public TFLite non-persistent arena ledger does not conserve its high-water mark.");
    expect(arenaHighWater("kTfLiteArenaRwPersistent") === Number(arena.persistent_arena_bytes), "Public TFLite persistent arena ledger does not conserve its high-water mark.");
    expect(Number(arena.non_persistent_arena_bytes) + Number(arena.persistent_arena_bytes) === Number(arena.combined_arena_bytes), "Public TFLite combined arena bytes do not conserve component arenas.");
  } else if (sample.format === "onnx") {
    analysis = analyzeOnnxModel(bytes, sample.filename, { id: "android_mid_a55", label: "Android mid", profile_sha256: "a".repeat(64), l1_data_bytes: 65_536 });
    analysis.model_sha256 = sample.sha256;
    analysis.on_device_llm = buildOnDeviceLlmContract(analysis);
    if (sample.companions.length) {
      const loaded = {};
      for (const companion of sample.companions) {
        const companionBytes = await readFile(path.join("web", companion.path));
        expect(companionBytes.byteLength === companion.byteLength, `${sample.id} ${companion.role} byte length changed.`);
        expect(createHash("sha256").update(companionBytes).digest("hex") === companion.sha256, `${sample.id} ${companion.role} SHA-256 changed.`);
        loaded[companion.role] = JSON.parse(companionBytes.toString("utf8"));
      }
      analysis.tensorrt_static_preflight = buildTensorRtStaticPreflight(
        analysis,
        loaded.tensorrt_build_profile,
        loaded.tensorrt_parser_observation,
      );
    }
    expect(Number(analysis.operator_count) === sample.expectedEvidence.operatorCount
      && Number(analysis.tensor_count) === sample.expectedEvidence.tensorCount, `${sample.id} ONNX graph identity changed.`);
    expect(Number(analysis.total_macs) === sample.expectedEvidence.totalMacs, `${sample.id} ONNX MAC conservation changed.`);
    expect(Number(analysis.size_breakdown?.constant_bytes) === sample.expectedEvidence.constantBytes, `${sample.id} ONNX initializer conservation changed.`);
    expect(Number(analysis.tensor_liveness?.peak_bytes) === sample.expectedEvidence.peakLiveBytes
      && Number(analysis.tensor_liveness?.peak_at_op) === sample.expectedEvidence.peakLiveAtOp, `${sample.id} ONNX live-payload projection changed.`);
  } else if (sample.format === "coreml") {
    analysis = (await readCoreMlModelFile(new File([bytes], sample.filename))).analysis;
    expect(analysis.coreml?.model_type === "neuralNetworkClassifier", "Public Core ML sample type changed.");
    expect(String(analysis.metadata_presence?.metadata_license || "").includes("LICENSE-MIT"), "Public Core ML sample does not retain its embedded MIT license reference.");
    expect(analysis.coreml?.description?.predicted_feature_name === "classLabel", "Public Core ML predicted feature binding changed.");
    expect(analysis.coreml?.description?.predicted_probabilities_name === "labelProbabilities", "Public Core ML probability binding changed.");
    expect(analysis.outputs?.find((output) => output.name === "labelProbabilities")?.constraints?.key_type === "INT64", "Public Core ML dictionary key contract changed.");
    expect(analysis.metadata_presence?.metadata_model_version === "1.0", "Public Core ML embedded model version changed.");
    expect(Number(analysis.total_macs) === 1_994_240 && analysis.mac_assessment?.status === "assessed_all_decoded_compute_ops", "Public Core ML source-backed MAC conservation changed.");
    expect(Number(analysis.size_breakdown?.constant_bytes) === 393_768
      && Number(analysis.size_breakdown?.constant_bytes) + Number(analysis.size_breakdown?.structure_overhead_bytes) === bytes.byteLength, "Public Core ML constant/file byte conservation changed.");
    expect(Number(analysis.tensor_liveness?.peak_bytes) === 100_352 && analysis.tensor_liveness?.status === "assessed", "Public Core ML static live-payload projection changed.");
  } else {
    analysis = (await readMetadataModelFile(new File([bytes], sample.filename), sample.format)).analysis;
    expect(Number(analysis.tensor_count) > 0, `${sample.id} tensor inventory is empty.`);
    if (sample.format === "gguf") {
      expect(Number(analysis.tensor_count) === 39 && Number(analysis.gguf?.declared_tensor_byte_length) === 507_392, "Public GGUF tensor-byte conservation changed.");
      expect(Number(analysis.gguf?.payload_byte_length) === 507_392 && analysis.gguf?.payload_coverage_status === "complete_without_gaps_or_overlaps", "Public GGUF payload coverage changed.");
      expect(analysis.tensor_numerical_integrity?.status === "assessed"
        && Number(analysis.tensor_numerical_integrity?.assessed_tensor_count) === 39
        && Number(analysis.tensor_numerical_integrity?.assessed_tensor_bytes) === 507_392
        && Number(analysis.tensor_numerical_integrity?.decoded_value_count) === 836_736
        && Number(analysis.tensor_numerical_integrity?.nonfinite_value_count) === 0
        && Number(analysis.tensor_numerical_integrity?.nonfinite_scale_block_count) === 0, "Public GGUF full source-pinned numerical decode changed.");
    } else {
      expect(Number(analysis.tensor_count) === 38 && Number(analysis.safetensors?.payload_byte_length) === 2_754_816, "Public SafeTensors payload conservation changed.");
      expect(analysis.tensor_numerical_integrity?.status === "assessed"
        && Number(analysis.tensor_numerical_integrity?.assessed_tensor_count) === 38
        && Number(analysis.tensor_numerical_integrity?.assessed_tensor_bytes) === 2_754_816
        && Number(analysis.tensor_numerical_integrity?.decoded_value_count) === 1_377_408
        && Number(analysis.tensor_numerical_integrity?.nonfinite_value_count) === 0, "Public SafeTensors full scalar numerical decode changed.");
    }
  }
  expect(analysis.format === sample.format, `${sample.id} parsed as ${analysis.format} instead of ${sample.format}.`);
  verifyExpectedEvidence(sample, analysis);
  analysis.model_sha256 = sample.sha256;
  const regression = comparePublicSampleEvidence(sample, analysis, {
    artifactSha256: sample.sha256,
    artifactByteLength: bytes.byteLength,
  });
  expect(regression.status === "pass" && regression.failed === 0 && regression.passed === regression.checks.length, `${sample.id} browser regression comparison did not pass its independently recalculated baseline.`);
  if (sample.format === "tflite" && sample.id === "tflite-mobilenet-v2-int8") {
    const tampered = comparePublicSampleEvidence(sample, { ...analysis, total_macs: Number(analysis.total_macs) + 1 }, {
      artifactSha256: sample.sha256,
      artifactByteLength: bytes.byteLength,
    });
    expect(tampered.status === "fail" && tampered.checks.some((row) => row.label === "Assessed MACs" && row.status === "fail"), "Public example verifier must fail closed on a recalculated evidence mismatch.");
  }
  const documents = buildDeploymentContractDocuments(analysis, { hash: sample.sha256, fileSizeBytes: bytes.byteLength, generatedAt: "2026-08-04T00:00:00.000Z" });
  const expectedSchema = {
    tflite: "TFLite schema 3",
    onnx: `IR ${analysis.onnx_ir_version} / opset ${(analysis.opsets || []).map((opset) => `${opset.domain || "ai.onnx"}:${opset.version}`).join(" / ")}`,
    gguf: "GGUF v3",
    safetensors: "SafeTensors format (unversioned)",
    coreml: "Core ML specification 1",
  }[sample.format];
  expect(documents.subject.byte_length === bytes.byteLength && documents.subject.sha256 === sample.sha256, `${sample.id} export identity is not byte-bound.`);
  expect(documents.subject.schema_or_opset === expectedSchema, `${sample.id} export schema identity is incorrect.`);
  const envelope = documents.documents.artifact_evidence_envelope;
  expect(envelope.identity.byte_length === bytes.byteLength && envelope.identity.sha256 === sample.sha256, `${sample.id} canonical evidence identity diverged from its export subject.`);
  const expectedOperatorCount = ["gguf", "safetensors"].includes(sample.format) ? null : (analysis.operator_count ?? analysis.ops?.length ?? null);
  expect(envelope.graph.operator_count === expectedOperatorCount, `${sample.id} exported operator count diverged from analysis.`);
  expect(envelope.graph.tensor_count === (analysis.tensor_count ?? analysis.tensors?.length ?? null), `${sample.id} exported tensor count diverged from analysis.`);
  expect(envelope.graph.total_macs === (analysis.total_macs ?? null), `${sample.id} exported MAC total diverged from analysis or collapsed an unassessed value to zero.`);
  if (sample.companions.length) {
    expect(envelope.format_extensions.onnx?.tensorrt_static_preflight?.schema === "deepbom.tensorrt_static_preflight.v1", `${sample.id} canonical evidence omitted TensorRT preflight.`);
    expect(envelope.format_extensions.onnx.tensorrt_static_preflight.projection?.state_counts?.CONDITIONALLY_ELIGIBLE
      === sample.expectedEvidence.tensorRtConditionallyEligibleOps, `${sample.id} canonical TensorRT state count diverged.`);
    expect(envelope.capabilities.assessed.includes("tensorrt_static_preflight")
      || envelope.capabilities.partial.includes("tensorrt_static_preflight"), `${sample.id} canonical capability manifest omitted TensorRT assessment state.`);
    const properties = new Map((documents.documents.cyclonedx_evidence.metadata?.component?.properties || []).map((row) => [row.name, row.value]));
    expect(properties.get("deepbom:model:tensorRtConditionallyEligibleOperatorCount")
      === String(sample.expectedEvidence.tensorRtConditionallyEligibleOps), `${sample.id} CycloneDX TensorRT count diverged.`);
    expect(properties.get("deepbom:model:tensorRtCollectorGitCommit") === "f3b8fb5adab237e490dbb930bd708049eb2764d8", `${sample.id} CycloneDX collector provenance diverged.`);
  }
  if (sample.id === "onnx-tiny-decoder-llm") {
    expect(envelope.format_extensions.onnx?.on_device_llm?.serialized_graph?.primitive_counts?.matrix_multiply === 8, "Tiny decoder LLM evidence was omitted from the canonical envelope.");
    expect(envelope.capabilities.partial.includes("llm_serialized_graph"), "Tiny decoder LLM motif must remain partial rather than being promoted to architecture proof.");
  }
  if (sample.format === "tflite") {
    expect(envelope.graph.mac_assessment_status === "assessed" && envelope.capabilities.assessed.includes("static_cost"), "Public TFLite deterministic MAC result is mislabeled not assessed in canonical evidence.");
  }
  if (["gguf", "safetensors"].includes(sample.format)) {
    expect(envelope.graph.total_macs === null, `${sample.id} canonical evidence fabricated a zero MAC calculation.`);
    expect(envelope.capabilities.assessed.includes("tensor_payloads")
      && envelope.format_extensions[sample.format]?.tensor_numerical_integrity?.status === "assessed", `${sample.id} canonical evidence omitted the full tensor-payload assessment.`);
  }
  if (sample.format === "coreml") {
    expect(envelope.graph.total_macs === analysis.total_macs && envelope.graph.mac_assessment_status === analysis.mac_assessment.status, "Public Core ML canonical evidence lost its exact serialized-graph MAC result.");
    expect(["graph", "tensor_payloads", "static_cost"].every((capability) => envelope.capabilities.assessed.includes(capability))
      && envelope.format_extensions.coreml?.weight_integrity?.status === "assessed"
      && envelope.format_extensions.coreml?.tensor_liveness?.status === "assessed", "Public Core ML canonical evidence omitted graph, payload, cost, or liveness evidence.");
  }
  if (["gguf", "safetensors"].includes(sample.format)) {
    expect(envelope.graph.operator_count === null && envelope.graph.mac_assessment_status === "not_applicable_weight_container", `${sample.id} weight-container status is not preserved in canonical evidence.`);
  }
  const expectedRuntime = {
    tflite: "TensorFlow Lite / LiteRT",
    onnx: "ONNX Runtime",
    gguf: "GGUF-compatible runtime (unbound)",
    safetensors: "SafeTensors-compatible loader (unbound)",
    coreml: "Core ML",
  }[sample.format];
  expect(documents.documents.runtime_requirement_manifest.necessary_runtime_floor.runtime === expectedRuntime, `${sample.id} export names the wrong runtime family.`);
}

console.log(`Public verified examples passed (5 local public artifacts, 3 deterministic accelerator/LLM probes, and 3 generation-pinned remote TFLite profiles${process.argv.includes("--remote") ? "; remote values recalculated" : "; remote identity-only mode"}).`);
