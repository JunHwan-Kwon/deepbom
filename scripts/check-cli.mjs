import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { loadCliInput, loadOnnxExternalData } from "../bin/deepbom-input.mjs";
import { createTensorRtBuildProfile, TENSORRT_PARSER_OBSERVATION_SCHEMA } from "../web/lib/tensorrt-static-preflight.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";
import { buildInterfaceQuantizationContractLedger } from "../web/lib/quantization-contract-summary.js";
import { EXECUTORCH_BACKEND_REGISTRY_SOURCE, EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA } from "../web/lib/executorch-build-binding.js";
import { TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA, tensorRtParserObservationIdentity } from "../web/lib/tensorrt-engine-inspector.js";
import { buildCoreMlComputePlanTemplate } from "../web/lib/coreml-compute-plan.js";
import { COREML_DEPLOYMENT_SOURCE } from "../web/lib/coreml-deployment-contract.js";
import { tfliteAcceleratorSourceManifest } from "../web/lib/tflite-accelerator-source-profiles.js";
import { decodeFixtureBase64, EXECUTORCH_ADD_PTE_BASE64 } from "./fixtures/executorch-fixtures.mjs";

const cases = [
  ["web/samples/mobilenet_v2_1.0_224_quant.tflite", "tflite"],
  ["web/samples/sample_cnn_float.onnx", "onnx"],
  ["web/samples/tinymqa1m.Q4_0.gguf", "gguf"],
  ["web/samples/nanofable-1m-fp16.safetensors", "safetensors"],
  ["web/samples/MNISTClassifier.mlmodel", "coreml"],
];

const emptyExecuTorchPtd = Buffer.from(
  "FAAAAEZUMDEAAAoAEAAEAAgADAAKAAAAAQAAAAgAAAAUAAAAAQAAAAgAAAAEAAQABAAAAAEAAAAMAAAAAAAGAAgABAAGAAAABAAAAAsAAAB3ZWlnaHRzLmJpbgA=",
  "base64",
);

const streamedGgufInput = await loadCliInput(path.resolve(cases[2][0]));
assert.equal(streamedGgufInput.kind, "file", "GGUF CLI input kind");
assert.equal(Object.hasOwn(streamedGgufInput, "bytes"), false, "GGUF input must remain disk-backed before analysis");
assert.equal(streamedGgufInput.prefix.byteLength <= 4096, true, "CLI format sniff is bounded");

const humanSummary = run(["audit", cases[1][0]]).stdout;
assert.match(humanSummary, /^DEEPBOM \S+ deployment-artifact audit/m, "default CLI output is human-readable");
assert.match(humanSummary, /Graph: 9 operators \| 16 tensors \| 6,488,384 MACs/, "human summary projects exact graph totals");
assert.match(humanSummary, /Evidence boundary:/, "human summary states its evidence boundary");
assert.equal(Buffer.byteLength(humanSummary, "utf8") < 8192, true, "human summary remains terminal-sized");
const incompleteMacSummary = run(["audit", "scripts/fixtures/onnx_dynamic_conv.onnx"]).stdout;
assert.match(incompleteMacSummary, /Graph: 1 operators \| 3 tensors \| MACs not assessable/, "human summary preserves an incomplete ONNX MAC total as not assessable");
assert.doesNotMatch(incompleteMacSummary, /\| 0 MACs/, "human summary must not coerce an unassessed ONNX MAC total to zero");
assert.equal(JSON.parse(run(["audit", cases[1][0], "--json"]).stdout).format, "onnx", "--json retains complete formatted machine output");
assert.match(run(["--help"]).stdout, /--json\s+Compatibility alias for --output-format json/, "JSON mode is discoverable");
assert.match(run(["--help"]).stdout, /deepbom verify <artifact> --contract <json>/, "verify command is discoverable");
assert.match(run(["--help"]).stdout, /deepbom diff <baseline\.tflite> <candidate\.tflite>/, "diff command is discoverable");
assert.match(run(["--help"]).stdout, /deepbom explore <artifact\.tflite>/, "explore command is discoverable");
assert.match(run(["--help"]).stdout, /--executorch-build <json>/, "ExecuTorch selected-build binding is discoverable");
assert.match(run(["--help"]).stdout, /--tensorrt-engine-inspector <json>/, "TensorRT optimized-engine evidence import is discoverable");
assert.match(run(["--help"]).stdout, /deepbom accelerator collect nvidia/, "NVIDIA accelerator observation is discoverable");
assert.match(run(["--help"]).stdout, /--include-device-identifiers/, "NVIDIA identifier privacy option is discoverable");
assert.match(run(["--help"]).stdout, /hf:\/\/owner\/repo@<40-hex-commit>\/path/, "immutable Hugging Face input is discoverable");
assert.match(run(["--help"]).stdout, /--offline\s+Refuse network access/, "offline cache behavior is discoverable");
assert.match(run(["--help"]).stdout, /deepbom graph <artifact>/, "deterministic graph export is discoverable");
assert.match(run(["--help"]).stdout, /--scan <mode>\s+auto, structure, integrity, or full/, "bounded scan modes are discoverable");
assert.match(run(["--help"]).stdout, /--accelerator-profile <json>/, "NVIDIA profile binding is discoverable");
assert.match(run(["--help"]).stdout, /--coreml-compute-plan <json>/, "Core ML plan binding is discoverable");
assert.match(run(["--help"]).stdout, /--edgetpu-compiler-evidence <json>/, "Edge TPU compiler binding is discoverable");
assert.match(run(["--help"]).stdout, /deepbom placement <artifact>/, "N-way placement comparison is discoverable");
assert.match(run(["--help"]).stdout, /--litert-qualcomm-evidence <json>/, "LiteRT Qualcomm compiler binding is discoverable");
assert.match(run(["--help"]).stdout, /--review-policy <json>/, "repeat-review policy is discoverable");
assert.match(run(["--help"]).stdout, /--images <count>/, "multimodal image-count scenario is discoverable");
assert.match(run(["--help"]).stdout, /--tokens-per-image <count>/, "explicit projector token budget is discoverable");

for (const [artifact, expectedFormat] of cases) {
  const result = run(["audit", artifact, "--compact"]);
  const document = JSON.parse(result.stdout);
  assert.equal(document.format, expectedFormat, `${artifact} format`);
  assert.equal(document.filename, path.basename(artifact), `${artifact} filename`);
  assert.match(document.model_sha256, /^[a-f0-9]{64}$/, `${artifact} SHA-256`);
  assert.equal(document.file_size_bytes > 0, true, `${artifact} byte size`);
  assert.equal(document.artifact_set?.schema, "deepbom.artifact_set.v1", `${artifact} artifact-set schema`);
  assert.equal(document.artifact_set?.files?.[0]?.sha256, document.model_sha256, `${artifact} artifact-set primary identity`);
}

const gguf = JSON.parse(run(["gguf", cases[2][0], "--compact"]).stdout);
assert.equal(gguf.gguf?.tensor_count > 0, true, "GGUF command tensor inventory");
const ggufStructure = JSON.parse(run(["gguf", cases[2][0], "--scan", "structure", "--compact"]).stdout);
assert.equal(ggufStructure.cli_scan_policy?.effective_mode, "structure", "GGUF structure scan policy");
assert.equal(ggufStructure.tensor_numerical_integrity?.status, "not_assessed_scan_policy_structure", "structure mode must not imply payload integrity");
assert.equal(ggufStructure.tensor_numerical_integrity?.assessed_tensor_count, 0, "structure mode decoded payload count");
assert.equal(ggufStructure.tensor_numerical_integrity?.unassessed_tensor_count, ggufStructure.tensor_count, "structure-mode unassessed tensor conservation");
const invalidOnnxScan = run(["audit", cases[1][0], "--scan", "structure"], false);
assert.notEqual(invalidOnnxScan.status, 0);
assert.match(invalidOnnxScan.stderr, /does not provide a truthful structure scan path/);
const ggufScenario = JSON.parse(run(["gguf", cases[2][0], "--context", "8192", "--batch", "2", "--state-bits", "8", "--memory-mib", "1", "--compact"]).stdout);
assert.equal(ggufScenario.cli_context_scenario?.schema, "deepbom.llm_token_budget_scenario.v1", "GGUF CLI uses the common LLM token-budget schema");
assert.equal(ggufScenario.cli_context_scenario?.context_length, 8192, "GGUF context scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.batch_size, 2, "GGUF batch scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.state_storage_bits, 8, "GGUF state-width scenario binding");
assert.equal(ggufScenario.cli_context_scenario?.context_source, "cli_argument", "GGUF context scenario source");
assert.match(ggufScenario.cli_context_scenario?.memory_feasibility?.status, /^lower_bound_(exceeds_capacity|at_or_below_capacity_fit_unresolved)$/, "GGUF memory-capacity lower-bound classification");
assert.match(ggufScenario.cli_context_scenario?.memory_feasibility?.residency_assumption, /simultaneously resident/, "GGUF memory scenario emits its residency assumption");
assert.equal(ggufScenario.cli_context_scenario?.memory_feasibility?.fit_claim, "not_emitted", "GGUF memory scenario must not emit a fit claim");
assert.equal(BigInt(ggufScenario.cli_context_scenario.memory_feasibility.static_lower_bound_bytes.decimal),
  BigInt(ggufScenario.cli_context_scenario.memory_feasibility.serialized_weight_floor_bytes.decimal)
    + BigInt(ggufScenario.cli_context_scenario.memory_feasibility.logical_kv_state_bytes.decimal), "GGUF memory lower-bound conservation");
assert.equal(ggufScenario.gguf?.semantic_contract?.context_length, gguf.gguf?.semantic_contract?.context_length, "GGUF scenario must not mutate serialized context");
const ggufImageScenario = JSON.parse(run(["gguf", cases[2][0], "--context", "1024", "--images", "2", "--tokens-per-image", "64", "--compact"]).stdout);
assert.equal(ggufImageScenario.cli_context_scenario?.token_budget?.text_tokens, 1024, "declared text-token count");
assert.equal(ggufImageScenario.cli_context_scenario?.token_budget?.image_tokens?.decimal, "128", "declared image-token product");
assert.equal(ggufImageScenario.cli_context_scenario?.token_budget?.total_context_tokens?.decimal, "1152", "total multimodal context conservation");
assert.equal(ggufImageScenario.cli_context_scenario?.context_length, 1152, "legacy context alias is the total token budget");
assert.match(ggufImageScenario.cli_context_scenario?.scenario_sha256, /^[a-f0-9]{64}$/, "token-budget scenario identity");
const missingImageWidth = run(["gguf", cases[2][0], "--context", "1024", "--images", "2"], false);
assert.notEqual(missingImageWidth.status, 0);
assert.match(missingImageWidth.stderr, /--images and --tokens-per-image must be provided together/);
assert.match(run(["gguf", "--help"]).stdout, /--context <tokens>/, "subcommand help");
assert.match(run(["gguf", "--help"]).stdout, /--memory-mib <MiB>/, "GGUF memory scenario help");
assert.match(run(["gguf", "--help"]).stdout, /--llm-memory-profile <json>/, "LLM static pool profile help");

const ggufScenarioCycloneDx = JSON.parse(run(["gguf", cases[2][0], "--context", "8192", "--batch", "2", "--state-bits", "8", "--memory-mib", "1", "--format", "cyclonedx", "--timestamp", "2026-08-18T00:00:00.000Z", "--compact"]).stdout);
const ggufScenarioProperties = new Map(ggufScenarioCycloneDx.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioContextLength"), "8192", "CycloneDX retains CLI context scenario");
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioBatchSize"), "2", "CycloneDX retains CLI batch scenario");
assert.match(ggufScenarioProperties.get("deepbom:model:llmCliScenarioResidencyAssumption"), /simultaneously resident/, "CycloneDX retains CLI residency assumption");
assert.equal(ggufScenarioProperties.get("deepbom:model:llmCliScenarioMemoryFitClaim"), "not_emitted", "CycloneDX retains lower-bound-only memory claim boundary");

const ggufImageScenarioCycloneDx = JSON.parse(run(["gguf", cases[2][0], "--context", "1024", "--images", "2", "--tokens-per-image", "64", "--format", "cyclonedx", "--timestamp", "2026-08-18T00:00:00.000Z", "--compact"]).stdout);
const ggufImageProperties = new Map(ggufImageScenarioCycloneDx.metadata.component.properties.map((row) => [row.name, row.value]));
assert.equal(ggufImageProperties.get("deepbom:model:llmCliScenarioImageCount"), "2", "CycloneDX retains image count");
assert.equal(ggufImageProperties.get("deepbom:model:llmCliScenarioTokensPerImage"), "64", "CycloneDX retains projector token width");
assert.equal(ggufImageProperties.get("deepbom:model:llmCliScenarioTotalContextTokens"), "1152", "CycloneDX retains total multimodal context");

const timestamp = "2026-08-18T00:00:00.000Z";
const cyclonedx = JSON.parse(run([
  "audit",
  cases[1][0],
  "--format",
  "cyclonedx",
  "--timestamp",
  timestamp,
  "--compact",
]).stdout);
assert.equal(cyclonedx.bomFormat, "CycloneDX");
assert.equal(cyclonedx.specVersion, "1.7");
assert.equal(cyclonedx.metadata.timestamp, timestamp);

const temp = await mkdtemp(path.join(tmpdir(), "deepbom-cli-tensorrt-"));
try {
  assert.equal(gguf.on_device_llm?.storage?.layer_storage?.status, "assessed_exact_serialized_layer_storage", "GGUF CLI fixture exact layer ledger");
  const ggufSerializedBytes = BigInt(gguf.on_device_llm.storage.serialized_tensor_bytes_decimal);
  const memoryProfilePath = path.join(temp, "memory-profile.json");
  await writeFile(memoryProfilePath, JSON.stringify({
    schema: "deepbom.llm_static_memory_profile.v1",
    artifact: { format: "gguf", sha256: gguf.model_sha256 },
    capacities: { cpu_bytes: String(ggufSerializedBytes * 4n), accelerator_bytes: String(ggufSerializedBytes * 4n) },
    reserves: { cpu_bytes: "0", accelerator_bytes: "0" },
    policy: {
      layer_order: "highest_index_first", non_layer_pool: "cpu", state_pool: "accelerator",
      context_length: gguf.on_device_llm.architecture.context_length, batch_size: 1, state_storage_bits: 16,
    },
  }), "utf8");
  const placedGguf = JSON.parse(run(["gguf", cases[2][0], "--llm-memory-profile", memoryProfilePath, "--compact"]).stdout);
  assert.equal(placedGguf.on_device_llm.static_memory_placement.status, "assessed_lower_bound_candidates", "GGUF conditional static pool placement");
  assert.equal(placedGguf.on_device_llm.static_memory_placement.maximum_accelerator_layer_count_not_disproven,
    placedGguf.on_device_llm.architecture.layer_count, "GGUF static pool candidate enumeration");
  assert.equal(placedGguf.on_device_llm.static_memory_placement.fit_claim, "not_emitted", "GGUF static pool placement must not claim fit");

  const ptdPath = path.join(temp, "weights.ptd");
  await writeFile(ptdPath, emptyExecuTorchPtd);
  const ptd = JSON.parse(run(["audit", ptdPath, "--compact"]).stdout);
  assert.equal(ptd.format, "executorch", "ExecuTorch PTD CLI format");
  assert.equal(ptd.executorch_container, "ptd", "ExecuTorch PTD container identity");
  assert.equal(ptd.tensor_count, 1, "ExecuTorch PTD tensor inventory");
  const ptdCycloneDx = JSON.parse(run(["audit", ptdPath, "--format", "cyclonedx", "--compact"]).stdout);
  assert.equal(ptdCycloneDx.bomFormat, "CycloneDX", "ExecuTorch PTD CycloneDX export");
  assert.match(ptdCycloneDx.serialNumber, /^urn:uuid:/, "ExecuTorch PTD CycloneDX serial number");

  const ptePath = path.join(temp, "add.pte");
  await writeFile(ptePath, decodeFixtureBase64(EXECUTORCH_ADD_PTE_BASE64));
  const execuTorchBuildPath = path.join(temp, "selected-build.json");
  await writeFile(execuTorchBuildPath, JSON.stringify(buildExecuTorchAttestation()), "utf8");
  const buildBoundPte = JSON.parse(run(["audit", ptePath, "--executorch-build", execuTorchBuildPath, "--compact"]).stdout);
  assert.equal(buildBoundPte.executorch_program.selected_build_binding.status,
    "SELECTED_BUILD_INVENTORY_SATISFIES_SERIALIZED_IDENTITIES", "ExecuTorch CLI selected-build inventory binding");
  assert.equal(buildBoundPte.executorch_program.selected_build_binding.selected_build_input.path,
    "selected-build.json", "ExecuTorch CLI selected-build source identity");
  assert.equal(buildBoundPte.executorch_program.selected_build_binding.selected_build_input.duplicate_key_validation,
    "complete", "ExecuTorch CLI selected-build duplicate-key validation");
  const wrongExecuTorchBuildFormat = run(["audit", cases[1][0], "--executorch-build", execuTorchBuildPath], false);
  assert.notEqual(wrongExecuTorchBuildFormat.status, 0);
  assert.match(wrongExecuTorchBuildFormat.stderr, /applies only to an ExecuTorch PTE/);
  const ptdExecuTorchBuild = run(["audit", ptdPath, "--executorch-build", execuTorchBuildPath], false);
  assert.notEqual(ptdExecuTorchBuild.status, 0);
  assert.match(ptdExecuTorchBuild.stderr, /PTD files do not accept|applies only to an ExecuTorch PTE/);

  const externalDataPath = path.join(temp, "weights.bin");
  await writeFile(externalDataPath, "external-weight-bytes", "utf8");
  const externalData = await loadOnnxExternalData(path.join(temp, "model.onnx"), {
    onnx_external_data: { tensors: [{ normalized_location: "weights.bin" }] },
  });
  assert.equal(externalData.length, 1, "ONNX external-data discovery");
  assert.equal(externalData[0].path, "weights.bin", "ONNX external-data canonical model-relative path");
  assert.equal(externalData[0].bytes.byteLength, 21, "ONNX external-data byte length");
  assert.equal(externalData[0].sha256, sha256TextHex("external-weight-bytes"), "ONNX external-data SHA-256");
  await assert.rejects(
    loadOnnxExternalData(path.join(temp, "model.onnx"), {
      onnx_external_data: { tensors: [{ normalized_location: "../weights.bin" }] },
    }),
    /not a safe relative path/,
    "ONNX external-data traversal must fail closed",
  );

  const onnxAnalysis = JSON.parse(run(["audit", cases[1][0], "--compact"]).stdout);
  const tfliteAnalysis = JSON.parse(run(["audit", cases[0][0], "--compact"]).stdout);
  const coreMlAnalysis = JSON.parse(run(["audit", cases[4][0], "--compact"]).stdout);
  assert.equal(tfliteAnalysis.cpu_cost_target_binding?.binding_source, "default_assumption", "default TFLite target must remain an explicit planning assumption");
  assert.equal(tfliteAnalysis.cpu_cost_target_binding?.host_observed, false, "CPU cost target must never claim host observation");
  assert.equal(tfliteAnalysis.cpu_cost_target_binding?.profile_sha256, tfliteAnalysis.target_profile.profile_sha256,
    "CPU cost target binding must retain the resolved profile digest");
  const explicitTarget = JSON.parse(run(["audit", cases[0][0], "--target", "android_mid_a55", "--compact"]).stdout);
  assert.equal(explicitTarget.cpu_cost_target_binding?.binding_source, "explicit_id", "an explicit built-in target id must be distinguishable from the default");
  assert.equal(explicitTarget.cpu_cost_target_binding?.profile_sha256, tfliteAnalysis.cpu_cost_target_binding.profile_sha256,
    "binding provenance must not change the selected target calculation");
  const targetEnvelope = JSON.parse(run(["audit", cases[0][0], "--format", "envelope", "--compact"]).stdout);
  assert.equal(targetEnvelope.cpu_cost_target_binding?.binding_source, "default_assumption", "canonical envelope must retain CPU target provenance");
  assert.equal(Array.isArray(targetEnvelope.accelerator_bindings), true, "canonical envelope must carry the accelerator binding ledger");
  assert.equal(targetEnvelope.accelerator_bindings.some((row) => row.profile_id === "tflite_coreml_delegate"), true,
    "canonical envelope must retain the source-pinned TFLite Core ML profile");
  assert.equal(targetEnvelope.accelerator_bindings.some((row) => row.profile_id === "litert_qualcomm_qnn"), true,
    "canonical envelope must retain the source-pinned LiteRT Qualcomm profile");

  const placement = JSON.parse(run(["placement", cases[0][0], "--compact"]).stdout);
  assert.equal(placement.schema, "deepbom.placement_comparison.v1", "placement CLI schema");
  assert.deepEqual(placement.available_profile_ids, ["xnnpack_cpu", "tflite_coreml_delegate", "litert_qualcomm_qnn"],
    "placement CLI must expose independent source profiles in deterministic order");
  const selectedPlacement = JSON.parse(run(["placement", cases[0][0], "--profiles", "xnnpack_cpu,tflite_coreml_delegate", "--compact"]).stdout);
  assert.deepEqual(selectedPlacement.selected_profile_ids, ["xnnpack_cpu", "tflite_coreml_delegate"], "placement CLI N-way selection");
  const unknownPlacement = run(["placement", cases[0][0], "--profiles", "missing"], false);
  assert.notEqual(unknownPlacement.status, 0, "unknown placement profile must fail closed");
  assert.match(unknownPlacement.stderr, /unavailable/);

  const reviewPolicyPath = path.join(temp, "review-policy.json");
  const reviewPolicyOutputPath = path.join(temp, "review-policy-result.json");
  await writeFile(reviewPolicyPath, JSON.stringify({
    schema: "deepbom.review_policy.v1",
    mode: "enforce",
    fail_on: "none",
    required_capabilities: ["artifact_identity", "graph"],
    exceptions: [],
  }), "utf8");
  const policyAudit = JSON.parse(run(["audit", cases[0][0], "--review-policy", reviewPolicyPath,
    "--policy-output", reviewPolicyOutputPath, "--format", "envelope", "--compact"]).stdout);
  assert.equal(policyAudit.policy_identity?.schema, "deepbom.review_policy.v1", "review policy identity must enter the envelope");
  assert.match(policyAudit.policy_identity?.policy_sha256, /^[a-f0-9]{64}$/, "review policy normalized digest");
  const policyConflict = run(["audit", cases[0][0], "--review-policy", reviewPolicyPath, "--fail-on", "high"], false);
  assert.notEqual(policyConflict.status, 0, "review policy and legacy threshold must be mutually exclusive");
  assert.match(policyConflict.stderr, /mutually exclusive/);

  const edgeTpuPath = path.join(temp, "edgetpu-compiler-evidence.json");
  await writeFile(edgeTpuPath, JSON.stringify({
    schema: "deepbom.edgetpu_compiler_evidence.v1",
    artifact_sha256: tfliteAnalysis.model_sha256,
    compiler: { name: "edgetpu_compiler", version: "fixture", binary_sha256: "6".repeat(64) },
    invocation: { options: ["--delegate_search_step=1"] },
    compiled_artifact_sha256: "7".repeat(64),
    compiler_report_sha256: "8".repeat(64),
    operations: tfliteAnalysis.ops.map((op, index) => ({
      op_index: op.index ?? index,
      op_name: op.name,
      mapping: index === tfliteAnalysis.ops.length - 1 ? "unmapped" : "mapped",
      reason: index === tfliteAnalysis.ops.length - 1 ? "fixture_compiler_report" : null,
    })),
  }), "utf8");
  const edgeTpuBound = JSON.parse(run(["audit", cases[0][0], "--edgetpu-compiler-evidence", edgeTpuPath, "--compact"]).stdout);
  const edgeTpuBinding = edgeTpuBound.accelerator_bindings.find((row) => row.profile_id === "google_edgetpu_compiler");
  assert.equal(edgeTpuBinding?.evidence_stage, "compiled_plan", "Edge TPU compiler evidence stage");
  assert.equal(edgeTpuBinding?.claims?.observed_assignment, false, "Edge TPU compiler report must not become observed assignment");
  assert.equal(edgeTpuBound.edgetpu_compiler_evidence.summary.operation_count, tfliteAnalysis.ops.length,
    "Edge TPU operation classification conservation");
  const wrongEdgeTpu = run(["audit", cases[1][0], "--edgetpu-compiler-evidence", edgeTpuPath], false);
  assert.notEqual(wrongEdgeTpu.status, 0);
  assert.match(wrongEdgeTpu.stderr, /only to a TFLite artifact/);

  const qualcommSource = tfliteAcceleratorSourceManifest().profiles.find((row) => row.id === "litert_qualcomm_qnn");
  const qualcommPath = path.join(temp, "litert-qualcomm-evidence.json");
  await writeFile(qualcommPath, JSON.stringify({
    schema: "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1",
    artifact_sha256: tfliteAnalysis.model_sha256,
    source: { litert_commit: qualcommSource.source.commit, rulepack_sha256: qualcommSource.rulepack_sha256 },
    compiler: { name: "litert-qualcomm-compiler", version: "fixture", binary_sha256: "1".repeat(64) },
    invocation: { options: ["--soc_model=fixture"] },
    compiled_plan_sha256: "2".repeat(64),
    operations: tfliteAnalysis.ops.map((op, index) => ({
      op_index: op.index ?? index,
      op_name: op.name,
      compile_status: index === tfliteAnalysis.ops.length - 1 ? "not_compiled" : "compiled",
      reason: index === tfliteAnalysis.ops.length - 1 ? "fixture_compiler_rejection" : null,
    })),
    dispatch: { status: "not_observed" },
  }), "utf8");
  const qualcommBound = JSON.parse(run(["audit", cases[0][0], "--litert-qualcomm-evidence", qualcommPath, "--compact"]).stdout);
  const qualcommBinding = qualcommBound.accelerator_bindings.find((row) => row.profile_id === "litert_qualcomm_qnn_compiled_plan");
  assert.equal(qualcommBinding?.evidence_stage, "compiled_plan", "LiteRT Qualcomm compiled plan evidence stage");
  assert.equal(qualcommBinding?.claims?.observed_assignment, false, "Qualcomm compiler result must not become observed assignment");
  const qualcommPlacement = JSON.parse(run(["placement", cases[0][0], "--litert-qualcomm-evidence", qualcommPath,
    "--profiles", "litert_qualcomm_qnn_compiled_plan", "--compact"]).stdout);
  assert.equal(qualcommPlacement.rows[0].conditionally_eligible_ops, tfliteAnalysis.ops.length - 1,
    "Qualcomm imported compiler projection must conserve the operation ledger");
  const wrongQualcomm = run(["placement", cases[1][0], "--litert-qualcomm-evidence", qualcommPath], false);
  assert.notEqual(wrongQualcomm.status, 0);
  assert.match(wrongQualcomm.stderr, /only to a TFLite artifact/);

  const coreMlTemplate = buildCoreMlComputePlanTemplate(coreMlAnalysis);
  coreMlTemplate.runtime = {
    coremltools_version: "9.0-fixture",
    coremltools_compute_plan_source_sha256: COREML_DEPLOYMENT_SOURCE.compute_plan_sha256,
    compiled_model_content_sha256: "9".repeat(64),
    platform: "macOS fixture",
    architecture: "arm64",
    platform_system: "Darwin",
    macos_version: "15.6",
    os_build: "24G84",
    hardware_model: "Mac-fixture",
    python_version: "3.12",
    available_compute_devices: [{ type: "CPU", source_class: "MLCPUComputeDevice", instance_count: 1 }],
  };
  coreMlTemplate.capture.capture_id = "cli-coreml-plan";
  coreMlTemplate.capture.collected_at = "2026-08-18T00:00:00.000Z";
  coreMlTemplate.capture.collector.source_sha256 = "a".repeat(64);
  coreMlTemplate.structure.rows = coreMlAnalysis.ops.map((op, index) => ({
    op_index: op.index ?? index,
    operator_type: coreMlTemplate.structure.kind === "program" ? op.mil_operation_type : op.name,
    identity: op.coreml_layer_name,
    preferred_compute_device: "CPU",
    supported_compute_devices: ["CPU"],
    estimated_cost_weight: coreMlAnalysis.ops.length ? 1 / coreMlAnalysis.ops.length : null,
  }));
  const coreMlPlanPath = path.join(temp, "coreml-plan.json");
  await writeFile(coreMlPlanPath, JSON.stringify(coreMlTemplate), "utf8");
  const coreMlBound = JSON.parse(run(["audit", cases[4][0], "--coreml-compute-plan", coreMlPlanPath, "--compact"]).stdout);
  const coreMlBinding = coreMlBound.accelerator_bindings.find((row) => row.provider === "apple");
  assert.equal(coreMlBinding?.evidence_stage, "compiled_plan", "Core ML compute plan evidence stage");
  assert.equal(coreMlBinding?.claims?.observed_assignment, false, "MLComputePlan must remain anticipated rather than observed assignment");
  const interfaceLedger = buildInterfaceQuantizationContractLedger(tfliteAnalysis);
  const interfaceContractPath = path.join(temp, "interface-contract.json");
  await writeFile(interfaceContractPath, JSON.stringify({
    schema: "deepbom.production_interface_contract.v1",
    artifact_sha256: tfliteAnalysis.model_sha256,
    implementation_sha256: "a".repeat(64),
    parameters: interfaceLedger.parameters,
  }), "utf8");
  const verified = JSON.parse(run(["verify", cases[0][0], "--contract", interfaceContractPath, "--compact"]).stdout);
  assert.equal(verified.schema, "deepbom.cli_interface_contract_verification.v1", "verify evidence schema");
  assert.equal(verified.comparison.status, "bound_exact_contract", "verify exact contract status");
  assert.equal(verified.comparison.gate_result, "pass", "verify exact contract gate");
  assert.match(run(["verify", cases[0][0], "--contract", interfaceContractPath]).stdout, /Result: PASS \| bound_exact_contract/, "verify human summary");

  const mismatchedContractPath = path.join(temp, "interface-contract-mismatch.json");
  const mismatchedParameters = structuredClone(interfaceLedger.parameters);
  mismatchedParameters[0].dtype = mismatchedParameters[0].dtype === "UINT8" ? "INT8" : "UINT8";
  await writeFile(mismatchedContractPath, JSON.stringify({
    schema: "deepbom.production_interface_contract.v1",
    artifact_sha256: tfliteAnalysis.model_sha256,
    implementation_sha256: "a".repeat(64),
    parameters: mismatchedParameters,
  }), "utf8");
  const mismatchRun = run(["verify", cases[0][0], "--contract", mismatchedContractPath, "--compact"], false);
  assert.equal(mismatchRun.status, 2, "verify mismatch exit code");
  const mismatch = JSON.parse(mismatchRun.stdout);
  assert.equal(mismatch.comparison.gate_result, "block", "verify mismatch gate");
  assert.equal(mismatch.comparison.mismatch_count > 0, true, "verify mismatch ledger");

  const customTargetPath = path.join(temp, "custom-target.json");
  await writeFile(customTargetPath, JSON.stringify({
    base: "android_mid_a55",
    id: "custom:cli-regression-a55",
    label: "CLI regression A55",
    evidence_class: "USER_DECLARED",
    evidence_note: "CLI regression fixture",
    overrides: { l2_bytes: 262144, compute_utilization_factor: 0.25 },
  }), "utf8");
  const customTarget = JSON.parse(run(["audit", cases[0][0], "--target-profile", customTargetPath, "--compact"]).stdout);
  assert.equal(customTarget.target_profile.id, "custom:cli-regression-a55", "custom target resolved identity");
  assert.match(customTarget.target_profile.profile_sha256, /^[a-f0-9]{64}$/, "custom target resolved SHA-256");
  assert.match(customTarget.cli_target_profile_input.source_sha256, /^[a-f0-9]{64}$/, "custom target source file SHA-256");
  assert.equal(customTarget.cli_target_profile_input.resolved_target_profile_sha256, customTarget.target_profile.profile_sha256,
    "custom target input evidence binds the resolved Rust profile");
  assert.equal(customTarget.cpu_cost_target_binding?.binding_source, "profile_file", "custom target binding source");
  assert.equal(customTarget.cpu_cost_target_binding?.source_input?.source_sha256, customTarget.cli_target_profile_input.source_sha256,
    "custom target binding must retain the profile-file identity");
  const duplicateTargetPath = path.join(temp, "duplicate-target.json");
  await writeFile(duplicateTargetPath, '{"base":"android_mid_a55","base":"rpi4_a72","id":"custom:duplicate","label":"Duplicate","overrides":{}}', "utf8");
  const duplicateTarget = run(["audit", cases[0][0], "--target-profile", duplicateTargetPath], false);
  assert.notEqual(duplicateTarget.status, 0, "duplicate target JSON must fail closed");
  assert.match(duplicateTarget.stderr, /duplicate JSON key base/);
  const conflictingTarget = run(["audit", cases[0][0], "--target", "android_mid_a55", "--target-profile", customTargetPath], false);
  assert.notEqual(conflictingTarget.status, 0, "target id/profile conflict must fail closed");
  assert.match(conflictingTarget.stderr, /mutually exclusive/);

  const delta = JSON.parse(run(["diff", "web/samples/mobilenet_v1_025_224_float.tflite", cases[0][0], "--compact"]).stdout);
  assert.equal(delta.schema, "deepbom.deployment_delta.v1.1", "diff uses the canonical deployment-delta schema");
  assert.equal(delta.target_count, 4, "diff default target denominator");
  assert.equal(delta.alignment.matched_op_count + delta.alignment.removed_op_count, delta.baseline.operator_count,
    "diff baseline op conservation");
  assert.equal(delta.alignment.matched_op_count + delta.alignment.added_op_count, delta.candidate.operator_count,
    "diff candidate op conservation");
  assert.equal(delta.target_deltas.every((row) => Math.abs(row.conservation_error_us) <= 1e-8), true,
    "diff target-ledger conservation");

  const pareto = JSON.parse(run(["explore", cases[0][0], "--compact"]).stdout);
  assert.equal(pareto.schema, "deepbom.redesign_pareto.v1", "explore uses the canonical Pareto schema");
  assert.equal(pareto.source_sha256, tfliteAnalysis.model_sha256, "explore source SHA-256 binding");
  assert.equal(pareto.evaluated_candidate_count,
    pareto.accepted_candidate_count + pareto.rejected_candidate_count, "explore candidate conservation");
  assert.equal(pareto.frontier_candidate_count, pareto.candidates.filter((row) => row.pareto_optimal).length,
    "explore frontier denominator");

  const trtConfig = {
    execution_path: "native_tensorrt",
    expected_tensorrt_version: "10.14.1",
    expected_cuda_version: "13.0",
    device_id: 0,
    device_compute_capability: "8.7",
    precision: { tf32: true, fp16: true, bf16: false, int8: false, fp8: false },
    workspace_limit_bytes: 1_073_741_824,
    builder_optimization_level: 3,
    dla_core: null,
    allow_gpu_fallback: false,
    calibration_cache_sha256: null,
    plugins: [],
    optimization_profiles: [],
  };
  const trtProfile = createTensorRtBuildProfile(trtConfig);
  const profilePath = path.join(temp, "profile.json");
  await writeFile(profilePath, JSON.stringify(trtConfig), "utf8");
  const configured = JSON.parse(run(["audit", cases[1][0], "--tensorrt-profile", profilePath, "--compact"]).stdout);
  assert.equal(configured.tensorrt_static_preflight.status, "configuration_valid_parser_observation_required");
  assert.equal(configured.tensorrt_static_preflight.build_profile.profile_sha256, trtProfile.profile_sha256);
  const evidencePath = path.join(temp, "parser.json");
  const parserObservation = {
    schema: TENSORRT_PARSER_OBSERVATION_SCHEMA,
    artifact_sha256: onnxAnalysis.model_sha256,
    build_profile_sha256: trtProfile.profile_sha256,
    build_profile_file_sha256: sha256TextHex(`${canonicalJson(trtProfile)}\n`),
    build_profile: trtProfile,
    execution_path: "native_tensorrt",
    tensorrt_version: "10.14.1",
    cuda_version: "13.0",
    device_id: 0,
    device_compute_capability: "8.7",
    device_identity: "CLI fixture CC 8.7",
    api_method: "supportsModelV2",
    subgraph_support_semantics: "per_subgraph_api_flag",
    parser_returned: true,
    collector: {
      binary_sha256: "b".repeat(64),
      source_set_sha256: "c".repeat(64),
      git_commit: "cli-fixture",
      git_state: "clean",
    },
    plugins: [],
    subgraphs: [{ subgraph_index: 0, supported: true, sdk_reported_flag: true, node_indices: Array.from({ length: onnxAnalysis.ops.length }, (_, index) => index) }],
    errors: [],
  };
  await writeFile(evidencePath, JSON.stringify(parserObservation), "utf8");
  const observed = JSON.parse(run(["audit", cases[1][0], "--tensorrt-parser-evidence", evidencePath, "--compact"]).stdout);
  assert.equal(observed.tensorrt_static_preflight.status, "parser_observed_all_supported");
  assert.equal(observed.tensorrt_static_preflight.projection.state_counts.CONDITIONALLY_ELIGIBLE, onnxAnalysis.ops.length);
  assert.equal(observed.accelerator_bindings.find((row) => row.backend === "tensorrt")?.evidence_stage,
    "selected_build", "TensorRT parser observation must remain selected-build evidence");
  const engineInformation = {
    Layers: [{
      Name: "conv [ONNX Layer: Conv_0]", LayerType: "Convolution",
      Inputs: [{ Name: "input", Dimensions: [1, 3, 32, 32], "Format/Datatype": "FP32" }],
      Outputs: [{ Name: "output", Dimensions: [1, 8, 30, 30], "Format/Datatype": "FP32" }],
      TacticName: "cli_fixture_tactic",
    }],
    Bindings: ["input", "output"],
  };
  const inspectorPath = path.join(temp, "engine-inspector.json");
  await writeFile(inspectorPath, JSON.stringify({
    schema: TENSORRT_ENGINE_INSPECTOR_EVIDENCE_SCHEMA,
    artifact_sha256: onnxAnalysis.model_sha256,
    build_profile_sha256: trtProfile.profile_sha256,
    parser_observation_sha256: tensorRtParserObservationIdentity(parserObservation),
    engine: { sha256: "d".repeat(64), byte_length: 4096 },
    runtime: { tensorrt_version: "10.14.1", cuda_version: "13.0", device_id: 0, device_compute_capability: "8.7", device_identity: "CLI fixture CC 8.7" },
    build_capture: {
      evidence_class: "DECLARED_BUILD_CAPTURE", binding_method: "cli_fixture_files",
      tool_name: "trtexec", tool_binary_sha256: "e".repeat(64), invocation_sha256: "f".repeat(64),
      collector_source_set_sha256: null, model_input_sha256: onnxAnalysis.model_sha256, serialized_engine_sha256: "d".repeat(64),
    },
    inspector: {
      source: "trtexec_exportLayerInfo", profiling_verbosity: "detailed", schema_generation: "tensorrt_10x",
      execution_context_bound: false, source_file_sha256: "1".repeat(64), source_file_byte_length: 1024,
      canonical_json_sha256: sha256TextHex(canonicalJson(engineInformation)), engine_information: engineInformation,
    },
  }), "utf8");
  const engineInspected = JSON.parse(run(["audit", cases[1][0], "--tensorrt-profile", profilePath,
    "--tensorrt-parser-evidence", evidencePath, "--tensorrt-engine-inspector", inspectorPath, "--compact"]).stdout);
  assert.equal(engineInspected.tensorrt_static_preflight.status, "engine_inspected_parser_observed_all_supported");
  assert.equal(engineInspected.tensorrt_static_preflight.engine_inspector_evidence.engine_layer_count, 1);
  const engineBinding = engineInspected.accelerator_bindings.find((row) => row.backend === "tensorrt");
  assert.equal(engineBinding?.evidence_stage, "compiled_plan", "TensorRT engine inspector evidence stage");
  assert.equal(engineBinding?.claims?.observed_assignment, false, "TensorRT engine metadata must not imply observed original-op assignment");
  const wrongFormatTrt = run(["audit", cases[0][0], "--tensorrt-profile", profilePath], false);
  assert.notEqual(wrongFormatTrt.status, 0);
  assert.match(wrongFormatTrt.stderr, /apply only to ONNX/);
  const trtLlmConfigPath = path.join(temp, "tensorrt-llm.json");
  await writeFile(trtLlmConfigPath, JSON.stringify({
    version: "1.2.0",
    pretrained_config: {
      architecture: "LlamaForCausalLM", dtype: "float16", hidden_size: 8, intermediate_size: 16,
      num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 1, head_size: 4,
      vocab_size: 32, max_position_embeddings: 64,
      mapping: { world_size: 1, tp_size: 1, pp_size: 1, cp_size: 1 },
      quantization: { quant_algo: null, kv_cache_quant_algo: null, group_size: 128, has_zero_point: false, exclude_modules: [] },
    },
    build_config: {
      max_input_len: 32, max_seq_len: 64, max_batch_size: 1, max_beam_width: 1,
      max_num_tokens: 64, opt_num_tokens: 32, kv_cache_type: "PAGED", strongly_typed: true,
      weight_streaming: false, plugin_config: { paged_kv_cache: true },
    },
  }), "utf8");
  const trtLlm = JSON.parse(run(["audit", cases[3][0], "--tensorrt-llm-config", trtLlmConfigPath, "--compact"]).stdout);
  assert.equal(trtLlm.on_device_llm.tensorrt_llm.status, "candidate_configuration_unbound");
  assert.equal(trtLlm.on_device_llm.tensorrt_llm.kv_cache_scenario.logical_bytes.decimal, "2048");
  const wrongTrtLlmFormat = run(["audit", cases[1][0], "--tensorrt-llm-config", trtLlmConfigPath], false);
  assert.notEqual(wrongTrtLlmFormat.status, 0);
  assert.match(wrongTrtLlmFormat.stderr, /only to a SafeTensors artifact/);
  const wrongExternalDataFormat = run(["audit", cases[0][0], "--external-data-dir", temp], false);
  assert.notEqual(wrongExternalDataFormat.status, 0);
  assert.match(wrongExternalDataFormat.stderr, /applies only to an ONNX or ExecuTorch PTE file/);
} finally {
  await rm(temp, { recursive: true, force: true });
}

const first = run(["audit", cases[0][0], "--compact"]).stdout;
const second = run(["audit", cases[0][0], "--compact"]).stdout;
assert.equal(first, second, "TFLite analysis JSON must be deterministic");

const wrongCommand = run(["gguf", cases[1][0]], false);
assert.notEqual(wrongCommand.status, 0);
assert.match(wrongCommand.stderr, /requires a GGUF artifact/);
const unboundOnnxContext = JSON.parse(run(["audit", cases[1][0], "--context", "8192", "--compact"]).stdout);
assert.equal(unboundOnnxContext.llm_token_budget_scenario?.status, "not_assessable_kv_contract_unbound");
assert.equal(unboundOnnxContext.llm_token_budget_scenario?.memory_feasibility?.fit_claim, "not_emitted");
const invalidContextFormat = run(["audit", cases[4][0], "--context", "8192"], false);
assert.notEqual(invalidContextFormat.status, 0);
assert.match(invalidContextFormat.stderr, /require a TFLite, ONNX, GGUF, or SafeTensors artifact/);
const invalidStateBits = run(["gguf", cases[2][0], "--context", "8192", "--state-bits", "12"], false);
assert.notEqual(invalidStateBits.status, 0);
assert.match(invalidStateBits.stderr, /must be 8, 16, or 32/);
const orphanMemoryBudget = run(["gguf", cases[2][0], "--memory-mib", "1024"], false);
assert.notEqual(orphanMemoryBudget.status, 0);
assert.match(orphanMemoryBudget.stderr, /require --context/);

console.log("CLI checks passed (six formats, strict target profiles, interface verify, deployment diff, redesign explore, external-data contracts, LLM/TensorRT evidence, CycloneDX projection, deterministic output, and fail-closed routing).");

function buildExecuTorchAttestation() {
  const hex = (character) => character.repeat(64);
  const binaryInventory = [{ path: "bin/executorch_runner", byte_length: 4096, sha256: hex("a") }];
  const normalized = {
    schema: EXECUTORCH_SELECTED_BUILD_ATTESTATION_SCHEMA,
    evidence_class: "REPRODUCIBLE_SELECTED_BUILD_ATTESTATION",
    source: {
      repository: "pytorch/executorch",
      release: "v1.4.1",
      commit: "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f",
      pristine_before_build: true,
      submodule_status_sha256: hex("b"),
      post_build_diff_sha256: hex("c"),
      backend_registry_sha256: EXECUTORCH_BACKEND_REGISTRY_SOURCE.registry_sha256,
    },
    build: {
      configuration: "cli-test-release",
      cmake_options: {
        EXECUTORCH_BUILD_COREML: false,
        EXECUTORCH_BUILD_CUDA: false,
        EXECUTORCH_BUILD_METAL: false,
        EXECUTORCH_BUILD_MPS: false,
        EXECUTORCH_BUILD_PORTABLE_OPS: true,
        EXECUTORCH_BUILD_QNN: false,
        EXECUTORCH_BUILD_VULKAN: false,
        EXECUTORCH_BUILD_XNNPACK: false,
      },
      linked_backend_ids: [],
      custom_backend_sources: [],
      portable_operator_names: ["aten::add.out"],
      custom_operator_names: [],
      cmake_cache_sha256: hex("d"),
      build_stdout_sha256: hex("e"),
      build_stderr_sha256: hex("f"),
    },
    runtime: {
      platform: "linux",
      arch: "x86_64",
      binary_inventory: binaryInventory,
      binary_inventory_sha256: sha256TextHex(canonicalJson(binaryInventory)),
      primary_binary_path: "bin/executorch_runner",
      primary_binary_sha256: hex("a"),
    },
    boundary: null,
  };
  return { ...normalized, attestation_sha256: sha256TextHex(canonicalJson(normalized)) };
}

function run(args, expectSuccess = true) {
  const result = spawnSync(process.execPath, ["bin/deepbom.mjs", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  if (expectSuccess && result.status !== 0) {
    throw new Error(`CLI failed: ${args.join(" ")}\n${result.stderr}`);
  }
  return result;
}
