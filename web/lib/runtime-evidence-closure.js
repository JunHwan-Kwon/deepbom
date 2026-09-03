import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { GGUF_BACKEND_SOURCE } from "./gguf-backend-contract.generated.js";
import { GGUF_RUNTIME_ENVIRONMENT_SCHEMA } from "./gguf-runtime-environment.js";
import { COREML_COMPUTE_PLAN_SCHEMA } from "./coreml-compute-plan.js";
import { buildRuntimeEvidenceSidecar } from "./runtime-evidence-sidecar.js";
import { buildOnnxRuntimeShapeBinding } from "./onnx-runtime-shape-binding.js";
import { buildRuntimeDataMovementEvidence } from "./runtime-data-movement-evidence.js";

export function buildRuntimeCapturePlan(analysis) {
  if (!analysis || !["tflite", "onnx", "gguf", "coreml"].includes(analysis.format)) {
    throw new Error("Runtime capture plans require a completed TFLite, ONNX, GGUF, or decoded Core ML graph audit.");
  }
  if (analysis.format === "coreml" && (!(artifactIrOperators(analysis) || []).length
    || !(analysis.coreml?.neural_network || analysis.coreml?.model_type === "mlProgram"))) {
    throw new Error("Core ML compute-plan capture requires a decoded NeuralNetwork or ML Program graph; classical and pipeline structures are outside the operation-level MLComputePlan API.");
  }
  const artifactSha256 = validSha(analysis.model_sha256, "artifact SHA-256");
  const inputs = ["gguf", "coreml"].includes(analysis.format) ? [] : (analysis.inputs || []).map((input, index) => ({
    index: Number.isSafeInteger(Number(input.index)) ? Number(input.index) : index,
    name: String(input.name || `input_${index}`),
    shape: staticShape(input.shape_signature || input.shape),
    dtype: String(input.dtype || "UNKNOWN"),
  }));
  if (!["gguf", "coreml"].includes(analysis.format) && (!inputs.length || inputs.some((input) => !input.shape.length))) {
    throw new Error("A capture plan requires at least one statically bound input shape.");
  }
  const target = analysis.target_profile || null;
  if (analysis.format === "tflite") validSha(target?.profile_sha256, "target-profile SHA-256");
  const command = analysis.format === "tflite"
    ? [
      "npm run capture:pinned-runtime --",
      "--model \"MODEL_PATH\"",
      "--output-dir \"CAPTURE_OUTPUT_DIR\"",
      `--target-profile-id ${shellToken(target.id)}`,
      `--target-profile-sha256 ${target.profile_sha256}`,
      ...inputs.map((input) => `--input ${input.index}:${shellToken(input.name)}:${input.shape.join("x")}`),
    ].join(" ")
    : analysis.format === "onnx" ? [
      "npm run prepare:pinned-ort -- --output \"ORT_BUILD_ATTESTATION.json\"",
      [
        "npm run capture:pinned-ort -- \"MODEL_PATH\" \"CAPTURE_OUTPUT_DIR\" --providers=cpu",
        "--runtime-module \"ORT_SOURCE/js/node/dist/index.js\"",
        "--build-attestation \"ORT_BUILD_ATTESTATION.json\"",
        ...inputs.map((input) => `--shape=${shellToken(input.name)}=${input.shape.join(",")}`),
      ].join(" "),
    ].join("\n") : analysis.format === "gguf" ? [
      "npm run prepare:gguf-runtime -- --output \"LLAMA_BUILD_ATTESTATION.json\"",
      [
        "npm run capture:gguf-runtime --",
        "--model \"MODEL_PATH\"",
        "--binary \"LLAMA_CLI_PATH\"",
        "--cmake-cache \"CMAKE_CACHE_PATH\"",
        "--build-attestation \"LLAMA_BUILD_ATTESTATION.json\"",
        "--output \"GGUF_RUNTIME_EVIDENCE.json\"",
        `--source-commit ${GGUF_BACKEND_SOURCE.source_commit}`,
        "--backend cpu --context 2048 --batch 512 --ubatch 128 --gpu-layers 0 --run-smoke",
      ].join(" "),
    ].join("\n") : [
      "npm run capture:coreml-plan --",
      "--model \"MODEL_OR_PACKAGE_PATH\"",
      `--artifact-sha256 ${artifactSha256}`,
      "--output \"COREML_COMPUTE_PLAN.json\"",
      `--compute-units ALL${analysis.coreml?.description?.default_function_name ? ` --function-name ${shellToken(analysis.coreml.description.default_function_name)}` : ""}`,
    ].join(" ");
  const body = {
    schema: "deepbom.runtime_capture_plan.v1.1",
    evidence_class: "REQUIREMENT",
    artifact: { format: analysis.format, filename: analysis.filename || null, sha256: artifactSha256 },
    target_profile: target ? { id: target.id || null, sha256: target.profile_sha256 || null } : null,
    invocation_inputs: inputs,
    command,
    required_identity: analysis.format === "coreml"
      ? ["artifact_sha256", "compiled_model_content_sha256", "coremltools_compute_plan_source_sha256", "collector_source_sha256", "macos_version_and_build", "hardware_model", "available_compute_devices", "compute_units", "function_name", "capture_timestamp"]
      : ["artifact_sha256", "runtime_binary_sha256", "source_commit", "build_flags", "cpu_features", "capture_timestamp"],
    required_observations: analysis.format === "tflite"
      ? ["delegate_partitions", "original_op_assignment", "lowering_ids", "microkernel_ids", "arena_allocations"]
      : analysis.format === "onnx"
        ? ["pinned_source_build_attestation", "selected_binary_inventory", "selected_provider_inventory", "optimized_graph_identity", "execution_provider_assignment", "internal_type_shapes", "kernel_events", "profiled_copy_node_payloads", "provider_transition_edges"]
        : analysis.format === "gguf"
          ? ["selected_build_attestation", "cmake_option_inventory", "compiled_backend_inventory", "requested_backend", "generated_graph_nodes", "scheduler_inserted_nodes", "scheduler_splits", "per_node_backend_assignment", "backend_transition_edges", "dispatch_status"]
          : ["compiled_model_identity", "compute_unit_configuration", "operation_identity", "preferred_compute_device", "supported_compute_devices", "estimated_cost_weight"],
    import_contract: {
      accepted_by: analysis.format === "gguf" ? "Evidence capability > Import runtime evidence" : "Graph > Kernel & Runtime > Import runtime evidence",
      schema: analysis.format === "gguf" ? GGUF_RUNTIME_ENVIRONMENT_SCHEMA : analysis.format === "coreml" ? COREML_COMPUTE_PLAN_SCHEMA : "deepbom.runtime_assignment.v1.10",
      fail_closed_on: analysis.format === "gguf"
        ? ["artifact_hash_mismatch", "source_or_patch_hash_mismatch", "build_attestation_mismatch", "runtime_binary_hash_mismatch", "cmake_cache_hash_mismatch", "build_option_inventory_incomplete", "requested_backend_not_compiled", "scheduler_trace_missing", "scheduler_graph_conservation_failure"]
        : analysis.format === "coreml"
          ? ["artifact_hash_mismatch", "compiled_model_hash_missing", "static_operation_identity_mismatch", "function_identity_mismatch", "compute_plan_source_hash_missing", "collector_source_hash_missing", "macos_build_missing", "compute_device_inventory_missing_or_inconsistent"]
          : analysis.format === "onnx"
            ? ["artifact_content_set_hash_mismatch", "pinned_source_or_build_script_hash_mismatch", "build_attestation_mismatch", "selected_binary_inventory_mismatch", "profile_hash_mismatch", "graph_mapping_ambiguity"]
            : ["artifact_hash_mismatch", "target_profile_hash_mismatch", "graph_mapping_ambiguity", "runtime_binary_hash_missing"],
    },
    boundary: "The plan is a reproducible collection requirement, not an observed result. Device latency and executed placement exist only after a verified capture is imported.",
  };
  return { ...body, plan_sha256: sha256TextHex(canonicalJson(body)) };
}

export function runtimeCapturePlanAvailability(analysis) {
  try {
    return { available: true, reason: "", plan: buildRuntimeCapturePlan(analysis) };
  } catch (error) {
    return { available: false, reason: error?.message || String(error), plan: null };
  }
}

export function renderRuntimeEvidenceClosure(root, analysis, runtimeEvidence) {
  if (!root) return;
  const capturePlanButton = root.querySelector("#downloadRuntimeCapturePlan");
  const sidecarButton = root.querySelector("#downloadRuntimeEvidenceSidecar");
  const capturePlan = runtimeCapturePlanAvailability(analysis);
  if (capturePlanButton) {
    capturePlanButton.disabled = !capturePlan.available;
    capturePlanButton.title = capturePlan.available ? "Download an identity-bound native runtime collection plan" : capturePlan.reason;
  }
  if (sidecarButton) {
    sidecarButton.disabled = !runtimeEvidence;
    sidecarButton.title = runtimeEvidence
      ? "Download the normalized, hash-bound cross-format runtime evidence index"
      : "Import runtime evidence before exporting a normalized sidecar";
  }
  const applicable = ["tflite", "onnx", "gguf", "coreml"].includes(analysis?.format);
  root.hidden = !applicable;
  if (!applicable) return;
  if (analysis.format === "gguf") {
    renderGgufRuntimeClosure(root, runtimeEvidence);
    return;
  }
  if (analysis.format === "coreml") {
    renderCoreMlComputePlanClosure(root, runtimeEvidence, analysis);
    return;
  }
  const comparison = runtimeEvidence?.comparison || null;
  const assignments = runtimeEvidence?.assignments || [];
  const observed = Number(comparison?.observed_assignment_count ?? comparison?.observed_op_count ?? assignments.length) || 0;
  const graphOps = Number(analysis?.operator_count ?? artifactIrOperators(analysis)?.length) || 0;
  const predicted = analysis.format === "tflite"
    ? (artifactIrOperators(analysis) || []).filter((op) => Number(op.xnnpack_chain_id) >= 0).length
    : null;
  const collector = runtimeEvidence?.source?.collector || null;
  const runtimeHash = runtimeEvidence?.runtime?.binary_sha256 || collector?.binary_sha256 || null;
  const lowering = assignments.filter((row) => row.lowering_id || row.lowerings?.length).length;
  const kernels = assignments.filter((row) => row.microkernel_id || row.microkernel_symbol || row.dispatches?.length).length;
  const onnxShapeBinding = analysis.format === "onnx" && runtimeEvidence
    ? buildOnnxRuntimeShapeBinding(analysis, runtimeEvidence)
    : null;
  const runtimeDataMovement = analysis.format === "onnx" && runtimeEvidence
    ? buildRuntimeDataMovementEvidence(runtimeEvidence)
    : null;
  const coverage = graphOps ? Math.min(100, observed / graphOps * 100) : 0;
  setText(root, "[data-runtime-closure-status]", observed
    ? `${observed}/${graphOps} graph ops carry imported runtime assignment evidence.${onnxShapeBinding ? ` Internal shapes ${onnxShapeBinding.observed_internal_tensor_count}; MAC ops newly closed ${onnxShapeBinding.runtime_closed_mac_op_count}, remaining ${onnxShapeBinding.remaining_unassessed_mac_op_count}.` : ""}`
    : "Static placement evidence is available; no observed runtime assignment is bound.");
  setText(root, "[data-runtime-closure-placement]", predicted == null ? `${observed}/${graphOps} observed` : `${predicted} predicted / ${observed} observed`);
  setText(root, "[data-runtime-closure-identity]", runtimeHash ? `Bound ${runtimeHash.slice(0, 12)}...` : "Runtime binary unbound");
  setText(root, "[data-runtime-closure-selector]", observed
    ? onnxShapeBinding
      ? `${onnxShapeBinding.status}; ${onnxShapeBinding.conflict_count} shape conflict(s); ${runtimeDataMovement ? `${runtimeDataMovement.observed_copy_node_count} profiled copy node(s), physical bytes not exposed` : "copy-node payload not assessed"}`
      : `${lowering} lowering / ${kernels} kernel rows`
    : "Not observed");
  setText(root, "[data-runtime-closure-collector]", collector ? `${collector.name} ${collector.version} / ${collector.attestation_status}` : "No collector manifest");
  const progress = root.querySelector("progress");
  progress.value = coverage;
  progress.setAttribute("aria-valuetext", `${coverage.toFixed(1)}% observed assignment coverage`);
  root.dataset.state = observed === graphOps && runtimeHash ? "bound" : observed ? "partial" : "missing";
}

export function runtimeEvidenceSidecarForDownload(analysis, runtimeEvidence) {
  if (!runtimeEvidence) throw new Error("Import runtime evidence before exporting a normalized sidecar.");
  return buildRuntimeEvidenceSidecar(analysis, runtimeEvidence);
}

function renderCoreMlComputePlanClosure(root, runtimeEvidence, analysis) {
  const bound = runtimeEvidence?.schema === COREML_COMPUTE_PLAN_SCHEMA;
  const rows = runtimeEvidence?.structure?.rows || [];
  const graphOps = Number(artifactIrOperators(analysis)?.length || 0);
  setText(root, "[data-runtime-closure-status]", bound
    ? `${rows.length}/${graphOps} decoded operations are bound to one MLComputePlan estimate. This is not an execution trace.`
    : "Serialized Core ML graph evidence is available; no compiled-model MLComputePlan is bound.");
  setText(root, "[data-runtime-closure-placement]", bound ? `${runtimeEvidence.configuration.compute_units}; anticipated devices only` : "Compute plan not imported");
  setText(root, "[data-runtime-closure-identity]", bound ? `Compiled ${runtimeEvidence.runtime.compiled_model_content_sha256.slice(0, 12)}...` : "Compiled model unbound");
  setText(root, "[data-runtime-closure-selector]", bound ? `${runtimeEvidence.summary.estimated_cost_operation_count}/${rows.length} cost rows` : "Not estimated");
  setText(root, "[data-runtime-closure-collector]", bound ? `${runtimeEvidence.capture.collector.name} ${runtimeEvidence.capture.collector.version} / ${runtimeEvidence.capture.capture_id}` : "No compute-plan manifest");
  const progress = root.querySelector("progress");
  progress.value = bound && graphOps ? rows.length / graphOps * 100 : 0;
  progress.setAttribute("aria-valuetext", bound ? `${rows.length}/${graphOps} compute-plan identity coverage; execution not observed` : "Core ML compute plan not bound");
  root.dataset.state = bound && rows.length === graphOps ? "partial" : "missing";
}

function renderGgufRuntimeClosure(root, runtimeEvidence) {
  const bound = runtimeEvidence?.schema === GGUF_RUNTIME_ENVIRONMENT_SCHEMA;
  const binaryHash = runtimeEvidence?.runtime?.binary_sha256 || null;
  const profiles = runtimeEvidence?.build?.compiled_backend_profile_ids || [];
  const observed = runtimeEvidence?.observations?.inference_status || "not_run";
  setText(root, "[data-runtime-closure-status]", bound
    ? `Runtime binary, CMake build inventory, and requested ${runtimeEvidence.selection.requested_backend_label} configuration are bound to this GGUF artifact.`
    : "Static llama.cpp prerequisites are available; no identity-bound runtime build manifest is imported.");
  setText(root, "[data-runtime-closure-placement]", bound
    ? `requested ${runtimeEvidence.selection.requested_backend_profile_id}; per-op placement N/A`
    : "Execution graph not serialized");
  setText(root, "[data-runtime-closure-identity]", binaryHash ? `Bound ${binaryHash.slice(0, 12)}...` : "Runtime binary unbound");
  setText(root, "[data-runtime-closure-selector]", bound ? `${profiles.join(" / ")} compiled; inference ${observed}` : "Build not imported");
  setText(root, "[data-runtime-closure-collector]", bound
    ? `${runtimeEvidence.capture.collector.name} ${runtimeEvidence.capture.collector.version} / ${runtimeEvidence.capture.capture_id}`
    : "No collector manifest");
  const progress = root.querySelector("progress");
  progress.value = bound ? observed === "observed_success" ? 100 : 70 : 0;
  progress.setAttribute("aria-valuetext", bound ? `Runtime environment bound; inference ${observed}` : "Runtime environment not bound");
  root.dataset.state = bound ? observed === "observed_success" ? "bound" : "partial" : "missing";
}

function setText(root, selector, value) {
  const node = root.querySelector(selector);
  if (node) node.textContent = value;
}

function validSha(value, label) {
  const normalized = String(value || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(normalized)) throw new Error(`Runtime capture plan requires ${label}.`);
  return normalized;
}

function staticShape(shape) {
  if (!Array.isArray(shape)) return [];
  const dimensions = shape.map(Number);
  return dimensions.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0) ? dimensions : [];
}

function shellToken(value) {
  const token = String(value || "").replaceAll(/[^A-Za-z0-9_.-]+/g, "-");
  if (!token) throw new Error("Runtime capture plan contains an empty shell token.");
  return token;
}
