import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";

const FORMAT_SCOPES = Object.freeze({
  tflite: Object.freeze({
    label: "TFLite",
    depth: "Deep graph and deployment-model audit",
    assessed: "Every serialized SubGraph and control-flow reference, independent per-subgraph tensor/quantization/cost/ArenaPlanner/source-pinned XNNPACK candidate proofs, all-subgraph sparse storage, and target model",
    runtimeBoundary: "Nested control-flow invocation counts, actual delegate assignment, materialized copies, selected microkernel, and device latency require imported runtime evidence",
    nextProof: "Import a target delegate assignment and runtime capture, then bind deployment requirements and representative task-output acceptance evidence",
    stagedDescriptor: "target-bound static graph and deployment audit",
    dashboardTitle: "TFLite Deployment Evidence",
    dashboardCopy: "Artifact facts, selected-target estimates, and conditional XNNPACK compatibility are separated from imported runtime observations.",
    runLabel: "Run TFLite Static Audit",
    completion: "TFLite static deployment audit run complete",
  }),
  onnx: Object.freeze({
    label: "ONNX",
    depth: "Static graph and ORT compatibility audit",
    assessed: "ModelProto graph, bounded shape/type inference, MAC coverage, Q/DQ contracts, serialized LLM graph signals, opsets, independent source-backed ORT EP portfolios, and optional TensorRT native-parser or ORT-EP preflight evidence",
    runtimeBoundary: "Source eligibility and native TensorRT parser capability acceptance are not ORT TensorRT GetCapability assignment or execution: actual provider assignment, lowering, kernels, memory pattern, and device latency require bound runtime rows",
    nextProof: "Import a target ORT profile with provider assignment and runtime identity, then bind deployment requirements and representative task-output acceptance evidence",
    stagedDescriptor: "static graph, opset, shape, and ORT compatibility audit",
    dashboardTitle: "ONNX Graph And Runtime Compatibility",
    dashboardCopy: "Serialized graph evidence and source-backed EP compatibility are shown without claiming observed provider assignment.",
    runLabel: "Run ONNX Static Audit",
    completion: "ONNX static graph audit run complete",
  }),
  gguf: Object.freeze({
    label: "GGUF",
    depth: "Container and tensor-payload audit",
    assessed: "Header, architecture and tokenizer metadata, tensor directory, GGML storage encodings, payload ranges, source-pinned numerical integrity, registered canonical-decoder parameter/KV/compute scenarios, lower-bound-only memory feasibility, and llama.cpp backend prerequisites",
    runtimeBoundary: "GGUF does not serialize an execution DAG or per-op placement; selected llama.cpp build, backend, context, offload, kernels, and latency require bound runtime evidence",
    nextProof: "Bind the consuming engine, backend, context, batch, and target runtime capture before making execution or latency claims",
    stagedDescriptor: "container, tensor encoding, and payload-integrity audit",
    dashboardTitle: "GGUF Container Evidence",
    dashboardCopy: "Stored tensor encodings and byte conservation are assessed without inventing an operator graph or runtime compute precision.",
    runLabel: "Audit GGUF Container",
    completion: "GGUF container audit run complete",
  }),
  safetensors: Object.freeze({
    label: "SafeTensors",
    depth: "Checkpoint and shard-integrity audit",
    assessed: "Header, tensor dtype and shape inventory, exact byte ranges, full-payload numerical integrity, shard-index completeness, source-pinned dense/MoE/SSM architecture and state scenarios, lower-bound-only memory feasibility, plus optional TensorRT-LLM TP/PP/CP, build-limit, quantization, and logical-KV configuration bound to a model-source digest",
    runtimeBoundary: "A checkpoint plus configuration still does not serialize the executable forward graph. TensorRT-LLM configuration also does not prove engine tactics, per-rank weight residency, preprocessing, runtime cache allocation, occupancy, throughput, or latency",
    nextProof: "Bind the executable graph, preprocessing contract, runtime configuration, and task-output acceptance evidence that consume this checkpoint",
    stagedDescriptor: "checkpoint inventory, payload, and shard-integrity audit",
    dashboardTitle: "SafeTensors Checkpoint Evidence",
    dashboardCopy: "Tensor storage and shard identity are assessed; graph and deployment behavior remain outside this container contract.",
    runLabel: "Audit SafeTensors Checkpoint",
    completion: "SafeTensors checkpoint integrity audit run complete",
  }),
  coreml: Object.freeze({
    label: "Core ML",
    depth: "Model/package contract and serialized-program audit",
    assessed: "Package identity, interfaces, decoded NeuralNetwork and ML Program DAGs, GLM/SVM/TreeEnsemble contracts, named Pipeline stage graphs, shape/arithmetic/liveness costs, numerical payload integrity, and a source-bound OS floor",
    runtimeBoundary: "For NeuralNetwork and ML Program only, MLComputePlan can add anticipated compute-device usage and relative cost; classical/pipeline execution plus all executed placement, fusion, allocation, and latency require Apple runtime evidence",
    nextProof: "Import an identity-bound MLComputePlan, then add executed Apple runtime placement and representative task-output acceptance evidence",
    stagedDescriptor: "model/package contract, serialized graph, and weight-encoding audit",
    dashboardTitle: "Core ML Serialized Artifact Evidence",
    dashboardCopy: "Decoded model and package contracts are separated from unobserved native Core ML execution behavior.",
    runLabel: "Audit Core ML Artifact",
    completion: "Core ML model/package audit run complete",
  }),
  executorch: Object.freeze({
    label: "ExecuTorch",
    depth: "Serialized execution-plan, delegate, tensor, and AOT memory audit",
    assessed: "ET12 execution plans, ordered instructions, EValue contracts, 209 source-pinned portable operator signatures, matching KernelCall input/output direction and nominal tensor-contraction MACs, serialized delegate IDs and compile specs, processed-payload byte identities and public-schema root envelopes, optional selected-build/backend/operator/binary attestation, segment conservation, AOT planned buffers, and FT01 named external tensor data",
    runtimeBoundary: "Custom or signature-mismatched KernelCall rows and delegate-internal semantics remain unbound; a build attestation does not establish dead-strip behavior, backend initialization, executed placement, runtime allocation, kernels, physical transfers, correctness, or latency",
    nextProof: "Add deepbom.executorch-build.json and required PTD data to close selected-build identity, then import native executed delegate, allocation, transfer, and timing evidence",
    stagedDescriptor: "serialized execution-plan, segment, delegate, and AOT memory audit",
    dashboardTitle: "ExecuTorch Program Evidence",
    dashboardCopy: "Observed ET12/FT01 contracts, source-described payload envelopes, and optional selected-build identity are separated from delegate-internal semantics and unobserved runtime behavior.",
    runLabel: "Audit ExecuTorch Artifact",
    completion: "ExecuTorch static program audit run complete",
  }),
});

function normalizedFormat(format) {
  const value = String(format || "").toLowerCase();
  return FORMAT_SCOPES[value] ? value : "tflite";
}

export function formatEvidenceScope(format, { analysis = null, runtimeEvidence = null } = {}) {
  const id = normalizedFormat(format || analysis?.format);
  const scope = FORMAT_SCOPES[id];
  let placement = null;
  let placementError = null;
  if (analysis) {
    try { placement = buildExecutionPlacementEvidence(analysis, runtimeEvidence); }
    catch (error) { placementError = error instanceof Error ? error.message : String(error); }
  }
  const runtimeObservation = placement?.runtime_observation || null;
  const observedRuntime = runtimeObservation?.status === "observed";
  const partialRuntime = observedRuntime && runtimeObservation.covered_item_count < runtimeObservation.total_item_count;
  const configurationLevel = placement?.levels?.find((level) => level.id === "configuration_bound");
  const runtimeConfigurationBound = Boolean(configurationLevel && !["UNBOUND", "EXTERNAL"].includes(configurationLevel.state));
  const placementEstimateBound = placement?.flow?.evidence_basis === "ANTICIPATED_MLCOMPUTEPLAN";
  const graphDecoded = Array.isArray(artifactIrOperators(analysis)) && artifactIrOperators(analysis).length > 0;
  const depth = id === "coreml" && analysis && !graphDecoded
    ? "Model/package contract audit; serialized graph not decoded for this model type"
    : scope.depth;
  return {
    id,
    ...scope,
    depth,
    runtimeObserved: observedRuntime,
    runtimePartiallyObserved: partialRuntime,
    runtimeConfigurationBound,
    placementEstimateBound,
    placementEvidenceInvalid: Boolean(placementError),
    evidenceClass: "STATIC ARTIFACT EVIDENCE",
    runtimeStatus: placementError
      ? `Imported placement evidence was rejected: ${placementError}`
      : observedRuntime
        ? partialRuntime
          ? `Imported runtime execution evidence is identity-bound for ${runtimeObservation.covered_item_count}/${runtimeObservation.total_item_count} items; remaining placement is unobserved`
          : "Imported runtime execution evidence is identity-bound for the complete assessed scope"
      : placementEstimateBound
        ? "An identity-bound MLComputePlan estimate is imported; execution remains unobserved"
        : runtimeConfigurationBound
          ? "Runtime configuration is bound; execution placement remains unobserved"
          : "Runtime execution not observed in this run",
    releaseStatus: "Release readiness not assessed: deployment requirements and representative task-output acceptance evidence are not bound",
  };
}

export function formatAuditButtonLabel(format, { rerun = false } = {}) {
  const scope = formatEvidenceScope(format);
  return rerun ? `Re-run ${scope.label} Audit` : scope.runLabel;
}

export function stagedArtifactContext(format) {
  const scope = formatEvidenceScope(format);
  const copy = scope.id === "tflite"
    ? "Choose the TFLite CPU cost profile below. GPU and NNAPI source eligibility is evaluated separately in Accelerator; build inclusion and observed placement require imported evidence."
    : scope.id === "onnx"
      ? "Run the target-independent graph and ORT compatibility audit first. Bind the exact ORT version, execution provider, build flags, and device through runtime evidence afterward."
      : `${scope.depth}. No generic CPU target is applied to this format.`;
  return { title: `${scope.label} artifact resolved`, copy };
}

export function renderStagedArtifactContext(doc, format) {
  const context = stagedArtifactContext(format);
  const title = doc?.getElementById("artifactContextTitle");
  const copy = doc?.getElementById("artifactContextCopy");
  if (title) title.textContent = context.title;
  if (copy) copy.textContent = context.copy;
}

export function formatWorkflowApplicability(format, analysis = null) {
  const id = normalizedFormat(format || analysis?.format);
  const graph = id === "tflite" || id === "onnx" || id === "executorch" && (artifactIrOperators(analysis) || []).length > 0 || id === "coreml" && (artifactIrOperators(analysis) || []).length > 0;
  const runtime = id === "tflite" || id === "onnx";
  const tfliteResearch = id === "tflite";
  return {
    graph,
    redesign: id === "tflite",
    runtime,
    protectedSourceAnalysis: id === "tflite" || id === "onnx",
    tfliteResearch,
  };
}

export const SUPPORTED_FORMAT_EVIDENCE_SCOPES = FORMAT_SCOPES;
