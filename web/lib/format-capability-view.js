import { runtimeCapturePlanAvailability } from "./runtime-evidence-closure.js";

const COLUMNS = Object.freeze([
  ["artifact", "Artifact"],
  ["graph", "Graph"],
  ["numeric", "Numerical"],
  ["deployment", "Deployment"],
  ["runtime", "Runtime"],
]);

const MATRIX = Object.freeze({
  tflite: Object.freeze({
    label: "TFLite",
    compact: "serialized graph and numerical contracts, source-pinned static placement candidates, and runtime evidence import",
    cells: ["full", "all_subgraphs", "quant", "tflite_source", "import"],
    summary: "Every serialized subgraph, local tensor/operator/I/O index, SignatureDef entrypoint, and schema-defined control-flow/computation reference is assessed. Each subgraph receives independent target-aware MAC, quantization, liveness, ArenaPlanner, weight-integrity, numerical, and source-pinned XNNPACK candidate ledgers. Cross-subgraph execution totals remain runtime-scoped because invocation multiplicity is not serialized.",
  }),
  onnx: Object.freeze({
    label: "ONNX",
    compact: "serialized graph and Q/DQ contracts, independent accelerator eligibility, TensorRT parser preflight, and observed profile import",
    cells: ["full", "dag", "quant", "source", "import"],
    summary: "Graph, opset, shape, Q/DQ, custom-domain floors, and source-backed ORT EP profiles including DirectML and WebGPU. TensorRT native parser support can be imported from a hash-bound, SDK-version-aware capability collector; ORT TensorRT EP remains a separate GetCapability evidence path. Neither parser acceptance nor source eligibility is promoted to executed assignment.",
  }),
  gguf: Object.freeze({
    label: "GGUF",
    compact: "container and tensor payload evidence, canonical decoder scenarios, and instrumented runtime-graph import",
    cells: ["container", "generated_graph", "payload", "source", "import"],
    summary: "Container, exact tensor payload, source-pinned GGML dequantization, architecture/tokenizer, registered canonical-decoder parameter/KV/compute scenarios, and lower-bound-only memory feasibility, plus llama.cpp backend prerequisite prechecks. The file does not serialize an execution DAG; an exact-source instrumented runtime manifest can separately import generated GGML graphs, scheduler-expanded nodes, split ranges, per-node backend assignment, transition edges, and dispatch status.",
  }),
  safetensors: Object.freeze({
    label: "SafeTensors",
    compact: "checkpoint payload evidence, source-pinned packed-weight layouts, bound architecture scenarios, TensorRT-LLM configuration, and external runtime contracts",
    cells: ["container", "canonical_scenario", "payload", "external", "external"],
    summary: "Checkpoint, shard, exact payload, numerical-integrity, and storage evidence; supported AWQ, GPTQ, HQQ, and compressed-tensors declarations add source-pinned packed-layout, group-cardinality, and bit-capacity conservation. A bound config for 18 source-pinned dense, sparse-MoE, and SSM families adds canonical tensors, parameter/state ledgers, bounded compute scenarios, and lower-bound-only memory feasibility. An optional TensorRT-LLM engine config validates TP/PP/CP and layer conservation, build limits, quantization declarations, and conditional logical KV state against a non-circular model-source digest. Engine execution, tactics, occupancy, latency, and task quality remain separate evidence.",
  }),
  coreml: Object.freeze({
    label: "Core ML",
    compact: "serialized model contracts and OS floor, representation-specific numerical evidence, and MLComputePlan import",
    cells: ["container", "dag", "quant", "floor", "import"],
    summary: "Package and interfaces plus NeuralNetwork, ML Program, GLM, SVM, TreeEnsemble, and Pipeline contracts; decoded representations expose DAG/structure, shape, arithmetic, numerical payload, logical liveness, source-backed serialized blockwise/LUT/sparse compression contracts, and an exact specification-to-OS floor. MLComputePlan import applies only to NeuralNetwork and ML Program and is never labeled an execution trace.",
  }),
  executorch: Object.freeze({
    label: "ExecuTorch",
    compact: "ET12 execution plans, EValue tensors, serialized delegate calls, AOT memory, and FT01 external data",
    cells: ["full", "dag", "tensor_contract", "serialized_delegate", "external"],
    summary: "ET12 exposes ordered execution plans, source-bound portable operator/delegate calls, tensor shape/dtype/layout contracts, segment-backed constants, processed payload identities, allocation offsets, and planned non-constant buffers. A selected-build sidecar can bind source/build options, backend/operator inventories, and runtime binary digests. FT01 exposes named external tensor/blob segments. Delegate-internal semantics, initialization, executed placement, runtime allocation, kernels, physical transfer, and latency remain external.",
  }),
});

const CELL = Object.freeze({
  full: ["Assessed", "complete", "Directly decoded and validated from the artifact."],
  container: ["Container", "complete", "Container identity, structure, declared ranges, and applicable package bindings are decoded and validated."],
  all_subgraphs: ["All scopes", "complete", "Every serialized subgraph and control-flow reference is decoded and independently deep-assessed; rows are not summed without runtime invocation counts."],
  quant: ["Assessed", "complete", "Affine or stored numerical contracts are assessed where serialized."],
  payload: ["Payload audit", "complete", "Every source-bound tensor payload is range-hashed and numerically decoded with explicit byte and logical-value conservation; unsupported producer-dependent encodings remain explicit."],
  tensor_contract: ["Tensor contract", "partial", "Serialized dtype, shape, layout, storage span, and memory-plan contracts are assessed; opaque backend blobs are not numerically decoded."],
  dag: ["DAG + costs", "partial", "Decoded model representations expose serialized operations, shapes, assessed MACs, weights, and logical liveness; coverage remains model-type and rule dependent."],
  generated_graph: ["Runtime-generated", "partial", "The artifact has no serialized execution DAG. An exact-source instrumented runtime manifest can import generated scheduler graphs, split ranges, per-node backend assignment, transition edges, and dispatch status."],
  canonical_scenario: ["Canonical scenario", "partial", "The container has no serialized execution DAG. A bound config for a registered source-pinned decoder family can derive canonical tensor, parameter, KV-state, prefill, and decode ledgers without claiming an executed graph."],
  predicted: ["Predicted", "partial", "Pinned source and selected-target static model; not executed placement."],
  tflite_source: ["Predicted + source", "partial", "XNNPACK is target-modeled; GPU and NNAPI are pinned-source artifact candidates. None establishes selected-build availability or executed placement."],
  source: ["Source-backed", "partial", "Pinned implementation registration and artifact-visible prerequisites; not selected-build availability, device capability, or executed assignment."],
  serialized_delegate: ["Serialized", "partial", "The artifact contains an AOT delegate call and backend identifier. This is not proof that the selected runtime can load or execute that backend."],
  floor: ["OS floor", "complete", "The serialized Core ML specification and feature use determine an exact necessary OS/Core ML availability floor within the pinned source range; compute-unit placement remains external."],
  runtime: ["Runtime needed", "partial", "Serialized graph evidence does not establish native compute-unit placement."],
  import: ["Import", "missing", "Observed evidence is accepted only when identity and graph binding validate."],
  external: ["External contract", "missing", "The container does not encode this evidence class."],
  na: ["Not applicable", "na", "This artifact class does not serialize an execution-operator graph."],
});

const CURRENT_CELL = Object.freeze({
  assessed: ["Assessed", "complete"],
  decoded: ["Decoded", "complete"],
  generated_observed: ["Generated graph", "complete"],
  canonical_scenario: ["Canonical scenario", "partial"],
  scoped: ["All scopes", "partial"],
  full_payload: ["Full payload", "complete"],
  partial: ["Partial", "partial"],
  predicted: ["Predicted", "partial"],
  source: ["Source-backed", "partial"],
  predicted_source: ["Predicted + source", "partial"],
  floor: ["OS floor", "complete"],
  runtime_needed: ["Runtime needed", "partial"],
  import_ready: ["Import ready", "missing"],
  not_loaded: ["Not loaded", "missing"],
  not_decoded: ["Not decoded", "missing"],
  unbound: ["Unbound", "missing"],
  external: ["External runtime", "missing"],
  observed: ["Observed", "complete"],
  plan_bound: ["Plan bound", "partial"],
  build_bound: ["Build bound", "partial"],
  serialized: ["Serialized", "partial"],
  invalid: ["Invalid", "missing"],
  na: ["Not applicable", "na"],
});

function normalizedStatus(value) {
  return String(value || "").toLowerCase();
}

function coverageState(status, { full = "assessed", absent = "not_decoded" } = {}) {
  const value = normalizedStatus(status);
  if (value === "assessed" || value.startsWith("assessed_") || value === "complete" || value.startsWith("complete_")) return full;
  if (value === "invalid" || value.startsWith("invalid_")) return "invalid";
  if (value === "partial" || value.startsWith("partial_")) return "partial";
  return absent;
}

function runtimeObserved(format, runtimeEvidence) {
  const comparison = runtimeEvidence?.comparison || runtimeEvidence?.runtime_assignment?.comparison;
  if (format === "gguf") {
    return Number(runtimeEvidence?.compute_graph?.graph_count || 0) > 0
      && (runtimeEvidence.compute_graph.graphs || []).some((graph) => (graph.scheduled_nodes || []).some((row) => row.backend));
  }
  if (!["tflite", "onnx"].includes(format)) return false;
  return Boolean(
    Number(runtimeEvidence?.assignment_count || 0) > 0
    || Number(comparison?.observed_op_count || 0) > 0
    || comparison?.op_comparisons?.some?.((row) => row?.observed_provider || row?.observed_delegated != null),
  );
}

function cell(id, title) {
  const [label, tone] = CURRENT_CELL[id];
  return Object.freeze({ id, label, tone, title });
}

function coverageFraction(assessed, total, noun) {
  if (assessed == null || total == null) return null;
  const assessedValue = Number(assessed);
  const totalValue = Number(total);
  return Number.isFinite(assessedValue) && Number.isFinite(totalValue)
    ? `${assessedValue}/${totalValue} ${noun}` : null;
}

export function deriveCurrentArtifactCapabilityRow(format, analysis, runtimeEvidence = null) {
  const id = MATRIX[String(format || analysis?.format || "").toLowerCase()] ? String(format || analysis?.format).toLowerCase() : "";
  if (!id || !analysis) return null;
  const ops = Array.isArray(analysis.ops) ? analysis.ops : [];
  const observed = runtimeObserved(id, runtimeEvidence);
  let artifact = cell("assessed", "The selected artifact identity and serialized container contract were decoded and validated.");
  let graph = cell("decoded", "The serialized operator graph was decoded from this artifact.");
  let numeric = cell("assessed", "Serialized numerical and quantization contracts were assessed for this artifact.");
  let deployment = cell("unbound", "No deployment implementation is selected by this artifact.");
  let runtime = observed
    ? cell("observed", "Identity-bound runtime evidence is present for this artifact.")
    : cell("external", "Runtime execution is outside this artifact and no observed evidence is bound.");

  if (id === "tflite") {
    const subgraphCount = Number(analysis.subgraphs || 1);
    const subgraphInventory = analysis.tflite_subgraph_inventory || {};
    const intrinsicRows = Array.isArray(subgraphInventory.rows) ? subgraphInventory.rows : [];
    const assessedIntrinsicRows = intrinsicRows.filter((row) => String(row?.intrinsic_cost?.status || "").startsWith("assessed")).length;
    const partialIntrinsicRows = intrinsicRows.filter((row) => row?.intrinsic_cost?.status === "partial").length;
    graph = subgraphCount > 1
      ? cell("scoped", `${subgraphInventory.parsed_subgraph_count || 0}/${subgraphCount} subgraphs, ${subgraphInventory.serialized_operator_count || 0} serialized operators, ${subgraphInventory.control_flow_reference_count || 0} control-flow/computation references, and ${subgraphInventory.assessed_control_flow_contract_count || 0}/${subgraphInventory.control_flow_contract_count || 0} pinned IF/WHILE/CALL_ONCE contracts are assessed. Deep rows: ${analysis.tflite_subgraph_deep_analysis?.assessed_subgraph_count || 0}/${subgraphCount}; intrinsic MAC/payload rows: ${assessedIntrinsicRows} assessed + ${partialIntrinsicRows} partial. Rows are not summed across conditional or repeated execution.`)
      : cell("decoded", `The artifact's single serialized subgraph is decoded and assessed, including local tensor/operator/I/O references and its one-invocation intrinsic nominal-MAC/payload ledger (${assessedIntrinsicRows} assessed + ${partialIntrinsicRows} partial row).`);
    numeric = cell("assessed", `${analysis.quantized_tensors ?? 0}/${analysis.tensor_count ?? 0} primary-subgraph tensors carry affine quantization metadata; ${analysis.tflite_sparse_storage_contract?.fully_decoded_tensor_count || 0}/${analysis.tflite_sparse_storage_contract?.sparse_tensor_count || 0} sparse storage contracts are fully reconstructed across all subgraphs. Every serialized subgraph has an independent quantization, dense-weight integrity, and applicable fixed-point proof row.`);
    const alternateDelegates = analysis.tflite_delegate_compatibility_evidence;
    const delegateBuild = runtimeEvidence?.tflite_delegate_build_inventory;
    deployment = analysis.target_profile && analysis.delegated_mac_percent != null && alternateDelegates && delegateBuild
      ? cell("predicted_source", `Pinned XNNPACK rules and GPU/NNAPI artifact prechecks are present. The selected build inventory reports GPU ${delegateBuild.gpu.compiled_status}, GPU quant option ${delegateBuild.gpu.quantized_model_flag_status}, and NNAPI ${delegateBuild.nnapi.compiled_status}; device acceptance and executed placement remain unobserved.`)
      : analysis.target_profile && analysis.delegated_mac_percent != null && alternateDelegates
        ? cell("predicted_source", "Pinned XNNPACK rules were evaluated against the selected target profile, while GPU and NNAPI source candidates were narrowed by artifact-visible checks. Selected build, device acceptance, and executed placement remain unobserved.")
      : analysis.target_profile && analysis.delegated_mac_percent != null
        ? cell("predicted", "Pinned XNNPACK rules were evaluated against the selected static target profile; the protected GPU/NNAPI candidate rulepack is not loaded and this is not executed placement.")
      : cell("not_loaded", "A selected target profile and complete static delegation result are not present.");
    runtime = observed ? runtime : cell("import_ready", "Identity- and graph-bound TFLite runtime evidence can be imported.");
  } else if (id === "onnx") {
    const shape = analysis.onnx_shape_inference || {};
    const macs = analysis.mac_assessment || {};
    const shapeStatus = coverageState(analysis.onnx_shape_inference?.status, { full: "decoded", absent: "partial" });
    graph = cell(shapeStatus, shapeStatus === "decoded"
      ? `The ModelProto graph and bounded shape/type contracts were decoded. ${coverageFraction(shape.rule_supported_nodes, shape.attempted_nodes, "shape-rule nodes") || "Shape-rule denominator is retained in structured evidence"}; ${coverageFraction(macs.assessed_compute_ops, macs.compute_ops, "MAC-bearing ops") || "MAC denominator is retained in structured evidence"}.`
      : `The graph was decoded, but one or more shape/type contracts remain partial. ${coverageFraction(shape.rule_supported_nodes, shape.attempted_nodes, "shape-rule nodes") || "Inspect the unresolved-node ledger"}; ${coverageFraction(macs.assessed_compute_ops, macs.compute_ops, "MAC-bearing ops") || "MAC coverage is not closed"}.`);
    const weightState = coverageState(analysis.weight_integrity?.status, { absent: "partial" });
    numeric = cell(weightState === "assessed" ? "assessed" : weightState, `Q/DQ, dtype, dense/sparse initializer, and numerical-integrity coverage follows the emitted per-artifact status: ${analysis.weight_integrity?.weight_tensors_scanned ?? "unresolved"}/${analysis.weight_integrity?.initializer_tensors_present ?? "unresolved"} initializer tensors assessed; ${analysis.weight_integrity?.initializer_tensors_unassessed ?? "unresolved"} residual.`);
    const compatibilityStatus = normalizedStatus(analysis.ort_compatibility_assessment_status);
    const selectedBuild = runtimeEvidence?.source?.adapter?.native_capture?.selected_build_provider_binding || null;
    const tensorRt = analysis.tensorrt_static_preflight || null;
    const tensorRtSummary = tensorRt?.status?.startsWith("parser_observed")
      ? ` TensorRT native parser observation is identity-bound (${tensorRt.projection?.state_counts?.CONDITIONALLY_ELIGIBLE || 0}/${ops.length} conditionally eligible); it is not ORT TensorRT EP assignment or execution.`
      : tensorRt?.status && tensorRt.status !== "not_selected"
        ? ` TensorRT preflight status is ${tensorRt.status}; parser/GetCapability observation remains required before support rows can be promoted.` : "";
    deployment = selectedBuild
      ? cell("build_bound", `The selected ORT ${selectedBuild.source_build_attestation ? "source build" : "package"} reports ${selectedBuild.bindings.map((row) => `${row.backend_name}:${row.bundled ? "bundled" : "available"}`).join(" / ")}; these build observations are cross-referenced to the pinned source profiles. ${selectedBuild.reduced_operator_inventory_status === "BUILD_INPUT_BINARY_ATTESTED" ? "The exact reduced-op config is attested as a selected-build input; this establishes configured inclusion intent, not kernel registration or acceptance." : "Reduced-op inclusion is not binary-attested."} GetCapability acceptance and assignment remain separate.`)
      : compatibilityStatus === "complete" || compatibilityStatus.startsWith("assessed")
        ? cell("source", `Pinned ORT execution-provider registrations and artifact-visible preconditions were evaluated; selected-build inclusion and GetCapability assignment were not observed.${tensorRtSummary}`)
      : tensorRt
        ? cell("source", `ORT source profiles are incomplete for this audit.${tensorRtSummary}`)
        : cell("not_loaded", "Source-backed ORT execution-provider evidence is not loaded for this audit.");
    runtime = observed ? runtime : cell("import_ready", "Identity- and graph-bound ORT profile evidence can be imported.");
  } else if (id === "gguf" || id === "safetensors") {
    const inventoryState = coverageState(analysis.tensor_inventory?.status, { absent: "partial" });
    artifact = cell(inventoryState, "Container structure, tensor directory, declared byte ranges, and payload conservation follow the emitted per-artifact status.");
    graph = cell("na", "This tensor container does not serialize an execution-operator graph.");
    const numericalState = coverageState(analysis.tensor_numerical_integrity?.status, { full: "full_payload", absent: "not_decoded" });
    const integrity = analysis.tensor_numerical_integrity || {};
    numeric = cell(numericalState, numericalState === "full_payload"
      ? `${integrity.assessed_tensor_count || 0}/${integrity.tensor_count || 0} tensor payloads and ${integrity.assessed_tensor_bytes || 0}/${integrity.declared_tensor_bytes || 0} declared bytes were SHA-256 bound and numerically decoded; ${integrity.nonfinite_value_count || 0} non-finite values, ${integrity.exact_zero_value_count || 0} exact zeros, and ${integrity.byte_conservation_status || "unknown"} byte conservation.`
      : `${integrity.assessed_tensor_count || 0}/${integrity.tensor_count || 0} tensor payloads and ${integrity.assessed_tensor_bytes || 0}/${integrity.declared_tensor_bytes || 0} declared bytes were decoded. ${integrity.unassessed_tensor_count || 0} encoding(s) remain explicit in the numerical-integrity limitation ledger; byte conservation is ${integrity.byte_conservation_status || "unresolved"}.`);
    if (id === "gguf") {
      const backendStatus = normalizedStatus(analysis.gguf?.backend_compatibility?.status);
      deployment = backendStatus === "source_candidate"
        ? cell("source", "Pinned llama.cpp architecture, GGML storage, backend build-option, and registration gates were evaluated. Runtime graph support, selected build, device, and execution remain unbound.")
        : backendStatus === "invalid"
          ? cell("invalid", "The artifact fails a pinned llama.cpp architecture or serialized-storage prerequisite; inspect the backend compatibility ledger.")
          : cell("unbound", "No source-backed llama.cpp architecture candidate was established for this artifact.");
      runtime = runtimeEvidence?.schema === "deepbom.gguf_runtime_environment.v2"
        ? cell("build_bound", `A pinned-source instrumented llama.cpp manifest is imported with selected-build attestation and ${runtimeEvidence.compute_graph?.dispatched_graph_count || 0} dispatched scheduler graph(s). Captured split and backend assignment are observed; microkernel identity remains unobserved.`)
        : cell("import_ready", "An identity-bound llama.cpp build and runtime-environment manifest can be imported.");
      if (runtimeEvidence?.schema === "deepbom.gguf_runtime_environment.v2") {
        const generated = runtimeEvidence.compute_graph || {};
        graph = cell("generated_observed", `No execution DAG is serialized in GGUF. The bound instrumented runtime observed ${generated.graph_count || 0} generated scheduler graph(s), ${generated.original_node_count || 0} original and ${generated.scheduled_node_count || 0} scheduled node rows, ${generated.split_count || 0} split(s), and ${generated.successful_dispatch_count || 0}/${generated.dispatch_count || 0} successful dispatch(es). These rows apply only to the captured calls.`);
      }
    } else {
      const hf = analysis.safetensors?.hf_architecture_contract || {};
      const packedQuant = analysis.safetensors?.quantization_contract || {};
      if (packedQuant.status === "assessed") {
        numeric = cell("source", `${numeric.title} ${String(packedQuant.method || "packed").toUpperCase()} ${packedQuant.bits || "?"}-bit ${packedQuant.granularity || "grouped"} layout: ${packedQuant.valid_module_count || 0}/${packedQuant.module_count || 0} modules conserve ${packedQuant.logical_weight_element_count || "?"} logical weight codes against ${packedQuant.packed_weight_code_capacity || "?"} packed capacity under ${packedQuant.source?.repository || "a pinned implementation"}@${packedQuant.source?.commit || "unbound"}. Packed values and calibration quality remain unassessed.`);
      } else if (packedQuant.status === "fail") {
        numeric = cell("invalid", `${numeric.title} The selected ${String(packedQuant.method || "packed").toUpperCase()} declaration has ${packedQuant.invalid_module_count || 0} invalid module layout(s) or conflicting configuration evidence; inspect the packed-weight ledger.`);
      } else {
        numeric = cell(numeric.id, `${numeric.title} No supported repository-bound packed-weight declaration was assessed; absence is not treated as an unquantized checkpoint.`);
      }
      if (["assessed", "partial"].includes(hf.status)) {
        const shapeChecks = hf.tensor_contract?.canonical_tensor_check_count || 0;
        const shapeMatches = hf.tensor_contract?.canonical_tensor_shape_match_count || 0;
        const architectureEvidence = hf.architecture_kind === "hybrid_attention_ssm_moe"
          ? "exact hybrid attention/Mamba layer schedule, KV plus recurrent-state, and active expert ledgers"
          : hf.architecture_kind === "sparse_moe_decoder"
          ? "exact total/active expert and router ledgers"
          : hf.architecture_kind === "ssm_recurrent"
            ? "exact recurrent-state and bounded projection/depthwise-convolution ledgers"
            : "exact parameter and KV-state plus prefill/decode ledgers";
        graph = cell("canonical_scenario", `SafeTensors serializes no execution DAG. Bound ${hf.model_type || "registered"} config and pinned source derive a ${hf.tensor_layout_id || "registered"} architecture scenario, ${shapeMatches}/${shapeChecks} canonical tensor-shape matches, and ${architectureEvidence}. This is not an executed framework graph.`);
      }
      const llmRuntime = analysis.on_device_llm?.runtime_contract || {};
      const tensorRtLlm = analysis.on_device_llm?.tensorrt_llm || {};
      const runtimeBound = String(llmRuntime.status || "").startsWith("artifact_bound_");
      deployment = tensorRtLlm.status === "artifact_bound_configuration"
        ? cell("build_bound", `A TensorRT-LLM engine config is bound to model-source and config digests. World/TP/PP/CP are ${tensorRtLlm.parallelism?.world_size || "?"}/${tensorRtLlm.parallelism?.tensor_parallel_size || "?"}/${tensorRtLlm.parallelism?.pipeline_parallel_size || "?"}/${tensorRtLlm.parallelism?.context_parallel_size || "?"}; layer partition conservation is ${tensorRtLlm.parallelism?.layer_partition_conservation || "unresolved"}. This is configuration evidence, not engine execution.`)
        : tensorRtLlm.status === "candidate_configuration_unbound"
          ? cell("source", "A TensorRT-LLM engine config was parsed and source-checked, but no model-source binding manifest identifies it as the selected deployment configuration.")
      : runtimeBound
        ? cell("source", `A hash-bound ${llmRuntime.evidence_class} runtime manifest conserves exclusive weight residency and complete layer placement for ${llmRuntime.deployment?.context_length || "?"} context / batch ${llmRuntime.deployment?.batch_size || "?"}. It does not establish kernels or latency.`)
        : cell("unbound", "The container does not select an engine, backend, build, or device target. A canonical architecture scenario, when present, is not deployment placement.");
      runtime = runtimeBound
        ? cell("build_bound", `${llmRuntime.runtime?.engine || "runtime"} ${llmRuntime.runtime?.version || ""} is artifact- and binary-bound; state residency ${llmRuntime.state_cache?.resident_bytes?.decimal || "?"}/${llmRuntime.state_cache?.allocated_bytes?.decimal || "?"} B.`)
        : cell("import_ready", `${tensorRtLlm.status && tensorRtLlm.status !== "not_selected" ? "TensorRT-LLM configuration does not establish engine execution. " : ""}A deepbom.runtime.json sidecar can bind runtime identity, residency, layer placement, and KV/SSM paging.`);
    }
  } else if (id === "executorch") {
    const pte = analysis.executorch_container === "pte";
    const program = analysis.executorch_program || {};
    const buildBinding = program.selected_build_binding || {};
    const selectedBuild = buildBinding.selected_build || null;
    const payloads = Array.isArray(program.processed_backend_payloads) ? program.processed_backend_payloads : [];
    const boundedPayloads = payloads.filter((row) => row.structural_status === "OBSERVED_BOUNDED_FLATBUFFER_ROOT_ENVELOPE").length;
    const payloadContradictions = payloads.filter((row) => String(row.structural_status || "").startsWith("CONTRADICTION_")).length;
    artifact = cell("assessed", `The ${pte ? "ET12 program" : "FT01 tensor-data container"}, FlatBuffer boundaries, extended header, and ${pte ? "program" : "data"} segments were decoded and range-validated.`);
    graph = pte
      ? cell("decoded", `${analysis.subgraphs || 0} execution plan(s), ${analysis.operator_count || 0} ordered instruction(s), and ${analysis.tensor_count || 0} Tensor EValue(s) were decoded. Matching portable KernelCall argument direction is bound to the pinned ${program.operator_signature_registry?.portable_operator_count || 0}-operator registry; custom and mismatched calls remain unassessed.`)
      : cell("na", "FT01 stores named external tensor/blob data and no execution plan.");
    numeric = cell(payloadContradictions ? "conflict" : "partial", `${analysis.weight_integrity?.assessed_tensors || 0}/${analysis.tensor_count || 0} tensor storage contract(s) have statically assessable serialized spans. Shape/cardinality, sub-byte storage, dim order, and segment conservation are checked. ${payloads.length} processed delegate payload(s) are byte-exact and SHA-256 bound; ${boundedPayloads} public-schema FlatBuffer root envelope(s) pass and ${payloadContradictions} contradict the pinned payload form. Delegate-internal semantics remain outside this decoder.`);
    deployment = pte && Number(program.delegate_instruction_count || 0) > 0
      ? cell(buildBinding.status === "CONTRADICTION_SELECTED_BUILD_CANNOT_SATISFY_SERIALIZED_PROGRAM" ? "conflict" : selectedBuild ? "build_bound" : "serialized", `${program.delegate_instruction_count} delegate call(s) and ${program.delegates?.length || 0} backend declaration(s) are serialized and checked against the pinned ${buildBinding.source_registry?.backend_count || 0}-backend registry. ${selectedBuild ? `${buildBinding.status}; selected-build attestation ${selectedBuild.attestation_sha256}.` : "No selected build or binary inventory is bound."} Successful backend initialization and execution are not observed.`)
      : pte
        ? cell("unbound", "No serialized delegate call establishes an accelerator backend for this program; runtime-selected or portable-kernel execution remains external.")
        : cell("na", "FT01 contains no execution placement contract.");
    runtime = selectedBuild
      ? cell("build_bound", `The selected source/build inputs, operator and backend inventories, binary inventory, and primary runtime binary digest are attested. This is build evidence, not proof of dead-strip behavior, backend initialization, executed placement, physical transfer, correctness, or latency.`)
      : cell("import_ready", "Add deepbom.executorch-build.json in the web sidecar selector or use CLI --executorch-build to bind source/build inputs, operator and backend inventories, and runtime binary digests. Execution remains a separate native evidence layer.");
  } else if (id === "coreml") {
    const macs = analysis.mac_assessment || {};
    const live = analysis.tensor_liveness || {};
    graph = ops.length
      ? cell("decoded", `Decoded ${ops.length} serialized Core ML operation${ops.length === 1 ? "" : "s"}; ${coverageFraction(macs.assessed_compute_ops, macs.compute_ops, "MAC-bearing ops") || "MAC coverage denominator is retained in evidence"}; logical liveness ${live.status || "not assessed"}.`)
      : cell("not_decoded", "This Core ML model representation did not yield a decoded executable graph.");
    const weightState = coverageState(analysis.weight_integrity?.status, { absent: "not_decoded" });
    const compression = analysis.coreml?.mil_compression_contract;
    numeric = cell(weightState, `Numerical-integrity coverage follows the emitted WeightParams, ML Program blob, classical FLOAT64, or nested pipeline status: ${analysis.weight_integrity?.assessed_parameter_count ?? "unresolved"}/${analysis.weight_integrity?.parameter_count ?? "unresolved"} parameter groups and ${analysis.weight_integrity?.assessed_payload_bytes ?? "unresolved"}/${analysis.weight_integrity?.payload_bytes ?? "unresolved"} payload bytes assessed; ${analysis.weight_integrity?.nonfinite_value_count ?? "unresolved"} non-finite values.${compression ? ` Serialized compression contracts: ${compression.exact_contract_count}/${compression.transform_count} exact; ${compression.partial_contract_count} explicit residual.` : ""}`);
    const floorStatus = normalizedStatus(analysis?.coreml?.deployment_floor?.status);
    deployment = floorStatus === "assessed"
      ? cell("floor", "The serialized specification and observed feature requirements determine an exact necessary OS/Core ML availability floor within the pinned source range. CPU/GPU/ANE placement remains external.")
      : cell("runtime_needed", "No complete source-bound Core ML OS floor was derived; CPU/GPU/ANE placement also requires an Apple runtime compute plan or execution capture.");
    const computePlanCapable = Boolean(analysis.coreml?.neural_network || analysis.coreml?.model_type === "mlProgram") && ops.length > 0;
    runtime = runtimeEvidence?.schema === "deepbom.coreml_compute_plan.v1"
      ? cell("plan_bound", "An identity- and op-order-bound MLComputePlan estimate is imported. It reports anticipated compute-device usage and cost, not executed placement or latency.")
      : computePlanCapable
        ? cell("import_ready", "An identity-bound Core ML MLComputePlan estimate can be imported for this decoded NeuralNetwork or ML Program graph.")
        : cell("external", "MLComputePlan's operation-level API does not address this decoded classical or pipeline representation; native execution evidence remains external.");
  }

  return Object.freeze({ id, label: MATRIX[id].label, cells: Object.freeze([artifact, graph, numeric, deployment, runtime]) });
}

export function renderFormatCapabilityMatrix(root, format, { analysis = null, runtimeEvidence = null } = {}) {
  if (!root) return;
  const id = MATRIX[String(format || "").toLowerCase()] ? String(format).toLowerCase() : "";
  const selected = MATRIX[id] || null;
  const summary = root.querySelector("[data-capability-summary]");
  const detail = root.querySelector("[data-capability-detail]");
  const scopeTable = root.querySelector("[data-scope-capability-table]");
  const head = scopeTable.querySelector("thead tr");
  const body = scopeTable.querySelector("tbody");
  const currentTitle = root.querySelector("[data-current-capability-title]");
  const currentWrap = root.querySelector("[data-current-capability-wrap]");
  const currentTable = root.querySelector("[data-current-capability-table]");
  const current = deriveCurrentArtifactCapabilityRow(id, analysis, runtimeEvidence);
  const runtimeBinding = root.querySelector("[data-format-runtime-binding]");
  summary.textContent = selected ? `${selected.label}: ${selected.compact}.` : "Compare deterministic evidence available from each supported artifact class.";
  detail.textContent = selected
    ? `${selected.summary}${runtimeObserved(id, runtimeEvidence) ? " Bound runtime execution evidence is present and shown separately from static results." : " Missing execution evidence remains explicit rather than being inferred."}`
    : "The same evidence spine is used across formats, while non-serialized evidence is marked not applicable or external.";
  head.replaceChildren(header("Format"), ...COLUMNS.map(([, label]) => header(label)));
  body.replaceChildren(...Object.entries(MATRIX).map(([formatId, row]) => {
    const tr = document.createElement("tr");
    if (formatId === id) tr.dataset.selected = "true";
    const name = document.createElement("th");
    name.scope = "row";
    name.dataset.column = "Format";
    name.textContent = row.label;
    tr.append(name, ...row.cells.map((cellId, index) => capabilityCell(cellId, COLUMNS[index][1], false)));
    return tr;
  }));
  currentTitle.hidden = !current;
  currentWrap.hidden = !current;
  if (current) {
    const currentHead = currentTable.querySelector("thead tr");
    const currentBody = currentTable.querySelector("tbody");
    currentHead.replaceChildren(header("Artifact"), ...COLUMNS.map(([, label]) => header(label)));
    const tr = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.dataset.column = "Artifact";
    name.textContent = analysis.filename || current.label;
    tr.append(name, ...current.cells.map((entry, index) => currentCapabilityCell(entry, COLUMNS[index][1])));
    currentBody.replaceChildren(tr);
  } else {
    currentTable.querySelector("thead tr").replaceChildren();
    currentTable.querySelector("tbody").replaceChildren();
  }
  renderFormatRuntimeBinding(runtimeBinding, id, analysis, runtimeEvidence);
}

function renderFormatRuntimeBinding(root, format, analysis, runtimeEvidence) {
  if (!root) return;
  const coreMlComputePlanCapable = format === "coreml"
    && Array.isArray(analysis?.ops) && analysis.ops.length > 0
    && Boolean(analysis.coreml?.neural_network || analysis.coreml?.model_type === "mlProgram");
  const applicable = format === "gguf" || coreMlComputePlanCapable;
  const importButton = root.querySelector("#importFormatRuntimeEvidence");
  const templateButton = root.querySelector("#downloadFormatRuntimeTemplate");
  const captureButton = root.querySelector("#downloadFormatRuntimeCapturePlan");
  const capturePlan = runtimeCapturePlanAvailability(analysis);
  if (importButton) importButton.disabled = !applicable;
  if (templateButton) templateButton.disabled = !applicable;
  if (captureButton) {
    captureButton.disabled = !applicable || !capturePlan.available;
    captureButton.title = capturePlan.available ? "Download an identity-bound native runtime collection plan" : capturePlan.reason;
  }
  root.hidden = !applicable;
  if (!applicable) return;
  const gguf = format === "gguf";
  const bound = runtimeEvidence?.schema === (gguf ? "deepbom.gguf_runtime_environment.v2" : "deepbom.coreml_compute_plan.v1");
  const title = root.querySelector("[data-format-runtime-title]");
  const detail = root.querySelector("[data-format-runtime-detail]");
  const clear = root.querySelector("#clearFormatRuntimeEvidence");
  title.textContent = bound
    ? gguf ? `${runtimeEvidence.selection.requested_backend_label} build bound` : `MLComputePlan bound (${runtimeEvidence.configuration.compute_units})`
    : gguf ? "llama.cpp build and backend unbound" : "Compiled Core ML compute plan unbound";
  detail.textContent = bound
    ? gguf
      ? `${runtimeEvidence.build.compiled_backend_profile_ids.join(" / ")} compiled; binary ${runtimeEvidence.runtime.binary_sha256.slice(0, 12)}...; inference ${runtimeEvidence.observations.inference_status}; scheduler graphs ${runtimeEvidence.compute_graph?.dispatched_graph_count || 0}/${runtimeEvidence.compute_graph?.graph_count || 0} dispatched.`
      : `${runtimeEvidence.structure.operation_count} operation estimates; compiled model ${runtimeEvidence.runtime.compiled_model_content_sha256.slice(0, 12)}...; execution remains unobserved.`
    : gguf
      ? "Import an instrumented collector manifest that binds the artifact, pinned llama.cpp source and patch, selected build, binary, CMake inventory, requested backend, device, generated scheduler graph, split ledger, and dispatch status."
      : "Import an Apple-hosted MLComputePlan manifest bound to the artifact, compiled model, compute-unit configuration, and exact decoded operation order.";
  if (clear) {
    clear.hidden = !bound;
    clear.disabled = !bound;
  }
}

function header(label) {
  const th = document.createElement("th");
  th.scope = "col";
  th.textContent = label;
  return th;
}

function capabilityCell(cellId, column, observed) {
  const [label, tone, title] = observed ? ["Observed", "complete", "Identity-bound imported runtime evidence is present."] : CELL[cellId];
  const td = document.createElement("td");
  td.className = "capability-value-cell";
  td.dataset.column = column;
  const badge = document.createElement("span");
  badge.className = `capability-state ${tone}`;
  badge.textContent = label;
  badge.title = `${column}: ${title}`;
  badge.tabIndex = 0;
  badge.setAttribute("aria-label", `${column}: ${label}. ${title}`);
  const detail = document.createElement("small");
  detail.className = "capability-cell-detail capability-scope-detail";
  detail.textContent = title;
  td.append(badge, detail);
  return td;
}

function currentCapabilityCell(entry, column) {
  const td = document.createElement("td");
  td.className = "capability-value-cell";
  td.dataset.column = column;
  const badge = document.createElement("span");
  badge.className = `capability-state ${entry.tone}`;
  badge.textContent = entry.label;
  badge.title = `${column}: ${entry.title}`;
  badge.tabIndex = 0;
  badge.setAttribute("aria-label", `${column}: ${entry.label}. ${entry.title}`);
  const detail = document.createElement("small");
  detail.className = "capability-cell-detail";
  detail.textContent = entry.title;
  td.append(badge, detail);
  return td;
}

export { MATRIX as FORMAT_CAPABILITY_MATRIX };
