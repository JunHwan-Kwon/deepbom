import {
  buildBackendPlacementProjection,
  ortProviderProjectionRows,
  tfliteDelegateProjectionRows,
  tfliteXnnpackProjectionRows,
  validateBackendPlacementProjection,
} from "./backend-placement-projection.js";

export const EXECUTION_PLACEMENT_EVIDENCE_SCHEMA = "deepbom.execution_placement_evidence.v1";

const LEVEL_IDS = Object.freeze([
  "artifact_observed",
  "source_pinned_eligibility",
  "configuration_bound",
  "runtime_evidence",
]);

export function buildExecutionPlacementEvidence(analysis, runtimeEvidence = null) {
  if (!analysis || typeof analysis !== "object") throw new Error("Execution placement requires analysis evidence.");
  const format = String(analysis.format || "").toLowerCase();
  const runtime = unwrapRuntimeEvidence(runtimeEvidence);
  let evidence;
  if (format === "tflite") evidence = tfliteEvidence(analysis, runtime);
  else if (format === "onnx") evidence = onnxEvidence(analysis, runtime);
  else if (format === "coreml") evidence = coreMlEvidence(analysis, runtime);
  else if (format === "gguf") evidence = ggufEvidence(analysis, runtime);
  else if (format === "safetensors") evidence = safeTensorsEvidence(analysis);
  else if (format === "executorch") evidence = execuTorchEvidence(analysis);
  else throw new Error(`Execution placement does not support format: ${format || "missing"}.`);
  const result = validateExecutionPlacementEvidence(evidence, { analysis });
  if (!result.valid) throw new Error(`Execution placement evidence is invalid: ${result.issues.join("; ")}`);
  return evidence;
}

export function validateExecutionPlacementEvidence(evidence, { analysis = null } = {}) {
  const issues = [];
  if (!evidence || typeof evidence !== "object") return { valid: false, issues: ["evidence must be an object"] };
  if (evidence.schema !== EXECUTION_PLACEMENT_EVIDENCE_SCHEMA) issues.push("schema mismatch");
  if (!["tflite", "onnx", "coreml", "gguf", "safetensors", "executorch"].includes(evidence.format)) issues.push("unsupported format");
  if (!Number.isSafeInteger(evidence.artifact_item_count) || evidence.artifact_item_count < 0) issues.push("artifact item count is invalid");
  if (!Array.isArray(evidence.levels) || evidence.levels.length !== LEVEL_IDS.length
    || evidence.levels.some((level, index) => level?.id !== LEVEL_IDS[index]
      || !level.label || !level.state || !level.tone || !level.evidence_class || !level.detail)) {
    issues.push("evidence level ladder is incomplete or out of order");
  }
  const flow = evidence.flow || {};
  const segments = Array.isArray(flow.segments) ? flow.segments : [];
  if (!Number.isSafeInteger(flow.scope_item_count) || flow.scope_item_count < 0) issues.push("flow scope item count is invalid");
  if (!Number.isSafeInteger(flow.covered_item_count) || flow.covered_item_count < 0
    || flow.covered_item_count > flow.scope_item_count) issues.push("flow covered item count is invalid");
  let expectedPosition = 0;
  for (const segment of segments) {
    if (!Number.isSafeInteger(segment.start_position) || !Number.isSafeInteger(segment.end_position)
      || !Number.isSafeInteger(segment.item_count) || segment.item_count <= 0
      || segment.start_position !== expectedPosition
      || segment.end_position !== segment.start_position + segment.item_count - 1
      || !segment.key || !segment.label || !segment.tone) {
      issues.push("flow segment positions or cardinality are invalid");
      break;
    }
    expectedPosition = segment.end_position + 1;
  }
  if (segments.length && expectedPosition !== flow.scope_item_count) issues.push("flow segments do not conserve the scope item count");
  if (!segments.length && flow.rendered_item_count !== 0) issues.push("empty flow has a non-zero rendered count");
  if (segments.length && flow.rendered_item_count !== expectedPosition) issues.push("rendered flow count does not match its segments");
  for (const profile of rows(evidence.portfolios)) {
    if (!profile.id || !profile.label || !profile.evidence_class
      || !Number.isSafeInteger(profile.candidate_count) || profile.candidate_count < 0
      || !Number.isSafeInteger(profile.total_count) || profile.total_count < profile.candidate_count) {
      issues.push("source portfolio count or identity is invalid");
      break;
    }
  }
  for (const projection of rows(evidence.static_profiles)) {
    try { validateBackendPlacementProjection(projection, { analysis }); }
    catch (error) { issues.push(String(error?.message || error)); break; }
  }
  for (const preflight of rows(evidence.configuration_preflights)) {
    if (!preflight?.schema || !preflight?.status || !preflight?.trust_boundary
      || !Number.isSafeInteger(preflight.blocking_issue_count) || !Number.isSafeInteger(preflight.unresolved_issue_count)) {
      issues.push("configuration preflight summary is invalid");
      break;
    }
  }
  const runtimeLevel = evidence.levels?.[3];
  if (evidence.format === "coreml" && runtimeLevel?.state !== "NOT OBSERVED") issues.push("Core ML plan was promoted to observed runtime evidence");
  if (evidence.format === "coreml" && evidence.portfolios.length
    && evidence.portfolios.reduce((sum, profile) => sum + profile.candidate_count, 0) !== flow.scope_item_count) {
    issues.push("Core ML anticipated-device counts do not conserve the plan rows");
  }
  if (evidence.format === "safetensors" && (segments.length || evidence.portfolios.length)) issues.push("SafeTensors contains fabricated placement evidence");
  if (["tflite", "onnx"].includes(evidence.format) && evidence.runtime_observation?.status === "observed"
    && evidence.runtime_observation.covered_item_count !== flow.covered_item_count) issues.push("runtime assignment coverage differs from flow coverage");
  if (analysis) {
    const expectedFormat = String(analysis.format || "").toLowerCase();
    const expectedCount = evidence.format === "gguf" || evidence.format === "safetensors"
      ? finiteInteger(analysis.tensor_count) : rows(analysis.ops).length;
    if (evidence.format !== expectedFormat) issues.push("analysis format binding mismatch");
    if (evidence.artifact_item_count !== expectedCount) issues.push("analysis item-count binding mismatch");
    if (String(evidence.artifact_sha256 || "") !== String(analysis.model_sha256 || "")) issues.push("analysis SHA-256 binding mismatch");
  }
  return { valid: issues.length === 0, issues };
}

export function unwrapRuntimeEvidence(value) {
  if (!value || typeof value !== "object") return null;
  return value.runtimeAssignmentEvidence || value.runtime_assignment || value;
}

function tfliteEvidence(analysis, runtime) {
  const ops = rows(analysis.ops);
  const assignments = canonicalAssignments(runtime, ops);
  const observed = assignments.rows.length > 0;
  if (observed) assertRuntimeArtifactBinding(analysis, runtime, "TFLite runtime assignment");
  const fullObservation = observed && assignments.rows.length === ops.length;
  const candidateCount = ops.filter((op) => signedInteger(op.xnnpack_chain_id, -1) >= 0).length;
  const buildRisks = rows(analysis.delegation_repair?.runtime_build_risks);
  const mismatchedBaseline = buildRisks.find((risk) => risk.baseline_conditionally_delegatable_op_count != null
    && finiteInteger(risk.baseline_conditionally_delegatable_op_count) !== candidateCount);
  if (mismatchedBaseline) throw new Error("TFLite runtime-build risk baseline differs from the canonical XNNPACK candidate count.");
  const alternate = analysis.tflite_delegate_compatibility_evidence;
  const buildBound = observed || Boolean(runtime?.tflite_delegate_build_inventory
    || runtime?.runtime_identity || runtime?.runtime_identity_status === "bound");
  const alternateProfiles = rows(alternate?.profiles)
    .filter((profile) => !/xnnpack/i.test(`${profile.id || ""} ${profile.label || ""}`))
    .map((profile) => portfolio(profile.id || profile.label, profile.label || profile.id,
      finiteInteger(profile.source_candidate_after_artifact_precheck_count), ops.length,
      "SOURCE_PINNED_ARTIFACT_PRECHECK", "artifact-visible source candidates"));
  const staticProfiles = Array.isArray(analysis.tensors) ? [
    buildBackendPlacementProjection({
      analysis,
      profileId: "xnnpack_cpu",
      label: "XNNPACK CPU",
      evidenceClass: "SOURCE_PINNED_PREDICTED",
      rows: tfliteXnnpackProjectionRows(analysis),
      source: {
        rule_basis: analysis.delegation_rule_basis || null,
        selector_assessment_status: analysis.xnnpack_selector_assessment_status || null,
      },
      interpretationBoundary: "CPU baseline derived from the pinned XNNPACK conditional-delegation ledger. It is not an accepted partition, selected microkernel, measured CPU workload, or timing result.",
    }),
    ...rows(alternate?.profiles)
      .filter((profile) => rows(profile?.rows).length === ops.length)
      .map((profile) => buildBackendPlacementProjection({
      analysis,
      profileId: profile.id,
      label: profile.label || profile.id,
      evidenceClass: profile.evidence_class || "SOURCE_PINNED/DERIVED_PARTIAL",
      rows: tfliteDelegateProjectionRows(profile),
      source: {
        tensorflow_source_commit: alternate.tensorflow_source_commit || null,
        rulepack_sha256: alternate.rulepack_sha256 || null,
      },
      })),
  ] : [];
  const segments = observed
    ? assignmentSegments(ops, assignments.byOp)
    : contiguous(ops, (op) => {
      const chainId = signedInteger(op.xnnpack_chain_id, -1);
      return chainId >= 0
        ? [`xnn-${chainId}`, `XNNPACK C${chainId}`, "candidate"]
        : [`cpu-${op.xnnpack_break_class || "fallback"}`, "CPU fallback", fallbackTone(op)];
    }, opItemId);
  const sourceDetail = alternate
    ? `XNNPACK ${candidateCount}/${ops.length} candidate ops; GPU and NNAPI artifact-visible candidates are separately source-pinned below.`
    : `XNNPACK ${candidateCount}/${ops.length} candidate ops. GPU/NNAPI source profiles are available from the protected source ledger.`;
  return baseEvidence(analysis, {
    format: "tflite",
    itemKind: "serialized_operator",
    itemCount: ops.length,
    title: "TFLite Execution Placement",
    subtitle: observed ? "Observed delegate assignment" : "XNNPACK predicted partition + alternate delegate eligibility",
    state: observed ? fullObservation ? "RUNTIME OBSERVED" : "PARTIAL RUNTIME OBSERVATION" : "SOURCE-PINNED PREDICTION",
    tone: observed ? fullObservation ? "complete" : "partial" : "partial",
    banner: !buildBound && buildRisks.length ? buildRiskBanner(buildRisks, candidateCount, ops.length) : null,
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", `${ops.length} serialized operators and their tensor contracts; the artifact does not select a delegate or runtime build.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", "ASSESSED", "partial", "SOURCE_PINNED_PREDICTED", sourceDetail),
      level("configuration_bound", "Configuration-bound", buildBound ? "BOUND" : "UNBOUND", buildBound ? "partial" : "missing", buildBound ? "CONFIGURATION_BOUND" : "NOT_OBSERVED", buildBound ? "Imported runtime/build identity is bound; device acceptance remains separate." : "Selected delegate build flags and device capability are not embedded in TFLite."),
      level("runtime_evidence", "Runtime evidence", observed ? fullObservation ? "OBSERVED" : "PARTIAL OBSERVED" : "NOT OBSERVED", observed ? fullObservation ? "complete" : "partial" : "missing", observed ? "OBSERVED_RUNTIME" : "NOT_OBSERVED", observed ? `${assignments.rows.length}/${ops.length} original-op assignments are identity-bound.` : "Executed delegate partitions, lowering, copies, and kernels require an imported target trace."),
    ],
    flow: flow(observed ? "Observed assignment flow" : "Conditional XNNPACK partition flow",
      observed ? "OBSERVED_RUNTIME_ASSIGNMENT" : "SOURCE_PINNED_PREDICTED_PARTITION",
      ops.length, observed ? assignments.rows.length : ops.length, segments),
    portfolios: [portfolio("xnnpack", "XNNPACK", candidateCount, ops.length, "SOURCE_PINNED_PREDICTED", "conditionally delegatable operators"), ...alternateProfiles],
    staticProfiles,
    note: observed
      ? "Observed rows apply only to the imported runtime capture; static XNNPACK and alternate-delegate ledgers remain separate comparisons."
      : "GPU and NNAPI rows are per-delegate source eligibility portfolios. They are not combined into a fictitious multi-delegate partition.",
    runtimeObservation: assignmentObservation(assignments, ops.length),
  });
}

function onnxEvidence(analysis, runtime) {
  const ops = rows(analysis.ops);
  const providers = rows(analysis.ort_compatibility_evidence?.execution_providers);
  const tensorRtPreflight = analysis.tensorrt_static_preflight;
  const assignments = canonicalAssignments(runtime, ops);
  const observed = assignments.rows.length > 0;
  if (observed) assertRuntimeArtifactBinding(analysis, runtime, "ONNX runtime assignment");
  const fullObservation = observed && assignments.rows.length === ops.length;
  const selectedBuild = runtime?.source?.adapter?.native_capture?.selected_build_provider_binding;
  const buildBound = observed || Boolean(selectedBuild);
  const portfolios = providers.map((provider) => portfolio(provider.execution_provider || provider.id || provider.label,
    provider.label || provider.execution_provider, finiteInteger(provider.source_candidate_after_artifact_precheck_count),
    ops.length, "SOURCE_PINNED_ARTIFACT_PRECHECK", "source candidates after artifact precheck"));
  const staticProfiles = Array.isArray(analysis.tensors) ? providers
    .filter((provider) => rows(provider?.ops).length === ops.length)
    .map((provider) => buildBackendPlacementProjection({
      analysis,
      profileId: provider.execution_provider || provider.id || provider.label,
      label: provider.label || provider.execution_provider || provider.id,
      evidenceClass: provider.support_evidence_class || "SOURCE_PINNED_ARTIFACT_PRECHECK",
      rows: ortProviderProjectionRows(provider),
      source: {
        source_id: provider.source_id || null,
        source_ref: provider.source_ref || null,
        source_sha256: provider.source_sha256 || null,
      },
    })) : [];
  if (tensorRtPreflight?.projection) staticProfiles.push(tensorRtPreflight.projection);
  return baseEvidence(analysis, {
    format: "onnx", itemKind: "modelproto_operator", itemCount: ops.length,
    title: "ONNX Execution Placement",
    subtitle: observed ? "Observed ORT provider assignment" : "Per-EP source eligibility portfolios",
    state: observed ? fullObservation ? "RUNTIME OBSERVED" : "PARTIAL RUNTIME OBSERVATION" : providers.length ? "SOURCE-PINNED ELIGIBILITY" : "SOURCE LEDGER NOT LOADED",
    tone: observed ? fullObservation ? "complete" : "partial" : providers.length ? "partial" : "missing",
    banner: observed ? null : "A provider-priority partition is intentionally not inferred from source registration alone. ORT graph optimization, provider order, selected build, GetCapability acceptance, and CPU fallback must be bound before a joint flow exists.",
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", `${ops.length} ModelProto operators, domains, opsets, and serialized graph contracts are decoded.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", providers.length ? `${providers.length} EP PROFILES` : "NOT LOADED", providers.length ? "partial" : "missing", providers.length ? "SOURCE_PINNED_ARTIFACT_PRECHECK" : "NOT_ASSESSED", providers.length ? "Each EP is assessed independently against pinned registrations and artifact-visible predicates." : "Load the source-backed ORT compatibility ledger to evaluate supported EP profiles."),
      level("configuration_bound", "Configuration-bound", buildBound ? "BUILD BOUND" : "UNBOUND", buildBound ? "partial" : "missing", buildBound ? "CONFIGURATION_BOUND" : "NOT_OBSERVED", buildBound ? observed && !selectedBuild ? "Imported runtime assignment binds the executed provider configuration; a separate selected-build inventory was not emitted." : "Selected package/source build provider inventory is bound; GetCapability acceptance is still not inferred." : "ORT version/build, provider priority, reduced-op configuration, and device are not selected by the ONNX artifact."),
      level("runtime_evidence", "Runtime evidence", observed ? fullObservation ? "OBSERVED" : "PARTIAL OBSERVED" : "NOT OBSERVED", observed ? fullObservation ? "complete" : "partial" : "missing", observed ? "OBSERVED_RUNTIME" : "NOT_OBSERVED", observed ? `${assignments.rows.length}/${ops.length} original-op provider assignments are bound to the imported profile.` : "Actual EP assignment, optimized nodes, lowering, memory pattern, and kernels require a bound ORT profile."),
    ],
    flow: flow(observed ? "Observed EP assignment flow" : "EP candidate portfolios",
      observed ? "OBSERVED_RUNTIME_ASSIGNMENT" : "INDEPENDENT_SOURCE_ELIGIBILITY_PORTFOLIOS",
      observed ? ops.length : 0, observed ? assignments.rows.length : 0,
      observed ? assignmentSegments(ops, assignments.byOp) : []),
    portfolios,
    staticProfiles,
    configurationPreflights: tensorRtPreflight ? [tensorRtPreflightSummary(tensorRtPreflight)] : [],
    note: observed
      ? "The flow is reconstructed from imported observed provider rows; it is not backfilled from static eligibility."
      : "Candidate counts answer whether each EP remains source-eligible. They do not answer which EP wins when providers are registered together.",
    runtimeObservation: assignmentObservation(assignments, ops.length),
  });
}

function coreMlEvidence(analysis, runtime) {
  const ops = rows(analysis.ops);
  const plan = runtime?.schema === "deepbom.coreml_compute_plan.v1" ? runtime : null;
  const planRows = rows(plan?.structure?.rows);
  if (plan) assertRuntimeArtifactBinding(analysis, plan, "Core ML compute plan");
  if (plan && planRows.length !== ops.length) {
    throw new Error("Core ML compute-plan operation count differs from the serialized graph.");
  }
  if (plan && planRows.some((row, position) => finiteInteger(row.op_index) !== finiteInteger(ops[position]?.index))) {
    throw new Error("Core ML compute-plan operation order differs from the serialized graph.");
  }
  const floor = analysis.coreml?.deployment_floor || {};
  const segments = planRows.length ? contiguous(planRows, (row) => {
    const device = row.preferred_compute_device || "not determined";
    return [device, device, "anticipated"];
  }, (row) => finiteInteger(row.op_index)) : [];
  return baseEvidence(analysis, {
    format: "coreml", itemKind: "serialized_operation", itemCount: ops.length,
    title: "Core ML Execution Placement",
    subtitle: plan ? "MLComputePlan anticipated device usage" : "Serialized graph + runtime-plan boundary",
    state: plan ? "ANTICIPATED - NOT EXECUTED" : "RUNTIME PLAN REQUIRED",
    tone: plan ? "estimated" : "missing",
    banner: plan ? "MLComputePlan preferred/supported compute devices and estimated relative cost are runtime-produced planning estimates for one compiled model and compute-unit configuration. They are not observed execution placement or measured timing." : null,
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", `${ops.length} serialized Core ML operation${ops.length === 1 ? "" : "s"} and interface contracts are decoded for this representation.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", floor.status === "assessed" ? "OS FLOOR DERIVED" : "PARTIAL", floor.status === "assessed" ? "complete" : "partial", "SOURCE_PINNED_DERIVED", "Serialized specification features determine a necessary OS/Core ML floor; they do not select CPU, GPU, or Neural Engine."),
      level("configuration_bound", "Configuration-bound", plan ? "PLAN BOUND" : "UNBOUND", plan ? "estimated" : "missing", plan ? "ANTICIPATED_RUNTIME_PLAN" : "NOT_OBSERVED", plan ? `${plan.configuration.compute_units}; compiled model and exact operation order are identity-bound.` : "Compile the model on Apple hardware and export an identity-bound MLComputePlan."),
      level("runtime_evidence", "Runtime evidence", "NOT OBSERVED", "missing", "NOT_OBSERVED", "Executed placement, fusion, allocation, and latency require a separate Apple runtime trace; MLComputePlan is not promoted to observed evidence."),
    ],
    flow: flow(plan ? "Anticipated preferred-device runs" : "No placement flow available",
      plan ? "ANTICIPATED_MLCOMPUTEPLAN" : "NOT_ASSESSED", planRows.length, planRows.length, segments),
    portfolios: plan ? Object.entries(countBy(planRows, (row) => row.preferred_compute_device || "not determined"))
      .map(([label, count]) => portfolio(label, label, count, planRows.length, "ANTICIPATED_RUNTIME_PLAN", "anticipated operations")) : [],
    note: plan ? plan.boundary : "Static Core ML decoding stops at artifact structure and OS availability. Placement is intentionally left unresolved.",
    runtimeObservation: { status: "not_observed", covered_item_count: 0, total_item_count: ops.length, rejected_row_count: 0 },
  });
}

function ggufEvidence(analysis, runtime) {
  const manifest = runtime?.schema === "deepbom.gguf_runtime_environment.v2" ? runtime : null;
  if (manifest) assertRuntimeArtifactBinding(analysis, manifest, "GGUF runtime environment");
  const computeGraph = manifest?.compute_graph || null;
  const graphCount = finiteInteger(computeGraph?.graph_count);
  const graphs = rows(computeGraph?.graphs);
  if (computeGraph && graphCount !== graphs.length) throw new Error("GGUF scheduler graph count differs from its graph inventory.");
  const graph = graphCount > 0 ? graphs[0] : null;
  const scheduled = rows(graph?.scheduled_nodes);
  if (graphCount > 0 && (!graph || !scheduled.length)) throw new Error("GGUF runtime evidence declares a captured graph without scheduled-node rows.");
  if (scheduled.some((row) => !String(row?.backend || "").trim())) throw new Error("GGUF scheduled-node evidence lacks a backend assignment.");
  const observed = graphCount > 0;
  const backend = analysis.gguf?.backend_compatibility || {};
  const segments = scheduled.length ? contiguous(scheduled, (row) => {
    const provider = row.backend || "unresolved";
    return [provider, provider, provider === "unresolved" ? "missing" : "observed"];
  }, (row, position) => finiteInteger(row.scheduled_index ?? row.node_index ?? position)) : [];
  return baseEvidence(analysis, {
    format: "gguf", itemKind: "serialized_tensor", itemCount: finiteInteger(analysis.tensor_count),
    title: "GGUF Execution Placement",
    subtitle: observed ? "Observed instrumented GGML scheduler assignment" : "Artifact prerequisites + external offload contract",
    state: observed ? "RUNTIME OBSERVED" : manifest ? "CONFIGURATION BOUND" : "EXECUTION GRAPH EXTERNAL",
    tone: observed ? "complete" : manifest ? "partial" : "missing",
    banner: manifest && !observed ? "The imported manifest binds the selected llama.cpp build, backend, device, context, batch, and GPU-layer request, but no generated scheduler graph was captured. Configuration is bound; execution placement is not observed." : null,
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", `${finiteInteger(analysis.tensor_count)} tensors and GGML storage contracts are decoded. GGUF does not serialize an execution DAG.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", backend.status === "source_candidate" ? "CANDIDATE" : backend.status === "invalid" ? "EXCLUDED" : "UNRESOLVED", backend.status === "source_candidate" ? "partial" : "missing", backend.status === "source_candidate" ? "SOURCE_PINNED_PREREQUISITE" : "NOT_ASSESSED", "Pinned llama.cpp architecture, storage, backend build-option, and registration prerequisites are assessed without inventing layer offload."),
      level("configuration_bound", "Configuration-bound", manifest ? "BOUND" : "UNBOUND", manifest ? "partial" : "missing", manifest ? "CONFIGURATION_BOUND" : "NOT_OBSERVED", manifest ? `${manifest.selection.requested_backend_label}; n_gpu_layers=${manifest.selection.gpu_layers}; context=${manifest.selection.context_size}; batch=${manifest.selection.batch_size}.` : "Engine build, backend inventory, n_gpu_layers, split policy, context, batch, and device remain external."),
      level("runtime_evidence", "Runtime evidence", observed ? "OBSERVED" : "NOT OBSERVED", observed ? "complete" : "missing", observed ? "OBSERVED_RUNTIME" : "NOT_OBSERVED", observed ? `${graphCount} generated graph(s), ${computeGraph.split_count} scheduler split(s), and ${computeGraph.successful_dispatch_count}/${computeGraph.dispatch_count} successful dispatch(es) are captured.` : "Per-node backend assignment and scheduler transitions require the pinned instrumented llama.cpp trace."),
    ],
    flow: flow(observed ? "Observed scheduled-node backend flow - first captured graph" : "No serialized placement flow",
      observed ? "OBSERVED_GGML_SCHEDULER_GRAPH" : "NOT_SERIALIZED", scheduled.length, scheduled.length, segments),
    portfolios: [],
    note: observed ? computeGraph.interpretation_boundary : "Architecture and storage compatibility do not determine generated GGML nodes, layer offload, tensor residency, or backend transitions.",
    runtimeObservation: { status: observed ? "observed" : "not_observed", covered_item_count: scheduled.length, total_item_count: scheduled.length, rejected_row_count: 0 },
  });
}

function safeTensorsEvidence(analysis) {
  return baseEvidence(analysis, {
    format: "safetensors", itemKind: "serialized_tensor", itemCount: finiteInteger(analysis.tensor_count),
    title: "SafeTensors Execution Placement", subtitle: "Checkpoint-container boundary",
    state: "NOT ASSESSABLE FROM CONTAINER", tone: "na",
    banner: "SafeTensors contains named tensor payloads, dtypes, shapes, and byte ranges. It does not contain an executable graph, runtime provider order, device policy, or placement trace, so a delegation map would be fabricated evidence.",
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", `${finiteInteger(analysis.tensor_count)} tensor descriptors and their payload ranges are assessed.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", "NOT APPLICABLE", "na", "NOT_APPLICABLE", "A bound config can derive a canonical architecture scenario, but that scenario is not an executed framework graph or backend-eligibility proof."),
      level("configuration_bound", "Configuration-bound", "EXTERNAL", "missing", "NOT_OBSERVED", "Bind the framework graph, preprocessing, runtime build, provider order, device policy, and cache/offload settings that consume this checkpoint."),
      level("runtime_evidence", "Runtime evidence", "EXTERNAL", "missing", "NOT_OBSERVED", "Import support requires a future framework-specific execution-trace schema; no generic placement is inferred today."),
    ],
    flow: flow("Placement unavailable by format semantics", "NOT_APPLICABLE", 0, 0, []),
    portfolios: [], note: "This explicit non-assessment is a format fact, not a missing parser feature.",
    runtimeObservation: { status: "not_applicable", covered_item_count: 0, total_item_count: 0, rejected_row_count: 0 },
  });
}

function execuTorchEvidence(analysis) {
  const ops = rows(analysis.ops);
  const pte = analysis.executorch_container === "pte";
  const segments = pte && ops.length ? contiguous(ops, (op) => {
    const delegated = op.instruction_kind === "DelegateCall";
    const label = delegated ? String(op.name || "serialized delegate") : "portable instruction";
    return [delegated ? label : "portable", label, delegated ? "anticipated" : "artifact"];
  }, (op, position) => finiteInteger(op.index ?? position)) : [];
  const delegateCount = ops.filter((op) => op.instruction_kind === "DelegateCall").length;
  return baseEvidence(analysis, {
    format: "executorch", itemKind: pte ? "serialized_instruction" : "named_tensor_data", itemCount: ops.length,
    title: "ExecuTorch Execution Placement", subtitle: pte ? "Serialized AOT instruction and delegate flow" : "FT01 data-container boundary",
    state: pte ? delegateCount ? "SERIALIZED DELEGATE CALLS" : "PORTABLE INSTRUCTIONS SERIALIZED" : "NOT APPLICABLE TO PTD",
    tone: pte ? "partial" : "na",
    banner: pte ? "ET12 delegate calls and backend IDs are artifact-observed AOT lowering evidence. They do not prove that the selected runtime build can load or execute the backend." : "FT01 stores named tensor/blob segments and no execution plan.",
    levels: [
      level("artifact_observed", "Artifact observed", "OBSERVED", "complete", "ARTIFACT_OBSERVED", pte ? `${ops.length} ordered ET12 instruction(s), including ${delegateCount} serialized delegate call(s), are decoded.` : `${analysis.tensor_count || 0} FT01 named tensor/blob descriptor(s) are decoded.`),
      level("source_pinned_eligibility", "Source-pinned eligibility", "SCHEMA PINNED", "complete", "SOURCE_PINNED_SCHEMA", "The ET12/FT01 wire contract is pinned to the declared ExecuTorch source release; backend implementation eligibility is not inferred from the backend ID alone."),
      level("configuration_bound", "Configuration-bound", "UNBOUND", "missing", "NOT_OBSERVED", pte ? "The serialized compile specs are observed, but the matching runtime binary, registered backend implementation, device, and allocator configuration are not bound." : "FT01 does not select a consuming PTE or runtime configuration."),
      level("runtime_evidence", "Runtime evidence", "NOT OBSERVED", "missing", "NOT_OBSERVED", "Delegate loading, executed instruction placement, runtime allocation, kernels, physical transfers, and latency require native ExecuTorch evidence."),
    ],
    flow: flow(pte ? "Serialized instruction placement flow" : "Placement unavailable by format semantics", pte ? "ARTIFACT_SERIALIZED_AOT_FLOW" : "NOT_APPLICABLE", pte ? ops.length : 0, pte ? ops.length : 0, segments),
    portfolios: [],
    note: pte ? "This flow reports serialized KernelCall and DelegateCall boundaries. It is not an execution trace and does not expose delegate-internal operations." : "FT01 is external tensor/blob data, not an executable graph.",
    runtimeObservation: { status: pte ? "not_observed" : "not_applicable", covered_item_count: 0, total_item_count: pte ? ops.length : 0, rejected_row_count: 0 },
  });
}

function baseEvidence(analysis, values) {
  return {
    schema: EXECUTION_PLACEMENT_EVIDENCE_SCHEMA,
    method_version: "1.0.0",
    format: values.format,
    artifact_sha256: String(analysis.model_sha256 || ""),
    artifact_item_kind: values.itemKind,
    artifact_item_count: values.itemCount,
    title: values.title,
    subtitle: values.subtitle,
    state: values.state,
    tone: values.tone,
    banner: values.banner || null,
    levels: values.levels,
    flow: values.flow,
    portfolios: values.portfolios,
    static_profiles: values.staticProfiles || [],
    configuration_preflights: values.configurationPreflights || [],
    interpretation_boundary: values.note,
    runtime_observation: values.runtimeObservation,
  };
}

function canonicalAssignments(runtime, ops) {
  const raw = rows(runtime?.assignments || runtime?.runtime_assignment?.assignments);
  const opIndices = new Set(ops.map((op) => finiteInteger(op.index)));
  const byOp = new Map();
  for (const row of raw) {
    const opIndex = Number(row?.op_index);
    const provider = String(row?.provider || row?.observed_provider || "").trim();
    if (!Number.isSafeInteger(opIndex) || !provider) throw new Error("Runtime assignment row lacks a valid original-op index or provider.");
    if (!opIndices.has(opIndex)) throw new Error(`Runtime assignment references out-of-scope op #${opIndex}.`);
    if (byOp.has(opIndex)) throw new Error(`Runtime assignment duplicates original op #${opIndex}.`);
    byOp.set(opIndex, { ...row, op_index: opIndex, provider });
  }
  return { rows: [...byOp.values()], byOp, rawCount: raw.length };
}

function assignmentObservation(assignments, total) {
  return {
    status: assignments.rows.length ? "observed" : "not_observed",
    covered_item_count: assignments.rows.length,
    total_item_count: total,
    rejected_row_count: assignments.rawCount - assignments.rows.length,
  };
}

function assignmentSegments(ops, byOp) {
  return contiguous(ops, (op) => {
    const assignment = byOp.get(finiteInteger(op.index));
    const provider = assignment?.provider || "unresolved";
    return [provider, provider, provider === "unresolved" ? "missing" : "observed"];
  }, opItemId);
}

function contiguous(values, descriptor, itemId = (_value, position) => position) {
  const segments = [];
  values.forEach((value, position) => {
    const [key, label, tone] = descriptor(value, position);
    const id = itemId(value, position);
    const previous = segments.at(-1);
    if (previous?.key === key) {
      previous.item_count += 1;
      previous.end_position = position;
      previous.end_item_id = id;
    } else {
      segments.push({ key, label, tone, start_position: position, end_position: position, start_item_id: id, end_item_id: id, item_count: 1 });
    }
  });
  return segments;
}

function flow(label, basis, scopeItemCount, coveredItemCount, segments) {
  return {
    label,
    evidence_basis: basis,
    scope_item_count: scopeItemCount,
    covered_item_count: coveredItemCount,
    rendered_item_count: segments.reduce((sum, segment) => sum + segment.item_count, 0),
    segments,
  };
}

function portfolio(id, label, candidateCount, totalCount, evidenceClass, detail) {
  return { id: String(id || label || "unknown"), label: String(label || id || "unknown"), candidate_count: candidateCount, total_count: totalCount, evidence_class: evidenceClass, detail, tone: "partial" };
}

function level(id, label, state, tone, evidenceClass, detail) {
  return { id, label, state, tone, evidence_class: evidenceClass, detail };
}

function tensorRtPreflightSummary(value) {
  return {
    schema: value.schema,
    label: "TensorRT Preflight",
    status: value.status,
    evidence_class: value.evidence_class,
    build_profile_sha256: value.build_profile?.profile_sha256 || null,
    execution_path: value.build_profile?.execution_path || null,
    blocking_issue_count: finiteInteger(value.blocking_issue_count),
    unresolved_issue_count: finiteInteger(value.unresolved_issue_count),
    profile_cost_status: value.optimization_profile_cost?.status || "not emitted",
    profile_cost_scenario_count: finiteInteger(value.optimization_profile_cost?.scenario_count),
    optimization_profile_cost: value.optimization_profile_cost || null,
    issues: rows(value.issues),
    trust_boundary: value.trust_boundary,
    interpretation_boundary: value.interpretation_boundary,
  };
}

function buildRiskBanner(risks, baseline, opCount) {
  const conditions = [...new Set(risks.map((risk) => risk.required_build_configuration).filter(Boolean))];
  const conditionText = conditions.length ? conditions.join("; ") : "unspecified selected-runtime build requirement";
  return `Conditional assignment: ${baseline}/${opCount} XNNPACK candidates depend on unbound selected-runtime build configuration (${conditionText}). If the requirement is absent, the candidate partition changes; no executed assignment is claimed.`;
}

function fallbackTone(op) {
  const kind = String(op.xnnpack_break_class || "");
  if (kind === "structural") return "structural";
  if (kind === "memory-traffic") return "traffic";
  if (kind === "high-adjacent-mac-exposure") return "compute";
  return "fallback";
}

function opItemId(op, position) { return finiteInteger(op?.index ?? position); }
function assertRuntimeArtifactBinding(analysis, runtime, label) {
  const expected = String(analysis?.model_sha256 || "").toLowerCase();
  const actual = String(runtime?.artifact_sha256 || runtime?.artifact?.sha256 || "").toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(expected) || actual !== expected) {
    throw new Error(`${label} is not bound to the active artifact SHA-256.`);
  }
  const expectedTarget = String(analysis?.target_profile?.profile_sha256 || "").toLowerCase();
  if (expectedTarget) {
    const actualTarget = String(runtime?.target_profile_sha256 || runtime?.target_profile?.profile_sha256 || "").toLowerCase();
    if (actualTarget !== expectedTarget) throw new Error(`${label} is not bound to the active target profile SHA-256.`);
  }
}
function countBy(values, selector) {
  const result = {};
  for (const value of values) {
    const key = String(selector(value));
    result[key] = (result[key] || 0) + 1;
  }
  return result;
}
function rows(value) { return Array.isArray(value) ? value : []; }
function finiteInteger(value) { const number = Number(value); return Number.isSafeInteger(number) && number >= 0 ? number : 0; }
function signedInteger(value, fallback = 0) { const number = Number(value); return Number.isSafeInteger(number) ? number : fallback; }
