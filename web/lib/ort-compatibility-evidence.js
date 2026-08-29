import { effectiveOnnxOpsetMap } from "./onnx-opset-imports.js";

export const ORT_COMPATIBILITY_EVIDENCE_SCHEMA = "deepbom.ort_source_compatibility.v1.15";
export const ORT_COMPATIBILITY_METHOD_VERSION = "2026-08-29.1";
export const ORT_EP_PORTABILITY_FRONTIER_SCHEMA = "deepbom.ort_ep_portability_frontier.v2";
const ORT_ARTIFACT_CONTRACT_SCHEMA = "deepbom.ort_source_artifact_contract.v1";

const SHA256 = /^[a-f0-9]{64}$/;
const SOURCE_COMMIT = /^[a-f0-9]{40}$/;
const VERSION = /^\d+\.\d+(?:\.\d+)?$/;
const EP_STATUSES = new Set([
  "SOURCE_SCHEMA_KERNEL_VERSION_MATCH_NO_ARTIFACT_CONDITION",
  "SOURCE_SCHEMA_KERNEL_VERSION_AND_ARTIFACT_PRECHECK_PASS",
  "SOURCE_SCHEMA_KERNEL_VERSION_MATCH_ARTIFACT_PRECHECK_UNRESOLVED",
  "SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL",
  "SOURCE_KERNEL_VERSION_NO_MATCH",
  "SOURCE_RULE_NOT_FOUND",
  "MODEL_LOCAL_FUNCTION_REQUIRES_RUNTIME_RESOLUTION",
  "EXTERNAL_CUSTOM_REGISTRY_REQUIRED",
  "OP_SCHEMA_VERSION_NOT_RESOLVED",
  "IMPORTED_DOMAIN_OPSET_MISSING",
]);
const VERSION_MATCH_STATUSES = new Set([
  "SOURCE_SCHEMA_KERNEL_VERSION_MATCH_NO_ARTIFACT_CONDITION",
  "SOURCE_SCHEMA_KERNEL_VERSION_AND_ARTIFACT_PRECHECK_PASS",
  "SOURCE_SCHEMA_KERNEL_VERSION_MATCH_ARTIFACT_PRECHECK_UNRESOLVED",
  "SOURCE_ARTIFACT_CONDITION_DEFINITE_FAIL",
]);
const ARTIFACT_PRECHECK_STATUSES = new Set([
  "NOT_APPLICABLE_SOURCE_VERSION_GAP",
  "NO_ARTIFACT_VISIBLE_CONDITION_DECLARED",
  "ARTIFACT_PRECHECK_PASS",
  "ARTIFACT_PRECHECK_DEFINITE_FAIL",
  "ARTIFACT_PRECHECK_UNRESOLVED",
]);

export function applyProtectedOrtCompatibilityEvidence(analysis, evidence) {
  validateProtectedOrtCompatibilityEvidence(analysis, evidence);
  analysis.ort_compatibility_assessment_status = evidence.assessment_status;
  analysis.ort_compatibility_evidence_schema = evidence.schema;
  analysis.ort_compatibility_evidence_access = evidence.access_scope;
  analysis.ort_compatibility_evidence = structuredCloneSafe(evidence);
  analysis.ort_assignment_capture_capability = structuredCloneSafe(evidence.assignment_capture_capability || null);
  analysis.ort_ep_portability_frontier = structuredCloneSafe(evidence.portability_frontier || null);
  if (evidence.assessment_status !== "complete") return analysis;

  const floor = evidence.runtime_floor;
  const completeFloor = isCompleteRuntimeFloor(floor.status);
  analysis.runtime_compat = {
    ...(analysis.runtime_compat || {}),
    min_runtime_version: "",
    derived_min_runtime_version: floor.minimum_ort_version || "",
    effective_min_runtime_version: completeFloor ? floor.minimum_ort_version || "" : "",
    runtime_floor_status: floor.status,
    runtime_floor_evidence_class: floor.evidence_class,
    unresolved_runtime_floor_domains: structuredCloneSafe(floor.unresolved_domains || []),
    model_local_function_domains: structuredCloneSafe(floor.model_local_function_domains || []),
    source_backed_runtime_floor_domains: structuredCloneSafe(floor.source_backed_external_domains || []),
    contrib_operator_runtime_floors: structuredCloneSafe(floor.contrib_operator_floors || []),
    runtime_version_basis: floor.basis,
    detail: completeFloor
      ? `Necessary ONNX/ONNX-ML${floor.contrib_operator_floors?.length ? "/com.microsoft" : ""} parser/schema floor ORT ${floor.minimum_ort_version}; model-local functions and EP kernel/build/device sufficiency remain separate.`
      : floor.minimum_ort_version
        ? `ONNX parser lower bound ORT ${floor.minimum_ort_version}; external runtime registries prevent a complete runtime floor.`
        : "Pinned release coverage does not determine a runtime floor for this artifact.",
  };
  analysis.onnx_sections_suppressed = (analysis.onnx_sections_suppressed || [])
    .filter((item) => item !== "ONNX Runtime minimum-version derivation");
  return analysis;
}

export function validateProtectedOrtCompatibilityEvidence(analysis, evidence) {
  if (!analysis || !Array.isArray(analysis.ops)) throw new Error("ORT compatibility evidence requires a current static analysis.");
  if (!evidence || evidence.schema !== ORT_COMPATIBILITY_EVIDENCE_SCHEMA) throw new Error("ORT compatibility evidence schema is unsupported.");
  if (evidence.method_version !== ORT_COMPATIBILITY_METHOD_VERSION) throw new Error("ORT compatibility method version does not match the browser contract.");
  if (evidence.access_scope !== "research" || !SOURCE_COMMIT.test(String(evidence.source_commit || ""))) throw new Error("ORT compatibility provenance is invalid.");

  const onnx = String(analysis.format || "").toLowerCase() === "onnx";
  if (!onnx) {
    if (evidence.assessment_status !== "not_applicable" || (evidence.execution_providers || []).length
      || evidence.assignment_capture_capability != null || evidence.portability_frontier != null) {
      throw new Error("Non-ONNX evidence must be not applicable and empty.");
    }
    return true;
  }
  if (evidence.assessment_status !== "complete" || !String(evidence.evidence_boundary || "").includes("necessary ONNX IR/opset parser floor")) {
    throw new Error("ONNX compatibility evidence assessment is incomplete or unbounded.");
  }

  validateRuntimeFloor(analysis, evidence.runtime_floor);
  validateSourceConditionInventory(evidence.source_condition_inventory);
  validateAssignmentCaptureCapability(evidence.assignment_capture_capability);
  const providers = evidence.execution_providers;
  if (!Array.isArray(providers) || providers.length === 0) throw new Error("ORT compatibility EP coverage is missing.");
  const providerIds = providers.map((provider) => String(provider.execution_provider || ""));
  if (providerIds.some((id) => !id) || new Set(providerIds).size !== providerIds.length) throw new Error("ORT compatibility EP identities are invalid.");
  for (const provider of providers) validateEpAssessment(analysis, provider);
  validatePortabilityFrontier(analysis, providers, evidence.portability_frontier);
  return true;
}

function validateSourceConditionInventory(inventory) {
  const expected = new Map([
    ["wasm_cpu", [292, 611, 0, 0, 0]],
    ["webgpu", [118, 0, 10, 12, 9]],
    ["webnn", [112, 0, 69, 4, 0]],
    ["qnn", [102, 0, 108, 102, 102]],
    ["directml", [194, 0, 194, 194, 194]],
    ["coreml", [57, 0, 57, 57, 57]],
    ["nnapi", [54, 0, 54, 54, 54]],
    ["xnnpack", [9, 0, 48, 9, 9]],
  ]);
  if (!inventory || inventory.schema !== ORT_ARTIFACT_CONTRACT_SCHEMA
    || inventory.status !== "complete_for_pinned_source_extractors"
    || Number(inventory.source_rule_count) !== 938
    || Number(inventory.cpu_registration_variant_count) !== 611
    || Number(inventory.cpu_registration_variant_with_signature_count) !== 611
    || Number(inventory.cpu_registration_variant_with_type_constraint_count) !== 611
    || Number(inventory.machine_condition_count) !== 540
    || Number(inventory.versioned_scalar_schema_default_binding_count) !== 80
    || Number(inventory.unresolved_source_fragment_count) !== 432
    || Number(inventory.informational_source_note_count) !== 425
    || !Array.isArray(inventory.execution_providers) || inventory.execution_providers.length !== expected.size
    || !String(inventory.method || "").trim() || !String(inventory.evidence_boundary || "").includes("does not establish")) {
    throw new Error("ORT source artifact-condition inventory is invalid.");
  }
  for (const [ep, counts] of expected) {
    const row = inventory.execution_providers.find((item) => item.execution_provider === ep);
    if (!row || [row.source_rule_count, row.registration_variant_count, row.machine_condition_count, row.unresolved_source_fragment_count, row.informational_source_note_count]
      .some((value, index) => Number(value) !== counts[index])) throw new Error(`ORT ${ep} source artifact-condition inventory is invalid.`);
  }
}

function validateRuntimeFloor(analysis, floor) {
  const opsets = importedOpsets(analysis);
  const localDomains = modelLocalFunctionDomains(analysis);
  const contribNames = contribOperatorNames(analysis);
  const contribRows = Array.isArray(floor?.contrib_operator_floors) ? floor.contrib_operator_floors : [];
  const contribImportedOpset = opsets.get("com.microsoft") ?? null;
  const rowNames = contribRows.map((row) => String(row?.op_name || ""));
  if (new Set(rowNames).size !== rowNames.length || JSON.stringify(rowNames) !== JSON.stringify([...rowNames].sort())) {
    throw new Error("ORT contrib runtime-floor operator identities are invalid or noncanonical.");
  }
  for (const row of contribRows) {
    if (row.domain !== "com.microsoft" || Number(row.imported_opset) !== 1
      || !rowNames.includes(row.op_name) || !contribNames.includes(row.op_name)
      || !VERSION.test(String(row.minimum_ort_version || ""))
      || !validSourceRef(row.source_ref) || !SHA256.test(String(row.source_sha256 || ""))
      || row.evidence_class !== "DERIVED_EARLIEST_PINNED_SCHEMA_INVENTORY") {
      throw new Error("ORT contrib runtime-floor row is invalid or not artifact-bound.");
    }
  }
  const completeContrib = contribNames.length === 0
    || (contribImportedOpset === 1 && JSON.stringify(rowNames) === JSON.stringify(contribNames));
  const sourceBackedDomains = contribNames.length && completeContrib ? ["com.microsoft:1"] : [];
  const unresolved = [...opsets]
    .filter(([domain]) => !["ai.onnx", "ai.onnx.ml"].includes(domain)
      && !(domain === "com.microsoft" && completeContrib)
      && !localDomains.includes(domain) && domainIsUsed(analysis, domain))
    .map(([domain, version]) => `${domain}:${version}`);
  if (contribNames.length && contribImportedOpset == null) unresolved.push("com.microsoft:missing_import");
  unresolved.sort();
  if (!floor || Number(floor.model_ir_version) !== Number(analysis.onnx_ir_version || 0)
    || floor.standard_domain_opset !== (opsets.get("ai.onnx") || null)
    || floor.standard_ml_domain_opset !== (opsets.get("ai.onnx.ml") || null)
    || JSON.stringify(floor.unresolved_domains || []) !== JSON.stringify(unresolved)
    || JSON.stringify(floor.model_local_function_domains || []) !== JSON.stringify(localDomains)
    || JSON.stringify(floor.source_backed_external_domains || []) !== JSON.stringify(sourceBackedDomains)) {
    throw new Error("ORT runtime floor is not bound to the observed IR/domain-opset identity.");
  }
  for (const value of [floor.minimum_ort_version, floor.standard_minimum_ort_version, floor.contrib_minimum_ort_version]) {
    if (value != null && !VERSION.test(String(value))) throw new Error("ORT runtime floor version is invalid.");
  }
  const expectedContribMinimum = maximumVersion(contribRows.map((row) => row.minimum_ort_version));
  const expectedCombinedMinimum = floor.standard_minimum_ort_version == null
    ? null : maximumVersion([floor.standard_minimum_ort_version, expectedContribMinimum].filter(Boolean));
  if (floor.contrib_minimum_ort_version !== expectedContribMinimum
    || floor.minimum_ort_version !== expectedCombinedMinimum) {
    throw new Error("ORT standard/contrib runtime-floor arithmetic is inconsistent.");
  }
  const assessed = [
    "assessed_onnx_and_model_local_domains",
    "assessed_onnx_model_local_and_source_backed_contrib_domains",
    "partial_external_domains_unresolved",
  ].includes(floor.status);
  if ((assessed && floor.evidence_class !== "DERIVED_NECESSARY_MINIMUM")
    || (!assessed && floor.evidence_class !== "NOT_ASSESSABLE")
    || !String(floor.basis || "").trim()) throw new Error("ORT runtime floor evidence class or basis is invalid.");
  if ((floor.status === "assessed_onnx_and_model_local_domains" && (contribNames.length || unresolved.length))
    || (floor.status === "assessed_onnx_model_local_and_source_backed_contrib_domains" && (!contribNames.length || !completeContrib || unresolved.length))
    || (floor.status === "partial_external_domains_unresolved" && !unresolved.length)) {
    throw new Error("ORT runtime floor status contradicts domain coverage.");
  }
  const documents = validateSourceDocuments(floor.source_documents, "ORT runtime floor");
  if (JSON.stringify(floor.source_refs || []) !== JSON.stringify(documents.map((source) => source.source_ref))) throw new Error("ORT runtime floor source ledger is inconsistent.");
  for (const row of contribRows) {
    if (!documents.some((source) => source.source_ref === row.source_ref && source.sha256 === row.source_sha256
      && source.role === "ort_contrib_first_release_schema_inventory")) {
      throw new Error("ORT contrib runtime-floor row is not bound to its source document.");
    }
  }
}

function isCompleteRuntimeFloor(status) {
  return [
    "assessed_onnx_and_model_local_domains",
    "assessed_onnx_model_local_and_source_backed_contrib_domains",
  ].includes(status);
}

function maximumVersion(values) {
  if (!values.length) return null;
  return [...values].sort((left, right) => {
    const a = String(left).split(".").map(Number);
    const b = String(right).split(".").map(Number);
    for (let index = 0; index < 3; index += 1) {
      const delta = (a[index] || 0) - (b[index] || 0);
      if (delta) return delta;
    }
    return 0;
  }).at(-1);
}

function validateAssignmentCaptureCapability(capability) {
  if (!capability
    || capability.status !== "browser_capture_unavailable_pinned_native_collector_available"
    || capability.evidence_class !== "SOURCE_OBSERVED_BROWSER_LIMIT_AND_NATIVE_PATH"
    || !VERSION.test(String(capability.pinned_ort_web_version || ""))
    || capability.automatic_browser_assignment_capture !== false
    || capability.external_runtime_profile_import !== true
    || capability.pinned_native_capture_available !== true
    || !String(capability.pinned_native_runtime || "").trim()
    || capability.native_capture_schema !== "deepbom.ort_native_capture.v1.5"
    || capability.native_profile_schema !== "deepbom.ort_native_profile.v1.4"
    || !String(capability.native_capture_command || "").trim()
    || !String(capability.native_profile_roles || "").trim()
    || !String(capability.required_observation_path || "").trim()) {
    throw new Error("ORT assignment-capture capability contract is invalid.");
  }
  validateSourceDocuments(capability.source_documents, "ORT assignment capture");
}

function validateEpAssessment(analysis, ep) {
  if (!ep || !String(ep.source_id || "").trim() || !validSourceRef(ep.source_ref) || !SHA256.test(String(ep.source_sha256 || ""))) {
    throw new Error("ORT EP source provenance is invalid.");
  }
  const nativeRegistration = ["qnn", "directml", "coreml", "nnapi", "xnnpack"].includes(ep.execution_provider);
  if (ep.assessment_status !== (nativeRegistration
    ? "source_registration_enumerated_get_capability_predicates_unresolved"
    : "schema_kernel_version_and_artifact_visible_conditions_assessed")
    || ep.support_evidence_class !== (nativeRegistration
      ? "SOURCE_REGISTRATION_CANDIDATE_WITH_UNRESOLVED_GET_CAPABILITY_PREDICATES"
      : "SOURCE_SCHEMA_KERNEL_VERSION_WITH_DEFINITE_ARTIFACT_EXCLUSIONS_ONLY")
    || ep.assignment_evidence_class !== "NOT_OBSERVED") throw new Error(`ORT ${ep.execution_provider} evidence classes are invalid.`);
  if (!String(ep.source_scope || "").trim() || !String(ep.evaluator_coverage || "").trim()
    || ep.execution_provider === "wasm_cpu" && !String(ep.evaluator_coverage).includes("reduced WebAssembly build inclusion remains unresolved")) throw new Error(`ORT ${ep.execution_provider} source scope is invalid.`);
  validateStringSet(ep.unresolved_dimensions, `ORT ${ep.execution_provider} unresolved dimensions`, { nonempty: true });
  if (!Array.isArray(ep.ops) || ep.ops.length !== analysis.ops.length || Number(ep.assessed_op_count) !== analysis.ops.length) throw new Error(`ORT ${ep.execution_provider} op coverage is incomplete.`);

  const opsets = importedOpsets(analysis);
  const seen = new Set();
  const counts = {
    matches: 0,
    noVersion: 0,
    missing: 0,
    modelLocal: 0,
    external: 0,
    schemaResolved: 0,
    schemaUnresolved: 0,
    conditions: 0,
    artifactConditions: 0,
    artifactPass: 0,
    artifactFail: 0,
    artifactUnresolved: 0,
    precheckPassOps: 0,
    precheckFailOps: 0,
    precheckUnresolvedOps: 0,
    precheckNoConditionOps: 0,
    candidates: 0,
  };
  for (const row of ep.ops) {
    const index = Number(row?.op_index);
    const op = analysis.ops.find((candidate) => Number(candidate.index) === index);
    const domain = normalizeDomain(op?.domain);
    if (!op || seen.has(index) || row.op_name !== op.name || row.domain !== domain
      || row.imported_opset !== (opsets.get(domain) || null)
      || row.resolution_class !== topLevelResolutionClass(analysis, index, domain, op.name)
      || row.source_ref !== ep.source_ref || row.source_sha256 !== ep.source_sha256) throw new Error(`ORT ${ep.execution_provider} op identity is invalid at #${index}.`);
    seen.add(index);
    if (!EP_STATUSES.has(row.status) || row.schema_kernel_version_match !== VERSION_MATCH_STATUSES.has(row.status)
      || !ARTIFACT_PRECHECK_STATUSES.has(row.artifact_precheck_status)) throw new Error(`ORT ${ep.execution_provider} status is invalid at #${index}.`);
    const artifactRows = row.artifact_conditions;
    if (!Array.isArray(artifactRows) || new Set(artifactRows.map((item) => item.condition_id)).size !== artifactRows.length
      || artifactRows.some((item) => !item?.condition_id || !item?.condition_kind
        || !["PASS", "DEFINITE_FAIL", "UNRESOLVED"].includes(item.status)
        || !new Set([
          item.status === "UNRESOLVED" ? "NOT_ASSESSABLE" : "DERIVED_FROM_ARTIFACT_AND_PINNED_SOURCE_CONDITION",
          "SOURCE_DOCUMENTED_NOT_MACHINE_ASSESSED",
          "SOURCE_PINNED_SCHEMA_DEFAULT_DERIVED",
        ]).has(item.evidence_class)
        || !String(item.subject || "").trim() || !String(item.expected || "").trim() || !String(item.reason || "").trim()
        || (item.source_ref && (!validSourceRef(item.source_ref) || !SHA256.test(String(item.source_sha256 || ""))))
        || (!item.source_ref && item.source_sha256))) {
      throw new Error(`ORT ${ep.execution_provider} artifact-condition ledger is invalid at #${index}.`);
    }
    const artifactPass = artifactRows.filter((item) => item.status === "PASS").length;
    const artifactFail = artifactRows.filter((item) => item.status === "DEFINITE_FAIL").length;
    const artifactUnresolved = artifactRows.filter((item) => item.status === "UNRESOLVED").length;
    const expectedPrecheck = !row.schema_kernel_version_match
      ? "NOT_APPLICABLE_SOURCE_VERSION_GAP"
      : artifactFail ? "ARTIFACT_PRECHECK_DEFINITE_FAIL"
        : artifactUnresolved ? "ARTIFACT_PRECHECK_UNRESOLVED"
          : artifactRows.length ? "ARTIFACT_PRECHECK_PASS" : "NO_ARTIFACT_VISIBLE_CONDITION_DECLARED";
    if (row.artifact_precheck_status !== expectedPrecheck
      || Number(row.artifact_condition_count) !== artifactRows.length
      || Number(row.artifact_condition_pass_count) !== artifactPass
      || Number(row.artifact_condition_fail_count) !== artifactFail
      || Number(row.artifact_condition_unresolved_count) !== artifactUnresolved
      || row.source_candidate_after_artifact_precheck !== (row.schema_kernel_version_match && artifactFail === 0)
      || row.definite_source_exclusion !== (row.schema_kernel_version_match && artifactFail > 0)
      || (artifactRows.length > 0 && row.artifact_contract_schema !== ORT_ARTIFACT_CONTRACT_SCHEMA)
      || !Array.isArray(row.unresolved_source_condition_fragments)
      || !Array.isArray(row.informational_source_notes)) throw new Error(`ORT ${ep.execution_provider} artifact precheck does not reproduce at #${index}.`);
    if ((row.schema_source_ref && (!validSourceRef(row.schema_source_ref) || !SHA256.test(String(row.schema_source_sha256 || ""))))
      || (!row.schema_source_ref && row.schema_source_sha256)) throw new Error(`ORT ${ep.execution_provider} schema provenance is invalid at #${index}.`);
    if (row.resolved_schema_version != null) {
      if (!Number.isSafeInteger(row.resolved_schema_version) || row.resolved_schema_version <= 0
        || row.imported_opset == null || row.resolved_schema_version > row.imported_opset || !row.schema_source_ref) throw new Error(`ORT ${ep.execution_provider} schema version is invalid at #${index}.`);
      counts.schemaResolved += 1;
    }
    if (row.schema_kernel_version_match) counts.matches += 1;
    else if (row.status === "SOURCE_KERNEL_VERSION_NO_MATCH") counts.noVersion += 1;
    else if (row.status === "SOURCE_RULE_NOT_FOUND") counts.missing += 1;
    else if (row.status === "MODEL_LOCAL_FUNCTION_REQUIRES_RUNTIME_RESOLUTION") counts.modelLocal += 1;
    else if (row.status === "EXTERNAL_CUSTOM_REGISTRY_REQUIRED") counts.external += 1;
    else if (row.status === "OP_SCHEMA_VERSION_NOT_RESOLVED") counts.schemaUnresolved += 1;
    if (row.schema_kernel_version_match && row.documented_condition) counts.conditions += 1;
    counts.artifactConditions += artifactRows.length;
    counts.artifactPass += artifactPass;
    counts.artifactFail += artifactFail;
    counts.artifactUnresolved += artifactUnresolved;
    if (row.artifact_precheck_status === "ARTIFACT_PRECHECK_PASS") counts.precheckPassOps += 1;
    if (row.artifact_precheck_status === "ARTIFACT_PRECHECK_DEFINITE_FAIL") counts.precheckFailOps += 1;
    if (row.artifact_precheck_status === "ARTIFACT_PRECHECK_UNRESOLVED") counts.precheckUnresolvedOps += 1;
    if (row.artifact_precheck_status === "NO_ARTIFACT_VISIBLE_CONDITION_DECLARED") counts.precheckNoConditionOps += 1;
    if (row.source_candidate_after_artifact_precheck) counts.candidates += 1;
  }
  const ratio = analysis.ops.length ? counts.matches / analysis.ops.length : 0;
  const candidateRatio = analysis.ops.length ? counts.candidates / analysis.ops.length : 0;
  if (Number(ep.schema_kernel_version_match_count) !== counts.matches
    || Number(ep.schema_kernel_version_no_match_count) !== counts.noVersion
    || Number(ep.source_rule_missing_count) !== counts.missing
    || Number(ep.model_local_function_count) !== counts.modelLocal
    || Number(ep.external_custom_registry_count) !== counts.external
    || Number(ep.schema_version_resolved_count) !== counts.schemaResolved
    || Number(ep.schema_version_unresolved_count) !== counts.schemaUnresolved
    || Number(ep.documented_condition_count) !== counts.conditions
    || Number(ep.artifact_condition_count) !== counts.artifactConditions
    || Number(ep.artifact_condition_pass_count) !== counts.artifactPass
    || Number(ep.artifact_condition_fail_count) !== counts.artifactFail
    || Number(ep.artifact_condition_unresolved_count) !== counts.artifactUnresolved
    || Number(ep.artifact_precheck_pass_op_count) !== counts.precheckPassOps
    || Number(ep.artifact_precheck_definite_fail_op_count) !== counts.precheckFailOps
    || Number(ep.artifact_precheck_unresolved_op_count) !== counts.precheckUnresolvedOps
    || Number(ep.artifact_precheck_no_condition_op_count) !== counts.precheckNoConditionOps
    || Number(ep.source_candidate_after_artifact_precheck_count) !== counts.candidates
    || !closeNumber(Number(ep.schema_kernel_version_match_ratio), ratio)
    || !closeNumber(Number(ep.source_candidate_after_artifact_precheck_ratio), candidateRatio)) throw new Error(`ORT ${ep.execution_provider} summary does not reproduce its op rows.`);
}

function validatePortabilityFrontier(analysis, providers, frontier) {
  const ops = analysis.ops || [];
  if (!frontier || frontier.schema !== ORT_EP_PORTABILITY_FRONTIER_SCHEMA
    || frontier.evidence_class !== "DERIVED_FROM_PINNED_SOURCE_AND_ARTIFACT_VISIBLE_DEFINITE_EXCLUSIONS"
    || frontier.op_count !== ops.length || frontier.execution_provider_count !== providers.length
    || !Array.isArray(frontier.ops) || frontier.ops.length !== ops.length
    || !Array.isArray(frontier.providers) || frontier.providers.length !== providers.length) throw new Error("ORT EP portability frontier identity is invalid.");

  const assessedMacTotal = ops.reduce((sum, op) => sum + (assessedMacs(op) ?? 0), 0);
  let allMatchCount = 0;
  let allMatchMacs = 0;
  let allCandidateCount = 0;
  let allCandidateMacs = 0;
  for (const [position, op] of ops.entries()) {
    const opIndex = Number.isSafeInteger(Number(op.index)) ? Number(op.index) : position;
    const expectedMatches = providers.filter((provider) => provider.ops.some((row) => Number(row.op_index) === opIndex && row.schema_kernel_version_match === true)).map((provider) => provider.execution_provider);
    const expectedGaps = providers.filter((provider) => !expectedMatches.includes(provider.execution_provider)).map((provider) => provider.execution_provider);
    const expectedCandidates = providers.filter((provider) => provider.ops.some((row) => Number(row.op_index) === opIndex && row.source_candidate_after_artifact_precheck === true)).map((provider) => provider.execution_provider);
    const expectedDefiniteExclusions = providers.filter((provider) => provider.ops.some((row) => Number(row.op_index) === opIndex && row.definite_source_exclusion === true)).map((provider) => provider.execution_provider);
    const expectedUnresolved = providers.filter((provider) => provider.ops.some((row) => Number(row.op_index) === opIndex && row.artifact_precheck_status === "ARTIFACT_PRECHECK_UNRESOLVED")).map((provider) => provider.execution_provider);
    const row = frontier.ops[position];
    const macs = assessedMacs(op);
    if (!row || row.op_index !== opIndex || row.op_name !== op.name || row.macs_status !== String(op.macs_status || "")
      || row.assessed_macs !== macs || row.source_match_ep_count !== expectedMatches.length
      || JSON.stringify(row.source_match_eps || []) !== JSON.stringify(expectedMatches)
      || JSON.stringify(row.source_gap_eps || []) !== JSON.stringify(expectedGaps)
      || row.artifact_precheck_candidate_ep_count !== expectedCandidates.length
      || JSON.stringify(row.artifact_precheck_candidate_eps || []) !== JSON.stringify(expectedCandidates)
      || JSON.stringify(row.artifact_precheck_definite_exclusion_eps || []) !== JSON.stringify(expectedDefiniteExclusions)
      || JSON.stringify(row.artifact_precheck_unresolved_eps || []) !== JSON.stringify(expectedUnresolved)) throw new Error(`ORT EP portability row is invalid at #${opIndex}.`);
    if (expectedMatches.length === providers.length) {
      allMatchCount += 1;
      allMatchMacs += macs ?? 0;
    }
    if (expectedCandidates.length === providers.length) {
      allCandidateCount += 1;
      allCandidateMacs += macs ?? 0;
    }
  }
  assertNear(frontier.assessed_mac_total, assessedMacTotal, "ORT EP portability assessed MAC total");
  assertNear(frontier.all_ep_source_match_macs, allMatchMacs, "ORT EP portability common MAC total");
  if (frontier.all_ep_source_match_op_count !== allMatchCount) throw new Error("ORT EP portability common op count is invalid.");
  assertNear(frontier.all_ep_source_match_op_ratio, ops.length ? allMatchCount / ops.length : 0, "ORT EP portability common op ratio");
  assertOptionalNear(frontier.all_ep_source_match_mac_ratio, assessedMacTotal > 0 ? allMatchMacs / assessedMacTotal : null, "ORT EP portability common MAC ratio");
  if (frontier.all_ep_artifact_precheck_candidate_op_count !== allCandidateCount) throw new Error("ORT EP portability narrowed candidate op count is invalid.");
  assertNear(frontier.all_ep_artifact_precheck_candidate_op_ratio, ops.length ? allCandidateCount / ops.length : 0, "ORT EP portability narrowed candidate op ratio");
  assertNear(frontier.all_ep_artifact_precheck_candidate_macs, allCandidateMacs, "ORT EP portability narrowed candidate MAC total");
  assertOptionalNear(frontier.all_ep_artifact_precheck_candidate_mac_ratio, assessedMacTotal > 0 ? allCandidateMacs / assessedMacTotal : null, "ORT EP portability narrowed candidate MAC ratio");

  for (const provider of providers) {
    const summary = frontier.providers.find((item) => item.execution_provider === provider.execution_provider);
    const matches = provider.ops.filter((row) => row.schema_kernel_version_match === true);
    const matchedMacs = matches.reduce((sum, row) => sum + (assessedMacs(ops.find((op) => Number(op.index) === Number(row.op_index))) ?? 0), 0);
    const candidates = provider.ops.filter((row) => row.source_candidate_after_artifact_precheck === true);
    const candidateMacs = candidates.reduce((sum, row) => sum + (assessedMacs(ops.find((op) => Number(op.index) === Number(row.op_index))) ?? 0), 0);
    if (!summary || summary.source_match_op_count !== matches.length) throw new Error(`ORT ${provider.execution_provider} portability summary is invalid.`);
    assertNear(summary.source_match_op_ratio, ops.length ? matches.length / ops.length : 0, `ORT ${provider.execution_provider} portability op ratio`);
    assertNear(summary.source_match_assessed_macs, matchedMacs, `ORT ${provider.execution_provider} portability MACs`);
    assertOptionalNear(summary.source_match_assessed_mac_ratio, assessedMacTotal > 0 ? matchedMacs / assessedMacTotal : null, `ORT ${provider.execution_provider} portability MAC ratio`);
    if (summary.artifact_precheck_candidate_op_count !== candidates.length
      || summary.artifact_precheck_definite_fail_op_count !== provider.ops.filter((row) => row.definite_source_exclusion).length) throw new Error(`ORT ${provider.execution_provider} narrowed candidate summary is invalid.`);
    assertNear(summary.artifact_precheck_candidate_op_ratio, ops.length ? candidates.length / ops.length : 0, `ORT ${provider.execution_provider} narrowed candidate op ratio`);
    assertNear(summary.artifact_precheck_candidate_assessed_macs, candidateMacs, `ORT ${provider.execution_provider} narrowed candidate MACs`);
    assertOptionalNear(summary.artifact_precheck_candidate_assessed_mac_ratio, assessedMacTotal > 0 ? candidateMacs / assessedMacTotal : null, `ORT ${provider.execution_provider} narrowed candidate MAC ratio`);
  }

  const expectedPairCount = providers.length * (providers.length - 1) / 2;
  if (!Array.isArray(frontier.provider_pairs) || frontier.provider_pairs.length !== expectedPairCount) throw new Error("ORT EP portability pair coverage is incomplete.");
  for (let left = 0; left < providers.length; left += 1) {
    for (let right = left + 1; right < providers.length; right += 1) {
      const leftSet = new Set(providers[left].ops.filter((row) => row.schema_kernel_version_match).map((row) => Number(row.op_index)));
      const rightSet = new Set(providers[right].ops.filter((row) => row.schema_kernel_version_match).map((row) => Number(row.op_index)));
      const leftCandidates = new Set(providers[left].ops.filter((row) => row.source_candidate_after_artifact_precheck).map((row) => Number(row.op_index)));
      const rightCandidates = new Set(providers[right].ops.filter((row) => row.source_candidate_after_artifact_precheck).map((row) => Number(row.op_index)));
      const union = new Set([...leftSet, ...rightSet]);
      const expected = union.size ? [...leftSet].filter((index) => rightSet.has(index)).length / union.size : 1;
      const pair = frontier.provider_pairs.find((item) => item.left_execution_provider === providers[left].execution_provider
        && item.right_execution_provider === providers[right].execution_provider);
      if (!pair) throw new Error("ORT EP portability pair identity is invalid.");
      assertNear(pair.source_match_set_jaccard, expected, "ORT EP portability pair Jaccard");
      const candidateUnion = new Set([...leftCandidates, ...rightCandidates]);
      const expectedCandidate = candidateUnion.size ? [...leftCandidates].filter((index) => rightCandidates.has(index)).length / candidateUnion.size : 1;
      assertNear(pair.artifact_precheck_candidate_set_jaccard, expectedCandidate, "ORT EP narrowed candidate pair Jaccard");
    }
  }
  if (!String(frontier.evidence_boundary || "").includes("not support or assignment")) throw new Error("ORT EP portability frontier boundary is missing.");
}

function validateSourceDocuments(value, label) {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`${label} source documents are missing.`);
  const seen = new Set();
  for (const source of value) {
    if (!source || !String(source.role || "").trim() || !validSourceRef(source.source_ref)
      || !SHA256.test(String(source.sha256 || "")) || seen.has(source.source_ref)) throw new Error(`${label} source document is invalid.`);
    seen.add(source.source_ref);
  }
  return value;
}

function validateStringSet(value, label, { nonempty = false } = {}) {
  if (!Array.isArray(value) || (nonempty && value.length === 0) || value.some((item) => !String(item || "").trim()) || new Set(value).size !== value.length) throw new Error(`${label} are invalid.`);
}

function validSourceRef(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" && Boolean(url.hostname);
  } catch {
    return false;
  }
}

function importedOpsets(analysis) {
  return effectiveOnnxOpsetMap(analysis.opsets || []);
}

function domainInventory(analysis) {
  return Array.isArray(analysis?.onnx_domain_analysis?.domains) ? analysis.onnx_domain_analysis.domains : [];
}

function modelLocalFunctionDomains(analysis) {
  return domainInventory(analysis)
    .filter((row) => (row.resolution_classes || []).includes("model_local_function") && !(row.resolution_classes || []).includes("external_custom_registry"))
    .map((row) => normalizeDomain(row.domain))
    .sort();
}

function domainIsUsed(analysis, domain) {
  const inventory = domainInventory(analysis);
  if (!inventory.length) return true;
  const row = inventory.find((item) => normalizeDomain(item.domain) === domain);
  return Number(row?.node_count || 0) > 0;
}

function contribOperatorNames(analysis) {
  const inventory = analysis?.onnx_domain_analysis?.nodes;
  const names = Array.isArray(inventory)
    ? inventory
      .filter((row) => normalizeDomain(row.domain) === "com.microsoft" && row.resolution_class === "ort_contrib_schema")
      .map((row) => String(row.op_name || ""))
    : (analysis?.ops || [])
      .filter((row) => normalizeDomain(row.domain) === "com.microsoft")
      .map((row) => String(row.name || ""));
  return [...new Set(names.filter(Boolean))].sort();
}

function topLevelResolutionClass(analysis, opIndex, domain, opName) {
  const row = (analysis?.onnx_domain_analysis?.nodes || []).find((item) => item.top_level_op_index != null && Number(item.top_level_op_index) === opIndex
    && normalizeDomain(item.domain) === domain && item.op_name === opName);
  if (row?.resolution_class) return row.resolution_class;
  if (domain === "ai.onnx") return "onnx_standard";
  if (domain === "ai.onnx.ml") return "onnx_standard_ml";
  if (domain === "com.microsoft") return "ort_contrib_schema";
  return "external_custom_registry";
}

function normalizeDomain(domain) {
  const value = String(domain || "").trim();
  return value && value !== "ai.onnx" ? value : "ai.onnx";
}

function assessedMacs(op) {
  if (!op || String(op.macs_status || "") !== "assessed") return null;
  const value = Number(op.macs);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function assertOptionalNear(actual, expected, label) {
  if (expected == null) {
    if (actual != null) throw new Error(`${label} must be null.`);
    return;
  }
  assertNear(actual, expected, label);
}

function assertNear(actual, expected, label) {
  const tolerance = Math.max(1e-12, Math.abs(Number(expected)) * 1e-12);
  if (!Number.isFinite(Number(actual)) || Math.abs(Number(actual) - Number(expected)) > tolerance) throw new Error(`${label} is invalid.`);
}

function closeNumber(left, right, tolerance = 1e-12) {
  return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= tolerance;
}

function structuredCloneSafe(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}
