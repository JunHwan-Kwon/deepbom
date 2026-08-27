export const ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA = "deepbom.on_device_llm_runtime_manifest.v2";
export const ON_DEVICE_LLM_RUNTIME_MANIFEST_LEGACY_SCHEMA = "deepbom.on_device_llm_runtime_manifest.v1";

const SECTION_EVIDENCE = new Set(["DECLARED", "OBSERVED_RUNTIME"]);
const LAYER_LOCATIONS = new Set(["cpu", "accelerator", "unresident"]);
const RUNTIME_MANIFEST_SCHEMAS = new Set([
  ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA,
  ON_DEVICE_LLM_RUNTIME_MANIFEST_LEGACY_SCHEMA,
]);
const WORKING_MEMORY_CATEGORIES = Object.freeze([
  "graph_workspace_bytes",
  "scratch_bytes",
  "packing_and_replica_bytes",
  "allocator_overhead_bytes",
  "backend_private_bytes",
  "other_runtime_bytes",
]);

function exactFrom(value) {
  if (value && typeof value === "object" && /^\d+$/.test(String(value.decimal || ""))) return BigInt(value.decimal);
  if (typeof value === "string" && /^\d+$/.test(value)) return BigInt(value);
  if (Number.isSafeInteger(value) && value >= 0) return BigInt(value);
  return null;
}

function exact(value) {
  return { value: value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null, decimal: String(value) };
}

function positive(value) { return Number.isSafeInteger(value) && value > 0 ? value : null; }
function sha(value) { return typeof value === "string" && /^[a-f0-9]{64}$/.test(value) ? value : null; }
function text(value) { return typeof value === "string" && value.trim() ? value.trim() : null; }

function sectionEvidence(section, path, issues) {
  const value = section?.evidence_class;
  if (!SECTION_EVIDENCE.has(value)) issues.push({ code: "LLM_RUNTIME_SECTION_EVIDENCE_INVALID", path, observed: value ?? null });
  return SECTION_EVIDENCE.has(value) ? value : null;
}

function expectedState(contract, context, batch, bits) {
  const kv = exactFrom(contract?.state?.kv_projection?.elements_per_token_per_batch);
  const recurrent = exactFrom(contract?.state?.recurrent_projection?.recurrent_state_elements_all_layers_per_batch);
  if (kv != null && recurrent != null) return {
    kind: "hybrid_kv_ssm",
    logicalBytes: (kv * BigInt(context) + recurrent) * BigInt(batch) * BigInt(bits / 8),
  };
  if (kv != null) return { kind: "transformer_kv", logicalBytes: kv * BigInt(context) * BigInt(batch) * BigInt(bits / 8) };
  if (recurrent != null) return { kind: "ssm_recurrent", logicalBytes: recurrent * BigInt(batch) * BigInt(bits / 8) };
  return null;
}

export function buildLlmStateScenarioMatrix(contract, {
  contexts = [512, 2048, 8192], batches = [1, 2, 4], storageBits = [8, 16, 32],
} = {}) {
  const kv = exactFrom(contract?.state?.kv_projection?.elements_per_token_per_batch);
  const recurrent = exactFrom(contract?.state?.recurrent_projection?.recurrent_state_elements_all_layers_per_batch);
  if (kv == null && recurrent == null) return [];
  const declaredContext = positive(contract?.architecture?.context_length);
  const normalizedContexts = kv == null ? [null] : [...new Set(contexts.filter(positive).map((value) => declaredContext ? Math.min(value, declaredContext) : value))].sort((a, b) => a - b);
  return normalizedContexts.flatMap((context) => batches.filter(positive).flatMap((batch) => storageBits.filter((bits) => positive(bits) && bits % 8 === 0).map((bits) => ({
    state_kind: kv != null && recurrent != null ? "hybrid_kv_ssm" : kv != null ? "transformer_kv" : "ssm_recurrent",
    context_length: context,
    batch_size: batch,
    storage_bits: bits,
    logical_bytes: exact((kv != null ? kv * BigInt(context) + (recurrent || 0n) : recurrent) * BigInt(batch) * BigInt(bits / 8)),
    evidence_class: "DERIVED_CONDITIONAL_SCENARIO",
  }))));
}

export function assessOnDeviceLlmRuntimeManifest(sidecar, analysis, contract) {
  if (!sidecar?.document) return {
    schema: ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA,
    status: "not_artifact_bound",
    evidence_class: "NOT_ASSESSABLE",
    required_bindings: requiredBindings(),
    source: null,
    issues: [],
  };
  const document = sidecar.document;
  const issues = [];
  if (!document || typeof document !== "object" || Array.isArray(document)) issues.push({ code: "LLM_RUNTIME_MANIFEST_NOT_OBJECT" });
  if (!RUNTIME_MANIFEST_SCHEMAS.has(document?.schema)) issues.push({ code: "LLM_RUNTIME_MANIFEST_SCHEMA_INVALID", observed: document?.schema ?? null });
  const activeSha = sha(analysis?.artifact_bundle?.model_source_sha256) || sha(analysis?.model_sha256);
  if (!activeSha || sha(document?.artifact?.sha256) !== activeSha) issues.push({ code: "LLM_RUNTIME_ARTIFACT_SHA256_MISMATCH", expected: activeSha, observed: document?.artifact?.sha256 ?? null });
  if (String(document?.artifact?.format || "").toLowerCase() !== String(analysis?.format || "").toLowerCase()) issues.push({ code: "LLM_RUNTIME_ARTIFACT_FORMAT_MISMATCH" });

  const runtimeEvidence = sectionEvidence(document?.runtime, "/runtime", issues);
  const runtime = {
    evidence_class: runtimeEvidence,
    engine: text(document?.runtime?.engine),
    version: text(document?.runtime?.version),
    source_repository: text(document?.runtime?.source_repository),
    source_commit: /^[a-f0-9]{40}$/.test(String(document?.runtime?.source_commit || "")) ? document.runtime.source_commit : null,
    binary_sha256: sha(document?.runtime?.binary_sha256),
    build_configuration_sha256: sha(document?.runtime?.build_configuration_sha256),
  };
  for (const key of ["engine", "version", "binary_sha256", "build_configuration_sha256"]) if (!runtime[key]) issues.push({ code: "LLM_RUNTIME_IDENTITY_FIELD_MISSING", field: key });

  const deploymentEvidence = sectionEvidence(document?.deployment, "/deployment", issues);
  const context = positive(document?.deployment?.context_length);
  const batch = positive(document?.deployment?.batch_size);
  const cacheBits = positive(document?.deployment?.state_storage_bits);
  if (!context || !batch || !cacheBits || cacheBits % 8 !== 0) issues.push({ code: "LLM_RUNTIME_DEPLOYMENT_DIMENSION_INVALID" });
  const declaredContext = positive(contract?.architecture?.context_length);
  if (context && declaredContext && context > declaredContext) issues.push({ code: "LLM_RUNTIME_CONTEXT_EXCEEDS_DECLARED_ARCHITECTURE", declared: declaredContext, observed: context });

  const weightEvidence = sectionEvidence(document?.weight_residency, "/weight_residency", issues);
  const artifactBytes = exactFrom(document?.weight_residency?.artifact_serialized_tensor_bytes);
  const expectedArtifactBytes = exactFrom(contract?.storage?.serialized_tensor_bytes_decimal);
  if (artifactBytes == null || expectedArtifactBytes == null || artifactBytes !== expectedArtifactBytes) issues.push({ code: "LLM_RUNTIME_SERIALIZED_WEIGHT_BYTES_MISMATCH", expected: expectedArtifactBytes == null ? null : String(expectedArtifactBytes), observed: artifactBytes == null ? null : String(artifactBytes) });
  const runtimeWeightBytes = exactFrom(document?.weight_residency?.runtime_weight_bytes);
  const cpuBytes = exactFrom(document?.weight_residency?.cpu_bytes);
  const acceleratorBytes = exactFrom(document?.weight_residency?.accelerator_bytes);
  const unresidentBytes = exactFrom(document?.weight_residency?.unresident_bytes);
  if ([runtimeWeightBytes, cpuBytes, acceleratorBytes, unresidentBytes].some((value) => value == null)
    || cpuBytes + acceleratorBytes + unresidentBytes !== runtimeWeightBytes) issues.push({ code: "LLM_RUNTIME_WEIGHT_RESIDENCY_CONSERVATION_FAILED" });

  const layerEvidence = sectionEvidence(document?.layer_placement, "/layer_placement", issues);
  const expectedLayers = positive(contract?.architecture?.layer_count);
  const declaredLayers = positive(document?.layer_placement?.layer_count);
  const assignments = Array.isArray(document?.layer_placement?.assignments) ? document.layer_placement.assignments : [];
  if (!expectedLayers || declaredLayers !== expectedLayers || assignments.length !== expectedLayers) issues.push({ code: "LLM_RUNTIME_LAYER_ASSIGNMENT_CARDINALITY_MISMATCH", expected: expectedLayers, observed: assignments.length });
  const seen = new Set();
  for (const row of assignments) {
    if (!Number.isSafeInteger(row?.layer_index) || row.layer_index < 0 || row.layer_index >= (expectedLayers || 0) || seen.has(row.layer_index) || !LAYER_LOCATIONS.has(row?.location)) {
      issues.push({ code: "LLM_RUNTIME_LAYER_ASSIGNMENT_INVALID", row });
    } else seen.add(row.layer_index);
  }

  const stateEvidence = sectionEvidence(document?.state_cache, "/state_cache", issues);
  const expected = context && batch && cacheBits && cacheBits % 8 === 0 ? expectedState(contract, context, batch, cacheBits) : null;
  const logicalBytes = exactFrom(document?.state_cache?.logical_bytes);
  if (!expected || document?.state_cache?.kind !== expected.kind || logicalBytes !== expected.logicalBytes) issues.push({ code: "LLM_RUNTIME_STATE_LOGICAL_BYTES_MISMATCH", expected_kind: expected?.kind || null, expected_bytes: expected ? String(expected.logicalBytes) : null });
  const allocatedBytes = exactFrom(document?.state_cache?.allocated_bytes);
  const residentBytes = exactFrom(document?.state_cache?.resident_bytes);
  const pageSize = exactFrom(document?.state_cache?.page_size_bytes);
  const residentPages = exactFrom(document?.state_cache?.resident_page_count);
  const totalPages = exactFrom(document?.state_cache?.total_page_count);
  const paging = document?.state_cache?.paging_enabled;
  if (typeof paging !== "boolean" || allocatedBytes == null || residentBytes == null || residentBytes > allocatedBytes) issues.push({ code: "LLM_RUNTIME_STATE_ALLOCATION_INVALID" });
  if (paging && (pageSize == null || pageSize === 0n || residentPages == null || totalPages == null || residentPages > totalPages || residentBytes !== pageSize * residentPages || allocatedBytes !== pageSize * totalPages)) {
    issues.push({ code: "LLM_RUNTIME_STATE_PAGING_CONSERVATION_FAILED" });
  }
  if (!paging && [pageSize, residentPages, totalPages].some((value) => value != null)) issues.push({ code: "LLM_RUNTIME_NONPAGED_STATE_HAS_PAGE_FIELDS" });

  const workingMemory = assessWorkingMemory(document, issues);

  const evidenceClasses = [runtimeEvidence, deploymentEvidence, weightEvidence, layerEvidence, stateEvidence, workingMemory?.evidence_class].filter(Boolean);
  const observed = evidenceClasses.includes("OBSERVED_RUNTIME");
  const capture = {
    capture_id: text(document?.capture?.capture_id),
    collected_at: typeof document?.capture?.collected_at === "string" && !Number.isNaN(Date.parse(document.capture.collected_at)) ? document.capture.collected_at : null,
    device_identity_sha256: sha(document?.capture?.device_identity_sha256),
    source_file_sha256: sidecar.sha256 || null,
  };
  if (observed && (!capture.capture_id || !capture.collected_at || !capture.device_identity_sha256)) issues.push({ code: "LLM_RUNTIME_OBSERVED_CAPTURE_IDENTITY_INCOMPLETE" });

  return {
    schema: document?.schema || ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA,
    status: issues.length ? "invalid" : observed ? "artifact_bound_observed_runtime" : "artifact_bound_declared_runtime",
    evidence_class: issues.length ? "INVALID" : observed ? "DECLARED_AND_OBSERVED_RUNTIME" : "DECLARED",
    source: sidecar.path || null,
    source_sha256: sidecar.sha256 || null,
    runtime,
    deployment: { evidence_class: deploymentEvidence, context_length: context, batch_size: batch, state_storage_bits: cacheBits },
    weight_residency: { evidence_class: weightEvidence, artifact_serialized_tensor_bytes: artifactBytes == null ? null : exact(artifactBytes), runtime_weight_bytes: runtimeWeightBytes == null ? null : exact(runtimeWeightBytes), cpu_bytes: cpuBytes == null ? null : exact(cpuBytes), accelerator_bytes: acceleratorBytes == null ? null : exact(acceleratorBytes), unresident_bytes: unresidentBytes == null ? null : exact(unresidentBytes) },
    layer_placement: { evidence_class: layerEvidence, layer_count: declaredLayers, assignments, accelerator_layer_count: assignments.filter((row) => row.location === "accelerator").length, cpu_layer_count: assignments.filter((row) => row.location === "cpu").length, unresident_layer_count: assignments.filter((row) => row.location === "unresident").length },
    state_cache: { evidence_class: stateEvidence, kind: document?.state_cache?.kind || null, logical_bytes: logicalBytes == null ? null : exact(logicalBytes), allocated_bytes: allocatedBytes == null ? null : exact(allocatedBytes), resident_bytes: residentBytes == null ? null : exact(residentBytes), paging_enabled: paging === true, page_size_bytes: pageSize == null ? null : exact(pageSize), resident_page_count: residentPages == null ? null : exact(residentPages), total_page_count: totalPages == null ? null : exact(totalPages) },
    working_memory: workingMemory,
    capture,
    required_bindings: requiredBindings(),
    issue_count: issues.length,
    issues,
    boundary: workingMemory
      ? "DECLARED sections describe requested configuration; OBSERVED_RUNTIME sections require capture identity. Exclusive primary weight residency, logical state bytes, and the six working-memory categories are conserved independently. The accounted allocation is not process RSS, device-pool fit, application memory, operating-system reserve, kernel selection, latency, or task quality."
      : "DECLARED sections describe requested configuration; OBSERVED_RUNTIME sections require capture identity. Exclusive primary weight residency and logical state bytes are conserved independently. Runtime packing, replicas, workspaces, allocator overhead, backend-private allocations, kernel selection, latency, and task quality are not inferred.",
  };
}

export function validateOnDeviceLlmRuntimeContract(runtime, analysis, contract) {
  const errors = [];
  if (!RUNTIME_MANIFEST_SCHEMAS.has(runtime?.schema)) return ["runtime_schema_mismatch"];
  if (runtime.status === "not_artifact_bound") return (runtime.required_bindings || []).length === 6 ? [] : ["runtime_unbound_requirement_cardinality"];
  if (!["artifact_bound_declared_runtime", "artifact_bound_observed_runtime"].includes(runtime.status) || runtime.issue_count !== 0 || (runtime.issues || []).length) return ["runtime_status_invalid"];
  if (!text(runtime.source) || !sha(runtime.source_sha256)) errors.push("runtime_source_identity_invalid");
  if (!["DECLARED", "DECLARED_AND_OBSERVED_RUNTIME"].includes(runtime.evidence_class)) errors.push("runtime_evidence_class_invalid");
  if (!text(runtime.runtime?.engine) || !text(runtime.runtime?.version) || !sha(runtime.runtime?.binary_sha256) || !sha(runtime.runtime?.build_configuration_sha256)) errors.push("runtime_identity_invalid");
  const context = positive(runtime.deployment?.context_length), batch = positive(runtime.deployment?.batch_size), bits = positive(runtime.deployment?.state_storage_bits);
  if (!context || !batch || !bits || bits % 8 !== 0) errors.push("runtime_deployment_invalid");
  const expectedArtifactBytes = exactFrom(contract?.storage?.serialized_tensor_bytes_decimal);
  const artifactBytes = exactFrom(runtime.weight_residency?.artifact_serialized_tensor_bytes);
  const total = exactFrom(runtime.weight_residency?.runtime_weight_bytes), cpu = exactFrom(runtime.weight_residency?.cpu_bytes);
  const accelerator = exactFrom(runtime.weight_residency?.accelerator_bytes), unresident = exactFrom(runtime.weight_residency?.unresident_bytes);
  if (artifactBytes !== expectedArtifactBytes || [total, cpu, accelerator, unresident].some((value) => value == null)
    || cpu + accelerator + unresident !== total) errors.push("runtime_weight_conservation_invalid");
  const layers = positive(contract?.architecture?.layer_count);
  const assignments = runtime.layer_placement?.assignments || [];
  const unique = new Set(assignments.map((row) => row.layer_index));
  if (!layers || runtime.layer_placement?.layer_count !== layers || assignments.length !== layers || unique.size !== layers
    || assignments.some((row) => !Number.isSafeInteger(row.layer_index) || row.layer_index < 0 || row.layer_index >= layers || !LAYER_LOCATIONS.has(row.location))
    || runtime.layer_placement.accelerator_layer_count !== assignments.filter((row) => row.location === "accelerator").length
    || runtime.layer_placement.cpu_layer_count !== assignments.filter((row) => row.location === "cpu").length
    || runtime.layer_placement.unresident_layer_count !== assignments.filter((row) => row.location === "unresident").length) errors.push("runtime_layer_placement_invalid");
  const expected = context && batch && bits ? expectedState(contract, context, batch, bits) : null;
  const logical = exactFrom(runtime.state_cache?.logical_bytes), allocated = exactFrom(runtime.state_cache?.allocated_bytes), resident = exactFrom(runtime.state_cache?.resident_bytes);
  if (!expected || runtime.state_cache?.kind !== expected.kind || logical !== expected.logicalBytes || allocated == null || resident == null || resident > allocated) errors.push("runtime_state_conservation_invalid");
  if (runtime.state_cache?.paging_enabled) {
    const page = exactFrom(runtime.state_cache.page_size_bytes), residentPages = exactFrom(runtime.state_cache.resident_page_count), totalPages = exactFrom(runtime.state_cache.total_page_count);
    if (page == null || page === 0n || residentPages == null || totalPages == null || residentPages > totalPages || resident !== page * residentPages || allocated !== page * totalPages) errors.push("runtime_paging_conservation_invalid");
  } else if ([runtime.state_cache?.page_size_bytes, runtime.state_cache?.resident_page_count, runtime.state_cache?.total_page_count].some((value) => value != null)) errors.push("runtime_nonpaged_page_fields_invalid");
  errors.push(...validateWorkingMemory(runtime));
  const sectionClasses = [runtime.runtime, runtime.deployment, runtime.weight_residency, runtime.layer_placement, runtime.state_cache, runtime.working_memory].filter(Boolean).map((row) => row?.evidence_class);
  if (sectionClasses.some((value) => !SECTION_EVIDENCE.has(value))) errors.push("runtime_section_evidence_invalid");
  const observed = sectionClasses.includes("OBSERVED_RUNTIME");
  if ((runtime.status === "artifact_bound_observed_runtime") !== observed) errors.push("runtime_observation_status_mismatch");
  if (observed && (!text(runtime.capture?.capture_id) || !runtime.capture?.collected_at || Number.isNaN(Date.parse(runtime.capture.collected_at)) || !sha(runtime.capture?.device_identity_sha256))) errors.push("runtime_capture_identity_invalid");
  if (String(analysis?.format || "").toLowerCase() !== String(contract?.format || "").toLowerCase()) errors.push("runtime_contract_format_mismatch");
  return errors;
}

function requiredBindings() {
  return ["runtime_engine_version_binary_and_build", "context_batch_and_state_dtype", "exclusive_weight_residency", "complete_layer_placement", "state_allocation_and_paging", "capture_identity_for_observed_sections"];
}

function assessWorkingMemory(document, issues) {
  const section = document?.working_memory;
  if (section == null) return null;
  if (document?.schema !== ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA) {
    issues.push({ code: "LLM_RUNTIME_WORKING_MEMORY_REQUIRES_V2" });
    return null;
  }
  const evidenceClass = sectionEvidence(section, "/working_memory", issues);
  const categories = {};
  let sum = 0n;
  for (const key of WORKING_MEMORY_CATEGORIES) {
    const value = exactFrom(section?.[key]);
    if (value == null) issues.push({ code: "LLM_RUNTIME_WORKING_MEMORY_CATEGORY_INVALID", field: key });
    else {
      categories[key] = exact(value);
      sum += value;
    }
  }
  const declaredTotal = exactFrom(section?.accounted_nonweight_runtime_bytes);
  if (declaredTotal == null || declaredTotal !== sum) {
    issues.push({
      code: "LLM_RUNTIME_WORKING_MEMORY_CONSERVATION_FAILED",
      expected: String(sum),
      observed: declaredTotal == null ? null : String(declaredTotal),
    });
  }
  const coverage = section?.coverage;
  const expectedCoverage = evidenceClass === "OBSERVED_RUNTIME"
    ? "complete_observed_runtime_categories"
    : evidenceClass === "DECLARED" ? "complete_declared_runtime_categories" : null;
  if (coverage !== expectedCoverage) issues.push({ code: "LLM_RUNTIME_WORKING_MEMORY_COVERAGE_INVALID", expected: expectedCoverage, observed: coverage ?? null });
  return {
    evidence_class: evidenceClass,
    coverage: expectedCoverage,
    ...Object.fromEntries(WORKING_MEMORY_CATEGORIES.map((key) => [key, categories[key] || null])),
    accounted_nonweight_runtime_bytes: exact(sum),
    category_count: WORKING_MEMORY_CATEGORIES.length,
    boundary: "All six runtime-allocation categories are explicit and arithmetically conserved. Completeness is the manifest producer's declared or observed capture scope; the section is not process RSS and does not include application memory or operating-system reserve.",
  };
}

function validateWorkingMemory(runtime) {
  const section = runtime?.working_memory;
  if (section == null) return [];
  const errors = [];
  if (runtime.schema !== ON_DEVICE_LLM_RUNTIME_MANIFEST_SCHEMA) errors.push("runtime_working_memory_requires_v2");
  if (!SECTION_EVIDENCE.has(section.evidence_class)) errors.push("runtime_working_memory_evidence_invalid");
  const expectedCoverage = section.evidence_class === "OBSERVED_RUNTIME"
    ? "complete_observed_runtime_categories"
    : "complete_declared_runtime_categories";
  if (section.coverage !== expectedCoverage || section.category_count !== WORKING_MEMORY_CATEGORIES.length) errors.push("runtime_working_memory_coverage_invalid");
  let sum = 0n;
  for (const key of WORKING_MEMORY_CATEGORIES) {
    const value = exactFrom(section[key]);
    if (value == null) errors.push(`runtime_working_memory_category_invalid:${key}`);
    else sum += value;
  }
  if (exactFrom(section.accounted_nonweight_runtime_bytes) !== sum) errors.push("runtime_working_memory_conservation_invalid");
  return errors;
}
