import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { validateArtifactEvidenceIr } from "./artifact-ir.js";
import { MODEL_IR_METHOD_VERSION, MODEL_IR_SCHEMA, MODEL_IR_SOURCE_SCHEMA, RELATIONSHIP_KINDS, SHA256 } from "./model-ir/internal/constants.js";
import { buildProgramModel } from "./model-ir/internal/program.js";
import { buildModelProfiles } from "./model-ir/internal/profiles.js";
import { buildModelIrNativeFactLedger, validateModelIrNativeFactLedger } from "./model-ir/internal/native-facts.js";
import { buildGenericModelIrAnalysis } from "./model-ir-generic-analysis.js";
import { clone, deepFreeze, evidenceClass, exact, list, unique } from "./model-ir/internal/shared.js";

export { MODEL_IR_METHOD_VERSION, MODEL_IR_SCHEMA };

export { buildModelIrNativeFactLedger };

export function buildModelIrFromArtifactIr(input, { nativeFactLedger = null } = {}) {
  const artifactIr = validateArtifactEvidenceIr(input);
  const nativeFacts = validateModelIrNativeFactLedger(
    nativeFactLedger || buildModelIrNativeFactLedger(null, artifactIr),
    artifactIr.artifact_ir_sha256,
  );
  const program = buildProgramModel(artifactIr, nativeFacts);
  const storage = buildStorageModel(artifactIr);
  const bindings = buildBindings(program, storage);
  const modelProfiles = buildModelProfiles(artifactIr, program, storage);
  const architecture = buildArchitectureModel(artifactIr, program, storage, modelProfiles);
  const quantization = buildQuantizationModel(artifactIr);
  const staticRuntime = buildStaticRuntime(artifactIr, program);
  const observedRuntime = buildObservedRuntime(artifactIr);
  const profiles = buildProfiles(artifactIr, program, storage, quantization, staticRuntime, observedRuntime);
  const lossLedger = buildLossLedger(artifactIr, program, bindings);
  const genericAnalysis = buildGenericModelIrAnalysis({ program, storage, quantization, staticRuntime });
  const body = {
    schema: MODEL_IR_SCHEMA,
    method_version: MODEL_IR_METHOD_VERSION,
    source_contract: {
      schema: MODEL_IR_SOURCE_SCHEMA,
      sha256: artifactIr.artifact_ir_sha256,
      projection_status: "loss_accounted_compatibility_migration",
      native_fact_ledger: nativeFacts,
    },
    hash_contract: { algorithm: "SHA-256", canonicalization: "RFC8785-JCS", source_encoding: "UTF-8", excluded_pointers: ["/model_ir_sha256"] },
    artifact: clone(artifactIr.artifact),
    artifact_set: {
      status: artifactIr.artifact.artifact_set_sha256 ? "identity_bound" : "single_artifact_or_set_not_declared",
      artifact_set_sha256: artifactIr.artifact.artifact_set_sha256,
      member_refs: ["artifact:primary"],
    },
    artifact_role: inferArtifactRole(artifactIr),
    format_profile: { id: artifactIr.artifact.format, native_schema: nativeSchemaName(artifactIr.artifact.format), applicability: "applicable" },
    profiles,
    capabilities: buildCapabilities(program, storage, bindings, quantization, staticRuntime, observedRuntime),
    evidence_vocabulary: {
      classes: ["OBSERVED_SERIALIZED_ARTIFACT", "OBSERVED_RUNTIME", "DERIVED", "PREDICTED", "DECLARED_UNVERIFIED", "NOT_ASSESSABLE"],
      unknown_is_zero: false,
      static_prediction_is_runtime_observation: false,
    },
    program,
    tensors_and_storage: storage,
    weight_bindings: bindings,
    quantization,
    architecture,
    static_runtime: staticRuntime,
    observed_runtime: observedRuntime,
    generic_analysis: genericAnalysis,
    lineage: clone(artifactIr.lineage_evidence),
    completeness: buildCompleteness(profiles, lossLedger),
    loss_ledger: lossLedger,
    native_extensions: {
      status: "available_through_hash_bound_source_contract",
      source_schema: artifactIr.schema,
      source_sha256: artifactIr.artifact_ir_sha256,
      embedded: false,
    },
    interpretation_boundary: "DEEPBOM Model IR preserves serialized program, logical tensor, physical storage, quantization, architecture grouping, static projection, and observed runtime as separate evidence layers. Missing structures are represented by applicability and loss records. Display order is deterministic but is not runtime order. This engineering evidence does not assert model quality, safety, regulatory conformity, or standard conformance.",
  };
  validateModelIrBody(body);
  return deepFreeze({ ...body, model_ir_sha256: sha256TextHex(canonicalJson(body)) });
}

export function validateModelIr(document) {
  const body = clone(document);
  const digest = String(body.model_ir_sha256 || "").toLowerCase();
  delete body.model_ir_sha256;
  validateModelIrBody(body);
  if (!SHA256.test(digest) || digest !== sha256TextHex(canonicalJson(body))) throw new Error("Model IR SHA-256 is invalid.");
  return deepFreeze({ ...body, model_ir_sha256: digest });
}

function buildStorageModel(artifactIr) {
  const graphValues = new Map(artifactIr.graph.values.map((row) => [row.id, row]));
  const quantBySubject = new Map(artifactIr.quantization_contracts.records.map((row) => [row.subject_ref, row]));
  const storageObjects = artifactIr.storage_topology.objects.map((row) => {
    const logical = [...graphValues.values()].find((value) => list(value.storage_refs).includes(row.id));
    const quant = quantBySubject.get(logical?.id) || quantBySubject.get(row.id);
    return {
      id: row.id,
      native_subject_ref: row.id,
      name: row.name,
      dtype: row.dtype,
      encoding: clone(row.encoding),
      shape: clone(row.shape),
      dimension_order: dimensionOrder(artifactIr.artifact.format),
      element_count: elementCount(row.shape),
      serialized_byte_length: clone(row.serialized_byte_length),
      logical_byte_length: clone(logical?.logical_byte_length || null),
      byte_range: clone(row.byte_range),
      payload_sha256: row.payload_sha256,
      external_storage: externalStorageStatus(row),
      aliases: [],
      quantization_ref: quant ? `quantization:${quant.id}` : null,
      native_source: clone(row.native_source),
      evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
    };
  });
  return {
    status: artifactIr.storage_topology.status,
    logical_values: artifactIr.graph.values.map((row) => ({ id: row.id, dtype: row.dtype, shape: clone(row.shape), storage_refs: clone(row.storage_refs), evidence_class: "OBSERVED_SERIALIZED_ARTIFACT" })),
    storage_objects: storageObjects,
    storage_ranges: storageObjects.filter((row) => row.byte_range?.status === "exact").map((row) => ({ storage_ref: row.id, ...clone(row.byte_range) })),
    external_storage: storageObjects.filter((row) => row.external_storage.status !== "inline_or_not_exposed").map((row) => ({ storage_ref: row.id, ...row.external_storage })),
    aliases: [],
    totals: {
      logical_value_count: artifactIr.graph.values.length,
      storage_object_count: storageObjects.length,
      serialized_object_bytes_sum: clone(artifactIr.storage_topology.totals.serialized_object_bytes_sum),
      exact_range_count: storageObjects.filter((row) => row.byte_range?.status === "exact").length,
      payload_digest_count: storageObjects.filter((row) => SHA256.test(String(row.payload_sha256 || ""))).length,
    },
    completeness: artifactIr.storage_topology.status,
    interpretation_boundary: "Logical tensor values and serialized storage objects are distinct. Serialized bytes are not runtime allocation, device residency, repacking, transfer volume, or peak memory.",
  };
}

function buildBindings(program, storage) {
  const storageIds = new Set(storage.storage_objects.map((row) => row.id));
  const bindings = [];
  for (const port of program.ports.filter((row) => row.direction === "input")) {
    const value = program.values.find((row) => row.id === port.value_ref);
    for (const storageRef of list(value?.storage_refs)) {
      if (!storageIds.has(storageRef)) continue;
      bindings.push({
        id: `binding:${port.id}:${storageRef}`,
        operation_ref: port.operation_ref,
        port_ref: port.id,
        value_ref: port.value_ref,
        storage_ref: storageRef,
        parameter_role: `input_port_${port.position}`,
        semantic_role: null,
        binding_status: "observed_from_graph",
        binding_basis: "serialized_operation_port_to_value_and_value_to_storage",
        evidence_class: "OBSERVED_SERIALIZED_ARTIFACT",
      });
    }
  }
  return {
    status: bindings.length ? "materialized" : program.status === "not_serialized" ? "not_assessable_graph_not_serialized" : "no_bound_serialized_parameters",
    bindings,
    totals: { binding_count: bindings.length, unresolved_count: 0 },
    interpretation_boundary: "A weight binding is emitted only when a serialized operation port resolves through a logical value to a serialized storage object. Input-port position is structural; semantic roles are not inferred from names.",
  };
}

function buildQuantizationModel(artifactIr) {
  const records = artifactIr.quantization_contracts.records.map((row) => ({
    id: `quantization:${row.id}`,
    native_subject_ref: row.id,
    subject_ref: row.subject_ref,
    related_storage_refs: clone(row.related_storage_refs || []),
    family: row.mapping.family,
    encoding: row.mapping.scheme,
    storage_dtype: row.storage.data_type,
    logical_dtype: null,
    scale: clone(row.parameters.scale || null),
    zero_point: clone(row.parameters.zero_point || null),
    granularity: row.parameterization.kind,
    axis: clone(row.parameterization.axes || []),
    block_size: row.parameterization.block_size,
    effective_bits_per_element: null,
    completeness: row.completeness,
    source: clone(row.source),
    evidence_class: evidenceClass(row.source?.evidence_class, "OBSERVED_SERIALIZED_ARTIFACT"),
  }));
  return {
    status: artifactIr.quantization_contracts.status,
    records,
    totals: { ...clone(artifactIr.quantization_contracts.totals), record_count: records.length },
    interpretation_boundary: artifactIr.quantization_contracts.interpretation_boundary,
  };
}

function buildArchitectureModel(artifactIr, program, storage, modelProfiles) {
  const rootMembers = program.status === "serialized" ? (program.blocks.length ? program.blocks.map((row) => row.id) : program.regions.map((row) => row.id)) : artifactIr.architecture_projection.nodes.map((row) => row.id);
  const root = {
    id: "architecture:model:0", kind: "model", label: artifactIr.artifact.filename, parent_ref: null,
    member_refs: rootMembers, boundary_refs: [], grouping_rule_id: "deepbom.model_root.v1", grouping_rule_version: "1.0.0",
    grouping_basis: "artifact_identity_and_serialized_program_or_storage_projection", evidence_class: "DERIVED",
  };
  const structuralGroups = program.blocks.map((row) => ({
    id: row.id,
    kind: row.kind,
    label: `${row.kind}${row.repeat_index == null ? "" : ` ${row.repeat_index}`}`,
    parent_ref: root.id,
    member_refs: clone(row.member_refs),
    boundary_refs: [],
    grouping_rule_id: row.grouping_rule_id,
    grouping_rule_version: row.grouping_rule_version,
    grouping_basis: clone(row.boundary_evidence),
    evidence_class: row.evidence_class,
  }));
  const storageGroups = artifactIr.architecture_projection.nodes.map((row) => ({
    id: row.id,
    kind: row.kind,
    label: row.label,
    parent_ref: root.id,
    member_refs: clone(row.storage_object_refs || []),
    boundary_refs: [],
    grouping_rule_id: row.kind === "decoder_layer_storage_group" ? "deepbom.serialized_layer_namespace.v1" : "deepbom.storage_namespace.v1",
    grouping_rule_version: "1.0.0",
    grouping_basis: clone(row.grouping_basis || "serialized_tensor_namespace"),
    evidence_class: "DERIVED",
  }));
  const groups = [...structuralGroups, ...storageGroups];
  const allMemberRefs = new Set([...program.regions.map((row) => row.id), ...program.operations.map((row) => row.id), ...program.blocks.map((row) => row.id), ...storage.storage_objects.map((row) => row.id), ...groups.map((row) => row.id)]);
  return {
    status: groups.length || program.regions.length ? "materialized" : "not_assessable",
    hierarchy: [root, ...groups],
    relationships: clone(artifactIr.architecture_projection.relationships),
    model_profiles: modelProfiles.profiles,
    profile_analysis: { status: modelProfiles.status, rule_contract: modelProfiles.rule_contract, unassigned_operation_refs: modelProfiles.unassigned_operation_refs, unassigned_storage_refs: modelProfiles.unassigned_storage_refs },
    totals: { hierarchy_node_count: 1 + groups.length, referenced_member_count: rootMembers.length + groups.reduce((sum, row) => sum + row.member_refs.length, 0) },
    unresolved_member_refs: [root, ...groups].flatMap((row) => row.member_refs).filter((ref) => !allMemberRefs.has(ref)),
    interpretation_boundary: "Architecture hierarchy groups existing IR subjects without changing or omitting their source identity. Storage namespace groups are review coordinates, not serialized execution blocks. Model-profile roles remain absent until a capability-gated profile pass supplies traceable evidence.",
  };
}

function buildStaticRuntime(artifactIr, program) {
  const dependencyByTarget = new Map();
  for (const edge of program.relationships.filter((row) => row.kind === "data_dependency")) {
    if (!dependencyByTarget.has(edge.to_ref)) dependencyByTarget.set(edge.to_ref, []);
    dependencyByTarget.get(edge.to_ref).push(edge.from_ref);
  }
  const projections = artifactIr.overlays.static.map((overlay, overlayIndex) => ({
    id: overlay.id,
    target_profile: clone(overlay.summary || null),
    candidate_backends: unique(overlay.rows.map((row) => row.backend).filter(Boolean)),
    segments: overlay.rows.map((row, index) => ({
      id: `static-segment:${overlayIndex}:${index}`,
      source_subject_refs: [row.subject_ref],
      predecessor_subject_refs: unique(dependencyByTarget.get(row.subject_ref) || []),
      candidate_backend: row.backend,
      state: row.state,
      conditions: clone(row.unresolved_predicates || []),
      exclusions: clone(row.reason_codes || []),
      evidence_class: evidenceClass(row.evidence_class, "PREDICTED"),
      actual_runtime_assignment_claim: false,
    })),
    interpretation_boundary: overlay.interpretation_boundary,
  }));
  return {
    status: projections.length ? "available" : "not_assessed_or_not_applicable",
    projections,
    completeness: projections.length ? "source_and_profile_bounded" : "not_assessed_or_not_applicable",
    interpretation_boundary: "Static runtime projections express conditional eligibility, exclusion, fallback, memory, or compute candidates. They do not establish selected-build acceptance, runtime scheduling, fusion, placement, latency, throughput, or model quality.",
  };
}

function buildObservedRuntime(artifactIr) {
  const overlays = artifactIr.overlays.runtime.map((overlay) => ({
    id: overlay.id,
    runtime_identity: clone(overlay.summary || null),
    runtime_nodes: clone(overlay.runtime_nodes || []),
    rows: clone(overlay.rows || []),
    artifact_binding: { artifact_sha256: artifactIr.artifact.sha256, status: "identity_bound" },
    reconciliation: clone(overlay.reconciliation || null),
    evidence_class: "OBSERVED_RUNTIME",
    interpretation_boundary: overlay.interpretation_boundary,
  }));
  return {
    status: overlays.length ? "imported_identity_bound" : "not_imported",
    overlays,
    reconciliation_states: ["confirmed", "contradicted", "not_observed", "partially_observed", "unmapped_runtime_node"],
    interpretation_boundary: "Observed runtime evidence is a separate identity-bound overlay. It may confirm or contradict a static projection but never rewrites serialized artifact facts.",
  };
}

function buildProfiles(artifactIr, program, storage, quantization, staticRuntime, observedRuntime) {
  return [
    profile("identity", "core_mandatory", true, "complete"),
    profile("program_graph", "profile_mandatory", program.status === "serialized", program.status === "serialized" ? program.completeness.status : "not_applicable_not_serialized"),
    profile("tensor_storage", "profile_mandatory", storage.storage_objects.length > 0, storage.completeness),
    profile("quantization", "profile_mandatory", quantization.records.length > 0, quantization.status),
    profile("architecture_hierarchy", "optional_extension", true, artifactIr.architecture_projection.status),
    profile("static_runtime_projection", "optional_extension", staticRuntime.projections.length > 0, staticRuntime.status),
    profile("observed_runtime", "optional_extension", observedRuntime.overlays.length > 0, observedRuntime.status),
  ];
}

function profile(id, requirement, applicable, status) { return { id, requirement, applicability: applicable ? "applicable" : "not_applicable", status }; }

function buildCapabilities(program, storage, bindings, quantization, staticRuntime, observedRuntime) {
  return {
    serialized_program_graph: program.status === "serialized",
    tensor_shapes: program.values.length > 0 || storage.storage_objects.length > 0,
    serialized_storage: storage.storage_objects.length > 0,
    weight_bindings: bindings.bindings.length > 0,
    quantization_contracts: quantization.records.length > 0,
    static_runtime_projection: staticRuntime.projections.length > 0,
    observed_runtime_overlay: observedRuntime.overlays.length > 0,
  };
}

function buildLossLedger(artifactIr, program, bindings) {
  const rows = [];
  if (program.status !== "serialized") rows.push(loss("program_graph", "not_serialized_by_format", "The artifact contains tensor storage but no executable program graph."));
  if (!program.blocks.length && program.status === "serialized" && program.operations.length) rows.push(loss("structural_blocks", "not_materialized", "No reversible structural partition could be materialized for serialized operations."));
  if (!program.relationships.some((row) => row.kind === "control_dependency") && program.status === "serialized") rows.push(loss("control_dependencies", "not_exposed", "No independent serialized control-dependency relation is available in the source projection."));
  if (!bindings.bindings.length) rows.push(loss("weight_bindings", bindings.status, "No operation-to-storage weight binding can be established from the serialized subjects."));
  if (!artifactIr.overlays.runtime.length) rows.push(loss("runtime_order_and_assignment", "not_observed", "No identity-bound runtime trace was imported."));
  if (artifactIr.lineage_evidence.status === "not_provided") rows.push(loss("conversion_lineage", "not_provided", "The artifact does not independently establish its source model or conversion procedure."));
  return { status: rows.length ? "bounded_losses_recorded" : "no_known_projection_loss", entries: rows, count: rows.length };
}

function loss(subject, status, explanation) { return { subject, status, explanation, evidence_class: "NOT_ASSESSABLE" }; }
function buildCompleteness(profiles, lossLedger) { return { profile_statuses: profiles.map(({ id, applicability, status }) => ({ id, applicability, status })), loss_count: lossLedger.count, unknown_is_zero: false }; }

function inferArtifactRole(artifactIr) {
  const format = artifactIr.artifact.format;
  if (["gguf", "safetensors"].includes(format)) return "weight_container";
  if (["hdf5", "pytorch_checkpoint"].includes(format)) return "training_checkpoint";
  if (format === "pt2") return "exported_program";
  if (["keras", "savedmodel"].includes(format)) return "model_package";
  if (["executorch", "pte", "ptd"].includes(format)) return "exported_program";
  if (format === "coreml" || format === "mlmodel") return "model_package";
  return artifactIr.graph.status === "serialized" ? "deployment_artifact" : "model_package";
}
function nativeSchemaName(format) { return ({ onnx: "ONNX ModelProto", tflite: "TensorFlow Lite FlatBuffer", gguf: "GGUF", safetensors: "SafeTensors", coreml: "Core ML Model", mlmodel: "Core ML Model", executorch: "ExecuTorch PTE", pte: "ExecuTorch PTE", ptd: "ExecuTorch PTD", graphdef: "TensorFlow GraphDef", savedmodel: "TensorFlow SavedModel", hdf5: "HDF5 safe envelope", keras: "Keras v3 archive safe envelope", pt2: "PyTorch PT2 archive safe envelope", pytorch_checkpoint: "PyTorch checkpoint safe envelope" })[format] || format; }
function dimensionOrder(format) { return format === "gguf" ? "format_native_ne0_innermost_first" : "format_native_declared_order"; }
function elementCount(shape) { if (!list(shape).length || list(shape).some((value) => !Number.isSafeInteger(Number(value)) || Number(value) < 0)) return null; return exact(list(shape).reduce((product, value) => product * BigInt(value), 1n)); }
function externalStorageStatus(row) { return row.native_source?.external_data ? { status: "external_bound", source: clone(row.native_source.external_data) } : { status: "inline_or_not_exposed", source: null }; }

function validateModelIrBody(value) {
  if (value?.schema !== MODEL_IR_SCHEMA || value?.method_version !== MODEL_IR_METHOD_VERSION) throw new Error("Model IR schema identity is invalid.");
  if (value?.source_contract?.schema !== MODEL_IR_SOURCE_SCHEMA || !SHA256.test(String(value?.source_contract?.sha256 || ""))) throw new Error("Model IR source contract binding is invalid.");
  validateModelIrNativeFactLedger(value.source_contract.native_fact_ledger, value.source_contract.sha256);
  if (value?.hash_contract?.algorithm !== "SHA-256" || value?.hash_contract?.canonicalization !== "RFC8785-JCS" || JSON.stringify(value?.hash_contract?.excluded_pointers) !== JSON.stringify(["/model_ir_sha256"])) throw new Error("Model IR hash contract is invalid.");
  if (!SHA256.test(String(value?.artifact?.sha256 || ""))) throw new Error("Model IR artifact identity is invalid.");
  const program = value.program;
  const expectedCapabilities = {
    serialized_program_graph: program?.status === "serialized",
    tensor_shapes: Boolean(program?.values?.length || value?.tensors_and_storage?.storage_objects?.length),
    serialized_storage: Boolean(value?.tensors_and_storage?.storage_objects?.length),
    weight_bindings: Boolean(value?.weight_bindings?.bindings?.length),
    quantization_contracts: Boolean(value?.quantization?.records?.length),
    static_runtime_projection: Boolean(value?.static_runtime?.projections?.length),
    observed_runtime_overlay: Boolean(value?.observed_runtime?.overlays?.length),
  };
  if (JSON.stringify(value.capabilities) !== JSON.stringify(expectedCapabilities)) throw new Error("Model IR capability applicability is inconsistent with its ledgers.");
  const profileById = new Map(value.profiles.map((profile) => [profile.id, profile]));
  if (profileById.size !== value.profiles.length || profileById.get("identity")?.requirement !== "core_mandatory" || profileById.get("identity")?.applicability !== "applicable") throw new Error("Model IR mandatory profile contract is invalid.");
  for (const [id, capability] of [["program_graph", "serialized_program_graph"], ["tensor_storage", "serialized_storage"], ["quantization", "quantization_contracts"], ["static_runtime_projection", "static_runtime_projection"], ["observed_runtime", "observed_runtime_overlay"]]) {
    const profile = profileById.get(id);
    if (!profile || (profile.applicability === "applicable") !== expectedCapabilities[capability]) throw new Error(`Model IR ${id} profile applicability is inconsistent.`);
  }
  for (const key of ["programs", "functions", "regions", "blocks", "operations", "values", "ports", "relationships"]) if (!Array.isArray(program?.[key])) throw new Error(`Model IR program ${key} ledger is invalid.`);
  const ids = new Set();
  const collect = (rows, label) => rows.forEach((row) => { if (!row?.id || ids.has(row.id)) throw new Error(`Model IR ${label} identity is invalid or duplicated.`); ids.add(row.id); });
  collect(program.programs, "program"); collect(program.functions, "function"); collect(program.regions, "region"); collect(program.blocks, "block"); collect(program.operations, "operation"); collect(program.values, "value"); collect(program.ports, "port");
  const storageIds = new Set(value.tensors_and_storage.storage_objects.map((row) => row.id));
  storageIds.forEach((id) => { if (ids.has(id)) throw new Error("Model IR storage identity collides with a program subject."); ids.add(id); });
  const relationshipIds = new Set();
  for (const row of program.relationships) {
    if (!row.id || relationshipIds.has(row.id) || !RELATIONSHIP_KINDS.includes(row.kind)) throw new Error("Model IR relationship identity or kind is invalid.");
    relationshipIds.add(row.id);
    if (!ids.has(row.from_ref) || !ids.has(row.to_ref) || (row.value_ref && !ids.has(row.value_ref))) throw new Error("Model IR relationship reference is invalid.");
  }
  const regionIds = new Set(program.regions.map((row) => row.id));
  const operationIds = new Set(program.operations.map((row) => row.id));
  const valueIds = new Set(program.values.map((row) => row.id));
  for (const row of program.operations) if (!regionIds.has(row.region_ref)) throw new Error("Model IR operation region reference is invalid.");
  const blockMembers = program.blocks.flatMap((row) => row.member_refs || []);
  if (new Set(blockMembers).size !== blockMembers.length || blockMembers.length !== program.operations.length || !blockMembers.every((ref) => operationIds.has(ref))) throw new Error("Model IR structural block operation conservation is invalid.");
  for (const row of program.blocks) if (!regionIds.has(row.region_ref) || !row.member_refs.length) throw new Error("Model IR structural block reference is invalid.");
  for (const row of program.ports) if (!operationIds.has(row.operation_ref) || !valueIds.has(row.value_ref)) throw new Error("Model IR port reference is invalid.");
  if (program.status === "not_serialized" && (program.operations.length || program.relationships.length || program.regions.length)) throw new Error("Model IR fabricated a program for a graphless artifact.");
  for (const binding of value.weight_bindings.bindings) if (!operationIds.has(binding.operation_ref) || !valueIds.has(binding.value_ref) || !storageIds.has(binding.storage_ref)) throw new Error("Model IR weight binding reference is invalid.");
  for (const row of value.quantization.records) if (!valueIds.has(row.subject_ref) && !storageIds.has(row.subject_ref)) throw new Error("Model IR quantization subject reference is invalid.");
  const profileIds = new Set();
  for (const profile of value.architecture.model_profiles) {
    if (!profile?.id || profileIds.has(profile.id)) throw new Error("Model IR model-profile identity is invalid or duplicated.");
    profileIds.add(profile.id);
    for (const assignment of list(profile.assignments)) if (!ids.has(assignment.subject_ref)) throw new Error("Model IR model-profile assignment reference is invalid.");
    for (const group of list(profile.groups)) {
      if (!list(group.member_refs).every((ref) => ids.has(ref))) throw new Error("Model IR model-profile group member reference is invalid.");
    }
  }
  for (const projection of value.static_runtime.projections) for (const segment of projection.segments) {
    if (!segment.source_subject_refs.every((ref) => operationIds.has(ref) || new Set(value.architecture.hierarchy.map((row) => row.id)).has(ref))) throw new Error("Model IR static runtime subject reference is invalid.");
    if (segment.actual_runtime_assignment_claim !== false) throw new Error("Model IR static projection was promoted to an actual runtime assignment.");
  }
  if (value.generic_analysis?.pass_contract?.format_specific_branch_count !== 0 || value.generic_analysis?.pass_contract?.native_ledger_access !== false) throw new Error("Model IR generic analysis crossed the format-neutral import boundary.");
  if (value.generic_analysis?.pass_count !== value.generic_analysis?.passes?.length) throw new Error("Model IR generic analysis pass conservation is invalid.");
  for (const pass of value.generic_analysis.passes) if (!Array.isArray(pass.subject_refs) || !pass.status || !pass.completeness) throw new Error("Model IR generic analysis pass contract is invalid.");
  if (value.completeness?.unknown_is_zero !== false || !Array.isArray(value.loss_ledger?.entries) || value.loss_ledger.count !== value.loss_ledger.entries.length) throw new Error("Model IR completeness or loss ledger is inconsistent.");
  if (!String(value.interpretation_boundary || "").trim()) throw new Error("Model IR interpretation boundary is missing.");
}
