export const MODEL_IR_GENERIC_ANALYSIS_SCHEMA = "deepbom.model_ir_generic_analysis.v1";

export function buildGenericModelIrAnalysis({ program, storage, quantization, staticRuntime }) {
  const passes = [
    interfacePass(program),
    topologyPass(program),
    storagePass(storage),
    quantizationPass(quantization),
    staticRuntimePass(staticRuntime),
  ];
  return {
    schema: MODEL_IR_GENERIC_ANALYSIS_SCHEMA,
    pass_contract: {
      input: "deepbom.model_ir.v1_sections",
      native_ledger_access: false,
      format_specific_branch_count: 0,
      unknown_value_policy: "preserve_not_assessable",
    },
    passes,
    pass_count: passes.length,
    interpretation_boundary: "These passes consume only format-neutral Model IR sections. They summarize serialized or explicitly projected facts and do not add native semantics, runtime observations, model quality, safety, regulatory conclusions, or standards-conformance verdicts.",
  };
}

function interfacePass(program) {
  const item = program.programs[0] || null;
  const applicable = program.status === "serialized";
  return pass("deepbom.generic_interface_contract.v1", {
    requiredCapabilities: ["serialized_program_graph"],
    status: applicable ? "assessed_from_serialized_program_contract" : "not_applicable_program_not_serialized",
    evidenceClass: applicable ? "OBSERVED_SERIALIZED_ARTIFACT" : "NOT_ASSESSABLE",
    subjectRefs: item ? [...item.input_value_refs, ...item.output_value_refs] : [],
    result: {
      input_value_refs: clone(item?.input_value_refs || []),
      output_value_refs: clone(item?.output_value_refs || []),
      input_count: item?.input_value_refs?.length || 0,
      output_count: item?.output_value_refs?.length || 0,
    },
    completeness: applicable ? (item ? "reported_as_serialized" : "missing_program_record") : "not_applicable",
  });
}

function topologyPass(program) {
  const executionEdges = program.relationships.filter((row) => ["data_dependency", "control_dependency", "state_dependency", "call"].includes(row.kind));
  const counts = Object.fromEntries(["data_dependency", "control_dependency", "call", "region_ownership", "state_dependency", "alias", "storage_binding", "parameter_binding"]
    .map((kind) => [kind, program.relationships.filter((row) => row.kind === kind).length]));
  return pass("deepbom.generic_program_topology.v1", {
    requiredCapabilities: ["serialized_program_graph"],
    status: program.status === "serialized" ? "assessed" : "not_applicable_program_not_serialized",
    evidenceClass: program.status === "serialized" ? "DERIVED" : "NOT_ASSESSABLE",
    subjectRefs: [...program.operations.map((row) => row.id), ...executionEdges.map((row) => row.id)],
    result: {
      region_count: program.regions.length,
      operation_count: program.operations.length,
      value_count: program.values.length,
      relationship_counts: counts,
      source_order_status: program.orders.source_order_status,
      dependency_order_status: program.orders.dependency_order_status,
      display_order_status: program.orders.display_order_status,
      runtime_order_status: program.orders.runtime_order_status,
    },
    completeness: program.status === "serialized" ? program.completeness.status : "not_applicable",
  });
}

function storagePass(storage) {
  const encodings = new Map();
  for (const row of storage.storage_objects) {
    const key = String(row.encoding?.name || row.encoding?.kind || row.dtype || "UNKNOWN");
    encodings.set(key, (encodings.get(key) || 0) + 1);
  }
  return pass("deepbom.generic_tensor_storage_inventory.v1", {
    requiredCapabilities: ["serialized_storage"],
    status: storage.storage_objects.length ? "assessed" : "not_applicable_no_serialized_storage_objects",
    evidenceClass: storage.storage_objects.length ? "DERIVED" : "NOT_ASSESSABLE",
    subjectRefs: storage.storage_objects.map((row) => row.id),
    result: {
      logical_value_count: storage.logical_values.length,
      storage_object_count: storage.storage_objects.length,
      serialized_object_bytes_sum: clone(storage.totals.serialized_object_bytes_sum),
      exact_range_count: storage.totals.exact_range_count,
      payload_digest_count: storage.totals.payload_digest_count,
      encoding_histogram: [...encodings].sort(([left], [right]) => left.localeCompare(right)).map(([encoding, count]) => ({ encoding, count })),
    },
    completeness: storage.completeness,
  });
}

function quantizationPass(quantization) {
  const families = new Map();
  for (const row of quantization.records) families.set(row.family, (families.get(row.family) || 0) + 1);
  return pass("deepbom.generic_quantization_inventory.v1", {
    requiredCapabilities: ["quantization_contracts"],
    status: quantization.records.length ? "assessed" : "not_applicable_no_quantization_contract",
    evidenceClass: quantization.records.length ? "DERIVED" : "NOT_ASSESSABLE",
    subjectRefs: quantization.records.map((row) => row.subject_ref),
    result: {
      record_count: quantization.records.length,
      family_histogram: [...families].sort(([left], [right]) => String(left).localeCompare(String(right))).map(([family, count]) => ({ family, count })),
      incomplete_record_count: quantization.records.filter((row) => !/complete/i.test(String(row.completeness))).length,
    },
    completeness: quantization.status,
  });
}

function staticRuntimePass(staticRuntime) {
  const segments = list(staticRuntime.projections).flatMap((row) => list(row.segments));
  return pass("deepbom.generic_static_runtime_projection_summary.v1", {
    requiredCapabilities: ["static_runtime_projection"],
    status: segments.length ? "assessed_conditional_projection" : "not_assessed_or_not_applicable",
    evidenceClass: segments.length ? "PREDICTED" : "NOT_ASSESSABLE",
    subjectRefs: segments.flatMap((row) => row.source_subject_refs),
    result: {
      projection_count: staticRuntime.projections.length,
      segment_count: segments.length,
      candidate_backends: [...new Set(staticRuntime.projections.flatMap((row) => row.candidate_backends || []))].sort(),
      actual_runtime_assignment_claim_count: segments.filter((row) => row.actual_runtime_assignment_claim !== false).length,
    },
    completeness: staticRuntime.completeness,
  });
}

function pass(id, { requiredCapabilities, status, evidenceClass, subjectRefs, result, completeness }) {
  return {
    id,
    version: "1.0.0",
    required_profiles: [],
    required_capabilities: requiredCapabilities,
    input_subject_types: inferSubjectTypes(id),
    status,
    evidence_class: evidenceClass,
    subject_refs: [...new Set(subjectRefs)].sort(),
    result,
    completeness,
  };
}

function inferSubjectTypes(id) {
  if (id.includes("interface")) return ["program", "value"];
  if (id.includes("topology")) return ["region", "operation", "value", "relationship"];
  if (id.includes("storage")) return ["logical_value", "storage_object"];
  if (id.includes("quantization")) return ["quantization_record"];
  return ["static_runtime_projection"];
}

function clone(value) { return value == null ? value : structuredClone(value); }
function list(value) { return Array.isArray(value) ? value : []; }
