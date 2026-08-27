export const QUANT_RESEARCH_COVERAGE_SCHEMA = "deepbom.quant_research_coverage.v1";

const ADVANCED_CLASSES = new Set(["full_integer", "mixed_integer"]);

export const QUANT_RESEARCH_LABS = Object.freeze([
  lab("weight_scale_integrity", "Weight scale integrity", "weight_integrity", false),
  lab("quantization_lattice", "Residual quantization lattice", "quantization_lattice"),
  lab("accumulator_atlas", "Accumulator atlas", "accumulator_atlas"),
  lab("requantization_fidelity", "Requantization fidelity", "requantization_fidelity"),
  lab("kernel_extremum_witness", "Kernel extremum witness", "kernel_extremum_witness"),
  lab("channel_vitality", "Channel vitality", "channel_vitality"),
  lab("rounding_equivalence", "Rounding equivalence", "rounding_equivalence"),
  lab("accumulator_reachability", "Accumulator reachability", "accumulator_reachability"),
  lab("numerical_abi_propagation", "Numerical ABI propagation", "numerical_abi_propagation"),
  lab("input_counterexample", "Input counterexample", "input_counterexample"),
  lab("preprocessing_realizability", "Preprocessing realizability", "preprocessing_realizability"),
  lab("preprocessing_consequence", "Preprocessing consequence", null),
  lab("contract_migration", "Residual contract migration", "contract_migration"),
  lab("residual_step_response", "Residual step response", "residual_step_response"),
  lab("residual_contract_distortion", "Residual contract distortion", "residual_contract_distortion"),
]);

function lab(id, label, evidenceKey, requiresIntegerActivations = true) {
  return Object.freeze({ id, label, evidenceKey, requiresIntegerActivations });
}

export function classifyQuantResearchArtifact(analysis) {
  if (String(analysis?.format || "tflite").toLowerCase() !== "tflite") {
    return artifactClass("not_applicable_format", "Not applicable", "Quantized TFLite research contracts do not apply to this artifact format.", "QR-CLASS-FORMAT");
  }
  const source = String(analysis?.quantization_status?.classification || "");
  const stateCounts = new Map((analysis?.quantization_status?.op_state_counts || [])
    .map((row) => [String(row.name || ""), Number(row.count || 0)]));
  if (source === "full_integer" || source === "integer_internal_float_io") {
    const boundary = source === "integer_internal_float_io"
      ? " Integer compute is present behind explicit floating-point model I/O boundaries."
      : "";
    return artifactClass("full_integer", "Full-integer activation path", `Quantized compute activations, weights, and outputs expose the integer arithmetic contracts required by the advanced labs.${boundary}`, "QR-CLASS-FULL-INTEGER", source);
  }
  if (source === "mixed_quantization" || Number(analysis?.quantization_status?.quantized_compute_ops || 0) > 0) {
    return artifactClass("mixed_integer", "Mixed integer/float path", "At least one integer compute region is present, but advanced-lab coverage can be partial at floating-point boundaries.", "QR-CLASS-MIXED", source);
  }
  if (source === "dynamic_range_or_weight_only") {
    if (Number(stateCounts.get("weight_only_or_dynamic_range") || 0) > 0) {
      return artifactClass("dynamic_range", "Dynamic-range quantized", "Stored 8-bit compute weights are present while compute activation inputs and outputs remain floating-point. Integer activation, accumulator, requantization, residual-code, and preprocessing-code proofs are outside this artifact contract.", "QR-CLASS-DYNAMIC-RANGE", source);
    }
    return artifactClass("weight_only", "Reduced-precision weight-only", "Reduced-precision storage signals exist without an artifact-visible integer activation compute path. Weight storage checks apply; integer activation-path proofs do not.", "QR-CLASS-WEIGHT-ONLY", source);
  }
  if (["qdq_signals_only", "float16_weight_storage"].includes(source) && hasReducedPrecisionConstantStorage(analysis)) {
    return artifactClass(
      "weight_only",
      "Reduced-precision weight-only",
      "Reduced-precision constant storage is expanded through DEQUANTIZE operators into a floating-point compute path. Integer activation, accumulator, requantization, residual-code, and preprocessing-code proofs do not apply.",
      "QR-CLASS-WEIGHT-ONLY",
      source,
    );
  }
  if (source === "not_quantized_float" || !hasQuantizedStorage(analysis)) {
    return artifactClass("float", "Floating-point", "No artifact-visible quantized storage and integer activation compute contract is available to the quantization research labs.", "QR-CLASS-FLOAT", source);
  }
  return artifactClass("weight_only", "Quantization signals without integer activation path", "Quantization metadata or boundary signals exist, but no complete integer activation compute contract is established.", "QR-CLASS-WEIGHT-ONLY", source);
}

export function buildQuantResearchCoverage(analysis) {
  const artifact = classifyQuantResearchArtifact(analysis);
  const rows = QUANT_RESEARCH_LABS.map((spec) => coverageRow(analysis, artifact, spec));
  const count = (predicate) => rows.filter(predicate).length;
  const classSupported = count((row) => row.class_supported);
  const artifactApplicable = count((row) => row.artifact_applicable);
  return {
    schema: QUANT_RESEARCH_COVERAGE_SCHEMA,
    evidence_class: "DERIVED",
    artifact_class: artifact.id,
    artifact_class_label: artifact.label,
    artifact_class_reason_code: artifact.reasonCode,
    artifact_class_detail: artifact.detail,
    source_quantization_classification: artifact.sourceClassification,
    lab_count: rows.length,
    class_supported_lab_count: classSupported,
    class_excluded_lab_count: rows.length - classSupported,
    artifact_applicable_lab_count: artifactApplicable,
    assessed_lab_count: count((row) => row.status === "assessed"),
    partial_lab_count: count((row) => row.status === "partial"),
    not_assessed_lab_count: count((row) => row.status === "not_assessed"),
    not_applicable_lab_count: count((row) => row.status === "not_applicable"),
    scan_denominator_policy: "Group artifacts by artifact_class. Exclude not_applicable labs from defect-free and defect-rate denominators; report not_assessed and rejected evidence separately from observed absence.",
    labs: rows,
  };
}

export function ensureQuantResearchCoverage(analysis) {
  if (!analysis || typeof analysis !== "object") return null;
  const coverage = buildQuantResearchCoverage(analysis);
  if (String(analysis.format || "tflite").toLowerCase() === "tflite") {
    analysis.quant_research_coverage = coverage;
  } else {
    // Non-TFLite viewers still consume the returned common-empty policy, but
    // the TFLite-only computation object must not contaminate format evidence.
    delete analysis.quant_research_coverage;
  }
  return coverage;
}

export function quantResearchLabCoverage(analysis, labId) {
  const coverage = analysis?.quant_research_coverage?.schema === QUANT_RESEARCH_COVERAGE_SCHEMA
    ? analysis.quant_research_coverage
    : buildQuantResearchCoverage(analysis);
  return coverage.labs.find((row) => row.id === labId) || null;
}

function coverageRow(analysis, artifact, spec) {
  const classSupported = spec.requiresIntegerActivations
    ? ADVANCED_CLASSES.has(artifact.id)
    : !["float", "not_applicable_format"].includes(artifact.id);
  if (!classSupported) {
    return {
      id: spec.id,
      label: spec.label,
      evidence_key: spec.evidenceKey,
      class_supported: false,
      artifact_applicable: false,
      status: "not_applicable",
      reason_code: artifact.reasonCode,
      reason: artifact.detail,
      render_policy: "common_empty",
    };
  }
  if (spec.id === "preprocessing_consequence") {
    return {
      id: spec.id,
      label: spec.label,
      evidence_key: null,
      class_supported: true,
      artifact_applicable: true,
      status: "not_assessed",
      reason_code: "QR-RUNTIME-INPUT-REQUIRED",
      reason: "This local runtime consequence lab becomes assessed only after an eligible preprocessing witness and runtime execution are bound.",
      render_policy: "lab",
    };
  }
  if (spec.id === "weight_scale_integrity") {
    const scanned = Number(analysis?.weight_integrity?.quantized_constant_tensors_scanned || 0);
    return {
      id: spec.id,
      label: spec.label,
      evidence_key: spec.evidenceKey,
      class_supported: true,
      artifact_applicable: scanned > 0,
      status: scanned > 0 ? "assessed" : "not_applicable",
      reason_code: scanned > 0 ? "QR-WEIGHT-SCALES-ASSESSED" : "QR-NO-QUANTIZED-CONSTANTS",
      reason: scanned > 0
        ? `${scanned} quantized constant tensor(s) were decoded for scale/grid/integrity checks.`
        : "No decodable quantized constant tensor is present.",
      render_policy: scanned > 0 ? "lab" : "common_empty",
    };
  }
  const evidence = analysis?.[spec.evidenceKey];
  const status = normalizedEvidenceStatus(evidence);
  const artifactApplicable = status !== "not_applicable";
  return {
    id: spec.id,
    label: spec.label,
    evidence_key: spec.evidenceKey,
    class_supported: true,
    artifact_applicable: artifactApplicable,
    status,
    reason_code: evidenceReasonCode(evidence, status),
    reason: evidenceReason(evidence, status),
    render_policy: status === "not_applicable" || status === "not_assessed" && zeroAssessedEvidence(evidence)
      ? "common_empty"
      : "lab",
  };
}

function normalizedEvidenceStatus(evidence) {
  if (!evidence || typeof evidence !== "object") return "not_assessed";
  const status = String(evidence.status || evidence.assessment_status || "not_assessed").toLowerCase();
  if (status === "assessed") return "assessed";
  if (status === "partial") return zeroAssessedEvidence(evidence) ? "not_assessed" : "partial";
  if (status === "not_applicable") return "not_applicable";
  return "not_assessed";
}

function zeroAssessedEvidence(evidence) {
  const rows = evidenceRows(evidence);
  const assessed = maximumFinite(evidence, [
    "assessed_op_count", "assessed_add_count", "assessed_source_count", "assessed_witness_count",
  ], rows.filter((row) => row?.assessment_status === "assessed").length);
  const candidate = maximumFinite(evidence, [
    "candidate_op_count", "candidate_add_count", "source_op_count",
  ], rows.length);
  return assessed === 0 && candidate > 0;
}

function maximumFinite(evidence, keys, fallback = 0) {
  let maximum = Number.isFinite(fallback) ? fallback : 0;
  for (const key of keys) {
    const value = Number(evidence?.[key]);
    if (Number.isFinite(value)) maximum = Math.max(maximum, value);
  }
  return maximum;
}

function evidenceRows(evidence) {
  for (const key of ["ops", "residual_adds", "sources", "candidates"]) {
    if (Array.isArray(evidence?.[key])) return evidence[key];
  }
  return [];
}

function evidenceReasonCode(evidence, status) {
  if (status === "assessed") return "QR-EVIDENCE-ASSESSED";
  if (status === "partial") return "QR-EVIDENCE-PARTIAL";
  const rowReason = evidenceRows(evidence).find((row) => row.assessment_status !== "assessed")?.reason_code;
  return String(evidence?.reason_code || rowReason || (status === "not_applicable" ? "QR-GRAPH-NOT-APPLICABLE" : "QR-EVIDENCE-NOT-ASSESSED"));
}

function evidenceReason(evidence, status) {
  if (status === "assessed") return "The emitted evidence ledger is assessed.";
  if (status === "partial") return "The artifact contains both assessed and explicitly unassessed evidence rows.";
  const row = evidenceRows(evidence).find((item) => item.assessment_status !== "assessed");
  return String(evidence?.reason || evidence?.detail || row?.reason || row?.status_detail
    || (status === "not_applicable" ? "The required graph pattern is absent." : "No assessable evidence row was emitted."));
}

function artifactClass(id, label, detail, reasonCode, sourceClassification = "") {
  return { id, label, detail, reasonCode, sourceClassification };
}

function hasQuantizedStorage(analysis) {
  return Number(analysis?.quantized_tensors || 0) > 0
    || Number(analysis?.quantization_status?.int8_tensors || 0) > 0
    || Number(analysis?.quantization_status?.uint8_tensors || 0) > 0;
}

function hasReducedPrecisionConstantStorage(analysis) {
  return (analysis?.tensors || []).some((tensor) => tensor?.constant_buffer === true
    && ["FLOAT16", "BFLOAT16", "INT8", "UINT8", "INT4", "UINT4"].includes(String(tensor.dtype || "")));
}
