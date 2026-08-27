import { validateTfIdfVectorizerRowAgainstEvidence } from "./onnx-tfidf-vectorizer-conformance.js";

const POINTER = "/evidence/static_analysis/onnx_shape_inference/tfidf_vectorizer_inference";
const FINDINGS = "/evidence/findings_register/findings";
const EXPECTED_SOURCES = new Map([
  ["onnx_schema_and_shape", "1619dd419d2eaa1da3ad4155206d58d86432829a534d5a8c587269abf5c1df02"],
  ["onnx_reference", "ac8e8495a50a0b85fb0f4adf5de2284efa2f5e14a999d4e1ac12e20c1079e69d"],
  ["onnx_backend_tests", "9b8e1d00174a66727864baf40964c6f3da5d4e3a72858f0ee34589990b14e9dc"],
  ["ort_cpu_kernel_header", "76308c5d7cf403eec02d9819fa919f24cd5a8567a42b8e4cf9695592d25f5645"],
  ["ort_cpu_kernel", "8d3494b5d9344d49d97fae6a0ca2ed41cb97be2a5bfa26d598c34c96de7321b8"],
  ["ort_cpu_tests", "21e16e4382809769a9cd8857b9a57c16a1e37348248378fc5f8cb28c5ae46687"],
]);
const MLBOM_BINDINGS = [
  ["deepbom:model:onnxTfIdfVectorizerAssessedNodes", "assessed_node_count"],
  ["deepbom:model:onnxTfIdfVectorizerPassedNodes", "passed_node_count"],
  ["deepbom:model:onnxTfIdfVectorizerPartialNodes", "partially_assessed_node_count"],
  ["deepbom:model:onnxTfIdfVectorizerFailedNodes", "failed_node_count"],
  ["deepbom:model:onnxTfIdfVectorizerExactStaticNodes", "exact_static_node_count"],
  ["deepbom:model:onnxTfIdfVectorizerExactNgramDefinitions", "exact_ngram_definition_count"],
  ["deepbom:model:onnxTfIdfVectorizerExactActiveNgramDefinitions", "exact_active_ngram_definition_count"],
  ["deepbom:model:onnxTfIdfVectorizerExactMatches", "exact_match_count"],
  ["deepbom:model:onnxTfIdfVectorizerExactOutputValues", "exact_output_value_count"],
  ["deepbom:model:onnxTfIdfVectorizerDuplicateOutputCoordinates", "exact_duplicate_output_coordinate_count"],
  ["deepbom:model:onnxTfIdfVectorizerWeightMappingDisagreements", "exact_weight_coordinate_value_disagreement_count"],
  ["deepbom:model:onnxTfIdfVectorizerOrtReferenceDivergentOutputs", "exact_ort_reference_divergent_output_count"],
];

export function buildOnnxTfIdfConformanceFacts(inference = {}) {
  const rows = inference.rows || [];
  const failedRows = rows.filter((row) => row.status === "fail");
  const partialRows = rows.filter((row) => row.status === "partial");
  return {
    rows, failedRows, partialRows,
    passedRows: rows.filter((row) => row.status === "pass"),
    exactRows: rows.filter((row) => row.static_execution_status === "assessed_exact"),
    weightRows: rows.filter((row) => (row.risk_codes || []).includes("tfidf_weight_coordinate_semantics_divergence")),
    referenceRows: rows.filter((row) => Number(row.exact_ort_reference_divergent_output_count || 0) > 0),
    noncanonicalRows: rows.filter((row) => (row.risk_codes || []).some((risk) => [
      "tfidf_ngram_counts_ignore_pool_prefix", "tfidf_duplicate_ngram_outside_active_length_range",
      "tfidf_multiple_ngrams_share_output_coordinate",
    ].includes(risk))),
    boundedRows: partialRows.filter((row) => ["not_assessed_output_element_limit", "not_assessed_work_limit"].includes(row.static_execution_status)),
    definitionCount: sum(rows, "exact_ngram_definition_count"),
    activeDefinitionCount: sum(rows, "exact_active_ngram_definition_count"),
    matchCount: nullableSum(rows, "exact_match_count"),
    outputValueCount: nullableSum(rows, "exact_output_value_count"),
    duplicateCoordinateCount: nullableSum(rows, "exact_duplicate_output_coordinate_count"),
    weightDisagreementCount: nullableSum(rows, "exact_weight_coordinate_value_disagreement_count"),
    referenceDivergenceCount: nullableSum(rows, "exact_ort_reference_divergent_output_count"),
  };
}

export function registerOnnxTfIdfConformanceChecks({ check, inference = {}, tensors = [], ops = [], findings = [], engineeringReport = "" }) {
  const facts = buildOnnxTfIdfConformanceFacts(inference);
  const sources = inference.source_documents || [];
  const roles = new Set(sources.map((source) => source.role));
  check("CF-SHAPE-TFIDF-001", inference.schema === "deepbom.onnx_tfidf_vectorizer_inference.v1"
    && inference.evidence_class === "SOURCE_PINNED_AND_DERIVED"
    && inference.source_release === "v1.21.0"
    && inference.source_commit === "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b"
    && inference.runtime_reference_release === "v1.26.0"
    && inference.runtime_reference_commit === "8c546c37b43caaca1fa25db430dab94b901cf277"
    && sources.length === EXPECTED_SOURCES.size && roles.size === sources.length
    && sources.every((source) => EXPECTED_SOURCES.get(source.role) === source.sha256
      && String(source.source_ref || "").includes(source.role.startsWith("onnx_") ? inference.source_commit : inference.runtime_reference_commit)),
  "TfIdfVectorizer source roles, full commits, or raw-file hashes do not match the independent pinned inventory.", [`${POINTER}/source_documents`]);
  check("CF-SHAPE-TFIDF-002", facts.rows.every((row) => validateTfIdfVectorizerRowAgainstEvidence(row, tensors, ops)),
    "A TfIdfVectorizer row does not independently reconstruct from public tensor and serialized node-attribute evidence.", [`${POINTER}/rows`, "/evidence/static_analysis/ops", "/evidence/static_analysis/tensors"]);
  check("CF-SHAPE-TFIDF-003", ledgerConserves(facts, inference),
    "TfIdfVectorizer node status, definition, match, output, coordinate, or divergence aggregates do not conserve.", [POINTER]);
  check("CF-SHAPE-TFIDF-004", sameRows(inference.failed_rows, facts.failedRows) && sameRows(inference.partial_rows, facts.partialRows),
    "TfIdfVectorizer failed/partial row projections do not match the canonical row ledger.", [`${POINTER}/rows`, `${POINTER}/failed_rows`, `${POINTER}/partial_rows`]);

  const finding = (id) => findings.find((item) => item.finding_id === id);
  findingCheck(check, finding, "EA-ONX-0066", facts.failedRows, "High", "CF-SHAPE-TFIDF-005");
  findingCheck(check, finding, "EA-ONX-0067", facts.weightRows, "High", "CF-SHAPE-TFIDF-006");
  findingCheck(check, finding, "EA-ONX-0068", facts.referenceRows, "Medium", "CF-SHAPE-TFIDF-007");
  findingCheck(check, finding, "EA-ONX-0069", facts.noncanonicalRows, "Medium", "CF-SHAPE-TFIDF-008");
  findingCheck(check, finding, "EA-ONX-0070", facts.boundedRows, "Informational", "CF-SHAPE-TFIDF-009");

  const report = String(engineeringReport || "");
  check("CF-SHAPE-TFIDF-010", !facts.rows.length || report.includes("TfIdfVectorizer-9 engine")
    && report.includes("TfIdfVectorizer-9 source documents")
    && report.includes("Exact TfIdfVectorizer facts")
    && report.includes("### TfIdfVectorizer-9 Contracts")
    && report.includes("coordinate/value mapping disagreements")
    && report.includes("ORT/reference divergence")
    && sources.every((source) => report.includes(source.sha256)),
  "Engineering Report must expose TfIdfVectorizer source, exact arithmetic, coordinate semantics, and bounded-runtime interpretation.", [POINTER, "/engineering_report.md"]);
}

export function onnxTfIdfMlBomConserves(propertyValue, inference = {}) {
  return propertyValue("deepbom:model:onnxTfIdfVectorizerInferenceSchema") === (inference.schema || "not_assessed")
    && propertyValue("deepbom:model:onnxTfIdfVectorizerInferenceStatus") === (inference.status || "not_assessed")
    && propertyValue("deepbom:model:onnxTfIdfVectorizerSourceCommit") === (inference.source_commit || "not_assessed")
    && propertyValue("deepbom:model:onnxTfIdfVectorizerRuntimeReferenceCommit") === (inference.runtime_reference_commit || "not_assessed")
    && MLBOM_BINDINGS.every(([propertyName, field]) => comparable(propertyValue(propertyName)) === comparable(inference[field]))
    && (inference.source_documents || []).every((source) => String(propertyValue("deepbom:model:onnxTfIdfVectorizerSourceDocuments") || "").includes(`${source.role}:${source.sha256}`));
}

function ledgerConserves(facts, inference) {
  const status = facts.failedRows.length ? "fail" : facts.partialRows.length ? "partial" : facts.rows.length ? "assessed" : "not_applicable";
  return inference.status === status
    && Number(inference.assessed_node_count || 0) === facts.rows.length
    && Number(inference.passed_node_count || 0) === facts.passedRows.length
    && Number(inference.partially_assessed_node_count || 0) === facts.partialRows.length
    && Number(inference.failed_node_count || 0) === facts.failedRows.length
    && Number(inference.exact_static_node_count || 0) === facts.exactRows.length
    && Number(inference.exact_ngram_definition_count || 0) === facts.definitionCount
    && Number(inference.exact_active_ngram_definition_count || 0) === facts.activeDefinitionCount
    && inference.exact_match_count === facts.matchCount
    && inference.exact_output_value_count === facts.outputValueCount
    && inference.exact_duplicate_output_coordinate_count === facts.duplicateCoordinateCount
    && inference.exact_weight_coordinate_value_disagreement_count === facts.weightDisagreementCount
    && inference.exact_ort_reference_divergent_output_count === facts.referenceDivergenceCount;
}

function findingCheck(check, finding, id, rows, priority, code) {
  check(code, rows.length ? finding(id)?.technical_priority === priority : !finding(id),
    `${id} must exist exactly when its independently reconstructed TfIdfVectorizer condition is present.`, [`${POINTER}/rows`, FINDINGS]);
}

function sum(rows, field) { return rows.reduce((total, row) => total + Number(row[field] || 0), 0); }
function nullableSum(rows, field) {
  if (!rows.length || rows.some((row) => !Number.isSafeInteger(row[field]) || row[field] < 0)) return null;
  const value = rows.reduce((total, row) => total + row[field], 0);
  return Number.isSafeInteger(value) ? value : null;
}
function sameRows(left = [], right = []) { return JSON.stringify(left) === JSON.stringify(right); }
function comparable(value) { return value == null || value === "not_assessed" ? "not_assessed" : String(value); }
