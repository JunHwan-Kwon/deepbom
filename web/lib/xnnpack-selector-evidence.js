import { csvCell } from "./report-utils.js";
import { formatPercentRange } from "./format.js";

export const XNNPACK_SELECTOR_EVIDENCE_SCHEMA = "deepbom.xnnpack_selector_evidence.v2";
export const XNNPACK_SELECTOR_METHOD_VERSION = "2026-07-16.2";

const SHA256 = /^[a-f0-9]{64}$/;
const SOURCE_COMMIT = /^[a-f0-9]{40}$/;
const CANDIDATE_UNRESOLVED_DIMENSIONS = Object.freeze([
  "runtime_architecture_identity",
  "compile_configuration",
  "lowering_shape",
  "runtime_dispatch",
]);
const NO_MATCH_UNRESOLVED_DIMENSIONS = Object.freeze(["unenumerated_lowering_paths"]);
const SELECTOR_FIELDS = Object.freeze([
  "xnnpack_kernel_candidate",
  "xnnpack_kernel_tile_mr",
  "xnnpack_kernel_tile_nr",
  "xnnpack_kernel_channel_tile",
  "xnnpack_kernel_primary_tile",
  "xnnpack_kernel_source",
  "xnnpack_kernel_evidence_class",
  "xnnpack_kernel_selector_status",
  "selector_artifact_facts",
  "unresolved_selector_dimensions",
  "no_match_reason_code",
  "xnnpack_kernel_candidates",
  "xnnpack_kernel_alignment_multiples",
  "channel_alignment_multiple",
  "channel_alignment_status",
  "channel_alignment_detail",
  "channel_tail_overhead_percent",
  "channel_tail_overhead_percent_min",
  "channel_tail_overhead_percent_max",
]);

export function applyProtectedXnnpackSelectorEvidence(analysis, evidence) {
  const staged = validateEvidence(analysis, evidence);
  const stagedByIndex = new Map(staged.map(({ op, patch }) => [Number(op.index), patch]));
  const mergedOps = (analysis.ops || []).map((op) => {
    const patch = stagedByIndex.get(Number(op.index));
    if (!patch) return op;
    return mergeSelectorPatch(op, patch);
  });
  const patchedRooflineCsv = patchRooflineCsv(analysis.roofline_csv, mergedOps);
  for (const { op, patch } of staged) {
    Object.assign(op, mergeSelectorPatch(op, patch));
  }
  reconcileSelectorDerivedSummary(analysis);
  analysis.xnnpack_selector_assessment_status = evidence.assessment_status;
  analysis.xnnpack_selector_evidence_schema = evidence.schema;
  analysis.xnnpack_selector_evidence_access = evidence.access_scope;
  analysis.xnnpack_selector_evidence_provenance = {
    schema: evidence.schema,
    method_version: evidence.method_version,
    target_profile_id: evidence.target_profile_id,
    target_profile_sha256: evidence.target_profile_sha256,
    xnnpack_source_commit: evidence.xnnpack_source_commit,
    gemm_config_sha256: evidence.gemm_config_sha256,
    dwconv_config_sha256: evidence.dwconv_config_sha256,
    assessed_op_count: evidence.assessed_op_count,
    candidate_op_count: evidence.candidate_op_count,
    candidate_configuration_count: evidence.candidate_configuration_count,
    unique_candidate_op_count: evidence.unique_candidate_op_count,
    ambiguous_candidate_op_count: evidence.ambiguous_candidate_op_count,
    no_match_op_count: evidence.no_match_op_count,
    tail_assessed_op_count: evidence.tail_assessed_op_count,
    worst_case_tail_ratio: evidence.worst_case_tail_ratio,
    worst_case_tail_op_indices: structuredCloneSafe(evidence.worst_case_tail_op_indices),
    unresolved_selector_op_count: evidence.unresolved_selector_op_count,
    unresolved_selector_dimension_count: evidence.unresolved_selector_dimension_count,
    evidence_boundary: evidence.evidence_boundary,
  };
  analysis.roofline_csv = patchedRooflineCsv;
  return analysis;
}

function mergeSelectorPatch(op, patch) {
  const semanticOutputContract = op.channel_alignment_status === "graph-output-contract"
    ? {
        channel_alignment_status: op.channel_alignment_status,
        channel_alignment_detail: op.channel_alignment_detail,
      }
    : null;
  const merged = { ...op };
  for (const field of SELECTOR_FIELDS) merged[field] = structuredCloneSafe(patch[field]);
  if (semanticOutputContract) Object.assign(merged, semanticOutputContract);
  return merged;
}

function reconcileSelectorDerivedSummary(analysis) {
  const misaligned = (analysis.ops || [])
    .filter((op) => op.channel_alignment_status === "misaligned")
    .sort((a, b) => Number(b.channel_tail_overhead_percent || 0) - Number(a.channel_tail_overhead_percent || 0)
      || Number(a.index || 0) - Number(b.index || 0));
  if (analysis.insights && typeof analysis.insights === "object") analysis.insights.misaligned_ops = misaligned.length;

  const recommendations = Array.isArray(analysis.recommendations)
    ? analysis.recommendations.filter((item) => !String(item?.title || "").includes("channel tail not aligned"))
    : [];
  const op = misaligned[0];
  if (op) {
    const multiples = (op.xnnpack_kernel_alignment_multiples || []).map(Number).filter((value) => value > 0);
    const alignment = multiples.length ? multiples.map((value) => `x${value}`).join("/") : `x${Number(op.channel_alignment_multiple || 0)}`;
    const min = Number(op.channel_tail_overhead_percent_min ?? op.channel_tail_overhead_percent ?? 0);
    const max = Number(op.channel_tail_overhead_percent_max ?? op.channel_tail_overhead_percent ?? 0);
    recommendations.push({
      priority: recommendations.length + 1,
      tone: "warn",
      title: `#${String(Number(op.index || 0)).padStart(3, "0")} ${op.name} channel tail not aligned`,
      detail: `${op.channel_alignment_detail || `Output channels ${Number(op.output_channels || 0)} vs ${alignment}.`} Source-enumerated candidate tail ${formatPercentRange(min, max)}; model-level latency impact is not estimated. Preserve semantic output axes and validate an aligned internal-channel variant with target profiling before changing the graph.`,
      op_index: Number(op.index),
    });
  }
  analysis.recommendations = recommendations.map((item, index) => ({ ...item, priority: index + 1 }));
}

export function validateProtectedXnnpackSelectorEvidence(analysis, evidence) {
  validateEvidence(analysis, evidence);
  return true;
}

function validateEvidence(analysis, evidence) {
  if (!analysis || !Array.isArray(analysis.ops)) throw new Error("Selector evidence requires a current static analysis.");
  if (!evidence || evidence.schema !== XNNPACK_SELECTOR_EVIDENCE_SCHEMA) throw new Error("Selector evidence schema is unsupported.");
  if (evidence.method_version !== XNNPACK_SELECTOR_METHOD_VERSION) throw new Error("Selector evidence method version does not match the browser contract.");
  if (evidence.access_scope !== "research") throw new Error("Selector evidence access scope is invalid.");
  if (!String(evidence.evidence_boundary || "").includes("executed runtime microkernel remains unobserved")) throw new Error("Selector evidence boundary is missing.");

  const target = analysis.target_profile || {};
  if (evidence.target_profile_id !== target.id) throw new Error("Selector evidence target profile does not match the current analysis.");
  if (!SHA256.test(String(target.profile_sha256 || "")) || evidence.target_profile_sha256 !== target.profile_sha256) throw new Error("Selector evidence target-profile SHA-256 does not match the current analysis.");
  if (!SOURCE_COMMIT.test(String(evidence.xnnpack_source_commit || ""))) throw new Error("Selector evidence source commit is invalid.");
  if (!SHA256.test(String(evidence.gemm_config_sha256 || "")) || !SHA256.test(String(evidence.dwconv_config_sha256 || ""))) throw new Error("Selector evidence source-file SHA-256 is invalid.");

  if (evidence.assessment_status === "not_available_for_profile") {
    const summaryFields = ["assessed_op_count", "candidate_op_count", "candidate_configuration_count", "unique_candidate_op_count", "ambiguous_candidate_op_count", "no_match_op_count", "tail_assessed_op_count", "worst_case_tail_ratio", "unresolved_selector_op_count", "unresolved_selector_dimension_count"];
    if ((evidence.ops || []).length || (evidence.worst_case_tail_op_indices || []).length
      || summaryFields.some((field) => Number(evidence[field] || 0) !== 0)) throw new Error("Unavailable selector evidence must not contain op assessments or summary values.");
    return [];
  }
  if (String(analysis.format || "").toLowerCase() !== "tflite" || evidence.assessment_status !== "complete" || !Array.isArray(evidence.ops)) {
    throw new Error("Selector evidence assessment is incomplete or not applicable to this artifact.");
  }

  const eligibleOps = analysis.ops.filter((op) => ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"].includes(op.name));
  if (evidence.ops.length !== eligibleOps.length || Number(evidence.assessed_op_count) !== eligibleOps.length) throw new Error("Selector evidence op coverage is incomplete.");
  const opByIndex = new Map(eligibleOps.map((op) => [Number(op.index), op]));
  const seen = new Set();
  let candidateOps = 0;
  let configurations = 0;
  let uniqueCandidateOps = 0;
  let ambiguousCandidateOps = 0;
  let noMatchOps = 0;
  let tailAssessedOps = 0;
  let unresolvedSelectorOps = 0;
  let unresolvedSelectorDimensions = 0;
  const assessedTailRows = [];
  const staged = [];

  for (const patch of evidence.ops) {
    const index = Number(patch?.op_index);
    if (!Number.isInteger(index) || seen.has(index)) throw new Error("Selector evidence contains a duplicate or invalid op index.");
    seen.add(index);
    const op = opByIndex.get(index);
    if (!op || patch.op_name !== op.name) throw new Error(`Selector evidence op identity mismatch at #${index}.`);
    const candidates = patch.xnnpack_kernel_candidates;
    if (!Array.isArray(candidates)) throw new Error(`Selector evidence candidates are missing at #${index}.`);
    const expectedFacts = deriveSelectorArtifactFacts(analysis, op);
    validateSelectorArtifactFacts(patch.selector_artifact_facts, expectedFacts, index);
    const expectedClass = candidates.length === 0 ? "SOURCE_ENUMERATED_NO_MATCH"
      : candidates.length === 1 ? "SOURCE_ENUMERATED_CANDIDATE" : "SOURCE_ENUMERATED_CANDIDATE_SET";
    if (patch.xnnpack_kernel_evidence_class !== expectedClass) throw new Error(`Selector evidence class/count mismatch at #${index}.`);

    if (candidates.length === 0) {
      noMatchOps += 1;
      validateUnresolvedDimensions(patch.unresolved_selector_dimensions, NO_MATCH_UNRESOLVED_DIMENSIONS, index);
      if (!String(patch.no_match_reason_code || "").trim()) throw new Error(`Selector no-match reason is missing at #${index}.`);
    } else {
      candidateOps += 1;
      configurations += candidates.length;
      if (candidates.length === 1) uniqueCandidateOps += 1;
      else ambiguousCandidateOps += 1;
      validateUnresolvedDimensions(patch.unresolved_selector_dimensions, CANDIDATE_UNRESOLVED_DIMENSIONS, index);
      if (patch.no_match_reason_code !== "NOT_APPLICABLE_CANDIDATES_REMAIN") throw new Error(`Selector candidate reason boundary is inconsistent at #${index}.`);
      if (expectedFacts.output_channels_status === "assessed") tailAssessedOps += 1;
    }
    unresolvedSelectorOps += patch.unresolved_selector_dimensions.length ? 1 : 0;
    unresolvedSelectorDimensions += patch.unresolved_selector_dimensions.length;

    const candidateKeys = new Set();
    const expectedMultiples = [...new Set(candidates.map((candidate) => {
      const multiple = validateCandidate(candidate, evidence, expectedFacts, index);
      const key = JSON.stringify(candidate);
      if (candidateKeys.has(key)) throw new Error(`Selector evidence contains a duplicate candidate at #${index}.`);
      candidateKeys.add(key);
      return multiple;
    }))].sort((a, b) => a - b);
    const actualMultiples = [...(patch.xnnpack_kernel_alignment_multiples || [])].map(Number).sort((a, b) => a - b);
    if (JSON.stringify(expectedMultiples) !== JSON.stringify(actualMultiples)) throw new Error(`Selector alignment set is inconsistent at #${index}.`);
    validateCandidateSummary(patch, candidates, index);
    validateTailProjection(op, patch, expectedMultiples);
    if (candidates.length && expectedFacts.output_channels_status === "assessed") assessedTailRows.push({ index, ratio: Number(patch.channel_tail_overhead_percent_max) });
    staged.push({ op, patch });
  }

  const worstRatio = assessedTailRows.length ? Math.max(...assessedTailRows.map((row) => row.ratio)) : 0;
  const worstIndices = assessedTailRows.filter((row) => closeNumber(row.ratio, worstRatio)).map((row) => row.index);
  if (candidateOps !== Number(evidence.candidate_op_count)
    || configurations !== Number(evidence.candidate_configuration_count)
    || uniqueCandidateOps !== Number(evidence.unique_candidate_op_count)
    || ambiguousCandidateOps !== Number(evidence.ambiguous_candidate_op_count)
    || noMatchOps !== Number(evidence.no_match_op_count)
    || tailAssessedOps !== Number(evidence.tail_assessed_op_count)
    || !closeNumber(worstRatio, Number(evidence.worst_case_tail_ratio))
    || JSON.stringify(worstIndices) !== JSON.stringify((evidence.worst_case_tail_op_indices || []).map(Number))
    || unresolvedSelectorOps !== Number(evidence.unresolved_selector_op_count)
    || unresolvedSelectorDimensions !== Number(evidence.unresolved_selector_dimension_count)) {
    throw new Error("Selector evidence summary counts or worst-tail ledger are inconsistent.");
  }
  return staged;
}

function validateCandidate(candidate, evidence, facts, opIndex) {
  if (!candidate?.family || !candidate.architecture_condition || !candidate.compile_condition || !candidate.runtime_condition) throw new Error(`Selector candidate conditions are incomplete at #${opIndex}.`);
  for (const field of ["tile_mr", "tile_nr", "channel_tile", "primary_tile"]) {
    const value = Number(candidate[field]);
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`Selector candidate ${field} is invalid at #${opIndex}.`);
  }
  const multiple = Math.max(Number(candidate.tile_nr || 0), Number(candidate.channel_tile || 0));
  if (!Number.isSafeInteger(multiple) || multiple <= 0) throw new Error(`Selector candidate tile is invalid at #${opIndex}.`);
  const channels = facts.output_channels_status === "assessed" ? facts.output_channels : 0;
  const padded = channels > 0 ? Math.ceil(channels / multiple) * multiple : 0;
  const inactive = padded > 0 ? padded - channels : 0;
  const ratio = padded > 0 ? inactive / padded : 0;
  const tailStatus = channels > 0 ? "assessed" : "not_assessed_output_channels_unavailable";
  if (Number(candidate.alignment_multiple) !== multiple || candidate.tail_projection_status !== tailStatus
    || Number(candidate.padded_output_channels) !== padded || Number(candidate.inactive_output_channels) !== inactive
    || !closeNumber(Number(candidate.inactive_lane_ratio), ratio)) throw new Error(`Selector candidate tail arithmetic is inconsistent at #${opIndex}.`);
  const source = String(candidate.source_ref || "");
  if (!source.includes(`@${evidence.xnnpack_source_commit}/src/configs/`)) throw new Error(`Selector candidate source commit is inconsistent at #${opIndex}.`);
  const expectedSha = source.includes("/gemm-config.c#") ? evidence.gemm_config_sha256
    : source.includes("/dwconv-config.c#") ? evidence.dwconv_config_sha256 : "";
  if (!expectedSha || candidate.source_file_sha256 !== expectedSha) throw new Error(`Selector candidate source-file hash is inconsistent at #${opIndex}.`);
  return multiple;
}

function deriveSelectorArtifactFacts(analysis, op) {
  const tensors = new Map((analysis?.tensors || []).map((tensor) => [Number(tensor.index), tensor]));
  const activation = tensors.get(Number(op.inputs?.[0]));
  const weights = tensors.get(Number(op.inputs?.[1]));
  const shape = Array.isArray(weights?.shape) ? weights.shape : [];
  const height = Number(shape[1]);
  const width = Number(shape[2]);
  const kernelProduct = height * width;
  const kernelArea = shape.length === 4 && Number.isSafeInteger(height) && height > 0
    && Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(kernelProduct) ? kernelProduct : 0;
  const rawChannels = Number(op.output_channels);
  const outputChannels = Number.isSafeInteger(rawChannels) && rawChannels > 0 ? rawChannels : 0;
  const kernelRelevant = ["CONV_2D", "DEPTHWISE_CONV_2D"].includes(op.name);
  return {
    activation_dtype: String(activation?.dtype || ""),
    kernel_area: kernelArea,
    kernel_area_status: kernelRelevant ? (kernelArea > 0 ? "assessed" : "not_assessed") : "not_applicable",
    per_channel_weights: Number(weights?.quant_scales || 0) > 1,
    output_channels: outputChannels,
    output_channels_status: outputChannels > 0 ? "assessed" : "not_assessed",
  };
}

function validateSelectorArtifactFacts(actual, expected, opIndex) {
  if (!actual || Object.keys(expected).some((field) => actual[field] !== expected[field])) throw new Error(`Selector artifact facts are inconsistent at #${opIndex}.`);
}

function validateUnresolvedDimensions(actual, expected, opIndex) {
  if (!Array.isArray(actual) || new Set(actual).size !== actual.length || JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Selector unresolved dimensions are inconsistent at #${opIndex}.`);
  }
}

function validateCandidateSummary(patch, candidates, opIndex) {
  const sources = [...new Set(candidates.map((candidate) => candidate.source_ref))].sort();
  if (candidates.length === 0) {
    if (["xnnpack_kernel_tile_mr", "xnnpack_kernel_tile_nr", "xnnpack_kernel_channel_tile", "xnnpack_kernel_primary_tile"].some((field) => Number(patch[field] || 0) !== 0)) {
      throw new Error(`Selector no-match summary is inconsistent at #${opIndex}.`);
    }
    return;
  }
  const one = candidates.length === 1;
  const expectedLabel = one ? candidates[0].family : `${candidates.length} protected source-enumerated XNNPACK configurations`;
  if (patch.xnnpack_kernel_candidate !== expectedLabel || patch.xnnpack_kernel_source !== sources.join("; ")) throw new Error(`Selector candidate summary is inconsistent at #${opIndex}.`);
  for (const field of ["tile_mr", "tile_nr", "channel_tile", "primary_tile"]) {
    const expected = one ? Number(candidates[0][field]) : 0;
    if (Number(patch[`xnnpack_kernel_${field}`] || 0) !== expected) throw new Error(`Selector legacy tile summary is inconsistent at #${opIndex}.`);
  }
}

function validateTailProjection(op, patch, multiples) {
  if (!multiples.length) {
    if (Number(patch.channel_alignment_multiple || 0) !== 0 || Number(patch.channel_tail_overhead_percent || 0) !== 0) throw new Error(`Selector no-match row contains a fabricated alignment projection at #${op.index}.`);
    return;
  }
  const channels = Number(op.output_channels || 0);
  const tails = multiples.map((multiple) => {
    const padded = channels > 0 ? Math.ceil(channels / multiple) * multiple : 0;
    return padded > 0 ? (padded - channels) / padded : 0;
  });
  const min = Math.min(...tails);
  const max = Math.max(...tails);
  const worstMultiple = Math.max(...multiples.filter((_multiple, index) => tails[index] === max));
  const expectedStatus = channels <= 0 ? "not-applicable" : max === 0 ? "aligned" : "misaligned";
  if (!closeNumber(Number(patch.channel_tail_overhead_percent_min), min)
    || !closeNumber(Number(patch.channel_tail_overhead_percent_max), max)
    || !closeNumber(Number(patch.channel_tail_overhead_percent), max)
    || Number(patch.channel_alignment_multiple) !== worstMultiple
    || patch.channel_alignment_status !== expectedStatus) throw new Error(`Selector tail projection is inconsistent at #${op.index}.`);
}

function patchRooflineCsv(csv, ops) {
  if (!String(csv || "").trim()) return String(csv || "");
  const rows = parseCsvWithRaw(csv);
  if (rows.length < 2) return String(csv || "");
  const headers = rows[0].map((cell) => cell.value);
  const column = new Map(headers.map((name, index) => [name, index]));
  const opByIndex = new Map((ops || []).map((op) => [Number(op.index), op]));
  const required = ["op_index", "xnnpack_kernel_candidate", "xnnpack_kernel_candidate_count", "xnnpack_kernel_evidence_class"];
  if (required.some((name) => !column.has(name))) throw new Error("Roofline CSV does not contain selector evidence columns.");
  for (const row of rows.slice(1)) {
    const op = opByIndex.get(Number(row[column.get("op_index")]?.value));
    if (!op) continue;
    const candidates = op.xnnpack_kernel_candidates || [];
    const updates = {
      xnnpack_kernel_candidate: op.xnnpack_kernel_candidate,
      xnnpack_kernel_candidate_count: candidates.length,
      xnnpack_kernel_candidate_families: candidates.map((item) => item.family).join(" || "),
      xnnpack_kernel_architecture_conditions: candidates.map((item) => item.architecture_condition).join(" || "),
      xnnpack_kernel_compile_conditions: candidates.map((item) => item.compile_condition).join(" || "),
      xnnpack_kernel_runtime_conditions: candidates.map((item) => item.runtime_condition).join(" || "),
      xnnpack_kernel_source_refs: candidates.map((item) => item.source_ref).join(" || "),
      xnnpack_kernel_source_file_sha256s: candidates.map((item) => item.source_file_sha256).join(" || "),
      xnnpack_kernel_alignment_multiples: (op.xnnpack_kernel_alignment_multiples || []).join("|"),
      xnnpack_kernel_tile_mr: op.xnnpack_kernel_tile_mr,
      xnnpack_kernel_tile_nr: op.xnnpack_kernel_tile_nr,
      xnnpack_kernel_channel_tile: op.xnnpack_kernel_channel_tile,
      xnnpack_kernel_primary_tile: op.xnnpack_kernel_primary_tile,
      xnnpack_kernel_source: op.xnnpack_kernel_source,
      xnnpack_kernel_evidence_class: op.xnnpack_kernel_evidence_class,
      xnnpack_kernel_selector_status: op.xnnpack_kernel_selector_status,
      xnnpack_selector_artifact_facts: selectorFactsText(op.selector_artifact_facts),
      xnnpack_unresolved_selector_dimensions: (op.unresolved_selector_dimensions || []).join("|"),
      xnnpack_no_match_reason_code: op.no_match_reason_code,
      xnnpack_candidate_tail_projections: candidates.map((candidate) => [
        candidate.family,
        `C${op.selector_artifact_facts?.output_channels || 0}->${candidate.padded_output_channels}`,
        `inactive=${candidate.inactive_output_channels}`,
        `ratio=${finiteFixed(candidate.inactive_lane_ratio)}`,
        `align=${candidate.alignment_multiple}`,
        candidate.tail_projection_status,
      ].join("; ")).join(" || "),
      channel_alignment_multiple: op.channel_alignment_multiple,
      channel_alignment_status: op.channel_alignment_status,
      channel_tail_overhead_percent: finiteFixed(op.channel_tail_overhead_percent),
      channel_tail_overhead_percent_min: finiteFixed(op.channel_tail_overhead_percent_min),
      channel_tail_overhead_percent_max: finiteFixed(op.channel_tail_overhead_percent_max),
    };
    for (const [name, value] of Object.entries(updates)) {
      const index = column.get(name);
      if (index == null) continue;
      row[index] = { value: String(value ?? ""), raw: csvCell(value) };
    }
  }
  return rows.map((row) => row.map((cell) => cell.raw).join(",")).join("\n");
}

function selectorFactsText(facts = {}) {
  return [
    `activation_dtype=${facts.activation_dtype || "unavailable"}`,
    `kernel_area=${Number(facts.kernel_area || 0)}`,
    `kernel_area_status=${facts.kernel_area_status || "not_assessed"}`,
    `per_channel_weights=${Boolean(facts.per_channel_weights)}`,
    `output_channels=${Number(facts.output_channels || 0)}`,
    `output_channels_status=${facts.output_channels_status || "not_assessed"}`,
  ].join("; ");
}

function parseCsvWithRaw(csv) {
  const rows = [];
  let row = [];
  let value = "";
  let raw = "";
  let quoted = false;
  const pushField = () => { row.push({ value, raw }); value = ""; raw = ""; };
  const pushRow = () => { pushField(); rows.push(row); row = []; };
  for (let index = 0; index < csv.length; index += 1) {
    const char = csv[index];
    if (quoted) {
      raw += char;
      if (char === '"' && csv[index + 1] === '"') { value += '"'; raw += csv[++index]; }
      else if (char === '"') quoted = false;
      else value += char;
      continue;
    }
    if (char === '"' && raw === "") { quoted = true; raw += char; continue; }
    if (char === ",") { pushField(); continue; }
    if (char === "\n") { pushRow(); continue; }
    if (char === "\r" && csv[index + 1] === "\n") continue;
    value += char;
    raw += char;
  }
  if (raw !== "" || value !== "" || row.length) pushRow();
  return rows;
}

function finiteFixed(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(6) : "0.000000";
}

function closeNumber(left, right, tolerance = 1e-9) {
  return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= tolerance;
}

function structuredCloneSafe(value) {
  if (value == null || typeof value !== "object") return value;
  return JSON.parse(JSON.stringify(value));
}
