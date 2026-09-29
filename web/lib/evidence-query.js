import { ANALYZER_SEMANTIC_VERSION } from "./app-config.js";
import { BROWSER_TARGET_PROFILES } from "./target-profiles.generated.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { canonicalJson, compareCanonicalText } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { EVIDENCE_QUERY_MAX_BYTES, EVIDENCE_QUERY_RESULT_SCHEMA, normalizeEvidenceQuery, sealEvidenceQueryResult, validateEvidenceQueryResult } from "./evidence-query-contract.js";

const BOUNDARY = "This is a query projection of the existing Evidence IR and source-pinned analysis rules. Static eligibility and serialized fused activation are not executed placement or runtime fusion. MACs are not measured latency. Improvement rows identify investigation priorities, not proven accuracy or speed gains. Missing evidence remains unknown. Digests establish consistency, not the authenticity of browser-produced evidence.";
const list = value => Array.isArray(value) ? value : [];

export function queryEvidence({ analysis, artifactIrContext, summary, envelope }, input) {
  const query = normalizeEvidenceQuery(input);
  const model = artifactIrContext.model_ir;
  const modelSummary = artifactIrContext.model_summary;
  if (analysis.model_sha256 !== model.artifact.sha256 || summary.artifact.sha256 !== model.artifact.sha256) throw new Error("Evidence query artifact identity mismatch.");
  if (query.target && (analysis.format !== "tflite" || analysis.target_profile?.id !== query.target)) throw new Error("Requested CPU planning target was not analyzed.");
  const all = modelSummary.rows;
  const operations = model.program.operations;
  const opByRef = new Map(operations.map(op => [op.id, op]));
  const nativeOps = artifactIrOperators(artifactIrContext.primary_view);
  const nativeByRef = new Map(nativeOps.filter(op => op.artifact_ir_subject_ref).map(op => [op.artifact_ir_subject_ref, op]));
  const nativeByIndex = new Map(nativeOps.map((op, position) => [op.index ?? position, op]));
  const rowsByRef = new Map(all.map(row => [row.subject_ref, row]));
  // Native indices are local to a scope. Never resolve a repeated index by
  // selecting the first row or equate a display-order number with that index.
  const nativeFor = row => nativeByRef.get(row.subject_ref) || null;
  const opRow = row => makeRow(row.subject_ref, row.kind, row.name, {
    ...row,
    metric_contracts: opByRef.get(row.subject_ref)?.metric_contracts ?? null,
    quantization_summary: opByRef.get(row.subject_ref)?.quantization_summary ?? null,
    attributes: opByRef.get(row.subject_ref)?.attributes ?? null,
    native_static_details: pick(nativeFor(row), ["fused_activation", "fusion_status", "fusion_detail", "quantization_state", "quantization_detail", "quant_scale_mode", "quant_risk", "xnnpack_reason", "static_action"]),
  });
  const context = {
    scan_policy: analysis.cli_scan_policy ?? null,
    target: summary.target,
    cpu_cost_target_binding: analysis.cpu_cost_target_binding ?? null,
    target_boundary: "CPU planning profile; not detected hardware, a selected delegate build, or a measured target.",
    rulepack: summary.rulepack,
    graph: summary.graph,
    selected_level: modelSummary.selected_level ?? modelSummary.projection.selected_level,
    ordering: modelSummary.projection.ordering,
    native_detail_scope: analysis.artifact_ir_primary_scope_ref ?? null,
    reproduction: summary.reproduction,
  };
  let rows;
  let unavailable = null;
  if (query.section === "operators" || query.section === "operator") {
    rows = all.map(opRow);
  } else if (query.section === "findings") {
    rows = list(envelope.findings).map((finding, index) => makeRow(`finding:${finding.id}:${index}`, "finding", finding.title || finding.id, finding));
    context.finding_boundary = "Finding classes are distinct. Severity is an investigation priority, not a measured loss of accuracy.";
  } else if (query.section === "fusion") {
    rows = all.filter(row => row.kind === "operation").map(row => {
      const native = nativeFor(row);
      const activation = native?.fused_activation ?? null;
      return makeRow(row.subject_ref, "fusion_evidence", row.name, {
        source_index: row.order.source,
        native_type: row.native_type,
        serialized_fused_activation: activation,
        serialized_activation_status: activation === null ? "not_assessable" : activation === "NONE" ? "not_encoded" : "encoded_in_artifact",
        static_review_status: native?.fusion_status ?? null,
        static_review_detail: native?.fusion_detail ?? null,
        predecessor_relationships: row.predecessor_relationships,
        runtime_fusion: "not_observed",
      });
    });
    context.fusion_boundary = "Encoded fused activation is an artifact fact. Neighboring depthwise/pointwise operators and static review hints do not establish a backend fusion candidate or executed fusion.";
    if (analysis.format !== "tflite") unavailable = "This query currently exposes TFLite serialized fused-activation evidence; no backend fusion rule is inferred for other formats.";
  } else if (query.section === "improvements") {
    const priority = { critical: 0, high: 1, medium: 2, low: 3, informational: 4 };
    const findings = [...list(envelope.findings)].sort((a, b) => (priority[a.severity] ?? 5) - (priority[b.severity] ?? 5) || compareCanonicalText(a.id, b.id));
    const candidates = [...all].sort((a, b) => compareExact(b.metrics?.macs, a.metrics?.macs) || compareExact(b.aggregates?.serialized_byte_length, a.aggregates?.serialized_byte_length) || compareCanonicalText(a.subject_ref, b.subject_ref));
    rows = [
      ...findings.map((finding, index) => makeRow(`finding:${finding.id}:${index}`, "investigation", finding.title || finding.id, { ...finding, priority_basis: "existing_finding_severity", effect_on_quality_or_latency: "not_measured" })),
      ...candidates.map(row => makeRow(row.subject_ref, "workload_review", row.name, {
        ...row, priority_basis: "descending_exact_serialized_MACs_then_bound_storage_bytes_unknown_last",
        static_action: nativeFor(row)?.static_action ?? null,
        effect_on_quality_or_latency: "not_measured",
      })),
    ];
    context.ranking_boundary = "Existing findings ordered by severity, followed by workload rows ordered by exact MACs and bound serialized storage. This ordering is not a performance bottleneck measurement or a model-quality score.";
  } else {
    const placement = buildExecutionPlacementEvidence(analysis);
    const profiles = list(placement.static_profiles);
    const ids = profiles.map(profile => profile.profile_id);
    if (query.profile_ids.some(id => !ids.includes(id))) throw new Error(`Unknown or unavailable static profile. Available profiles: ${ids.join(", ") || "none"}.`);
    const selected = query.profile_ids.length ? profiles.filter(profile => query.profile_ids.includes(profile.profile_id)) : profiles;
    context.available_profile_ids = ids;
    context.placement_scope = "Independent source-pinned profiles; each profile counts the same graph separately. Do not add profile coverage counts together.";
    context.profiles = selected.map(profile => ({
      profile_id: profile.profile_id, label: profile.label, source: profile.source,
      evidence_class: profile.evidence_class, op_count: profile.op_count,
      state_counts: profile.state_counts, conservation: profile.conservation,
      boundary_edge_count: profile.boundary_edge_count,
      boundary_payload: profile.boundary_payload,
      conditionally_eligible_mac_share: profile.workload_envelope?.conditionally_eligible_mac_share_decimal ?? null,
      interpretation_boundary: profile.interpretation_boundary,
    }));
    if (query.section === "profiles") {
      rows = selected.map(profile => makeRow(profile.profile_id, "backend_profile", profile.label, context.profiles.find(row => row.profile_id === profile.profile_id)));
      if (analysis.format === "tflite") rows.push(...BROWSER_TARGET_PROFILES.map(profile => makeRow(profile.id, "cpu_planning_profile", profile.label, pick(profile, ["id", "label", "profile_sha256", "architecture", "hardware_spec", "performance_model_evidence_class", "performance_model_assumption"]))));
    } else {
      rows = selected.flatMap(profile => profile.rows.map(row => {
        const match = rowsByRef.get(nativeByIndex.get(row.op_index)?.artifact_ir_subject_ref);
        if (!match) throw new Error("Static placement row has no identity-bound primary-scope Model IR subject.");
        const ref = match.subject_ref;
        return makeRow(ref, "static_placement", row.op_name || `Operator ${row.op_index}`, {
          ...row, profile_id: profile.profile_id, source: profile.source,
          model_subject_ref: ref,
          predecessor_relationships: match.predecessor_relationships,
          runtime_assignment: "not_observed",
        });
      }));
      if (!rows.length) unavailable = "No per-operation static backend eligibility profile is available for this artifact. Storage-only artifacts cannot supply an execution graph.";
    }
  }
  const total = rows.length;
  if (query.subject_ref !== undefined) rows = rows.filter(row => row.subject_ref === query.subject_ref);
  if (query.source_index !== undefined) {
    const matches = all.filter(row => row.kind === "operation" && row.order.source === query.source_index);
    if (matches.length > 1) throw new Error("This native index occurs in multiple scopes. Query operators to obtain an exact subject_ref.");
    rows = rows.filter(row => matches.some(match => row.subject_ref === match.subject_ref));
  }
  if (query.search) {
    const term = query.search.toLowerCase();
    rows = rows.filter(row => `${row.title} ${row.subject_ref} ${row.kind} ${row.details.native_type?.name || ""}`.toLowerCase().includes(term));
  }
  if (query.section === "operator" && rows.length !== 1) throw new Error("Operator detail did not resolve exactly one subject. Use an exact subject_ref from the operators query.");
  if (unavailable) context.unavailable_reason = unavailable;
  // Bound nested ledgers explicitly, retaining truncation paths. Full rows can
  // be inspected in local exports; no omitted detail is presented as complete.
  const page = rows.slice(query.offset, query.offset + query.limit);
  const boundedContext = boundDetails(context);
  if (boundedContext.paths.length) boundedContext.value.detail_truncations = boundedContext.paths;
  const body = {
    schema: EVIDENCE_QUERY_RESULT_SCHEMA, analyzer_version: ANALYZER_SEMANTIC_VERSION,
    analysis_location: "chatgpt_browser_sandbox",
    artifact: { filename: model.artifact.filename, format: model.artifact.format, sha256: model.artifact.sha256 },
    model_ir_sha256: model.model_ir_sha256,
    query, query_sha256: sha256TextHex(canonicalJson(query)),
    status: unavailable ? "not_assessable" : rows.length ? "complete" : "no_matches",
    coverage: { total_rows: total, matched_rows: rows.length, returned_rows: 0, offset: query.offset, next_offset: null, detail_truncated: false },
    context: boundedContext.value, rows: page,
    evidence_boundary: BOUNDARY, transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
  };
  const seal = () => {
    body.coverage.returned_rows = page.length;
    body.coverage.next_offset = query.offset + page.length < rows.length ? query.offset + page.length : null;
    body.coverage.detail_truncated = boundedContext.paths.length > 0 || page.some(row => row.detail_truncations.length > 0);
    if (!unavailable && rows.length) body.status = body.coverage.next_offset !== null || body.coverage.detail_truncated ? "partial" : "complete";
    return sealEvidenceQueryResult(body);
  };
  let result = seal();
  while (new TextEncoder().encode(JSON.stringify(result)).length > EVIDENCE_QUERY_MAX_BYTES && page.length > 1) { page.pop(); result = seal(); }
  return validateEvidenceQueryResult(result);
}

function makeRow(ref, kind, title, details) {
  const bounded = boundDetails(details);
  const text = String(title || ref);
  if (text.length > 512) bounded.paths.push("title");
  return { subject_ref: String(ref), kind, title: text.slice(0, 512), details: bounded.value, detail_truncations: bounded.paths };
}
function pick(value, keys) { return value ? Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]])) : null; }
function boundDetails(value) {
  const paths = [];
  const visit = (item, path, depth) => {
    if (depth > 8) { paths.push(path); return null; }
    if (typeof item === "string" && item.length > 1200) { paths.push(path); return item.slice(0, 1200); }
    if (Array.isArray(item)) { if (item.length > 32) paths.push(path); return item.slice(0, 32).map((child, index) => visit(child, `${path}/${index}`, depth + 1)); }
    if (item && typeof item === "object") return Object.fromEntries(Object.entries(item).map(([key, child]) => [key, visit(child, `${path}/${key}`, depth + 1)]));
    return item;
  };
  return { value: visit(value, "", 0), paths };
}
function compareExact(a, b) {
  const decimal = value => typeof value?.decimal === "string" && /^\d+$/.test(value.decimal) ? BigInt(value.decimal) : null;
  const x = decimal(a), y = decimal(b);
  return x === y ? 0 : x === null ? -1 : y === null ? 1 : x > y ? 1 : -1;
}
