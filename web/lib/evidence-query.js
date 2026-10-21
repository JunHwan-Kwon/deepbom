import { ANALYZER_SEMANTIC_VERSION } from "./app-config.js";
import { BROWSER_TARGET_PROFILES } from "./target-profiles.generated.js";
import { buildExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { artifactIrOperators } from "./artifact-ir-selectors.js";
import { validateWeightAnalysis } from "./weight-analysis.js";
import { workflowUrl } from "./workflow-catalog.js";
import { canonicalJson, compareCanonicalText } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";
import { EVIDENCE_QUERY_MAX_BYTES, EVIDENCE_QUERY_RESULT_SCHEMA, normalizeEvidenceQuery, sealEvidenceQueryResult, validateEvidenceQueryResult } from "./evidence-query-contract.js";

const BOUNDARY = "This is a query projection of the existing Evidence IR and source-pinned analysis rules. Static eligibility and serialized fused activation are not executed placement or runtime fusion. MACs are not measured latency. Improvement rows identify investigation priorities, not proven accuracy or speed gains. Missing evidence remains unknown. Digests establish consistency, not the authenticity of browser-produced evidence.";
const list = value => Array.isArray(value) ? value : [];

export function queryEvidence({ analysis, artifactIrContext, summary, envelope, weightEvidence }, input) {
  const query = normalizeEvidenceQuery(input);
  const model = artifactIrContext.model_ir;
  const modelSummary = artifactIrContext.model_summary;
  if (analysis.model_sha256 !== model.artifact.sha256 || summary.artifact.sha256 !== model.artifact.sha256 || envelope.identity?.sha256 !== model.artifact.sha256) throw new Error("Evidence query artifact identity mismatch.");
  if (query.target && (analysis.format !== "tflite" || analysis.target_profile?.id !== query.target)) throw new Error("Requested CPU planning target was not analyzed.");
  const all = modelSummary.rows;
  const operations = model.program.operations;
  const opByRef = new Map(operations.map(op => [op.id, op]));
  const nativeOps = artifactIrOperators(artifactIrContext.primary_view);
  const nativeByRef = new Map(nativeOps.filter(op => op.artifact_ir_subject_ref).map(op => [op.artifact_ir_subject_ref, op]));
  const nativeByIndex = new Map(nativeOps.map((op, position) => [op.index ?? position, op]));
  const rowsByRef = new Map(all.map(row => [row.subject_ref, row]));
  const successors = new Map();
  for (const row of all) for (const edge of row.predecessor_relationships || []) {
    if (!successors.has(edge.from_ref)) successors.set(edge.from_ref, []);
    successors.get(edge.from_ref).push({ ...edge, to_ref: row.subject_ref });
  }
  const hierarchy = artifactIrContext.model_hierarchy;
  const sourceByRef = new Map((hierarchy?.assignments || []).map(row => [row.subject_ref, row]));
  const groupByRef = new Map((hierarchy?.groups || []).map(row => [row.id, row]));
  // Native indices are local to a scope. Never resolve a repeated index by
  // selecting the first row or equate a display-order number with that index.
  const nativeFor = row => nativeByRef.get(row.subject_ref) || null;
  const opRow = row => makeRow(row.subject_ref, row.kind, row.name, {
    ...row,
    ...(sourceByRef.has(row.subject_ref) ? { source_hierarchy: {
      ...sourceByRef.get(row.subject_ref),
      path: groupByRef.get(sourceByRef.get(row.subject_ref).group_ref)?.path || null,
      hierarchy_sha256: hierarchy.hierarchy_sha256,
      group_kind: groupByRef.get(sourceByRef.get(row.subject_ref).group_ref)?.kind || null,
      interpretation_boundary: hierarchy.interpretation_boundary,
    } } : {}),
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
    summary_scope: "graph, finding_summary and analysis_coverage retain the source analysis scope before filtering or pagination. graph is not recomputed across nested graphs; see native_detail_scope. Row coverage describes this query section; never sum page-local rows to infer whole-model totals.",
    finding_summary: summary.verdict,
    evidence_envelope_sha256: envelope.envelope_sha256,
    analysis_coverage: summary.coverage,
    ...(hierarchy ? { hierarchy: { status: hierarchy.status, hierarchy_sha256: hierarchy.hierarchy_sha256, coverage: hierarchy.coverage, interpretation_boundary: hierarchy.interpretation_boundary } } : {}),
    reference_scope: "Use subject_ref with this artifact SHA-256. Finding references also require the same evidence_envelope_sha256 and scan policy: their source envelope index is preserved across findings and improvements, not across changed analyses or releases.",
    native_detail_scope: analysis.artifact_ir_primary_scope_ref ?? null,
    reproduction: summary.reproduction,
    continuation: { url: workflowUrl(({ operators: "layers", operator: "layers", findings: "findings", improvements: "findings", placement: "placement", profiles: "placement", fusion: "fusion", weights: "weights" })[query.section]), instruction: "For richer interaction, select the model again on deepbom.org and compare its SHA-256. No file or result is transferred by this link." },
  };
  let rows;
  let unavailable = null;
  if (query.section === "operators" || query.section === "operator") {
    rows = all.map(opRow);
    if (query.section === "operator") {
      for (const row of rows) {
        if (row.kind !== "operation") continue;
        const neighbors = (edges, field) => edges.filter(edge => rowsByRef.get(edge[field])?.kind === "operation").map(edge => ({
          subject_ref: edge[field], title: rowsByRef.get(edge[field]).name || edge[field],
          relationship_ref: edge.relationship_ref, kind: edge.kind,
        }));
        const incoming = neighbors(row.details.predecessor_relationships || [], "from_ref");
        const outgoing = neighbors(successors.get(row.subject_ref) || [], "to_ref");
        row.details.navigation = { incoming_count: incoming.length, outgoing_count: outgoing.length, incoming, outgoing,
          interpretation_boundary: "Direct serialized relationships in the full Model Summary, including relationships outside the current page. Relationship kind is preserved; these are not observed execution or gradient paths." };
      }
    }
  } else if (query.section === "findings") {
    rows = list(envelope.findings).map((finding, index) => makeRow(`finding:${finding.id}:${index}`, "finding", finding.title || finding.id, finding));
    context.finding_boundary = "Finding classes are distinct. Severity is an investigation priority, not a measured loss of accuracy.";
  } else if (query.section === "weights") {
    if (!weightEvidence) throw new Error("Weight analysis must complete before a weight query is projected.");
    const { weight_ir: weight, weight_analysis: advanced } = weightEvidence;
    validateWeightAnalysis(advanced, weight, model);
    const view = query.weight_view || "distribution";
    context.weight_evidence = { weight_ir_sha256: weight.weight_ir_sha256, weight_analysis_sha256: advanced.weight_analysis_sha256, coverage: advanced.coverage, options: advanced.options, view, interpretation_boundary: advanced.interpretation_boundary };
    const weightsByRef = new Map(weight.tensors.map(row => [row.id, row]));
    rows = advanced.tensors.map(row => makeRow(row.weight_ref, "weight_evidence", row.name || row.weight_ref, {
      ...pick(row, ["weight_ref", "storage_ref", "shape", "dtype", "status", "reason", "value_count", "representation", "axes", "statistics"]),
      feature: view === "distribution" ? { status: row.status, reason: row.reason, statistics: row.statistics } : row[view],
      binding_refs: weightsByRef.get(row.weight_ref)?.binding_refs ?? [],
    }));
    if (!rows.some(row => row.details.status === "assessed")) unavailable = "No weights could be assessed within the decoder and budget contracts. Inspect per-tensor reasons; absent evidence is not zero.";
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
    const findings = list(envelope.findings).map((finding, index) => ({ finding, index }))
      .sort((a, b) => (priority[a.finding.severity] ?? 5) - (priority[b.finding.severity] ?? 5) || compareCanonicalText(a.finding.id, b.finding.id) || a.index - b.index);
    const candidates = [...all].sort((a, b) => compareExact(b.metrics?.macs, a.metrics?.macs) || compareExact(b.aggregates?.serialized_byte_length, a.aggregates?.serialized_byte_length) || compareCanonicalText(a.subject_ref, b.subject_ref));
    rows = [
      ...findings.map(({ finding, index }) => makeRow(`finding:${finding.id}:${index}`, "investigation", finding.title || finding.id, { ...finding, priority_basis: "existing_finding_severity", effect_on_quality_or_latency: "not_measured" })),
      ...candidates.map(row => makeRow(row.subject_ref, "workload_review", row.name, {
        ...opRow(row).details, priority_basis: "descending_exact_serialized_MACs_then_bound_storage_bytes_unknown_last",
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
    rows = rows.filter(row => `${row.title} ${row.subject_ref} ${row.kind} ${row.details.native_type?.name || ""} ${row.details.source_hierarchy?.path || ""}`.toLowerCase().includes(term));
  }
  if (query.section === "operator" && rows.length !== 1) throw new Error("Operator detail did not resolve exactly one subject. Use an exact subject_ref from the operators query.");
  if (unavailable) context.unavailable_reason = unavailable;
  // Bound nested ledgers explicitly, retaining truncation paths. Full rows can
  // be inspected in local exports; no omitted detail is presented as complete.
  const page = rows.slice(query.offset, query.offset + query.limit).map(boundRow);
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
    body.context.follow_up = queryFollowUps(body);
    return sealEvidenceQueryResult(body);
  };
  let result = seal();
  while (new TextEncoder().encode(JSON.stringify(result)).length > EVIDENCE_QUERY_MAX_BYTES && page.length > 1) { page.pop(); result = seal(); }
  return validateEvidenceQueryResult(result);
}

// Suggestions are navigation over the final returned page, never analysis or
// natural-language parsing. Recompute after byte-budget trimming so cursors and
// subjects cannot refer to rows omitted from the result.
export function queryFollowUps(result, selectedRow = null) {
  const { query, coverage } = result;
  if (selectedRow && !result.rows.includes(selectedRow)) throw new Error("Selected evidence is not part of this query result.");
  const rows = selectedRow ? [selectedRow] : result.rows;
  const base = { limit: query.limit, ...(query.target ? { target: query.target } : {}) };
  const actions = [];
  const add = (id, ko, en, next) => actions.push({ id, prompt: {
    ko: selectedRow ? ko.replace("표시된 첫", "선택한").replace("첫 backend", "선택한 backend") : ko,
    en: selectedRow ? en.replace("first displayed", "selected").replace("first backend", "selected backend") : en,
  }, query: normalizeEvidenceQuery(next) });
  const usable = ref => typeof ref === "string" && ref.length > 0 && ref.length <= 512;
  if (coverage.next_offset !== null) add("next_page", "같은 조건으로 다음 페이지를 보여줘.", "Show the next page with the same filters.", { ...query, offset: coverage.next_offset });
  if (query.search || query.subject_ref || query.source_index !== undefined || query.offset) {
    add("reset", "선택과 검색을 풀고 처음부터 보여줘.", "Clear the selection and search, and start at the first page.", { ...base, section: query.section === "operator" ? "operators" : query.section, ...(query.profile_ids.length ? { profile_ids: query.profile_ids } : {}), ...(query.weight_view ? { weight_view: query.weight_view } : {}) });
  }
  if (query.section === "operator") {
    for (const [direction, ko, en] of [["incoming", "이 연산으로 들어오는 첫 연결을 설명해줘.", "Explain the first incoming relationship."], ["outgoing", "이 연산에서 나가는 첫 연결을 설명해줘.", "Explain the first outgoing relationship."]]) {
      const edge = rows[0]?.details.navigation?.[direction]?.find(edge => usable(edge.subject_ref));
      if (edge) add(direction, ko, en, { ...base, section: "operator", subject_ref: edge.subject_ref });
    }
  } else if (query.section === "weights") {
    const ref = rows[0]?.subject_ref;
    if (usable(ref)) for (const [view, ko, en] of [["distribution", "이 텐서의 분포를 보여줘.", "Show this tensor's distribution."], ["channels", "이 텐서의 채널별 통계를 보여줘.", "Show this tensor's channel statistics."], ["spectrum", "이 텐서의 특이값 스펙트럼을 보여줘.", "Show this tensor's singular spectrum."]]) {
      if (view !== (query.weight_view || "distribution") || !query.subject_ref) add(`weight_${view}`, ko, en, { ...base, section: "weights", weight_view: view, subject_ref: ref });
    }
  } else if (query.section === "profiles") {
    const profile = rows.find(row => row.kind === "backend_profile");
    if (profile) add("profile_eligibility", "첫 backend profile의 연산별 정적 지원 조건을 보여줘.", "Show static operation eligibility for the first backend profile.", { ...base, section: "placement", profile_ids: [profile.subject_ref] });
  } else {
    const row = rows.find(row => ["operation", "serialized_storage", "fusion_evidence", "static_placement", "workload_review"].includes(row.kind) && usable(row.subject_ref));
    if (row) add("inspect_subject", "표시된 첫 레이어 또는 저장 텐서를 자세히 설명해줘.", "Explain the first displayed layer or storage tensor in detail.", { ...base, section: "operator", subject_ref: row.subject_ref });
    const finding = rows.find(row => ["finding", "investigation"].includes(row.kind) && usable(row.subject_ref));
    if (finding && !query.subject_ref) add("inspect_finding", "표시된 첫 finding의 근거와 한계를 설명해줘.", "Explain the evidence and limits of the first displayed finding.", { ...base, section: "findings", subject_ref: finding.subject_ref });
  }
  return { tool: "deepbom_query_file", expected_sha256: result.artifact.sha256,
    instruction: "Reuse the authorized attachment and analysis depth. These are optional requests, not executed results. Resolve ambiguous references with the user; do not substitute display row numbers for subject_ref. Read the current evidence before repeating a query.", actions };
}

function makeRow(ref, kind, title, details) {
  // Search full source names/paths before bounding the returned page. Otherwise
  // a valid match past a display limit disappears from search and pagination.
  return { subject_ref: String(ref), kind, title: String(title || ref), details };
}
function boundRow(row) {
  const bounded = boundDetails(row.details);
  if (row.title.length > 512) bounded.paths.push("title");
  return { ...row, title: row.title.slice(0, 512), details: bounded.value, detail_truncations: bounded.paths };
}
function pick(value, keys) { return value ? Object.fromEntries(keys.filter(key => value[key] !== undefined).map(key => [key, value[key]])) : null; }
function boundDetails(value) {
  const paths = [];
  const visit = (item, path, depth) => {
    if (depth > 8) { paths.push(path); return null; }
    if (typeof item === "string" && item.length > 1200) { paths.push(path); return item.slice(0, 1200); }
    if (Array.isArray(item)) { const cap = /\/histogram\/(counts|edges)$/.test(path) ? 128 : 32; if (item.length > cap) paths.push(path); return item.slice(0, cap).map((child, index) => visit(child, `${path}/${index}`, depth + 1)); }
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
