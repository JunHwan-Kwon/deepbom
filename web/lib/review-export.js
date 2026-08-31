import { normalizeEvidenceExplanation } from "./evidence-why-drawer.js";
import { buildFindingsRegister } from "./report-findings.js";

export const REVIEW_STATE_SCHEMA = "deepbom.review_state.v1";

export function buildReviewState({ analysis, cursor, graphView, workspace, auditTab, acceleratorProfileId, comparison, runtimeEvidence, externalOverlay } = {}) {
  return {
    schema: REVIEW_STATE_SCHEMA,
    artifact_sha256: String(analysis?.model_sha256 || ""),
    format: String(analysis?.format || "").toLowerCase(),
    analyzer_version: String(analysis?.analyzer_version || ""),
    rulepack_version: String(analysis?.rulepack_version || ""),
    generated_at: analysis?._reportGeneratedAt || null,
    workspace: workspace || null,
    audit_tab: auditTab || null,
    accelerator_profile_id: acceleratorProfileId || null,
    evidence_cursor: cursor || null,
    graph_view: graphView || null,
    comparison: comparison || null,
    runtime_evidence: runtimeSummary(runtimeEvidence),
    external_overlay: externalOverlay ? {
      schema: externalOverlay.schema,
      source: externalOverlay.source,
      node_count: externalOverlay.nodes?.length || 0,
      edge_count: externalOverlay.edges?.length || 0,
    } : null,
    interpretation_boundary: "Review state contains navigation and evidence references, not model payload bytes, tensor values, or proof of runtime behavior beyond imported evidence.",
  };
}

export function buildSelfContainedReviewHtml({ analysis, graphSvg, reviewState, runtimeEvidence } = {}) {
  const findings = buildFindingsRegister(analysis, { runtimeEvidence }).map(reviewFinding);
  const explanations = findings.map((finding) => normalizeEvidenceExplanation({
    title: finding.title,
    value: `${finding.severity || "unrated"} / ${finding.category || "uncategorized"}`,
    evidence_class: finding.confidence || "NOT_ASSESSABLE",
    method: "Finding assembled from the cited evidence rows and action policy.",
    source_pointers: (finding.evidence || []).map((row) => row.source).filter(Boolean),
    limitations: finding.impact || "No limitation statement supplied.",
    report_pointer: `findings[id=${finding.id}]`,
  }));
  const runtimeRows = runtimeReconciliationRows(analysis, runtimeEvidence);
  const safeSvg = String(graphSvg || "").replace(/^<\?xml[^>]+>\s*/, "");
  const graphDataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(safeSvg)}`;
  const stateJson = escapeHtml(JSON.stringify(reviewState, null, 2));
  const findingHtml = findings.map((finding, index) => `<details${index === 0 ? " open" : ""}><summary><strong>${escapeHtml(finding.title)}</strong><span>${escapeHtml(`${finding.severity || ""} / ${finding.category || ""}`)}</span></summary><p>${escapeHtml(finding.impact || "")}</p><table><tbody>${(finding.evidence || []).map((row) => `<tr><th>${escapeHtml(row.source || "Evidence")}</th><td>${escapeHtml(row.text || "")}</td></tr>`).join("")}</tbody></table>${finding.actions?.length ? `<h4>Recommended action</h4><ul>${finding.actions.map((action) => `<li>${escapeHtml(action)}</li>`).join("")}</ul>` : ""}<h4>Evidence explanation</h4><pre>${escapeHtml(JSON.stringify(explanations[index], null, 2))}</pre></details>`).join("");
  const runtimeHtml = runtimeRows.length
    ? `<table><thead><tr><th>Source op</th><th>Runtime node</th><th>Provider</th><th>Mapping</th></tr></thead><tbody>${runtimeRows.map((row) => `<tr><td>${escapeHtml(row.source)}</td><td>${escapeHtml(row.runtime)}</td><td>${escapeHtml(row.provider)}</td><td>${escapeHtml(row.mapping)}</td></tr>`).join("")}</tbody></table>`
    : "<p>No explicit runtime-to-source mapping is bound.</p>";
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline';"><title>${escapeHtml(analysis?.filename || "artifact")} | DEEPBOM review</title><style>${reviewCss()}</style></head><body>
<header><p>DEEPBOM READ-ONLY REVIEW</p><h1>${escapeHtml(analysis?.filename || "Deployment artifact")}</h1><dl><div><dt>Artifact SHA-256</dt><dd>${escapeHtml(analysis?.model_sha256 || "unbound")}</dd></div><div><dt>Format</dt><dd>${escapeHtml(analysis?.format || "unknown")}</dd></div><div><dt>Evidence boundary</dt><dd>Artifact evidence plus explicitly imported, hash-bound runtime evidence only.</dd></div></dl></header>
<main><section><h2>Graph</h2><div class="graph"><img src="${escapeHtml(graphDataUrl)}" alt="Static artifact graph"></div></section><section><h2>Findings and evidence</h2>${findingHtml || "<p>No findings were emitted.</p>"}</section><section><h2>Source to runtime reconciliation</h2>${runtimeHtml}<p class="boundary">Runtime nodes are linked only through explicit original-op identity. Name similarity is not used.</p></section><section><h2>Review state</h2><pre>${stateJson}</pre></section></main>
<footer>Generated locally. This file contains no original model bytes or raw tensor values. It is a review aid, not a task-accuracy, clinical-validity, or release-readiness certificate.</footer></body></html>\n`;
}

function reviewFinding(finding) {
  const pointers = Array.isArray(finding?.evidence_json_pointers) ? finding.evidence_json_pointers : [];
  return {
    id: finding?.finding_id || "unidentified",
    title: finding?.title || finding?.observation || "Untitled finding",
    severity: String(finding?.technical_priority || "unrated").toLowerCase(),
    category: finding?.category || finding?.finding_class || "uncategorized",
    confidence: finding?.evidence_class || "NOT_ASSESSABLE",
    evidence: pointers.map((pointer) => ({ source: pointer, text: finding?.observation || "See the cited evidence field." })),
    impact: finding?.interpretation || finding?.limitations || "No interpretation supplied.",
    actions: finding?.recommendation ? [finding.recommendation] : [],
  };
}

function runtimeSummary(value) {
  const runtime = value?.runtimeAssignmentEvidence || value?.runtime_assignment || value;
  if (!runtime) return null;
  const assignments = Array.isArray(runtime.assignments) ? runtime.assignments : [];
  return {
    artifact_sha256: runtime.model_sha256 || runtime.artifact_sha256 || null,
    binary_sha256: runtime.runtime_binary_sha256 || runtime.binary_sha256 || null,
    assignment_count: assignments.length,
    explicitly_mapped_source_op_count: assignments.filter((row) => Number.isSafeInteger(Number(row.op_index))).length,
  };
}

function runtimeReconciliationRows(analysis, value) {
  const runtime = value?.runtimeAssignmentEvidence || value?.runtime_assignment || value;
  const assignments = Array.isArray(runtime?.assignments) ? runtime.assignments : [];
  const byIndex = new Map((analysis?.ops || []).map((op) => [Number(op.index), op]));
  return assignments.map((row) => {
    const opIndex = Number(row.op_index);
    const source = Number.isSafeInteger(opIndex) && byIndex.has(opIndex) ? `#${opIndex} ${byIndex.get(opIndex).name}` : "Unmapped";
    const runtimeIdentity = row.runtime_node_name || (row.runtime_node_index == null ? null : `runtime #${row.runtime_node_index}`) || "not exposed";
    return { source, runtime: runtimeIdentity, provider: row.provider || "not exposed", mapping: row.mapping_method || (source === "Unmapped" ? "UNMAPPED" : "explicit op_index") };
  });
}

function reviewCss() {
  return `:root{color-scheme:light;--bg:#f4f1e9;--surface:#fffdf8;--ink:#20231f;--muted:#636961;--line:#cec9bd;--accent:#246b61}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,Arial,sans-serif}header,main,footer{width:min(1320px,calc(100% - 32px));margin:auto}header{padding:28px 0 18px}header>p{color:var(--accent);font-size:12px;font-weight:700}h1{margin:4px 0 16px;font-size:clamp(26px,4vw,44px)}h2{font-size:20px}dl{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-block:1px solid var(--line)}dl div{padding:10px 12px;border-inline-end:1px solid var(--line)}dt{color:var(--muted);font-size:11px;text-transform:uppercase}dd{margin:4px 0;overflow-wrap:anywhere}main{display:grid;gap:14px;padding:0 0 24px}section,details{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px}details{margin:8px 0}summary{display:flex;justify-content:space-between;gap:12px;cursor:pointer}.graph{overflow:auto;max-height:72vh;border:1px solid var(--line);background:#f5f2ea}.graph img{display:block;max-width:none}table{width:100%;border-collapse:collapse}th,td{padding:8px;text-align:left;vertical-align:top;border-block-end:1px solid var(--line)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#efede6;padding:12px;border-radius:4px}.boundary,footer{color:var(--muted)}footer{padding:16px 0 28px;border-block-start:1px solid var(--line)}@media(max-width:700px){header,main,footer{width:min(100% - 16px,1320px)}dl{grid-template-columns:1fr}dl div{border-inline-end:0;border-block-end:1px solid var(--line)}summary{display:block}.graph{max-height:55vh}}`;
}

function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]); }
