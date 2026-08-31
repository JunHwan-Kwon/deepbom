const EVIDENCE_PREVIEW_LIMIT = 420;
const EVIDENCE_SEGMENT_LIMIT = 3;

function evidenceTextNode(value) {
  const text = String(value ?? "");
  if (text.length <= EVIDENCE_PREVIEW_LIMIT) return document.createTextNode(text);

  const segments = text.split(/;\s+/).filter(Boolean);
  let preview = "";
  if (segments.length > EVIDENCE_SEGMENT_LIMIT) {
    preview = `${segments.slice(0, EVIDENCE_SEGMENT_LIMIT).join("; ")}; +${segments.length - EVIDENCE_SEGMENT_LIMIT} more evidence segments`;
  } else {
    const boundary = text.lastIndexOf(" ", EVIDENCE_PREVIEW_LIMIT);
    preview = `${text.slice(0, boundary > 280 ? boundary : EVIDENCE_PREVIEW_LIMIT).trim()}...`;
  }

  const fragment = document.createDocumentFragment();
  const summary = document.createElement("span");
  summary.className = "finding-evidence-preview";
  summary.textContent = preview;
  const detail = document.createElement("details");
  detail.className = "finding-evidence-detail";
  const control = document.createElement("summary");
  control.textContent = "View complete evidence";
  const full = document.createElement("p");
  full.textContent = text;
  detail.append(control, full);
  fragment.append(summary, detail);
  return fragment;
}

function appendEvidenceRow(table, source, text) {
  const tr = document.createElement("tr");
  const tdSrc = document.createElement("td");
  tdSrc.className = "ev-source";
  tdSrc.textContent = source;
  const tdText = document.createElement("td");
  tdText.className = "ev-text";
  tdText.append(evidenceTextNode(text));
  tr.append(tdSrc, tdText);
  table.append(tr);
}

export function renderFindings(findingsBody, analysis, { onSelectEvidence = null, onExplain = null } = {}) {
  if (!findingsBody) return;

  const findings = Array.isArray(analysis.findings) ? analysis.findings : [];
  if (!findings.length) {
    const p = document.createElement("p");
    p.className = "findings-empty";
    p.textContent = "No significant findings for this model.";
    findingsBody.replaceChildren(p);
    return;
  }

  const SEV_LABEL = { high: "HIGH", medium: "MED", low: "LOW", informational: "INFO" };
  const CAT_LABEL = {
    delegation: "Delegation",
    quant: "Quantization",
    memory: "Memory",
    latency: "Latency",
    input: "Input Contract",
  };

  const compact = typeof window !== "undefined" && window.matchMedia?.("(max-width: 720px)").matches;
  const cards = findings.map((f, index) => {
    const card = document.createElement("details");
    card.className = `finding-card finding-${f.severity} finding-cat-${f.category}`;
    card.dataset.findingId = f.id;
    card.open = !compact || index === 0;

    // Header row: severity badge + category + title
    const hdr = document.createElement("summary");
    hdr.className = "finding-card-header";
    const sevBadge = document.createElement("span");
    sevBadge.className = `finding-sev finding-sev-${f.severity}`;
    sevBadge.textContent = SEV_LABEL[f.severity] || f.severity.toUpperCase();
    const catBadge = document.createElement("span");
    catBadge.className = "finding-cat-badge";
    catBadge.textContent = CAT_LABEL[f.category] ?? f.category;
    const title = document.createElement("h4");
    title.className = "finding-title";
    title.textContent = f.title;
    hdr.append(sevBadge, catBadge, title);
    hdr.addEventListener("click", () => onSelectEvidence?.({
      finding_id: f.id,
      report_anchor: `finding:${f.id}`,
    }));

    // Evidence table
    const evTable = document.createElement("table");
    evTable.className = "finding-evidence";
    for (const ev of f.evidence) {
      appendEvidenceRow(evTable, ev.source, ev.text);
    }

    // Impact
    const impactRow = document.createElement("p");
    impactRow.className = "finding-impact";
    const impactLabel = document.createElement("strong");
    impactLabel.textContent = "Possible effect: ";
    impactRow.append(impactLabel, document.createTextNode(f.impact));

    // Actions
    const actionList = document.createElement("ul");
    actionList.className = "finding-actions";
    for (const action of f.actions) {
      const li = document.createElement("li");
      li.textContent = action;
      actionList.append(li);
    }

    const references = findingReferences(f, analysis);
    const evidenceActions = document.createElement("div");
    evidenceActions.className = "finding-evidence-actions";
    for (const opIndex of references.opIndices.slice(0, 4)) {
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "secondary-action finding-evidence-link";
      inspect.textContent = `Inspect op #${opIndex}`;
      inspect.addEventListener("click", () => onSelectEvidence?.({
        finding_id: f.id,
        op_index: opIndex,
        report_anchor: `finding:${f.id}`,
      }));
      evidenceActions.append(inspect);
    }
    for (const tensorIndex of references.tensorIndices.slice(0, 3)) {
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "secondary-action finding-evidence-link";
      inspect.textContent = `Inspect tensor T${tensorIndex}`;
      inspect.addEventListener("click", () => onSelectEvidence?.({
        finding_id: f.id,
        tensor_index: tensorIndex,
        report_anchor: `finding:${f.id}`,
      }));
      evidenceActions.append(inspect);
    }
    if (references.opIndices.length > 4 || references.tensorIndices.length > 3) {
      const remainder = document.createElement("span");
      remainder.className = "finding-evidence-remainder";
      remainder.textContent = `+${Math.max(0, references.opIndices.length - 4) + Math.max(0, references.tensorIndices.length - 3)} linked items`;
      evidenceActions.append(remainder);
    }
    if (typeof onExplain === "function") {
      const why = document.createElement("button");
      why.type = "button";
      why.className = "secondary-action finding-evidence-link";
      why.textContent = "Why?";
      why.addEventListener("click", () => onExplain({
        title: f.title,
        value: `${f.severity} / ${f.category}`,
        evidence_class: f.confidence || "NOT_ASSESSED",
        method: "Finding assembled from the cited evidence rows and action policy.",
        source_pointers: (f.evidence || []).map((row) => row.source).filter(Boolean),
        limitations: f.impact || "No limitation statement was supplied.",
        report_pointer: `findings[id=${f.id}]`,
      }));
      evidenceActions.append(why);
    }

    // Footer: confidence level
    const footer = document.createElement("div");
    footer.className = "finding-footer";
    const confBadge = document.createElement("span");
    confBadge.className = `finding-confidence finding-conf-${f.confidence.replace(/\+/g, "-")}`;
    confBadge.textContent = f.confidence === "static" ? "Static only"
      : f.confidence === "static+wasm" ? "Static + WASM runtime"
      : f.confidence === "static+device" ? "Static + device benchmark"
      : f.confidence;
    footer.append(confBadge);

    const content = document.createElement("div");
    content.className = "finding-card-content";
    content.append(evTable, impactRow, actionList, evidenceActions, footer);
    card.append(hdr, content);
    return card;
  });

  // Summary line
  const summary = document.createElement("p");
  summary.className = "findings-summary";
  const highCount = findings.filter((item) => item.severity === "high").length;
  const medCount = findings.filter((item) => item.severity === "medium").length;
  const lowCount = findings.filter((item) => item.severity === "low").length;
  const infoCount = findings.filter((item) => item.severity === "informational").length;
  summary.textContent = `${findings.length} findings - ${highCount} high / ${medCount} medium / ${lowCount} low / ${infoCount} informational`;

  const mobileActions = document.createElement("div");
  mobileActions.className = "findings-mobile-actions";
  const expandAll = document.createElement("button");
  expandAll.type = "button";
  expandAll.textContent = "Expand all";
  expandAll.addEventListener("click", () => cards.forEach((card) => { card.open = true; }));
  const collapseAll = document.createElement("button");
  collapseAll.type = "button";
  collapseAll.textContent = "Collapse all";
  collapseAll.addEventListener("click", () => cards.forEach((card) => { card.open = false; }));
  mobileActions.append(expandAll, collapseAll);

  findingsBody.replaceChildren(summary, mobileActions, ...cards);
}

export function findingReferences(finding, analysis) {
  const validOps = new Set((analysis?.ops || []).map((op) => Number(op.index)));
  const validTensors = new Set((analysis?.tensors || []).map((tensor) => Number(tensor?.index)).filter(Number.isSafeInteger));
  const opIndices = new Set();
  const tensorIndices = new Set();
  const addIndex = (target, valid, value) => {
    const index = Number(value);
    if (Number.isSafeInteger(index) && index >= 0 && valid.has(index)) target.add(index);
  };
  addIndex(opIndices, validOps, finding?.op_index);
  addIndex(tensorIndices, validTensors, finding?.tensor_index);
  for (const value of finding?.op_indices || []) addIndex(opIndices, validOps, value);
  for (const value of finding?.tensor_indices || []) addIndex(tensorIndices, validTensors, value);
  const evidenceText = (finding?.evidence || []).map((row) => `${row?.source || ""} ${row?.text || ""}`).join("\n");
  for (const match of evidenceText.matchAll(/(?:op(?:erator)?\s*)?#(\d+)/gi)) addIndex(opIndices, validOps, match[1]);
  for (const match of evidenceText.matchAll(/\bT(\d+)\b/g)) addIndex(tensorIndices, validTensors, match[1]);
  return {
    opIndices: [...opIndices].sort((left, right) => left - right),
    tensorIndices: [...tensorIndices].sort((left, right) => left - right),
  };
}

export function renderFindingsCalibration(findingsBody, cal) {
  if (!Number.isFinite(cal?.correction_factor)) return;
  if (!findingsBody) return;
  // Remove any existing calibration card
  findingsBody.querySelector(".finding-calibration")?.remove();

  const card = document.createElement("article");
  card.className = "finding-card finding-calibration finding-low finding-cat-latency";

  const hdr = document.createElement("div");
  hdr.className = "finding-card-header";
  const sevBadge = document.createElement("span");
  sevBadge.className = "finding-sev finding-sev-low";
  sevBadge.textContent = "INFO";
  const catBadge = document.createElement("span");
  catBadge.className = "finding-cat-badge";
  catBadge.textContent = "Latency";
  const title = document.createElement("h4");
  title.className = "finding-title";
  title.textContent = `Static vs WASM runtime: ×${cal.correction_factor.toFixed(2)} correction`;
  hdr.append(sevBadge, catBadge, title);

  const evTable = document.createElement("table");
  evTable.className = "finding-evidence";
  for (const [src, text] of [
    ["Static steady", `Roofline estimate: ${cal.static_estimate_ms.toFixed(1)} ms`],
    ["Static cold", `Steady + packing + setup: ${Number(cal.cold_start_static_estimate_ms ?? cal.static_estimate_ms).toFixed(1)} ms (${Number(cal.one_time_packing_ms || 0).toFixed(1)} ms packing; ${Number(cal.boundary_setup_ms || 0).toFixed(1)} ms unmeasured planning setup)`],
    ["WASM p50", `Measured: ${cal.measured_ms.toFixed(1)} ms`],
    ["Result", cal.interpretation],
  ]) {
    appendEvidenceRow(evTable, src, text);
  }

  const footer = document.createElement("div");
  footer.className = "finding-footer";
  const confBadge = document.createElement("span");
  confBadge.className = "finding-confidence finding-conf-static-wasm";
  confBadge.textContent = "Static + WASM runtime";
  footer.append(confBadge);

  card.append(hdr, evTable, footer);
  // Insert at top (after summary line)
  const summary = findingsBody.querySelector(".findings-summary");
  if (summary) summary.after(card);
  else findingsBody.prepend(card);
}
