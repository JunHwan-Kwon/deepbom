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

export function renderFindings(findingsBody, analysis) {
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
    content.append(evTable, impactRow, actionList, footer);
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
