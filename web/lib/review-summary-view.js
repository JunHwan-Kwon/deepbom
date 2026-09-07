const COUNT_FIELDS = Object.freeze([
  ["artifact_defect_count", "Artifact defects", "defect"],
  ["caution_count", "Cautions", "caution"],
  ["evidence_needed_count", "Evidence needed", "evidence"],
]);

export function renderReviewSummary(root, summary) {
  if (!root) return;
  if (!summary) {
    root.hidden = true;
    root.replaceChildren();
    return;
  }

  const doc = root.ownerDocument;
  const heading = doc.createElement("header");
  heading.className = "review-summary-head";
  const titleWrap = doc.createElement("div");
  const eyebrow = doc.createElement("span");
  eyebrow.textContent = "Static review result";
  const title = doc.createElement("h2");
  title.textContent = summary.verdict.artifact_defect_count
    ? "Artifact defects require review"
    : "No artifact defect was observed";
  const scope = doc.createElement("p");
  scope.textContent = summary.verdict.scope;
  titleWrap.append(eyebrow, title, scope);
  const identity = doc.createElement("code");
  identity.textContent = shortDigest(summary.artifact.sha256);
  identity.title = summary.artifact.sha256;
  heading.append(titleWrap, identity);

  const counts = doc.createElement("div");
  counts.className = "review-summary-counts";
  counts.setAttribute("aria-label", "Review verdict counts");
  for (const [field, label, tone] of COUNT_FIELDS) {
    const item = doc.createElement("div");
    item.dataset.tone = tone;
    const value = doc.createElement("strong");
    value.textContent = String(summary.verdict[field]);
    const caption = doc.createElement("span");
    caption.textContent = label;
    item.append(value, caption);
    counts.append(item);
  }

  const coverage = doc.createElement("p");
  coverage.className = "review-summary-coverage";
  coverage.textContent = `${graphCopy(summary)} ${coverageCopy(summary)} ${targetCopy(summary)}`.replace(/\s+/g, " ").trim();

  const actions = doc.createElement("div");
  actions.className = "review-summary-actions";
  actions.append(
    actionButton(doc, "Review findings", "findings"),
    actionButton(doc, "Open report", "report"),
    actionButton(doc, "Why these counts?", "explain"),
  );

  root.replaceChildren(heading, counts, coverage, actions);
  root.hidden = false;
}

export function bindReviewSummaryActions(root, handlers = {}) {
  root?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-review-summary-action]");
    if (!button || !root.contains(button)) return;
    handlers[button.dataset.reviewSummaryAction]?.();
  });
}

function actionButton(doc, label, action) {
  const button = doc.createElement("button");
  button.type = "button";
  button.className = "secondary-action";
  button.dataset.reviewSummaryAction = action;
  button.textContent = label;
  return button;
}

function targetCopy(summary) {
  const target = summary.target?.label || summary.target?.id;
  const binding = summary.target?.binding_source;
  if (!target) return "No target-specific cost profile was applied.";
  return `Target: ${target}${binding ? ` (${binding})` : ""}.`;
}

// 셋은 evidence capability 의 상태 분포다. 세는 대상을 문장에 남겨 두어야
// "10 assessed, 2 partial, 1 need" 처럼 잘려 보이지 않는다.
function coverageCopy(summary) {
  const coverage = summary.coverage || {};
  const assessed = Number(coverage.assessed || 0);
  const partial = Number(coverage.partial || 0);
  const external = Number(coverage.needs_external_evidence || 0);
  const total = assessed + partial + external;
  if (!total) return "";
  const parts = [`${assessed} assessed`];
  if (partial) parts.push(`${partial} partial`);
  parts.push(`${external} ${external === 1 ? "needs" : "need"} external evidence`);
  return `Evidence capabilities: ${total} in scope — ${parts.join(", ")}.`;
}

function graphCopy(summary) {
  const graph = summary.graph || {};
  if (graph.operator_count == null && graph.tensor_count == null) return "";
  const macs = graph.total_macs == null ? "MAC total not assessable" : `${Number(graph.total_macs).toLocaleString("en-US")} MACs`;
  return `Graph: ${graph.operator_count ?? "unknown"} operators, ${graph.tensor_count ?? "unknown"} tensors, ${macs}.`;
}

function shortDigest(value) {
  const digest = String(value || "");
  return digest.length > 16 ? `sha256:${digest.slice(0, 12)}...${digest.slice(-4)}` : digest;
}
