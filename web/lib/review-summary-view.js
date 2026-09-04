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
  coverage.textContent = `${graphCopy(summary)} ${summary.coverage.assessed} assessed, ${summary.coverage.partial} partial, ${summary.coverage.needs_external_evidence} need external evidence. `
    + targetCopy(summary);

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
