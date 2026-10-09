// Explorer renders the same hash-bound evidence and charts as Weight workspace.
import { escapeXml as esc, numberLabel, countLabel, zeroLabel, histogramSvg, intervalLabel } from "./weight-visuals.js";
import { describeDistribution } from "./numerical-ir/distribution.js";
import { distributionDetailsView } from "./numerical-details-view.js";
import { renderWeightAnalysisView } from "./weight-analysis-view.js";

const colors = {surface:"var(--surface)",ink:"var(--ink)",muted:"var(--muted)",line:"var(--line)",accent:"var(--accent)",blue:"var(--blue)"};
const phrase = value => String(value ?? "").replaceAll("_", " ");
const metric = (label, value) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;

export function renderWeightHistogram({ tensor, advanced }, identity) {
  const card = document.createElement("article");
  card.className = "whist-card weight-explorer-evidence";
  const s = tensor.statistics;
  card.innerHTML = `<h5>${esc(tensor.name || tensor.native_locator)}</h5>
    <p>${esc(tensor.dtype)} · ${esc(tensor.shape?.join(" × ") ?? "Unknown shape")} · ${esc(phrase(tensor.representation || "not_assessed"))}</p>
    <p class="weight-chart-note">${esc(tensor.id)} · ${esc(tensor.storage_ref || "Unlinked storage")} · ${esc(identity.operation_ref)}</p>`;
  if (s) {
    card.insertAdjacentHTML("beforeend", `<dl class="weight-stat-grid">${[
      ["Stored elements",countLabel(s.value_count)], ["Minimum",numberLabel(s.minimum)],
      ["Maximum",numberLabel(s.maximum)], ["Mean",numberLabel(s.mean)],
      ["Population std dev",numberLabel(s.population_stddev)], ["Exact zeros / finite",zeroLabel(s)],
    ].map(([k,v])=>metric(k,v)).join("")}</dl>
    ${histogramSvg(tensor, colors) || "<p>Histogram not assessed.</p>"}
    ${distributionDetailsView(describeDistribution(s))}`);
    const readout = document.createElement("p");
    readout.className = "weight-chart-note";
    readout.setAttribute("role", "status");
    readout.textContent = "Select a histogram bin to inspect its exact count and interval.";
    card.append(readout);
    const inspect = event => {
      const target = event.target.closest?.("[data-bin]");
      if (!target || event.type === "keydown" && !["Enter", " "].includes(event.key)) return;
      if (event.type === "keydown") event.preventDefault();
      const bin = Number(target.dataset.bin), h = s.histogram;
      if (!h || !Number.isSafeInteger(bin) || bin < 0 || bin >= h.counts.length) return;
      readout.textContent = `${intervalLabel(h, bin)}: ${countLabel(h.counts[bin])} stored values`;
    };
    card.addEventListener("click", inspect);
    card.addEventListener("keydown", inspect);
  } else {
    card.insertAdjacentHTML("beforeend", `<p>Not assessed: ${esc(phrase(tensor.reason))}</p>`);
  }
  const detail = document.createElement("details");
  detail.className = "weight-method";
  detail.innerHTML = "<summary>Channels, sparsity and singular spectrum</summary>";
  // These common projections already expose feature budgets, axes and value basis.
  for (const [view, label] of [["channels","Channels"],["sparsity","Sparsity"],["spectrum","Singular spectrum"],["similarity","Similarity"],["quantization","Quantization"]]) {
    const section = document.createElement("details");
    section.innerHTML = `<summary>${label}</summary>`;
    section.addEventListener("toggle", () => {
      if (!section.open || section.dataset.rendered) return;
      const result = renderWeightAnalysisView(view, advanced, null, colors);
      section.insertAdjacentHTML("beforeend", result.html);
      // Channel selection belongs to the full Weight workspace; this is a read-only projection.
      for (const button of section.querySelectorAll("button")) button.replaceWith(document.createTextNode(button.textContent));
      for (const button of section.querySelectorAll('[role="button"]')) { button.removeAttribute("role"); button.removeAttribute("tabindex"); }
      section.dataset.rendered = "true";
    });
    detail.append(section);
  }
  card.append(detail);
  card.insertAdjacentHTML("beforeend", `<details class="weight-method"><summary>Evidence identity and scope</summary>
    <p>Weight IR: <code>${esc(identity.weight_ir_sha256)}</code></p>
    <p>Weight analysis: <code>${esc(identity.weight_analysis_sha256 || "Not requested")}</code></p>
    <p>Model IR: <code>${esc(identity.source.model_ir_sha256)}</code></p>
    <p>Artifact: <code>${esc(identity.source.artifact_sha256)}</code></p>
    <p>Stored scalar statistics and explicitly dequantized advanced results have separate representations. Counts describe stored values, not inferred trainable parameters. Numerical rank is computed only by the common singular-spectrum analysis. Missing evidence is not zero.</p></details>`);
  return card;
}
