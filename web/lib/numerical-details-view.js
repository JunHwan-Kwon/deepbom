import { escapeXml as esc, countLabel, numberLabel, histogramSvg } from "./weight-visuals.js";

const phrase = value => String(value).replaceAll("_", " ");
const metric = (label, value) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
const fraction = value => value?.value == null ? "Not assessed" : `${numberLabel(value.value * 100)}%`;

export function distributionDetailsView(profile) {
  if (!profile) return "";
  const q = profile.quantiles;
  return `<details class="weight-method" data-distribution-details><summary>Distribution detail · signs and percentile bounds</summary>
    <dl class="weight-stat-grid">${metric("Negative values", countLabel(profile.sign_counts.negative))}${metric("Exact zeros", countLabel(profile.sign_counts.zero))}${metric("Positive values", countLabel(profile.sign_counts.positive))}${metric("Finite fraction", fraction(profile.finite_fraction))}${metric("Constant finite value", numberLabel(profile.constant_finite_value))}${metric("All values zero", profile.all_values_zero === null ? "Not assessed" : profile.all_values_zero ? "Yes" : "No")}</dl>
    <p>Percentiles use finite values only. An interval contains the nearest-rank value; it is not an interpolated estimate. Exact endpoints and zero mass are shown when resolved.</p>
    ${q.status === "assessed" ? `<div class="weight-table-scroll"><table><thead><tr><th>Percentile</th><th>Finite rank</th><th>Value or interval</th><th>Precision</th></tr></thead><tbody>${q.intervals.map(row => `<tr><td>P${row.percent}</td><td>${countLabel(row.rank)}</td><td><code>${row.exact ? esc(String(row.lower)) : esc(`[${row.lower}, ${row.upper}${row.upper_closed ? "]" : ")"}`)}</code></td><td>${row.exact ? "Exact value" : "Histogram bound"}</td></tr>`).join("")}</tbody></table></div>` : `<p>${esc(phrase(q.reason))}</p>`}
  </details>`;
}

export const FLOW_PAGE_SIZE = 24;
export function activationFlowView(detail, page = 0) {
  const a = detail?.activation;
  if (!a) return "";
  const rows = a.flow.slice(page * FLOW_PAGE_SIZE, (page + 1) * FLOW_PAGE_SIZE), c = a.coverage;
  return `<h3>Activation capture coverage</h3><p>${c.captured_count} / ${c.requested_count} requested values captured · ${c.missing_count} missing · ${c.not_requested_value_refs.length} not requested · ${c.unmapped_capture_count} unmapped captures.</p>
    <p>Run <strong>${esc(a.run.id)}</strong> · ${esc(a.run.runtime.name)} ${esc(a.run.runtime.version)} · ${esc(a.run.started_at)}. Inputs, stored values and missing captures remain distinct. Port order describes the serialized graph, not runtime execution order.</p>
    <div class="weight-table-scroll"><table><thead><tr><th>Operation</th><th>Input ports</th><th>Output ports</th></tr></thead><tbody>${rows.map(op => `<tr><td>${esc(op.name)}<br><code>${esc(op.operation_ref)}</code></td>${["input", "output"].map(direction => `<td>${op.ports.filter(p => p.direction === direction).map(p => `<div>${p.tensor_ref ? `<button type="button" data-flow-tensor="${esc(p.tensor_ref)}" title="${esc(p.value_ref)}">${p.position}: ${esc(phrase(p.capture_status))}</button>` : `<span title="${esc(p.value_ref)}">${p.position}: ${esc(phrase(p.capture_status))}</span>`}</div>`).join("") || "—"}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <div class="weight-pagination"><button type="button" data-flow-page="prev" ${page === 0 ? "disabled" : ""}>Previous operations</button><span>${rows.length ? `${page * FLOW_PAGE_SIZE + 1}–${page * FLOW_PAGE_SIZE + rows.length}` : "0"} / ${a.flow.length}</span><button type="button" data-flow-page="next" ${(page + 1) * FLOW_PAGE_SIZE >= a.flow.length ? "disabled" : ""}>Next operations</button></div>`;
}

export function activationComparisonView(detail, row, baseline, colors) {
  const comparison = detail?.activation_comparison;
  if (!comparison) return "";
  const result = comparison.tensors.find(t => t.value_ref === row.value_ref);
  const before = baseline?.tensors.find(t => t.id === result?.baseline_ref);
  const labels = { same_inputs: "Input values", same_runtime: "Runtime", same_collector: "Collector", same_configuration: "Configuration", same_instrumentation: "Instrumentation", same_probe: "Probe description", same_entry_region: "Entry region" };
  return `<section class="weight-method" data-activation-comparison><h3>Reference capture comparison</h3><p>${esc(comparison.baseline_run_id)} → ${esc(comparison.candidate_run_id)}. ${comparison.coverage.compared_count} / ${comparison.coverage.union_requested_count} requested values compared.</p>
    <dl class="weight-stat-grid">${Object.entries(labels).map(([key, label]) => metric(label, comparison[key] ? "Same recorded identity" : "Different")).join("")}</dl>
    <p>Observed distribution changes are descriptive. Changes in inputs, runtime or capture settings can affect them; they do not establish a training improvement, paired-value error or causal effect.</p>
    ${result?.status === "assessed" ? `<dl class="weight-stat-grid">${metric("Histogram total variation", numberLabel(result.histogram_total_variation.value))}${metric("Mean change", numberLabel(result.mean_delta))}${metric("RMS change", numberLabel(result.rms_delta))}${metric("Standard deviation change", numberLabel(result.population_stddev_delta))}</dl><p>Total variation uses normalized finite counts and ranges from 0 to 1. Differences inside a bin remain unresolved; 0 does not mean identical values. Changes are current minus reference.</p><h4>Reference distribution</h4>${histogramSvg(before, colors) || ""}` : `<p>Not compared: ${esc(phrase(result?.reason || (row.capture_status === "input" ? "input_identity_reported_separately" : "value_not_requested_in_either_capture")))}</p>`}
  </section>`;
}
