import {
  buildCalibrationValidationLedger,
  representativeDatasetInterfaceFromAnalysis,
  validateCalibrationValidationLedger,
} from "./calibration-validation-ledger.js";
import { formatNumber, formatPercent } from "./format.js";

export function createCalibrationValidationController({
  input,
  selectButton,
  downloadButton,
  status,
  result,
  getArtifactSha256,
  getAnalysis,
  onResult = () => {},
  onDownload = () => {},
} = {}) {
  let ledger = null;
  let artifactSha256 = null;

  selectButton?.addEventListener("click", () => input?.click());
  input?.addEventListener("change", async () => {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    try {
      setStatus(status, "Validating capture...", "running");
      const expected = getArtifactSha256?.() || null;
      if (!expected) throw new Error("Run a static audit before importing representative dataset evidence.");
      const expectedInterface = representativeDatasetInterfaceFromAnalysis(getAnalysis?.());
      if (!expectedInterface) throw new Error("The static audit did not expose a complete external input/output interface for capture binding.");
      const capture = JSON.parse(await file.text());
      ledger = buildCalibrationValidationLedger(capture, { expectedArtifactSha256: expected, expectedInterface });
      validateCalibrationValidationLedger(ledger, capture, { expectedArtifactSha256: expected, expectedInterface });
      artifactSha256 = expected;
      renderCalibrationValidationLedger(result, ledger);
      downloadButton.disabled = false;
      setStatus(status, `Verified ${formatNumber(ledger.sample_count)} sample(s) against artifact ${expected.slice(0, 12)}...`, "ok");
      onResult(ledger);
    } catch (error) {
      ledger = null;
      artifactSha256 = null;
      downloadButton.disabled = true;
      renderEmpty(result, error?.message || String(error), true);
      setStatus(status, `Rejected: ${error?.message || error}`, "error");
      onResult(null);
    }
  });
  downloadButton?.addEventListener("click", () => {
    if (ledger) onDownload(ledger);
  });

  return {
    reset(nextArtifactSha256 = null) {
      if (ledger && nextArtifactSha256 && artifactSha256 === nextArtifactSha256) return;
      ledger = null;
      artifactSha256 = nextArtifactSha256;
      if (downloadButton) downloadButton.disabled = true;
      renderEmpty(result, nextArtifactSha256
        ? "Import a hash-bound representative dataset capture to calculate interface saturation, reference-output drift, and repeated-run nondeterminism."
        : "Run a static audit before importing a capture.");
      setStatus(status, nextArtifactSha256 ? "No capture imported" : "Waiting for an audited artifact", "idle");
    },
    getResult() { return ledger; },
  };
}

export function renderCalibrationValidationLedger(root, ledger) {
  if (!root) return;
  const endpoint = ledger.input_endpoint_saturation;
  const drift = ledger.reference_output_drift;
  const repeat = ledger.repeat_nondeterminism;
  const metrics = document.createElement("div");
  metrics.className = "calibration-validation-metrics";
  metrics.append(
    metric("Dataset samples", formatNumber(ledger.sample_count), `${formatNumber(ledger.captured_value_count)} captured tensor values`, "neutral"),
    metric("Input endpoints", endpoint.endpoint_ratio == null ? "N/A" : formatPercent(endpoint.endpoint_ratio), endpoint.status === "assessed" ? `${formatNumber(endpoint.endpoint_count)} / ${formatNumber(endpoint.assessed_value_count)} bounded-integer input values` : "No bounded-integer interface input was present.", endpoint.endpoint_count ? "warn" : "good"),
    metric("Reference drift", drift.maximum_absolute_difference == null ? "Not provided" : formatMetric(drift.maximum_absolute_difference), drift.status === "assessed" ? `${formatNumber(drift.changed_value_count)} / ${formatNumber(drift.compared_value_count)} values changed across ${formatNumber(drift.comparison_count)} comparison(s)` : "No same-contract reference outputs were supplied.", drift.changed_value_count ? "warn" : drift.status === "assessed" ? "good" : "neutral"),
    metric("Repeat variance", repeat.maximum_absolute_difference == null ? "Not assessed" : formatMetric(repeat.maximum_absolute_difference), repeat.status === "assessed" ? `${formatNumber(repeat.changed_value_count)} / ${formatNumber(repeat.compared_value_count)} values changed across ${formatNumber(repeat.comparison_count)} repeat comparison(s)` : "At least two runs per sample are required.", repeat.changed_value_count ? "risk" : repeat.status === "assessed" ? "good" : "neutral"),
  );

  const tableWrap = document.createElement("div");
  tableWrap.className = "calibration-validation-table-wrap";
  const table = document.createElement("table");
  table.className = "calibration-validation-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Sample", "Runs", "Input endpoints", "Reference delta", "Repeat delta"]) headRow.append(cell("th", label));
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const sample of ledger.samples) {
    const row = document.createElement("tr");
    const endpointRow = sample.input_endpoint_saturation;
    const referenceMax = finiteMax(sample.reference_comparisons.map((item) => item.maximum_absolute_difference));
    const repeatMax = finiteMax(sample.repeat_comparisons.map((item) => item.maximum_absolute_difference));
    row.append(
      cell("td", sample.sample_id),
      cell("td", String(sample.run_count)),
      cell("td", endpointRow.endpoint_ratio == null ? "N/A" : `${formatPercent(endpointRow.endpoint_ratio)} (${endpointRow.endpoint_count}/${endpointRow.assessed_value_count})`),
      cell("td", referenceMax == null ? "not provided" : formatMetric(referenceMax)),
      cell("td", repeatMax == null ? "not assessed" : formatMetric(repeatMax)),
    );
    body.append(row);
  }
  table.append(head, body);
  tableWrap.append(table);

  const provenance = document.createElement("div");
  provenance.className = "calibration-validation-provenance";
  provenance.textContent = `Dataset ${ledger.dataset.id}@${ledger.dataset.version} | manifest ${ledger.dataset.manifest_sha256.slice(0, 16)}... | runtime ${ledger.runtime.name}@${ledger.runtime.version}/${ledger.runtime.backend} | ledger ${ledger.ledger_sha256.slice(0, 16)}...`;
  const boundary = document.createElement("p");
  boundary.className = "calibration-validation-boundary";
  boundary.textContent = ledger.interpretation_boundary;
  root.replaceChildren(metrics, tableWrap, provenance, boundary);
}

function renderEmpty(root, message, error = false) {
  if (!root) return;
  const node = document.createElement("p");
  node.className = `calibration-validation-empty${error ? " error" : ""}`;
  node.textContent = message;
  root.replaceChildren(node);
}

function metric(label, value, detail, tone) {
  const node = document.createElement("div");
  node.className = `calibration-validation-metric ${tone}`;
  const small = document.createElement("span");
  small.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  const p = document.createElement("p");
  p.textContent = detail;
  node.append(small, strong, p);
  return node;
}

function cell(tag, value) {
  const node = document.createElement(tag);
  node.textContent = value;
  return node;
}

function finiteMax(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  return finite.length ? Math.max(...finite) : null;
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  if (number === 0) return "0";
  if (Math.abs(number) < 0.001 || Math.abs(number) >= 10000) return number.toExponential(3);
  return number.toPrecision(5);
}

function setStatus(node, text, state) {
  if (!node) return;
  node.textContent = text;
  node.dataset.state = state;
}
