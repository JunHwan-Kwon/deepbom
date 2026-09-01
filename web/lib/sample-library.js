import { publicSampleManifestDocument, publicSampleModel } from "./sample-models.js";
import { comparePublicSampleEvidence } from "./sample-verification.js";

const number = new Intl.NumberFormat("en-US");

function textElement(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = value;
  return element;
}

function expectedSummary(expected) {
  const parts = [];
  if (expected.operatorCount == null) parts.push("operator graph N/A");
  else parts.push(`${number.format(expected.operatorCount)} ops`);
  parts.push(`${number.format(expected.tensorCount)} tensors`);
  if (expected.totalMacs == null) parts.push("MACs N/A");
  else parts.push(`${number.format(expected.totalMacs)} MACs`);
  return parts.join(" / ");
}

async function readExampleBytes(response, expectedBytes, onProgress) {
  if (!response.body?.getReader || !Number.isSafeInteger(expectedBytes) || expectedBytes <= 0) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, expectedBytes || buffer.byteLength);
    return buffer;
  }
  const output = new Uint8Array(expectedBytes);
  const reader = response.body.getReader();
  let offset = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.byteLength > expectedBytes) {
      await reader.cancel();
      throw new Error(`byte mismatch (more than ${expectedBytes})`);
    }
    output.set(value, offset);
    offset += value.byteLength;
    onProgress?.(offset, expectedBytes);
  }
  if (offset !== expectedBytes) return output.slice(0, offset).buffer;
  return output.buffer;
}

function renderEvidenceGlance(container, sample) {
  if (!container || !sample) return;
  const expected = sample.expectedEvidence;
  const header = document.createElement("header");
  const title = document.createElement("div");
  title.append(
    textElement("span", "sample-format", `${sample.format === "coreml" ? "CORE ML" : sample.format.toUpperCase()} VERIFIED BASELINE`),
    textElement("h3", "", sample.label.replace(/^[^-]+-\s*/, "")),
  );
  header.append(title, textElement("span", "sample-glance-hash", `SHA-256 ${sample.sha256.slice(0, 12)}...`));

  const facts = document.createElement("dl");
  const rows = [
    ["Static baseline", expectedSummary(expected)],
  ];
  if (expected.quantizeOps != null || expected.dequantizeOps != null) {
    rows.push(["Serialized Q / DQ", `${number.format(expected.quantizeOps || 0)} / ${number.format(expected.dequantizeOps || 0)}`]);
  }
  if (expected.predictedBreakOps != null) rows.push(["Predicted break ops", number.format(expected.predictedBreakOps)]);
  else if (expected.tensorRtConditionallyEligibleOps != null) {
    rows.push([
      "TensorRT parser capability",
      `${number.format(expected.tensorRtConditionallyEligibleOps)}/${number.format(expected.operatorCount)} conditionally eligible`,
    ]);
  }
  if (expected.llmMatrixMultiplyOps != null) {
    rows.push([
      "Serialized LLM signals",
      `${number.format(expected.llmMatrixMultiplyOps)} MatMul / ${number.format(expected.llmExternalStateCandidates || 0)} state candidates`,
    ]);
  }
  else if (expected.payloadBytes != null) rows.push(["Validated payload", `${number.format(expected.payloadBytes)} B`]);
  for (const [term, value] of rows) {
    const row = document.createElement("div");
    row.append(textElement("dt", "", term), textElement("dd", "", value));
    facts.append(row);
  }

  const interpretation = document.createElement("div");
  interpretation.className = "sample-glance-interpretation";
  const establishes = document.createElement("p");
  establishes.append(textElement("strong", "", "Establishes: "), document.createTextNode(sample.focus));
  const boundary = document.createElement("p");
  boundary.append(
    textElement("strong", "", "Does not establish: "),
    document.createTextNode(`${sample.coverageNote} No verified example establishes task accuracy, clinical performance, or release readiness.`),
  );
  interpretation.append(establishes, boundary);
  container.replaceChildren(header, facts, interpretation);
}

function downloadManifest() {
  const blob = new Blob([`${JSON.stringify(publicSampleManifestDocument(), null, 2)}\n`], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "deepbom-public-sample-expected-evidence.v1.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function installPublicSampleLibrary({ models, select, focus, glance, grid, verificationPanel, downloadButton, runButton, digestArtifact, onArtifact, onError, onProgress }) {
  if (!select) return;
  let activeSample = null;
  select.replaceChildren(...models.map((sample) => {
    const option = document.createElement("option");
    option.value = sample.id;
    option.textContent = sample.label;
    option.title = `${sample.filename}; ${sample.license}; SHA-256 ${sample.sha256}`;
    option.setAttribute("aria-label", `${sample.format}, ${sample.label}, ${sample.purpose}, ${sample.analysisDepth}`);
    return option;
  }));

  const rows = new Map();
  for (const sample of models) {
    if (!grid) break;
    const row = document.createElement("article");
    row.className = "sample-profile";
    row.dataset.sampleId = sample.id;

    const heading = document.createElement("div");
    heading.className = "sample-profile-head";
    const title = document.createElement("div");
    title.append(
      textElement("span", "sample-format", sample.format === "coreml" ? "CORE ML" : sample.format.toUpperCase()),
      textElement("h3", "", sample.label.replace(/^[^-]+-\s*/, "")),
      textElement("p", "", sample.purpose),
    );
    const run = textElement("button", "secondary-action", "Use example");
    run.type = "button";
    run.addEventListener("click", () => {
      select.value = sample.id;
      select.dispatchEvent(new Event("change"));
      runSample(sample, run);
    });
    heading.append(title, run);

    const facts = document.createElement("dl");
    for (const [term, value] of [
      ["Depth", sample.analysisDepth],
      ["Expected", expectedSummary(sample.expectedEvidence)],
      ["Identity", `${number.format(sample.byteLength)} B / ${sample.sha256.slice(0, 12)}...`],
      ["Provenance", `${sample.license} / ${sample.sourceRevision}`],
    ]) {
      facts.append(textElement("dt", "", term), textElement("dd", "", value));
    }

    const evidence = document.createElement("p");
    evidence.className = "sample-profile-evidence";
    evidence.append(textElement("strong", "", "Key evidence: "), document.createTextNode(sample.focus));
    const boundary = document.createElement("p");
    boundary.className = "sample-profile-boundary";
    boundary.append(textElement("strong", "", "Claim boundary: "), document.createTextNode(sample.coverageNote));
    const source = textElement("a", "sample-profile-source", "Source record");
    source.href = sample.source;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    source.setAttribute("aria-label", `Open source record for ${sample.label}`);

    row.append(heading, facts, evidence, boundary, source);
    grid.append(row);
    rows.set(sample.id, row);
  }

  const sync = () => {
    const selected = publicSampleModel(select.value);
    if (focus) focus.textContent = selected ? "Hash-pinned identity is verified before analysis." : "";
    renderEvidenceGlance(glance, selected);
    for (const [id, row] of rows) {
      const active = id === selected?.id;
      row.classList.toggle("selected", active);
      if (active) row.setAttribute("aria-current", "true");
      else row.removeAttribute("aria-current");
    }
  };
  select.addEventListener("change", sync);
  downloadButton?.addEventListener("click", downloadManifest);
  runButton?.addEventListener("click", () => {
    const sample = publicSampleModel(select.value);
    if (sample) runSample(sample, runButton);
  });
  sync();

  async function runSample(sample, button) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "Verifying example";
    try {
      onProgress?.({ phase: "download", loaded: 0, total: sample.byteLength, percent: 0, sample });
      const response = await fetch(sample.path, { cache: "no-cache" });
      if (!response.ok) throw new Error(`example fetch failed (${response.status})`);
      const bytes = await readExampleBytes(response, sample.byteLength, (loaded, total) => {
        onProgress?.({ phase: "download", loaded, total, percent: total ? loaded / total : null, sample });
      });
      if (bytes.byteLength !== sample.byteLength) throw new Error(`byte mismatch (${bytes.byteLength} != ${sample.byteLength})`);
      onProgress?.({ phase: "hash", loaded: bytes.byteLength, total: bytes.byteLength, percent: 1, sample });
      const digest = await digestArtifact(bytes);
      if (digest !== sample.sha256) throw new Error(`example SHA-256 mismatch (${digest})`);
      const companions = {};
      for (const companion of sample.companions || []) {
        if (companions[companion.role]) throw new Error(`duplicate example companion role (${companion.role})`);
        const companionResponse = await fetch(companion.path, { cache: "no-cache" });
        if (!companionResponse.ok) throw new Error(`example companion fetch failed (${companionResponse.status})`);
        const companionBytes = await readExampleBytes(companionResponse, companion.byteLength);
        if (companionBytes.byteLength !== companion.byteLength) {
          throw new Error(`companion byte mismatch (${companionBytes.byteLength} != ${companion.byteLength})`);
        }
        const companionDigest = await digestArtifact(companionBytes);
        if (companionDigest !== companion.sha256) throw new Error(`example companion SHA-256 mismatch (${companionDigest})`);
        const decoded = new TextDecoder("utf-8", { fatal: true }).decode(companionBytes);
        const parsed = JSON.parse(decoded);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error(`example companion ${companion.role} is not a JSON object`);
        companions[companion.role] = parsed;
      }
      onProgress?.({ phase: "verified", loaded: bytes.byteLength, total: bytes.byteLength, percent: 1, sample });
      await onArtifact(new File([bytes], sample.filename), sample, companions);
    } catch (error) {
      onError?.(error);
    } finally {
      button.disabled = false;
      button.textContent = original;
    }
  }

  const clearVerification = () => {
    if (!verificationPanel) return;
    verificationPanel.hidden = true;
    verificationPanel.replaceChildren();
  };
  const renderVerification = (sample, analysis, identity = {}) => {
    if (!verificationPanel) return null;
    const result = comparePublicSampleEvidence(sample, analysis, identity);
    if (!result) return null;
    const head = document.createElement("div");
    head.className = "sample-verification-head";
    const title = document.createElement("div");
    title.append(
      textElement("span", "sample-format", "VERIFIED RESULT"),
      textElement("h3", "", sample.label),
      textElement("p", "", `${result.checks.length} checks against this hash-pinned artifact.`),
    );
    const status = textElement("strong", `sample-verification-status ${result.status}`, result.status === "pass"
      ? `${result.passed}/${result.checks.length} passed`
      : `${result.failed}/${result.checks.length} failed`);
    status.setAttribute("role", "status");
    head.append(title, status);

    const table = document.createElement("table");
    table.className = "sample-verification-table";
    const thead = document.createElement("thead");
    const header = document.createElement("tr");
    for (const value of ["Evidence", "Expected", "Observed", "Status"]) {
      const cell = textElement("th", "", value);
      cell.scope = "col";
      header.append(cell);
    }
    thead.append(header);
    const tbody = document.createElement("tbody");
    for (const row of result.checks) {
      const tr = document.createElement("tr");
      tr.className = row.status;
      const evidence = textElement("th", "", row.label);
      evidence.scope = "row";
      const expected = textElement("td", "", displayEvidenceValue(row.expected, row.kind));
      expected.dataset.label = "Expected";
      const observed = textElement("td", "", displayEvidenceValue(row.observed, row.kind));
      observed.dataset.label = "Observed";
      const statusCell = textElement("td", "sample-verification-cell-status", row.status === "pass" ? "Pass" : "Mismatch");
      statusCell.dataset.label = "Status";
      tr.append(evidence, expected, observed, statusCell);
      tbody.append(tr);
    }
    table.append(thead, tbody);
    const details = document.createElement("details");
    details.className = "sample-verification-details";
    details.open = result.status !== "pass";
    details.append(textElement("summary", "", result.status === "pass" ? "Review expected versus observed checks" : "Review regression mismatches"), table);
    const boundary = textElement("p", "sample-profile-boundary", "Deterministic static evidence only; runtime assignment, device timing, and task accuracy remain outside this profile.");
    verificationPanel.replaceChildren(head, details, boundary);
    verificationPanel.hidden = false;
    return result;
  };
  return {
    setActive(sample) {
      activeSample = sample || null;
      clearVerification();
    },
    verifyActive(analysis, artifactByteLength = null) {
      if (!activeSample) return null;
      const result = renderVerification(activeSample, analysis, {
        artifactSha256: analysis?.model_sha256,
        artifactByteLength: analysis?.file_size_bytes ?? analysis?.file_size ?? artifactByteLength,
      });
      if (result?.status !== "pass") {
        throw new Error(`Regression mismatch (${result?.failed || 0}/${result?.checks?.length || 0} checks failed).`);
      }
      return result;
    },
  };
}

function displayEvidenceValue(value, kind) {
  if (value == null) return "Not applicable";
  if (kind === "sha256") return String(value).length > 18 ? `${String(value).slice(0, 16)}...` : String(value);
  if (kind === "number" && Number.isFinite(Number(value))) return number.format(Number(value));
  return String(value);
}
