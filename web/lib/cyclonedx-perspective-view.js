import {
  auditCycloneDxPerspectives,
  renderCycloneDxPerspectiveAuditHtml,
  validateCycloneDxPerspectiveAudit,
} from "./cyclonedx-perspective-audit.js";
import { downloadText } from "./download.js";
import { parseStrictJson } from "./metadata-model-adapters.js";
import { jsonForDownload } from "./report-utils.js";

const MAX_JSON_BYTES = 16 * 1024 * 1024;

export function createCycloneDxPerspectiveController(elements, { onStatus } = {}) {
  let audit = null;
  let sourceName = "cyclonedx-document";

  elements?.runPerspectiveAudit?.addEventListener("click", run);
  elements?.downloadPerspectiveJson?.addEventListener("click", () => {
    if (!audit) return;
    downloadText(`${safeBaseName(sourceName)}.perspective-audit.json`, jsonForDownload(audit), "application/json");
  });
  elements?.downloadPerspectiveHtml?.addEventListener("click", () => {
    if (!audit) return;
    downloadText(
      `${safeBaseName(sourceName)}.perspective-audit.html`,
      renderCycloneDxPerspectiveAuditHtml(audit, { title: `CycloneDX Perspective Audit - ${sourceName}` }),
      "text/html",
    );
  });
  for (const input of [elements?.perspectiveBomInput, elements?.perspectiveSourceInput, elements?.perspectiveProjectionInput]) {
    input?.addEventListener("change", () => {
      audit = null;
      render();
    });
  }

  async function run() {
    const bomFile = elements?.perspectiveBomInput?.files?.[0];
    if (!bomFile) return setStatus("Select a CycloneDX JSON document first", "error");
    const button = elements.runPerspectiveAudit;
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Evaluating";
    try {
      const document = await readJsonFile(bomFile, "CycloneDX document");
      const perspectiveFile = elements?.perspectiveSourceInput?.files?.[0];
      const perspectiveDocument = perspectiveFile
        ? await readJsonFile(perspectiveFile, "Perspective source")
        : document;
      const projectionFile = elements?.perspectiveProjectionInput?.files?.[0];
      const projection = projectionFile
        ? await readJsonFile(projectionFile, "Perspective projection")
        : null;
      audit = auditCycloneDxPerspectives(document, {
        perspectiveDocument,
        mode: projection ? "explicit_candidate_projection" : "raw_document",
        projection,
        expectedTypes: projection?.expected_types || {},
      });
      const validation = validateCycloneDxPerspectiveAudit(audit);
      if (!validation.valid) throw new Error(validation.errors.join("; "));
      sourceName = bomFile.name;
      setStatus(`Perspective audit complete: ${audit.mapping_count} mappings`, "ok");
    } catch (error) {
      audit = null;
      console.error("[cyclonedx-perspective-audit]", error);
      setStatus(`Perspective audit failed: ${error?.message || error}`, "error");
    } finally {
      button.textContent = label;
      render();
    }
  }

  function setStatus(message, state) {
    if (elements?.perspectiveAuditStatus) {
      elements.perspectiveAuditStatus.textContent = message;
      elements.perspectiveAuditStatus.dataset.state = state;
    }
    onStatus?.(message, state);
  }

  function render() {
    const ready = Boolean(elements?.perspectiveBomInput?.files?.length);
    if (elements?.runPerspectiveAudit) elements.runPerspectiveAudit.disabled = !ready;
    if (elements?.downloadPerspectiveJson) elements.downloadPerspectiveJson.disabled = !audit;
    if (elements?.downloadPerspectiveHtml) elements.downloadPerspectiveHtml.disabled = !audit;
    if (!elements?.perspectiveAuditSummary) return;
    if (!audit) {
      elements.perspectiveAuditSummary.textContent = ready
        ? "Ready for raw-document RFC 9535 evaluation. A projection is used only when its sidecar is explicitly selected."
        : "No document selected.";
      return;
    }
    const summary = audit.summary;
    elements.perspectiveAuditSummary.textContent = `${audit.perspective_count} perspectives / ${audit.mapping_count} mappings / ${summary.zero_match_count} zero-match / ${summary.multiple_match_count} multi-match. Policy decision remains NOT_ASSESSABLE.`;
  }

  render();
  return { render, getAudit: () => audit };
}

async function readJsonFile(file, label) {
  if (!file || file.size > MAX_JSON_BYTES) throw new Error(`${label} exceeds the 16 MiB browser limit.`);
  return parseStrictJson(await file.text(), label);
}

function safeBaseName(filename) {
  return String(filename || "cyclonedx-document").replace(/\.(?:cdx\.)?json$/i, "").replace(/[^a-z0-9._-]+/gi, "-") || "cyclonedx-document";
}
