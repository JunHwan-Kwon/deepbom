import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  EVIDENCE_CLASSES,
  normalizeEvidenceClass,
} from "../web/lib/evidence-class.js";
import {
  APPLICABILITY_STATUS,
  auditTabApplicability,
} from "../web/lib/evidence-applicability.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const expectedClasses = [
  "OBSERVED",
  "SOURCE_BACKED",
  "DERIVED",
  "DERIVED_WITH_HEURISTIC_THRESHOLD",
  "PREDICTED",
  "ESTIMATED",
  "DECLARED_UNVERIFIED",
  "MEASURED",
  "NOT_ASSESSABLE",
  "NOT_APPLICABLE",
];
const actualClasses = [...EVIDENCE_CLASSES];
if (JSON.stringify(actualClasses) !== JSON.stringify(expectedClasses)) {
  throw new Error(`Canonical evidence classes changed: ${JSON.stringify(actualClasses)}`);
}
if (normalizeEvidenceClass("IMPORTED_IDENTITY_BOUND_RUNTIME_EVIDENCE") !== "MEASURED"
  || normalizeEvidenceClass("OBSERVED_SERIALIZED_ARTIFACT") !== "OBSERVED"
  || normalizeEvidenceClass("PREDICTED_SOURCE_AND_ARTIFACT_ELIGIBILITY") !== "PREDICTED"
  || normalizeEvidenceClass("unknown") !== "NOT_ASSESSABLE") {
  throw new Error("Evidence-class aliases do not preserve the canonical visual contract.");
}

const statuses = Object.values(APPLICABILITY_STATUS);
if (JSON.stringify(statuses) !== JSON.stringify(["applicable", "not_applicable", "not_assessable", "not_assessed_yet"])) {
  throw new Error(`Applicability status vocabulary changed: ${JSON.stringify(statuses)}`);
}

const formatCases = {
  tflite: { format: "tflite", ops: [{ op_type: "CONV_2D" }] },
  onnx: { format: "onnx", ops: [{ op_type: "Conv" }] },
  coreml: { format: "coreml", ops: [{ op_type: "convolution" }] },
  gguf: { format: "gguf", ops: [], on_device_llm: { architecture: { family: "llama" } } },
  safetensors: { format: "safetensors", ops: [], on_device_llm: { architecture: { family: "llama" } } },
  executorch: { format: "executorch", executorch_container: "pte", ops: [{ op_type: "aten.add.Tensor" }] },
};
const ledgers = Object.fromEntries(Object.entries(formatCases).map(([format, analysis]) => [format, auditTabApplicability(format, analysis)]));
const gguf = ledgers.gguf;
const tflite = ledgers.tflite;
for (const [format, ledger] of Object.entries(ledgers)) {
  if (Object.keys(ledger).length !== 8) throw new Error(`${format} applicability ledger is incomplete.`);
  for (const [tab, row] of Object.entries(ledger)) {
    if (!statuses.includes(row.applicability_status) || !row.reason_code || !row.reason_text) {
      throw new Error(`${format}/${tab} applicability record is incomplete.`);
    }
  }
}
const pending = auditTabApplicability("onnx", null);
if (!Object.values(pending).every((row) => row.applicability_status === "not_assessed_yet" && row.required_evidence)) {
  throw new Error("Pre-audit applicability must remain explicitly not assessed yet.");
}
const coreMlUnmaterialized = auditTabApplicability("coreml", { format: "coreml", ops: [] });
if (coreMlUnmaterialized.stage.applicability_status !== "not_assessable" || !coreMlUnmaterialized.stage.required_evidence) {
  throw new Error("Materializable-but-unresolved Core ML graph scope must remain not assessable with required evidence.");
}
for (const [format, ledger] of Object.entries(ledgers)) {
  for (const lens of ["xnnpack", "quant-labs", "llm"]) {
    if (!ledger[lens]) throw new Error(`${format}/${lens} specialized lens disappeared from the applicability ledger.`);
  }
}
if (gguf.stage.applicability_status !== "not_applicable"
  || gguf.stage.reason_code !== "EXECUTABLE_GRAPH_NOT_SERIALIZED"
  || tflite.xnnpack.applicability_status !== "applicable") {
  throw new Error("Graphless-container or TFLite applicability semantics changed.");
}

const formatTabsSource = readFileSync(path.join(ROOT, "web", "lib", "format-audit-tabs.js"), "utf8");
const appSource = readFileSync(path.join(ROOT, "web", "app.js"), "utf8");
if (!formatTabsSource.includes("tab.hidden = false") || !formatTabsSource.includes("option.hidden = false") || !formatTabsSource.includes("option.disabled = false")) {
  throw new Error("Desktop or mobile audit lenses can disappear instead of exposing applicability.");
}
if (/getActiveAuditTab\(\) === ["']xnnpack["'][\s\S]{0,100}setActiveAuditTab\(["']overview["']\)/.test(appSource)) {
  throw new Error("Format changes must not silently replace an active non-applicable lens with Overview.");
}

const html = readFileSync(path.join(ROOT, "web", "index.html"), "utf8");
const css = readFileSync(path.join(ROOT, "web", "research-theme.css"), "utf8");
const shellCss = readFileSync(path.join(ROOT, "web", "app-shell.css"), "utf8");
const workflow = [...html.matchAll(/data-navigation-role="workflow"/g)].length;
const domains = [...html.matchAll(/data-audit-primary="true"/g)].length;
if (workflow !== 4 || domains !== 5) {
  throw new Error(`Expected four workflow stages and five evidence domains, found ${workflow}/${domains}.`);
}
for (const evidenceClass of expectedClasses) {
  if (!css.includes(`[data-evidence-class="${evidenceClass}"]`)) {
    throw new Error(`Missing visual contract for ${evidenceClass}.`);
  }
}
if (!shellCss.includes('[data-format-scope="tflite"]:not([data-audit-tab])')) {
  throw new Error("Legacy format-scope CSS can hide a specialized audit lens instead of exposing applicability.");
}

console.log("Evidence class, applicability, workflow, and fixed-domain UI contracts passed.");
