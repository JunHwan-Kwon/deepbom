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

const gguf = auditTabApplicability("gguf", { ops: [] });
const tflite = auditTabApplicability("tflite", { ops: [{ op_type: "CONV_2D" }] });
for (const [format, ledger] of [["gguf", gguf], ["tflite", tflite]]) {
  if (Object.keys(ledger).length !== 8) throw new Error(`${format} applicability ledger is incomplete.`);
  for (const [tab, row] of Object.entries(ledger)) {
    if (!statuses.includes(row.applicability_status) || !row.reason_code || !row.reason_text) {
      throw new Error(`${format}/${tab} applicability record is incomplete.`);
    }
  }
}
if (gguf.stage.applicability_status !== "not_applicable"
  || gguf.stage.reason_code !== "EXECUTABLE_GRAPH_NOT_SERIALIZED"
  || tflite.xnnpack.applicability_status !== "applicable") {
  throw new Error("Graphless-container or TFLite applicability semantics changed.");
}

const html = readFileSync(path.join(ROOT, "web", "index.html"), "utf8");
const css = readFileSync(path.join(ROOT, "web", "research-theme.css"), "utf8");
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

console.log("Evidence class, applicability, workflow, and fixed-domain UI contracts passed.");
