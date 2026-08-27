import { validateExecutionPlacementEvidence } from "./execution-placement-evidence.js";
import { ANALYZER_METADATA } from "./report-metadata.js";

export function registerExecutionPlacementConformance({
  check,
  id,
  analysis,
  executionPlacement,
  engineeringReport,
  serialized = false,
}) {
  const validation = validateExecutionPlacementEvidence(executionPlacement, { analysis });
  const report = String(engineeringReport || "");
  check(id, validation.valid
    && executionPlacement?.schema === ANALYZER_METADATA.schemas.executionPlacementEvidence
    && report.includes("## Execution Placement Evidence")
    && report.includes(executionPlacement?.schema || "missing-schema"),
  validation.issues[0] || `${serialized ? "Serialized-artifact " : ""}execution-placement evidence is invalid or absent from the Engineering Report.`,
  ["/evidence/execution_placement", "/engineering_report.md"]);
}
