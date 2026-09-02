import { normalizeEvidenceClass } from "./evidence-class.js";

export const EVIDENCE_VISUAL_CONTRACT_SCHEMA = "deepbom.evidence_visual_contract.v1";

const LABELS = Object.freeze({
  OBSERVED: "Observed",
  SOURCE_BACKED: "Source-backed",
  DERIVED: "Derived",
  DERIVED_WITH_HEURISTIC_THRESHOLD: "Derived with heuristic threshold",
  PREDICTED: "Predicted",
  ESTIMATED: "Estimated",
  DECLARED_UNVERIFIED: "Declared, unverified",
  MEASURED: "Measured",
  NOT_ASSESSABLE: "Not assessable",
  NOT_APPLICABLE: "Not applicable",
});

export function decorateEvidenceElement(element, evidenceClass, { label = true } = {}) {
  if (!element) return null;
  const canonical = normalizeEvidenceClass(evidenceClass);
  element.dataset.evidenceClass = canonical;
  if (label && !String(element.textContent || "").trim()) element.textContent = LABELS[canonical];
  element.setAttribute("aria-label", element.getAttribute("aria-label") || `Evidence class: ${LABELS[canonical]}`);
  element.title = element.title || `Evidence class: ${LABELS[canonical]}`;
  return canonical;
}

export function evidenceClassLabel(value) {
  const canonical = normalizeEvidenceClass(value);
  return LABELS[canonical];
}
