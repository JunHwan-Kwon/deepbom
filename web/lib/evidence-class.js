export const EVIDENCE_CLASSES = Object.freeze([
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
]);

const EVIDENCE_CLASS_SET = new Set(EVIDENCE_CLASSES);

export function isCanonicalEvidenceClass(value) {
  return EVIDENCE_CLASS_SET.has(String(value || "").trim().toUpperCase());
}

export function normalizeEvidenceClass(value, fallback = "NOT_ASSESSABLE") {
  const normalized = String(value || "").trim().toUpperCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (EVIDENCE_CLASS_SET.has(normalized)) return normalized;
  if (normalized.includes("NOT_ASSESSABLE") || normalized.includes("NOT_ASSESSED")) return "NOT_ASSESSABLE";
  if (normalized.includes("NOT_APPLICABLE")) return "NOT_APPLICABLE";
  if (normalized.includes("MEASURED") || normalized.includes("OBSERVED_RUNTIME")
    || normalized.includes("IMPORTED_IDENTITY_BOUND_RUNTIME_EVIDENCE")) return "MEASURED";
  if (normalized.includes("PREDICTED")) return "PREDICTED";
  if (normalized.includes("ESTIMATED")) return "ESTIMATED";
  if (normalized.includes("SOURCE_BACKED") || normalized.includes("SOURCE_BASED") || normalized.includes("SOURCE_PINNED")) return "SOURCE_BACKED";
  if (normalized.includes("DECLARED")) return "DECLARED_UNVERIFIED";
  if (normalized.includes("HEURISTIC")) return "DERIVED_WITH_HEURISTIC_THRESHOLD";
  if (normalized.includes("DERIVED")) return "DERIVED";
  if (normalized.includes("OBSERVED")) return "OBSERVED";
  return EVIDENCE_CLASS_SET.has(fallback) ? fallback : "NOT_ASSESSABLE";
}
