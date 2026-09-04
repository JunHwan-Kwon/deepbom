export const FINDING_EVIDENCE_POINTER_VALIDATION_SCHEMA = "deepbom.finding_evidence_pointer_validation.v1";

export const FINDING_PRIORITIES = Object.freeze(["High", "Medium", "Low", "Informational"]);
export const FINDING_KINDS = Object.freeze(["artifact_defect", "caution", "evidence_gap"]);

const PRIORITY_RANK = new Map(FINDING_PRIORITIES.map((priority, index) => [priority, index]));
const CONFIDENCE_LEVELS = new Set(["high", "medium", "low"]);

export function sortFindingsByPriority(findings = []) {
  return findings.map((item, index) => ({ item, index })).sort((left, right) =>
    (PRIORITY_RANK.get(left.item.technical_priority) ?? 99) - (PRIORITY_RANK.get(right.item.technical_priority) ?? 99)
    || left.index - right.index).map(({ item }) => item);
}

export function validateFindingEvidenceBindings(findings = [], evidenceRoot) {
  const ids = new Set();
  const schemaErrors = [];
  const unresolvedPointers = [];
  let pointerCount = 0;
  let resolvedPointerCount = 0;
  for (const [index, finding] of findings.entries()) {
    const id = String(finding?.finding_id || "");
    const pointers = Array.isArray(finding?.evidence_json_pointers) ? finding.evidence_json_pointers : [];
    const valid = id && !ids.has(id)
      && PRIORITY_RANK.has(finding?.technical_priority)
      && FINDING_KINDS.includes(finding?.finding_kind)
      && CONFIDENCE_LEVELS.has(finding?.confidence)
      && String(finding?.evidence_class || "").trim()
      && String(finding?.source_rule_id || "").trim()
      && String(finding?.method_version || "").trim()
      && finding?.origin === "report_synthesis"
      && pointers.length > 0 && new Set(pointers).size === pointers.length;
    if (!valid) schemaErrors.push({ finding_id: id || `index:${index}`, message: `Finding ${id || index} has a duplicate/empty ID, invalid priority/confidence, missing source/method/evidence, non-synthesis origin, or empty/duplicate pointer list.` });
    ids.add(id);
    for (const pointer of pointers) {
      pointerCount += 1;
      const resolution = resolveEvidenceJsonPointer(evidenceRoot, pointer);
      if (resolution.found) resolvedPointerCount += 1;
      else unresolvedPointers.push({ finding_id: id || `index:${index}`, pointer, message: `Finding ${id || index} evidence pointer ${pointer} does not resolve: ${resolution.reason}.` });
    }
  }
  return {
    schema: FINDING_EVIDENCE_POINTER_VALIDATION_SCHEMA,
    finding_count: findings.length,
    pointer_count: pointerCount,
    resolved_pointer_count: resolvedPointerCount,
    unresolved_pointer_count: unresolvedPointers.length,
    schema_error_count: schemaErrors.length,
    schema_errors: schemaErrors,
    unresolved_pointers: unresolvedPointers,
    method: "Resolve every authoritative finding evidence_json_pointers entry as an RFC 6901 JSON Pointer against the exact pre-conformance engineering-evidence document; require unique IDs and pointers plus enumerated priority and confidence contracts.",
  };
}

function resolveEvidenceJsonPointer(root, pointer) {
  if (pointer === "") return { found: true, value: root };
  if (typeof pointer !== "string" || !pointer.startsWith("/") || /~(?:[^01]|$)/.test(pointer)) return { found: false, reason: "invalid_rfc6901_syntax" };
  let value = root;
  for (const encoded of pointer.slice(1).split("/")) {
    const token = encoded.replace(/~1/g, "/").replace(/~0/g, "~");
    if (value == null || (typeof value !== "object" && typeof value !== "function") || !Object.prototype.hasOwnProperty.call(value, token)) {
      return { found: false, reason: `missing_token:${token}` };
    }
    value = value[token];
  }
  return { found: true, value };
}
