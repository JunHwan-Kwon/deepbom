import { exec as executeJsonPath } from "../vendor/jsonpath-rfc9535.mjs";

import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const CYCLONEDX_PERSPECTIVE_AUDIT_SCHEMA = "deepbom.cyclonedx_perspective_audit.v1";
export const CYCLONEDX_PERSPECTIVE_PROJECTION_SCHEMA = "deepbom.cyclonedx_perspective_projection.v1";
export const CYCLONEDX_PERSPECTIVE_EVALUATOR = Object.freeze({
  standard: "RFC 9535",
  package: "jsonpath-rfc9535",
  version: "1.3.0",
  package_integrity: "sha512-3jFHya7oZ45aDxIIdx+/zQARahHXxFSMWBkcBUldfXpLS9VCXDJyTKt35kQfEXLqh0K3Ixw/9xFnvcDStaxh7Q==",
});

export function auditCycloneDxPerspectives(document, options = {}) {
  assertJsonObject(document, "CycloneDX document");
  const perspectiveSource = options.perspectiveDocument || document;
  assertJsonObject(perspectiveSource, "Perspective source");
  const mode = options.mode || "raw_document";
  if (!new Set(["raw_document", "explicit_candidate_projection"]).has(mode)) {
    throw new Error(`Unsupported perspective audit mode: ${mode}`);
  }

  const projection = mode === "explicit_candidate_projection"
    ? applyExplicitProjection(document, options.projection)
    : { document: structuredClone(document), record: null };
  const evaluationDocument = projection.document;
  const perspectives = perspectiveSource.perspectives;
  if (!Array.isArray(perspectives)) throw new Error("Perspective source must contain a perspectives array.");

  const rows = perspectives.map((perspective, perspectiveIndex) => auditPerspective(
    evaluationDocument,
    perspective,
    perspectiveIndex,
    options.expectedTypes || {},
  ));
  const mappings = rows.flatMap((row) => row.mappings);
  const summary = summarizeMappings(mappings);
  const documentSha256 = digest(document);
  const perspectiveSha256 = digest(perspectiveSource);
  const evaluationSha256 = digest(evaluationDocument);

  return {
    schema: CYCLONEDX_PERSPECTIVE_AUDIT_SCHEMA,
    evidence_class: "DERIVED",
    evaluator: CYCLONEDX_PERSPECTIVE_EVALUATOR,
    mode,
    subject: {
      document_sha256: documentSha256,
      perspective_source_sha256: perspectiveSha256,
      evaluation_document_sha256: evaluationSha256,
      projection_changed_document: documentSha256 !== evaluationSha256,
    },
    projection: projection.record,
    perspective_count: rows.length,
    mapping_count: mappings.length,
    summary,
    policy_interpretation: {
      relevance_vocabulary_observed: [...new Set(mappings.map((row) => row.relevance).filter(Boolean))].sort(),
      required_match_cardinality_semantics: "NOT_DEFINED_BY_EVALUATOR",
      decision_status: "NOT_ASSESSABLE",
      reason: "JSONPath match evidence is reported without converting relevance labels into conformance or release decisions.",
    },
    perspectives: rows,
    claim_boundary: "This audit evaluates JSONPath mappings and records exact matches. It does not define CycloneDX perspective scoping, reference traversal, required cardinality, conformance, or release policy.",
  };
}

export function applyExplicitProjection(document, projection) {
  assertJsonObject(projection, "Perspective projection");
  if (projection.schema !== CYCLONEDX_PERSPECTIVE_PROJECTION_SCHEMA) {
    throw new Error(`Perspective projection schema must be ${CYCLONEDX_PERSPECTIVE_PROJECTION_SCHEMA}.`);
  }
  if (!Array.isArray(projection.rules) || !projection.rules.length) {
    throw new Error("Perspective projection must contain at least one explicit rule.");
  }
  const output = structuredClone(document);
  const appliedRules = [];
  for (const [index, rule] of projection.rules.entries()) {
    if (rule?.operation !== "copy_if_absent") throw new Error(`Projection rule ${index} must use copy_if_absent.`);
    const sourcePointer = normalizePointer(rule.source_pointer, `Projection rule ${index} source_pointer`);
    const targetPointer = normalizePointer(rule.target_pointer, `Projection rule ${index} target_pointer`);
    const source = readPointer(document, sourcePointer);
    if (!source.found) throw new Error(`Projection source does not exist: ${sourcePointer}`);
    const target = readPointer(output, targetPointer);
    if (target.found) throw new Error(`Projection target already exists: ${targetPointer}`);
    writePointer(output, targetPointer, structuredClone(source.value));
    appliedRules.push({ operation: rule.operation, source_pointer: sourcePointer, target_pointer: targetPointer });
  }
  return {
    document: output,
    record: {
      schema: CYCLONEDX_PERSPECTIVE_PROJECTION_SCHEMA,
      profile_id: cleanText(projection.profile_id) || null,
      source_sha256: digest(projection),
      evidence_class: "DECLARED_UNVERIFIED",
      applied_rule_count: appliedRules.length,
      applied_rules: appliedRules,
      normative_status: "CANDIDATE_ONLY",
    },
  };
}

export function validateCycloneDxPerspectiveAudit(audit) {
  const errors = [];
  if (audit?.schema !== CYCLONEDX_PERSPECTIVE_AUDIT_SCHEMA) errors.push("Perspective audit schema is invalid.");
  if (audit?.evaluator?.standard !== "RFC 9535") errors.push("Perspective audit evaluator standard is invalid.");
  if (!Array.isArray(audit?.perspectives)) errors.push("Perspective audit perspectives must be an array.");
  if (!Number.isSafeInteger(audit?.mapping_count) || audit.mapping_count < 0) errors.push("Perspective audit mapping_count is invalid.");
  const mappings = (audit?.perspectives || []).flatMap((row) => Array.isArray(row?.mappings) ? row.mappings : []);
  if (mappings.length !== audit?.mapping_count) errors.push("Perspective audit mapping_count does not match materialized mappings.");
  for (const row of mappings) {
    if (!cleanText(row.expression)) errors.push("Perspective mapping expression is missing.");
    if (!Number.isSafeInteger(row.match_count) || row.match_count < 0) errors.push(`Perspective mapping ${row.expression || "?"} has an invalid match_count.`);
    if (!Array.isArray(row.matches) || row.matches.length !== row.match_count) errors.push(`Perspective mapping ${row.expression || "?"} does not conserve matches.`);
    for (const match of row.matches || []) {
      if (typeof match.json_pointer !== "string" || (match.json_pointer !== "" && !match.json_pointer.startsWith("/"))) {
        errors.push(`Perspective mapping ${row.expression || "?"} has an invalid JSON Pointer.`);
      }
      if (!/^[a-f0-9]{64}$/.test(String(match.value_sha256 || ""))) errors.push(`Perspective mapping ${row.expression || "?"} has an invalid value digest.`);
    }
  }
  if (audit?.policy_interpretation?.decision_status !== "NOT_ASSESSABLE") errors.push("Perspective relevance policy must remain NOT_ASSESSABLE.");
  return { valid: errors.length === 0, errors };
}

export function renderCycloneDxPerspectiveAuditHtml(audit, options = {}) {
  const validation = validateCycloneDxPerspectiveAudit(audit);
  if (!validation.valid) throw new Error(`Cannot render invalid perspective audit: ${validation.errors.join("; ")}`);
  const title = cleanText(options.title) || "CycloneDX Perspective Audit";
  const rows = audit.perspectives.flatMap((perspective) => perspective.mappings.map((mapping) => `
    <tr>
      <td>${escapeHtml(perspective.name)}</td>
      <td><code>${escapeHtml(mapping.expression)}</code></td>
      <td>${escapeHtml(mapping.relevance || "not declared")}</td>
      <td>${escapeHtml(mapping.match_status)}</td>
      <td>${mapping.match_count}</td>
      <td>${mapping.matches.map((match) => `<code>${escapeHtml(match.json_pointer)}</code>`).join("<br>") || "None"}</td>
    </tr>`)).join("");
  const embedded = JSON.stringify(audit).replaceAll("<", "\\u003c");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(title)}</title><style>
:root{color-scheme:light dark;--bg:#f7f7f4;--fg:#171916;--muted:#5c625d;--line:#b7beb8;--panel:#fff;--accent:#185c4c} @media(prefers-color-scheme:dark){:root{--bg:#111512;--fg:#eef3ef;--muted:#b8c1ba;--line:#536058;--panel:#18201b;--accent:#8fd4bd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 4px}p{margin:4px 0 16px;color:var(--muted)}dl{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);background:var(--panel)}dl div{padding:12px;border-right:1px solid var(--line)}dt{font-size:12px;color:var(--muted)}dd{margin:2px 0 0;font-weight:700}table{width:100%;border-collapse:collapse;margin-top:18px;background:var(--panel)}th,td{padding:9px;text-align:left;vertical-align:top;border:1px solid var(--line)}th{color:var(--muted);font-size:12px}code{overflow-wrap:anywhere;color:var(--accent)}.boundary{border-left:4px solid var(--accent);padding:10px 12px;background:var(--panel);color:var(--fg)}@media(max-width:700px){main{padding:14px}dl{grid-template-columns:1fr 1fr}.table{overflow:auto}table{min-width:860px}}
</style></head><body><main><h1>${escapeHtml(title)}</h1>
<p>RFC 9535 mapping evidence. Relevance is not converted into a conformance or release decision.</p>
<dl><div><dt>Mode</dt><dd>${escapeHtml(audit.mode)}</dd></div><div><dt>Perspectives</dt><dd>${audit.perspective_count}</dd></div><div><dt>Mappings</dt><dd>${audit.mapping_count}</dd></div><div><dt>Zero matches</dt><dd>${audit.summary.zero_match_count}</dd></div></dl>
<p class="boundary">${escapeHtml(audit.claim_boundary)}</p><div class="table"><table><thead><tr><th>Perspective</th><th>RFC 9535 expression</th><th>Relevance</th><th>Observed state</th><th>Matches</th><th>JSON Pointers</th></tr></thead><tbody>${rows}</tbody></table></div>
<script type="application/json" id="deepbom-perspective-audit">${embedded}</script></main></body></html>`;
}

function auditPerspective(document, perspective, perspectiveIndex, expectedTypes) {
  if (!perspective || typeof perspective !== "object" || Array.isArray(perspective)) {
    throw new Error(`Perspective ${perspectiveIndex} must be an object.`);
  }
  if (!Array.isArray(perspective.mappings)) throw new Error(`Perspective ${perspectiveIndex} must contain a mappings array.`);
  return {
    perspective_index: perspectiveIndex,
    bom_ref: cleanText(perspective["bom-ref"]) || null,
    name: cleanText(perspective.name) || `Perspective ${perspectiveIndex + 1}`,
    mapping_count: perspective.mappings.length,
    mappings: perspective.mappings.map((mapping, mappingIndex) => auditMapping(
      document,
      mapping,
      perspectiveIndex,
      mappingIndex,
      expectedTypes,
    )),
  };
}

function auditMapping(document, mapping, perspectiveIndex, mappingIndex, expectedTypes) {
  const expression = cleanText(mapping?.expression);
  if (!expression) throw new Error(`Perspective ${perspectiveIndex} mapping ${mappingIndex} has no expression.`);
  const matches = [];
  let expressionError = null;
  try {
    executeJsonPath(document, expression, (value, path) => {
      matches.push({
        json_pointer: pathToJsonPointer(path),
        value_type: jsonType(value),
        empty: isEmptyValue(value),
        value_sha256: digest(value),
        value_preview: scalarPreview(value),
      });
    });
  } catch (error) {
    expressionError = cleanText(error?.message) || String(error);
  }
  const declaredExpected = normalizeExpectedTypes(mapping?.expectedTypes ?? expectedTypes[expression]);
  const typeMismatchCount = declaredExpected.length
    ? matches.filter((row) => !declaredExpected.includes(row.value_type)).length
    : 0;
  const emptyValueCount = matches.filter((row) => row.empty).length;
  return {
    perspective_index: perspectiveIndex,
    mapping_index: mappingIndex,
    native_name: cleanText(mapping?.nativeName) || null,
    expression,
    relevance: cleanText(mapping?.relevance) || null,
    expression_status: expressionError ? "invalid" : "valid",
    expression_error: expressionError,
    match_status: primaryMatchStatus(expressionError, matches.length, emptyValueCount, typeMismatchCount),
    match_count: matches.length,
    cardinality_status: matches.length === 0 ? "zero_matches" : matches.length === 1 ? "single_match" : "multiple_matches",
    empty_value_count: emptyValueCount,
    expected_types: declaredExpected.length ? declaredExpected : null,
    type_assessment_status: declaredExpected.length ? (typeMismatchCount ? "type_mismatch" : "pass") : "not_assessed_expected_type_unbound",
    type_mismatch_count: declaredExpected.length ? typeMismatchCount : null,
    policy_status: "NOT_ASSESSABLE",
    matches,
  };
}

function summarizeMappings(rows) {
  const count = (key, value) => rows.filter((row) => row[key] === value).length;
  return {
    valid_expression_count: count("expression_status", "valid"),
    invalid_expression_count: count("expression_status", "invalid"),
    zero_match_count: count("cardinality_status", "zero_matches"),
    single_match_count: count("cardinality_status", "single_match"),
    multiple_match_count: count("cardinality_status", "multiple_matches"),
    mappings_with_empty_values: rows.filter((row) => row.empty_value_count > 0).length,
    mappings_with_type_mismatch: rows.filter((row) => Number(row.type_mismatch_count) > 0).length,
    required_mapping_count: rows.filter((row) => row.relevance === "required").length,
    required_zero_match_count: rows.filter((row) => row.relevance === "required" && row.match_count === 0).length,
  };
}

function primaryMatchStatus(error, count, emptyCount, typeMismatchCount) {
  if (error) return "expression_error";
  if (count === 0) return "zero_matches";
  if (typeMismatchCount > 0) return "type_mismatch";
  if (emptyCount > 0) return "empty_value";
  if (count > 1) return "multiple_matches";
  return "matched";
}

function pathToJsonPointer(path) {
  if (!Array.isArray(path)) throw new Error("RFC 9535 evaluator returned a non-array normalized path.");
  if (!path.length) return "";
  return `/${path.map((segment) => String(segment).replaceAll("~", "~0").replaceAll("/", "~1")).join("/")}`;
}

function scalarPreview(value) {
  if (value === null || typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value !== "string") return null;
  return value.length <= 160 ? value : `${value.slice(0, 157)}...`;
}

function normalizeExpectedTypes(value) {
  if (value == null) return [];
  const types = Array.isArray(value) ? value : [value];
  const allowed = new Set(["null", "boolean", "number", "string", "array", "object"]);
  const normalized = [...new Set(types.map((item) => cleanText(item).toLowerCase()).filter(Boolean))].sort();
  if (normalized.some((item) => !allowed.has(item))) throw new Error(`Unsupported expected JSON type: ${normalized.find((item) => !allowed.has(item))}`);
  return normalized;
}

function jsonType(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value === "object" ? "object" : typeof value;
}

function isEmptyValue(value) {
  if (value === null || value === "") return true;
  if (Array.isArray(value)) return value.length === 0;
  return typeof value === "object" && value != null && Object.keys(value).length === 0;
}

function digest(value) {
  return sha256TextHex(canonicalJson(value));
}

function assertJsonObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be a JSON object.`);
}

function normalizePointer(value, label) {
  const pointer = cleanText(value);
  if (!pointer.startsWith("/") || pointer.endsWith("/")) throw new Error(`${label} must be a non-root RFC 6901 JSON Pointer.`);
  parsePointer(pointer);
  return pointer;
}

function parsePointer(pointer) {
  return pointer.slice(1).split("/").map((segment) => {
    if (/~(?:[^01]|$)/.test(segment)) throw new Error(`Invalid JSON Pointer escape in ${pointer}.`);
    const decoded = segment.replaceAll("~1", "/").replaceAll("~0", "~");
    if (["__proto__", "prototype", "constructor"].includes(decoded)) throw new Error(`Unsafe JSON Pointer segment in ${pointer}.`);
    return decoded;
  });
}

function readPointer(document, pointer) {
  let current = document;
  for (const segment of parsePointer(pointer)) {
    if (!current || typeof current !== "object" || !Object.hasOwn(current, segment)) return { found: false, value: undefined };
    current = current[segment];
  }
  return { found: true, value: current };
}

function writePointer(document, pointer, value) {
  const segments = parsePointer(pointer);
  let current = document;
  for (const segment of segments.slice(0, -1)) {
    if (!current || typeof current !== "object" || !Object.hasOwn(current, segment)) {
      throw new Error(`Projection target parent does not exist: ${pointer}`);
    }
    current = current[segment];
  }
  const finalSegment = segments.at(-1);
  if (!current || typeof current !== "object" || Object.hasOwn(current, finalSegment)) {
    throw new Error(`Projection target cannot be written: ${pointer}`);
  }
  current[finalSegment] = value;
}

function cleanText(value) {
  return String(value ?? "").trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
