import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

// A bounded transport projection of existing Evidence IR, not another IR or
// an attestation that the receiving service inspected the original bytes.
export const EVIDENCE_QUERY_RESULT_SCHEMA = "deepbom.evidence_query_result.v1";
export const EVIDENCE_QUERY_SECTIONS = Object.freeze(["operators", "operator", "findings", "placement", "fusion", "improvements", "profiles", "weights"]);
export const EVIDENCE_WEIGHT_VIEWS = Object.freeze(["distribution", "channels", "similarity", "spectrum", "sparsity", "quantization"]);
export const EVIDENCE_QUERY_MAX_BYTES = 64 * 1024;
const SHA = /^[a-f0-9]{64}$/;
export const EVIDENCE_QUERY_JSON_SCHEMA = Object.freeze({
  type: "object", additionalProperties: false,
  properties: {
    section: { type: "string", enum: [...EVIDENCE_QUERY_SECTIONS] },
    subject_ref: { type: "string", minLength: 1, maxLength: 512, description: "Exact Model IR subject_ref from a previous result; preferred over an index." },
    source_index: { type: "integer", minimum: 0, description: "Native serialized operator index, not display order. Ambiguous indices require subject_ref." },
    search: { type: "string", maxLength: 160, description: "Literal case-insensitive filter over row names and references." },
    offset: { type: "integer", minimum: 0, default: 0 },
    limit: { type: "integer", minimum: 1, maximum: 40, default: 20 },
    profile_ids: { type: "array", items: { type: "string", minLength: 1, maxLength: 128 }, uniqueItems: true, maxItems: 4, description: "Exact static backend profile IDs returned by section=profiles. Independent eligibility profiles, not runtime assignments." },
    target: { type: "string", minLength: 1, maxLength: 128, description: "Optional TFLite CPU planning profile ID from capabilities. Not detected hardware or a GPU delegate selector." },
    weight_view: { type: "string", enum: [...EVIDENCE_WEIGHT_VIEWS], description: "Only for section=weights: explicitly opt into bounded payload analysis; use returned weight subject_ref to select a tensor. Default distribution." },
  },
  required: ["section"],
});

export function normalizeEvidenceQuery(value) {
  object(value, "query");
  allowed(value, Object.keys(EVIDENCE_QUERY_JSON_SCHEMA.properties), "query");
  if (!EVIDENCE_QUERY_SECTIONS.includes(value.section)) throw new Error("Unknown evidence query section.");
  if (value.weight_view !== undefined && (value.section !== "weights" || !EVIDENCE_WEIGHT_VIEWS.includes(value.weight_view))) throw new Error("Invalid weight view or query section.");
  if (value.section === "weights" && value.source_index !== undefined) throw new Error("Weight queries require a weight subject_ref, not an operator index.");
  for (const [key, max] of [["subject_ref", 512], ["search", 160], ["target", 128]]) {
    if (value[key] !== undefined && (typeof value[key] !== "string" || value[key].length > max || (key !== "search" && !value[key].length))) throw new Error(`Invalid query ${key}.`);
  }
  for (const [key, min, max] of [["source_index", 0, Number.MAX_SAFE_INTEGER], ["offset", 0, Number.MAX_SAFE_INTEGER], ["limit", 1, 40]]) {
    if (value[key] !== undefined && (!Number.isSafeInteger(value[key]) || value[key] < min || value[key] > max)) throw new Error(`Invalid query ${key}.`);
  }
  if (value.subject_ref !== undefined && value.source_index !== undefined) throw new Error("Use subject_ref or source_index, not both.");
  if (value.section === "operator" && value.subject_ref === undefined && value.source_index === undefined) throw new Error("Operator detail requires subject_ref or source_index.");
  const ids = value.profile_ids ?? [];
  if (!Array.isArray(ids) || ids.length > 4 || new Set(ids).size !== ids.length || ids.some(id => typeof id !== "string" || !id.length || id.length > 128)) throw new Error("Invalid query profile_ids.");
  if (ids.length && !["placement", "profiles"].includes(value.section)) throw new Error("profile_ids applies only to placement or profiles.");
  return { ...value, offset: value.offset ?? 0, limit: value.limit ?? 20, profile_ids: [...ids] };
}

export const EVIDENCE_QUERY_RESULT_JSON_SCHEMA = Object.freeze({
  type: "object", additionalProperties: false,
  properties: {
    schema: { const: EVIDENCE_QUERY_RESULT_SCHEMA, type: "string" },
    analyzer_version: { type: "string" },
    analysis_location: { const: "chatgpt_browser_sandbox", type: "string" },
    artifact: { type: "object", additionalProperties: false, properties: { filename: { type: "string" }, format: { type: "string" }, sha256: { type: "string", pattern: "^[a-f0-9]{64}$" } }, required: ["filename", "format", "sha256"] },
    model_ir_sha256: { type: "string", pattern: "^[a-f0-9]{64}$" },
    query: EVIDENCE_QUERY_JSON_SCHEMA,
    query_sha256: { type: "string", pattern: "^[a-f0-9]{64}$" },
    status: { type: "string", enum: ["complete", "partial", "not_assessable", "no_matches"] },
    coverage: { type: "object", additionalProperties: false, properties: {
      total_rows: { type: "integer", minimum: 0 }, matched_rows: { type: "integer", minimum: 0 }, returned_rows: { type: "integer", minimum: 0, maximum: 40 },
      offset: { type: "integer", minimum: 0 }, next_offset: { type: ["integer", "null"], minimum: 0 }, detail_truncated: { type: "boolean" },
    }, required: ["total_rows", "matched_rows", "returned_rows", "offset", "next_offset", "detail_truncated"] },
    context: { type: "object" },
    rows: { type: "array", maxItems: 40, items: { type: "object", additionalProperties: false, properties: {
      subject_ref: { type: "string" }, kind: { type: "string" }, title: { type: "string" }, details: { type: "object" },
      detail_truncations: { type: "array", items: { type: "string" } },
    }, required: ["subject_ref", "kind", "title", "details", "detail_truncations"] } },
    evidence_boundary: { type: "string" },
    transfer_boundary: { type: "string", const: "model_bytes_not_sent_to_deepbom_service" },
    result_sha256: { type: "string", pattern: "^[a-f0-9]{64}$" },
  },
  required: ["schema", "analyzer_version", "analysis_location", "artifact", "model_ir_sha256", "query", "query_sha256", "status", "coverage", "context", "rows", "evidence_boundary", "transfer_boundary", "result_sha256"],
});

export function sealEvidenceQueryResult(body) {
  return { ...body, result_sha256: sha256TextHex(canonicalJson(body)) };
}

export function validateEvidenceQueryResult(result) {
  object(result, "query result");
  allowed(result, EVIDENCE_QUERY_RESULT_JSON_SCHEMA.required, "query result");
  for (const field of EVIDENCE_QUERY_RESULT_JSON_SCHEMA.required) if (!Object.hasOwn(result, field)) throw new Error(`Query result lacks ${field}.`);
  if (new TextEncoder().encode(JSON.stringify(result)).length > EVIDENCE_QUERY_MAX_BYTES) throw new Error("Evidence query result exceeds 64 KiB.");
  if (result.schema !== EVIDENCE_QUERY_RESULT_SCHEMA || result.analysis_location !== "chatgpt_browser_sandbox" || result.transfer_boundary !== "model_bytes_not_sent_to_deepbom_service") throw new Error("Evidence query result boundary is invalid.");
  object(result.artifact, "artifact");
  allowed(result.artifact, ["filename", "format", "sha256"], "artifact");
  if (typeof result.artifact.filename !== "string" || result.artifact.filename.length > 512 || !["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"].includes(result.artifact.format)) throw new Error("Invalid query artifact.");
  for (const hash of [result.artifact.sha256, result.model_ir_sha256, result.query_sha256, result.result_sha256]) if (!SHA.test(hash)) throw new Error("Evidence query digest is invalid.");
  if (typeof result.analyzer_version !== "string" || !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(result.analyzer_version) || result.analyzer_version.length > 64) throw new Error("Invalid query producer version.");
  const query = normalizeEvidenceQuery(result.query);
  if (canonicalJson(query) !== canonicalJson(result.query) || result.query_sha256 !== sha256TextHex(canonicalJson(query))) throw new Error("Evidence query identity mismatch.");
  if (!["complete", "partial", "not_assessable", "no_matches"].includes(result.status)) throw new Error("Invalid query status.");
  const c = result.coverage;
  object(c, "coverage");
  allowed(c, ["total_rows", "matched_rows", "returned_rows", "offset", "next_offset", "detail_truncated"], "coverage");
  for (const key of ["total_rows", "matched_rows", "returned_rows", "offset"]) if (!Number.isSafeInteger(c[key]) || c[key] < 0) throw new Error("Invalid evidence query coverage.");
  if (!Array.isArray(result.rows) || result.rows.length !== c.returned_rows || c.returned_rows > query.limit || c.matched_rows > c.total_rows || c.offset !== query.offset || c.returned_rows > Math.max(0, c.matched_rows - c.offset)) throw new Error("Evidence query row conservation failed.");
  const next = c.offset + c.returned_rows < c.matched_rows ? c.offset + c.returned_rows : null;
  if (c.next_offset !== next || (next !== null && c.returned_rows === 0)) throw new Error("Evidence query cursor does not advance.");
  for (const row of result.rows) {
    object(row, "row"); allowed(row, ["subject_ref", "kind", "title", "details", "detail_truncations"], "row");
    for (const key of ["subject_ref", "kind", "title"]) if (typeof row[key] !== "string" || !row[key].length) throw new Error(`Invalid query row ${key}.`);
    object(row.details, "row details");
    if (!Array.isArray(row.detail_truncations) || row.detail_truncations.some(x => typeof x !== "string")) throw new Error("Invalid query detail coverage.");
  }
  if (c.detail_truncated !== (result.rows.some(row => row.detail_truncations.length > 0) || (result.context?.detail_truncations?.length > 0))) throw new Error("Evidence query detail coverage mismatch.");
  if (result.status === "complete" && (next !== null || c.detail_truncated)) throw new Error("Incomplete query was reported complete.");
  object(result.context, "query context");
  if (typeof result.evidence_boundary !== "string" || !result.evidence_boundary.length) throw new Error("Missing query evidence boundary.");
  const { result_sha256, ...body } = result;
  if (result_sha256 !== sha256TextHex(canonicalJson(body))) throw new Error("Evidence query result digest mismatch.");
  return result;
}

function object(value, name) { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid ${name}.`); }
function allowed(value, keys, name) { if (Object.keys(value).some(key => !keys.includes(key))) throw new Error(`Unexpected ${name} field.`); }
