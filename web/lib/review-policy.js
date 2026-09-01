import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

export const REVIEW_POLICY_SCHEMA = "deepbom.review_policy.v1";
export const REVIEW_POLICY_RESULT_SCHEMA = "deepbom.review_policy_result.v1";
const SHA256 = /^[a-f0-9]{64}$/;
const LEVELS = ["informational", "low", "medium", "high"];
const RANK = new Map(LEVELS.map((level, index) => [level, index]));

export function validateReviewPolicy(document) {
  const body = copy(document);
  if (!body || body.schema !== REVIEW_POLICY_SCHEMA) throw new Error("Review policy schema is invalid.");
  const declared = body.policy_sha256 == null ? null : String(body.policy_sha256).toLowerCase();
  delete body.policy_sha256;
  if (!["observe", "enforce"].includes(body.mode)) throw new Error("Review policy mode must be observe or enforce.");
  body.fail_on = normalizeLevel(body.fail_on ?? "none", true);
  body.required_capabilities = uniqueStrings(body.required_capabilities || [], "required capability", 256);
  body.exceptions = Array.isArray(body.exceptions) ? body.exceptions.map(validateException) : [];
  const ids = body.exceptions.map((row) => row.id);
  if (new Set(ids).size !== ids.length) throw new Error("Review policy exception ids must be unique.");
  const digest = sha256TextHex(canonicalJson(body));
  if (declared && (!SHA256.test(declared) || declared !== digest)) throw new Error("Review policy SHA-256 is invalid.");
  return Object.freeze({ ...body, policy_sha256: digest });
}

export function evaluateReviewPolicy(envelope, policyDocument, {
  analyzerVersion = null,
  rulepackVersion = null,
  evaluatedAt = null,
  sourceFileSha256 = null,
} = {}) {
  if (envelope?.schema !== "deepbom.artifact_evidence_envelope.v1") throw new Error("Review policy requires a canonical artifact evidence envelope.");
  const policy = validateReviewPolicy(policyDocument);
  const now = evaluatedAt == null ? null : normalizeTimestamp(evaluatedAt, "evaluation timestamp");
  const capabilities = envelope.capabilities || {};
  const assessed = new Set(capabilities.assessed || []);
  const partial = new Set(capabilities.partial || []);
  const unavailable = new Set(capabilities.unavailable || []);
  const missingCapabilities = policy.required_capabilities.filter((id) => !assessed.has(id));
  const coverage = {
    status: missingCapabilities.length ? "incomplete" : "complete",
    required_capability_count: policy.required_capabilities.length,
    assessed_required_capability_count: policy.required_capabilities.length - missingCapabilities.length,
    missing_required_capabilities: missingCapabilities,
    partial_required_capabilities: policy.required_capabilities.filter((id) => partial.has(id)),
    unavailable_required_capabilities: policy.required_capabilities.filter((id) => unavailable.has(id)),
  };
  const identity = {
    artifact_sha256: envelope.identity?.sha256 || null,
    cpu_target_profile_sha256: envelope.cpu_cost_target_binding?.profile_sha256 || null,
    analyzer_version: analyzerVersion,
    rulepack_version: rulepackVersion,
  };
  const exceptionRows = policy.exceptions.map((exception) => exceptionState(exception, identity, now));
  const applicableIds = new Set(exceptionRows.filter((row) => row.status === "applicable").map((row) => row.finding_id));
  const findings = Array.isArray(envelope.findings) ? envelope.findings : [];
  const threshold = policy.fail_on === "none" ? Number.POSITIVE_INFINITY : RANK.get(policy.fail_on);
  const blocking = findings.filter((finding) => !applicableIds.has(finding.id)
    && RANK.get(normalizeLevel(finding.severity)) >= threshold);
  const findingPolicy = {
    status: blocking.length ? "block" : "pass",
    fail_on: policy.fail_on,
    finding_count: findings.length,
    exception_suppressed_finding_count: findings.filter((finding) => applicableIds.has(finding.id)).length,
    blocking_finding_count: blocking.length,
    blocking_finding_ids: blocking.map((finding) => finding.id),
  };
  const enforcementBlock = coverage.status !== "complete" || findingPolicy.status === "block";
  return {
    schema: REVIEW_POLICY_RESULT_SCHEMA,
    status: policy.mode === "observe" ? "observe" : enforcementBlock ? "block" : "pass",
    mode: policy.mode,
    evaluated_at: now,
    execution_status: "completed",
    coverage_status: coverage.status,
    finding_policy_status: findingPolicy.status,
    policy_identity: {
      schema: policy.schema,
      policy_sha256: policy.policy_sha256,
      source_file_sha256: validSha(sourceFileSha256) ? sourceFileSha256 : null,
    },
    evaluation_identity: identity,
    coverage,
    finding_policy: findingPolicy,
    exceptions: exceptionRows,
    evidence_envelope_sha256: envelope.envelope_sha256,
    interpretation_boundary: "Execution completion, analysis coverage, and finding policy are independent states. An exception applies only to its exact identity scope and cannot turn unavailable evidence into an assessed zero or a resolved finding.",
  };
}

function validateException(value, index) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Review policy exception ${index} is invalid.`);
  const row = {
    id: requiredText(value.id, `exception ${index} id`, 160),
    finding_id: requiredText(value.finding_id, `exception ${index} finding id`, 200),
    subject_ref: value.subject_ref == null ? null : requiredText(value.subject_ref, `exception ${index} subject ref`, 300),
    reason: requiredText(value.reason, `exception ${index} reason`, 1200),
    owner: requiredText(value.owner, `exception ${index} owner`, 300),
    created_at: normalizeTimestamp(value.created_at, `exception ${index} created_at`),
    expires_at: value.expires_at == null ? null : normalizeTimestamp(value.expires_at, `exception ${index} expires_at`),
    expires_when: value.expires_when == null ? null : requiredText(value.expires_when, `exception ${index} expires_when`, 600),
    scope: validateScope(value.scope, index),
  };
  if (!row.expires_at && !row.expires_when) throw new Error(`Review policy exception ${index} requires expires_at or expires_when.`);
  if (row.expires_at && Date.parse(row.expires_at) <= Date.parse(row.created_at)) throw new Error(`Review policy exception ${index} expires_at must be after created_at.`);
  return row;
}

function validateScope(value, index) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Review policy exception ${index} scope is required.`);
  const scope = {
    artifact_sha256: digestOrNull(value.artifact_sha256, `exception ${index} artifact SHA-256`),
    cpu_target_profile_sha256: digestOrNull(value.cpu_target_profile_sha256, `exception ${index} target-profile SHA-256`),
    analyzer_version: optionalText(value.analyzer_version, 100),
    rulepack_version: optionalText(value.rulepack_version, 100),
  };
  if (!scope.artifact_sha256) throw new Error(`Review policy exception ${index} must bind an artifact SHA-256.`);
  return scope;
}

function exceptionState(exception, identity, now) {
  const mismatches = [];
  for (const key of ["artifact_sha256", "cpu_target_profile_sha256", "analyzer_version", "rulepack_version"]) {
    if (exception.scope[key] != null && exception.scope[key] !== identity[key]) mismatches.push(key);
  }
  const expired = exception.expires_at != null && (now == null || Date.parse(now) >= Date.parse(exception.expires_at));
  return {
    id: exception.id,
    finding_id: exception.finding_id,
    subject_ref: exception.subject_ref,
    status: expired ? "expired" : mismatches.length ? "scope_mismatch" : "applicable",
    scope_mismatches: mismatches,
    expires_at: exception.expires_at,
    expires_when: exception.expires_when,
    reason: exception.reason,
    owner: exception.owner,
  };
}

function normalizeLevel(value, allowNone = false) {
  const level = String(value || "informational").toLowerCase();
  if (level === "info") return "informational";
  if ((allowNone && level === "none") || RANK.has(level)) return level;
  throw new Error(`Review policy severity ${value} is invalid.`);
}

function uniqueStrings(values, label, maximum) {
  if (!Array.isArray(values)) throw new Error(`Review policy ${label} list is invalid.`);
  const normalized = values.map((value) => requiredText(value, label, maximum));
  if (new Set(normalized).size !== normalized.length) throw new Error(`Review policy ${label} values must be unique.`);
  return normalized.sort();
}

function requiredText(value, label, maximum) {
  const text = String(value || "").trim();
  if (!text || text.length > maximum) throw new Error(`Review policy ${label} is invalid.`);
  return text;
}
function optionalText(value, maximum) { return value == null ? null : requiredText(value, "scope text", maximum); }
function normalizeTimestamp(value, label) {
  const time = Date.parse(String(value || ""));
  if (!Number.isFinite(time)) throw new Error(`Review policy ${label} is invalid.`);
  return new Date(time).toISOString();
}
function digestOrNull(value, label) {
  if (value == null) return null;
  const digest = String(value).toLowerCase();
  if (!SHA256.test(digest)) throw new Error(`Review policy ${label} is invalid.`);
  return digest;
}
function validSha(value) { return typeof value === "string" && SHA256.test(value); }
function copy(value) { return value == null ? null : JSON.parse(JSON.stringify(value)); }
