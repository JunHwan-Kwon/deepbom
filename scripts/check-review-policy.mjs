import assert from "node:assert/strict";
import { evaluateReviewPolicy, validateReviewPolicy } from "../web/lib/review-policy.js";
import { canonicalJson } from "../web/lib/report-utils.js";
import { sha256TextHex } from "../web/lib/sha256-sync.js";

const artifact = "a".repeat(64);
const target = "b".repeat(64);
const policy = {
  schema: "deepbom.review_policy.v1",
  mode: "enforce",
  fail_on: "medium",
  required_capabilities: ["artifact_identity", "graph"],
  exceptions: [{
    id: "accepted-EA-1",
    finding_id: "EA-1",
    subject_ref: null,
    reason: "Accepted until the pinned deployment is replaced.",
    owner: "release-owner",
    created_at: "2026-08-01T00:00:00.000Z",
    expires_at: "2026-09-01T00:00:00.000Z",
    scope: { artifact_sha256: artifact, cpu_target_profile_sha256: target, analyzer_version: "1.96.0", rulepack_version: "1.96.0" },
  }],
};
const normalized = validateReviewPolicy(policy);
const body = { ...normalized };
delete body.policy_sha256;
assert.equal(normalized.policy_sha256, sha256TextHex(canonicalJson(body)));

const envelope = {
  schema: "deepbom.artifact_evidence_envelope.v1",
  identity: { sha256: artifact },
  cpu_cost_target_binding: { profile_sha256: target },
  capabilities: { assessed: ["artifact_identity", "graph"], partial: [], unavailable: [] },
  findings: [{ id: "EA-1", severity: "high" }],
  envelope_sha256: "c".repeat(64),
};
const pass = evaluateReviewPolicy(envelope, policy, { analyzerVersion: "1.96.0", rulepackVersion: "1.96.0", evaluatedAt: "2026-08-31T00:00:00.000Z" });
assert.equal(pass.status, "pass");
assert.equal(pass.finding_policy.exception_suppressed_finding_count, 1);
const evaluation = { analyzerVersion: "1.96.0", rulepackVersion: "1.96.0", evaluatedAt: "2026-08-31T00:00:00.000Z" };
for (const subject_ref of ["op:7", "nonexistent-subject"]) {
  const scoped = evaluateReviewPolicy(envelope, { ...policy, exceptions: [{ ...policy.exceptions[0], subject_ref }] }, evaluation);
  assert.equal(scoped.status, "block");
  assert.equal(scoped.exceptions[0].status, "unsupported_subject_scope");
}
const future = evaluateReviewPolicy(envelope, policy, { ...evaluation, evaluatedAt: "2026-07-01T00:00:00Z" });
assert.equal(future.exceptions[0].status, "not_yet_effective");
assert.equal(future.status, "block");
assert.equal(evaluateReviewPolicy(envelope, policy, { ...evaluation, evaluatedAt: null }).exceptions[0].status, "time_not_assessed");

const expired = evaluateReviewPolicy(envelope, policy, { analyzerVersion: "1.96.0", rulepackVersion: "1.96.0", evaluatedAt: "2026-09-01T00:00:00.000Z" });
assert.equal(expired.status, "block");
assert.equal(expired.exceptions[0].status, "expired");

const reducedCoverage = evaluateReviewPolicy({ ...envelope, capabilities: { assessed: ["artifact_identity"], partial: ["graph"], unavailable: [] } }, policy,
  { analyzerVersion: "1.96.0", rulepackVersion: "1.96.0", evaluatedAt: "2026-08-31T00:00:00.000Z" });
assert.equal(reducedCoverage.status, "block");
assert.equal(reducedCoverage.coverage_status, "incomplete");
assert.equal(reducedCoverage.finding_policy_status, "pass", "Coverage regression must not be counted as a resolved finding.");

const observed = evaluateReviewPolicy(envelope, { ...policy, mode: "observe", exceptions: [] },
  { analyzerVersion: "1.96.0", rulepackVersion: "1.96.0", evaluatedAt: "2026-08-31T00:00:00.000Z" });
assert.equal(observed.status, "observe");
assert.equal(observed.finding_policy_status, "block");
const conditionBound = validateReviewPolicy({
  ...policy,
  exceptions: [{ ...policy.exceptions[0], expires_at: null, expires_when: "The bound target profile is replaced." }],
});
assert.equal(conditionBound.exceptions[0].expires_when, "The bound target profile is replaced.");
assert.equal(evaluateReviewPolicy(envelope, conditionBound, evaluation).exceptions[0].status, "condition_not_assessed");
assert.throws(() => validateReviewPolicy({
  ...policy,
  exceptions: [{ ...policy.exceptions[0], expires_at: null }],
}), /expires_at or expires_when/);
assert.throws(() => validateReviewPolicy({ ...policy, policy_sha256: "d".repeat(64) }), /SHA-256/);
console.log("Review policy contract passed (identity-bound exceptions, independent coverage, observation/enforcement modes).");
