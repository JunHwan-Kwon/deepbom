export function buildPublicShareAnalysis(analysis) {
  const sourceArtifactSha256 = String(analysis?.model_sha256 || "").toLowerCase();
  const redacted = structuredClone(analysis || {});
  redactExactArtifactIdentity(redacted, sourceArtifactSha256);
  redactPublicArtifactBundle(redacted.artifact_bundle);
  redacted.filename = "ARTIFACT-001";
  redacted.model_sha256 = "";
  redacted._markdown = "";
  redacted.markdown = "";
  if (redacted.target_profile) {
    redacted.target_profile.id = "PUBLIC-TARGET";
    redacted.target_profile.label = "Redacted planning target";
  }
  redactDeploymentFrontier(redacted.deployment_frontier);
  redactDeploymentDelta(redacted.deployment_delta);
  redactDelegationRepair(redacted.delegation_repair);
  return redacted;
}

function redactPublicArtifactBundle(bundle) {
  if (!bundle || typeof bundle !== "object") return;
  bundle.bundle_sha256 = "";
  if (Object.hasOwn(bundle, "model_source_sha256")) bundle.model_source_sha256 = "";
  for (const file of Array.isArray(bundle.files) ? bundle.files : []) {
    if (file && typeof file === "object") file.sha256 = "";
  }
  bundle.public_redaction = {
    schema: "deepbom.public_artifact_bundle_redaction.v1",
    artifact_identity_removed: true,
    canonical_bundle_sha256_removed: true,
    model_source_sha256_removed: true,
    member_content_sha256_removed: true,
    retained_fields: ["path", "byte_length", "role", "required", "binding_status"],
    interpretation_boundary: "This public projection preserves package structure and byte cardinality but cannot independently verify source member contents because private content digests are removed.",
  };
}

function redactExactArtifactIdentity(value, sourceArtifactSha256) {
  if (!sourceArtifactSha256 || !value || typeof value !== "object") return;
  const pending = [value];
  const visited = new Set();
  while (pending.length) {
    const current = pending.pop();
    if (!current || typeof current !== "object" || visited.has(current)) continue;
    visited.add(current);
    for (const [key, child] of Object.entries(current)) {
      if (typeof child === "string" && child.toLowerCase() === sourceArtifactSha256) current[key] = "";
      else if (child && typeof child === "object") pending.push(child);
    }
  }
}

function redactDelegationRepair(repair) {
  if (!repair || typeof repair !== "object") return;
  repair.artifact_filename = "ARTIFACT-001";
  repair.artifact_sha256 = "";
  repair.target_id = "PUBLIC-TARGET";
  repair.target_label = "Redacted planning target";
}

function redactDeploymentDelta(delta) {
  if (!delta || typeof delta !== "object") return;
  delta.baseline.filename = "ARTIFACT-BASELINE";
  delta.baseline.sha256 = "";
  delta.candidate.filename = "ARTIFACT-001";
  delta.candidate.sha256 = "";
  const targetIds = new Map((delta.target_deltas || []).map((target, index) => [
    target.target_id,
    `PUBLIC-PLANNING-TARGET-${String(index + 1).padStart(2, "0")}`,
  ]));
  const publicTargetId = (targetId) => targetIds.get(targetId) || "PUBLIC-PLANNING-TARGET";
  for (const [index, target] of (delta.target_deltas || []).entries()) {
    target.target_id = publicTargetId(target.target_id);
    target.target_label = `Redacted planning target ${index + 1}`;
  }
  if (delta.worst_relative_delta_target_id) delta.worst_relative_delta_target_id = publicTargetId(delta.worst_relative_delta_target_id);
}

function redactDeploymentFrontier(frontier) {
  if (!frontier || typeof frontier !== "object") return;
  frontier.artifact_sha256 = "";
  frontier.artifact_filename = "ARTIFACT-001";
  const targetIds = new Map((frontier.targets || []).map((target, index) => [
    target.target_id,
    `PUBLIC-PLANNING-TARGET-${String(index + 1).padStart(2, "0")}`,
  ]));
  const publicTargetId = (targetId) => targetIds.get(targetId) || "PUBLIC-PLANNING-TARGET";
  for (const [index, target] of (frontier.targets || []).entries()) {
    target.target_id = publicTargetId(target.target_id);
    target.target_label = `Redacted planning target ${index + 1}`;
  }
  for (const target of frontier.robust_coverage?.per_target || []) target.target_id = publicTargetId(target.target_id);
  for (const pair of frontier.target_divergence?.pairs || []) {
    pair.left_target_id = publicTargetId(pair.left_target_id);
    pair.right_target_id = publicTargetId(pair.right_target_id);
  }
  for (const intervention of frontier.interventions || []) {
    for (const target of intervention.per_target || []) target.target_id = publicTargetId(target.target_id);
  }
  for (const op of frontier.ops || []) {
    for (const estimate of op.target_estimates || []) estimate.target_id = publicTargetId(estimate.target_id);
  }
}

export function buildPublicShareIdentity(analysis) {
  const targetBound = ["tflite", "onnx"].includes(String(analysis?.format || "").toLowerCase());
  return {
    artifact_id: "ARTIFACT-001",
    filename: "ARTIFACT-001",
    format: analysis?.format || "",
    sha256: "redacted",
    target_id: targetBound ? "PUBLIC-TARGET" : "",
    target_label: targetBound ? "Redacted planning target" : "",
    target_profile_sha256: targetBound ? analysis?.target_profile?.profile_sha256 || "" : "",
    operator_count: analysis?.operator_count == null ? null : Number(analysis.operator_count),
    tensor_count: analysis?.tensor_count == null ? null : Number(analysis.tensor_count),
    total_macs: analysis?.total_macs == null ? null : Number(analysis.total_macs),
  };
}

export function publicShareTimestamp(now = new Date()) {
  return `${now.toISOString().slice(0, 10)}T00:00:00.000Z`;
}
