import { exact, nonNegativeInteger, normalizeSha256 } from "./shared.js";

export function artifactIdentity(analysis, artifact, format) {
  const sha256 = normalizeSha256(artifact.sha256 || analysis.model_sha256 || analysis.artifact_sha256);
  if (!sha256) throw new Error("Artifact IR requires an artifact SHA-256.");
  const size = nonNegativeInteger(artifact.size ?? artifact.byte_length ?? analysis.file_size_bytes ?? analysis.file_size);
  return {
    filename: String(artifact.filename || analysis.filename || `model.${format}`),
    format,
    sha256,
    byte_length: size == null ? null : exact(BigInt(size)),
    artifact_set_sha256: normalizeSha256(artifact.artifact_set_sha256 || analysis?.artifact_set?.artifact_set_sha256),
  };
}
