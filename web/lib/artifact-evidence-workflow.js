import { requireArtifactIrConsumerView } from "./artifact-ir-context.js";
import { buildArtifactEvidenceEnvelope, validateArtifactEvidenceEnvelope } from "./artifact-evidence-envelope.js";
import { buildReviewSummary } from "./review-summary.js";

// Channel-independent finalization. Acquisition and explicit execution rights
// remain in adapters; every renderer receives this same validated projection.
export function finishArtifactEvidenceWorkflow(artifactIrContext, envelopeOptions = {}) {
  const analysis = requireArtifactIrConsumerView(artifactIrContext?.primary_view);
  const envelope = buildArtifactEvidenceEnvelope(analysis, envelopeOptions);
  const validation = validateArtifactEvidenceEnvelope(envelope);
  if (!validation.valid) throw new Error(`Canonical evidence envelope validation failed: ${validation.errors.join(", ")}`);
  return { artifactIrContext, analysis, envelope, summary: buildReviewSummary({ analysis, envelope, artifactIrContext }) };
}
