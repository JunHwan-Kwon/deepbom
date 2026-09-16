import { validateArtifactEvidenceEnvelope } from "./artifact-evidence-envelope.js";
import { ANALYZER_SEMANTIC_VERSION } from "./app-config.js";
import { canonicalJson } from "./report-utils.js";
import { sha256TextHex } from "./sha256-sync.js";

// SPDX 2.3 FILE-purpose packages: file-level license scanning is not performed.
// These preserve the measured SHA-256 without inventing the SHA-1 required by
// SPDX File records, an SPDX 3 AI profile, or runtime software dependencies.
export function buildSpdxArtifactDocument(envelope, { generatedAt = new Date().toISOString() } = {}) {
  const validation = validateArtifactEvidenceEnvelope(envelope);
  if (!validation.valid || !/^[a-f0-9]{64}$/.test(envelope.identity.sha256 || "")) {
    throw new Error("SPDX export requires a valid evidence envelope and artifact SHA-256.");
  }
  const created = new Date(generatedAt).toISOString().replace(/\.\d{3}Z$/, "Z");
  const creator = `Tool: DEEPBOM-${ANALYZER_SEMANTIC_VERSION}`;
  const identity = envelope.identity;
  const boundary = "Inventory of the selected serialized model artifact and verified external files only. Software dependencies, training data, runtime placement, model quality, and license conclusions are not inferred. Static findings are DEEPBOM annotations, not SPDX AI-profile fields.";
  const packages = [{ path: identity.filename, sha256: identity.sha256 }, ...(envelope.external_files || [])]
    .map((file, index) => ({
      SPDXID: index === 0 ? "SPDXRef-Artifact" : `SPDXRef-File-${sha256TextHex(file.path)}`,
      name: file.path,
      packageFileName: file.path,
      primaryPackagePurpose: "FILE",
      downloadLocation: "NOASSERTION",
      filesAnalyzed: false,
      checksums: [{ algorithm: "SHA256", checksumValue: file.sha256 }],
      licenseConcluded: "NOASSERTION",
      licenseDeclared: "NOASSERTION",
      copyrightText: "NOASSERTION",
      licenseComments: "DEEPBOM static artifact inspection does not determine the file's license or copyright.",
    }));
  const document = {
    spdxVersion: "SPDX-2.3",
    dataLicense: "CC0-1.0",
    SPDXID: "SPDXRef-DOCUMENT",
    name: `DEEPBOM artifact inventory: ${identity.filename}`,
    creationInfo: { created, creators: [creator] },
    documentDescribes: ["SPDXRef-Artifact"],
    packages,
    relationships: packages.slice(1).map((file) => ({
      spdxElementId: "SPDXRef-Artifact",
      relationshipType: "DEPENDS_ON",
      relatedSpdxElement: file.SPDXID,
      comment: "Verified serialized external-file dependency; not an inferred runtime software dependency.",
    })),
    annotations: [{
      annotationDate: created,
      annotationType: "OTHER",
      annotator: creator,
      comment: canonicalJson({
        schema: "deepbom.spdx_artifact_annotation.v1",
        identity,
        evidence_envelope_sha256: envelope.envelope_sha256,
        graph: envelope.graph,
        findings: envelope.findings,
        evidence_boundary: envelope.evidence_boundary,
      }),
    }],
    comment: boundary,
  };
  return {
    ...document,
    documentNamespace: `https://deepbom.org/spdx/${sha256TextHex(canonicalJson(document))}`,
  };
}
