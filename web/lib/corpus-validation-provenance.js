export const INTERFACE_CORPUS_VALIDATION = Object.freeze({
  schema: "deepbom.interface_boundary_corpus_validation_profile.v1",
  profileId: "quant-policy-boundary-public-50-2026-08-05",
  profileUrl: "https://deepbom.org/reference/quant-policy-boundary-validation.v1.json",
  profileSha256: "991eb12596486da9841022efaf3b9a13a79f632d5c6349e7ad3dd677a55338f9",
  manifestSha256: "9ec7fdf97c8fcc8e14d3b9b5a61ffafe05d0e90e5a58005cdca51a1d886f0540",
  verifierSchema: "deepbom.interface_boundary_corpus_review.v1",
  verifierImplementationSha256: "d316401f638ce32642c52afad59b33858326e852f9f63d94c84ebcaacf0ec5c6",
  reviewSha256: "617630133a28f10cb6d121e0a265e4d9fa051381357e0c32473eaf17aedc62c6",
  ledgerSha256: "3abd2f6148e150b6e1b6ed523c4f2ed037a972053a740ad2b0af1e8132577e12",
  analyzerWasmSha256: "95aec2444fa76c1ef6fd636c1003b7e01c78f79f24962a3180af0158fa454a1e",
  artifactCount: 50,
  externalParameterCount: 114,
  interpretationBoundary: "Pinned corpus validation evidence; it does not validate the current artifact or estimate ecosystem prevalence.",
});

export function interfaceCorpusValidationProperties() {
  const profile = INTERFACE_CORPUS_VALIDATION;
  return [
    ["deepbom:validation:interfaceCorpusProfileId", profile.profileId],
    ["deepbom:validation:interfaceCorpusProfileSha256", profile.profileSha256],
    ["deepbom:validation:interfaceCorpusManifestSha256", profile.manifestSha256],
    ["deepbom:validation:interfaceCorpusVerifierSchema", profile.verifierSchema],
    ["deepbom:validation:interfaceCorpusVerifierImplementationSha256", profile.verifierImplementationSha256],
    ["deepbom:validation:interfaceCorpusReviewSha256", profile.reviewSha256],
    ["deepbom:validation:interfaceCorpusLedgerSha256", profile.ledgerSha256],
    ["deepbom:validation:interfaceCorpusAnalyzerWasmSha256", profile.analyzerWasmSha256],
    ["deepbom:validation:interfaceCorpusArtifactCount", profile.artifactCount],
    ["deepbom:validation:interfaceCorpusExternalParameterCount", profile.externalParameterCount],
    ["deepbom:validation:interfaceCorpusBoundary", profile.interpretationBoundary],
  ];
}

export function interfaceCorpusValidationExternalReference() {
  const profile = INTERFACE_CORPUS_VALIDATION;
  return {
    type: "evidence",
    url: profile.profileUrl,
    comment: "Hash-pinned validation profile for the external interface quantization-contract corpus; not evidence about the currently analyzed artifact.",
    hashes: [{ alg: "SHA-256", content: profile.profileSha256 }],
  };
}
