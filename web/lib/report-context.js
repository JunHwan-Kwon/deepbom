export function buildFindingContext({
  runtimeEvidence = null,
  runtimeBasinResult = null,
  deepBomResult = null,
  deployCurvatureResult = null,
} = {}) {
  return {
    runtimeEvidence,
    runtimeBasinResult,
    deepBomResult,
    deployCurvatureResult,
  };
}

export function buildReportContext({
  identity = {},
  runtimeBenchmarkResults = [],
  runtimeEvidence = {},
  deepBomResult = null,
  perturbationResult = null,
  runtimeBasinResult = null,
  deployCurvatureResult = null,
  findingsContext = {},
  generatedAt = new Date().toISOString(),
  artifactIrContext = null,
} = {}) {
  return {
    identity,
    runtimeBenchmarkResults,
    runtimeEvidence,
    deepBomResult,
    perturbationResult,
    runtimeBasinResult,
    deployCurvatureResult,
    findingsContext,
    generatedAt,
    artifactIrContext,
    artifactIr: artifactIrContext?.artifact_ir || null,
  };
}

export function buildRegulatoryReportContext({
  reportContext = {},
  fileSizeBytes = 0,
  securityPosture = null,
} = {}) {
  return {
    ...reportContext,
    fileSizeBytes,
    securityPosture,
  };
}

export function buildBundleSummaryContext({
  files = [],
  moduleLog = [],
  capabilities = {},
  analysis = null,
  identity = {},
  user = {},
  generatedAt = new Date().toISOString(),
} = {}) {
  return {
    files,
    moduleLog,
    capabilities,
    analysis,
    identity,
    user,
    generatedAt,
  };
}

export function buildBundleManifestContext({
  files = [],
  moduleLog = [],
  capabilities = {},
  analysis = null,
  model = {},
  user = {},
  generatedAt = new Date().toISOString(),
  exportMode = "account_redacted_external",
} = {}) {
  return {
    files,
    moduleLog,
    capabilities,
    analysis,
    model,
    user,
    generatedAt,
    exportMode,
  };
}

export function buildRawEvidenceContext({
  identity = {},
  runtimeEvidence = {},
  weightIndicatorEvidence = {},
  securityPosture = null,
  findingsContext = {},
  artifactIrContext = null,
} = {}) {
  return {
    identity,
    runtimeEvidence,
    weightIndicatorEvidence,
    securityPosture,
    findingsContext,
    artifactIrContext,
    artifactIr: artifactIrContext?.artifact_ir || null,
  };
}

export function buildReportContextSet({
  analysis = null,
  identity = {},
  user = {},
  capabilities = {},
  files = [],
  moduleLog = [],
  runtimeBenchmarkResults = [],
  deepBomResult = null,
  perturbationResult = null,
  runtimeBasinResult = null,
  deployCurvatureResult = null,
  runtimeEvidence = {},
  weightIndicatorEvidence = {},
  fileSizeBytes = 0,
  securityPosture = null,
  generatedAt = new Date().toISOString(),
  artifactIrContext = null,
} = {}) {
  const findingsContext = buildFindingContext({
    runtimeEvidence,
    runtimeBasinResult,
    deepBomResult,
    deployCurvatureResult,
  });
  const reportContext = buildReportContext({
    identity,
    runtimeBenchmarkResults,
    runtimeEvidence,
    deepBomResult,
    perturbationResult,
    runtimeBasinResult,
    deployCurvatureResult,
    findingsContext,
    generatedAt,
    artifactIrContext,
  });
  return {
    findingsContext,
    reportContext,
    regulatoryReportContext: buildRegulatoryReportContext({
      reportContext,
      fileSizeBytes,
      securityPosture,
    }),
    bundleSummaryContext: buildBundleSummaryContext({
      files,
      moduleLog,
      capabilities,
      analysis,
      identity,
      user,
      generatedAt,
    }),
    bundleManifestContext: buildBundleManifestContext({
      files,
      moduleLog,
      capabilities,
      analysis,
      model: identity,
      user,
      generatedAt,
    }),
    rawEvidenceContext: buildRawEvidenceContext({
      identity,
      runtimeEvidence,
      weightIndicatorEvidence,
      securityPosture,
      findingsContext,
      artifactIrContext,
    }),
  };
}
