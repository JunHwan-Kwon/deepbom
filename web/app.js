import init, {
  analyze_tflite_for_target,
  compute_deployment_delta,
  compute_input_influence,
  compute_output_influence,
  compute_static_runtime_calibration,
  compute_weight_histogram,
  compute_kernel_haar_decomposition,
  compute_activation_haar,
  landscape_directions,
  synthetic_landscape_grid,
  compute_model_tomography,
  landscape_tomography,
  layer_landscape_grid,
  explore_tflite_redesign_pareto,
  project_tflite_redesign,
  runtime_guard,
  target_profiles,
} from "../pkg/tflite_wasm_audit.js";
import { prepareExternalDataFiles } from "./lib/onnx-external-data.js";
import {
  loadCustomTargets,
  resolveTargetSpec,
  customTargetStub,
  isCustomTargetId,
} from "./lib/custom-targets.js";
import { createCustomTargetEditor } from "./lib/custom-target-editor.js";
import {
  createLiteRtRuntimeLoader,
  loadOnnxBenchmark,
  loadTfliteBenchmark,
} from "./lib/runtime-module-loader.js";
import {
  ANALYZER_VERSION,
  deploymentFrontierMatchesTargetIds,
  deploymentFrontierTargetIds,
  RULEPACK_VERSION,
  MODULE_WORKSPACES,
  REPORT_WORKSPACES,
  populateReportTargetSelect,
  reportTargetControlCopy,
  reportTargetLabel as reportTargetLabelContract,
  resolveReportTargetBinding,
  resolveReportTargetId,
  WORKFLOW_ORDER,
} from "./lib/app-config.js";
import { copyTextToClipboard } from "./lib/clipboard.js";
import {
  detectAppSurface,
  isMedicalSurface,
  prepareAppSurface,
} from "./lib/app-surface.js";
import {
  artifactFilename,
  downloadBlob,
  downloadText,
  downloadTextArtifact,
  registerTextExport,
} from "./lib/download.js";
import { TEXT_EXPORT_ARTIFACTS } from "./lib/export-artifacts.js";
import { createExportContractController } from "./lib/export-contract-view.js";
import { buildPublicCycloneDxDocuments } from "./lib/public-cyclonedx-export.js";
import { syncPublicEvidencePackageButton } from "./lib/public-evidence-package.js";
import {
  buildEvidencePackageProfileFiles,
  evidencePackageProfile as resolveEvidencePackageProfile,
} from "./lib/evidence-package-profiles.js";
import { evidenceLevelProfile as resolveEvidenceLevelProfile } from "./lib/evidence-level-report.js";
import {
  ACCESS_REQUEST_PROFILES,
  applyGatedExportLabels,
  bundleUserIdentityForUser,
  combineModuleResults,
  capabilitiesForUser,
  accessLabel,
  moduleAccessStatesFor,
  moduleTabStatusTextFor,
  moduleWorkflowDescription,
  providerLabel,
  requestDraftTitle,
  roleLabel,
  selectableModuleIdFor,
  profileIdForCapability,
} from "./lib/auth-labels.js";
import { applyAuthConfigView, applyAuthModeView } from "./lib/auth-ui.js";
import {
  appendRegulatoryBundleModule,
  buildBundleEnvelope,
  engineeringBundleItems,
  initialEvidenceBundleProgress,
  REGULATORY_BUNDLE_MODULE_SPECS,
  renderBundleScope,
  regulatoryBundleItems,
} from "./lib/bundle.js";
import {
  appendBenchmarkRow,
  p99EvidenceForSampleCount,
  runtimeReadinessSignals,
  updateBenchmarkRow,
} from "./lib/benchmark-ui.js";
import {
  histogramRow,
  rooflineTableRows,
  stageCard,
  summaryMetricCards,
  topMacRows,
} from "./lib/audit-ui.js";
import {
  artifactOverviewHeader,
  artifactOverviewPanels,
} from "./lib/artifact-overview.js";
import { buildRepresentableKernelChannelCheck } from "./lib/quantization-contract-summary.js";
import {
  canvasToPngBytes,
  deepBomMetric,
  insightCard,
  protocolBlock,
  runtimeSignal,
  withBusyButton,
} from "./lib/dom.js";
import {
  buildDualRadialSvg,
  drawLandscapeCanvas,
  drawMvDepth,
  drawMvFilter,
} from "./lib/research-visuals.js";
import {
  assessedOpLogicalBytes,
  buildGraphIndex,
  opLogicalL1Ratio,
  opLogicalRowPayloadBytes,
  normalizeUnassessedCostValues,
  opSteadyStateUs,
  opMatchesSearch,
} from "./lib/analysis.js";
import {
  buildTensorInventory,
  classifyTensorRoles,
  tensorQuantizationMode,
} from "./lib/tensor-inventory.js";
import { applyProtectedXnnpackSelectorEvidence } from "./lib/xnnpack-selector-evidence.js";
import { applyProtectedOrtCompatibilityEvidence } from "./lib/ort-compatibility-evidence.js";
import { applyProtectedTfliteDelegateCompatibilityEvidence } from "./lib/tflite-delegate-compatibility.js";
import { buildTensorRtStaticPreflight } from "./lib/tensorrt-static-preflight.js";
import { buildOnDeviceLlmContract } from "./lib/on-device-llm-contract.js";
import {
  clampInt,
  cloneTypedArray,
  formatBytes,
  formatDrift,
  formatNumber,
  formatPercent,
  formatPercent1,
  formatUs,
  maxBy,
  padOp,
  score100,
  shortError,
} from "./lib/format.js";
import { renderFindings, renderFindingsCalibration } from "./lib/findings-viewer.js";
import { updateFormatSpecificAuditLabels } from "./lib/format-audit-tabs.js";
import {
  driftSeverity,
  severityRank,
  statusBlocked,
  statusForBackendCoverage,
  statusForCosineDistance,
  statusForEntropy,
  statusForMaxDrift,
  statusForRmsDrift,
  statusForTop1Flip,
  statusFromSeverityLabel,
  statusInfo,
} from "./lib/status.js";
import {
  deploymentSensitivityProtocolGroups,
  perturbationProtocolGroups,
} from "./lib/protocols.js";
import { registerServiceWorker } from "./lib/service-worker.js";
import {
  artifactUuidFromSha256,
  bindPublicAuditPrintButton,
  buildPublicEngineeringReportHtml,
  buildSessionPrivacy,
  buildSessionReportContextSet,
  syncPublicPrintButton,
} from "./lib/session-evidence.js";
import { formatAuditButtonLabel, formatEvidenceScope, renderStagedArtifactContext } from "./lib/format-evidence-scope.js";
import { renderAuditClaimBoundaryView, renderInsightDashboardView, syncFormatWorkflowVisibilityView } from "./lib/format-workflow-ui.js";
import { bindPublicVerificationButton, syncPublicVerificationButton } from "./lib/public-verification-ui.js";
import { renderKernelInspector } from "./lib/kernel-inspector.js";
import { createRuntimeProfileModal } from "./lib/runtime-profile-modal.js";
import { renderTensorArenaViewer } from "./lib/arena-viewer.js";
import {
  buildCanonicalPackageDigest,
  jsonForDownload,
  validatePackageAttestation,
  zipBinaryFile,
  zipTextFile,
} from "./lib/report-utils.js";
import { appendPublicKeySignature } from "./lib/public-key-signature.js";
import { renderExternalDataStatus } from "./lib/onnx-external-data-status.js";
import { buildStaticAuditMarkdown } from "./lib/markdown-report.js";
import {
  adminShortcutCard,
  renderAccountCapabilityList,
  renderRequestList,
  requestCard,
  requestLoading,
} from "./lib/account-ui.js";
import {
  buildGraphEvidenceMaps,
  graphOpRow,
  graphSvgText,
  renderGraphMapContent,
  renderOpDetailPanel,
} from "./lib/graph-ui.js";
import { getArtifactIrContext } from "./lib/artifact-ir-context.js";
import { buildSingleFileArtifactSet } from "./lib/artifact-set.js";
import { exportGraphVisualization } from "./lib/graph-export.js";
import {
  buildModelIdentity,
  detectModelFormat,
  estimateModelAnalysis,
  formatMeasuredAudit,
  modelExportAvailability,
  modelFormatAdapter,
  modelFormatGate,
  modelReportBindingMatchesAnalysis as reportBindingMatchesAnalysis,
  modelSupportsCapability,
  inspectModelFile,
  selectedModelCopy,
  stagedModelCopy,
} from "./lib/model-file.js";
import { readMetadataModelFile } from "./lib/metadata-model-adapters.js";
import { readCoreMlModelFile } from "./lib/coreml-metadata-adapter.js";
import { inspectArtifactBundle, readArtifactBundle } from "./lib/artifact-bundle.js";
import { initPrivacyAgreementUi } from "./lib/privacy-ui.js";
import { closeModal, installModalKeyboard, openModal } from "./lib/modal-accessibility.js";
import {
  backendCandidates,
  benchmarkErrorStatus,
  browserBucket,
  deleteTensors,
  RUNTIME_OK,
  resolveFakeInputShape,
  selectWasmCalibrationResult,
  runtimeGuardMessage,
  runtimeGuardTitle,
} from "./lib/runtime.js";
import {
  AGREEMENT_POLICY_VERSION,
  readAgreementRecord,
  readAuditTimings,
  readResearchConsent,
  readResearchConsentRecord,
  readSavedTarget,
  recordAuditTiming,
  syncResearchConsent,
  writeSavedTarget,
  writeResearchConsent,
} from "./lib/storage.js";
import { createWorkflowController } from "./lib/workflow-controller.js";
import { createAnalysisDepthMode } from "./lib/analysis-depth-mode.js";
import { createOfflineDeviceController } from "./lib/offline-device-controller.js";
import {
  installWorkspaceNavigation,
  syncTabSelection,
} from "./lib/workspace-navigation.js";
import { renderFormatCapabilityMatrix } from "./lib/format-capability-view.js";
import { renderRuntimeEvidenceClosure, runtimeEvidenceSidecarForDownload } from "./lib/runtime-evidence-closure.js";
import { installRuntimeEvidenceController } from "./lib/runtime-evidence-controller.js";
import {
  buildBenchmarkTelemetryPayload,
  buildStructureTelemetryPayload,
  postJson,
} from "./lib/telemetry.js";
import { createPerformanceVisualController } from "./lib/performance-visuals.js";
import { createCoreIsolationController } from "./lib/core-isolation-view.js";
import { createAuditProgressController } from "./lib/audit-progress.js";
import { createStaticAuditWorkerClient } from "./lib/static-audit-worker-client.js";
import { buildCpuCostTargetBinding } from "./lib/cpu-target-binding.js";
import { createExplorerDecisionView } from "./lib/explorer-decision-view.js";
import { createExplorerRedesignController } from "./lib/explorer-redesign.js";
import { createQuantEvidenceController } from "./lib/quant-evidence-view.js";
import { createNodeViewController } from "./lib/node-view.js";
import { renderExecutionPlacementView as renderExecutionPlacementViewBase } from "./lib/execution-placement-view.js";
import {
  acceleratorProfilesForAnalysis,
  renderAcceleratorProfileSwitcher,
} from "./lib/accelerator-profile-switcher.js";
import { createCalibrationValidationController } from "./lib/calibration-validation-view.js";
import { createDeploymentFrontierController } from "./lib/deployment-frontier.js";
import { createDeploymentDeltaController } from "./lib/deployment-delta.js";
import { createDelegationRepairController } from "./lib/delegation-repair.js";
import { createQuantizationResearchSuite } from "./lib/quantization-research-suite.js";
import {
  collectFullGraph,
  collectNeighborhood,
  layoutFoldedGraph,
  layoutNeighborhood,
} from "./lib/graph-layout.js";
import { sha256FileHex, sha256Hex } from "./lib/hash.js";
import { PUBLIC_SAMPLE_MODELS } from "./lib/sample-models.js";
import { installPublicSampleLibrary } from "./lib/sample-library.js";
import {
  aggregateGrids,
  aggregateHessian,
  applyLandscapePatch,
  compareOutputArrays,
  computeHessian2D,
  computeRadialProfileSEM,
  computeDeployBasinProxy,
  computeDirectionalCurvature,
  decisionMargin,
  deployProbeConsistencyWarning,
  driftDeltaSummary,
  driftLooksIdentical,
  createResearchInputData,
  createHaarPatternData,
  computePatternInputStats,
  HAAR_PATTERN_SPECS,
  HAAR_SWEEP_CONFIG,
  haarAmplitudeSweepLevels,
  haarSensitivityProfile,
  haarTranslationProfile,
  haarRotationProfile,
  haarPhaseProfile,
  haarAmplitudeSweepProfile,
  haarPolarityProfile,
  normalizedZeroPointForDtype,
  computeStaticRuntimeAlignment,
  COCO_HAAR_PRIOR,
  isPerturbationMode,
  linspaceArr,
  outputDriftEnsembleCopy,
  outputDriftProjectionCopy,
  outputDriftProfileForAnalysis,
  perturbTypedArray,
  perturbationOptions,
  perturbationStatsFromMode,
  perturbModelWeightBytes,
  perturbationProtocolStatus,
  requantRatio,
  robustnessScoreFromDrift,
  runtimeAttemptInterpretation,
  runtimeBasinProtocolStatus,
  sanitizeCurvature,
  sanitizeDrift,
  sanitizeRuntimeAttempt,
  sanitizeTimingVariance,
  selectWeightPerturbationCandidates,
  summarizeProbeResult,
  summarizeOutputDriftProjectionEnsemble,
  subtractCenter,
  timingVarianceSummary,
  timingContextNote,
} from "./lib/research.js";
import { createZipBlob } from "./lib/zip.js";
import { visualPngSpecs } from "./lib/visual-export.js";
import { bindAppElements } from "./lib/elements.js";
import { installQuantEvidenceChains, renderQuantEvidenceChains } from "./lib/quant-evidence-chains.js";
import { ensureQuantResearchCoverage } from "./lib/quant-research-applicability.js";
import { createGraphWorkspace } from "./lib/app-graph-workspace.js";
import { createEvidenceCursor } from "./lib/evidence-cursor.js";
import { buildOnnxRuntimeShapeBinding } from "./lib/onnx-runtime-shape-binding.js";
import { createDeepBomWorkspace } from "./lib/app-deepbom-workspace.js";
import {
  buildAuditSnapshot,
  saveAuditSnapshot,
  recordReportArtifact,
  listSnapshots,
  deleteSnapshot,
  buildComparisonReport,
  updateSnapshotNote,
  readReportHistorySettings,
  writeReportHistorySettings,
  pruneAuditSnapshots,
} from "./lib/report-store.js";
import { renderArtifactDiffWorkspace } from "./lib/artifact-diff-view.js";
import { createEvidenceWhyDrawer } from "./lib/evidence-why-drawer.js";
import { buildReviewState, buildSelfContainedReviewHtml } from "./lib/review-export.js";
import {
  buildNodeEdgeEvidenceOverlayTemplate,
  validateNodeEdgeEvidenceOverlay,
} from "./lib/node-edge-evidence-overlay.js";

function mk(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }

let graphWorkspace = null;
const evidenceCursor = createEvidenceCursor();
function jumpToGraphOp(opIndex) {
  return graphWorkspace?.jumpToGraphOp(opIndex);
}
function previewDelegationScenario(scenario) {
  return graphWorkspace?.previewDelegationScenario(scenario);
}
function clearGraphScenarioPreview() {
  return graphWorkspace?.clearGraphScenarioPreview();
}
function switchExplorerTab(tab) {
  return graphWorkspace?.switchExplorerTab(tab);
}
function selectGraphOp(analysis, opIndex, options = {}) {
  return graphWorkspace?.selectGraphOp(analysis, opIndex, options);
}

const appSurface = detectAppSurface(window.location);
prepareAppSurface(appSurface, document);
installQuantEvidenceChains(document, { onOpenOp: (opIndex) => jumpToGraphOp(opIndex) });

const appElements = bindAppElements();
const {
  topbar,
  statusEl,
  sessionAnchor,
  authWidget,
  authOpen,
  authUser,
  authRole,
  authName,
  authEmail,
  authProvider,
  authVerify,
  resendVerify,
  accountOpen,
  adminOpen,
  authLogout,
  authBackdrop,
  authClose,
  authMessage,
  authLoginTab,
  authSignupTab,
  authPasswordTabs,
  authForm,
  authDivider,
  authNameWrap,
  authFormName,
  authFormEmail,
  authFormPassword,
  authSubmit,
  googleLogin,
  runtimeProfileBackdrop,
  runtimeProfileClose,
  runtimeProfileTitle,
  runtimeProfilePreview,
  runtimeProfileForm,
  runtimeProfileVersion,
  runtimeProfileBackend,
  runtimeProfileOptimization,
  runtimeProfileExecutionMode,
  runtimeProfileCollectedAt,
  runtimeProfileCaptureLabel,
  runtimeProfileCapture,
  runtimeProfileBinarySha,
  runtimeProfileBuild,
  runtimeProfileStatus,
  runtimeProfileCancel,
  runtimeProfileImport,
  accountBackdrop,
  accountClose,
  accountProfileStatus,
  accountCapabilityCount,
  accountCapabilityList,
  accountRequestForm,
  accountRequestProfile,
  accessProfileInfo,
  accountRequestTitle,
  accountRequestMessage,
  accountRequestSubmit,
  accountRequestStatus,
  accountRequestCount,
  accountRequestList,
  consentStatusBadge,
  consentMetadataLog,
  withdrawConsentBtn,
  restoreConsentBtn,
  localReportsList,
  saveReportHistory,
  reportHistoryRetention,
  compareReportsBtn,
  clearReportsBtn,
  comparisonResult,
  comparisonPre,
  comparisonVisual,
  evidenceWhyDrawer,
  evidenceWhyTitle,
  evidenceWhySubtitle,
  evidenceWhyBody,
  copyEvidenceWhy,
  closeEvidenceWhy,
  downloadComparison,
  closeComparison,
  adminPanel,
  adminRefresh,
  adminRequestList,
  adminBenchList,
  adminBenchRefresh,
  adminBenchStatusFilter,
  fileInput,
  artifactBundleInput,
  onnxExternalDataControl,
  onnxExternalDataInput,
  onnxExternalDataDirectoryInput,
  onnxExternalDataStatus,
  dropzone,
  targetSelect,
  runAudit,
  sampleModelSelect,
  trySampleModel,
  modelPlan,
  formatCapabilityPanel,
  workflowConsole,
  targetStaleNotice,
  selectedModelName,
  selectedModelMeta,
  analysisEstimate,
  analysisEstimateNote,
  analysisPlanStatus,
  auditProgress,
  auditProgressBar,
  auditProgressLabel,
  workflowMode,
  evidenceCursorStatus,
  workflowNextAction,
  workflowNextDetail,
  workflowSteps,
  workflowModuleSteps,
  outputModuleSelector,
  moduleRunConsole,
  actions,
  moduleAccessStatus,
  moduleTabs,
  moduleRunPanels,
  modulePanels,
  perturbationPanelNote,
  perturbationPanelAction,
  perturbationResultPanel,
  perturbationStatus,
  perturbationGrid,
  haarSweepPanel,
  lossLandscapePanel,
  modelViewerPanel,
  modelViewerBtn,
  lossTomoPanel,
  perturbationProtocols,
  perturbationNotes,
  runtimeBasinPanelNote,
  runtimeBasinResultPanel,
  runtimeBasinStatus,
  runtimeBasinGrid,
  runtimeBasinProtocols,
  runtimeBasinNotes,
  deploymentSensitivityPanelNote,
  deploymentSensitivityPanelAction,
  deploymentSensitivityResultPanel,
  deploymentSensitivityStatus,
  deploymentSensitivityGrid,
  deploymentSensitivityProtocols,
  deploymentSensitivityNotes,
  auditWorkbench,
  auditTabs,
  auditFocusTitle,
  auditFocusCopy,
  auditApplicabilityBoundary,
  auditApplicabilityStatus,
  auditApplicabilityTitle,
  auditApplicabilityReason,
  auditApplicabilityRequired,
  summary,
  agreementBackdrop,
  privacyAgree,
  researchConsent,
  acceptAgreement,
  offlineTestModels,
  offlineTestStatus,
  calibrationValidationInput,
  calibrationValidationSelect,
  calibrationValidationDownload,
  calibrationValidationStatus,
  calibrationValidationResult: calibrationValidationResultEl,
  deviceRegistryList,
  deviceRegistryRefresh,
  targetSwitcherBar,
  acceleratorSwitcherBar,
  insightDashboard,
  modelGlancePanel,
  perfVisuals,
  perfVisualTitle,
  perfVisualSubtitle,
  perfVisualStatus,
  llmEvidencePanel,
  visualPanels,
  macFlame,
  macTopList,
  bottleneckFlame,
  bottleneckList,
  targetCompareGrid,
  deploymentFrontierPanel,
  deploymentFrontierStatus,
  deploymentFrontierSummary,
  deploymentFrontierBody,
  downloadDeploymentFrontier,
  deploymentDeltaPanel,
  deploymentDeltaStatus,
  deploymentDeltaSummary,
  deploymentDeltaBody,
  pinDeploymentBaseline,
  clearDeploymentBaseline,
  downloadDeploymentDelta,
  chainFlow,
  xnnpackFallbackMap,
  xnnpackFallbackCount,
  xnnpackBreakTable,
  delegationRepairPanel,
  delegationRepairStatus,
  delegationRepairSummary,
  delegationRepairBody,
  downloadDelegationRepair,
  quantHeatmap,
  quantHeatmapCount,
  quantStateBreakdown,
  quantExposureMap,
  quantScaleScatter,
  quantRiskTable,
  quantRiskCount,
  quantHoleList,
  quantHoleCount,
  rooflineChart,
  rooflineChartLegend,
  coreIsolationPanel,
  coreIsolationStatus,
  coreIsolationControls,
  coreIsolationSummary,
  coreIsolationChart,
  coreIsolationBody,
  coreIsolationBoundary,
  stageMemoryMix,
  perfTimeline,
  perfTimelineSubtitle,
  inferencePanel,
  runtimeStatus,
  backendSelect,
  warmupInput,
  runsInput,
  runInference,
  runtimeNotes,
  benchmarkWrap,
  benchmarkBody,
  graphExplorer,
  resourceMapPanel,
  blocksExplorerPanel,
  quantEvidencePanel,
  nodeViewPanel,
  cacheExplorerPanel,
  graphDetailLayout,
  redesignPanel,
  graphOpsView,
  graphScenarioBanner,
  graphScenarioLabel,
  graphScenarioDetail,
  clearGraphScenario,
  opParetoBar,
  opTimeline,
  xnnSegmentBar,
  tensorMemoryTimeline,
  layeredViewPanel,
  graphStats,
  graphSearch,
  graphDepth,
  graphZoomOut,
  graphZoomIn,
  graphFit,
  downloadGraphSvg,
  graphMapStatus,
  graphModeHint,
  graphMapSvg,
  graphOpBody,
  graphOpHead,
  tensorExplorerPanel,
  kernelInspectorPanel,
  explorerExecutionPlacementPanel,
  kernelInspectorSearch,
  kernelInspectorSummary,
  runtimeAssignmentComparison,
  kernelBoundaryInventory,
  kernelInspectorBody,
  runtimeAssignmentInput,
  downloadRuntimeAssignmentTemplate,
  runtimeEvidenceClosure,
  downloadRuntimeCapturePlan,
  runtimeAssignmentStatus,
  clearRuntimeAssignment,
  nodeEvidenceOverlayInput,
  downloadNodeEvidenceOverlayTemplate,
  clearNodeEvidenceOverlay,
  nodeEvidenceOverlayStatus,
  tensorStatsBar,
  tensorBody,
  tensorSearch,
  opFilterBar,
  opFilterCount,
  opDetail,
  opNavPrev,
  opNavNext,
  opNavLabel,
  diagramSection,
  stageCount,
  stageStrip,
  tables,
  histogramBody,
  topMacBody,
  rooflineBody,
  reportPreview,
  reportPreviewTitle,
  reportPreviewStatus,
  copyReportBtn,
  registerVerifyBtn,
  reportTargetSelect,
  reportTargetStatus,
  reportTargetAnalyzeBtn,
  regulatoryReportPreview,
  regulatoryReportPreviewTitle,
  regulatoryReportPreviewStatus,
  downloadMarkdown,
  downloadReviewHtml,
  printPublicReport,
  downloadPublicVerificationManifest,
  downloadRegulatoryReport,
  downloadRawData,
  downloadCsv,
  downloadMermaid,
  downloadVisualPngs,
  downloadEngineeringBundle,
  evidencePackageProfile,
  evidencePackageLevel,
  downloadPublicBundle,
  engineeringBundleNote,
  engineeringBundleScope,
  downloadEvidenceBundle,
  evidenceBundleNote,
  evidenceBundleScope,
  findingsPanel,
  findingsBody,
  runDeepBom,
  requestDeepBomAccess,
  deepBomPanel,
  deepBomStatus,
  deepBomGrid,
  deepBomProtocols,
  deepBomCaveats,
  deepBomNotes,
  downloadDeepBom,
  deepBomAccessNote,
} = appElements;

const evidenceWhyController = createEvidenceWhyDrawer({
  root: evidenceWhyDrawer,
  title: evidenceWhyTitle,
  subtitle: evidenceWhySubtitle,
  body: evidenceWhyBody,
  copyButton: copyEvidenceWhy,
  closeButton: closeEvidenceWhy,
});

function renderEvidenceCursorStatus(state = evidenceCursor.get()) {
  if (!evidenceCursorStatus) return;
  const coordinates = [
    state.finding_id ? `finding ${state.finding_id}` : "",
    state.op_index != null ? `op #${String(state.op_index).padStart(3, "0")}` : "",
    state.tensor_index != null ? `tensor T${state.tensor_index}` : "",
    state.runtime_node_id ? `runtime ${state.runtime_node_id}` : "",
    state.report_anchor ? `report ${state.report_anchor}` : "",
  ].filter(Boolean);
  evidenceCursorStatus.textContent = coordinates.length
    ? `Evidence cursor: ${coordinates.join(" · ")}`
    : state.artifact_sha256
      ? "Evidence cursor: artifact bound; no item selected"
      : "Evidence cursor: no selection";
}

evidenceCursor.subscribe((state) => renderEvidenceCursorStatus(state));
renderEvidenceCursorStatus();

const auditProgressController = createAuditProgressController({
  root: auditProgress,
  bar: auditProgressBar,
  label: auditProgressLabel,
});
const staticAuditWorkerClient = createStaticAuditWorkerClient();
const liteRtRuntime = createLiteRtRuntimeLoader({
  onStatus: (message) => { runtimeStatus.textContent = message; },
});
const ensureLiteRtRuntime = liteRtRuntime.ensure;

let current = null;
let currentArtifactIrContext = null;

function currentAnalysisView() {
  return currentArtifactIrContext?.primary_view || current;
}

function artifactIrBackedView(analysis) {
  return analysis === current ? currentAnalysisView() : analysis;
}
let selectedAcceleratorProfileId = "";
let selectedPlacementProfileIds = [];
let currentDeploymentFrontier = null;
let currentDeploymentDelta = null;
let currentDelegationRepair = null;
let deploymentDeltaBaseline = null;
let currentModelBytes = null;
let currentModelPayloadLoaded = false;
let currentLowNormStatMap = new Map(); // opIndex -> { low_norm, total } for conv-family ops
let currentGraphMode = "deploy";    // "raw" | "deploy" | "stage"
let activeGraphScenario = null;
let currentTopologyAnnotations = null;
let currentTensorFilter = "";       // tensor explorer: search term
let currentTensorRoleFilter = "";   // tensor explorer: all|constant|activation|fanout|quant
let currentKernelFilter = "all";
let runtimeAssignmentEvidence = null;
let nodeEdgeEvidenceOverlay = null;
let productionInterfaceContract = null;

function clearNodeEdgeEvidenceOverlayState() {
  nodeEdgeEvidenceOverlay = null;
  if (nodeEvidenceOverlayStatus) nodeEvidenceOverlayStatus.textContent = "No external node/edge evidence overlay imported.";
  if (clearNodeEvidenceOverlay) clearNodeEvidenceOverlay.hidden = true;
}
let pendingRuntimeProfile = null;
let pendingModelFile = null;
let pendingModelInspection = null;
let pendingArtifactBundleFiles = [];
let pendingArtifactBundleName = "";
let sampleLibraryController = null;
let currentExternalDataFiles = [];
let pendingPublicSampleCompanions = null;
let selectedOpIndex = null;
let activeTargetId = "";
let reportTargetRequestedId = "";
let targetAnalysisTransition = null;
let targetAnalysisTransitionPromise = null;
const targetAnalysisCache = new Map();
let opTableSortKey = "";
let opTableSortDir = -1;
let opFilterBound = "";
let opFilterXnn = "";
let opFilterQuant = "";
let graphMapBounds = null;
let graphViewBox = null;
let graphDrag = null;
let graphRenderToken = 0;
let runtimeGuardCode = null;
let currentFilename = "";
let targetProfiles = [];
let customTargetSpecs = [];

function rebuildCurrentArtifactIrContext(analysis = current) {
  if (!analysis) {
    currentArtifactIrContext = null;
    return null;
  }
  currentArtifactIrContext = getArtifactIrContext(analysis, {
    filename: analysis.filename || currentFilename || "model",
    format: analysis.format,
    sha256: analysis.model_sha256,
    size: analysis.file_size_bytes ?? analysis.file_size ?? currentModelBytes?.length ?? null,
    artifact_set_sha256: analysis.artifact_set?.artifact_set_sha256 || null,
  }, { runtimeEvidence: runtimeAssignmentEvidence });
  return currentArtifactIrContext;
}
let customTargetEditor = null;
function openCustomTargetEditor(existingId = null) {
  customTargetEditor ||= createCustomTargetEditor({
    getBuiltInProfiles: () => targetProfiles.filter((profile) => !isCustomTargetId(profile.id)),
    onSaved: () => {
      populateTargetProfiles();
      renderTargetSwitcher();
    },
  });
  customTargetEditor.open(existingId);
}
const TARGET_PILL_LABELS = Object.freeze({
  rpi4_a72: "RPi4 A72",
  android_mid_a55: "A55 / 32 KiB",
  android_mid_a55_l1_16k: "A55 / 16 KiB",
  android_mid_a55_l1_64k: "A55 / 64 KiB",
  zynq_ultrascale_plus_a53: "Zynq A53",
  android_flagship_x3_a715: "X3/A715",
  x86_avx2: "x86 AVX2",
  x86_sse4: "x86 SSE4",
  wasm_simd: "WASM SIMD",
});
let authMode = "login";
let currentAuthUser = null;
let authConfigState = { enabled: false, google: false, password: false };
let accessGrantState = null;
let googleAuthPopup = null;
let googleAuthCloseTimer = 0;
let googleAuthCompletionPromise = null;
let accountProfileState = null;
let deepBomModule = null;
let deepBomResult = null;
let perturbationResult = null;
let runtimeBasinResult = null;
let deployCurvatureResult = null;
let preprocessingConsequenceResult = null;
let calibrationValidationResult = null;
let runtimeBenchmarkResults = [];
let workflowController = null;
let activeModule = "engineering_report";
let structureTelemetryState = null;
let engineeringBundleProgress = null;
let evidenceBundleProgress = null;
const calibrationValidationController = createCalibrationValidationController({
  input: calibrationValidationInput,
  selectButton: calibrationValidationSelect,
  downloadButton: calibrationValidationDownload,
  status: calibrationValidationStatus,
  result: calibrationValidationResultEl,
  getArtifactSha256: () => current?.model_sha256 || null,
  getAnalysis: () => current,
  onResult: (result) => {
    calibrationValidationResult = result;
    renderReportPanel();
  },
  onDownload: (result) => downloadText(
    currentArtifactFilename("representative_dataset_validation_ledger.json"),
    jsonForDownload(result),
    "application/json",
  ),
});
const medicalReportSurface = isMedicalSurface(appSurface);
const performanceVisualController = createPerformanceVisualController({
  elements: {
    perfVisualTitle,
    perfVisualSubtitle,
    perfVisualStatus,
    llmEvidencePanel,
    visualPanels,
    macFlame,
    macTopList,
    bottleneckFlame,
    bottleneckList,
    targetCompareGrid,
    chainFlow,
    xnnpackFallbackMap,
    xnnpackFallbackCount,
    xnnpackBreakTable,
    quantHeatmap,
    quantHeatmapCount,
    quantStateBreakdown,
    quantExposureMap,
    quantScaleScatter,
    quantRiskTable,
    quantRiskCount,
    quantHoleList,
    quantHoleCount,
    rooflineChart,
    rooflineChartLegend,
    stageMemoryMix,
    perfTimeline,
    perfTimelineSubtitle,
  },
  getContext: () => ({
    current: currentAnalysisView(),
    currentModelBytes,
    currentFilename,
    targetProfiles,
    activeAuditTab: getActiveAuditTab(),
    selectedTargetId: selectedTargetId(),
    selectedTargetProfile: selectedTargetProfile(),
  }),
  analyzeForTarget: (bytes, filename, targetId) =>
    analyze_tflite_for_target(bytes, filename, resolveTargetSpec(targetId, customTargetSpecs)),
  jumpToGraphOp,
});
const coreIsolationController = createCoreIsolationController({
  elements: {
    panel: coreIsolationPanel,
    status: coreIsolationStatus,
    controls: coreIsolationControls,
    summary: coreIsolationSummary,
    chart: coreIsolationChart,
    body: coreIsolationBody,
    boundary: coreIsolationBoundary,
  },
  getContext: () => ({ analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence }),
});
workflowController = createWorkflowController({
  elements: {
    body: document.body,
    workflowMode,
    workflowNextAction,
    workflowNextDetail,
    workflowSteps,
    auditTabs,
    auditFocusTitle,
    auditFocusCopy,
    auditApplicabilityBoundary,
    auditApplicabilityStatus,
    auditApplicabilityTitle,
    auditApplicabilityReason,
    auditApplicabilityRequired,
    dropzone,
    modelPlan,
    workflowConsole,
    auditWorkbench,
    summary,
    insightDashboard,
    perfVisuals,
    tables,
    diagramSection,
    findingsPanel,
    graphExplorer,
    redesignPanel,
    inferencePanel,
    outputModuleSelector,
    moduleRunConsole,
    actions,
  },
  order: WORKFLOW_ORDER,
  moduleWorkspaces: MODULE_WORKSPACES,
  reportWorkspaces: REPORT_WORKSPACES,
  getSelectedArtifactName: () => pendingModelFile?.name || currentFilename || "selected artifact",
  getAnalysis: () => current,
  getFormat: () => current?.format || pendingModelInspection?.formatId || "tflite",
  getActiveModule: () => activeModule,
  setActiveModule,
  syncSelection: syncTabSelection,
  updatePerformanceVisibility: () => performanceVisualController.updateVisibility(),
});
createAnalysisDepthMode();
const offlineDeviceController = createOfflineDeviceController({
  registryList: deviceRegistryList,
  refreshButton: deviceRegistryRefresh,
  modelList: offlineTestModels,
  status: offlineTestStatus,
  getTargetProfiles: () => targetProfiles,
  getStructure: () => structureTelemetryState,
  getAnalysis: () => current,
  getFilename: () => currentFilename || current?.filename || "model",
  queueTarget: (fingerprint, target) => authFetch("/api/benchmark/queue", {
    method: "POST",
    body: JSON.stringify({ fingerprint, target }),
  }),
});
const explorerDecisionController = createExplorerDecisionView({
  root: modelGlancePanel,
  onSelectOp: jumpToGraphOp,
  onPreviewScenario: previewDelegationScenario,
  onClearScenario: clearGraphScenarioPreview,
  getRuntimeEvidence: () => ({ runtimeAssignmentEvidence }),
  onOpenAuditTab: (tabId) => setActiveAuditTab(tabId),
});
const explorerRedesignController = createExplorerRedesignController({
  blocksRoot: blocksExplorerPanel,
  cacheRoot: cacheExplorerPanel,
  redesignRoot: redesignPanel,
  project: (request) => {
    if (!current || current.format !== "tflite" || !currentModelBytes) {
      throw new Error("A bound TFLite source audit is required for Redesign.");
    }
    return project_tflite_redesign(
      currentModelBytes,
      currentFilename || current.filename || "model.tflite",
      current.target_profile?.id || selectedTargetId(),
      request,
    );
  },
  explorePareto: (request) => {
    if (!current || current.format !== "tflite" || !currentModelBytes) {
      throw new Error("A bound TFLite source audit is required for Pareto exploration.");
    }
    return explore_tflite_redesign_pareto(
      currentModelBytes,
      currentFilename || current.filename || "model.tflite",
      current.target_profile?.id || selectedTargetId(),
      request,
    );
  },
  selectOp: jumpToGraphOp,
  openWorkspace: (workspace) => setActiveWorkspace(workspace, { force: true }),
  filenameForExport: currentArtifactFilename,
});
const deploymentFrontierController = createDeploymentFrontierController({
  root: deploymentFrontierPanel,
  status: deploymentFrontierStatus,
  summary: deploymentFrontierSummary,
  body: deploymentFrontierBody,
  downloadButton: downloadDeploymentFrontier,
  getContext: () => ({ analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence }),
  jumpToGraphOp,
  onDownload: (frontier, suffix) => downloadText(
    currentArtifactFilename(suffix),
    jsonForDownload(frontier),
    "application/json",
  ),
});
const deploymentDeltaController = createDeploymentDeltaController({
  root: deploymentDeltaPanel,
  status: deploymentDeltaStatus,
  summary: deploymentDeltaSummary,
  body: deploymentDeltaBody,
  downloadButton: downloadDeploymentDelta,
  getDelta: () => currentDeploymentDelta,
  getBaseline: () => deploymentDeltaBaseline,
  jumpToGraphOp,
  onDownload: (delta, suffix) => downloadText(
    currentArtifactFilename(suffix),
    jsonForDownload(delta),
    "application/json",
  ),
});
const delegationRepairController = createDelegationRepairController({
  root: delegationRepairPanel,
  status: delegationRepairStatus,
  summary: delegationRepairSummary,
  body: delegationRepairBody,
  downloadButton: downloadDelegationRepair,
  getAnalysis: () => currentAnalysisView(),
  jumpToGraphOp,
  onPreviewScenario: previewDelegationScenario,
  onDownload: (result, suffix) => downloadText(
    currentArtifactFilename(suffix),
    jsonForDownload(result),
    "application/json",
  ),
});
clearGraphScenario?.addEventListener("click", clearGraphScenarioPreview);
const {
  quantizationLatticeController,
  accumulatorAtlasController,
  requantizationFidelityController,
  kernelWitnessController,
  channelVitalityController,
  roundingEquivalenceController,
  accumulatorReachabilityController,
  numericalAbiPropagationController,
  inputCounterexampleController,
  preprocessingRealizabilityController,
  preprocessingConsequenceController,
  contractMigrationController,
  residualStepResponseController,
  residualContractDistortionController,
} = createQuantizationResearchSuite({
  elements: appElements,
  getContext: () => ({ analysis: currentAnalysisView(), modelBytes: currentModelBytes }),
  jumpToGraphOp,
  onDownload: (result, suffix) => downloadText(
    currentArtifactFilename(suffix),
    jsonForDownload(result),
    "application/json",
  ),
  onDownloadBinary: (bytes, suffix, type = "application/octet-stream") => downloadBlob(
    currentArtifactFilename(suffix),
    new Blob([bytes], { type }),
  ),
  ensureRuntime: ensureLiteRtRuntime,
  onPreprocessingConsequenceResult: (result) => {
    preprocessingConsequenceResult = result;
    renderReportPanel();
  },
});
const quantEvidenceController = createQuantEvidenceController({
  root: quantEvidencePanel,
  onOpenOp: (opIndex) => {
    if (!current) return;
    switchExplorerTab("ops");
    selectGraphOp(currentAnalysisView(), Number(opIndex), { scrollTable: true });
  },
  onOpenNode: (opIndex) => {
    if (!current) return;
    switchExplorerTab("node");
    selectGraphOp(currentAnalysisView(), Number(opIndex), { scrollTable: false });
  },
});
const nodeViewController = createNodeViewController({
  root: nodeViewPanel,
  onSelectOp: (opIndex) => {
    if (!current) return;
    selectGraphOp(currentAnalysisView(), Number(opIndex), { scrollTable: false, fromNode: true });
  },
  onOpenOps: (opIndex) => {
    if (!current) return;
    switchExplorerTab("ops");
    selectGraphOp(currentAnalysisView(), Number(opIndex), { scrollTable: true });
  },
  onOpenQuant: (opIndex) => {
    if (!current) return;
    if (!quantEvidenceController.selectOp(Number(opIndex))) return;
    switchExplorerTab("quant");
  },
  canOpenQuant: (opIndex) => quantEvidenceController.hasEvidenceForOp(Number(opIndex)),
});
const runtimeProfileModal = createRuntimeProfileModal({
  elements: {
    backdrop: runtimeProfileBackdrop,
    closeButton: runtimeProfileClose,
    cancelButton: runtimeProfileCancel,
    title: runtimeProfileTitle,
    preview: runtimeProfilePreview,
    version: runtimeProfileVersion,
    backend: runtimeProfileBackend,
    optimization: runtimeProfileOptimization,
    executionMode: runtimeProfileExecutionMode,
    collectedAt: runtimeProfileCollectedAt,
    captureLabel: runtimeProfileCaptureLabel,
    capture: runtimeProfileCapture,
    binarySha: runtimeProfileBinarySha,
    build: runtimeProfileBuild,
    status: runtimeProfileStatus,
    importButton: runtimeProfileImport,
    fallbackFocus: kernelInspectorSearch,
  },
  getPending: () => pendingRuntimeProfile,
  clearPending: () => { pendingRuntimeProfile = null; },
  getAnalysis: () => current,
  getAssignmentEvidence: () => runtimeAssignmentEvidence,
});

function resetAdvancedResultState({
  includeDeepBom = true,
  includeRuntimeBenchmarks = true,
  includeBundleProgress = true,
  includeCalibrationValidation = true,
} = {}) {
  if (includeDeepBom) deepBomResult = null;
  perturbationResult = null;
  runtimeBasinResult = null;
  deployCurvatureResult = null;
  preprocessingConsequenceResult = null;
  if (includeCalibrationValidation) {
    calibrationValidationResult = null;
    calibrationValidationController.reset(null);
  }
  if (includeRuntimeBenchmarks) runtimeBenchmarkResults = [];
  if (includeBundleProgress) {
    engineeringBundleProgress = null;
    evidenceBundleProgress = null;
  }
}

registerServiceWorker();

initPrivacyAgreement();
initPinnedSessionOffset();
updateFormatSpecificAuditLabels({
  modelFormat: "",
  analysis: null,
  auditTabs,
  activeTab: getActiveAuditTab,
  selectTab: setActiveAuditTab,
});
updateWorkflowState("idle");
updateModuleAccessState();
initAuth();
refreshSessionNonce();
renderConsentPanel();
renderLocalReports().catch(() => {});
setStatus("Ready", "ok");

let wasmReady = null;

function loadTfliteAnalyzer() {
  if (wasmReady) return wasmReady;
  setStatus("Loading TFLite analyzer");
  wasmReady = init({ module_or_path: new URL("../pkg/tflite_wasm_audit_bg.wasm", import.meta.url) })
    .then(() => {
      const code = refreshRuntimeGuard();
      if (code === RUNTIME_OK) {
        populateTargetProfiles();
        setStatus("Ready", "ok");
      }
    })
    .catch((error) => {
      console.error(error);
      setStatus("Analyzer failed", "error");
      wasmReady = null;
      throw error;
    });
  return wasmReady;
}

{
  const link = document.getElementById("authorEmailLink");
  if (link) {
    const user = ["kjh", "0442"].join("");
    const host = ["yuhs", "ac"].join(".");
    link.href = `mailto:${user}@${host}`;
    link.textContent = `${user}@${host}`;
  }
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (file) {
    await stageModelFile(file);
    if (pendingRerun) await handlePendingRerun(file);
  }
});

artifactBundleInput?.addEventListener("change", async (event) => {
  const files = [...(event.target.files || [])];
  event.target.value = "";
  if (files.length) await stageArtifactBundle(files);
});

for (const input of [onnxExternalDataInput, onnxExternalDataDirectoryInput]) {
  input?.addEventListener("change", async (event) => {
    const files = [...(event.target.files || [])];
    event.target.value = "";
    if (files.length) await stageExternalDataSelection(files);
  });
}

pinDeploymentBaseline?.addEventListener("click", async () => {
  if (!current || !currentModelBytes || String(current.format).toLowerCase() !== "tflite") return;
  await ensureModelHash();
  deploymentDeltaBaseline = Object.freeze({
    bytes: currentModelBytes,
    filename: currentFilename || current.filename || "baseline.tflite",
    sha256: current.model_sha256,
  });
  currentDeploymentDelta = null;
  delete current.deployment_delta;
  syncDeploymentDeltaControls();
  deploymentDeltaController.render(null, current);
  renderReportPanel();
});

clearDeploymentBaseline?.addEventListener("click", () => {
  deploymentDeltaBaseline = null;
  currentDeploymentDelta = null;
  if (current) delete current.deployment_delta;
  syncDeploymentDeltaControls();
  deploymentDeltaController.render(null, current);
  renderReportPanel();
});

sampleLibraryController = installPublicSampleLibrary({
  models: PUBLIC_SAMPLE_MODELS,
  select: sampleModelSelect,
  focus: document.getElementById("sampleModelFocus"),
  glance: document.getElementById("sampleEvidenceGlance"),
  grid: document.getElementById("sampleLibraryGrid"),
  verificationPanel: document.getElementById("sampleVerificationPanel"),
  downloadButton: document.getElementById("downloadSampleManifest"),
  runButton: trySampleModel,
  digestArtifact: sha256Hex,
  onArtifact: async (file, sample, companions) => {
    await stageModelFile(file, { publicSample: sample, publicSampleCompanions: companions });
    if (!runAudit.disabled) runAudit.click();
  },
  onProgress: ({ phase, loaded, total, percent }) => {
    updateWorkflowState("running");
    if (phase === "download") {
      const bounded = Number.isFinite(percent) ? Math.max(0, Math.min(1, percent)) : 0;
      const progress = 1 + Math.round(bounded * 5);
      const detail = total > 0 ? `${formatBytes(loaded)} / ${formatBytes(total)}` : formatBytes(loaded);
      setStatus("Downloading verified example");
      analysisPlanStatus.textContent = `Downloading verified example (${detail})`;
      auditProgressController.set(progress, `Downloading hash-pinned example (${detail})`, "running", { step: 1 });
    } else if (phase === "hash") {
      setStatus("Verifying example identity");
      analysisPlanStatus.textContent = "Verifying example SHA-256";
      auditProgressController.begin(7, "Verifying example SHA-256", { ceiling: 8, step: 1 });
    } else if (phase === "verified") {
      auditProgressController.set(8, "Verified example identity", "running", { step: 1 });
    }
  },
  onError: (error) => {
    console.error("[example]", error);
    auditProgressController.set(null, "Example load failed", "error", { step: 1 });
    setStatus("Example load failed", "error");
  },
});

targetSelect.addEventListener("change", async () => {
  const requestedTargetId = targetSelect.value;
  const preserveReportWorkspace = getActiveWorkspace() === "output" && REPORT_WORKSPACES.has(activeModule);
  const preservedReportModule = preserveReportWorkspace ? activeModule : "";
  if (currentModelBytes && currentFilename) {
    try {
      await requestTargetAnalysis(requestedTargetId, {
        keepTab: true,
        keepModule: preserveReportWorkspace,
      });
      if (preserveReportWorkspace) {
        setActiveModule(preservedReportModule);
        setActiveWorkspace("output", { force: true });
      }
    } catch (error) {
      console.error("[targetChange]", error);
      const errorMsg = error?.message || String(error) || "Unknown error";
      updateWorkflowState("error");
      auditProgressController.set(null, "Target analysis failed", "error");
      setStatus(`Audit failed: ${errorMsg}`, "error");
      if (preserveReportWorkspace) {
        setActiveModule(preservedReportModule);
        setActiveWorkspace("output", { force: true });
      }
    }
  } else if (pendingModelFile) {
    writeSavedTarget(requestedTargetId);
    renderStagedModel(pendingModelFile, pendingModelInspection);
  }
});

reportTargetSelect?.addEventListener("change", () => {
  reportTargetRequestedId = reportTargetSelect.value;
  renderReportPanel();
  updateExportLockState();
});

reportTargetAnalyzeBtn?.addEventListener("click", async () => {
  const binding = reportTargetBinding();
  if (!binding.targetId || !currentModelBytes || !currentFilename) {
    setStatus("Run a static audit before selecting a report target", "error");
    return;
  }
  const original = reportTargetAnalyzeBtn.textContent;
  reportTargetAnalyzeBtn.disabled = true;
  reportTargetAnalyzeBtn.textContent = binding.state === "cached" ? "Loading analyzed target..." : "Analyzing target...";
  try {
    await requestTargetAnalysis(binding.targetId, {
      keepTab: true,
      keepModule: true,
    });
    setStatus(
      binding.state === "cached"
        ? `Loaded analyzed report target: ${selectedTargetLabel()}`
        : `Report target analysis complete: ${selectedTargetLabel()}`,
      "ok",
    );
  } catch (error) {
    console.error("[reportTarget]", error);
    updateWorkflowState("error");
    setStatus(`Report target analysis failed: ${shortError(error)}`, "error");
  } finally {
    setActiveModule("engineering_report");
    setActiveWorkspace("output", { force: true });
    reportTargetAnalyzeBtn.textContent = original;
    syncReportTargetControls();
    updateExportLockState();
  }
});

authOpen.addEventListener("click", () => openAuthModal("login"));
authClose.addEventListener("click", closeAuthModal);
authBackdrop.addEventListener("click", (event) => {
  if (event.target === authBackdrop) closeAuthModal();
});
installModalKeyboard(authBackdrop, closeAuthModal);
authLoginTab.addEventListener("click", () => setAuthMode("login"));
authSignupTab.addEventListener("click", () => setAuthMode("signup"));
authForm.addEventListener("submit", submitAuthForm);
googleLogin.addEventListener("click", startGoogleSignIn);
window.addEventListener("message", (event) => {
  if (event.origin !== location.origin || event.data?.type !== "deepbom:auth-complete") return;
  if (googleAuthPopup && event.source && event.source !== googleAuthPopup) return;
  completeGoogleSignIn(event.data);
});
if ("BroadcastChannel" in window) {
  const authChannel = new BroadcastChannel("deepbom-auth");
  authChannel.addEventListener("message", (event) => {
    if (event.data?.type === "deepbom:auth-complete") completeGoogleSignIn(event.data);
  });
}
resendVerify.addEventListener("click", resendVerificationEmail);
accountOpen.addEventListener("click", openAccountPanel);
adminOpen.addEventListener("click", openAdminConsole);
accountClose.addEventListener("click", closeAccountPanel);
accountBackdrop.addEventListener("click", (event) => {
  if (event.target === accountBackdrop) closeAccountPanel();
});
installModalKeyboard(accountBackdrop, closeAccountPanel);
accountRequestForm.addEventListener("submit", submitAccountRequest);
accountRequestProfile.addEventListener("change", () => {
  updateAccessProfileInfo(accountRequestProfile.value);
  accountRequestTitle.value = requestDraftTitle(accountRequestProfile.value);
});
adminRefresh.addEventListener("click", openAdminConsole);
adminBenchRefresh.addEventListener("click", () => loadAdminBenchmarks());
adminBenchStatusFilter.addEventListener("change", () => loadAdminBenchmarks());
authLogout.addEventListener("click", logoutAuth);

const explorerTabs = [...document.querySelectorAll("[data-explorer-tab]")];
installWorkspaceNavigation({
  workflowSteps,
  evidenceSteps: [...document.querySelectorAll("[data-evidence-workflow]")],
  auditTabs,
  moduleTabs,
  explorerTabs,
  onWorkspace: setActiveWorkspace,
  onAudit: setActiveAuditTab,
  onModule: setActiveModule,
  onLockedModule: openAccountPanelForCapability,
  onExplorer: switchExplorerTab,
});
for (const button of document.querySelectorAll("[data-request-feature]")) {
  button.addEventListener("click", () => openAccountPanelForCapability(button.dataset.requestFeature));
}

for (const button of document.querySelectorAll(".module-panel [data-scroll-target]")) {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.scrollTarget || "");
    if (!target || target.hidden) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

for (const name of ["dragenter", "dragover"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}

for (const name of ["dragleave", "drop"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}

dropzone.addEventListener("drop", async (event) => {
  const files = [...(event.dataTransfer?.files || [])];
  if (!files.length) return;
  const includesSafeTensor = files.some((candidate) => /\.safetensors$/i.test(candidate.webkitRelativePath || candidate.name));
  const includesHfConfig = files.some((candidate) => /(^|[\\/])config\.json$/i.test(candidate.webkitRelativePath || candidate.name));
  if (files.some((candidate) => /(^|[\\/])Manifest\.json$/i.test(candidate.webkitRelativePath || candidate.name)
    || /\.safetensors\.index\.json$/i.test(candidate.webkitRelativePath || candidate.name))
    || includesSafeTensor && includesHfConfig) {
    await stageArtifactBundle(files);
    return;
  }
  const modelCandidates = files.filter((file) => /\.(onnx|tflite|gguf|safetensors|mlmodel|pte|ptd)$/i.test(file.name));
  const primaryPte = modelCandidates.find((candidate) => /\.pte$/i.test(candidate.name));
  const primaryOnnx = modelCandidates.find((candidate) => /\.onnx$/i.test(candidate.name));
  const file = primaryPte || primaryOnnx || (modelCandidates.length === 1 ? modelCandidates[0] : files[0]);
  await stageModelFile(file);
  if (/\.(onnx|pte)$/i.test(file.name)) {
    const sidecars = files.filter((candidate) => candidate !== file && (/\.pte$/i.test(file.name) ? /\.ptd$/i.test(candidate.name) : true));
    if (sidecars.length) await stageExternalDataSelection(sidecars);
  }
});

runAudit.addEventListener("click", async () => {
  if (!pendingModelFile) return;
  await analyzeFile(pendingModelFile);
});

const textExportOptions = {
  getFilename: currentArtifactFilename,
  isReady: (artifact) => Boolean(
    current
    && reportTargetBinding().canCopy
    && (!artifact.requireModelBytes || currentModelBytes)
  ),
  ensureAllowed: (artifact) => artifact.raw
    ? ensureRawExportAllowed(artifact.permissionLabel)
    : true,
  ensureHash: ensureModelHash,
};
registerTextExport(downloadMarkdown, TEXT_EXPORT_ARTIFACTS.engineeringReport, async () => {
  const formatter = await loadEngineeringFormatter();
  if (current?.model_sha256) {
    recordReportArtifact(current.model_sha256, current.target_profile?.id || "", { kind: "engineering_report" })
      .then(() => renderLocalReports()).catch(() => {});
  }
  const reportContext = currentReportContext();
  const reportBody = formatter.buildEngineeringReport(currentAnalysisView(), reportContext);
  const reportFingerprint = await sha256Hex(new TextEncoder().encode(reportBody));
  return buildPublicEngineeringReportHtml(reportBody, {
    generatedAt: reportContext.generatedAt,
    modelName: current?.filename || currentFilename || "model",
    origin: location.origin,
    reportFingerprint,
    profile: "engineering",
  });
}, textExportOptions);

downloadReviewHtml?.addEventListener("click", async () => {
  if (!current || !reportTargetBinding().canCopy) return;
  await withBusyButton(downloadReviewHtml, "Preparing", async () => {
    try {
      await ensureModelHash();
      downloadText(currentArtifactFilename("review.html"), currentReviewHtml(), "text/html");
      setStatus("Read-only review HTML downloaded", "ok");
    } catch (error) {
      setStatus(`Review HTML failed: ${shortError(error)}`, "error");
    }
  }, updateExportLockState);
});
registerTextExport(downloadRegulatoryReport, TEXT_EXPORT_ARTIFACTS.regulatoryReport, async () => {
  const formatter = await loadRegulatoryFormatter();
  const reportContext = currentReportContext();
  const reportBody = formatter.buildRegulatoryReport(currentAnalysisView(), currentRegulatoryReportContext());
  const reportFingerprint = await sha256Hex(new TextEncoder().encode(reportBody));
  return buildPublicEngineeringReportHtml(reportBody, {
    generatedAt: reportContext.generatedAt,
    modelName: current?.filename || currentFilename || "model",
    origin: location.origin,
    reportFingerprint,
    profile: "regulatory",
  });
}, textExportOptions);
registerTextExport(downloadCsv, TEXT_EXPORT_ARTIFACTS.rooflineCsv, () => current.roofline_csv, textExportOptions);
registerTextExport(downloadMermaid, TEXT_EXPORT_ARTIFACTS.mermaidStageGraph, () => current.stage_mermaid, textExportOptions);
const exportContractController = createExportContractController({
  elements: appElements,
  getContext: () => ({ analysis: currentAnalysisView(), modelBytes: currentModelBytes, runtimeEvidence: runtimeAssignmentEvidence }),
  getDocuments: buildCurrentDeploymentContractDocuments,
  getPublicDocuments: buildCurrentPublicCycloneDxDocuments,
  getFilename: currentArtifactFilename,
  ensureAllowed: ensureRawExportAllowed,
  ensureHash: ensureModelHash,
  getAccess: () => ({ rawExportAllowed: currentCapabilities().raw_export }),
  onStatus: setStatus,
  onProductionContractChange: (value) => { productionInterfaceContract = value; },
  onProductionComparison: updateProductionInterfaceFinding,
});

graphWorkspace = createGraphWorkspace({
  ...appElements,
  TEXT_EXPORT_ARTIFACTS,
  appendBenchmarkRow,
  artifactOverviewHeader,
  artifactOverviewPanels,
  assessedOpLogicalBytes,
  backendCandidates,
  backendSelect,
  benchmarkBody,
  benchmarkErrorStatus,
  benchmarkWrap,
  resourceMapPanel,
  blocksExplorerPanel,
  buildGraphEvidenceMaps,
  buildGraphIndex,
  buildRepresentableKernelChannelCheck,
  buildOnnxRuntimeShapeBinding,
  buildTensorInventory,
  cacheExplorerPanel,
  canvasToPngBytes,
  clampInt,
  classifyTensorRoles,
  tensorQuantizationMode,
  clearRuntimeAssignment,
  collectFullGraph,
  collectNeighborhood,
  compute_input_influence,
  compute_output_influence,
  compute_static_runtime_calibration,
  compute_weight_histogram,
  downloadTextArtifact,
  ensureLiteRtRuntime,
  evidenceCursor,
  explorerDecisionController,
  explorerExecutionPlacementPanel,
  explorerRedesignController,
  explorerTabs,
  findingsBody,
  formatBytes,
  formatNumber,
  formatPercent,
  formatUs,
  graphDepth,
  graphDetailLayout,
  graphExplorer,
  graphMapStatus,
  graphMapSvg,
  graphModeHint,
  graphOpBody,
  graphOpHead,
  graphOpRow,
  graphOpsView,
  graphScenarioBanner,
  graphScenarioDetail,
  graphScenarioLabel,
  graphSearch,
  graphStats,
  graphSvgText,
  canonicalGraphSvgText,
  histogramBody,
  histogramRow,
  kernelBoundaryInventory,
  kernelInspectorBody,
  kernelInspectorPanel,
  kernelInspectorSearch,
  kernelInspectorSummary,
  layeredViewPanel,
  layoutFoldedGraph,
  layoutNeighborhood,
  loadOnnxBenchmark,
  loadProtectedSourceAnalysis,
  loadTfliteBenchmark,
  mk,
  modelFormatAdapter,
  modelSupportsCapability,
  nodeViewController,
  nodeViewPanel,
  opDetail,
  opFilterCount,
  opLogicalL1Ratio,
  opLogicalRowPayloadBytes,
  opMatchesSearch,
  opNavLabel,
  opNavNext,
  opNavPrev,
  opParetoBar,
  opSteadyStateUs,
  opTimeline,
  p99EvidenceForSampleCount,
  padOp,
  performanceVisualController,
  preprocessingConsequenceController,
  quantEvidenceController,
  quantEvidencePanel,
  renderExecutionPlacementView,
  renderFindingsCalibration,
  renderGraphMapContent,
  renderInsightDashboardView,
  renderKernelInspector,
  renderOpDetailPanel,
  renderRuntimeEvidenceClosure,
  renderTensorArenaViewer,
  rooflineBody,
  rooflineTableRows,
  runInference,
  runsInput,
  runtimeAssignmentComparison,
  runtimeAssignmentStatus,
  runtimeEvidenceClosure,
  runtimeNotes,
  runtimeReadinessSignals,
  runtimeStatus,
  selectWasmCalibrationResult,
  selectedTargetId,
  selectedTargetProfile,
  setActiveAuditTab,
  setActiveWorkspace,
  shortError,
  stageCard,
  stageCount,
  stageStrip,
  submitBenchmarkTelemetry,
  summary,
  summaryMetricCards,
  syncTabSelection,
  tensorBody,
  tensorExplorerPanel,
  tensorMemoryTimeline,
  tensorStatsBar,
  textExportOptions,
  topMacBody,
  topMacRows,
  updateBenchmarkRow,
  updateWorkflowState,
  visualPngSpecs,
  warmupInput,
  xnnSegmentBar,
  zipBinaryFile,
  zipTextFile,
  get activeGraphScenario() { return activeGraphScenario; },
  set activeGraphScenario(value) { activeGraphScenario = value; },
  get activeTargetId() { return activeTargetId; },
  get current() { return current; },
  get currentFilename() { return currentFilename; },
  get currentGraphMode() { return currentGraphMode; },
  set currentGraphMode(value) { currentGraphMode = value; },
  get currentKernelFilter() { return currentKernelFilter; },
  get currentLowNormStatMap() { return currentLowNormStatMap; },
  set currentLowNormStatMap(value) { currentLowNormStatMap = value; },
  get currentModelBytes() { return currentModelBytes; },
  get currentModelPayloadLoaded() { return currentModelPayloadLoaded; },
  get currentArtifactIrContext() { return currentArtifactIrContext; },
  get currentOnnxExternalDataFiles() { return currentExternalDataFiles; },
  get currentTensorFilter() { return currentTensorFilter; },
  get currentTensorRoleFilter() { return currentTensorRoleFilter; },
  get currentTopologyAnnotations() { return currentTopologyAnnotations; },
  set currentTopologyAnnotations(value) { currentTopologyAnnotations = value; },
  get graphMapBounds() { return graphMapBounds; },
  set graphMapBounds(value) { graphMapBounds = value; },
  get graphRenderToken() { return graphRenderToken; },
  set graphRenderToken(value) { graphRenderToken = value; },
  get graphViewBox() { return graphViewBox; },
  set graphViewBox(value) { graphViewBox = value; },
  get opFilterBound() { return opFilterBound; },
  get opFilterQuant() { return opFilterQuant; },
  get opFilterXnn() { return opFilterXnn; },
  get opTableSortDir() { return opTableSortDir; },
  get opTableSortKey() { return opTableSortKey; },
  get preprocessingConsequenceResult() { return preprocessingConsequenceResult; },
  get runtimeAssignmentEvidence() { return runtimeAssignmentEvidence; },
  get runtimeBenchmarkResults() { return runtimeBenchmarkResults; },
  set runtimeBenchmarkResults(value) { runtimeBenchmarkResults = value; },
  get selectedOpIndex() { return selectedOpIndex; },
  set selectedOpIndex(value) { selectedOpIndex = value; },
});
const {
  renderSummary,
  adaptInsightsForUI,
  renderInsightDashboard,
  buildVisualPngFiles,
  graphScenarioMatchesAnalysis,
  renderGraphScenarioState,
  renderInferencePanel,
  renderGraphExplorer,
  renderOpTimeline,
  renderXnnSegmentBar,
  renderOpParetoBar,
  renderGraphOpRows,
  buildTensorConsumers,
  buildDuplicateWeightGroups,
  renderTensorExplorer,
  renderTensorMemoryTimeline,
  renderLayeredView,
  switchGraphMode,
  updateGraphModeHint,
  renderCurrentKernelInspector,
  updateOpNav,
  selectGraphNodeFromMap,
  scrollGraphTableToOp,
  closestGraphNode,
  deferGraphMap,
  renderOpDetail,
  renderGraphMap,
  fitGraphMap,
  zoomGraphMap,
  applyGraphViewBox,
  downloadCurrentGraphSvg,
  runInferenceBenchmark,
  renderStages,
  jumpToStage,
  renderHistogram,
  renderTopMacs,
  renderRoofline,
} = graphWorkspace;

const deepBomWorkspace = createDeepBomWorkspace({
  ...appElements,
  applyProtectedOrtCompatibilityEvidence,
  applyProtectedTfliteDelegateCompatibilityEvidence,
  applyProtectedXnnpackSelectorEvidence,
  assessedOpLogicalBytes,
  compute_model_tomography,
  deepBomMetric,
  deploymentFrontierController,
  drawMvDepth,
  drawMvFilter,
  formatBytes,
  formatNumber,
  formatPercent1,
  layer_landscape_grid,
  mk,
  modelIdentity,
  modelSupportsCapability,
  nextPaint,
  protocolBlock,
  renderAuditClaimBoundary,
  renderCurrentKernelInspector,
  renderGraphOpRows,
  renderOpDetail,
  score100,
  selectedTargetId,
  setStatus,
  shortError,
  statusForEntropy,
  updateModuleAccessState,
  updateWorkflowState,
  get current() { return current; },
  get currentFilename() { return currentFilename; },
  get currentModelBytes() { return currentModelBytes; },
  get deepBomModule() { return deepBomModule; },
  set deepBomModule(value) { deepBomModule = value; },
  get deepBomResult() { return deepBomResult; },
  set deepBomResult(value) { deepBomResult = value; },
  get selectedOpIndex() { return selectedOpIndex; },
});
const {
  runDeepBomAnalysis,
  resetDeepBomPanel,
  renderDeepBomSkeleton,
  renderDriftSkeleton,
  renderDeploymentSensitivitySkeleton,
  deepBomSignalValue,
  renderDeepBomResult,
  renderProtocolGroups,
  activeThemeColor,
  renderModelViewer,
  runModelViewer,
  hasModelTomography,
} = deepBomWorkspace;

function updateProductionInterfaceFinding(comparison) {
  if (!current) return;
  const findingId = "EA-ABI-0001";
  current.findings = (current.findings || []).filter((finding) => finding.id !== findingId);
  if (comparison?.gate_result === "block") {
    const details = (comparison.mismatches || []).slice(0, 4)
      .map((item) => `${item.parameter_id || "document"} ${item.field}: artifact ${JSON.stringify(item.expected)}; declaration ${JSON.stringify(item.declared)}`);
    current.findings.push({
      id: findingId,
      title: "Production interface contract contradicts the artifact ABI",
      category: "input",
      severity: "high",
      confidence: "high",
      evidence: details.map((text) => ({ source: "Contract diff", text })),
      impact: "The deployed encoder or decoder can apply a dtype, shape, scale, zero-point, axis, or artifact identity that differs from the audited model boundary.",
      actions: ["Block release until every named external parameter and the implementation SHA-256 are bound to the audited artifact."],
    });
  }
  renderFindings(findingsBody, currentArtifactIrContext?.primary_view || current);
}

downloadVisualPngs.addEventListener("click", async () => {
  if (!current) return;
  if (!(await ensureRawExportAllowed("Visual PNGs"))) return;
  await withBusyButton(downloadVisualPngs, "Rendering", async () => {
    try {
      const files = await buildVisualPngFiles();
      const zip = createZipBlob(files);
      downloadBlob(currentArtifactFilename("deepbom_visuals.zip"), zip);
      setStatus("Visual PNGs downloaded", "ok");
    } catch (error) {
      console.error(error);
      setStatus("Visual PNG export failed", "error");
    }
  }, updateExportLockState);
});

downloadRawData.addEventListener("click", async () => {
  await downloadRawDataZip();
});

downloadEngineeringBundle.addEventListener("click", async () => {
  await downloadEngineeringBundleZip();
});

downloadPublicBundle?.addEventListener("click", async () => {
  await downloadEvidencePackageProfileZip();
});

evidencePackageProfile?.addEventListener("change", () => updateExportLockState());
evidencePackageLevel?.addEventListener("change", () => updateExportLockState());

if (downloadEvidenceBundle) {
  downloadEvidenceBundle.addEventListener("click", async () => {
    await downloadEvidenceBundleZip();
  });
}

runDeepBom.addEventListener("click", () => loadProtectedSourceAnalysis({ openModule: true }));

async function loadProtectedSourceAnalysis({ openModule = false } = {}) {
  if (!current || !currentModelBytes || !currentModelPayloadLoaded || !modelSupportsCapability(current.format, "protected_source_analysis")) return;
  runDeepBom.disabled = true;
  runDeepBom.setAttribute("aria-busy", "true");
  deepBomStatus.textContent = "Authorizing…";
  try {
    if (openModule) setActiveModule("deepbom");
    const manifest = await ensureDeepBomAllowed();
    await runDeepBomAnalysis(manifest, { activateWorkflow: openModule });
    renderAcceleratorSwitcher();
  } catch (error) {
    console.error(error);
    deepBomStatus.textContent = "Blocked";
    setStatus("DEEPBOM blocked", "error");
  } finally {
    runDeepBom.disabled = !modelSupportsCapability(current?.format, "protected_source_analysis");
    runDeepBom.removeAttribute("aria-busy");
  }
}

perturbationPanelAction.addEventListener("click", async () => {
  perturbationPanelAction.disabled = true;
  perturbationPanelAction.setAttribute("aria-busy", "true");
  perturbationStatus.textContent = "Authorizing…";
  runtimeBasinStatus.textContent = "Waiting";
  try {
    if (!(await ensureResearchModuleAllowed("runtime_basin", "Drift Analysis"))) {
      perturbationStatus.textContent = "Blocked";
      runtimeBasinStatus.textContent = "Blocked";
      return;
    }
    await runPerturbationAnalysis();
    perturbationPanelAction.disabled = true;
    await runRuntimeBasinValidation();
  } finally {
    perturbationPanelAction.disabled = !current || !modelSupportsCapability(current.format, "experimental_tflite_research");
    perturbationPanelAction.removeAttribute("aria-busy");
  }
});

modelViewerBtn.addEventListener("click", async () => {
  if (!modelSupportsCapability(current?.format, "model_tomography")) return;
  modelViewerBtn.disabled = true;
  modelViewerBtn.textContent = "Computing…";
  try { await runModelViewer(); } finally {
    modelViewerBtn.disabled = false;
    modelViewerBtn.textContent = hasModelTomography() ? "Re-run Artifact Viewer" : "Artifact Viewer";
  }
});

deploymentSensitivityPanelAction.addEventListener("click", async () => {
  deploymentSensitivityPanelAction.disabled = true;
  deploymentSensitivityPanelAction.setAttribute("aria-busy", "true");
  deploymentSensitivityStatus.textContent = "Authorizing…";
  try {
    if (!(await ensureResearchModuleAllowed("deployment_sensitivity", "Deployment Sensitivity analysis"))) {
      deploymentSensitivityStatus.textContent = "Blocked";
      return;
    }
    await runDeployCurvatureBasinAnalysis();
  } finally {
    deploymentSensitivityPanelAction.disabled = !current || !modelSupportsCapability(current.format, "experimental_tflite_research");
    deploymentSensitivityPanelAction.removeAttribute("aria-busy");
  }
});


downloadDeepBom.addEventListener("click", async () => {
  if (!deepBomResult || !current) return;
  await ensureDeepBomAllowed();
  downloadText(currentArtifactFilename("deepbom.json"), jsonForDownload(deepBomResult), "application/json");
});

runInference.addEventListener("click", async () => {
  if (!current || !currentModelBytes || !modelSupportsCapability(current.format, "runtime_execution")) return;
  try {
    await ensureAnalyzerReady();
    await runInferenceBenchmark();
  } catch (error) {
    console.error(error);
    if (runtimeGuardCode === RUNTIME_OK) {
      setStatus("Benchmark blocked", "error");
    }
  }
});

for (const control of [backendSelect, warmupInput, runsInput]) {
  control.addEventListener("input", () => {
    if (current && getActiveWorkspace() === "runtime") renderInferencePanel(currentAnalysisView());
  });
  control.addEventListener("change", () => {
    if (current && getActiveWorkspace() === "runtime") renderInferencePanel(currentAnalysisView());
  });
}

graphSearch.addEventListener("input", () => {
  if (current) renderGraphOpRows(currentAnalysisView());
});

opFilterBar.addEventListener("click", (e) => {
  const chip = e.target.closest(".filter-chip");
  if (!chip) return;
  const group = chip.dataset.filterGroup;
  const value = chip.dataset.filterValue;
  if (group === "bound") opFilterBound = value;
  else if (group === "xnn") opFilterXnn = value;
  else if (group === "quant") opFilterQuant = value;
  for (const c of opFilterBar.querySelectorAll(`[data-filter-group="${group}"]`)) {
    c.classList.toggle("active", c.dataset.filterValue === value);
  }
  if (current) renderGraphOpRows(currentAnalysisView());
});

graphOpHead.addEventListener("click", (e) => {
  const th = e.target.closest("th[data-sort-key]");
  if (!th) return;
  const key = th.dataset.sortKey;
  if (opTableSortKey === key) {
    opTableSortDir = -opTableSortDir;
  } else {
    opTableSortKey = key;
    opTableSortDir = key === "index" ? 1 : -1;
  }
  if (current) renderGraphOpRows(currentAnalysisView());
});

graphDepth.addEventListener("change", () => {
  if (current) renderGraphMap(currentAnalysisView(), selectedOpIndex);
});

document.addEventListener("click", (e) => {
  const modeBtn = e.target.closest("[data-graph-mode]");
  if (!modeBtn) return;
  const mode = modeBtn.dataset.graphMode;
  for (const btn of document.querySelectorAll("[data-graph-mode]")) {
    btn.classList.toggle("active", btn.dataset.graphMode === mode);
  }
  switchGraphMode(mode);
});

kernelInspectorSearch?.addEventListener("input", () => {
  if (current) renderCurrentKernelInspector();
});
document.addEventListener("click", (event) => {
  const chip = event.target.closest("[data-kernel-filter]");
  if (!chip) return;
  currentKernelFilter = chip.dataset.kernelFilter || "all";
  for (const item of document.querySelectorAll("[data-kernel-filter]")) item.classList.toggle("active", item === chip);
  if (current) renderCurrentKernelInspector();
});
installRuntimeEvidenceController({
  elements: {
    input: runtimeAssignmentInput, form: runtimeProfileForm, collectedAt: runtimeProfileCollectedAt,
    version: runtimeProfileVersion, backend: runtimeProfileBackend, build: runtimeProfileBuild,
    binarySha: runtimeProfileBinarySha, optimization: runtimeProfileOptimization,
    executionMode: runtimeProfileExecutionMode, capture: runtimeProfileCapture,
    formStatus: runtimeProfileStatus, templateButton: downloadRuntimeAssignmentTemplate,
    capturePlanButton: downloadRuntimeCapturePlan, clearButton: clearRuntimeAssignment,
  },
  modal: runtimeProfileModal,
  getAnalysis: () => current,
  getEvidence: () => runtimeAssignmentEvidence,
  setEvidence: (value) => { runtimeAssignmentEvidence = value; },
  getPending: () => pendingRuntimeProfile,
  setPending: (value) => { pendingRuntimeProfile = value; },
  ensureArtifactHash: ensureModelHash,
  artifactFilename: currentArtifactFilename,
  onChanged: () => {
    rebuildCurrentArtifactIrContext();
    renderCurrentKernelInspector(true);
    renderTensorExplorer(currentAnalysisView());
    deploymentFrontierController.render(currentAnalysisView());
    renderFormatCapabilityMatrix(formatCapabilityPanel, current?.format, { analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence });
    renderRuntimeEvidenceClosure(runtimeEvidenceClosure, currentAnalysisView(), runtimeAssignmentEvidence);
    coreIsolationController.render();
    renderAuditClaimBoundary(current?.format, currentAnalysisView());
    renderReportPanel();
  },
  setStatus,
});
document.getElementById("importFormatRuntimeEvidence")?.addEventListener("click", () => runtimeAssignmentInput?.click());
document.getElementById("downloadFormatRuntimeTemplate")?.addEventListener("click", () => downloadRuntimeAssignmentTemplate?.click());
document.getElementById("downloadFormatRuntimeCapturePlan")?.addEventListener("click", () => downloadRuntimeCapturePlan?.click());
document.getElementById("clearFormatRuntimeEvidence")?.addEventListener("click", () => clearRuntimeAssignment?.click());
document.getElementById("downloadRuntimeEvidenceSidecar")?.addEventListener("click", () => {
  try {
    const sidecar = runtimeEvidenceSidecarForDownload(current, runtimeAssignmentEvidence);
    downloadText(
      currentArtifactFilename("runtime_evidence_sidecar.json"),
      `${JSON.stringify(sidecar, null, 2)}\n`,
      "application/json",
    );
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
});

nodeEvidenceOverlayInput?.addEventListener("change", async () => {
  const file = nodeEvidenceOverlayInput.files?.[0];
  nodeEvidenceOverlayInput.value = "";
  if (!file || !current) return;
  try {
    if (file.size > 32 * 1024 * 1024) throw new Error("Node/edge evidence overlay exceeds the 32 MiB JSON limit.");
    await ensureModelHash();
    const parsed = JSON.parse(await file.text());
    nodeEdgeEvidenceOverlay = validateNodeEdgeEvidenceOverlay(parsed, current);
    current.external_node_edge_evidence_overlay = nodeEdgeEvidenceOverlay;
    if (nodeEvidenceOverlayStatus) {
      nodeEvidenceOverlayStatus.textContent = `${nodeEdgeEvidenceOverlay.nodes.length} node rows and ${nodeEdgeEvidenceOverlay.edges.length} edge rows imported from ${nodeEdgeEvidenceOverlay.source.label}.`;
    }
    if (clearNodeEvidenceOverlay) clearNodeEvidenceOverlay.hidden = false;
    renderGraphExplorer(currentAnalysisView());
    renderReportPanel();
    setStatus("Hash-bound node/edge evidence overlay imported.", "ok");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
});

downloadNodeEvidenceOverlayTemplate?.addEventListener("click", async () => {
  if (!current) return setStatus("Run a static audit before exporting an overlay template.", "error");
  try {
    await ensureModelHash();
    const template = buildNodeEdgeEvidenceOverlayTemplate(currentAnalysisView());
    downloadText(currentArtifactFilename("node_edge_evidence_overlay.template.json"), `${JSON.stringify(template, null, 2)}\n`, "application/json");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
});

clearNodeEvidenceOverlay?.addEventListener("click", () => {
  nodeEdgeEvidenceOverlay = null;
  if (current) delete current.external_node_edge_evidence_overlay;
  if (nodeEvidenceOverlayStatus) nodeEvidenceOverlayStatus.textContent = "No external node/edge evidence overlay imported.";
  clearNodeEvidenceOverlay.hidden = true;
  if (current) renderGraphExplorer(currentAnalysisView());
  renderReportPanel();
});

globalThis.addEventListener("deepbom:evidence-select", (event) => handleEvidenceSelection(event.detail || {}));
globalThis.addEventListener("deepbom:evidence-explain", (event) => evidenceWhyController.open(event.detail || {}));

document.addEventListener("click", (e) => {
  const tabBtn = e.target.closest("[data-explorer-tab]");
  if (!tabBtn) return;
  switchExplorerTab(tabBtn.dataset.explorerTab);
});

if (tensorSearch) {
  tensorSearch.addEventListener("input", () => {
    currentTensorFilter = tensorSearch.value.trim();
    if (current) renderTensorExplorer(currentAnalysisView());
  });
}
document.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-tfilter]");
  if (!chip) return;
  for (const c of document.querySelectorAll("[data-tfilter]")) c.classList.remove("active");
  chip.classList.add("active");
  currentTensorRoleFilter = chip.dataset.tfilter === "all" ? "" : chip.dataset.tfilter;
  if (current) renderTensorExplorer(currentAnalysisView());
});

graphZoomOut.addEventListener("click", () => zoomGraphMap(1.25));
graphZoomIn.addEventListener("click", () => zoomGraphMap(0.8));
graphFit.addEventListener("click", () => fitGraphMap());
downloadGraphSvg.addEventListener("click", () => {
  void downloadCurrentGraphSvg();
});

graphMapSvg.addEventListener("pointerdown", (event) => {
  if (!graphViewBox) return;
  if (closestGraphNode(event.target)) return;
  graphDrag = {
    x: event.clientX,
    y: event.clientY,
    viewBox: { ...graphViewBox },
    moved: false,
  };
  graphMapSvg.classList.add("dragging");
  graphMapSvg.setPointerCapture(event.pointerId);
});

graphMapSvg.addEventListener("pointermove", (event) => {
  if (!graphDrag || !graphViewBox) return;
  const rect = graphMapSvg.getBoundingClientRect();
  const dx = event.clientX - graphDrag.x;
  const dy = event.clientY - graphDrag.y;
  if (Math.abs(dx) > 2 || Math.abs(dy) > 2) graphDrag.moved = true;
  graphViewBox = {
    ...graphDrag.viewBox,
    x: graphDrag.viewBox.x - (dx * graphDrag.viewBox.width) / rect.width,
    y: graphDrag.viewBox.y - (dy * graphDrag.viewBox.height) / rect.height,
  };
  applyGraphViewBox();
});

for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
  graphMapSvg.addEventListener(eventName, () => {
    graphDrag = null;
    graphMapSvg.classList.remove("dragging");
  });
}

function initPrivacyAgreement() {
  initPrivacyAgreementUi({
    privacyAgree,
    acceptAgreement,
    researchConsent,
    agreementBackdrop,
    body: document.body,
    fallbackFocus: fileInput,
    onAccept: () => {
      appendConsentLog({ kind: "consent-restored", policyVersion: AGREEMENT_POLICY_VERSION });
      renderConsentPanel();
    },
  });
}

async function ensureAnalyzerReady() {
  await loadTfliteAnalyzer();
  const code = refreshRuntimeGuard();
  if (code !== RUNTIME_OK) {
    throw new Error(runtimeGuardMessage(code));
  }
}

function refreshRuntimeGuard() {
  runtimeGuardCode = runtime_guard();
  if (runtimeGuardCode !== RUNTIME_OK) {
    lockRuntime(runtimeGuardCode);
  }
  return runtimeGuardCode;
}

function lockRuntime(code) {
  const title = runtimeGuardTitle(code);
  const message = runtimeGuardMessage(code);
  updateWorkflowState("locked", { title, detail: message });
  setStatus(title, "error");
  fileInput.disabled = true;
  runAudit.disabled = true;
  runInference.disabled = true;
  runtimeStatus.textContent = title;
  runtimeNotes.replaceChildren(runtimeSignal("Runtime", title, "risk"));
  runtimeNotes.title = message;
  summary.replaceChildren(insightCard("Runtime Availability", title, message, "risk"));
}

function updateWorkflowState(state, detail = {}) {
  workflowController.updateState(state, detail);
}

function setActiveWorkspace(workspace = "input", options = {}) {
  return workflowController.setWorkspace(workspace, options);
}

function getActiveWorkspace() {
  return workflowController?.activeWorkspace || "input";
}

function getActiveAuditTab() {
  return workflowController?.activeAuditTab || "overview";
}

function setActiveAuditTab(tabId = "overview") {
  workflowController.setAuditTab(tabId);
}

function initPinnedSessionOffset() {
  const root = document.documentElement;
  const update = () => {
    const stickyHeight = (element) => {
      const position = element ? getComputedStyle(element).position : "static";
      return ["fixed", "sticky"].includes(position)
        ? Math.round(element.getBoundingClientRect().height || 0)
        : 0;
    };
    const topbarH = stickyHeight(topbar);
    const anchorH = stickyHeight(sessionAnchor);
    const staticMobileChrome = window.matchMedia("(max-width: 820px)").matches;
    root.style.setProperty("--session-sticky-top", `${staticMobileChrome ? 0 : topbarH}px`);
    const coverH = staticMobileChrome ? 0 : topbarH + anchorH + 8;
    root.style.setProperty("--sticky-cover-height", `${coverH}px`);
    root.style.scrollPaddingTop = `${coverH}px`;
  };
  update();
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(update);
    if (topbar) ro.observe(topbar);
    if (sessionAnchor) ro.observe(sessionAnchor);
  }
  window.addEventListener("resize", update);
}

document.addEventListener("keydown", (e) => {
  if (!current || graphExplorer.hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) return;
  if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
  e.preventDefault();
  const analysisView = currentAnalysisView();
  const ops = analysisView.ops;
  const idx = ops.findIndex((op) => op.index === selectedOpIndex);
  if (idx < 0) return;
  const next = e.key === "ArrowUp" ? idx - 1 : idx + 1;
  if (next >= 0 && next < ops.length) {
    selectGraphOp(analysisView, ops[next].index, { scrollTable: true });
  }
});

function setActiveModule(moduleId = "engineering_report") {
  const resolved = moduleId === "perturbation" ? "runtime_basin" : moduleId;
  const fallback = selectableModuleIdFor(moduleTabs, resolved);
  activeModule = fallback;
  for (const tab of moduleTabs) {
    tab.classList.toggle("active", tab.dataset.moduleTab === fallback);
  }
  syncTabSelection(moduleTabs, (tab) => tab.dataset.moduleTab === fallback);
  for (const panel of moduleRunPanels) {
    panel.classList.toggle("active", panel.dataset.moduleRunPanel === fallback);
  }
  for (const panel of modulePanels) {
    panel.classList.toggle("active", panel.dataset.modulePanel === fallback);
  }
  if (fallback === "offline_test") {
    offlineDeviceController.loadModels();
    offlineDeviceController.loadRegistry();
  }
}

async function initAuth() {
  try {
    const config = await authFetch("/api/auth/config");
    authConfigState = config;
    renderAuthConfig(config);
    if (config.enabled) {
      const session = await authFetch("/api/auth/me");
      renderAuthUser(session.user);
    }
    await refreshAccessStatus();
    const authError = new URLSearchParams(location.search).get("auth_error");
    if (authError) {
      openAuthModal("login", `Google sign-in failed: ${authError}`);
    }
    const verified = new URLSearchParams(location.search).get("verified");
    if (verified === "ok") {
      setStatus("Email verified", "ok");
      authMessage.textContent = "Email verification complete. Account-bound exports and requests are now available where authorized.";
      await refreshAccessStatus({ force: true });
    }
  } catch (error) {
    console.warn("Auth init failed", error);
    authWidget.classList.add("auth-unavailable");
    authOpen.textContent = "Account unavailable";
  }
}

function renderAuthConfig(config) {
  applyAuthConfigView({
    widget: authWidget,
    openButton: authOpen,
    googleButton: googleLogin,
    submitButton: authSubmit,
    passwordTabs: authPasswordTabs,
    passwordForm: authForm,
    divider: authDivider,
    message: authMessage,
  }, config);
  updateExportLockState();
}

function renderAuthUser(user) {
  currentAuthUser = user || null;
  if (!user) {
    authOpen.hidden = false;
    authOpen.removeAttribute("aria-hidden");
    authUser.hidden = true;
    authUser.inert = true;
    authUser.setAttribute("aria-hidden", "true");
    authUser.dataset.role = "";
    authRole.hidden = true;
    authRole.textContent = "";
    authName.textContent = "";
    authEmail.textContent = "";
    authProvider.textContent = "";
    authVerify.hidden = true;
    authVerify.textContent = "";
    resendVerify.hidden = true;
    adminOpen.hidden = true;
    syncOfflineTestVisibility(null);
    renderLocalReports().catch(() => {});
    accessGrantState = null;
    updateExportLockState();
    return;
  }
  authOpen.hidden = true;
  authOpen.setAttribute("aria-hidden", "true");
  authUser.hidden = false;
  authUser.inert = false;
  authUser.removeAttribute("aria-hidden");
  authUser.dataset.role = user.role || "user";
  const externalTest = Boolean(user.test_access?.active);
  authRole.hidden = user.role !== "admin" && !externalTest;
  authRole.textContent = user.role === "admin" ? "Admin" : externalTest ? "External test" : "";
  authName.textContent = user.name || user.email;
  authEmail.textContent = externalTest ? "No account required" : user.email;
  authProvider.textContent = externalTest
    ? `Private link / expires ${new Date(user.test_access.expires_at).toLocaleString()}`
    : `${roleLabel(user.role)} / ${accessLabel(user, accessGrantState)} / ${providerLabel(user.provider)} account / ${user.email_verified ? "verified email" : "email pending"}`;
  authVerify.hidden = Boolean(user.email_verified);
  authVerify.textContent = user.email_verified ? "" : "Email verification required";
  resendVerify.hidden = Boolean(user.email_verified || user.provider === "google");
  adminOpen.hidden = user.role !== "admin";
  syncOfflineTestVisibility(user);
  renderLocalReports().catch(() => {});
  persistAuditSnapshot().catch(() => {}); // capture the already-run audit right after sign-in
  updateExportLockState();
  syncResearchConsent().catch((error) => console.warn("Consent sync skipped", error));
  closeAuthModal();
}

function syncOfflineTestVisibility(user) {
  const isAdmin = user?.role === "admin";
  for (const el of document.querySelectorAll('[data-workflow-step="offline_test"], [data-module-tab="offline_test"]')) {
    el.hidden = !isAdmin || el.dataset.formatApplicable === "false";
  }
}

function openAuthModal(mode = "login", message = "") {
  setAuthMode(mode);
  authMessage.textContent = message || (authConfigState.enabled
    ? authConfigState.password ? "Analysis is open. Account-bound report exports require sign-in." : "Continue with Google to access account-bound exports."
    : "Authentication is not configured yet. Add D1 and Worker secrets to enable signups.");
  openModal(authBackdrop, { focus: authConfigState.password ? authFormEmail : googleLogin });
}

function closeAuthModal() {
  closeModal(authBackdrop, { fallbackFocus: authOpen });
  authForm.reset();
}

function setAuthMode(mode) {
  authMode = mode === "signup" ? "signup" : "login";
  applyAuthModeView({
    loginTab: authLoginTab,
    signupTab: authSignupTab,
    nameWrap: authNameWrap,
    passwordInput: authFormPassword,
    submitButton: authSubmit,
  }, authMode);
}

function startGoogleSignIn() {
  if (googleAuthPopup && !googleAuthPopup.closed) {
    googleAuthPopup.focus();
    return;
  }
  const returnTo = "/web/auth-complete.html?auth_popup=complete";
  const url = `/api/auth/google/start?return_to=${encodeURIComponent(returnTo)}`;
  googleAuthPopup = window.open(url, "deepbom-google-auth", "popup,width=520,height=720");
  if (!googleAuthPopup) {
    authMessage.textContent = "The sign-in window was blocked. Allow popups and try again; the current audit has been kept.";
    setStatus("Sign-in popup blocked", "error");
    return;
  }
  authMessage.textContent = "Complete sign-in in the new window. This audit stays open in the current tab.";
  clearInterval(googleAuthCloseTimer);
  googleAuthCloseTimer = window.setInterval(() => {
    if (!googleAuthPopup?.closed) return;
    clearInterval(googleAuthCloseTimer);
    googleAuthCloseTimer = 0;
    googleAuthPopup = null;
    if (!googleAuthCompletionPromise) authMessage.textContent = "Sign-in window closed. The current audit was kept.";
  }, 500);
}

function completeGoogleSignIn(result) {
  if (googleAuthCompletionPromise) return googleAuthCompletionPromise;
  googleAuthCompletionPromise = (async () => {
    clearInterval(googleAuthCloseTimer);
    googleAuthCloseTimer = 0;
    if (googleAuthPopup && !googleAuthPopup.closed) googleAuthPopup.close();
    googleAuthPopup = null;
    if (!result?.ok) {
      const reason = result?.error || "oauth_failed";
      authMessage.textContent = `Google sign-in failed (${reason}). The current audit was kept.`;
      setStatus("Google sign-in failed", "error");
      return;
    }
    try {
      const session = await authFetch("/api/auth/me");
      renderAuthUser(session.user);
      await refreshAccessStatus({ force: true });
      setStatus(current ? "Signed in; audit retained" : "Signed in", "ok");
    } catch (error) {
      authMessage.textContent = `${error.message || "Session refresh failed."} The current audit was kept.`;
      setStatus("Session refresh failed", "error");
    }
  })().finally(() => {
    googleAuthCompletionPromise = null;
  });
  return googleAuthCompletionPromise;
}

async function submitAuthForm(event) {
  event.preventDefault();
  if (!authConfigState.password) {
    authMessage.textContent = "Password signup/login is not configured yet.";
    return;
  }
  authSubmit.disabled = true;
  try {
    const payload = {
      email: authFormEmail.value,
      password: authFormPassword.value,
    };
    if (authMode === "signup") payload.name = authFormName.value;
    const result = await authFetch(authMode === "signup" ? "/api/auth/signup" : "/api/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderAuthUser(result.user);
    if (result.verification?.required) {
      setStatus(result.verification.sent ? "Verify email" : "Verification setup needed", result.verification.sent ? "error" : "error");
      authMessage.textContent = result.verification.sent
        ? "Verification email sent. Please verify your email before using exports, feedback requests, or gated analysis."
        : "Account created, but email delivery is not configured yet. Ask the admin to enable verification email delivery.";
    }
    await refreshAccessStatus({ force: true });
  } catch (error) {
    authMessage.textContent = error.message || "Authentication failed.";
  } finally {
    authSubmit.disabled = !authConfigState.password;
  }
}

async function logoutAuth() {
  try {
    await authFetch("/api/auth/logout", { method: "POST" });
  } catch (error) {
    console.warn("Logout failed", error);
  }
  closeAccountPanel();
  accountProfileState = null;
  renderAuthUser(null);
  await refreshAccessStatus({ force: true });
}

async function resendVerificationEmail() {
  if (!currentAuthUser || currentAuthUser.email_verified) return;
  resendVerify.disabled = true;
  resendVerify.textContent = "Sending";
  try {
    const result = await authFetch("/api/auth/verification/send", { method: "POST" });
    if (result.verified) {
      setStatus("Email verified", "ok");
      currentAuthUser.email_verified = true;
      renderAuthUser(currentAuthUser);
      await refreshAccessStatus({ force: true });
      return;
    }
    setStatus(result.sent ? "Verification sent" : "Email delivery unavailable", result.sent ? "ok" : "error");
    authMessage.textContent = result.sent
      ? "Verification email sent. Please check your inbox."
      : "Email delivery is not configured yet. Ask the admin to enable verification email delivery.";
  } catch (error) {
    setStatus("Verification failed", "error");
    authMessage.textContent = error.message || "Could not send verification email.";
  } finally {
    resendVerify.disabled = false;
    resendVerify.textContent = "Resend verification";
  }
}

async function openAccountPanel({ consentOnly = false } = {}) {
  const anon = !currentAuthUser;
  openModal(accountBackdrop, { focus: accountClose });
  document.body.classList.toggle("consent-only-modal", anon || consentOnly);
  renderConsentPanel();
  if (!anon) await loadAccountProfile();
}

function updateAccessProfileInfo(profileId) {
  const profileDef = ACCESS_REQUEST_PROFILES[profileId];
  if (!profileDef) {
    accessProfileInfo.hidden = true;
    accessProfileInfo.replaceChildren();
    return;
  }
  accessProfileInfo.hidden = false;
  const p = document.createElement("p");
  p.textContent = profileDef.description;
  const ul = document.createElement("ul");
  for (const feat of profileDef.features) {
    const li = document.createElement("li");
    li.textContent = feat;
    ul.append(li);
  }
  accessProfileInfo.replaceChildren(p, ul);
}

async function openAccountPanelForCapability(featureIdOrTierId = "feedback") {
  if (!currentAuthUser) {
    openAuthModal("signup", "Create an account to request module access or send feedback. Static analysis remains open without login.");
    return;
  }
  document.body.classList.remove("consent-only-modal");
  openModal(accountBackdrop, { focus: accountRequestMessage });
  await loadAccountProfile();
  const profileId = profileIdForCapability(featureIdOrTierId) || featureIdOrTierId || "feedback";
  const validProfiles = [...accountRequestProfile.options].map((o) => o.value);
  accountRequestProfile.value = validProfiles.includes(profileId) ? profileId : "feedback";
  updateAccessProfileInfo(accountRequestProfile.value);
  if (!accountRequestTitle.value) {
    accountRequestTitle.value = requestDraftTitle(accountRequestProfile.value);
  }
  accountRequestMessage.focus();
}

function closeAccountPanel() {
  closeModal(accountBackdrop, { fallbackFocus: accountOpen });
  document.body.classList.remove("consent-only-modal");
}

async function loadAccountProfile() {
  accountProfileStatus.textContent = "Loading account workspace";
  accountRequestStatus.textContent = "Ready";
  try {
    const profile = await accountFetch("/api/account/profile");
    accountProfileState = profile;
    renderAccountCapabilities(profile.features || []);
    renderAccountRequests(profile.requests || []);
    const accountBound = profile.user.test_access?.account_bound !== false;
    accountProfileStatus.textContent = !accountBound
      ? "External test session / account required for requests"
      : profile.user.email_verified
        ? `${profile.user.name || profile.user.email} / ${profile.user.email}`
      : `${profile.user.name || profile.user.email} / ${profile.user.email} / email verification required before submitting requests`;
    accountRequestSubmit.disabled = !profile.user.email_verified || !accountBound;
    accountRequestStatus.textContent = accountBound ? profile.user.email_verified ? "Ready" : "Verify email first" : "Account required";
    if (profile.user.role === "admin") {
      adminPanel.hidden = false;
      renderAdminShortcut();
    } else {
      adminPanel.hidden = true;
      adminRequestList.replaceChildren();
    }
  } catch (error) {
    accountProfileStatus.textContent = error.message || "Account workspace failed to load.";
  }
}

function renderAccountCapabilities(features) {
  const visibleFeatures = medicalReportSurface
    ? features
    : features.filter((feature) => feature.id !== "regulatory_report");
  renderAccountCapabilityList(accountCapabilityList, accountCapabilityCount, visibleFeatures, (profileId) => {
    openAccountPanelForCapability(profileId);
  });
}

function renderAccountRequests(requests) {
  renderRequestList(accountRequestList, requests, { countNode: accountRequestCount });
}

async function submitAccountRequest(event) {
  event.preventDefault();
  if (!currentAuthUser) {
    openAuthModal("signup", "Sign in before sending a request.");
    return;
  }
  if (!currentAuthUser.email_verified) {
    accountRequestStatus.textContent = "Email verification required.";
    return;
  }
  accountRequestSubmit.disabled = true;
  accountRequestStatus.textContent = "Submitting";
  try {
    const profileId = accountRequestProfile.value;
    const profileDef = ACCESS_REQUEST_PROFILES[profileId];
    const result = await accountFetch("/api/account/requests", {
      method: "POST",
      body: JSON.stringify({
        type: profileDef ? profileDef.type : profileId === "bug" ? "bug" : "feedback",
        capability: profileDef ? profileDef.capability : profileId,
        title: accountRequestTitle.value,
        message: accountRequestMessage.value,
      }),
    });
    accountRequestTitle.value = "";
    accountRequestMessage.value = "";
    accountRequestStatus.textContent = "Submitted";
    renderAccountRequests(result.requests || []);
    if (currentAuthUser.role === "admin") renderAdminShortcut();
  } catch (error) {
    accountRequestStatus.textContent = error.message || "Submit failed.";
  } finally {
    accountRequestSubmit.disabled = false;
  }
}

function openAdminConsole() {
  window.open("/web/admin", "_blank", "noopener,noreferrer");
}

function renderAdminShortcut() {
  adminRequestList.replaceChildren(adminShortcutCard());
}

async function loadAdminRequests() {
  if (currentAuthUser?.role !== "admin") return;
  adminRequestList.replaceChildren(requestLoading("Loading admin queue"));
  try {
    const data = await accountFetch("/api/admin/requests");
    const requests = data.requests || [];
    renderRequestList(adminRequestList, requests, {
      emptyMessage: "No account requests yet.",
      cardFactory: adminRequestCard,
    });
  } catch (error) {
    adminRequestList.replaceChildren(requestLoading(error.message || "Admin queue failed."));
  }
}

function adminRequestCard(request) {
  const node = requestCard(request);
  node.classList.add("admin-request-card");
  const user = document.createElement("p");
  user.className = "request-user";
  user.textContent = `${request.user_name || request.user_email} / ${request.user_email} / ${request.user_access_profile || "verified"}`;
  const controls = document.createElement("div");
  controls.className = "admin-request-controls";
  const select = document.createElement("select");
  for (const value of ["new", "reviewing", "planned", "granted", "declined", "closed"]) {
    select.append(new Option(value, value));
  }
  select.value = request.status || "new";
  const note = document.createElement("textarea");
  note.rows = 3;
  note.value = request.admin_note || "";
  note.placeholder = "Admin note";
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Save";
  save.addEventListener("click", async () => {
    save.disabled = true;
    save.textContent = "Saving";
    try {
      await accountFetch(`/api/admin/requests/${encodeURIComponent(request.id)}`, {
        method: "PATCH",
        body: JSON.stringify({ status: select.value, admin_note: note.value }),
      });
      await loadAdminRequests();
      await loadAccountProfile();
    } catch (error) {
      save.textContent = error.message || "Failed";
      setTimeout(() => {
        save.textContent = "Save";
        save.disabled = false;
      }, 1200);
    }
  });
  controls.append(select, note, save);
  node.prepend(user);
  node.append(controls);
  return node;
}

async function accountFetch(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function currentCapabilities() {
  return capabilitiesForUser(currentAuthUser, accessGrantState);
}

let engineeringFormatterPromise = null;
let rawExportFormatterPromise = null;
let regulatoryFormatterPromise = null;
let reportRenderToken = 0;

async function loadEngineeringFormatter() {
  engineeringFormatterPromise ||= import("./lib/report-engineering-entry.js");
  return engineeringFormatterPromise;
}

async function loadRawExportFormatter() {
  rawExportFormatterPromise ||= import("./lib/report-raw-entry.js");
  return rawExportFormatterPromise;
}

async function loadRegulatoryFormatter() {
  regulatoryFormatterPromise ||= import("./lib/report-regulatory-entry.js");
  return regulatoryFormatterPromise;
}

async function ensureRawExportAllowed(label) {
  if (currentAuthUser?.role === "admin") {
    setStatus("Admin raw export authorized", "ok");
    return true;
  }
  if (currentAuthUser) {
    try {
      const access = await refreshAccessStatus({ check: true, force: true });
      if (access?.allowed?.raw_export) {
        setStatus("Account export authorized", "ok");
        return true;
      }
      setStatus("Raw export locked", "error");
      authMessage.textContent = `${label} requires a verified, authorized account.`;
      return false;
    } catch (error) {
      setStatus("Authorization check failed", "error");
      authMessage.textContent = `${label} requires a current account authorization check. Model bytes and reports were not uploaded.`;
      console.warn("Raw export authorization check failed", error);
      return false;
    }
  }
  openAuthModal("signup", `${label} is available after account registration. Static analysis remains usable without login.`);
  return false;
}

async function ensureRegulatoryReportAllowed(label = "Regulatory report") {
  if (currentAuthUser?.role === "admin") {
    setStatus("Admin regulatory export authorized", "ok");
    return true;
  }
  if (!currentAuthUser) {
    const message = `${label} requires a signed-in account authorized for the regulatory workspace. Engineering analysis remains usable without login.`;
    setStatus("Regulatory report requires sign in", "error");
    openAuthModal("signup", message);
    return false;
  }
  try {
    const access = await refreshAccessStatus({ check: true, force: true });
    const allowed = access?.allowed || {};
    if (allowed.regulatory_report) {
      setStatus("Regulatory report authorized", "ok");
      return true;
    }
    setStatus("Regulatory report locked", "error");
    authMessage.textContent = `${label} requires regulatory workspace authorization. Engineering Report access can remain separately enabled.`;
    await openAccountPanelForCapability("regulatory_report");
    return false;
  } catch (error) {
    setStatus("Authorization check failed", "error");
    authMessage.textContent = `${label} requires a current account authorization check. Model bytes and reports were not uploaded.`;
    console.warn("Regulatory authorization check failed", error);
    return false;
  }
}

async function ensureDeepBomAllowed() {
  if (!currentAuthUser) {
    setStatus("DEEPBOM requires sign in", "error");
    openAuthModal("signup", "DEEPBOM loads its controlled WASM module only for authorized accounts. Model bytes are not uploaded.");
    throw new Error("sign_in_required");
  }
  const response = await fetch("/api/analysis-module/deepbom/manifest", {
    method: "GET",
    credentials: "same-origin",
    headers: { accept: "application/json" },
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    accessGrantState = data.allowed ? data : accessGrantState;
    updateExportLockState();
    const reason = data.reason || data.error || `HTTP ${response.status}`;
    authMessage.textContent = `DEEPBOM is not enabled for this account (${reason}).`;
    setStatus("DEEPBOM authorization blocked", "error");
    throw new Error(reason);
  }
  setStatus("DEEPBOM authorized", "ok");
  return data;
}

async function ensureResearchModuleAllowed(feature, label) {
  if (!current || !currentModelBytes) {
    setStatus("Run audit first", "error");
    return false;
  }
  if (!modelSupportsCapability(current.format, "experimental_tflite_research")) {
    setStatus(`${label} is not available for ${modelFormatAdapter(current.format).label} metadata analysis`, "error");
    return false;
  }
  if (!currentAuthUser) {
    openAuthModal("signup", `${label} requires a signed-in account. Model bytes still stay in the browser.`);
    return false;
  }
  const capabilities = currentCapabilities();
  const capabilityAllowed = Boolean(
    feature === "perturbation_analysis" && capabilities.perturbation ||
    feature === "runtime_basin" && capabilities.runtime_basin ||
    feature === "deployment_sensitivity" && capabilities.deployment_sensitivity,
  );
  if (capabilityAllowed) return true;
  await openAccountPanelForCapability(feature);
  return false;
}

function updateExportLockState() {
  const capabilities = currentCapabilities();
  const reportTargetReady = reportTargetBinding().canCopy;
  const rawExportAllowed = capabilities.raw_export;
  const regulatoryAllowed = capabilities.regulatory_report;
  const deepBomAllowed = capabilities.deepbom;
  const rawLocked = !rawExportAllowed;
  applyGatedExportLabels(downloadMarkdown, [downloadRawData, downloadCsv, downloadMermaid, downloadVisualPngs, downloadEngineeringBundle], false, rawLocked, currentAuthUser);
  syncPublicPrintButton(printPublicReport, { hasAnalysis: Boolean(current), reportTargetReady });
  syncPublicVerificationButton(downloadPublicVerificationManifest, { hasAnalysis: Boolean(current), reportTargetReady });
  const selectedEvidencePackageProfile = resolveEvidencePackageProfile(evidencePackageProfile?.value);
  const selectedEvidenceLevel = resolveEvidenceLevelProfile(evidencePackageLevel?.value);
  if (evidencePackageProfile) evidencePackageProfile.disabled = !current || !reportTargetReady;
  if (evidencePackageLevel) evidencePackageLevel.disabled = !current || !reportTargetReady;
  syncPublicEvidencePackageButton(downloadPublicBundle, {
    hasAnalysis: Boolean(current),
    reportTargetReady,
    profileLabel: `${selectedEvidencePackageProfile.label} / ${selectedEvidenceLevel.label}`,
  });
  setAccountLockedButtons(
    [downloadMarkdown],
    false,
    "Download a login-free, watermarked Engineering Report whose visible body is bound by SHA-256.",
  );
  downloadMarkdown.disabled = !current || !reportTargetReady;
  if (downloadReviewHtml) downloadReviewHtml.disabled = !current || !reportTargetReady;
  setAccountLockedButtons(
    [downloadRawData, downloadCsv, downloadMermaid, downloadGraphSvg, downloadVisualPngs, downloadEngineeringBundle],
    rawLocked,
    rawLocked
      ? currentAuthUser
        ? "Download Raw Data requires an authorized account."
        : "Sign in to download raw data."
      : `Account-bound export enabled for ${currentAuthUser.email || currentAuthUser.name}.`,
  );
  const rawArtifactDisabled = !current || !rawExportAllowed || !reportTargetReady;
  const exportAvailability = modelExportAvailability(current);
  for (const button of [downloadRawData, downloadVisualPngs, downloadEngineeringBundle]) {
    if (button) button.disabled = rawArtifactDisabled;
  }
  if (downloadCsv) downloadCsv.disabled = rawArtifactDisabled || !exportAvailability.performanceDerivatives;
  if (downloadMermaid) downloadMermaid.disabled = rawArtifactDisabled || !exportAvailability.performanceDerivatives;
  if (downloadGraphSvg) downloadGraphSvg.disabled = rawArtifactDisabled || !exportAvailability.graph;
  downloadRawData.title = rawExportAllowed
    ? "Raw audit, ML-BOM, graphs, PNGs, ES256 signature, and server digest attestation."
    : currentAuthUser
      ? "Download Raw Data requires an authorized account."
      : "Sign in to unlock Download Raw Data.";
  downloadEngineeringBundle.title = rawExportAllowed
    ? "Download a compact ZIP with one Engineering Report and one consolidated evidence JSON."
    : currentAuthUser
      ? "Engineering bundle requires an authorized account."
      : "Sign in to unlock the engineering bundle.";
  if (engineeringBundleNote) {
    engineeringBundleNote.textContent = !current
      ? "Run a static audit first. Evidence Package profiles never include original model bytes or raw tensor values."
      : !reportTargetReady
        ? "Analyze or load the selected report target before downloading the selected package profile."
        : `${selectedEvidencePackageProfile.label}; ${selectedEvidenceLevel.label}: ${selectedEvidenceLevel.detail}. Lower levels omit higher-class findings, metrics, and unscoped CycloneDX documents. No sign-in is required; individual raw exports remain separately controlled.`;
  }
  if (medicalReportSurface) {
    setAccountLockedButtons(
      [downloadRegulatoryReport],
      false,
      "Download the login-free Regulatory Support Report. It is not a submission, approval, or release authorization.",
    );
    setAccountLockedButtons(
      [downloadEvidenceBundle],
      !regulatoryAllowed,
      regulatoryAllowed
        ? `Regulatory evidence bundle enabled for ${currentAuthUser.email || currentAuthUser.name}.`
        : currentAuthUser
          ? "The full regulatory evidence bundle requires regulatory workspace authorization."
          : "Sign in to request access to the full regulatory evidence bundle.",
    );
    if (downloadRegulatoryReport) downloadRegulatoryReport.disabled = !current || !reportTargetReady;
    downloadEvidenceBundle.disabled = !current || !regulatoryAllowed || !reportTargetReady;
    downloadEvidenceBundle.title = regulatoryAllowed
        ? "Download a ZIP with the full Engineering Bundle plus Regulatory Report and enabled Research evidence."
        : currentAuthUser
        ? "Regulatory evidence bundle requires regulatory workspace authorization."
        : "Sign in to unlock the regulatory evidence bundle.";
    if (evidenceBundleNote) {
      evidenceBundleNote.textContent = !current
        ? "Run a static audit first. The bundle never includes raw model weights."
        : regulatoryAllowed
          ? "Includes the full Engineering Bundle, then adds Regulatory Report and enabled Research module JSON. Raw model weights are not included."
          : currentAuthUser
            ? "Regulatory workspace authorization is required before the evidence bundle is enabled."
            : "Sign in to download a regulatory evidence bundle. Static analysis remains usable without login.";
    }
  }
  runDeepBom.classList.toggle("research-locked", !deepBomAllowed);
  const protectedSourceReady = Boolean(current && currentModelPayloadLoaded && modelSupportsCapability(current.format, "protected_source_analysis"));
  runDeepBom.disabled = !protectedSourceReady || !deepBomAllowed;
  runDeepBom.title = deepBomAllowed
    ? "DEEPBOM loads on demand for this account."
    : currentAuthUser
      ? "DEEPBOM requires authorization for advanced modules."
      : "Sign in to request advanced module access.";
  if (deepBomAccessNote) {
    deepBomAccessNote.textContent = deepBomAllowed
      ? "Access confirmed. The controlled WASM module loads only after Run Artifact Geometry is clicked. Browser delivery controls access, but is not cryptographic source secrecy."
      : "The controlled WASM module is not fetched for this account state. Request advanced module access to enable on-demand loading.";
  }
  exportContractController.render();
  updateModuleAccessState();
  renderReportPanel();
}

function setAccountLockedButtons(buttons, locked, title, lockedClass = "account-locked") {
  for (const button of buttons) {
    if (!button) continue;
    button.classList.remove("account-locked");
    if (locked) button.classList.add(lockedClass);
    button.title = title;
  }
}

function updateModuleAccessState() {
  const capabilities = currentCapabilities();
  const admin = capabilities.admin;
  const verified = Boolean(currentAuthUser?.email_verified || currentAuthUser?.role === "admin");
  const rawExportAllowed = capabilities.raw_export;
  const regulatoryAllowed = capabilities.regulatory_report;
  const deepBomAllowed = capabilities.deepbom;
  const perturbationAllowed = capabilities.perturbation;
  const runtimeBasinAllowed = capabilities.runtime_basin;
  const deploymentSensitivityAllowed = capabilities.deployment_sensitivity;
  const states = moduleAccessStatesFor(capabilities, currentAuthUser);
  const tfliteResearchReady = Boolean(current && currentModelPayloadLoaded && modelSupportsCapability(current.format, "experimental_tflite_research"));
  const moduleResults = {
    deepbom: deepBomResult,
    perturbation: perturbationResult,
    runtime_basin: combineModuleResults([perturbationResult, runtimeBasinResult]),
    deployment_sensitivity: deployCurvatureResult,
  };

  for (const tab of moduleTabs) {
    const state = states[tab.dataset.moduleTab] || states.static;
    tab.classList.toggle("available", state.className === "available");
    tab.classList.toggle("locked", state.className === "locked");
    tab.classList.toggle("planned", state.className === "planned");
    tab.setAttribute("aria-disabled", state.locked ? "true" : "false");
    const badge = tab.querySelector("em");
    if (badge) {
      badge.textContent = moduleTabStatusTextFor({
        moduleId: tab.dataset.moduleTab,
        accessState: state,
        result: moduleResults[tab.dataset.moduleTab],
        capabilities,
        hasCurrent: Boolean(current),
      });
    }
  }
  const activeState = states[activeModule] || states.static;
  if (activeState.locked) {
    setActiveModule("engineering_report");
  }

  for (const step of workflowModuleSteps) {
    const state = states[step.dataset.workflowModule] || states.static;
    step.classList.toggle("module-available", state.className === "available");
    step.classList.toggle("module-locked", state.className === "locked");
    step.classList.toggle("module-planned", state.className === "planned");
    const status = step.querySelector("p");
    if (status) status.textContent = moduleWorkflowDescription(step.dataset.workflowModule, state.label);
    step.title = state.locked ? `${state.label}. ${moduleWorkflowDescription(step.dataset.workflowModule, "Module access required.")}` : "";
  }

  if (moduleAccessStatus) {
    moduleAccessStatus.textContent = admin
      ? "Admin access: all modules"
    : regulatoryAllowed
      ? "Regulatory workspace enabled"
      : perturbationAllowed || runtimeBasinAllowed || deploymentSensitivityAllowed
      ? "Advanced modules enabled"
      : rawExportAllowed
        ? "Raw exports enabled"
        : currentAuthUser
          ? verified
            ? "Open reports + requests"
            : "Email verification required"
          : "Open audit + reports";
  }
  if (requestDeepBomAccess) requestDeepBomAccess.hidden = deepBomAllowed;
  const localAnalysisAllowed = perturbationAllowed || runtimeBasinAllowed;
  updateResearchModulePanel(runtimeBasinPanelNote, perturbationPanelAction, localAnalysisAllowed, current, {
    availableNote: "Runs perturbation drift and runtime basin checks locally in the browser.",
    availableAction: "Run Drift Analysis",
    requestNote: "Request Researcher access to run local analysis on this model.",
    requestAction: "Request Researcher access",
  });
  if (perturbationPanelAction && current && !tfliteResearchReady) {
    perturbationPanelAction.disabled = true;
    perturbationPanelAction.title = `${modelFormatAdapter(current.format).label} supports static evidence export here; perturbation requires a loaded TFLite runtime artifact.`;
  }
  updateResearchModulePanel(deploymentSensitivityPanelNote, deploymentSensitivityPanelAction, deploymentSensitivityAllowed, current, {
    availableNote: "Runs deploy-domain finite-difference stability probes on the TFLite runtime path.",
    availableAction: "Run Deployment Sensitivity",
    requestNote: "Request Researcher access to run deploy-domain finite-difference probes.",
    requestAction: "Request Researcher access",
  });
  if (deploymentSensitivityPanelAction && current && !tfliteResearchReady) {
    deploymentSensitivityPanelAction.disabled = true;
    deploymentSensitivityPanelAction.title = `${modelFormatAdapter(current.format).label} does not expose the TFLite finite-difference runtime path.`;
  }
  if (modelViewerBtn) {
    modelViewerBtn.hidden = !current || !modelSupportsCapability(current.format, "model_tomography");
    modelViewerBtn.textContent = "Artifact Viewer";
    modelViewerBtn.disabled = false;
  }
}

bindPublicAuditPrintButton(printPublicReport, {
  getAnalysis: () => current,
  getBinding: reportTargetBinding,
  bindingMatches: reportBindingMatchesAnalysis,
  getContext: currentReportContext,
  getScope: (analysis) => formatEvidenceScope(analysis.format, { analysis, runtimeEvidence: runtimeAssignmentEvidence }),
  sha256Hex,
  origin: location.origin,
  setStatus,
  formatError: shortError,
});
bindPublicVerificationButton(downloadPublicVerificationManifest, {
  getAnalysis: () => current,
  getBinding: reportTargetBinding,
  bindingMatches: reportBindingMatchesAnalysis,
  ensureHash: ensureModelHash,
  getContext: currentReportContext,
  getScope: (analysis) => formatEvidenceScope(analysis.format, { analysis, runtimeEvidence: runtimeAssignmentEvidence }),
  getRuntimeEvidence: () => runtimeAssignmentEvidence,
  origin: location.origin,
  filename: () => currentArtifactFilename("public_report_verification_manifest.json"),
  download: downloadText,
  serialize: jsonForDownload,
  setStatus,
});

copyReportBtn?.addEventListener("click", async () => {
  const binding = reportTargetBinding();
  const analysis = current;
  if (!binding.canCopy || !reportBindingMatchesAnalysis(binding, analysis)) {
    setStatus("Analyze or load the report binding before copying", "error");
    return;
  }
  try {
    const formatter = await loadEngineeringFormatter();
    if (current !== analysis
      || reportTargetBinding().bindingScope !== binding.bindingScope
      || reportTargetBinding().targetId !== binding.targetId
      || !reportBindingMatchesAnalysis(binding, current)) {
      throw new Error("report binding changed while the report was being generated");
    }
    const text = formatter.buildEngineeringReport(artifactIrBackedView(analysis), currentReportContext());
    if (!text.startsWith("#")) throw new Error("generated report is not valid markdown");
    await copyTextToClipboard(text);
    const original = copyReportBtn.textContent;
    copyReportBtn.textContent = "Copied!";
    setStatus(`Copied report for ${binding.bindingScope === "artifact" ? "the analyzed artifact" : analysis.target_profile?.label || binding.targetId}`, "ok");
    setTimeout(() => { copyReportBtn.textContent = original; }, 1500);
  } catch (error) {
    console.error("[copyReport]", error);
    setStatus(`Report copy failed: ${shortError(error)}`, "error");
  }
});

let reportVerification = null; // { code, reportHash, origin, authenticationTag, authenticationAlgorithm }

registerVerifyBtn?.addEventListener("click", async () => {
  if (!currentAuthUser) {
    openAuthModal("signup", "Registering a verification code requires a signed-in account. Analysis stays open without login.");
    return;
  }
  if (!current) { setStatus("Run an audit before registering a code", "error"); return; }
  if (!reportTargetBinding().canCopy) {
    setStatus("Analyze or load the selected report target before registering a fingerprint", "error");
    return;
  }
  const btn = registerVerifyBtn;
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = "Registering…";
  try {
    const formatter = await loadEngineeringFormatter();
    reportVerification = null;
    const body = formatter.reportBodyForFingerprint(formatter.buildEngineeringReport(currentAnalysisView(), currentReportContext()));
    const reportHash = await sha256Hex(new TextEncoder().encode(body));
    const res = await fetch("/api/report/verify/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_hash: reportHash,
        artifact_sha256: current.model_sha256 || "",
        analyzer_version: ANALYZER_VERSION,
        rulepack_version: RULEPACK_VERSION,
      }),
    });
    if (!res.ok) throw new Error(`register failed (${res.status})`);
    const data = await res.json();
    reportVerification = {
      code: data.code,
      reportHash: data.report_hash,
      origin: location.origin,
      authenticationTag: data.authentication_tag,
      authenticationAlgorithm: data.authentication_algorithm,
    };
    renderReportPanel(); // re-render so the report now carries the code footer
    setStatus(`Report fingerprint registered: ${data.code}`, "ok");
  } catch (error) {
    console.error("[verify-register]", error);
    setStatus("Could not register verification code", "error");
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
});

function reportTargetId() {
  return resolveReportTargetId({
    targetProfileApplicable: !(current && !modelSupportsCapability(current.format, "target_profiles")),
    availableTargetIds: targetProfiles.map((profile) => profile.id),
    requestedTargetId: reportTargetRequestedId,
    activeTargetId: current?.target_profile?.id,
    selectedTargetId: targetSelect?.value,
    fallbackTargetId: targetProfiles[0]?.id,
  });
}

function reportTargetBinding() {
  const cachedTargetIds = [...targetAnalysisCache.entries()]
    .filter(([targetId, analysis]) => analysis?.target_profile?.id === targetId)
    .map(([targetId]) => targetId);
  return resolveReportTargetBinding({
    requestedTargetId: reportTargetId(),
    activeTargetId: current?.target_profile?.id || "",
    cachedTargetIds,
    hasArtifact: Boolean(current && currentModelBytes && currentFilename),
    artifactOnly: Boolean(current && !modelSupportsCapability(current.format, "target_profiles")),
  });
}

function populateReportTargetProfiles() {
  populateReportTargetSelect(reportTargetSelect, {
    artifactOnly: Boolean(current && !modelSupportsCapability(current.format, "target_profiles")),
    targetProfiles,
    selectedTargetId: reportTargetId(),
  });
}

function syncReportTargetControls() {
  const targetId = reportTargetId();
  if (reportTargetSelect) {
    const artifactOnly = Boolean(current && !modelSupportsCapability(current.format, "target_profiles"));
    if (artifactOnly || reportTargetSelect.options.length !== targetProfiles.length) populateReportTargetProfiles();
    reportTargetSelect.value = targetId;
    reportTargetSelect.disabled = artifactOnly || !currentModelBytes || !currentFilename;
  }
  const binding = reportTargetBinding();
  const copy = reportTargetControlCopy(binding);
  if (reportTargetStatus) {
    reportTargetStatus.textContent = copy.statusText;
    reportTargetStatus.className = binding.analyzed ? "ready" : binding.state === "required" ? "required" : "";
  }
  if (reportTargetAnalyzeBtn) {
    reportTargetAnalyzeBtn.textContent = copy.analyzeText;
    reportTargetAnalyzeBtn.disabled = copy.analyzeDisabled;
  }
  if (copyReportBtn) {
    copyReportBtn.disabled = !binding.canCopy;
  }
  if (registerVerifyBtn) registerVerifyBtn.disabled = !binding.canCopy;
  return binding;
}

function renderReportPanel() {
  const token = ++reportRenderToken;
  const binding = syncReportTargetControls();
  const boundTargetLabel = binding.bindingScope === "artifact"
    ? "Artifact-only / no execution target"
    : reportTargetLabelContract({ targetId: binding.targetId, targetProfiles });
  if (reportPreview) {
    reportPreview.classList.remove("report-protected");
    if (!current) {
      reportPreview.textContent = "Run a static audit to generate the report preview.";
    } else if (binding.state === "required") {
      reportPreview.textContent = `Analysis required for ${boundTargetLabel}. Select Analyze target before previewing, copying, or exporting this report.`;
    } else if (binding.state === "cached") {
      reportPreview.textContent = `${boundTargetLabel} was already analyzed for this artifact. Select Use analyzed to bind the preview and exports to that result.`;
    } else {
      reportPreview.textContent = "Loading Engineering Report formatter...";
      const analysis = current;
      const expectedTargetId = binding.targetId;
      const expectedBindingScope = binding.bindingScope;
      loadEngineeringFormatter()
        .then((formatter) => {
          if (token === reportRenderToken
            && current === analysis
            && reportTargetBinding().bindingScope === expectedBindingScope
            && reportTargetBinding().targetId === expectedTargetId
            && reportBindingMatchesAnalysis(binding, analysis)) {
            reportPreview.textContent = formatter.buildEngineeringReport(artifactIrBackedView(analysis), currentReportContext());
          }
        })
        .catch((error) => {
          if (token === reportRenderToken) {
            reportPreview.textContent = `Engineering Report formatter is unavailable: ${shortError(error)}`;
          }
        });
    }
  }
  if (reportPreviewTitle) {
    reportPreviewTitle.textContent = binding.bindingScope === "artifact" || binding.targetId
      ? `Engineering report / ${boundTargetLabel}`
      : "Engineering report";
  }
  if (reportPreviewStatus) {
    reportPreviewStatus.textContent = !current
      ? "Not generated"
      : !binding.canCopy
        ? binding.state === "cached" ? "Load analyzed" : "Analysis required"
        : "Report ready";
    reportPreviewStatus.className = current && binding.canCopy ? "ready" : "";
  }
  if (regulatoryReportPreview) {
    if (!current) {
      regulatoryReportPreview.textContent = "Run a static audit to generate the regulatory report preview.";
    } else {
      regulatoryReportPreview.textContent = "Loading Regulatory Report formatter...";
      loadRegulatoryFormatter()
        .then((formatter) => {
          if (token === reportRenderToken && current) {
            regulatoryReportPreview.textContent = formatter.buildRegulatoryReport(currentAnalysisView(), currentRegulatoryReportContext());
          }
        })
        .catch((error) => {
          if (token === reportRenderToken) {
            regulatoryReportPreview.textContent = `Regulatory Report formatter is unavailable: ${shortError(error)}`;
          }
        });
    }
  }
  if (regulatoryReportPreviewTitle) {
    regulatoryReportPreviewTitle.textContent = "Regulatory report";
  }
  if (regulatoryReportPreviewStatus) {
    regulatoryReportPreviewStatus.textContent = !current ? "Not generated" : "Report ready";
    regulatoryReportPreviewStatus.className = current ? "ready" : "";
  }
  renderEvidenceBundleScope();
}

function renderEvidenceBundleScope() {
  renderEngineeringBundleScope();
  renderRegulatoryBundleScope();
}

function renderEngineeringBundleScope() {
  renderBundleScope(engineeringBundleScope, engineeringBundleItems(Boolean(current)), engineeringBundleProgress);
}

function renderRegulatoryBundleScope() {
  renderBundleScope(evidenceBundleScope, regulatoryBundleItems(currentCapabilities(), Boolean(current)), evidenceBundleProgress);
}

function setEvidenceBundleProgress(id, status, detail = "") {
  setBundleProgress("regulatory", id, status, detail);
}

function setEngineeringBundleProgress(id, status, detail = "") {
  setBundleProgress("engineering", id, status, detail);
}

function setBundleProgress(scope, id, status, detail = "") {
  if (scope === "engineering") {
    if (!engineeringBundleProgress) engineeringBundleProgress = {};
    engineeringBundleProgress[id] = { status, detail };
  } else {
    if (!evidenceBundleProgress) evidenceBundleProgress = {};
    evidenceBundleProgress[id] = { status, detail };
  }
  renderEvidenceBundleScope();
}

function updateResearchModulePanel(note, action, allowed, hasModel, copy) {
  if (note) note.textContent = allowed ? copy.availableNote : copy.requestNote;
  if (action) {
    action.textContent = allowed ? copy.availableAction : copy.requestAction;
    action.disabled = Boolean(allowed && !hasModel);
    action.classList.toggle("available-static", Boolean(allowed));
    action.removeAttribute("aria-disabled");
    action.title = allowed && !hasModel ? "Run a static audit first." : "";
  }
}

async function refreshAccessStatus({ check = false, force = false } = {}) {
  if (!force && accessGrantState) return accessGrantState;
  const path = check ? "/api/access/check" : "/api/access/status";
  const options = check ? { method: "POST" } : {};
  const status = await accessFetch(path, options);
  accessGrantState = status;
  if (status.user) {
    currentAuthUser = status.user;
    authProvider.textContent = `${accessLabel(status.user, accessGrantState)} / ${providerLabel(status.user.provider)}`;
  }
  updateExportLockState();
  return status;
}

async function accessFetch(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok && !data.allowed) {
    return data;
  }
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function authFetch(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function populateTargetProfiles() {
  let builtIn = [];
  try {
    builtIn = target_profiles() || [];
  } catch (error) {
    console.warn("Target profile load failed", error);
  }
  // A custom target is stored as a specification, not as a resolved profile; it
  // is listed beside the built-ins and resolved at the analyzer boundary.
  customTargetSpecs = loadCustomTargets();
  const customProfiles = customTargetSpecs.map((spec) =>
    customTargetStub(spec, builtIn.find((profile) => profile.id === spec.base)));
  targetProfiles = [...builtIn, ...customProfiles];
  if (!targetProfiles.length) {
    targetSelect.replaceChildren(new Option("Android mid-range / Cortex-A55", "android_mid_a55"));
    renderTargetSwitcher();
    return;
  }
  targetSelect.replaceChildren(
    ...targetProfiles.map((profile) => new Option(profile.label, profile.id)),
  );
  const saved = readSavedTarget();
  const fallback = targetProfiles.find((profile) => profile.id === "android_mid_a55")?.id || targetProfiles[0].id;
  targetSelect.value = targetProfiles.some((profile) => profile.id === saved) ? saved : fallback;
  populateReportTargetProfiles();
  renderTargetSwitcher();
}

function selectedTargetId() {
  return activeTargetId || targetSelect.value || targetProfiles[0]?.id || "android_mid_a55";
}

function targetLabel(targetId) {
  return targetProfiles.find((profile) => profile.id === targetId)?.label
    || TARGET_PILL_LABELS[targetId]
    || targetId
    || "unbound target";
}

function syncTargetTransitionUi() {
  const transition = targetAnalysisTransition;
  const analyzedId = current?.target_profile?.id || activeTargetId || "";
  const requestedId = transition?.requestedTargetId || "";
  const stale = Boolean(transition && analyzedId && analyzedId !== requestedId);
  workflowConsole?.classList.toggle("target-results-stale", stale);
  if (transition) workflowConsole?.setAttribute("aria-busy", "true");
  else workflowConsole?.removeAttribute("aria-busy");
  if (!targetStaleNotice) return;
  targetStaleNotice.hidden = !transition;
  targetStaleNotice.classList.toggle("stale", stale);
  if (!transition) {
    targetStaleNotice.textContent = "";
    return;
  }
  targetStaleNotice.textContent = stale
    ? `Displayed results remain bound to ${targetLabel(analyzedId)}. Reanalysis for ${targetLabel(requestedId)} is running; do not interpret the existing panels as results for the requested target.`
    : `Analysis for ${targetLabel(requestedId)} is running. Target-bound results will be available after the audit completes.`;
}

async function requestTargetAnalysis(requestedTargetId, { keepTab = true, keepModule = false } = {}) {
  if (!requestedTargetId) throw new Error("A target profile is required.");
  if (targetAnalysisTransitionPromise) {
    if (targetAnalysisTransition?.requestedTargetId === requestedTargetId) return targetAnalysisTransitionPromise;
    throw new Error(`Target analysis for ${targetLabel(targetAnalysisTransition?.requestedTargetId)} is already running.`);
  }
  if (!currentModelBytes || !currentFilename) throw new Error("Select an artifact before changing the analyzed target.");
  const previousWorkspace = getActiveWorkspace();
  const previousModule = activeModule;
  const previous = {
    current,
    currentDeploymentFrontier,
    currentDeploymentDelta,
    currentDelegationRepair,
    runtimeAssignmentEvidence,
    activeTargetId,
    reportTargetRequestedId,
    targetSelectValue: targetSelect.value,
    runAuditDisabled: runAudit.disabled,
  };
  targetAnalysisTransition = {
    requestedTargetId,
    previousTargetId: current?.target_profile?.id || activeTargetId || "",
  };
  targetSelect.value = requestedTargetId;
  targetSelect.disabled = true;
  runAudit.disabled = true;
  updateWorkflowState("running");
  renderTargetSwitcher();
  const transitionPromise = (async () => {
    try {
      const result = await analyzeLoadedModel(currentFilename, requestedTargetId, { keepTab, keepModule });
      writeSavedTarget(requestedTargetId);
      return result;
    } catch (error) {
      current = previous.current;
      currentDeploymentFrontier = previous.currentDeploymentFrontier;
      currentDeploymentDelta = previous.currentDeploymentDelta;
      currentDelegationRepair = previous.currentDelegationRepair;
      runtimeAssignmentEvidence = previous.runtimeAssignmentEvidence;
      activeTargetId = previous.activeTargetId;
      reportTargetRequestedId = previous.reportTargetRequestedId;
      targetSelect.value = previous.targetSelectValue;
      if (current) {
        await render(current, { keepTab: true, keepModule: true });
        setActiveModule(previousModule);
        setActiveWorkspace(previousWorkspace, { force: true });
      }
      throw error;
    } finally {
      targetAnalysisTransition = null;
      targetAnalysisTransitionPromise = null;
      targetSelect.disabled = false;
      runAudit.disabled = previous.runAuditDisabled;
      syncTargetTransitionUi();
      renderTargetSwitcher();
    }
  })();
  targetAnalysisTransitionPromise = transitionPromise;
  renderTargetSwitcher();
  return transitionPromise;
}

let _sessionNonce = null;
let _sessionNonceFetchedAt = 0;

async function refreshSessionNonce() {
  try {
    const res = await fetch("/api/nonce");
    if (!res.ok) return;
    const data = await res.json();
    _sessionNonce = data.token;
    _sessionNonceFetchedAt = Date.now();
  } catch {
  }
}

function renderAuditClaimBoundary(format, analysis = null) {
  renderAuditClaimBoundaryView({
    format,
    analysis,
    runtimeEvidence: runtimeAssignmentEvidence,
    placementOptions: executionPlacementOptions(),
  });
  renderFormatCapabilityMatrix(formatCapabilityPanel, format, { analysis, runtimeEvidence: runtimeAssignmentEvidence });
  renderRuntimeEvidenceClosure(runtimeEvidenceClosure, analysis, runtimeAssignmentEvidence);
}

function executionPlacementOptions() {
  return {
    selectedProfileId: selectedAcceleratorProfileId,
    selectedProfileIds: selectedPlacementProfileIds,
    onProfileSelect: selectAcceleratorProfile,
    onProfileSelectionChange: selectPlacementProfiles,
  };
}

function renderExecutionPlacementView(root, analysis, runtimeEvidence = null) {
  return renderExecutionPlacementViewBase(root, analysis, runtimeEvidence, executionPlacementOptions());
}

function syncFormatWorkflowVisibility(analysis = current) {
  syncFormatWorkflowVisibilityView({
    format: analysis?.format || pendingModelInspection?.formatId || "",
    analysis,
    workflowSteps,
    moduleTabs,
    currentUser: currentAuthUser,
    activeModule,
    setActiveModule,
  });
}

async function analyzeFile(file) {
  const auditStarted = performance.now();
  const controlTop = runAudit.getBoundingClientRect().top;
  try {
    setStatus("Analyzing");
    auditProgressController.begin(2, "Preparing analyzer", { ceiling: 7, step: 1 });
    updateWorkflowState("running");
    runAudit.disabled = true;
    runAudit.textContent = "Running";
    analysisPlanStatus.textContent = "Running audit";
    await nextPaint();
    const format = pendingModelInspection?.formatId || detectModelFormat(file.name);
    if (modelSupportsCapability(format, "target_profiles")) await ensureAnalyzerReady();
    setStatus("Reading model");
    auditProgressController.begin(8, "Reading model bytes", { ceiling: 15, step: 2 });
    await nextPaint();
    const formatGate = modelFormatGate(format);
    const { adapter } = formatGate;
    if (formatGate.blocked) throw new Error(formatGate.message);
    if (pendingArtifactBundleFiles.length) {
      const parsed = await readArtifactBundle(pendingArtifactBundleFiles, {
        onProgress: ({ index, count, phase }) => auditProgressController.set(8 + Math.round((index + 1) / count * 7), `${phase} ${index + 1}/${count}`, "running"),
      });
      current = parsed.analysis;
      currentModelBytes = parsed.retainedBytes;
      currentModelPayloadLoaded = false;
    } else if (["gguf", "safetensors"].includes(format)) {
      const parsed = await readMetadataModelFile(file, format, {
        onProgress: ({ index, count }) => auditProgressController.set(8 + Math.round((index + 1) / count * 7), `Payload ${index + 1}/${count}`, "running"),
      });
      current = parsed.analysis;
      currentModelBytes = parsed.retainedBytes;
      currentModelPayloadLoaded = false;
    } else if (format === "coreml") {
      const parsed = await readCoreMlModelFile(file);
      current = parsed.analysis;
      currentModelBytes = parsed.retainedBytes;
      currentModelPayloadLoaded = false;
    } else {
      currentModelBytes = new Uint8Array(await file.arrayBuffer());
      currentModelPayloadLoaded = true;
    }
    currentFilename = current?.filename || file.name;
    if (pendingRuntimeProfile) runtimeProfileModal.close();
    runtimeAssignmentEvidence = null;
    resetAdvancedResultState();
    performanceVisualController.resetTargetComparisonCache();
    resetDeepBomPanel();
    const timing = await analyzeLoadedModel(currentFilename, "", { finalize: false });
    calibrationValidationController.reset(current?.model_sha256 || null);
    const durationMs = performance.now() - auditStarted;
    const comparisonTargetCount = Number(current?.deployment_frontier?.target_count || 1);
    recordAuditTiming({
      format: current?.format || detectModelFormat(file.name, currentModelBytes),
      sizeBytes: current?.file_size_bytes || file.size,
      comparisonTargetCount,
      durationMs,
    });
    if (current) {
      current.static_audit_timing = {
        schema: "deepbom.static_audit_timing.v1",
        evidence_class: "MEASURED_BROWSER_WALL_CLOCK",
        wall_ms: durationMs,
        core_static_analysis_ms: Number(timing?.coreAnalysisMs || 0),
        comparison_target_count: comparisonTargetCount,
      };
    }
    analysisEstimate.textContent = formatMeasuredAudit(durationMs);
    analysisEstimateNote.textContent = "Measured from Run Static Audit to rendered completion on this browser; retained locally to calibrate future estimates.";
    pendingModelFile = file;
    const scope = formatEvidenceScope(current?.format || format, { analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence });
    analysisPlanStatus.textContent = scope.completion;
    renderAuditClaimBoundary(current?.format || format, currentAnalysisView());
    syncFormatWorkflowVisibility(current);
    updateWorkflowState("audited");
    renderDeepBomSkeleton();
    renderDriftSkeleton();
    renderDeploymentSensitivitySkeleton();
    runAudit.disabled = false;
    runAudit.textContent = formatAuditButtonLabel(current?.format || format, { rerun: true });
    auditProgressController.set(100, "Complete", "complete", { step: 12 });
    await nextPaint();
    await nextPaint();
    setStatus(scope.completion, "ok");
    if (matchMedia("(max-width: 820px)").matches) {
      auditWorkbench.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      window.scrollBy(0, runAudit.getBoundingClientRect().top - controlTop);
    }
  } catch (error) {
    console.error("[analyzeFile]", error);
    const errorMsg = error?.message || String(error) || "Unknown error";
    runAudit.disabled = !pendingModelFile;
    runAudit.textContent = formatAuditButtonLabel(pendingModelInspection?.formatId);
    analysisPlanStatus.textContent = `Audit failed: ${errorMsg}`;
    auditProgressController.set(null, "Audit failed", "error");
    if (runtimeGuardCode !== RUNTIME_OK) {
      return;
    }
    updateWorkflowState("error");
    setStatus(`Audit failed: ${errorMsg}`, "error");
    summary.innerHTML = "";
    actions.hidden = true;
    auditWorkbench.hidden = true;
    insightDashboard.hidden = true;
    perfVisuals.hidden = true;
    inferencePanel.hidden = true;
    deepBomPanel.hidden = true;
    resetResearchModulePanels();
    graphExplorer.hidden = true;
    if (redesignPanel) redesignPanel.hidden = true;
    diagramSection.hidden = true;
    tables.hidden = true;
  }
}

async function stageArtifactBundle(files) {
  try {
    setStatus("Inspecting package");
    const bundleInspection = await inspectArtifactBundle(files);
    await stageModelFile(bundleInspection.rootFile, { bundleFiles: files, bundleName: bundleInspection.displayName, bundleFormat: bundleInspection.format });
    selectedModelName.textContent = bundleInspection.displayName;
    selectedModelName.title = bundleInspection.displayName;
    selectedModelName.setAttribute("aria-label", `Selected artifact package: ${bundleInspection.displayName}`);
    const packageBinding = bundleInspection.kind === "safetensors_single_repository"
      ? `single tensor file${bundleInspection.config ? " + config.json" : ""}`
      : bundleInspection.kind === "safetensors_shards"
        ? `shard index${bundleInspection.config ? " + config.json" : ""}`
        : "package manifest";
    selectedModelMeta.textContent = `${modelFormatAdapter(bundleInspection.format).label} package / ${files.length} selected file(s) / ${packageBinding} resolved`;
  } catch (error) {
    console.error("[stageArtifactBundle]", error);
    pendingArtifactBundleFiles = [];
    pendingArtifactBundleName = "";
    runAudit.disabled = true;
    runAudit.textContent = "Invalid package";
    analysisPlanStatus.textContent = `Package rejected: ${error?.message || error}`;
    setStatus("Package rejected", "error");
  }
}

async function stageModelFile(file, { bundleFiles = [], bundleName = "", bundleFormat = "", publicSample = null, publicSampleCompanions = null } = {}) {
  if (pendingRuntimeProfile) runtimeProfileModal.close();
  reportVerification = null;
  pendingModelFile = file;
  pendingArtifactBundleFiles = [...bundleFiles];
  pendingArtifactBundleName = bundleName;
  sampleLibraryController?.setActive(publicSample);
  pendingPublicSampleCompanions = publicSampleCompanions && Object.keys(publicSampleCompanions).length
    ? publicSampleCompanions : null;
  currentExternalDataFiles = [];
  syncExternalDataControl(detectModelFormat(file.name));
  pendingModelInspection = null;
  clearNodeEdgeEvidenceOverlayState();
  current = null;
  currentArtifactIrContext = null;
  currentDeploymentFrontier = null;
  currentDeploymentDelta = null;
  currentDelegationRepair = null;
  currentModelBytes = null;
  currentModelPayloadLoaded = false;
  currentFilename = "";
  staticAuditWorkerClient.reset();
  auditProgressController.reset();
  activeTargetId = "";
  reportTargetRequestedId = targetSelect.value || "";
  targetAnalysisCache.clear();
  currentLowNormStatMap = new Map();
  resetAdvancedResultState();
  performanceVisualController.resetTargetComparisonCache();
  selectedOpIndex = null;
  resetExplorerInteractionState();
  resetDeepBomPanel();
  resetAnalysisViews();
  modelPlan.hidden = false;
  runAudit.disabled = true;
  runAudit.textContent = "Run Static Audit";
  renderAuditClaimBoundary("");
  renderModelPlan(selectedModelCopy(file));
  setStatus("Artifact selected", "ok");
  updateWorkflowState("selected");
  try {
    pendingModelInspection = await inspectModelFile(file, (header) => auditEstimateOptions(file, header));
    if (bundleFormat) {
      pendingModelInspection = {
        ...pendingModelInspection,
        formatId: bundleFormat,
        format: modelFormatAdapter(bundleFormat).label,
      };
    }
    renderStagedModel(file, pendingModelInspection);
    if (pendingArtifactBundleName) selectedModelName.textContent = pendingArtifactBundleName;
    const adapter = modelFormatAdapter(pendingModelInspection.formatId);
    runAudit.disabled = adapter.load === "rejected" || adapter.analyzer === "not_implemented";
    runAudit.textContent = runAudit.disabled ? "Format unavailable" : formatAuditButtonLabel(pendingModelInspection.formatId);
    prepareStagedTfliteTargets(file, pendingModelInspection);
  } catch (error) {
    console.warn("Light model inspection failed", error);
    pendingModelInspection = estimateModelAnalysis(file, null, auditEstimateOptions(file));
    if (bundleFormat) {
      pendingModelInspection = {
        ...pendingModelInspection,
        formatId: bundleFormat,
        format: modelFormatAdapter(bundleFormat).label,
      };
    }
    renderStagedModel(file, pendingModelInspection);
    if (pendingArtifactBundleName) selectedModelName.textContent = pendingArtifactBundleName;
    runAudit.disabled = false;
    runAudit.textContent = formatAuditButtonLabel(pendingModelInspection.formatId);
    prepareStagedTfliteTargets(file, pendingModelInspection);
  }
}

async function prepareStagedTfliteTargets(file, inspection) {
  if (inspection?.formatId !== "tflite") return;
  try {
    await ensureAnalyzerReady();
    if (pendingModelFile !== file || current || pendingModelInspection !== inspection) return;
    pendingModelInspection = estimateModelAnalysis(file, {
      readBytes: inspection.readBytes,
      elapsed: inspection.probeMs,
    }, auditEstimateOptions(file));
    renderTargetSwitcher();
    renderModelPlan(stagedModelCopy(file, pendingModelInspection, selectedTargetLabel()));
    setStatus("Artifact selected", "ok");
  } catch (error) {
    if (pendingModelFile !== file || pendingModelInspection !== inspection) return;
    console.error("[stagedTfliteTargets]", error);
    runAudit.disabled = true;
    runAudit.textContent = "Analyzer unavailable";
  }
}

function syncExternalDataControl(format) {
  const normalized = String(format || "").toLowerCase();
  const ptdArtifact = normalized === "executorch" && /\.ptd$/i.test(pendingModelFile?.name || current?.filename || "");
  renderExternalDataStatus({
    control: onnxExternalDataControl,
    status: onnxExternalDataStatus,
    format: ptdArtifact ? "" : normalized,
    files: currentExternalDataFiles,
    evidence: normalized === "onnx" ? current?.onnx_external_data : current?.executorch_program,
  });
}

async function stageExternalDataSelection(files) {
  if (!pendingModelFile) return;
  const format = detectModelFormat(pendingModelFile.name);
  if (!["onnx", "executorch"].includes(format) || format === "executorch" && !/\.pte$/i.test(pendingModelFile.name)) return;
  const inputs = [onnxExternalDataInput, onnxExternalDataDirectoryInput].filter(Boolean);
  inputs.forEach((input) => { input.disabled = true; });
  runAudit.disabled = true;
  onnxExternalDataStatus.textContent = "Reading selected data";
  try {
    const records = await prepareExternalDataFiles(files, {
      label: format === "onnx" ? "ONNX external data" : "ExecuTorch PTD or selected-build sidecar",
      onProgress: ({ index, count, phase }) => {
        onnxExternalDataStatus.textContent = `${phase === "hashing" ? "Hashing" : "Reading"} ${formatNumber(index + 1)}/${formatNumber(count)}`;
      },
    });
    currentExternalDataFiles = records;
    reportVerification = null;
    clearNodeEdgeEvidenceOverlayState();
    current = null;
    currentArtifactIrContext = null;
    currentDeploymentFrontier = null;
    currentDeploymentDelta = null;
    currentDelegationRepair = null;
    currentModelBytes = null;
    currentFilename = "";
    runtimeAssignmentEvidence = null;
    activeTargetId = "";
    reportTargetRequestedId = targetSelect.value || "";
    targetAnalysisCache.clear();
    currentLowNormStatMap = new Map();
    resetAdvancedResultState();
    performanceVisualController.resetTargetComparisonCache();
    resetDeepBomPanel();
    resetAnalysisViews();
    syncExternalDataControl(format);
    analysisPlanStatus.textContent = "External data ready; run audit";
    runAudit.textContent = formatAuditButtonLabel(format);
    setStatus(`${format === "onnx" ? "ONNX external data" : "ExecuTorch sidecar evidence"} ready`, "ok");
  } catch (error) {
    console.error("[artifactExternalData]", error);
    onnxExternalDataStatus.textContent = `Selection failed: ${shortError(error)}`;
    analysisPlanStatus.textContent = "External data selection failed";
    setStatus("External data selection failed", "error");
  } finally {
    inputs.forEach((input) => { input.disabled = false; });
    runAudit.disabled = !pendingModelFile;
  }
}

function resetAnalysisViews() {
  activeGraphScenario = null;
  actions.hidden = true;
  auditWorkbench.hidden = true;
  summary.innerHTML = "";
  summary.hidden = true;
  insightDashboard.hidden = true;
  perfVisuals.hidden = true;
  inferencePanel.hidden = true;
  graphExplorer.hidden = true;
  if (redesignPanel) redesignPanel.hidden = true;
  diagramSection.hidden = true;
  tables.hidden = true;
  benchmarkWrap.hidden = true;
  benchmarkBody.replaceChildren();
  runtimeNotes.replaceChildren();
  structureTelemetryState = null;
  graphMapSvg.replaceChildren();
  resetResearchModulePanels();
  renderReportPanel();
  explorerDecisionController.render(null);
  explorerRedesignController.render(null);
  renderGraphScenarioState();
  deploymentFrontierController.render(null);
  deploymentDeltaController.render(null);
  delegationRepairController.render(null);
  quantizationLatticeController.render(null);
  accumulatorAtlasController.render(null);
  requantizationFidelityController.render(null);
  kernelWitnessController.render(null);
  channelVitalityController.render(null);
  roundingEquivalenceController.render(null);
  accumulatorReachabilityController.render(null);
  numericalAbiPropagationController.render(null);
  inputCounterexampleController.render(null);
  preprocessingRealizabilityController.render(null);
  preprocessingConsequenceController.render(null);
  contractMigrationController.render(null);
  residualStepResponseController.render(null);
  residualContractDistortionController.render(null);
  renderQuantEvidenceChains(null);
  renderExecutionPlacementView(explorerExecutionPlacementPanel, null, null);
  syncDeploymentDeltaControls();
  graphOpBody.replaceChildren();
  resourceMapPanel?.replaceChildren();
  quantExposureMap?.replaceChildren();
  stageStrip.replaceChildren();
  histogramBody.replaceChildren();
  topMacBody.replaceChildren();
  rooflineBody.replaceChildren();
  updateWorkflowState(pendingModelFile ? "selected" : "idle");
}

function resetExplorerInteractionState() {
  currentTensorFilter = "";
  currentTensorRoleFilter = "";
  currentKernelFilter = "all";
  currentGraphMode = "deploy";
  opTableSortKey = "";
  opTableSortDir = -1;
  opFilterBound = "";
  opFilterXnn = "";
  opFilterQuant = "";
  if (graphSearch) graphSearch.value = "";
  if (graphDepth) graphDepth.value = "2";
  if (tensorSearch) tensorSearch.value = "";
  if (kernelInspectorSearch) kernelInspectorSearch.value = "";
  for (const chip of opFilterBar?.querySelectorAll("[data-filter-group]") || []) {
    chip.classList.toggle("active", chip.dataset.filterValue === "");
  }
  for (const chip of document.querySelectorAll("[data-tfilter]")) {
    chip.classList.toggle("active", chip.dataset.tfilter === "all");
  }
  for (const chip of document.querySelectorAll("[data-kernel-filter]")) {
    chip.classList.toggle("active", chip.dataset.kernelFilter === "all");
  }
  for (const button of document.querySelectorAll("[data-graph-mode]")) {
    button.classList.toggle("active", button.dataset.graphMode === "deploy");
  }
  nodeViewController.resetInteractionState();
  quantEvidenceController.resetInteractionState();
  explorerRedesignController.resetInteractionState();
}

function auditEstimateOptions(file, header = null) {
  return {
    auditTimings: readAuditTimings(),
    comparisonTargetCount: detectModelFormat(file.name, header) === "tflite"
      ? Math.max(1, deploymentFrontierTargetIds(targetProfiles, selectedTargetId()).length) : 1,
  };
}

function renderStagedModel(file, inspection) {
  modelPlan.hidden = false;
  const modelFormat = String(inspection?.formatId || detectModelFormat(file?.name || "")).toLowerCase();
  document.body.dataset.modelFormat = modelFormat;
  renderStagedArtifactContext(document, modelFormat);
  syncExternalDataControl(modelFormat);
  renderModelPlan(stagedModelCopy(file, inspection, selectedTargetLabel()));
  renderAuditClaimBoundary(modelFormat);
  syncFormatWorkflowVisibility(null);
  updateWorkflowState("selected");
  renderTargetSwitcher();
}

function renderModelPlan(copy) {
  selectedModelName.textContent = copy.name;
  selectedModelName.title = copy.name;
  selectedModelName.setAttribute("aria-label", `Selected artifact: ${copy.name}`);
  selectedModelMeta.textContent = copy.meta;
  analysisEstimate.textContent = copy.estimate;
  analysisEstimateNote.textContent = copy.note;
  analysisPlanStatus.textContent = copy.status;
}

function selectedTargetLabel() {
  return targetProfiles.find((profile) => profile.id === selectedTargetId())?.label || selectedTargetId();
}

async function analyzeLoadedModel(filename, targetOverride = "", { keepTab = false, keepModule = false, finalize = true, targetBindingSource = null } = {}) {
  setStatus("Parsing artifact");
  auditProgressController.begin(16, "Starting isolated static analysis", { ceiling: 31, step: 3 });
  await nextPaint();
  const started = performance.now();
  const format = String(current?.format || detectModelFormat(filename, currentModelBytes)).toLowerCase();
  const targetProfileApplicable = modelSupportsCapability(format, "target_profiles");
  const targetId = targetProfileApplicable ? targetOverride || selectedTargetId() : "";
  const cacheKey = targetId || `${format}:artifact`;
  if (targetId && runtimeAssignmentEvidence?.target_profile_id && runtimeAssignmentEvidence.target_profile_id !== targetId) {
    runtimeAssignmentEvidence = null;
  }
  if (pendingRuntimeProfile) runtimeProfileModal.close();
  await nextPaint();
  const metadataOnly = ["gguf", "safetensors", "coreml"].includes(format);
  const cachedAnalysis = metadataOnly ? null : targetAnalysisCache.get(cacheKey);
  if (cachedAnalysis && (!targetProfileApplicable || cachedAnalysis?.target_profile?.id === targetId)) {
    current = cachedAnalysis;
  } else {
    if (cachedAnalysis) targetAnalysisCache.delete(cacheKey);
    const workerOperation = format === "onnx" ? "onnx_analyze" : format === "executorch" ? "executorch_analyze" : "tflite_analyze";
    current = metadataOnly ? current : await staticAuditWorkerClient.run(
      workerOperation,
      {
        bytes: currentModelBytes,
        filename,
        targetId: targetProfileApplicable ? resolveTargetSpec(targetId, customTargetSpecs) : null,
        targetProfile: targetProfileApplicable ? targetProfiles.find((p) => p.id === targetId) || selectedTargetProfile() : null,
        externalDataFiles: currentExternalDataFiles,
        onStatus: (phase) => auditProgressController.describe(phase),
      },
    );
    if (targetProfileApplicable && !metadataOnly && current?.target_profile?.id !== targetId) {
      throw new Error(`analysis target mismatch: requested ${targetId}, received ${current?.target_profile?.id || "unbound"}`);
    }
    if (!metadataOnly) targetAnalysisCache.set(cacheKey, current);
  }
  normalizeUnassessedCostValues(current);
  if (["onnx", "executorch"].includes(format)) syncExternalDataControl(format);
  auditProgressController.begin(32, "Binding artifact SHA-256", { ceiling: 41, step: 4 });
  await ensureModelHash();
  if (targetProfileApplicable && current?.target_profile) {
    const bindingSource = targetBindingSource
      || (targetOverride || readSavedTarget() ? "explicit_id" : "default_assumption");
    current.cpu_cost_target_binding = buildCpuCostTargetBinding(current.target_profile, {
      bindingSource,
    });
  } else if (current) {
    delete current.cpu_cost_target_binding;
  }
  if (["onnx", "tflite"].includes(format)) {
    current.on_device_llm = buildOnDeviceLlmContract(current);
  }
  if (format === "onnx") {
    current.tensorrt_static_preflight = buildTensorRtStaticPreflight(
      current,
      pendingPublicSampleCompanions?.tensorrt_build_profile || null,
      pendingPublicSampleCompanions?.tensorrt_parser_observation || null,
      pendingPublicSampleCompanions?.tensorrt_engine_inspector || null,
    );
  }
  if (format === "tflite") {
    const frontierTargetIds = deploymentFrontierTargetIds(targetProfiles, targetId);
    auditProgressController.begin(42, `Building ${frontierTargetIds.length}-target frontier`, { ceiling: 65, step: 5 });
    if (!currentDeploymentFrontier
      || currentDeploymentFrontier.artifact_sha256 !== current.model_sha256
      || !deploymentFrontierMatchesTargetIds(currentDeploymentFrontier, frontierTargetIds)) {
      try {
        await nextPaint();
        currentDeploymentFrontier = await staticAuditWorkerClient.run("tflite_frontier", {
          bytes: currentModelBytes,
          filename,
          targetIdsJson: JSON.stringify(frontierTargetIds.map((id) => resolveTargetSpec(id, customTargetSpecs))),
          onStatus: (phase) => auditProgressController.describe(phase),
        });
        delete current.deployment_frontier_error;
      } catch (error) {
        console.warn("Deployment frontier unavailable", error);
        currentDeploymentFrontier = null;
        delete current.deployment_frontier;
        current.deployment_frontier_error = shortError(error);
      }
    }
    if (currentDeploymentFrontier) {
      current.deployment_frontier = currentDeploymentFrontier;
      delete current.deployment_frontier_error;
    }
    auditProgressController.begin(66, "Evaluating delegation repair", { ceiling: 77, step: 6 });
    if (currentDelegationRepair?.artifact_sha256 !== current.model_sha256
      || currentDelegationRepair?.target_profile_sha256 !== current.target_profile?.profile_sha256) {
      try {
        await nextPaint();
        currentDelegationRepair = await staticAuditWorkerClient.run("tflite_delegation_repair", {
          bytes: currentModelBytes,
          filename,
          targetId: resolveTargetSpec(targetId, customTargetSpecs),
          onStatus: (phase) => auditProgressController.describe(phase),
        });
        current.delegation_repair = currentDelegationRepair;
        delete current.delegation_repair_error;
      } catch (error) {
        console.warn("Delegation repair analysis unavailable", error);
        currentDelegationRepair = null;
        delete current.delegation_repair;
        current.delegation_repair_error = shortError(error);
      }
    } else {
      current.delegation_repair = currentDelegationRepair;
    }
    auditProgressController.set(78, "Assembling static evidence", "running", { step: 7 });
  } else {
    currentDelegationRepair = null;
    delete current.delegation_repair;
    auditProgressController.set(78, format === "onnx" ? "Assembling ONNX evidence" : "Assembling container evidence", "running", { step: 7 });
  }
  await nextPaint();
  updateDeploymentDeltaForCurrent(format, targetId);
  const elapsedMs = performance.now() - started;
  setStatus("Rendering audit");
  auditProgressController.begin(86, "Rendering audit overview", { ceiling: 87, step: 8 });
  await nextPaint();
  activeTargetId = targetId;
  reportTargetRequestedId = targetId;
  if (targetSelect.querySelector(`option[value="${targetId}"]`)) targetSelect.value = targetId;
  await render(current, { keepTab, keepModule });
  sampleLibraryController?.verifyActive(current);
  submitStructureTelemetry().catch((error) => console.warn("Structure telemetry skipped", error));
  refreshPriorAuditSnapshot()
    .then(() => persistAuditSnapshot())
    .catch((error) => console.warn("Snapshot save skipped", error));
  if (finalize) {
    updateWorkflowState("audited");
    setStatus(formatEvidenceScope(current?.format, { analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence }).completion, "ok");
    auditProgressController.set(100, "Complete", "complete", { step: 12 });
  }
  return {
    coreAnalysisMs: elapsedMs,
    fullAnalysisMs: performance.now() - started,
  };
}

function selectedTargetProfile() {
  return targetProfiles.find((profile) => profile.id === selectedTargetId()) || targetProfiles[0] || null;
}

function renderTargetSwitcher() {
  if (!targetSwitcherBar) return;
  const activeFormat = String(current?.format || pendingModelInspection?.formatId || "").toLowerCase();
  if (activeFormat !== "tflite") { targetSwitcherBar.hidden = true; return; }
  targetSwitcherBar.hidden = false;
  const analyzedId = current?.target_profile?.id || activeTargetId || "";
  const selectedId = analyzedId || targetSelect.value || selectedTargetId();
  const pendingId = targetAnalysisTransition?.requestedTargetId || "";
  const transitionRunning = Boolean(targetAnalysisTransitionPromise);
  const pills = targetProfiles.map((profile) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `target-pill${profile.id === selectedId ? " active" : ""}${profile.id === pendingId && profile.id !== selectedId ? " pending" : ""}`;
    btn.dataset.targetId = profile.id;
    btn.disabled = transitionRunning;
    if (profile.id === selectedId) btn.setAttribute("aria-current", "true");
    const cached = targetAnalysisCache.has(profile.id);
    const ready = document.createElement("span");
    ready.className = `target-pill-ready${cached ? " loaded" : ""}`;
    ready.title = cached ? "Analyzed and cached" : "Not analyzed in this session";
    const label = document.createElement("strong");
    label.textContent = TARGET_PILL_LABELS[profile.id] || profile.label || profile.id;
    btn.title = isCustomTargetId(profile.id)
      ? `${profile.label || profile.id} — custom profile, right-click to edit`
      : profile.label || profile.id;
    btn.append(ready, label);
    if (isCustomTargetId(profile.id)) {
      btn.classList.add("target-pill-custom");
      btn.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        openCustomTargetEditor(profile.id);
      });
    }
    btn.addEventListener("click", async () => {
      if (profile.id === selectedTargetId()) return;
      if (!currentModelBytes || !currentFilename) {
        targetSelect.value = profile.id;
        writeSavedTarget(profile.id);
        if (pendingModelFile) renderStagedModel(pendingModelFile, pendingModelInspection);
        else renderTargetSwitcher();
        return;
      }
      try {
        await requestTargetAnalysis(profile.id, { keepTab: true });
      } catch (err) {
        console.error("[targetSwitch]", err);
        updateWorkflowState("error");
        auditProgressController.set(null, "Target switch failed", "error");
        setStatus(`Switch failed: ${err?.message || err}`, "error");
      }
    });
    return btn;
  });
  const label = document.createElement("span");
  label.className = "target-switcher-label";
  label.textContent = "CPU cost profile:";
  const addCustom = document.createElement("button");
  addCustom.type = "button";
  addCustom.className = "target-pill target-pill-add";
  addCustom.title = "Create a custom CPU cost profile from a built-in one";
  addCustom.disabled = transitionRunning;
  addCustom.append(Object.assign(document.createElement("strong"), { textContent: "+ Custom" }));
  addCustom.addEventListener("click", () => openCustomTargetEditor());
  targetSwitcherBar.replaceChildren(label, ...pills, addCustom);
  syncTargetTransitionUi();
}

function selectAcceleratorProfile(profileId, { navigate = false } = {}) {
  const profiles = acceleratorProfilesForAnalysis(current, runtimeAssignmentEvidence);
  const profile = profiles.find((item) => item.profile_id === profileId);
  if (!profile) return false;
  selectedAcceleratorProfileId = profile.profile_id;
  if (!selectedPlacementProfileIds.includes(profile.profile_id)) {
    selectedPlacementProfileIds = [...selectedPlacementProfileIds, profile.profile_id];
  }
  renderAcceleratorSwitcher();
  renderExecutionPlacementViewBase(document.getElementById("executionPlacementPanel"), currentAnalysisView(), runtimeAssignmentEvidence, executionPlacementOptions());
  renderExecutionPlacementViewBase(explorerExecutionPlacementPanel, currentAnalysisView(), runtimeAssignmentEvidence, executionPlacementOptions());
  if (navigate) {
    setActiveAuditTab("accelerator");
    setActiveWorkspace("audit", { force: true });
    document.getElementById("executionPlacementPanel")?.scrollIntoView({ block: "start", behavior: "smooth" });
  }
  return true;
}

function selectPlacementProfiles(profileIds) {
  selectedPlacementProfileIds = [...new Set((profileIds || []).map(String))];
  renderExecutionPlacementViewBase(document.getElementById("executionPlacementPanel"), currentAnalysisView(), runtimeAssignmentEvidence, executionPlacementOptions());
  renderExecutionPlacementViewBase(explorerExecutionPlacementPanel, currentAnalysisView(), runtimeAssignmentEvidence, executionPlacementOptions());
}

function renderAcceleratorSwitcher() {
  selectedAcceleratorProfileId = renderAcceleratorProfileSwitcher(acceleratorSwitcherBar, {
    analysis: currentAnalysisView(),
    runtimeEvidence: runtimeAssignmentEvidence,
    selectedProfileId: selectedAcceleratorProfileId,
    onSelect: (profileId) => selectAcceleratorProfile(profileId, { navigate: true }),
    onLoadSourceLedgers: () => runDeepBom?.click(),
  });
}

function handleEvidenceSelection(selection) {
  if (!current) return;
  const state = evidenceCursor.select({
    artifact_sha256: current.model_sha256 || null,
    ...selection,
  }, { source: "finding" });
  if (state.op_index != null) {
    setActiveWorkspace("graph", { force: true });
    switchExplorerTab("node");
    selectGraphOp(currentAnalysisView(), state.op_index, { scrollTable: false, fromEvidenceCursor: true });
    nodeViewPanel?.scrollIntoView({ block: "start", behavior: "smooth" });
  } else if (state.tensor_index != null) {
    setActiveWorkspace("graph", { force: true });
    graphWorkspace?.selectTensor(currentAnalysisView(), state.tensor_index, { fromEvidenceCursor: true });
  }
}

async function render(analysis, { keepTab = false, keepModule = false } = {}) {
  rebuildCurrentArtifactIrContext(analysis);
  const artifactView = currentArtifactIrContext?.primary_view || analysis;
  const modelFormat = String(analysis?.format || "tflite").toLowerCase();
  if (nodeEdgeEvidenceOverlay?.artifact_sha256 === analysis?.model_sha256) {
    analysis.external_node_edge_evidence_overlay = nodeEdgeEvidenceOverlay;
  }
  if (analysis?.model_sha256 && evidenceCursor.get().artifact_sha256 !== analysis.model_sha256) {
    evidenceCursor.reset(analysis.model_sha256, { source: "artifact-render" });
  }
  ensureQuantResearchCoverage(analysis);
  document.body.dataset.modelFormat = modelFormat;
  renderAuditClaimBoundary(modelFormat, artifactView);
  syncFormatWorkflowVisibility(artifactView);
  if (modelFormat !== "tflite") {
    opFilterXnn = "";
  }
  updateFormatSpecificAuditLabels({
    modelFormat,
    analysis: artifactView,
    auditTabs,
    activeTab: getActiveAuditTab,
    selectTab: setActiveAuditTab,
  });
  if (!keepModule) setActiveModule("deepbom");
  if (!keepTab) setActiveAuditTab("overview");
  updateExportLockState();
  resetResearchModulePanels();
  renderSummary(artifactView);
  renderReportPanel();
  renderInsightDashboard(artifactView);
  if (!graphScenarioMatchesAnalysis(activeGraphScenario, artifactView)) activeGraphScenario = null;
  explorerDecisionController.render(artifactView, { activeScenario: activeGraphScenario });
  explorerRedesignController.render(artifactView);
  if (!keepTab) switchExplorerTab("node");
  renderGraphScenarioState();
  renderTargetSwitcher();
  renderAcceleratorSwitcher();
  auditProgressController.begin(88, "Rendering deployment evidence", { ceiling: 90, step: 9 });
  await nextPaint();
  deploymentFrontierController.render(artifactView);
  deploymentDeltaController.render(currentDeploymentDelta, artifactView);
  delegationRepairController.render(artifactView);
  quantizationLatticeController.render(artifactView);
  accumulatorAtlasController.render(artifactView);
  requantizationFidelityController.render(artifactView);
  auditProgressController.begin(91, "Rendering numerical labs", { ceiling: 94, step: 10 });
  await nextPaint();
  kernelWitnessController.render(artifactView);
  channelVitalityController.render(artifactView);
  roundingEquivalenceController.render(artifactView);
  accumulatorReachabilityController.render(artifactView);
  numericalAbiPropagationController.render(artifactView);
  inputCounterexampleController.render(artifactView);
  preprocessingRealizabilityController.render(artifactView);
  preprocessingConsequenceController.render(artifactView);
  contractMigrationController.render(artifactView);
  residualStepResponseController.render(artifactView);
  residualContractDistortionController.render(artifactView);
  renderQuantEvidenceChains(artifactView);
  syncDeploymentDeltaControls();
  auditProgressController.begin(95, "Rendering findings", { ceiling: 97, step: 11 });
  await nextPaint();
  performanceVisualController.render(artifactView);
  coreIsolationController.render();
  renderInferencePanel(artifactView);
  resetDeepBomPanel();
  renderFindings(findingsBody, artifactView, {
    onSelectEvidence: handleEvidenceSelection,
    onExplain: (explanation) => evidenceWhyController.open(explanation),
  });
  auditProgressController.begin(98, "Rendering graph explorer", { ceiling: 99, step: 12 });
  await nextPaint();
  renderGraphExplorer(artifactView);
  renderStages(artifactView);
  renderHistogram(artifactView);
  renderTopMacs(artifactView);
  renderRoofline(artifactView);
  setActiveWorkspace("audit", { force: true });
}

async function ensureModelHash() {
  if (!current || !currentModelBytes) return "";
  if (!current.model_sha256) {
    current.model_sha256 = currentModelPayloadLoaded ? await sha256Hex(currentModelBytes) : await sha256FileHex(pendingModelFile);
  }
  if (!current.artifact_set && currentExternalDataFiles.length === 0) {
    current.artifact_set = buildSingleFileArtifactSet({
      filename: current.filename || currentFilename || pendingModelFile?.name || "model",
      format: current.format || detectModelFormat(currentFilename || pendingModelFile?.name || ""),
      sha256: current.model_sha256,
      byteLength: current.file_size_bytes ?? current.file_size ?? currentModelBytes.byteLength,
    });
  }
  if (!current._markdown) {
    current._markdown = current.format === "onnx" && current.markdown
      ? current.markdown
      : buildStaticAuditMarkdown(currentAnalysisView(), current.model_sha256);
  }
  renderReportPanel();
  return current.model_sha256;
}

function updateDeploymentDeltaForCurrent(format, targetId = selectedTargetId()) {
  if (currentDeploymentDelta?.baseline?.sha256 === deploymentDeltaBaseline?.sha256
    && currentDeploymentDelta?.candidate?.sha256 === current?.model_sha256) {
    current.deployment_delta = currentDeploymentDelta;
    return;
  }
  currentDeploymentDelta = null;
  if (current) delete current.deployment_delta;
  if (format !== "tflite" || !deploymentDeltaBaseline || !current?.model_sha256 || deploymentDeltaBaseline.sha256 === current.model_sha256) return;
  const targetIds = deploymentFrontierTargetIds(targetProfiles, targetId);
  if (targetIds.length < 2) {
    current.deployment_delta_error = "At least two pinned target profiles are required.";
    return;
  }
  try {
    currentDeploymentDelta = compute_deployment_delta(
      deploymentDeltaBaseline.bytes,
      deploymentDeltaBaseline.filename,
      currentModelBytes,
      currentFilename || current.filename || "candidate.tflite",
      JSON.stringify(targetIds),
    );
    current.deployment_delta = currentDeploymentDelta;
    delete current.deployment_delta_error;
  } catch (error) {
    console.warn("Deployment delta unavailable", error);
    current.deployment_delta_error = shortError(error);
  }
}

function syncDeploymentDeltaControls() {
  if (pinDeploymentBaseline) {
    pinDeploymentBaseline.disabled = !current || !currentModelBytes || String(current.format).toLowerCase() !== "tflite";
    pinDeploymentBaseline.textContent = deploymentDeltaBaseline ? "Replace baseline" : "Pin baseline";
  }
  if (clearDeploymentBaseline) clearDeploymentBaseline.disabled = !deploymentDeltaBaseline;
}

async function buildMlBom(analysis, bytes) {
  const formatter = await loadRawExportFormatter();
  const analysisView = artifactIrBackedView(analysis);
  const hash = analysis.model_sha256 || await sha256Hex(bytes);
  const target = ["tflite", "onnx"].includes(analysis.format) ? analysis.target_profile || selectedTargetProfile() || {} : {};
  const targetBound = modelSupportsCapability(analysis.format, "target_profiles");
  return formatter.buildMlBomDocument(analysisView, {
    hash,
    fileSizeBytes: analysis.file_size_bytes ?? bytes.length,
    target,
    targetId: targetBound ? selectedTargetId() : "",
    endpointOrigin: location.origin,
    serialNumber: `urn:uuid:${artifactUuidFromSha256(hash)}`,
    timestamp: analysis._reportGeneratedAt || (analysis._reportGeneratedAt = new Date().toISOString()),
    preprocessingConsequenceResult: analysis === current ? preprocessingConsequenceResult : null,
    runtimeAssignmentEvidence: analysis === current ? runtimeAssignmentEvidence : null,
    artifactIr: analysis === current ? currentArtifactIrContext?.artifact_ir || null : null,
  });
}

async function buildCurrentDeploymentContractDocuments() {
  if (!current) throw new Error("Analyzed model evidence is required for artifact evidence export.");
  const formatter = await loadRawExportFormatter();
  const generatedAt = current._reportGeneratedAt || (current._reportGeneratedAt = new Date().toISOString());
  return formatter.buildDeploymentContractDocuments(currentAnalysisView(), {
    hash: current.model_sha256,
    fileSizeBytes: current.file_size_bytes ?? currentModelBytes?.length ?? 0,
    runtimeAssignmentEvidence,
    productionInterfaceContract,
    declaredFormulation: current.declared_formulation,
    releaseManifest: current.release_manifest,
    generatedAt,
    artifactIr: currentArtifactIrContext?.artifact_ir || null,
  });
}

async function buildCurrentPublicCycloneDxDocuments() {
  if (!current) throw new Error("Analyzed model evidence is required for CycloneDX export.");
  return buildPublicCycloneDxDocuments(currentAnalysisView(), {
    hash: current.model_sha256,
    fileSizeBytes: current.file_size_bytes ?? currentModelBytes?.length ?? 0,
    generatedAt: current._reportGeneratedAt || (current._reportGeneratedAt = new Date().toISOString()),
    artifactIr: currentArtifactIrContext?.artifact_ir || null,
  });
}

function currentArtifactFilename(suffix) {
  return artifactFilename(current?.filename || currentFilename || "model", suffix);
}

function canonicalGraphSvgText() {
  if (!current?.model_sha256) return graphSvgText(graphMapSvg);
  const size = Number(current.file_size_bytes ?? current.file_size ?? currentModelBytes?.length ?? 0);
  if (!Number.isSafeInteger(size) || size < 0) return graphSvgText(graphMapSvg);
  const graph = currentArtifactIrContext?.graph_ir;
  if (!graph) return graphSvgText(graphMapSvg);
  const view = currentGraphMode === "deploy" ? "placement" : "structure";
  return exportGraphVisualization(graph, { view, format: "svg" }).text;
}

function currentReviewState() {
  return buildReviewState({
    analysis: currentAnalysisView(),
    cursor: evidenceCursor.get(),
    graphView: nodeViewController.reviewState(),
    workspace: getActiveWorkspace(),
    auditTab: getActiveAuditTab(),
    acceleratorProfileId: selectedAcceleratorProfileId,
    comparison: {
      schema: "deepbom.review_placement_selection.v1",
      selected_profile_ids: selectedPlacementProfileIds,
    },
    runtimeEvidence: runtimeAssignmentEvidence,
    externalOverlay: nodeEdgeEvidenceOverlay,
  });
}

function currentReviewHtml() {
  return buildSelfContainedReviewHtml({
    analysis: currentAnalysisView(),
    graphSvg: canonicalGraphSvgText(),
    reviewState: currentReviewState(),
    runtimeEvidence: runtimeAssignmentEvidence,
  });
}

async function buildEngineeringBundleFiles() {
  const formatter = await loadRawExportFormatter();
  const files = formatter.buildEngineeringBundleArtifactFiles(currentAnalysisView(), {
    reportContext: currentReportContext(),
    rawEvidenceContext: currentRawEvidenceContext(),
    mlBomDocument: await buildMlBom(current, currentModelBytes),
    graphSvgText: canonicalGraphSvgText(),
  });
  files.push(zipTextFile("reports/review.html", currentReviewHtml()));
  files.push(zipTextFile("evidence/review_state.json", `${JSON.stringify(currentReviewState(), null, 2)}\n`));
  if (nodeEdgeEvidenceOverlay) {
    files.push(zipTextFile("evidence/external_node_edge_overlay.json", `${JSON.stringify(nodeEdgeEvidenceOverlay, null, 2)}\n`));
  }
  return files;
}

async function buildRawDataFiles() {
  const formatter = await loadRawExportFormatter();
  return formatter.buildRawDataArtifactFiles(currentAnalysisView(), {
    rawEvidenceContext: currentRawEvidenceContext(),
    mlBomDocument: await buildMlBom(current, currentModelBytes),
    graphSvgText: canonicalGraphSvgText(),
    visualPngFiles: await buildVisualPngFiles(),
  });
}

async function currentMetricCoverageForEvidencePackage() {
  const formatter = await loadRawExportFormatter();
  const index = formatter.buildEvidenceLevelIndex(currentAnalysisView(), currentRawEvidenceContext());
  if (!Array.isArray(index?.metric_coverage?.entries) || !Array.isArray(index?.findings)) {
    throw new Error("Evidence-level index is invalid.");
  }
  return index;
}

async function appendPackageAttestation(files, scope, subject = {}) {
  if (!files?.length || !current) return files;
  const packageDigest = await buildCanonicalPackageDigest(files);
  const targetBound = modelSupportsCapability(current.format, "target_profiles");
  const signature = await postJson("/api/report/sign", {
    scope,
    model_sha256: subject.model_sha256 ?? current.model_sha256 ?? "",
    artifact_id: subject.artifact_id || "",
    target_id: subject.target_id ?? (targetBound ? selectedTargetId() : ""),
    target_label: subject.target_label ?? (targetBound ? selectedTargetLabel() : ""),
    target_profile_sha256: subject.target_profile_sha256 ?? (targetBound ? current.target_profile?.profile_sha256 ?? "" : ""),
    package_hash_sha256: packageDigest.package_hash_sha256,
    package_hash_method: packageDigest.package_hash_method,
    canonicalization: packageDigest.canonicalization,
    package_members: packageDigest.files,
    attestation_member: "attestation.json",
  });
  validatePackageAttestation(signature, packageDigest);
  files.push(zipTextFile("attestation.json", jsonForDownload(signature)));
  return files;
}

const REGULATORY_BUNDLE_RUNNERS = {
  deepbom: async () => {
    const manifest = await ensureDeepBomAllowed();
    return runDeepBomAnalysis(manifest, { activateWorkflow: false });
  },
  perturbation: () => runPerturbationAnalysis(),
  runtime_basin: () => runRuntimeBasinValidation(),
  deploy_curvature: () => runDeployCurvatureBasinAnalysis(),
};

async function downloadEngineeringBundleZip() {
  if (!current || !currentModelBytes) return;
  await withBusyButton(downloadEngineeringBundle, "Preparing", async () => {
    try {
      if (!(await ensureRawExportAllowed("Engineering bundle ZIP"))) return;
      await ensureModelHash();
      engineeringBundleProgress = {
        engineering_static: {
          status: "queued",
          detail: "Waiting to package engineering report and technical artifacts.",
        },
      };
      renderEvidenceBundleScope();
      setStatus("Preparing engineering bundle");
      setEngineeringBundleProgress("engineering_static", "running", "Building the Engineering Report and consolidated technical evidence document.");
      const formatter = await loadRawExportFormatter();
      let files = await buildEngineeringBundleFiles();
      setEngineeringBundleProgress("engineering_static", "done", "Engineering Bundle is ready.");
      files = buildBundleEnvelope(files, {
        buildSummary: (currentFiles) => formatter.buildEngineeringBundleSummary(currentBundleSummaryContext({ files: currentFiles })),
        buildManifest: (currentFiles) => jsonForDownload(formatter.buildEngineeringBundleManifest(currentBundleManifestContext({ files: currentFiles }))),
        includeSummary: false,
      });
      await appendPublicKeySignature(files, "engineering_bundle");
      await appendPackageAttestation(files, "engineering_bundle");
      downloadBlob(currentArtifactFilename("deepbom_engineering_bundle.zip"), createZipBlob(files));
      setStatus("Engineering bundle downloaded", "ok");
    } catch (error) {
      console.error(error);
      setEngineeringBundleProgress("engineering_static", "failed", shortError(error));
      setStatus("Engineering bundle failed", "error");
    }
  }, updateExportLockState);
}

async function downloadEvidencePackageProfileZip() {
  if (!current || !downloadPublicBundle) return;
  await withBusyButton(downloadPublicBundle, "Preparing", async () => {
    try {
      const binding = reportTargetBinding();
      if (!binding.canCopy || !reportBindingMatchesAnalysis(binding, current)) {
        throw new Error("Analyze or load the selected report binding before exporting public evidence.");
      }
      await ensureModelHash();
      const profile = resolveEvidencePackageProfile(evidencePackageProfile?.value);
      const evidenceLevel = resolveEvidenceLevelProfile(evidencePackageLevel?.value);
      let reportBody = null;
      if (evidenceLevel.id === "all_available" && profile.id === "engineering") {
        const formatter = await loadEngineeringFormatter();
        reportBody = formatter.buildEngineeringReport(currentAnalysisView(), currentReportContext());
      } else if (evidenceLevel.id === "all_available" && profile.id === "regulatory") {
        const formatter = await loadRegulatoryFormatter();
        reportBody = formatter.buildRegulatoryReport(currentAnalysisView(), currentRegulatoryReportContext());
      }
      const evidenceIndex = await currentMetricCoverageForEvidencePackage();
      const files = buildEvidencePackageProfileFiles({
        profileId: profile.id,
        evidenceLevelId: evidenceLevel.id,
        analysis: currentAnalysisView(),
        context: currentReportContext(),
        scope: formatEvidenceScope(current.format, { analysis: currentAnalysisView(), runtimeEvidence: runtimeAssignmentEvidence }),
        runtimeEvidence: runtimeAssignmentEvidence,
        metricCoverage: evidenceIndex.metric_coverage,
        findings: evidenceIndex.findings,
        origin: location.origin,
        reportBody,
      });
      await appendPublicKeySignature(files, `evidence_package_${profile.id}`);
      downloadBlob(currentArtifactFilename(`deepbom_${profile.id}_${evidenceLevel.id}_evidence_package.zip`), createZipBlob(files));
      setStatus(`${profile.label} / ${evidenceLevel.label} Evidence Package downloaded`, "ok");
    } catch (error) {
      console.error(error);
      setStatus(`Evidence Package failed: ${shortError(error)}`, "error");
    }
  }, updateExportLockState);
}

async function downloadRawDataZip() {
  if (!current || !currentModelBytes) return;
  await withBusyButton(downloadRawData, "Preparing", async () => {
    try {
      if (!(await ensureRawExportAllowed("Download Raw Data"))) return;
      await ensureModelHash();
      let files = await buildRawDataFiles();
      await appendPublicKeySignature(files, "raw_data");
      await appendPackageAttestation(files, "raw_data");
      downloadBlob(currentArtifactFilename("deepbom_raw_data.zip"), createZipBlob(files));
      setStatus("Raw data downloaded", "ok");
    } catch (error) {
      console.error(error);
      setStatus("Raw data download failed", "error");
    }
  }, updateExportLockState);
}

async function downloadEvidenceBundleZip() {
  if (!current || !currentModelBytes) return;
  await withBusyButton(downloadEvidenceBundle, "Preparing", async () => {
    const moduleLog = [];
    const files = [];
    try {
      if (!(await ensureRegulatoryReportAllowed("Evidence bundle ZIP"))) return;
      await ensureModelHash();
      const capabilities = currentCapabilities();
      evidenceBundleProgress = initialEvidenceBundleProgress(capabilities);
      renderEvidenceBundleScope();
      setStatus("Preparing regulatory bundle");
      setEvidenceBundleProgress("engineering_bundle", "running", "Building the full Engineering Bundle first.");
      files.push(...await buildEngineeringBundleFiles());
      setEvidenceBundleProgress("engineering_bundle", "done", "Full Engineering Bundle is included.");
      setEvidenceBundleProgress("regulatory_report", "running", "Adding Regulatory Report and evidence-class framing.");
      const regulatoryFormatter = await loadRegulatoryFormatter();
      const rawFormatter = await loadRawExportFormatter();
      files.push(zipTextFile("reports/regulatory_report.md", regulatoryFormatter.buildRegulatoryReport(currentAnalysisView(), currentRegulatoryReportContext())));
      setEvidenceBundleProgress("regulatory_report", "done", "Regulatory Report is included.");

      for (const bundleModule of REGULATORY_BUNDLE_MODULE_SPECS) {
        await appendRegulatoryBundleModule({
          files,
          moduleLog,
          capabilities,
          bundleModule,
          runners: REGULATORY_BUNDLE_RUNNERS,
          setProgress: setEvidenceBundleProgress,
          setStatus,
          beforeRun: nextPaint,
          formatError: shortError,
          onError: (error, item) => console.warn(`${item.label} bundle step failed`, error),
        });
      }

      const packagedFiles = buildBundleEnvelope(files, {
        buildSummary: (currentFiles) => rawFormatter.buildEvidenceBundleSummary(currentBundleSummaryContext({
          files: currentFiles,
          moduleLog,
          capabilities,
        })),
        buildManifest: (currentFiles) => jsonForDownload(rawFormatter.buildEvidenceBundleManifest(currentBundleManifestContext({
          files: currentFiles,
          moduleLog,
          capabilities,
        }))),
      });
      await appendPublicKeySignature(packagedFiles, "regulatory_bundle");
      await appendPackageAttestation(packagedFiles, "regulatory_bundle");

      const zip = createZipBlob(packagedFiles);
      renderReportPanel();
      downloadBlob(currentArtifactFilename("deepbom_regulatory_bundle.zip"), zip);
      setStatus("Regulatory bundle downloaded", "ok");
    } catch (error) {
      console.error(error);
      const runningStep = evidenceBundleProgress
        ? Object.entries(evidenceBundleProgress).find(([, step]) => step.status === "running")?.[0]
        : "";
      if (runningStep) setEvidenceBundleProgress(runningStep, "failed", shortError(error));
      setStatus("Bundle failed", "error");
    }
  }, updateExportLockState);
}

function currentReportContextSet({ files = [], moduleLog = [], capabilities = currentCapabilities() } = {}) {
  const identity = modelIdentity();
  return buildSessionReportContextSet({
    analysis: currentAnalysisView(),
    identity,
    user: bundleUserIdentityForUser(currentAuthUser),
    capabilities,
    files,
    moduleLog,
    runtimeBenchmarkResults,
    deepBomResult,
    perturbationResult,
    runtimeBasinResult,
    deployCurvatureResult,
    preprocessingConsequenceResult,
    calibrationValidationResult,
    runtimeAssignmentEvidence,
    browserRuntime: {
      browserBucket: browserBucket(),
      sharedArrayBufferAvailable: typeof SharedArrayBuffer !== "undefined",
      webgpuAvailable: "gpu" in navigator,
      webnnAvailable: "ml" in navigator,
    },
    fileSizeBytes: current?.file_size || currentModelBytes?.length || 0,
    generatedAt: new Date().toISOString(),
    artifactIrContext: currentArtifactIrContext,
  });
}

function currentSessionPrivacy() {
  return buildSessionPrivacy({
    consentLog: readConsentLog(),
    agreementRecord: readAgreementRecord(),
    researchConsentRecord: readResearchConsentRecord(),
    researchConsent: readResearchConsent(),
    structureTelemetryState,
    policyVersion: AGREEMENT_POLICY_VERSION,
    historySaved: Boolean(currentAuthUser && readReportHistorySettings().enabled),
  });
}

let priorAuditSnapshot = null; // most recent stored snapshot of a *different* build of this model

async function refreshPriorAuditSnapshot() {
  priorAuditSnapshot = null;
  if (!currentAuthUser || !current?.model_sha256) return;
  try {
    const snapshots = await listSnapshots();
    const currentFormat = String(current.format || "").toLowerCase();
    const currentTarget = current.target_profile?.id || "";
    const candidates = snapshots.filter((snapshot) => {
      if (!snapshot?.sha256 || snapshot.sha256 === current.model_sha256) return false;
      if (String(snapshot.format || "").toLowerCase() !== currentFormat) return false;
      if ((snapshot.target || "") !== currentTarget) return false;
      if (snapshot.modelLineageId && current.model_lineage_id) {
        return snapshot.modelLineageId === current.model_lineage_id;
      }
      if (current.previous_artifact_sha256 && snapshot.sha256 === current.previous_artifact_sha256) return true;
      if (snapshot.derivationManifestId && current.derivation_manifest_id) return snapshot.derivationManifestId === current.derivation_manifest_id;
      return snapshot.explicitlySelectedForComparison === true;
    });
    priorAuditSnapshot = candidates[0] || null;
  } catch { priorAuditSnapshot = null; }
}

function currentReportContext() {
  if (current && !current._reportGeneratedAt) current._reportGeneratedAt = new Date().toISOString();
  return {
    ...currentReportContextSet().reportContext,
    generatedAt: current?._reportGeneratedAt,
    sessionPrivacy: currentSessionPrivacy(),
    priorSnapshot: priorAuditSnapshot,
    verification: reportVerification,
    productionInterfaceContract,
  };
}

function currentRegulatoryReportContext() {
  return currentReportContextSet().regulatoryReportContext;
}

function currentBundleSummaryContext({ files = [], moduleLog = [], capabilities = currentCapabilities() } = {}) {
  return currentReportContextSet({ files, moduleLog, capabilities }).bundleSummaryContext;
}

function currentBundleManifestContext({ files = [], moduleLog = [], capabilities = currentCapabilities() } = {}) {
  return currentReportContextSet({ files, moduleLog, capabilities }).bundleManifestContext;
}

function currentRawEvidenceContext() {
  return currentReportContextSet().rawEvidenceContext;
}


function readConsentLog() {
  try { return JSON.parse(localStorage.getItem("deepbom.consent_log") || "[]"); }
  catch { return []; }
}
function appendConsentLog(entry) {
  try {
    const log = readConsentLog();
    const eventId = globalThis.crypto?.randomUUID?.() || `evt_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
    log.push({ event_id: eventId, ...entry, at: new Date().toISOString() });
    localStorage.setItem("deepbom.consent_log", JSON.stringify(log.slice(-50)));
  } catch { /* storage disabled — provenance is best-effort */ }
}
function logMetadataShared(structure) {
  appendConsentLog({
    kind: "structure-shared",
    fingerprint: structure?.fingerprint || "",
    ops: structure?.op_count ?? structure?.operator_count ?? null,
    macs: structure?.total_macs ?? null,
    target: selectedTargetId(),
  });
  renderConsentPanel();
}

function renderConsentPanel() {
  if (!consentStatusBadge) return;
  const consented = readResearchConsent();
  consentStatusBadge.textContent = consented ? "Sharing enabled" : "Withdrawn";
  consentStatusBadge.className = consented ? "consent-badge on" : "consent-badge off";
  if (withdrawConsentBtn) withdrawConsentBtn.hidden = !consented;
  if (restoreConsentBtn) restoreConsentBtn.hidden = consented;

  if (consentMetadataLog) {
    const log = readConsentLog().filter((e) => e.kind === "structure-shared").slice(-10).reverse();
    if (!log.length) {
      consentMetadataLog.innerHTML = `<p class="muted-text">No structure metadata has been shared from this browser yet.</p>`;
    } else {
      consentMetadataLog.replaceChildren(...log.map((e) => {
        const row = mk("div", "consent-log-row");
        const when = mk("span", "consent-log-when"); when.textContent = new Date(e.at).toLocaleString();
        const what = mk("span", "consent-log-what");
        what.textContent = `${e.ops != null ? `${e.ops} ops` : "structure"}${e.macs != null ? ` · ${formatNumber(e.macs)} MACs` : ""} · target ${e.target || "?"} · fp ${(e.fingerprint || "—").slice(0, 12)}`;
        row.append(when, what);
        return row;
      }));
    }
  }
}

function setConsent(enabled) {
  writeResearchConsent(enabled);
  syncResearchConsent().catch((error) => console.warn("Consent sync skipped", error));
  appendConsentLog({ kind: enabled ? "consent-restored" : "consent-withdrawn" });
  renderConsentPanel();
  setStatus(enabled ? "Research metadata sharing re-enabled" : "Research metadata sharing withdrawn", "ok");
}

withdrawConsentBtn?.addEventListener("click", () => setConsent(false));
restoreConsentBtn?.addEventListener("click", () => setConsent(true));


const selectedSnapshotIds = new Set();
let lastComparisonMarkdown = "";

async function persistAuditSnapshot() {
  // Audit history persistence is a signed-in benefit; anonymous sessions keep
  // the current session's views/exports but nothing is written to IndexedDB.
  if (!currentAuthUser) return;
  const historySettings = readReportHistorySettings();
  if (!historySettings.enabled) return;
  if (!current || !current.model_sha256) return;
  try {
    const snapshot = buildAuditSnapshot(currentArtifactIrContext?.primary_view || current, {
      analyzerVersion: ANALYZER_VERSION,
      rulepackVersion: RULEPACK_VERSION,
      runtimeBenchmarkResults: runtimeBenchmarkResults || [],
      sha256: current.model_sha256,
      reportMarkdown: current._markdown || "",
    });
    await saveAuditSnapshot(snapshot);
    await pruneAuditSnapshots(historySettings);
    await renderLocalReports();
  } catch (error) {
    console.warn("Snapshot save skipped", error);
  }
}

let pendingRerun = null; // { sha256, target, filename } — set by "Rerun" while waiting for file re-selection

async function handlePendingRerun(file) {
  const expected = pendingRerun;
  pendingRerun = null;
  const bytes = new Uint8Array(await file.arrayBuffer());
  const sha = await sha256Hex(bytes);
  if (sha === expected.sha256) {
    setStatus("Selected artifact matches the original audit — re-running", "ok");
    if (expected.target && targetSelect.querySelector(`option[value="${expected.target}"]`)) {
      targetSelect.value = expected.target;
      writeSavedTarget(expected.target);
    }
    if (!runAudit.disabled) runAudit.click();
  } else {
    setStatus(`Selected file does not match the original audit (SHA-256 ${sha.slice(0, 12)}… vs ${expected.sha256.slice(0, 12)}…) — not re-run`, "error");
  }
}

async function renderLocalReports() {
  if (!localReportsList) return;
  if (!currentAuthUser) {
    localReportsList.innerHTML = "";
    const teaser = mk("div", "local-reports-teaser");
    const copy = mk("p");
    copy.textContent = "Sign in to keep your audit history in this browser: every completed audit is saved locally (never uploaded), so you can reopen past reports, re-run with the current rulepack, and compare artifact versions by SHA-256.";
    const cta = mk("button", "secondary-action");
    cta.type = "button";
    cta.textContent = "Sign in to enable audit history";
    cta.addEventListener("click", () => openAuthModal("signup", "Audit history, reopen, and version comparison are enabled for signed-in accounts. Analysis itself stays open without login."));
    teaser.append(copy, cta);
    localReportsList.append(teaser);
    if (compareReportsBtn) compareReportsBtn.hidden = true;
    if (clearReportsBtn) clearReportsBtn.hidden = true;
    if (saveReportHistory) saveReportHistory.closest("label").hidden = true;
    if (reportHistoryRetention) reportHistoryRetention.closest("label").hidden = true;
    return;
  }
  const historySettings = readReportHistorySettings();
  if (saveReportHistory) {
    saveReportHistory.closest("label").hidden = false;
    saveReportHistory.checked = historySettings.enabled;
  }
  if (reportHistoryRetention) {
    reportHistoryRetention.closest("label").hidden = false;
    reportHistoryRetention.value = String(historySettings.retentionDays);
  }
  if (compareReportsBtn) compareReportsBtn.hidden = false;
  if (clearReportsBtn) clearReportsBtn.hidden = false;
  let snapshots = [];
  try { snapshots = await listSnapshots(); } catch { snapshots = []; }
  const snapshotIds = new Set(snapshots.map((snapshot) => snapshot.id));
  for (const id of selectedSnapshotIds) if (!snapshotIds.has(id)) selectedSnapshotIds.delete(id);
  if (!snapshots.length) {
    localReportsList.innerHTML = `<p class="muted-text">${historySettings.enabled ? "No saved reports yet. Completed audits will be stored locally." : "Local history is paused. Enable Save completed audits to retain future snapshots."}</p>`;
    updateCompareButton();
    return;
  }
  localReportsList.replaceChildren(...snapshots.map((snap) => {
    const row = mk("div", "local-report-row");
    if (selectedSnapshotIds.has(snap.id)) row.classList.add("selected");

    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selectedSnapshotIds.has(snap.id);
    check.addEventListener("change", () => {
      if (check.checked) selectedSnapshotIds.add(snap.id); else selectedSnapshotIds.delete(snap.id);
      row.classList.toggle("selected", check.checked);
      updateCompareButton();
    });

    const info = mk("div", "local-report-info");
    const title = mk("strong");
    title.textContent = snap.filename || "(unnamed)";
    if (snap.rulepackVersion && snap.rulepackVersion !== RULEPACK_VERSION) {
      const drift = mk("em", "maturity-badge maturity-beta");
      drift.textContent = "rulepack updated";
      drift.title = `Audited with ${snap.rulepackVersion}; current is ${RULEPACK_VERSION}. Re-run to refresh estimates.`;
      title.append(" ", drift);
    }
    const meta = mk("span", "local-report-meta");
    meta.textContent = `${snap.target} · ${snap.operatorCount} ops · ${new Date(snap.updatedAt).toLocaleString()} · run ×${snap.runCount} · ${snap.analyzerVersion}/${snap.rulepackVersion}${snap.userNote ? ` · 📝 ${snap.userNote}` : ""}`;
    const sha = mk("span", "local-report-sha");
    sha.textContent = `SHA-256 ${(snap.sha256 || "").slice(0, 16)}…`;
    info.append(title, meta, sha);

    const actions = mk("div", "local-report-actions");
    // Reopen: view the stored report — no original model required
    const reopen = mk("button", "secondary-action"); reopen.type = "button"; reopen.textContent = "Reopen";
    reopen.title = "Reopen the saved report (original model not required)";
    reopen.addEventListener("click", () => showSnapshotSummary(snap));
    const dl = mk("button", "secondary-action"); dl.type = "button"; dl.textContent = "Download";
    dl.title = "Download the stored report markdown + snapshot JSON";
    dl.addEventListener("click", () => {
      if (snap.reportMarkdown) downloadText(`deepbom-report-${(snap.sha256 || "x").slice(0, 8)}.md`, snap.reportMarkdown);
      downloadText(`deepbom-snapshot-${(snap.sha256 || "x").slice(0, 8)}.json`, JSON.stringify(snap, null, 2));
    });
    // Rerun: needs the original artifact — re-select and verify by SHA-256
    const rerun = mk("button", "secondary-action"); rerun.type = "button"; rerun.textContent = "Rerun";
    rerun.title = "Rerun with the current rulepack — original model selection required; the file is verified against the stored SHA-256";
    rerun.addEventListener("click", async () => {
      if (current && currentModelBytes && current.model_sha256 === snap.sha256) {
        setStatus("Loaded artifact matches the original audit — re-running", "ok");
        updateWorkflowState("running"); await analyzeLoadedModel(currentFilename);
        return;
      }
      pendingRerun = { sha256: snap.sha256, target: snap.target, filename: snap.filename };
      setStatus(`Select the original model file (${snap.filename}) to re-run — it will be verified against the stored SHA-256`, "ok");
      fileInput.click();
    });
    const note = mk("button", "secondary-action"); note.type = "button"; note.textContent = "Note";
    note.title = "Attach a note to this saved report";
    note.addEventListener("click", async () => {
      const text = window.prompt("Note for this report (max 500 chars):", snap.userNote || "");
      if (text !== null) { await updateSnapshotNote(snap.id, text); await renderLocalReports(); }
    });
    const del = mk("button", "secondary-action danger"); del.type = "button"; del.textContent = "Delete";
    del.addEventListener("click", async () => {
      await deleteSnapshot(snap.id);
      selectedSnapshotIds.delete(snap.id);
      await renderLocalReports();
    });
    actions.append(reopen, dl, rerun, note, del);

    row.append(check, info, actions);
    return row;
  }));
  updateCompareButton();
}

function updateCompareButton() {
  if (!compareReportsBtn) return;
  compareReportsBtn.textContent = `Compare selected (${selectedSnapshotIds.size})`;
  compareReportsBtn.disabled = selectedSnapshotIds.size !== 2;
}

function showSnapshotSummary(snap) {
  if (!comparisonResult || !comparisonPre) return;
  lastComparisonMarkdown = snap.reportMarkdown
    ? snap.reportMarkdown
    : `# Stored Audit Snapshot\n\n\`\`\`json\n${JSON.stringify(snap, null, 2)}\n\`\`\`\n`;
  comparisonPre.textContent = lastComparisonMarkdown;
  comparisonVisual?.replaceChildren();
  comparisonResult.hidden = false;
  comparisonResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runComparison() {
  if (selectedSnapshotIds.size !== 2) return;
  const snapshots = await listSnapshots();
  const [a, b] = [...selectedSnapshotIds].map((id) => snapshots.find((s) => s.id === id)).filter(Boolean);
  if (!a || !b) return;
  lastComparisonMarkdown = buildComparisonReport(a, b);
  comparisonPre.textContent = lastComparisonMarkdown;
  renderArtifactDiffWorkspace(comparisonVisual, a, b, {
    onSelect: (match) => {
      evidenceCursor.select({
        artifact_sha256: b.sha256 || null,
        op_index: match.right.index,
        report_anchor: `artifact-diff:${match.left.index}:${match.right.index}`,
      }, { source: "artifact-diff" });
    },
  });
  comparisonResult.hidden = false;
  comparisonResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

compareReportsBtn?.addEventListener("click", () => { runComparison().catch((e) => console.warn(e)); });
saveReportHistory?.addEventListener("change", () => {
  const settings = writeReportHistorySettings({ ...readReportHistorySettings(), enabled: saveReportHistory.checked });
  setStatus(settings.enabled ? "Local audit history enabled" : "Local audit history paused; existing snapshots were retained", "ok");
  renderLocalReports().catch((error) => console.warn(error));
});
reportHistoryRetention?.addEventListener("change", async () => {
  const settings = writeReportHistorySettings({ ...readReportHistorySettings(), retentionDays: Number(reportHistoryRetention.value) });
  const removed = await pruneAuditSnapshots(settings);
  setStatus(`Local history retention updated${removed ? `; ${removed} expired snapshot(s) removed` : ""}`, "ok");
  await renderLocalReports();
});
clearReportsBtn?.addEventListener("click", async () => {
  const snapshots = await listSnapshots();
  for (const snap of snapshots) await deleteSnapshot(snap.id);
  selectedSnapshotIds.clear();
  if (comparisonResult) comparisonResult.hidden = true;
  await renderLocalReports();
});
downloadComparison?.addEventListener("click", () => {
  if (lastComparisonMarkdown) downloadText("deepbom-comparison.md", lastComparisonMarkdown);
});
closeComparison?.addEventListener("click", () => { if (comparisonResult) comparisonResult.hidden = true; });

async function submitStructureTelemetry() {
  if (!readResearchConsent() || !current) return;
  const structure = await buildStructureTelemetryPayload(current, currentTelemetryContext());
  structureTelemetryState = structure;
  await postJson("/api/benchmark/structure", {
    consent: true,
    structure,
  });
  logMetadataShared(structure); // record what left this browser, for the consent panel
}

async function submitBenchmarkTelemetry(result) {
  if (!readResearchConsent() || !current) return;
  const structure = structureTelemetryState || await buildStructureTelemetryPayload(current, currentTelemetryContext());
  structureTelemetryState = structure;
  await postJson("/api/benchmark/structure", buildBenchmarkTelemetryPayload(structure, result, {
    targetId: selectedTargetId(),
    browserBucket: browserBucket(),
    preparedInput: false,
    runtimeStatus: runtimeStatus.textContent,
  }));
}

function currentTelemetryContext() {
  return {
    targetId: selectedTargetId(),
    targetProfileLabel: selectedTargetLabel(),
    browserBucket: browserBucket(),
  };
}

function renderLossLandscapePanel(result) {
  if (!lossLandscapePanel) return;
  if (!result) { lossLandscapePanel.hidden = true; return; }
  lossLandscapePanel.hidden = false;
  lossLandscapePanel.replaceChildren();

  const { int8, f64, axes, seeds, grid: G, radius, hessian, int8Radial, f64Radial, requantRatio: rqr } = result;
  const wrap = mk("div", "haar-ext-panels");
  const sec  = mk("div", "haar-ext-section");

  const title = mk("div", "haar-ext-title");
  const geometryCopy = outputDriftProjectionCopy({ seeds, gridSize: G, radius });
  title.textContent = geometryCopy.title;
  sec.append(title);

  const chips = mk("div", "haar-ext-chips");
  const addChip = (l, v, tone) => {
    const c = mk("span", `haar-chip haar-chip-${tone || "info"}`);
    c.textContent = `${l}: ${v}`; chips.append(c);
  };
  if (int8?.centerLoss != null) addChip("INT8 center drift", int8.centerLoss.toFixed(4), "info");
  if (f64?.centerLoss  != null) addChip("f64 center drift",  f64.centerLoss.toFixed(4),  "info");
  if (rqr != null) addChip("INT8/f64 OLS gain", rqr.toFixed(2) + "×", Math.abs(rqr - 1) > 0.5 ? "warn" : "ok");

  const h8  = hessian?.int8, hf  = hessian?.f64;
  if (h8)  addChip("λ_max INT8",  `${h8.lambdaMax_mean.toFixed(4)} ±${h8.lambdaMax_sem.toFixed(4)}`,  h8.lambdaMax_mean > 0.1 ? "warn" : "ok");
  if (hf)  addChip("λ_max f64",   `${hf.lambdaMax_mean.toFixed(4)} ±${hf.lambdaMax_sem.toFixed(4)}`,  hf.lambdaMax_mean > 0.1 ? "warn" : "ok");
  if (h8)  addChip("tr(H) INT8",  `${h8.trace_mean.toFixed(4)} ±${h8.trace_sem.toFixed(4)}`,          "info");
  if (hf)  addChip("tr(H) f64",   `${hf.trace_mean.toFixed(4)} ±${hf.trace_sem.toFixed(4)}`,          "info");
  sec.append(chips);

  const heatRow = mk("div", ""); heatRow.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;margin:8px 0";
  const vmaxInt8 = Math.max(1e-8, int8?.maxDmean ?? 0);
  const vmaxF64  = Math.max(1e-8, f64?.maxDmean  ?? 0);
  const vmaxShared = Math.max(vmaxInt8, vmaxF64);
  const vmaxSem  = Math.max(1e-8, (int8?.meanSem ?? 0) * 6);

  const heatSpecs = [
    ["INT8 mean Δdrift", int8?.dmean, vmaxShared, "Quantized model landscape"],
    ["f64 mean Δdrift",  f64?.dmean,  vmaxShared, "Float forward pass landscape"],
    ["INT8 SEM (cross-seed)", int8?.sem, vmaxSem, "Statistical uncertainty"],
  ];
  for (const [label, grid, vmax, note] of heatSpecs) {
    if (!grid) continue;
    const box = mk("div", ""); box.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:4px";
    const lbl = mk("div", "haar-ext-bar-group-label"); lbl.textContent = label;
    const cv = document.createElement("canvas");
    cv.width = 160; cv.height = 160;
    cv.style.cssText = "border:1px solid var(--line-strong);border-radius:4px;image-rendering:pixelated";
    drawLandscapeCanvas(cv, grid, G, vmax);
    const noteLbl = mk("div", "haar-ext-note"); noteLbl.textContent = note;
    noteLbl.style.cssText = "font-size:11px;color:var(--ink-soft);margin:0";
    box.append(lbl, cv, noteLbl);
    heatRow.append(box);
  }
  sec.append(heatRow);

  const radialSvg = buildDualRadialSvg(int8Radial, f64Radial, axes);
  if (radialSvg) {
    const rTitle = mk("div", "haar-ext-bar-group-label");
    rTitle.textContent = "Radial Δdrift — INT8 (brown) vs f64 (blue), bands = cross-seed SEM";
    rTitle.style.marginTop = "10px";
    sec.append(rTitle, radialSvg);
  }

  const note = mk("p", "haar-ext-note");
  note.textContent = geometryCopy.method;
  sec.append(note);
  wrap.append(sec); lossLandscapePanel.append(wrap);
}


async function runLandscapeTomography({ numProjections = 12, gridSize = 9, radius = 0.4 } = {}) {
  if (!currentModelBytes) return null;
  const raw = landscape_tomography(currentModelBytes, numProjections, gridSize, radius);
  return summarizeOutputDriftProjectionEnsemble(raw, { numProjections, gridSize, radius });
}

function renderLossTomoPanel(result) {
  if (!lossTomoPanel) return;
  if (!result) { lossTomoPanel.hidden = true; return; }
  lossTomoPanel.hidden = false;
  lossTomoPanel.replaceChildren();

  const {
    dmGrids,
    meanGrid,
    varGrid,
    axes,
    G,
    lambdas = [],
    lambdaMean,
    lambdaStd,
    directionalLambdaMaxCv,
    hessianAssessedCount = 0,
    numProjections,
  } = result;

  const wrap = mk("div", "haar-ext-section");
  const title = mk("h3", "haar-ext-title");
  const ensembleCopy = outputDriftEnsembleCopy({ numProjections, assessedCount: hessianAssessedCount });
  title.textContent = ensembleCopy.title;
  wrap.append(title);

  // Chips row
  const chips = mk("div"); chips.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px";
  const chip = (label, val) => {
    const c = mk("span"); c.style.cssText = "background:var(--surface-tint);border:1px solid var(--line-strong);border-radius:4px;padding:3px 8px;font-size:11px;color:var(--ink-soft)";
    const name = mk("span");
    name.style.color = "var(--muted)";
    name.textContent = label;
    c.append(name, document.createTextNode(` ${val}`));
    return c;
  };
  const scientificOrNA = (value) => typeof value === "number" && Number.isFinite(value) ? value.toExponential(2) : "N/A";
  const fixedOrNA = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "N/A";
  chips.append(
    chip("projections", numProjections),
    chip("grid", `${G}×${G}`),
    chip("projected curvature assessed", `${hessianAssessedCount}/${numProjections}`),
    chip("λ_max mean", scientificOrNA(lambdaMean)),
    chip("λ_max std", scientificOrNA(lambdaStd)),
    chip("directional λ_max CV", fixedOrNA(directionalLambdaMaxCv)),
  );
  wrap.append(chips);

  // Two heatmaps: mean Δdrift + directional variance
  const heatRow = mk("div"); heatRow.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px";

  const drawTomoHeatmap = (grid, label, colorFn) => {
    const box = mk("div");
    const lbl = mk("div"); lbl.style.cssText = "font-size:11px;color:var(--muted);margin-bottom:4px"; lbl.textContent = label;
    const cv = document.createElement("canvas");
    cv.width = 160; cv.height = 160; cv.style.cssText = "display:block;image-rendering:pixelated;border-radius:4px";
    const ctx = cv.getContext("2d");
    const flat = grid.flat().filter(v => !isNaN(v));
    const vMax = Math.max(...flat.map(Math.abs), 1e-10);
    const CW = 160;
    // Float-based cell sizing: x0/x1 via Math.floor to avoid 1-px gaps at any canvas size/G combo
    const cellX = i => Math.floor((i / G) * CW);
    for (let bi = 0; bi < G; bi++) {
      const y0 = cellX(G - 1 - bi), y1 = cellX(G - bi); // y-flip: negative β at bottom
      for (let ai = 0; ai < G; ai++) {
        const x0 = cellX(ai), x1 = cellX(ai + 1);
        const v = grid[bi]?.[ai] ?? 0;
        ctx.fillStyle = colorFn(v, vMax);
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
    // center crosshair
    const ci = Math.floor(G / 2);
    ctx.strokeStyle = "rgba(255,210,80,0.8)"; ctx.lineWidth = 1.5;
    const cx = (cellX(ci) + cellX(ci + 1)) / 2, cy = (cellX(G - 1 - ci) + cellX(G - ci)) / 2;
    ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();
    box.append(lbl, cv);
    return box;
  };

  const divColor = (v, vMax) => {
    const t = Math.max(-1, Math.min(1, v / vMax));
    if (t >= 0) {
      const r = Math.round(255 * t), g = Math.round(40 * (1 - t)), b = Math.round(40 * (1 - t));
      return `rgb(${r},${g},${b})`;
    }
    const a = -t;
    return `rgb(${Math.round(40*(1-a))},${Math.round(40*(1-a))},${Math.round(255*a)})`;
  };
  const magColor2 = (v, vMax) => {
    const t = Math.max(0, Math.min(1, Math.abs(v) / vMax));
    const r = Math.round(10 + 240 * t), g = Math.round(15 + 60 * t), b = Math.round(20 + 30 * t);
    return `rgb(${r},${g},${b})`;
  };

  heatRow.append(
    drawTomoHeatmap(meanGrid, "Mean Δdrift (across projections)", divColor),
    drawTomoHeatmap(varGrid, "Directional variance (std across projections)", magColor2),
  );
  wrap.append(heatRow);

  // λ_max dot plot (sorted) — filter nulls from degenerate projections
  const sorted = lambdas.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length) {
  const svgNs = "http://www.w3.org/2000/svg";
  const DW = 340, DH = 80, DPL = 40, DPR = 10, DPT = 10, DPB = 24;
  const DIW = DW - DPL - DPR, DIH = DH - DPT - DPB;
  const lMin = sorted[0], lMax = sorted[sorted.length - 1];
  const lRange = Math.max(lMax - lMin, 1e-12);
  const dotSvg = document.createElementNS(svgNs, "svg");
  dotSvg.setAttribute("viewBox", `0 0 ${DW} ${DH}`);
  dotSvg.setAttribute("width", DW); dotSvg.setAttribute("height", DH);
  dotSvg.style.cssText = "display:block;overflow:visible";
  // x-axis line
  const axLine = document.createElementNS(svgNs, "line");
  axLine.setAttribute("x1", DPL); axLine.setAttribute("x2", DPL + DIW);
  axLine.setAttribute("y1", DPT + DIH); axLine.setAttribute("y2", DPT + DIH);
  axLine.setAttribute("stroke", "var(--line-strong)"); axLine.setAttribute("stroke-width", "1");
  dotSvg.append(axLine);
  sorted.forEach((v, i) => {
    const x = DPL + (sorted.length > 1 ? (i / (sorted.length - 1)) * DIW : DIW / 2);
    const t = (v - lMin) / lRange;
    const r = Math.round(80 + 175 * t), g = Math.round(100 + 60 * t), b = Math.round(200 - 120 * t);
    const dot = document.createElementNS(svgNs, "circle");
    dot.setAttribute("cx", x); dot.setAttribute("cy", DPT + DIH * 0.4);
    dot.setAttribute("r", "5"); dot.setAttribute("fill", `rgb(${r},${g},${b})`);
    const tt = document.createElementNS(svgNs, "title");
    tt.textContent = `rank ${i + 1}/${sorted.length}: λ_max=${v.toExponential(3)}`;
    dot.append(tt); dotSvg.append(dot);
  });
  // labels
  const mkT = (x, y, txt, attrs = {}) => {
    const t = document.createElementNS(svgNs, "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "var(--viz-meta)");
    Object.entries(attrs).forEach(([k, v]) => t.setAttribute(k, v));
    t.textContent = txt; dotSvg.append(t);
  };
  mkT(DPL, DPT + DIH + 14, lMin.toExponential(2));
  mkT(DPL + DIW, DPT + DIH + 14, lMax.toExponential(2), { "text-anchor": "end" });
  mkT(DPL + DIW / 2, DPT + DIH + 14, "λ_max spectrum (sorted)", { "text-anchor": "middle" });
  mkT(DPL - 4, DPT + DIH * 0.4 + 4, "↓", { "text-anchor": "end" });
  const dotSection = mk("div");
  const dotLbl = mk("div"); dotLbl.style.cssText = "font-size:11px;color:var(--muted);margin-bottom:4px";
  dotLbl.textContent = "Projected curvature λ_max per slice (sorted)";
  dotSection.append(dotLbl, dotSvg);
  wrap.append(dotSection);
  } else {
    const unavailable = mk("p", "haar-ext-note");
    unavailable.textContent = "Projected curvature is not assessable: every finite-difference grid was degenerate at the selected perturbation radius.";
    wrap.append(unavailable);
  }

  // K mini-strip canvases (each projection as a small heatmap row)
  const stripsTitle = mk("div");
  stripsTitle.style.cssText = "font-size:11px;color:var(--muted);margin:10px 0 4px";
  stripsTitle.textContent = "Per-projection Δdrift slices";
  wrap.append(stripsTitle);
  const stripsRow = mk("div"); stripsRow.style.cssText = "display:flex;flex-wrap:wrap;gap:6px";

  dmGrids.forEach((g, proj) => {
    const flat = g.flat().filter(v => !isNaN(v));
    const vMax = Math.max(...flat.map(Math.abs), 1e-10);
    const SW = 72, SH = 72;
    const cv = document.createElement("canvas");
    cv.width = SW; cv.height = SH;
    cv.style.cssText = "display:block;image-rendering:pixelated;border-radius:3px;cursor:default";
    cv.title = `Projection ${proj + 1}${Number.isFinite(lambdas[proj]) ? ` — λ_max=${lambdas[proj].toExponential(3)}` : " — Hessian N/A"}`;
    const ctx = cv.getContext("2d");
    const cX = i => Math.floor((i / G) * SW); // float-based, no gap
    for (let bi = 0; bi < G; bi++) {
      const y0 = cX(G - 1 - bi), y1 = cX(G - bi); // y-flip: negative β at bottom
      for (let ai = 0; ai < G; ai++) {
        const x0 = cX(ai), x1 = cX(ai + 1);
        const v = g[bi]?.[ai] ?? 0;
        ctx.fillStyle = divColor(v, vMax);
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
    const lbl = mk("div");
    lbl.style.cssText = "font-size:9px;color:var(--muted);text-align:center;margin-top:2px";
    lbl.textContent = `P${proj + 1}`;
    const box = mk("div");
    box.append(cv, lbl);
    stripsRow.append(box);
  });
  wrap.append(stripsRow);

  const note = mk("p", "haar-ext-note");
  note.textContent = ensembleCopy.method;
  wrap.append(note);
  lossTomoPanel.append(wrap);
}


// Per-seed radial mean (single seed dmean grid → { rc, mu })
// Cross-seed SEM for radial profile: compute per-seed radial means, aggregate with SEM
// 2D projected Hessian at landscape center (finite differences)

async function runLossLandscape(baseline, { numSeeds = 5, gridSize = 9, radius = 0.4 } = {}) {
  const G = gridSize;
  const axes = linspaceArr(-radius, radius, G);

  const int8Grids = [];   // per-seed G×G arrays (LiteRT.js INT8)
  const f64Grids  = [];   // per-seed G×G arrays (WASM synthetic float)

  for (let k = 0; k < numSeeds; k++) {
    const s1 = 1000 + 7 * k, s2 = 5000 + 13 * k;

    // f64 WASM landscape — synchronous, fast (no LiteRT.js round-trip)
    try {
      const res = synthetic_landscape_grid(currentModelBytes, s1, s2, G, radius);
      if (res?.grid?.length === G * G) {
        f64Grids.push(Array.from({ length: G }, (_, bi) =>
          Array.from({ length: G }, (_, ai) => res.grid[bi * G + ai])));
      }
    } catch (_) {}

    // INT8 LiteRT.js landscape
    let dirs;
    try { dirs = landscape_directions(currentModelBytes, s1, s2); } catch (_) { continue; }
    if (!dirs?.metas?.length) continue;
    const d1 = dirs.d1 instanceof Float32Array ? dirs.d1 : new Float32Array(dirs.d1);
    const d2 = dirs.d2 instanceof Float32Array ? dirs.d2 : new Float32Array(dirs.d2);
    const seedGrid = Array.from({ length: G }, () => new Array(G).fill(NaN));
    for (let bi = 0; bi < G; bi++) {
      for (let ai = 0; ai < G; ai++) {
        try {
          const patched = applyLandscapePatch(currentModelBytes, d1, d2, axes[ai], axes[bi], dirs.metas);
          const probe = await runTfliteOutputProbe("wasm", "landscape_probe", { modelBytes: patched });
          const drift = probe?.outputs ? compareOutputArrays(baseline.outputs, probe.outputs) : null;
          seedGrid[bi][ai] = drift?.rms ?? NaN;
        } catch (_) {}
      }
    }
    int8Grids.push(seedGrid);
  }

  const int8Agg = aggregateGrids(int8Grids, G);
  const f64Agg  = aggregateGrids(f64Grids, G);

  const int8Hess = int8Grids.map(g => computeHessian2D(g, axes, G)).filter(Boolean);
  const f64Hess  = f64Grids.map(g => computeHessian2D(g, axes, G)).filter(Boolean);

  const int8Dmeans = int8Grids.map(g => subtractCenter(g, G));
  const f64Dmeans  = f64Grids.map(g => subtractCenter(g, G));

  return {
    int8: int8Agg,
    f64:  f64Agg,
    axes,
    seeds: numSeeds,
    grid: G,
    radius,
    hessian: { int8: aggregateHessian(int8Hess), f64: aggregateHessian(f64Hess) },
    int8Radial: computeRadialProfileSEM(int8Dmeans, axes, G),
    f64Radial:  computeRadialProfileSEM(f64Dmeans,  axes, G),
    requantRatio: (int8Grids.length && f64Grids.length)
      ? requantRatio(int8Agg.dmean, f64Agg.dmean, G) : null,
  };
}

function renderHaarSweepPanel(sweep, haarProfile, outputProfile, extended, kernelHaar = null, alignment = null, activationHaar = null) {
  if (!haarSweepPanel) return;
  const ok = sweep.filter((r) => !r.error && r.drift);
  if (!ok.length) { haarSweepPanel.hidden = true; return; }
  haarSweepPanel.hidden = false;
  haarSweepPanel.replaceChildren();

  const BAND_COLORS = { dc: "#94a3b8", very_low: "#3b82f6", low: "#0d9488", mid: "#22c55e", mid_high: "#f59e0b", high: "#f97316", very_high: "#ef4444" };
  const BAND_ORDER = ["dc", "very_low", "low", "mid", "mid_high", "high", "very_high", "multiscale", "natural"];
  const sortedByDrift = [...ok].sort((a, b) => Number(b.drift?.rms || 0) - Number(a.drift?.rms || 0));
  const sortedByFreq = [...ok].sort((a, b) => {
    const d = BAND_ORDER.indexOf(a.freqBand) - BAND_ORDER.indexOf(b.freqBand);
    return d !== 0 ? d : Number(b.drift?.rms || 0) - Number(a.drift?.rms || 0);
  });
  let sortMode = "drift";
  let selectedId = null;

  function haarChip(label, value) {
    const chip = mk("div", "haar-chip");
    const l = mk("span", "haar-chip-label"); l.textContent = label;
    const v = mk("span", "haar-chip-value"); v.textContent = value;
    chip.append(l, v);
    return chip;
  }

  function haarDm(label, value) {
    const d = mk("div", "haar-dm");
    const l = mk("div", "haar-dm-label"); l.textContent = label;
    const v = mk("div", "haar-dm-value"); v.textContent = value;
    d.append(l, v); return d;
  }

  function patternThumb(patternId, color) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 48 48"); svg.setAttribute("width", "48"); svg.setAttribute("height", "48");
    svg.classList.add("haar-pattern-thumb");
    const bg = document.createElementNS(ns, "rect");
    bg.setAttribute("width", "48"); bg.setAttribute("height", "48"); bg.setAttribute("fill", "#f1f5f9");
    svg.append(bg);
    const neg = "#e2e8f0";
    const cells = [];
    switch (patternId) {
      case "dc":   cells.push([0, 0, 48, 48, color]); break;
      case "h_edge": cells.push([0, 0, 48, 24, color], [0, 24, 48, 24, neg]); break;
      case "v_edge": cells.push([0, 0, 24, 48, color], [24, 0, 24, 48, neg]); break;
      case "h_line": cells.push([0, 0, 48, 16, color], [0, 16, 48, 16, neg], [0, 32, 48, 16, color]); break;
      case "v_line": cells.push([0, 0, 16, 48, color], [16, 0, 16, 48, neg], [32, 0, 16, 48, color]); break;
      case "diag": case "diag_sv": case "diag_sh":
        cells.push([0, 0, 24, 24, neg], [24, 0, 24, 24, color], [0, 24, 24, 24, color], [24, 24, 24, 24, neg]);
        break;
      default: {
        // Multi-scale Haar bank: haar_LH_S, haar_HL_S, haar_HH_S
        const haarM = patternId.match(/^haar_(LH|HL|HH)_(\d+)$/);
        if (haarM) {
          const type = haarM[1];
          const S    = Math.max(2, Math.min(parseInt(haarM[2]), 24));
          if (type === "LH") {
            // Horizontal stripes (tiled 2S blocks)
            for (let y = 0; y < 48; y += 2 * S)
              cells.push([0, y, 48, S, color], [0, y + S, 48, S, neg]);
          } else if (type === "HL") {
            // Vertical stripes
            for (let x = 0; x < 48; x += 2 * S)
              cells.push([x, 0, S, 48, color], [x + S, 0, S, 48, neg]);
          } else {
            // HH: diagonal checker
            for (let r = 0; r < Math.ceil(48 / S); r++)
              for (let c = 0; c < Math.ceil(48 / S); c++)
                cells.push([c * S, r * S, S, S, (r + c) % 2 === 0 ? color : neg]);
          }
        } else if (patternId.startsWith("fractal_") || patternId === "coco_prior") {
          // Fractal/COCO: multi-scale checker overlay (representative icon)
          for (const S of [2, 4, 8]) {
            for (let r = 0; r < Math.ceil(48 / S); r++)
              for (let c = 0; c < Math.ceil(48 / S); c++) {
                const alpha = S === 2 ? "33" : S === 4 ? "66" : "99";
                cells.push([c * S, r * S, S, S, ((r + c) % 2 === 0 ? color : neg) + alpha]);
              }
          }
        } else {
          const f = parseInt(patternId.replace("checker_", ""), 10) || 4;
          for (let r = 0; r < Math.ceil(48 / f); r++)
            for (let c = 0; c < Math.ceil(48 / f); c++)
              cells.push([c * f, r * f, f, f, (r + c) % 2 === 0 ? color : neg]);
        }
      }
    }
    for (const [x, y, w, h, fill] of cells) {
      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", w); rect.setAttribute("height", h);
      rect.setAttribute("fill", fill); svg.append(rect);
    }
    if (patternId === "diag_sv" || patternId === "diag_sh") {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", "34"); t.setAttribute("y", "44");
      t.setAttribute("font-size", "12"); t.setAttribute("fill", "rgba(0,0,0,0.45)");
      t.setAttribute("font-family", "sans-serif");
      t.textContent = patternId === "diag_sv" ? "↓" : "→";
      svg.append(t);
    }
    return svg;
  }

  function haarInterp(pattern, drift) {
    const rms = Number(drift?.rms || 0);
    const maxRms = ok.reduce((m, r) => Math.max(m, Number(r.drift?.rms || 0)), 1e-9);
    const isHigh = rms > maxRms * 0.6;
    const isMid = rms > maxRms * 0.25;
    const { id } = pattern;
    if (id === "dc") return isHigh ? "High DC drift: output is strongly driven by absolute pixel magnitude. Verify input normalization (mean/std)." : "Low DC drift: model is not strongly driven by absolute intensity.";
    if (id === "h_edge") return (haarProfile.orientation_bias === "horizontal_dominant" && isHigh) ? "Horizontal-edge sensitivity candidate: model output shifts more strongly with horizontal transitions vs zero baseline — possibly architecture-driven (asymmetric pooling) or input-distribution skew." : "Horizontal edge response within expected range.";
    if (id === "v_edge") return (haarProfile.orientation_bias === "vertical_dominant" && isHigh) ? "Vertical-edge sensitivity candidate: elevated response to vertical transitions vs zero baseline — possible vertical-bias in pooling or input distribution." : "Vertical edge drift is consistent with orientation balance.";
    if (id === "diag") return isHigh ? "High diagonal sensitivity: model reacts strongly to 4-quadrant ± patterns — common in high-resolution detectors or corner-sensitive architectures." : "Low diagonal sensitivity: model relies less on corner-like diagonal transitions.";
    if (id === "h_line") return isHigh ? "High horizontal ridge response: the model reacts to thin horizontal bright/dark bands — common in face detection (eyes/mouth) or document/text models." : "Horizontal ridge response is moderate.";
    if (id === "v_line") return isHigh ? "High vertical ridge response: thin vertical bands influence outputs strongly." : "Vertical ridge response is moderate.";
    if (id.startsWith("checker_")) {
      const freqLabel = { checker_2: "Nyquist (2px)", checker_4: "high (4px)", checker_8: "medium (8px)", checker_16: "low (16px)" }[id];
      const trend = haarProfile.frequency_trend;
      const highNote = trend === "high_freq_sensitive"
        ? "Consistent with high-freq-sensitive profile — fine detail drives outputs."
        : trend === "mid_high_freq_peak"
          ? `Mid-high peak profile: resonance at ${haarProfile.peak_checker_label || "4–8px"} — peak sensitivity is not at the finest scale.`
          : "Locally dominant frequency band.";
      const lowNote = trend === "low_freq_sensitive"
        ? "Consistent with low-freq-sensitive profile — coarse structure dominates."
        : trend === "mid_low_freq_peak"
          ? `Mid-low peak profile: peak response at ${haarProfile.peak_checker_label || "8px"}, not at the coarsest scale.`
          : "This frequency band is less influential.";
      return isHigh
        ? `High response at ${freqLabel} frequency. ${highNote}`
        : `Low response at ${freqLabel}. ${lowNote}`;
    }
    if (id === "diag_sv" || id === "diag_sh") {
      const base = ok.find((r) => r.id === "diag");
      const baseRms = Number(base?.drift?.rms || 0);
      const axis = id === "diag_sv" ? "vertical" : "horizontal";
      const ratio = baseRms > 1e-9 ? rms / baseRms : 1;
      if (ratio > 1.15) return `${axis.charAt(0).toUpperCase() + axis.slice(1)} shift increases drift by ${Math.round((ratio - 1) * 100)}% — model shows ${axis} position sensitivity.`;
      if (ratio < 0.85) return `${axis.charAt(0).toUpperCase() + axis.slice(1)} shift reduces drift — position-specific suppression of the diagonal pattern.`;
      return `Drift is stable under ${axis} stride shift (±${Math.round(Math.abs(ratio - 1) * 100)}%), consistent with spatial translation invariance.`;
    }
    // Multi-scale Haar bank patterns
    const haarM = id.match(/^haar_(LH|HL|HH)_(\d+)$/);
    if (haarM) {
      const type = haarM[1];
      const scale = parseInt(haarM[2]);
      const typeLabel = { LH: "horizontal-edge", HL: "vertical-edge", HH: "diagonal/checker" }[type];
      const peakType = haarProfile.peak_by_haar_type?.[type];
      const isPeak   = peakType?.scale === scale;
      const bankBias = haarProfile.bank_orientation_bias;
      if (!isHigh && !isMid) return `Low synthetic-pattern sensitivity at ${typeLabel} ${scale}px scale. Not a dominant frequency/orientation candidate for this model.`;
      if (isPeak) {
        const bankNote = type === "LH" && bankBias === "horizontal_dominant" ? " Consistent with horizontal-dominant bank orientation bias."
          : type === "HL" && bankBias === "vertical_dominant" ? " Consistent with vertical-dominant bank orientation bias."
          : bankBias === "isotropic" ? " Bank orientation appears isotropic." : "";
        return `Peak response within ${type} sub-band: model output is most sensitive to ${typeLabel} transitions at ${scale}px scale.${bankNote} This is a synthetic-pattern signal — compare against same model at different builds.`;
      }
      return `Elevated response to ${typeLabel} patterns at ${scale}px scale — not the peak ${type} scale (peak: ${peakType?.scale ?? "N/A"}px). Partial sensitivity candidate.`;
    }
    // Fractal patterns
    if (id.startsWith("fractal_")) {
      const fType = id.replace("fractal_", "");
      const label = fType === "full" ? "LH+HL+HH (full)" : fType.toUpperCase();
      return isHigh
        ? `High response to fractal ${label} probe — 1/f-weighted superposition of ${label} atoms across scales 2–32px. The model is sensitive to natural-image-like multiscale structure, not just single-frequency patterns.`
        : `Fractal ${label} probe elicits lower response than expected — model sensitivity is more frequency-selective (responds to individual scales but not their 1/f combination).`;
    }
    if (id === "coco_prior") {
      return isHigh
        ? `High response to COCO-prior probe (LH:HL:HH weighted by COCO val2017 Haar statistics). Model reacts to natural-image-like detail-band mixtures — supports ecological validity of the Haar probe suite.`
        : `Low response to COCO-prior probe — model may not respond to detail-band energy at the COCO-typical magnitude; consider higher amplitude or check input normalization.`;
    }
    return "";
  }

  function renderDetail(pattern) {
    detailEl.hidden = false;
    const color = BAND_COLORS[pattern.freqBand] || "#94a3b8";
    const d = pattern.drift;

    const head = mk("div", "haar-detail-head");
    const thumb = patternThumb(pattern.id, color);
    const info = mk("div", "haar-detail-info");
    const title = mk("div", "haar-detail-title"); title.textContent = pattern.label;
    const band = mk("span", "haar-detail-band");
    band.textContent = (pattern.freqBand || "").replace(/_/g, " ");
    band.style.background = color + "22"; band.style.color = color;
    info.append(title, band);
    head.append(thumb, info);

    const metrics = mk("div", "haar-detail-metrics");
    metrics.append(
      haarDm("RMS Drift", formatDrift(d?.rms)),
      haarDm("Max Abs", formatDrift(d?.maxAbs)),
      haarDm("Mean Abs", formatDrift(d?.meanAbs)),
      haarDm("Cosine Dist", formatDrift(d?.cosineDistance)),
      haarDm("Top-1 Flip", d?.top1Flip ? "YES" : "No"),
      haarDm("Severity", d?.severity || "—"),
    );

    const interp = mk("p", "haar-detail-interp");
    interp.textContent = haarInterp(pattern, d) || "No interpretation rule for this pattern.";

    detailEl.replaceChildren(head, metrics, interp);
  }

  function buildChart(sorted) {
    const maxRms = sorted.reduce((m, r) => Math.max(m, Number(r.drift?.rms || 0)), 1e-9);
    const wrap = mk("div", "haar-chart");
    const rowEls = [];
    for (const pattern of sorted) {
      const color = BAND_COLORS[pattern.freqBand] || "#94a3b8";
      const rms = Number(pattern.drift?.rms || 0);
      const row = mk("div", "haar-row");
      row.dataset.id = pattern.id;
      if (selectedId === pattern.id) row.classList.add("haar-selected");
      const labelEl = mk("span", "haar-row-label");
      const dot = mk("i", "haar-dot"); dot.style.background = color; labelEl.append(dot);
      labelEl.append(pattern.label);
      const track = mk("div", "haar-row-track");
      const bar = mk("div", "haar-row-bar");
      bar.style.width = `${Math.max(1.5, (rms / maxRms) * 100)}%`;
      bar.style.background = color;
      const val = mk("span", "haar-row-value");
      val.textContent = `${formatDrift(rms)} ${outputProfile.unit}`;
      track.append(bar, val);
      row.append(labelEl, track);
      row.addEventListener("click", () => {
        for (const r of rowEls) r.classList.remove("haar-selected");
        row.classList.add("haar-selected");
        selectedId = pattern.id;
        renderDetail(pattern);
      });
      wrap.append(row);
      rowEls.push(row);
    }
    for (const pattern of sweep.filter((r) => r.error)) {
      const row = mk("div", "haar-row haar-row-err");
      const labelEl = mk("span", "haar-row-label"); labelEl.textContent = pattern.label;
      const errEl = mk("span", "haar-row-error"); errEl.textContent = `error: ${pattern.error}`;
      row.append(labelEl, errEl); wrap.append(row);
    }
    return { wrap, rowEls };
  }

  const head = mk("div", "haar-head");
  const headTitle = mk("h3"); headTitle.textContent = "Haar Pattern Sweep";
  const headMeta = mk("p", "haar-head-meta");
  headMeta.textContent = `${ok.length}/${sweep.length} patterns · drift vs zero-fill baseline`;
  head.append(headTitle, headMeta);

  const chips = mk("div", "haar-chips");
  const freqMap = {
    high_freq_sensitive:  "High-freq sensitive",
    mid_high_freq_peak:   "Mid-high peak",
    broadband:            "Broadband",
    mid_low_freq_peak:    "Mid-low peak",
    low_freq_sensitive:   "Low-freq sensitive",
    indeterminate:        "Indeterminate",
  };
  const biasMap = { isotropic: "Isotropic", horizontal_dominant: "H dominant", vertical_dominant: "V dominant" };
  const posMap = {
    translation_invariant:       "Translation invariant",
    vertical_position_sensitive: "V-position sensitive",
    horizontal_position_sensitive: "H-position sensitive",
    position_sensitive:          "Position sensitive",
  };
  // Append "(peak Npx)" to frequency label when peak is not at the expected endpoint
  const freqTrend = haarProfile.frequency_trend;
  const peakLabel = haarProfile.peak_checker_label;
  const freqEndpointMismatch = peakLabel && freqTrend === "high_freq_sensitive" && peakLabel !== "2px"
    || freqTrend === "low_freq_sensitive" && peakLabel !== "16px";
  const freqChipValue = (freqMap[freqTrend] || freqTrend || "—")
    + (freqEndpointMismatch ? ` (peak ${peakLabel})` : "");
  chips.append(
    haarChip("Frequency", freqChipValue),
    haarChip("Orientation", biasMap[haarProfile.orientation_bias] || haarProfile.orientation_bias || "—"),
    haarChip("Position", posMap[haarProfile.position_sensitivity] || haarProfile.position_sensitivity || "—"),
    ...(haarProfile.freq_ratio_fine_coarse !== null ? [haarChip("Fine/Coarse", `${haarProfile.freq_ratio_fine_coarse}×`)] : []),
    haarChip("Most sensitive", haarProfile.most_sensitive_pattern || "—"),
  );

  const legend = mk("div", "haar-legend");
  const legendEntries = [["dc", "DC"], ["very_low", "Edge"], ["low", "Diagonal/Ridge"], ["mid_high", "Checker (mid)"], ["very_high", "Checker (fine)"]];
  for (const [band, label] of legendEntries) {
    const item = mk("span", "haar-legend-item");
    const dot = mk("i", "haar-dot"); dot.style.background = BAND_COLORS[band] || "#ccc";
    item.append(dot, label); legend.append(item);
  }

  const sortBar = mk("div", "haar-sort-bar");
  const sortLbl = mk("span", "haar-sort-label"); sortLbl.textContent = "Sort:";
  const btnDrift = mk("button", "haar-sort-btn active"); btnDrift.textContent = "By drift"; btnDrift.type = "button";
  const btnFreq = mk("button", "haar-sort-btn"); btnFreq.textContent = "By frequency"; btnFreq.type = "button";
  sortBar.append(sortLbl, btnDrift, btnFreq);

  const chartWrap = mk("div");
  const detailEl = mk("div", "haar-detail"); detailEl.hidden = true;

  function rebuildChart(sorted) {
    const { wrap } = buildChart(sorted);
    chartWrap.replaceChildren(wrap);
  }

  btnDrift.addEventListener("click", () => {
    if (sortMode === "drift") return;
    sortMode = "drift"; btnDrift.classList.add("active"); btnFreq.classList.remove("active");
    rebuildChart(sortedByDrift);
  });
  btnFreq.addEventListener("click", () => {
    if (sortMode === "freq") return;
    sortMode = "freq"; btnFreq.classList.add("active"); btnDrift.classList.remove("active");
    rebuildChart(sortedByFreq);
  });

  const extChips = mk("div", "haar-chips haar-ext-chips");
  const bankBiasMap = { isotropic: "Bank: isotropic", horizontal_dominant: "Bank: H-dominant", vertical_dominant: "Bank: V-dominant" };
  const bankBiasVal = bankBiasMap[haarProfile.bank_orientation_bias] || haarProfile.bank_orientation_bias || "—";
  const peakLH = haarProfile.peak_by_haar_type?.["LH"];
  const peakHL = haarProfile.peak_by_haar_type?.["HL"];
  const peakHH = haarProfile.peak_by_haar_type?.["HH"];
  if (peakLH || peakHL || peakHH) {
    extChips.append(haarChip("Bank bias", bankBiasVal));
    if (peakLH) extChips.append(haarChip("LH peak", `${peakLH.scale}px`));
    if (peakHL) extChips.append(haarChip("HL peak", `${peakHL.scale}px`));
    if (peakHH) extChips.append(haarChip("HH peak", `${peakHH.scale}px`));
  }
  const tranProf = extended?.translation?.profile;
  if (tranProf) extChips.append(haarChip("Trans. sensitivity", tranProf.label?.replace(/_/g, " ") || tranProf.translation_sensitivity));
  const rotProf0 = extended?.rotation?.by_pattern?.["haar_HL_8"] || extended?.rotation?.by_pattern?.["haar_LH_8"];
  if (rotProf0) extChips.append(haarChip("Rotation", rotProf0.label?.replace(/_/g, " ") || `asym ${rotProf0.rotation_asymmetry}×`));
  const phaseProf0 = extended?.phase?.by_pattern?.["haar_HH_4"] || extended?.phase?.by_pattern?.["haar_HH_8"];
  if (phaseProf0) extChips.append(haarChip("Phase jitter", phaseProf0.label?.replace(/_/g, " ") || `${phaseProf0.phase_jitter}`));
  const ampProf = extended?.amplitude?.profile;
  if (ampProf) extChips.append(haarChip("Amplitude", ampProf.label?.replace(/_/g, " ") || "—"));
  if (extChips.children.length) haarSweepPanel.append(extChips);  // defer to after sortBar

  haarSweepPanel.append(head, chips, legend, sortBar, chartWrap, detailEl);
  if (extChips.children.length) haarSweepPanel.insertBefore(extChips, sortBar);
  rebuildChart(sortedByDrift);

  // Auto-select most sensitive
  const topPattern = sortedByDrift[0];
  if (topPattern) {
    selectedId = topPattern.id;
    renderDetail(topPattern);
    const firstRow = chartWrap.querySelector(".haar-row");
    if (firstRow) firstRow.classList.add("haar-selected");
  }

  if (extended) {
    const extWrap = mk("div", "haar-ext-panels");

    // Translation heatmap (3×3)
    if (extended.translation?.profile?.heatmap?.length) {
      const tp = extended.translation.profile;
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title");
      title.textContent = `Translation probe (${HAAR_SWEEP_CONFIG.translationPatternId}) — sensitivity: ${tp.translation_sensitivity}`;
      const grid = mk("div", "haar-spatial-grid");
      const maxRms = Math.max(...tp.heatmap.map((c) => c.rms || 0), 1e-9);
      for (const cell of tp.heatmap) {
        const c = mk("div", "haar-spatial-cell");
        const t = (cell.rms || 0) / maxRms;
        c.style.background = `rgba(239,68,68,${(0.1 + t * 0.85).toFixed(2)})`;
        c.title = `${cell.region}: ${formatDrift(cell.rms)}`;
        const lbl = mk("span", "haar-spatial-label"); lbl.textContent = cell.region?.split("-")[0]?.[0]?.toUpperCase() || "";
        const val = mk("span", "haar-spatial-val"); val.textContent = formatDrift(cell.rms);
        c.append(lbl, val); grid.append(c);
      }
      const note = mk("p", "haar-ext-note");
      note.textContent = `Peak: ${tp.peak_region} · Trough: ${tp.trough_region} · Label: ${tp.label?.replace(/_/g, " ")}`;
      sec.append(title, grid, note); extWrap.append(sec);
    }

    // Rotation bars
    if (extended.rotation?.results?.length) {
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title"); title.textContent = "Rotation sensitivity (0° / 90° / 180° / 270°)";
      const barsWrap = mk("div", "haar-ext-bars");
      for (const patId of HAAR_SWEEP_CONFIG.rotationPatternIds) {
        const rp = extended.rotation.by_pattern[patId];
        if (!rp) continue;
        const patLabel = mk("div", "haar-ext-bar-group-label"); patLabel.textContent = patId.replace("haar_", "");
        const barGroup = mk("div", "haar-ext-bar-group");
        const maxRms = Math.max(...Object.values(rp.by_rotation), 1e-9);
        for (const deg of [0, 90, 180, 270]) {
          const rms = rp.by_rotation[deg] || 0;
          const row = mk("div", "haar-ext-bar-row");
          const lbl = mk("span", "haar-ext-bar-label"); lbl.textContent = `${deg}°`;
          const track = mk("div", "haar-ext-bar-track");
          const bar = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (rms / maxRms) * 100).toFixed(1)}%`;
          const val = mk("span", "haar-ext-bar-val"); val.textContent = formatDrift(rms);
          track.append(bar, val); row.append(lbl, track); barGroup.append(row);
        }
        const note = mk("p", "haar-ext-note"); note.textContent = `Asymmetry: ${rp.rotation_asymmetry}× · ${rp.label?.replace(/_/g, " ")}`;
        barsWrap.append(patLabel, barGroup, note);
      }
      sec.append(title, barsWrap); extWrap.append(sec);
    }

    // Phase jitter bars
    if (extended.phase?.results?.length) {
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title"); title.textContent = "Phase jitter (shift offset in pixels)";
      const barsWrap = mk("div", "haar-ext-bars");
      for (const patId of HAAR_SWEEP_CONFIG.phasePatternIds) {
        const pp = extended.phase.by_pattern[patId];
        if (!pp) continue;
        const patResults = extended.phase.results.filter((r) => r.pattern_id === patId);
        const patLabel = mk("div", "haar-ext-bar-group-label"); patLabel.textContent = patId.replace("haar_", "");
        const barGroup = mk("div", "haar-ext-bar-group");
        const maxRms = Math.max(...patResults.map((r) => Number(r.drift?.rms || 0)), 1e-9);
        for (const r of patResults) {
          const rms = Number(r.drift?.rms || 0);
          const row = mk("div", "haar-ext-bar-row");
          const lbl = mk("span", "haar-ext-bar-label"); lbl.textContent = `+${r.phaseOffset}px`;
          const track = mk("div", "haar-ext-bar-track");
          const bar = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (rms / maxRms) * 100).toFixed(1)}%`;
          bar.style.background = "#0d9488";
          const val = mk("span", "haar-ext-bar-val"); val.textContent = formatDrift(rms);
          track.append(bar, val); row.append(lbl, track); barGroup.append(row);
        }
        const note = mk("p", "haar-ext-note"); note.textContent = `Phase jitter: ${pp.phase_jitter} · Aliasing risk: ${pp.aliasing_risk} · ${pp.label?.replace(/_/g, " ")}`;
        barsWrap.append(patLabel, barGroup, note);
      }
      sec.append(title, barsWrap); extWrap.append(sec);
    }

    // Amplitude sweep gain curve
    if (extended.amplitude?.profile?.amplitudes?.length) {
      const ap = extended.amplitude.profile;
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title"); title.textContent = `Amplitude sweep — ${HAAR_SWEEP_CONFIG.amplitudePatternId.replace("haar_", "")} · response_gain = output_rms / input_l2`;
      const barGroup = mk("div", "haar-ext-bar-group");
      const maxGain = Math.max(...ap.amplitudes.map((a) => a.gain), 1e-9);
      for (const a of ap.amplitudes) {
        const row = mk("div", "haar-ext-bar-row");
        const lbl = mk("span", "haar-ext-bar-label"); lbl.textContent = `amp ${a.amplitude}`;
        const track = mk("div", "haar-ext-bar-track");
        const bar = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (a.gain / maxGain) * 100).toFixed(1)}%`;
        bar.style.background = "#8b5cf6";
        const val = mk("span", "haar-ext-bar-val"); val.textContent = `gain ${a.gain.toFixed(3)}`;
        track.append(bar, val); row.append(lbl, track); barGroup.append(row);
      }
      const note = mk("p", "haar-ext-note");
      note.textContent = `Low-amp gain: ${ap.low_gain} · High-amp gain: ${ap.high_gain} · Saturation: ${ap.saturation_ratio} · ${ap.label?.replace(/_/g, " ")}`;
      sec.append(title, barGroup, note); extWrap.append(sec);
    }

    // Polarity asymmetry
    if (extended.polarity?.profile?.length) {
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title"); title.textContent = "Polarity pairs (+/− pattern)";
      const rows = mk("div", "haar-ext-bars");
      for (const pp of extended.polarity.profile) {
        const patLabel = mk("div", "haar-ext-bar-group-label"); patLabel.textContent = pp.pattern_id.replace("haar_", "");
        const barGroup = mk("div", "haar-ext-bar-group");
        const maxRms = Math.max(pp.plus_rms, pp.minus_rms, 1e-9);
        for (const [label, rms] of [["+pattern", pp.plus_rms], ["−pattern", pp.minus_rms]]) {
          const row = mk("div", "haar-ext-bar-row");
          const lbl = mk("span", "haar-ext-bar-label"); lbl.textContent = label;
          const track = mk("div", "haar-ext-bar-track");
          const bar = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (rms / maxRms) * 100).toFixed(1)}%`;
          bar.style.background = label.startsWith("+") ? "#3b82f6" : "#f97316";
          const val = mk("span", "haar-ext-bar-val"); val.textContent = formatDrift(rms);
          track.append(bar, val); row.append(lbl, track); barGroup.append(row);
        }
        const note = mk("p", "haar-ext-note");
        note.textContent = `Polarity asymmetry: ${pp.polarity_asymmetry} · Dominant: ${pp.dominant} · ${pp.label?.replace(/_/g, " ")}`;
        rows.append(patLabel, barGroup, note);
      }
      sec.append(title, rows); extWrap.append(sec);
    }

    // Channel sweep
    if (extended.channels?.results?.length) {
      const sec = mk("div", "haar-ext-section");
      const title = mk("div", "haar-ext-title"); title.textContent = `Channel sensitivity (${HAAR_SWEEP_CONFIG.channelPatternId.replace("haar_", "")})`;
      const barGroup = mk("div", "haar-ext-bar-group");
      const chanResults = extended.channels.results;
      const maxRms = Math.max(...chanResults.map((r) => Number(r.drift?.rms || 0)), 1e-9);
      const chanColors = { all: "#64748b", r: "#ef4444", g: "#22c55e", b: "#3b82f6" };
      for (const r of chanResults) {
        const rms = Number(r.drift?.rms || 0);
        const row = mk("div", "haar-ext-bar-row");
        const lbl = mk("span", "haar-ext-bar-label"); lbl.textContent = r.channelMode;
        const track = mk("div", "haar-ext-bar-track");
        const bar = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (rms / maxRms) * 100).toFixed(1)}%`;
        bar.style.background = chanColors[r.channelMode] || "#94a3b8";
        const val = mk("span", "haar-ext-bar-val"); val.textContent = formatDrift(rms);
        track.append(bar, val); row.append(lbl, track); barGroup.append(row);
      }
      sec.append(title, barGroup); extWrap.append(sec);
    }

    if (extWrap.children.length) haarSweepPanel.append(extWrap);
  }

  const fractalPatterns = sweep.filter((r) => !r.error && r.drift &&
    (r.id?.startsWith("fractal_") || r.id === "coco_prior"));
  if (fractalPatterns.length) {
    const fWrap  = mk("div", "haar-ext-panels");
    const fSec   = mk("div", "haar-ext-section");
    const fTitle = mk("div", "haar-ext-title");
    fTitle.textContent = "Fractal / COCO-prior probes (multiscale natural-image-like inputs)";
    const fBars  = mk("div", "haar-ext-bars");
    const maxRms = Math.max(1e-12, ...fractalPatterns.map((r) => Number(r.drift?.rms || 0)));
    for (const r of fractalPatterns) {
      const rms = Number(r.drift?.rms || 0);
      const row   = mk("div", "haar-ext-bar-row");
      const lbl   = mk("span", "haar-ext-bar-label"); lbl.textContent = r.label || r.id;
      const track = mk("div", "haar-ext-bar-track");
      const bar   = mk("div", "haar-ext-bar"); bar.style.width = `${Math.max(2, (rms / maxRms) * 100).toFixed(1)}%`;
      const gainTxt = r.response_gain != null ? ` · gain ${r.response_gain.toFixed(4)}` : "";
      const val   = mk("span", "haar-ext-bar-val"); val.textContent = formatDrift(rms) + gainTxt;
      track.append(bar, val); row.append(lbl, track); fBars.append(row);
    }
    const fNote  = mk("p", "haar-ext-note");
    fNote.textContent = `Fractal probes use 1/f-weighted superposition of LH+HL+HH atoms (scales 2–32px). COCO prior weights LH:HL:HH ≈ ${
      COCO_HAAR_PRIOR.detail_weight_lh.toFixed(1)}:${COCO_HAAR_PRIOR.detail_weight_hl.toFixed(1)}:${COCO_HAAR_PRIOR.detail_weight_hh.toFixed(1)} from empirical COCO val2017 Haar statistics.`;
    fSec.append(fTitle, fBars, fNote); fWrap.append(fSec);
    haarSweepPanel.append(fWrap);
  }

  if (kernelHaar?.summary && kernelHaar.ops?.length) {
    const ks = kernelHaar.summary;
    const kWrap = mk("div", "haar-ext-panels");
    const kSec  = mk("div", "haar-ext-section");
    const kTitle= mk("div", "haar-ext-title");
    kTitle.textContent = `Kernel Haar decomposition — ${ks.conv_op_count} conv/depthwise ops`;
    const chips = mk("div", "haar-ext-chips");
    const addKChip = (label, val, tone) => {
      const c = mk("span", `haar-chip haar-chip-${tone || "info"}`);
      c.textContent = `${label}: ${val}`; chips.append(c);
    };
    addKChip("global dominant", ks.global_dominant.toUpperCase(), "info");
    addKChip("edge-heavy ops", `${ks.edge_heavy_ops}/${ks.conv_op_count}`, ks.edge_heavy_ops > ks.dc_heavy_ops ? "ok" : "info");
    addKChip("orient bias", ks.orientation_bias > 0.01 ? `H-edge (${ks.orientation_bias.toFixed(3)})` : ks.orientation_bias < -0.01 ? `V-edge (${ks.orientation_bias.toFixed(3)})` : "isotropic", "info");
    kSec.append(kTitle, chips);

    // Per-op table (top ops by edge_dc_ratio)
    const topOps = [...kernelHaar.ops]
      .sort((a, b) => b.edge_dc_ratio - a.edge_dc_ratio)
      .slice(0, 8);
    const table = mk("div", "haar-ext-bars");
    for (const op of topOps) {
      const me = op.mean_energy;
      const row = mk("div", "haar-ext-bar-row");
      const lbl = mk("span", "haar-ext-bar-label");
      lbl.textContent = `#${op.op_index} ${op.op_name} ${op.kernel_h}×${op.kernel_w}`;
      lbl.style.minWidth = "16ch";
      const track = mk("div", "haar-ext-bar-track");
      // Stacked mini-bars: LL(grey) LH(blue) HL(teal) HH(orange)
      const total = me.ll + me.lh + me.hl + me.hh || 1;
      for (const [key, color] of [["ll","#94a3b8"],["lh","#3b82f6"],["hl","#0d9488"],["hh","#f59e0b"]]) {
        const seg = mk("div", "haar-ext-bar");
        seg.style.cssText = `width:${((me[key]/total)*100).toFixed(1)}%;background:${color};display:inline-block;height:100%`;
        seg.title = `${key.toUpperCase()}: ${(me[key]*100).toFixed(1)}%`;
        track.append(seg);
      }
      const val = mk("span", "haar-ext-bar-val");
      val.textContent = `${op.dominant.toUpperCase()} dom · edge/DC=${op.edge_dc_ratio.toFixed(2)} · CS=${op.mean_energy.center_surround.toFixed(3)}`;
      row.append(lbl, track, val); table.append(row);
    }
    kSec.append(table);
    const kLegend = mk("p", "haar-ext-note");
    kLegend.innerHTML = `<span style="color:#94a3b8">■ LL(DC)</span> &nbsp;<span style="color:#3b82f6">■ LH(H-edge)</span> &nbsp;<span style="color:#0d9488">■ HL(V-edge)</span> &nbsp;<span style="color:#f59e0b">■ HH(diag)</span> &nbsp;— averaged across input channels per filter, top 8 by edge/DC ratio`;
    kSec.append(kLegend);
    kWrap.append(kSec);
    haarSweepPanel.append(kWrap);
  }

  if (activationHaar?.summary && activationHaar.ops?.length) {
    const as_ = activationHaar.summary;
    const aWrap = mk("div", "haar-ext-panels");
    const aSec  = mk("div", "haar-ext-section");
    const aTitle= mk("div", "haar-ext-title");
    aTitle.textContent = `Activation Haar decomposition — ${as_.executed_count} of ${as_.conv_op_count} ops (${as_.input_size})`;
    const chips = mk("div", "haar-ext-chips");
    const addChip = (lbl, val, tone) => {
      const c = mk("span", `haar-chip haar-chip-${tone || "info"}`);
      c.textContent = `${lbl}: ${val}`; chips.append(c);
    };
    addChip("activation dominant", (as_.dominant_band || "?").toUpperCase(), "info");
    addChip("mean smoothness", (as_.mean_spatial_smoothness * 100).toFixed(1) + "%", as_.mean_spatial_smoothness > 0.5 ? "ok" : "warn");
    addChip("LL", (as_.ll_mean * 100).toFixed(1) + "%", "info");
    addChip("LH+HL", ((as_.lh_mean + as_.hl_mean) * 100).toFixed(1) + "%", "info");
    addChip("HH", (as_.hh_mean * 100).toFixed(1) + "%", "info");
    if (as_.skipped_count > 0) addChip("skipped", as_.skipped_count, "warn");
    aSec.append(aTitle, chips);

    // Per-op bars for executed conv/dw ops
    const execOps = activationHaar.ops
      .filter(r => !r.skipped && !r.spatial_too_small)
      .slice(0, 12);
    if (execOps.length) {
      const bars = mk("div", "haar-ext-bars");
      for (const op of execOps) {
        const row = mk("div", "haar-ext-bar-row");
        const lbl = mk("span", "haar-ext-bar-label");
        const shp = op.output_shape?.length >= 4
          ? `${op.output_shape[1]}×${op.output_shape[2]}×${op.output_shape[3]}` : "";
        lbl.textContent = `#${op.op_index} ${op.op_name} ${shp}`;
        lbl.style.minWidth = "18ch";
        const track = mk("div", "haar-ext-bar-track");
        const total = op.ll_energy + op.lh_energy + op.hl_energy + op.hh_energy || 1;
        for (const [key, color] of [["ll_energy","#94a3b8"],["lh_energy","#3b82f6"],["hl_energy","#0d9488"],["hh_energy","#f59e0b"]]) {
          const seg = mk("div", "haar-ext-bar");
          seg.style.cssText = `width:${((op[key]/total)*100).toFixed(1)}%;background:${color};display:inline-block;height:100%`;
          seg.title = `${key.replace("_energy","").toUpperCase()}: ${(op[key]*100).toFixed(1)}%`;
          track.append(seg);
        }
        const val = mk("span", "haar-ext-bar-val");
        val.textContent = `${op.dominant_band} · smooth=${(op.spatial_smoothness*100).toFixed(0)}%`;
        row.append(lbl, track, val); bars.append(row);
      }
      aSec.append(bars);
    }
    const note = mk("p", "haar-ext-note");
    note.innerHTML = `<span style="color:#94a3b8">■ LL(low-freq)</span> &nbsp;<span style="color:#3b82f6">■ LH(H-edge)</span> &nbsp;<span style="color:#0d9488">■ HL(V-edge)</span> &nbsp;<span style="color:#f59e0b">■ HH(diag)</span> &nbsp;— synthetic 32×32 input, full f64 forward pass in WASM. Skipped=${as_.skipped_count}, spatial&lt;2×2=${as_.spatial_too_small_count}.`;
    aSec.append(note);
    aWrap.append(aSec);
    haarSweepPanel.append(aWrap);
  }

  if (alignment?.score != null) {
    const aWrap = mk("div", "haar-ext-panels");
    const aSec  = mk("div", "haar-ext-section");
    const aTitle= mk("div", "haar-ext-title");
    const pct   = Math.round(alignment.score * 100);
    aTitle.textContent = `Static-runtime alignment — ${pct}% (${alignment.aligned_count}/${alignment.total_conv_ops} ops)`;
    const aChips= mk("div", "haar-ext-chips");
    const addAChip = (label, val, tone) => {
      const c = mk("span", `haar-chip haar-chip-${tone || "info"}`);
      c.textContent = `${label}: ${val}`; aChips.append(c);
    };
    addAChip("kernel dominant", (alignment.kernel_global_dominant || "?").toUpperCase(), "info");
    addAChip("runtime dominant", (alignment.runtime_dominant || "?").toUpperCase(), "info");
    addAChip("orient match", alignment.orientation_match ? "yes" : "no", alignment.orientation_match ? "ok" : "warn");
    const scoreChipTone = alignment.score >= 0.7 ? "ok" : alignment.score >= 0.4 ? "warn" : "error";
    addAChip("alignment", `${pct}%`, scoreChipTone);

    const aInterp = mk("p", "haar-ext-note"); aInterp.textContent = alignment.interpretation;
    aSec.append(aTitle, aChips, aInterp);

    // Misaligned ops detail
    const misaligned = (alignment.per_op || []).filter((o) => !o.aligned);
    if (misaligned.length) {
      const mTitle = mk("div", "haar-ext-bar-group-label");
      mTitle.textContent = "Misaligned ops (kernel dominant ≠ runtime dominant):";
      mTitle.style.marginTop = "6px";
      const mBars = mk("div", "haar-ext-bars");
      for (const op of misaligned.slice(0, 6)) {
        const row = mk("div", "haar-ext-bar-row");
        const lbl = mk("span", "haar-ext-bar-label");
        lbl.textContent = `#${op.op_index} ${op.op_name}`;
        const val = mk("span", "haar-ext-bar-val");
        val.textContent = `kernel=${op.kernel_dominant.toUpperCase()} ≠ runtime=${op.runtime_dominant.toUpperCase()}`;
        row.append(lbl, val); mBars.append(row);
      }
      aSec.append(mTitle, mBars);
    }
    aWrap.append(aSec);
    haarSweepPanel.append(aWrap);
  }
}

function renderPerturbationProtocols(drift, baseline, perturbed, weightProbe = null, layerRobustness = [], haarSweep = null) {
  const profile = outputDriftProfileForAnalysis(current);
  renderProtocolGroups(perturbationProtocols, perturbationProtocolGroups({ drift, baseline, perturbed, weightProbe, layerRobustness, haarSweep, profile }));
}

function renderDeploymentSensitivityProtocols(basin = null, curvature = null) {
  renderProtocolGroups(deploymentSensitivityProtocols, deploymentSensitivityProtocolGroups({ basin, curvature }));
}

function resetResearchModulePanels() {
  resetAdvancedResultState({
    includeDeepBom: false,
    includeRuntimeBenchmarks: false,
    includeBundleProgress: false,
    includeCalibrationValidation: false,
  });
  for (const panel of [perturbationResultPanel, runtimeBasinResultPanel, deploymentSensitivityResultPanel, haarSweepPanel, lossLandscapePanel, modelViewerPanel, lossTomoPanel]) {
    if (panel) panel.hidden = true;
  }
  for (const grid of [perturbationGrid, runtimeBasinGrid, deploymentSensitivityGrid]) {
    grid?.replaceChildren();
  }
  for (const protocols of [perturbationProtocols, runtimeBasinProtocols, deploymentSensitivityProtocols]) {
    protocols?.replaceChildren();
  }
  if (perturbationStatus) perturbationStatus.textContent = "Not run";
  if (runtimeBasinStatus) runtimeBasinStatus.textContent = "Not run";
  if (deploymentSensitivityStatus) deploymentSensitivityStatus.textContent = "Not run";
  if (perturbationNotes) perturbationNotes.textContent = "";
  if (runtimeBasinNotes) runtimeBasinNotes.textContent = "";
  if (deploymentSensitivityNotes) deploymentSensitivityNotes.textContent = "";
}

function modelIdentity() {
  return buildModelIdentity({
    analysis: currentAnalysisView(),
    filename: currentFilename,
    modelBytes: currentModelBytes,
    selectedTargetProfile: selectedTargetProfile(),
    selectedTargetId: selectedTargetId(),
    selectedTargetLabel: selectedTargetLabel(),
  });
}

async function runWeightPerturbationProbe(baseline, candidates, strategy) {
  const mutation = perturbModelWeightBytes(currentModelBytes, current, candidates, { direction: 1, amplitude: 1 });
  if (!mutation.touchedValues) return null;
  const probe = await runTfliteOutputProbe("wasm", "baseline", { modelBytes: mutation.bytes });
  const drift = compareOutputArrays(baseline.outputs, probe.outputs);
  return {
    strategy,
    probe,
    drift,
    touchedTensors: mutation.touchedTensors,
    touchedValues: mutation.touchedValues,
    candidateOps: candidates.map(({ op, tensorIds }) => ({
      index: op.index,
      name: op.name,
      tensor_ids: tensorIds,
    })),
  };
}

async function runHaarPatternSweep(baselineOutputs) {
  await ensureLiteRtRuntime("wasm");
  const { Tensor, loadAndCompile } = await liteRtRuntime.loadCore();
  let model = null;
  const patterns = [];
  const extended = { translation: null, rotation: null, phase: null, amplitude: null, polarity: null, channels: null };
  try {
    model = await loadAndCompile(currentModelBytes, { accelerator: "wasm" });
    const inputDetailsList = model.getInputDetails();
    const primary = inputDetailsList[0];
    if (!primary) return { patterns, extended };

    const dtype   = primary.dtype;
    const staticT = current?.inputs?.[0];
    const zp      = normalizedZeroPointForDtype(staticT, dtype);
    const profile = outputDriftProfileForAnalysis(current);

    // Infer channel count from input shape (for channel sweep gating)
    const primaryShape = resolveFakeInputShape(Array.from(primary.shape), staticT);
    const channelCount = primaryShape.length === 4
      ? (primaryShape[3] <= 4 && primaryShape[1] > 4 ? primaryShape[3] : primaryShape[1] <= 4 ? primaryShape[1] : 1)
      : 1;

    async function runOne(patternId, opts, extraMeta) {
      const inputs = [];
      let patternOutputs = null;
      try {
        let inputStats = null;
        for (const [idx, details] of inputDetailsList.entries()) {
          const s = resolveFakeInputShape(Array.from(details.shape), current?.inputs?.[idx]);
          const data = idx === 0
            ? createHaarPatternData(details.dtype, s, patternId, current?.inputs?.[0], opts || {})
            : createResearchInputData(details.dtype, s.reduce((a, d) => a * d, 1), current?.inputs?.[idx]);
          if (idx === 0) inputStats = computePatternInputStats(data, details.dtype, zp);
          inputs.push(Tensor.fromTypedArray(data, s));
        }
        patternOutputs = await model.run(inputs);
        const outputArrays = await collectTensorArrays(patternOutputs);
        const drift = sanitizeDrift(compareOutputArrays(baselineOutputs, outputArrays), profile);
        return { drift, inputStats, ...extraMeta };
      } catch (e) {
        return { error: shortError(e), ...extraMeta };
      } finally {
        deleteTensors(patternOutputs);
        deleteTensors(inputs);
      }
    }

    // 1. Base pattern sweep (all HAAR_PATTERN_SPECS including fractal and COCO prior)
    for (const pattern of HAAR_PATTERN_SPECS) {
      const result = await runOne(pattern.id, {}, {});
      // Attach response_gain = output_rms / input_l2 for alignment analysis
      if (result.inputStats?.l2 > 0 && result.drift?.rms != null) {
        result.response_gain = result.drift.rms / result.inputStats.l2;
      }
      patterns.push({ patternId: pattern.id, ...pattern, ...result });
    }

    // 2. Translation sweep — localized 3×3 patch positions for haar_HL_8
    {
      const gridN = HAAR_SWEEP_CONFIG.translationGridN;
      const transResults = [];
      for (let gr = 0; gr < gridN; gr++)
        for (let gc = 0; gc < gridN; gc++) {
          const r = await runOne(HAAR_SWEEP_CONFIG.translationPatternId, { patchGrid: { gridRow: gr, gridCol: gc, gridN } }, { gridRow: gr, gridCol: gc });
          transResults.push(r);
        }
      extended.translation = {
        pattern_id: HAAR_SWEEP_CONFIG.translationPatternId,
        results:    transResults,
        profile:    haarTranslationProfile(transResults),
      };
    }

    // 3. Rotation sweep — 0/90/180/270° for haar_LH_8 and haar_HL_8
    {
      const rotResults = [];
      for (const patId of HAAR_SWEEP_CONFIG.rotationPatternIds)
        for (const rot of [0, 90, 180, 270]) {
          const r = await runOne(patId, { rotation: rot }, { pattern_id: patId, rotation: rot });
          rotResults.push(r);
        }
      extended.rotation = {
        pattern_ids: HAAR_SWEEP_CONFIG.rotationPatternIds,
        results:     rotResults,
        by_pattern:  Object.fromEntries(
          HAAR_SWEEP_CONFIG.rotationPatternIds.map((id) => [
            id, haarRotationProfile(rotResults.filter((r) => r.pattern_id === id)),
          ])
        ),
      };
    }

    // 4. Phase sweep — offsets [0,1,2,4] for haar_HH_4 and haar_HH_8
    {
      const phaseResults = [];
      for (const patId of HAAR_SWEEP_CONFIG.phasePatternIds)
        for (const off of HAAR_SWEEP_CONFIG.phaseOffsets) {
          const r = await runOne(patId, { phaseX: off, phaseY: off }, { pattern_id: patId, phaseOffset: off });
          phaseResults.push(r);
        }
      extended.phase = {
        pattern_ids: HAAR_SWEEP_CONFIG.phasePatternIds,
        results:     phaseResults,
        by_pattern:  Object.fromEntries(
          HAAR_SWEEP_CONFIG.phasePatternIds.map((id) => [
            id, haarPhaseProfile(phaseResults.filter((r) => r.pattern_id === id)),
          ])
        ),
      };
    }

    // 5. Amplitude sweep — dtype-specific levels for haar_HH_4
    {
      const ampLevels = haarAmplitudeSweepLevels(dtype);
      const ampResults = [];
      for (const amp of ampLevels) {
        const r = await runOne(HAAR_SWEEP_CONFIG.amplitudePatternId, { amplitude: amp }, { amplitude: amp });
        ampResults.push(r);
      }
      extended.amplitude = {
        pattern_id: HAAR_SWEEP_CONFIG.amplitudePatternId,
        results:    ampResults,
        profile:    haarAmplitudeSweepProfile(ampResults),
      };
    }

    // 6. Polarity sweep — +1 / -1 for three key patterns
    {
      const polarityPairs = [];
      for (const patId of HAAR_SWEEP_CONFIG.polarityPatternIds) {
        const plus  = await runOne(patId, { polarity:  1 }, {});
        const minus = await runOne(patId, { polarity: -1 }, {});
        polarityPairs.push({ patternId: patId, plus, minus });
      }
      extended.polarity = {
        pattern_ids: HAAR_SWEEP_CONFIG.polarityPatternIds,
        pairs:       polarityPairs,
        profile:     haarPolarityProfile(polarityPairs),
      };
    }

    // 7. Channel sweep — all/R/G/B for haar_HL_8 (only if input has ≥3 channels)
    if (channelCount >= 3) {
      const chanResults = [];
      for (const mode of HAAR_SWEEP_CONFIG.channelModes) {
        const r = await runOne(HAAR_SWEEP_CONFIG.channelPatternId, { channelMode: mode }, { channelMode: mode });
        chanResults.push(r);
      }
      extended.channels = {
        pattern_id: HAAR_SWEEP_CONFIG.channelPatternId,
        results:    chanResults,
      };
    }

  } finally {
    model?.delete?.();
  }

  // Kernel Haar decomposition (static weight analysis)
  let kernelHaar = null;
  let alignment  = null;
  let activationHaar = null;
  try {
    kernelHaar = compute_kernel_haar_decomposition(currentModelBytes, currentFilename || "model.tflite", selectedTargetId());
    if (kernelHaar && patterns.length > 0) {
      const haarProfile = haarSensitivityProfile(patterns);
      alignment = computeStaticRuntimeAlignment(kernelHaar, patterns, haarProfile);
    }
  } catch (_) {
    // Non-fatal: skip if model has no conv ops or decomposition fails
  }
  try {
    // Activation Haar: synthetic forward pass in WASM (32×32 input, f64 precision)
    const rawActHaar = compute_activation_haar(currentModelBytes, currentFilename || "model.tflite", selectedTargetId());
    if (Array.isArray(rawActHaar) && rawActHaar.length > 0) {
      const convOps = rawActHaar.filter(r => !r.skipped && !r.spatial_too_small);
      const allBands = ["LL", "LH", "HL", "HH"];
      const energySums = Object.fromEntries(allBands.map(b => [b, 0]));
      convOps.forEach(r => {
        energySums.LL += r.ll_energy;
        energySums.LH += r.lh_energy;
        energySums.HL += r.hl_energy;
        energySums.HH += r.hh_energy;
      });
      const total = energySums.LL + energySums.LH + energySums.HL + energySums.HH;
      const dominant = total > 0 ? allBands.reduce((a, b) => energySums[a] > energySums[b] ? a : b) : "LL";
      activationHaar = {
        ops: rawActHaar,
        summary: {
          conv_op_count: rawActHaar.length,
          executed_count: convOps.length,
          skipped_count: rawActHaar.filter(r => r.skipped).length,
          spatial_too_small_count: rawActHaar.filter(r => r.spatial_too_small).length,
          dominant_band: dominant,
          mean_spatial_smoothness: convOps.length > 0
            ? convOps.reduce((s, r) => s + r.spatial_smoothness, 0) / convOps.length : 0,
          ll_mean: total > 0 ? energySums.LL / total : 0,
          lh_mean: total > 0 ? energySums.LH / total : 0,
          hl_mean: total > 0 ? energySums.HL / total : 0,
          hh_mean: total > 0 ? energySums.HH / total : 0,
          input_size: "32×32 synthetic",
          method: "wasm_forward_pass_f64",
        },
      };
    }
  } catch (_) {
    // Non-fatal: skip if forward pass fails
  }

  return { patterns, extended, kernelHaar, alignment, activationHaar };
}

async function runLayerRobustnessSweep(baseline, candidates, profile) {
  const rows = [];
  for (const candidate of candidates) {
    const mutation = perturbModelWeightBytes(currentModelBytes, current, [candidate], { direction: 1, amplitude: 1 });
    if (!mutation.touchedValues) continue;
    try {
      const probe = await runTfliteOutputProbe("wasm", "baseline", { modelBytes: mutation.bytes });
      const drift = compareOutputArrays(baseline.outputs, probe.outputs);
      rows.push({
        op_index: candidate.op.index,
        op: `#${padOp(candidate.op.index)} ${candidate.op.name}`,
        tensor_ids: candidate.tensorIds,
        touched_values: mutation.touchedValues,
        drift: sanitizeDrift(drift, profile),
        robustness_score: robustnessScoreFromDrift(drift, profile),
      });
    } catch (error) {
      rows.push({
        op_index: candidate.op.index,
        op: `#${padOp(candidate.op.index)} ${candidate.op.name}`,
        tensor_ids: candidate.tensorIds,
        touched_values: mutation.touchedValues,
        status: "failed",
        error: shortError(error),
      });
    }
  }
  return rows;
}

async function runPerturbationAnalysis() {
  perturbationResultPanel.hidden = false;
  perturbationStatus.textContent = "Running…";
  perturbationNotes.textContent = "Running perturbation locally — this may take a few seconds.";
  perturbationGrid.querySelectorAll(".deepbom-metric.skeleton").forEach(c => c.classList.add("running"));
  perturbationPanelAction.disabled = true;
  setStatus("Running Perturbation");
  await nextPaint();
  try {
    if (!modelSupportsCapability(current?.format, "experimental_tflite_research")) {
      perturbationResult = {
        schema: "deepbom.perturbation_analysis.v1.1",
        generated_at: new Date().toISOString(),
        model: modelIdentity(),
        status: "blocked",
        reason: "tflite_runtime_path_required",
        executed: [],
        not_executed: ["input_perturbation", "output_drift", "weight_perturbation", "layer_wise_robustness"],
        scope: {
          executed: [],
          not_executed: ["input_perturbation", "output_drift", "weight_perturbation", "layer_wise_robustness"],
          protocols: perturbationProtocolStatus(false, false, false),
        },
      };
      perturbationStatus.textContent = "TFLite path required";
      perturbationGrid.replaceChildren(
        deepBomMetric("Runtime path", "TFLite required", "Perturbation output probes currently run through LiteRT.js and therefore require a TFLite artifact.", statusBlocked("ONNX output probes require an ONNX Runtime Web implementation for controlled input perturbation.")),
        deepBomMetric("Model format", modelFormatAdapter(current?.format).label, "No TFLite perturbation path is declared for this artifact format.", statusInfo()),
      );
      renderPerturbationProtocols(null, null, null);
      perturbationNotes.textContent = "No model bytes were uploaded. This module stopped before execution because the loaded artifact is not a LiteRT/TFLite runtime path.";
      return perturbationResult;
    }
    const baseline = await runTfliteOutputProbe("wasm", "baseline");
    const perturbed = await runTfliteOutputProbe("wasm", "perturb");
    const drift = compareOutputArrays(baseline.outputs, perturbed.outputs);
    const profile = outputDriftProfileForAnalysis(current);
    const weightCandidates = selectWeightPerturbationCandidates(current, 12);
    const weightProbe = weightCandidates.length
      ? await runWeightPerturbationProbe(baseline, weightCandidates, "global_weight_plus_lsb")
      : null;
    const layerRobustness = weightCandidates.length
      ? await runLayerRobustnessSweep(baseline, weightCandidates.slice(0, 4), profile)
      : [];
    // Ensure SHA-256 is computed before perturbationResult is assembled (lazy cache)
    if (!current.model_sha256) current.model_sha256 = await sha256Hex(currentModelBytes);
    const { patterns: haarPatterns, extended: haarExtended, kernelHaar, alignment: haarAlignment, activationHaar } = await runHaarPatternSweep(baseline.outputs);
    const haarProfile = haarSensitivityProfile(haarPatterns);
    const haarMostSensitive = haarPatterns.filter((r) => !r.error).reduce((a, b) => (Number(b.drift?.rms || 0) > Number(a.drift?.rms || 0) ? b : a), haarPatterns[0] || {});
    const worstLayer = maxBy(layerRobustness, (item) => severityRank(item.drift?.severity) * 1e9 + Number(item.drift?.max_abs || 0));
    let lossLandscape = null;
    try {
      lossLandscape = await runLossLandscape(baseline);
    } catch (_) { /* non-fatal */ }
    let lossTomo = null;
    try {
      lossTomo = await runLandscapeTomography();
    } catch (_) { /* non-fatal */ }
    const variance = timingVarianceSummary(baseline.runMs, perturbed.runMs);
    perturbationResult = {
      schema: "deepbom.perturbation_analysis.v1.1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "complete",
      scope: {
        executed: [
          "input_perturbation",
          "output_drift",
          ...(weightProbe ? ["weight_perturbation"] : []),
          ...(layerRobustness.length ? ["layer_wise_robustness"] : []),
          ...(haarPatterns.some((r) => !r.error) ? ["haar_pattern_sweep"] : []),
          ...(kernelHaar?.ops?.length ? ["kernel_haar_decomposition"] : []),
          ...(activationHaar?.summary?.executed_count > 0 ? ["activation_haar_decomposition"] : []),
          ...(lossLandscape ? ["output_drift_geometry_projection"] : []),
          ...(lossTomo ? ["multi_projection_output_drift_geometry"] : []),
        ],
        not_executed: [
          ...(!weightProbe ? ["weight_perturbation"] : []),
          ...(!layerRobustness.length ? ["layer_wise_robustness"] : []),
          ...(!haarPatterns.some((r) => !r.error) ? ["haar_pattern_sweep"] : []),
          ...(!kernelHaar?.ops?.length ? ["kernel_haar_decomposition"] : []),
          ...(!activationHaar?.summary?.executed_count ? ["activation_haar_decomposition"] : []),
          ...(!lossLandscape ? ["output_drift_geometry_projection"] : []),
          ...(!lossTomo ? ["multi_projection_output_drift_geometry"] : []),
        ],
        protocols: perturbationProtocolStatus(true, Boolean(weightProbe), Boolean(layerRobustness.length), haarPatterns.some((r) => !r.error)),
      },
      backend: "wasm",
      output_profile: profile,
      timing_context: timingContextNote(),
      input_source: "synthetic_tensor",
      baseline: summarizeProbeResult(baseline),
      perturbed: summarizeProbeResult(perturbed),
      drift: sanitizeDrift(drift, profile),
      weight_perturbation: weightProbe
        ? {
            strategy: weightProbe.strategy,
            touched_tensors: weightProbe.touchedTensors,
            touched_values: weightProbe.touchedValues,
            candidate_ops: weightProbe.candidateOps,
            probe: summarizeProbeResult(weightProbe.probe),
            drift: sanitizeDrift(weightProbe.drift, profile),
          }
        : {
            status: "not_available",
            reason: "No constant weight buffers with supported dtype/offset metadata were detected.",
          },
      layer_wise_robustness: layerRobustness,
      haar_pattern_sweep: {
        patterns_tested: haarPatterns.filter((r) => !r.error).length,
        results: haarPatterns,
        profile: haarProfile,
      },
      timing_variance: sanitizeTimingVariance(variance),
      kernel_haar_decomposition: kernelHaar ?? null,
      activation_haar_decomposition: activationHaar ?? null,
      output_drift_geometry_projection: lossLandscape ?? null,
      multi_projection_output_drift_geometry: lossTomo ? {
        evidenceClass: lossTomo.protocol?.evidenceClass || "MEASURED_SYNTHETIC_PROXY",
        numProjections: lossTomo.numProjections,
        hessianAssessedCount: lossTomo.hessianAssessedCount,
        directionalLambdaMaxCv: lossTomo.directionalLambdaMaxCv,
        lambdaMean: lossTomo.lambdaMean,
        lambdaStd: lossTomo.lambdaStd,
        protocol: lossTomo.protocol,
      } : null,
      static_runtime_alignment: haarAlignment ?? null,
      anomalies: Array.isArray(current?.anomalies) ? current.anomalies : [],
      xnnpack_chains: Array.isArray(current?.xnnpack_chains) ? current.xnnpack_chains : [],
      privacy: "local-only; model bytes and outputs were not uploaded",
    };
    perturbationStatus.textContent = drift.top1Flip ? "Rank changed" : "Complete";
    perturbationGrid.replaceChildren(
      deepBomMetric("Backend", "WASM", `Compile ${baseline.compileMs.toFixed(2)} ms / baseline run ${baseline.runMs.toFixed(2)} ms / perturbed run ${perturbed.runMs.toFixed(2)} ms.`, statusInfo("Single-run timing only; use Runtime Benchmark p50/p90/p95 for latency decisions.")),
      deepBomMetric("Run Variance", variance.label, variance.detail, variance.status),
      deepBomMetric("RMS Drift", `${formatDrift(drift.rms)} ${profile.unit}`, "Root-mean-square output change after local input perturbation.", statusForRmsDrift(drift.rms, profile, drift.leftRms)),
      deepBomMetric("Mean Abs Drift", `${formatDrift(drift.meanAbs)} ${profile.unit}`, "Average absolute output change across all returned output tensors.", statusForRmsDrift(drift.meanAbs, profile, drift.leftRms)),
      deepBomMetric("Max Abs Drift", `${formatDrift(drift.maxAbs)} ${profile.unit}`, "Largest absolute output change observed in the returned tensors.", statusForMaxDrift(drift.maxAbs, profile, drift.leftRms)),
      deepBomMetric("Cosine Distance", formatDrift(drift.cosineDistance), "1 - cosine similarity across flattened outputs.", statusForCosineDistance(drift.cosineDistance)),
      deepBomMetric("Top-1 Flip", drift.top1Flip ? "Yes" : "No", "Whether argmax of the first output tensor changed under perturbation.", statusForTop1Flip(drift.top1Flip)),
      deepBomMetric("Weight Perturbation", weightProbe ? `${formatDrift(weightProbe.drift.maxAbs)} ${profile.unit}` : "N/A", weightProbe ? `${weightProbe.touchedTensors} tensor(s), ${formatNumber(weightProbe.touchedValues)} value(s) touched in a local model-byte copy.` : "No supported constant weight buffer was detected for local mutation.", weightProbe ? driftSeverity(weightProbe.drift, profile) : statusInfo("Unavailable when the deployment artifact does not expose supported constant buffers.")),
      deepBomMetric("Layer Robustness", worstLayer ? `${worstLayer.op} / ${worstLayer.drift.severity}` : "N/A", worstLayer ? `Worst sampled layer max drift ${formatDrift(worstLayer.drift.max_abs)} ${profile.unit}; ${layerRobustness.length} layer(s) swept.` : "No layer-wise weight sweep was executable for this artifact.", worstLayer ? statusFromSeverityLabel(worstLayer.drift.severity) : statusInfo("Layer sweep requires supported constant weight buffers.")),
      deepBomMetric("Haar Sweep", haarMostSensitive.label ? `${haarMostSensitive.label} / ${formatDrift(haarMostSensitive.drift?.rms)} ${profile.unit}` : "N/A", `${haarPatterns.filter((r) => !r.error).length}/${HAAR_PATTERN_SPECS.length} patterns (legacy bank + multi-scale LH/HL/HH × 5 scales) vs zero-fill baseline. Most sensitive: ${haarMostSensitive.label || "—"}. Extended: translation 3×3, rotation, phase sweep, amplitude sweep, polarity pairs.`, statusInfo("Spatial frequency and orientation sensitivity profile. Includes full multi-scale Haar wavelet bank (2–32px), translation stability, rotation invariance, phase jitter, and polarity asymmetry probes.")),
      deepBomMetric("Freq Profile", haarProfile.frequency_trend || "N/A", haarProfile.freq_ratio_fine_coarse !== null ? `2px/16px ratio: ${haarProfile.freq_ratio_fine_coarse}×. Peak: ${haarProfile.peak_checker_label || "2px"} checker. ${haarProfile.orientation_bias || ""}.` : "Frequency ratio unavailable.", statusInfo("high_freq_sensitive/mid_high_freq_peak: fine patterns dominate. broadband: flat. mid_low_freq_peak/low_freq_sensitive: coarse patterns dominate. Peak checker shows actual maximum-response scale.")),
      deepBomMetric("Position Sensitivity", haarProfile.position_sensitivity || "N/A", "Compares diagonal Haar pattern vs vertically/horizontally stride-shifted variants to detect translation-invariance.", statusInfo("translation_invariant is expected for well-regularized CNNs; position_sensitive indicates sensitivity to input position (architecture or input-distribution effect — not a definitive training inference).")),
    );
    perturbationGrid.querySelectorAll(".deepbom-metric").forEach((card, i) => {
      card.classList.add("filling");
      card.style.animationDelay = `${i * 35}ms`;
      card.addEventListener("animationend", () => { card.classList.remove("filling"); card.style.animationDelay = ""; }, { once: true });
    });
    renderHaarSweepPanel(haarPatterns, haarProfile, profile, haarExtended, kernelHaar, haarAlignment, activationHaar);
    renderLossLandscapePanel(lossLandscape);
    renderLossTomoPanel(lossTomo);
    renderPerturbationProtocols(drift, baseline, perturbed, weightProbe, layerRobustness, haarPatterns);
    perturbationNotes.textContent = `Executed scope: input perturbation, output drift${weightProbe ? ", weight perturbation" : ""}${layerRobustness.length ? ", layer-wise robustness sweep" : ""}, Haar pattern sweep (${haarPatterns.filter((r) => !r.error).length}/${HAAR_PATTERN_SPECS.length} base patterns + translation/rotation/phase/amplitude/polarity extended probes). Perturbations run on local tensors or local model-byte copies only. Compared ${formatNumber(drift.count)} output values across ${baseline.outputs.length} output tensor(s). Timing values here are single local runs and can invert by normal browser/runtime variance; use the Runtime Benchmark table for latency claims.`;
    updateWorkflowState("pro");
    setStatus("Perturbation complete", "ok");
    return perturbationResult;
  } catch (error) {
    console.error(error);
    perturbationResult = {
      schema: "deepbom.perturbation_analysis.v1.1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "failed",
      error: shortError(error),
    };
    perturbationStatus.textContent = "Failed";
    perturbationGrid.replaceChildren(deepBomMetric("Execution", "Failed", shortError(error), { tone: "risk", label: "risk", criteria: "Execution must complete before drift status can be interpreted." }));
    renderPerturbationProtocols(null, null, null);
    perturbationNotes.textContent = "The module attempted local LiteRT.js execution and failed before producing drift evidence.";
    setStatus("Perturbation failed", "error");
    return perturbationResult;
  } finally {
    perturbationPanelAction.disabled = !current;
    updateModuleAccessState();
  }
}

async function runRuntimeBasinValidation() {
  runtimeBasinResultPanel.hidden = false;
  runtimeBasinStatus.textContent = "Running…";
  runtimeBasinNotes.textContent = "Probing backend availability and cross-backend output drift locally.";
  runtimeBasinGrid.querySelectorAll(".deepbom-metric.skeleton").forEach(c => c.classList.add("running"));
  setStatus("Running Backend Consistency");
  await nextPaint();
  try {
    if (!modelSupportsCapability(current?.format, "experimental_tflite_research")) {
      runtimeBasinResult = {
        schema: "deepbom.runtime_basin.v1",
        generated_at: new Date().toISOString(),
        model: modelIdentity(),
        status: "blocked",
        reason: "tflite_runtime_path_required",
        executed: [],
        not_executed: ["backend_availability", "backend_output_drift", "preprocessing_drift"],
        scope: {
          executed: [],
          not_executed: ["backend_availability", "backend_output_drift", "preprocessing_drift"],
          protocols: runtimeBasinProtocolStatus().map((item) => item.status === "executed" ? { ...item, status: "not_run" } : item),
        },
      };
      runtimeBasinStatus.textContent = "TFLite path required";
      runtimeBasinGrid.replaceChildren(
        deepBomMetric("Runtime path", "TFLite required", "Backend Consistency output probes currently run through LiteRT.js and therefore require a TFLite artifact.", statusBlocked("ONNX backend consistency requires ONNX Runtime Web output probes for multiple execution providers.")),
        deepBomMetric("Model format", modelFormatAdapter(current?.format).label, modelSupportsCapability(current?.format, "runtime_execution") ? "A format-specific runtime benchmark remains available in the Runtime tab." : "Static artifact, payload, and applicable numerical analysis remain available; this format has no local browser runtime in the current build.", statusInfo()),
      );
      runtimeBasinProtocols.replaceChildren();
      runtimeBasinNotes.textContent = "No model bytes were uploaded. This module stopped before execution because the loaded artifact is not a LiteRT/TFLite runtime path.";
      return runtimeBasinResult;
    }
    const candidates = [...new Set(["wasm", ...backendCandidates("auto", current?.format, navigator)])];
    const results = [];
    for (const backend of candidates) {
      try {
        const result = await runTfliteOutputProbe(backend, "baseline");
        results.push({ backend, ok: true, result });
      } catch (error) {
        results.push({ backend, ok: false, error: shortError(error) });
      }
    }
    const successful = results.filter((item) => item.ok);
    const reference = successful[0];
    let maxDrift = null;
    if (reference) {
      for (const item of successful.slice(1)) {
        const drift = compareOutputArrays(reference.result.outputs, item.result.outputs);
        item.drift = drift;
        if (!maxDrift || drift.maxAbs > maxDrift.maxAbs) maxDrift = drift;
      }
    }
    const avgRunMs = successful.length
      ? successful.reduce((acc, item) => acc + item.result.runMs, 0) / successful.length
      : 0;
    const profile = outputDriftProfileForAnalysis(current);
    const runtimeReference = successful.find((item) => item.backend === "wasm") || reference || null;
    const backendInterpretations = results
      .map((item) => runtimeAttemptInterpretation(item, runtimeReference, current))
      .map((interpretation, index) => ({ backend: results[index].backend, ...interpretation }))
      .filter((item) => item.note);
    const topBackendCaveat = backendInterpretations.find((item) => item.severity === "warn") || backendInterpretations[0] || null;
    runtimeBasinResult = {
      schema: "deepbom.runtime_basin.v1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: successful.length ? "complete" : "failed",
      error: successful.length ? null : "no_browser_runtime_backend_completed",
      scope: {
        executed: ["backend_availability", "backend_output_drift"],
        not_executed: ["preprocessing_drift"],
        protocols: runtimeBasinProtocolStatus(false),
      },
      output_profile: profile,
      attempted_backends: results.map((item) => sanitizeRuntimeAttempt(item, profile, runtimeReference, current)),
      backend_interpretations: backendInterpretations,
      failures: results
        .filter((item) => !item.ok)
        .map((item) => ({ backend: item.backend, error: item.error })),
      successful_count: successful.length,
      attempted_count: results.length,
      reference_backend: reference?.backend || null,
      max_drift: maxDrift ? sanitizeDrift(maxDrift, profile) : null,
      preprocessing_drift: {
        status: "not_available",
        reason: "No prepared image tensor is active.",
      },
      mean_run_ms: avgRunMs,
      privacy: "local-only; model bytes and outputs were not uploaded",
    };
    runtimeBasinStatus.textContent = successful.length ? "Complete" : "No backend completed";
    runtimeBasinGrid.replaceChildren(
      deepBomMetric("Backends OK", `${successful.length}/${results.length}`, results.map((item) => `${item.backend}:${item.ok ? "ok" : "fail"}`).join(" / "), statusForBackendCoverage(successful.length, results.length)),
      deepBomMetric("Reference", reference?.backend || "-", reference ? "First successful backend used as the drift reference." : "No backend produced outputs.", statusInfo("Reference is chosen from the first successful local backend path.")),
      deepBomMetric("Max Backend Drift", maxDrift ? `${formatDrift(maxDrift.maxAbs)} ${profile.unit}` : "N/A", maxDrift ? `Largest absolute drift against the reference backend. Output dtype basis: ${profile.dtype}.` : "Only one or zero backend paths completed.", maxDrift ? statusForMaxDrift(maxDrift.maxAbs, profile) : statusInfo("Backend drift requires at least two successful backend paths.")),
      deepBomMetric("Mean Run", successful.length ? `${avgRunMs.toFixed(2)} ms` : "N/A", "Average single local run time across successful browser backends.", statusInfo("Single-run timing; use Runtime Benchmark p50/p90/p95/p99 for latency claims.")),
      deepBomMetric("Failures", formatNumber(results.length - successful.length), results.filter((item) => !item.ok).map((item) => `${item.backend}: ${item.error}`).join(" / ") || "No backend failure.", results.length === successful.length ? { tone: "good", label: "ok", criteria: "ok = no attempted backend failed; warn/risk when failures reduce coverage." } : { tone: "warn", label: "warn", criteria: "Backend failures reduce runtime-basin coverage; inspect browser support and model dtype." }),
      deepBomMetric("Backend Caveat", topBackendCaveat ? `${topBackendCaveat.backend} / ${topBackendCaveat.severity}` : "None", topBackendCaveat?.note || "No backend-specific caveat was generated.", topBackendCaveat?.severity === "warn" ? { tone: "warn", label: "warn", criteria: "warn when browser accelerator path is much slower than the WASM/reference path for the current dtype contract." } : statusInfo("Backend-specific interpretation note.")),
      deepBomMetric("Privacy", "Local only", "Model bytes, generated tensors, and outputs stayed in browser memory.", { tone: "good", label: "ok", criteria: "No model, input, output, or timing report upload is required for this local run." }),
    );
    runtimeBasinGrid.querySelectorAll(".deepbom-metric").forEach((card, i) => {
      card.classList.add("filling");
      card.style.animationDelay = `${i * 35}ms`;
      card.addEventListener("animationend", () => { card.classList.remove("filling"); card.style.animationDelay = ""; }, { once: true });
    });
    runtimeBasinProtocols.replaceChildren();
    runtimeBasinNotes.textContent = successful.length
      ? `Executed ${successful.length} of ${results.length} attempted local LiteRT backend path(s). ${maxDrift ? `Largest backend max-abs drift was ${formatDrift(maxDrift.maxAbs)} ${profile.unit}.` : "Only one backend completed, so backend drift could not be compared."} ${topBackendCaveat?.severity === "warn" ? topBackendCaveat.note : ""} WebNN/WebGPU availability is browser-dependent and is reported through attempted backend success/failure, not assumed.`
      : "All local backend paths failed before returning outputs. Use the Runtime tab error messages and browser backend support to diagnose the path.";
    updateWorkflowState("pro");
    setStatus("Backend Consistency complete", successful.length ? "ok" : "error");
    return runtimeBasinResult;
  } catch (error) {
    console.error(error);
    runtimeBasinResult = {
      schema: "deepbom.runtime_basin.v1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "failed",
      error: shortError(error),
    };
    runtimeBasinStatus.textContent = "Failed";
    runtimeBasinGrid.replaceChildren(deepBomMetric("Execution", "Failed", shortError(error), { tone: "risk", label: "risk", criteria: "Execution must complete before runtime-basin status can be interpreted." }));
    runtimeBasinProtocols.replaceChildren();
    runtimeBasinNotes.textContent = "The module attempted local backend execution and failed before producing runtime-basin evidence.";
    setStatus("Backend Consistency failed", "error");
    return runtimeBasinResult;
  } finally {
    updateModuleAccessState();
  }
}

async function runDeployCurvatureBasinAnalysis() {
  deploymentSensitivityResultPanel.hidden = false;
  deploymentSensitivityGrid.replaceChildren();
  deploymentSensitivityNotes.textContent = "";
  deploymentSensitivityStatus.textContent = "Running locally";
  deploymentSensitivityPanelAction.disabled = true;
  setStatus("Running curvature");
  await nextPaint();
  try {
    if (!modelSupportsCapability(current?.format, "experimental_tflite_research")) {
      deployCurvatureResult = {
        schema: "deepbom.deploy_curvature_basin.v1",
        generated_at: new Date().toISOString(),
        model: modelIdentity(),
        status: "blocked",
        reason: "tflite_runtime_path_required",
        deploy_curvature_available: false,
        training_curvature_evidence: {
          status: "blocked",
          required: ["checkpoint_or_training_artifact", "loss_function", "representative_data", "gradient_capable_runtime"],
        },
      };
      deploymentSensitivityStatus.textContent = "TFLite path required";
      deploymentSensitivityGrid.replaceChildren(
        deepBomMetric("Runtime path", "TFLite required", "Deployment Sensitivity observations currently run through LiteRT.js and therefore require a TFLite artifact.", statusBlocked("ONNX deployment-sensitivity analysis requires ONNX Runtime output probes with finite-difference input control.")),
        deepBomMetric("Model format", modelFormatAdapter(current?.format).label, "No TFLite finite-difference path is declared for this artifact format.", statusInfo()),
      );
      renderDeploymentSensitivityProtocols(null, null);
      deploymentSensitivityNotes.textContent = "No model bytes were uploaded. The module stopped before execution because the loaded artifact is not a LiteRT/TFLite runtime path.";
      return deployCurvatureResult;
    }
    const baseline = await runTfliteOutputProbe("wasm", "baseline");
    const plus = await runTfliteOutputProbe("wasm", "curve_plus");
    const minus = await runTfliteOutputProbe("wasm", "curve_minus");
    const wide = await runTfliteOutputProbe("wasm", "curve_wide");
    const plusDrift = compareOutputArrays(baseline.outputs, plus.outputs);
    const minusDrift = compareOutputArrays(baseline.outputs, minus.outputs);
    const wideDrift = compareOutputArrays(baseline.outputs, wide.outputs);
    const curvature = computeDirectionalCurvature(baseline.outputs, plus.outputs, minus.outputs, plus.inputPerturbation);
    const margin = decisionMargin(baseline.outputs[0]);
    const basin = computeDeployBasinProxy(plusDrift, minusDrift, wideDrift, curvature, margin);
    const profile = outputDriftProfileForAnalysis(current);
    const wideMinusIdentical = driftLooksIdentical(wideDrift, minusDrift);
    const widePlusIdentical = driftLooksIdentical(wideDrift, plusDrift);
    const widePlusDelta = driftDeltaSummary(wideDrift, plusDrift);
    const consistencyWarning = deployProbeConsistencyWarning(widePlusIdentical, wideMinusIdentical);
    deployCurvatureResult = {
      schema: "deepbom.deploy_curvature_basin.v1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "complete",
      scope: {
        executed: ["deploy_finite_difference_curvature", "deploy_basin_proxy"],
        blocked_training_evidence: ["hessian_trace", "sharpness_proxy", "pac_bayes_flatness"],
      },
      backend: "wasm",
      output_profile: profile,
      timing_context: timingContextNote(),
      baseline: summarizeProbeResult(baseline),
      plus: summarizeProbeResult(plus),
      minus: summarizeProbeResult(minus),
      wide: summarizeProbeResult(wide),
      drifts: {
        plus: sanitizeDrift(plusDrift, profile),
        minus: sanitizeDrift(minusDrift, profile),
        wide: sanitizeDrift(wideDrift, profile),
      },
      drift_consistency: {
        wide_minus_identical: wideMinusIdentical,
        wide_plus_identical: widePlusIdentical,
        wide_plus_delta: widePlusDelta,
        warning: consistencyWarning,
      },
      curvature: sanitizeCurvature(curvature),
      decision_margin: margin,
      basin,
      training_curvature_evidence: {
        status: "blocked",
        required: ["checkpoint_or_training_artifact", "loss_function", "representative_data", "gradient_capable_runtime"],
      },
      privacy: "local-only; model bytes and outputs were not uploaded",
    };
    deploymentSensitivityStatus.textContent = "Complete";
    deploymentSensitivityGrid.replaceChildren(
      deepBomMetric("Deploy Curvature", "Complete", "Finite-difference probes executed on the deployed TFLite/LiteRT function in this browser.", { tone: "good", label: "ok", criteria: "Computed from the deployment artifact through local runtime probes." }),
      deepBomMetric("Directional Curvature", formatDrift(curvature.normalizedRms), "Central finite-difference RMS normalized by input perturbation L2^2 along one local input direction.", statusInfo("Relative metric; compare across same model/input contract and perturbation scale.")),
      deepBomMetric("Raw 2nd Diff RMS", `${formatDrift(curvature.rawRms)} ${profile.unit}`, "RMS of y(x+eps)-2y(x)+y(x-eps) across returned output tensors.", statusInfo("Raw output-unit second difference; compare within the same dtype/output contract.")),
      deepBomMetric("Local Lipschitz", formatDrift(curvature.localLipschitz), "Max first-order RMS output drift divided by input perturbation L2 norm.", statusInfo("Relative sensitivity metric; lower is calmer for the same input scale.")),
      deepBomMetric("Tested Rank-Stability Radius", basin.radiusLabel, "Largest tested epsilon band that preserved first-output argmax for this synthetic local direction; no pass/fail threshold is implied.", statusInfo("Observed rank behavior for this input, direction, dtype, and perturbation scale only.")),
      deepBomMetric("Experimental Stability Composite", `${basin.score.toFixed(1)} / 100`, "Unvalidated fixed-weight summary of finite-difference curvature, output drift, cosine distance, decision margin, and top-1 stability. Inspect the component observations instead of using this value as a decision threshold.", { tone: "info", label: "experimental", criteria: "No demonstrated correlation with device latency, representative-data accuracy, robustness, or release readiness." }),
      deepBomMetric("Decision Margin", `${formatDrift(margin.margin)} ${profile.unit}`, margin.ready ? `Top-1 minus top-2 margin on the first output tensor. Status: ${margin.status}.` : "First output tensor did not expose at least two values.", margin.ready ? statusInfo(margin.detail || "Compare with max output drift; margin below drift can indicate rank instability.") : statusBlocked("Requires at least two output values.")),
      deepBomMetric("Top-1 Stability", basin.top1Stable ? "Stable" : "Changed", `+eps flip=${plusDrift.top1Flip ? "yes" : "no"} / -eps flip=${minusDrift.top1Flip ? "yes" : "no"} / 2eps flip=${wideDrift.top1Flip ? "yes" : "no"}.`, statusForTop1Flip(!basin.top1Stable)),
      deepBomMetric("Probe Consistency", consistencyWarning ? "Check" : "Distinct", consistencyWarning || "2eps drift is distinct from both +eps and -eps drift summaries.", consistencyWarning ? { tone: "warn", label: "watch", criteria: "watch when 2eps collapses to the same drift summary as +/-eps; common with quantized plateaus or saturated output regions." } : statusInfo("Independent perturbation probes produced distinct drift summaries.")),
      deepBomMetric("Input Probe", `${formatNumber(plus.inputPerturbation.touched)} values`, `epsilon=${plus.inputPerturbation.epsilonLabel}; L2=${formatDrift(plus.inputPerturbation.l2)}; source=synthetic tensor.`, statusInfo()),
    );
    renderDeploymentSensitivityProtocols(basin, curvature);
    deploymentSensitivityNotes.textContent = `Executed research-stage deploy-domain finite-difference probes around the current input contract. Compared ${formatNumber(curvature.count)} output values across ${baseline.outputs.length} output tensor(s). Baseline run ${baseline.runMs.toFixed(2)} ms; +eps ${plus.runMs.toFixed(2)} ms; -eps ${minus.runMs.toFixed(2)} ms; 2eps ${wide.runMs.toFixed(2)} ms. Raw observations are measured synthetic evidence; the 0-100 composite is unvalidated.${consistencyWarning ? ` ${consistencyWarning}` : ""}`;
    updateWorkflowState("pro");
    setStatus("Curvature complete", "ok");
    return deployCurvatureResult;
  } catch (error) {
    console.error(error);
    deployCurvatureResult = {
      schema: "deepbom.deploy_curvature_basin.v1",
      generated_at: new Date().toISOString(),
      model: modelIdentity(),
      status: "failed",
      error: shortError(error),
    };
    deploymentSensitivityStatus.textContent = "Failed";
    deploymentSensitivityGrid.replaceChildren(deepBomMetric("Execution", "Failed", shortError(error), { tone: "risk", label: "risk", criteria: "Execution must complete before the deployment-sensitivity proxy can be interpreted." }));
    renderDeploymentSensitivityProtocols(null, null);
    deploymentSensitivityNotes.textContent = "The module attempted local deploy-stability execution and failed before producing finite-difference evidence.";
    setStatus("Curvature failed", "error");
    return deployCurvatureResult;
  } finally {
    deploymentSensitivityPanelAction.disabled = !current;
    updateModuleAccessState();
  }
}

async function runTfliteOutputProbe(backend, mode, options = {}) {
  await ensureLiteRtRuntime(backend);
  const { Tensor, loadAndCompile } = await liteRtRuntime.loadCore();
  const compileStarted = performance.now();
  let model = null;
  const inputs = [];
  let outputs = null;
  let inputPerturbation = perturbationStatsFromMode(mode);
  try {
    model = await loadAndCompile(options.modelBytes || currentModelBytes, { accelerator: backend });
    const compileMs = performance.now() - compileStarted;
    for (const [index, details] of model.getInputDetails().entries()) {
      if (index === 0) {
        inputPerturbation = estimateInputPerturbation(details, current?.inputs?.[index], mode);
      }
      inputs.push(createResearchInputTensor(Tensor, details, current?.inputs?.[index], index, mode, options));
    }
    const runStarted = performance.now();
    outputs = await model.run(inputs);
    const runMs = performance.now() - runStarted;
    const outputArrays = await collectTensorArrays(outputs);
    return {
      backend,
      compileMs,
      runMs,
      outputs: outputArrays,
      outputCount: outputArrays.length,
      inputPerturbation,
    };
  } finally {
    deleteTensors(outputs);
    deleteTensors(inputs);
    model?.delete?.();
  }
}

function createResearchInputTensor(Tensor, details, staticTensor, index, mode, options = {}) {
  const shape = resolveFakeInputShape(Array.from(details.shape), staticTensor);
  const data = createResearchInputData(details.dtype, shape.reduce((acc, dim) => acc * dim, 1), staticTensor);
  if (index === 0 && isPerturbationMode(mode)) perturbTypedArray(data, details.dtype, staticTensor, perturbationOptions(mode));
  return Tensor.fromTypedArray(data, shape);
}

function estimateInputPerturbation(details, staticTensor, mode) {
  if (!isPerturbationMode(mode)) return perturbationStatsFromMode(mode);
  const options = perturbationOptions(mode);
  const shape = resolveFakeInputShape(Array.from(details.shape), staticTensor);
  const total = shape.reduce((acc, dim) => acc * dim, 1);
  const step = Math.max(1, Math.floor(total / 2048));
  const touched = Math.ceil(total / step);
  const unit = details.dtype === "float32" ? 0.01 : 1;
  const epsilon = unit * options.amplitude;
  return {
    active: true,
    mode,
    touched,
    total,
    epsilon,
    epsilonLabel: details.dtype === "float32" ? `${epsilon}` : `${epsilon} LSB`,
    l2: Math.sqrt(touched) * epsilon,
    linf: epsilon,
    direction: options.direction,
    amplitude: options.amplitude,
    pattern: options.pattern || "alternating",
  };
}

async function collectTensorArrays(value) {
  const tensors = Array.isArray(value) ? value : Object.values(value || {});
  const arrays = [];
  for (const tensor of tensors) {
    const data = await tensor.data();
    arrays.push(cloneTypedArray(data));
  }
  return arrays;
}

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
