export function tensorRtCycloneDxPropertyEntries(analysis) {
  const preflight = analysis?.tensorrt_static_preflight;
  if (preflight?.schema !== "deepbom.tensorrt_static_preflight.v1") return [];
  const observation = preflight.parser_observation;
  const engine = preflight.engine_inspector_evidence;
  const runtime = observation || engine?.runtime || null;
  const counts = preflight.projection?.state_counts || {};
  return [
    ["deepbom:model:tensorRtStaticPreflightSchema", preflight.schema],
    ["deepbom:model:tensorRtStaticPreflightStatus", preflight.status],
    ["deepbom:model:tensorRtStaticPreflightEvidenceClass", preflight.evidence_class],
    ["deepbom:model:tensorRtBuildProfileSha256", preflight.build_profile?.profile_sha256],
    ["deepbom:model:tensorRtExecutionPath", preflight.build_profile?.execution_path],
    ["deepbom:model:tensorRtParserApiMethod", observation?.api_method],
    ["deepbom:model:tensorRtVersion", runtime?.tensorrt_version],
    ["deepbom:model:tensorRtCudaVersion", runtime?.cuda_version],
    ["deepbom:model:tensorRtDeviceIdentity", runtime?.device_identity],
    ["deepbom:model:tensorRtConditionallyEligibleOperatorCount", counts.CONDITIONALLY_ELIGIBLE],
    ["deepbom:model:tensorRtDefiniteExclusionOperatorCount", counts.DEFINITE_EXCLUSION],
    ["deepbom:model:tensorRtUnresolvedOperatorCount", counts.UNRESOLVED],
    ["deepbom:model:tensorRtCollectorBinarySha256", observation?.collector?.binary_sha256],
    ["deepbom:model:tensorRtCollectorSourceSetSha256", observation?.collector?.source_set_sha256],
    ["deepbom:model:tensorRtCollectorGitCommit", observation?.collector?.git_commit],
    ["deepbom:model:tensorRtCollectorGitState", observation?.collector?.git_state],
    ["deepbom:model:tensorRtOptimizationProfileCostSchema", preflight.optimization_profile_cost?.schema],
    ["deepbom:model:tensorRtOptimizationProfileCostStatus", preflight.optimization_profile_cost?.status],
    ["deepbom:model:tensorRtOptimizationProfileCostScenarioCount", preflight.optimization_profile_cost?.scenario_count],
    ["deepbom:model:tensorRtEngineInspectorSchema", engine?.schema],
    ["deepbom:model:tensorRtEngineInspectorStatus", engine?.status],
    ["deepbom:model:tensorRtEngineInspectorEvidenceClass", engine?.evidence_class],
    ["deepbom:model:tensorRtEngineSha256", engine?.engine?.sha256],
    ["deepbom:model:tensorRtEngineByteLength", engine?.engine?.byte_length],
    ["deepbom:model:tensorRtEngineLayerCount", engine?.engine_layer_count],
    ["deepbom:model:tensorRtEngineIoTensorCount", engine?.io_tensor_count],
    ["deepbom:model:tensorRtTacticAnnotatedLayerCount", engine?.tactic_annotated_layer_count],
    ["deepbom:model:tensorRtMultiSourceMetadataLayerCount", engine?.multi_source_metadata_layer_count],
    ["deepbom:model:tensorRtEngineSourceMappingStatus", engine?.source_mapping_status],
    ["deepbom:model:tensorRtArtifactEngineRelation", engine?.artifact_engine_relation],
    ["deepbom:model:tensorRtInspectorSchemaGeneration", engine?.inspector?.schema_generation],
    ["deepbom:model:tensorRtInspectorProfilingVerbosity", engine?.inspector?.profiling_verbosity],
    ["deepbom:model:tensorRtInspectorExecutionContextBound", engine?.inspector?.execution_context_bound],
    ["deepbom:model:tensorRtInspectorCanonicalJsonSha256", engine?.inspector?.canonical_json_sha256],
    ["deepbom:model:tensorRtEngineBuildCaptureClass", engine?.build_capture?.evidence_class],
    ["deepbom:model:tensorRtEngineBuildToolBinarySha256", engine?.build_capture?.tool_binary_sha256],
    ["deepbom:model:tensorRtEngineBuildInvocationSha256", engine?.build_capture?.invocation_sha256],
    ["deepbom:model:tensorRtEvidencePointer", "/format_extensions/onnx/tensorrt_static_preflight"],
  ];
}
