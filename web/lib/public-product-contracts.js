export const PUBLIC_PRODUCT_CONTRACT_SCHEMA = "deepbom.public_product_contracts.v1";

const FORMAT_MATURITY = Object.freeze([
  format("tflite", "stable", "stable", "stable", "source_bounded", "Full FlatBuffer artifact analysis; placement remains a source/build projection until runtime evidence is imported."),
  format("onnx", "stable", "stable", "stable", "source_bounded", "Full protobuf and external-data analysis; execution-provider assignment is never inferred from eligibility."),
  format("gguf", "stable", "not_applicable", "stable_storage_encoding", "not_applicable", "GGUF serializes tensor storage and architecture metadata, not an executable operator DAG."),
  format("safetensors", "stable", "not_applicable", "stable_metadata_bound", "not_applicable", "SafeTensors serializes named tensor storage; graph and placement require a separately bound executable artifact."),
  format("coreml", "preview", "preview", "preview", "imported_evidence_only", "NeuralNetwork and ML Program evidence is supported; compressed and flexible contracts remain fixture-bounded."),
  format("executorch", "preview", "preview", "limited", "selected_build_required", "ET12/FT01 parsing is bounded; backend and operator completeness require selected-build evidence."),
]);

const MACHINE_CONTRACTS = Object.freeze([
  machineContract("deepbom.cli_capabilities.v1", "stable", "Additive fields only within v1; removals or semantic changes require a new schema id."),
  machineContract("deepbom.artifact_evidence_envelope.v1", "stable", "Canonical cross-format automation envelope."),
  machineContract("deepbom.artifact_ir.v2", "stable", "Canonical artifact evidence identity, graph/storage topology, and overlay contract."),
  machineContract("deepbom.semantic_artifact_diff.v1", "stable", "Same-format graph, storage, and quantization-contract comparison derived from two Artifact IR documents."),
  machineContract("deepbom.review_summary.v1", "stable", "Structured source for bounded human summaries; rendered prose is not a machine API."),
  machineContract("sarif-2.1.0/deepbomFinding-v1", "stable", "Finding identity and artifact fingerprints are stable; additive SARIF properties are allowed."),
  machineContract("cyclonedx-1.7-json", "stable_projection", "CycloneDX 1.7 is the only product ML-BOM export contract."),
  machineContract("format_specific_analysis", "compatibility_surface", "Richer native analysis may add or revise fields; automation should consume the envelope or Artifact IR."),
]);

export const PUBLIC_PRODUCT_CONTRACTS = deepFreeze({
  schema: PUBLIC_PRODUCT_CONTRACT_SCHEMA,
  format_maturity: FORMAT_MATURITY,
  machine_contracts: MACHINE_CONTRACTS,
  release_policy: {
    schema: "deepbom.release_channel_policy.v1",
    stable_channel: {
      source_channel: "release",
      npm_dist_tag: "latest",
      version_pattern: "MAJOR.MINOR.PATCH",
      promotion_gate: "clean full quality, deploy, six-platform channel matrix, and installed-channel equivalence",
      target_minimum_interval_days: 14,
      interval_exception: "P0 correctness or security fix with a bounded regression fixture",
    },
    prerelease_channel: {
      source_channel: "prerelease",
      npm_dist_tag: "next",
      version_pattern: "MAJOR.MINOR.PATCH-(alpha|beta|rc).N",
      promotion_gate: "affected quality tiers and channel packaging pass; never promoted to latest",
    },
    development_channel: {
      source_channel: "dev",
      publishable: false,
      version_pattern: "MAJOR.MINOR.PATCH-dev",
    },
  },
});

export function clonePublicProductContracts() {
  return JSON.parse(JSON.stringify(PUBLIC_PRODUCT_CONTRACTS));
}

export function formatMaturity(formatId) {
  return FORMAT_MATURITY.find((row) => row.format === String(formatId || "").toLowerCase()) || null;
}

function format(id, parser, graph, quantization, placement, boundary) {
  return Object.freeze({
    format: id,
    overall: [parser, graph, quantization].includes("preview") || [parser, graph, quantization].includes("limited") ? "preview" : "stable",
    parser,
    serialized_graph: graph,
    quantization: quantization,
    execution_placement: placement,
    interpretation_boundary: boundary,
  });
}

function machineContract(schema, stability, boundary) {
  return Object.freeze({ schema, stability, compatibility_boundary: boundary });
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.freeze(value);
  for (const child of Object.values(value)) deepFreeze(child);
  return value;
}
