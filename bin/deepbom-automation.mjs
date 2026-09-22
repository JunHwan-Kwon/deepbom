import { createHash, randomBytes } from "node:crypto";
import { access, mkdir, open, rename, unlink } from "node:fs/promises";
import path from "node:path";

import {
  AUDIT_DEFAULT_OUTPUT_FORMAT,
  AUDIT_OUTPUT_FORMATS,
  cloneAuditOutputContracts,
} from "../web/lib/audit-output-contracts.js";
import { clonePublicProductContracts } from "../web/lib/public-product-contracts.js";
import { findFindingRule } from "../web/lib/finding-rule-catalog.js";
import { evaluateGatePolicyProfile, listGatePolicyProfiles } from "../web/lib/gate-policy-profiles.js";

export const CLI_CAPABILITIES_SCHEMA = "deepbom.cli_capabilities.v1";
export const CLI_ERROR_SCHEMA = "deepbom.cli_error.v1";
export const CLI_POLICY_RESULT_SCHEMA = "deepbom.cli_finding_policy_result.v1";
export const CLI_DEFECT_GATE_RESULT_SCHEMA = "deepbom.cli_defect_gate_result.v1";
export const SARIF_VERSION = "2.1.0";

const FINDING_LEVELS = Object.freeze(["informational", "low", "medium", "high"]);
const FINDING_LEVEL_RANK = new Map(FINDING_LEVELS.map((level, index) => [level, index]));

export function buildCliCapabilities(version, { defaultTarget, deltaTargets } = {}) {
  return {
    schema: CLI_CAPABILITIES_SCHEMA,
    cli_version: String(version),
    optional_numerical_ir: { weight: { schema: "deepbom.weight_ir.v1", flag: "--weight-analysis", section: "weight_ir", analysis_schema: "deepbom.weight_analysis.v1", analysis_section: "weight_analysis", comparison_schema: "deepbom.weight_comparison.v1", comparison_section: "weight_comparison", baseline_flag: "--weight-baseline", options_flag: "--weight-options", mapping_flag: "--weight-mapping", features: ["channels", "signed_cosine_similarity", "singular_spectrum", "sparsity_2_4", "affine_quantization", "aligned_baseline_comparison"], formats: ["onnx", "tflite", "gguf", "safetensors", "coreml", "executorch"], coverage: "decoder-dependent; every storage object is accounted for", execution: false }, activation: { schema: "deepbom.activation_ir.v1", capture_schema: "deepbom.activation_capture.v1", flag: "--activation-evidence", section: "activation_ir", execution: false, trust: "imported evidence; consistency checked, not attested" }, default_enabled: false, output: "json", maximum_weight_values: 100_000_000, maximum_capture_values: 1_000_000 },
    analysis_engine: "shared_browser_cli_javascript_and_tflite_wasm",
    commands: [
      { name: "audit", input_count: 1, outputs: [...AUDIT_OUTPUT_FORMATS] },
      { name: "gguf", input_count: 1, outputs: [...AUDIT_OUTPUT_FORMATS] },
      { name: "batch", input_count: 1, outputs: ["summary", "deepbom.batch_result.v1"], manifest_schema: "deepbom.batch_manifest.v1" },
      { name: "verify", input_count: 1, declaration_count: 1, declaration_kinds: ["artifact_derived_or_supplied_interface_contract", "cyclonedx_1_7_bom"], outputs: ["summary", "markdown", "deepbom.cli_interface_contract_verification.v1", "deepbom.bom_artifact_reconciliation.v1"] },
      { name: "contract capture", input_count: 1, outputs: ["summary", "deepbom.artifact_derived_interface_baseline.v1"], production_approval_inferred: false },
      { name: "diff", input_count: 2, outputs: ["summary", "markdown", "deepbom.semantic_artifact_diff.v1", "deepbom.tensor_encoding_diff.v1"], supported_formats: ["tflite", "onnx", "coreml", "gguf", "safetensors", "executorch"], matching_format_required: true },
      { name: "explore", input_count: 1, outputs: ["summary", "deepbom.redesign_pareto.v1"] },
      { name: "graph", input_count: 1, outputs: ["svg", "png", "html", "mermaid", "dot", "deepbom.artifact_ir.v2", "deepbom.model_ir.v1", "deepbom.graph_ir.v1", "deepbom.visualization_manifest.v1"] },
      { name: "model-summary", input_count: 1, outputs: ["table", "markdown", "deepbom.model_summary.v1"], levels: ["auto", "operation", "block", "storage"], trainability_inferred: false, runtime_order_inferred: false },
      { name: "visualize", input_count: 1, outputs: ["monochrome_a4_svg", "black_white_300dpi_png", "caption_sidecars", "deepbom.model_ir_visualization_manifest.v1"], views: ["identity-boundary", "architecture-overview", "block-detail", "exhaustive", "static-runtime", "observed-runtime"], regulatory_conclusion_inferred: false },
      { name: "placement", input_count: 1, outputs: ["deepbom.placement_comparison.v1"] },
      { name: "accelerator collect nvidia", input_count: 0, outputs: ["deepbom.accelerator_profile.v1"] },
      { name: "explain-rule", input_count: 0, optional_identifier_count: 1, outputs: ["deepbom.rule_explanation.v1", "deepbom.rule_explanation_index.v1"] },
      { name: "self-test", input_count: 0, outputs: ["deepbom.cli_self_test.v1"] },
      { name: "capabilities", input_count: 0, outputs: [CLI_CAPABILITIES_SCHEMA] },
      { name: "integrate", input_count: 0, outputs: ["deepbom.agent_integration.v1"], targets: ["codex", "claude-code", "generic"], writes_require_apply: true },
    ],
    inputs: {
      standalone_extensions: [".tflite", ".onnx", ".gguf", ".safetensors", ".mlmodel", ".pte", ".ptd", ".pb", ".h5", ".hdf5", ".keras", ".pt2", ".pt", ".pth", ".ckpt"],
      package_kinds: ["coreml_mlpackage", "onnx_external_data", "safetensors_sharded_repository", "executorch_ptd", "tensorflow_savedmodel"],
      preview_boundaries: {
        static_graph: ["tensorflow_graphdef", "tensorflow_savedmodel_first_metagraph", "keras_declarative_config", "pt2_declarative_graph"],
        safe_envelope_only: ["generic_hdf5", "pytorch_pt_pth_ckpt"],
        arbitrary_framework_object_construction: false,
      },
      remote_sources: {
        huggingface: "full_commit_required",
        gcs: "object_generation_required",
        https: "sha256_required",
        kaggle: "not_yet_available_use_verified_local_or_https_sha256",
        content_addressed_cache: true,
        remote_code_execution: false,
      },
      symbolic_stdin: false,
      symbolic_stdin_reason: "Artifact identity, sidecar resolution, and bounded range reads require a stable regular file or package directory.",
    },
    default_audit_output: AUDIT_DEFAULT_OUTPUT_FORMAT,
    output_contracts: {
      ...cloneAuditOutputContracts(),
      analysis: {
        media_type: "application/json",
        compatibility_alias_for: ["json", "json-compact"],
        stability: "format_specific_complete_evidence",
      },
    },
    bounded_projections: {
      gguf_tensor_table: {
        invocation: "deepbom gguf <artifact.gguf> --tensors --compact",
        schema: "deepbom.tensor_table.v1",
        pagination: "--tensor-offset <n> --tensor-limit <1..1000>",
        default_scan: "structure",
        excludes: ["decoded_tensor_values", "tensor_numerical_integrity_ledgers"],
      },
      tensor_encoding_inventory: {
        invocation: "deepbom audit <artifact> --encoding-inventory --compact",
        schema: "deepbom.tensor_encoding_inventory.v1",
        supported_formats: ["gguf", "safetensors", "onnx", "tflite"],
        default_scan_for_range_read_containers: "structure",
        evidence_scope: "serialized_tensor_encoding_only",
      },
      model_ir_document_views: {
        invocation: "deepbom visualize <artifact> --view all --orientation portrait -o model-views.zip",
        schema: "deepbom.model_ir_visualization_manifest.v1",
        canonical_vector: "monochrome ISO A4 SVG with source subject references",
        compatibility_raster: "300-DPI opaque black-white PNG",
        boundary: "engineering evidence only; no standard-conformance or regulatory-approval claim",
      },
      format_neutral_model_summary: {
        invocation: "deepbom model-summary <artifact> --level auto --format table",
        schema: "deepbom.model_summary.v1",
        levels: ["auto", "operation", "block", "storage"],
        boundary: "serialized storage is not framework trainability; display order is not runtime order; unknown values are not zero-filled",
      },
    },
    provenance_inputs: {
      conversion_receipt: "deepbom.conversion_receipt.v1",
      source_code_serialization_loaded: false,
      output_artifact_binding: "exact_sha256_and_format",
      converter_claim_evidence_class: "DECLARED_UNVERIFIED",
    },
    scan_policies: {
      modes: ["auto", "structure", "integrity", "full"],
      gguf_safetensors_range_read: true,
      gguf_safetensors_auto_structure_above_bytes: "10737418240",
      gguf_safetensors_auto_integrity_above_bytes: "2147483648",
      monolithic_executable_fail_closed_above_bytes: "1073741824",
      payload_execution: false,
    },
    public_product_contracts: clonePublicProductContracts(),
    accelerator_profiles: {
      nvidia_collector_schema: "deepbom.accelerator_profile.v1",
      legacy_nvidia_binding_schema: "deepbom.accelerator_profile_binding.v1",
      binding_schema: "deepbom.accelerator_binding.v1",
      evidence_stages: ["serialized_artifact", "source_eligibility", "selected_build", "compiled_plan", "observed_assignment", "measured_execution"],
      imports: {
        coreml_compute_plan: "deepbom.coreml_compute_plan.v1",
        edgetpu_compiler_evidence: "deepbom.edgetpu_compiler_evidence.v1",
        litert_qualcomm_compiler_dispatch_evidence: "deepbom.litert_qualcomm_compiler_dispatch_evidence.v1",
        tensorrt_parser_observation: "deepbom.tensorrt_parser_observation.v1",
        tensorrt_engine_inspector: "deepbom.tensorrt_engine_inspector_evidence.v1",
      },
      cpu_cost_profile_separate: true,
      selected_build_inferred_from_host_profile: false,
      runtime_assignment_inferred_from_host_profile: false,
      llm_vram_comparison: "conditional_static_lower_bound_only",
    },
    automation: {
      finding_gate_levels: FINDING_LEVELS,
      finding_kinds: ["artifact_defect", "caution", "evidence_gap"],
      default_gate: "artifact_defect_only",
      review_policy_schema: "deepbom.review_policy.v1",
      review_policy_states: ["execution_status", "coverage_status", "finding_policy_status"],
      builtin_policy_profiles: listGatePolicyProfiles(),
      identity_scoped_expiring_exceptions: true,
      deterministic_json: true,
      reproducible_timestamp_sources: ["--timestamp", "SOURCE_DATE_EPOCH"],
      atomic_file_output: true,
      no_clobber_output: true,
      structured_stderr: ["text", "json"],
      // A transport, not an analysis command: it emits no evidence document of
      // its own, so it is declared here rather than in `commands`.
      mcp_stdio_server: {
        invocation: "deepbom mcp",
        transport: "stdio_jsonrpc",
        protocol_versions: ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"],
        tools: ["deepbom_capabilities", "deepbom_audit", "deepbom_diff", "deepbom_explain_rule"],
        default_audit_output: AUDIT_DEFAULT_OUTPUT_FORMAT,
        detailed_outputs_are_explicit: true,
        local_path_policy: "DEEPBOM_MCP_ALLOWED_ROOTS_or_launch_directory",
        bounded_environment_controls: [
          "DEEPBOM_MCP_MAX_RESPONSE_BYTES",
          "DEEPBOM_MCP_MAX_FRAME_BYTES",
          "DEEPBOM_MCP_TOOL_TIMEOUT_MS",
          "DEEPBOM_MCP_MAX_CONCURRENT",
          "DEEPBOM_MCP_MAX_QUEUE",
        ],
        hosted_endpoint: false,
      },
    },
    targets: {
      default_tflite_target: defaultTarget || null,
      default_diff_targets: Array.isArray(deltaTargets) ? [...deltaTargets] : [],
      custom_tflite_target_profile: true,
    },
    exit_codes: {
      "0": "command completed and any requested gate passed",
      "1": "invalid invocation, unreadable input, unsupported artifact, or analysis/output failure",
      "2": "verification contradiction or finding policy blocked",
      "3": "verification could not establish a complete release binding",
      "4": "independently supplied artifact SHA-256 did not match the observed artifact",
    },
    privacy: {
      model_bytes_network_transfer: false,
      telemetry: false,
      analysis_location: "local_process",
    },
  };
}

export function buildSarifDocument(envelope, { version, policyResult = null } = {}) {
  if (envelope?.schema !== "deepbom.artifact_evidence_envelope.v1") {
    throw new Error("SARIF projection requires a deepbom.artifact_evidence_envelope.v1 document.");
  }
  const findings = Array.isArray(envelope.findings) ? envelope.findings : [];
  const uniqueRuleFindings = [...new Map(findings.map((finding) => [finding.id, finding])).values()];
  const ruleIndexById = new Map(uniqueRuleFindings.map((finding, index) => [finding.id, index]));
  const rules = uniqueRuleFindings.map((finding) => {
    const catalog = findFindingRule(finding.id);
    return {
      id: finding.id,
      name: sarifName(finding.id),
      shortDescription: { text: finding.title || catalog?.title || finding.id },
      fullDescription: { text: finding.summary || finding.title || catalog?.title || finding.id },
      help: {
        text: catalog?.remediation || finding.recommendation || finding.interpretation || "Review the hash-bound DEEPBOM evidence envelope.",
      },
      defaultConfiguration: { level: sarifLevel(finding) },
      properties: {
        category: finding.rule_id || null,
        deepbomEvidenceClass: finding.evidence_class || null,
        deepbomSeverity: normalizeFindingLevel(finding.severity),
        deepbomFindingKind: finding.finding_kind || "caution",
        deepbomRuleCatalogSchema: catalog ? "deepbom.finding_rule_explanation.v1" : null,
        deepbomRuleTriggerContract: catalog?.trigger_contract || null,
      },
    };
  });
  const artifactUri = artifactUriFor(envelope.identity?.filename || "model");
  const artifact = {
    location: { uri: artifactUri },
    roles: ["analysisTarget"],
    ...(envelope.identity?.sha256 ? { hashes: { "sha-256": envelope.identity.sha256 } } : {}),
    properties: {
      format: envelope.identity?.format || null,
      byteLength: envelope.identity?.byte_length ?? null,
    },
  };
  const results = findings.map((finding) => ({
    ruleId: finding.id,
    ruleIndex: ruleIndexById.get(finding.id),
    level: sarifLevel(finding),
    message: { text: finding.summary || finding.title || finding.id },
    locations: [{ physicalLocation: { artifactLocation: { uri: artifactUri, index: 0 } } }],
    partialFingerprints: {
      "deepbomFinding/v1": sha256(JSON.stringify({
        artifact_sha256: envelope.identity?.sha256 || null,
        finding_id: finding.id,
        evidence_class: finding.evidence_class || null,
        source_pointers: finding.source_pointers || [],
      })),
    },
    properties: {
      deepbomEvidenceClass: finding.evidence_class || null,
      deepbomSeverity: normalizeFindingLevel(finding.severity),
      deepbomFindingKind: finding.finding_kind || "caution",
      deepbomStatus: finding.status || null,
      deepbomInterpretation: finding.interpretation || null,
      deepbomRecommendation: finding.recommendation || null,
      deepbomSourcePointers: finding.source_pointers || [],
    },
  }));
  return {
    $schema: "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
    version: SARIF_VERSION,
    runs: [{
      tool: {
        driver: {
          name: "DEEPBOM",
          semanticVersion: String(version),
          informationUri: "https://deepbom.org",
          rules,
        },
      },
      automationDetails: {
        id: `deepbom/${envelope.identity?.sha256 || envelope.envelope_sha256}`,
      },
      artifacts: [artifact],
      results,
      invocations: [{
        executionSuccessful: true,
        properties: {
          evidenceEnvelopeSha256: envelope.envelope_sha256,
          findingPolicy: policyResult,
        },
      }],
      properties: {
        deepbomEvidenceEnvelopeSchema: envelope.schema,
        deepbomEvidenceEnvelopeSha256: envelope.envelope_sha256,
        deepbomEvidenceBoundary: envelope.evidence_boundary,
      },
    }],
  };
}

export function evaluateFindingPolicy(envelope, failOn = "none") {
  const threshold = normalizeFailOn(failOn);
  const findings = Array.isArray(envelope?.findings) ? envelope.findings : [];
  const counts = Object.fromEntries(FINDING_LEVELS.map((level) => [level, 0]));
  for (const finding of findings) counts[normalizeFindingLevel(finding?.severity)] += 1;
  const thresholdRank = threshold === "none" ? Number.POSITIVE_INFINITY : FINDING_LEVEL_RANK.get(threshold);
  const blocking = findings.filter((finding) => FINDING_LEVEL_RANK.get(normalizeFindingLevel(finding?.severity)) >= thresholdRank);
  return {
    schema: CLI_POLICY_RESULT_SCHEMA,
    status: blocking.length ? "block" : "pass",
    fail_on: threshold,
    finding_count: findings.length,
    severity_counts: counts,
    blocking_finding_count: blocking.length,
    blocking_finding_ids: blocking.map((finding) => finding.id),
    evidence_envelope_sha256: envelope?.envelope_sha256 || null,
  };
}

export function evaluateDefectGate(envelope) {
  const findings = Array.isArray(envelope?.findings) ? envelope.findings : [];
  const counts = { artifact_defect: 0, caution: 0, evidence_gap: 0 };
  for (const finding of findings) counts[finding.finding_kind] = (counts[finding.finding_kind] || 0) + 1;
  const blocking = findings.filter((finding) => finding.finding_kind === "artifact_defect");
  return {
    schema: CLI_DEFECT_GATE_RESULT_SCHEMA,
    status: blocking.length ? "block" : "pass",
    gate: "defects",
    finding_count: findings.length,
    finding_kind_counts: counts,
    blocking_finding_count: blocking.length,
    blocking_finding_ids: blocking.map((finding) => finding.id),
    evidence_envelope_sha256: envelope?.envelope_sha256 || null,
  };
}

export { evaluateGatePolicyProfile };

export function normalizeFailOn(value) {
  const normalized = String(value || "none").trim().toLowerCase();
  if (normalized === "info") return "informational";
  if (normalized !== "none" && !FINDING_LEVEL_RANK.has(normalized)) {
    throw new Error("--fail-on must be none, informational, low, medium, or high.");
  }
  return normalized;
}

export function resolveGenerationTimestamp(explicit, environment = process.env) {
  if (explicit) return normalizeIsoTimestamp(explicit, "--timestamp");
  const sourceDateEpoch = String(environment?.SOURCE_DATE_EPOCH || "").trim();
  if (!sourceDateEpoch) return null;
  if (!/^\d+$/.test(sourceDateEpoch)) throw new Error("SOURCE_DATE_EPOCH must be a non-negative integer number of seconds.");
  const seconds = Number(sourceDateEpoch);
  if (!Number.isSafeInteger(seconds)) throw new Error("SOURCE_DATE_EPOCH exceeds the safe integer range.");
  return new Date(seconds * 1000).toISOString();
}

export async function writeOutputAtomically(outputPath, text, { noClobber = false } = {}) {
  const destination = path.resolve(outputPath);
  const directory = path.dirname(destination);
  await mkdir(directory, { recursive: true });
  const temporary = path.join(directory, `.${path.basename(destination)}.${process.pid}.${randomBytes(8).toString("hex")}.tmp`);
  let handle;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(text, "utf8");
    await handle.sync();
    await handle.close();
    handle = null;
    if (noClobber) {
      let reservation;
      try {
        reservation = await open(destination, "wx", 0o600);
        await reservation.close();
        reservation = null;
        await rename(temporary, destination);
      } catch (error) {
        if (reservation) await reservation.close().catch(() => {});
        if (error?.code !== "EEXIST") await unlink(destination).catch(() => {});
        throw error;
      }
    } else {
      await rename(temporary, destination);
    }
  } catch (error) {
    if (handle) await handle.close().catch(() => {});
    await unlink(temporary).catch(() => {});
    if (noClobber && error?.code === "EEXIST") throw new Error(`Output already exists: ${destination}`);
    throw error;
  }
}

export function renderCliError(error, argv = process.argv.slice(2), environment = process.env) {
  const document = cliErrorDocument(error);
  const format = errorFormat(argv, environment);
  return format === "json" ? `${JSON.stringify(document)}\n` : `deepbom: ${document.message}\n`;
}

export function cliErrorDocument(error) {
  const message = error?.message || String(error);
  const code = classifyCliError(message);
  const details = errorDetails(code, message);
  return {
    schema: CLI_ERROR_SCHEMA,
    code,
    message,
    exit_code: code === "artifact_identity_mismatch" ? 4 : 1,
    expected: details.expected,
    observed: details.observed,
    suggested_action: details.suggested_action,
  };
}

export function exitCodeForCliError(error) {
  return cliErrorDocument(error).exit_code;
}

function normalizeIsoTimestamp(value, label) {
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds)) throw new Error(`${label} must be an ISO-8601 timestamp.`);
  return new Date(milliseconds).toISOString();
}

function normalizeFindingLevel(value) {
  const normalized = String(value || "informational").trim().toLowerCase();
  if (normalized === "info" || normalized === "note") return "informational";
  return FINDING_LEVEL_RANK.has(normalized) ? normalized : "informational";
}

function sarifLevel(finding) {
  const severity = normalizeFindingLevel(finding.severity);
  // Missing evidence is not an observed defect. Keep original severity/kind
  // in SARIF properties so a consumer can apply its own evidence policy.
  if (finding.finding_kind === "evidence_gap") return "note";
  if (finding.finding_kind !== "artifact_defect") return ["high", "medium"].includes(severity) ? "warning" : "note";
  if (severity === "high") return "error";
  if (severity === "medium") return "warning";
  return "note";
}

function sarifName(value) {
  const name = String(value || "DEEPBOMFinding").replace(/[^A-Za-z0-9_]/g, "_");
  return /^[A-Za-z_]/.test(name) ? name : `DEEPBOM_${name}`;
}

function artifactUriFor(value) {
  return String(value).replaceAll("\\", "/").split("/").map((segment) => encodeURIComponent(segment)).join("/");
}

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function errorFormat(argv, environment) {
  const index = argv.indexOf("--error-format");
  const candidate = index >= 0 ? argv[index + 1] : environment?.DEEPBOM_ERROR_FORMAT;
  return String(candidate || "text").toLowerCase() === "json" ? "json" : "text";
}

function classifyCliError(message) {
  if (/Artifact SHA-256 mismatch: expected [a-f0-9]{64}, observed [a-f0-9]{64}/i.test(message)) return "artifact_identity_mismatch";
  if (/unknown option|unexpected positional|required|must be|mutually exclusive|valid only|does not accept/i.test(message)) return "invalid_invocation";
  if (/unsupported artifact format|no analyzer is registered|requires a .* artifact/i.test(message)) return "unsupported_artifact";
  if (/cannot read|unavailable|does not exist|not found|no such file or directory|must be a regular file or directory/i.test(message)) return "input_unavailable";
  if (/output already exists|EACCES|EPERM|ENOSPC/i.test(message)) return "output_failure";
  if (/invalid|mismatch|failed|unsafe|changed during analysis/i.test(message)) return "artifact_or_evidence_invalid";
  return "analysis_failure";
}

function errorDetails(code, message) {
  const digestMismatch = message.match(/Artifact SHA-256 mismatch: expected ([a-f0-9]{64}), observed ([a-f0-9]{64})/i);
  if (digestMismatch) return {
    expected: { sha256: digestMismatch[1].toLowerCase() },
    observed: { sha256: digestMismatch[2].toLowerCase() },
    suggested_action: "Stop the pipeline and confirm the immutable artifact source or update the independently reviewed digest.",
  };
  const truncated = message.match(/payload range (\d+):(\d+) exceeds source length (\d+)/i);
  if (truncated) return {
    expected: { minimum_byte_length: Number(truncated[2]), payload_range: [Number(truncated[1]), Number(truncated[2])] },
    observed: { byte_length: Number(truncated[3]) },
    suggested_action: "The file is likely truncated or its tensor directory is corrupt. Reacquire it from an immutable source and verify its SHA-256 before retrying.",
  };
  const suggestions = {
    invalid_invocation: "Correct the command arguments; run deepbom --help or deepbom capabilities --format agent-text before retrying.",
    unsupported_artifact: "Supply one supported serialized deployment artifact or package directory and inspect deepbom capabilities for accepted formats.",
    input_unavailable: "Confirm that the path exists and is readable, or use an explicitly immutable remote source with the required digest binding.",
    output_failure: "Choose a writable output path, free sufficient space, and use --no-clobber only when replacement must be refused.",
    artifact_or_evidence_invalid: "Treat the input as invalid or contradictory; inspect the complete message and reacquire or regenerate the affected evidence.",
    analysis_failure: "Preserve the failing bytes and command, then rerun with --error-format json for a machine-readable diagnostic.",
  };
  return { expected: null, observed: null, suggested_action: suggestions[code] || suggestions.analysis_failure };
}

export async function outputExists(outputPath) {
  try { await access(path.resolve(outputPath)); return true; } catch { return false; }
}
