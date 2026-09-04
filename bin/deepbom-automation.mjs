import { createHash, randomBytes } from "node:crypto";
import { access, mkdir, open, rename, unlink } from "node:fs/promises";
import path from "node:path";

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
    analysis_engine: "shared_browser_cli_javascript_and_tflite_wasm",
    commands: [
      { name: "audit", input_count: 1, outputs: ["analysis", "envelope", "cyclonedx", "sarif"] },
      { name: "gguf", input_count: 1, outputs: ["analysis", "envelope", "cyclonedx", "sarif"] },
      { name: "verify", input_count: 1, outputs: ["deepbom.cli_interface_contract_verification.v1"] },
      { name: "diff", input_count: 2, outputs: ["deepbom.deployment_delta.v1.1"] },
      { name: "explore", input_count: 1, outputs: ["deepbom.redesign_pareto.v1"] },
        { name: "graph", input_count: 1, outputs: ["svg", "png", "html", "mermaid", "dot", "deepbom.artifact_ir.v2", "deepbom.graph_ir.v1", "deepbom.visualization_manifest.v1"] },
        { name: "placement", input_count: 1, outputs: ["deepbom.placement_comparison.v1"] },
      { name: "accelerator collect nvidia", input_count: 0, outputs: ["deepbom.accelerator_profile.v1"] },
      { name: "explain-rule", input_count: 0, optional_identifier_count: 1, outputs: ["deepbom.rule_explanation.v1", "deepbom.rule_explanation_index.v1"] },
      { name: "self-test", input_count: 0, outputs: ["deepbom.cli_self_test.v1"] },
      { name: "capabilities", input_count: 0, outputs: [CLI_CAPABILITIES_SCHEMA] },
    ],
    inputs: {
      standalone_extensions: [".tflite", ".onnx", ".gguf", ".safetensors", ".mlmodel", ".pte", ".ptd"],
      package_kinds: ["coreml_mlpackage", "onnx_external_data", "safetensors_sharded_repository", "executorch_ptd"],
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
    output_contracts: {
      analysis: { media_type: "application/json", stability: "format_specific_complete_evidence" },
      envelope: { media_type: "application/json", schema: "deepbom.artifact_evidence_envelope.v1", stability: "canonical_cross_format_contract" },
      cyclonedx: { media_type: "application/vnd.cyclonedx+json", spec_version: "1.7" },
      sarif: { media_type: "application/sarif+json", version: SARIF_VERSION },
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
      identity_scoped_expiring_exceptions: true,
      deterministic_json: true,
      reproducible_timestamp_sources: ["--timestamp", "SOURCE_DATE_EPOCH"],
      atomic_file_output: true,
      no_clobber_output: true,
      structured_stderr: ["text", "json"],
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
  const rules = uniqueRuleFindings.map((finding) => ({
    id: finding.id,
    name: sarifName(finding.id),
    shortDescription: { text: finding.title || finding.id },
    fullDescription: { text: finding.summary || finding.title || finding.id },
    help: {
      text: finding.recommendation || finding.interpretation || "Review the hash-bound DEEPBOM evidence envelope.",
    },
    defaultConfiguration: { level: sarifLevel(finding.severity) },
    properties: {
      category: finding.rule_id || null,
      deepbomEvidenceClass: finding.evidence_class || null,
      deepbomSeverity: normalizeFindingLevel(finding.severity),
      deepbomFindingKind: finding.finding_kind || "caution",
    },
  }));
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
    level: sarifLevel(finding.severity),
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
  const message = error?.message || String(error);
  const format = errorFormat(argv, environment);
  const document = {
    schema: CLI_ERROR_SCHEMA,
    code: classifyCliError(message),
    message,
    exit_code: 1,
  };
  return format === "json" ? `${JSON.stringify(document)}\n` : `deepbom: ${message}\n`;
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

function sarifLevel(value) {
  const severity = normalizeFindingLevel(value);
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
  if (/unknown option|unexpected positional|required|must be|mutually exclusive|valid only|does not accept/i.test(message)) return "invalid_invocation";
  if (/unsupported artifact format|no analyzer is registered|requires a .* artifact/i.test(message)) return "unsupported_artifact";
  if (/cannot read|unavailable|does not exist|not found|no such file or directory|must be a regular file or directory/i.test(message)) return "input_unavailable";
  if (/output already exists|EACCES|EPERM|ENOSPC/i.test(message)) return "output_failure";
  if (/invalid|mismatch|failed|unsafe|changed during analysis/i.test(message)) return "artifact_or_evidence_invalid";
  return "analysis_failure";
}

export async function outputExists(outputPath) {
  try { await access(path.resolve(outputPath)); return true; } catch { return false; }
}
