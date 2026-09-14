import {
  AGENT_CONTRACT,
  EVIDENCE_CONTRACT,
  cloneContractIdentity,
} from "./deepbom-public-contract-versions.mjs";

export const AGENT_CAPABILITIES_SCHEMA = "deepbom.agent_capabilities.v1";

export function buildAgentCapabilities(cliCapabilities) {
  if (cliCapabilities?.schema !== "deepbom.cli_capabilities.v1") {
    throw new Error("Agent capability projection requires deepbom.cli_capabilities.v1.");
  }
  const version = cliCapabilities.cli_version;
  const packageSpec = `deepbom@${version}`;
  return {
    schema: AGENT_CAPABILITIES_SCHEMA,
    product: "DEEPBOM",
    version,
    agent_contract: cloneContractIdentity(AGENT_CONTRACT),
    evidence_schema_contract: cloneContractIdentity(EVIDENCE_CONTRACT),
    purpose: "Evidence-bounded static analysis of serialized AI deployment artifacts through local tools or the ChatGPT browser sandbox.",
    selection: {
      use_when: [
        "A supported serialized deployment artifact must be inspected without executing model code.",
        "The task asks about graph or tensor contracts, MAC or symbolic-shape coverage, quantization, memory feasibility, accelerator eligibility, artifact diff, graph export, SARIF, or CycloneDX 1.7.",
      ],
      do_not_use_when: [
        "The only input is a training checkpoint that would require deserializing executable Python or HDF5 state.",
        "The requested conclusion is measured latency, energy, thermal behavior, task accuracy, clinical validity, or actual runtime assignment without imported evidence.",
      ],
    },
    invocation: {
      package: packageSpec,
      compatible_registry_package: "deepbom@latest",
      resolution_order: ["DEEPBOM_BIN", "bundled_npm_package", "compatible_agent_contract_on_PATH", "authorized_npm_download"],
      resolution_helper: "skills/deepbom/scripts/verify-deepbom.mjs",
      registry_download_default: "disabled_requires_explicit_allow_download",
      discovery: "deepbom capabilities --format agent-json",
      self_test: "deepbom self-test --compact",
      audit_template: "deepbom audit \"<artifact>\" --summary",
      download_fallback: "npx -y deepbom@latest",
      detail_strategy: "Request one section, JSON Pointer, or evidence envelope only after reading the bounded summary.",
      default_output: cliCapabilities.default_audit_output,
    },
    inputs: cliCapabilities.inputs,
    outputs: cliCapabilities.output_contracts,
    evidence_contract: {
      finding_kinds: cliCapabilities.automation.finding_kinds,
      applicability_states: ["NOT_APPLICABLE", "NOT_ASSESSABLE", "NOT_ASSESSED_YET"],
      required_answer_fields: [
        "artifact_filename",
        "artifact_sha256",
        "artifact_format",
        "analyzer_version",
        "target_and_binding_source_when_applicable",
        "assessment_coverage",
        "defects_cautions_and_evidence_gaps_separately",
        "reproduction_command",
      ],
    },
    execution_boundary: {
      analysis_location: "local_process",
      hosted_analysis_endpoint: false,
      artifact_bytes_network_transfer: false,
      remote_inputs_require_immutable_identity: true,
      runtime_measurement_inferred: false,
      optional_browser_runtime_measurement: "separate_web_workspace_path_not_part_of_static_cli_or_agent_audit",
    },
    local_integrations: {
      codex: { path: ".agents/skills/deepbom", automatic_matching: true },
      claude_code: { path: ".claude/skills/deepbom", automatic_matching: true },
      claude_desktop: {
        distribution: "MCPB local desktop extension",
        release_asset: `https://github.com/JunHwan-Kwon/deepbom/releases/download/channels-v${version}/deepbom-${version}.mcpb`,
        automatic_matching: "host_dependent_after_installation",
      },
      generic: { path: "skills/deepbom", automatic_matching: "host_dependent" },
      install_preview: "deepbom integrate <codex|claude-code|generic>",
      install_apply: "deepbom integrate <codex|claude-code|generic> --apply",
    },
    chatgpt_integration: {
      status: "developer_mode_ready_public_listing_requires_openai_plugin_review",
      endpoint: "https://deepbom.org/mcp",
      transport: "streamable_http",
      primary_tool: "deepbom_analyze_file",
      input: "one ChatGPT-authorized attachment",
      execution_location: "chatgpt_browser_sandbox",
      result: "bounded hash-bound static evidence returned to the conversation",
      transfer_boundary: "the DEEPBOM service does not fetch or retain model bytes",
      use_local_instead_when: [
        "the artifact or result fields are confidential",
        "the artifact exceeds the browser or ChatGPT attachment boundary",
        "the review needs package-directory, sharded-repository, or ONNX external-data closure",
        "the workflow needs a complete evidence export or repeated automation",
      ],
    },
    discovery: {
      agent_page: "https://deepbom.org/for-agents/",
      chatgpt_page: "https://deepbom.org/chatgpt/",
      chatgpt_mcp: "https://deepbom.org/mcp",
      machine_contract: "https://deepbom.org/agent-capabilities.json",
      plain_text: "https://deepbom.org/llms.txt",
      skill_source: "https://github.com/JunHwan-Kwon/deepbom/tree/main/skills/deepbom",
      portable_plugin_manifest: "https://github.com/JunHwan-Kwon/deepbom/blob/main/plugin.json",
      problem_guides: "https://deepbom.org/guides/",
      desktop_extension_release: `https://github.com/JunHwan-Kwon/deepbom/releases/tag/channels-v${version}`,
    },
  };
}
