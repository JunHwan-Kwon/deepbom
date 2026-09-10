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
    purpose: "Local evidence-bounded static analysis of serialized AI deployment artifacts.",
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
      discovery: `npx -y ${packageSpec} capabilities --format agent-json`,
      self_test: `npx -y ${packageSpec} self-test --compact`,
      audit_template: `npx -y ${packageSpec} audit \"<artifact>\" --summary`,
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
    },
    local_integrations: {
      codex: { path: ".agents/skills/deepbom", automatic_matching: true },
      claude_code: { path: ".claude/skills/deepbom", automatic_matching: true },
      claude_desktop: {
        distribution: "MCPB local desktop extension",
        release_asset: `https://github.com/JunHwan-Kwon/deepbom/releases/download/channels-v${version}/deepbom-${version}.mcpb`,
        automatic_matching: false,
      },
      generic: { path: "skills/deepbom", automatic_matching: "host_dependent" },
      install_preview: "deepbom integrate <codex|claude-code|generic>",
      install_apply: "deepbom integrate <codex|claude-code|generic> --apply",
    },
    discovery: {
      agent_page: "https://deepbom.org/for-agents/",
      machine_contract: "https://deepbom.org/agent-capabilities.json",
      plain_text: "https://deepbom.org/llms.txt",
      skill_source: "https://github.com/JunHwan-Kwon/deepbom/tree/main/skills/deepbom",
      desktop_extension_release: `https://github.com/JunHwan-Kwon/deepbom/releases/tag/channels-v${version}`,
    },
  };
}
