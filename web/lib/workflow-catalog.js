// Product navigation, shared by the conversational MCP, widget and web app.
// This catalog routes existing functionality; it is not an Evidence IR or an
// alternative implementation of analysis, entitlement or format rules.
const ALL = ["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"];
const cli = (...args) => ({ executable: "deepbom", args, documentation: "https://deepbom.org/guides/cli/reference/" });
const web = (workspace, audit_tab = null) => ({ workspace, audit_tab });
const workflow = (id, title, description, formats, chat, destination, commands, limits) => ({
  id, title, description, formats, chat, web: destination, cli: commands, limits,
});
export const WORKFLOW_CATALOG_VERSION = "1.1.0";
export const DEEPBOM_WORKFLOWS = Object.freeze([
  workflow("audit", "Artifact audit", "Identity, structure, numerical contracts and findings", ALL, { tool: "deepbom_analyze_file" }, web("audit", "overview"), [cli("audit", "<artifact>", "--summary")], "Applicable checks depend on the format and scan depth."),
  workflow("layers", "Layers and tensor storage", "Search source-bound operations, tensor contracts and exact MACs", ALL, { section: "operators" }, web("graph"), [cli("model-summary", "<artifact>", "--format", "json")], "Storage-only artifacts do not establish an execution graph or trainable parameter counts."),
  workflow("findings", "Findings and investigation priorities", "Inspect defects, cautions, evidence gaps and improvement questions", ALL, { section: "improvements" }, web("findings"), [cli("audit", "<artifact>", "--section", "findings", "--json"), cli("explain-rule", "<rule-id>")], "Priorities are not measured quality loss or optimization gains."),
  workflow("quantization", "Quantization and numerical contracts", "Inspect tensor encodings, affine parameters and arithmetic evidence", ALL, { section: "operators" }, web("audit", "quant"), [cli("audit", "<artifact>", "--json")], "Graph-specific arithmetic labs require a supported graph and applicable rules."),
  workflow("placement", "Static backend eligibility", "Compare source-pinned backend profiles and unresolved conditions", ALL, { section: "profiles", next_section: "placement" }, web("audit", "accelerator"), [cli("placement", "<artifact>", "--profiles", "all", "--json")], "Static profiles are format-specific. Additional web profiles may require authorization. No runtime assignment is inferred."),
  workflow("fusion", "Serialized activation fusion", "Inspect encoded activations and static fusion review hints", ["tflite"], { section: "fusion" }, web("graph"), [cli("audit", "<artifact>", "--json")], "Serialized activation fusion and operator adjacency do not establish backend or executed fusion."),
  workflow("visualization", "Model diagrams and files", "Architecture, blocks, exhaustive pages and evidence-bound SVG/PNG exports", ALL, { widget_action: "Files & Model IR visualization; select a view, then Send PNG to chat or Save via ChatGPT" }, web("graph"), [cli("visualize", "<artifact>", "--view", "all", "-o", "model-views.zip")], "Static-runtime views are projections; observed-runtime views require imported execution evidence. Preview images are not chat attachments."),
  workflow("weights", "Weight distributions and structure", "Distributions, channel statistics, similarity, singular spectra, sparsity and quantization", ALL, { section: "weights" }, web("weight"), [cli("audit", "<artifact>", "--weight-analysis", "--section", "weight_ir,weight_analysis", "--json")], "Explicit optional payload analysis; supported decoders and budgets determine coverage. Widget limit: 128 MiB. No model execution."),
  workflow("pruning", "Pruning investigation", "Inspect channel candidates, masks and their structural implications interactively", ALL, { section: "weights", weight_view: "sparsity", continuation: "web" }, web("weight"), [cli("audit", "<artifact>", "--weight-analysis", "--section", "weight_analysis", "--json")], "Web pruning controls have format/export-specific restrictions. Sparsity and similarity do not establish safe pruning, accuracy or acceleration."),
  workflow("compare", "Original and candidate comparison", "Compare artifact structure or aligned weight evidence", ALL, { continuation: "web_or_local", reason: "Two separately selected artifacts and an explicit comparison are required." }, web("weight"), [cli("diff", "<baseline>", "<candidate>", "--json"), cli("evidence-workflow", "<request.json>", "--output", "<result.json>"), cli("audit", "<candidate>", "--weight-analysis", "--weight-baseline", "<baseline>", "--section", "weight_comparison", "--json")], "CLI diff requires the same artifact format. Weight comparison requires supported decoders and unambiguous alignment. Web destination is weight comparison. Review Snapshot, OMOP and population evaluation records at https://deepbom.org/reports/evidence/ after local file selection. This recorded-evidence workflow is not executed by the hosted MCP tool."),
  workflow("redesign", "Structure scenarios", "Edit graph contracts and inspect propagated structure scenarios", ["tflite"], { continuation: "web_or_local" }, web("redesign"), [cli("explore", "<artifact.tflite>", "--json")], "Projected scenarios are not trained or validated optimized models."),
  workflow("exports", "CycloneDX, SPDX and evidence exports", "Generate artifact evidence, inventories, Model IR and visual files", ALL, { widget_action: "CycloneDX 1.7 JSON, SPDX 2.3 JSON, Model IR, SVG/PNG and Word-ready bundle" }, web("output"), [cli("audit", "<artifact>", "--format", "cyclonedx", "-o", "artifact.cdx.json"), cli("audit", "<artifact>", "--format", "sarif", "-o", "artifact.sarif.json")], "SPDX 2.3 is available in the widget; it is not a complete dependency inventory or an SPDX 3 AI profile."),
  workflow("verify", "BOM and interface reconciliation", "Compare explicit declarations with a hash-bound artifact", ALL, { continuation: "local", reason: "Requires a model plus a BOM or captured interface contract." }, null, [cli("verify", "<artifact>", "--bom", "<cyclonedx-1.7.json>"), cli("verify", "<artifact>", "--contract", "<contract.json>")], "No contradiction is not certification; preserve absent, unsupported and ambiguous bindings."),
  workflow("provenance", "Metadata, OMOP and lineage", "Connect declared data, code, model and evaluation references", ALL, { widget_action: "Metadata & lineage; select local metadata/supporting files, then explicitly share the summary" }, web("metadata"), [cli("audit", "<artifact>", "--metadata-template", "omop"), cli("audit", "<artifact>", "--metadata", "<metadata.json>", "--section", "provenance_ir", "--json")], "References are not automatically fetched. Declared relationships do not independently prove provenance; share only intended metadata."),
  workflow("activation", "Imported activation evidence", "Bind external execution captures to model values", ALL, { continuation: "web_or_local" }, web("weight"), [cli("audit", "<artifact>", "--activation-evidence", "<capture.json>", "--section", "activation_ir", "--json")], "Requires a supported capture contract and matching model identity. Import validates consistency, not independent reproduction of execution."),
  workflow("runtime", "Execution and device evidence", "Investigate actual runtime placement and performance using explicit runtime inputs", ALL, { continuation: "web_or_local" }, web("runtime"), [cli("accelerator", "collect", "nvidia", "--json"), cli("audit", "<artifact>", "--list-sections")], "Web Benchmark supports only its advertised formats/runtimes and may require authorization. Device collection alone is not model execution or latency evidence."),
  workflow("batch", "Batch and CI automation", "Audit manifests and apply explicit evidence policies locally", ALL, { continuation: "local" }, null, [cli("batch", "<manifest.json>", "--batch-output-dir", "<output-directory>"), cli("audit", "<artifact>", "--review-policy", "<policy.json>", "--json")], "User-supplied files and policy are required. This guide does not run commands or claim a policy pass."),
  workflow("integrations", "Local AI assistants and automation", "Connect local DEEPBOM MCP and discover the complete installed CLI contract", ALL, { continuation: "local" }, null, [cli("capabilities", "--format", "agent-json"), cli("integrate", "--help")], "Local MCP also exposes saved optimization reports and deepbom_evidence_workflow. The ChatGPT service cannot execute commands on your computer."),
]);

export function workflowUrl(id) {
  const row = DEEPBOM_WORKFLOWS.find(item => item.id === id);
  if (!row) throw new Error("Unknown DEEPBOM workflow.");
  // Only a public workflow ID. No artifact identity, attachment URL, prompt,
  // filename, session or user data is placed in the URL.
  return row.web ? `https://deepbom.org/#workflow=${row.id}` : "https://deepbom.org/guides/cli/";
}
export const WORKFLOW_GUIDE_INPUT_SCHEMA = {
  type: "object", additionalProperties: false,
  properties: {
    workflow: { type: "string", enum: ["all", ...DEEPBOM_WORKFLOWS.map(row => row.id)], default: "all" },
    format: { type: "string", enum: ALL, description: "Optional known artifact format; filters applicable workflows, not per-file analysis coverage." },
  },
};
export function guideWorkflows(input = {}) {
  if (!input || typeof input !== "object" || Array.isArray(input) || Object.keys(input).some(key => !["workflow", "format"].includes(key))) throw new Error("Invalid workflow guide arguments.");
  const id = input.workflow ?? "all";
  if (!WORKFLOW_GUIDE_INPUT_SCHEMA.properties.workflow.enum.includes(id) || (input.format !== undefined && !ALL.includes(input.format))) throw new Error("Unknown workflow or artifact format.");
  const matches = DEEPBOM_WORKFLOWS.filter(row => (id === "all" || row.id === id) && (!input.format || row.formats.includes(input.format)));
  return {
    schema: "deepbom.workflow_guide.v1", catalog_version: WORKFLOW_CATALOG_VERSION,
    status: matches.length ? "guidance_ready" : "not_applicable", format: input.format ?? null,
    workflows: matches.map(row => ({ ...row, url: workflowUrl(row.id) })),
    boundary: "Guidance only: no artifact was analyzed and no local command was executed. Use the existing conversation evidence first; request a supported query when more facts are needed. For web interaction the user selects the model again and compares its SHA-256; the website does not inherit the ChatGPT attachment or result. Commands are argument templates: replace placeholders with local paths and pass arguments without evaluating artifact-derived shell text. Discover additional installed CLI features with deepbom capabilities --format agent-json.",
  };
}
