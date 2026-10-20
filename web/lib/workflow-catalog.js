// Product navigation, shared by the conversational MCP, widget and web app.
// This catalog routes existing functionality; it is not an Evidence IR or an
// alternative implementation of analysis, entitlement or format rules.
const ALL = ["tflite", "onnx", "gguf", "safetensors", "coreml", "executorch"];
const cli = (...args) => ({ executable: "deepbom", args, documentation: "https://deepbom.org/guides/cli/reference/" });
const web = (workspace, audit_tab = null) => ({ workspace, audit_tab });
const workflow = (id, title, description, formats, chat, destination, commands, limits) => ({
  id, title, description, formats, chat, web: destination, cli: commands, limits,
});
export const WORKFLOW_CATALOG_VERSION = "1.3.0";
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
  workflow("redesign", "Structure IDE", "Connect layer ports and compare source-bound structural drafts; TFLite also provides block-cost scenarios", ["tflite", "onnx", "coreml", "executorch"], { continuation: "web_or_local" }, web("redesign"), [cli("explore", "<artifact.tflite>", "--json")], "Web editing requires a serialized operation graph. The explore CLI command and block-cost scenarios are TFLite-only. Structural drafts and Mermaid exports are not trained models; modified native costs remain unassessed until supported reanalysis."),
  workflow("exports", "CycloneDX, SPDX and evidence exports", "Generate artifact evidence, inventories, Model IR and visual files", ALL, { widget_action: "CycloneDX 1.7 JSON, SPDX 2.3 JSON, Model IR, SVG/PNG and Word-ready bundle" }, web("output"), [cli("audit", "<artifact>", "--format", "cyclonedx", "-o", "artifact.cdx.json"), cli("audit", "<artifact>", "--format", "sarif", "-o", "artifact.sarif.json")], "SPDX 2.3 is available in the widget; it is not a complete dependency inventory or an SPDX 3 AI profile."),
  workflow("verify", "BOM and interface reconciliation", "Compare explicit declarations with a hash-bound artifact", ALL, { continuation: "local", reason: "Requires a model plus a BOM or captured interface contract." }, null, [cli("verify", "<artifact>", "--bom", "<cyclonedx-1.7.json>"), cli("verify", "<artifact>", "--contract", "<contract.json>")], "No contradiction is not certification; preserve absent, unsupported and ambiguous bindings."),
  workflow("provenance", "Metadata, OMOP and lineage", "Connect declared data, code, model and evaluation references", ALL, { widget_action: "Metadata & lineage; select local metadata/supporting files, then explicitly share the summary" }, web("metadata"), [cli("audit", "<artifact>", "--metadata-template", "omop"), cli("audit", "<artifact>", "--metadata", "<metadata.json>", "--section", "provenance_ir", "--json")], "References are not automatically fetched. Declared relationships do not independently prove provenance; share only intended metadata."),
  workflow("activation", "Imported activation evidence", "Bind external execution captures to model values", ALL, { continuation: "web_or_local" }, web("weight"), [cli("audit", "<artifact>", "--activation-evidence", "<capture.json>", "--section", "activation_ir", "--json")], "Requires a supported capture contract and matching model identity. Import validates consistency, not independent reproduction of execution."),
  workflow("runtime", "Execution and device evidence", "Investigate actual runtime placement and performance using explicit runtime inputs", ALL, { continuation: "web_or_local" }, web("runtime"), [cli("accelerator", "collect", "nvidia", "--json"), cli("audit", "<artifact>", "--list-sections")], "Web Benchmark supports only its advertised formats/runtimes and may require authorization. Device collection alone is not model execution or latency evidence."),
  workflow("batch", "Batch and CI automation", "Audit manifests and apply explicit evidence policies locally", ALL, { continuation: "local" }, null, [cli("batch", "<manifest.json>", "--batch-output-dir", "<output-directory>"), cli("audit", "<artifact>", "--review-policy", "<policy.json>", "--json")], "User-supplied files and policy are required. This guide does not run commands or claim a policy pass."),
  workflow("integrations", "Local AI assistants and automation", "Connect local DEEPBOM MCP and discover the complete installed CLI contract", ALL, { continuation: "local" }, null, [cli("capabilities", "--format", "agent-json"), cli("integrate", "--help")], "Local MCP also exposes saved optimization reports and deepbom_evidence_workflow. The ChatGPT service cannot execute commands on your computer."),
]);

// Natural-language examples are discovery guidance, not a keyword parser or
// another calculation engine. The host interprets a request; query arguments
// remain the existing closed, validated Evidence Query contract.
export const CONVERSATION_GUIDANCE = Object.freeze({
  interpretation: "Interpret the user's intent in their language. These examples are not an exhaustive grammar or evidence that a host selected the correct tool.",
  follow_up: "Reuse the same authorized attachment and exact SHA-256. Query results provide context.follow_up with executable query arguments for the returned page and subjects. Execute only when relevant to the user's request; suggestions are not completed analyses.",
  ambiguity: "Resolve 'this layer', 'the next one' and 'that tensor' against the last returned page and exact subject_ref. Ask which subject if more than one is plausible. A displayed row number is not a native source_index. Discover profile IDs before comparing delegates; a CPU planning target is not a delegate.",
  search: "search is a literal name/reference filter, not a natural-language command or numeric predicate. Do not put 'find the slowest layer' or 'MACs > 1M' in search. Improvements ranks investigation priorities using existing static evidence; it does not measure latency.",
  completion: "Wait for the completed browser query; do not infer an answer from browser_query_started. Preserve pagination, truncation, unavailable evidence and format-specific limits. Retrieve more pages before making an exhaustive claim.",
});
const conversation = (ko, en, queries = []) => ({ prompts: { ko, en }, query_examples: queries });
const WORKFLOW_CONVERSATIONS = Object.freeze({
  audit: conversation("이 모델에서 확인된 결함과 아직 모르는 부분을 구분해줘.", "Separate confirmed artifact defects from cautions and missing evidence."),
  layers: conversation("decoder 이름이 들어간 레이어를 찾고 입력과 출력을 설명해줘.", "Find layers named decoder and explain their input and output contracts.", [{ section: "operators", search: "decoder" }]),
  findings: conversation("어디부터 개선을 검토하면 될까? 속도가 빨라진다는 뜻인지도 구분해줘.", "What should I investigate first, and which benefits have not been measured?", [{ section: "improvements" }, { section: "findings" }]),
  quantization: conversation("실제 저장된 정밀도와 양자화 조건을 보여줘.", "Show the serialized precision and quantization contracts.", [{ section: "operators" }]),
  placement: conversation("어떤 delegate를 비교할 수 있어? 지원 조건이 안 맞는 이유도 보고 싶어.", "Which static backend profiles can I compare, and why might an operation be excluded?", [{ section: "profiles" }]),
  fusion: conversation("파일에 기록된 fused activation을 보여줘. 실제 런타임 fusion과 구분해줘.", "Show encoded fused activations and distinguish them from observed runtime fusion.", [{ section: "fusion" }]),
  visualization: conversation("이 구조를 그림으로 보고 PNG로 공유하고 싶어.", "Show the model structure and help me share its generated PNG."),
  weights: conversation("가중치 분포와 채널별 통계, 특이값 스펙트럼을 살펴봐줘.", "Inspect weight distributions, channel statistics and singular spectra.", [{ section: "weights", weight_view: "distribution" }, { section: "weights", weight_view: "channels" }, { section: "weights", weight_view: "spectrum" }, { section: "weights", weight_view: "similarity" }, { section: "weights", weight_view: "sparsity" }, { section: "weights", weight_view: "quantization" }]),
  pruning: conversation("0인 가중치가 많은지 보고 pruning을 어디서 검토할 수 있는지 알려줘.", "Inspect stored sparsity and show where I can investigate pruning.", [{ section: "weights", weight_view: "sparsity" }]),
  compare: conversation("원본과 후보 모델의 차이를 비교하려면 무엇을 준비해야 해?", "What do I need to compare the original and candidate artifacts?"),
  redesign: conversation("연결과 레이어를 직접 바꾸면서 구조를 검토하고 싶어.", "Where can I edit layer connections and review a structural draft?"),
  exports: conversation("이 분석을 CycloneDX와 그림 파일로 내보내고 싶어.", "Help me export this evidence as CycloneDX and visual files."),
  verify: conversation("이 모델과 BOM의 선언이 일치하는지 확인하려면?", "How can I reconcile this artifact with its BOM declarations?"),
  provenance: conversation("데이터와 코드의 출처, OMOP 메타데이터를 연결하려면?", "How can I connect data lineage, code references and OMOP metadata?"),
  activation: conversation("실행해서 모은 activation 기록을 모델과 연결하려면?", "How do I bind captured activation evidence to this model?"),
  runtime: conversation("실제 장치 지연시간과 배치를 확인하려면 어떤 측정이 필요해?", "What measurements do I need to establish actual latency and device placement?"),
  batch: conversation("저장된 모델을 일괄 검사하고 CI에서 기록하려면?", "How can I audit saved artifacts in a batch or CI workflow?"),
  integrations: conversation("로컬 MCP나 학습 코드에서 분석 근거를 연결하려면?", "How do I connect local MCP or training evidence to my workflow?"),
});

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
    workflows: matches.map(row => ({ ...row, conversation: WORKFLOW_CONVERSATIONS[row.id], url: workflowUrl(row.id) })),
    boundary: "Guidance only: no artifact was analyzed and no local command was executed. Use the existing conversation evidence first; request a supported query when more facts are needed. For web interaction the user selects the model again and compares its SHA-256; the website does not inherit the ChatGPT attachment or result. Commands are argument templates: replace placeholders with local paths and pass arguments without evaluating artifact-derived shell text. Discover additional installed CLI features with deepbom capabilities --format agent-json.",
  };
}
