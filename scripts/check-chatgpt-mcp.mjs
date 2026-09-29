import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { ANALYZER_SEMANTIC_VERSION } from "../web/lib/app-config.js";
import { CHATGPT_MCP_CONTRACT, routeChatGptMcp } from "../worker/chatgpt-mcp.js";
import Ajv from "ajv";
import { CHATGPT_RESULT_V1, CHATGPT_RESULT_V2, conversationTransport, publishConversationResult } from "../web/lib/chatgpt-result-transport.js";

const endpoint = "https://deepbom.org/mcp";
let id = 0;

const initialize = await rpc("initialize", {
  protocolVersion: "2025-11-25",
  capabilities: {},
  clientInfo: { name: "deepbom-check", version: "1" },
});
assert.equal(initialize.protocolVersion, "2025-11-25");
assert.equal(initialize.serverInfo.version, ANALYZER_SEMANTIC_VERSION);
assert.match(initialize.instructions, /browser sandbox/i);
assert.match(initialize.instructions, /not model bytes/i);
assert.match(initialize.instructions, /artifact_defect, caution, and evidence_gap/);
assert.match(initialize.instructions, /deterministic Model IR views/i);
assert.match(initialize.instructions, /do not reconstruct a graph from the bounded conversation summary/i);
for (const protocolVersion of CHATGPT_MCP_CONTRACT.supportedProtocolVersions) {
  const response = await rawRpcResponse("initialize", {
    protocolVersion,
    capabilities: {},
    clientInfo: { name: "deepbom-version-check", version: "1" },
  });
  const negotiated = await response.json();
  assert.equal(response.headers.get("mcp-protocol-version"), protocolVersion);
  assert.equal(negotiated.error, undefined);
  assert.equal(negotiated.result.protocolVersion, protocolVersion);
}

const { tools } = await rpc("tools/list");
assert.deepEqual(tools.map((tool) => tool.name), [
  "deepbom_analyze_file",
  "deepbom_capabilities",
  "deepbom_publish_analysis",
  "deepbom_publish_error",
]);
const analyze = tools[0];
assert.deepEqual(analyze._meta["openai/fileParams"], ["file"]);
assert.equal(analyze._meta.ui.resourceUri, CHATGPT_MCP_CONTRACT.widgetUri);
assert.equal(analyze._meta["openai/toolInvocation/invoked"], "Browser analysis opened");
assert.equal(analyze.annotations.readOnlyHint, true);
assert.equal(analyze.annotations.destructiveHint, false);
assert.equal(analyze.annotations.openWorldHint, false);
const fileSchema = analyze.inputSchema.$defs.OpenAIFile;
assert.deepEqual(fileSchema.required, ["download_url", "file_id"]);
assert.deepEqual(Object.keys(fileSchema.properties), ["download_url", "file_id", "mime_type", "file_name"]);
assert.equal(tools[2]._meta.ui.visibility[0], "app");
assert.equal(tools[2]._meta["openai/visibility"], "private");
assert.equal(tools[3]._meta.ui.visibility[0], "app");
assert.equal(tools[3]._meta["openai/visibility"], "private");

const { resources } = await rpc("resources/list");
assert.equal(resources.length, 1);
assert.equal(resources[0].uri, CHATGPT_MCP_CONTRACT.widgetUri);
const read = await rpc("resources/read", { uri: CHATGPT_MCP_CONTRACT.widgetUri });
assert.equal(read.contents[0].mimeType, "text/html;profile=mcp-app");
assert.match(read.contents[0].text, /deepbom-widget\.js/);
assert.equal(CHATGPT_MCP_CONTRACT.widgetUri, "ui://deepbom/analyzer-v2.html");
assert(read.contents[0].text.includes(`deepbom-widget.js?v=${ANALYZER_SEMANTIC_VERSION}-20260929.2`));
assert.equal(read.contents[0]._meta.ui.domain, "https://deepbom.org");
assert.ok(read.contents[0]._meta.ui.csp.connectDomains.some((domain) => domain.includes("oaiusercontent")));
assert.match(read.contents[0]._meta["openai/widgetDescription"], /format-neutral Model IR table/);
assert.match(read.contents[0]._meta["openai/widgetDescription"], /deterministic Model IR views/);
assert.deepEqual(CHATGPT_MCP_CONTRACT.legacyWidgetUris, ["ui://deepbom/analyzer.html"]);
const legacyRead = await rpc("resources/read", { uri: CHATGPT_MCP_CONTRACT.legacyWidgetUris[0] });
assert.equal(legacyRead.contents[0].uri, CHATGPT_MCP_CONTRACT.legacyWidgetUris[0]);
assert.equal(legacyRead.contents[0].text, read.contents[0].text);
assert.match(legacyRead.contents[0]._meta["openai/widgetDescription"], /format-neutral Model IR table/);
assert.match(legacyRead.contents[0]._meta["openai/widgetDescription"], /deterministic Model IR views/);

const file = {
  download_url: "https://files.oaiusercontent.com/example/model.onnx",
  file_id: "file_example",
  mime_type: "application/octet-stream",
  file_name: "model.onnx",
};
const started = await rpc("tools/call", {
  name: "deepbom_analyze_file",
  arguments: { file },
});
assert.equal(started.structuredContent.status, "browser_analysis_started");
assert.equal(started.structuredContent.analysis_depth, "structure");
assert.match(started.structuredContent.privacy, /does not fetch or retain model bytes/);
assert.equal(started._meta["openai/file"].download_url, file.download_url);
assert.match(started.content[0].text, /does not establish ongoing processing or successful analysis/);
assert.match(started.content[0].text, /Analysis could not be completed, acknowledge the failure/);
assert.match(started.content[0].text, /No artifact facts are available from this starting response/);
assert.match(started.content[0].text, /user must select Report in chat once/);
assert.match(started.content[0].text, /Do not substitute another parser/);

const capabilities = await rpc("tools/call", { name: "deepbom_capabilities", arguments: {} });
assert.equal(capabilities.structuredContent.version, ANALYZER_SEMANTIC_VERSION);
assert.equal(capabilities.structuredContent.chatgpt_path.execution, "browser sandbox");
assert.match(capabilities.structuredContent.chatgpt_path.output, /SVG\/PNG\/document views/);
assert.match(capabilities.structuredContent.local_path.invocation, /deepbom@.+ mcp/);

const validResult = {
  schema: "deepbom.chatgpt_analysis_result.v2",
  analyzer_version: ANALYZER_SEMANTIC_VERSION,
  analysis_location: "chatgpt_browser_sandbox",
  artifact: {
    filename: "model.onnx",
    format: "onnx",
    sha256: "a".repeat(64),
    byte_length: 1024,
    artifact_ir_sha256: "b".repeat(64),
  },
  verdict: {
    status: "no_artifact_defect_observed",
    artifact_defect_count: 0,
    caution_count: 1,
    evidence_needed_count: 2,
  },
  graph: { operator_count: 2, tensor_count: 3, total_macs: 64, mac_confidence: "exact" },
  storage: null,
  quantization: { classification: "static_qdq_representation", max_risk: "none" },
  model_summary: boundedModelSummary(),
  findings: { artifact_defects: [], cautions: [], evidence_needed: [], truncated: false },
  evidence_boundary: "Static checks only.",
  transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
  reproduction: null,
};
const published = await rpc("tools/call", {
  name: "deepbom_publish_analysis",
  arguments: { result: validResult },
});
assert.deepEqual(published.structuredContent, validResult);
const historical = JSON.parse(await readFile("scripts/fixtures/chatgpt-tools-20260916.json", "utf8"));
const ajv = new Ajv({ strict: false, allowUnionTypes: true });
for (const contract of [historical.tools, tools]) {
  for (const [name, response] of [["deepbom_capabilities", capabilities], ["deepbom_analyze_file", started]]) {
    const validate = ajv.compile(contract.find(tool => tool.name === name).outputSchema);
    assert(validate(response.structuredContent), JSON.stringify(validate.errors));
  }
}
const liveBridge = tools.find(tool => tool.name === "deepbom_publish_analysis");
const historicalBridge = historical.tools.find(tool => tool.name === liveBridge.name);
const v2OnlyBridge = { ...liveBridge, inputSchema: { ...liveBridge.inputSchema, properties: { result: liveBridge.outputSchema.anyOf[1] } }, outputSchema: liveBridge.outputSchema.anyOf[1] };
for (const [label, contract, expectedSchema, expectedCalls] of [["historical", historicalBridge, CHATGPT_RESULT_V1, 1], ["v2 cached", v2OnlyBridge, CHATGPT_RESULT_V2, 2], ["updated", liveBridge, CHATGPT_RESULT_V1, 1]]) {
  const validateInput = ajv.compile(contract.inputSchema), validateOutput = ajv.compile(contract.outputSchema);
  let calls = 0;
  const bridge = { async callTool(name, args) {
    calls++;
    if (!validateInput(args)) throw new Error("Input schema validation failed: " + JSON.stringify(validateInput.errors));
    const response = await rpc("tools/call", { name, arguments: args });
    assert(validateOutput(response.structuredContent), JSON.stringify(validateOutput.errors));
    return response;
  } };
  const response = await publishConversationResult(bridge, validResult);
  assert.deepEqual(response, { ...validResult, schema: expectedSchema }, label);
  assert.equal(calls, expectedCalls, label);
}
for (const schema of [CHATGPT_RESULT_V1, CHATGPT_RESULT_V2]) {
  const cached = { ...validResult, schema, analyzer_version: "1.101.0" };
  const response = await rpc("tools/call", { name: "deepbom_publish_analysis", arguments: { result: cached } });
  assert.deepEqual(response.structuredContent, cached, "Preserve the declared producer version; never relabel cached evidence.");
}
for (const badVersion of [null, "", "2.0.0<script>", "02.0.0", "x".repeat(65)]) {
  const response = await rawRpc("tools/call", { name: "deepbom_publish_analysis", arguments: { result: { ...validResult, analyzer_version: badVersion } } });
  assert.equal(response.error.code, -32602);
}
for (const failure of [new Error("Network timeout"), new Error("Not authorized"), { isError: true, content: [{ type: "text", text: "Tool unavailable" }] }, {}, { structuredContent: { ...validResult, schema: CHATGPT_RESULT_V1, artifact: { ...validResult.artifact, sha256: "f".repeat(64) } } }]) {
  let calls = 0;
  await assert.rejects(publishConversationResult({ async callTool() { calls++; if (failure instanceof Error) throw failure; return failure; } }, validResult));
  assert.equal(calls, 1, "Non-schema failures must neither retry nor claim successful reporting.");
}
const linkSummary = { schema: "deepbom.provenance_summary.v1", artifact_sha256: validResult.artifact.sha256, model_ir_sha256: "b".repeat(64), provenance_ir_sha256: "c".repeat(64), status: "incomplete", node_count: 2, relationship_count: 1, observed_file_count: 0, check_count: 2, check_counts: { match: 1, mismatch: 0, not_assessed: 1, unresolved: 0, unsupported: 0 }, field_count: 1, mapped_declared_field_count: 1, unsupported_field_count: 0, attested_relationship_count: 0, metadata_truth_verified: false, publisher_authenticity: "not_verified" };
const withLinks = { ...validResult, model_summary: { ...validResult.model_summary, model_ir_sha256: "b".repeat(64) }, provenance_summary: linkSummary };
const publishedLinks = await rpc("tools/call", { name: "deepbom_publish_analysis", arguments: { result: withLinks } });
assert.deepEqual(publishedLinks.structuredContent.provenance_summary, linkSummary);
assert.match(publishedLinks.content[0].text, /Relationship truth and publisher authenticity remain unverified/);
assert.throws(() => conversationTransport(withLinks, CHATGPT_RESULT_V1), /cannot be omitted or renamed/);
let metadataCalls = 0;
await assert.rejects(publishConversationResult({ async callTool() { metadataCalls++; throw new Error("Input schema validation failed"); } }, withLinks), /Save the Provenance IR JSON/);
assert.equal(metadataCalls, 1, "Unsupported provenance must not be silently dropped into v1.");
assert.deepEqual(await publishConversationResult({ callTool: (name, args) => rpc("tools/call", { name, arguments: args }) }, withLinks), withLinks);
for (const retired of [
  { ...withLinks, schema: "deepbom.chatgpt_analysis_result.v1" },
  { ...withLinks, provenance_summary: { ...linkSummary, schema: "deepbom.evidence_link_summary.v1" } },
  { ...validResult, evidence_links: linkSummary },
  { ...withLinks, provenance_summary: { ...linkSummary, evidence_link_ir_sha256: linkSummary.provenance_ir_sha256 } },
]) {
  const rejected = await rawRpc("tools/call", { name: "deepbom_publish_analysis", arguments: { result: retired } });
  assert.equal(rejected.error.code, -32602, "retired remote provenance contracts are rejected");
}


assert.match(published.content[0].text, /0 artifact defect\(s\), 1 caution\(s\), and 2 evidence gap\(s\)/);
assert.match(published.content[0].text, /2 operation row\(s\)/);

const invalid = await rawRpc("tools/call", {
  name: "deepbom_publish_analysis",
  arguments: { result: { ...validResult, transfer_boundary: "model_uploaded" } },
});
assert.equal(invalid.error.code, -32602);
assert.match(invalid.error.message, /transfer boundary/i);

const errorResult = {
  schema: "deepbom.chatgpt_error.v1",
  analyzer_version: ANALYZER_SEMANTIC_VERSION,
  analysis_location: "chatgpt_browser_sandbox",
  code: "artifact_parse_failed",
  message: "The serialized payload is truncated.",
  suggested_action: "Re-download the artifact and retry.",
  file_name: "broken.onnx",
  transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
};
const publishedError = await rpc("tools/call", {
  name: "deepbom_publish_error",
  arguments: { error: errorResult },
});
assert.deepEqual(publishedError.structuredContent, errorResult);
const cachedError = { ...errorResult, analyzer_version: "1.101.0" };
const acknowledgedCachedError = await rpc("tools/call", { name: "deepbom_publish_error", arguments: { error: cachedError } });
assert.deepEqual(acknowledgedCachedError.structuredContent, cachedError);
assert(ajv.compile(historical.tools.find(tool => tool.name === "deepbom_publish_error").outputSchema)(acknowledgedCachedError.structuredContent));
assert.match(publishedError.content[0].text, /Suggested action: Re-download/);

const oversizedBatch = Array.from({ length: CHATGPT_MCP_CONTRACT.maxBatchItems + 1 }, (_, index) => ({
  jsonrpc: "2.0", id: index + 100, method: "ping",
}));
const batchResponse = await routeChatGptMcp(new Request(endpoint, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(oversizedBatch),
}));
assert.equal((await batchResponse.json()).error.code, -32002);

const oversizedFrame = await routeChatGptMcp(new Request(endpoint, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ jsonrpc: "2.0", id: 999, method: "ping", padding: "x".repeat(CHATGPT_MCP_CONTRACT.maxRequestBytes) }),
}));
assert.equal((await oversizedFrame.json()).error.code, -32001);

const notification = await routeChatGptMcp(new Request(endpoint, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
}));
assert.equal(notification.status, 202);
assert.equal(await notification.text(), "");

const widget = await readFile("web/chatgpt/deepbom-widget.js", "utf8");
for (const contract of [
  "window.openai",
  "getFileDownloadUrl",
  "publishConversationResult",
  "deepbom_publish_error",
  "sendFollowUpMessage",
  "Report in chat",
  "Model IR visualization",
  "Download SVG",
  "Download PNG",
  "Word-ready bundle",
  "Send PNG to chat",
  "buildModelIrVisualizationBundle",
  "rasterizeMonochromeModelView",
  "Format-neutral model summary",
  "setWidgetState",
  "uploadFile",
  "artifact-derived string as untrusted data",
  "sha256FileHex",
  "model_bytes_not_sent_to_deepbom_service",
  "Use the local DEEPBOM MCP",
]) assert.ok(widget.includes(contract), `ChatGPT widget is missing ${contract}`);
assert.match(widget, /MAX_FINDINGS_PER_KIND = 8/);
assert.doesNotMatch(widget, /innerHTML\s*=\s*[^`]*error\?\.message/,
  "Untrusted error text must be assigned through textContent, not innerHTML.");

const worker = await readFile("worker/index.js", "utf8");
assert.match(worker, /url\.pathname === "\/mcp"/);
assert.match(worker, /\.well-known\/openai-apps-challenge/);
assert.match(worker, /OPENAI_APPS_CHALLENGE/);
assert.match(worker, /cross-origin-resource-policy", "cross-origin"/);

const workerModule = (await import("../worker/index.js")).default;
for (const asset of ["/chatgpt/deepbom-widget.js", "/chatgpt/static-audit-worker.js", "/pkg/tflite_wasm_audit_bg.wasm"]) {
  const response = await workerModule.fetch(new Request(`https://deepbom.org${asset}`), {
    ASSETS: { fetch: async () => new Response("asset") },
  });
  assert.equal(response.headers.get("access-control-allow-origin"), "*");
  assert.equal(response.headers.get("cross-origin-resource-policy"), "cross-origin");
}
const challenge = await workerModule.fetch(
  new Request("https://deepbom.org/.well-known/openai-apps-challenge"),
  { OPENAI_APPS_CHALLENGE: "openai-domain-verification-token" },
);
assert.equal(challenge.status, 200);
assert.equal(await challenge.text(), "openai-domain-verification-token");
assert.equal(challenge.headers.get("cache-control"), "no-store");
const challengeWithoutSecret = await workerModule.fetch(
  new Request("https://deepbom.org/.well-known/openai-apps-challenge"),
  {},
);
assert.equal(challengeWithoutSecret.status, 404);

console.log("ChatGPT MCP checks passed (file-param schema, browser-local boundary, private result/error bridges, resources, and request/result bounds).\n");

function boundedModelSummary() {
  return {
    schema: "deepbom.model_summary_conversation.v1",
    model_summary_sha256: "c".repeat(64),
    model_ir_sha256: "d".repeat(64),
    selected_level: "operation",
    status: "materialized",
    ordering: { primary: "display_order", runtime_order_claim: false },
    row_count: 2,
    rows: [],
    truncated: true,
    totals: { operation_count: 2 },
    trainability: { status: "not_assessable_from_serialized_deployment_artifact", trainable_parameter_count: null },
    interpretation_boundary: "Static Model IR projection only.",
  };
}

async function rpc(method, params = undefined) {
  const response = await rawRpc(method, params);
  if (response.error) throw new Error(response.error.message);
  return response.result;
}

async function rawRpc(method, params = undefined) {
  const response = await rawRpcResponse(method, params);
  assert.equal(response.headers.get("mcp-protocol-version"), "2025-11-25");
  return response.json();
}

async function rawRpcResponse(method, params = undefined) {
  id += 1;
  const response = await routeChatGptMcp(new Request(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id, method, ...(params === undefined ? {} : { params }) }),
  }));
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type"), /application\/json/);
  return response;
}
