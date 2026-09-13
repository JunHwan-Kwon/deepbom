import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { ANALYZER_SEMANTIC_VERSION } from "../web/lib/app-config.js";
import { CHATGPT_MCP_CONTRACT, routeChatGptMcp } from "../worker/chatgpt-mcp.js";

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
assert.equal(read.contents[0]._meta.ui.domain, "https://deepbom.org");
assert.ok(read.contents[0]._meta.ui.csp.connectDomains.some((domain) => domain.includes("oaiusercontent")));

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

const capabilities = await rpc("tools/call", { name: "deepbom_capabilities", arguments: {} });
assert.equal(capabilities.structuredContent.version, ANALYZER_SEMANTIC_VERSION);
assert.equal(capabilities.structuredContent.chatgpt_path.execution, "browser sandbox");
assert.match(capabilities.structuredContent.local_path.invocation, /deepbom@.+ mcp/);

const validResult = {
  schema: "deepbom.chatgpt_analysis_result.v1",
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
assert.match(published.content[0].text, /0 artifact defect\(s\), 1 caution\(s\), and 2 evidence gap\(s\)/);

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
  "deepbom_publish_analysis",
  "deepbom_publish_error",
  "sendFollowUpMessage",
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
