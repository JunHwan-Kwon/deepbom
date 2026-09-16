import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { ANALYZER_SEMANTIC_VERSION } from "../web/lib/app-config.js";
import { CLAUDE_MCP_CONTRACT, routeClaudeMcp } from "../worker/claude-mcp.js";

const endpoint = "https://deepbom.org/mcp/claude";
let id = 0;

const initialize = await rpc("initialize", {
  protocolVersion: "2025-11-25",
  capabilities: {},
  clientInfo: { name: "deepbom-claude-check", version: "1" },
});
assert.equal(initialize.protocolVersion, "2025-11-25");
assert.equal(initialize.serverInfo.version, ANALYZER_SEMANTIC_VERSION);
assert.match(initialize.instructions, /deliberately selects/i);
assert.match(initialize.instructions, /cannot automatically read Claude attachments/i);
assert.match(initialize.instructions, /not model bytes/i);

for (const protocolVersion of CLAUDE_MCP_CONTRACT.supportedProtocolVersions) {
  const response = await rawRpcResponse("initialize", {
    protocolVersion,
    capabilities: {},
    clientInfo: { name: "deepbom-claude-version-check", version: "1" },
  });
  const document = await response.json();
  assert.equal(response.headers.get("mcp-protocol-version"), protocolVersion);
  assert.equal(document.result.protocolVersion, protocolVersion);
}

const { tools } = await rpc("tools/list");
assert.deepEqual(tools.map((tool) => tool.name), [
  "deepbom_open_local_analyzer",
  "deepbom_capabilities",
  "deepbom_publish_browser_analysis",
  "deepbom_publish_browser_error",
]);
for (const tool of tools) {
  assert.ok(tool.title, `${tool.name}: title is required.`);
  assert.equal(tool.annotations.readOnlyHint, true, `${tool.name}: readOnlyHint`);
  assert.equal(tool.annotations.destructiveHint, false, `${tool.name}: destructiveHint`);
  assert.equal(tool.annotations.openWorldHint, false, `${tool.name}: openWorldHint`);
  assert.equal(tool.annotations.idempotentHint, true, `${tool.name}: idempotentHint`);
  assert.equal(tool.inputSchema.type, "object");
}
assert.equal(tools[0]._meta.ui.resourceUri, CLAUDE_MCP_CONTRACT.widgetUri);
assert.deepEqual(tools[2]._meta.ui.visibility, ["app"]);
assert.deepEqual(tools[3]._meta.ui.visibility, ["app"]);

const { resources } = await rpc("resources/list");
assert.equal(resources.length, 1);
assert.equal(resources[0].uri, CLAUDE_MCP_CONTRACT.widgetUri);
const resource = await rpc("resources/read", { uri: CLAUDE_MCP_CONTRACT.widgetUri });
assert.equal(resource.contents[0].mimeType, "text/html;profile=mcp-app");
assert.match(resource.contents[0].text, /claude\/deepbom-widget\.js/);
assert.equal(resource.contents[0]._meta.ui.domain, CLAUDE_MCP_CONTRACT.widgetDomain);
assert.deepEqual(resource.contents[0]._meta.ui.csp.connectDomains, ["https://deepbom.org"]);

const opened = await rpc("tools/call", {
  name: "deepbom_open_local_analyzer",
  arguments: { analysis_depth: "payload_integrity" },
});
assert.equal(opened.structuredContent.status, "awaiting_user_file_selection");
assert.equal(opened.structuredContent.analysis_depth, "payload_integrity");
assert.match(opened.structuredContent.privacy, /never the selected model bytes/i);

const capabilities = await rpc("tools/call", { name: "deepbom_capabilities", arguments: {} });
assert.equal(capabilities.structuredContent.version, ANALYZER_SEMANTIC_VERSION);
assert.equal(capabilities.structuredContent.remote_browser_path.automatic_claude_attachment_access, false);
assert.equal(capabilities.structuredContent.remote_browser_path.service_receives_model_bytes, false);
assert.match(capabilities.structuredContent.evidence_boundary, /Static serialized-artifact evidence only/);

const validResult = {
  schema: "deepbom.browser_analysis_result.v1",
  analyzer_version: ANALYZER_SEMANTIC_VERSION,
  analysis_location: "mcp_app_browser_sandbox",
  artifact: { filename: "model.onnx", format: "onnx", sha256: "a".repeat(64), byte_length: 1024, artifact_ir_sha256: "b".repeat(64) },
  verdict: { status: "no_artifact_defect_observed", artifact_defect_count: 0, caution_count: 1, evidence_needed_count: 2 },
  graph: { operator_count: 2, tensor_count: 3, total_macs: 64, mac_confidence: "exact" },
  storage: null,
  quantization: { classification: "static_qdq_representation", max_risk: "none" },
  model_summary: boundedModelSummary(),
  findings: { artifact_defects: [], cautions: [], evidence_needed: [], truncated: false },
  evidence_boundary: "Static checks only.",
  transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
  reproduction: null,
};
const published = await rpc("tools/call", { name: "deepbom_publish_browser_analysis", arguments: { result: validResult } });
assert.deepEqual(published.structuredContent, validResult);
assert.match(published.content[0].text, /0 artifact defect\(s\), 1 caution\(s\), and 2 evidence gap\(s\)/);
assert.match(published.content[0].text, /2 format-neutral operation summary row\(s\)/);

const invalid = await rawRpc("tools/call", {
  name: "deepbom_publish_browser_analysis",
  arguments: { result: { ...validResult, transfer_boundary: "model_uploaded" } },
});
assert.equal(invalid.error.code, -32602);
assert.match(invalid.error.message, /transfer boundary/i);

const oversizedBoundary = await rawRpc("tools/call", {
  name: "deepbom_publish_browser_analysis",
  arguments: { result: { ...validResult, evidence_boundary: "x".repeat(4097) } },
});
assert.equal(oversizedBoundary.error.code, -32602);
assert.match(oversizedBoundary.error.message, /4096/);

const errorResult = {
  schema: "deepbom.browser_error.v1",
  analyzer_version: ANALYZER_SEMANTIC_VERSION,
  analysis_location: "mcp_app_browser_sandbox",
  code: "artifact_parse_failed",
  message: "The serialized payload is truncated.",
  suggested_action: "Re-download the artifact and retry.",
  file_name: "broken.onnx",
  transfer_boundary: "model_bytes_not_sent_to_deepbom_service",
};
const publishedError = await rpc("tools/call", { name: "deepbom_publish_browser_error", arguments: { error: errorResult } });
assert.deepEqual(publishedError.structuredContent, errorResult);
const oversizedError = await rawRpc("tools/call", {
  name: "deepbom_publish_browser_error",
  arguments: { error: { ...errorResult, message: "x".repeat(2049) } },
});
assert.equal(oversizedError.error.code, -32602);
assert.match(oversizedError.error.message, /2048/);

const blockedOrigin = await routeClaudeMcp(new Request(endpoint, {
  method: "POST",
  headers: { "content-type": "application/json", origin: "https://example.invalid" },
  body: JSON.stringify({ jsonrpc: "2.0", id: 900, method: "ping" }),
}));
assert.equal(blockedOrigin.status, 403);

const widget = await readFile("web/claude/deepbom-widget.js", "utf8");
for (const contract of [
  "McpAppClient",
  "type=\"file\"",
  "deepbom_publish_browser_analysis",
  "deepbom_publish_browser_error",
  "updateModelContext",
  "mcp_app_browser_sandbox",
  "model_bytes_not_sent_to_deepbom_service",
  "renderModelSummaryTable",
]) assert.ok(widget.includes(contract), `Claude widget is missing ${contract}`);
assert.doesNotMatch(widget, /window\.openai|getFileDownloadUrl|download_url/);
assert.doesNotMatch(widget, /innerHTML\s*=\s*[^`]*error\?\.message/,
  "Untrusted error text must be assigned through textContent, not innerHTML.");
const bridge = await readFile("web/lib/mcp-app-client.js", "utf8");
for (const method of [
  "ui/initialize",
  "ui/notifications/initialized",
  "ui/notifications/tool-input",
  "tools/call",
  "ui/update-model-context",
  "ui/notifications/size-changed",
  "ui/resource-teardown",
]) assert.ok(bridge.includes(method), `MCP App client is missing ${method}`);
assert.match(bridge, /event\.source !== this\.parent/);
await checkMcpAppClientRoundTrip();

const worker = await readFile("worker/index.js", "utf8");
assert.match(worker, /url\.pathname === "\/mcp\/claude"/);
assert.match(worker, /routeClaudeMcp/);
assert.match(worker, /\/claude\/deepbom-widget\.js/);

console.log("Claude remote MCP checks passed (separate file-picker contract, MCP App resource, runtime annotations, bounded result bridge, and transfer boundary).\n");

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
  const response = await routeClaudeMcp(new Request(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
    body: JSON.stringify({ jsonrpc: "2.0", id, method, ...(params === undefined ? {} : { params }) }),
  }));
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type"), /application\/json/);
  return response;
}

async function checkMcpAppClientRoundTrip() {
  const listeners = new Set();
  const sent = [];
  const parent = { postMessage: (message) => sent.push(message) };
  const originalWindow = globalThis.window;
  globalThis.window = {
    parent,
    addEventListener: (name, handler) => { if (name === "message") listeners.add(handler); },
    removeEventListener: (name, handler) => { if (name === "message") listeners.delete(handler); },
  };
  try {
    const { McpAppClient } = await import(`../web/lib/mcp-app-client.js?check=${Date.now()}`);
    const client = new McpAppClient({ name: "DEEPBOM check", version: ANALYZER_SEMANTIC_VERSION }, { autoResize: false, timeoutMs: 1000 });
    const connecting = client.connect();
    const initializeMessage = sent.shift();
    assert.equal(initializeMessage.method, "ui/initialize");
    deliver({ jsonrpc: "2.0", id: initializeMessage.id, result: { hostInfo: { name: "test-host" }, hostCapabilities: {}, hostContext: {} } });
    await connecting;
    assert.equal(sent.shift().method, "ui/notifications/initialized");

    const calling = client.callServerTool({ name: "deepbom_capabilities", arguments: {} });
    const callMessage = sent.shift();
    assert.equal(callMessage.method, "tools/call");
    deliver({ jsonrpc: "2.0", id: callMessage.id, result: { structuredContent: { ok: true } } });
    assert.deepEqual(await calling, { structuredContent: { ok: true } });

    const updating = client.updateModelContext({ content: [{ type: "text", text: "bounded" }] });
    const updateMessage = sent.shift();
    assert.equal(updateMessage.method, "ui/update-model-context");
    deliver({ jsonrpc: "2.0", id: updateMessage.id, result: {} });
    await updating;
    client.close();
    assert.equal(listeners.size, 0);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }

  function deliver(data) {
    for (const listener of listeners) listener({ source: parent, data });
  }
}
