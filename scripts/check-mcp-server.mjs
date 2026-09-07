import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

// The MCP surface is read by a model, not a person. Its tool descriptions are
// the only place the evidence boundary is stated before an assistant writes a
// summary, so they are checked here the same way the CLI contract is.

const frames = await exchange([
  { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "check", version: "0" } } },
  { jsonrpc: "2.0", method: "notifications/initialized" },
  { jsonrpc: "2.0", id: 2, method: "tools/list" },
  { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "deepbom_capabilities", arguments: {} } },
  { jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: "" } } },
  { jsonrpc: "2.0", id: 5, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: path.resolve("web/samples/gpu_partition_probe.onnx") } } },
  { jsonrpc: "2.0", id: 6, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: "model.onnx", passthrough: "--help" } } },
  { jsonrpc: "2.0", id: 7, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: "model.onnx", pointer: "/format", section: "summary" } } },
  { jsonrpc: "2.0", id: 8, method: "resources/list" },
]);

const byId = new Map(frames.map((frame) => [frame.id, frame]));
assert.equal(frames.length, 8, "Each request receives exactly one frame and the notification receives none.");

const initialize = byId.get(1).result;
assert.equal(initialize.protocolVersion, "2025-06-18", "The server must accept the protocol version the client offered.");
assert.equal(initialize.serverInfo.name, "deepbom");
assert.match(initialize.serverInfo.version, /^\d+\.\d+\.\d+/, "The server reports the CLI release version.");
assert.equal(initialize.capabilities.tools.listChanged, false);
for (const phrase of ["never uploaded", "evidence_gap", "latency"]) {
  assert.ok(initialize.instructions.includes(phrase),
    `Server instructions must keep the "${phrase}" boundary statement.`);
}

const tools = byId.get(2).result.tools;
assert.deepEqual(tools.map((tool) => tool.name), ["deepbom_capabilities", "deepbom_audit", "deepbom_diff"]);
for (const tool of tools) {
  assert.equal(tool.inputSchema.type, "object");
  assert.equal(tool.inputSchema.additionalProperties, false,
    `${tool.name} must not accept undeclared arguments; option pass-through would bypass the checked contract.`);
}

const audit = tools.find((tool) => tool.name === "deepbom_audit");
assert.deepEqual(audit.inputSchema.required, ["path"]);
for (const kind of ["artifact_defect", "caution", "evidence_gap"]) {
  assert.ok(audit.description.includes(kind), `deepbom_audit must name the ${kind} finding kind.`);
}
assert.ok(audit.description.includes("An evidence_gap is not a defect."),
  "deepbom_audit must state that an evidence gap is not a defect.");
assert.ok(/never establishes executed accelerator assignment, latency, energy, accuracy, or device fit/.test(audit.description),
  "deepbom_audit must state what the static result never establishes.");
const diff = tools.find((tool) => tool.name === "deepbom_diff");
assert.deepEqual(diff.inputSchema.required, ["baseline", "candidate"]);
assert.ok(diff.description.includes("not measured behaviour"),
  "deepbom_diff must state that its static estimates are not measured behaviour.");

const capabilities = JSON.parse(byId.get(3).result.content[0].text);
assert.equal(capabilities.schema, "deepbom.cli_capabilities.v1");
assert.equal(capabilities.privacy.model_bytes_network_transfer, false,
  "The MCP path must expose the same local-analysis privacy contract as the CLI.");
assert.equal(capabilities.privacy.analysis_location, "local_process");
// Discovery must reach the transport too, or an assistant that only reads the
// capability document never learns the MCP entry point exists.
const declared = capabilities.automation.mcp_stdio_server;
assert.equal(declared.invocation, "deepbom mcp");
assert.equal(declared.transport, "stdio_jsonrpc");
assert.equal(declared.hosted_endpoint, false);
assert.deepEqual(declared.tools, tools.map((tool) => tool.name),
  "The declared MCP tool list must match what tools/list actually serves.");

const rejected = byId.get(4).result;
assert.equal(rejected.isError, true, "A missing artifact path is rejected before the CLI is spawned.");
assert.match(rejected.content[0].text, /non-empty path/);

const auditResult = JSON.parse(byId.get(5).result.content[0].text);
assert.equal(auditResult.schema, "deepbom.artifact_evidence_envelope.v1");
assert.equal(auditResult.identity.format, "onnx");
assert.match(auditResult.identity.sha256, /^[a-f0-9]{64}$/);

assert.equal(byId.get(6).result.isError, true, "Undeclared tool arguments must be rejected at runtime.");
assert.match(byId.get(6).result.content[0].text, /Undeclared tool argument: passthrough/);
assert.equal(byId.get(7).result.isError, true, "Conflicting audit selectors must not be silently ignored.");
assert.match(byId.get(7).result.content[0].text, /mutually exclusive/);

assert.equal(byId.get(8).error.code, -32601, "An unsupported method returns method-not-found, not a crash.");

console.log("MCP server check passed (protocol handshake, tool contract, evidence boundary, and local-analysis privacy).\n");

function exchange(messages) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["bin/deepbom.mjs", "mcp"], { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) return reject(new Error(`deepbom mcp exited ${code}: ${stderr.trim()}`));
      // stdout is the JSON-RPC transport; a stray diagnostic write would corrupt it.
      if (stderr.trim()) return reject(new Error(`deepbom mcp wrote to stderr: ${stderr.trim()}`));
      try {
        resolve(stdout.split("\n").filter(Boolean).map((line) => JSON.parse(line)));
      } catch (error) {
        reject(new Error(`deepbom mcp emitted a non-JSON frame: ${error.message}`));
      }
    });
    for (const message of messages) child.stdin.write(`${JSON.stringify(message)}\n`);
    child.stdin.end();
  });
}
