import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

// The MCP surface is read by a model, not a person. These checks exercise the
// live transport so cancellation, control-message responsiveness, bounded
// output, and the evidence boundary cannot drift behind the CLI contract.

const root = process.cwd();
const onnxPath = path.resolve("web/samples/gpu_partition_probe.onnx");
const externalReviewTfliteRoot = path.resolve("corpus/external-review/fixtures");
const scratch = path.resolve(".local-validation/mcp-server");
await rm(scratch, { recursive: true, force: true });
await mkdir(scratch, { recursive: true });

async function checkRealServerContract() {
  const nanPath = path.join(scratch, "nan-weight.safetensors");
  await writeFile(nanPath, safetensorsF32("weight", [Number.NaN, 1]));
  const session = new McpSession(process.execPath, ["bin/deepbom.mjs", "mcp"]);
  try {
    const initialize = await initializeSession(session);
    assert.equal(initialize.protocolVersion, "2025-11-25");
    assert.equal(initialize.serverInfo.name, "deepbom");
    assert.match(initialize.serverInfo.version, /^\d+\.\d+\.\d+/);
    assert.equal(initialize.capabilities.tools.listChanged, false);
    for (const phrase of ["never uploaded", "DEEPBOM_MCP_ALLOWED_ROOTS", "human-readable summary", "evidence_gap", "latency"]) {
      assert.ok(initialize.instructions.includes(phrase), `Server instructions must retain the "${phrase}" boundary.`);
    }

    session.request(2, "tools/list");
    const tools = (await session.response(2)).result.tools;
    assert.deepEqual(tools.map((tool) => tool.name), [
      "deepbom_capabilities",
      "deepbom_audit",
      "deepbom_diff",
      "deepbom_explain_rule",
    ]);
    for (const tool of tools) {
      assert.equal(tool.inputSchema.type, "object");
      assert.equal(tool.inputSchema.additionalProperties, false);
      assert.equal(typeof tool.annotations.readOnlyHint, "boolean");
      assert.equal(tool.annotations.destructiveHint, false);
      assert.equal(typeof tool.title, "string");
    }
    assert.equal(tools.find((tool) => tool.name === "deepbom_capabilities").annotations.readOnlyHint, true);
    assert.equal(tools.find((tool) => tool.name === "deepbom_explain_rule").annotations.readOnlyHint, true);
    assert.equal(tools.find((tool) => tool.name === "deepbom_audit").annotations.readOnlyHint, false,
      "Remote audit may populate a verified local cache and must not claim strict read-only behaviour.");
    assert.equal(tools.find((tool) => tool.name === "deepbom_diff").annotations.readOnlyHint, false,
      "Remote diff may populate a verified local cache and must not claim strict read-only behaviour.");
    const auditTool = tools.find((tool) => tool.name === "deepbom_audit");
    assert.deepEqual(auditTool.inputSchema.properties.scan.enum, ["auto", "structure", "integrity", "full"]);
    assert.deepEqual(auditTool.inputSchema.properties.output_format.enum,
      ["summary", "envelope", "json", "json-compact", "cyclonedx", "sarif"]);
    assert.match(auditTool.description, /immutable remote/);
    assert.match(auditTool.description, /an evidence_gap is not a defect/i);
    assert.match(auditTool.description, /never establishes executed accelerator assignment, latency, energy, accuracy, or device fit/);
    assert.match(tools.find((tool) => tool.name === "deepbom_diff").description, /standalone TFLite artifacts/);

    session.request(3, "tools/call", { name: "deepbom_capabilities", arguments: {} });
    const capabilityResult = (await session.response(3)).result;
    const capabilities = JSON.parse(capabilityResult.content[0].text);
    assert.deepEqual(capabilityResult.structuredContent, capabilities);
    assert.equal(capabilities.schema, "deepbom.cli_capabilities.v1");
    assert.equal(capabilities.privacy.model_bytes_network_transfer, false);
    assert.equal(capabilities.privacy.analysis_location, "local_process");
    const declared = capabilities.automation.mcp_stdio_server;
    assert.equal(declared.invocation, "deepbom mcp");
    assert.equal(declared.default_audit_output, "summary");
    assert.deepEqual(declared.protocol_versions, ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]);
    assert.deepEqual(declared.tools, tools.map((tool) => tool.name));

    session.request(4, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath } });
    const summary = (await session.response(4)).result;
    assert.equal(summary.isError, undefined);
    assert.equal(summary.structuredContent, undefined, "The human-readable default must not pretend to be structured JSON.");
    assert.ok(Buffer.byteLength(summary.content[0].text, "utf8") < 20 * 1024, "The default agent response must stay concise.");
    assert.match(summary.content[0].text, /deployment-artifact audit/);
    assert.match(summary.content[0].text, /sha256:/);

    session.request(5, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath, output_format: "envelope", scan: "auto" } });
    const envelopeResult = (await session.response(5)).result;
    const envelope = JSON.parse(envelopeResult.content[0].text);
    assert.deepEqual(envelopeResult.structuredContent, envelope);
    assert.equal(envelope.schema, "deepbom.artifact_evidence_envelope.v1");
    assert.equal(envelope.identity.format, "onnx");
    assert.match(envelope.identity.sha256, /^[a-f0-9]{64}$/);

    session.request(6, "tools/call", { name: "deepbom_explain_rule", arguments: { rule: "onnx.conv.output-shape" } });
    const explanation = (await session.response(6)).result;
    assert.equal(explanation.structuredContent.schema, "deepbom.rule_explanation.v1");
    assert.equal(JSON.parse(explanation.content[0].text).rule_id, "onnx.conv.output-shape");

    session.request(7, "tools/call", { name: "deepbom_audit", arguments: { path: "" } });
    assert.equal((await session.response(7)).result.isError, true);
    session.request(8, "tools/call", { name: "missing_tool", arguments: {} });
    assert.equal((await session.response(8)).error.code, -32602, "Unknown tools are invalid JSON-RPC params, not tool results.");
    session.request(9, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath, passthrough: "--help" } });
    assert.match((await session.response(9)).result.content[0].text, /Undeclared tool argument/);
    session.request(10, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath, pointer: "/format", section: "summary" } });
    assert.match((await session.response(10)).result.content[0].text, /mutually exclusive/);
    session.request(11, "tools/call", { name: "deepbom_audit", arguments: { path: path.resolve("..", "outside.onnx") } });
    assert.match((await session.response(11)).result.content[0].text, /outside DEEPBOM_MCP_ALLOWED_ROOTS/);
    session.request(12, "tools/call", { name: "deepbom_audit", arguments: { path: "https://example.com/model.onnx" } });
    assert.match((await session.response(12)).result.content[0].text, /must include #sha256/);
    session.request(13, "resources/list");
    assert.equal((await session.response(13)).error.code, -32601);
    session.request(14, "tools/call", { name: "deepbom_explain_rule", arguments: { rule: "--list" } });
    assert.match((await session.response(14)).result.content[0].text, /not a CLI option/);
    session.request(15, "tools/call", { name: "deepbom_audit", arguments: { path: nanPath, output_format: "envelope" } });
    const nanEnvelopeResult = (await session.response(15)).result;
    const nanEnvelope = JSON.parse(nanEnvelopeResult.content[0].text);
    const nanFinding = nanEnvelope.findings.find((finding) => finding.id === "EA-SER-0001");
    assert.equal(nanFinding?.finding_kind, "artifact_defect", "MCP must preserve the canonical serialized-defect classification.");
    session.request(16, "tools/call", { name: "deepbom_audit", arguments: { path: nanPath, output_format: "envelope", gate: "defects" } });
    const nanGate = (await session.response(16)).result;
    assert.equal(nanGate._meta?.deepbom?.exit_code, 2, "MCP must expose the blocked artifact-defect gate without corrupting the envelope.");
    assert.equal(JSON.parse(nanGate.content[0].text).findings.some((finding) => finding.id === "EA-SER-0001"), true);

    session.request(17, "tools/call", {
      name: "deepbom_audit",
      arguments: { path: path.join(externalReviewTfliteRoot, "space-to-batch-stale-shapes.tflite"), output_format: "envelope" },
    });
    const staleShapeEnvelope = (await session.response(17)).result.structuredContent;
    assert.equal(staleShapeEnvelope.graph.total_macs, 589_824);
    assert.equal(staleShapeEnvelope.graph.mac_assessment_status, "assessed_complete");

    session.request(18, "tools/call", {
      name: "deepbom_audit",
      arguments: { path: path.join(externalReviewTfliteRoot, "conv-16x8.tflite"), output_format: "json-compact" },
    });
    const sixteenByEight = (await session.response(18)).result.structuredContent;
    assert.equal(sixteenByEight.quantization_status.classification, "full_integer_16x8");
    assert.equal(sixteenByEight.quantization_status.quantized_compute_mac_percent, 1);
    assert.equal(sixteenByEight.estimated_int8_speedup, 1);

    session.request(19, "tools/call", {
      name: "deepbom_audit",
      arguments: { path: path.join(externalReviewTfliteRoot, "dynamic-reshape.tflite"), output_format: "json-compact" },
    });
    const dynamicShape = (await session.response(19)).result.structuredContent;
    assert.equal(dynamicShape.total_macs, null);
    assert.equal(dynamicShape.total_macs_decimal, null);
    assert.equal(dynamicShape.mac_confidence, "symbolic");
    assert.equal(dynamicShape.dynamic_shape_cost_contract.total_macs_formula.expression, "4096*D2");

    session.request(20, "tools/call", {
      name: "deepbom_audit",
      arguments: { path: path.join(externalReviewTfliteRoot, "quant-scale-risk.tflite") },
    });
    const quantRiskSummary = (await session.response(20)).result.content[0].text;
    assert.match(quantRiskSummary, /Quantization: risk at #0 CONV_2D/);
  } finally {
    await session.close();
  }
}

function safetensorsF32(name, values) {
  const payload = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
  const rawHeader = Buffer.from(JSON.stringify({
    [name]: { dtype: "F32", shape: [values.length], data_offsets: [0, payload.length] },
  }), "utf8");
  const padding = (8 - (rawHeader.length % 8)) % 8;
  const header = Buffer.concat([rawHeader, Buffer.alloc(padding, 0x20)]);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  return Buffer.concat([length, header, payload]);
}

async function checkCancellationAndQueue() {
  const fixture = await writeFixtureCli();
  const session = new McpSession(process.execPath, [fixture, "serve"], {
    DEEPBOM_MCP_MAX_CONCURRENT: "1",
    DEEPBOM_MCP_MAX_QUEUE: "0",
  });
  try {
    await initializeSession(session);
    const slowPath = path.join(scratch, "slow.onnx");
    await writeFile(slowPath, "fixture");
    session.request(20, "tools/call", {
      name: "deepbom_audit",
      arguments: { path: slowPath, output_format: "envelope" },
      _meta: { progressToken: "audit-progress" },
    });
    await session.frame((frame) => frame.method === "notifications/progress" && frame.params?.progressToken === "audit-progress");

    const pingStarted = Date.now();
    session.request(21, "ping");
    assert.deepEqual((await session.response(21, 1000)).result, {});
    assert.ok(Date.now() - pingStarted < 1000, "ping must not wait behind an audit child process.");

    session.request(22, "tools/call", { name: "deepbom_audit", arguments: { path: slowPath } });
    assert.equal((await session.response(22, 1000)).error.code, -32001, "A full zero-length queue must fail closed.");

    session.notify("notifications/cancelled", { requestId: 20, reason: "test cancellation" });
    await delay(500);
    assert.equal(session.frames.some((frame) => frame.id === 20), false, "A cancelled request must not emit a response.");
    session.request(23, "ping");
    assert.deepEqual((await session.response(23, 1000)).result, {});
  } finally {
    const closeStarted = Date.now();
    await session.close();
    assert.ok(Date.now() - closeStarted < 2500, "Cancellation must terminate the child instead of waiting for its full delay.");
  }
}

async function checkGateJsonPreservation() {
  const fixture = await writeFixtureCli();
  const session = new McpSession(process.execPath, [fixture, "serve"]);
  try {
    await initializeSession(session);
    session.request(30, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath, output_format: "envelope", gate: "defects" } });
    const result = (await session.response(30)).result;
    const firstBlock = JSON.parse(result.content[0].text);
    assert.equal(firstBlock.schema, "deepbom.test_gate_result.v1");
    assert.deepEqual(result.structuredContent, firstBlock);
    assert.equal(result.content.length, 2);
    assert.match(result.content[1].text, /policy outcome/);
    assert.deepEqual(result._meta.deepbom, { exit_code: 2, policy_status: "blocked", analysis_completed: true });
  } finally {
    await session.close();
  }
}

async function checkResponseAndFrameLimits() {
  const bounded = new McpSession(process.execPath, ["bin/deepbom.mjs", "mcp"], {
    DEEPBOM_MCP_MAX_RESPONSE_BYTES: String(64 * 1024),
  });
  try {
    await initializeSession(bounded);
    bounded.request(40, "tools/call", { name: "deepbom_audit", arguments: { path: onnxPath, output_format: "json-compact" } });
    const result = (await bounded.response(40, 5000)).result;
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /exceeded the 64 KiB MCP response limit/);
    assert.ok(Buffer.byteLength(JSON.stringify(result), "utf8") < 64 * 1024);
  } finally {
    await bounded.close();
  }

  const framed = new McpSession(process.execPath, ["bin/deepbom.mjs", "mcp"], {
    DEEPBOM_MCP_MAX_FRAME_BYTES: "4096",
  });
  try {
    await initializeSession(framed);
    framed.raw(`${JSON.stringify({ jsonrpc: "2.0", id: 41, method: "tools/call", params: { name: "deepbom_audit", arguments: { path: onnxPath, padding: "x".repeat(5000) } } })}\n`);
    const rejection = await framed.frame((frame) => frame.id === null && frame.error?.code === -32600);
    assert.match(rejection.error.message, /frame exceeds/);
    framed.request(42, "ping");
    assert.deepEqual((await framed.response(42)).result, {}, "The server must recover after discarding one oversized frame.");
  } finally {
    await framed.close();
  }
}

async function initializeSession(session) {
  session.request(1, "initialize", { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "deepbom-check", version: "0" } });
  const frame = await session.response(1);
  assert.equal(frame.error, undefined, frame.error?.message);
  session.notify("notifications/initialized");
  return frame.result;
}

async function writeFixtureCli() {
  const fixturePath = path.join(scratch, "mcp-fixture-cli.mjs");
  const mcpModule = pathToFileURL(path.resolve("bin/deepbom-mcp.mjs")).href;
  await writeFile(fixturePath, `import process from "node:process";\nimport { runMcpServer } from ${JSON.stringify(mcpModule)};\nif (process.argv[2] === "serve") {\n  await runMcpServer({ cliEntry: process.argv[1], version: "0.0.0-test" });\n} else if (process.argv.some((value) => value.endsWith("slow.onnx"))) {\n  await new Promise((resolve) => setTimeout(resolve, 10000));\n  process.stdout.write('{"schema":"deepbom.slow_fixture.v1"}\\n');\n} else {\n  process.stdout.write('{"schema":"deepbom.test_gate_result.v1","valid_json":true}\\n');\n  process.exitCode = 2;\n}\n`, "utf8");
  return fixturePath;
}

class McpSession {
  constructor(command, args, extraEnv = {}) {
    this.frames = [];
    this.waiters = [];
    this.stderr = "";
    this.closed = false;
    this.child = spawn(command, args, {
      cwd: root,
      env: { ...process.env, ...extraEnv },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk) => {
      stdout += chunk;
      let newline = stdout.indexOf("\n");
      while (newline !== -1) {
        const line = stdout.slice(0, newline);
        stdout = stdout.slice(newline + 1);
        if (line) this.pushFrame(JSON.parse(line));
        newline = stdout.indexOf("\n");
      }
    });
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk) => { this.stderr += chunk; });
    this.exit = new Promise((resolve, reject) => {
      this.child.on("error", reject);
      this.child.on("close", (code) => {
        this.closed = true;
        if (code !== 0) reject(new Error(`MCP fixture exited ${code}: ${this.stderr.trim()}`));
        else if (this.stderr.trim()) reject(new Error(`MCP server wrote to stderr: ${this.stderr.trim()}`));
        else resolve();
      });
    });
  }

  request(id, method, params) {
    this.raw(`${JSON.stringify({ jsonrpc: "2.0", id, method, ...(params === undefined ? {} : { params }) })}\n`);
  }

  notify(method, params) {
    this.raw(`${JSON.stringify({ jsonrpc: "2.0", method, ...(params === undefined ? {} : { params }) })}\n`);
  }

  raw(value) {
    this.child.stdin.write(value);
  }

  response(id, timeout = 10000) {
    return this.frame((candidate) => candidate.id === id, timeout);
  }

  frame(predicate, timeout = 10000) {
    const index = this.frames.findIndex(predicate);
    if (index !== -1) return Promise.resolve(this.frames.splice(index, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = { predicate, resolve, reject };
      waiter.timer = setTimeout(() => {
        const waiterIndex = this.waiters.indexOf(waiter);
        if (waiterIndex !== -1) this.waiters.splice(waiterIndex, 1);
        reject(new Error(`Timed out waiting for an MCP frame. Seen: ${JSON.stringify(this.frames)}`));
      }, timeout);
      this.waiters.push(waiter);
    });
  }

  pushFrame(frame) {
    const index = this.waiters.findIndex((waiter) => waiter.predicate(frame));
    if (index === -1) {
      this.frames.push(frame);
      return;
    }
    const [waiter] = this.waiters.splice(index, 1);
    clearTimeout(waiter.timer);
    waiter.resolve(frame);
  }

  async close() {
    if (!this.closed) this.child.stdin.end();
    await this.exit;
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

await checkRealServerContract();
await checkCancellationAndQueue();
await checkGateJsonPreservation();
await checkResponseAndFrameLimits();

console.log("MCP server checks passed (2025-11-25 contract, bounded output, structured results, cancellation, responsive ping, queue/root limits, and gate JSON preservation).\n");
