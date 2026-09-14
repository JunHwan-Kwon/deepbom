import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dirname, isAbsolute, resolve } from "node:path";
import process from "node:process";
import { resolveNpmCommand } from "./run-utils.mjs";

const inspectorVersion = process.env.DEEPBOM_MCP_INSPECTOR_VERSION || "2.6.0";
const options = parseOptions(process.argv.slice(2));
const entry = resolve(options.entry || "bin/deepbom.mjs");
const fixture = resolve(options.fixture || "web/samples/mnist-8.onnx");
const allowedRoot = resolve(options.allowedRoot || dirname(fixture));
const outsidePath = resolve(options.outsidePath || "package.json");
const environment = { ...process.env, DEEPBOM_MCP_ALLOWED_ROOTS: allowedRoot };

const listed = unwrap(runInspector("tools/list"));
assert.deepEqual(listed.tools?.map((tool) => tool.name), [
  "deepbom_capabilities",
  "deepbom_audit",
  "deepbom_diff",
  "deepbom_explain_rule",
]);
for (const tool of listed.tools) {
  assert.ok(tool.title, `${tool.name}: Inspector did not receive a tool title.`);
  assert.equal(tool.inputSchema?.type, "object", `${tool.name}: Inspector did not receive an object input schema.`);
  assert.equal(tool.inputSchema?.additionalProperties, false, `${tool.name}: Inspector did not receive a closed input schema.`);
  assert.equal(typeof tool.annotations?.readOnlyHint, "boolean", `${tool.name}: readOnlyHint is required.`);
  assert.equal(typeof tool.annotations?.destructiveHint, "boolean", `${tool.name}: destructiveHint is required.`);
}

const capabilities = unwrap(runInspector("tools/call", "deepbom_capabilities", {}));
assert.equal(capabilities.isError, undefined);
assert.match(capabilities.content?.[0]?.text || "", /deepbom\.cli_capabilities\.v1/);

const audit = unwrap(runInspector("tools/call", "deepbom_audit", {
  path: fixture,
  output_format: "summary",
  offline: true,
}));
assert.notEqual(audit.isError, true, JSON.stringify(audit));
assert.match(audit.content?.[0]?.text || "", /DEEPBOM|artifact|model/i);

const diff = unwrap(runInspector("tools/call", "deepbom_diff", {
  baseline: fixture,
  candidate: fixture,
  offline: true,
}));
assert.notEqual(diff.isError, true, JSON.stringify(diff));
assert.match(diff.content?.[0]?.text || "", /deepbom|baseline|candidate|change/i);

const explanation = unwrap(runInspector("tools/call", "deepbom_explain_rule", { rule: "EA-SER-0002" }));
assert.notEqual(explanation.isError, true, JSON.stringify(explanation));
assert.match(explanation.content?.[0]?.text || "", /EA-SER-0002/);

const outside = unwrap(runInspector("tools/call", "deepbom_audit", {
  path: outsidePath,
  output_format: "summary",
  offline: true,
}, { allowedStatuses: [0, 5] }));
assert.equal(outside.isError, true, `Outside-root path was not rejected: ${JSON.stringify(outside)}`);
assert.match(outside.content?.[0]?.text || "", /outside DEEPBOM_MCP_ALLOWED_ROOTS/i);

console.log(`Official MCP Inspector ${inspectorVersion} passed (four runtime tool calls, annotations, and outside-root rejection; entry ${entry}).`);

function runInspector(method, toolName = null, toolArguments = null, { allowedStatuses = [0] } = {}) {
  const invocation = resolveNpmCommand([
    "exec",
    "--yes",
    `--package=@modelcontextprotocol/inspector@${inspectorVersion}`,
    "--",
    "mcp-inspector",
    "--cli",
    process.execPath,
    entry,
    "mcp",
    "-e",
    `DEEPBOM_MCP_ALLOWED_ROOTS=${allowedRoot}`,
    "--method",
    method,
    ...(toolName ? ["--tool-name", toolName] : []),
    ...(toolArguments ? ["--tool-args-json", JSON.stringify(toolArguments)] : []),
    "--format",
    "json",
  ]);
  const result = spawnSync(invocation.command, invocation.args, {
    cwd: process.cwd(),
    encoding: "utf8",
    env: environment,
    maxBuffer: 32 * 1024 * 1024,
    timeout: 180_000,
  });
  if (result.error?.code === "ETIMEDOUT") throw new Error(`MCP Inspector ${inspectorVersion} timed out for ${toolName || method}.`);
  assert.ok(allowedStatuses.includes(result.status), `${result.stdout}\n${result.stderr}`);
  return JSON.parse(result.stdout);
}

function unwrap(document) {
  return document?.result ?? document;
}

function parseOptions(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const raw = args[index];
    const [name, inlineValue] = raw.split(/=(.*)/s, 2);
    if (!name.startsWith("--")) throw new Error(`Unexpected argument: ${raw}`);
    const key = ({
      "--entry": "entry",
      "--fixture": "fixture",
      "--allowed-root": "allowedRoot",
      "--outside-path": "outsidePath",
    })[name];
    if (!key) throw new Error(`Unknown option: ${name}`);
    const value = inlineValue ?? args[++index];
    if (!value) throw new Error(`${name} requires a value.`);
    parsed[key] = isAbsolute(value) ? value : resolve(value);
  }
  return parsed;
}
