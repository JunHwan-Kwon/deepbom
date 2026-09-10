import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import process from "node:process";
import { resolveNpmCommand } from "./run-utils.mjs";

const inspectorVersion = process.env.DEEPBOM_MCP_INSPECTOR_VERSION || "2.6.0";
const invocation = resolveNpmCommand([
  "exec",
  "--yes",
  `--package=@modelcontextprotocol/inspector@${inspectorVersion}`,
  "--",
  "mcp-inspector",
  "--cli",
  process.execPath,
  "bin/deepbom.mjs",
  "mcp",
  "--method",
  "tools/list",
]);
const result = spawnSync(invocation.command, invocation.args, {
  cwd: process.cwd(),
  encoding: "utf8",
  maxBuffer: 16 * 1024 * 1024,
  timeout: 120_000,
});
if (result.error?.code === "ETIMEDOUT") throw new Error(`MCP Inspector ${inspectorVersion} timed out.`);
assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
const document = JSON.parse(result.stdout);
assert.deepEqual(document.tools?.map((tool) => tool.name), [
  "deepbom_capabilities",
  "deepbom_audit",
  "deepbom_diff",
  "deepbom_explain_rule",
]);
for (const tool of document.tools) {
  assert.equal(tool.inputSchema?.type, "object", `${tool.name}: Inspector did not receive an object input schema.`);
  assert.equal(tool.inputSchema?.additionalProperties, false, `${tool.name}: Inspector did not receive a closed input schema.`);
}
console.log(`Official MCP Inspector ${inspectorVersion} passed (stdio handshake and four-tool discovery).`);
