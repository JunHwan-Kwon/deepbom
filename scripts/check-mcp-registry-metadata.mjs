import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const server = JSON.parse(await readFile(path.join(root, "server.json"), "utf8"));
const workflow = await readFile(path.join(root, ".github", "workflows", "publish-mcp-registry.yml"), "utf8");
const expectedName = "io.github.JunHwan-Kwon/deepbom";

assert.equal(packageDocument.mcpName, expectedName, "package.json mcpName must bind the GitHub namespace.");
assert.equal(server.$schema, "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json");
assert.equal(server.name, packageDocument.mcpName, "server.json name must match npm mcpName exactly.");
assert.equal(server.version, packageDocument.version, "MCP server and npm package versions must match.");
assert.equal(server.repository?.source, "github");
assert.equal(server.repository?.url, "https://github.com/JunHwan-Kwon/deepbom");
assert.equal(server.websiteUrl, "https://deepbom.org");
assert.ok(server.description.length <= 100, "Registry description exceeds the schema limit.");
assert.deepEqual(server.remotes, undefined, "DEEPBOM exposes no hosted MCP endpoint.");
assert.equal(server.packages?.length, 1, "Registry metadata must expose one canonical npm package.");

const entry = server.packages[0];
assert.equal(entry.registryType, "npm");
assert.equal(entry.identifier, packageDocument.name);
assert.equal(entry.version, packageDocument.version);
assert.deepEqual(entry.transport, { type: "stdio" });
assert.deepEqual(entry.packageArguments, [{ type: "positional", value: "mcp" }]);
assert.equal(entry.runtimeArguments, undefined, "The npm runtime is registry-selected; only the package subcommand is fixed.");

assert.match(workflow, /^\s*workflow_dispatch:\s*$/m, "Registry publishing must remain manually dispatched.");
assert.doesNotMatch(workflow, /^\s*(?:push|pull_request|schedule|release):\s*$/m,
  "Registry publishing must not consume Actions minutes on unrelated repository events.");
assert.match(workflow, /^\s*id-token:\s*write\s*$/m, "Registry publishing requires GitHub OIDC permission.");
assert.match(workflow, /modelcontextprotocol\/registry\/releases\/download\/v1\.8\.1\/mcp-publisher_linux_amd64\.tar\.gz/,
  "Registry publisher release must remain version-pinned.");
assert.match(workflow, /a06c9096dcb9727c13555b6be26c7effa707b01f06a4c561ba7a3635443cf2cc/,
  "Registry publisher archive digest must remain pinned.");
assert.match(workflow, /mcp-publisher login github-oidc/, "Registry publishing must authenticate without a stored token.");
assert.match(workflow, /mcp-publisher publish server\.json/, "Registry workflow must publish the reviewed metadata file.");

console.log("MCP Registry metadata check passed (namespace, npm ownership, version, stdio transport, positional subcommand, and OIDC workflow).");
