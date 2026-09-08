import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const server = JSON.parse(await readFile(path.join(root, "server.json"), "utf8"));
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

console.log("MCP Registry metadata check passed (namespace, npm ownership, version, stdio transport, and positional subcommand).");
