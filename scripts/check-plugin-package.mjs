import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { AGENT_PLUGIN_VERSION } from "../bin/deepbom-public-contract-versions.mjs";

const packageDocument = await json("package.json");
const portable = await json("plugin.json");
const portableMcp = await json("mcp.json");
const compatibility = await json(".codex-plugin/plugin.json");
const compatibilityMcp = await json(".mcp.json");
const skill = await readFile("skills/deepbom/SKILL.md", "utf8");
const openaiSkill = await readFile("skills/deepbom/agents/openai.yaml", "utf8");
const forbiddenStandardsText = /CycloneDX\s*2\.0|cyclonedx-20|WG activity|pullrequestreview-|(?:issues|pull)\/\d{2,}/i;

assert.equal(portable.$schema, "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json");
assert.equal(portable.name, "deepbom");
assert.notEqual(portable.version, packageDocument.version, "Agent plugin version must not track the engine release version.");
assert.equal(portable.version, AGENT_PLUGIN_VERSION);
assert.equal(portable.license, "Apache-2.0");
assert.equal(portable.repository, "https://github.com/JunHwan-Kwon/deepbom");
assert.match(portable.description, /serialized AI deployment artifacts/);
assert.deepEqual(Object.keys(portableMcp.mcpServers), ["deepbom-artifact-evidence"]);
assert.equal(portableMcp.$schema, "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json");
assert.deepEqual(portableMcp.mcpServers["deepbom-artifact-evidence"], {
  type: "streamable-http",
  url: "https://deepbom.org/mcp",
});

assert.equal(compatibility.name, portable.name);
assert.equal(compatibility.version, portable.version);
assert.equal(compatibility.description, portable.description);
assert.equal(compatibility.skills, "./skills/");
assert.equal(compatibility.mcpServers, "./.mcp.json");
assert.deepEqual(compatibilityMcp.mcpServers["deepbom-artifact-evidence"], {
  type: "http",
  url: "https://deepbom.org/mcp",
});

const portableInterface = portable.extensions?.["com.openai"]?.interface;
assert(portableInterface, "Portable plugin lacks the OpenAI presentation extension.");
assert.equal(portableInterface.displayName, "DEEPBOM Artifact Evidence");
assert.equal(portableInterface.displayName, compatibility.interface.displayName);
assert.equal(portableInterface.privacyPolicyURL, "https://deepbom.org/privacy");
assert.equal(portableInterface.termsOfServiceURL, "https://deepbom.org/terms");
assert.deepEqual(portableInterface.defaultPrompt, compatibility.interface.defaultPrompt);
assert(portableInterface.defaultPrompt.length >= 3 && portableInterface.defaultPrompt.length <= 3);
assert(portableInterface.defaultPrompt.every((prompt) => prompt.length <= 128));

assert.match(skill, /serialized AI model artifacts/);
assert.match(skill, /does not access a package registry by default/);
assert.match(skill, /ask before running/);
assert.match(openaiSkill, /allow_implicit_invocation: true/);
assert.doesNotMatch(JSON.stringify({ portable, portableMcp, compatibility, compatibilityMcp }), forbiddenStandardsText);

console.log("Portable Agent Plugin package passed (portable and compatibility manifests, remote MCP identity, Skill discovery, and evidence boundary).\n");

async function json(file) {
  return JSON.parse(await readFile(file, "utf8"));
}
