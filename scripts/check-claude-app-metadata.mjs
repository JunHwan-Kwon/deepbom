import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { buildMcpbBundle } from "./build-mcpb-bundle.mjs";

const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const profile = JSON.parse(await readFile("docs/claude-app/submission-profile.json", "utf8"));
const readme = await readFile("docs/claude-app/README.md", "utf8");
const forbiddenStandardsText = /CycloneDX\s*2\.0|cyclonedx-20|WG activity|pullrequestreview-|(?:issues|pull)\/\d{2,}/i;

assert.equal(profile.schema, "deepbom.claude_desktop_submission.v1");
assert.equal(profile.display_name, "DEEPBOM Artifact Evidence");
assert.equal(profile.delivery, "local_mcpb");
assert.equal(profile.version, packageDocument.version);
assert.match(profile.release_asset, new RegExp(`channels-v${escapeRegExp(packageDocument.version)}/deepbom-${escapeRegExp(packageDocument.version)}\\.mcpb$`));
assert.equal(profile.remote_connector_status, "not_claimed_or_tested_for_claude_by_this_package");
assert.equal(profile.listing_status, "not_submitted_or_approved_by_this_file");
assert.match(profile.local_file_boundary, /one allowed directory/);
assert.match(readme, /current\s+ChatGPT Streamable HTTP endpoint/);
assert.match(readme, /not claimed here as a tested Claude remote connector/);
assert.match(readme, /outside-root path/);
assert.match(readme, /account-owned submission form manually/);
assert.doesNotMatch(`${readme}\n${JSON.stringify(profile)}`, forbiddenStandardsText);

// Keep the listing copy aligned with the manifest builder even before a full
// release package is assembled.
assert.match(buildMcpbBundle.toString(), /buildMcpbBundle/);

console.log("Claude Desktop review metadata passed (MCPB release identity, local path boundary, host eval route, and manual submission boundary).\n");

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
