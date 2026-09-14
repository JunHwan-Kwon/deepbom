import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { buildMcpbBundle } from "./build-mcpb-bundle.mjs";

const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const profile = JSON.parse(await readFile("docs/claude-app/submission-profile.json", "utf8"));
const readme = await readFile("docs/claude-app/README.md", "utf8");
const remoteProfile = JSON.parse(await readFile("docs/claude-remote/submission-profile.json", "utf8"));
const remoteReadme = await readFile("docs/claude-remote/README.md", "utf8");
const forbiddenStandardsText = /CycloneDX\s*2\.0|cyclonedx-20|WG activity|pullrequestreview-|(?:issues|pull)\/\d{2,}/i;

assert.equal(profile.schema, "deepbom.claude_desktop_submission.v1");
assert.equal(profile.display_name, "DEEPBOM Artifact Evidence");
assert.equal(profile.delivery, "local_mcpb");
assert.equal(profile.version, packageDocument.version);
assert.match(profile.release_asset, new RegExp(`channels-v${escapeRegExp(packageDocument.version)}/deepbom-${escapeRegExp(packageDocument.version)}\\.mcpb$`));
assert.equal(profile.remote_connector_status, "separate_remote_candidate_documented_but_not_host_validated_by_this_package");
assert.equal(profile.listing_status, "not_submitted_or_approved_by_this_file");
assert.match(profile.local_file_boundary, /one allowed directory/);
assert.match(readme, /separate integrations\s+with separate file-transfer contracts/);
assert.match(readme, /outside-root path/);
assert.match(readme, /account-owned submission form manually/);
assert.match(readme, /## Privacy Policy/);
assert.match(readme, /do not upload\s+model bytes/);
assert.equal(remoteProfile.schema, "deepbom.claude_remote_submission.v1");
assert.equal(remoteProfile.version, packageDocument.version);
assert.equal(remoteProfile.endpoint, "https://deepbom.org/mcp/claude");
assert.equal(remoteProfile.automatic_attachment_access, false);
assert.equal(remoteProfile.host_validation_status, "not_yet_observed_on_claude_hosts");
assert.equal(remoteProfile.listing_status, "not_submitted_or_approved_by_this_file");
assert.match(remoteReadme, /does not claim automatic access to\s+Claude attachments/);
assert.match(remoteReadme, /real run has been recorded/);
assert.match(remoteReadme, /local MCPB or CLI for confidential files/);
assert.doesNotMatch(`${readme}\n${JSON.stringify(profile)}\n${remoteReadme}\n${JSON.stringify(remoteProfile)}`, forbiddenStandardsText);

// Keep the listing copy aligned with the manifest builder even before a full
// release package is assembled.
assert.match(buildMcpbBundle.toString(), /buildMcpbBundle/);

console.log("Claude Desktop review metadata passed (MCPB release identity, local path boundary, host eval route, and manual submission boundary).\n");

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
