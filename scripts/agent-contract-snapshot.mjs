import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import {
  AGENT_CONTRACT,
  EVIDENCE_CONTRACT,
  cloneContractIdentity,
} from "../bin/deepbom-public-contract-versions.mjs";
import { routeChatGptMcp } from "../worker/chatgpt-mcp.js";
import { routeClaudeMcp } from "../worker/claude-mcp.js";

const PROTOCOL_VERSION = "2025-11-25";
const SKILL_FILES = Object.freeze([
  "skills/deepbom/SKILL.md",
  "skills/deepbom/agents/openai.yaml",
  "skills/deepbom/references/capability-selection.md",
  "skills/deepbom/references/evidence-semantics.md",
  "skills/deepbom/references/failure-recovery.md",
  "skills/deepbom/scripts/verify-deepbom.mjs",
]);

export async function buildAgentContractSnapshot({ root = process.cwd() } = {}) {
  const packageDocument = await json(path.join(root, "package.json"));
  const chatGptProfile = await json(path.join(root, "docs/chatgpt-app/submission-profile.json"));
  const claudeProfile = await json(path.join(root, "docs/claude-remote/submission-profile.json"));
  const openAiMcp = await remoteMcpMaterial(routeChatGptMcp, chatGptProfile.endpoint);
  const claudeMcp = await remoteMcpMaterial(routeClaudeMcp, claudeProfile.endpoint);
  const pluginPackage = {
    portable_manifest: await json(path.join(root, "plugin.json")),
    portable_mcp: await json(path.join(root, "mcp.json")),
    codex_manifest: await json(path.join(root, ".codex-plugin/plugin.json")),
    codex_mcp: await json(path.join(root, ".mcp.json")),
    skill_files: await Promise.all(SKILL_FILES.map(async (relative) => ({
      path: relative,
      sha256: sha256(await readFile(path.join(root, relative))),
    }))),
  };
  const openAiPrivacy = {
    privacy_url: chatGptProfile.privacy_url,
    terms_url: chatGptProfile.terms_url,
    support_url: chatGptProfile.support_url,
    data_flow: chatGptProfile.data_flow,
    evidence_boundary: chatGptProfile.evidence_boundary,
    privacy_page_sha256: sha256(await readFile(path.join(root, "web/legal/privacy.html"))),
    terms_page_sha256: sha256(await readFile(path.join(root, "web/legal/terms.html"))),
  };
  const claudeListing = {
    endpoint: endpointIdentity(claudeProfile.endpoint),
    display_name: claudeProfile.display_name,
    description: await listingDescription(claudeProfile),
    documentation_url: claudeProfile.documentation_url,
    privacy_url: claudeProfile.privacy_url,
    terms_url: claudeProfile.terms_url,
    support_url: claudeProfile.support_url,
    automatic_attachment_access: claudeProfile.automatic_attachment_access,
    model_byte_transfer: claudeProfile.model_byte_transfer,
    conversation_data: claudeProfile.conversation_data,
  };
  const evidenceIdentity = {
    contract: cloneContractIdentity(EVIDENCE_CONTRACT),
    compatibility_boundary: "Additive fields may be introduced within v1; removals or semantic changes require a new evidence schema id and contract version.",
  };

  const openAiSections = sectionHashes({
    endpoint: endpointIdentity(chatGptProfile.endpoint),
    mcp_metadata: openAiMcp,
    plugin_package: pluginPackage,
    privacy_and_transfer: openAiPrivacy,
  });
  const claudeSections = sectionHashes({
    endpoint: endpointIdentity(claudeProfile.endpoint),
    mcp_metadata: claudeMcp,
    listing_and_transfer: claudeListing,
  });
  const evidenceSections = sectionHashes({ identity: evidenceIdentity });

  return {
    schema: "deepbom.public_agent_contract_snapshot.v1",
    engine_version: packageDocument.version,
    agent_contract: cloneContractIdentity(AGENT_CONTRACT),
    evidence_contract: cloneContractIdentity(EVIDENCE_CONTRACT),
    surfaces: {
      openai: openAiSections,
      claude_remote: claudeSections,
      evidence: evidenceSections,
    },
    interpretation_boundary: "Fingerprints cover reviewed metadata, schemas, annotations, instructions, UI resource identity and policy metadata, plugin Skill content, and declared transfer boundaries. They intentionally exclude analyzer version values, live analysis results, and backward-compatible UI content served at an unchanged resource URI.",
  };
}

export function classifyAgentContractChange(current, baseline) {
  const changed = [];
  for (const surface of ["openai", "claude_remote", "evidence"]) {
    const currentSurface = current?.surfaces?.[surface] || {};
    const baselineSurface = baseline?.surfaces?.[surface] || {};
    for (const name of new Set([...Object.keys(currentSurface.sections || {}), ...Object.keys(baselineSurface.sections || {})])) {
      if (currentSurface.sections?.[name] !== baselineSurface.sections?.[name]) changed.push(`${surface}.${name}`);
    }
  }
  const openAiChanged = changed.some((value) => value.startsWith("openai."));
  const claudeChanged = changed.some((value) => value.startsWith("claude_remote."));
  const evidenceChanged = changed.some((value) => value.startsWith("evidence."));
  return {
    schema: "deepbom.agent_release_classification.v1",
    release_class: changed.length ? "public_contract_change" : "engine_update_with_stable_public_contracts",
    engine_version: current.engine_version,
    agent_contract: current.agent_contract,
    evidence_contract: current.evidence_contract,
    changed_sections: changed,
    openai: {
      review_required: openAiChanged,
      action: openAiChanged ? "create_new_plugin_version_scan_review_and_publish" : "host_smoke_test_only",
    },
    claude_remote: {
      resubmission_required: false,
      host_validation_required: claudeChanged,
      action: claudeChanged ? "validate_the_refreshed_surface_and_listing_impact" : "host_smoke_test_only",
    },
    evidence: {
      contract_version_review_required: evidenceChanged,
      action: evidenceChanged ? "review_schema_compatibility_and_bump_contract_when_semantics_changed" : "no_contract_version_change",
    },
    interpretation_boundary: "This is a source-contract classification, not platform approval evidence. Account-owned review, publication, listing, and host observations remain manual facts.",
  };
}

function sectionHashes(materials) {
  const sections = Object.fromEntries(Object.entries(materials).map(([name, value]) => [name, fingerprint(value)]));
  return { fingerprint: fingerprint(sections), sections };
}

async function remoteMcpMaterial(route, endpoint) {
  const initialize = await rpc(route, endpoint, "initialize", {
    protocolVersion: PROTOCOL_VERSION,
    capabilities: {},
    clientInfo: { name: "deepbom-contract-snapshot", version: "1.0.0" },
  }, 1);
  const tools = await rpc(route, endpoint, "tools/list", {}, 2);
  const resources = await rpc(route, endpoint, "resources/list", {}, 3);
  const resourceMetadata = [];
  for (const resource of resources.resources || []) {
    const read = await rpc(route, endpoint, "resources/read", { uri: resource.uri }, 4 + resourceMetadata.length);
    for (const content of read.contents || []) {
      resourceMetadata.push({ uri: content.uri, mimeType: content.mimeType, _meta: content._meta || null });
    }
  }
  return {
    protocol_version: initialize.protocolVersion,
    capabilities: initialize.capabilities,
    server_info: {
      name: initialize.serverInfo?.name,
      title: initialize.serverInfo?.title || null,
      version: "<engine-version>",
    },
    instructions: initialize.instructions,
    tools: tools.tools,
    resources: resources.resources,
    resource_metadata: resourceMetadata,
  };
}

async function rpc(route, endpoint, method, params, id) {
  const request = new Request(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json", "mcp-protocol-version": PROTOCOL_VERSION },
    body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
  });
  const response = await route(request);
  const document = await response.json();
  if (document.error) throw new Error(`${method} failed: ${document.error.message}`);
  return document.result;
}

function endpointIdentity(value) {
  const url = new URL(value);
  return { origin: url.origin, path: url.pathname, authentication: "none", transport: "streamable_http" };
}

async function listingDescription(profile) {
  return String(profile.description || "").trim();
}

async function json(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

function fingerprint(value) {
  return sha256(Buffer.from(canonicalJson(value), "utf8"));
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
