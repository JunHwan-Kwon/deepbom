import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  buildAgentContractSnapshot,
  classifyAgentContractChange,
} from "./agent-contract-snapshot.mjs";
import {
  AGENT_CONTRACT,
  EVIDENCE_CONTRACT,
} from "../bin/deepbom-public-contract-versions.mjs";

const root = path.resolve(".");
const baselinePath = path.join(root, "release", "agent-contract-baseline.v1.json");
const current = await buildAgentContractSnapshot({ root });

if (process.argv.includes("--write-baseline")) {
  if (!process.argv.includes("--acknowledge-contract-review-candidate")) {
    throw new Error("Writing the public contract baseline requires --acknowledge-contract-review-candidate.");
  }
  const baseline = {
    schema: "deepbom.public_agent_contract_baseline.v1",
    review_state: {
      openai: "candidate_unsubmitted",
      claude_remote: "candidate_unsubmitted",
    },
    captured_engine_version: current.engine_version,
    agent_contract: current.agent_contract,
    evidence_contract: current.evidence_contract,
    surfaces: current.surfaces,
    interpretation_boundary: "This baseline is a reviewed source snapshot and does not assert OpenAI or Anthropic approval. Replace it only as an explicit contract-change action; never during an ordinary engine version sync.",
  };
  await writeFile(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`, "utf8");
  console.log(`Wrote candidate public contract baseline for Agent ${AGENT_CONTRACT.version} and evidence ${EVIDENCE_CONTRACT.version}.`);
  process.exit(0);
}

const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
assert.equal(baseline.schema, "deepbom.public_agent_contract_baseline.v1");
assert.deepEqual(baseline.agent_contract, AGENT_CONTRACT);
assert.deepEqual(baseline.evidence_contract, EVIDENCE_CONTRACT);
assert(["candidate_unsubmitted", "approved"].includes(baseline.review_state?.openai));
assert(["candidate_unsubmitted", "listed"].includes(baseline.review_state?.claude_remote));

const classification = classifyAgentContractChange(current, baseline);
const publicExportEquivalent = await buildAgentContractSnapshot({ root, remoteMcpMode: "baseline" });
assert.deepEqual(classifyAgentContractChange(publicExportEquivalent, baseline).changed_sections, [],
  "The public-export Agent gate must reproduce the live-route contract classification from pinned MCP metadata hashes.");
const engineOnlyMutation = classifyAgentContractChange({ ...current, engine_version: "9.9.9" }, baseline);
assert.deepEqual(engineOnlyMutation.changed_sections, [], "Engine version alone must not change the reviewed Agent contract fingerprint.");
const openAiMutation = structuredClone(current);
openAiMutation.surfaces.openai.sections.mcp_metadata = "0".repeat(64);
assert.equal(classifyAgentContractChange(openAiMutation, baseline).openai.review_required, true,
  "An OpenAI tool-contract mutation must require a reviewed plugin version.");
const claudeMutation = structuredClone(current);
claudeMutation.surfaces.claude_remote.sections.mcp_metadata = "1".repeat(64);
const claudeClassification = classifyAgentContractChange(claudeMutation, baseline);
assert.equal(claudeClassification.claude_remote.resubmission_required, false);
assert.equal(claudeClassification.claude_remote.host_validation_required, true,
  "A Claude remote surface mutation must require refreshed host validation.");
const evidenceMutation = structuredClone(current);
evidenceMutation.surfaces.evidence.sections.identity = "2".repeat(64);
assert.equal(classifyAgentContractChange(evidenceMutation, baseline).evidence.contract_version_review_required, true,
  "An evidence-contract mutation must require compatibility review.");
if (process.argv.includes("--json")) process.stdout.write(`${JSON.stringify(classification, null, 2)}\n`);
assert.deepEqual(classification.changed_sections, [],
  `Public Agent contract drift requires an explicit contract-version review before deployment: ${classification.changed_sections.join(", ")}. Do not refresh the baseline during an ordinary engine release.`);
assert.equal(classification.openai.review_required, false);
assert.equal(classification.claude_remote.resubmission_required, false);
assert.equal(classification.evidence.contract_version_review_required, false);

if (!process.argv.includes("--json")) {
  console.log(`Public contract release gate passed: engine ${current.engine_version}, Agent ${AGENT_CONTRACT.version}, evidence ${EVIDENCE_CONTRACT.version}; OpenAI and Claude remote metadata unchanged.`);
}
