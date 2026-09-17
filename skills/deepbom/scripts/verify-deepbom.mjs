#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const AGENT_CONTRACT = {"id":"deepbom.agent_contract.v1","version":"1.0.2","compatibility":"backward_compatible_within_v1"};
const EVIDENCE_CONTRACT = {"id":"deepbom.artifact_evidence_envelope.v1","version":"1.0.0","compatibility":"additive_fields_only_within_v1","required_top_level_fields":["schema","generated_at","identity","artifact_set","conversion_receipt","cpu_cost_target_binding","accelerator_profile_binding","accelerator_bindings","policy_identity","capabilities","interfaces","graph","external_files","metadata","findings","format_extensions","provenance","evidence_boundary","envelope_sha256"],"conditional_top_level_fields":["llm_token_budget_scenario","structured_details"],"semantic_boundaries":["artifact_defects_cautions_and_evidence_gaps_remain_distinct","static_predictions_are_not_runtime_observations","unknown_and_not_assessable_are_not_coerced_to_absence_or_zero"]};
const REGISTRY_PACKAGE = "deepbom@latest";
const allowDownload = process.argv.slice(2).includes("--allow-download");
const attempts = [];
const candidates = localCandidates();
if (allowDownload) candidates.push({ source: "npm_contract_candidate", command: process.platform === "win32" ? "npx.cmd" : "npx", prefix_args: ["-y", REGISTRY_PACKAGE], network: true });

for (const candidate of candidates) {
  const versionRun = invoke(candidate, ["--version"]);
  const observedVersion = versionRun.stdout.trim();
  if (versionRun.status !== 0 || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(observedVersion)) {
    attempts.push({ source: candidate.source, status: versionRun.status, observed_version: observedVersion || null, error: bounded(versionRun.stderr) });
    continue;
  }
  const capabilitiesRun = invoke(candidate, ["capabilities", "--format", "agent-json", "--compact"]);
  let capabilities;
  try { capabilities = JSON.parse(capabilitiesRun.stdout); } catch {}
  if (capabilitiesRun.status !== 0 || capabilities?.schema !== "deepbom.agent_capabilities.v1"
    || capabilities?.agent_contract?.id !== AGENT_CONTRACT.id || capabilities?.agent_contract?.version !== AGENT_CONTRACT.version
    || capabilities?.evidence_schema_contract?.id !== EVIDENCE_CONTRACT.id || capabilities?.evidence_schema_contract?.version !== EVIDENCE_CONTRACT.version) {
    attempts.push({ source: candidate.source, status: capabilitiesRun.status, observed_version: observedVersion, observed_agent_contract: capabilities?.agent_contract || null, observed_evidence_contract: capabilities?.evidence_schema_contract || null, error: bounded(capabilitiesRun.stderr || capabilitiesRun.stdout || "public contract mismatch") });
    continue;
  }
  const selfTest = invoke(candidate, ["self-test", "--compact"]);
  try {
    const document = JSON.parse(selfTest.stdout);
    if (selfTest.status !== 0 || document.schema !== "deepbom.cli_self_test.v1" || document.status !== "pass" || document.version !== observedVersion || capabilities.version !== observedVersion) throw new Error("self-test or analyzer identity mismatch");
    process.stdout.write(`${JSON.stringify({ schema: "deepbom.skill_runtime_resolution.v1", version: observedVersion, agent_contract: AGENT_CONTRACT, evidence_contract: EVIDENCE_CONTRACT, status: "pass", source: candidate.source, network_used: candidate.network, invocation: { command: candidate.command, prefix_args: candidate.prefix_args }, self_test: document })}\n`);
    process.exit(0);
  } catch (error) {
    attempts.push({ source: candidate.source, status: selfTest.status, observed_version: observedVersion, error: bounded(selfTest.stderr || selfTest.stdout || error.message) });
  }
}

process.stderr.write(`${JSON.stringify({ schema: "deepbom.skill_runtime_resolution.v1", agent_contract: AGENT_CONTRACT, evidence_contract: EVIDENCE_CONTRACT, status: "unavailable", registry_download_attempted: allowDownload, attempts, suggested_action: allowDownload ? "Review the recorded installation, compatibility, or network error; do not claim an analysis ran." : "Install a contract-compatible release or rerun with --allow-download only after registry access is authorized." })}\n`);
process.exit(3);

function localCandidates() {
  const rows = [];
  if (process.env.DEEPBOM_BIN?.trim()) rows.push(candidateForPath(path.resolve(process.env.DEEPBOM_BIN.trim()), "DEEPBOM_BIN"));
  const packaged = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../bin/deepbom.mjs");
  if (existsSync(packaged)) rows.push(candidateForPath(packaged, "bundled_npm_package"));
  const installed = findOnPath("deepbom");
  if (installed && !rows.some((row) => path.resolve(row.identity) === path.resolve(installed))) rows.push(candidateForPath(installed, "installed_path"));
  return rows;
}

function candidateForPath(file, source) {
  return /\.(?:mjs|js|cjs)$/i.test(file)
    ? { source, identity: file, command: process.execPath, prefix_args: [file], network: false }
    : { source, identity: file, command: file, prefix_args: [], network: false };
}

function findOnPath(name) {
  const extensions = process.platform === "win32" ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";") : [""];
  for (const directory of (process.env.PATH || "").split(path.delimiter).filter(Boolean)) {
    for (const extension of extensions) {
      const file = path.join(directory.replace(/^"|"$/g, ""), process.platform === "win32" ? `${name}${extension.toLowerCase()}` : name);
      if (existsSync(file)) return file;
    }
  }
  return null;
}

function invoke(candidate, args) {
  const shell = process.platform === "win32" && /\.(?:cmd|bat)$/i.test(candidate.command);
  const result = spawnSync(candidate.command, [...candidate.prefix_args, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], shell, timeout: 120000, windowsHide: true });
  return { status: Number.isInteger(result.status) ? result.status : 1, stdout: result.stdout || "", stderr: result.stderr || result.error?.message || "" };
}

function bounded(value) {
  const text = String(value || "").trim();
  return text ? text.slice(0, 1000) : null;
}
