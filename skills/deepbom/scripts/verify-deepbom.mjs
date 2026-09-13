#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const VERSION = "1.97.4";
const allowDownload = process.argv.slice(2).includes("--allow-download");
const attempts = [];
const candidates = localCandidates();
if (allowDownload) candidates.push({ source: "npm_exact_version", command: process.platform === "win32" ? "npx.cmd" : "npx", prefix_args: ["-y", `deepbom@${VERSION}`], network: true });

for (const candidate of candidates) {
  const versionRun = invoke(candidate, ["--version"]);
  if (versionRun.status !== 0 || versionRun.stdout.trim() !== VERSION) {
    attempts.push({ source: candidate.source, status: versionRun.status, observed_version: versionRun.stdout.trim() || null, error: bounded(versionRun.stderr) });
    continue;
  }
  const selfTest = invoke(candidate, ["self-test", "--compact"]);
  try {
    const document = JSON.parse(selfTest.stdout);
    if (selfTest.status !== 0 || document.schema !== "deepbom.cli_self_test.v1" || document.status !== "pass" || document.version !== VERSION) throw new Error("self-test contract mismatch");
    process.stdout.write(`${JSON.stringify({ schema: "deepbom.skill_runtime_resolution.v1", version: VERSION, status: "pass", source: candidate.source, network_used: candidate.network, invocation: { command: candidate.command, prefix_args: candidate.prefix_args }, self_test: document })}\n`);
    process.exit(0);
  } catch (error) {
    attempts.push({ source: candidate.source, status: selfTest.status, error: bounded(selfTest.stderr || selfTest.stdout || error.message) });
  }
}

process.stderr.write(`${JSON.stringify({ schema: "deepbom.skill_runtime_resolution.v1", version: VERSION, status: "unavailable", registry_download_attempted: allowDownload, attempts, suggested_action: allowDownload ? "Review the recorded installation or network error; do not claim an analysis ran." : "Install the exact release or rerun with --allow-download only after registry access is authorized." })}\n`);
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
