import { readFileSync } from "node:fs";
import { MODULE_WORKSPACES } from "../web/lib/app-config.js";
import {
  appendRegulatoryBundleModule,
  buildBundleEnvelope,
  executeBundleStep,
  initialEvidenceBundleProgress,
  REGULATORY_BUNDLE_MODULE_SPECS,
  regulatoryBundleItems,
} from "../web/lib/bundle.js";

const appSource = readFileSync("web/app.js", "utf8");
const errors = [];

const runnerIds = parseRegulatoryBundleRunnerIds(appSource);
const specIds = REGULATORY_BUNDLE_MODULE_SPECS.map((item) => item.id);
const workspaceIds = new Set(MODULE_WORKSPACES);
const allCapabilities = Object.fromEntries(REGULATORY_BUNDLE_MODULE_SPECS.map((item) => [item.capability, true]));
allCapabilities.regulatory_report = true;
const lockedCapabilities = Object.fromEntries(REGULATORY_BUNDLE_MODULE_SPECS.map((item) => [item.capability, false]));
lockedCapabilities.regulatory_report = false;

expectUnique("regulatory bundle spec ids", specIds);
if (!appSource.includes('attestation_member: "attestation.json"') || appSource.includes('zipTextFile("signed_manifest.json"')) {
  errors.push("Bundle exports should use attestation.json and retire the ambiguous signed_manifest.json member name.");
}
expectUnique("regulatory bundle runner ids", runnerIds);
expectExactSet("regulatory bundle runners", runnerIds, specIds);
expectExactSet("regulatory bundle UI scope ids", regulatoryBundleItems(allCapabilities, true).map((item) => item.id), [
  "engineering_bundle",
  "regulatory_report",
  ...specIds,
]);
expectExactSet("regulatory bundle progress ids", Object.keys(initialEvidenceBundleProgress(allCapabilities)), [
  "engineering_bundle",
  "regulatory_report",
  ...specIds,
]);
expectExactSet("locked regulatory bundle progress ids", Object.keys(initialEvidenceBundleProgress(lockedCapabilities)), [
  "engineering_bundle",
  "regulatory_report",
  ...specIds,
]);

for (const spec of REGULATORY_BUNDLE_MODULE_SPECS) {
  expectNonEmpty(spec.id, `${spec.id || "<missing>"}.id`);
  expectNonEmpty(spec.capability, `${spec.id}.capability`);
  expectNonEmpty(spec.workspace, `${spec.id}.workspace`);
  expectNonEmpty(spec.label, `${spec.id}.label`);
  expectNonEmpty(spec.path, `${spec.id}.path`);
  expectNonEmpty(spec.detail, `${spec.id}.detail`);
  expectNonEmpty(spec.queuedDetail, `${spec.id}.queuedDetail`);

  if (spec.path && (!spec.path.startsWith("research/") || !spec.path.endsWith(".json"))) {
    errors.push(`${spec.id}.path should be a research/*.json bundle artifact path, got ${spec.path}.`);
  }
  if (spec.workspace && !workspaceIds.has(spec.workspace)) {
    errors.push(`${spec.id}.workspace=${spec.workspace} is not in MODULE_WORKSPACES.`);
  }
}

expectUnique("regulatory bundle artifact paths", REGULATORY_BUNDLE_MODULE_SPECS.map((item) => item.path));
expectLockedProgress(lockedCapabilities);
expectBundleEnvelopeContract();
await expectExecuteBundleStepContract();
await expectAppendRegulatoryBundleModuleContract();

if (errors.length) {
  console.error("Regulatory bundle contract check failed:");
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(`Regulatory bundle contract passed (${specIds.length} specs and ${runnerIds.length} runners).`);

function parseRegulatoryBundleRunnerIds(source) {
  const block = /^const REGULATORY_BUNDLE_RUNNERS = \{([\s\S]*?)^};/m.exec(source)?.[1];
  if (!block) {
    errors.push("web/app.js must define REGULATORY_BUNDLE_RUNNERS.");
    return [];
  }
  return [...block.matchAll(/^\s*([a-zA-Z_$][\w$]*)\s*:/gm)].map((match) => match[1]).sort();
}

function expectNonEmpty(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    errors.push(`${label} must be a non-empty string.`);
  }
}

function expectExactSet(label, actual, expected) {
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  for (const value of expectedSet) {
    if (!actualSet.has(value)) {
      errors.push(`${label} is missing ${value}.`);
    }
  }
  for (const value of actualSet) {
    if (!expectedSet.has(value)) {
      errors.push(`${label} has unexpected value ${value}.`);
    }
  }
}

function expectUnique(label, values) {
  const seen = new Set();
  for (const value of values) {
    if (seen.has(value)) {
      errors.push(`${label} contains duplicate ${value}.`);
    }
    seen.add(value);
  }
}

function expectLockedProgress(capabilities) {
  const progress = initialEvidenceBundleProgress(capabilities);
  for (const spec of REGULATORY_BUNDLE_MODULE_SPECS) {
    if (progress[spec.id]?.status !== "locked") {
      errors.push(`${spec.id} should start locked when ${spec.capability} is false.`);
    }
  }
}

function expectBundleEnvelopeContract() {
  const envelope = buildBundleEnvelope([{ name: "static/a.json", data: "{}" }], {
    buildSummary: (files) => `summary sees ${files.length}`,
    buildManifest: (files) => JSON.stringify({ files: files.map((file) => file.name) }),
  });
  if (envelope[0]?.name !== "summary.md" || envelope.at(-1)?.name !== "manifest.json") {
    errors.push("buildBundleEnvelope should place summary.md first and manifest.json last.");
  }
  if (envelope[0]?.data !== "summary sees 1") {
    errors.push("buildBundleEnvelope summary builder should see the base files before summary insertion.");
  }
  if (!String(envelope.at(-1)?.data || "").includes("summary.md")) {
    errors.push("buildBundleEnvelope manifest builder should see summary.md in the packaged file list.");
  }

  const compactEnvelope = buildBundleEnvelope([{ name: "engineering_report.md", data: "report" }], {
    buildSummary: () => "unused",
    buildManifest: (files) => JSON.stringify({ files: files.map((file) => file.name) }),
    includeSummary: false,
  });
  if (compactEnvelope.length !== 2 || compactEnvelope[0]?.name !== "engineering_report.md" || compactEnvelope[1]?.name !== "manifest.json") {
    errors.push("buildBundleEnvelope should omit summary.md for compact engineering packages.");
  }
  if (String(compactEnvelope[1]?.data || "").includes("summary.md")) {
    errors.push("Compact engineering package manifest should not list an omitted summary.md.");
  }
}

async function expectExecuteBundleStepContract() {
  const successEvents = [];
  const success = await executeBundleStep({
    id: "deepbom",
    label: "DEEPBOM",
    runner: async () => ({ status: "complete" }),
    setProgress: (...args) => successEvents.push(args),
    setStatus: () => {},
  });
  if (success.result?.status !== "complete" || success.logItem.status !== "complete" || success.logItem.included !== true) {
    errors.push("executeBundleStep should return a complete included log item for successful module results.");
  }
  if (!successEvents.some(([, status]) => status === "running") || !successEvents.some(([, status]) => status === "done")) {
    errors.push("executeBundleStep should emit running and done progress for successful module results.");
  }

  const failed = await executeBundleStep({
    id: "runtime_basin",
    label: "Backend Consistency",
    runner: async () => {
      throw new Error("backend unavailable");
    },
    setProgress: () => {},
    setStatus: () => {},
    formatError: (error) => error.message,
  });
  if (failed.result !== null || failed.logItem.status !== "failed" || failed.logItem.error !== "backend unavailable") {
    errors.push("executeBundleStep should return a failed log item when a bundle runner throws.");
  }
}

async function expectAppendRegulatoryBundleModuleContract() {
  const spec = REGULATORY_BUNDLE_MODULE_SPECS[0];
  const files = [];
  const moduleLog = [];
  const progress = [];
  const result = await appendRegulatoryBundleModule({
    files,
    moduleLog,
    capabilities: { [spec.capability]: true },
    bundleModule: spec,
    runners: { [spec.id]: async () => ({ status: "complete", module: spec.id }) },
    setProgress: (...args) => progress.push(args),
    setStatus: () => {},
  });
  if (result?.module !== spec.id || moduleLog[0]?.included !== true || files[0]?.name !== spec.path) {
    errors.push("appendRegulatoryBundleModule should run authorized modules and append their JSON artifact.");
  }
  if (!progress.some(([, status]) => status === "running") || !progress.some(([, status]) => status === "done")) {
    errors.push("appendRegulatoryBundleModule should forward running/done progress for authorized modules.");
  }

  const skippedFiles = [];
  const skippedLog = [];
  const skippedProgress = [];
  await appendRegulatoryBundleModule({
    files: skippedFiles,
    moduleLog: skippedLog,
    capabilities: { [spec.capability]: false },
    bundleModule: spec,
    runners: {},
    setProgress: (...args) => skippedProgress.push(args),
    setStatus: () => {},
  });
  if (skippedFiles.length || skippedLog[0]?.status !== "skipped" || skippedLog[0]?.reason !== "capability_not_enabled") {
    errors.push("appendRegulatoryBundleModule should skip locked modules without appending files.");
  }
  if (!skippedProgress.some(([, status]) => status === "skipped")) {
    errors.push("appendRegulatoryBundleModule should emit skipped progress for locked modules.");
  }
}
