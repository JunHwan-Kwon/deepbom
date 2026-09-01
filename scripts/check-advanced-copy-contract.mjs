import { readFileSync } from "node:fs";
import {
  combineModuleResults,
  moduleAccessStatesFor,
  moduleTabStatusTextFor,
  moduleWorkflowDescription,
} from "../web/lib/auth-labels.js";
import {
  deepBomProtocolGroups,
  deploymentSensitivityProtocolGroups,
  perturbationProtocolGroups,
  runtimeBasinProtocolGroups,
} from "../web/lib/protocols.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Research module copy contract check");
const bannedVisibleCopy = /\b(MVP|Planned|Available|Pro v2|Enterprise|Research Beta)\b/i;
const driftProfile = {
  dtype: "INT8",
  unit: "raw output LSB",
  rmsOk: 0.5,
  rmsWarn: 2,
  maxOk: 1,
  maxWarn: 4,
};

const allCapabilities = {
  report: true,
  export: true,
  raw_export: true,
  regulatory_report: true,
  deepbom: true,
  perturbation: true,
  runtime_basin: true,
  deployment_sensitivity: true,
};
const adminUser = { role: "admin", email_verified: true };
const anonymousStates = moduleAccessStatesFor({}, null);
const authorizedStates = moduleAccessStatesFor(allCapabilities, adminUser);

expectEqual(combineModuleResults([null, null]), null, "combined module status should remain not run without results.");
expectEqual(combineModuleResults([{ status: "complete" }, null]).status, "running", "combined module status should wait for every result.");
expectEqual(combineModuleResults([{ status: "failed" }, { status: "complete" }]).status, "failed", "combined module status should preserve a failed stage.");
expectEqual(combineModuleResults([{ status: "complete" }, { status: "complete" }]).status, "complete", "combined module status should complete only when every stage completes.");
const htmlSource = readFileSync("web/index.html", "utf8");
const deepBomSource = readFileSync("protected/deepbom_wasm/src/lib.rs", "utf8");
const appSource = readFileSync("web/app.js", "utf8");
const deepBomWorkspaceSource = readFileSync("web/lib/app-deepbom-workspace.js", "utf8");

expect(!htmlSource.includes(">Request feedback</button>"), "Module run buttons should not default to ambiguous Request feedback copy.");
expect(!deepBomSource.includes("DEEPBOM Pro"), "DEEPBOM report title should use Advanced-era product copy, not Pro branding.");
expect(!bannedVisibleCopy.test(deepBomSource), "DEEPBOM user-facing strings should not use ambiguous legacy module labels.");
expect(appSource.includes('from "./lib/app-deepbom-workspace.js"') && appSource.includes("createDeepBomWorkspace({"), "Research workspace module should remain wired into the application controller.");
expect(deepBomWorkspaceSource.includes('deepBomSection("XNNPACK Selector Evidence"'), "Research result should expose protected selector coverage as a first-class result section.");
expect(deepBomWorkspaceSource.includes("Open Kernel Inspector") && deepBomWorkspaceSource.includes('data-explorer-tab="kernels"'), "Research selector result should provide a direct Kernel Inspector command.");

expectEqual(moduleWorkflowDescription("deepbom"), "Artifact descriptors", "DEEPBOM workflow description should be capability-oriented.");
expectEqual(moduleWorkflowDescription("perturbation"), "Input/output drift", "Perturbation workflow description should avoid version branding.");
expectEqual(moduleWorkflowDescription("runtime_basin"), "Backend drift", "Runtime basin workflow description should describe the check.");
expectEqual(moduleWorkflowDescription("deployment_sensitivity"), "Finite-difference observations", "Deployment Sensitivity workflow description should describe research-stage evidence.");

expectEqual(authorizedStates.engineering_report.label, "Report", "Engineering Report should remain an open report surface.");
expectEqual(authorizedStates.regulatory_report.label, "Report", "Regulatory Support Report should remain an open report surface.");
expectEqual(authorizedStates.deepbom.label, "Run", "DEEPBOM should show Run when authorized.");
expectEqual(authorizedStates.perturbation.label, "Run", "Perturbation should show Run when authorized.");
expectEqual(authorizedStates.runtime_basin.label, "Run", "Backend Consistency should show Run when authorized.");
expectEqual(authorizedStates.deployment_sensitivity.label, "Run", "Deployment Sensitivity should show Run when authorized.");
expectEqual(anonymousStates.deepbom.label, "Sign in", "Locked Research module should ask anonymous users to sign in.");

expectEqual(
  moduleTabStatusTextFor({
    moduleId: "engineering_report",
    accessState: authorizedStates.engineering_report,
    capabilities: allCapabilities,
    hasCurrent: true,
  }),
  "Report",
  "Engineering Report tab status should be Report after audit.",
);
expectEqual(
  moduleTabStatusTextFor({
    moduleId: "deepbom",
    accessState: authorizedStates.deepbom,
    capabilities: allCapabilities,
    hasCurrent: true,
  }),
  "Not run",
  "Runnable modules should show Not run before execution.",
);
expectEqual(
  moduleTabStatusTextFor({
    moduleId: "deepbom",
    accessState: authorizedStates.deepbom,
    result: { status: "complete" },
    capabilities: allCapabilities,
    hasCurrent: true,
  }),
  "Complete",
  "Runnable modules should show Complete after execution.",
);
expectEqual(
  moduleTabStatusTextFor({
    moduleId: "deepbom",
    accessState: authorizedStates.deepbom,
    result: { basin_proxy_score: 0.52, topology_stress_score: 0.53 },
    capabilities: allCapabilities,
    hasCurrent: true,
  }),
  "Complete",
  "Runnable modules should show Complete when a result object exists without an explicit status.",
);

const protocolSets = [
  deepBomProtocolGroups(),
  perturbationProtocolGroups({
    drift: { rms: 0.1, maxAbs: 1, cosineDistance: 0, top1Flip: false },
    baseline: { outputs: [{ values: [1, 2, 3] }] },
    perturbed: { outputs: [{ values: [1, 2, 3] }] },
    profile: driftProfile,
  }),
  runtimeBasinProtocolGroups({
    maxDrift: { rms: 0.2, maxAbs: 2 },
    results: [{ backend: "wasm", ok: true }, { backend: "webgpu", ok: true }],
    profile: driftProfile,
  }),
  deploymentSensitivityProtocolGroups({
    basin: { score: 80 },
    curvature: { rms: 0.001 },
  }),
];

for (const group of protocolSets.flat()) {
  expect(!bannedVisibleCopy.test(group.title), `Protocol group title should not use ambiguous visible copy: ${group.title}`);
  for (const item of group.items || []) {
    const text = [item.name, item.status, item.detail].filter(Boolean).join(" ");
    expect(!bannedVisibleCopy.test(text), `Protocol item should not use ambiguous visible copy: ${text}`);
  }
}

const protocolCopy = JSON.stringify(protocolSets);
expect(protocolCopy.includes("No validated") || protocolCopy.includes("not empirically calibrated"), "Experimental composite protocols must disclose their unvalidated interpretation boundary.");
expect(!/ok\s*[<>]=?\s*70|ok\s*[<>]=?\s*40|risk\s*[<>]=?\s*65/i.test(protocolCopy), "Experimental composites must not expose unvalidated decision thresholds.");
expect(!deepBomWorkspaceSource.includes("Calmer artifact") && !deepBomWorkspaceSource.includes("High quant pressure"), "Artifact composites must not be translated into unvalidated qualitative grades.");
const compositeOnlyFindings = buildFindingsRegister(
  { format: "tflite", ops: [], tensors: [], summary: {} },
  { deepBomResult: { score_assessment: { status: "ASSESSED" }, topology_stress_score: 0.99, quant_stress_score: 0.99, basin_proxy_score: 0.01 } },
);
expect(!compositeOnlyFindings.some((finding) => finding.finding_id === "EA-PRX-0001"), "An unvalidated artifact composite must not enter the actionable findings register.");

done("Research module copy contract passed (module action labels, statuses, and protocol wording).");
