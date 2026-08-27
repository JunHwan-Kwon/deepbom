import { readFileSync } from "node:fs";
import { createCheck } from "./check-assert.mjs";
import {
  classifyChangeSet,
  packageJsonDeployReason,
  packageJsonPrivateWasmCheckReason,
  packageJsonRustCheckReason,
  parseDeployArgs,
  privateWasmCheckReasonForFile,
  rustCheckReasonForFile,
  stripRustTests,
} from "./detect-deployable-changes.mjs";
import { privateModuleValidationCases } from "./private-wasm-modules.mjs";

const { done, expect, expectDeepEqual, expectEqual } = createCheck("CI deploy contract check");
const workflow = readFileSync(".github/workflows/pages.yml", "utf8");
const qualityWorkflow = readFileSync(".github/workflows/quality.yml", "utf8");
const publicQualityWorkflow = readFileSync(".github/workflows/public-quality.yml", "utf8");
const channelReleaseWorkflow = readFileSync(".github/workflows/release-channels.yml", "utf8");
const channelBuildSource = readFileSync("scripts/build-channel-artifacts.mjs", "utf8");
const checkTierSource = readFileSync("scripts/check-tier.mjs", "utf8");
const deployGateSource = readFileSync("scripts/check-deploy.mjs", "utf8");
const releaseBuildSource = readFileSync("scripts/build-release.mjs", "utf8");
const buildPagesSource = readFileSync("scripts/build-pages.mjs", "utf8");
const buildMetadataSource = readFileSync("scripts/write-build-metadata.mjs", "utf8");
const packageManifest = JSON.parse(readFileSync("package.json", "utf8"));
const releaseGeneratedSource = readFileSync("scripts/release-generated-artifacts.mjs", "utf8");
const wasmBuildSource = readFileSync("scripts/build-wasm.mjs", "utf8");
const privateModuleCases = privateModuleValidationCases();
expect(Array.isArray(privateModuleCases), "CI deploy contract should expose private optional WASM module cases as an array.");
expect(
  /^on:\s*\r?\n\s+workflow_dispatch:/m.test(workflow)
    && !/^\s+(?:push|pull_request|schedule|workflow_run):/m.test(workflow),
  "Cloudflare deployment must remain manual-only in this repository.",
);
expect(
  workflow.includes("vars.DEEPBOM_ENABLE_DEPLOY == 'true'"),
  "Manual deployment must require explicit repository opt-in.",
);
expect(
  /^on:\s*\r?\n\s+workflow_dispatch:/m.test(qualityWorkflow)
    && !/^\s+(?:push|pull_request|schedule|workflow_run):/m.test(qualityWorkflow),
  "Full Quality must remain manual-only in this repository.",
);
expect(/^on:\s*\r?\n\s+push:/m.test(publicQualityWorkflow)
  && /^\s+pull_request:/m.test(publicQualityWorkflow)
  && /^\s+workflow_dispatch:/m.test(publicQualityWorkflow),
"Public core quality should run on public pushes, pull requests, and manual dispatch.");
expect(publicQualityWorkflow.includes("if: github.event.repository.private == false"),
  "Public core quality must consume no private-repository runner minutes.");
for (const snippet of [
  "actions/checkout@v5",
  "actions/setup-node@v6",
  "actions/setup-python@v6",
  "npm run check:public-source",
  "npx playwright install --with-deps chromium",
  "npm run check:public-package-boundary",
  "npm run check:channels -- --no-build",
  "npm run check:rust",
]) expect(publicQualityWorkflow.includes(snippet), `Public core quality should contain: ${snippet}`);
expect(
  /quality:\s*\r?\n\s+if: vars\.DEEPBOM_ENABLE_CI == 'true'/.test(qualityWorkflow),
  "Full Quality must require explicit repository opt-in before consuming hosted runner minutes.",
);
expect(
  /supported-format-evidence:\s*\r?\n\s+if: vars\.DEEPBOM_ENABLE_CI == 'true'/.test(qualityWorkflow),
  "Supported-format evidence must require explicit repository opt-in before consuming hosted runner minutes.",
);
expect(
  workflow.includes("ref: ${{ github.sha }}"),
  "Manual deployment checkout must bind to the selected workflow SHA.",
);

for (const snippet of [
  'reason=forced-workflow_dispatch',
  'echo "changed=true" >> "$GITHUB_OUTPUT"',
  'echo "rust_check=false" >> "$GITHUB_OUTPUT"',
  "private_wasm_check",
  "Run private WASM build/load smoke",
  "npm audit --audit-level=high",
  "npm run check:deploy",
  "node scripts/write-cloudflare-deploy-config.mjs wrangler.jsonc .wrangler-deploy.json",
  "npx --yes wrangler@latest deploy --config .wrangler-deploy.json",
  "CLOUDFLARE_ZONE_NAME: deepbom.org",
  '--data-urlencode "name=${CLOUDFLARE_ZONE_NAME}"',
  'zones/${zone_id}/purge_cache',
  "npm run build:release",
]) {
  expectWorkflowContains(snippet);
}
expect(workflow.includes("actions/checkout@v5"), "Deploy workflow should use the Node 24 checkout action.");
expect(workflow.includes("actions/setup-node@v5"), "Deploy workflow should use the Node 24 setup-node action.");
expect(
  /jobs:\s*[\s\S]*?quality:\s*[\s\S]*?Checkout[\s\S]*?fetch-depth:\s*100[\s\S]*?supported-format-evidence:/.test(qualityWorkflow),
  "The quality job must retain bounded history for recorded corpus-analyzer provenance checks.",
);
expect(workflow.includes("cancel-in-progress: true"), "A newer deployment should cancel a stale deployment run.");
expect(!workflow.includes("npx playwright install"), "Browser installation belongs in the Full Quality workflow, not deployment.");
expect(!workflow.includes("check:explorer-redesign-dist"), "Minified browser viewer checks must not restore Chromium to the deployment path.");
expect(!workflow.includes("- name: Run Rust checks"), "Rust checks belong in the Full Quality workflow, not the time-bounded deployment path.");
expect(!workflow.includes("- name: Run native trace sink checks"), "Native trace checks belong in the Full Quality workflow, not deployment.");
expect(workflow.includes("wasm-bindgen/wasm-pack/releases/download/v0.13.1"), "Deployment must use the pinned official wasm-pack release binary.");
expect(workflow.includes("c539d91ccab2591a7e975bcf82c82e1911b03335c80aa83d67ad25ed2ad06539"), "Pinned wasm-pack archive must be SHA-256 verified.");
for (const snippet of [
  "actions/checkout@v5",
  "actions/setup-node@v5",
  "npx playwright install --with-deps chromium",
  "npm run check",
  "npm run check:rust",
  "check:source-budget",
  "npm run validate:supported-formats",
  "npm run verify:supported-format-outputs",
]) {
  expect(qualityWorkflow.includes(snippet), `Full Quality workflow should contain: ${snippet}`);
}
expect(/^on:\s*\r?\n\s+workflow_dispatch:/m.test(channelReleaseWorkflow), "Channel release must be manual-only.");
expect(!/^\s+(?:push|schedule|workflow_run):/m.test(channelReleaseWorkflow), "Channel release must not have an automatic trigger.");
for (const snippet of [
  "actions/checkout@v5",
  "actions/setup-node@v6",
  "actions/setup-python@v6",
  "actions/upload-artifact@v6",
  "actions/download-artifact@v6",
  "node-version: 24.12.0",
  "npm run check:channels -- --no-build",
  "npm run check:public-package-boundary",
  "verify-python-wheel-matrix.py",
  "pypa/gh-action-pypi-publish@v1.14.2",
  "npm publish dist/deepbom-${{ inputs.expected_version }}.tgz --access public",
  "environment: pypi",
  "environment: npm",
  "npm >= 11.5.1 is required for Trusted Publishing",
  "test \"${GITHUB_REF}\" = \"refs/tags/channels-v${actual}\"",
]) expect(channelReleaseWorkflow.includes(snippet), `Channel release workflow should contain: ${snippet}`);
for (const identity of ["windows-x64", "windows-arm64", "linux-x64", "linux-arm64", "macos-x64", "macos-arm64"]) {
  expect(channelReleaseWorkflow.includes(`id: ${identity}`), `Channel release matrix should contain ${identity}.`);
}
expect((channelReleaseWorkflow.match(/id-token:\s*write/g) || []).length === 2, "Only the npm and PyPI publishing jobs should receive OIDC identity-token permission.");
expect(!channelReleaseWorkflow.includes("NPM_TOKEN"), "npm Trusted Publishing must not retain a long-lived publication token.");
expect(!channelReleaseWorkflow.includes("PYPI_API_TOKEN"), "PyPI Trusted Publishing must not retain a long-lived publication token.");
expect(!channelReleaseWorkflow.includes("--provenance=false"), "The package workflow must not permanently suppress provenance when the repository later becomes public.");
expect(!/(?:npm_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{20,})/.test(channelReleaseWorkflow), "Registry credentials must be referenced from GitHub Secrets, never embedded in the workflow.");
for (const snippet of ["resolveWindowsSignTool", '["remove", "/s", executable]', '["--remove-signature", executable]', '["--force", "--sign", "-", executable]']) {
  expect(channelBuildSource.includes(snippet), `Channel engine assembly should contain: ${snippet}`);
}
expect(!deployGateSource.includes("-viewer.mjs"), "Deployment gate must not restore repeated browser viewer checks.");
expect(deployGateSource.includes("check-parser-robustness.mjs"), "Deployment gate must reject parser robustness regressions.");
expect(deployGateSource.includes("check-format-routing.mjs"), "Deployment gate must exercise the production format gate.");
expect(deployGateSource.includes("check-onnx-shape-inference.mjs"), "Deployment gate should retain deterministic ONNX inference coverage.");
expect(
  !workflow.includes("deploy --config wrangler.jsonc"),
  "CI must not reconcile dashboard-managed production routes during a Worker code deployment.",
);
for (const snippet of ["--untracked-files=all", "RELEASE_GENERATED_TRACKED_ARTIFACTS", "sourceMetadataSnapshot", 'DEEPBOM_RELEASE_BUILD: "1"', "APP_EXPIRES_AT_EPOCH_MS", "APP_NOT_BEFORE_EPOCH_MS", "Formal release builds require a clean source tree"]) {
  expect(releaseBuildSource.includes(snippet), `Release build gate should contain: ${snippet}`);
}
expect(buildMetadataSource.includes("RELEASE_GENERATED_TRACKED_ARTIFACTS"), "Build metadata clean-tree gate should share the release-generated artifact inventory.");
expect(
  packageManifest.scripts?.["validate:supported-formats"] === "node scripts/write-build-metadata.mjs && node scripts/validate-supported-formats.mjs",
  "Supported-format validation must generate ignored build metadata before importing report modules in a clean checkout.",
);
expect(
  packageManifest.scripts?.postinstall === "node scripts/patch-litert-int8.mjs && node scripts/write-build-metadata.mjs --auto-distribution",
  "A clean source install must generate build metadata for its detected public/private distribution.",
);
expect(
  packageManifest.scripts?.["check:format-evidence"] === "node scripts/check-tier.mjs format-evidence",
  "The full supported-format evidence capture must remain available through an explicit slow tier.",
);
const formatTierBlock = /const FORMATS = \[([\s\S]*?)\n\];/.exec(checkTierSource)?.[1] || "";
const formatEvidenceTierBlock = /const FORMAT_EVIDENCE = \[([\s\S]*?)\n\];/.exec(checkTierSource)?.[1] || "";
expect(!formatTierBlock.includes("validate-supported-formats.mjs"), "The fast format-contract tier must not run full browser evidence capture.");
expect(formatEvidenceTierBlock.includes("validate-supported-formats.mjs"), "The explicit format-evidence tier must retain full browser evidence capture.");
expect(formatEvidenceTierBlock.includes("verify-supported-format-outputs.mjs"), "The explicit format-evidence tier must independently verify captured outputs.");
expect(wasmBuildSource.includes("Math.floor(Date.now() / 86400_000) * 86400_000"), "Development WASM expiry epochs should be byte-reproducible within one UTC build day.");
for (const artifact of ["pkg/tflite_wasm_audit_bg.wasm", "web/protected/deepbom/pkg/deepbom_wasm_bg.wasm"]) {
  expect(wasmBuildSource.includes(artifact) && wasmBuildSource.includes("hardenWasmFile"), `Release WASM build should harden: ${artifact}`);
  expect(buildPagesSource.includes(artifact) && buildPagesSource.includes("hardenWasmFile"), `Deploy assembly should re-harden before hashing and copying: ${artifact}`);
}
for (const artifact of ["pkg/tflite_wasm_audit_bg.wasm", "web/protected/deepbom/pkg/deepbom_wasm_bg.wasm"]) {
  expect(releaseGeneratedSource.includes(artifact), `Release-generated artifact inventory should contain: ${artifact}`);
}
expect(buildPagesSource.includes('await import("./check-build-metadata.mjs?build-pages-verification")'), "Build pipeline should verify generated build-content metadata before formal-build artifact restoration.");

const beforePackage = packageText();
const afterCheckPackage = packageText({ extraScripts: { "check:rust": "node scripts/check-rust.mjs" } });
const afterDeployCheckPackage = packageText({ extraScripts: { "check:deploy": "node scripts/check-deploy.mjs" } });
const afterRuntimePackage = packageText({
  buildWorker: "node scripts/build-pages.mjs",
  dependency: "^2.5.3",
});
const runtimeRust = { "src/lib.rs": ["fn answer() -> u32 { 1 }\n", "fn answer() -> u32 { 2 }\n"] };
const rustTestOnly = {
  "src/lib.rs": [
    "fn answer() -> u32 { 1 }\n",
    "fn answer() -> u32 { 1 }\n\n#[cfg(test)]\nmod tests {\n  #[test]\n  fn ok() { assert_eq!(1, 1); }\n}\n",
  ],
};
for (const item of [
  [["web/app.js"], "web asset changes deploy"],
  [[" web\\app.js "], "trimmed Windows web asset path changes deploy"],
  [["worker/index.js"], "worker changes deploy"],
  [["pkg/tflite_wasm_audit_bg.wasm"], "tracked wasm changes deploy"],
  [["scripts/build-pages.mjs"], "build script changes deploy"],
  [["scripts/build-release.mjs"], "formal release wrapper changes deploy"],
  [["scripts/release-generated-artifacts.mjs"], "release artifact inventory changes deploy"],
  [["scripts/write-build-metadata.mjs"], "build provenance writer changes deploy"],
  [["scripts/write-cloudflare-deploy-config.mjs"], "Cloudflare deploy config writer changes deploy"],
  [["protected/deepbom_wasm/Cargo.toml"], "public DEEPBOM manifest changes deploy"],
  [["protected/deepbom_wasm/Cargo.lock"], "public DEEPBOM lockfile changes deploy"],
  [["src/lib.rs"], "rust runtime changes deploy", runtimeRust],
  [["package.json"], "package dependency/build changes deploy", { "package.json": [beforePackage, afterRuntimePackage] }],
]) expectDeploy(...item);

for (const item of [
  [["README.md"], "readme skips deploy"],
  [[".gitignore"], "gitignore privacy-only changes skip deploy"],
  [["pkg/README.md"], "generated wasm-pack README skips deploy"],
  [["docs/PROJECT_STATUS.md"], "docs skip deploy"],
  [[" docs\\PROJECT_STATUS.md "], "trimmed Windows docs path skips deploy"],
  [[".github/workflows/pages.yml"], "workflow-only changes skip deploy"],
  [["scripts/check-rust.mjs"], "check script skips deploy"],
  [["scripts/verify-local.mjs"], "local verification script skips deploy"],
  [["docs/PROJECT_STATUS.md", "scripts/check-js.mjs"], "mixed docs and check-script changes skip deploy"],
  [["src/lib.rs"], "rust test-only changes skip deploy", rustTestOnly],
  [["package.json"], "package check-script-only changes skip deploy", { "package.json": [beforePackage, afterCheckPackage] }],
]) expectSkip(...item);
for (const moduleCase of privateModuleCases) {
  expectSkip([moduleCase.cargoManifest], `private ${moduleCase.id} manifest skips public deploy`);
  expectSkip([moduleCase.primarySource], `private ${moduleCase.id} source skips public deploy`, privateRustFixture(moduleCase));
}

for (const item of [
  [["src/lib.rs"], "rust runtime changes run rust checks", runtimeRust],
  [[" src\\lib.rs "], "trimmed Windows Rust source path runs rust checks", runtimeRust],
  [["src/lib.rs"], "rust test-only changes run rust checks", rustTestOnly],
  [["protected/deepbom_wasm/src/lib.rs"], "DEEPBOM rust changes run rust checks", {
    "protected/deepbom_wasm/src/lib.rs": ["fn score() -> f64 { 0.1 }\n", "fn score() -> f64 { 0.2 }\n"],
  }],
  [["Cargo.toml"], "Cargo manifest changes run rust checks"],
  [["native/runtime_collector/src/main.rs"], "native collector changes run rust checks", {
    "native/runtime_collector/src/main.rs": ["fn main() {}\n", "fn main() { println!(\"capture\"); }\n"],
  }],
  [["native/instrumentation/deepbom_runtime_trace.cc"], "native trace sink changes run native checks"],
  [["package.json"], "package changes run rust checks", { "package.json": [beforePackage, afterCheckPackage] }],
]) expectRustCheck(...item);
for (const moduleCase of privateModuleCases) {
  expectRustCheck([moduleCase.primarySource], `private ${moduleCase.id} rust changes run rust checks`, privateRustFixture(moduleCase));
  expectRustCheck([moduleCase.cargoManifest], `private ${moduleCase.id} manifest changes run rust checks`);
}

for (const item of [
  [["scripts/private-wasm-modules.mjs"], "private WASM registry changes run private WASM smoke"],
  [["scripts/check-private-wasm-build.mjs"], "private WASM build checker changes run private WASM smoke"],
]) expectPrivateWasmCheck(...item);
for (const moduleCase of privateModuleCases) {
  expectPrivateWasmCheck([moduleCase.primarySource], `private ${moduleCase.id} source runs private WASM smoke`, privateRustFixture(moduleCase));
  expectPrivateWasmCheck([moduleCase.cargoManifest], `private ${moduleCase.id} manifest runs private WASM smoke`);
}

for (const item of [
  [["web/app.js"], "web-only changes skip rust checks"],
  [["docs/PROJECT_STATUS.md"], "docs skip rust checks"],
]) expectNoRustCheck(...item);

expectEqual(
  stripRustTests("#[cfg(test)]\nmod tests { fn helper() { let x = \"}\"; } }\nfn live() {}\n"),
  "fn live() {}",
  "stripRustTests should remove cfg(test) module without touching live code.",
);
expectEqual(packageJsonDeployReason(beforePackage, afterCheckPackage), "", "packageJsonDeployReason should skip check-script-only changes.");
expect(Boolean(packageJsonDeployReason(beforePackage, afterRuntimePackage)), "packageJsonDeployReason should deploy dependency/build changes.");
expectEqual(packageJsonRustCheckReason(beforePackage, afterDeployCheckPackage), "", "Deployment-only package scripts should skip Rust checks.");
expect(Boolean(packageJsonRustCheckReason(beforePackage, afterCheckPackage)), "Rust check script changes should run Rust checks.");
expectEqual(packageJsonPrivateWasmCheckReason(beforePackage, afterDeployCheckPackage), "", "Deployment-only package scripts should skip private WASM checks.");
for (const item of [
  [["base", "head"], { before: "base", after: "head", workingTree: false }, "parseDeployArgs should preserve commit range mode."],
  [["HEAD", "--working-tree"], { before: "HEAD", after: "--working-tree", workingTree: true }, "parseDeployArgs should support base followed by --working-tree."],
  [["--working-tree", "main"], { before: "main", after: "--working-tree", workingTree: true }, "parseDeployArgs should support --working-tree followed by base."],
]) expectDeepEqual(parseDeployArgs(item[0]), item[1], item[2]);
expect(
  Boolean(rustCheckReasonForFile("src/lib.rs", { readBefore: () => "fn a() {}\n", readAfter: () => "fn b() {}\n" })),
  "rustCheckReasonForFile should flag Rust source changes.",
);
for (const moduleCase of privateModuleCases) {
  expect(
    Boolean(privateWasmCheckReasonForFile(moduleCase.primarySource)),
    `privateWasmCheckReasonForFile should flag ${moduleCase.id} private optional WASM source changes.`,
  );
}

for (const stepName of [
  "Prepare Rust WASM target",
  "Install pinned wasm-pack binary",
]) expectStepIf(stepName, "steps.deployable.outputs.changed == 'true' || steps.deployable.outputs.private_wasm_check == 'true'");
expectStepIf("Run private WASM build/load smoke", "steps.deployable.outputs.private_wasm_check == 'true'");
for (const stepName of [
  "Set rolling build expiry",
  "Build Worker static asset artifact",
  "Check static deploy artifact",
]) expectStepIf(stepName, "steps.deployable.outputs.changed == 'true'");
for (const stepName of [
  "Write route-preserving Cloudflare deploy config",
  "Deploy to Cloudflare Workers Static Assets",
  "Purge configured Cloudflare zone cache",
]) {
  expectStepIf(stepName, "steps.deployable.outputs.changed == 'true' && steps.cloudflare-config.outputs.configured == 'true'");
}

done("CI deploy contract passed (deployable, check-only, package, and rust test-only cases).");

function packageText({
  buildWorker = "npm run build:wasm && node scripts/build-pages.mjs",
  dependency = "^2.5.2",
  extraScripts = {},
} = {}) {
  return JSON.stringify({
    scripts: { "build:worker": buildWorker, check: "node scripts/check-all.mjs", ...extraScripts },
    dependencies: { "@litertjs/core": dependency },
  });
}

function expectDeploy(files, label, contents = {}) {
  const decision = decisionFor(files, contents);
  expect(decision.changed, `${label}: expected deploy, got skip.`);
}

function expectSkip(files, label, contents = {}) {
  const decision = decisionFor(files, contents);
  expect(!decision.changed, `${label}: expected skip, got deploy (${decision.details.join("; ")}).`);
}

function expectRustCheck(files, label, contents = {}) {
  const decision = decisionFor(files, contents);
  expect(decision.rustCheck, `${label}: expected rust check.`);
}

function expectNoRustCheck(files, label, contents = {}) {
  const decision = decisionFor(files, contents);
  expect(!decision.rustCheck, `${label}: expected no rust check (${decision.rustCheckDetails.join("; ")}).`);
}

function expectPrivateWasmCheck(files, label, contents = {}) {
  const decision = decisionFor(files, contents);
  expect(decision.privateWasmCheck, `${label}: expected private WASM build/load smoke.`);
}

function decisionFor(files, contents) {
  return classifyChangeSet(files, {
    readBefore: (file) => contents[file]?.[0] || "",
    readAfter: (file) => contents[file]?.[1] || "",
  });
}

function expectWorkflowContains(text) {
  expect(workflow.includes(text), `Workflow should contain: ${text}`);
}

function expectStepIf(stepName, expectedIf) {
  const block = sourceBlockForStep(workflow, escapeRegExp(stepName));
  expect(Boolean(block), `Workflow step is missing: ${stepName}.`);
  if (block) expect(block.includes(`if: ${expectedIf}`), `Workflow step "${stepName}" must be gated by: if: ${expectedIf}`);
}

function sourceBlockForStep(source, escapedName) {
  return new RegExp(`- name: ${escapedName}[\\s\\S]*?(?=\\n\\s+- name: |\\n\\s*$)`).exec(source)?.[0] || "";
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function privateRustFixture(moduleCase) {
  const probeName = `${moduleCase.id}_ci_probe`;
  return {
    [moduleCase.primarySource]: [
      `pub fn ${probeName}() -> u32 { 1 }\n`,
      `pub fn ${probeName}() -> u32 { 2 }\n`,
    ],
  };
}
