import {
  addRuntimeSourceBytes,
  collectSourceFiles,
  formatBytes,
  kibToBytes,
  sourceTotals,
} from "./source-size-utils.mjs";

// Budget history belongs in git/CHANGELOG; this executable keeps only current policy.
const handwrittenRuntimeBudgetKiB = Number(process.env.HANDWRITTEN_RUNTIME_SOURCE_BUDGET_KIB || process.env.SOURCE_BUDGET_KIB || 11392);
const generatedRuntimeDataBudgetKiB = Number(process.env.GENERATED_RUNTIME_DATA_BUDGET_KIB || 1536);
const docsSourceBudgetKiB = Number(process.env.DOCS_SOURCE_BUDGET_KIB || 352);
const verificationSourceBudgetKiB = Number(process.env.VERIFICATION_SOURCE_BUDGET_KIB || 3136);
const devToolingSourceBudgetKiB = Number(process.env.DEV_TOOLING_SOURCE_BUDGET_KIB || 1296);
const nativeToolingSourceBudgetKiB = Number(process.env.NATIVE_TOOLING_SOURCE_BUDGET_KIB || 256);
const corpusEvidenceSourceBudgetKiB = Number(process.env.CORPUS_EVIDENCE_SOURCE_BUDGET_KIB || 512);
const runtimeFileBudgetKiB = Number(process.env.RUNTIME_SOURCE_FILE_BUDGET_KIB || 512);
const appJsBudgetKiB = Number(process.env.APP_JS_BUDGET_KIB || 300);
const stylesCssBudgetKiB = Number(process.env.STYLES_CSS_BUDGET_KIB || 280);
const researchLabsCssBudgetKiB = Number(process.env.RESEARCH_LABS_CSS_BUDGET_KIB || 48);
const appShellCssBudgetKiB = Number(process.env.APP_SHELL_CSS_BUDGET_KIB || 64);
const deviceWorkspaceCssBudgetKiB = Number(process.env.DEVICE_WORKSPACE_CSS_BUDGET_KIB || 64);
const engineeringReportBudgetKiB = Number(process.env.ENGINEERING_REPORT_BUDGET_KIB || 352);
const rustLibBudgetKiB = Number(process.env.RUST_LIB_BUDGET_KIB || 420);
const privateRuntimeSourceBudgetKiB = Number(process.env.PRIVATE_RUNTIME_SOURCE_BUDGET_KIB || process.env.PRIVATE_SOURCE_BUDGET_KIB || 224);
const privateTestSourceBudgetKiB = Number(process.env.PRIVATE_TEST_SOURCE_BUDGET_KIB || 96);

const files = await addRuntimeSourceBytes(await collectSourceFiles());
const totals = sourceTotals(files);
const appJs = files.find((file) => file.path === "web/app.js");
const stylesCss = files.find((file) => file.path === "web/styles.css");
const researchLabsCss = files.find((file) => file.path === "web/research-labs.css");
const appShellCss = files.find((file) => file.path === "web/app-shell.css");
const deviceWorkspaceCss = files.find((file) => file.path === "web/device-workspace.css");
const engineeringReport = files.find((file) => file.path === "web/lib/report-engineering.js");
const rustLib = files.find((file) => file.path === "src/lib.rs");
const handwrittenRuntimeBudgetBytes = kibToBytes(handwrittenRuntimeBudgetKiB);
const generatedRuntimeDataBudgetBytes = kibToBytes(generatedRuntimeDataBudgetKiB);
const docsSourceBudgetBytes = kibToBytes(docsSourceBudgetKiB);
const verificationSourceBudgetBytes = kibToBytes(verificationSourceBudgetKiB);
const devToolingSourceBudgetBytes = kibToBytes(devToolingSourceBudgetKiB);
const nativeToolingSourceBudgetBytes = kibToBytes(nativeToolingSourceBudgetKiB);
const corpusEvidenceSourceBudgetBytes = kibToBytes(corpusEvidenceSourceBudgetKiB);
const runtimeFileBudgetBytes = kibToBytes(runtimeFileBudgetKiB);
const appJsBudgetBytes = kibToBytes(appJsBudgetKiB);
const stylesCssBudgetBytes = kibToBytes(stylesCssBudgetKiB);
const researchLabsCssBudgetBytes = kibToBytes(researchLabsCssBudgetKiB);
const appShellCssBudgetBytes = kibToBytes(appShellCssBudgetKiB);
const deviceWorkspaceCssBudgetBytes = kibToBytes(deviceWorkspaceCssBudgetKiB);
const engineeringReportBudgetBytes = kibToBytes(engineeringReportBudgetKiB);
const rustLibBudgetBytes = kibToBytes(rustLibBudgetKiB);
const privateRuntimeSourceBudgetBytes = kibToBytes(privateRuntimeSourceBudgetKiB);
const privateTestSourceBudgetBytes = kibToBytes(privateTestSourceBudgetKiB);
const runtimeFiles = files
  .filter((file) => file.runtimeSource && file.runtimeBytes > 0)
  .sort((a, b) => b.runtimeBytes - a.runtimeBytes);
const generatedRuntimeDataFiles = runtimeFiles.filter((file) => file.generatedRuntimeData);
const generatedRuntimeDataBytes = generatedRuntimeDataFiles.reduce((total, file) => total + file.runtimeBytes, 0);
const handwrittenRuntimeFiles = runtimeFiles.filter((file) => !file.generatedRuntimeData);
const oversizedRuntimeFiles = handwrittenRuntimeFiles.filter((file) => file.runtimeBytes > runtimeFileBudgetBytes);
const largestRuntimeFile = handwrittenRuntimeFiles[0];

if (totals.handwrittenRuntimeBytes > handwrittenRuntimeBudgetBytes) {
  throw new Error(`handwritten runtime source budget exceeded: ${formatBytes(totals.handwrittenRuntimeBytes)} > ${formatBytes(handwrittenRuntimeBudgetBytes)}. Split runtime responsibilities or set HANDWRITTEN_RUNTIME_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.generatedRuntimeDataBytes > generatedRuntimeDataBudgetBytes) {
  throw new Error(`generated pinned runtime-data budget exceeded: ${formatBytes(totals.generatedRuntimeDataBytes)} > ${formatBytes(generatedRuntimeDataBudgetBytes)}. Compact the generated rulepacks or set GENERATED_RUNTIME_DATA_BUDGET_KIB intentionally.`);
}
if (totals.docsBytes > docsSourceBudgetBytes) {
  throw new Error(`docs source budget exceeded: ${formatBytes(totals.docsBytes)} > ${formatBytes(docsSourceBudgetBytes)}. Set DOCS_SOURCE_BUDGET_KIB to adjust intentionally.`);
}
if (totals.verificationBytes > verificationSourceBudgetBytes) {
  throw new Error(`verification source budget exceeded: ${formatBytes(totals.verificationBytes)} > ${formatBytes(verificationSourceBudgetBytes)}. Split fixtures from executable checks or set VERIFICATION_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.devToolingBytes > devToolingSourceBudgetBytes) {
  throw new Error(`development tooling source budget exceeded: ${formatBytes(totals.devToolingBytes)} > ${formatBytes(devToolingSourceBudgetBytes)}. Split generator/build responsibilities or set DEV_TOOLING_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.nativeToolingBytes > nativeToolingSourceBudgetBytes) {
  throw new Error(`native tooling source budget exceeded: ${formatBytes(totals.nativeToolingBytes)} > ${formatBytes(nativeToolingSourceBudgetBytes)}. Split native collectors/instrumentation or set NATIVE_TOOLING_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.corpusEvidenceBytes > corpusEvidenceSourceBudgetBytes) {
  throw new Error(`corpus evidence source budget exceeded: ${formatBytes(totals.corpusEvidenceBytes)} > ${formatBytes(corpusEvidenceSourceBudgetBytes)}. Compact generated evidence or set CORPUS_EVIDENCE_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.privateRuntimeBytes > privateRuntimeSourceBudgetBytes) {
  throw new Error(`private optional runtime source budget exceeded: ${formatBytes(totals.privateRuntimeBytes)} > ${formatBytes(privateRuntimeSourceBudgetBytes)}. Split private modules or set PRIVATE_RUNTIME_SOURCE_BUDGET_KIB intentionally.`);
}
if (totals.privateTestBytes > privateTestSourceBudgetBytes) {
  throw new Error(`private optional test source budget exceeded: ${formatBytes(totals.privateTestBytes)} > ${formatBytes(privateTestSourceBudgetBytes)}. Split private tests or set PRIVATE_TEST_SOURCE_BUDGET_KIB intentionally.`);
}
if (oversizedRuntimeFiles.length) {
  throw new Error(
    `runtime source file budget exceeded: ${oversizedRuntimeFiles.map((file) => `${file.path}=${formatBytes(file.runtimeBytes)}`).join(", ")} > ${formatBytes(runtimeFileBudgetBytes)}. Split large runtime modules or set RUNTIME_SOURCE_FILE_BUDGET_KIB intentionally.`,
  );
}
if (!appJs) throw new Error("web/app.js is missing from source budget scan.");
if (appJs.bytes > appJsBudgetBytes) {
  throw new Error(`web/app.js budget exceeded: ${formatBytes(appJs.bytes)} > ${formatBytes(appJsBudgetBytes)}. Split browser code or set APP_JS_BUDGET_KIB intentionally.`);
}
for (const [file, budgetBytes, environmentName, responsibility] of [
  [stylesCss, stylesCssBudgetBytes, "STYLES_CSS_BUDGET_KIB", "visual domains"],
  [researchLabsCss, researchLabsCssBudgetBytes, "RESEARCH_LABS_CSS_BUDGET_KIB", "research-lab visual domains"],
  [appShellCss, appShellCssBudgetBytes, "APP_SHELL_CSS_BUDGET_KIB", "application shell and access-control visual domains"],
  [deviceWorkspaceCss, deviceWorkspaceCssBudgetBytes, "DEVICE_WORKSPACE_CSS_BUDGET_KIB", "device benchmark and audit-control visual domains"],
  [engineeringReport, engineeringReportBudgetBytes, "ENGINEERING_REPORT_BUDGET_KIB", "format-specific report assembly"],
  [rustLib, rustLibBudgetBytes, "RUST_LIB_BUDGET_KIB", "Rust analysis ownership"],
]) {
  if (!file) throw new Error(`Expected budgeted source file is missing for ${environmentName}.`);
  if (file.bytes > budgetBytes) {
    throw new Error(`${file.path} budget exceeded: ${formatBytes(file.bytes)} > ${formatBytes(budgetBytes)}. Split ${responsibility} or set ${environmentName} intentionally.`);
  }
}

console.log(
  `Source budget check passed (handwritten runtime ${formatBytes(totals.handwrittenRuntimeBytes)} / ${formatBytes(handwrittenRuntimeBudgetBytes)}, generated pinned runtime data ${formatBytes(generatedRuntimeDataBytes)} / ${formatBytes(generatedRuntimeDataBudgetBytes)}, verification ${formatBytes(totals.verificationBytes)} / ${formatBytes(verificationSourceBudgetBytes)}, development tooling ${formatBytes(totals.devToolingBytes)} / ${formatBytes(devToolingSourceBudgetBytes)}, docs ${formatBytes(totals.docsBytes)} / ${formatBytes(docsSourceBudgetBytes)}, corpus evidence ${formatBytes(totals.corpusEvidenceBytes)} / ${formatBytes(corpusEvidenceSourceBudgetBytes)}, native tooling ${formatBytes(totals.nativeToolingBytes)} / ${formatBytes(nativeToolingSourceBudgetBytes)}, private runtime ${formatBytes(totals.privateRuntimeBytes)} / ${formatBytes(privateRuntimeSourceBudgetBytes)}, private tests ${formatBytes(totals.privateTestBytes)} / ${formatBytes(privateTestSourceBudgetBytes)}, private total ${formatBytes(totals.privateBytes)}, total runtime ${formatBytes(totals.runtimeBytes)}, largest handwritten runtime file ${largestRuntimeFile ? `${largestRuntimeFile.path} ${formatBytes(largestRuntimeFile.runtimeBytes)}` : "n/a"} / ${formatBytes(runtimeFileBudgetBytes)}, web/app.js ${formatBytes(appJs.bytes)} / ${formatBytes(appJsBudgetBytes)}, web/styles.css ${formatBytes(stylesCss.bytes)} / ${formatBytes(stylesCssBudgetBytes)}, web/research-labs.css ${formatBytes(researchLabsCss.bytes)} / ${formatBytes(researchLabsCssBudgetBytes)}, web/app-shell.css ${formatBytes(appShellCss.bytes)} / ${formatBytes(appShellCssBudgetBytes)}, web/device-workspace.css ${formatBytes(deviceWorkspaceCss.bytes)} / ${formatBytes(deviceWorkspaceCssBudgetBytes)}, report-engineering.js ${formatBytes(engineeringReport.bytes)} / ${formatBytes(engineeringReportBudgetBytes)}, src/lib.rs ${formatBytes(rustLib.bytes)} / ${formatBytes(rustLibBudgetBytes)}).`,
);
