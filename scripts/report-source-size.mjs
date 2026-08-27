import {
  addRuntimeSourceBytes,
  collectSourceFiles,
  formatBytes,
  sourceTotals,
} from "./source-size-utils.mjs";

const topCount = Number(process.argv[2] || 15);

const files = await addRuntimeSourceBytes(await collectSourceFiles());
const totals = sourceTotals(files);
const largest = [...files].sort((a, b) => b.bytes - a.bytes).slice(0, Math.max(1, topCount));
const largestRuntime = [...files]
  .filter((file) => file.runtimeSource && file.runtimeBytes > 0)
  .sort((a, b) => b.runtimeBytes - a.runtimeBytes)
  .slice(0, Math.max(1, topCount));
const largestPrivate = [...files]
  .filter((file) => file.privateSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));
const largestPrivateRuntime = [...files]
  .filter((file) => file.privateSource && !file.privateTestSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));
const largestPrivateTest = [...files]
  .filter((file) => file.privateTestSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));
const largestDocs = [...files]
  .filter((file) => file.docsSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));
const largestNativeTooling = [...files]
  .filter((file) => file.nativeToolingSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));
const largestDev = [...files]
  .filter((file) => !file.runtimeSource && !file.privateSource && !file.docsSource && !file.nativeToolingSource)
  .sort((a, b) => b.bytes - a.bytes)
  .slice(0, Math.max(1, topCount));

console.log(`source_files=${files.length}`);
console.log(`source_total=${formatBytes(totals.sourceBytes)} (${totals.sourceBytes} bytes)`);
console.log(`public_source_total=${formatBytes(totals.publicSourceBytes)} (${totals.publicSourceBytes} bytes)`);
console.log(`public_code_source_total=${formatBytes(totals.publicCodeBytes)} (${totals.publicCodeBytes} bytes)`);
console.log(`handwritten_runtime_source_total=${formatBytes(totals.handwrittenRuntimeBytes)} (${totals.handwrittenRuntimeBytes} bytes)`);
console.log(`generated_runtime_data_total=${formatBytes(totals.generatedRuntimeDataBytes)} (${totals.generatedRuntimeDataBytes} bytes)`);
console.log(`docs_source_total=${formatBytes(totals.docsBytes)} (${totals.docsBytes} bytes)`);
console.log(`private_optional_source_total=${formatBytes(totals.privateBytes)} (${totals.privateBytes} bytes)`);
console.log(`private_optional_runtime_source_total=${formatBytes(totals.privateRuntimeBytes)} (${totals.privateRuntimeBytes} bytes)`);
console.log(`private_optional_test_source_total=${formatBytes(totals.privateTestBytes)} (${totals.privateTestBytes} bytes)`);
console.log(`runtime_source_total=${formatBytes(totals.runtimeBytes)} (${totals.runtimeBytes} bytes)`);
console.log(`native_tooling_source_total=${formatBytes(totals.nativeToolingBytes)} (${totals.nativeToolingBytes} bytes)`);
console.log(`dev_check_source_total=${formatBytes(totals.devBytes)} (${totals.devBytes} bytes)`);
console.log(`verification_source_total=${formatBytes(totals.verificationBytes)} (${totals.verificationBytes} bytes)`);
console.log(`development_tooling_source_total=${formatBytes(totals.devToolingBytes)} (${totals.devToolingBytes} bytes)`);
console.log(`largest_source_files_top=${largest.length}`);
for (const file of largest) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_runtime_source_files_top=${largestRuntime.length}`);
for (const file of largestRuntime) {
  const note = file.runtimeBytes === file.bytes ? "" : ` (raw ${formatBytes(file.bytes)})`;
  console.log(`${formatBytes(file.runtimeBytes).padStart(10)}  ${file.path}${note}`);
}
console.log(`largest_private_optional_source_files_top=${largestPrivate.length}`);
for (const file of largestPrivate) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_private_optional_runtime_source_files_top=${largestPrivateRuntime.length}`);
for (const file of largestPrivateRuntime) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_private_optional_test_source_files_top=${largestPrivateTest.length}`);
for (const file of largestPrivateTest) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_docs_source_files_top=${largestDocs.length}`);
for (const file of largestDocs) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_native_tooling_source_files_top=${largestNativeTooling.length}`);
for (const file of largestNativeTooling) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
console.log(`largest_dev_check_source_files_top=${largestDev.length}`);
for (const file of largestDev) {
  console.log(`${formatBytes(file.bytes).padStart(10)}  ${file.path}`);
}
