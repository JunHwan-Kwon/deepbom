import { collectFileSizes, formatBytes, kibToBytes, mibToBytes } from "./size-utils.mjs";
import { normalizePath } from "./path-utils.mjs";
import { privateModuleValidationCases } from "./private-wasm-modules.mjs";
import {
  canonicalSourceText,
  DEFAULT_IGNORED_DIRS,
  isDocsSourcePath,
  isIgnoredSourceFile,
  isPrivateSourcePath,
  isPrivateTestSourcePath,
  isRuntimeSourcePath,
  sourceTotals,
} from "./source-size-utils.mjs";
import { stripRustTests } from "./rust-source-utils.mjs";
import { createCheck } from "./check-assert.mjs";

const { done, expectEqual } = createCheck("Size utility contract check");
const privateModuleCases = privateModuleValidationCases();

expectEqual(formatBytes(1023), "1023 B", "formatBytes should keep byte values below 1 KiB.");
expectEqual(formatBytes(1024), "1.0 KiB", "formatBytes should format KiB with one decimal.");
expectEqual(formatBytes(1024 * 1024), "1.00 MiB", "formatBytes should format MiB with two decimals.");
expectEqual(kibToBytes(64), 65536, "kibToBytes should convert KiB to bytes.");
expectEqual(mibToBytes(16), 16777216, "mibToBytes should convert MiB to bytes.");
expectEqual(typeof collectFileSizes, "function", "collectFileSizes should be available to dist/source report tools.");
expectEqual(normalizePath("web\\app.js"), "web/app.js", "normalizePath should normalize Windows separators.");
expectEqual(canonicalSourceText("a\r\nb\rc\n"), "a\nb\nc\n", "source budgets should be independent of checkout line endings.");

expectEqual(isRuntimeSourcePath("web/app.js"), true, "web app code should count as runtime source.");
expectEqual(isRuntimeSourcePath(" web\\app.js "), true, "runtime path detection should preserve legacy trim and separator behavior.");
expectEqual(isRuntimeSourcePath("worker/index.js"), true, "worker code should count as runtime source.");
expectEqual(isRuntimeSourcePath("src/lib.rs"), true, "main Rust WASM code should count as runtime source.");
expectEqual(isRuntimeSourcePath("protected/deepbom_wasm/src/lib.rs"), true, "protected WASM module code should count as runtime source.");
expectEqual(isPrivateSourcePath("protected/deepbom_wasm/src/lib.rs"), false, "shipped protected DEEPBOM source should remain in runtime budget.");
for (const moduleCase of privateModuleCases) {
  expectEqual(isRuntimeSourcePath(moduleCase.primarySource), false, `${moduleCase.id} private optional WASM code should not count as public runtime source.`);
  expectEqual(isPrivateSourcePath(moduleCase.primarySource), true, `${moduleCase.id} private optional WASM code should have a separate source budget.`);
  expectEqual(isPrivateSourcePath(moduleCase.sourceRoot), false, `${moduleCase.id} source root without trailing file prefix should not match accidentally.`);
  expectEqual(isPrivateTestSourcePath(moduleCase.testSource), true, `${moduleCase.id} private optional tests should have their own source budget.`);
  expectEqual(isPrivateTestSourcePath(moduleCase.primarySource), false, `${moduleCase.id} private optional production code should not count as private test source.`);
}
expectEqual(isRuntimeSourcePath("scripts/check-all.mjs"), false, "local check scripts should not count as runtime source.");
expectEqual(isRuntimeSourcePath("docs/PROJECT_STATUS.md"), false, "docs should not count as runtime source.");
expectEqual(isDocsSourcePath("README.md"), true, "root README should count as docs source.");
expectEqual(isDocsSourcePath("docs/PROJECT_STATUS.md"), true, "public docs should count as docs source.");
expectEqual(isDocsSourcePath("docs-site/src/content/docs/index.mdx"), true, "the standalone documentation site should count as docs source.");
expectEqual(isDocsSourcePath("web/app.js"), false, "runtime app code should not count as docs source.");
expectEqual(isIgnoredSourceFile("LOCAL_PRIVATE_ROADMAP.local.md"), true, "local-private roadmap should not count toward source budget.");
expectEqual(isIgnoredSourceFile("package-lock.json"), true, "dependency lockfile should stay tracked but not count toward source budget.");
expectEqual(isIgnoredSourceFile("docs-site/package-lock.json"), true, "nested documentation lockfiles should not count toward source budget.");
expectEqual(isIgnoredSourceFile("PUBLIC_SOURCE_MANIFEST.json"), true, "generated public-source provenance should not count as handwritten development source.");
expectEqual(DEFAULT_IGNORED_DIRS.has("node_modules.incomplete"), true, "incomplete dependency staging directories should not count toward source budget.");
expectEqual(DEFAULT_IGNORED_DIRS.has(".wrangler"), true, "local Worker build output should not count toward source budget.");
expectEqual(isIgnoredSourceFile("docs/PROJECT_STATUS.md"), false, "public project status should count toward source budget.");

const totals = sourceTotals([
  { bytes: 10, runtimeBytes: 8, privateSource: false },
  { bytes: 4, runtimeBytes: 0 },
  { bytes: 6, privateSource: true },
  { bytes: 2, privateSource: true, privateTestSource: true },
  { bytes: 3, runtimeBytes: 0, docsSource: true },
]);
expectEqual(totals.sourceBytes, 25, "sourceTotals should sum raw source bytes.");
expectEqual(totals.publicSourceBytes, 17, "sourceTotals should keep public source distinct from private optional source.");
expectEqual(totals.publicCodeBytes, 8, "sourceTotals should use runtime-stripped bytes as public code source.");
expectEqual(totals.runtimeBytes, 8, "sourceTotals should sum runtime bytes only.");
expectEqual(totals.privateBytes, 8, "sourceTotals should sum private optional source separately.");
expectEqual(totals.privateRuntimeBytes, 6, "sourceTotals should keep private optional runtime source distinct from tests.");
expectEqual(totals.privateTestBytes, 2, "sourceTotals should keep private optional tests visible.");
expectEqual(totals.docsBytes, 3, "sourceTotals should sum docs source separately.");
expectEqual(totals.devBytes, 6, "sourceTotals should keep non-runtime non-private non-doc source visible.");

const stripped = stripRustTests(`
fn live() -> &'static str { "}" }

#[cfg(test)]
mod tests {
  #[test]
  fn keeps_parser_balanced() {
    assert_eq!("{", "{");
  }
}
`);
expectEqual(stripped, `fn live() -> &'static str { "}" }`, "stripRustTests should remove cfg(test) modules without touching live code.");

done("Size utility contract passed (bytes, budgets, runtime paths, and Rust test stripping).");
