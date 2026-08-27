import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createCheck } from "./check-assert.mjs";

if (existsSync("scripts/generate-xnnpack-delegate-rulepack.mjs")) {
  execFileSync(process.execPath, ["scripts/generate-xnnpack-delegate-rulepack.mjs", "--check"], {
    cwd: process.cwd(),
    stdio: "inherit",
  });
}

const { done, expect, expectEqual } = createCheck("Generated XNNPACK delegate rulepack check");
const manifest = JSON.parse(readFileSync("reference/xnnpack-readme/rule-manifest-v2.21.0.json", "utf8"));
const versionDiff = JSON.parse(readFileSync("reference/xnnpack-readme/version-diff.json", "utf8"));
const rust = readFileSync("src/xnnpack_rulepack_generated.rs", "utf8");

expectEqual(manifest.source.checked_ref, "87bbf65b8d23d3f06912b1b2183587e1884bc45c", "source commit must remain exact");
expectEqual(manifest.source.tag_commit, "a481b10260dfdf833a1b16007eead49c1d7febf3", "v2.21.0 tag commit must remain distinct and exact");
expectEqual(manifest.source.sha256, "85524b3e6acfe5429b9d8b7c1ef47dde79a4543ccbe2f73a0e276f8e7e0eb93c", "README SHA-256 must remain exact");
expectEqual(manifest.coverage.fp32_operator_count, 41, "FP32 op count must remain source-derived");
expectEqual(manifest.coverage.quantized_operator_count, 24, "quantized op count must remain source-derived");
expectEqual(manifest.coverage.documented_operator_constraint_count, 133, "source constraint count must remain deterministic");
expectEqual(manifest.coverage.implemented_operator_constraint_count, 133, "all artifact-visible constraints must remain mapped");
expectEqual(manifest.coverage.unmapped_operator_constraint_count, 0, "unmapped source constraints must fail closed");
expectEqual((rust.match(/    XnnpackDocumentRule \{/g) || []).length, 65, "generated Rust must carry every precision/op rule");

const releases = new Map(versionDiff.releases.map((release) => [release.tag, release]));
for (const tag of ["v2.14.0", "v2.15.0", "v2.16.1", "v2.17.0", "v2.18.0", "v2.19.0", "v2.20.0", "v2.21.0"]) {
  const changes = releases.get(tag)?.changes_from_previous;
  if (!changes) throw new Error(`Missing version diff for ${tag}.`);
  if (tag !== "v2.15.0") {
    expectEqual(changes.fp32.changed_operator_constraints.length, 0, `${tag} FP32 op constraints should remain unchanged`);
    expectEqual(changes.quantized.changed_operator_constraints.length, 0, `${tag} quantized op constraints should remain unchanged`);
  }
}
expect(releases.get("v2.21.0")?.fp16_contract === "inherits_fp32_feature_parity", "v2.21 FP16 must be represented as inherited FP32 parity, not zero support.");

done(existsSync("scripts/generate-xnnpack-delegate-rulepack.mjs")
  ? "Generated XNNPACK delegate rulepack and private regeneration path passed."
  : "Public XNNPACK generated snapshot passed; private regeneration path is intentionally absent.");
