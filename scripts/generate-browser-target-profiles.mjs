import { readFile, writeFile } from "node:fs/promises";
import { initSync, target_profiles as targetProfiles } from "../pkg/tflite_wasm_audit.js";

const output = "web/lib/target-profiles.generated.js";
initSync({ module: await readFile("pkg/tflite_wasm_audit_bg.wasm") });
const profiles = targetProfiles();
if (!Array.isArray(profiles) || !profiles.length) throw new Error("TFLite WASM returned no target profiles.");

const source = [
  "// Generated from pkg/tflite_wasm_audit_bg.wasm target_profiles(). Do not edit manually.",
  `export const BROWSER_TARGET_PROFILES = Object.freeze(${JSON.stringify(profiles, null, 2)}.map((profile) => Object.freeze(profile)));`,
  "",
].join("\n");

if (process.argv.includes("--check")) {
  const existing = await readFile(output, "utf8").catch(() => "");
  if (existing !== source) throw new Error(`${output} is stale. Run npm run generate:browser-target-profiles.`);
  console.log(`Browser target profile registry is current (${profiles.length} profiles).`);
} else {
  await writeFile(output, source, "utf8");
  console.log(`Generated ${output} from TFLite WASM (${profiles.length} profiles).`);
}
