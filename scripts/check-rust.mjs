import { existsSync } from "node:fs";
import { privateModuleCargoManifests } from "./private-wasm-modules.mjs";
import { runCommand } from "./run-utils.mjs";

const SHIPPED_PROTECTED_CRATES = [
  "protected/deepbom_wasm/Cargo.toml",
];
const NATIVE_TOOL_CRATES = [
  "native/runtime_collector/Cargo.toml",
  "native/runtime_probe/Cargo.toml",
];
const PRIVATE_OPTIONAL_CRATES = privateModuleCargoManifests();

const CHECKS = [
  ["cargo", ["fmt", "--check"]],
  ["cargo", ["clippy", "--all-targets", "--", "-D", "warnings"]],
  ...[...SHIPPED_PROTECTED_CRATES, ...PRIVATE_OPTIONAL_CRATES].filter(existsSync).map((manifestPath) => [
    "cargo",
    ["clippy", "--manifest-path", manifestPath, "--all-targets", "--", "-D", "warnings"],
  ]),
  ...NATIVE_TOOL_CRATES.filter(existsSync).map((manifestPath) => ["cargo", ["clippy", "--manifest-path", manifestPath, "--all-targets", "--", "-D", "warnings"]]),
  ["cargo", ["test"]],
  ...[...SHIPPED_PROTECTED_CRATES, ...PRIVATE_OPTIONAL_CRATES].filter(existsSync).map((manifestPath) => [
    "cargo",
    ["test", "--manifest-path", manifestPath],
  ]),
  ...NATIVE_TOOL_CRATES.filter(existsSync).map((manifestPath) => ["cargo", ["test", "--manifest-path", manifestPath]]),
];

for (const [command, args] of CHECKS) {
  await runCommand(command, args);
}

await runCommand(process.execPath, ["scripts/check-native-capture-pipeline.mjs"]);

console.log(`Rust quality check passed (${CHECKS.length} checks).`);
