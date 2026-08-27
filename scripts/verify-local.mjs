import { runCommand, runNpm } from "./run-utils.mjs";

const parityArgs = process.argv.slice(2);

await runNpm(["run", "check:deployable"]);
await runNpm(["run", "check"]);
await runCommand("git", ["diff", "--check"]);
await runNpm(["run", "check:rust"]);
await runNpm(["run", "report:source-size", "--", "12"]);
await runNpm(["run", "check:source-budget"]);
await runNpm(["run", "check:private-wasm-build"]);
await runNpm(["run", "build:worker"]);
await runNpm(["run", "check:dist-assets"]);
await runNpm(["run", "check:explorer-redesign-dist"]);
await runNpm(["run", "check:protected-selector-viewer"]);
await runNpm(["run", "check:dist-budget"]);
await runNpm(["run", "report:dist-size", "--", "12"]);

if (parityArgs.length) {
  await runNpm(["run", "parity:visualizers", "--", ...parityArgs]);
} else {
  console.log("\n> parity:visualizers skipped; pass model paths after `--` to include parity.");
}

console.log("\nLocal verification passed.");
