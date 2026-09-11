#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const VERSION = "1.96.15";
const executable = process.platform === "win32" ? process.env.ComSpec || "cmd.exe" : "npx";
const args = process.platform === "win32"
  ? ["/d", "/s", "/c", "npx.cmd", "-y", `deepbom@${VERSION}`, "self-test", "--compact"]
  : ["-y", `deepbom@${VERSION}`, "self-test", "--compact"];
const result = spawnSync(executable, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
if (result.status !== 0) {
  process.stderr.write(result.stderr || result.stdout || `DEEPBOM self-test exited ${result.status}.\n`);
  process.exit(result.status || 1);
}
const document = JSON.parse(result.stdout);
if (document.schema !== "deepbom.cli_self_test.v1" || document.status !== "pass") {
  throw new Error("DEEPBOM self-test did not return a passing deepbom.cli_self_test.v1 document.");
}
process.stdout.write(`${JSON.stringify(document)}\n`);
