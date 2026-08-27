import { execFileSync } from "node:child_process";

execFileSync(process.execPath, ["scripts/generate-gguf-backend-contract.mjs", "--check"], {
  cwd: new URL("..", import.meta.url),
  stdio: "inherit",
});

console.log("GGUF backend source contract check passed.");
