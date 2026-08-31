import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { collectFileSizes } from "./size-utils.mjs";

const requiredRoots = ["web", "scripts"];
for (const root of requiredRoots) {
  if (!existsSync(root)) throw new Error(`Required JavaScript source root is missing: ${root}`);
}
const roots = [
  "web",
  "worker",
  "scripts",
].filter((root) => existsSync(root));
const deadFunctionGuardFiles = [
  "web/app.js",
  "web/admin.js",
  "worker/index.js",
].filter((file) => existsSync(file));

const files = (await Promise.all(roots.map((root) => collectFileSizes(root, {
  relativeRoot: ".",
  extensions: new Set([".js", ".mjs"]),
  ignoredDirs: new Set(["node_modules", "vendor"]),
}))))
  .flat()
  .map((file) => file.path)
  .sort();

for (const file of files) {
  const result = spawnSync(process.execPath, ["--check", file], {
    encoding: "utf8",
    stdio: "pipe",
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
    throw new Error(`Syntax check failed: ${file}`);
  }
}

const deadFunctions = deadFunctionGuardFiles.flatMap((file) =>
  unusedFunctionDeclarations(file).map((name) => `${file}:${name}`),
);
if (deadFunctions.length) {
  throw new Error(`Unused function declarations: ${deadFunctions.join(", ")}`);
}

console.log(`JavaScript syntax check passed (${files.length} files, dead-function guard enabled for ${deadFunctionGuardFiles.length} entry modules).`);

function unusedFunctionDeclarations(file) {
  const source = readFileSync(file, "utf8");
  const names = [...source.matchAll(/^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(/gm)]
    .map((match) => match[1]);
  return [...new Set(names)].filter((name) => {
    const count = [...source.matchAll(new RegExp(`\\b${escapeRegExp(name)}\\b`, "g"))].length;
    return count <= 1;
  });
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
