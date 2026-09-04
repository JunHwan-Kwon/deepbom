import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(".");
const outputRoot = path.join(root, ".local-validation", "release-validation");
const manifestPath = path.join(outputRoot, "deepbom.release_validation.v1.json");
const logRoot = path.join(outputRoot, "logs");
const requested = new Set(process.argv.slice(2).filter((value) => value.startsWith("--only=")).flatMap((value) => value.slice(7).split(",")).filter(Boolean));
const selected = (name) => requested.size === 0 || requested.has(name);
const sourceCommit = capture("git", ["rev-parse", "HEAD"]).trim();
const initialState = sourceState();
if (initialState !== "clean") throw new Error("Release validation requires a clean source tree.");

await mkdir(logRoot, { recursive: true });
const packageDocument = JSON.parse(await readFile(path.join(root, "package.json"), "utf8"));
const startedAt = new Date().toISOString();
const manifest = {
  schema: "deepbom.release_validation.v1",
  status: "running",
  version: packageDocument.version,
  source: { git_commit: sourceCommit, git_state_before: initialState, git_state_after: null },
  runtime: {
    platform: process.platform,
    arch: process.arch,
    node: process.versions.node,
    npm: captureNpmVersion(),
    python: captureOptional("python", ["--version"]),
    rustc: captureOptional("rustc", ["--version"]),
    cargo: captureOptional("cargo", ["--version"]),
  },
  started_at: startedAt,
  completed_at: null,
  duration_seconds: null,
  commands: [],
  artifacts: {},
};
await writeManifest();

const expiry = releaseGuardEpochs();
const releaseEnvironment = {
  APP_NOT_BEFORE_EPOCH_MS: String(expiry.not_before_epoch_ms),
  APP_EXPIRES_AT_EPOCH_MS: String(expiry.expires_at_epoch_ms),
};

try {
  if (selected("quality")) {
    await run("quality", process.execPath, ["scripts/check-all.mjs"]);
    await run("source-budget", process.execPath, ["scripts/check-source-budget.mjs"]);
  }
  if (selected("deploy")) {
    await run("deploy-gate", process.execPath, ["scripts/check-deploy.mjs"]);
    await run("release-build", npmCommand(), npmArgs(["run", "build:release"]), releaseEnvironment);
    await run("dist-assets", process.execPath, ["scripts/check-dist-assets.mjs"]);
    await run("dist-budget", process.execPath, ["scripts/check-dist-budget.mjs"]);
  }
  if (selected("public")) {
    await run("public-source-build", process.execPath, ["scripts/build-public-source-export.mjs", "--export"]);
    await run("public-source-verify", process.execPath, ["scripts/verify-public-source-export.mjs"]);
  }
  if (selected("channels")) await run("channel-equivalence", process.execPath, ["scripts/check-channel-equivalence.mjs", "--release-contract"]);
  manifest.artifacts = await collectArtifactRecords();
  manifest.source.git_state_after = sourceState();
  if (manifest.source.git_state_after !== "clean") throw new Error("Release validation changed tracked source files.");
  manifest.status = "pass";
} catch (error) {
  manifest.status = "fail";
  manifest.failure = String(error?.stack || error);
  manifest.source.git_state_after = sourceState();
  throw error;
} finally {
  manifest.completed_at = new Date().toISOString();
  manifest.duration_seconds = secondsBetween(startedAt, manifest.completed_at);
  manifest.release_guard = expiry;
  await writeManifest();
  console.log(`Release validation manifest: ${manifestPath}`);
}

async function run(id, command, args, extraEnvironment = {}) {
  const started = new Date().toISOString();
  const logPath = path.join(logRoot, `${String(manifest.commands.length + 1).padStart(2, "0")}-${id}.log`);
  const record = { id, command: [command, ...args].join(" "), status: "running", started_at: started, completed_at: null, duration_seconds: null, log: portable(path.relative(root, logPath)) };
  manifest.commands.push(record);
  await writeManifest();
  const lines = [];
  const child = spawn(command, args, {
    cwd: root,
    env: { ...process.env, ...extraEnvironment },
    shell: false,
  });
  const append = (chunk, stream) => {
    const text = chunk.toString();
    lines.push(text);
    stream.write(text);
  };
  child.stdout.on("data", (chunk) => append(chunk, process.stdout));
  child.stderr.on("data", (chunk) => append(chunk, process.stderr));
  const code = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  await writeFile(logPath, lines.join(""), "utf8");
  record.completed_at = new Date().toISOString();
  record.duration_seconds = secondsBetween(started, record.completed_at);
  record.exit_code = code;
  record.reported_check_count = reportedCheckCount(lines.join(""));
  record.status = code === 0 ? "pass" : "fail";
  await writeManifest();
  if (code !== 0) throw new Error(`${id} failed with exit code ${code}.`);
}

async function collectArtifactRecords() {
  const candidates = [
    ["public_source_manifest", ".local-validation/public-source/PUBLIC_SOURCE_MANIFEST.json"],
    ["channel_release_manifest", ".local-validation/channel-release/channel-release-manifest.json"],
    ["ui_regression_baseline", ".local-validation/1.96-stabilization/ui/ui-regression-baseline.v1.json"],
    ["deployment_hardening", "dist/deployment-hardening.json"],
  ];
  const records = {};
  for (const [name, relative] of candidates) {
    try {
      const bytes = await readFile(path.join(root, relative));
      records[name] = { path: relative, byte_length: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
      if (name === "channel_release_manifest") {
        const channel = JSON.parse(bytes.toString("utf8"));
        records[name].source_git_commit = channel.source?.git_commit || null;
        records[name].source_git_state = channel.source?.git_state || null;
      } else if (name === "ui_regression_baseline") {
        const baseline = JSON.parse(bytes.toString("utf8"));
        if (baseline.schema !== "deepbom.ir_stabilization_ui_baseline.v1"
          || baseline.source_commit !== sourceCommit
          || !Array.isArray(baseline.rows) || baseline.rows.length < 16) {
          throw new Error("UI regression baseline is stale or incomplete for the release source commit.");
        }
        const semanticRows = baseline.rows.filter((row) => row.web_cli_semantic_digest?.status === "equal");
        const requiredSemanticFormats = ["tflite", "onnx", "coreml", "gguf", "safetensors", "executorch"];
        const observedSemanticFormats = new Set(semanticRows.map((row) => row.artifact_format));
        const missingSemanticFormats = requiredSemanticFormats.filter((format) => !observedSemanticFormats.has(format));
        if (missingSemanticFormats.length) {
          throw new Error(`UI regression baseline is missing Web/CLI semantic-digest equality for: ${missingSemanticFormats.join(", ")}.`);
        }
        records[name].source_git_commit = baseline.source_commit;
        records[name].row_count = baseline.rows.length;
        records[name].web_cli_semantic_digest_formats = semanticRows.map((row) => row.artifact_format).sort();
      }
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  return records;
}

function releaseGuardEpochs() {
  const days = Number.parseInt(process.env.DEEPBOM_VALIDATION_EXPIRY_DAYS || "30", 10);
  if (!Number.isSafeInteger(days) || days < 1 || days > 90) throw new Error("DEEPBOM_VALIDATION_EXPIRY_DAYS must be 1-90.");
  const utcDay = Math.floor(Date.now() / 86_400_000) * 86_400_000;
  return {
    derivation: "validation_utc_day",
    expiry_days: days,
    not_before_epoch_ms: utcDay - 86_400_000,
    expires_at_epoch_ms: utcDay + days * 86_400_000,
  };
}

function sourceState() {
  return capture("git", ["status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":!.local-validation", ":!dist", ":!web/lib/build-metadata.js"]).trim() ? "dirty" : "clean";
}

function capture(command, args) {
  const result = spawnCapture(command, args);
  if (result.status !== 0) throw new Error(`${command} ${args.join(" ")} failed: ${result.stderr}`);
  return result.stdout;
}

function captureOptional(command, args) {
  try { return capture(command, args).trim(); } catch { return "not_available"; }
}

function captureNpmVersion() {
  return captureOptional(npmCommand(), npmArgs(["--version"]));
}

function spawnCapture(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", shell: false });
  return { status: result.status, stdout: String(result.stdout || ""), stderr: String(result.stderr || "") };
}

function npmCommand() { return process.platform === "win32" ? process.env.ComSpec || "cmd.exe" : "npm"; }
function npmArgs(args) { return process.platform === "win32" ? ["/d", "/s", "/c", "npm.cmd", ...args] : args; }
function reportedCheckCount(output) {
  const matches = [...output.matchAll(/(?:passed \(|checks passed \()(?:(?:\d+-)?\d+\/)?(\d+)(?: checks)?/gi)];
  return matches.length ? Number(matches.at(-1)[1]) : null;
}
function secondsBetween(a, b) { return Number(((Date.parse(b) - Date.parse(a)) / 1000).toFixed(3)); }
function portable(value) { return value.split(path.sep).join("/"); }
async function writeManifest() { await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8"); }
