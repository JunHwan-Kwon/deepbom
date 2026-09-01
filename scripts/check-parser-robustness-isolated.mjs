import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SUITE = path.join(ROOT, "scripts", "check-parser-robustness.mjs");
const OUTPUT_LIMIT = 2 * 1024 * 1024;
const RESULTS_DIR = path.join(ROOT, ".local-validation", "1.96-stabilization");

const listed = await runProcess({
  name: "case discovery",
  args: [SUITE],
  env: { DEEPBOM_PARSER_ROBUSTNESS_MODE: "list" },
  timeoutMs: 30000,
});
if (listed.status !== "accepted") fail(`Parser case discovery failed: ${listed.detail}`);
const descriptors = extractJson(listed.stdout, "DEEPBOM_PARSER_CASES_JSON=");
if (!Array.isArray(descriptors) || !descriptors.length) fail("Parser case discovery returned no cases.");

const results = [];
for (const descriptor of descriptors) {
  const execution = await runProcess({
    name: descriptor.id,
    args: ["--max-old-space-size=384", SUITE],
    env: {
      DEEPBOM_PARSER_ROBUSTNESS_MODE: "case",
      DEEPBOM_PARSER_ROBUSTNESS_CASE: descriptor.id,
    },
    timeoutMs: Number(descriptor.timeout_ms || 15000),
  });
  let row = null;
  if (!["timeout", "resource_limit"].includes(execution.status)) {
    try { row = extractJson(execution.stdout, "DEEPBOM_PARSER_RESULT_JSON="); }
    catch (error) {
      if (execution.status === "accepted") {
        execution.status = "crash";
        execution.detail = error.message;
      }
    }
  }
  const status = row?.status || execution.status;
  const verdict = row?.verdict || (execution.status === "accepted" ? "FAIL" : execution.status);
  results.push({
    ...descriptor,
    status,
    verdict,
    duration_ms: execution.duration_ms,
    detail: row?.detail || execution.detail,
    exit_code: execution.exit_code,
    signal: execution.signal,
  });
  const mark = verdict === "ok" ? "ok" : "FAIL";
  console.log(`${mark.padEnd(4)} [${status.padEnd(14)}] ${descriptor.id}`);
}

const harness = [
  await runProcess({
    name: "isolation synchronous CPU loop",
    args: ["-e", "while (true) {}"],
    timeoutMs: 1200,
  }),
  await runProcess({
    name: "isolation heap exhaustion",
    args: ["--max-old-space-size=16", "-e", "const x=[]; while (true) x.push(new Array(1e6).fill(1));"],
    timeoutMs: 15000,
  }),
];
results.push({ id: "isolation-harness:cpu-loop", group: "isolation-harness", name: "synchronous CPU loop", expectation: "timeout", ...harness[0], verdict: harness[0].status === "timeout" ? "ok" : "FAIL" });
results.push({ id: "isolation-harness:heap-exhaustion", group: "isolation-harness", name: "heap exhaustion", expectation: "resource_limit", ...harness[1], verdict: harness[1].status === "resource_limit" ? "ok" : "FAIL" });

const summary = {
  schema: "deepbom.parser_robustness_isolation.v1",
  generated_at: new Date().toISOString(),
  process_boundary: "one_node_process_per_case",
  outcome_vocabulary: ["accepted", "rejected", "timeout", "crash", "resource_limit"],
  case_count: results.length,
  unexpected_count: results.filter((row) => row.verdict !== "ok").length,
  results,
};
await mkdir(RESULTS_DIR, { recursive: true });
await writeFile(path.join(RESULTS_DIR, "parser-robustness-isolated.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(`Parser isolation: ${summary.case_count} cases, ${summary.unexpected_count} unexpected.`);
if (summary.unexpected_count) process.exitCode = 1;

function extractJson(output, marker) {
  const line = String(output).split(/\r?\n/).find((value) => value.startsWith(marker));
  if (!line) throw new Error(`Missing child result marker ${marker}`);
  return JSON.parse(line.slice(marker.length));
}

function runProcess({ name, args, env = {}, timeoutMs }) {
  return new Promise((resolve) => {
    const started = performance.now();
    const child = spawn(process.execPath, args, {
      cwd: ROOT,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    let outputExceeded = false;
    let timedOut = false;
    const collect = (kind) => (chunk) => {
      const text = chunk.toString("utf8");
      if (kind === "stdout") stdout += text;
      else stderr += text;
      if (Buffer.byteLength(stdout) + Buffer.byteLength(stderr) > OUTPUT_LIMIT) {
        outputExceeded = true;
        child.kill("SIGKILL");
      }
    };
    child.stdout.on("data", collect("stdout"));
    child.stderr.on("data", collect("stderr"));
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);
    child.on("error", (error) => {
      clearTimeout(timer);
      resolve(result("crash", error.message, null, null));
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      const elapsed = Math.round(performance.now() - started);
      const resource = /heap out of memory|allocation failed|reached heap limit|fatal process out of memory/i.test(stderr);
      const status = timedOut ? "timeout"
        : outputExceeded ? "resource_limit"
          : resource ? "resource_limit"
            : code === 0 ? "accepted" : "crash";
      const detail = timedOut ? `terminated after ${timeoutMs} ms`
        : outputExceeded ? `combined child output exceeded ${OUTPUT_LIMIT} bytes`
          : resource ? "child process exceeded its configured heap budget"
            : code === 0 ? "child completed" : `child exited with code ${code}${signal ? ` signal ${signal}` : ""}`;
      resolve({ status, detail, stdout, stderr, exit_code: code, signal, duration_ms: elapsed });
    });
    function result(status, detail, exitCode, signal) {
      return { status, detail, stdout, stderr, exit_code: exitCode, signal, duration_ms: Math.round(performance.now() - started) };
    }
  });
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
