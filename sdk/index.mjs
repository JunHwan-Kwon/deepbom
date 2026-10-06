/** Public Node SDK. All analysis and policy evaluation run in the packaged CLI. */
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const cli = fileURLToPath(new URL("../bin/deepbom.mjs", import.meta.url));
const defaults = { timeoutMs: 300_000, maxOutputBytes: 64 * 1024 * 1024 };
const limitKeys = Object.keys(defaults);
const auditKeys = [...limitKeys, "scan", "output", "sections", "gate", "policy", "expectedSha256"];

export class DeepBomError extends Error {
  constructor(message, { exitCode = null, document = null, cause } = {}) {
    super(message, { cause });
    this.name = this.constructor.name;
    this.code = this.constructor.code;
    this.exitCode = exitCode;
    this.document = document;
  }
  static code = "DEEPBOM_ERROR";
}
export class DeepBomInvocationError extends DeepBomError { static code = "INVOCATION_FAILED"; }
export class DeepBomPolicyBlocked extends DeepBomError { static code = "POLICY_BLOCKED"; }
export class DeepBomIncompleteBinding extends DeepBomError { static code = "INCOMPLETE_BINDING"; }
export class DeepBomIdentityMismatch extends DeepBomError { static code = "IDENTITY_MISMATCH"; }
export class DeepBomTimeout extends DeepBomError { static code = "TIMEOUT"; }
export class DeepBomOutputTooLarge extends DeepBomError { static code = "OUTPUT_TOO_LARGE"; }

export async function capabilities(options = {}) {
  checkOptions(options, limitKeys);
  return invoke(["capabilities", "--compact"], options);
}

export async function audit(artifact, options = {}) {
  checkOptions(options, auditKeys);
  const sections = names(options.sections);
  const output = choice(options.output ?? (sections.length ? "analysis" : "envelope"), "output", ["analysis", "envelope", "cyclonedx", "sarif"]);
  if (sections.length && output !== "analysis") throw new TypeError("sections require output='analysis'");
  if (options.gate != null && options.policy != null) throw new TypeError("gate and policy are mutually exclusive");
  const argv = ["audit", localPath(artifact), "--scan", choice(options.scan ?? "auto", "scan", ["auto", "structure", "integrity", "full"])];
  if (sections.length) argv.push("--section", sections.join(","));
  if (output !== "analysis") argv.push("--output-format", output);
  argv.push("--compact");
  if (options.gate != null) argv.push("--gate", choice(options.gate, "gate", ["defects"]));
  if (options.policy != null) argv.push("--policy", choice(options.policy, "policy", ["engineering", "regulatory"]));
  if (options.expectedSha256 != null) {
    if (typeof options.expectedSha256 !== "string" || !/^[0-9a-f]{64}$/i.test(options.expectedSha256)) throw new TypeError("expectedSha256 must contain 64 hexadecimal characters");
    argv.push("--expected-sha256", options.expectedSha256.toLowerCase());
  }
  return invoke(argv, options);
}

export async function inspect(artifact, options = {}) {
  checkOptions(options, auditKeys.filter(key => !["output", "sections"].includes(key)));
  try {
    return summaryFrom(await audit(artifact, { ...options, sections: ["summary"] }));
  } catch (error) {
    // A policy rejection still contains the same summary shape as a success.
    if (error instanceof DeepBomPolicyBlocked && error.document) error.document = summaryFrom(error.document);
    throw error;
  }
}

export async function diff(baseline, candidate, options = {}) {
  checkOptions(options, [...limitKeys, "tensorsOnly"]);
  if (options.tensorsOnly != null && typeof options.tensorsOnly !== "boolean") throw new TypeError("tensorsOnly must be boolean");
  return invoke(["diff", localPath(baseline), localPath(candidate), "--compact", ...(options.tensorsOnly ? ["--tensors"] : [])], options);
}

export async function captureContract(artifact, options = {}) {
  checkOptions(options, limitKeys);
  return invoke(["contract", "capture", localPath(artifact), "--compact"], options);
}

export async function verifyContract(artifact, contract, options = {}) {
  checkOptions(options, limitKeys);
  return invoke(["verify", localPath(artifact), "--contract", localPath(contract), "--compact"], options);
}

export async function verifyBom(artifact, bom, options = {}) {
  checkOptions(options, [...limitKeys, "componentRef"]);
  const argv = ["verify", localPath(artifact), "--bom", localPath(bom), "--compact"];
  if (options.componentRef != null) {
    if (typeof options.componentRef !== "string" || !options.componentRef.trim() || options.componentRef.includes("\0")) throw new TypeError("componentRef must be a non-empty bom-ref");
    argv.push("--component-ref", options.componentRef.trim());
  }
  return invoke(argv, options);
}

function summaryFrom(selection) {
  const summary = selection.sections?.summary;
  if (selection.schema !== "deepbom.analysis_selection.v1" || summary?.schema !== "deepbom.review_summary.v1") {
    throw new DeepBomInvocationError("The engine returned an incompatible review summary.");
  }
  return summary;
}

function localPath(value) {
  if (typeof value !== "string" || !value || value.includes("\0") || /^[a-z][a-z0-9+.-]*:\/\//i.test(value)) throw new TypeError("SDK paths must be non-empty local filesystem paths");
  return path.resolve(value);
}
function checkOptions(options, allowed) {
  if (!options || typeof options !== "object" || ![Object.prototype, null].includes(Object.getPrototypeOf(options))) throw new TypeError("options must be a plain object with own properties");
  for (const key of Reflect.ownKeys(options)) {
    if (typeof key !== "string" || !allowed.includes(key)) throw new TypeError(`Unsupported SDK option: ${String(key)}`);
    if (!Object.getOwnPropertyDescriptor(options, key).enumerable) throw new TypeError(`SDK option must be enumerable: ${key}`);
  }
}
function choice(value, field, allowed) {
  if (!allowed.includes(value)) throw new TypeError(`${field} must be one of: ${allowed.join(", ")}`);
  return value;
}
function names(values) {
  if (values == null) return [];
  if (!Array.isArray(values) || values.some(value => typeof value !== "string" || !/^[a-z0-9_.-]+$/i.test(value))) throw new TypeError("sections must be an array of section names");
  return [...new Set(values)];
}

async function invoke(argv, options) {
  const timeoutMs = options.timeoutMs ?? defaults.timeoutMs;
  const maxOutputBytes = options.maxOutputBytes ?? defaults.maxOutputBytes;
  for (const [key, value] of Object.entries({ timeoutMs, maxOutputBytes })) {
    if (!Number.isSafeInteger(value) || value <= 0 || (key === "timeoutMs" && value > 2_147_483_647)) throw new TypeError(`${key} must be a positive bounded integer`);
  }
  let directory;
  let failure;
  try {
    directory = await mkdtemp(path.join(tmpdir(), "deepbom-sdk-"));
    const output = path.join(directory, "result.json");
    const { exitCode, diagnostic } = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [cli, ...argv, "--output", output], { shell: false, windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
      let diagnostic = Buffer.alloc(0);
      let timedOut = false;
      let spawnError;
      const timer = setTimeout(() => { timedOut = true; child.kill("SIGKILL"); }, timeoutMs);
      child.stderr.on("data", chunk => { diagnostic = Buffer.concat([diagnostic, chunk]).subarray(-16_384); });
      child.on("error", error => { spawnError = error; });
      child.on("close", (exitCode, signal) => {
        clearTimeout(timer);
        if (timedOut) reject(new DeepBomTimeout(`DEEPBOM exceeded timeoutMs=${timeoutMs}.`));
        else if (spawnError) reject(new DeepBomInvocationError("Could not start the packaged DEEPBOM engine.", { cause: spawnError }));
        else if (signal) reject(new DeepBomInvocationError(`DEEPBOM was terminated by ${signal}.`));
        else resolve({ exitCode, diagnostic: diagnostic.toString("utf8").trim() });
      });
    });
    let document = null;
    const info = await stat(output).catch(error => { if (error.code === "ENOENT") return null; throw error; });
    if (info) {
      if (info.size > maxOutputBytes) throw new DeepBomOutputTooLarge(`DEEPBOM result is ${info.size} bytes; maxOutputBytes is ${maxOutputBytes}.`, { exitCode });
      try {
        document = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(await readFile(output)), (_key, value) => {
          if (typeof value === "number" && !Number.isFinite(value)) throw new Error("JSON number exceeds the finite floating-point range");
          return value;
        });
      }
      catch (cause) { throw new DeepBomInvocationError("DEEPBOM returned invalid JSON.", { exitCode, cause }); }
      if (!document || typeof document !== "object" || Array.isArray(document)) throw new DeepBomInvocationError("DEEPBOM JSON result must be an object.", { exitCode });
    }
    if (exitCode === 0 && document) return document;
    const Failure = { 2: DeepBomPolicyBlocked, 3: DeepBomIncompleteBinding, 4: DeepBomIdentityMismatch }[exitCode] || DeepBomInvocationError;
    throw new Failure(diagnostic || `DEEPBOM exited ${exitCode} without a usable result.`, { exitCode, document });
  } catch (error) {
    failure = error instanceof DeepBomError ? error : new DeepBomInvocationError("Could not invoke or read the packaged DEEPBOM engine.", { cause: error });
    throw failure;
  } finally {
    if (directory) {
      try { await rm(directory, { recursive: true, force: true }); }
      catch (cause) { if (!failure) throw new DeepBomInvocationError("Could not clean up DEEPBOM temporary output.", { cause }); }
    }
  }
}
