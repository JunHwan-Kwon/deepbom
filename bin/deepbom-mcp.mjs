import { existsSync, realpathSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

import { AUDIT_DEFAULT_OUTPUT_FORMAT, AUDIT_OUTPUT_FORMATS } from "../web/lib/audit-output-contracts.js";

// Model Context Protocol server over stdio. Each analysis runs in a child
// process so cancellation can terminate work without corrupting the JSON-RPC
// transport or leaving the server unable to answer control requests.

const PROTOCOL_VERSIONS = Object.freeze(["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]);
const SCAN_MODES = Object.freeze(["auto", "structure", "integrity", "full"]);
const MAX_RESPONSE_BYTES = boundedEnvironmentInteger("DEEPBOM_MCP_MAX_RESPONSE_BYTES", 4 * 1024 * 1024, 64 * 1024, 64 * 1024 * 1024);
const MAX_FRAME_BYTES = boundedEnvironmentInteger("DEEPBOM_MCP_MAX_FRAME_BYTES", 1024 * 1024, 4096, 8 * 1024 * 1024);
const MAX_STDERR_BYTES = boundedEnvironmentInteger("DEEPBOM_MCP_MAX_STDERR_BYTES", 1024 * 1024, 4096, 8 * 1024 * 1024);
const TOOL_TIMEOUT_MS = boundedEnvironmentInteger("DEEPBOM_MCP_TOOL_TIMEOUT_MS", 15 * 60 * 1000, 1000, 60 * 60 * 1000);
const MAX_CONCURRENT_TOOLS = boundedEnvironmentInteger("DEEPBOM_MCP_MAX_CONCURRENT", 2, 1, 8);
const MAX_QUEUED_TOOLS = boundedEnvironmentInteger("DEEPBOM_MCP_MAX_QUEUE", 8, 0, 64);

const READ_ONLY_ANNOTATIONS = Object.freeze({
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
});
const CACHE_WRITING_ANNOTATIONS = Object.freeze({
  readOnlyHint: false,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: true,
});

const TOOLS = Object.freeze([
  {
    name: "deepbom_capabilities",
    title: "Inspect DEEPBOM capabilities",
    description: "Return deepbom.cli_capabilities.v1: supported artifact formats, output contracts, scan policies, target profiles, and exit codes. Call this before assuming an option exists.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    outputSchema: { type: "object", required: ["schema"], properties: { schema: { const: "deepbom.cli_capabilities.v1" } }, additionalProperties: true },
    annotations: READ_ONLY_ANNOTATIONS,
  },
  {
    name: "deepbom_audit",
    title: "Audit a deployed model artifact",
    description: "Audit one local or immutable remote model artifact and return local static evidence. Supported remote identities are hf:// with a full commit, gs:// with an object generation, and HTTPS with SHA-256. Remote bytes may be downloaded into a content-addressed local cache but are never uploaded by DEEPBOM. Findings distinguish artifact_defect, caution, and evidence_gap; an evidence_gap is not a defect. The result never establishes executed accelerator assignment, latency, energy, accuracy, or device fit.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Local artifact/package path under DEEPBOM_MCP_ALLOWED_ROOTS, or an immutable hf://, gs://, or hash-pinned HTTPS source." },
        output_format: {
          type: "string",
          enum: [...AUDIT_OUTPUT_FORMATS],
          description: "summary is the bounded human-readable default. Request envelope for the canonical cross-format contract, json/json-compact for format-specific evidence, CycloneDX 1.7, or SARIF explicitly.",
        },
        scan: { type: "string", enum: [...SCAN_MODES], description: "Bounded scan policy. structure avoids payload integrity work; integrity streams supported payload checks; full requests all supported static analysis." },
        section: { type: "string", description: "Emit only these analysis sections, comma-separated. Use list_sections first. Applies to json formats." },
        list_sections: { type: "boolean", description: "List selectable analysis sections instead of auditing." },
        pointer: { type: "string", description: "Emit one RFC 6901 JSON Pointer result, such as /operator_count." },
        target: { type: "string", description: "TFLite target profile id for cost-model binding; it is not host detection." },
        external_data_dir: { type: "string", description: "Local ONNX external_data or ExecuTorch .ptd sidecar directory under the allowed roots." },
        expected_sha256: { type: "string", pattern: "^[A-Fa-f0-9]{64}$", description: "Independent expected digest for a local or remote artifact." },
        cache_dir: { type: "string", description: "Content-addressed cache directory under the allowed roots." },
        offline: { type: "boolean", description: "Refuse network access and require a verified cache receipt." },
        max_download_gib: { type: "integer", minimum: 1, maximum: 1024, description: "Upper bound for one remote download in GiB." },
        gate: { type: "string", enum: ["defects"], description: "Return a policy-blocked status when an artifact_defect is present. Nothing is gated by default." },
        policy: { type: "string", enum: ["engineering", "regulatory"], description: "Built-in gate profile. Engineering blocks artifact defects; regulatory also requires evidence gaps to be resolved but does not determine legal compliance." },
      },
      required: ["path"],
      additionalProperties: false,
    },
    annotations: CACHE_WRITING_ANNOTATIONS,
  },
  {
    name: "deepbom_diff",
    title: "Compare two model artifacts",
    description: "Compare two artifacts of the same supported format and return deepbom.semantic_artifact_diff.v1. TFLite results also retain the target-bound deployment delta. The comparison does not prove lineage, runtime behaviour, task quality, or regulatory significance.",
    inputSchema: {
      type: "object",
      properties: {
        baseline: { type: "string", description: "Local or immutable remote baseline artifact or package." },
        candidate: { type: "string", description: "Local or immutable remote candidate artifact or package of the same format." },
        target: { type: "string", description: "Optional TFLite target profile id for the embedded pinned cost model." },
        expected_sha256: { type: "string", pattern: "^[A-Fa-f0-9]{64}$", description: "Expected digest for the baseline source. Use independently pinned remote URLs for both artifacts when possible." },
        cache_dir: { type: "string", description: "Content-addressed cache directory under the allowed roots." },
        offline: { type: "boolean", description: "Refuse network access and require verified cache receipts." },
        max_download_gib: { type: "integer", minimum: 1, maximum: 1024 },
      },
      required: ["baseline", "candidate"],
      additionalProperties: false,
    },
    outputSchema: { type: "object", required: ["schema"], properties: { schema: { const: "deepbom.semantic_artifact_diff.v1" } }, additionalProperties: true },
    annotations: CACHE_WRITING_ANNOTATIONS,
  },
  {
    name: "deepbom_explain_rule",
    title: "Explain a DEEPBOM rule",
    description: "Return the machine-readable meaning, evidence boundary, source binding, and verification status of one DEEPBOM rule or analysis term. Use deepbom_capabilities or the CLI rule index to discover identifiers instead of guessing their meaning.",
    inputSchema: {
      type: "object",
      properties: { rule: { type: "string", minLength: 1, description: "Rule or analysis-term identifier accepted by deepbom explain-rule." } },
      required: ["rule"],
      additionalProperties: false,
    },
    outputSchema: { type: "object", required: ["schema"], properties: { schema: { const: "deepbom.rule_explanation.v1" } }, additionalProperties: true },
    annotations: READ_ONLY_ANNOTATIONS,
  },
]);

export async function runMcpServer({ cliEntry, version }) {
  const state = {
    entry: resolveCliEntry(cliEntry),
    version,
    initialized: false,
    pending: new Map(),
    roots: resolveAllowedRoots(),
    scheduler: new ToolScheduler(MAX_CONCURRENT_TOOLS, MAX_QUEUED_TOOLS),
  };
  process.stdin.setEncoding("utf8");
  let buffer = "";
  let discardingOversizeFrame = false;
  const inflight = new Set();

  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (discardingOversizeFrame) {
        discardingOversizeFrame = false;
      } else if (line) {
        if (Buffer.byteLength(line, "utf8") > MAX_FRAME_BYTES) {
          send(rpcFailure(null, -32600, `JSON-RPC frame exceeds the ${MAX_FRAME_BYTES}-byte MCP input limit.`));
        } else {
          const task = handleLine(line, state).finally(() => inflight.delete(task));
          inflight.add(task);
        }
      }
      newline = buffer.indexOf("\n");
    }
    if (Buffer.byteLength(buffer, "utf8") > MAX_FRAME_BYTES) {
      buffer = "";
      discardingOversizeFrame = true;
      send(rpcFailure(null, -32600, `JSON-RPC frame exceeds the ${MAX_FRAME_BYTES}-byte MCP input limit.`));
    }
  });

  await new Promise((resolve) => {
    process.stdin.on("end", resolve);
    process.stdin.on("close", resolve);
  });
  await Promise.allSettled([...inflight]);
}

async function handleLine(line, state) {
  let message;
  try {
    message = JSON.parse(line);
  } catch {
    send(rpcFailure(null, -32700, "Parse error"));
    return;
  }
  if (!message || typeof message !== "object" || Array.isArray(message) || message.jsonrpc !== "2.0" || typeof message.method !== "string") {
    send(rpcFailure(validResponseId(message?.id), -32600, "Invalid JSON-RPC request"));
    return;
  }
  if (message.id === undefined) {
    handleNotification(message, state);
    return;
  }
  const id = validRequestId(message.id);
  if (id === undefined) {
    send(rpcFailure(null, -32600, "A JSON-RPC request id must be a string or number."));
    return;
  }

  const key = requestKey(id);
  if (state.pending.has(key)) {
    send(rpcFailure(id, -32600, "A request with this id is already in progress."));
    return;
  }
  const controller = new AbortController();
  const record = { id, controller, method: message.method };
  state.pending.set(key, record);
  try {
    const result = await dispatch(message, state, controller.signal);
    if (!controller.signal.aborted && result !== undefined) send({ jsonrpc: "2.0", id, result });
  } catch (error) {
    if (!controller.signal.aborted && error?.name !== "AbortError") {
      send(rpcFailure(id, Number.isInteger(error?.code) ? error.code : -32603, String(error?.message || error)));
    }
  } finally {
    state.pending.delete(key);
  }
}

function handleNotification(message, state) {
  if (message.method === "notifications/initialized") return;
  if (message.method !== "notifications/cancelled") return;
  const id = validRequestId(message.params?.requestId);
  if (id === undefined) return;
  state.pending.get(requestKey(id))?.controller.abort(cancellationError("The MCP client cancelled this request."));
}

async function dispatch(message, state, signal) {
  if (message.method === "initialize") {
    if (state.initialized) throw rpcError(-32600, "The MCP session is already initialized.");
    state.initialized = true;
    const requested = message.params?.protocolVersion;
    return {
      protocolVersion: PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0],
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "deepbom", version: state.version },
      instructions: "Every tool runs the local DEEPBOM CLI; artifact bytes are never uploaded. Local paths are limited to DEEPBOM_MCP_ALLOWED_ROOTS (the launch directory by default). Call deepbom_capabilities before assuming an option exists. The bounded audit default is a human-readable summary; request detailed formats explicitly. Keep artifact_defect, caution, and evidence_gap distinct: an evidence_gap is not a defect. Never report executed accelerator placement, latency, energy, accuracy, or device fit from static output.",
    };
  }
  if (!state.initialized) throw rpcError(-32002, "Initialize the MCP session before calling this method.");
  if (message.method === "ping") return {};
  if (message.method === "tools/list") return { tools: TOOLS };
  if (message.method === "tools/call") {
    const name = message.params?.name;
    if (!TOOLS.some((tool) => tool.name === name)) throw rpcError(-32602, `Unknown tool: ${String(name)}`);
    return state.scheduler.run(signal, () => callTool(message.params || {}, state, signal));
  }
  throw rpcError(-32601, `Unsupported method: ${message.method}`);
}

async function callTool(params, state, signal) {
  const name = params.name;
  const args = params.arguments ?? {};
  let argv;
  try {
    validateToolArguments(name, args);
    argv = commandArguments(name, args, state.roots);
  } catch (error) {
    return toolError(String(error.message));
  }

  const progressToken = validProgressToken(params?._meta?.progressToken);
  if (progressToken !== undefined) sendProgress(progressToken, 0, 1, `Running ${name}`);
  const run = await runCli(state.entry, argv, signal);
  if (signal.aborted || run.cancelled) throw cancellationError("The MCP request was cancelled.");
  if (progressToken !== undefined) sendProgress(progressToken, 1, 1, `${name} complete`);

  if (run.limitExceeded) {
    return toolError(`DEEPBOM output exceeded the ${formatBytes(MAX_RESPONSE_BYTES)} MCP response limit. Retry with output_format summary, envelope, section, or pointer, or raise DEEPBOM_MCP_MAX_RESPONSE_BYTES deliberately.`);
  }
  if (run.stderrLimitExceeded) {
    return toolError(`DEEPBOM diagnostic output exceeded the ${formatBytes(MAX_STDERR_BYTES)} MCP stderr limit and the child process was terminated.`);
  }
  if (run.timedOut) {
    return toolError(`DEEPBOM exceeded the ${TOOL_TIMEOUT_MS} ms MCP tool timeout. Retry with scan structure/integrity, narrow the result with section or pointer, or raise DEEPBOM_MCP_TOOL_TIMEOUT_MS deliberately.`);
  }
  if (run.code !== 0 && run.code !== 2) {
    return toolError(run.stderr.trim() || run.stdout.trim() || `deepbom exited ${run.code}`);
  }

  const text = run.stdout.trim();
  const result = { content: [{ type: "text", text }] };
  const structured = parseStructuredObject(text);
  if (structured) result.structuredContent = structured;
  if (run.code === 2) {
    result.content.push({ type: "text", text: "The requested gate policy blocked this run (exit 2). The first content block remains the complete, unmodified result; treat this as a policy outcome, not an analysis failure." });
    result._meta = { deepbom: { exit_code: 2, policy_status: "blocked", analysis_completed: true } };
  }
  if (Buffer.byteLength(JSON.stringify(result), "utf8") > MAX_RESPONSE_BYTES) {
    return toolError(`The serialized MCP result exceeded the ${formatBytes(MAX_RESPONSE_BYTES)} response limit after compatibility content was added. Retry with output_format summary, envelope, section, or pointer.`);
  }
  return result;
}

function commandArguments(name, args, roots) {
  if (name === "deepbom_capabilities") return ["capabilities", "--json"];
  if (name === "deepbom_explain_rule") return ["explain-rule", requiredRuleId(args.rule), "--compact"];
  if (name === "deepbom_audit") {
    const argv = ["audit", requiredArtifactSource(args.path, "path", roots, args.expected_sha256)];
    if (args.list_sections) {
      argv.push("--list-sections");
    } else if (Object.hasOwn(args, "pointer")) {
      argv.push("--pointer", String(args.pointer));
    } else {
      const format = args.output_format || AUDIT_DEFAULT_OUTPUT_FORMAT;
      argv.push("--output-format", format);
      if (args.section) argv.push("--section", String(args.section));
    }
    if (args.scan) argv.push("--scan", args.scan);
    if (args.target) argv.push("--target", String(args.target));
    if (args.external_data_dir) argv.push("--external-data-dir", requiredLocalPath(args.external_data_dir, "external_data_dir", roots));
    appendRemoteControls(argv, args, roots);
    if (args.gate === "defects") argv.push("--gate", "defects");
    if (args.policy) argv.push("--policy", String(args.policy));
    return argv;
  }
  if (name === "deepbom_diff") {
    const argv = [
      "diff",
      requiredArtifactSource(args.baseline, "baseline", roots, args.expected_sha256),
      requiredArtifactSource(args.candidate, "candidate", roots),
      "--json",
    ];
    if (args.target) argv.push("--target", String(args.target));
    appendRemoteControls(argv, args, roots);
    return argv;
  }
  throw rpcError(-32602, `Unknown tool: ${name}`);
}

function appendRemoteControls(argv, args, roots) {
  if (args.expected_sha256) argv.push("--expected-sha256", args.expected_sha256.toLowerCase());
  if (args.cache_dir) argv.push("--cache-dir", requiredLocalPath(args.cache_dir, "cache_dir", roots));
  if (args.offline) argv.push("--offline");
  if (args.max_download_gib) argv.push("--max-download-gib", String(args.max_download_gib));
}

function validateToolArguments(name, args) {
  if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Tool arguments must be a JSON object.");
  const allowed = {
    deepbom_capabilities: [],
    deepbom_audit: ["path", "output_format", "scan", "section", "list_sections", "pointer", "target", "external_data_dir", "expected_sha256", "cache_dir", "offline", "max_download_gib", "gate", "policy"],
    deepbom_diff: ["baseline", "candidate", "target", "expected_sha256", "cache_dir", "offline", "max_download_gib"],
    deepbom_explain_rule: ["rule"],
  }[name];
  if (!allowed) throw rpcError(-32602, `Unknown tool: ${name}`);
  const extra = Object.keys(args).filter((key) => !allowed.includes(key));
  if (extra.length) throw new Error(`Undeclared tool argument${extra.length === 1 ? "" : "s"}: ${extra.sort().join(", ")}.`);

  const booleanFields = ["list_sections", "offline"];
  for (const key of allowed.filter((key) => !booleanFields.includes(key) && key !== "max_download_gib")) {
    if (Object.hasOwn(args, key) && typeof args[key] !== "string") throw new Error(`The ${key} argument must be a string.`);
  }
  for (const key of booleanFields) {
    if (Object.hasOwn(args, key) && typeof args[key] !== "boolean") throw new Error(`The ${key} argument must be a boolean.`);
  }
  if (Object.hasOwn(args, "max_download_gib") && (!Number.isInteger(args.max_download_gib) || args.max_download_gib < 1 || args.max_download_gib > 1024)) {
    throw new Error("The max_download_gib argument must be an integer from 1 through 1024.");
  }
  if (Object.hasOwn(args, "expected_sha256") && !/^[a-f0-9]{64}$/i.test(args.expected_sha256)) {
    throw new Error("The expected_sha256 argument must contain exactly 64 hexadecimal characters.");
  }
  if (name === "deepbom_capabilities") return;
  if (name === "deepbom_explain_rule") {
    requiredRuleId(args.rule);
    return;
  }
  if (name === "deepbom_diff") return;

  const selectors = [
    args.list_sections ? "list_sections" : null,
    Object.hasOwn(args, "pointer") ? "pointer" : null,
    Object.hasOwn(args, "section") ? "section" : null,
  ].filter(Boolean);
  if (selectors.length > 1) throw new Error(`Audit selectors are mutually exclusive: ${selectors.join(", ")}.`);
  if ((args.list_sections || Object.hasOwn(args, "pointer")) && Object.hasOwn(args, "output_format")) {
    throw new Error("output_format cannot be combined with list_sections or pointer.");
  }
  if (Object.hasOwn(args, "section") && !["json", "json-compact"].includes(args.output_format || "")) {
    throw new Error("section requires output_format json or json-compact.");
  }
  if (Object.hasOwn(args, "output_format") && !AUDIT_OUTPUT_FORMATS.includes(args.output_format)) {
    throw new Error(`Unsupported output_format: ${args.output_format}. Use one of ${AUDIT_OUTPUT_FORMATS.join(", ")}.`);
  }
  if (Object.hasOwn(args, "scan") && !SCAN_MODES.includes(args.scan)) throw new Error(`Unsupported scan: ${args.scan}.`);
  if (Object.hasOwn(args, "gate") && args.gate !== "defects") throw new Error("The gate argument must be defects.");
}

function requiredText(value, field) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`The ${field} argument must be a non-empty string.`);
  return value.trim();
}

function requiredRuleId(value) {
  const rule = requiredText(value, "rule");
  if (!/^[a-z0-9][a-z0-9._:-]*$/i.test(rule)) {
    throw new Error("The rule argument must be one rule identifier, not a CLI option or free-form command.");
  }
  return rule;
}

function requiredArtifactSource(value, field, roots, expectedSha256 = "") {
  const source = requiredText(value, field);
  if (/^https:\/\//i.test(source)) {
    if (!/#sha256=[a-f0-9]{64}(?:$|&)/i.test(source) && !expectedSha256) {
      throw new Error(`The ${field} HTTPS source must include #sha256=<64-hex> or expected_sha256.`);
    }
    return source;
  }
  if (/^(?:hf|gs):\/\//i.test(source)) return source;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(source)) throw new Error(`The ${field} source scheme is not supported.`);
  return requiredLocalPath(source, field, roots);
}

function requiredLocalPath(value, field, roots) {
  const localPath = requiredText(value, field);
  const canonical = canonicalPath(localPath);
  if (!roots.some((root) => pathContains(root, canonical))) {
    throw new Error(`The ${field} path is outside DEEPBOM_MCP_ALLOWED_ROOTS. Allowed roots: ${roots.join(path.delimiter)}.`);
  }
  return canonical;
}

function resolveAllowedRoots() {
  const declared = process.env.DEEPBOM_MCP_ALLOWED_ROOTS;
  const values = declared ? declared.split(path.delimiter).filter((value) => value.trim()) : [process.cwd()];
  if (!values.length) throw new Error("DEEPBOM_MCP_ALLOWED_ROOTS does not contain a path.");
  return values.map((value) => {
    const absolute = path.resolve(value.trim());
    if (!existsSync(absolute)) throw new Error(`MCP allowed root does not exist: ${absolute}`);
    return canonicalPath(absolute);
  });
}

function canonicalPath(value) {
  const absolute = path.resolve(value);
  let cursor = absolute;
  const suffix = [];
  while (!existsSync(cursor)) {
    const parent = path.dirname(cursor);
    if (parent === cursor) break;
    suffix.unshift(path.basename(cursor));
    cursor = parent;
  }
  const resolved = existsSync(cursor) ? realpathSync.native(cursor) : cursor;
  return path.resolve(resolved, ...suffix);
}

function pathContains(root, candidate) {
  const normalize = (value) => process.platform === "win32" ? value.toLowerCase() : value;
  const normalizedRoot = normalize(path.resolve(root));
  const normalizedCandidate = normalize(path.resolve(candidate));
  return normalizedCandidate === normalizedRoot || normalizedCandidate.startsWith(`${normalizedRoot}${path.sep}`);
}

function runCli(entry, argv, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(cancellationError("The MCP request was cancelled before execution."));
    const child = spawn(process.execPath, [entry, ...argv], {
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      env: { ...process.env, DEEPBOM_PROGRESS: "0" },
    });
    let stdout = "";
    let stderr = "";
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let limitExceeded = false;
    let stderrLimitExceeded = false;
    let timedOut = false;
    let cancelled = false;
    let settled = false;

    const terminate = () => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM");
      const force = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      }, 1000);
      force.unref();
    };
    const onAbort = () => {
      cancelled = true;
      terminate();
    };
    signal.addEventListener("abort", onAbort, { once: true });
    const timeout = setTimeout(() => {
      timedOut = true;
      terminate();
    }, TOOL_TIMEOUT_MS);
    timeout.unref();

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdoutBytes += Buffer.byteLength(chunk, "utf8");
      if (stdoutBytes > MAX_RESPONSE_BYTES) {
        limitExceeded = true;
        terminate();
      } else if (!limitExceeded) stdout += chunk;
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderrBytes += Buffer.byteLength(chunk, "utf8");
      if (stderrBytes > MAX_STDERR_BYTES) {
        stderrLimitExceeded = true;
        terminate();
      } else if (!stderrLimitExceeded) stderr += chunk;
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal.removeEventListener("abort", onAbort);
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal.removeEventListener("abort", onAbort);
      resolve({ code: Number.isInteger(code) ? code : 1, stdout, stderr, limitExceeded, stderrLimitExceeded, timedOut, cancelled });
    });
  });
}

class ToolScheduler {
  constructor(maxConcurrent, maxQueued) {
    this.maxConcurrent = maxConcurrent;
    this.maxQueued = maxQueued;
    this.active = 0;
    this.queue = [];
  }

  run(signal, task) {
    if (signal.aborted) return Promise.reject(cancellationError("The MCP request was cancelled."));
    if (this.active < this.maxConcurrent) return this.start(task);
    if (this.queue.length >= this.maxQueued) return Promise.reject(rpcError(-32001, `The MCP tool queue is full (${this.maxQueued}); retry after an active analysis completes.`));
    return new Promise((resolve, reject) => {
      const item = { signal, task, resolve, reject };
      const onAbort = () => {
        const index = this.queue.indexOf(item);
        if (index !== -1) this.queue.splice(index, 1);
        reject(cancellationError("The queued MCP request was cancelled."));
      };
      item.onAbort = onAbort;
      signal.addEventListener("abort", onAbort, { once: true });
      this.queue.push(item);
    });
  }

  start(task) {
    this.active += 1;
    return Promise.resolve().then(task).finally(() => {
      this.active -= 1;
      this.drain();
    });
  }

  drain() {
    while (this.active < this.maxConcurrent && this.queue.length) {
      const item = this.queue.shift();
      item.signal.removeEventListener("abort", item.onAbort);
      if (item.signal.aborted) {
        item.reject(cancellationError("The queued MCP request was cancelled."));
        continue;
      }
      this.start(item.task).then(item.resolve, item.reject);
    }
  }
}

function resolveCliEntry(cliEntry) {
  const candidate = cliEntry || process.argv[1];
  if (!candidate) throw new Error("The MCP server could not locate its own CLI entry point.");
  return path.resolve(candidate);
}

function parseStructuredObject(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function toolError(message) {
  return { content: [{ type: "text", text: message }], isError: true };
}

function rpcError(code, message) {
  return Object.assign(new Error(message), { code });
}

function cancellationError(message) {
  return Object.assign(new Error(message), { name: "AbortError" });
}

function validRequestId(value) {
  return typeof value === "string" || (typeof value === "number" && Number.isFinite(value)) ? value : undefined;
}

function validResponseId(value) {
  return validRequestId(value) ?? null;
}

function validProgressToken(value) {
  return validRequestId(value);
}

function requestKey(id) {
  return `${typeof id}:${String(id)}`;
}

function rpcFailure(id, code, message) {
  return { jsonrpc: "2.0", id, error: { code, message } };
}

function sendProgress(progressToken, progress, total, message) {
  send({ jsonrpc: "2.0", method: "notifications/progress", params: { progressToken, progress, total, message } });
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function boundedEnvironmentInteger(name, fallback, minimum, maximum) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} through ${maximum}.`);
  }
  return value;
}

function formatBytes(value) {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KiB`;
  return `${Math.round((value / (1024 * 1024)) * 10) / 10} MiB`;
}
