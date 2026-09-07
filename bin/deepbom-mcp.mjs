import { execFile } from "node:child_process";
import path from "node:path";
import process from "node:process";

// Model Context Protocol server over stdio. It exists so an assistant can reach
// the same local analysis the CLI performs without a hosted endpoint: every
// request is answered by running this CLI on this machine, and artifact bytes
// never leave it.
//
// The analysis runs in a child process rather than in this one on purpose. The
// stdio transport owns this process's stdout for JSON-RPC frames, and the CLI
// writes its documents to stdout; sharing the stream would interleave an
// evidence document into the protocol.

const PROTOCOL_VERSIONS = Object.freeze(["2025-06-18", "2025-03-26", "2024-11-05"]);
const MAX_DOCUMENT_BYTES = 256 * 1024 * 1024;
const AUDIT_OUTPUT_FORMATS = Object.freeze(["envelope", "json", "json-compact", "cyclonedx", "sarif"]);

const TOOLS = Object.freeze([
  {
    name: "deepbom_capabilities",
    description: "Return deepbom.cli_capabilities.v1: the commands, supported artifact formats, output contracts, scan policies, target profiles, and exit codes of this installation. Call this before the other tools instead of assuming an option exists.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
  },
  {
    name: "deepbom_audit",
    description: "Audit one local model artifact (.tflite, .onnx, .gguf, .safetensors, .mlmodel, .pte, .ptd, or a .mlpackage / sharded-repository directory) and return its evidence document. Findings carry three distinct kinds: artifact_defect is a problem in the artifact, caution is for reviewer attention, and evidence_gap is a claim the artifact cannot settle on its own. An evidence_gap is not a defect. The result never establishes executed accelerator assignment, latency, energy, accuracy, or device fit.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Path to the artifact file or package directory on this machine." },
        output_format: {
          type: "string",
          enum: [...AUDIT_OUTPUT_FORMATS],
          description: "envelope is the canonical cross-format contract (default). json is the richer format-specific analysis. cyclonedx emits a CycloneDX 1.7 ML-BOM, sarif a SARIF 2.1.0 findings document.",
        },
        section: { type: "string", description: "Emit only these analysis sections, comma-separated. Use list_sections first to discover the names. Applies to the json formats." },
        list_sections: { type: "boolean", description: "List the selectable analysis section names for this artifact instead of auditing it." },
        pointer: { type: "string", description: "Emit a single RFC 6901 JSON Pointer result from the analysis, such as /operator_count." },
        target: { type: "string", description: "TFLite target profile id for cost-model binding. Results bound to it are assumptions about that profile, not host measurements." },
        external_data_dir: { type: "string", description: "Directory holding ONNX external_data or ExecuTorch .ptd sidecars." },
        gate: { type: "string", enum: ["defects"], description: "Exit 2 when an artifact_defect finding is present. Nothing is gated by default." },
      },
      required: ["path"],
      additionalProperties: false,
    },
  },
  {
    name: "deepbom_diff",
    description: "Compare two local artifacts and return deepbom.deployment_delta.v1.1: what changed between a baseline and a candidate build. Static estimates are bound to a pinned target profile and are not measured behaviour.",
    inputSchema: {
      type: "object",
      properties: {
        baseline: { type: "string", description: "Path to the baseline artifact." },
        candidate: { type: "string", description: "Path to the candidate artifact." },
        target: { type: "string", description: "TFLite target profile id for the pinned cost model." },
      },
      required: ["baseline", "candidate"],
      additionalProperties: false,
    },
  },
]);

export async function runMcpServer({ cliEntry, version }) {
  const entry = resolveCliEntry(cliEntry);
  process.stdin.setEncoding("utf8");
  let buffer = "";
  let queue = Promise.resolve();

  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      // Serialize handling so responses keep the order the client sent, and one
      // slow audit cannot interleave its frame into another.
      if (line) queue = queue.then(() => handleLine(line, entry, version));
      newline = buffer.indexOf("\n");
    }
  });

  await new Promise((resolve) => {
    process.stdin.on("end", resolve);
    process.stdin.on("close", resolve);
  });
  await queue;
}

async function handleLine(line, entry, version) {
  let message;
  try {
    message = JSON.parse(line);
  } catch {
    return send({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } });
  }
  // A notification carries no id and must never be answered.
  if (message.id === undefined || message.id === null) return;
  try {
    const result = await dispatch(message, entry, version);
    if (result !== undefined) send({ jsonrpc: "2.0", id: message.id, result });
  } catch (error) {
    send({ jsonrpc: "2.0", id: message.id, error: { code: Number.isInteger(error?.code) ? error.code : -32603, message: String(error?.message || error) } });
  }
}

async function dispatch(message, entry, version) {
  if (message.method === "initialize") {
    const requested = message.params?.protocolVersion;
    return {
      protocolVersion: PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0],
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: "deepbom", version },
      instructions: "Every tool runs the local DEEPBOM CLI on this machine; artifact bytes are never uploaded. Call deepbom_capabilities before assuming an option exists. Keep artifact_defect, caution, and evidence_gap findings distinct: an evidence_gap means the artifact cannot settle the claim, not that it is defective. Never report executed accelerator placement, latency, energy, accuracy, or device fit from this output.",
    };
  }
  if (message.method === "ping") return {};
  if (message.method === "tools/list") return { tools: TOOLS };
  if (message.method === "tools/call") return callTool(message.params || {}, entry);
  throw Object.assign(new Error(`Unsupported method: ${message.method}`), { code: -32601 });
}

async function callTool(params, entry) {
  const name = params.name;
  const args = params.arguments ?? {};
  let argv;
  try {
    validateToolArguments(name, args);
    argv = commandArguments(name, args);
  } catch (error) {
    return { content: [{ type: "text", text: String(error.message) }], isError: true };
  }
  const run = await runCli(entry, argv);
  if (run.code === 0 || run.code === 2) {
    const gate = run.code === 2
      ? "\n\nThe finding gate blocked this run (exit 2). The document above is complete; report it as a policy result, not a failure to analyse."
      : "";
    return { content: [{ type: "text", text: `${run.stdout.trim()}${gate}` }] };
  }
  return {
    content: [{ type: "text", text: run.stderr.trim() || run.stdout.trim() || `deepbom exited ${run.code}` }],
    isError: true,
  };
}

function commandArguments(name, args) {
  if (name === "deepbom_capabilities") return ["capabilities", "--json"];
  if (name === "deepbom_audit") {
    const argv = ["audit", requiredPath(args.path, "path")];
    if (args.list_sections) {
      argv.push("--list-sections");
    } else if (Object.hasOwn(args, "pointer")) {
      argv.push("--pointer", String(args.pointer));
    } else {
      const format = args.output_format || "envelope";
      if (!AUDIT_OUTPUT_FORMATS.includes(format)) {
        throw new Error(`Unsupported output_format: ${format}. Use one of ${AUDIT_OUTPUT_FORMATS.join(", ")}.`);
      }
      argv.push("--output-format", format);
      if (args.section) argv.push("--section", String(args.section));
    }
    if (args.target) argv.push("--target", String(args.target));
    if (args.external_data_dir) argv.push("--external-data-dir", requiredPath(args.external_data_dir, "external_data_dir"));
    if (args.gate === "defects") argv.push("--gate", "defects");
    return argv;
  }
  if (name === "deepbom_diff") {
    const argv = ["diff", requiredPath(args.baseline, "baseline"), requiredPath(args.candidate, "candidate"), "--json"];
    if (args.target) argv.push("--target", String(args.target));
    return argv;
  }
  throw new Error(`Unknown tool: ${name}`);
}

function validateToolArguments(name, args) {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    throw new Error("Tool arguments must be a JSON object.");
  }
  const allowed = {
    deepbom_capabilities: [],
    deepbom_audit: ["path", "output_format", "section", "list_sections", "pointer", "target", "external_data_dir", "gate"],
    deepbom_diff: ["baseline", "candidate", "target"],
  }[name];
  if (!allowed) throw new Error(`Unknown tool: ${name}`);
  const extra = Object.keys(args).filter((key) => !allowed.includes(key));
  if (extra.length) throw new Error(`Undeclared tool argument${extra.length === 1 ? "" : "s"}: ${extra.sort().join(", ")}.`);

  if (name === "deepbom_capabilities") return;
  for (const key of allowed.filter((key) => !["list_sections"].includes(key))) {
    if (Object.hasOwn(args, key) && typeof args[key] !== "string") {
      throw new Error(`The ${key} argument must be a string.`);
    }
  }
  if (Object.hasOwn(args, "list_sections") && typeof args.list_sections !== "boolean") {
    throw new Error("The list_sections argument must be a boolean.");
  }
  if (name !== "deepbom_audit") return;

  const selectors = [
    args.list_sections ? "list_sections" : null,
    Object.hasOwn(args, "pointer") ? "pointer" : null,
    Object.hasOwn(args, "section") ? "section" : null,
  ].filter(Boolean);
  if (selectors.length > 1) {
    throw new Error(`Audit selectors are mutually exclusive: ${selectors.join(", ")}.`);
  }
  if ((args.list_sections || Object.hasOwn(args, "pointer")) && Object.hasOwn(args, "output_format")) {
    throw new Error("output_format cannot be combined with list_sections or pointer.");
  }
  if (Object.hasOwn(args, "section") && !["json", "json-compact"].includes(args.output_format || "")) {
    throw new Error("section requires output_format json or json-compact.");
  }
  if (Object.hasOwn(args, "output_format") && !AUDIT_OUTPUT_FORMATS.includes(args.output_format)) {
    throw new Error(`Unsupported output_format: ${args.output_format}. Use one of ${AUDIT_OUTPUT_FORMATS.join(", ")}.`);
  }
  if (Object.hasOwn(args, "gate") && args.gate !== "defects") {
    throw new Error("The gate argument must be defects.");
  }
}

function requiredPath(value, field) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`The ${field} argument must be a non-empty path.`);
  return value;
}

function runCli(entry, argv) {
  return new Promise((resolve) => {
    execFile(process.execPath, [entry, ...argv], { maxBuffer: MAX_DOCUMENT_BYTES }, (error, stdout, stderr) => {
      const code = error ? (Number.isInteger(error.code) ? error.code : 1) : 0;
      resolve({ code, stdout: stdout || "", stderr: stderr || "" });
    });
  });
}

function resolveCliEntry(cliEntry) {
  const candidate = cliEntry || process.argv[1];
  if (!candidate) throw new Error("The MCP server could not locate its own CLI entry point.");
  return path.resolve(candidate);
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}
