import { readFile, stat, unlink } from "node:fs/promises";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import {
  buildNativeEvidence,
  analyzeNativeTensor,
  compareNativeEvidence,
  buildTrainingIr,
  validateTrainingIr,
  validateNativeDocument,
  validateNativeBundle,
  validateTrainingBundle,
  selectTrainingResult,
} from "../web/lib/native-evidence.js";
import { parseStrictJson } from "../web/lib/metadata-model-adapters.js";
import { buildNativeOptimizationReport } from "../web/lib/native-optimization-report.js";
import { buildNativeOptimizationDiff } from "../web/lib/native-optimization-diff.js";
import { openOptimizationReport } from "../web/lib/optimization-report-access.js";
import { writeOutputAtomically } from "./deepbom-automation.mjs";
import { optimizationFacts } from "../web/lib/analysis-facts.js";
import { inspectNative } from "../web/lib/native-inspection.js";
import { buildTransformationCorrespondence, buildMappedModelComparison } from "../web/lib/transformation-correspondence.js";
import { validateTrainingAnalysis } from "../web/lib/training-analysis-bindings.js";
import { numericalWasmIdentity } from "../web/lib/numerical-wasm.js";
import { evaluateBuiltinMetric } from "../web/lib/builtin-metrics.js";
import { ANALYZER_BUNDLE_CONTENT_SHA256 } from "../web/lib/build-metadata.js";

import { canonicalEvidenceDigest } from "../web/lib/evidence-identity.js";

const nativeEngineIdentity = () => ({...numericalWasmIdentity(), analyzer_content_sha256: ANALYZER_BUNDLE_CONTENT_SHA256});

export async function runNativeEvidence(argv) {
  if (argv.length === 2 && argv[0] === "--serve") return serveNativeEvidence(resolve(argv[1]));
  const options = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    if (
      !["--request", "--output"].includes(key) ||
      options[key] ||
      !argv[i + 1]
    )
      throw new Error(
        "evidence-native requires --request <json> [--output <json>]",
      );
    options[key] = argv[i + 1];
  }
  if (!options["--request"])
    throw new Error("Missing native evidence request.");
  if ((await stat(options["--request"])).size > 128 * 1024 * 1024)
    throw new Error("Native request exceeds 128 MiB.");
  const request = parseStrictJson(await readFile(options["--request"], "utf8"));
  const result = await dispatchNativeEvidence(request), json = JSON.stringify(result) + "\n";
  if (options["--output"])
    await writeOutputAtomically(options["--output"], json);
  else process.stdout.write(json);
}

export async function dispatchNativeEvidence(request) {
  if (!request || Object.keys(request).sort().join(",") !== "operation,payload")
    throw new Error("Invalid native bridge request.");
  const handlers = {
    content_hash: (value) => ({sha256: canonicalEvidenceDigest(value)}),
    numerical_identity: nativeEngineIdentity,
    metric: evaluateBuiltinMetric,
    validate_training_analysis: validateTrainingAnalysis,
    inspect_native: inspectNative,
    optimization_report: buildNativeOptimizationReport,
    optimization_diff: buildNativeOptimizationDiff,
    optimization_facts: optimizationFacts,
    correspondence: buildTransformationCorrespondence,
    mapped_comparison: buildMappedModelComparison,
    optimization_import: ({text, expected_sha256}) => openOptimizationReport(text, expected_sha256),
    select_result: selectTrainingResult,
    snapshot: buildNativeEvidence,
    training: buildTrainingIr,
    validate_training: validateTrainingIr,
    validate_native: validateNativeDocument,
    validate_bundle: validateNativeBundle,
    validate_run: validateTrainingBundle,
    tensor: (p) => analyzeNativeTensor(p.tensor, p.options),
    tensor_batch: async (p) => {
      if (
        Object.keys(p).sort().join(",") !== "options,tensors" ||
        !Array.isArray(p.tensors)
      )
        throw new Error("Invalid tensor batch");
      const rows = [];
      for (const tensor of p.tensors)
        rows.push(await analyzeNativeTensor(tensor, p.options));
      return { tensors: rows };
    },
    compare: (p) => compareNativeEvidence(p.baseline, p.candidate),
  };
  if (!Object.hasOwn(handlers, request.operation))
    throw new Error("Unknown native evidence operation.");
  return handlers[request.operation](request.payload);
}

// Private single-caller transport. The same dispatch owns both one-shot and
// session requests; the host controls resource limits, cancellation and lifetime.
async function serveNativeEvidence(directory) {
  const parent = process.ppid;
  await writeOutputAtomically(join(directory, "ready.json"), JSON.stringify({
    schema: "deepbom.native_session.v1", ...nativeEngineIdentity(),
  }));
  let sequence = 0;
  while (process.ppid === parent) {
    const input = join(directory, "request.json");
    let info;
    try { info = await stat(input); }
    catch (error) { if (error.code !== "ENOENT") throw error; await delay(2); continue; }
    if (info.size > 128 * 1024 * 1024) throw new Error("Session request exceeds its budget.");
    const envelope = parseStrictJson(await readFile(input, "utf8"));
    if (Object.keys(envelope).sort().join(",") !== "request,sequence" || envelope.sequence !== sequence)
      throw new Error("Invalid native session sequence.");
    await unlink(input);
    let reply;
    try { reply = {sequence, result: await dispatchNativeEvidence(envelope.request)}; }
    catch (error) { reply = {sequence, error: {code: "native_analysis_failed", message: String(error.message).slice(0, 4096)}}; }
    const text = JSON.stringify(reply);
    if (Buffer.byteLength(text) > 128 * 1024 * 1024) throw new Error("Session response exceeds its budget.");
    await writeOutputAtomically(join(directory, "result.json"), text);
    sequence += 1;
  }
}
