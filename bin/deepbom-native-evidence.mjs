import { readFile, stat } from "node:fs/promises";
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

export async function runNativeEvidence(argv) {
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
  if (Object.keys(request).sort().join(",") !== "operation,payload")
    throw new Error("Invalid native bridge request.");
  const handlers = {
    optimization_report: buildNativeOptimizationReport,
    optimization_diff: buildNativeOptimizationDiff,
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
  const result = await handlers[request.operation](request.payload),
    json = JSON.stringify(result) + "\n";
  if (options["--output"])
    await writeOutputAtomically(options["--output"], json);
  else process.stdout.write(json);
}
