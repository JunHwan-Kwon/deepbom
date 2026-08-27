import { Tensor, loadAndCompile } from "@litertjs/core";
import { sha256TypedArrayListHex } from "./hash.js";
import { cloneTypedArray } from "./format.js";
import {
  createBenchmarkInputSpec,
  createBenchmarkOutputContract,
  deleteTensors,
  measureBrowserBenchmarkPhases,
} from "./runtime.js";

export async function benchmarkTfliteModel({
  modelBytes,
  analysis,
  backend,
  warmup,
  runs,
  preparedInput = null,
  ensureRuntime = async () => {},
} = {}, dependencies = {}) {
  const compile = dependencies.loadAndCompile || loadAndCompile;
  const makeTensor = dependencies.tensorFromTypedArray || ((data, shape) => Tensor.fromTypedArray(data, shape));
  await ensureRuntime(backend);
  const compileStarted = performance.now();
  let model = null;
  const inputs = [];
  const inputContracts = [];

  try {
    model = await compile(modelBytes, { accelerator: backend });
    const compileMs = performance.now() - compileStarted;
    for (const [index, details] of model.getInputDetails().entries()) {
      const artifact = analysis?.inputs?.find((input) => input.name === details.name) || analysis?.inputs?.[index] || null;
      if (!artifact) throw new Error(`Runtime input ${details.name || index} has no bound artifact input contract`);
      const spec = createBenchmarkInputSpec(details, artifact, preparedInput, index);
      inputs.push(makeTensor(spec.data, spec.shape));
      inputContracts.push({
        input_index: index,
        input_name: details.name || artifact.name || `input_${index}`,
        artifact_dtype: artifact.dtype || "UNKNOWN",
        runtime_dtype: details.dtype,
        declared_shape: Array.from(artifact.shape || []),
        artifact_shape_signature: artifact.shape_signature?.length === artifact.shape?.length ? Array.from(artifact.shape_signature) : null,
        runtime_declared_shape: Array.from(details.shape || []),
        executed_shape: [...spec.shape],
        element_count: spec.data.length,
        basis: String(spec.basis || "synthetic tensor").replaceAll(" ", "_"),
        synthetic_fill_value: spec.basis === "synthetic tensor" && spec.data.length ? Number(spec.data[0]) : null,
      });
    }
    const phases = await measureBrowserBenchmarkPhases({
      execute: () => model.run(inputs),
      dispose: deleteTensors,
      warmupRuns: warmup,
      measuredRuns: runs,
      yieldEvery: 10,
      yieldControl: () => new Promise((resolve) => setTimeout(resolve, 0)),
      observeFinal: (outputs) => observeTfliteOutputs(outputs, model.getOutputDetails(), analysis?.outputs || []),
    });
    return {
      compileMs,
      firstRunMs: phases.firstRunMs,
      timings: phases.timings,
      outputCount: phases.finalObservation.output_count,
      outputDigest: phases.finalObservation.output_digest,
      outputContracts: phases.finalObservation.output_contracts,
      stats: phases.stats,
      steadyStats: phases.steadyStats,
      timingMethod: phases.timingMethod,
      statisticsMethod: phases.statisticsMethod,
      noiseMethod: phases.noiseMethod,
      noiseDiagnostics: phases.noiseDiagnostics,
      phaseCounts: phases.phaseCounts,
      inputContracts,
    };
  } finally {
    deleteTensors(inputs);
    model?.delete?.();
  }
}

async function observeTfliteOutputs(outputs, detailsRows, artifactOutputs) {
  const tensors = Array.isArray(outputs) ? outputs : detailsRows.map((details) => outputs?.[details.name]);
  if (tensors.length !== detailsRows.length) throw new Error(`Runtime returned ${tensors.length} outputs for ${detailsRows.length} declared outputs`);
  const arrays = [];
  const contracts = [];
  for (const [index, tensor] of tensors.entries()) {
    const details = detailsRows[index];
    if (!tensor) throw new Error(`Runtime did not return declared output ${details.name || index}`);
    const artifact = artifactOutputs.find((item) => item.name === details.name) || artifactOutputs[index] || null;
    if (!artifact) throw new Error(`Runtime output ${details.name || index} has no bound artifact output contract`);
    const data = cloneTypedArray(await tensor.data());
    arrays.push(data);
    contracts.push(createBenchmarkOutputContract({ index, name: details.name, artifactDtype: artifact.dtype, runtimeDtype: tensor.type?.dtype || details.dtype, declaredShape: artifact.shape, artifactShapeSignature: artifact.shape_signature?.length === artifact.shape?.length ? artifact.shape_signature : null, runtimeDeclaredShape: details.shape, executedShape: tensor.type?.layout?.dimensions, data }));
  }
  return { output_count: arrays.length, output_digest: await sha256TypedArrayListHex(arrays), output_contracts: contracts };
}
