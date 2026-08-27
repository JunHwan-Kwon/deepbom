import { Tensor, loadAndCompile } from "@litertjs/core";
import { cloneTypedArray, sameShape } from "./format.js";
import { sha256Hex } from "./hash.js";
import {
  buildCandidateReplayInput,
  buildCanonicalWitnessInput,
  buildPreprocessingConsequenceEvidence,
} from "./preprocessing-consequence-core.js";
import { validatePreprocessingConsequenceCapture } from "./preprocessing-consequence-validator.js";
import { validatePreprocessingRealizabilityAnalysis } from "./preprocessing-realizability.js";
import { createFakeInputSpec, deleteTensors, resolveFakeInputShape } from "./runtime.js";

const MAX_CAPTURE_BYTES = 64 * 1024 * 1024;

export async function runPreprocessingConsequenceAtlas({
  analysis,
  modelBytes,
  ensureRuntime,
  runtimeVersion = "2.5.2",
  onProgress = null,
}) {
  assert(String(analysis?.format || "").toLowerCase() === "tflite", "Preprocessing consequence replay requires a TFLite artifact.");
  assert(modelBytes?.byteLength > 0, "Preprocessing consequence replay requires model bytes.");
  const variableCount = (analysis.tensors || []).filter((tensor) => tensor.is_variable).length;
  assert(variableCount === 0, `Preprocessing consequence replay is withheld for ${variableCount} stateful variable tensor(s).`);
  const preprocessingValidation = await validatePreprocessingRealizabilityAnalysis(analysis);
  assert(preprocessingValidation.evidence.candidates.length > 0, "Preprocessing consequence replay has no eligible candidate contracts.");
  const witness = preprocessingValidation.input.evidence.witnesses[preprocessingValidation.evidence.candidates[0].witness_index];
  assert(witness, "Preprocessing consequence replay witness is unavailable.");
  await ensureRuntime?.("wasm");
  onProgress?.({ completed: 0, total: preprocessingValidation.evidence.candidates.length + 1, label: "Compiling LiteRT.js WASM" });

  let model = null;
  try {
    model = await loadAndCompile(modelBytes, { accelerator: "wasm" });
    const inputDetails = Array.from(model.getInputDetails() || []);
    const outputDetails = Array.from(model.getOutputDetails() || []).map((details, index) => ({
      name: String(details.name || `output_${index}`),
      dtype: String(details.dtype || "unknown"),
      shape: Array.from(details.shape || []),
    }));
    assert(inputDetails.length > 0 && outputDetails.length > 0, "LiteRT.js did not expose model input/output details.");
    validatePrimaryInput(inputDetails[0], analysis.inputs?.[0], witness);
    const otherInputs = inputDetails.slice(1).map((details, index) => {
      const spec = createFakeInputSpec(details, analysis.inputs?.[index + 1]);
      return { input_index: index + 1, name: String(details.name || `input_${index + 1}`), dtype: details.dtype, shape: spec.shape, basis: spec.basis };
    });
    const baselineInput = buildCanonicalWitnessInput(witness);
    const baselineCapture = await captureDeterministicReplay(model, inputDetails, analysis, baselineInput);
    enforceCaptureBudget(baselineCapture.outputs, preprocessingValidation.evidence.candidates.length + 1);
    onProgress?.({ completed: 1, total: preprocessingValidation.evidence.candidates.length + 1, label: "Tensor-ABI witness" });

    const candidateCaptures = [];
    for (let index = 0; index < preprocessingValidation.evidence.candidates.length; index += 1) {
      const candidate = preprocessingValidation.evidence.candidates[index];
      const replayInput = await buildCandidateReplayInput(candidate, witness);
      const capture = await captureDeterministicReplay(model, inputDetails, analysis, replayInput.data);
      candidateCaptures.push({ ...capture, fixture: replayInput.fixture });
      onProgress?.({ completed: index + 2, total: preprocessingValidation.evidence.candidates.length + 1, label: candidate.contract_label });
      await yieldToBrowser();
    }
    const artifactSha256 = await sha256Hex(modelBytes);
    assert(!analysis.model_sha256 || analysis.model_sha256 === artifactSha256, "Preprocessing consequence artifact bytes do not match the active analysis SHA-256.");
    analysis.model_sha256 = artifactSha256;
    const evidence = await buildPreprocessingConsequenceEvidence({
      analysis,
      artifactSha256,
      runtime: { name: "@litertjs/core", version: runtimeVersion, backend: "wasm" },
      preprocessingValidation,
      baselineCapture,
      candidateCaptures,
      outputDetails,
      otherInputs,
    });
    const verification = await validatePreprocessingConsequenceCapture({
      analysis,
      evidence,
      baselineCapture,
      candidateCaptures,
      outputDetails,
    });
    return {
      evidence,
      verification,
      capture: {
        baseline: stripRepeat(baselineCapture),
        candidates: candidateCaptures.map(stripRepeat),
        outputDetails,
      },
    };
  } finally {
    model?.delete?.();
  }
}

async function captureDeterministicReplay(model, inputDetails, analysis, primaryInput) {
  const first = await executeOnce(model, inputDetails, analysis, primaryInput);
  const second = await executeOnce(model, inputDetails, analysis, primaryInput);
  assert(outputSetsEqual(first, second), "LiteRT.js replay was not byte-deterministic for an identical input tensor.");
  return {
    input: cloneTypedArray(primaryInput),
    outputs: first,
    repeat_outputs: second,
    deterministic_replay: true,
  };
}

async function executeOnce(model, inputDetails, analysis, primaryInput) {
  const inputs = [];
  let outputs = null;
  try {
    for (const [index, details] of inputDetails.entries()) {
      const shape = resolveFakeInputShape(Array.from(details.shape), analysis.inputs?.[index]);
      const data = index === 0 ? cloneTypedArray(primaryInput) : createFakeInputSpec(details, analysis.inputs?.[index]).data;
      inputs.push(Tensor.fromTypedArray(data, shape));
    }
    outputs = await model.run(inputs);
    return collectTensorArrays(outputs);
  } finally {
    deleteTensors(outputs);
    deleteTensors(inputs);
  }
}

async function collectTensorArrays(value) {
  const tensors = Array.isArray(value) ? value : Object.values(value || {});
  const arrays = [];
  for (const tensor of tensors) arrays.push(cloneTypedArray(await tensor.data()));
  return arrays;
}

function validatePrimaryInput(details, staticTensor, witness) {
  const shape = resolveFakeInputShape(Array.from(details.shape), staticTensor);
  const dtype = String(details.dtype || "").toLowerCase();
  assert(dtype === witness.model_input_dtype.toLowerCase(), `LiteRT.js input dtype ${dtype} does not match witness ${witness.model_input_dtype}.`);
  assert(sameShape(shape, witness.model_input_shape), `LiteRT.js input shape ${shape.join("x")} does not match witness ${witness.model_input_shape.join("x")}.`);
}

function enforceCaptureBudget(outputs, replayCount) {
  const outputBytes = outputs.reduce((total, output) => total + output.byteLength, 0);
  const projected = outputBytes * replayCount * 2;
  assert(projected <= MAX_CAPTURE_BYTES, `Preprocessing consequence capture would retain ${projected} bytes; the browser-local evidence budget is ${MAX_CAPTURE_BYTES} bytes.`);
}

function outputSetsEqual(left, right) {
  return left.length === right.length && left.every((array, index) => typedArrayBytesEqual(array, right[index]));
}

function typedArrayBytesEqual(left, right) {
  if (left.constructor !== right.constructor || left.byteLength !== right.byteLength || left.length !== right.length) return false;
  const a = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const b = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  return a.every((value, index) => value === b[index]);
}

function stripRepeat(capture) {
  return {
    input: capture.input,
    outputs: capture.outputs,
    deterministic_replay: capture.deterministic_replay,
    fixture: capture.fixture || null,
  };
}

function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
