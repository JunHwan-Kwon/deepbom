import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { analyze_tflite_for_target, initSync } from "../pkg/tflite_wasm_audit.js";
import {
  buildCandidateReplayInput,
  buildCanonicalWitnessInput,
  buildPreprocessingConsequenceEvidence,
} from "../web/lib/preprocessing-consequence-core.js";
import { validatePreprocessingConsequenceCapture } from "../web/lib/preprocessing-consequence-validator.js";
import { validatePreprocessingRealizabilityAnalysis } from "../web/lib/preprocessing-realizability.js";
import { createCheck } from "./check-assert.mjs";

const { done, expect, expectEqual } = createCheck("Preprocessing consequence core check");
const filename = "mobilenet_v2_1.0_224_quant.tflite";
initSync({ module: readFileSync("pkg/tflite_wasm_audit_bg.wasm") });
const modelBytes = new Uint8Array(readFileSync(`web/samples/${filename}`));
const analysis = analyze_tflite_for_target(modelBytes, filename, "android_mid_a55");
analysis.model_sha256 = createHash("sha256").update(modelBytes).digest("hex");
const preprocessingValidation = await validatePreprocessingRealizabilityAnalysis(analysis);
const witness = preprocessingValidation.input.evidence.witnesses[0];
const baselineInput = buildCanonicalWitnessInput(witness);
const baselineCapture = capture(baselineInput);
const candidateCaptures = [];
for (const candidate of preprocessingValidation.evidence.candidates) {
  const replay = await buildCandidateReplayInput(candidate, witness);
  candidateCaptures.push({ ...capture(replay.data), fixture: replay.fixture });
}
const outputDetails = [{ name: "deterministic_test_output", dtype: "uint8", shape: [1, 1001] }];
const evidence = await buildPreprocessingConsequenceEvidence({
  analysis,
  artifactSha256: analysis.model_sha256,
  runtime: { name: "deepbom-test-runtime", version: "1", backend: "deterministic" },
  preprocessingValidation,
  baselineCapture,
  candidateCaptures,
  outputDetails,
});
const verification = await validatePreprocessingConsequenceCapture({ analysis, evidence, baselineCapture, candidateCaptures, outputDetails });

expectEqual(`${evidence.schema}:${evidence.method_version}:${evidence.evidence_class}:${evidence.status}`, "deepbom.preprocessing_consequence_atlas.v1:2026-07-18.1:MEASURED_SYNTHETIC:assessed", "Consequence schema and evidence class should remain stable.");
expectEqual(`${evidence.candidate_count}:${evidence.exact_source_contract_count}:${evidence.non_exact_source_contract_count}`, "8:4:4", "The full preprocessing candidate matrix should replay.");
expectEqual(`${evidence.unique_input_tensor_count}:${evidence.input_equivalence_classes.map((row) => row.candidate_count).join(",")}`, "4:4,1,1,2", "Input fingerprint equivalence classes should remain deterministic.");
expect(evidence.exact_contract_output_conservation, "Every exact source contract should conserve the witness input and deterministic output.");
expect(evidence.candidates.every((row) => row.deterministic_replay && /^[a-f0-9]{64}$/.test(row.candidate_ledger_sha256)), "Every candidate should retain deterministic replay and a ledger digest.");
expectEqual(`${verification.status}:${verification.candidate_count}:${verification.captured_repetitions_per_input}`, "independently_verified:8:2", "Independent capture verification should cover every candidate and repetition.");
expect(/^[a-f0-9]{64}$/.test(evidence.portfolio_ledger_sha256), "The consequence portfolio should expose SHA-256.");

await expectRejected((copy) => { copy.candidates[0].input_changed_element_count = 1; }, "input difference mismatch", "Input-difference tampering");
await expectRejected((copy) => { copy.candidates[0].output_tensor_set_sha256 = "0".repeat(64); }, "digest mismatch", "Output-digest tampering");
await expectRejected((copy) => { copy.candidates[0].candidate_ledger_sha256 = "0".repeat(64); }, "candidate ledger mismatch", "Candidate-ledger tampering");
await expectRejected((copy) => { copy.portfolio_ledger_sha256 = "0".repeat(64); }, "portfolio ledger mismatch", "Portfolio-ledger tampering");
const repeatTamper = structuredClone(candidateCaptures);
repeatTamper[0].repeat_outputs[0][0] ^= 1;
await expectCaptureRejected(structuredClone(evidence), repeatTamper, "non-deterministic", "Repeated-output tampering");

done("Preprocessing consequence core passed (8 candidate tensors, deterministic duplicate captures, exact input/output equivalence classes, independent reconstruction, and tamper rejection)." );

function capture(input) {
  const output = deterministicOutput(input);
  return {
    input: input.slice(),
    outputs: [output],
    repeat_outputs: [output.slice()],
    deterministic_replay: true,
  };
}

function deterministicOutput(input) {
  const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  const output = new Uint8Array(1001);
  for (let index = 0; index < output.length; index += 1) {
    const a = bytes[(index * 149 + 17) % bytes.length];
    const b = bytes[(index * 509 + 31) % bytes.length];
    output[index] = (a + 3 * b + index) & 0xff;
  }
  return output;
}

async function expectRejected(mutate, message, label) {
  const copy = structuredClone(evidence);
  mutate(copy);
  await expectCaptureRejected(copy, candidateCaptures, message, label);
}

async function expectCaptureRejected(copy, captures, message, label) {
  try {
    await validatePreprocessingConsequenceCapture({ analysis, evidence: copy, baselineCapture, candidateCaptures: captures, outputDetails });
    expect(false, `${label} should be rejected.`);
  } catch (error) {
    expect(String(error?.message || error).toLowerCase().includes(message.toLowerCase()), `${label} expected ${message}, received ${error?.message || error}.`);
  }
}
