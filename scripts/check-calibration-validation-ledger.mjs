import assert from "node:assert/strict";

import {
  buildCalibrationValidationLedger,
  CALIBRATION_VALIDATION_LEDGER_SCHEMA,
  validateCalibrationValidationLedger,
} from "../web/lib/calibration-validation-ledger.js";
import { buildRuntimeEvidence } from "../web/lib/report-evidence.js";
import { calibrationValidationMarkdown } from "../web/lib/report-sections.js";
import { buildMetricCoverageEntries } from "../web/lib/metric-coverage.js";

const artifactSha256 = "a".repeat(64);
const capture = {
  schema: "deepbom.representative_dataset_capture.v1",
  artifact_sha256: artifactSha256,
  dataset: {
    id: "fixture/calibration-v1",
    version: "1.0.0",
    manifest_sha256: "b".repeat(64),
    preprocessing_contract_sha256: "c".repeat(64),
    representativeness_claim: "fixture_only",
  },
  runtime: {
    name: "fixture-runtime",
    version: "1.2.3",
    backend: "cpu",
    binary_sha256: "d".repeat(64),
    build_inventory_sha256: "e".repeat(64),
  },
  samples: [
    {
      sample_id: "sample-0",
      sample_manifest_entry_sha256: "f".repeat(64),
      inputs: [{
        tensor_index: 0,
        name: "image",
        dtype: "INT8",
        shape: [1, 2, 2, 1],
        values: [-128, 0, 127, 127],
        quantization: { scale: 0.25, zero_point: 0 },
      }],
      reference_outputs: [{ tensor_index: 3, name: "scores", dtype: "FLOAT32", shape: [1, 3], values: [0, 1, 2] }],
      runs: [
        { run_index: 0, outputs: [{ tensor_index: 3, name: "scores", dtype: "FLOAT32", shape: [1, 3], values: [0, 1, 2] }] },
        { run_index: 1, outputs: [{ tensor_index: 3, name: "scores", dtype: "FLOAT32", shape: [1, 3], values: [0, 1.5, 2] }] },
      ],
    },
    {
      sample_id: "sample-1",
      inputs: [{ tensor_index: 0, name: "image", dtype: "UINT8", shape: [4], values: [0, 1, 255, 128] }],
      runs: [{ run_index: 0, outputs: [{ tensor_index: 3, name: "scores", dtype: "FLOAT32", shape: [1, 3], values: [3, 2, 1] }] }],
    },
  ],
};
const expectedInterface = {
  inputs: [{ parameter_index: 0, tensor_index: 0, name: "image", dtype: "INT8", shape: [-1, 2, 2, 1] }],
  outputs: [{ parameter_index: 0, tensor_index: 3, name: "scores", dtype: "FLOAT32", shape: [1, 3] }],
};

const interfaceBoundCapture = structuredClone(capture);
interfaceBoundCapture.samples[1].inputs[0] = { tensor_index: 0, name: "image", dtype: "INT8", shape: [1, 2, 2, 1], values: [-128, -1, 0, 127] };
const ledger = buildCalibrationValidationLedger(interfaceBoundCapture, { expectedArtifactSha256: artifactSha256, expectedInterface });
assert.equal(ledger.schema, CALIBRATION_VALIDATION_LEDGER_SCHEMA);
assert.equal(ledger.sample_count, 2);
assert.equal(ledger.input_endpoint_saturation.assessed_value_count, 8);
assert.equal(ledger.input_endpoint_saturation.endpoint_count, 5);
assert.equal(ledger.input_endpoint_saturation.endpoint_ratio, 5 / 8);
assert.equal(ledger.interface_binding.status, "matched_to_static_audit_external_interface");
assert.equal(ledger.reference_output_drift.assessed_sample_count, 1);
assert.equal(ledger.reference_output_drift.comparison_count, 2);
assert.equal(ledger.reference_output_drift.compared_value_count, 6);
assert.equal(ledger.reference_output_drift.changed_value_count, 1);
assert.equal(ledger.reference_output_drift.maximum_absolute_difference, 0.5);
assert.equal(ledger.repeat_nondeterminism.assessed_sample_count, 1);
assert.equal(ledger.repeat_nondeterminism.comparison_count, 1);
assert.equal(ledger.repeat_nondeterminism.compared_value_count, 3);
assert.equal(ledger.repeat_nondeterminism.changed_value_count, 1);
assert.equal(ledger.repeat_nondeterminism.maximum_absolute_difference, 0.5);
assert.equal(ledger.samples[1].reference_status, "not_provided");
assert.equal(ledger.samples[1].repeat_status, "not_assessed_single_run");
assert.deepEqual(ledger.hash_contract.excluded_pointers, ["/ledger_sha256"]);
assert.match(ledger.source_capture_sha256, /^[a-f0-9]{64}$/);
assert.match(ledger.ledger_sha256, /^[a-f0-9]{64}$/);

const second = buildCalibrationValidationLedger(structuredClone(interfaceBoundCapture), { expectedArtifactSha256: artifactSha256, expectedInterface });
assert.deepEqual(second, ledger, "Ledger construction must be deterministic.");
assert.equal(validateCalibrationValidationLedger(ledger, interfaceBoundCapture, { expectedArtifactSha256: artifactSha256, expectedInterface }).status, "independently_reconstructed");
const runtimeEvidence = buildRuntimeEvidence({ calibrationValidationResult: ledger });
assert.equal(runtimeEvidence.representative_dataset_validation.ledger_sha256, ledger.ledger_sha256);
assert.equal(runtimeEvidence.assessments.representative_dataset_validation.status, "assessed");
const markdown = calibrationValidationMarkdown(ledger);
assert.match(markdown, /Representative Dataset Validation/);
assert.match(markdown, new RegExp(ledger.ledger_sha256));
assert.match(markdown, /5 \/ 8 assessed values/);
const coverage = buildMetricCoverageEntries({ format: "coreml" }, { runtimeResults: runtimeEvidence });
const coverageRow = coverage.find((row) => row.metric_id === "validation.representative_dataset_capture");
assert.equal(coverageRow.status, "assessed");
assert.deepEqual(coverageRow.viewer_tabs, ["Drift Analysis", "Reports"]);

const tamperedLedger = structuredClone(ledger);
tamperedLedger.repeat_nondeterminism.changed_value_count = 0;
assert.throws(() => validateCalibrationValidationLedger(tamperedLedger, interfaceBoundCapture, { expectedArtifactSha256: artifactSha256, expectedInterface }), /does not reconstruct/);
assert.throws(() => buildCalibrationValidationLedger(interfaceBoundCapture, { expectedArtifactSha256: "0".repeat(64), expectedInterface }), /different artifact/);

const badInterface = structuredClone(interfaceBoundCapture);
badInterface.samples[0].inputs[0].shape = [1, 4, 1, 1];
assert.throws(() => buildCalibrationValidationLedger(badInterface, { expectedInterface }), /shape axis 1/);

const badShape = structuredClone(capture);
badShape.samples[0].inputs[0].shape = [5];
assert.throws(() => buildCalibrationValidationLedger(badShape), /cardinality/);
const badFinite = structuredClone(capture);
badFinite.samples[0].runs[0].outputs[0].values[0] = Number.POSITIVE_INFINITY;
assert.throws(() => buildCalibrationValidationLedger(badFinite), /finite JSON number/);
const badContract = structuredClone(capture);
badContract.samples[0].runs[1].outputs[0].dtype = "FLOAT64";
assert.throws(() => buildCalibrationValidationLedger(badContract), /dtype mismatch/);
const badQuant = structuredClone(capture);
badQuant.samples[0].inputs[0].quantization.scale = 0;
assert.throws(() => buildCalibrationValidationLedger(badQuant), /greater than zero/);

console.log("Calibration validation ledger passed (artifact/dataset binding, exact endpoint counts, reference drift, repeat nondeterminism, deterministic digest, and fail-closed mutation checks). ");
