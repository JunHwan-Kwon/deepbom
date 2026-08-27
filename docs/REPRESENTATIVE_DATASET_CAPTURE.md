# Representative Dataset Capture Contract

DEEPBOM accepts `deepbom.representative_dataset_capture.v1` JSON in the
Reports workspace after a static audit. The capture remains local. It is
accepted only when its artifact SHA-256 and complete external tensor contract
match the active audit.

## Required Identity

- `artifact_sha256`: lowercase SHA-256 of the exact audited artifact.
- `dataset.id`, `dataset.version`, and `dataset.manifest_sha256`: identity of
  the sampled dataset and its immutable manifest.
- `dataset.preprocessing_contract_sha256`: optional SHA-256 of the exact
  source-to-tensor implementation or contract. Absence remains explicit.
- `runtime.name`, `runtime.version`, and `runtime.backend`: runtime identity.
- `runtime.binary_sha256`, `runtime.build_inventory_sha256`, and
  `runtime.device_profile_sha256`: optional stronger runtime bindings.

The dataset `representativeness_claim` is an external declaration. DEEPBOM
hashes and preserves it but does not verify population representativeness.

## Tensor Capture

Each sample contains every external input and at least one runtime run. A
reference output set is optional.

```json
{
  "schema": "deepbom.representative_dataset_capture.v1",
  "artifact_sha256": "<64 lowercase hex characters>",
  "dataset": {
    "id": "dataset-name",
    "version": "1.0.0",
    "manifest_sha256": "<64 lowercase hex characters>",
    "preprocessing_contract_sha256": null,
    "representativeness_claim": "externally_declared_not_verified_by_deepbom"
  },
  "runtime": {
    "name": "runtime-name",
    "version": "runtime-version",
    "backend": "backend-name",
    "binary_sha256": null,
    "build_inventory_sha256": null,
    "device_profile_sha256": null
  },
  "samples": [
    {
      "sample_id": "sample-000001",
      "sample_manifest_entry_sha256": null,
      "inputs": [{
        "tensor_index": 0,
        "name": "input",
        "dtype": "UINT8",
        "shape": [1, 1, 1, 3],
        "quantization": { "scale": 0.0078125, "zero_point": 128 },
        "values": [0, 128, 255]
      }],
      "reference_outputs": null,
      "runs": [{
        "run_index": 0,
        "outputs": [{
          "tensor_index": 171,
          "name": "output",
          "dtype": "UINT8",
          "shape": [1, 3],
          "values": [58, 58, 59]
        }]
      }]
    }
  ]
}
```

The small shapes above illustrate fields only. Every `values` array must have
exactly the product of its declared shape. Every sample must contain the same
number of inputs and outputs as the audited external interface. Dtype, rank,
all static dimensions, tensor index when present, and nonempty tensor name
when present must match. A negative or symbolic audited dimension may bind to
a concrete captured runtime dimension. Runs and references must have identical
output dtype and shape contracts.

Supported capture dtypes are `INT8`, `UINT8`, `INT16`, `UINT16`, `INT32`,
`UINT32`, `FLOAT16`, `BFLOAT16`, `FLOAT32`, `FLOAT64`, and `BOOL`. Values must
be finite JSON numbers; integer values must be integral and within the declared
storage range. The parser rejects malformed hashes, duplicate run indices,
shape/cardinality differences, non-finite values, unsupported dtypes, and
cross-artifact captures.

## Deterministic Calculations

For every bounded-integer external input, endpoint saturation is:

```text
endpoint_ratio = count(value == dtype_min or value == dtype_max)
                 / assessed_integer_input_value_count
```

For each same-contract reference/run pair and first-run/repeated-run pair,
DEEPBOM calculates changed-value count, mean and maximum absolute difference,
RMS difference, relative L2 difference, cosine distance when both norms are
nonzero, and raw argmax changes. Aggregates retain sample, comparison, tensor,
and value denominators. Integer output comparisons are storage-code
comparisons; floating output comparisons use the captured finite numerical
values.

The emitted `deepbom.calibration_validation_ledger.v1` contains the source
capture SHA-256 and a reproducible ledger SHA-256. The ledger hash is SHA-256
over UTF-8 RFC 8785 JCS canonical JSON with `/ledger_sha256` omitted. The
`hash_contract.excluded_pointers` field states that exclusion explicitly.

## Evidence Boundary

The ledger establishes only what is present in the bound capture:

- exact interface storage endpoint counts;
- numerical difference from supplied same-contract reference outputs; and
- numerical difference among supplied repeated runtime outputs.

It does not establish dataset representativeness, calibration quality, task
accuracy, clinical validity, production preprocessing identity, production
workload frequency, device-wide determinism, or release readiness. Labeled
task evaluation requires a separately bound label ontology, metric definition,
acceptance threshold, preprocessing/postprocessing implementation, and
evaluation protocol.
