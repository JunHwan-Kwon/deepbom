#!/usr/bin/env python3
"""Generate the bounded ER-C2 Core ML fixtures with coremltools 9.0."""

from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.proto import Model_pb2

mil_load = importlib.import_module("coremltools.converters.mil.backend.mil.load")


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "corpus" / "external-review" / "fixtures"
PACKAGE = FIXTURES / "coreml-affine-attributes.mlpackage"
MODEL_PATH = PACKAGE / "Data" / "com.deepbom.fixture" / "model.mlmodel"
INVALID_PATH = FIXTURES / "coreml-affine-missing-axis.mlmodel"
MANIFEST_PATH = FIXTURES / "coreml-affine-attributes.manifest.json"

COREMLTOOLS_VERSION = "9.0"
COREMLTOOLS_SDIST_SHA256 = "4ff346b29c31c4b45acd19a20e0f0a1ac65180a96776e62f15bd5c46f4926687"
EXPORTER_SHA256 = "1a01006e1d87eb37626869c249c3e9df755cbb9a83a9c1e233d68a4b435620ae"
OP_DEFINITION_SHA256 = "2575e57c32f99c18e8e9d32af152ee19f9414835cb1c141ab19a92ec0b659a6d"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_spec() -> Model_pb2.Model:
    if ct.__version__ != COREMLTOOLS_VERSION:
        raise RuntimeError(f"coremltools {COREMLTOOLS_VERSION} is required, observed {ct.__version__}")

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))], opset_version=ct.target.iOS16)
    def program(x):
        weight = mb.constexpr_affine_dequantize(
            quantized_data=np.arange(6, dtype=np.int8).reshape(2, 3),
            zero_point=np.zeros(3, dtype=np.int8),
            scale=np.array([0.25, 0.5, 0.75], dtype=np.float32),
            axis=np.int32(1),
            name="dequantized_weight",
        )
        return mb.add(x=x, y=weight, name="output")

    # Windows wheels do not ship Apple's blob writer. This fixture keeps every
    # value immediate, so a non-null sentinel only bypasses the exporter's
    # unconditional availability check; any attempted blob write still fails.
    if mil_load.BlobWriter is None:
        mil_load.BlobWriter = object
    with tempfile.TemporaryDirectory() as weights_dir:
        return mil_load.load(program, weights_dir, specification_version=7)


def main() -> None:
    spec = build_spec()
    valid = spec.SerializeToString(deterministic=True)
    operation = spec.mlProgram.functions["main"].block_specializations["CoreML6"].operations[0]
    if operation.type != "constexpr_affine_dequantize" or operation.inputs:
        raise RuntimeError("Pinned exporter no longer emits the expected CoreML6 constexpr operation")
    if set(operation.attributes) != {"name", "quantized_data", "zero_point", "scale", "axis"}:
        raise RuntimeError(f"Unexpected affine attribute set: {sorted(operation.attributes)}")

    malformed = Model_pb2.Model()
    malformed.CopyFrom(spec)
    del malformed.mlProgram.functions["main"].block_specializations["CoreML6"].operations[0].attributes["axis"]
    invalid = malformed.SerializeToString(deterministic=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(valid)
    INVALID_PATH.write_bytes(invalid)
    package_manifest = {
        "fileFormatVersion": "1.0.0",
        "rootModelIdentifier": "root",
        "itemInfoEntries": {
            "root": {
                "path": "com.deepbom.fixture",
                "name": "Model",
                "author": "DEEPBOM",
                "description": "coremltools 9.0 affine-attribute conformance fixture",
            }
        },
    }
    (PACKAGE / "Manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "schema": "deepbom.external_review_coreml_fixtures.v1",
        "generator": {
            "name": "coremltools",
            "version": COREMLTOOLS_VERSION,
            "sdist_sha256": COREMLTOOLS_SDIST_SHA256,
            "backend_exporter_sha256": EXPORTER_SHA256,
            "affine_op_definition_sha256": OP_DEFINITION_SHA256,
        },
        "fixtures": [
            {
                "path": MODEL_PATH.relative_to(FIXTURES).as_posix(),
                "bytes": len(valid),
                "sha256": sha256(valid),
                "expected": {
                    "status": "assessed",
                    "binding_source": "attribute",
                    "normalized_axis": 1,
                    "scale_elements": 3,
                },
            },
            {
                "path": INVALID_PATH.relative_to(FIXTURES).as_posix(),
                "bytes": len(invalid),
                "sha256": sha256(invalid),
                "expected": {"status": "malformed", "missing_argument": "axis"},
            },
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
