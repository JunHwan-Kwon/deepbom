from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "channels" / "python" / "src"))

deepbom = importlib.import_module("deepbom")
api = importlib.import_module("deepbom.api")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


expect(callable(deepbom.audit), "audit must be exported")
expect(callable(deepbom.tensors), "tensors must be exported")
expect(callable(deepbom.capabilities), "capabilities must be exported")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.artifact_evidence_envelope.v1"}) as invoke:
    result = deepbom.audit("model.gguf")
    expect(result["schema"].endswith("envelope.v1"), "audit must return the parsed document")
    expect(invoke.call_args.args[0] == ["audit", "model.gguf", "--scan", "auto", "--output-format", "envelope", "--compact"],
           "audit must build the canonical envelope invocation")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.tensor_table.v1", "tensors": [{"name": "w"}]}) as invoke:
    rows = deepbom.tensors("model.gguf")
    expect(rows == [{"name": "w"}], "tensors must return only compact rows")
    expect(invoke.call_args.args[0] == ["gguf", "model.gguf", "--tensors", "--compact"],
           "tensors must use the bounded CLI projection")

for callable_value in (
    lambda: deepbom.audit("model.gguf", output="summary"),
    lambda: deepbom.audit("model.gguf", sections=["bad/name"]),
    lambda: deepbom.audit("model.gguf", expected_sha256="0"),
    lambda: deepbom.audit("model.gguf", timeout_seconds=0),
    lambda: deepbom.audit("model.gguf", timeout_seconds=float("nan")),
    lambda: deepbom.audit("model.gguf", sections="summary"),
):
    try:
        callable_value()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid API input must fail before engine execution")

with tempfile.TemporaryDirectory(prefix="deepbom-api-test-") as directory:
    output = Path(directory) / "result.json"
    output.write_text(json.dumps({"schema": "test"}), encoding="utf-8")
    expect(api._read_bounded_json(output, 100) == {"schema": "test"}, "bounded JSON reader")
    try:
        api._read_bounded_json(output, 1)
    except deepbom.DeepBomOutputTooLarge:
        pass
    else:
        raise AssertionError("oversized JSON must fail before reading")

print("Python facade contract checks passed.")
