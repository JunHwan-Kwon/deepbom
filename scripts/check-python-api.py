from __future__ import annotations

import importlib
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from subprocess import CompletedProcess
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

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.tensor_table.v1", "tensors": [{
    "name": "w", "element_count": "16", "effective_bits_per_element": "4.5", "byte_length": 9,
}]}) as invoke:
    rows = deepbom.tensors("model.gguf")
    expect(rows == [{"name": "w", "element_count": 16, "effective_bits_per_element": Decimal("4.5"), "byte_length": 9}],
           "tensors must return Python-native exact values")
    expect(invoke.call_args.args[0] == ["gguf", "model.gguf", "--tensors", "--compact"],
           "tensors must use the bounded CLI projection")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.analysis_selection.v1"}) as invoke:
    deepbom.audit("model.gguf", sections=["summary", "findings"])
    expect(invoke.call_args.args[0] == ["audit", "model.gguf", "--scan", "auto", "--section", "summary,findings", "--compact"],
           "sections without an explicit output must select analysis")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.artifact_evidence_envelope.v1"}) as invoke:
    deepbom.audit("model.safetensors", gate="defects")
    expect(invoke.call_args.args[0][-2:] == ["--gate", "defects"], "gate must reach the engine")
    deepbom.audit("model.safetensors", policy="regulatory")
    expect(invoke.call_args.args[0][-2:] == ["--policy", "regulatory"], "policy must reach the engine")

for callable_value in (
    lambda: deepbom.audit("model.gguf", output="summary"),
    lambda: deepbom.audit("model.gguf", sections=["bad/name"]),
    lambda: deepbom.audit("model.gguf", expected_sha256="0"),
    lambda: deepbom.audit("model.gguf", timeout_seconds=0),
    lambda: deepbom.audit("model.gguf", timeout_seconds=float("nan")),
    lambda: deepbom.audit("model.gguf", sections="summary"),
    lambda: deepbom.audit("model.gguf", gate="defects", policy="engineering"),
    lambda: deepbom.audit("model.gguf", gate="warnings"),
    lambda: deepbom.audit("model.gguf", policy="legal"),
    lambda: deepbom.audit("model.gguf", sections=["summary"], output="envelope"),
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

with tempfile.TemporaryDirectory(prefix="deepbom-policy-test-") as directory:
    completed_document = {"schema": "deepbom.artifact_evidence_envelope.v1", "findings": [{"id": "EA-SER-0001"}]}

    def blocked_run(command, **kwargs):
        output_index = command.index("--output") + 1
        Path(command[output_index]).write_text(json.dumps(completed_document), encoding="utf-8")
        return CompletedProcess(command, 2, stdout="", stderr="deepbom: review policy blocked")

    with patch.object(api, "_verified_engine", return_value=(Path(sys.executable), None)), patch.object(api.subprocess, "run", side_effect=blocked_run):
        try:
            deepbom.audit("nan-weight.safetensors", gate="defects")
        except deepbom.DeepBomPolicyBlocked as error:
            expect(error.exit_code == 2 and error.document == completed_document,
                   "policy block must retain the completed evidence document")
        else:
            raise AssertionError("a gate exit 2 must raise DeepBomPolicyBlocked")

print("Python facade contract checks passed.")
