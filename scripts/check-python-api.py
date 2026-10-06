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
expect(callable(deepbom.model_ir), "model_ir must be exported")
expect(callable(deepbom.visualization_manifest), "visualization_manifest must be exported")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.artifact_evidence_envelope.v1"}) as invoke:
    result = deepbom.audit("model.gguf")
    expect(result["schema"].endswith("envelope.v1"), "audit must return the parsed document")
    expect(invoke.call_args.args[0] == ["audit", str(Path("model.gguf").absolute()), "--scan", "auto", "--output-format", "envelope", "--compact"],
           "audit must build the canonical envelope invocation")

with patch.object(api, "_invoke_json", return_value={
    "schema": "deepbom.analysis_selection.v1",
    "sections": {"model_ir": {"schema": "deepbom.model_ir.v1", "model_ir_sha256": "a" * 64}},
}) as invoke:
    document = deepbom.model_ir("model.onnx")
    expect(document["schema"] == "deepbom.model_ir.v1", "model_ir must unwrap the selected section")
    expect(invoke.call_args.args[0] == ["audit", str(Path("model.onnx").absolute()), "--scan", "auto", "--section", "model_ir", "--compact"],
           "model_ir must use the public bounded section contract")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.model_ir_visualization_manifest.v1"}) as invoke:
    deepbom.visualization_manifest("model.onnx", views=["architecture-overview"], orientation="landscape")
    expect(invoke.call_args.args[0] == ["visualize", str(Path("model.onnx").absolute()), "--view", "architecture-overview", "--orientation", "landscape", "--output-format", "json", "--compact"],
           "visualization_manifest must use the deterministic JSON projection")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.tensor_table.v1", "tensors": [{
    "name": "w", "element_count": "16", "effective_bits_per_element": "4.5", "byte_length": 9,
}]}) as invoke:
    rows = deepbom.tensors("model.gguf")
    expect(rows == [{"name": "w", "element_count": 16, "effective_bits_per_element": Decimal("4.5"), "byte_length": 9}],
           "tensors must return Python-native exact values")
    expect(invoke.call_args.args[0] == ["gguf", str(Path("model.gguf").absolute()), "--tensors", "--compact"],
           "tensors must use the bounded CLI projection")

with patch.object(api, "_invoke_json", return_value={"schema": "deepbom.analysis_selection.v1"}) as invoke:
    deepbom.audit("model.gguf", sections=["summary", "findings"])
    expect(invoke.call_args.args[0] == ["audit", str(Path("model.gguf").absolute()), "--scan", "auto", "--section", "summary,findings", "--compact"],
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
    lambda: deepbom.visualization_manifest("model.onnx", views=["unknown"]),
    lambda: deepbom.visualization_manifest("model.onnx", orientation="diagonal"),
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

summary = {"schema": "deepbom.review_summary.v1", "graph": {"total_macs": None}}
selection = {"schema": "deepbom.analysis_selection.v1", "sections": {"summary": summary}}
with patch.object(api, "_invoke_json", return_value=selection):
    expect(deepbom.inspect("model.onnx") is summary, "inspect must forward the engine summary without recomputing it")
with patch.object(api, "_invoke_json", side_effect=deepbom.DeepBomPolicyBlocked("blocked", exit_code=2, document=selection)):
    try:
        deepbom.inspect("model.onnx", gate="defects")
    except deepbom.DeepBomPolicyBlocked as error:
        expect(error.document is summary and error.code == "POLICY_BLOCKED", "inspect policy failure must retain the summary")
    else:
        raise AssertionError("expected policy failure")
with patch.object(api, "_verified_engine", side_effect=RuntimeError("checksum mismatch")):
    try:
        deepbom.capabilities()
    except deepbom.DeepBomInvocationError as error:
        expect(error.code == "INVOCATION_FAILED", "engine-resolution error must be typed")
    else:
        raise AssertionError("invalid engine must fail")
with tempfile.TemporaryDirectory(prefix="deepbom-sdk-json-") as directory:
    invalid = Path(directory) / "invalid.json"
    invalid.write_text('{"value": NaN}', encoding="utf-8")
    try:
        api._read_bounded_json(invalid, 1024)
    except deepbom.DeepBomInvocationError:
        pass
    else:
        raise AssertionError("non-JSON numeric constants must be rejected consistently with Node")
expect(api._path_text("-model.onnx") == str(Path("-model.onnx").absolute()), "leading '-' filenames must not become options")
with tempfile.TemporaryDirectory(prefix="deepbom-sdk-strict-json-") as directory:
    invalid = Path(directory) / "invalid.json"
    for raw in [b'{"value": 1e999}', b'{"value": "\xff"}', b'[]']:
        invalid.write_bytes(raw)
        try:
            api._read_bounded_json(invalid, 1024)
        except deepbom.DeepBomInvocationError:
            pass
        else:
            raise AssertionError("non-finite, invalid UTF-8 and non-object results must fail")

def noisy_run(command, **kwargs):
    kwargs["stderr"].write(b"x" * 100000 + b" diagnostic-tail")
    return CompletedProcess(command, 1)

with patch.object(api, "_verified_engine", return_value=(Path(sys.executable), None)), patch.object(api.subprocess, "run", side_effect=noisy_run):
    try:
        deepbom.capabilities()
    except deepbom.DeepBomInvocationError as error:
        expect(len(str(error)) <= 16384 and str(error).endswith("diagnostic-tail"), "retain a bounded diagnostic tail")
        expect(error.exit_code == 1, "retain engine exit status")
    else:
        raise AssertionError("failed engine must not succeed")
with patch.object(api, "_verified_engine", return_value=(Path(sys.executable), None)), patch.object(api.tempfile, "TemporaryDirectory", side_effect=OSError("disk unavailable")):
    try:
        deepbom.capabilities()
    except deepbom.DeepBomInvocationError:
        pass
    else:
        raise AssertionError("temporary filesystem failures need the public exception contract")
print("Python SDK contract checks passed.")
