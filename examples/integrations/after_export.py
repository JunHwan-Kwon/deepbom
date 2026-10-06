"""Export deterministic toy ONNX models, then inspect the exact saved bytes.

Run with the installed DEEPBOM wheel and requirements.txt. No training,
inference, clinical claim, or optimization benefit is demonstrated here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import deepbom


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def export_model(path, columns=2, first_weight=0.25):
    weights = [first_weight] + [0.25] * (4 * columns - 1)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["linear"], name="projection"),
         helper.make_node("Relu", ["linear"], ["output"], name="activation")],
        "sdk-integration-example",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, columns])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [4, columns], weights)],
    )
    model = helper.make_model(graph, producer_name="DEEPBOM integration example",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, columns, first_weight in [("baseline", 2, 0.25), ("candidate", 2, 0.5), ("incompatible", 3, 0.25)]:
        artifact = directory / f"{name}.onnx"
        export_model(artifact, columns, first_weight)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        summary = deepbom.inspect(artifact, expected_sha256=digest, scan="full")
        envelope = deepbom.audit(artifact, expected_sha256=digest, scan="full")
        require(summary["artifact"]["sha256"] == envelope["identity"]["sha256"] == digest, "Artifact identity changed between inspections")
        require(summary["evidence_envelope_sha256"] == envelope["envelope_sha256"], "Summary and envelope do not describe the same evidence")
        # Independent oracle: [1,4] @ [4,columns] has 4*columns MACs; Relu has none.
        require(summary["graph"]["total_macs"] == 4 * columns, "Toy MatMul cost differs from the analytic count")
        require(summary["graph"]["operator_count"] == 2, "Toy graph must retain MatMul and Relu")
        write_json(directory / f"{name}.summary.json", summary)
        write_json(directory / f"{name}.evidence.json", envelope)
        entries[name] = {"file": artifact.name, "sha256": digest}
        print(f"{name}: {digest}; {summary['verdict']['status']}; "
              f"unavailable capabilities: {summary['coverage']['unavailable']}; "
              f"evidence-gap findings: {summary['verdict']['evidence_needed_count']}")
    write_json(directory / "inputs.json", entries)
    print("Saved source-bound evidence. An unchanged interface does not establish output equivalence.")


if __name__ == "__main__":
    main()
