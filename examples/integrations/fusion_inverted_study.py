"""Development experiment: real Conv/BN folding versus untrained block replacement.

Only executes deterministic synthetic models created here. DEEPBOM performs
static inspection; ONNX Runtime performs the explicitly separate execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import deepbom
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from mlflow.tracking import MlflowClient
from after_export import require, write_json


SHAPE = [1, 16, 32, 32]
ATOL, RTOL = 1e-5, 1e-4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(path, nodes, arrays):
    model = helper.make_model(helper.make_graph(nodes, "development-block",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
        [numpy_helper.from_array(value.astype(np.float32), name) for name, value in arrays.items()]),
        producer_name="DEEPBOM synthetic development example", ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)])
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def create_baseline(path):
    rng = np.random.default_rng(1234)
    arrays = {"weight": rng.normal(0, .08, [16, 16, 3, 3]), "bias": rng.normal(0, .05, [16]),
              "scale": rng.uniform(.7, 1.3, [16]), "offset": rng.normal(0, .1, [16]),
              "mean": rng.normal(0, .2, [16]), "variance": rng.uniform(.6, 1.4, [16])}
    save_model(path, [helper.make_node("Conv", ["input", "weight", "bias"], ["conv"],
                                     name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                      helper.make_node("BatchNormalization", ["conv", "scale", "offset", "mean", "variance"],
                                       ["normalized"], name="bn", epsilon=1e-5),
                      helper.make_node("Relu", ["normalized"], ["output"], name="relu")], arrays)


def create_inverted(path, expansion):
    rng = np.random.default_rng(1234 + expansion)
    hidden = 16 * expansion
    arrays = {"expand_weight": rng.normal(0, .08, [hidden, 16, 1, 1]),
              "depthwise_weight": rng.normal(0, .08, [hidden, 1, 3, 3]),
              "project_weight": rng.normal(0, .08, [16, hidden, 1, 1]),
              "clip_min": np.array(0.), "clip_max": np.array(6.)}
    save_model(path, [helper.make_node("Conv", ["input", "expand_weight"], ["expanded"], name="expand", kernel_shape=[1, 1]),
                      helper.make_node("Clip", ["expanded", "clip_min", "clip_max"], ["expand_relu6"], name="expand_relu6"),
                      helper.make_node("Conv", ["expand_relu6", "depthwise_weight"], ["spatial"], name="depthwise",
                                       kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=hidden),
                      helper.make_node("Clip", ["spatial", "clip_min", "clip_max"], ["spatial_relu6"], name="spatial_relu6"),
                      helper.make_node("Conv", ["spatial_relu6", "project_weight"], ["projected"], name="linear_projection", kernel_shape=[1, 1]),
                      helper.make_node("Add", ["input", "projected"], ["output"], name="residual")], arrays)


def session(path, *, optimized_output=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if optimized_output
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    if optimized_output:
        options.optimized_model_filepath = str(optimized_output)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def validation_inputs():
    rng = np.random.default_rng(5678)
    values = [np.zeros(SHAPE, np.float32), np.ones(SHAPE, np.float32), -np.ones(SHAPE, np.float32)]
    impulse = np.zeros(SHAPE, np.float32)
    impulse[0, 0, 16, 16] = 1
    values.append(impulse)
    values.extend(rng.normal(0, scale, SHAPE).astype(np.float32) for scale in [.01, .1, 1., 3.] for _ in range(3))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--store", type=Path, default=root / ".local-validation/mlflow-architecture")
    args = parser.parse_args()
    store = args.store.resolve()
    study_id = uuid4().hex
    bundle = store / f"study-{study_id}"
    bundle.mkdir(parents=True)
    for name in ["baseline", "conv_bn_folded", "inverted_e2", "inverted_e6"]:
        (bundle / name).mkdir()
    baseline = bundle / "baseline/model.onnx"
    folded = bundle / "conv_bn_folded/model.onnx"
    create_baseline(baseline)
    # Use the runtime's maintained rewrite; no duplicate BN-fold formula.
    session(baseline, optimized_output=folded)
    require([node.op_type for node in onnx.load(folded).graph.node] == ["Conv", "Relu"],
            "Conv/BatchNorm folding was not observed in the exported graph")
    for expansion in [2, 6]:
        create_inverted(bundle / f"inverted_e{expansion}/model.onnx", expansion)
    inputs = validation_inputs()
    baseline_session = session(baseline)
    baseline_outputs = [baseline_session.run(None, {"input": value})[0] for value in inputs]
    # Preserve the exact numerical test inputs, not only the random seed.
    np.savez(bundle / "validation-inputs.npz", **{f"case_{i}": x for i, x in enumerate(inputs)})
    execution = {"runtime": "onnxruntime", "version": ort.__version__, "provider": "CPUExecutionProvider",
                 "platform": platform.platform(), "numpy_version": np.__version__,
                 "onnx_version": onnx.__version__, "intra_op_threads": 1, "inter_op_threads": 1,
                 "validation_optimization_level": "ORT_DISABLE_ALL", "rewrite_level": "ORT_ENABLE_BASIC",
                 "input_file_sha256": digest(bundle / "validation-inputs.npz"),
                 "example_script_sha256": digest(Path(__file__).resolve()),
                 "atol": ATOL, "rtol": RTOL, "case_count": len(inputs),
                 "boundary": "Numerical regression on synthetic inputs only; no task dataset, training, latency measurement or full-domain equivalence proof."}
    write_json(bundle / "execution.json", execution)
    records = []
    for name in ["baseline", "conv_bn_folded", "inverted_e2", "inverted_e6"]:
        directory = bundle / name
        artifact = directory / "model.onnx"
        sha = digest(artifact)
        summary = deepbom.inspect(artifact, expected_sha256=sha, scan="full")
        write_json(directory / "summary.json", summary)
        comparison = deepbom.diff(baseline, artifact)
        require(comparison["baseline"]["sha256"] == digest(baseline) and comparison["candidate"]["sha256"] == sha,
                "Comparison subjects changed")
        require(comparison["graph_delta"]["interface_comparison_status"] == "assessed_serialized_contracts" and
                not comparison["graph_delta"]["input_contract_changed"] and
                not comparison["graph_delta"]["output_contract_changed"], "Block interface changed")
        write_json(directory / "comparison.json", comparison)
        subprocess.run(["node", str(root / "bin/deepbom.mjs"), "audit", str(artifact),
                        "--weight-analysis", "--expected-sha256", sha, "--output-format", "json",
                        "-o", str(directory / "weight-analysis.json")], check=True, timeout=120)
        subprocess.run(["node", str(root / "bin/deepbom.mjs"), "graph", str(artifact),
                        "--format", "svg", "--expected-sha256", sha,
                        "-o", str(directory / "graph.svg")], check=True, timeout=120)
        current = session(artifact)
        outcomes = []
        for expected, value in zip(baseline_outputs, inputs):
            observed = current.run(None, {"input": value})[0]
            require(list(observed.shape) == SHAPE and np.isfinite(observed).all(), "Invalid execution output")
            outcomes.append({"max_absolute_error": float(np.max(np.abs(observed.astype(np.float64) - expected))),
                             "within_tolerance": bool(np.allclose(observed, expected, atol=ATOL, rtol=RTOL))})
        if name == "conv_bn_folded":
            require(all(row["within_tolerance"] for row in outcomes), "Fusion numerical regression failed")
        evaluation = {"artifact_sha256": sha, "baseline_sha256": digest(baseline),
                      "comparison_purpose": "output_regression" if "inverted" not in name else "show_untrained_output_difference",
                      "cases": outcomes, "max_absolute_error": max(row["max_absolute_error"] for row in outcomes),
                      "all_within_tolerance": all(row["within_tolerance"] for row in outcomes),
                      "execution_record_sha256": digest(bundle / "execution.json")}
        write_json(directory / "execution-comparison.json", evaluation)
        # Small independent MAC oracle tests group handling and conv shape accounting.
        expected_macs = 32 * 32 * 16 * 16 * 9 if "inverted" not in name else 32 * 32 * (16 * (16 * int(name[-1])) * 2 + 9 * (16 * int(name[-1])))
        require(summary["graph"]["total_macs"] == expected_macs, "Shared engine MAC result differs from block oracle")
        record = {"name": name, "artifact_sha256": sha, "operator_count": summary["graph"]["operator_count"],
                  "macs": summary["graph"]["total_macs"], "external_interface_preserved": True,
                  "state": ("numerical_regression_passed_on_test_inputs" if name == "conv_bn_folded" else
                            "requires_training_and_task_evaluation" if name.startswith("inverted") else "synthetic_reference"),
                  "max_absolute_error": evaluation["max_absolute_error"], "within_tolerance": evaluation["all_within_tolerance"]}
        records.append(record)
    write_json(bundle / "study.json", {"scenario": "fusion-vs-inverted-development", "records": records,
        "boundary": "Synthetic ONNX block demonstration. Fusion by ONNX Runtime; alternative blocks by this example; static evidence by DEEPBOM. No general automatic block-replacement feature or trained optimization benefit is claimed."})
    lines = ["# Fusion · inverted residual 개발 실험", "", "미학습 합성 블록이며 임상·태스크 정확도나 속도 실험은 아닙니다.", "",
             "| 변형 | 연산 수 | MACs | 원본과 출력 비교 | 개발 상태 |", "| --- | ---: | ---: | --- | --- |"]
    for record in records:
        lines.append(f"| {record['name']} | {record['operator_count']} | {record['macs']:,} | "
                     f"{'16개 입력 오차 기준 통과' if record['within_tolerance'] else '출력 다름'} | {record['state']} |")
    lines += ["", "Fusion은 Conv–BatchNorm folding입니다. Relu는 별도 연산으로 남습니다. Conv MACs는 유지됩니다.", "",
              "Inverted는 1×1 확장 → ReLU6 → 3×3 depthwise → ReLU6 → 1×1 선형 투영 → residual add 구조입니다.",
              "외부 입출력은 [1,16,32,32]로 같지만 내부 연산과 가중치는 바뀌며 재학습이 필요합니다.", "",
              "실행은 ONNX Runtime이 했습니다. DEEPBOM의 정적 검사와 실행 증거를 구분하여 기록했습니다.", "",
              "각 폴더의 graph.svg에서 구조를 볼 수 있고, weight-analysis.json에서 Weight IR를 확인할 수 있습니다.", ""]
    (bundle / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    tracking_uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    artifact_uri = (store / "artifacts").as_uri()
    experiment = client.get_experiment_by_name("deepbom-fusion-inverted-development")
    if experiment is None:
        experiment_id = client.create_experiment("deepbom-fusion-inverted-development", artifact_location=artifact_uri)
    else:
        require(experiment.artifact_location.rstrip("/") == artifact_uri, "Unexpected artifact store")
        experiment_id = experiment.experiment_id
    receipt = {"tracking_uri": tracking_uri, "study_id": study_id, "runs": {}}
    for record in records:
        tags = {"development.state": record["state"], "development.study_id": study_id,
                "deepbom.artifact_sha256": record["artifact_sha256"], "development.training": "not_run",
                "execution.runtime": "onnxruntime", "execution.version": ort.__version__,
                "execution.inputs_sha256": execution["input_file_sha256"]}
        run_id = client.create_run(experiment_id, tags=tags, run_name=record["name"]).info.run_id
        receipt["runs"][record["name"]] = run_id
        try:
            metrics = {"static.macs": record["macs"], "static.operators": record["operator_count"],
                       "validation.max_absolute_error": record["max_absolute_error"]}
            for key, value in metrics.items():
                client.log_metric(run_id, key, value, synchronous=True)
            files = list((bundle / record["name"]).iterdir()) + [bundle / "execution.json", bundle / "study.json", bundle / "validation-inputs.npz"]
            readback = bundle / "readback" / record["name"]
            readback.mkdir(parents=True)
            for file in files:
                client.log_artifact(run_id, str(file), artifact_path="development")
                restored = Path(client.download_artifacts(run_id, "development/" + file.name, str(readback)))
                require(digest(file) == digest(restored), "MLflow artifact readback mismatch")
            stored = client.get_run(run_id).data
            require(all(stored.tags[key] == value for key, value in tags.items()), "MLflow tag mismatch")
            require(all(stored.metrics[key] == value for key, value in metrics.items()), "MLflow metric mismatch")
            client.set_terminated(run_id, status="FINISHED")
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
    receipt["status"] = "complete"
    write_json(bundle / "completed.json", receipt)
    print(json.dumps({"directory": str(bundle), "records": records, "tracking_uri": tracking_uri}, indent=2))


if __name__ == "__main__":
    main()
