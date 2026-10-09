"""Run a checkout-based Weight IR/redesign study and record it in local MLflow."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from uuid import uuid4

from mlflow.tracking import MlflowClient
from after_export import require, write_json


def checked_artifacts(bundle):
    manifest = json.loads((bundle / "completed.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Development study is incomplete")
    files = {}
    for entry in manifest["artifacts"]:
        relative = PurePosixPath(entry["file"])
        require(not relative.is_absolute() and ".." not in relative.parts and
                "\\" not in entry["file"] and entry["file"] not in files, "Invalid artifact path")
        source = bundle / relative
        require(source.resolve().is_relative_to(bundle.resolve()), "Artifact escapes study directory")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"],
                f"Changed study artifact: {relative}")
        files[entry["file"]] = entry["sha256"]
    require("study.json" in files and "source/weight-ir.json" in files, "Missing study evidence")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--model", type=Path, default=root / "web/samples/mobilenet_v1_025_224_float.tflite")
    parser.add_argument("--target", default="android_mid_a55")
    parser.add_argument("--store", type=Path, default=root / ".local-validation/mlflow-development")
    args = parser.parse_args()
    store = args.store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    scenario_id = uuid4().hex
    bundle = store / f"study-{scenario_id}"
    subprocess.run(["node", str(Path(__file__).with_name("development_study.mjs")),
                    str(args.model.resolve()), str(bundle), args.target], check=True, timeout=300)
    files = checked_artifacts(bundle)
    study = json.loads((bundle / "study.json").read_text(encoding="utf-8"))
    tracking_uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    artifact_uri = (store / "artifacts").as_uri()
    # Never inherit a production tracking URI or an active training run.
    client = MlflowClient(tracking_uri=tracking_uri)
    name = "deepbom-weight-guided-development"
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(name, artifact_location=artifact_uri)
    else:
        require(experiment.artifact_location.rstrip("/") == artifact_uri, "Unexpected artifact store")
        experiment_id = experiment.experiment_id
    receipt = {"scenario_id": scenario_id, "tracking_uri": tracking_uri,
               "experiment_id": experiment_id, "study_directory": str(bundle), "runs": {}}
    baseline_id = None
    for index, record in enumerate(study["runs"]):
        tags = {"development.scenario_id": scenario_id, "development.state": record["state"],
                "development.evidence_kind": record["kind"],
                "development.source_sha256": study["source"]["sha256"],
                "development.request_sha256": record["request_sha256"],
                "development.projection_sha256": record["projection_sha256"],
                "development.weight_ir_sha256": study["source"]["weight_ir_sha256"],
                "development.target": study["source"]["target"],
                "development.engine_version": study["engine_version"],
                "development.training": "not_run", "development.measured_quality": "not_provided",
                "development.measured_latency": "not_provided"}
        if baseline_id:
            tags["development.baseline_run_id"] = baseline_id
        run_id = client.create_run(experiment_id, tags=tags, run_name=record["name"]).info.run_id
        receipt["runs"][record["name"]] = run_id
        write_json(bundle / "mlflow-progress.json", receipt)
        try:
            metrics = {key: value for key, value in record["metrics"].items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and abs(value) <= 2**53}
            for key, value in metrics.items():
                client.log_metric(run_id, key, value, synchronous=True)
            client.log_param(run_id, "selected_block", study["selected"]["block_id"])
            client.log_param(run_id, "block_output_channels", record["proposed_output_channels"])
            # Baseline owns the full report and source plots. Each proposal also
            # receives its request, projection and weight-free structure scaffold.
            upload = files if index == 0 else {
                key: value for key, value in files.items()
                if key.startswith(record["name"] + "/") or key in ["study.json", "source/proposal-basis.json"]}
            restored_directory = bundle / "mlflow-readback" / record["name"]
            restored_directory.mkdir(parents=True)
            for filename, digest in upload.items():
                parent = str(PurePosixPath(filename).parent)
                artifact_path = "development" if parent == "." else "development/" + parent
                client.log_artifact(run_id, str(bundle / filename), artifact_path=artifact_path)
                remote_file = "development/" + filename
                restored = Path(client.download_artifacts(run_id, remote_file, str(restored_directory)))
                require(hashlib.sha256(restored.read_bytes()).hexdigest() == digest,
                        f"MLflow artifact mismatch: {filename}")
            stored = client.get_run(run_id).data
            require(all(stored.tags[key] == value for key, value in tags.items()), "MLflow tags changed")
            require(all(stored.metrics[key] == value for key, value in metrics.items()), "MLflow metrics changed")
            require(stored.params["block_output_channels"] == str(record["proposed_output_channels"]),
                    "MLflow experiment parameter changed")
            client.set_terminated(run_id, status="FINISHED")
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        baseline_id = baseline_id or run_id
        print(f"{record['name']}: {record['state']} · {run_id}")
    receipt["status"] = "complete"
    write_json(bundle / "mlflow-completed.json", receipt)
    print(f"Report: {bundle / 'index.html'}")
    print(f"MLflow: mlflow ui --backend-store-uri {tracking_uri} --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
