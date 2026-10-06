"""Record static evidence in a local MLflow run and verify retrieval."""
import argparse
import json
import os
from pathlib import Path

import deepbom
import mlflow
from mlflow.tracking import MlflowClient
from after_export import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("example_directory", type=Path)
    args = parser.parse_args()
    directory = args.example_directory.resolve()
    manifest = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    model = manifest["candidate"]
    artifact = directory / model["file"]
    envelope = deepbom.audit(artifact, expected_sha256=model["sha256"], scan="full")
    summary = deepbom.inspect(artifact, expected_sha256=model["sha256"], scan="full")
    require(summary["artifact"]["sha256"] == envelope["identity"]["sha256"] == model["sha256"], "Artifact identity changed")
    require(summary["evidence_envelope_sha256"] == envelope["envelope_sha256"], "Summary/envelope binding mismatch")
    require(not os.environ.get("MLFLOW_RUN_ID"), "Unset MLFLOW_RUN_ID so this example creates a new isolated run")

    # Deliberately local; do not silently inherit a user's production tracking URI.
    tracking_uri = f"sqlite:///{(directory / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "deepbom-sdk-example"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=(directory / "mlflow-artifacts").as_uri())
    else:
        require(experiment.artifact_location.rstrip("/") == (directory / "mlflow-artifacts").as_uri(), "Existing experiment is bound to a different artifact store")
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name="candidate-static-evidence") as run:
        mlflow.set_tags({"deepbom.artifact_sha256": model["sha256"],
                         "deepbom.envelope_sha256": envelope["envelope_sha256"],
                         "deepbom.engine_version": envelope["provenance"]["version"],
                         "deepbom.evidence_kind": "static_artifact_inspection"})
        mlflow.log_dict(envelope, "deepbom/evidence.json")
        mlflow.log_dict(summary, "deepbom/summary.json")
        metrics = {"deepbom.findings.artifact_defects": summary["verdict"]["artifact_defect_count"],
                   "deepbom.findings.cautions": summary["verdict"]["caution_count"],
                   "deepbom.findings.evidence_gaps": summary["verdict"]["evidence_needed_count"],
                   **{f"deepbom.coverage.{key}": summary["coverage"][key] for key in ["declared", "assessed", "partial", "unavailable"]}}
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
    restored = Path(client.download_artifacts(run_id, "deepbom/evidence.json"))
    require(json.loads(restored.read_text(encoding="utf-8")) == envelope, "Logged envelope differs from the source")
    restored_summary = Path(client.download_artifacts(run_id, "deepbom/summary.json"))
    require(json.loads(restored_summary.read_text(encoding="utf-8")) == summary, "Logged summary differs from the source")
    stored = client.get_run(run_id).data
    require(stored.tags["deepbom.artifact_sha256"] == model["sha256"], "Logged identity differs from the source")
    require(all(stored.metrics[key] == value for key, value in metrics.items()), "Logged metrics differ from the source")
    (directory / "mlflow-run.json").write_text(json.dumps({"run_id": run_id, "tracking_uri": tracking_uri}, indent=2) + "\n", encoding="utf-8")
    print(f"Retrieved and checked evidence from local MLflow run {run_id}")
    print("No model weights were logged. JSON evidence can contain model names and structure.")


if __name__ == "__main__":
    main()
