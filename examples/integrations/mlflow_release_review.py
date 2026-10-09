"""Record a completed release_review.mjs bundle as three local MLflow runs.

This consumes the existing example policy; it neither recalculates model facts
nor registers/deploys a model. Run after after_export.py and release_review.mjs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import deepbom
from mlflow.tracking import MlflowClient
from after_export import require, write_json


def load_bundle(bundle):
    def document(name):
        return json.loads((bundle / name).read_text(encoding="utf-8"))

    completed = document("completed.json")
    require(completed == {"status": "complete", "example_policy": "static-interface-review-v1"},
            "A completed static-interface-review-v1 bundle is required")
    summaries = {name: document(f"{name}.summary.json")
                 for name in ["baseline", "candidate", "incompatible"]}
    baseline_hash = summaries["baseline"]["artifact"]["sha256"]
    decisions = {}
    comparisons = {}
    for name in ["candidate", "incompatible"]:
        decision = document(f"{name}.decision.json")
        expected_files = {"baseline.onnx", f"{name}.onnx", "baseline.summary.json",
                          "baseline.contract.json", "baseline.verification.json",
                          f"{name}.summary.json", f"{name}.comparison.json"}
        entries = decision["evidence_files"]
        require(len(entries) == len(expected_files) and
                {entry["file"] for entry in entries} == expected_files,
                f"{name}: unexpected evidence file inventory")
        for entry in entries:
            require(hashlib.sha256((bundle / entry["file"]).read_bytes()).hexdigest() == entry["sha256"],
                    f"{name}: changed evidence file {entry['file']}")
        comparison = document(f"{name}.comparison.json")
        require(decision["example_policy"] == completed["example_policy"], "Unknown example policy")
        require(decision["disposition"] in ["reject", "hold_for_external_evaluation"],
                "This example never authorizes deployment")
        require(decision["external_evaluation"] == "not_provided" and
                decision["actual_deployment"] == "not_performed", "Unexpected example decision scope")
        require(comparison["schema"] == "deepbom.semantic_artifact_diff.v1", "Unexpected diff schema")
        require(decision["baseline_sha256"] == comparison["baseline"]["sha256"] == baseline_hash,
                "Baseline identity mismatch")
        require(decision["candidate_sha256"] == comparison["candidate"]["sha256"] ==
                summaries[name]["artifact"]["sha256"], "Candidate identity mismatch")
        decisions[name], comparisons[name] = decision, comparison

    records = {}
    for name, summary in summaries.items():
        require(summary["schema"] == "deepbom.review_summary.v1", "Unexpected summary schema")
        envelope = deepbom.audit(bundle / f"{name}.onnx",
                                 expected_sha256=summary["artifact"]["sha256"], scan="full")
        require(envelope["identity"]["sha256"] == summary["artifact"]["sha256"], "Artifact identity mismatch")
        require(envelope["envelope_sha256"] == summary["evidence_envelope_sha256"],
                "Summary/envelope mismatch; regenerate the review using the same engine version")
        records[name] = {"summary.json": summary, "evidence.json": envelope}
        if name in decisions:
            records[name].update({"decision.json": decisions[name], "comparison.json": comparisons[name]})
        else:
            records[name].update({"contract.json": document("baseline.contract.json"),
                                  "verification.json": document("baseline.verification.json")})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_bundle", type=Path)
    parser.add_argument("--store", type=Path, default=Path(".local-validation/mlflow-release-scenario"))
    args = parser.parse_args()
    # Validate all subjects before creating any tracking runs. Explicit client
    # URIs avoid inheriting MLFLOW_TRACKING_URI, an active run or a production store.
    records = load_bundle(args.review_bundle.resolve())
    store = args.store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    artifact_uri = (store / "artifacts").as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "deepbom-release-review"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_uri)
    else:
        require(experiment.artifact_location.rstrip("/") == artifact_uri, "Unexpected artifact store")
        experiment_id = experiment.experiment_id
    scenario_id = uuid4().hex
    receipt_directory = store / "scenarios" / scenario_id
    receipt_directory.mkdir(parents=True)
    receipt = {"scenario_id": scenario_id, "tracking_uri": tracking_uri,
               "experiment_id": experiment_id, "runs": {}, "status": "running"}
    write_json(receipt_directory / "progress.json", receipt)
    baseline_run_id = None
    for name, documents in records.items():
        summary, envelope = documents["summary.json"], documents["evidence.json"]
        disposition = documents.get("decision.json", {}).get("disposition", "reference_only")
        tags = {"deepbom.scenario_id": scenario_id, "deepbom.role": name,
                "deepbom.artifact_sha256": summary["artifact"]["sha256"],
                "deepbom.envelope_sha256": envelope["envelope_sha256"],
                "deepbom.engine_version": envelope["provenance"]["version"],
                "deepbom.evidence_kind": "static_artifact_inspection",
                "review.disposition": disposition, "review.external_evaluation": "not_provided",
                "review.example_policy": "static-interface-review-v1"}
        if baseline_run_id:
            tags["review.baseline_run_id"] = baseline_run_id
        run_id = client.create_run(experiment_id, tags=tags, run_name=name).info.run_id
        receipt["runs"][name] = {"run_id": run_id, "sha256": summary["artifact"]["sha256"],
                                 "disposition": disposition}
        write_json(receipt_directory / "progress.json", receipt)
        try:
            metrics = {"artifact.bytes": summary["artifact"]["byte_length"],
                       "static.operators": summary["graph"]["operator_count"],
                       **{f"findings.{key}": summary["verdict"][key] for key in
                          ["artifact_defect_count", "caution_count", "evidence_needed_count"]},
                       **{f"coverage.{key}": summary["coverage"][key] for key in
                          ["declared", "assessed", "partial", "unavailable"]}}
            macs = summary["graph"]["total_macs"]
            # An unknown count must not become zero in a dashboard. MLflow
            # metrics are floating point; the canonical JSON retains exact values.
            if type(macs) is int and 0 <= macs <= 2**53:
                metrics["static.macs"] = macs
            for key, value in metrics.items():
                client.log_metric(run_id, key, value, synchronous=True)
            download_directory = receipt_directory / name
            download_directory.mkdir()
            for filename, original in documents.items():
                artifact_path = f"deepbom/{filename}"
                client.log_dict(run_id, original, artifact_path)
                restored = Path(client.download_artifacts(run_id, artifact_path, str(download_directory)))
                require(json.loads(restored.read_text(encoding="utf-8")) == original,
                        f"{name}: MLflow changed {filename}")
            stored = client.get_run(run_id).data
            require(all(stored.tags[key] == value for key, value in tags.items()), "Stored tag mismatch")
            require(all(stored.metrics[key] == value for key, value in metrics.items()), "Stored metric mismatch")
            client.set_terminated(run_id, status="FINISHED")
        except BaseException:
            client.set_terminated(run_id, status="FAILED")
            raise
        if name == "baseline":
            baseline_run_id = run_id
        print(f"{name}: {disposition}; MLflow run {run_id}")
    receipt["status"] = "complete"
    write_json(receipt_directory / "completed.json", receipt)
    print(f"Verified receipt: {receipt_directory / 'completed.json'}")
    print(f"UI: mlflow ui --backend-store-uri {tracking_uri} --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
