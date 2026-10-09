"""Explicit imports/exports. Raw models and inputs are never uploaded here."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from .collector import Recorder
from .._native.core import EvidenceStore, read_json, atomic_json, content_hash, engine


def read_run(directory):
    root = Path(directory)
    store = EvidenceStore(root)
    head = read_json(root / "latest.json")
    document = store.get(head["training_ir_sha256"])
    events = []
    objects = {document["training_ir_sha256"]: document}
    expected = 0
    for ref in document["chunks"]:
        chunk = store.get(ref["sha256"])
        objects[ref["sha256"]] = chunk
        if (
            chunk["run_id"] != document["run_id"]
            or chunk["segment_id"] != document["segment_id"]
        ):
            raise ValueError("Foreign training chunk")
        if ref["offset"] != str(expected) or ref["count"] != str(len(chunk["events"])):
            raise ValueError("Chunk count/range mismatch")
        for event in chunk["events"]:
            if (
                event["sequence"] != str(expected)
                or event["id"] != f"{document['segment_id']}:{expected}"
            ):
                raise ValueError("Event order/identity mismatch")
            expected += 1
            events.append(event)
            refs = event["payload"].get("evidence", {})
            if isinstance(refs, dict):
                refs = [refs] if "sha256" in refs else refs.values()
                for item in refs:
                    if isinstance(item, dict) and "sha256" in item:
                        objects[item["sha256"]] = store.get(item["sha256"])
    if str(expected) != document["event_count"]:
        raise ValueError("Incomplete training snapshot")
    return engine(
        "validate_run", {"training": document, "events": events, "objects": objects}
    )


def import_logs(
    source, *, output, logdir=None, tracking_uri=None, run_id=None, project=None
):
    if Path(output).exists():
        raise FileExistsError("Import uses a new destination")
    with Recorder(logdir=output, capture=("metrics",)) as recorder:
        if source == "csv":
            with open(logdir, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not {"step", "name", "value"} <= row.keys():
                        raise ValueError("CSV requires step,name,value columns")
                    recorder.record_metrics(
                        {row["name"]: float(row["value"])},
                        at={"native_step": row["step"], "step_meaning": "unknown"},
                        definition="imported CSV scalar",
                    )
        elif source == "tensorboard":
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            accumulator = EventAccumulator(
                str(logdir), size_guidance={"scalars": 0, "tensors": 0, "histograms": 0}
            ).Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    recorder.record_metrics(
                        {tag: event.value},
                        at={
                            "native_step": str(event.step),
                            "wall_time": event.wall_time,
                            "step_meaning": "logger_defined",
                        },
                        definition="TensorBoard scalar",
                    )
            for tag in accumulator.Tags().get("tensors", []):
                from tensorboard.util.tensor_util import make_ndarray

                for event in accumulator.Tensors(tag):
                    array = make_ndarray(event.tensor_proto)
                    if array.size == 1 and array.dtype.kind in "fiub":
                        recorder.record_metrics(
                            {tag: array.item()},
                            at={
                                "native_step": str(event.step),
                                "wall_time": event.wall_time,
                                "step_meaning": "logger_defined",
                            },
                            definition="TensorBoard tensor scalar",
                        )
                    else:
                        recorder.event(
                            "external_tensor_summary",
                            {"native_step": str(event.step)},
                            {
                                "tag": tag,
                                "shape": list(array.shape),
                                "dtype": str(array.dtype),
                                "status": "not_reinterpreted_as_model_tensor",
                            },
                        )
            for tag in accumulator.Tags().get("histograms", []):
                for event in accumulator.Histograms(tag):
                    h = event.histogram_value
                    recorder.event(
                        "external_histogram",
                        {"native_step": str(event.step)},
                        {
                            "tag": tag,
                            "bucket_limits": list(h.bucket_limit),
                            "bucket_counts": list(h.bucket),
                            "count": h.num,
                            "minimum": h.min,
                            "maximum": h.max,
                            "scope": "logger_summary_not_raw_weight_values",
                        },
                    )
        elif source == "mlflow":
            from mlflow.tracking import MlflowClient

            if not tracking_uri or not run_id:
                raise ValueError(
                    "MLflow import requires explicit tracking_uri and run_id"
                )
            client = MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)
            for key in run.data.metrics:
                for m in client.get_metric_history(run_id, key):
                    recorder.record_metrics(
                        {key: m.value},
                        at={
                            "native_step": str(m.step),
                            "timestamp_ms": str(m.timestamp),
                            "step_meaning": "logger_defined",
                        },
                        definition="MLflow metric history",
                    )
        elif source == "wandb":
            import wandb

            if not project or not run_id:
                raise ValueError("W&B import requires entity/project and run_id")
            run = wandb.Api().run(f"{project}/{run_id}")
            for row in run.scan_history():
                values = {
                    k: v
                    for k, v in row.items()
                    if not k.startswith("_") and isinstance(v, (int, float))
                }
                recorder.record_metrics(
                    values,
                    at={
                        "native_step": str(row.get("_step")),
                        "step_meaning": "logger_defined",
                    },
                    definition="W&B run history",
                )
        else:
            raise ValueError("Sources: csv, tensorboard, mlflow, wandb")
        recorder.event(
            "import_complete",
            {},
            {
                "source": source,
                "training_completion_inferred": False,
                "binding": "run_only_unbound",
            },
        )
    return Path(output)


def export(
    directory,
    *,
    destination,
    tracking_uri=None,
    run_id=None,
    logdir=None,
    wandb_run=None,
):
    bundle = read_run(directory)
    root = Path(directory)
    digest = bundle["training"]["training_ir_sha256"]
    if logdir is not None:
        logdir = str(Path(logdir).expanduser().resolve())
    if destination == "wandb":
        if wandb_run is None:
            raise ValueError("Pass the explicitly selected W&B run")
        run_id = "/".join(
            str(getattr(wandb_run, k, "")) for k in ("entity", "project", "id")
        )
        if not getattr(wandb_run, "id", None):
            raise ValueError("W&B run has no stable identity")
    key = content_hash(
        {
            "destination": destination,
            "tracking_uri": tracking_uri,
            "run_id": run_id,
            "logdir": str(logdir) if logdir else None,
            "digest": digest,
            "projection": "2.0.0",
        }
    )
    receipt = root / "exports" / (key + ".json")
    if receipt.exists():
        return read_json(receipt)
    # A frozen staging directory contains only selected evidence, never state.json/model bytes.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="deepbom-export-") as tmp:
        stage = Path(tmp)
        for sha, doc in bundle["objects"].items():
            atomic_json(stage / (sha + ".json"), doc)
        atomic_json(stage / "index.json", {"training_ir_sha256": digest})
        if destination == "mlflow":
            from mlflow.tracking import MlflowClient

            if not tracking_uri or not run_id:
                raise ValueError(
                    "MLflow export requires explicit tracking_uri and run_id"
                )
            client = MlflowClient(tracking_uri=tracking_uri)
            client.log_artifacts(run_id, str(stage), artifact_path="deepbom/" + digest)
        elif destination == "tensorboard":
            if not logdir:
                raise ValueError("TensorBoard export requires logdir")
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
            from tensorboard.compat.proto.tensor_pb2 import TensorProto
            from tensorboard.compat.proto.types_pb2 import DT_DOUBLE
            import time
            import shutil

            target = Path(logdir) / "deepbom-evidence" / digest
            target.mkdir(parents=True, exist_ok=True)
            for f in stage.iterdir():
                shutil.copy2(f, target / f.name)
            # Each immutable head is one TensorBoard run. Appending whole heads
            # to one directory duplicates older observations and mixes runs.
            writer = EventFileWriter(str(Path(logdir) / "deepbom" / digest))
            try:
                for event in bundle["events"]:
                    for name, value in event["payload"].get("values", {}).items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            writer.add_event(
                                Event(
                                    wall_time=time.time(),
                                    step=int(event["sequence"]),
                                    summary=Summary(
                                        value=[
                                            Summary.Value(
                                                tag=name,
                                                tensor=TensorProto(
                                                    dtype=DT_DOUBLE,
                                                    double_val=[float(value)],
                                                ),
                                                metadata=SummaryMetadata(
                                                    plugin_data=SummaryMetadata.PluginData(
                                                        plugin_name="scalars"
                                                    )
                                                ),
                                            )
                                        ]
                                    ),
                                )
                            )
                writer.flush()
            finally:
                writer.close()
        elif destination == "wandb":
            import wandb

            if wandb_run is None:
                raise ValueError("Pass the explicitly selected W&B run")
            artifact = wandb.Artifact(
                "deepbom-" + digest[:16],
                type="model-evidence",
                metadata={"training_ir_sha256": digest},
            )
            artifact.add_dir(str(stage))
            wandb_run.log_artifact(artifact)
        else:
            raise ValueError("Destinations: mlflow, tensorboard, wandb")
    result = {
        "schema": "deepbom.export_receipt.v1",
        "key": key,
        "source_digest": digest,
        "destination": destination,
        "status": "submitted",
        "remote_exactly_once": False,
        "projection": "2.0.0",
        "event_logdir": (
            str(Path(logdir) / "deepbom" / digest)
            if destination == "tensorboard"
            else None
        ),
    }
    atomic_json(receipt, result)
    return result
