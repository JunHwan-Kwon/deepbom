"""Write-ahead intent for the existing common Training IR builder."""
from .._native.core import atomic_json, content_hash


def training_input(recorder):
    return {"schema":"deepbom.training_input.v1", "run_id":recorder.run_id,
            "segment_id":recorder.segment_id, "lifecycle":recorder.lifecycle,
            "events":recorder.events, "previous":recorder.previous}


def write_intent(root, payload):
    atomic_json(root / "pending-training.json", {
        "schema":"deepbom.training_commit_intent.v1", "input":payload,
        "input_sha256":content_hash(payload)}, replace=True, compact=True,
        max_bytes=128*1024*1024)


def publish(root, store, result):
    store.put(result["chunk"])
    store.put(result["document"])
    atomic_json(root / "latest.json", {"schema":"deepbom.training_head.v1",
        "training_ir_sha256":result["document"]["training_ir_sha256"]}, replace=True)
    (root / "pending-training.json").unlink(missing_ok=True)
