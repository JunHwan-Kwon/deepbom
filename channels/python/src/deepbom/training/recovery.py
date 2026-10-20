"""Recover persisted observations, never resume training or infer missing events."""
from pathlib import Path
import time
from .._native.core import EvidenceStore, atomic_json, content_hash, engine, read_json
from .._ownership import WriterOwnership
from .._process import RunControl, execution_control
from ._persistence import publish, write_intent


def _read(path, maximum=128*1024*1024):
    if path.stat().st_size > maximum: raise ValueError("Recovery document exceeds its byte budget")
    return read_json(path)


def _validate_result(store, result):
    """Use the same semantic validator as live readers before publishing a head."""
    document, chunk = result["document"], result["chunk"]
    objects = {document["training_ir_sha256"]:document, chunk["chunk_sha256"]:chunk}
    events = []
    for ref in document["chunks"]:
        item = objects.get(ref["sha256"]) or store.get(ref["sha256"])
        objects[ref["sha256"]] = item
        events.extend(item["events"])
    for event in events:
        refs = event["payload"].get("evidence", {})
        if isinstance(refs, dict):
            refs = [refs] if "sha256" in refs else refs.values()
            for ref in refs:
                if isinstance(ref, dict) and "sha256" in ref:
                    objects[ref["sha256"]] = store.get(ref["sha256"])
    return engine("validate_run", {"training":document, "events":events, "objects":objects})


def _recover_intent(root, store):
    path = root / "pending-training.json"
    if not path.exists(): return False
    intent = _read(path)
    if set(intent) != {"schema","input","input_sha256"} or intent["schema"] != "deepbom.training_commit_intent.v1" or content_hash(intent["input"]) != intent["input_sha256"]:
        raise ValueError("Training recovery intent identity mismatch")
    payload = intent["input"]
    result = engine("training", payload)
    _validate_result(store, result)
    head = _read(root / "latest.json")["training_ir_sha256"] if (root / "latest.json").exists() else None
    previous = (payload.get("previous") or {}).get("training_ir_sha256")
    if head not in (previous, result["document"]["training_ir_sha256"]):
        raise ValueError("Training head does not match the recovery predecessor or completed intent")
    publish(root, store, result)
    return True


def _recover_unsealed(root, store):
    from .integrations import read_run
    path = root / "unsealed-events.json"
    if not path.exists(): return False
    journal = _read(path)
    if journal.get("schema") != "deepbom.unsealed_training_journal.v1":
        raise ValueError("Unsupported unsealed training journal")
    bundle = read_run(root)
    current = bundle["training"]
    if any(journal.get(key) != current[key] for key in ("run_id","segment_id")):
        raise ValueError("Foreign unsealed training journal")
    previous = journal.get("previous")
    payload = {"schema":"deepbom.training_input.v1", "run_id":current["run_id"],
               "segment_id":current["segment_id"], "previous":previous,
               "lifecycle":current["lifecycle"], "events":journal["events"]}
    # Validate the complete journal, including its original sequence and refs,
    # before removing an already committed prefix.
    original = engine("training", payload)
    _validate_result(store, original)
    start = int((previous or {}).get("event_count", "0"))
    if previous and bundle["events"][:start] != _validate_result(store, {"document":previous, "chunk":store.get(previous["chunks"][-1]["sha256"])})["events"]:
        raise ValueError("Unsealed journal predecessor differs from durable history")
    overlap = min(len(journal["events"]), max(0, len(bundle["events"])-start))
    if start > len(bundle["events"]) or journal["events"][:overlap] != bundle["events"][start:start+overlap]:
        raise ValueError("Unsealed events conflict with durable history")
    remaining = journal["events"][overlap:]
    if remaining:
        payload.update(previous=current, events=remaining)
        write_intent(root, payload)
        _recover_intent(root, store)
    # Keep the original recovery input addressable instead of deleting evidence.
    atomic_json(root / "recovery-inputs" / (content_hash(journal)+".json"), journal, compact=True)
    path.unlink()
    return True


def _recover_bindings(root, store, deadline):
    from .integrations import read_analysis_view, read_run
    from .managed import managed_producer
    from ..jobs import AnalysisQueue
    path = root / "analysis-bindings.json"
    if not path.exists(): return None
    view = read_analysis_view(root)  # common identity, counts and evidence checks
    index = view["index"]
    pending = [r for r in index["bindings"] if r["status"] in ("pending", "commit_pending")]
    if not pending: return index["counts"]
    # No live model is reconstructed. Only the immutable requests already
    # captured at a known boundary may be re-executed, with the same producer.
    queue = AnalysisQueue(root / "analysis-jobs", producer=managed_producer(),
                          max_request_bytes=128*1024*1024, close_timeout=max(.001,deadline-time.monotonic()))
    try:
        while time.monotonic() < deadline:
            queue.recover()
            with queue.lock: active = list(queue.futures)
            for job in active:
                queue.wait(job, timeout=max(.001,deadline-time.monotonic()))
            remaining = False
            events = read_run(root)["events"]
            for row in pending:
                if row["status"] not in ("pending", "commit_pending"): continue
                status = queue.status(row["job_sha256"])["execution_status"]
                if status == "completed":
                    bundle = queue.result(row["job_sha256"])
                    refs = {k:store.put(v) for k,v in bundle.items() if v is not None}
                    matches = [e for e in events if e.get("at",{}).get("analysis_binding_sha256") == row["binding_sha256"]]
                    if len(matches)>1 or matches and matches[0]["payload"].get("evidence") != refs:
                        raise ValueError("Recovered binding conflicts with a Training event")
                    row.update(status="completed", evidence=refs, completed_after_capture=True, late=not bool(matches))
                elif status in ("failed", "cancelled"):
                    row["status"] = status
                else:
                    remaining = True
            index["counts"]["completed"] = sum(r["status"]=="completed" for r in index["bindings"])
            index["counts"]["failed"] = sum(r["status"] in ("failed","cancelled") for r in index["bindings"])
            objects = {}
            for row in index["bindings"]:
                for ref in row.get("evidence",{}).values(): objects[ref["sha256"]] = store.get(ref["sha256"])
            engine("validate_training_analysis", {"bundle":read_run(root), "index":index, "objects":objects})
            atomic_json(path, index, replace=True, compact=True)
            if not remaining: break
    finally:
        queue.close(deadline=deadline)
    return index["counts"]


def recover(directory, *, timeout=30, analyze_pending=True):
    """Seal durable observations and optionally finish captured CPU analyses.

    The source run remains an immutable chain. Buffered observations lost to a
    process crash are unknown, and no optimizer/model execution is resumed.
    """
    import math
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError("Recovery timeout must be finite and positive")
    root = Path(directory).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    deadline = time.monotonic()+timeout
    from .integrations import read_run
    with WriterOwnership(root / ".writer.lock", purpose="training_writer"), execution_control(RunControl(deadline=deadline)):
        store = EvidenceStore(root)
        restored = _recover_intent(root, store)
        restored = _recover_unsealed(root, store) or restored
        bundle = read_run(root)
        current = bundle["training"]
        if current["lifecycle"] == "running":
            sequence = current["event_count"]
            payload = {"schema":"deepbom.training_input.v1", "run_id":current["run_id"],
                       "segment_id":current["segment_id"], "previous":current, "lifecycle":"interrupted",
                       "events":[{"id":f"{current['segment_id']}:{sequence}", "sequence":sequence,
                                  "kind":"collector_recovered", "at":{}, "payload":{
                                      "origin":"explicit_evidence_recovery", "training_completion_inferred":False,
                                      "lost_buffered_observations":"unknown", "training_resumed":False}}]}
            write_intent(root,payload)
            _recover_intent(root,store)
            restored = True
        counts = _recover_bindings(root,store,deadline) if analyze_pending else None
        document = read_run(root)["training"]
        receipt = {"schema":"deepbom.training_recovery_receipt.v1", "run_id":document["run_id"],
                   "training_ir_sha256":document["training_ir_sha256"], "training_lifecycle":document["lifecycle"],
                   "persisted_observations_recovered":restored, "analysis_counts":counts,
                   "boundary":"Only persisted observations were recovered. Missing buffers and training completion remain unknown; training was not resumed."}
        atomic_json(root / "recovery.json",receipt,replace=True)
        return receipt
