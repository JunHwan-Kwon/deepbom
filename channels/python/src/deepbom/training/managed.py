"""Monitor-owned immutable snapshot jobs; only the caller thread writes history."""
import hashlib
from pathlib import Path
import time
import sys
from .._native.adapters import snapshot_input, state_byte_count, require_supported_execution
from .._native.core import atomic_json, content_hash, json_value, engine
from ..jobs import AnalysisQueue


def managed_producer():
    identity=engine("numerical_identity",{})
    return {"name":"deepbom-managed-snapshot", "version":"1",
            "code_sha256":content_hash({"adapter":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"engine":identity})}


class ManagedSnapshots:
    def __init__(self, recorder, *, max_pending=2, max_total_bytes=None, close_timeout=30):
        require_supported_execution(recorder.model)
        if sys.platform!="linux":
            raise ValueError("Managed snapshot analysis is currently qualified on Linux CPU only")
        from .._native.adapters import framework_of
        if framework_of(recorder.model)=="pytorch":
            import torch
            if any(v.device.type!="cpu" for v in [*recorder.model.parameters(),*recorder.model.buffers()]):
                raise ValueError("Managed CPU snapshots do not yet qualify asynchronous GPU copying")
            if torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
                raise ValueError("Managed snapshots require a single unsharded process")
        else:
            import tensorflow as tf
            if tf.distribute.get_strategy().num_replicas_in_sync!=1 or any("GPU" in str(getattr(getattr(v,"value",v),"device","")).upper() for v in recorder.model.weights):
                raise ValueError("Managed TensorFlow snapshots require single-process CPU state")
        self.recorder = recorder
        self.sequence = 0
        self.bindings = []
        self.jobs = {}
        self.counts = {key:0 for key in ("requested", "admitted", "completed", "skipped", "failed")}
        self.copy_seconds = 0.0
        self.copy_bytes = 0
        # Native transport includes base64 and graph metadata. Reserve the
        # configured maximum before capture, not only the tensor payload size.
        maximum = min(128*1024*1024, max(1024*1024, recorder.max_state_bytes*2 + 1024*1024))
        self.queue = AnalysisQueue(recorder.logdir / "analysis-jobs", producer=managed_producer(),
                                   max_pending=max_pending, max_request_bytes=maximum,
                                   max_total_bytes=max_total_bytes if max_total_bytes is not None else max_pending*(maximum+recorder.max_state_bytes),
                                   close_timeout=close_timeout)
        self.index_path = recorder.logdir / "analysis-bindings.json"

    def capture(self, *, at=None, advanced=False):
        self.recorder._check(running=True)
        self.drain()
        self.counts["requested"] += 1
        if len(self.bindings)+len(self.jobs)>=10000:
            self.counts["skipped"]+=1
            self.write_index()
            return None
        size = state_byte_count(self.recorder.model)
        coordinate = json_value(at or {})
        binding = {"run_id":self.recorder.run_id, "segment_id":self.recorder.segment_id,
                   "capture_sequence":str(self.sequence), "at":coordinate}
        self.sequence += 1
        def copy():
            started=time.monotonic()
            raw=snapshot_input(self.recorder.model, max_values=self.recorder.max_values,
                               advanced=advanced, max_state_bytes=self.recorder.max_state_bytes)
            self.copy_seconds += time.monotonic()-started
            self.copy_bytes += size
            # Framework version wrappers (e.g. TorchVersion) are metadata,
            # not tensor/live-object transport. Normalize to JSON primitives.
            return json_value(raw)
        try:
            job=self.queue.submit_capture("snapshot", copy, configuration={"analysis":"snapshot"}, copy_bytes=size)
        except BufferError:
            self.counts["skipped"] += 1
            # Coalesced counters bound diagnostics even in a long saturated run.
            self.write_index()
            return None
        except Exception:
            self.counts["skipped"] += 1
            self.write_index()
            raise
        binding["job_sha256"]=job
        binding["binding_sha256"]=content_hash(binding)
        binding["status"]="pending"
        self.jobs[binding["binding_sha256"]]=binding
        self.counts["admitted"] += 1
        self.write_index()
        return None  # automatic boundary never masquerades as a Snapshot IR SHA

    def drain(self):
        for key,binding in list(self.jobs.items()):
            state=self.queue.poll(binding["job_sha256"])["execution_status"]
            if state not in ("completed","failed","cancelled"):
                continue
            row=dict(binding)
            if state=="completed":
                bundle=self.queue.result(binding["job_sha256"])
                refs={k:self.recorder.store.put(v) for k,v in bundle.items() if v is not None}
                row.update(status="commit_pending", evidence=refs,
                           completed_after_capture=True, late=self.recorder.lifecycle!="running")
                # Write intent before the single owner commits a Training event.
                self.jobs[key]=row
                self.write_index()
                if not row["late"]:
                    coordinate={**binding["at"],"analysis_binding_sha256":key}
                    already=self._committed(key)
                    if not already:
                        self.recorder._commit_snapshot(bundle, at=coordinate)
                    # A completed binding acknowledges a durable Training head,
                    # not merely an event still buffered in Python memory.
                    self.recorder.flush()
                row["status"]="completed"
                self.counts["completed"]+=1
            else:
                row["status"]=state
                self.counts["failed"]+=1
            self.bindings.append(row)
            self.jobs.pop(key)
            self.write_index()

    def _committed(self, key):
        if any(e.get("at",{}).get("analysis_binding_sha256")==key for e in self.recorder.events):
            return True
        for ref in (self.recorder.previous or {}).get("chunks",[]):
            chunk=self.recorder.store.get(ref["sha256"])
            if any(e.get("at",{}).get("analysis_binding_sha256")==key for e in chunk["events"]):
                return True
        return False

    def write_index(self):
        body={"schema":"deepbom.analysis_bindings.v1", "run_id":self.recorder.run_id,
              "segment_id":self.recorder.segment_id,"bindings":[*self.bindings,*self.jobs.values()],
              "counts":self.counts,"copy_bytes":self.copy_bytes,"copy_seconds":self.copy_seconds,
              "boundary":"Supporting analysis completions; does not rewrite Training IR lifecycle or prove update effects."}
        atomic_json(self.index_path,body,replace=True,compact=True)

    def close(self, deadline):
        receipt=self.queue.close(deadline=deadline)
        self.drain()
        self.write_index()
        return receipt
