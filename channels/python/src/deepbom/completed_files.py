"""Producer-declared completed files, verified staging, and restart-safe polling.

A filesystem event is only a wake-up signal. Each scan reads atomic completion
manifests and verifies every selected byte before submitting to AnalysisQueue.
"""
import hashlib
from pathlib import Path, PurePosixPath
import tempfile
from ._native.core import atomic_json, content_hash, read_json

SUFFIX = ".deepbom-ready.json"


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def _member(root, name):
    part = PurePosixPath(name)
    if not isinstance(name,str) or not name or "\\" in name or part.is_absolute() or any(p in ("..", ".", "") for p in name.split("/")):
        raise ValueError("Completion member must be a safe relative path")
    path = root.joinpath(*part.parts)
    if path.resolve() != path.absolute(): raise ValueError("Completion members cannot traverse symlinks")
    if not path.is_relative_to(root): raise ValueError("Completion member escapes source root")
    return path


def mark_complete(path, *, manifest, producer):
    """Call only after the producer has finished/closed one artifact.

    The manifest is atomically created beside the artifact. Stable file size is
    never interpreted as completion. Directory producers may write the documented
    explicit member-list contract themselves with an engine artifact-set digest.
    """
    path, manifest = Path(path).resolve(), Path(manifest).absolute()
    if not manifest.name.endswith(SUFFIX) or path.parent != manifest.parent or not path.is_file():
        raise ValueError("Single-file completion manifest must be beside the completed file")
    sha = _hash(path)
    document = {"schema":"deepbom.completed_artifact.v1", "producer":producer,
                "artifact":{"path":path.name, "kind":"file", "sha256":sha},
                "files":[{"path":path.name, "sha256":sha, "bytes":str(path.stat().st_size)}]}
    _validate(document)
    atomic_json(manifest, document)
    return document


def _validate(document):
    import re
    if not isinstance(document,dict) or set(document) != {"schema", "producer", "artifact", "files"} or document["schema"] != "deepbom.completed_artifact.v1":
        raise ValueError("Unsupported completed artifact manifest")
    producer = document["producer"]
    if not isinstance(producer,dict) or set(producer) != {"name","version","code_sha256"} or not all(isinstance(v,str) and 0<len(v)<=256 for v in producer.values()) or not re.fullmatch("[a-f0-9]{64}",producer["code_sha256"]):
        raise ValueError("Completion producer must identify its implementation")
    artifact = document["artifact"]
    if not isinstance(artifact,dict) or set(artifact) != {"path","kind","sha256"} or artifact["kind"] not in ("file","directory") or not re.fullmatch("[a-f0-9]{64}",str(artifact["sha256"])):
        raise ValueError("Invalid completed artifact identity")
    if not isinstance(document["files"],list) or not 1 <= len(document["files"]) <= 100_000: raise ValueError("Invalid member count")
    seen = set()
    for row in document["files"]:
        if not isinstance(row,dict) or set(row) != {"path","sha256","bytes"} or not re.fullmatch("[a-f0-9]{64}",str(row["sha256"])) or not re.fullmatch("0|[1-9][0-9]*",str(row["bytes"])):
            raise ValueError("Invalid completed member contract")
        if row["path"] in seen: raise ValueError("Repeated member path")
        seen.add(row["path"])
    if artifact["kind"] == "file" and (len(seen)!=1 or artifact["path"] not in seen or artifact["sha256"] != document["files"][0]["sha256"]):
        raise ValueError("Single file identity does not match its member")
    if artifact["kind"] == "directory" and any(not row["path"].startswith(artifact["path"]+"/") for row in document["files"]):
        raise ValueError("Directory members must belong to the selected package")


class CompletedFileWatcher:
    def __init__(self, directory, queue, *, scan="auto", max_staged_bytes=1024*1024*1024, max_total_staged_bytes=4*1024*1024*1024):
        self.root = Path(directory).resolve()
        if not self.root.is_dir(): raise ValueError("Watch directory does not exist")
        if queue.directory.is_relative_to(self.root) or self.root.is_relative_to(queue.directory):
            raise ValueError("Keep analysis output and watched source trees separate")
        if scan not in ("auto","structure","integrity","full") or type(max_staged_bytes) is not int or max_staged_bytes<=0:
            raise ValueError("Invalid watch configuration")
        if type(max_total_staged_bytes) is not int or max_total_staged_bytes < max_staged_bytes:
            raise ValueError("Total staging budget must cover one artifact budget")
        self.queue, self.scan, self.maximum, self.total_maximum = queue, scan, max_staged_bytes, max_total_staged_bytes
        self.staging = queue.directory / "inputs"; self.staging.mkdir(exist_ok=True)

    def scan_once(self):
        receipts = []
        # Polling is authoritative, so lost NAS/watchdog notifications are harmless.
        for path in sorted(self.root.rglob("*"+SUFFIX)):
            try:
                if path.is_symlink() or path.resolve()!=path.absolute() or path.stat().st_size>16*1024*1024: raise ValueError("Unsafe or oversized completion manifest")
                document = read_json(path); _validate(document)
                identity = content_hash(document)
                if sum(int(r["bytes"]) for r in document["files"]) > self.maximum: raise ValueError("Staging byte budget exceeded")
                _member(path.parent, document["artifact"]["path"])
                destination = self.staging / identity
                if not destination.exists():
                    used = sum(p.stat().st_size for p in self.staging.rglob('*') if p.is_file())
                    if used + sum(int(r['bytes']) for r in document['files']) > self.total_maximum:
                        raise ValueError("Total staging budget exceeded; archive completed jobs before retrying")
                    with tempfile.TemporaryDirectory(prefix=".staging-",dir=self.staging) as tmp:
                        tmp = Path(tmp)
                        for row in document["files"]:
                            source = _member(path.parent,row["path"])
                            if source.stat().st_size != int(row["bytes"]): raise ValueError("Completed member size mismatch")
                            target = _member(tmp,row["path"]); target.parent.mkdir(parents=True,exist_ok=True)
                            # Bounded streaming copy also verifies the copied bytes.
                            digest, count = hashlib.sha256(), 0
                            with source.open("rb") as reader, target.open("xb") as writer:
                                for block in iter(lambda: reader.read(4*1024*1024),b""):
                                    count += len(block)
                                    if count > int(row["bytes"]): raise ValueError("Completed member grew during staging")
                                    digest.update(block); writer.write(block)
                            if count!=int(row["bytes"]) or digest.hexdigest()!=row["sha256"]: raise ValueError("Completed member hash mismatch")
                        tmp.rename(destination)
                for row in document["files"]:
                    staged = _member(destination,row["path"])
                    if staged.stat().st_size!=int(row["bytes"]) or _hash(staged)!=row["sha256"]: raise ValueError("Staged immutable member changed")
                job = self.queue.submit("audit_file", {"path":str(_member(destination,document["artifact"]["path"])),
                    "sha256":document["artifact"]["sha256"], "scan":self.scan},
                    configuration={"completion_manifest_sha256":identity,"source_producer":document["producer"],"scan":self.scan})
                receipts.append({"manifest":str(path),"job_sha256":job,"status":"submitted"})
            except Exception as error:
                receipts.append({"manifest":str(path),"status":"not_submitted","error":{"code":type(error).__name__,"message":str(error)}})
        return receipts

    def run(self, stop_event, *, poll_seconds=1.0, on_scan=None):
        if not 0.05 <= poll_seconds <= 60: raise ValueError("Poll interval must be 0.05–60 seconds")
        while not stop_event.is_set():
            rows = self.scan_once()
            if on_scan: on_scan(rows)  # Explicit caller callback; no implicit external export.
            stop_event.wait(poll_seconds)
