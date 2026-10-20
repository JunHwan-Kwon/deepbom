"""Transport and content-addressed storage; analysis uses the common engine."""

from __future__ import annotations
import base64
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

_active_engine = ContextVar("deepbom_native_engine", default=None)


def bind_engine(request):
    """Bind an owner-controlled session to this context; never share globally."""
    return _active_engine.set((os.getpid(), threading.get_ident(), request))


def reset_engine(token):
    _active_engine.reset(token)



def engine(operation: str, payload: dict, *, timeout_seconds=300) -> dict:
    from ..api import _invoke_json
    from .._process import remaining_seconds

    bound = _active_engine.get()
    if bound is not None:
        pid, thread, request = bound
        if pid == os.getpid() and thread == threading.get_ident():
            return _json_binary64(request(operation, payload, remaining_seconds(timeout_seconds)))
    # A copied context or forked data loader must not use its parent's session.
    with tempfile.TemporaryDirectory(prefix="deepbom-native-") as tmp:
        request = Path(tmp) / "request.json"
        atomic_json(request, {"operation": operation, "payload": payload},
                    compact=True, max_bytes=128 * 1024 * 1024)
        return _json_binary64(
            _invoke_json(
                ["evidence-native", "--request", str(request)], remaining_seconds(timeout_seconds), 128 * 1024 * 1024
            )
        )


def _json_binary64(value):
    # JSON numbers from the JS owner have IEEE-754 semantics. Integral-valued
    # histogram boundaries (e.g. 2**64) may parse as Python ints; preserve the
    # original binary64 value. Exact quantities in this contract use strings.
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) > 2**53 - 1
    ):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_binary64(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_binary64(x) for k, x in value.items()}
    return value


def content_hash(value: Any) -> str:
    # Host admission only; canonical bytes and SHA-256 belong to Rust/WASM.
    def admit(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Canonical evidence keys must be strings")
            return {key: admit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [admit(child) for child in item]
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Canonical evidence numbers must be finite")
            return item
        raise TypeError(f"Unsupported canonical evidence value: {type(item).__name__}")
    result = engine("content_hash", admit(_json_binary64(value)))
    digest = result.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Native engine returned an invalid content digest")
    return digest


def json_value(value):
    if hasattr(value, "detach"):
        detached = value.detach().cpu()
        if str(detached.dtype) == "torch.bfloat16":
            detached = detached.float()
        value = detached.numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    if str(getattr(value, "dtype", "")) == "bfloat16":
        value = value.astype("float32")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = str(key)
            if key in result:
                raise ValueError(f"Evidence keys collide after JSON conversion: {key}")
            result[key] = json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def tensor_row(value, name, *, role="variable", trainable=False, aliases=()):
    import numpy as np

    if hasattr(value, "detach"):
        if value.layout.__str__() != "torch.strided" or value.is_quantized:
            raise ValueError(
                "Native dense transport does not silently densify sparse/quantized tensors"
            )
        raw = value.detach().cpu().contiguous()
        if str(raw.dtype) == "torch.bfloat16":
            import torch

            array = raw.view(torch.uint16).numpy().copy()
            dtype = "BF16"
        else:
            array = raw.numpy().copy()
            dtype = None
    else:
        array = np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy()
        dtype = None
    dtypes = {
        "float16": "F16",
        "float32": "F32",
        "float64": "F64",
        "int8": "I8",
        "uint8": "U8",
        "int16": "I16",
        "uint16": "U16",
        "int32": "I32",
        "uint32": "U32",
        "int64": "I64",
        "uint64": "U64",
        "bool": "BOOL",
        "bfloat16": "BF16",
    }
    dtype = dtype or dtypes.get(array.dtype.name)
    if dtype is None:
        raise ValueError(f"Unsupported native scalar dtype {array.dtype}")
    array = (
        np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        if array.ndim
        else array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    return {
        "id": name,
        "name": name,
        "role": role,
        "trainable": bool(trainable),
        "aliases": list(aliases),
        "dtype": dtype,
        "shape": list(array.shape),
        "byte_order": "little",
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def atomic_json(path, value, *, replace=False, compact=False, max_bytes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, allow_nan=False,
                   indent=None if compact else 2) + ("" if compact else "\n")
    ).encode()
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError("JSON transport exceeds its byte budget")
    if path.exists() and not replace:
        if path.read_bytes() != data:
            raise ValueError(f"Immutable evidence conflict: {path}")
        return path
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(temp, path)
        else:
            try:
                os.link(temp, path)  # atomic create; no overwrite
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"Immutable evidence conflict: {path}")
            os.unlink(temp)
        # Persist the directory entry as well as file bytes. An atomic rename
        # alone does not make the new head durable across host power loss.
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def read_json(path, *, expected_type=dict, label="native evidence"):
    from .._engine import _strict_json

    return _strict_json(Path(path), expected_type=expected_type, label=label)


def verify_document(doc):
    fields = {
        "deepbom.training_result_manifest.v1": "training_result_manifest_sha256",
        "deepbom.model_state_snapshot.v1": "snapshot_sha256",
        "deepbom.model_ir.v2": "model_ir_sha256",
        "deepbom.weight_ir.v2": "weight_ir_sha256",
        "deepbom.activation_ir.v2": "activation_ir_sha256",
        "deepbom.training_ir.v1": "training_ir_sha256",
        "deepbom.training_chunk.v1": "chunk_sha256",
        "deepbom.native_model_comparison.v1": "comparison_sha256",
        "deepbom.model_candidate_manifest.v1": "manifest_sha256",
    }
    field = fields.get(doc.get("schema"))
    if field is None:
        raise ValueError("Unknown native evidence document")
    body = {k: v for k, v in doc.items() if k != field}
    if content_hash(body) != doc[field]:
        raise ValueError(f"{field} mismatch")
    return doc[field]


class EvidenceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, document):
        digest = verify_document(document)
        atomic_json(self.root / "objects" / (digest + ".json"), document)
        return {"schema": document["schema"], "sha256": digest}

    def get(self, digest):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid object digest")
        doc = read_json(self.root / "objects" / (digest + ".json"))
        if verify_document(doc) != digest:
            raise ValueError("Object path/content mismatch")
        return doc
