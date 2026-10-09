"""Transport and content-addressed storage. Numerical definitions live in JS."""

from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def engine(operation: str, payload: dict) -> dict:
    from ..api import _invoke_json

    with tempfile.TemporaryDirectory(prefix="deepbom-native-") as tmp:
        request = Path(tmp) / "request.json"
        request.write_text(
            json.dumps({"operation": operation, "payload": payload}, allow_nan=False),
            encoding="utf-8",
        )
        return _json_binary64(
            _invoke_json(
                ["evidence-native", "--request", str(request)], 300, 128 * 1024 * 1024
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
    if isinstance(value, list):
        return [_json_binary64(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_binary64(x) for k, x in value.items()}
    return value


def content_hash(value: Any) -> str:
    import rfc8785

    return hashlib.sha256(rfc8785.dumps(_json_binary64(value))).hexdigest()


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


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode()
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
