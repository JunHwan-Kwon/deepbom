#!/usr/bin/env python3
"""
Reconstruct a synthetic .tflite from op_sequence stored in D1.
Weights are zeroed — shape/dtype/quant-equivalent for benchmark_model latency testing.

Usage:
    python scripts/reconstruct_tflite.py <fingerprint> [--out output.tflite]
    python scripts/reconstruct_tflite.py --list
"""

import argparse
import json
import math
import struct
import sys
import urllib.request
import urllib.error
from pathlib import Path

import flatbuffers
import numpy as np

# ---------------------------------------------------------------------------
# TFLite schema constants (flatbuffers field indices, stable across TFLite 2.x)
# TensorType enum
TTYPE = {"FLOAT32": 0, "FLOAT16": 1, "INT32": 2, "UINT8": 3, "INT64": 4,
         "STRING": 5, "BOOL": 6, "INT16": 7, "COMPLEX64": 8, "INT8": 9,
         "FLOAT64": 10, "COMPLEX128": 11, "UINT64": 12, "RESOURCE": 13,
         "VARIANT": 14, "UINT32": 15, "UINT16": 16, "INT4": 17}

TTYPE_BYTES = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 6: 1, 7: 2, 9: 1, 10: 8, 15: 4, 16: 2, 17: 1}

# TFLite builtin op codes (partial — most common ops)
OP_CODES = {
    "ADD": 0, "AVERAGE_POOL_2D": 1, "CONCATENATION": 2, "CONV_2D": 3,
    "DEPTHWISE_CONV_2D": 4, "DEPTH_TO_SPACE": 5, "DEQUANTIZE": 6,
    "EMBEDDING_LOOKUP": 7, "FLOOR": 8, "FULLY_CONNECTED": 9,
    "HASHTABLE_LOOKUP": 10, "L2_NORMALIZATION": 11, "L2_POOL_2D": 12,
    "LOCAL_RESPONSE_NORMALIZATION": 13, "LOGISTIC": 14, "LSH_PROJECTION": 15,
    "LSTM": 16, "MAX_POOL_2D": 17, "MUL": 18, "RELU": 19, "RELU_N1_TO_1": 20,
    "RESHAPE": 21, "RESIZE_BILINEAR": 22, "RNN": 23, "SOFTMAX": 24,
    "SPACE_TO_DEPTH": 25, "SVDF": 26, "TANH": 27, "CONCAT_EMBEDDINGS": 28,
    "SKIP_GRAM": 29, "CALL": 30, "CUSTOM": 32, "EMBEDDING_LOOKUP_SPARSE": 33,
    "PAD": 34, "UNIDIRECTIONAL_SEQUENCE_RNN": 35, "GATHER": 36,
    "BATCH_TO_SPACE_ND": 37, "SPACE_TO_BATCH_ND": 38, "TRANSPOSE": 39,
    "MEAN": 40, "SUB": 41, "DIV": 42, "SQUEEZE": 43, "UNIDIRECTIONAL_SEQUENCE_LSTM": 44,
    "STRIDED_SLICE": 45, "BIDIRECTIONAL_SEQUENCE_RNN": 46, "EXP": 47,
    "TOPK_V2": 48, "SPLIT": 49, "LOG_SOFTMAX": 50, "DELEGATE": 51,
    "BIDIRECTIONAL_SEQUENCE_LSTM": 52, "CAST": 53, "PRELU": 54, "MAXIMUM": 55,
    "ARG_MAX": 56, "MINIMUM": 57, "LESS": 58, "NEG": 59, "PADV2": 60,
    "GREATER": 61, "GREATER_EQUAL": 62, "LESS_EQUAL": 63, "SELECT": 64,
    "SLICE": 65, "SIN": 66, "TRANSPOSE_CONV": 67, "SPARSE_TO_DENSE": 68,
    "TILE": 69, "EXPAND_DIMS": 70, "EQUAL": 71, "NOT_EQUAL": 72,
    "LOG": 73, "SUM": 74, "SQRT": 75, "RSQRT": 76, "SHAPE": 77,
    "POW": 78, "ARG_MIN": 79, "FAKE_QUANT": 80, "REDUCE_PROD": 81,
    "REDUCE_MAX": 82, "PACK": 83, "LOGICAL_OR": 84, "ONE_HOT": 85,
    "LOGICAL_AND": 86, "LOGICAL_NOT": 87, "UNPACK": 88, "REDUCE_MIN": 89,
    "FLOOR_DIV": 90, "REDUCE_ANY": 91, "SQUARE": 92, "ZEROS_LIKE": 93,
    "FILL": 94, "FLOOR_MOD": 95, "RANGE": 96, "RESIZE_NEAREST_NEIGHBOR": 97,
    "LEAKY_RELU": 98, "SQUARED_DIFFERENCE": 99, "MIRROR_PAD": 100,
    "ABS": 101, "SPLIT_V": 102, "UNIQUE": 103, "CEIL": 104, "REVERSE_V2": 105,
    "ADD_N": 106, "GATHER_ND": 107, "COS": 108, "WHERE": 109, "RANK": 110,
    "ELU": 111, "REVERSE_SEQUENCE": 112, "MATRIX_DIAG": 113,
    "QUANTIZE": 114, "MATRIX_SET_DIAG": 115, "ROUND": 116, "HARD_SWISH": 117,
    "IF": 118, "WHILE": 119, "NON_MAX_SUPPRESSION_V4": 120,
    "NON_MAX_SUPPRESSION_V5": 121, "SCATTER_ND": 122, "SELECT_V2": 123,
    "DENSIFY": 124, "SEGMENT_SUM": 125, "BATCH_MATMUL": 126,
    "PLACEHOLDER_FOR_GREATER_OP_CODES": 127,
    # codes >= 128 are "extended" — use PLACEHOLDER_FOR_GREATER_OP_CODES mapping
}


def op_code_for(name: str) -> int:
    code = OP_CODES.get(name.upper())
    if code is None:
        print(f"  [warn] unknown op '{name}', treating as CUSTOM (32)", file=sys.stderr)
        return 32
    return code


# ---------------------------------------------------------------------------
# Minimal flatbuffer builder for TFLite
# We write the binary manually to avoid needing the generated TFLite Python bindings.

def write_string(b: flatbuffers.Builder, s: str) -> int:
    return b.CreateString(s)


def write_int_vector(b: flatbuffers.Builder, values: list) -> int:
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependInt32(v)
    return b.EndVector()


def write_shape(b: flatbuffers.Builder, dims: list) -> int:
    b.StartVector(4, len(dims), 4)
    for d in reversed(dims):
        b.PrependInt32(int(d))
    return b.EndVector()


def write_buffer(b: flatbuffers.Builder, data: bytes) -> int:
    if data:
        data_vec = b.CreateByteVector(data)
        # Buffer table: field 1 = data
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, data_vec, 0)
        return b.EndObject()
    else:
        # Empty buffer
        b.StartObject(2)
        return b.EndObject()


def write_quant_params(b: flatbuffers.Builder, scale=1.0 / 128, zero_point=0) -> int:
    scales_vec = write_shape(b, [scale])
    b.StartVector(8, 1, 8)
    b.PrependInt64(zero_point)
    zp_vec = b.EndVector()
    # QuantizationParameters: field 2=scale, 3=zero_point
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(1, scales_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, zp_vec, 0)
    return b.EndObject()


def write_tensor(b: flatbuffers.Builder, name: str, shape: list, dtype_str: str,
                 buffer_idx: int, is_quantized: bool) -> int:
    name_off = write_string(b, name)
    shape_off = write_shape(b, shape)
    quant_off = write_quant_params(b) if is_quantized else None
    ttype = TTYPE.get(dtype_str.upper(), 0)
    # Tensor table: 0=name, 1=type, 2=shape, 4=quantization, 5=is_variable
    b.StartObject(7)
    b.PrependUOffsetTRelativeSlot(0, name_off, 0)
    b.PrependInt8Slot(1, ttype, 0)
    b.PrependUOffsetTRelativeSlot(2, shape_off, 0)
    b.PrependInt32Slot(3, buffer_idx, 0)  # field 3 = buffer
    if quant_off is not None:
        b.PrependUOffsetTRelativeSlot(4, quant_off, 0)
    return b.EndObject()


def write_operator_code(b: flatbuffers.Builder, builtin_code: int) -> int:
    b.StartObject(4)
    if builtin_code >= 127:
        b.PrependInt8Slot(0, 127, 0)
        b.PrependInt32Slot(3, builtin_code, 0)
    else:
        b.PrependInt8Slot(0, builtin_code, 0)
    return b.EndObject()


def write_operator(b: flatbuffers.Builder, opcode_idx: int,
                   input_idxs: list, output_idxs: list) -> int:
    inputs_off = write_int_vector(b, input_idxs)
    outputs_off = write_int_vector(b, output_idxs)
    # Operator table: 0=opcode_index, 1=inputs, 2=outputs
    b.StartObject(6)
    b.PrependUint32Slot(0, opcode_idx, 0)
    b.PrependUOffsetTRelativeSlot(1, inputs_off, 0)
    b.PrependUOffsetTRelativeSlot(2, outputs_off, 0)
    return b.EndObject()


def write_vector_of_offsets(b: flatbuffers.Builder, offsets: list) -> int:
    b.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        b.PrependUOffsetTRelative(off)
    return b.EndVector()


# ---------------------------------------------------------------------------

def tensor_size_bytes(shape: list, dtype_str: str) -> int:
    nbytes = TTYPE_BYTES.get(TTYPE.get(dtype_str.upper(), 0), 4)
    n = 1
    for d in shape:
        if d > 0:
            n *= d
    return n * nbytes


def reconstruct(op_sequence: list, input_contract: list, output_contract: list) -> bytes:
    """
    Build a minimal valid TFLite flatbuffer from op_sequence.

    op_sequence items: {index, name, input_shapes, output_shapes, input_dtypes, output_dtypes, ...}
    input_contract: ["UINT8[1x320x320x3]", ...]  (used only for naming)
    output_contract: ["FLOAT32[1x19206x4]", ...]
    """

    # --- Collect unique tensors from op sequence ---
    # tensor_id = (shape_tuple, dtype_str)
    # We assign a flat tensor index and track which ones need weight buffers.

    # First pass: assign a unique tensor for each op input/output by (shape, dtype).
    # In a real model, tensors are shared; here we unify by canonical key.
    # Key strategy: op[i].output_shapes[j] produced by op i, consumed later.

    # Build a graph: for each (op_idx, output_slot) → assign tensor_idx
    # Then wire op inputs to matching prior outputs (or create new constant tensors).

    tensor_list = []  # [(shape, dtype, name, is_weight)]
    tensor_key_to_idx = {}  # (tuple(shape), dtype) -> tensor_idx

    def get_or_create_tensor(shape, dtype, name_hint="t", is_weight=False):
        key = (tuple(shape), dtype.upper())
        if key in tensor_key_to_idx:
            return tensor_key_to_idx[key]
        idx = len(tensor_list)
        tensor_list.append({"shape": shape, "dtype": dtype.upper(),
                             "name": f"{name_hint}_{idx}", "is_weight": is_weight})
        tensor_key_to_idx[key] = idx
        return idx

    # Track which tensor index each op's output occupies
    op_output_tensors = []  # [list of tensor_idx per op]

    for op in op_sequence:
        out_idxs = []
        for j, (shape, dtype) in enumerate(zip(op.get("output_shapes", []), op.get("output_dtypes", []))):
            t_idx = get_or_create_tensor(shape, dtype, name_hint=f"{op['name']}_out")
            out_idxs.append(t_idx)
        op_output_tensors.append(out_idxs)

    # Assign input tensors: find matching output from prior op, else create weight/input tensor
    produced_keys = set()
    op_input_tensors = []

    for i, op in enumerate(op_sequence):
        in_idxs = []
        for j, (shape, dtype) in enumerate(zip(op.get("input_shapes", []), op.get("input_dtypes", []))):
            key = (tuple(shape), dtype.upper())
            if key in tensor_key_to_idx and key in produced_keys:
                in_idxs.append(tensor_key_to_idx[key])
            else:
                # Constant (weight) tensor
                t_idx = get_or_create_tensor(shape, dtype, name_hint=f"{op['name']}_w", is_weight=True)
                tensor_list[t_idx]["is_weight"] = True
                in_idxs.append(t_idx)
        op_input_tensors.append(in_idxs)

        # Mark this op's outputs as produced
        for key_shape, key_dtype in zip(op.get("output_shapes", []), op.get("output_dtypes", [])):
            produced_keys.add((tuple(key_shape), key_dtype.upper()))

    # Identify graph inputs/outputs from contracts
    graph_input_idxs = []
    for contract in input_contract:
        # Parse "UINT8[1x320x320x3]" → dtype="UINT8", shape=[1,320,320,3]
        try:
            dtype_part, shape_part = contract.split("[", 1)
            shape = [int(d) for d in shape_part.rstrip("]").split("x")]
            key = (tuple(shape), dtype_part.upper())
            if key not in tensor_key_to_idx:
                get_or_create_tensor(shape, dtype_part, "input")
            graph_input_idxs.append(tensor_key_to_idx[key])
        except Exception:
            pass

    graph_output_idxs = []
    for contract in output_contract:
        try:
            dtype_part, shape_part = contract.split("[", 1)
            shape = [int(d) for d in shape_part.rstrip("]").split("x")]
            key = (tuple(shape), dtype_part.upper())
            if key in tensor_key_to_idx:
                graph_output_idxs.append(tensor_key_to_idx[key])
        except Exception:
            pass

    # Fallback: use last op's outputs as graph outputs
    if not graph_output_idxs and op_sequence:
        graph_output_idxs = op_output_tensors[-1]

    # --- Collect unique op codes ---
    unique_ops = []
    opcode_name_to_idx = {}
    for op in op_sequence:
        name = op["name"].upper()
        if name not in opcode_name_to_idx:
            opcode_name_to_idx[name] = len(unique_ops)
            unique_ops.append(name)

    print(f"  Tensors: {len(tensor_list)} (weights: {sum(1 for t in tensor_list if t['is_weight'])})")
    print(f"  Unique op types: {len(unique_ops)}")
    print(f"  Graph inputs: {graph_input_idxs}, outputs: {graph_output_idxs}")

    # --- Build flatbuffer ---
    b = flatbuffers.Builder(1024 * 512)

    # Buffers: buffer[0] = empty sentinel, then one per weight tensor
    weight_tensor_idxs = [i for i, t in enumerate(tensor_list) if t["is_weight"]]
    tensor_to_buffer = {}
    buffer_offsets = []

    # buffer 0: empty
    buffer_offsets.append(write_buffer(b, b""))
    buf_idx = 1

    for t_idx in range(len(tensor_list)):
        t = tensor_list[t_idx]
        if t["is_weight"]:
            size = tensor_size_bytes(t["shape"], t["dtype"])
            data = bytes(size)  # zeroed weights
            buffer_offsets.append(write_buffer(b, data))
            tensor_to_buffer[t_idx] = buf_idx
            buf_idx += 1
        else:
            tensor_to_buffer[t_idx] = 0  # activation tensors use empty buffer

    # Operator codes
    opcode_offsets = [write_operator_code(b, op_code_for(name)) for name in unique_ops]

    # Tensors
    tensor_offsets = []
    for i, t in enumerate(tensor_list):
        is_q = t["dtype"] in ("INT8", "UINT8", "INT16")
        tensor_offsets.append(write_tensor(b, t["name"], t["shape"], t["dtype"],
                                           tensor_to_buffer[i], is_q))

    # Operators
    op_offsets = []
    for i, op in enumerate(op_sequence):
        opcode_idx = opcode_name_to_idx.get(op["name"].upper(), 0)
        op_offsets.append(write_operator(b, opcode_idx, op_input_tensors[i], op_output_tensors[i]))

    # Subgraph
    tensors_vec = write_vector_of_offsets(b, tensor_offsets)
    ops_vec = write_vector_of_offsets(b, op_offsets)
    inputs_vec = write_int_vector(b, graph_input_idxs)
    outputs_vec = write_int_vector(b, graph_output_idxs)
    sg_name = write_string(b, "main")

    b.StartObject(7)
    b.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    b.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(3, ops_vec, 0)
    b.PrependUOffsetTRelativeSlot(4, sg_name, 0)
    subgraph = b.EndObject()

    # Model
    buffers_vec = write_vector_of_offsets(b, buffer_offsets)
    opcodes_vec = write_vector_of_offsets(b, opcode_offsets)
    subgraphs_vec = write_vector_of_offsets(b, [subgraph])
    desc = write_string(b, "Synthetic zero-weight reconstruction for benchmark_model")

    b.StartObject(8)
    b.PrependUint32Slot(0, 3, 0)  # version = 3
    b.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    b.PrependUOffsetTRelativeSlot(3, desc, 0)
    b.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
    model = b.EndObject()

    b.Finish(model)
    buf = bytes(b.Output())

    # Prepend TFLite file identifier ("TFL3")
    size = len(buf)
    header = struct.pack("<I", size) + b"TFL3"
    # flatbuffers already encodes size in first 4 bytes; prepend file_identifier
    # TFLite format: first 4 bytes = root table offset, bytes 4-8 = "TFL3" file_identifier
    # The flatbuffers builder puts the root offset at position 0; we insert the identifier at offset 4.
    result = buf[:4] + b"TFL3" + buf[8:]
    return result


# ---------------------------------------------------------------------------

def fetch_structure(fingerprint: str, api_base: str) -> dict:
    url = f"{api_base}/api/admin/structures/{fingerprint}" if "/admin/" in api_base else f"{api_base}/api/benchmark/structure-export/{fingerprint}"
    # Try direct D1 export endpoint
    url = f"{api_base}/api/debug/structure/{fingerprint}"
    print(f"  Fetching: {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")


def _wrangler(sql: str) -> list:
    import subprocess, shutil
    cwd = Path(__file__).parent.parent
    cmd = f'npx wrangler d1 execute deepbom_auth --remote --command "{sql}"'
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, shell=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    out = result.stdout or ""
    start = out.find("[")
    if start == -1:
        raise RuntimeError(f"Unexpected wrangler output:\n{out}")
    data = json.loads(out[start:])
    return data[0]["results"]


def list_structures_local() -> list:
    return _wrangler("SELECT fingerprint, format, op_count, tensor_count, created_at FROM model_structures ORDER BY created_at DESC LIMIT 20")


def fetch_structure_local(fingerprint: str) -> dict:
    rows = _wrangler(f"SELECT metadata_json, op_count FROM model_structures WHERE fingerprint='{fingerprint}'")
    if not rows:
        raise RuntimeError(f"fingerprint not found: {fingerprint}")
    meta = json.loads(rows[0]["metadata_json"] or "{}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Reconstruct synthetic TFLite from D1 op_sequence")
    parser.add_argument("fingerprint", nargs="?", help="Model fingerprint (sf_...)")
    parser.add_argument("--list", action="store_true", help="List available structures")
    parser.add_argument("--out", default="reconstructed.tflite", help="Output .tflite path")
    args = parser.parse_args()

    if args.list or not args.fingerprint:
        print("Fetching structures from D1...")
        rows = list_structures_local()
        print(f"\n{'fingerprint':<70} {'fmt':<8} {'ops':>5} {'tensors':>8} {'created_at'}")
        print("-" * 110)
        for r in rows:
            print(f"{r['fingerprint']:<70} {r['format']:<8} {r['op_count']:>5} {r['tensor_count']:>8} {r['created_at']}")
        return

    print(f"Fetching structure: {args.fingerprint}")
    meta = fetch_structure_local(args.fingerprint)

    op_sequence = meta.get("op_sequence", [])
    if not op_sequence:
        print("ERROR: op_sequence is empty — model was saved before telemetry was updated.")
        print("Re-analyze the model in the browser (with research consent enabled) to populate op_sequence.")
        sys.exit(1)

    input_contract = meta.get("input_contract", [])
    output_contract = meta.get("output_contract", [])

    print(f"op_sequence: {len(op_sequence)} ops")
    print(f"input: {input_contract}")
    print(f"output: {output_contract}")
    print("Reconstructing...")

    tflite_bytes = reconstruct(op_sequence, input_contract, output_contract)

    out_path = Path(args.out)
    out_path.write_bytes(tflite_bytes)
    print(f"\nWrote {len(tflite_bytes):,} bytes → {out_path}")
    print("\nVerify with:")
    print(f"  python -c \"import tflite_runtime.interpreter as tflite; i=tflite.Interpreter('{out_path}'); i.allocate_tensors(); print('OK', i.get_input_details())\"")
    print(f"\nBenchmark on device:")
    print(f"  adb push {out_path} /data/local/tmp/")
    print(f"  adb shell benchmark_model --graph=/data/local/tmp/{out_path.name} --num_runs=50")


if __name__ == "__main__":
    main()
