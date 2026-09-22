#!/usr/bin/env python3
"""Explicit local ONNX/LiteRT execution capture. Never imported by static audit."""
import argparse
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--model-ir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-execution", action="store_true")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs-npz", type=Path)
    group.add_argument("--probe", choices=["ones", "zeros", "identity"])
    p.add_argument("--max-values", type=int, default=1_000_000)
    p.add_argument("--outputs-only", action="store_true")
    p.add_argument("--entry-region", help="Exact Model IR entry region ID for a model with multiple entries")
    args = p.parse_args()
    if not args.allow_execution:
        p.error("Execution requires --allow-execution. Static audit never invokes this collector.")
    if args.output.exists():
        p.error("Output already exists; choose a new capture path.")
    if args.max_values < 1 or args.max_values > 1_000_000:
        p.error("--max-values must be between 1 and 1000000.")
    import numpy as np
    model_ir = json.loads(args.model_ir.read_text())
    if model_ir.get("schema") == "deepbom.analysis_selection.v1":
        model_ir = model_ir["sections"]["model_ir"]
    raw = args.model.read_bytes()
    if sha(raw) != model_ir["artifact"]["sha256"]:
        raise ValueError("Model bytes do not match Model IR.")
    source = {"model_ir_sha256": model_ir["model_ir_sha256"], "artifact_sha256": sha(raw), "artifact_set_sha256": model_ir["artifact_set"]["artifact_set_sha256"]}
    values = model_ir["program"]["values"]
    entries = [entry for program in model_ir["program"]["programs"] for entry in program["entry_region_refs"]]
    if len(entries) != 1 and not args.entry_region:
        raise ValueError("Multiple entry regions require --entry-region.")
    entry = args.entry_region or entries[0]
    if entry not in entries:
        raise ValueError("Unknown entry region.")
    expected_entry = {"onnx": "region:scope:onnx:main_graph", "tflite": "region:scope:tflite:subgraph:0"}.get(model_ir["artifact"]["format"])
    if entry != expected_entry:
        raise ValueError("This collector supports the primary ONNX graph or TFLite subgraph only.")
    inputs = [v for v in values if "graph_input" in v["roles"] and v["region_ref"] == entry and not v["storage_refs"]]
    eligible = [v for v in values if not ("graph_input" in v["roles"] and v["region_ref"] == entry) and not v["storage_refs"]]
    requested = [v for v in eligible if not args.outputs_only or "graph_output" in v["roles"]]
    dtype_map = {"FLOAT32": np.float32, "FLOAT16": np.float16, "FLOAT64": np.float64, "INT64": np.int64, "INT32": np.int32, "INT16": np.int16, "INT8": np.int8, "UINT64": np.uint64, "UINT32": np.uint32, "UINT16": np.uint16, "UINT8": np.uint8, "BOOL": np.bool_}
    input_bytes = args.inputs_npz.read_bytes() if args.inputs_npz else None
    npz = np.load(io.BytesIO(input_bytes), allow_pickle=False) if input_bytes is not None else None
    feeds = {}
    for item in inputs:
        if item["name"] in feeds:
            raise ValueError("Input names are ambiguous across serialized regions.")
        dtype = dtype_map[item["dtype"]]
        if npz is not None:
            array = npz[item["name"]]
            if array.dtype != dtype:
                raise ValueError(f"Input {item['name']} dtype mismatch; implicit casts are disabled.")
        else:
            shape = item["shape"]
            if any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError("Dynamic inputs require --inputs-npz with explicit dimensions.")
            if math.prod(shape) > args.max_values:
                raise ValueError("Synthetic input exceeds value budget.")
            if args.probe == "identity":
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError("Identity probe requires a square rank-2 input.")
                array = np.eye(shape[0], dtype=dtype)
            else:
                array = np.full(shape, int(args.probe == "ones"), dtype=dtype)
        feeds[item["name"]] = array
    if npz is not None:
        if set(npz.files) != set(feeds):
            raise ValueError("NPZ keys must exactly match serialized graph inputs.")
        npz.close()
    consumed = sum(array.size for array in feeds.values())
    if consumed > args.max_values:
        raise ValueError("Inputs exceed value budget.")
    missing, selected = [], []
    for item in requested:
        shape = item["shape"]
        expected = math.prod(shape) if all(isinstance(d, int) and d >= 0 for d in shape) else None
        if item["dtype"] not in dtype_map:
            missing.append({"value_ref": item["id"], "reason": "collector_dtype_unsupported"})
        elif expected is None or consumed + expected > args.max_values:
            missing.append({"value_ref": item["id"], "reason": "shape_or_capture_budget_unavailable"})
        else:
            selected.append(item)
            consumed += expected
    captures = []
    def encode(item, array):
        if str(array.dtype) not in [str(np.dtype(dtype_map[item["dtype"]]))]:
            raise ValueError("Runtime capture dtype differs from Model IR.")
        if array.size > args.max_values:
            raise ValueError("Runtime tensor exceeds capture budget.")
        numeric = []
        for v in array.reshape(-1):
            if array.dtype.kind in "iu": numeric.append(str(int(v)) if array.dtype.itemsize == 8 else int(v))
            elif array.dtype.kind == "b": numeric.append(int(v))
            else:
                n = float(v)
                numeric.append("NaN" if math.isnan(n) else "+Infinity" if n == math.inf else "-Infinity" if n == -math.inf else "-0" if n == 0 and math.copysign(1, n) < 0 else n)
        return {"value_ref": item["id"], "native_locator": item["name"], "dtype": item["dtype"], "shape": list(array.shape), "values": numeric}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    instrumented_hash = None
    configuration = {"collector_version": "1.0.0", "max_values": args.max_values, "outputs_only": args.outputs_only, "probe": args.probe, "threads": 1, "numpy_version": np.__version__, "python_version": sys.version.split()[0], "inputs_npz_sha256": sha(input_bytes) if input_bytes is not None else None}
    if model_ir["artifact"]["format"] == "onnx":
        import onnx
        import onnxruntime as ort
        configuration["onnx_version"] = onnx.__version__
        parsed = onnx.load_model_from_string(raw)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in parsed.graph.initializer):
            raise ValueError("This collector requires inline ONNX data; external artifacts need an independently bound collector.")
        try: parsed = onnx.shape_inference.infer_shapes(parsed)
        except (ValueError, RuntimeError): pass
        known = {v.name: v for v in list(parsed.graph.value_info) + list(parsed.graph.input) + list(parsed.graph.output)}
        outputs = {v.name for v in parsed.graph.output}
        accepted = []
        for item in selected:
            if item["region_ref"] != "region:scope:onnx:main_graph":
                missing.append({"value_ref": item["id"], "reason": "nested_region_capture_not_supported"})
                continue
            if item["name"] not in outputs:
                if item["name"] not in known:
                    missing.append({"value_ref": item["id"], "reason": "onnx_value_type_not_inferred"})
                    continue
                parsed.graph.output.append(known[item["name"]])
                outputs.add(item["name"])
            accepted.append(item)
        instrumented = parsed.SerializeToString()
        instrumented_hash = sha(instrumented)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(instrumented, options, providers=["CPUExecutionProvider"])
        if not accepted:
            raise ValueError("No requested activations are collectible within the bound; no run was made.")
        arrays = session.run([v["name"] for v in accepted], feeds)
        captures = [encode(item, array) for item, array in zip(accepted, arrays)]
        runtime = {"name": "onnxruntime", "version": ort.__version__, "configured_providers": session.get_providers(), "device": "CPU"}
        configuration["graph_optimizations"] = "disabled"
    elif model_ir["artifact"]["format"] == "tflite":
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        from importlib.metadata import version
        interpreter = Interpreter(model_content=raw, num_threads=1, experimental_preserve_all_tensors=True, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        for item in interpreter.get_input_details():
            interpreter.resize_tensor_input(item["index"], feeds[item["name"]].shape, strict=True)
        interpreter.allocate_tensors()
        for item in interpreter.get_input_details(): interpreter.set_tensor(item["index"], feeds[item["name"]])
        interpreter.invoke()
        for item in selected:
            if item["region_ref"] != "region:scope:tflite:subgraph:0":
                missing.append({"value_ref": item["id"], "reason": "non_primary_subgraph_not_collected"})
                continue
            try: array = interpreter.get_tensor(item["native_index"])
            except ValueError:
                missing.append({"value_ref": item["id"], "reason": "runtime_did_not_preserve_tensor"})
                continue
            captures.append(encode(item, array))
        runtime = {"name": "ai-edge-litert", "version": version("ai-edge-litert"), "configured_providers": ["builtin_without_default_delegates"], "device": "CPU"}
        configuration["preserve_all_tensors"] = True
    else:
        raise ValueError("Built-in capture supports ONNX and TFLite. Other runtimes can produce the same capture contract; runtime support is not inferred from file format.")
    if sum(len(t["values"]) for t in captures) + sum(a.size for a in feeds.values()) > args.max_values:
        raise ValueError("Actual run exceeds capture value budget; no partial success is emitted.")
    if args.model.read_bytes() != raw:
        raise ValueError("Artifact changed during capture.")
    run = {"id": str(uuid.uuid4()), "started_at": started, "entry_region_ref": entry, "runtime": runtime,
           "collector": {"name": "deepbom-local-activation-collector", "version": "1.0.0", "sha256": sha(Path(__file__).read_bytes())},
           "execution": {"artifact_sha256": source["artifact_sha256"], "instrumented_artifact_sha256": instrumented_hash, "configuration_sha256": sha(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()), "configuration": configuration},
           "probe": {"kind": "custom" if args.inputs_npz else "synthetic_" + args.probe, "description": "User supplied numeric NPZ; preprocessing and representativeness are not independently established." if args.inputs_npz else "Synthetic " + args.probe + " input in serialized dtype; not a representative-data evaluation."}, "runtime_evidence": None}
    capture = {"schema": "deepbom.activation_capture.v1", "source": source, "run": run, "inputs": [encode(item, feeds[item["name"]]) for item in inputs], "requested_value_refs": [v["id"] for v in requested], "captures": captures, "missing": missing}
    encoded = json.dumps(capture, allow_nan=False, separators=(",", ":")) + "\n"
    if len(encoded.encode()) > 16 * 1024 * 1024:
        raise ValueError("Capture exceeds the 16 MiB import limit.")
    with args.output.open("x") as output: output.write(encoded)
    print(f"Captured {len(captures)}/{len(requested)} requested values; {len(missing)} unavailable. {args.output}")


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"Activation capture failed: {error}", file=sys.stderr)
        sys.exit(1)
