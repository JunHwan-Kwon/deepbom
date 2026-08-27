import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def sha256_file(filename):
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_name(value):
    dtype = np.dtype(value)
    return "STRING" if dtype.kind in ("O", "S", "U") else dtype.name.upper()


def interpreter_parameters(filename):
    interpreter = Interpreter(model_path=str(filename))
    rows = []
    for direction, details in (
        ("input", interpreter.get_input_details()),
        ("output", interpreter.get_output_details()),
    ):
        for ordinal, detail in enumerate(details):
            quant = detail.get("quantization_parameters") or {}
            rows.append({
                "direction": direction,
                "ordinal": ordinal,
                "index": int(detail["index"]),
                "dtype": dtype_name(detail["dtype"]),
                "shape": [int(value) for value in detail["shape"].tolist()],
                "scales": [float(value) for value in quant.get("scales", [])],
                "zero_points": [
                    int(value) for value in quant.get("zero_points", [])
                ],
                "quantized_dimension": int(
                    quant.get("quantized_dimension", 0)
                ),
            })
    return rows


def expected_parameters(row):
    return [{
        "direction": parameter["direction"],
        "ordinal": parameter["ordinal"],
        "index": parameter["tensor_index"],
        "dtype": parameter["dtype"],
        "shape": parameter["shape"],
        "scales": parameter["quantization"]["scales"],
        "zero_points": parameter["quantization"]["zero_points"],
        "quantized_dimension":
            parameter["quantization"]["quantized_dimension"] or 0,
    } for parameter in row["interface_contracts"]["parameters"]]


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check a DeepBOM interface-contract corpus with LiteRT."
    )
    parser.add_argument("--sweep", required=True)
    parser.add_argument(
        "--cache-root",
        default=str(
            Path(os.environ.get("LOCALAPPDATA", Path.home()))
            / "DeepBOM"
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    sweep_path = Path(args.sweep).resolve()
    cache_root = Path(args.cache_root).resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    expected = {
        row["artifact_sha256"]: row
        for row in sweep["rows"]
        if row.get("status") == "passed"
        and row.get("public_corpus_member")
    }
    located = {}
    for filename in cache_root.rglob("*.tflite"):
        try:
            digest = sha256_file(filename)
        except OSError:
            continue
        if digest in expected and digest not in located:
            located[digest] = filename

    verified = []
    mismatches = []
    for digest, row in sorted(expected.items()):
        filename = located.get(digest)
        if filename is None:
            continue
        try:
            actual = interpreter_parameters(filename)
            wanted = expected_parameters(row)
            if actual == wanted:
                verified.append({
                    "qualified_id": row["qualified_id"],
                    "artifact_sha256": digest,
                    "parameter_count": len(actual),
                    "interface_ledger_sha256":
                        row["interface_contracts"]["ledger_sha256"],
                })
            else:
                mismatches.append({
                    "qualified_id": row["qualified_id"],
                    "artifact_sha256": digest,
                    "expected": wanted,
                    "actual": actual,
                })
        except Exception as error:
            mismatches.append({
                "qualified_id": row["qualified_id"],
                "artifact_sha256": digest,
                "error": f"{type(error).__name__}: {error}",
            })

    missing = sorted(set(expected) - set(located))
    result = {
        "schema": "deepbom.litert_interface_crosscheck.v1",
        "parser": f"ai-edge-litert {version('ai-edge-litert')}",
        "method": (
            "Interpreter.get_input_details/get_output_details only; "
            "no allocation or inference."
        ),
        "dtype_normalization":
            "NumPy object/bytes/unicode -> TFLite STRING",
        "source_sweep_sha256": sha256_file(sweep_path),
        "source_interface_corpus_ledger_sha256":
            sweep["interface_quantization_contract_summary"]["ledger_sha256"],
        "expected_artifact_count": len(expected),
        "located_artifact_count": len(located),
        "verified_artifact_count": len(verified),
        "verified_parameter_count":
            sum(row["parameter_count"] for row in verified),
        "missing_artifact_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_artifact_sha256": missing,
        "verified": verified,
        "mismatches": mismatches,
    }
    canonical = json.dumps(
        result, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    result["ledger_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    if missing or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
