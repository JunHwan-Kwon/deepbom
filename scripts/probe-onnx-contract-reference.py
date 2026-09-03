import argparse
import hashlib
import json
from pathlib import Path

import onnx


def probe(path_string, operation):
    path = Path(path_string)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "artifact_sha256": digest,
        "onnx_version": onnx.__version__,
    }
    if operation == "checker":
        result["checker"] = run_check(path)
    elif operation == "strict_shape_inference":
        result["strict_shape_inference"] = run_shape_inference(path)
    return result


def run_check(path):
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model, full_check=True, check_custom_domain=True)
        return {"status": "pass", "exception_type": None, "message": None}
    except Exception as error:  # ONNX exposes several checker/inference exception classes.
        return failure(error, path)


def run_shape_inference(path):
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        return {"status": "pass", "exception_type": None, "message": None}
    except Exception as error:
        return failure(error, path)


def failure(error, path):
    message = str(error).replace(str(path), "<artifact>").replace(str(path.resolve()), "<artifact>")
    return {
        "status": "fail",
        "exception_type": type(error).__name__,
        "message": message[:2000],
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-check ONNX contract fixtures with the official ONNX package.")
    parser.add_argument("--operation", choices=("checker", "strict_shape_inference"), required=True)
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps({"schema": "deepbom.onnx_official_reference_probe.v1", "row": probe(args.path, args.operation)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
