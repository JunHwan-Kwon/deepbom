#!/usr/bin/env python3
"""Capture a hash-bound Core ML MLComputePlan without treating it as execution."""

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import tempfile
import uuid


SCHEMA = "deepbom.coreml_compute_plan.v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compute-units", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"], default="ALL")
    parser.add_argument("--function-name")
    args = parser.parse_args()

    import coremltools as ct
    from coremltools.models import compute_plan as compute_plan_module

    if platform.system() != "Darwin":
        raise RuntimeError("Core ML compute-plan capture requires macOS and the native Core ML runtime")

    model_path = pathlib.Path(args.model).resolve()
    artifact_sha = artifact_digest(model_path)
    expected_sha = normalized_sha(args.artifact_sha256, "artifact SHA-256")
    if artifact_sha != expected_sha:
        raise ValueError(f"Core ML artifact digest mismatch: computed {artifact_sha}, expected {expected_sha}")

    with tempfile.TemporaryDirectory(prefix="deepbom-coreml-") as temporary:
        destination = pathlib.Path(temporary) / "DeepBOM.mlmodelc"
        compiled_path = pathlib.Path(ct.models.utils.compile_model(str(model_path), destination_path=str(destination)))
        compiled_digest = directory_digest(compiled_path, "deepbom.coreml_compiled_model_digest.v1")
        compute_units = getattr(ct.ComputeUnit, args.compute_units)
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled_path), compute_units=compute_units)
        structure = collect_structure(plan, args.function_name)

    source_path = pathlib.Path(compute_plan_module.__file__).resolve()
    collector_path = pathlib.Path(__file__).resolve()
    host = capture_host_environment(ct)
    document = {
        "schema": SCHEMA,
        "evidence_class": "COREML_COMPUTE_PLAN_ESTIMATE",
        "artifact": {"format": "coreml", "filename": model_path.name, "sha256": artifact_sha},
        "runtime": {
            "coremltools_version": str(ct.__version__),
            "coremltools_compute_plan_source_sha256": hash_file(source_path),
            "compiled_model_content_sha256": compiled_digest,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            **host,
        },
        "configuration": {"compute_units": args.compute_units, "function_name": structure.pop("function_name", None)},
        "capture": {
            "capture_id": str(uuid.uuid4()),
            "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "collector": {
                "name": "deepbom-coreml-compute-plan-collector",
                "version": "2",
                "source_sha256": hash_file(collector_path),
            },
        },
        "structure": structure,
        "boundary": "MLComputePlan provides anticipated compute-device usage and estimated relative costs for a compiled model. These rows are not executed placement, latency, allocation, fusion, or lowering observations.",
    }
    pathlib.Path(args.output).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Core ML compute plan written to {args.output}")
    print(f"artifact {artifact_sha}")
    print(f"compiled model {compiled_digest}")
    print(f"operations {structure['operation_count']}")


def collect_structure(plan, function_name):
    model_structure = plan.model_structure
    if model_structure.program is not None:
        functions = model_structure.program.functions
        selected_name = function_name or ("main" if "main" in functions else next(iter(functions)))
        if selected_name not in functions:
            raise ValueError(f"Core ML function {selected_name!r} is absent from MLComputePlan")
        rows = []
        collect_program_block(plan, functions[selected_name].block, rows)
        return {"kind": "program", "function_name": selected_name, "operation_count": len(rows), "rows": rows}
    if model_structure.neuralnetwork is not None:
        rows = []
        for index, layer in enumerate(model_structure.neuralnetwork.layers):
            usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
            rows.append(row(index, layer.type, layer.name, usage, None))
        return {"kind": "neuralnetwork", "function_name": None, "operation_count": len(rows), "rows": rows}
    if model_structure.pipeline is not None:
        raise ValueError("Pipeline MLComputePlan is not imported until DeepBOM decodes and binds every nested pipeline model")
    raise ValueError("MLComputePlan returned an unsupported model structure")


def collect_program_block(plan, block, rows):
    for operation in block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
        identity = ", ".join(output.name for output in operation.outputs)
        rows.append(row(len(rows), operation.operator_name, identity, usage, None if cost is None else cost.weight))
        for nested in operation.blocks:
            collect_program_block(plan, nested, rows)


def row(index, operator_type, identity, usage, cost):
    preferred = None if usage is None else device_name(usage.preferred_compute_device)
    supported = [] if usage is None else [device_name(device) for device in usage.supported_compute_devices]
    supported = list(dict.fromkeys(supported))
    return {
        "op_index": index,
        "operator_type": str(operator_type),
        "identity": str(identity),
        "preferred_compute_device": preferred,
        "supported_compute_devices": supported,
        "estimated_cost_weight": None if cost is None else float(cost),
    }


def device_name(device):
    name = type(device).__name__
    mapping = {"MLCPUComputeDevice": "CPU", "MLGPUComputeDevice": "GPU", "MLNeuralEngineComputeDevice": "NEURAL_ENGINE"}
    if name not in mapping:
        raise ValueError(f"Unsupported Core ML compute device class: {name}")
    return mapping[name]


def capture_host_environment(coremltools):
    inventory = {}
    for device in coremltools.models.compute_device.MLComputeDevice.get_all_compute_devices():
        normalized_type = device_name(device)
        normalized = inventory.setdefault(normalized_type, {
            "type": normalized_type,
            "source_class": type(device).__name__,
            "instance_count": 0,
        })
        normalized["instance_count"] += 1
        if normalized_type == "NEURAL_ENGINE":
            normalized["total_core_count"] = normalized.get("total_core_count", 0) + int(device.total_core_count)
    devices = list(inventory.values())
    devices.sort(key=lambda item: item["type"])
    if not devices:
        raise ValueError("Core ML compute-device inventory is empty")
    return {
        "platform_system": platform.system(),
        "macos_version": platform.mac_ver()[0] or command_output(["sw_vers", "-productVersion"]),
        "os_build": command_output(["sw_vers", "-buildVersion"]),
        "hardware_model": command_output(["sysctl", "-n", "hw.model"]),
        "python_version": platform.python_version(),
        "available_compute_devices": devices,
    }


def command_output(command):
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"Host identity command returned no value: {' '.join(command)}")
    return value


def artifact_digest(path):
    if path.is_file():
        return hash_file(path)
    if path.is_dir() and path.suffix.lower() == ".mlpackage":
        return deepbom_package_digest(path)
    raise ValueError("--model must be a .mlmodel file or .mlpackage directory")


def deepbom_package_digest(package):
    manifest_path = package / "Manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fileFormatVersion") != "1.0.0":
        raise ValueError("Unsupported Core ML package Manifest.json version")
    entries = manifest.get("itemInfoEntries")
    root_id = manifest.get("rootModelIdentifier")
    if not isinstance(entries, dict) or root_id not in entries:
        raise ValueError("Core ML package manifest has no resolved root model")
    package_prefix = package.name + "/"
    required = {package_prefix + "Manifest.json": ("package_manifest", True)}
    for identifier, item in entries.items():
        item_root = package / "Data" / pathlib.PurePosixPath(item["path"])
        physical = [item_root] if item_root.is_file() else sorted(row for row in item_root.rglob("*") if row.is_file())
        if not physical:
            raise ValueError(f"Core ML package item {identifier} has no files")
        root_models = [row for row in physical if row.suffix.lower() == ".mlmodel"] if identifier == root_id else []
        if identifier == root_id and len(root_models) != 1:
            raise ValueError("Core ML package root item does not contain exactly one .mlmodel")
        for file_path in physical:
            relative = package_prefix + file_path.relative_to(package).as_posix()
            role = "root_model" if identifier == root_id and file_path in root_models else (
                "weights" if "weight" in f"{item.get('name', '')} {item.get('description', '')} {relative}".lower() else "package_item"
            )
            required[relative] = (role, True)
    rows = []
    for file_path in sorted(row for row in package.rglob("*") if row.is_file()):
        relative = package_prefix + file_path.relative_to(package).as_posix()
        role, is_required = required.get(relative, ("unreferenced_package_file", False))
        rows.append({"path": relative, "byte_length": file_path.stat().st_size, "sha256": hash_file(file_path), "role": role, "required": is_required})
    body = {"schema": "deepbom.artifact_bundle_digest.v1", "files": sorted(rows, key=lambda item: item["path"])}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def directory_digest(path, schema):
    rows = []
    for file_path in sorted(row for row in path.rglob("*") if row.is_file()):
        rows.append({"path": file_path.relative_to(path).as_posix(), "byte_length": file_path.stat().st_size, "sha256": hash_file(file_path)})
    return hashlib.sha256(canonical_json({"schema": schema, "files": rows})).hexdigest()


def canonical_json(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sha(value, label):
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return text


if __name__ == "__main__":
    main()
