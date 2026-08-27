import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def read_recipe(filename, expected_schema):
    recipe_path = Path(filename).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("schema") != expected_schema:
        raise ValueError(
            f"Expected recipe schema {expected_schema}, got {recipe.get('schema')!r}."
        )
    return recipe_path, recipe


def resolve_recipe_path(recipe_path, value):
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = recipe_path.parent / candidate
    return candidate.resolve()


def require_file(recipe_path, descriptor, label):
    filename = resolve_recipe_path(recipe_path, descriptor["path"])
    if not filename.is_file():
        raise FileNotFoundError(f"{label} does not exist: {filename}")
    expected = descriptor.get("sha256")
    observed = sha256_file(filename)
    if expected and observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}."
        )
    return filename, observed


def load_function(filename, function_name):
    module_name = f"deepbom_recipe_{sha256_file(filename)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load recipe module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"{filename}:{function_name} is not callable.")
    return function


def sha256_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def pip_freeze():
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def base_record(recipe_path, recipe_sha256, package_names):
    return {
        "schema": "deepbom.converter_run_record.v1",
        "status": "completed",
        "recipe": {
            "path": str(recipe_path),
            "sha256": recipe_sha256,
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(package_names),
            "pip_freeze": pip_freeze(),
            "environment_variables_recorded": [
                name
                for name in ["PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "TF_DETERMINISTIC_OPS"]
                if name in os.environ
            ],
        },
    }


def write_record(filename, record):
    output = Path(filename).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
