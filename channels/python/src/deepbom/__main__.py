"""Execute the packaged DEEPBOM engine without reimplementing analysis."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=reject_duplicates)
    if not isinstance(document, dict):
        raise ValueError("engine manifest must be a JSON object")
    return document


def _runtime_identity() -> tuple[str, str]:
    system = {"win32": "win32", "linux": "linux", "darwin": "darwin"}.get(sys.platform)
    machine = platform.machine().lower()
    architecture = {
        "amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"
    }.get(machine)
    if not system or not architecture:
        raise RuntimeError(f"Unsupported Python platform: {sys.platform}/{machine or 'unknown'}")
    return system, architecture


def _verified_engine() -> tuple[Path, Optional[Path]]:
    override = os.environ.get("DEEPBOM_ENGINE", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        expected = os.environ.get("DEEPBOM_ENGINE_SHA256", "").strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RuntimeError("DEEPBOM_ENGINE requires a 64-character DEEPBOM_ENGINE_SHA256 binding.")
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise RuntimeError("The supplied DEEPBOM_ENGINE does not match DEEPBOM_ENGINE_SHA256.")
        asset_root = Path(os.environ.get("DEEPBOM_RUNTIME_ASSET_DIR", candidate.parent / "pkg")).expanduser().resolve()
        return candidate, asset_root

    engine_root = Path(__file__).resolve().parent / "_engine"
    manifest_path = engine_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("The packaged DEEPBOM engine manifest is unavailable.")
    try:
        manifest = _strict_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The packaged DEEPBOM engine manifest is invalid: {error}") from error
    system, architecture = _runtime_identity()
    version_contract = manifest.get("version_contract")
    if (
        manifest.get("schema") != "deepbom.packaged_engine.v1"
        or not isinstance(version_contract, dict)
        or version_contract.get("python_version") != __version__
    ):
        raise RuntimeError("The packaged DEEPBOM engine manifest has an incompatible schema or version.")
    if manifest.get("platform") != system or manifest.get("arch") != architecture:
        raise RuntimeError(
            f"The packaged DEEPBOM engine targets {manifest.get('platform')}/{manifest.get('arch')}, "
            f"not {system}/{architecture}."
        )
    executable = manifest.get("executable")
    wasm = manifest.get("tflite_wasm")
    if not isinstance(executable, dict) or not isinstance(wasm, dict):
        raise RuntimeError("The packaged DEEPBOM engine manifest is missing artifact records.")
    filename = "deepbom-core.exe" if os.name == "nt" else "deepbom-core"
    if executable.get("filename") != filename or executable.get("path") != filename:
        raise RuntimeError("The packaged DEEPBOM executable identity is invalid.")
    candidate = engine_root / filename
    wasm_path = engine_root / "pkg" / "tflite_wasm_audit_bg.wasm"
    for label, artifact, record in (("engine", candidate, executable), ("TFLite WASM", wasm_path, wasm)):
        if not artifact.is_file() or artifact.stat().st_size != record.get("byte_length"):
            raise RuntimeError(f"The packaged DEEPBOM {label} size does not match its manifest.")
        if _sha256(artifact) != record.get("sha256"):
            raise RuntimeError(f"The packaged DEEPBOM {label} failed its SHA-256 check.")
    return candidate, wasm_path.parent


def main() -> int:
    try:
        engine, asset_root = _verified_engine()
        environment = os.environ.copy()
        if asset_root is not None:
            environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(asset_root)
        completed = subprocess.run([str(engine), *sys.argv[1:]], check=False, env=environment)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"deepbom: {error}", file=sys.stderr)
        return 2
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
