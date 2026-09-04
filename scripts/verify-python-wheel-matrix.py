from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


EXPECTED_DEFAULT = {
    "win32-x64", "win32-arm64",
    "linux-x64", "linux-arm64",
    "darwin-x64", "darwin-arm64",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the complete DEEPBOM Python wheel matrix.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected", default=",".join(sorted(EXPECTED_DEFAULT)))
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    expected = {value.strip() for value in args.expected.split(",") if value.strip()}
    wheels = sorted(args.directory.rglob("deepbom-*.whl"))
    if not wheels:
        raise SystemExit("No DEEPBOM wheels were found.")
    package_version = json.loads(Path("package.json").read_text(encoding="utf-8"))["version"]
    observed = {}
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            manifest_name = "deepbom/_engine/manifest.json"
            if manifest_name not in names:
                raise ValueError(f"{wheel.name}: packaged engine manifest is missing")
            manifest = json.loads(archive.read(manifest_name), object_pairs_hook=unique_object)
            identity = f"{manifest.get('platform')}-{manifest.get('arch')}"
            if identity in observed:
                raise ValueError(f"duplicate wheel identity {identity}: {wheel.name} and {observed[identity]}")
            if manifest.get("schema") != "deepbom.packaged_engine.v1" or manifest.get("version") != package_version:
                raise ValueError(f"{wheel.name}: engine schema/version mismatch")
            if args.require_clean and manifest.get("source", {}).get("git_state") != "clean":
                raise ValueError(f"{wheel.name}: dirty source state is not publishable")
            validate_platform_tag(wheel.name, identity)
            validate_artifact(archive, manifest["executable"], "deepbom/_engine/")
            validate_artifact(archive, manifest["tflite_wasm"], "deepbom/_engine/")
            validate_artifact(archive, manifest["self_test"], "deepbom/_engine/")
            validate_record(archive, wheel.name)
            observed[identity] = wheel.name

    actual = set(observed)
    if actual != expected:
        raise ValueError(f"wheel matrix mismatch: expected {sorted(expected)}, observed {sorted(actual)}")
    print(json.dumps({"status": "pass", "version": package_version, "wheel_count": len(wheels), "matrix": observed}, indent=2))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def validate_platform_tag(filename: str, identity: str) -> None:
    patterns = {
        "win32-x64": r"-win_amd64\.whl$",
        "win32-arm64": r"-win_arm64\.whl$",
        "linux-x64": r"-manylinux_2_28_x86_64\.whl$",
        "linux-arm64": r"-manylinux_2_28_aarch64\.whl$",
        "darwin-x64": r"-macosx_[0-9_]+_x86_64\.whl$",
        "darwin-arm64": r"-macosx_[0-9_]+_arm64\.whl$",
    }
    if identity not in patterns or not re.search(patterns[identity], filename):
        raise ValueError(f"{filename}: wheel tag does not match {identity}")


def validate_artifact(archive: zipfile.ZipFile, record: dict, prefix: str) -> None:
    relative = str(record.get("path", "")).replace("\\", "/")
    name = f"{prefix}{relative}"
    if name not in archive.namelist():
        raise ValueError(f"wheel is missing manifest artifact {name}")
    data = archive.read(name)
    if len(data) != record.get("byte_length") or sha256(data) != record.get("sha256"):
        raise ValueError(f"wheel artifact does not match manifest: {name}")


def validate_record(archive: zipfile.ZipFile, filename: str) -> None:
    record_names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
    if len(record_names) != 1:
        raise ValueError(f"{filename}: expected exactly one RECORD")
    rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
    by_name = {row[0]: row for row in rows}
    for name in archive.namelist():
        row = by_name.get(name)
        if not row:
            raise ValueError(f"{filename}: RECORD omits {name}")
        if name == record_names[0]:
            continue
        data = archive.read(name)
        if row[1] != f"sha256={record_digest(data)}" or row[2] != str(len(data)):
            raise ValueError(f"{filename}: RECORD mismatch for {name}")


if __name__ == "__main__":
    main()
