"""Experimental typed Python facade over the verified DEEPBOM engine.

The analysis remains in the packaged engine. This module only constructs a
bounded subprocess invocation, reads one JSON result, and maps documented CLI
exit codes to Python exceptions.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from .__main__ import _verified_engine

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class DeepBomError(RuntimeError):
    """Base class for Python-facade failures."""

    def __init__(self, message: str, *, exit_code: Optional[int] = None,
                 document: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.document = document


class DeepBomInvocationError(DeepBomError):
    """The CLI rejected the invocation or could not analyze/write output."""


class DeepBomPolicyBlocked(DeepBomError):
    """Analysis completed, but the requested policy blocked it (exit 2)."""


class DeepBomIncompleteBinding(DeepBomError):
    """Verification could not establish a complete release binding (exit 3)."""


class DeepBomTimeout(DeepBomError):
    """The verified engine exceeded the caller's timeout."""


class DeepBomOutputTooLarge(DeepBomError):
    """The JSON result exceeded the caller's output-size bound."""


def capabilities(*, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return ``deepbom.cli_capabilities.v1`` from the installed engine."""
    return _invoke_json(["capabilities", "--compact"], timeout_seconds, max_output_bytes)


def audit(path: os.PathLike[str] | str, *, output: str = "envelope",
          scan: str = "auto", sections: Optional[Iterable[str]] = None,
          expected_sha256: Optional[str] = None,
          timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
          max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Audit one artifact and return a parsed JSON document.

    ``output`` is ``analysis``, ``envelope``, ``cyclonedx``, or ``sarif``.
    Section selection is available only with ``analysis`` output.
    """
    output = _choice(output, "output", {"analysis", "envelope", "cyclonedx", "sarif"})
    scan = _choice(scan, "scan", {"auto", "structure", "integrity", "full"})
    argv = ["audit", _path_text(path), "--scan", scan]
    section_values = _sections(sections)
    if section_values and output != "analysis":
        raise ValueError("sections require output='analysis'")
    if section_values:
        argv.extend(["--section", ",".join(section_values)])
    if output == "analysis":
        argv.append("--compact")
    else:
        argv.extend(["--output-format", output, "--compact"])
    if expected_sha256 is not None:
        argv.extend(["--expected-sha256", _sha256_text(expected_sha256)])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def tensors(path: os.PathLike[str] | str, *, expected_sha256: Optional[str] = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
            max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> list[dict[str, Any]]:
    """Return the bounded GGUF tensor table as a list of dictionaries."""
    document = tensor_inventory(
        path,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    rows = document.get("tensors")
    if document.get("schema") != "deepbom.tensor_table.v1" or not isinstance(rows, list):
        raise DeepBomInvocationError("The engine returned an incompatible tensor-table document.")
    return rows


def tensor_inventory(path: os.PathLike[str] | str, *, expected_sha256: Optional[str] = None,
                     timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                     max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return the complete bounded ``deepbom.tensor_table.v1`` document."""
    argv = ["gguf", _path_text(path), "--tensors", "--compact"]
    if expected_sha256 is not None:
        argv.extend(["--expected-sha256", _sha256_text(expected_sha256)])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def _invoke_json(argv: list[str], timeout_seconds: float, max_output_bytes: int) -> dict[str, Any]:
    timeout = _positive_number(timeout_seconds, "timeout_seconds")
    output_limit = _positive_integer(max_output_bytes, "max_output_bytes")
    engine, asset_root = _verified_engine()
    environment = os.environ.copy()
    if asset_root is not None:
        environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(asset_root)

    # A temporary output permits a size check before Python allocates the
    # complete JSON string. The engine writes it atomically.
    with tempfile.TemporaryDirectory(prefix="deepbom-python-") as directory:
        output_path = Path(directory) / "result.json"
        command = [str(engine), *argv, "--output", str(output_path)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=timeout,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as error:
            raise DeepBomTimeout(
                f"DEEPBOM exceeded timeout_seconds={timeout:g}.", exit_code=None
            ) from error

        document = _read_bounded_json(output_path, output_limit)
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        if completed.returncode == 0:
            if document is None:
                raise DeepBomInvocationError(
                    diagnostic or "DEEPBOM completed without a JSON result.", exit_code=0
                )
            return document
        if completed.returncode == 2:
            raise DeepBomPolicyBlocked(
                diagnostic or "DEEPBOM policy blocked the completed analysis.",
                exit_code=2,
                document=document,
            )
        if completed.returncode == 3:
            raise DeepBomIncompleteBinding(
                diagnostic or "DEEPBOM could not establish a complete binding.",
                exit_code=3,
                document=document,
            )
        raise DeepBomInvocationError(
            diagnostic or f"DEEPBOM exited {completed.returncode}.",
            exit_code=completed.returncode,
            document=document,
        )


def _read_bounded_json(path: Path, maximum: int) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size > maximum:
        raise DeepBomOutputTooLarge(
            f"DEEPBOM result is {size} bytes; max_output_bytes is {maximum}."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DeepBomInvocationError(f"DEEPBOM returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise DeepBomInvocationError("DEEPBOM JSON result must be an object.")
    return document


def _choice(value: str, field: str, allowed: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return normalized


def _path_text(value: os.PathLike[str] | str) -> str:
    text = os.fspath(value)
    if not text or "\x00" in text:
        raise ValueError("path must be a non-empty path or immutable remote source")
    return text


def _sections(values: Optional[Iterable[str]]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError("sections must be an iterable of section names, not one string")
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or not all(char.isalnum() or char in "_.-" for char in item):
            raise ValueError("section names may contain only letters, digits, underscore, dot, and hyphen")
        if item not in result:
            result.append(item)
    return result


def _sha256_text(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return normalized


def _positive_number(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value
