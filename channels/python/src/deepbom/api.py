"""Public Python SDK over the verified DEEPBOM engine.

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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

from ._engine import _verified_engine

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class DeepBomError(RuntimeError):
    """Base class for Python-facade failures."""

    code = "DEEPBOM_ERROR"

    def __init__(self, message: str, *, exit_code: Optional[int] = None,
                 document: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.document = document


class DeepBomInvocationError(DeepBomError):
    """The CLI rejected the invocation or could not analyze/write output."""
    code = "INVOCATION_FAILED"


class DeepBomPolicyBlocked(DeepBomError):
    """Analysis completed, but the requested policy blocked it (exit 2)."""
    code = "POLICY_BLOCKED"


class DeepBomIncompleteBinding(DeepBomError):
    """Verification could not establish a complete release binding (exit 3)."""
    code = "INCOMPLETE_BINDING"


class DeepBomIdentityMismatch(DeepBomError):
    """An independently supplied artifact SHA-256 did not match (exit 4)."""
    code = "IDENTITY_MISMATCH"


class DeepBomTimeout(DeepBomError):
    """The verified engine exceeded the caller's timeout."""
    code = "TIMEOUT"


class DeepBomOutputTooLarge(DeepBomError):
    """The JSON result exceeded the caller's output-size bound."""
    code = "OUTPUT_TOO_LARGE"


def capabilities(*, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                 max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return ``deepbom.cli_capabilities.v1`` from the installed engine."""
    return _invoke_json(["capabilities", "--compact"], timeout_seconds, max_output_bytes)


def audit(path: os.PathLike[str] | str, *, output: Optional[str] = None,
          scan: str = "auto", sections: Optional[Iterable[str]] = None,
          gate: Optional[str] = None, policy: Optional[str] = None,
          expected_sha256: Optional[str] = None,
          timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
          max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Audit one artifact and return a parsed JSON document.

    ``output`` is ``analysis``, ``envelope``, ``cyclonedx``, or ``sarif``.
    Omitting it selects ``analysis`` when ``sections`` are requested and the
    canonical ``envelope`` otherwise. ``gate`` accepts ``defects``; ``policy``
    accepts the built-in ``engineering`` or ``regulatory`` profile. The two
    policy sources are mutually exclusive.
    """
    scan = _choice(scan, "scan", {"auto", "structure", "integrity", "full"})
    section_values = _sections(sections)
    resolved_output = "analysis" if output is None and section_values else "envelope" if output is None else _choice(
        output, "output", {"analysis", "envelope", "cyclonedx", "sarif"}
    )
    if section_values and resolved_output != "analysis":
        raise ValueError("sections require output='analysis'")
    gate_value = None if gate is None else _choice(gate, "gate", {"defects"})
    policy_value = None if policy is None else _choice(policy, "policy", {"engineering", "regulatory"})
    if gate_value and policy_value:
        raise ValueError("gate and policy are mutually exclusive")
    argv = ["audit", _path_text(path), "--scan", scan]
    if section_values:
        argv.extend(["--section", ",".join(section_values)])
    if resolved_output == "analysis":
        argv.append("--compact")
    else:
        argv.extend(["--output-format", resolved_output, "--compact"])
    if gate_value:
        argv.extend(["--gate", gate_value])
    if policy_value:
        argv.extend(["--policy", policy_value])
    if expected_sha256 is not None:
        argv.extend(["--expected-sha256", _sha256_text(expected_sha256)])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def numerical_evidence(path: os.PathLike[str] | str, *, weights: bool = False,
                       weight_baseline: Optional[os.PathLike[str] | str] = None,
                       weight_options: Optional[os.PathLike[str] | str] = None,
                       weight_mapping: Optional[os.PathLike[str] | str] = None,
                       activation_capture: Optional[os.PathLike[str] | str] = None,
                       activation_baseline: Optional[os.PathLike[str] | str] = None,
                       expected_sha256: Optional[str] = None,
                       timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                       max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Inspect weights or import captures through the common engine; never execute a model.

    Returns an analysis selection containing Model IR, numerical detail and the
    requested source documents. A reference capture must bind to the same model.
    """
    if not isinstance(weights, bool):
        raise ValueError("weights must be boolean")
    inspect_weights = weights or any(value is not None for value in (weight_baseline, weight_options, weight_mapping))
    if not inspect_weights and activation_capture is None:
        raise ValueError("Select weights or provide activation_capture")
    if activation_baseline is not None and activation_capture is None:
        raise ValueError("activation_baseline requires activation_capture")
    if weight_mapping is not None and weight_baseline is None:
        raise ValueError("weight_mapping requires weight_baseline")
    argv = ["audit", _path_text(path), "--compact"]
    sections = ["model_ir", "numerical_evidence_bundle", "numerical_details"]
    if inspect_weights:
        argv.append("--weight-analysis")
        sections.extend(["weight_ir", "weight_analysis"])
    if weight_baseline is not None:
        sections.append("weight_comparison")
    if activation_capture is not None:
        sections.append("activation_ir")
    if activation_baseline is not None:
        sections.append("activation_baseline_ir")
    for flag, value in (("--weight-baseline", weight_baseline), ("--weight-options", weight_options),
                        ("--weight-mapping", weight_mapping), ("--activation-evidence", activation_capture),
                        ("--activation-baseline", activation_baseline)):
        if value is not None:
            argv.extend([flag, _path_text(value)])
    if expected_sha256 is not None:
        argv.extend(["--expected-sha256", _sha256_text(expected_sha256)])
    argv.extend(["--section", ",".join(sections)])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def inspect(path: os.PathLike[str] | str, *, scan: str = "auto",
            expected_sha256: Optional[str] = None,
            gate: Optional[str] = None, policy: Optional[str] = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
            max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return the engine's existing ``deepbom.review_summary.v1`` unchanged.

    This is a bounded static summary, not deployment approval. Policy rejection
    retains this same summary shape in ``DeepBomPolicyBlocked.document``.
    """
    try:
        selection = audit(path, sections=["summary"], scan=scan,
                          expected_sha256=expected_sha256, gate=gate, policy=policy,
                          timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
    except DeepBomPolicyBlocked as error:
        if error.document is not None:
            error.document = _review_summary(error.document)
        raise
    return _review_summary(selection)


def _review_summary(selection: dict[str, Any]) -> dict[str, Any]:
    sections = selection.get("sections")
    summary = sections.get("summary") if isinstance(sections, dict) else None
    if (selection.get("schema") != "deepbom.analysis_selection.v1"
            or not isinstance(summary, dict) or summary.get("schema") != "deepbom.review_summary.v1"):
        raise DeepBomInvocationError("The engine returned an incompatible review summary.")
    return summary


def model_ir(path: os.PathLike[str] | str, *, scan: str = "auto",
             expected_sha256: Optional[str] = None,
             timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
             max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return the preview ``deepbom.model_ir.v1`` projection.

    The result remains static engineering evidence. It does not imply runtime
    execution order, standards conformance, or regulatory acceptance.
    """
    selection = audit(
        path,
        scan=scan,
        sections=["model_ir"],
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    document = selection.get("sections", {}).get("model_ir")
    if not isinstance(document, dict) or document.get("schema") != "deepbom.model_ir.v1":
        raise DeepBomInvocationError("The engine returned an incompatible Model IR document.")
    return document


def visualization_manifest(path: os.PathLike[str] | str, *,
                           views: Optional[Iterable[str]] = None,
                           orientation: str = "portrait",
                           timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                           max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return the deterministic Model IR visualization manifest as JSON.

    SVG/PNG/Word bundle bytes are intentionally written by the CLI or Web
    export surface; this facade returns the bounded manifest for automation.
    """
    allowed_views = {
        "identity-boundary", "architecture-overview", "block-detail",
        "exhaustive", "static-runtime", "observed-runtime",
    }
    view_values = _sections(views)
    if any(value not in allowed_views for value in view_values):
        raise ValueError(f"views must be selected from: {', '.join(sorted(allowed_views))}")
    orientation_value = _choice(orientation, "orientation", {"portrait", "landscape"})
    argv = ["visualize", _path_text(path), "--view", ",".join(view_values) if view_values else "all",
            "--orientation", orientation_value, "--output-format", "json", "--compact"]
    document = _invoke_json(argv, timeout_seconds, max_output_bytes)
    if document.get("schema") != "deepbom.model_ir_visualization_manifest.v1":
        raise DeepBomInvocationError("The engine returned an incompatible Model IR visualization manifest.")
    return document


def tensors(path: os.PathLike[str] | str, *, expected_sha256: Optional[str] = None,
            timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
            max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> list[dict[str, Any]]:
    """Return Python-native rows from the bounded GGUF tensor table.

    Exact counts are converted to :class:`int`; exact decimal ratios are
    converted to :class:`~decimal.Decimal`. Use :func:`tensor_inventory` when
    the raw JSON-compatible ``deepbom.tensor_table.v1`` document is required.
    """
    document = tensor_inventory(
        path,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    rows = document.get("tensors")
    if document.get("schema") != "deepbom.tensor_table.v1" or not isinstance(rows, list):
        raise DeepBomInvocationError("The engine returned an incompatible tensor-table document.")
    return [_python_tensor_row(row) for row in rows]


def _python_tensor_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DeepBomInvocationError("The engine returned a non-object tensor-table row.")
    result = dict(row)
    if isinstance(result.get("element_count"), str):
        try:
            result["element_count"] = int(result["element_count"])
        except ValueError as error:
            raise DeepBomInvocationError("The engine returned an invalid exact tensor element count.") from error
    if isinstance(result.get("effective_bits_per_element"), str):
        try:
            result["effective_bits_per_element"] = Decimal(result["effective_bits_per_element"])
        except InvalidOperation as error:
            raise DeepBomInvocationError("The engine returned an invalid exact tensor bit ratio.") from error
    return result


def tensor_inventory(path: os.PathLike[str] | str, *, expected_sha256: Optional[str] = None,
                     timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                     max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Return the complete bounded ``deepbom.tensor_table.v1`` document."""
    argv = ["gguf", _path_text(path), "--tensors", "--compact"]
    if expected_sha256 is not None:
        argv.extend(["--expected-sha256", _sha256_text(expected_sha256)])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def verify_bom(path: os.PathLike[str] | str, bom: os.PathLike[str] | str, *,
               component_ref: Optional[str] = None,
               timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
               max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Reconcile one CycloneDX 1.7 component with artifact-observable facts."""
    argv = ["verify", _path_text(path), "--bom", _path_text(bom), "--compact"]
    if component_ref is not None:
        if not str(component_ref).strip():
            raise ValueError("component_ref must be a non-empty bom-ref")
        argv.extend(["--component-ref", str(component_ref).strip()])
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def verify_contract(path: os.PathLike[str] | str, contract: os.PathLike[str] | str, *,
                    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Compare a serialized external interface with an explicit contract or artifact baseline."""
    return _invoke_json(
        ["verify", _path_text(path), "--contract", _path_text(contract), "--compact"],
        timeout_seconds,
        max_output_bytes,
    )


def capture_contract(path: os.PathLike[str] | str, *,
                     timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                     max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Capture an artifact-derived external-interface baseline; this is not an approved declaration."""
    return _invoke_json(["contract", "capture", _path_text(path), "--compact"], timeout_seconds, max_output_bytes)


def diff(baseline: os.PathLike[str] | str, candidate: os.PathLike[str] | str, *,
         tensors_only: bool = False,
         timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
         max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> dict[str, Any]:
    """Compare two same-format artifacts, optionally returning the tensor-encoding projection."""
    if not isinstance(tensors_only, bool):
        raise ValueError("tensors_only must be boolean")
    argv = ["diff", _path_text(baseline), _path_text(candidate), "--compact"]
    if tensors_only:
        argv.append("--tensors")
    return _invoke_json(argv, timeout_seconds, max_output_bytes)


def _invoke_json(argv: list[str], timeout_seconds: float, max_output_bytes: int) -> dict[str, Any]:
    timeout = _positive_number(timeout_seconds, "timeout_seconds")
    output_limit = _positive_integer(max_output_bytes, "max_output_bytes")
    try:
        engine, asset_root = _verified_engine()
    except (OSError, RuntimeError) as error:
        raise DeepBomInvocationError(f"Could not resolve the verified DEEPBOM engine: {error}") from error
    environment = os.environ.copy()
    if asset_root is not None:
        environment["DEEPBOM_RUNTIME_ASSET_DIR"] = str(asset_root)

    # A temporary output permits a size check before Python allocates the
    # complete JSON string. The engine writes it atomically.
    try:
        return _run_engine_json(engine, environment, argv, timeout, output_limit)
    except DeepBomError:
        raise
    except OSError as error:
        raise DeepBomInvocationError(f"Could not create, read or clean up DEEPBOM temporary output: {error}") from error


def _run_engine_json(engine, environment, argv, timeout, output_limit):
    with tempfile.TemporaryDirectory(prefix="deepbom-python-") as directory, tempfile.TemporaryFile() as diagnostic_stream:
        output_path = Path(directory) / "result.json"
        command = [str(engine), *argv, "--output", str(output_path)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=diagnostic_stream,
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
        except OSError as error:
            raise DeepBomInvocationError(f"Could not start the verified DEEPBOM engine: {error}") from error

        diagnostic_stream.seek(0, os.SEEK_END)
        diagnostic_stream.seek(max(0, diagnostic_stream.tell() - 16_384))
        diagnostic = diagnostic_stream.read().decode("utf-8", errors="replace").strip()
        try:
            document = _read_bounded_json(output_path, output_limit)
        except DeepBomError as error:
            error.exit_code = completed.returncode
            raise
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
        if completed.returncode == 4:
            raise DeepBomIdentityMismatch(
                diagnostic or "DEEPBOM observed an artifact identity mismatch.",
                exit_code=4,
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
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise DeepBomOutputTooLarge(f"DEEPBOM result grew beyond max_output_bytes={maximum}.")
        document = json.loads(raw.decode("utf-8"), parse_constant=_invalid_json_constant, parse_float=_finite_json_float)
    except (OSError, UnicodeError, ValueError) as error:
        raise DeepBomInvocationError(f"DEEPBOM returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise DeepBomInvocationError("DEEPBOM JSON result must be an object.")
    return document


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON number exceeds the finite floating-point range")
    return result


def _choice(value: str, field: str, allowed: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return normalized


def _path_text(value: os.PathLike[str] | str) -> str:
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise ValueError("path must be a non-empty path or immutable remote source")
    # Preserve the existing CLI remote-source path; local leading '-' names
    # must never be interpreted as engine options.
    return text if "://" in text else str(Path(text).absolute())


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
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value
