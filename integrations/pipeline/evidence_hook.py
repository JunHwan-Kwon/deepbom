"""Generic DeepBOM evidence hook for experiment and release pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from deepbom import audit


def record_deepbom_evidence(
    artifact: str | Path,
    destination: str | Path,
    attach: Callable[[Path], None] | None = None,
) -> dict:
    """Audit locally, persist deterministic JSON, then optionally attach it.

    ``attach`` is supplied by the caller's tracking or registry client. DeepBOM
    does not select a vendor, transmit model bytes, or claim that attachment
    implies approval.
    """

    result = audit(artifact, output="envelope")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if attach is not None:
        attach(target)
    return result
