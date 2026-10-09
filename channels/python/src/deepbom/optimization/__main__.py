"""Explicit local native entry points; selected factories execute local code."""

from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
from . import optimize, load_model, load_evidence, compare, suggest
from .._native.core import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("optimize", "suggest"):
        p = sub.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--factory", help="Trusted local module:function returning a built model"
        )
        source.add_argument(
            "--keras", help="Trusted local .keras file (compile=False, safe_mode=True)"
        )
        source.add_argument("--candidate", help="Previously selected model package")
        p.add_argument(
            "--objective", choices=["structure", "cpu_inference"], default="structure"
        )
        p.add_argument(
            "--state-file",
            help="PyTorch weights-only state dictionary; requires --factory",
        )
        if name == "optimize":
            p.add_argument(
                "--rules",
                help="JSON file containing an ordered rule list; omit to propose automatically",
            )
            p.add_argument(
                "--inputs",
                help="JSON {args:[{shape:[...],dtype:float32}],kwargs:{...}} for independent input/output shape probes and synthetic forward/backward validation",
            )
            p.add_argument("--output", required=True)
            p.add_argument("--advanced", action="store_true")
            p.add_argument(
                "--rationale",
                help="Optional user-stated reason, recorded separately from rule semantics",
            )
            p.add_argument(
                "--report-format",
                choices=["html", "pdf", "json", "all"],
                default="html",
                help="HTML and JSON are always saved; pdf/all additionally writes the monochrome PDF",
            )
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p = sub.add_parser(
        "report",
        help="Export an official monochrome report from a saved package, without executing model code",
    )
    p.add_argument("candidate")
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["html", "pdf", "json"])
    p.add_argument(
        "--baseline",
        help="Original baseline package, required only for older packages without report-source.json",
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        from .report import export_report

        path = export_report(
            args.candidate, args.output, format=args.format, baseline=args.baseline
        )
        print(json.dumps({"report": str(path)}, indent=2))
        return 0
    if args.command == "compare":
        print(json.dumps(compare(args.baseline, args.candidate), indent=2))
        return 0
    if args.command == "optimize" and args.report_format in ("pdf", "all"):
        from .report import require_pdf

        require_pdf()
    if args.factory:
        module, name = args.factory.split(":", 1)
        model = getattr(importlib.import_module(module), name)()
    elif args.keras:
        import keras

        model = keras.models.load_model(args.keras, compile=False, safe_mode=True)
    else:
        model = load_model(args.candidate)
    if args.state_file:
        if not args.factory:
            raise ValueError("--state-file requires a trusted --factory")
        from .._native.adapters import framework_of

        if framework_of(model) != "pytorch":
            raise ValueError("--state-file is the PyTorch weights-only path")
        import torch

        model.load_state_dict(
            torch.load(args.state_file, map_location="cpu", weights_only=True),
            strict=True,
        )
    if args.command == "suggest":
        print(json.dumps(suggest(model, objective=args.objective), indent=2))
        return 0
    examples = []
    kwargs = {}
    if args.inputs:
        spec = read_json(args.inputs)
        if set(spec) - {"args", "kwargs"}:
            raise ValueError("Unknown input specification key")
        from .._native.adapters import framework_of

        fw = framework_of(model)
        for tensor in spec.get("args", []):
            if set(tensor) != {"shape", "dtype"}:
                raise ValueError("Each synthetic input requires shape and dtype")
            if not isinstance(tensor["shape"], list) or any(
                type(d) is not int or d < 0 for d in tensor["shape"]
            ):
                raise ValueError("Invalid input dimensions")
            if fw == "pytorch":
                import torch

                allowed = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int64": torch.int64,
                    "int32": torch.int32,
                }
                examples.append(
                    torch.zeros(tensor["shape"], dtype=allowed[tensor["dtype"]])
                )
            else:
                import tensorflow as tf

                allowed = {
                    "float32": tf.float32,
                    "float64": tf.float64,
                    "int64": tf.int64,
                    "int32": tf.int32,
                }
                examples.append(
                    tf.zeros(tensor["shape"], dtype=allowed[tensor["dtype"]])
                )
        kwargs = spec.get("kwargs", {})
    result = optimize(
        model,
        objective=args.objective,
        rules=(
            read_json(args.rules, expected_type=list, label="Optimization rules")
            if args.rules
            else None
        ),
        example_args=tuple(examples),
        example_kwargs=kwargs,
        output_dir=args.output,
        factory=args.factory,
        advanced=args.advanced,
        rationale=args.rationale,
    )
    pdf = None
    if args.report_format in ("pdf", "all"):
        from .report import export_report

        pdf = str(export_report(args.output, Path(args.output) / "report.pdf"))
    print(
        json.dumps(
            {
                "snapshot_sha256": result.snapshot_sha256,
                "manifest": result.manifest_path,
                "report": result.report_path,
                "report_json": str(Path(args.output).resolve() / "report.json"),
                "report_pdf": pdf,
                "validation": result.validation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
