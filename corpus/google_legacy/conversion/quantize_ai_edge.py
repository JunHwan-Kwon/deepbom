import argparse
from pathlib import Path

from repro_common import (
    base_record,
    load_function,
    read_recipe,
    require_file,
    resolve_recipe_path,
    sha256_file,
    write_record,
)


SCHEMA = "deepbom.ai_edge_quantizer_recipe.v1"


def main():
    parser = argparse.ArgumentParser(
        description="Run a provenance-recorded AI Edge Quantizer recipe."
    )
    parser.add_argument("recipe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    recipe_path, recipe = read_recipe(args.recipe, SCHEMA)
    source, source_sha = require_file(
        recipe_path, recipe["source_model"], "source FP32 TFLite model"
    )
    recipe_module, recipe_module_sha = require_file(
        recipe_path, recipe["quantization_recipe"], "quantization recipe module"
    )
    calibration_manifest = None
    calibration_sha = None
    if recipe.get("requires_calibration", False):
        if not recipe.get("calibration_manifest"):
            raise ValueError(
                "Static/selective calibration requires a pinned calibration_manifest."
            )
        calibration_manifest, calibration_sha = require_file(
            recipe_path, recipe["calibration_manifest"], "calibration manifest"
        )
    output = resolve_recipe_path(recipe_path, recipe["output"])
    record_path = resolve_recipe_path(
        recipe_path, recipe.get("record", f"{recipe['output']}.provenance.json")
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    from ai_edge_quantizer import quantizer

    factory = load_function(
        recipe_module, recipe["quantization_recipe"]["function"]
    )
    quantization_recipe = factory(
        **recipe["quantization_recipe"].get("kwargs", {})
    )
    instance = quantizer.Quantizer(str(source))
    instance.load_quantization_recipe(quantization_recipe)
    instance.quantize().export_model(str(output))
    output_sha = sha256_file(output)
    expected = recipe.get("expected_output_sha256")
    if expected and output_sha != expected:
        output.unlink(missing_ok=True)
        raise ValueError(
            f"Output SHA-256 mismatch: expected {expected}, observed {output_sha}."
        )

    record = base_record(
        recipe_path,
        sha256_file(recipe_path),
        ["ai-edge-quantizer", "tensorflow", "ai-edge-litert"],
    )
    record.update(
        {
            "converter": "ai_edge_quantizer",
            "source": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": source_sha,
            },
            "quantization_recipe": {
                "module_path": str(recipe_module),
                "module_sha256": recipe_module_sha,
                "function": recipe["quantization_recipe"]["function"],
                "kwargs": recipe["quantization_recipe"].get("kwargs", {}),
                "requires_calibration": bool(
                    recipe.get("requires_calibration", False)
                ),
                "calibration_manifest_path": (
                    str(calibration_manifest) if calibration_manifest else None
                ),
                "calibration_manifest_sha256": calibration_sha,
            },
            "output": {
                "path": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": output_sha,
            },
            "quality": {
                "status": "not_assessed",
                "reason": "Record task-specific FP32-versus-quantized quality metrics before admitting this artifact to a paper cohort.",
            },
        }
    )
    write_record(record_path, record)
    print(f"Wrote {output}")
    print(f"Wrote {record_path}")


if __name__ == "__main__":
    main()
