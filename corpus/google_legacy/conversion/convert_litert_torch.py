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


SCHEMA = "deepbom.litert_torch_recipe.v1"


def main():
    parser = argparse.ArgumentParser(
        description="Run a provenance-recorded LiteRT Torch conversion."
    )
    parser.add_argument("recipe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    recipe_path, recipe = read_recipe(args.recipe, SCHEMA)
    factory_file, factory_sha = require_file(
        recipe_path, recipe["model_factory"], "model factory"
    )
    checkpoint = None
    checkpoint_sha = None
    if recipe.get("checkpoint"):
        checkpoint, checkpoint_sha = require_file(
            recipe_path, recipe["checkpoint"], "checkpoint"
        )
    output = resolve_recipe_path(recipe_path, recipe["output"])
    record_path = resolve_recipe_path(
        recipe_path, recipe.get("record", f"{recipe['output']}.provenance.json")
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import litert_torch

    torch.manual_seed(int(recipe.get("seed", 0)))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=False)

    factory = load_function(factory_file, recipe["model_factory"]["function"])
    model = factory(**recipe["model_factory"].get("kwargs", {}))
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=bool(recipe.get("strict_state_dict", True)))
    model.eval()

    inputs = []
    for index, spec in enumerate(recipe["inputs"]):
        dtype = getattr(torch, spec["dtype"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(spec.get("seed", recipe.get("seed", 0))) + index)
        if spec.get("fill", "seeded_normal") == "zeros":
            tensor = torch.zeros(tuple(spec["shape"]), dtype=dtype)
        else:
            tensor = torch.randn(
                tuple(spec["shape"]), dtype=dtype, generator=generator
            )
        inputs.append(tensor)

    converted = litert_torch.convert(
        model,
        tuple(inputs),
        **recipe.get("converter_options", {}),
    )
    converted.export(str(output))
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
        ["litert-torch", "torch", "ai-edge-litert", "ai-edge-quantizer"],
    )
    record.update(
        {
            "converter": "litert_torch",
            "source": {
                "factory_path": str(factory_file),
                "factory_sha256": factory_sha,
                "factory_function": recipe["model_factory"]["function"],
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "checkpoint_sha256": checkpoint_sha,
            },
            "example_inputs": recipe["inputs"],
            "converter_options": recipe.get("converter_options", {}),
            "output": {
                "path": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": output_sha,
            },
            "parity": {
                "status": "not_assessed",
                "reason": "Run source-versus-LiteRT numerical parity before admitting this artifact to a paper cohort.",
            },
        }
    )
    write_record(record_path, record)
    print(f"Wrote {output}")
    print(f"Wrote {record_path}")


if __name__ == "__main__":
    main()
