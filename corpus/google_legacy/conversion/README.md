# Reproducible Converter Harnesses

These scripts create future converter-generation cohorts. They are not evidence
that a conversion has already run.

Use a fresh Linux or macOS Python 3.10-3.13 environment. AI Edge Quantizer 0.8.0
does not advertise Windows as a supported operating system.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.direct.txt
```

The direct package pins and official distribution hashes are recorded in
`../converter-cohorts.v1.json`. Every run also records the complete resolved
`pip freeze`, Python/platform identity, recipe hash, source/checkpoint hash, and
output hash. A generated artifact is not admitted until its run record and
task-specific numerical parity or quality evidence are complete.

## LiteRT Torch

Create a JSON recipe with schema `deepbom.litert_torch_recipe.v1`. It must name
a local Python factory file and function, fixed example input shapes/dtypes, an
optional hash-pinned checkpoint, output path, and converter options.

```bash
python convert_litert_torch.py path/to/recipe.json
```

This v1 harness deliberately uses static example inputs. That is a study design
choice, not a claim that the underlying converter can never support dynamic
shape specifications.

## AI Edge Quantizer

Create a JSON recipe with schema `deepbom.ai_edge_quantizer_recipe.v1`. It must
pin the source FP32 `.tflite` and a Python recipe factory. A static or selective
recipe must additionally pin a calibration manifest containing sample hashes and
ordering.

```bash
python quantize_ai_edge.py path/to/recipe.json
```

The recipe factory is the authoritative per-operator scheme definition. DeepBOM
then measures the emitted artifact per op and per tensor; the coarse
`mixed_integer` class is only a routing summary.
