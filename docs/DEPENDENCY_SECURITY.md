# Dependency security record

DeepBOM separates runtime dependencies from development and validation dependencies. Release gates run `npm audit` and record all remaining findings.

## Current decisions

- `adm-zip` is pinned and overridden to `0.6.1`, which removes the known destination-symlink overwrite advisory affecting versions through `0.6.0`.
- `onnxruntime-node` remains pinned to `1.26.0` as a development-only measurement oracle. It is not shipped in npm, Python, Cargo, standalone, MCP bundle, or browser runtime artifacts. Moving the oracle to `1.29.x` requires a separate operator, quantization, and numerical-output compatibility experiment; a version bump is not treated as a mechanical security fix.
- Release acceptance requires zero high or critical npm audit findings. Any lower-severity residual must have a scoped, dated acceptance record and must not enter published runtime artifacts.
