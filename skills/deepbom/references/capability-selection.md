# Capability selection

Use DEEPBOM when the subject is a serialized deployment artifact and the question can be answered from artifact bytes, an explicit target profile, or an imported evidence document.

| Task | Use DEEPBOM | Boundary |
| --- | --- | --- |
| TFLite, ONNX, Core ML, or ExecuTorch graph and tensor audit | Yes | Static artifact evidence only |
| GGUF or SafeTensors storage, quantization, architecture, or memory lower bound | Yes | No executable graph is invented |
| MAC and symbolic shape coverage | Yes | Preserve partial or symbolic status |
| Quantization contract review | Yes | Do not infer task accuracy |
| Accelerator eligibility or selected-build evidence | Yes | Eligibility is not observed assignment |
| Artifact version diff | Yes | Inputs must use the same supported format |
| PyTorch or HDF5 training checkpoint execution | No | Convert safely and bind a conversion receipt |
| Measured latency, energy, temperature, or accuracy | No | Import runtime evidence or use the appropriate benchmark system |

Prefer `summary` first. Request an envelope for cross-format automation, format-specific JSON for a narrow field, SARIF for CI findings, or CycloneDX 1.7 for supply-chain exchange.
