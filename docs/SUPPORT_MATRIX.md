# DEEPBOM Support Matrix

This matrix describes evidence that DEEPBOM can derive from serialized artifacts and explicitly imported, hash-bound sidecars. It is not a runtime, task-accuracy, clinical-validity, or release-readiness claim.

| Format or evidence | Artifact and graph | Numerical contract | Deployment evidence | Runtime boundary |
| --- | --- | --- | --- | --- |
| TFLite / LiteRT | Serialized graph, tensors, subgraphs, memory and arithmetic ledgers | Affine quantization, stored constants, integer arithmetic checks | CPU cost profile; source-pinned XNNPACK, GPU, NNAPI, Core ML, and Qualcomm QNN candidates; selected-build or compiled-plan evidence when imported | Actual partition, transfer, and latency require runtime evidence |
| ONNX | Recursive graph, initializer, shape, type, and symbolic cost contracts | Q/DQ, initializer, sparse and external-data contracts | Source-pinned ORT EP eligibility and TensorRT build/parser/engine evidence | Actual EP assignment and physical movement require runtime evidence |
| Core ML | NeuralNetwork, ML Program, classical and package contracts | Stored weight, compression, interface and deployment-floor evidence | Hash-bound MLComputePlan import is `compiled_plan` anticipated evidence | MLComputePlan is not executed CPU/GPU/ANE assignment or timing |
| GGUF | Header, tensor directory, architecture and tokenizer contracts; no serialized executable DAG | Encoding inventory and supported payload integrity | Hash-bound runtime/build manifests and lower-bound memory scenarios | Lowering, layer placement, private workspace and execution remain external |
| SafeTensors | Tensor directory plus hash-bound configuration/package closure; no serialized executable DAG | Storage, packed quantization and architecture-specific contracts where declared and decodable | TensorRT-LLM and memory-profile sidecars when identity-bound | Runtime graph, allocator behavior and execution remain external |
| ExecuTorch PTE/PTD | Program/container, selected operator and external segment contracts | Stored tensor and planned-memory evidence where represented | Selected-build operator/backend attestation | Delegate internals and execution require runtime evidence |
| Edge TPU compiler evidence | Original TFLite graph plus exact compiler operation ledger | Preserves the TFLite numerical contract | `compiled_plan` only when compiler binary, invocation, report and compiled artifact are hash-bound | Device execution, transfer and latency remain unobserved |
| LiteRT Qualcomm evidence | Original TFLite graph plus exact compiler operation ledger | Preserves the TFLite numerical contract | `compiled_plan` only when LiteRT source/rulepack, compiler binary, normalized invocation and QNN plan are hash-bound | Dispatch state is recorded separately; per-source-op execution, transfer and latency remain unobserved |
| NVIDIA host profile | Artifact plus observed device/software/VRAM inventory | No new artifact numerical claim | Host capability only; selected build remains unbound | No fit, tactic, assignment or execution claim |

## Evidence Stages

Accelerator evidence uses one ordered vocabulary without implying promotion between stages:

1. `serialized_artifact`
2. `source_eligibility`
3. `selected_build`
4. `compiled_plan`
5. `observed_assignment`
6. `measured_execution`

Evidence at one stage does not establish any later stage. CPU cost profiles remain separate planning inputs and are never interpreted as accelerator targets or host detection.

Independent placement profiles can be compared with `deepbom placement <artifact>` or the Web N-way selector. The comparison conserves one graph ledger but does not infer backend priority or combine profiles into a fictitious multi-delegate partition.

## Explicit Non-Claims

- Static eligibility is not observed delegation or provider assignment.
- Logical boundary payload is not physical bus traffic, zero-copy behavior, synchronization cost, or latency.
- A lower-bound memory scenario above capacity can prove insufficiency under that scenario; a value below capacity does not prove fit.
- Model-file evidence does not establish source-data preprocessing, task accuracy, clinical utility, safety/effectiveness, or release readiness.
