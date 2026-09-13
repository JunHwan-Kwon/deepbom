# DeepBOM terminology and numeric contract

DeepBOM uses four related terms deliberately:

- **dtype** is the format-declared tensor element type or storage code.
- **encoding** is the format-neutral projection of that serialized dtype. It does not assert a runtime kernel or arithmetic path.
- **precision** describes the representational width or class derived from an encoding; it is not a task-quality measurement.
- **quantization** is used only when the artifact or a separately bound declaration supplies a quantization contract. A reduced-width encoding alone does not prove the quantizer, calibration data, or runtime execution mode.

`deepbom.tensor_table.v1` retains `encoding` for readability. Its assignment-signature basis explicitly projects that field to the canonical key `dtype`, so the hash is reproducible without implementation knowledge. GGUF shapes use `gguf_ne_order_innermost_first`; other formats retain their declared dimension order.

## Numeric values

Machine JSON preserves exact integers that can exceed the JavaScript safe-integer range as decimal strings. Fields with a `_decimal` suffix always follow this rule. Bounded counts and byte offsets that fit the safe-integer range are JSON numbers. Derived non-integral ratios are canonical decimal strings when exact cross-language reproduction is required.

The Python facade converts tensor-table `element_count` to `int` and `effective_bits_per_element` to `Decimal`. It preserves machine-document fields outside that typed facade exactly as emitted by the shared engine. Consumers should use the schema identifier and field contract rather than infer a type from a visually similar value elsewhere.

These conventions describe serialized evidence only. They do not establish runtime precision, numerical equivalence, accuracy, safety, quality, or regulatory status.
