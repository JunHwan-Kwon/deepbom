import { onnxNodeSemantic } from "./onnx-portable.js";
const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const MAX_EXACT_SEQUENCE_ELEMENTS = 4_096;

export const ONNX_CONTAINER_VALUE_OPS = new Set([
  "ConcatFromSequence", "Optional", "OptionalGetElement", "OptionalHasElement", "SequenceAt",
  "SequenceConstruct", "SequenceEmpty", "SequenceErase", "SequenceInsert", "SequenceLength",
  "SplitToSequence",
]);

export const ONNX_CONTAINER_VALUE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  documents: Object.freeze([
    Object.freeze({ role: "current_sequence_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/defs.cc`, sha256: "a57eb7cd7d58b70f26c561aad318c7998a11ffcf7bf03675fcec1a7cc7efabaa" }),
    Object.freeze({ role: "historical_sequence_schema", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/old.cc`, sha256: "f0815ecb7ce2ba994826e521bcbedb012229025f46e80b80e1e7edbee9a13643" }),
    Object.freeze({ role: "split_to_sequence_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/sequence/utils.cc`, sha256: "7c3ec816bc676a5d7c0213ae1d4ec225446608cf87ab930e39987a6c087dfce8" }),
    Object.freeze({ role: "optional_inference", source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/optional/defs.cc`, sha256: "d9e8bacff27ab70539a9cd5c84646bb7b384e6116c3f529379814f4f88e9effb" }),
  ]),
});

export function inferOnnxContainerNode({node,tensorMap,tensorTypeName,nodeIndex,importedOpset,scope="main_graph"}) {
  const a=node.attributes?.get("dtype");
  const encoded=a?.i??(a?.iExactDecimal == null ? 1:Number(a.iExactDecimal));
  return onnxNodeSemantic("container",node,tensorMap,{node_index:nodeIndex,imported_opset:importedOpset,scope,empty_dtype:tensorTypeName(encoded)});
}
