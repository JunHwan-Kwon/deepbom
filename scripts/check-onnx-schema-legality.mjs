import { assessOnnxNodeSchemaForm } from "../web/lib/onnx-schema-legality.js";
import {
  ONNX_SHAPE_SCHEMA_FORMS,
  ONNX_SHAPE_SCHEMA_SOURCE,
} from "../web/lib/onnx-shape-schema-generated.js";
import { inferOnnxShapes } from "../web/lib/onnx-shape-inference.js";

expectEqual(ONNX_SHAPE_SCHEMA_SOURCE.op_count, 152, "generated supported-op count");
expectEqual(ONNX_SHAPE_SCHEMA_SOURCE.schema_version_count, 514, "generated schema-history count");
expectEqual(ONNX_SHAPE_SCHEMA_FORMS.size, 152, "generated map size");
expectEqual([...ONNX_SHAPE_SCHEMA_FORMS.values()].reduce((sum, forms) => sum + forms.length, 0), 514, "generated row conservation");

const attr = (type, fields = {}) => ({ type, ...fields });
const node = (opType, inputs, outputs = ["y"], attributes = []) => ({
  opType,
  domain: "",
  inputs,
  outputs,
  attributes: new Map(attributes),
});
const pass = (value, opset, label) => expectEqual(assessOnnxNodeSchemaForm(value, opset).status, "pass", label);
const fail = (value, opset, reason, label) => {
  const result = assessOnnxNodeSchemaForm(value, opset);
  expectEqual(result.status, "fail", label);
  if (!result.reason_codes.some((item) => item.startsWith(reason))) throw new Error(`${label}: missing ${reason} in ${result.reason_codes.join(", ")}`);
};

pass(node("Slice", ["x"], ["y"], [["starts", attr(7)], ["ends", attr(7)]]), 9, "Slice-1 attribute form");
fail(node("Slice", ["x", "starts", "ends"]), 9, "input_count_above_maximum", "pre-10 Slice rejects control inputs");
pass(node("Slice", ["x", "starts", "ends"]), 10, "Slice-10 input form");
fail(node("Slice", ["x"], ["y"], [["starts", attr(7)], ["ends", attr(7)]]), 10, "input_count_below_minimum", "Slice-10 rejects legacy attributes");

pass(node("Unsqueeze", ["x"], ["y"], [["axes", attr(7)]]), 12, "Unsqueeze-11 attribute form");
pass(node("Unsqueeze", ["x", "axes"]), 13, "Unsqueeze-13 input form");
fail(node("Unsqueeze", ["x"], ["y"], [["axes", attr(7)]]), 13, "input_count_below_minimum", "Unsqueeze-13 rejects attribute form");

pass(node("Pad", ["x"], ["y"], [["pads", attr(7)]]), 10, "Pad-2 attribute form");
pass(node("Pad", ["x", "pads"]), 11, "Pad-11 input form");
fail(node("Pad", ["x", "pads"], ["y"], [["pads", attr(7)]]), 11, "attribute_not_defined:pads", "Pad-11 rejects pads attribute");

pass(node("ReduceSum", ["x"], ["y"], [["axes", attr(7)]]), 12, "ReduceSum-11 attribute form");
pass(node("ReduceSum", ["x", "axes"]), 13, "ReduceSum-13 input form");
fail(node("ReduceSum", ["x", "axes"], ["y"], [["axes", attr(7)]]), 13, "attribute_not_defined:axes", "ReduceSum-13 rejects axes attribute");
pass(node("ReduceMean", ["x"], ["y"], [["axes", attr(7)]]), 17, "ReduceMean-13 attribute form");
pass(node("ReduceMean", ["x", "axes"]), 18, "ReduceMean-18 input form");

fail(node("Reshape", ["x", "shape"], ["y"], [["allowzero", attr(2, { i: 1 })]]), 13, "attribute_not_defined:allowzero", "Reshape allowzero version gate");
pass(node("Reshape", ["x", "shape"], ["y"], [["allowzero", attr(2, { i: 1 })]]), 14, "Reshape-14 allowzero");
fail(node("Shape", ["x"], ["y"], [["start", attr(2, { i: 0 })]]), 14, "attribute_not_defined:start", "Shape start version gate");
pass(node("Shape", ["x"], ["y"], [["start", attr(2, { i: 0 })]]), 15, "Shape-15 start");

pass(node("TopK", ["x"], ["values", "indices"], [["k", attr(2, { i: 3 })]]), 9, "TopK-1 k attribute");
fail(node("TopK", ["x"], ["values", "indices"]), 9, "required_attribute_missing:k", "TopK-1 requires k");
pass(node("TopK", ["x", "k"], ["values", "indices"]), 10, "TopK-10 k input");

fail(node("QuantizeLinear", ["x", "scale"], ["y"], [["output_dtype", attr(2, { i: 2 })]]), 20, "attribute_not_defined:output_dtype", "QuantizeLinear output_dtype version gate");
pass(node("QuantizeLinear", ["x", "scale"], ["y"], [["output_dtype", attr(2, { i: 2 })]]), 21, "QuantizeLinear-21 output_dtype");
fail(node("QuantizeLinear", ["x", "scale"], ["y"], [["block_size", attr(2, { i: 4 })]]), 20, "attribute_not_defined:block_size", "QuantizeLinear block_size version gate");
pass(node("QuantizeLinear", ["x", "scale"], ["y"], [["block_size", attr(2, { i: 4 })]]), 21, "QuantizeLinear-21 block_size");

fail(node("Conv", ["x", "w"], ["y"], [["group", attr(3, { s: "1" })]]), 13, "attribute_type_mismatch:group", "attribute type mismatch");
fail({ ...node("Conv", ["x", "w"], ["y"], [["group", attr(2, { i: 1 })]]), duplicateAttributeNames: ["group"] }, 13, "duplicate_attribute_name:group", "duplicate attribute name");
fail(node("Conv", ["x", "w"], ["y"], [["group", attr(2, { i: 1, valueTypesPresent: [3] })]]), 13, "attribute_discriminator_payload_mismatch:INT:STRING:group", "attribute discriminator/payload mismatch");
fail(node("Conv", ["x", "w"], ["y"], [["group", attr(2, { i: 1, valueTypesPresent: [2, 3] })]]), 13, "attribute_multiple_value_fields:INT+STRING:group", "multiple AttributeProto value fields");
fail(node("Conv", ["x", "w"], ["y"], [["group", attr(2, { valueTypesPresent: [] })]]), 13, "attribute_value_missing:INT:group", "missing scalar attribute value");
fail(node("Conv", ["x", "w"], ["y"], [["group", attr(2, { refAttrName: "parent_group" })]]), 13, "attribute_reference_not_allowed_in_main_graph:group", "main-graph attribute reference");
fail(node("Relu", [""], ["y"]), 13, "required_input_omitted:0", "required input omission");
fail(node("Relu", ["x"], ["y"], [["invented", attr(2, { i: 1 })]]), 13, "attribute_not_defined:invented", "unknown attribute");
fail(node("Acos", ["x"]), 6, "operator_not_defined_at_imported_opset", "operator availability gate");

const graph = { nodes: [node("Unsqueeze", ["x"], ["y"], [["axes", attr(7)]])] };
const tensors = new Map([
  ["x", { name: "x", dtype: "FLOAT32", shape: [2, 3], shapeDeclared: true }],
  ["y", { name: "y", dtype: "FLOAT32", shape: [], shapeDeclared: false }],
]);
const evidence = inferOnnxShapes(graph, tensors, [{ domain: "", version: 13 }], () => "UNKNOWN");
expectEqual(evidence.status, "fail", "invalid schema form fails shape evidence");
expectEqual(evidence.schema_form_invalid_node_count, 1, "invalid schema-form count");
expectEqual(evidence.known_node_output_count, 0, "invalid schema form must not be inferred");
expectEqual(evidence.rule_unresolved_nodes[0].reason, "opset_schema_form_invalid", "fail-closed reason");

const duplicateImportTensors = new Map([
  ["x", { name: "x", dtype: "FLOAT32", shape: [2, 3], shapeDeclared: true }],
  ["y", { name: "y", dtype: "FLOAT32", shape: [], shapeDeclared: false }],
]);
const duplicateImport = inferOnnxShapes(
  { nodes: [node("Relu", ["x"])] },
  duplicateImportTensors,
  [{ domain: "", version: 12 }, { domain: "ai.onnx", version: 13 }],
  () => "UNKNOWN",
);
expectEqual(duplicateImport.status, "assessed", "multiple valid standard-domain imports use the source-defined highest version");
expectEqual(duplicateImport.opset_import_contract.schema, "deepbom.onnx_opset_import_contract.v1.1", "opset contract schema");
expectEqual(duplicateImport.opset_import_contract.invalid_import_count, 0, "valid repeated import records are not fabricated as invalid");
expectEqual(duplicateImport.opset_import_contract.duplicate_domain_count, 1, "duplicate normalized domain count");
expectEqual(duplicateImport.opset_import_contract.duplicate_version_variant_domain_count, 1, "version-variant repeated domain count");
expectEqual(duplicateImport.opset_import_contract.effective_imports[0].version, 13, "nodes bind to the highest referenced standard-domain version");
expectEqual(duplicateImport.schema_form_rows[0].imported_opset, 13, "schema resolution uses the effective highest import");
expectEqual(duplicateImport.known_node_output_count, 1, "an unambiguous highest import must retain deterministic shape inference");

const identicalDuplicateImport = inferOnnxShapes(
  { nodes: [node("Relu", ["x"])] },
  new Map(duplicateImportTensors),
  [{ domain: "", version: 13 }, { domain: "ai.onnx", version: 13 }],
  () => "UNKNOWN",
);
expectEqual(identicalDuplicateImport.status, "assessed", "identical repeated standard-domain records remain assessable");
expectEqual(identicalDuplicateImport.opset_import_contract.duplicate_identical_domain_count, 1, "identical repeated domain count");
expectEqual(identicalDuplicateImport.opset_import_contract.rows.filter((row) => row.selected_effective_import).length, 2, "all identical maximum-version records preserve their source identity");

const missingImport = inferOnnxShapes(
  { nodes: [node("Relu", ["x"])] },
  new Map(duplicateImportTensors),
  [],
  () => "UNKNOWN",
);
expectEqual(missingImport.status, "fail", "missing standard-domain import fails shape evidence");
expectEqual(missingImport.schema_form_rows[0].reason_codes[0], "standard_domain_opset_missing", "missing import reason");

const invalidVersion = inferOnnxShapes(
  { nodes: [node("Relu", ["x"])] },
  new Map(duplicateImportTensors),
  [{ domain: "", version: 0 }],
  () => "UNKNOWN",
);
expectEqual(invalidVersion.opset_import_contract.invalid_version_count, 1, "invalid import version count");
expectEqual(invalidVersion.status, "fail", "invalid import version fails shape evidence");

console.log(`${ONNX_SHAPE_SCHEMA_SOURCE.op_count}-op / ${ONNX_SHAPE_SCHEMA_SOURCE.schema_version_count}-version pinned OpSchema formal legality and transition fixtures`);

function expectEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`);
}
