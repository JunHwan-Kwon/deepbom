import { inferOnnxShapes } from "../web/lib/onnx-shape-inference.js";
import { makeOnnxMapType, makeOnnxTensorType, onnxValueDescriptorFromType } from "../web/lib/onnx-type-proto.js";
import { analyzeOnnxModel } from "../web/onnx.js";
import { buildEngineeringBundleArtifactFiles, buildMlBomDocument } from "../web/lib/report.js";
import { buildFindingsRegister } from "../web/lib/report-findings.js";
import { createCheck } from "./check-assert.mjs";
import { assertCompactMlBomProjection } from "./compact-mlbom-assert.mjs";
import * as ort from "onnxruntime-node";

const { done, expect, expectEqual, expectThrows } = createCheck("ONNX-ML value inference check");
const expectCompactMlBom = (document, omittedProperties, label) => assertCompactMlBomProjection(document, {
  expect,
  expectEqual,
  omittedProperties,
  label,
});

const stringResult = run(
  mlNode({ classlabels_strings: stringsAttribute(["cat", "dog", "bird"]) }),
  tensor("scores", "FLOAT32", [2, 3]),
);
const stringRow = stringResult.evidence.ml_value_inference.rows[0];
const stringOutput = stringResult.tensorMap.get("probabilities");
expectEqual(stringResult.evidence.status, "assessed", "A static rank-2 ZipMap contract should be fully assessed.");
expectEqual(stringOutput.valueKind, "sequence", "ZipMap should emit a sequence value rather than a dense tensor.");
expectEqual(stringOutput.typeProto.elementType.kind, "map", "ZipMap sequence elements should be maps.");
expectEqual(stringOutput.typeProto.elementType.keyTypeName, "STRING", "String labels should fix the map key type.");
expectEqual(stringOutput.typeProto.elementType.valueType.elementTypeName, "FLOAT32", "ZipMap map values should be FLOAT.");
expectEqual(stringOutput.typeProto.elementType.valueType.shapeDeclared, true, "The pinned schema initializes an explicit rank-0 TensorShapeProto for map values.");
expectEqual(stringOutput.sequenceLength, 2, "Rank-2 ZipMap output length should equal the static batch dimension.");
expectEqual(stringOutput.sequenceElementTypes.length, 2, "A bounded static batch should preserve an exact element inventory.");
expectEqual(stringRow.exact_feature_count, 3, "The final input dimension should determine feature count.");
expectEqual(stringRow.class_key_count, 3, "Class-label cardinality should be observed exactly.");
expectEqual(stringRow.canonical_output_type, "sequence<map<STRING,tensor<FLOAT32[]>>>", "Canonical TypeProto should preserve the source-pinned scalar FLOAT value type.");

const integerResult = run(
  mlNode({ classlabels_int64s: intsAttribute(["-3", "9007199254740993", "9223372036854775807"]) }),
  tensor("scores", "FLOAT32", [3]),
);
const integerRow = integerResult.evidence.ml_value_inference.rows[0];
expectEqual(integerResult.tensorMap.get("probabilities").sequenceLength, 1, "Rank-1 ZipMap input represents one output map.");
expectEqual(integerRow.class_key_type, "INT64", "Integer labels should fix the map key type.");
expectEqual(integerRow.class_key_preview[1], "9007199254740993", "INT64 keys must remain exact beyond JavaScript's safe integer range.");

const dynamicBatch = run(
  mlNode({ classlabels_strings: stringsAttribute(["a", "b", "c"]) }),
  tensor("scores", "FLOAT32", [-1, 3]),
);
expectEqual(dynamicBatch.evidence.status, "partial", "A known non-dense output with runtime batch cardinality should remain partial, not not_assessed.");
expectEqual(dynamicBatch.evidence.known_non_dense_node_output_count, 1, "Dynamic sequence length should not erase the exact sequence/map TypeProto.");
expectEqual(dynamicBatch.evidence.ml_value_inference.partially_assessed_node_count, 1, "Dynamic batch cardinality should be explicit in the ML ledger.");

expectEqual(run(
  mlNode({ classlabels_strings: stringsAttribute(["a", "b"]) }),
  tensor("scores", "FLOAT32", [1, 3]),
).evidence.status, "fail", "Feature/key cardinality mismatch should fail deterministically.");
expectEqual(run(
  mlNode({ classlabels_strings: stringsAttribute(["a"]), classlabels_int64s: intsAttribute(["1"]) }),
  tensor("scores", "FLOAT32", [1]),
).evidence.status, "fail", "ZipMap must carry exactly one nonempty class-label attribute.");
expectEqual(run(mlNode({}), tensor("scores", "FLOAT32", [1])).evidence.status, "fail", "A missing class-label contract should fail.");
expectEqual(run(
  mlNode({ classlabels_strings: stringsAttribute(["a", "b", "c"]) }),
  tensor("scores", "FLOAT32", [1, 1, 3]),
).evidence.status, "fail", "ZipMap rank must be one or two.");
expectEqual(run(
  mlNode({ classlabels_strings: stringsAttribute(["a", "b", "c"]) }),
  tensor("scores", "FLOAT64", [1, 3]),
).evidence.status, "fail", "ZipMap input must be FLOAT32 under the pinned schema.");

const duplicates = run(
  mlNode({ classlabels_strings: stringsAttribute(["a", "a", "b"]) }),
  tensor("scores", "FLOAT32", [1, 3]),
);
expectEqual(duplicates.evidence.status, "assessed", "Duplicate labels are an artifact risk, not an invented OpSchema rejection.");
expectEqual(duplicates.evidence.ml_value_inference.duplicate_class_key_count, 1, "Duplicate class keys should be counted exactly.");
expectEqual(duplicates.evidence.ml_value_inference.duplicate_class_key_node_count, 1, "The affected ZipMap node count should be exact.");
expectEqual(duplicates.evidence.ml_value_inference.rows[0].risk_codes[0], "zip_map_duplicate_class_keys_information_loss_risk", "Duplicate keys should retain an actionable risk code.");

const castSparse = run(
  mlOp("CastMap", {
    cast_to: stringAttribute("TO_INT64"),
    map_form: stringAttribute("SPARSE"),
    max_map: intAttribute("7"),
  }),
  mapValue("scores", "INT64", "FLOAT32", 3, ["0", "3", "6"]),
);
const castSparseRow = castSparse.evidence.ml_value_inference.rows[0];
expectEqual(castSparse.evidence.status, "assessed", "A valid SPARSE CastMap contract should be fully assessed.");
expectEqual(castSparse.tensorMap.get("probabilities").dtype, "INT64", "cast_to must determine CastMap output dtype.");
expectEqual(JSON.stringify(castSparse.tensorMap.get("probabilities").shape), "[7]", "SPARSE max_map must determine exact 1-D output shape.");
expectEqual(castSparseRow.exact_dense_output_element_count, 7, "SPARSE max_map must determine exact output cardinality.");
expectEqual(castSparseRow.sparse_key_bounds_status, "assessed_pass", "Exact SPARSE keys should be checked against max_map without numeric coercion.");

const castDenseUnknown = run(mlOp("CastMap", {}), mapValue("scores", "INT64", "STRING"));
expectEqual(castDenseUnknown.evidence.ml_value_inference.status, "partial", "DENSE CastMap must retain an explicit ML residual when map cardinality is unavailable.");
expectEqual(castDenseUnknown.evidence.known_node_output_count, 0, "Unknown DENSE length must not be promoted to a fully known dense output.");
expectEqual(castDenseUnknown.tensorMap.get("probabilities").dtype, "FLOAT32", "Unknown DENSE length must still preserve source-derived output dtype.");
expectEqual(castDenseUnknown.evidence.ml_value_inference.rows[0].canonical_output_type, "tensor<FLOAT32[?]>", "Default CastMap attributes should derive FLOAT32 rank-1 output.");

const castDenseKnown = run(mlOp("CastMap", {}), mapValue("scores", "INT64", "FLOAT32", 3));
expectEqual(castDenseKnown.evidence.status, "assessed", "Artifact-known map cardinality should close DENSE CastMap output shape.");
expectEqual(JSON.stringify(castDenseKnown.tensorMap.get("probabilities").shape), "[3]", "DENSE output length must equal exact input-map cardinality.");
expectEqual(run(mlOp("CastMap", {}), mapValue("scores", "STRING", "FLOAT32")).evidence.status, "fail", "CastMap map keys must be INT64.");
expectEqual(run(mlOp("CastMap", {}), mapValue("scores", "INT64", "INT64")).evidence.status, "fail", "CastMap map values must be FLOAT32 or STRING.");
expectEqual(run(mlOp("CastMap", { cast_to: stringAttribute("TO_DOUBLE") }), mapValue("scores", "INT64", "FLOAT32")).evidence.status, "fail", "Unknown CastMap cast_to values must fail closed.");
expectEqual(run(mlOp("CastMap", { map_form: stringAttribute("PACKED") }), mapValue("scores", "INT64", "FLOAT32")).evidence.status, "fail", "Unknown CastMap map_form values must fail closed.");
expectEqual(run(mlOp("CastMap", { map_form: stringAttribute("SPARSE"), max_map: intAttribute("-1") }), mapValue("scores", "INT64", "FLOAT32")).evidence.status, "fail", "Negative SPARSE max_map must fail closed.");
expectEqual(run(mlOp("CastMap", { map_form: stringAttribute("SPARSE"), max_map: intAttribute("9007199254740993") }), mapValue("scores", "INT64", "FLOAT32")).evidence.ml_value_inference.status, "partial", "A SPARSE bound beyond exact JavaScript integer arithmetic must remain explicit, not rounded.");
expectEqual(run(mlOp("CastMap", { map_form: stringAttribute("SPARSE"), max_map: intAttribute("7") }), mapValue("scores", "INT64", "FLOAT32")).evidence.ml_value_inference.status, "partial", "SPARSE output shape must not hide unresolved runtime key bounds.");
expectEqual(run(mlOp("CastMap", { map_form: stringAttribute("SPARSE"), max_map: intAttribute("7") }), mapValue("scores", "INT64", "FLOAT32", 2, ["0", "7"])).evidence.status, "fail", "An exact SPARSE key outside [0,max_map) must fail deterministically.");

const dict = run(
  mlOp("DictVectorizer", { string_vocabulary: stringsAttribute(["a", "b", "c", "d"]) }),
  mapValue("scores", "STRING", "INT64"),
);
const dictRow = dict.evidence.ml_value_inference.rows[0];
expectEqual(dict.evidence.status, "assessed", "A valid DictVectorizer contract should be fully assessed.");
expectEqual(dict.tensorMap.get("probabilities").dtype, "INT64", "DictVectorizer output dtype must equal map value dtype.");
expectEqual(JSON.stringify(dict.tensorMap.get("probabilities").shape), "[1,4]", "Pinned ORT behavior must close DictVectorizer output shape from vocabulary length.");
expectEqual(dictRow.exact_dense_output_element_count, 4, "Vocabulary cardinality must determine exact dense element count.");

const dictExactInt64 = run(
  mlOp("DictVectorizer", { int64_vocabulary: intsAttribute(["-3", "9007199254740993", "9223372036854775807"]) }),
  mapValue("scores", "INT64", "STRING"),
);
expectEqual(dictExactInt64.evidence.ml_value_inference.rows[0].vocabulary_preview[1], "9007199254740993", "DictVectorizer INT64 vocabulary must remain exact beyond JavaScript's safe integer range.");

const dictDuplicates = run(
  mlOp("DictVectorizer", { string_vocabulary: stringsAttribute(["a", "a", "c"]) }),
  mapValue("scores", "STRING", "FLOAT64"),
);
expectEqual(dictDuplicates.evidence.status, "assessed", "Duplicate vocabulary entries are supported runtime behavior, not an invented schema failure.");
expectEqual(dictDuplicates.evidence.ml_value_inference.duplicate_vocabulary_entry_count, 1, "Duplicate vocabulary occurrences should be counted exactly.");
expectEqual(dictDuplicates.evidence.ml_value_inference.duplicate_vocabulary_node_count, 1, "Affected DictVectorizer nodes should be counted exactly.");
expectEqual(dictDuplicates.evidence.ml_value_inference.rows[0].risk_codes[0], "dict_vectorizer_duplicate_vocabulary_columns", "Duplicate vocabulary should retain the pinned-runtime risk code.");

expectEqual(run(mlOp("DictVectorizer", {}), mapValue("scores", "STRING", "FLOAT32")).evidence.status, "fail", "DictVectorizer requires exactly one vocabulary attribute.");
expectEqual(run(mlOp("DictVectorizer", { string_vocabulary: stringsAttribute([]), int64_vocabulary: intsAttribute([]) }), mapValue("scores", "STRING", "FLOAT32")).evidence.status, "fail", "DictVectorizer must reject conflicting vocabulary attributes even when both are empty.");
expectEqual(run(mlOp("DictVectorizer", { string_vocabulary: stringsAttribute(["a"]) }), mapValue("scores", "INT64", "FLOAT32")).evidence.status, "fail", "Vocabulary type must equal DictVectorizer map key type.");
expectEqual(run(mlOp("DictVectorizer", { string_vocabulary: stringsAttribute(["a"]) }), mapValue("scores", "STRING", "STRING")).evidence.status, "fail", "DictVectorizer must reject same-type string keys and values.");
expectEqual(run(mlOp("DictVectorizer", { string_vocabulary: stringsAttribute([]) }), mapValue("scores", "STRING", "FLOAT32")).evidence.status, "assessed", "An explicitly present empty vocabulary has exact [1,0] runtime shape.");

const badAttribute = run(
  mlNode({ classlabels_strings: { ...stringsAttribute(["a"]), type: 7 } }),
  tensor("scores", "FLOAT32", [1]),
);
expectEqual(badAttribute.evidence.status, "fail", "Attribute discriminator/payload mismatch should fail the formal schema contract.");
expectEqual(badAttribute.evidence.schema_form_invalid_node_count, 1, "Malformed ZipMap attributes should enter schema-form evidence.");

const badCategoryAttribute = run(
  mlOp("CategoryMapper", {
    cats_strings: { ...stringsAttribute(["a"]), type: 7 },
    cats_int64s: intsAttribute(["1"]),
  }),
  tensor("scores", "STRING", [1]),
);
expectEqual(badCategoryAttribute.evidence.status, "fail", "Malformed CategoryMapper attribute typing should fail the formal schema contract.");
expectEqual(badCategoryAttribute.evidence.schema_form_invalid_node_count, 1, "Malformed CategoryMapper attributes should enter schema-form evidence.");

const missingImport = run(
  mlNode({ classlabels_strings: stringsAttribute(["a"]) }),
  tensor("scores", "FLOAT32", [1]),
  [],
);
expectEqual(missingImport.evidence.status, "fail", "A missing ai.onnx.ml opset import should fail closed.");

const duplicateImport = run(
  mlNode({ classlabels_strings: stringsAttribute(["a"]) }),
  tensor("scores", "FLOAT32", [1]),
  [{ domain: "ai.onnx.ml", version: 1 }, { domain: "ai.onnx.ml", version: 1 }],
);
expectEqual(duplicateImport.evidence.status, "assessed", "Identical repeated ai.onnx.ml imports should retain deterministic inference.");
expectEqual(duplicateImport.evidence.opset_import_contract.duplicate_identical_domain_count, 1, "Identical ai.onnx.ml imports should remain explicit diagnostics.");
expectEqual(duplicateImport.evidence.opset_import_contract.effective_imports[0]?.version, 1, "Identical ai.onnx.ml imports should preserve their effective version.");

const multiVersionImport = run(
  mlNode({ classlabels_strings: stringsAttribute(["a"]) }),
  tensor("scores", "FLOAT32", [1]),
  [{ domain: "ai.onnx.ml", version: 1 }, { domain: "ai.onnx.ml", version: 3 }],
);
expectEqual(multiVersionImport.evidence.status, "assessed", "Multiple valid ai.onnx.ml imports should bind to the highest referenced version.");
expectEqual(multiVersionImport.evidence.opset_import_contract.duplicate_version_variant_domain_count, 1, "Multi-version ai.onnx.ml imports should remain explicit diagnostics.");
expectEqual(multiVersionImport.evidence.opset_import_contract.effective_imports[0]?.version, 3, "ai.onnx.ml schema resolution should use the highest referenced version.");

const customDomain = run(
  mlNode({ classlabels_strings: stringsAttribute(["a"]) }, "com.acme"),
  tensor("scores", "FLOAT32", [1]),
  [{ domain: "com.acme", version: 1 }],
);
expectEqual(customDomain.evidence.rule_supported_nodes, 0, "A custom-domain ZipMap must never inherit ai.onnx.ml semantics.");
expectEqual(customDomain.evidence.ml_value_inference.assessed_node_count, 0, "Domain isolation should leave the ML ledger untouched.");

const customCategoryDomain = run(
  mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a"]), cats_int64s: intsAttribute(["1"]) }, "com.acme"),
  tensor("scores", "STRING", [1]),
  [{ domain: "com.acme", version: 1 }],
);
expectEqual(customCategoryDomain.evidence.rule_supported_nodes, 0, "A custom-domain CategoryMapper must never inherit ai.onnx.ml semantics.");
expectEqual(customCategoryDomain.evidence.ml_value_inference.assessed_node_count, 0, "Custom CategoryMapper domain isolation should leave the ML ledger untouched.");

expectEqual(stringResult.evidence.schema_form_assessed_node_count, stringResult.evidence.rule_supported_nodes, "Every supported ZipMap node should have one formal schema row.");

const serialized = analyzeOnnxModel(serializedZipMapModel(["cat", "dog", "bird"], [2, 3]), "serialized_zip_map.onnx");
expectEqual(serialized.onnx_shape_inference.status, "assessed", "Serialized ZipMap bytes should preserve the assessed contract through the real parser.");
expectEqual(serialized.onnx_shape_inference.ml_value_inference.exact_class_key_count, 3, "Serialized string labels should survive the AttributeProto parser exactly.");
expectEqual(serialized.onnx_shape_inference.ml_value_inference.rows[0].exact_output_sequence_length, 2, "Serialized input batch should determine exact output sequence length.");
const serializedBundle = bundle(serialized);
expectEqual(serializedBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized ZipMap report bundle should pass export conformance.");
expect(serializedBundle.report.includes("### ONNX-ML Value Contracts"), "Engineering Report should expose serialized ONNX-ML rows.");
expect(serializedBundle.report.includes("4588b9efe493ea820d54c5b65b6af6ad8a7860f625f97f5d6edcfd5bf06125e6"), "Engineering Report should preserve the pinned ZipMap schema hash.");
expect(serializedBundle.report.includes("f9205124962c59fcaf2f56aee5e0f47af05a7f21b2bb897e0e08dc39ef7f481a"), "Engineering Report should preserve the pinned ORT ZipMap kernel hash.");
expectCompactMlBom(serializedBundle.mlBom, ["deepbom:model:onnxMlExactClassKeys"], "ZipMap compact ML-BOM");

const serializedDuplicate = analyzeOnnxModel(serializedZipMapModel(["cat", "cat", "bird"], [1, 3]), "serialized_zip_map_duplicate.onnx");
const duplicateFinding = buildFindingsRegister(serializedDuplicate).find((item) => item.finding_id === "EA-ONX-0014");
expectEqual(duplicateFinding?.technical_priority, "High", "Serialized duplicate keys should enter the High action queue without inventing an OpSchema failure.");
expectEqual(bundle(serializedDuplicate).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported duplicate-key risk should pass export conformance.");

const serializedMismatch = analyzeOnnxModel(serializedZipMapModel(["cat", "dog"], [1, 3]), "serialized_zip_map_mismatch.onnx");
const mismatchFinding = buildFindingsRegister(serializedMismatch).find((item) => item.finding_id === "EA-ONX-0013");
expectEqual(serializedMismatch.onnx_shape_inference.status, "fail", "Serialized feature/key mismatch should fail the artifact contract.");
expectEqual(mismatchFinding?.technical_priority, "High", "Serialized ZipMap failure should enter the High action queue.");
expectEqual(bundle(serializedMismatch).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported invalid ZipMap artifact should still pass export conformance.");

const serializedCast = analyzeOnnxModel(serializedCastMapModel(), "serialized_cast_map.onnx");
const serializedCastRows = serializedCast.onnx_shape_inference.ml_value_inference.rows;
expectEqual(serializedCastRows.length, 1, `Serialized CastMap should produce one map/value row; operators=${JSON.stringify(serializedCast.ops || [])}; shape=${JSON.stringify(serializedCast.onnx_shape_inference)}`);
const serializedCastRow = serializedCastRows[0] || {};
expectEqual(serializedCast.onnx_shape_inference.status, "partial", "Serialized CastMap should preserve exact SPARSE shape while retaining runtime key-bound residuals.");
expectEqual(serializedCastRow.output_dtype, "STRING", "Serialized cast_to should survive AttributeProto parsing.");
expectEqual(JSON.stringify(serializedCastRow.exact_output_shape), "[5]", "Serialized max_map should close SPARSE output shape.");
expectEqual(serializedCastRow.sparse_key_bounds_status, "not_assessed_runtime_keys", "Serialized graph-input map keys must remain an explicit runtime residual.");
const serializedCastBundle = bundle(serializedCast);
expectEqual(serializedCastBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized CastMap report bundle should pass export conformance.");
expect(serializedCastBundle.report.includes("cast_to STRING") === false, "Report should retain the canonical CastMap attribute spelling rather than inventing aliases.");
expect(serializedCastBundle.report.includes("cast_to TO_STRING; map_form SPARSE; max_map 5"), "Engineering Report should expose the exact serialized CastMap conversion contract.");
expectCompactMlBom(serializedCastBundle.mlBom, ["deepbom:model:onnxMlExactDenseOutputShapes"], "CastMap compact ML-BOM");

const serializedDict = analyzeOnnxModel(serializedDictVectorizerModel(["a", "b", "c"]), "serialized_dict_vectorizer.onnx");
const serializedDictRow = serializedDict.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedDict.onnx_shape_inference.status, "assessed", "Serialized DictVectorizer bytes should preserve the vocabulary contract through the real parser.");
expectEqual(serializedDictRow.output_dtype, "FLOAT64", "Serialized map value dtype should determine DictVectorizer output dtype.");
expectEqual(JSON.stringify(serializedDictRow.exact_output_shape), "[1,3]", "Serialized vocabulary cardinality should determine [1,C] output shape.");
const serializedDictBundle = bundle(serializedDict);
expectEqual(serializedDictBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized DictVectorizer report bundle should pass export conformance.");
expect(serializedDictBundle.report.includes("vocabulary STRING x3"), "Engineering Report should expose exact DictVectorizer vocabulary cardinality.");
expect(serializedDictBundle.report.includes("b372ee05451ff430e7a6f112addabde1da826792411061b456dbca0ce78b69b8"), "Engineering Report should preserve the pinned ORT DictVectorizer contract hash.");
expectCompactMlBom(serializedDictBundle.mlBom, ["deepbom:model:onnxMlExactVocabularyEntries"], "DictVectorizer compact ML-BOM");

const serializedDictDuplicate = analyzeOnnxModel(serializedDictVectorizerModel(["a", "a", "c"]), "serialized_dict_vectorizer_duplicate.onnx");
const duplicateVocabularyFinding = buildFindingsRegister(serializedDictDuplicate).find((item) => item.finding_id === "EA-ONX-0015");
expectEqual(duplicateVocabularyFinding?.technical_priority, "Medium", "Serialized duplicate vocabulary columns should enter the Medium action queue without inventing an OpSchema failure.");
expectEqual(bundle(serializedDictDuplicate).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported duplicate-vocabulary risk should pass export conformance.");

const categoryString = run(
  mlOp("CategoryMapper", {
    cats_strings: stringsAttribute(["Three", "Two", "One"]),
    cats_int64s: intsAttribute(["3", "2", "1"]),
    default_int64: intAttribute("99"),
  }),
  tensor("scores", "STRING", [2, 2, 2]),
);
const categoryStringRow = categoryString.evidence.ml_value_inference.rows[0];
expectEqual(categoryString.evidence.status, "assessed", "CategoryMapper should preserve the pinned ORT rank-3 same-shape behavior.");
expectEqual(categoryString.tensorMap.get("probabilities").dtype, "INT64", "STRING CategoryMapper input must produce INT64 output.");
expectEqual(JSON.stringify(categoryString.tensorMap.get("probabilities").shape), "[2,2,2]", "CategoryMapper output shape must equal input shape.");
expectEqual(categoryStringRow.mapping_direction, "STRING_TO_INT64", "Input dtype must determine the active mapping direction.");
expectEqual(categoryStringRow.category_pair_count, 3, "Parallel category arrays should preserve exact pair cardinality.");
expectEqual(categoryStringRow.active_default_value, "99", "The active INT64 default must remain exact.");
expectEqual(categoryStringRow.exact_dense_output_element_count, 8, "Static CategoryMapper shape should determine exact output elements.");

const categoryInt = run(
  mlOp("CategoryMapper", {
    cats_strings: stringsAttribute(["small", "large"]),
    cats_int64s: intsAttribute(["9007199254740993", "9223372036854775807"]),
    default_string: stringAttribute("unknown"),
  }),
  tensor("scores", "INT64", [2]),
);
const categoryIntRow = categoryInt.evidence.ml_value_inference.rows[0];
expectEqual(categoryInt.tensorMap.get("probabilities").dtype, "STRING", "INT64 CategoryMapper input must produce STRING output.");
expectEqual(categoryIntRow.category_int64_preview[0], "9007199254740993", "CategoryMapper category INT64 values must remain exact beyond JavaScript safe integers.");
expectEqual(categoryIntRow.active_default_value, "unknown", "The active STRING default should be preserved.");

const categoryDynamic = run(
  mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a"]), cats_int64s: intsAttribute(["1"]) }),
  tensor("scores", "STRING", [-1, 3]),
);
expectEqual(categoryDynamic.evidence.ml_value_inference.status, "partial", "Dynamic CategoryMapper dimensions must remain explicit while preserving dtype/rank.");
expectEqual(categoryDynamic.evidence.ml_value_inference.rows[0].canonical_output_type, "tensor<INT64[?,3]>", "Dynamic dimensions should be preserved canonically.");

const categoryDuplicateString = run(
  mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a", "a", "b"]), cats_int64s: intsAttribute(["1", "2", "3"]) }),
  tensor("scores", "STRING", [1]),
);
expectEqual(categoryDuplicateString.evidence.status, "assessed", "Duplicate active categories are an artifact risk, not an invented schema failure.");
expectEqual(categoryDuplicateString.evidence.ml_value_inference.duplicate_category_active_key_count, 1, "Active duplicate category keys should be counted exactly.");
expectEqual(categoryDuplicateString.evidence.ml_value_inference.rows[0].risk_codes[0], "category_mapper_duplicate_active_keys_last_write_wins", "Active duplicate keys should retain a source-backed risk code.");
const categoryInactiveDuplicate = run(
  mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a", "b"]), cats_int64s: intsAttribute(["1", "1"]) }),
  tensor("scores", "STRING", [1]),
);
expectEqual(categoryInactiveDuplicate.evidence.ml_value_inference.duplicate_category_active_key_count, 0, "Duplicate values in the inactive reverse-key direction must not be mislabeled as active overwrite risk.");

expectEqual(run(mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a"]) }), tensor("scores", "STRING", [1])).evidence.status, "fail", "CategoryMapper requires cats_int64s operationally.");
expectEqual(run(mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a"]), cats_int64s: intsAttribute(["1", "2"]) }), tensor("scores", "STRING", [1])).evidence.status, "fail", "CategoryMapper category arrays must have equal length.");
expectEqual(run(mlOp("CategoryMapper", { cats_strings: stringsAttribute(["a"]), cats_int64s: intsAttribute(["1"]) }), tensor("scores", "FLOAT32", [1])).evidence.status, "fail", "CategoryMapper input dtype must be STRING or INT64.");
expectEqual(run(mlOp("CategoryMapper", { cats_strings: stringsAttribute([]), cats_int64s: intsAttribute([]) }), tensor("scores", "STRING", [0])).evidence.status, "assessed", "Explicit empty parallel categories should retain an exact zero-shape contract.");

const serializedCategory = analyzeOnnxModel(serializedCategoryMapperModel(["red", "green", "blue"], [10, 20, 30]), "serialized_category_mapper.onnx");
const serializedCategoryRow = serializedCategory.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedCategory.onnx_shape_inference.status, "assessed", "Serialized CategoryMapper bytes should preserve dtype, shape, defaults, and pairs through the real parser.");
expectEqual(serializedCategoryRow.mapping_direction, "STRING_TO_INT64", "Serialized input dtype should determine CategoryMapper direction.");
expectEqual(serializedCategoryRow.category_pair_count, 3, "Serialized parallel attributes should retain exact cardinality.");
expectEqual(JSON.stringify(serializedCategoryRow.exact_output_shape), "[2,2]", "Serialized CategoryMapper should preserve input shape exactly.");
const serializedCategoryBundle = bundle(serializedCategory);
expectEqual(serializedCategoryBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized CategoryMapper report bundle should pass export conformance.");
expect(serializedCategoryBundle.report.includes("STRING_TO_INT64; 3 pair(s)"), "Engineering Report should expose CategoryMapper direction and exact pair count.");
expect(serializedCategoryBundle.report.includes("0ed0df8a1616d08c291d8fd41d6c6bada42d489bf801ff86071187314f12d248"), "Engineering Report should preserve the pinned ORT CategoryMapper contract hash.");
expectCompactMlBom(serializedCategoryBundle.mlBom, ["deepbom:model:onnxMlExactCategoryPairs"], "CategoryMapper compact ML-BOM");

const serializedCategoryMismatch = analyzeOnnxModel(serializedCategoryMapperModel(["red", "green"], [10]), "serialized_category_mapper_mismatch.onnx");
const categoryMismatchFinding = buildFindingsRegister(serializedCategoryMismatch).find((item) => item.finding_id === "EA-ONX-0013");
expectEqual(serializedCategoryMismatch.onnx_shape_inference.status, "fail", "Serialized CategoryMapper arrays with unequal lengths should fail the artifact contract.");
expectEqual(categoryMismatchFinding?.technical_priority, "High", "Serialized CategoryMapper cardinality failure should enter the High action queue.");
expectEqual(bundle(serializedCategoryMismatch).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported invalid CategoryMapper artifact should still pass export conformance.");

const serializedCategoryDuplicate = analyzeOnnxModel(serializedCategoryMapperModel(["red", "red", "blue"], [10, 20, 30]), "serialized_category_mapper_duplicate.onnx");
const duplicateCategoryFinding = buildFindingsRegister(serializedCategoryDuplicate).find((item) => item.finding_id === "EA-ONX-0016");
expectEqual(duplicateCategoryFinding?.technical_priority, "High", "Serialized active duplicate categories should enter the High action queue.");
expectEqual(bundle(serializedCategoryDuplicate).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported CategoryMapper overwrite risk should pass export conformance.");

const binarizerRuntime = run(
  mlOp("Binarizer", { threshold: floatAttribute(0.25) }),
  tensor("scores", "FLOAT32", [2, 2]),
);
const binarizerRuntimeRow = binarizerRuntime.evidence.ml_value_inference.rows[0];
expectEqual(binarizerRuntime.evidence.status, "assessed", "Runtime-valued Binarizer should still close the source-pinned dtype and shape contract.");
expectEqual(binarizerRuntimeRow.static_value_assessment_status, "not_assessed_runtime_values", "Runtime Binarizer values must remain explicitly unassessed.");
expectEqual(JSON.stringify(binarizerRuntime.tensorMap.get("probabilities").shape), "[2,2]", "Binarizer should preserve exact input shape.");
expectEqual(binarizerRuntime.tensorMap.get("probabilities").dtype, "FLOAT32", "Binarizer should preserve exact input dtype.");

const binarizerStatic = run(
  mlOp("Binarizer", { threshold: floatAttribute(0.25) }),
  staticNumericTensor("scores", "FLOAT32", [4], [-1, 0.25, 0.5, 2]),
);
const binarizerStaticRow = binarizerStatic.evidence.ml_value_inference.rows[0];
expectEqual(binarizerStatic.evidence.status, "assessed", "A finite static Binarizer contract should be fully assessed.");
expectEqual(binarizerStaticRow.exact_static_input_value_count, 4, "Every static Binarizer input value should be counted.");
expectEqual(binarizerStaticRow.exact_above_threshold_count, 2, "Strictly greater values should map to one.");
expectEqual(binarizerStaticRow.exact_at_or_below_threshold_count, 2, "Equal and lower values should map to zero.");
expectEqual(binarizerStaticRow.exact_equal_threshold_count, 1, "Threshold equality should be counted separately.");
expectEqual(JSON.stringify(binarizerStatic.tensorMap.get("probabilities").staticValues), "[0,0,1,1]", "Exact Binarizer outputs should propagate to downstream static inference.");

const binarizerDefault = run(
  mlOp("Binarizer", {}),
  staticNumericTensor("scores", "FLOAT64", [3], [-1, 0, 1]),
);
const binarizerDefaultRow = binarizerDefault.evidence.ml_value_inference.rows[0];
expectEqual(binarizerDefaultRow.threshold_source, "onnx_schema_default_0", "An omitted Binarizer threshold must use the pinned ONNX default 0.0.");
expectEqual(binarizerDefaultRow.threshold_value_text, "0", "The schema-default threshold must be serialized canonically.");
expectEqual(JSON.stringify(binarizerDefault.tensorMap.get("probabilities").staticValues), "[0,0,1]", "The default threshold must use strict greater-than semantics.");
expect(binarizerDefaultRow.risk_codes.includes("binarizer_dtype_unsupported_by_pinned_ort_cpu"), "A schema-valid FLOAT64 Binarizer must expose the pinned ORT CPU kernel gap.");

const binarizerInt64 = run(
  mlOp("Binarizer", { threshold: floatAttribute(0.5) }),
  exactIntegerTensor("scores", [3], ["-9223372036854775808", "0", "9007199254740993"]),
);
expectEqual(binarizerInt64.evidence.ml_value_inference.rows[0].exact_above_threshold_count, 1, "INT64 comparison must stay exact beyond JavaScript safe integers.");
expectEqual(JSON.stringify(binarizerInt64.tensorMap.get("probabilities").staticValues), "[0,0,1]", "Exact INT64 Binarizer outputs should be represented as safe 0/1 values.");
expect(binarizerInt64.evidence.ml_value_inference.rows[0].risk_codes.includes("binarizer_dtype_unsupported_by_pinned_ort_cpu"), "A schema-valid INT64 Binarizer must not be mislabeled executable on the pinned ORT CPU path.");
expectEqual(run(mlOp("Binarizer", {}), tensor("scores", "UINT8", [1])).evidence.status, "fail", "Binarizer must reject a dtype outside its pinned numeric type set.");
expectEqual(run(mlOp("Binarizer", { threshold: stringAttribute("0.5") }), tensor("scores", "FLOAT32", [1])).evidence.status, "fail", "Binarizer must reject a non-FLOAT threshold AttributeProto.");
const customBinarizer = run(mlOp("Binarizer", {}, "com.example"), tensor("scores", "FLOAT32", [1]), [{ domain: "com.example", version: 1 }]);
expectEqual(customBinarizer.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain Binarizer name must not inherit ai.onnx.ml semantics.");

const normalizerRuntime = run(mlOp("Normalizer", { norm: stringAttribute("L2") }), tensor("scores", "FLOAT64", [2, 3]));
const normalizerRuntimeRow = normalizerRuntime.evidence.ml_value_inference.rows[0];
expectEqual(normalizerRuntime.evidence.status, "assessed", "Runtime-valued Normalizer should close its source-pinned type, rank, shape, and mode contract.");
expectEqual(normalizerRuntime.tensorMap.get("probabilities").dtype, "FLOAT32", "Normalizer output must be FLOAT32 independently of input dtype.");
expectEqual(JSON.stringify(normalizerRuntime.tensorMap.get("probabilities").shape), "[2,3]", "Normalizer must preserve rank-2 shape.");
expectEqual(normalizerRuntimeRow.normalizer_static_assessment_status, "not_assessed_runtime_values", "Runtime Normalizer values must remain explicitly unresolved.");

const normalizerMax = run(
  mlOp("Normalizer", {}),
  staticNumericTensor("scores", "FLOAT32", [2, 2], [-2, -1, 0, 0]),
);
const normalizerMaxRow = normalizerMax.evidence.ml_value_inference.rows[0];
expectEqual(normalizerMaxRow.normalizer_mode, "MAX", "An omitted Normalizer norm must use schema default MAX.");
expectEqual(normalizerMaxRow.normalizer_mode_source, "onnx_schema_default_MAX", "Default Normalizer mode provenance must remain explicit.");
expectEqual(normalizerMaxRow.normalizer_negative_max_divisor_row_count, 1, "Signed MAX semantics must detect an all-negative row.");
expectEqual(normalizerMaxRow.normalizer_zero_divisor_row_count, 1, "A zero MAX divisor row must be counted exactly.");
expectEqual(JSON.stringify(normalizerMax.tensorMap.get("probabilities").staticValues), "[2,1,0,0]", "MAX and zero-divisor passthrough must reproduce pinned ORT FLOAT32 behavior.");

const normalizerL1 = run(
  mlOp("Normalizer", { norm: stringAttribute("L1") }),
  staticNumericTensor("scores", "FLOAT32", [3], [-1, 0, 1]),
);
expectEqual(JSON.stringify(normalizerL1.tensorMap.get("probabilities").staticValues), "[-0.5,0,0.5]", "L1 must use the absolute row sum and preserve signs.");
const normalizerL2 = run(
  mlOp("Normalizer", { norm: stringAttribute("L2") }),
  staticNumericTensor("scores", "FLOAT32", [3], [-1, 0, 1]),
);
const expectedL2 = normalizerL2.tensorMap.get("probabilities").staticValues;
expect(Math.abs(expectedL2[0] + Math.fround(Math.SQRT1_2)) < 1e-7 && expectedL2[1] === 0 && Math.abs(expectedL2[2] - Math.fround(Math.SQRT1_2)) < 1e-7, "L2 must reproduce the pinned square-sum/sqrt path.");

const normalizerLarge = run(
  mlOp("Normalizer", { norm: stringAttribute("L2") }),
  staticNumericTensor("scores", "FLOAT32", [1_000_001], new Array(1_000_001).fill(1)),
);
const normalizerLargeRow = normalizerLarge.evidence.ml_value_inference.rows[0];
expectEqual(normalizerLargeRow.normalizer_static_assessment_status, "assessed_counts_output_not_materialized_limit", "The materialization ceiling must not suppress exact Normalizer row arithmetic.");
expectEqual(normalizerLargeRow.normalizer_non_finite_output_count, 0, "Large static Normalizer inputs must still stream every output finiteness check.");
expectEqual(normalizerLargeRow.normalizer_output_materialized, false, "Large Normalizer outputs must not allocate a second million-element evidence array.");
expectEqual(normalizerLargeRow.normalizer_output_preview.length, 8, "Aggregate-only Normalizer assessment must retain a bounded output preview.");

const normalizerNonfiniteProjection = run(
  mlOp("Normalizer", { norm: stringAttribute("L2") }),
  staticNumericTensor("scores", "FLOAT32", [1], [Math.fround(3.4028234663852886e38)]),
);
const normalizerNonfiniteRow = normalizerNonfiniteProjection.evidence.ml_value_inference.rows[0];
expectEqual(normalizerNonfiniteRow.normalizer_non_finite_output_count, 1, "Finite FLOAT32 input whose pinned L2 square overflows must expose the resulting NaN exactly.");
expectEqual(normalizerNonfiniteRow.normalizer_output_preview[0], "NaN", "Non-finite Normalizer output evidence must preserve a canonical bounded preview.");
expect(normalizerNonfiniteRow.risk_codes.includes("normalizer_non_finite_float32_projection"), "A statically proven non-finite Normalizer projection must carry an explicit risk code.");

const normalizerEmpty = run(
  mlOp("Normalizer", {}),
  staticNumericTensor("scores", "FLOAT32", [0], []),
);
expectEqual(normalizerEmpty.evidence.ml_value_inference.rows[0].normalizer_negative_max_divisor_row_count, 0, "An empty row must not turn ORT's internal lowest-float sentinel into a signed-MAX risk.");

const normalizerInt32 = run(
  mlOp("Normalizer", { norm: stringAttribute("MAX") }),
  staticNumericTensor("scores", "INT32", [2], [16_777_217, 1]),
);
expectEqual(normalizerInt32.evidence.ml_value_inference.rows[0].normalizer_integer_float32_rounding_count, 1, "INT32 values changed by the required FLOAT32 output cast must be counted exactly.");
expect(normalizerInt32.evidence.ml_value_inference.rows[0].risk_codes.includes("normalizer_integer_to_float32_precision_loss"), "Integer-to-FLOAT32 precision loss must remain an explicit risk.");
const normalizerInt64 = run(
  mlOp("Normalizer", { norm: stringAttribute("MAX") }),
  exactIntegerTensor("scores", [2], ["9007199254740993", "1"]),
);
expectEqual(normalizerInt64.evidence.ml_value_inference.rows[0].normalizer_integer_float32_rounding_count, 1, "INT64-to-FLOAT32 comparison must not round through JavaScript Number before accounting.");

const normalizerL2Overflow = run(
  mlOp("Normalizer", { norm: stringAttribute("L2") }),
  staticNumericTensor("scores", "INT32", [1], [46_341]),
);
expectEqual(normalizerL2Overflow.evidence.status, "partial", "Proven INT32 square overflow must prevent a fabricated static output while retaining shape/type evidence.");
expectEqual(normalizerL2Overflow.evidence.ml_value_inference.rows[0].normalizer_signed_overflow_value_count, 1, "INT32 L2 overflow values must be counted exactly.");
expectEqual(normalizerL2Overflow.tensorMap.get("probabilities").staticValuesComplete, undefined, "Overflow-affected Normalizer output values must not propagate.");
const normalizerL1Overflow = run(
  mlOp("Normalizer", { norm: stringAttribute("L1") }),
  staticNumericTensor("scores", "INT32", [1], [-2_147_483_648]),
);
expectEqual(normalizerL1Overflow.evidence.ml_value_inference.rows[0].normalizer_signed_overflow_value_count, 1, "INT32 minimum abs overflow must be detected before emulation.");
expectEqual(run(mlOp("Normalizer", {}), tensor("scores", "FLOAT32", [1, 2, 3])).evidence.status, "fail", "Normalizer rank greater than two must fail.");
expectEqual(run(mlOp("Normalizer", { norm: stringAttribute("EUCLIDEAN") }), tensor("scores", "FLOAT32", [2])).evidence.status, "fail", "Normalizer must reject a mode outside MAX/L1/L2.");
expectEqual(run(mlOp("Normalizer", {}), tensor("scores", "UINT8", [2])).evidence.status, "fail", "Normalizer must reject a dtype outside its pinned schema set.");
const customNormalizer = run(mlOp("Normalizer", {}, "com.example"), tensor("scores", "FLOAT32", [2]), [{ domain: "com.example", version: 1 }]);
expectEqual(customNormalizer.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain Normalizer name must not inherit ai.onnx.ml semantics.");

const scalerRuntime = run(
  mlOp("Scaler", { scale: floatsAttribute([2, -3, 0.5]), offset: floatsAttribute([1, 2, -4]) }),
  tensor("scores", "FLOAT64", [2, 3]),
);
const scalerRuntimeRow = scalerRuntime.evidence.ml_value_inference.rows[0];
expectEqual(scalerRuntime.evidence.status, "assessed", "Runtime-valued Scaler should close its source-pinned type, shape, and parameter contract.");
expectEqual(scalerRuntimeRow.scaler_feature_stride, 3, "Rank-2 Scaler must use dimension 1 as the pinned ORT feature stride.");
expectEqual(scalerRuntimeRow.scaler_static_assessment_status, "not_assessed_runtime_values", "Runtime Scaler values must remain explicitly unassessed.");
expectEqual(scalerRuntime.tensorMap.get("probabilities").dtype, "FLOAT32", "Scaler output must be FLOAT32 for every accepted input dtype.");

const scalerPerFeature = run(
  mlOp("Scaler", { scale: floatsAttribute([2, -3, 0.5]), offset: floatsAttribute([1, 2, -4]) }),
  staticNumericTensor("scores", "FLOAT32", [2, 3], [2, 3, -2, 5, -1, 6]),
);
const scalerPerFeatureRow = scalerPerFeature.evidence.ml_value_inference.rows[0];
expectEqual(JSON.stringify(scalerPerFeature.tensorMap.get("probabilities").staticValues), "[2,-3,1,8,9,5]", "Per-feature Scaler must apply source-ordered offset then scale arithmetic.");
expectEqual(scalerPerFeatureRow.scaler_parameter_mode, "per_feature", "A parameter count equal to the feature stride must be classified as per-feature.");
expectEqual(scalerPerFeatureRow.scaler_exact_input_value_count, 6, "Every exact Scaler input value must be counted.");

const scalerScalar = run(
  mlOp("Scaler", { scale: floatsAttribute([2]), offset: floatsAttribute([1]) }),
  staticNumericTensor("scores", "FLOAT32", [3], [0, 1, 2]),
);
expectEqual(JSON.stringify(scalerScalar.tensorMap.get("probabilities").staticValues), "[-2,0,2]", "Scalar Scaler parameters must broadcast across the complete tensor.");
expectEqual(scalerScalar.evidence.ml_value_inference.rows[0].scaler_parameter_mode, "scalar", "Single-value scale/offset arrays must be classified as scalar parameters.");

const scalerRank3 = run(
  mlOp("Scaler", { scale: floatsAttribute([1, 10, 100]), offset: floatsAttribute([0, 0, 0]) }),
  staticNumericTensor("scores", "FLOAT32", [2, 3, 2], new Array(12).fill(1)),
);
expectEqual(JSON.stringify(scalerRank3.tensorMap.get("probabilities").staticValues), "[1,10,100,1,10,100,1,10,100,1,10,100]", "Rank-3 Scaler must follow the pinned ORT i%dim1 parameter cycle rather than the last axis.");

const scalerLarge = run(
  mlOp("Scaler", { scale: floatsAttribute([1]), offset: floatsAttribute([0]) }),
  staticNumericTensor("scores", "FLOAT32", [1_000_001], new Array(1_000_001).fill(1)),
);
const scalerLargeRow = scalerLarge.evidence.ml_value_inference.rows[0];
expectEqual(scalerLargeRow.scaler_static_assessment_status, "assessed_counts_output_not_materialized_limit", "The Scaler materialization ceiling must preserve full aggregate arithmetic.");
expectEqual(scalerLargeRow.scaler_exact_input_value_count, 1_000_001, "Aggregate-only Scaler assessment must count every exact input value.");
expectEqual(scalerLargeRow.scaler_output_materialized, false, "Large Scaler output evidence must not allocate a second million-element array.");
expectEqual(scalerLargeRow.scaler_output_preview.length, 8, "Aggregate-only Scaler evidence must retain a bounded preview.");

const scalerInt64 = run(
  mlOp("Scaler", { scale: floatsAttribute([1]), offset: floatsAttribute([0]) }),
  exactIntegerTensor("scores", [2], ["9007199254740993", "1"]),
);
expectEqual(scalerInt64.evidence.ml_value_inference.rows[0].scaler_integer_float32_rounding_count, 1, "Scaler INT64-to-FLOAT32 accounting must not round through JavaScript Number first.");
expect(scalerInt64.evidence.ml_value_inference.rows[0].risk_codes.includes("scaler_integer_to_float32_precision_loss"), "Scaler integer precision loss must remain an explicit risk.");

for (const [label, attributes, input] of [
  ["missing attributes", {}, tensor("scores", "FLOAT32", [2])],
  ["empty arrays", { scale: floatsAttribute([]), offset: floatsAttribute([]) }, tensor("scores", "FLOAT32", [2])],
  ["unequal arrays", { scale: floatsAttribute([1, 2]), offset: floatsAttribute([0]) }, tensor("scores", "FLOAT32", [2])],
  ["feature mismatch", { scale: floatsAttribute([1, 2]), offset: floatsAttribute([0, 0]) }, tensor("scores", "FLOAT32", [1, 3])],
  ["rank zero", { scale: floatsAttribute([1]), offset: floatsAttribute([0]) }, tensor("scores", "FLOAT32", [])],
]) {
  const invalid = run(mlOp("Scaler", attributes), input);
  const row = invalid.evidence.ml_value_inference.rows[0];
  expectEqual(row.scaler_parameter_contract_status, "fail", `Scaler ${label} must fail the pinned ORT runtime contract.`);
  expect(row.risk_codes.includes("scaler_pinned_ort_attribute_or_shape_contract_invalid"), `Scaler ${label} must retain the deterministic schema/runtime gap risk.`);
  expectEqual(invalid.tensorMap.get("probabilities").dtype, "UNKNOWN", `Scaler ${label} must suppress invalid output propagation.`);
}
expectEqual(run(mlOp("Scaler", { scale: stringAttribute("1"), offset: floatsAttribute([0]) }), tensor("scores", "FLOAT32", [1])).evidence.status, "fail", "Scaler must reject a non-FLOATS scale AttributeProto.");
expectEqual(run(mlOp("Scaler", { scale: floatsAttribute([1]), offset: floatsAttribute([0]) }), tensor("scores", "UINT8", [1])).evidence.status, "fail", "Scaler must reject an input dtype outside its pinned type set.");

const scalerNonfinite = run(
  mlOp("Scaler", { scale: floatsAttribute([Number.POSITIVE_INFINITY]), offset: floatsAttribute([0]) }),
  staticNumericTensor("scores", "FLOAT32", [1], [0]),
);
expectEqual(scalerNonfinite.evidence.ml_value_inference.rows[0].scaler_non_finite_parameter_count, 1, "Scaler must count non-finite parameters exactly.");
expectEqual(scalerNonfinite.evidence.ml_value_inference.rows[0].scaler_non_finite_output_count, 1, "Scaler must count the resulting non-finite output exactly.");
expect(scalerNonfinite.evidence.ml_value_inference.rows[0].risk_codes.includes("scaler_non_finite_parameter_input_or_output"), "Scaler non-finite arithmetic must carry an explicit risk.");

const scalerSignedZero = run(
  mlOp("Scaler", { scale: floatsAttribute([-0]), offset: floatsAttribute([0]) }),
  staticNumericTensor("scores", "FLOAT32", [2], [1, -1]),
);
expectEqual(scalerSignedZero.evidence.ml_value_inference.rows[0].scaler_signed_zero_output_count, 1, "Scaler must distinguish negative and positive zero outputs before public JSON canonicalization.");
expectEqual(JSON.stringify(scalerSignedZero.evidence.ml_value_inference.rows[0].scaler_output_preview), '["-0","0"]', "Scaler output preview must retain canonical signed-zero text.");
const customScaler = run(mlOp("Scaler", { scale: floatsAttribute([1]), offset: floatsAttribute([0]) }, "com.example"), tensor("scores", "FLOAT32", [1]), [{ domain: "com.example", version: 1 }]);
expectEqual(customScaler.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain Scaler name must not inherit ai.onnx.ml semantics.");

const imputerPerFeature = run(
  mlOp("Imputer", { imputed_value_floats: floatsAttribute([10, 20, 30]), replaced_value_float: floatAttribute(0) }),
  staticNumericTensor("scores", "FLOAT32", [2, 3], [0, 1, 2, 0, 1, 2]),
);
const imputerPerFeatureRow = imputerPerFeature.evidence.ml_value_inference.rows[0];
expectEqual(JSON.stringify(imputerPerFeature.tensorMap.get("probabilities").staticValues), "[10,1,2,10,1,2]", "Per-feature Imputer must cycle values with the pinned ORT i%dim1 rule.");
expectEqual(imputerPerFeatureRow.imputer_parameter_mode, "per_feature", "An imputed-value count equal to the feature stride must be per-feature.");
expectEqual(imputerPerFeatureRow.imputer_exact_replacement_count, 2, "Imputer must count exact replacements across the complete tensor.");
expectEqual(imputerPerFeatureRow.imputer_exact_unchanged_count, 4, "Imputer replacement and unchanged counts must conserve input cardinality.");
expectEqual(imputerPerFeature.evidence.ml_value_inference.tensor_imputation_node_count, 1, "Imputer must enter its dedicated contract family.");

const imputerDefaultMarker = run(
  mlOp("Imputer", { imputed_value_floats: floatsAttribute([9]) }),
  staticNumericTensor("scores", "FLOAT32", [3], [1, 0, 2]),
);
expectEqual(JSON.stringify(imputerDefaultMarker.tensorMap.get("probabilities").staticValues), "[1,9,2]", "An omitted replacement marker must use the ONNX schema default zero injected by ORT.");
expectEqual(imputerDefaultMarker.evidence.ml_value_inference.rows[0].imputer_replaced_value_source, "onnx_schema_default_0", "Default marker provenance must remain explicit.");

const imputerNan = run(
  mlOp("Imputer", { imputed_value_floats: floatsAttribute([7]), replaced_value_float: floatAttribute(Number.NaN) }),
  staticNumericTensor("scores", "FLOAT32", [3], [Number.NaN, 1, Number.NaN]),
);
expectEqual(imputerNan.evidence.ml_value_inference.rows[0].imputer_exact_nan_replacement_count, 2, "NaN marker matching must follow the pinned ORT isnan special case.");
expectEqual(JSON.stringify(imputerNan.tensorMap.get("probabilities").staticValues), "[7,1,7]", "NaN marker replacement must produce the exact finite output.");

const imputerScalarFirst = run(
  mlOp("Imputer", { imputed_value_floats: floatsAttribute([9, 8]) }),
  staticNumericTensor("scores", "FLOAT32", [1, 3], [0, 0, 1]),
);
const imputerScalarFirstRow = imputerScalarFirst.evidence.ml_value_inference.rows[0];
expectEqual(JSON.stringify(imputerScalarFirst.tensorMap.get("probabilities").staticValues), "[9,9,1]", "Pinned ORT must use the first value when imputed-value length is neither one nor the feature stride.");
expectEqual(imputerScalarFirstRow.imputer_parameter_mode, "scalar_first_fallback", "Out-of-contract cardinality must expose the actual pinned runtime fallback.");
expectEqual(imputerScalarFirstRow.imputer_ignored_imputed_value_count, 1, "Ignored trailing imputed values must be counted exactly.");
expect(imputerScalarFirstRow.risk_codes.includes("imputer_attribute_length_outside_onnx_one_or_feature_count"), "ORT scalar-first fallback must remain an explicit portability risk.");

const imputerInt64 = run(
  mlOp("Imputer", { imputed_value_int64s: intsAttribute(["9223372036854775807"]), replaced_value_int64: intAttribute("9007199254740993") }),
  exactIntegerTensor("scores", [2], ["9007199254740993", "1"]),
);
const imputerInt64Row = imputerInt64.evidence.ml_value_inference.rows[0];
expectEqual(imputerInt64Row.imputer_exact_replacement_count, 1, "INT64 Imputer comparison must remain exact beyond JavaScript Number range.");
expectEqual(imputerInt64Row.imputer_output_preview[0], "9223372036854775807", "INT64 output preview must preserve the exact replacement decimal.");
expectEqual(imputerInt64Row.imputer_output_materialized, false, "Unsafe JSON integers must remain exact aggregate evidence instead of rounded static values.");

for (const [label, attributes, input] of [
  ["missing imputed values", {}, tensor("scores", "FLOAT32", [2])],
  ["both value lists", { imputed_value_floats: floatsAttribute([1]), imputed_value_int64s: intsAttribute(["1"]) }, tensor("scores", "FLOAT32", [2])],
  ["float/int dtype mismatch", { imputed_value_int64s: intsAttribute(["1"]) }, tensor("scores", "FLOAT32", [2])],
  ["rank zero", { imputed_value_floats: floatsAttribute([1]) }, tensor("scores", "FLOAT32", [])],
]) {
  const invalid = run(mlOp("Imputer", attributes), input);
  const row = invalid.evidence.ml_value_inference.rows[0];
  expectEqual(row.imputer_parameter_contract_status, "fail", `Imputer ${label} must fail the pinned runtime contract.`);
  expect(row.risk_codes.includes("imputer_pinned_ort_attribute_or_shape_contract_invalid"), `Imputer ${label} must retain the invalid-runtime risk.`);
  expectEqual(invalid.tensorMap.get("probabilities").dtype, "UNKNOWN", `Imputer ${label} must suppress output propagation.`);
}
for (const dtype of ["FLOAT64", "INT32"]) {
  const attributes = dtype === "FLOAT64" ? { imputed_value_floats: floatsAttribute([1]) } : { imputed_value_int64s: intsAttribute(["1"]) };
  const gap = run(mlOp("Imputer", attributes), tensor("scores", dtype, [2]));
  expect(gap.evidence.ml_value_inference.rows[0].risk_codes.includes("imputer_schema_dtype_missing_pinned_ort_cpu_kernel"), `Schema-valid ${dtype} Imputer must expose the pinned ORT CPU kernel gap.`);
}
const customImputer = run(mlOp("Imputer", { imputed_value_floats: floatsAttribute([1]) }, "com.example"), tensor("scores", "FLOAT32", [1]), [{ domain: "com.example", version: 1 }]);
expectEqual(customImputer.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain Imputer name must not inherit ai.onnx.ml semantics.");

const oneHotInt64 = run(
  mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1", "2", "3"]) }),
  exactIntegerTensor("scores", [3], ["1", "3", "9"]),
);
const oneHotInt64Row = oneHotInt64.evidence.ml_value_inference.rows[0];
expectEqual(JSON.stringify(oneHotInt64.tensorMap.get("probabilities").staticValues), "[1,0,0,0,0,1,0,0,0]", "OneHotEncoder must append the category axis and emit all-zero unknown slices under default zeros=1.");
expectEqual(oneHotInt64Row.onehot_zeros_source, "onnx_schema_default_1", "Omitted zeros must preserve schema-default provenance.");
expectEqual(oneHotInt64Row.onehot_exact_matched_input_count, 2, "OneHotEncoder must count exact vocabulary matches.");
expectEqual(oneHotInt64Row.onehot_exact_unknown_input_count, 1, "OneHotEncoder must count exact unknown categories.");
expectEqual(oneHotInt64Row.onehot_exact_output_one_count, 2, "OneHotEncoder one count must equal exact matches.");
expectEqual(oneHotInt64Row.onehot_exact_output_zero_count, 7, "OneHotEncoder zero count must conserve exact output cardinality.");
expectEqual(oneHotInt64.evidence.ml_value_inference.tensor_encoder_node_count, 1, "OneHotEncoder must enter its dedicated tensor-encoder contract family.");

const oneHotFloat = run(
  mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1", "-2", "3"]) }),
  staticNumericTensor("scores", "FLOAT64", [3], [1.9, -2.2, 3]),
);
expectEqual(JSON.stringify(oneHotFloat.tensorMap.get("probabilities").staticValues), "[1,0,0,0,1,0,0,0,1]", "FLOAT64 OneHotEncoder lookup must truncate toward zero before INT64 matching.");
expectEqual(oneHotFloat.evidence.ml_value_inference.rows[0].onehot_numeric_to_int64_changed_count, 2, "Numeric truncation changes must be counted exactly.");

const oneHotString = run(
  mlOp("OneHotEncoder", { cats_strings: stringsAttribute(["cat", "dog"]) }),
  staticStringTensor("scores", [3], ["cat", "bird", "dog"]),
);
expectEqual(JSON.stringify(oneHotString.tensorMap.get("probabilities").staticValues), "[1,0,0,0,0,1]", "STRING OneHotEncoder must retain exact string identity and all-zero unknown encoding.");

const oneHotDuplicate = run(
  mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1", "1", "2"]) }),
  exactIntegerTensor("scores", [2], ["1", "2"]),
);
const oneHotDuplicateRow = oneHotDuplicate.evidence.ml_value_inference.rows[0];
expectEqual(JSON.stringify(oneHotDuplicate.tensorMap.get("probabilities").staticValues), "[0,1,0,0,0,1]", "Pinned ORT last-write-wins lookup must make the earlier duplicate category column unreachable.");
expectEqual(oneHotDuplicateRow.onehot_duplicate_category_count, 1, "Duplicate OneHotEncoder categories must be counted exactly.");
expectEqual(JSON.stringify(oneHotDuplicateRow.onehot_unreachable_duplicate_column_indices), "[0]", "The exact unreachable duplicate column index must be retained.");

const oneHotGuaranteedFailure = run(
  mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1", "2"]), zeros: intAttribute("0") }),
  exactIntegerTensor("scores", [2], ["1", "9"]),
);
expectEqual(oneHotGuaranteedFailure.evidence.ml_value_inference.rows[0].onehot_guaranteed_runtime_failure, true, "zeros=0 plus an exact unknown category must prove runtime failure.");
expectEqual(oneHotGuaranteedFailure.tensorMap.get("probabilities").dtype, "UNKNOWN", "A proven runtime failure must suppress output propagation.");

for (const [label, attributes, input] of [
  ["missing vocabulary", {}, tensor("scores", "INT64", [2])],
  ["both vocabularies", { cats_int64s: intsAttribute(["1"]), cats_strings: stringsAttribute(["1"]) }, tensor("scores", "INT64", [2])],
  ["integer/string vocabulary mismatch", { cats_strings: stringsAttribute(["1"]) }, tensor("scores", "INT64", [2])],
  ["string/integer vocabulary mismatch", { cats_int64s: intsAttribute(["1"]) }, tensor("scores", "STRING", [2])],
]) {
  const invalid = run(mlOp("OneHotEncoder", attributes), input);
  const row = invalid.evidence.ml_value_inference.rows[0];
  expectEqual(row.onehot_parameter_contract_status, "fail", `OneHotEncoder ${label} must fail its pinned runtime category contract.`);
  expect(row.risk_codes.includes("onehot_pinned_ort_attribute_contract_invalid"), `OneHotEncoder ${label} must retain an invalid-contract risk.`);
  expectEqual(invalid.tensorMap.get("probabilities").dtype, "UNKNOWN", `OneHotEncoder ${label} must suppress output propagation.`);
}

const oneHotInt32 = run(mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1"]) }), tensor("scores", "INT32", [1]));
expect(oneHotInt32.evidence.ml_value_inference.rows[0].risk_codes.includes("onehot_schema_dtype_missing_pinned_ort_cpu_kernel"), "Schema-valid INT32 OneHotEncoder must expose the pinned ORT CPU kernel gap.");
const oneHotNoncanonicalZeros = run(mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1"]), zeros: intAttribute("2") }), exactIntegerTensor("scores", [1], ["1"]));
expect(oneHotNoncanonicalZeros.evidence.ml_value_inference.rows[0].risk_codes.includes("onehot_noncanonical_zeros_boolean"), "A nonzero zeros value outside 1 must remain a portability risk while preserving pinned behavior.");
for (const value of [Number.NaN, Number.POSITIVE_INFINITY, 2 ** 63]) {
  const invalidCast = run(mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1"]) }), staticNumericTensor("scores", "FLOAT64", [1], [value]));
  expectEqual(invalidCast.evidence.ml_value_inference.rows[0].onehot_numeric_to_int64_invalid_count, 1, "Unrepresentable numeric OneHotEncoder input must be counted without emulating undefined conversion.");
  expectEqual(invalidCast.tensorMap.get("probabilities").staticValuesComplete, undefined, "Unrepresentable numeric lookup input must suppress static output values.");
}
const customOneHot = run(mlOp("OneHotEncoder", { cats_int64s: intsAttribute(["1"]) }, "com.example"), tensor("scores", "INT64", [1]), [{ domain: "com.example", version: 1 }]);
expectEqual(customOneHot.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain OneHotEncoder name must not inherit ai.onnx.ml semantics.");

const serializedNormalizer = analyzeOnnxModel(serializedNormalizerModel([-2, -1, 0, 0], null, 1, [2, 2]), "serialized_normalizer.onnx");
const serializedNormalizerRow = serializedNormalizer.onnx_shape_inference.ml_value_inference.rows[0];
const serializedNormalizerOutput = serializedNormalizer.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(serializedNormalizerRow.normalizer_mode_source, "onnx_schema_default_MAX", "Serialized Normalizer attribute omission must retain the schema-default MAX provenance.");
expectEqual(serializedNormalizerRow.normalizer_negative_max_divisor_row_count, 1, "Serialized signed-MAX arithmetic must count all-negative rows.");
expectEqual(serializedNormalizerRow.normalizer_zero_divisor_row_count, 1, "Serialized zero rows must retain pinned ORT passthrough accounting.");
expectEqual(JSON.stringify(serializedNormalizerOutput.static_values), "[2,1,0,0]", "Serialized Normalizer output must propagate exact pinned FLOAT32 values.");
const serializedNormalizerFindings = buildFindingsRegister(serializedNormalizer);
expectEqual(serializedNormalizerFindings.find((item) => item.finding_id === "EA-ONX-0021")?.technical_priority, "Medium", "Negative signed-MAX semantics must enter the Medium action queue.");
const serializedNormalizerBundle = bundle(serializedNormalizer);
expectEqual(serializedNormalizerBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized Normalizer row arithmetic must pass independent export reconstruction.");
expect(serializedNormalizerBundle.report.includes("Normalizer exact row effects") && serializedNormalizerBundle.report.includes("negative signed-MAX rows 1"), "Engineering Report must expose exact Normalizer row effects.");
expect(serializedNormalizerBundle.report.includes("50b0a8eb826fd730b3c895f5493d36a8c12e477e9b91337b10170413b73af20c"), "Engineering Report must preserve the pinned ORT Normalizer kernel hash.");
expectCompactMlBom(serializedNormalizerBundle.mlBom, ["deepbom:model:onnxMlExactNormalizerNegativeMaxRows"], "Normalizer compact ML-BOM");
const ortNormalizerSession = await ort.InferenceSession.create(serializedNormalizerModel([-2, -1, 0, 0], null, 1, [2, 2]), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortNormalizerOutput = await ortNormalizerSession.run({});
expectEqual(JSON.stringify([...ortNormalizerOutput.probabilities.data]), JSON.stringify(serializedNormalizerOutput.static_values), "Pinned ORT default-MAX output must equal the independently emulated static tensor.");

const serializedNormalizerSignedZero = analyzeOnnxModel(serializedNormalizerModel([-0, 0]), "serialized_normalizer_signed_zero.onnx");
const serializedNormalizerSignedZeroRow = serializedNormalizerSignedZero.onnx_shape_inference.ml_value_inference.rows[0];
const serializedNormalizerSignedZeroInput = serializedNormalizerSignedZero.tensors.find((tensor) => tensor.name === "scores");
const serializedNormalizerSignedZeroOutput = serializedNormalizerSignedZero.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(serializedNormalizerSignedZeroRow.normalizer_divisor_preview[0], "-0", "MAX tie handling must retain the first signed-zero value like std::max.");
expectEqual(JSON.stringify(serializedNormalizerSignedZeroRow.normalizer_output_preview), '["-0","0"]', "Zero-divisor passthrough must preserve signed-zero output evidence.");
expectEqual(serializedNormalizerSignedZeroRow.normalizer_signed_zero_output_count, 1, "Normalizer arithmetic must count signed-zero outputs exactly.");
for (const tensor of [serializedNormalizerSignedZeroInput, serializedNormalizerSignedZeroOutput]) {
  expectEqual(JSON.stringify(tensor.static_values), "[0,0]", "Public static values must canonicalize signed zero for JSON safety.");
  expect(!tensor.static_values.some((value) => Object.is(value, -0)), "Public static values must not contain a non-JSON-distinguishable negative zero.");
  expectEqual(tensor.static_values_negative_zero_count, 1, "Public static evidence must count canonicalized signed zeros.");
  expectEqual(JSON.stringify(tensor.static_values_negative_zero_indices), "[0]", "Public static evidence must retain each signed-zero index exactly.");
}
const serializedNormalizerSignedZeroBundle = bundle(serializedNormalizerSignedZero);
expectEqual(serializedNormalizerSignedZeroBundle.evidence.evidence?.conformance_report?.status, "pass", "Independent reconstruction must preserve signed-zero tie order.");
const signedZeroFieldCoverage = serializedNormalizerSignedZeroBundle.evidence.evidence?.metric_coverage_manifest?.field_coverage;
for (const fieldPath of [
  "/tensors/[]/static_values_negative_zero_count",
  "/tensors/[]/static_values_negative_zero_indices/[]",
  "/onnx_shape_inference/ml_value_inference/exact_normalizer_signed_zero_output_count",
  "/onnx_shape_inference/ml_value_inference/rows/[]/normalizer_signed_zero_output_count",
]) {
  expect(signedZeroFieldCoverage.required_report_field_paths.includes(fieldPath), `Metric coverage must classify ${fieldPath} as a required Engineering Report field.`);
  expectEqual(signedZeroFieldCoverage.field_ledger.find((item) => item.field_path === fieldPath)?.engineering_report_access, "consumed", `Engineering Report must consume ${fieldPath}.`);
}
expect(serializedNormalizerSignedZeroBundle.report.includes("1 signed-zero output(s)")
  && serializedNormalizerSignedZeroBundle.report.includes("2 signed-zero value(s) across 2 tensor(s)"), "Engineering Report must expose Normalizer and tensor-ledger signed-zero counts.");
expectCompactMlBom(serializedNormalizerSignedZeroBundle.mlBom, [
  "deepbom:model:onnxMlExactNormalizerSignedZeroOutputs",
  "deepbom:model:onnxStaticSignedZeroValues",
], "Normalizer signed-zero compact ML-BOM");
const ortNormalizerSignedZeroSession = await ort.InferenceSession.create(serializedNormalizerModel([-0, 0]), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortNormalizerSignedZeroOutput = await ortNormalizerSignedZeroSession.run({});
expect(Object.is(ortNormalizerSignedZeroOutput.probabilities.data[0], -0) && Object.is(ortNormalizerSignedZeroOutput.probabilities.data[1], 0), "Pinned ORT zero-divisor passthrough must preserve both signed zeros.");

const serializedNormalizerL2 = analyzeOnnxModel(serializedNormalizerModel([-1, 0, 1], "L2"), "serialized_normalizer_l2.onnx");
const serializedNormalizerL2Values = serializedNormalizerL2.tensors.find((tensor) => tensor.name === "probabilities").static_values;
const ortNormalizerL2Session = await ort.InferenceSession.create(serializedNormalizerModel([-1, 0, 1], "L2"), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortNormalizerL2Output = await ortNormalizerL2Session.run({});
expect([...ortNormalizerL2Output.probabilities.data].every((value, index) => Object.is(value, serializedNormalizerL2Values[index]) || value === serializedNormalizerL2Values[index]), "Pinned ORT L2 output bits must equal the source-order FLOAT32 reconstruction.");

const serializedNormalizerInt = analyzeOnnxModel(serializedNormalizerModel([16_777_217, 1], "MAX", 6), "serialized_normalizer_int32.onnx");
expectEqual(buildFindingsRegister(serializedNormalizerInt).find((item) => item.finding_id === "EA-ONX-0022")?.technical_priority, "Medium", "Exact integer-to-FLOAT32 changes must enter the Medium action queue.");
expectEqual(bundle(serializedNormalizerInt).evidence.evidence?.conformance_report?.status, "pass", "Integer Normalizer precision evidence must pass independent reconstruction.");
const serializedNormalizerInt64 = analyzeOnnxModel(serializedNormalizerModel([9_007_199_254_740_993n, 1n], "MAX", 7), "serialized_normalizer_int64.onnx");
const serializedNormalizerInt64Values = serializedNormalizerInt64.tensors.find((tensor) => tensor.name === "probabilities").static_values;
expectEqual(serializedNormalizerInt64.onnx_shape_inference.ml_value_inference.rows[0].normalizer_integer_float32_rounding_count, 1, "Serialized INT64 values must be compared to FLOAT32 without an intermediate JavaScript Number rounding.");
const ortNormalizerInt64Session = await ort.InferenceSession.create(serializedNormalizerModel([9_007_199_254_740_993n, 1n], "MAX", 7), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortNormalizerInt64Output = await ortNormalizerInt64Session.run({});
expect([...ortNormalizerInt64Output.probabilities.data].every((value, index) => Object.is(value, serializedNormalizerInt64Values[index]) || value === serializedNormalizerInt64Values[index]), "Pinned ORT INT64 MAX output bits must equal exact-decimal-to-FLOAT32 reconstruction.");

const serializedNormalizerOverflow = analyzeOnnxModel(serializedNormalizerModel([46_341], "L2", 6), "serialized_normalizer_overflow.onnx");
expectEqual(buildFindingsRegister(serializedNormalizerOverflow).find((item) => item.finding_id === "EA-ONX-0020")?.technical_priority, "High", "Proven signed square overflow must enter the High action queue.");
expectEqual(bundle(serializedNormalizerOverflow).evidence.evidence?.conformance_report?.status, "pass", "A faithfully suppressed overflow output must pass independent reconstruction.");

const serializedNormalizerNonfinite = analyzeOnnxModel(serializedNormalizerModel([Math.fround(3.4028234663852886e38)], "L2"), "serialized_normalizer_nonfinite.onnx");
expectEqual(buildFindingsRegister(serializedNormalizerNonfinite).find((item) => item.finding_id === "EA-ONX-0023")?.technical_priority, "High", "A proven non-finite FLOAT32 projection must enter the High action queue.");
expectEqual(bundle(serializedNormalizerNonfinite).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported non-finite Normalizer projection must pass independent reconstruction.");
const ortNormalizerNonfiniteSession = await ort.InferenceSession.create(serializedNormalizerModel([Math.fround(3.4028234663852886e38)], "L2"), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortNormalizerNonfiniteOutput = await ortNormalizerNonfiniteSession.run({});
expect(Number.isNaN(ortNormalizerNonfiniteOutput.probabilities.data[0]), "Pinned ORT must reproduce the statically proven Infinity/Infinity Normalizer projection as NaN.");

const serializedScaler = analyzeOnnxModel(
  serializedScalerModel([2, 3, -2, 5, -1, 6], [2, -3, 0.5], [1, 2, -4], 1, [2, 3]),
  "serialized_scaler.onnx",
);
const serializedScalerRow = serializedScaler.onnx_shape_inference.ml_value_inference.rows[0];
const serializedScalerOutput = serializedScaler.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(serializedScaler.onnx_shape_inference.status, "assessed", "Serialized Scaler should preserve exact source-pinned affine arithmetic through the real parser.");
expectEqual(JSON.stringify(serializedScalerOutput.static_values), "[2,-3,1,8,9,5]", "Serialized Scaler output must propagate exact FLOAT32 values.");
expectEqual(serializedScalerRow.scaler_feature_stride, 3, "Serialized rank-2 Scaler must bind per-feature attributes to dimension 1.");
expectEqual(serializedScalerRow.scaler_scale_values.join(","), "2,-3,0.5", "Serialized Scaler must preserve every canonical FLOATS scale value.");
const serializedScalerBundle = bundle(serializedScaler);
expectEqual(serializedScalerBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized Scaler arithmetic must pass independent export reconstruction.");
expect(serializedScalerBundle.report.includes("Scaler exact affine effects")
  && serializedScalerBundle.report.includes("per_feature; feature stride 3")
  && serializedScalerBundle.report.includes("integer->FLOAT32 changed 0"), "Engineering Report must expose exact Scaler parameter and arithmetic evidence.");
expect(serializedScalerBundle.report.includes("08ee63f5e1b4a2341f190537198761d648528262752e4cf24b083cbee1fdaee3"), "Engineering Report must preserve the pinned ORT Scaler kernel hash.");
expectCompactMlBom(serializedScalerBundle.mlBom, ["deepbom:model:onnxMlExactScalerInputValues"], "Scaler compact ML-BOM");
const scalerFieldCoverage = serializedScalerBundle.evidence.evidence?.metric_coverage_manifest?.field_coverage;
for (const fieldPath of [
  "/onnx_shape_inference/ml_value_inference/tensor_affine_scaler_node_count",
  "/onnx_shape_inference/ml_value_inference/exact_scaler_input_value_count",
  "/onnx_shape_inference/ml_value_inference/exact_scaler_integer_float32_rounding_count",
  "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_parameter_contract_status",
  "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_scale_values/[]",
  "/onnx_shape_inference/ml_value_inference/rows/[]/scaler_output_preview/[]",
]) {
  expect(scalerFieldCoverage.required_report_field_paths.includes(fieldPath), `Metric coverage must classify ${fieldPath} as a required Engineering Report field.`);
  expectEqual(scalerFieldCoverage.field_ledger.find((item) => item.field_path === fieldPath)?.engineering_report_access, "consumed", `Engineering Report must consume ${fieldPath}.`);
}
const ortScalerSession = await ort.InferenceSession.create(
  serializedScalerModel([2, 3, -2, 5, -1, 6], [2, -3, 0.5], [1, 2, -4], 1, [2, 3]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortScalerOutput = await ortScalerSession.run({});
expect([...ortScalerOutput.probabilities.data].every((value, index) => Object.is(value, serializedScalerOutput.static_values[index]) || value === serializedScalerOutput.static_values[index]), "Pinned ORT Scaler output bits must equal the independent source-order reconstruction.");

const serializedScalerRank3 = analyzeOnnxModel(
  serializedScalerModel(new Array(12).fill(1), [1, 10, 100], [0, 0, 0], 1, [2, 3, 2]),
  "serialized_scaler_rank3.onnx",
);
const serializedScalerRank3Values = serializedScalerRank3.tensors.find((tensor) => tensor.name === "probabilities").static_values;
expectEqual(JSON.stringify(serializedScalerRank3Values), "[1,10,100,1,10,100,1,10,100,1,10,100]", "Serialized rank-3 Scaler must retain pinned dim1 cyclic indexing.");
const ortScalerRank3Session = await ort.InferenceSession.create(
  serializedScalerModel(new Array(12).fill(1), [1, 10, 100], [0, 0, 0], 1, [2, 3, 2]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortScalerRank3Output = await ortScalerRank3Session.run({});
expectEqual(JSON.stringify([...ortScalerRank3Output.probabilities.data]), JSON.stringify(serializedScalerRank3Values), "Pinned ORT rank-3 execution must confirm the unusual dim1 feature cycle.");

const scalerDoubleInput = 9.537696838378906e-7;
const scalerDoubleOffset = 9.5367431640625e-7;
const scalerDoubleScale = 1.0000001192092896;
const serializedScalerFloat64 = analyzeOnnxModel(
  serializedScalerModel([scalerDoubleInput], [scalerDoubleScale], [scalerDoubleOffset], 11),
  "serialized_scaler_float64.onnx",
);
const serializedScalerFloat64Value = serializedScalerFloat64.tensors.find((tensor) => tensor.name === "probabilities").static_values[0];
const incorrectlyPrecastScalerValue = Math.fround(Math.fround(Math.fround(scalerDoubleInput) - Math.fround(scalerDoubleOffset)) * Math.fround(scalerDoubleScale));
expect(!Object.is(serializedScalerFloat64Value, incorrectlyPrecastScalerValue), "FLOAT64 Scaler fixture must distinguish double source arithmetic from an incorrect input pre-cast.");
const ortScalerFloat64Session = await ort.InferenceSession.create(
  serializedScalerModel([scalerDoubleInput], [scalerDoubleScale], [scalerDoubleOffset], 11),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortScalerFloat64Output = await ortScalerFloat64Session.run({});
expect(Object.is(ortScalerFloat64Output.probabilities.data[0], serializedScalerFloat64Value), "Pinned ORT FLOAT64 Scaler output must match double arithmetic followed by one FLOAT32 cast.");

const serializedScalerInt64 = analyzeOnnxModel(
  serializedScalerModel([9_007_199_254_740_993n, 1n], [1], [0], 7),
  "serialized_scaler_int64.onnx",
);
const serializedScalerInt64Row = serializedScalerInt64.onnx_shape_inference.ml_value_inference.rows[0];
const serializedScalerInt64Values = serializedScalerInt64.tensors.find((tensor) => tensor.name === "probabilities").static_values;
expectEqual(serializedScalerInt64Row.scaler_integer_float32_rounding_count, 1, "Serialized Scaler INT64 precision accounting must use exact decimal values.");
expectEqual(buildFindingsRegister(serializedScalerInt64).find((item) => item.finding_id === "EA-ONX-0025")?.technical_priority, "Medium", "Scaler integer precision loss must enter the Medium action queue.");
expectEqual(bundle(serializedScalerInt64).evidence.evidence?.conformance_report?.status, "pass", "Exact INT64 Scaler evidence must pass independent reconstruction.");
const ortScalerInt64Session = await ort.InferenceSession.create(
  serializedScalerModel([9_007_199_254_740_993n, 1n], [1], [0], 7),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortScalerInt64Output = await ortScalerInt64Session.run({});
expect([...ortScalerInt64Output.probabilities.data].every((value, index) => Object.is(value, serializedScalerInt64Values[index]) || value === serializedScalerInt64Values[index]), "Pinned ORT INT64 Scaler output bits must match exact-decimal-to-FLOAT32 reconstruction.");

const serializedScalerInvalid = analyzeOnnxModel(serializedScalerModel([1, 2], null, null), "serialized_scaler_missing_attributes.onnx");
const serializedScalerInvalidRow = serializedScalerInvalid.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedScalerInvalidRow.scaler_parameter_contract_status, "fail", "Missing optional-by-schema Scaler attributes must be classified as pinned-runtime invalid.");
expectEqual(buildFindingsRegister(serializedScalerInvalid).find((item) => item.finding_id === "EA-ONX-0024")?.technical_priority, "High", "A deterministic Scaler schema/runtime gap must enter the High action queue.");
expectEqual(bundle(serializedScalerInvalid).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported pinned-runtime-invalid Scaler must pass export conformance.");
let ortScalerMissingRejected = false;
try {
  await ort.InferenceSession.create(serializedScalerModel([1, 2], null, null), {
    executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
  });
} catch {
  ortScalerMissingRejected = true;
}
expect(ortScalerMissingRejected, "Pinned ORT must reject Scaler with schema-optional but kernel-required attributes missing.");

const serializedScalerSignedZero = analyzeOnnxModel(
  serializedScalerModel([1, -1], [-0], [0]),
  "serialized_scaler_signed_zero.onnx",
);
const serializedScalerSignedZeroRow = serializedScalerSignedZero.onnx_shape_inference.ml_value_inference.rows[0];
const serializedScalerSignedZeroOutput = serializedScalerSignedZero.tensors.find((tensor) => tensor.name === "probabilities");
const serializedScalerScaleAttribute = serializedScalerSignedZero.ops[0].onnx_attributes.find((attribute) => attribute.name === "scale");
expectEqual(serializedScalerScaleAttribute.float_values[0], 0, "Public Scaler attribute JSON must canonicalize negative zero numerically.");
expect(!Object.is(serializedScalerScaleAttribute.float_values[0], -0), "Public Scaler attribute numeric mirror must not leak JSON-indistinguishable negative zero.");
expectEqual(serializedScalerScaleAttribute.float_values_text[0], "-0", "Public Scaler attribute evidence must preserve the exact signed-zero text.");
expectEqual(serializedScalerSignedZeroRow.scaler_signed_zero_output_count, 1, "Serialized Scaler must count signed-zero output positions.");
expectEqual(JSON.stringify(serializedScalerSignedZeroOutput.static_values), "[0,0]", "Public Scaler output values must be JSON-safe.");
expectEqual(JSON.stringify(serializedScalerSignedZeroOutput.static_values_negative_zero_indices), "[0]", "Public Scaler output evidence must preserve the signed-zero index.");
expectEqual(bundle(serializedScalerSignedZero).evidence.evidence?.conformance_report?.status, "pass", "Scaler signed-zero attribute and output ledgers must pass independent reconstruction.");

const serializedScalerNonfinite = analyzeOnnxModel(
  serializedScalerModel([0], [Number.POSITIVE_INFINITY], [0]),
  "serialized_scaler_nonfinite.onnx",
);
expectEqual(buildFindingsRegister(serializedScalerNonfinite).find((item) => item.finding_id === "EA-ONX-0026")?.technical_priority, "High", "Scaler non-finite arithmetic must enter the High action queue.");
expectEqual(bundle(serializedScalerNonfinite).evidence.evidence?.conformance_report?.status, "pass", "Faithfully reported non-finite Scaler arithmetic must pass independent reconstruction.");
const ortScalerNonfiniteSession = await ort.InferenceSession.create(
  serializedScalerModel([0], [Number.POSITIVE_INFINITY], [0]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortScalerNonfiniteOutput = await ortScalerNonfiniteSession.run({});
expect(Number.isNaN(ortScalerNonfiniteOutput.probabilities.data[0]), "Pinned ORT must reproduce zero multiplied by infinite scale as NaN.");

const serializedImputer = analyzeOnnxModel(
  serializedImputerModel([0, 1, 2, 0, 1, 2], [10, 20, 30], null, 1, [2, 3]),
  "serialized_imputer.onnx",
);
const serializedImputerRow = serializedImputer.onnx_shape_inference.ml_value_inference.rows[0];
const serializedImputerOutput = serializedImputer.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(serializedImputer.onnx_shape_inference.status, "assessed", "Serialized Imputer must preserve exact source-pinned replacement arithmetic through the real parser.");
expectEqual(JSON.stringify(serializedImputerOutput.static_values), "[10,1,2,10,1,2]", "Serialized Imputer must propagate exact per-feature output values.");
expectEqual(serializedImputerRow.imputer_replaced_value_source, "onnx_schema_default_0", "Serialized Imputer must preserve schema-default marker provenance.");
expectEqual(serializedImputerRow.imputer_exact_replacement_count, 2, "Serialized Imputer must count exact replacements.");
const serializedImputerBundle = bundle(serializedImputer);
expectEqual(serializedImputerBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized Imputer arithmetic must pass independent export reconstruction.");
expect(serializedImputerBundle.report.includes("Imputer exact replacement effects")
  && serializedImputerBundle.report.includes("2 replaced + 4 unchanged")
  && serializedImputerBundle.report.includes("per_feature"), "Engineering Report must expose exact Imputer contract and replacement arithmetic.");
expect(serializedImputerBundle.report.includes("10de709f7625306815ef374afdf3fd6dd930b4c8a6bd65e8f0fc348cda5f4dea"), "Engineering Report must preserve the pinned ORT Imputer kernel hash.");
expectCompactMlBom(serializedImputerBundle.mlBom, ["deepbom:model:onnxMlExactImputerReplacements"], "Imputer compact ML-BOM");
const imputerFieldCoverage = serializedImputerBundle.evidence.evidence?.metric_coverage_manifest?.field_coverage;
for (const fieldPath of [
  "/onnx_shape_inference/ml_value_inference/tensor_imputation_node_count",
  "/onnx_shape_inference/ml_value_inference/exact_imputer_replacement_count",
  "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_parameter_contract_status",
  "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_imputed_values/[]",
  "/onnx_shape_inference/ml_value_inference/rows/[]/imputer_output_preview/[]",
]) {
  expect(imputerFieldCoverage.required_report_field_paths.includes(fieldPath), `Metric coverage must classify ${fieldPath} as a required Engineering Report field.`);
  expectEqual(imputerFieldCoverage.field_ledger.find((item) => item.field_path === fieldPath)?.engineering_report_access, "consumed", `Engineering Report must consume ${fieldPath}.`);
}
const ortImputerSession = await ort.InferenceSession.create(
  serializedImputerModel([0, 1, 2, 0, 1, 2], [10, 20, 30], null, 1, [2, 3]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortImputerOutput = await ortImputerSession.run({});
expectEqual(JSON.stringify([...ortImputerOutput.probabilities.data]), JSON.stringify(serializedImputerOutput.static_values), "Pinned ORT Imputer output must equal independent replacement reconstruction.");

const serializedImputerFallback = analyzeOnnxModel(
  serializedImputerModel([0, 0, 1], [9, 8], null, 1, [1, 3]),
  "serialized_imputer_scalar_first.onnx",
);
expectEqual(JSON.stringify(serializedImputerFallback.tensors.find((tensor) => tensor.name === "probabilities").static_values), "[9,9,1]", "Serialized Imputer fallback must use only the first configured value.");
expectEqual(buildFindingsRegister(serializedImputerFallback).find((item) => item.finding_id === "EA-ONX-0028")?.technical_priority, "High", "Imputer scalar-first fallback must enter the High action queue.");
expectEqual(bundle(serializedImputerFallback).evidence.evidence?.conformance_report?.status, "pass", "Faithfully reported Imputer fallback must pass export conformance.");
const ortImputerFallbackSession = await ort.InferenceSession.create(
  serializedImputerModel([0, 0, 1], [9, 8], null, 1, [1, 3]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortImputerFallbackOutput = await ortImputerFallbackSession.run({});
expectEqual(JSON.stringify([...ortImputerFallbackOutput.probabilities.data]), "[9,9,1]", "Pinned ORT must confirm scalar-first fallback for a length outside one or F.");

const serializedImputerInt64 = analyzeOnnxModel(
  serializedImputerModel([2n, 0n, 2n], [10n, 20n, 30n], 2n, 7, [1, 3]),
  "serialized_imputer_int64.onnx",
);
expectEqual(JSON.stringify(serializedImputerInt64.tensors.find((tensor) => tensor.name === "probabilities").static_values), "[10,0,30]", "Serialized INT64 Imputer must preserve exact per-feature replacement values within JSON-safe range.");
expectEqual(bundle(serializedImputerInt64).evidence.evidence?.conformance_report?.status, "pass", "INT64 Imputer evidence must pass independent reconstruction.");
const ortImputerInt64Session = await ort.InferenceSession.create(
  serializedImputerModel([2n, 0n, 2n], [10n, 20n, 30n], 2n, 7, [1, 3]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortImputerInt64Output = await ortImputerInt64Session.run({});
expectEqual(JSON.stringify([...ortImputerInt64Output.probabilities.data].map(String)), '["10","0","30"]', "Pinned ORT INT64 Imputer output must match exact integer reconstruction.");

const serializedImputerMissing = analyzeOnnxModel(serializedImputerModel([0, 1], null), "serialized_imputer_missing.onnx");
expectEqual(buildFindingsRegister(serializedImputerMissing).find((item) => item.finding_id === "EA-ONX-0027")?.technical_priority, "High", "Missing Imputer values must enter the High action queue.");
expectEqual(bundle(serializedImputerMissing).evidence.evidence?.conformance_report?.status, "pass", "Faithfully reported invalid Imputer contract must pass export conformance.");
let ortImputerMissingRejected = false;
try {
  await ort.InferenceSession.create(serializedImputerModel([0, 1], null), { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" });
} catch {
  ortImputerMissingRejected = true;
}
expect(ortImputerMissingRejected, "Pinned ORT must reject Imputer without a nonempty imputed-value list.");

const serializedImputerFloat64 = analyzeOnnxModel(serializedImputerModel([0, 1], [9], null, 11), "serialized_imputer_float64.onnx");
expectEqual(buildFindingsRegister(serializedImputerFloat64).find((item) => item.finding_id === "EA-ONX-0029")?.technical_priority, "High", "Schema-valid FLOAT64 Imputer without a pinned CPU kernel must enter the High action queue.");
expectEqual(bundle(serializedImputerFloat64).evidence.evidence?.conformance_report?.status, "pass", "Imputer schema/runtime dtype separation must pass export conformance.");
let ortImputerFloat64Rejected = false;
try {
  await ort.InferenceSession.create(serializedImputerModel([0, 1], [9], null, 11), { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" });
} catch {
  ortImputerFloat64Rejected = true;
}
expect(ortImputerFloat64Rejected, "Pinned ORT CPU must reject schema-valid FLOAT64 Imputer.");

const serializedOneHotInt64 = analyzeOnnxModel(
  serializedOneHotEncoderModel([1n, 3n, 9n], [1n, 2n, 3n]),
  "serialized_one_hot_int64.onnx",
);
const serializedOneHotInt64Row = serializedOneHotInt64.onnx_shape_inference.ml_value_inference.rows[0];
const serializedOneHotInt64Output = serializedOneHotInt64.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(JSON.stringify(serializedOneHotInt64Output.static_values), "[1,0,0,0,0,1,0,0,0]", "Serialized INT64 OneHotEncoder must preserve exact match and all-zero unknown output values.");
expectEqual(JSON.stringify(serializedOneHotInt64Row.exact_output_shape), "[3,3]", "Serialized OneHotEncoder must append exact vocabulary cardinality to output shape.");
expectEqual(serializedOneHotInt64Row.onehot_exact_input_value_count, 3, "Serialized OneHotEncoder must retain exact input cardinality.");
expectEqual(serializedOneHotInt64Row.onehot_exact_matched_input_count, 2, "Serialized OneHotEncoder must retain exact match count.");
expectEqual(serializedOneHotInt64Row.onehot_exact_unknown_input_count, 1, "Serialized OneHotEncoder must retain exact unknown count.");
const serializedOneHotInt64Finding = buildFindingsRegister(serializedOneHotInt64).find((item) => item.finding_id === "EA-ONX-0033");
expectEqual(serializedOneHotInt64Finding?.technical_priority, "Medium", "Artifact-known all-zero unknown encoding must enter the Medium action queue.");
const serializedOneHotInt64Bundle = bundle(serializedOneHotInt64);
expectEqual(serializedOneHotInt64Bundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized OneHotEncoder evidence must pass independent export reconstruction.");
expect(serializedOneHotInt64Bundle.report.includes("OneHotEncoder exact encoding effects")
  && serializedOneHotInt64Bundle.report.includes("2 matched + 1 unknown")
  && serializedOneHotInt64Bundle.report.includes("guaranteed runtime failure no"), "Engineering Report must expose exact OneHotEncoder arithmetic and runtime branch state.");
expect(serializedOneHotInt64Bundle.report.includes("8b5bd9bcdf8455326ec857b540743465cf57bdb50f7f26820f734a179c3431ef"), "Engineering Report must preserve the pinned ORT OneHotEncoder kernel hash.");
expectCompactMlBom(serializedOneHotInt64Bundle.mlBom, ["deepbom:model:onnxMlExactOneHotUnknownInputs"], "OneHotEncoder compact ML-BOM");
const oneHotFieldCoverage = serializedOneHotInt64Bundle.evidence.evidence?.metric_coverage_manifest?.field_coverage;
for (const path of [
  "/onnx_shape_inference/ml_value_inference/tensor_encoder_node_count",
  "/onnx_shape_inference/ml_value_inference/exact_onehot_matched_input_count",
  "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_parameter_contract_status",
  "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_category_values/[]",
  "/onnx_shape_inference/ml_value_inference/rows/[]/onehot_output_preview/[]",
]) {
  expect(oneHotFieldCoverage.required_report_field_paths.includes(path), `Metric coverage must classify ${path} as a required Engineering Report field.`);
  expectEqual(oneHotFieldCoverage.field_ledger.find((item) => item.field_path === path)?.engineering_report_access, "consumed", `Engineering Report must consume ${path}.`);
}
const ortOneHotInt64Session = await ort.InferenceSession.create(
  serializedOneHotEncoderModel([1n, 3n, 9n], [1n, 2n, 3n]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortOneHotInt64Output = await ortOneHotInt64Session.run({});
expectEqual(JSON.stringify([...ortOneHotInt64Output.probabilities.data]), JSON.stringify(serializedOneHotInt64Output.static_values), "Pinned ORT INT64 OneHotEncoder output must equal independent exact reconstruction.");

const serializedOneHotFloat64 = analyzeOnnxModel(
  serializedOneHotEncoderModel([1.9, -2.2, 3], [1n, -2n, 3n], null, null, 11),
  "serialized_one_hot_float64.onnx",
);
const serializedOneHotFloat64Output = serializedOneHotFloat64.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(JSON.stringify(serializedOneHotFloat64Output.static_values), "[1,0,0,0,1,0,0,0,1]", "Serialized FLOAT64 OneHotEncoder must use pinned truncation-toward-zero lookup semantics.");
expectEqual(serializedOneHotFloat64.onnx_shape_inference.ml_value_inference.rows[0].onehot_numeric_to_int64_changed_count, 2, "Serialized FLOAT64 lookup must count exact cast changes.");
expectEqual(bundle(serializedOneHotFloat64).evidence.evidence?.conformance_report?.status, "pass", "FLOAT64 OneHotEncoder cast evidence must pass independent reconstruction.");
const ortOneHotFloat64Session = await ort.InferenceSession.create(
  serializedOneHotEncoderModel([1.9, -2.2, 3], [1n, -2n, 3n], null, null, 11),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortOneHotFloat64Output = await ortOneHotFloat64Session.run({});
expectEqual(JSON.stringify([...ortOneHotFloat64Output.probabilities.data]), JSON.stringify(serializedOneHotFloat64Output.static_values), "Pinned ORT FLOAT64 OneHotEncoder output must confirm numeric truncation semantics.");

const serializedOneHotString = analyzeOnnxModel(
  serializedOneHotEncoderModel(["cat", "bird", "dog"], null, ["cat", "dog"], null, 8),
  "serialized_one_hot_string.onnx",
);
const serializedOneHotStringInput = serializedOneHotString.tensors.find((tensor) => tensor.name === "scores");
const serializedOneHotStringOutput = serializedOneHotString.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(JSON.stringify(serializedOneHotStringInput.static_values), '["cat","bird","dog"]', "The real parser must decode TensorProto string_data exactly.");
expectEqual(JSON.stringify(serializedOneHotStringOutput.static_values), "[1,0,0,0,0,1]", "Serialized STRING OneHotEncoder must propagate exact output values.");
expectEqual(bundle(serializedOneHotString).evidence.evidence?.conformance_report?.status, "pass", "STRING OneHotEncoder parser and arithmetic evidence must pass independent reconstruction.");
const ortOneHotStringSession = await ort.InferenceSession.create(
  serializedOneHotEncoderModel(["cat", "bird", "dog"], null, ["cat", "dog"], null, 8),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortOneHotStringOutput = await ortOneHotStringSession.run({});
expectEqual(JSON.stringify([...ortOneHotStringOutput.probabilities.data]), JSON.stringify(serializedOneHotStringOutput.static_values), "Pinned ORT STRING OneHotEncoder output must equal exact string reconstruction.");

const serializedOneHotDuplicate = analyzeOnnxModel(
  serializedOneHotEncoderModel([1n, 2n], [1n, 1n, 2n]),
  "serialized_one_hot_duplicate.onnx",
);
expectEqual(JSON.stringify(serializedOneHotDuplicate.tensors.find((tensor) => tensor.name === "probabilities").static_values), "[0,1,0,0,0,1]", "Serialized duplicate vocabulary must preserve pinned last-write-wins output.");
expectEqual(buildFindingsRegister(serializedOneHotDuplicate).find((item) => item.finding_id === "EA-ONX-0032")?.technical_priority, "Medium", "Duplicate OneHotEncoder vocabulary must enter the Medium action queue.");
expectEqual(bundle(serializedOneHotDuplicate).evidence.evidence?.conformance_report?.status, "pass", "Duplicate-vocabulary risk must pass faithful export conformance.");
const ortOneHotDuplicateSession = await ort.InferenceSession.create(
  serializedOneHotEncoderModel([1n, 2n], [1n, 1n, 2n]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortOneHotDuplicateOutput = await ortOneHotDuplicateSession.run({});
expectEqual(JSON.stringify([...ortOneHotDuplicateOutput.probabilities.data]), "[0,1,0,0,0,1]", "Pinned ORT must confirm last-write-wins duplicate vocabulary behavior.");

const serializedOneHotFailure = analyzeOnnxModel(
  serializedOneHotEncoderModel([1n, 9n], [1n, 2n], null, 0n),
  "serialized_one_hot_unknown_failure.onnx",
);
expectEqual(buildFindingsRegister(serializedOneHotFailure).find((item) => item.finding_id === "EA-ONX-0034")?.technical_priority, "High", "zeros=0 with an exact unknown must enter the High action queue.");
expectEqual(bundle(serializedOneHotFailure).evidence.evidence?.conformance_report?.status, "pass", "Proven OneHotEncoder runtime failure must pass faithful export conformance.");
let ortOneHotUnknownRejected = false;
try {
  const session = await ort.InferenceSession.create(serializedOneHotEncoderModel([1n, 9n], [1n, 2n], null, 0n), { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" });
  await session.run({});
} catch {
  ortOneHotUnknownRejected = true;
}
expect(ortOneHotUnknownRejected, "Pinned ORT must reject an unknown category when zeros=0.");

const serializedOneHotMissing = analyzeOnnxModel(serializedOneHotEncoderModel([1n], null, null), "serialized_one_hot_missing.onnx");
expectEqual(buildFindingsRegister(serializedOneHotMissing).find((item) => item.finding_id === "EA-ONX-0031")?.technical_priority, "High", "Missing OneHotEncoder vocabulary must enter the High action queue.");
expectEqual(bundle(serializedOneHotMissing).evidence.evidence?.conformance_report?.status, "pass", "Invalid OneHotEncoder category contract must pass faithful export conformance.");
let ortOneHotMissingRejected = false;
try {
  await ort.InferenceSession.create(serializedOneHotEncoderModel([1n], null, null), { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" });
} catch {
  ortOneHotMissingRejected = true;
}
expect(ortOneHotMissingRejected, "Pinned ORT must reject OneHotEncoder without a nonempty category list.");

const serializedOneHotInt32 = analyzeOnnxModel(serializedOneHotEncoderModel([1], [1n], null, null, 6), "serialized_one_hot_int32.onnx");
expectEqual(buildFindingsRegister(serializedOneHotInt32).find((item) => item.finding_id === "EA-ONX-0035")?.technical_priority, "High", "Schema-valid INT32 OneHotEncoder must enter the High action queue for the pinned CPU kernel gap.");
expectEqual(bundle(serializedOneHotInt32).evidence.evidence?.conformance_report?.status, "pass", "INT32 OneHotEncoder schema/runtime gap must pass faithful export conformance.");
let ortOneHotInt32Rejected = false;
try {
  await ort.InferenceSession.create(serializedOneHotEncoderModel([1], [1n], null, null, 6), { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" });
} catch {
  ortOneHotInt32Rejected = true;
}
expect(ortOneHotInt32Rejected, "Pinned ORT CPU must reject schema-valid INT32 OneHotEncoder.");

const serializedOneHotNoncanonical = analyzeOnnxModel(serializedOneHotEncoderModel([1n], [1n], null, 2n), "serialized_one_hot_noncanonical_zeros.onnx");
expectEqual(buildFindingsRegister(serializedOneHotNoncanonical).find((item) => item.finding_id === "EA-ONX-0036")?.technical_priority, "Medium", "Noncanonical OneHotEncoder zeros must enter the Medium action queue.");
expectEqual(bundle(serializedOneHotNoncanonical).evidence.evidence?.conformance_report?.status, "pass", "Noncanonical zeros behavior must pass faithful export conformance.");

const serializedOneHotInvalidCast = analyzeOnnxModel(serializedOneHotEncoderModel([2 ** 63], [1n], null, null, 11), "serialized_one_hot_invalid_cast.onnx");
expectEqual(buildFindingsRegister(serializedOneHotInvalidCast).find((item) => item.finding_id === "EA-ONX-0037")?.technical_priority, "High", "Unrepresentable OneHotEncoder cast must enter the High action queue.");
expectEqual(bundle(serializedOneHotInvalidCast).evidence.evidence?.conformance_report?.status, "pass", "Refusal to emulate an unrepresentable cast must pass independent conformance.");

const directLinearClassifier = runLinearClassifier(
  linearClassifierOp({
    coefficients: floatsAttribute([1, 0, 0, 1]), intercepts: floatsAttribute([0, 0]),
    classlabels_ints: intsAttribute(["10", "20"]),
  }),
  staticNumericTensor("scores", "FLOAT32", [2, 2], [1, 2, 3, 1]),
);
const directLinearClassifierRow = directLinearClassifier.evidence.ml_value_inference.rows[0];
expectEqual(directLinearClassifier.evidence.status, "assessed", "A complete LinearClassifier-1 contract must be fully assessed.");
expectEqual(JSON.stringify(directLinearClassifier.tensorMap.get("labels").shape), "[2]", "LinearClassifier labels must have exact batch shape.");
expectEqual(JSON.stringify(directLinearClassifier.tensorMap.get("probabilities").shape), "[2,2]", "LinearClassifier scores must have exact batch/class shape.");
expectEqual(directLinearClassifierRow.linear_used_coefficient_count, 4, "LinearClassifier must consume classes multiplied by features coefficients.");
expectEqual(JSON.stringify(directLinearClassifierRow.linear_reference_raw_score_preview), '["1","2","3","1"]', "LinearClassifier scalar reference must retain exact raw-score order.");
expectEqual(JSON.stringify(directLinearClassifierRow.linear_reference_label_preview), '["20","10"]', "LinearClassifier scalar reference must apply pinned greater-than/argmax label selection.");
expectEqual(directLinearClassifier.tensorMap.get("probabilities").staticValuesComplete, undefined, "Scalar linear reference must not be promoted to runtime-bit-exact tensor evidence.");

const directBinaryClassifier = runLinearClassifier(
  linearClassifierOp({
    coefficients: floatsAttribute([1, -1]), intercepts: floatsAttribute([0]),
    classlabels_ints: intsAttribute(["0", "1"]),
  }),
  staticNumericTensor("scores", "FLOAT32", [1, 2], [3, 1]),
);
const directBinaryClassifierRow = directBinaryClassifier.evidence.ml_value_inference.rows[0];
expectEqual(directBinaryClassifierRow.classifier_binary_score_expansion, true, "One intercept plus two labels must activate pinned binary score expansion.");
expectEqual(JSON.stringify(directBinaryClassifierRow.classifier_score_output_shape), "[1,2]", "Expanded binary scores must expose two columns.");
expectEqual(JSON.stringify(directBinaryClassifierRow.linear_reference_output_preview), '["-1","2"]', "Pinned non-PROBIT binary expansion must produce complement and raw score.");

const directLinearClassifierExtra = runLinearClassifier(
  linearClassifierOp({
    coefficients: floatsAttribute([1, 0, 0, 1, 99]), intercepts: floatsAttribute([0, 0]),
    classlabels_strings: stringsAttribute(["left", "right"]),
  }),
  tensor("scores", "FLOAT32", [1, 2]),
);
expectEqual(directLinearClassifierExtra.evidence.ml_value_inference.rows[0].linear_unused_coefficient_count, 1, "Trailing LinearClassifier coefficients must be counted as ignored by pinned ORT.");
expectEqual(runLinearClassifier(
  linearClassifierOp({ coefficients: floatsAttribute([1, 0]), intercepts: floatsAttribute([0]) }),
  tensor("scores", "FLOAT32", [1, 2]),
).evidence.status, "fail", "LinearClassifier without exactly one nonempty label list must fail closed.");

const directLinearRegressor = run(
  mlOp("LinearRegressor", {
    coefficients: floatsAttribute([1, 0, 0, 1]), intercepts: floatsAttribute([0.5, -0.5]), targets: intAttribute("2"),
  }),
  staticNumericTensor("scores", "FLOAT32", [2, 2], [1, 2, 3, 1]),
);
const directLinearRegressorRow = directLinearRegressor.evidence.ml_value_inference.rows[0];
expectEqual(directLinearRegressor.evidence.status, "assessed", "A complete LinearRegressor-1 FLOAT32 contract must be fully assessed.");
expectEqual(JSON.stringify(directLinearRegressor.tensorMap.get("probabilities").shape), "[2,2]", "LinearRegressor output must have exact batch/target shape.");
expectEqual(JSON.stringify(directLinearRegressorRow.linear_reference_output_preview), '["1.5","1.5","3.5","0.5"]', "LinearRegressor scalar reference must preserve coefficient and intercept order.");
expectEqual(run(
  mlOp("LinearRegressor", { coefficients: floatsAttribute([1, 0]), intercepts: floatsAttribute([0]) }),
  tensor("scores", "FLOAT32", [1, 2]),
).evidence.ml_value_inference.rows[0].linear_pinned_ort_contract_reason, "linear_regressor_runtime_contract_resolved", "ORT schema resolution must materialize the ONNX targets=1 default before the kernel constructor.");
expectEqual(run(
  mlOp("LinearRegressor", { coefficients: floatsAttribute([1, 0]), intercepts: floatsAttribute([0]), targets: intAttribute("1") }),
  tensor("scores", "FLOAT64", [1, 2]),
).evidence.ml_value_inference.rows[0].risk_codes.includes("linear_regressor_schema_dtype_missing_pinned_ort_cpu_kernel"), true, "Schema-valid FLOAT64 LinearRegressor must retain the pinned CPU dtype gap.");

const serializedLinearClassifier = analyzeOnnxModel(
  serializedLinearClassifierModel([1, 2, 3, 1], [2, 2], [1, 0, 0, 1], [0, 0], [10n, 20n]),
  "serialized_linear_classifier.onnx",
);
const serializedLinearClassifierRow = serializedLinearClassifier.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedLinearClassifierRow.linear_pinned_ort_contract_status, "pass", "Serialized LinearClassifier must preserve the executable pinned ORT contract.");
expectEqual(JSON.stringify(serializedLinearClassifierRow.canonical_output_shapes), "[[2],[2,2]]", "Serialized LinearClassifier must preserve both output shapes in structured evidence.");
const serializedLinearClassifierBundle = bundle(serializedLinearClassifier);
expectEqual(serializedLinearClassifierBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized LinearClassifier report and ML-BOM must pass independent reconstruction.");
expect(serializedLinearClassifierBundle.report.includes("Linear model coefficient/runtime contracts")
  && serializedLinearClassifierBundle.report.includes("coefficients expected/used/serialized")
  && serializedLinearClassifierBundle.report.includes("not promoted as runtime-bit-exact tensor evidence"), "Engineering Report must expose linear parameter conservation and scalar-reference boundary.");
expect(serializedLinearClassifierBundle.report.includes("8fcef175e9db50c017a94ac4db1b3c118294dea7eab93315d58dabfdae95d052"), "Engineering Report must preserve the pinned ORT LinearClassifier kernel hash.");
expectCompactMlBom(serializedLinearClassifierBundle.mlBom, ["deepbom:model:onnxMlExactLinearUsedCoefficients"], "LinearClassifier compact ML-BOM");
const linearFieldCoverage = serializedLinearClassifierBundle.evidence.evidence?.metric_coverage_manifest?.field_coverage;
for (const path of [
  "/onnx_shape_inference/ml_value_inference/linear_classifier_node_count",
  "/onnx_shape_inference/ml_value_inference/exact_linear_used_coefficient_count",
  "/onnx_shape_inference/ml_value_inference/rows/[]/canonical_output_shapes/[]/[]",
  "/onnx_shape_inference/ml_value_inference/rows/[]/linear_pinned_ort_contract_reason",
  "/onnx_shape_inference/ml_value_inference/rows/[]/linear_reference_boundary",
]) {
  expect(linearFieldCoverage.required_report_field_paths.includes(path), `Metric coverage must classify ${path} as a required Engineering Report field.`);
  expectEqual(linearFieldCoverage.field_ledger.find((item) => item.field_path === path)?.engineering_report_access, "consumed", `Engineering Report must consume ${path}.`);
}
const ortLinearClassifierSession = await ort.InferenceSession.create(
  serializedLinearClassifierModel([1, 2, 3, 1], [2, 2], [1, 0, 0, 1], [0, 0], [10n, 20n]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortLinearClassifierOutput = await ortLinearClassifierSession.run({});
expectEqual(JSON.stringify([...ortLinearClassifierOutput.probabilities.data]), "[1,2,3,1]", "Pinned native ORT LinearClassifier scores must match the deterministic scalar fixture.");
expectEqual(JSON.stringify([...ortLinearClassifierOutput.labels.data].map(String)), '["20","10"]', "Pinned native ORT LinearClassifier labels must match the deterministic scalar fixture.");

const serializedBinaryClassifier = analyzeOnnxModel(
  serializedLinearClassifierModel([3, 1], [1, 2], [1, -1], [0], [0n, 1n]),
  "serialized_binary_linear_classifier.onnx",
);
const ortBinaryClassifierSession = await ort.InferenceSession.create(
  serializedLinearClassifierModel([3, 1], [1, 2], [1, -1], [0], [0n, 1n]),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortBinaryClassifierOutput = await ortBinaryClassifierSession.run({});
expectEqual(JSON.stringify([...ortBinaryClassifierOutput.probabilities.data]), "[-1,2]", "Pinned native ORT must confirm binary complement/raw-score expansion.");
expectEqual(bundle(serializedBinaryClassifier).evidence.evidence?.conformance_report?.status, "pass", "Binary LinearClassifier expansion evidence must pass independent reconstruction.");

const serializedLinearRegressor = analyzeOnnxModel(
  serializedLinearRegressorModel([1, 2, 3, 1], [2, 2], [1, 0, 0, 1], [0.5, -0.5], { targets: 2 }),
  "serialized_linear_regressor.onnx",
);
const serializedLinearRegressorBundle = bundle(serializedLinearRegressor);
expectEqual(serializedLinearRegressorBundle.evidence.evidence?.conformance_report?.status, "pass", "Serialized LinearRegressor report and ML-BOM must pass independent reconstruction.");
expect(serializedLinearRegressorBundle.report.includes("615259243fec59bd088299bb4778f6f484b84af6f1ea8985497e73bc58ee11e2"), "Engineering Report must preserve the pinned ORT LinearRegressor kernel hash.");
const ortLinearRegressorSession = await ort.InferenceSession.create(
  serializedLinearRegressorModel([1, 2, 3, 1], [2, 2], [1, 0, 0, 1], [0.5, -0.5], { targets: 2 }),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortLinearRegressorOutput = await ortLinearRegressorSession.run({});
expectEqual(JSON.stringify([...ortLinearRegressorOutput.probabilities.data]), "[1.5,1.5,3.5,0.5]", "Pinned native ORT LinearRegressor output must match the deterministic scalar fixture.");

const serializedLinearExtra = analyzeOnnxModel(
  serializedLinearClassifierModel([1, 2], [1, 2], [1, 0, 0, 1, 99], [0, 0], [10n, 20n]),
  "serialized_linear_extra_coefficients.onnx",
);
expectEqual(buildFindingsRegister(serializedLinearExtra).find((item) => item.finding_id === "EA-ONX-0039")?.technical_priority, "Medium", "Ignored trailing linear coefficients must enter the Medium action queue.");
expectEqual(bundle(serializedLinearExtra).evidence.evidence?.conformance_report?.status, "pass", "Ignored linear parameters must pass faithful export conformance.");

const serializedLinearIntentRisks = analyzeOnnxModel(
  serializedLinearClassifierModel([0, 0], [1, 2], [1, 0, 0, 1], [0, 0], [10n, 10n], { multiClass: 1 }),
  "serialized_linear_intent_risks.onnx",
);
const linearIntentFindings = buildFindingsRegister(serializedLinearIntentRisks);
expectEqual(linearIntentFindings.find((item) => item.finding_id === "EA-ONX-0043")?.technical_priority, "Medium", "Ignored nonzero multi_class must enter the Medium action queue.");
expectEqual(linearIntentFindings.find((item) => item.finding_id === "EA-ONX-0044")?.technical_priority, "Medium", "Duplicate classifier labels must enter the Medium action queue.");
expectEqual(linearIntentFindings.find((item) => item.finding_id === "EA-ONX-0045")?.technical_priority, "High", "Exact zero-margin or tied classifier reference scores must enter the High numerical queue.");
expectEqual(bundle(serializedLinearIntentRisks).evidence.evidence?.conformance_report?.status, "pass", "Classifier intent and numerical risks must pass faithful export conformance.");

const serializedLinearProbit = analyzeOnnxModel(
  serializedLinearClassifierModel([3, 1], [1, 2], [1, -1], [0], [0n, 1n], { postTransform: "PROBIT" }),
  "serialized_binary_linear_probit.onnx",
);
expectEqual(buildFindingsRegister(serializedLinearProbit).find((item) => item.finding_id === "EA-ONX-0042")?.technical_priority, "High", "Pinned binary PROBIT second-score write gap must enter the High action queue.");
expectEqual(bundle(serializedLinearProbit).evidence.evidence?.conformance_report?.status, "pass", "Pinned-source binary PROBIT hazard must pass faithful export conformance.");

const serializedRegressorDefault = analyzeOnnxModel(
  serializedLinearRegressorModel([1, 2], [1, 2], [1, 0], [0], { includeTargets: false }),
  "serialized_linear_regressor_default_targets.onnx",
);
const regressorDefaultFindings = buildFindingsRegister(serializedRegressorDefault);
expectEqual(regressorDefaultFindings.find((item) => item.finding_id === "EA-ONX-0038"), undefined, "Resolved targets=1 schema defaults must not emit a false pinned-runtime failure.");
expectEqual(regressorDefaultFindings.find((item) => item.finding_id === "EA-ONX-0041"), undefined, "The retired false-positive targets-default finding must remain absent.");
expectEqual(serializedRegressorDefault.onnx_shape_inference.ml_value_inference.rows[0].linear_targets_source, "onnx_schema_default_1_materialized_by_ort_schema_resolution", "Structured evidence must identify the runtime-resolved schema default.");
expectEqual(bundle(serializedRegressorDefault).evidence.evidence?.conformance_report?.status, "pass", "Resolved targets-default evidence must pass faithful export conformance.");
const ortLinearRegressorDefaultSession = await ort.InferenceSession.create(
  serializedLinearRegressorModel([1, 2], [1, 2], [1, 0], [0], { includeTargets: false }),
  { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
);
const ortLinearRegressorDefaultOutput = await ortLinearRegressorDefaultSession.run({});
expectEqual(JSON.stringify([...ortLinearRegressorDefaultOutput.probabilities.data]), "[1]", "Pinned native ORT must confirm targets=1 schema-default materialization.");

const serializedRegressorFloat64 = analyzeOnnxModel(
  serializedLinearRegressorModel([1, 2], [1, 2], [1, 0], [0], { targets: 1, dtype: 11 }),
  "serialized_linear_regressor_float64.onnx",
);
expectEqual(buildFindingsRegister(serializedRegressorFloat64).find((item) => item.finding_id === "EA-ONX-0040")?.technical_priority, "High", "Schema-valid FLOAT64 LinearRegressor CPU gap must enter the High action queue.");
expectEqual(buildFindingsRegister(serializedRegressorFloat64).find((item) => item.finding_id === "EA-ONX-0038")?.technical_priority, "High", "Pinned ORT linear runtime failure must enter the High action queue.");
expectEqual(bundle(serializedRegressorFloat64).evidence.evidence?.conformance_report?.status, "pass", "FLOAT64 LinearRegressor schema/runtime gap must pass faithful export conformance.");
let ortLinearRegressorFloat64Rejected = false;
try {
  const session = await ort.InferenceSession.create(
    serializedLinearRegressorModel([1, 2], [1, 2], [1, 0], [0], { targets: 1, dtype: 11 }),
    { executionProviders: ["cpu"], graphOptimizationLevel: "disabled" },
  );
  await session.run({});
} catch {
  ortLinearRegressorFloat64Rejected = true;
}
expect(ortLinearRegressorFloat64Rejected, "Pinned native ORT CPU must reject schema-valid FLOAT64 LinearRegressor execution.");

const serializedRegressorIgnoredIntercept = analyzeOnnxModel(
  serializedLinearRegressorModel([1, 2], [1, 2], [1, 0, 0, 1], [9], { targets: 2 }),
  "serialized_linear_regressor_ignored_intercept.onnx",
);
expectEqual(serializedRegressorIgnoredIntercept.onnx_shape_inference.ml_value_inference.rows[0].linear_ignored_intercept_count, 1, "Mismatched LinearRegressor intercept vectors must be counted as ignored.");
expectEqual(buildFindingsRegister(serializedRegressorIgnoredIntercept).find((item) => item.finding_id === "EA-ONX-0039")?.technical_priority, "Medium", "Ignored LinearRegressor intercepts must enter the Medium action queue.");
expectEqual(bundle(serializedRegressorIgnoredIntercept).evidence.evidence?.conformance_report?.status, "pass", "Ignored-intercept evidence must pass faithful export conformance.");

const serializedBinarizer = analyzeOnnxModel(serializedBinarizerModel([-1, 0.25, 0.5, 2], 0.25), "serialized_binarizer.onnx");
const serializedBinarizerRow = serializedBinarizer.onnx_shape_inference.ml_value_inference.rows[0];
const serializedBinarizerOutput = serializedBinarizer.tensors.find((tensor) => tensor.name === "probabilities");
expectEqual(serializedBinarizer.onnx_shape_inference.status, "assessed", "Serialized Binarizer should preserve exact threshold arithmetic through the real parser.");
expectEqual(serializedBinarizerRow.exact_above_threshold_count, 2, "Serialized Binarizer should count strict threshold crossings exactly.");
expectEqual(JSON.stringify(serializedBinarizerOutput.static_values), "[0,0,1,1]", "Serialized Binarizer output values should remain in public tensor evidence.");
expectEqual(serializedBinarizer.onnx_shape_inference.propagated_static_value_tensor_count, 1, "Exact Binarizer output values should enter static-value propagation conservation.");
const serializedBinarizerBundle = bundle(serializedBinarizer);
expectEqual(serializedBinarizerBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized Binarizer report bundle should pass independent conformance.");
expect(serializedBinarizerBundle.report.includes("Binarizer exact threshold effects") && serializedBinarizerBundle.report.includes("threshold `0.25` (explicit_attribute)"), "Engineering Report should expose the exact Binarizer threshold and output distribution.");
expect(serializedBinarizerBundle.report.includes("5253671998fd3c5493d2c8acfcca34ba5747b575f05c635ad7533696a53a43bf"), "Engineering Report should preserve the pinned ORT Binarizer kernel hash.");
expectCompactMlBom(serializedBinarizerBundle.mlBom, ["deepbom:model:onnxMlExactBinarizerAboveThreshold"], "Binarizer compact ML-BOM");

const serializedBinarizerDefault = analyzeOnnxModel(serializedBinarizerModel([-1, 0, 1], null), "serialized_binarizer_default.onnx");
expectEqual(serializedBinarizerDefault.onnx_shape_inference.ml_value_inference.rows[0].threshold_source, "onnx_schema_default_0", "Serialized attribute omission must retain the ONNX schema default rather than the kernel constructor fallback.");
expectEqual(JSON.stringify(serializedBinarizerDefault.tensors.find((tensor) => tensor.name === "probabilities").static_values), "[0,0,1]", "Serialized default-threshold output must match the pinned ONNX schema and observed ORT 1.26 behavior.");
const ortDefaultSession = await ort.InferenceSession.create(serializedBinarizerModel([-1, 0, 1], null), {
  executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
});
const ortDefaultOutput = await ortDefaultSession.run({});
expectEqual(JSON.stringify([...ortDefaultOutput.probabilities.data]), "[0,0,1]", "Pinned ORT must inject the ONNX schema default threshold before the CPU kernel constructor fallback can apply.");

const serializedBinarizerInfinite = analyzeOnnxModel(serializedBinarizerModel([-1, 0, 1], Number.POSITIVE_INFINITY), "serialized_binarizer_infinite.onnx");
const binarizerInfiniteFinding = buildFindingsRegister(serializedBinarizerInfinite).find((item) => item.finding_id === "EA-ONX-0018");
expectEqual(serializedBinarizerInfinite.onnx_shape_inference.ml_value_inference.binarizer_nonfinite_threshold_node_count, 1, "A non-finite threshold must be counted without JSON null ambiguity.");
expectEqual(binarizerInfiniteFinding?.technical_priority, "High", "A non-finite Binarizer numerical contract should enter the High action queue.");
expectEqual(bundle(serializedBinarizerInfinite).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported non-finite Binarizer risk should pass export conformance.");

const serializedBinarizerFloat64 = analyzeOnnxModel(serializedBinarizerModel([-1, 0, 1], 0, 11), "serialized_binarizer_float64.onnx");
const binarizerFloat64Finding = buildFindingsRegister(serializedBinarizerFloat64).find((item) => item.finding_id === "EA-ONX-0019");
expectEqual(serializedBinarizerFloat64.onnx_shape_inference.ml_value_inference.rows[0].status, "pass", "FLOAT64 Binarizer should remain valid under the pinned ONNX schema.");
expectEqual(binarizerFloat64Finding?.technical_priority, "High", "A schema-valid dtype without a pinned ORT CPU kernel must enter the High action queue.");
const serializedBinarizerFloat64Bundle = bundle(serializedBinarizerFloat64);
expectEqual(serializedBinarizerFloat64Bundle.evidence.evidence?.conformance_report?.status, "pass", "A faithfully separated ONNX schema pass and ORT CPU dtype gap should pass export conformance.");
expect(serializedBinarizerFloat64Bundle.report.includes("EA-ONX-0019") && serializedBinarizerFloat64Bundle.report.includes("no pinned ORT CPU kernel"), "Engineering Report must front the schema/runtime dtype gap as an actionable finding.");
let ortFloat64Rejected = false;
try {
  await ort.InferenceSession.create(serializedBinarizerModel([-1, 0, 1], 0, 11), {
    executionProviders: ["cpu"], graphOptimizationLevel: "disabled",
  });
} catch {
  ortFloat64Rejected = true;
}
expect(ortFloat64Rejected, "Pinned ORT CPU must reject a FLOAT64 Binarizer for which it registers no kernel.");

const featureVectorizer = run(
  mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["2", "5"]) }, ["x0", "x1"]),
  [tensor("x0", "INT32", [2, 3]), tensor("x1", "INT32", [2, 2, 2])],
);
const featureRow = featureVectorizer.evidence.ml_value_inference.rows[0];
expectEqual(featureVectorizer.evidence.status, "assessed", "Pinned ORT rank-3 flattening with exact widths should be assessed.");
expectEqual(JSON.stringify(featureVectorizer.tensorMap.get("probabilities").shape), "[2,7]", "FeatureVectorizer output should be [batch,sum(inputdimensions)].");
expectEqual(JSON.stringify(featureRow.exact_input_row_feature_counts), "[3,4]", "Every input row width should be derived from post-batch dimensions.");
expectEqual(JSON.stringify(featureRow.copied_feature_counts_per_input), "[2,4]", "Copied widths should equal min(actual,configured).");
expectEqual(JSON.stringify(featureRow.padded_feature_counts_per_input), "[0,1]", "Configured width beyond actual width should be exact zero padding.");
expectEqual(JSON.stringify(featureRow.truncated_feature_counts_per_input), "[1,0]", "Actual width beyond configured width should be exact truncation.");
expectEqual(featureRow.exact_padded_feature_count_per_batch, 1, "FeatureVectorizer padding total should be exact per batch row.");
expectEqual(featureRow.exact_truncated_feature_count_per_batch, 1, "FeatureVectorizer truncation total should be exact per batch row.");
expect(featureRow.risk_codes.includes("feature_vectorizer_truncates_input_features"), "Exact FeatureVectorizer information loss should carry a source-backed risk code.");

const featureDynamic = run(
  mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["3", "2"]) }, ["x0", "x1"]),
  [tensor("x0", "FLOAT32", [-1, 3]), tensor("x1", "FLOAT32", [-1, 2])],
);
expectEqual(featureDynamic.evidence.ml_value_inference.status, "partial", "Dynamic batch consistency should remain explicit while output width stays exact.");
expectEqual(JSON.stringify(featureDynamic.tensorMap.get("probabilities").shape), "[-1,5]", "Dynamic FeatureVectorizer batch should preserve exact output width.");
expectEqual(run(mlVariadicOp("FeatureVectorizer", {}, ["x0"]), [tensor("x0", "INT32", [1, 2])]).evidence.status, "fail", "Pinned ORT requires a nonempty inputdimensions attribute.");
expectEqual(run(mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["2"]) }, ["x0", "x1"]), [tensor("x0", "INT32", [1, 2]), tensor("x1", "INT32", [1, 2])]).evidence.status, "fail", "FeatureVectorizer input and configured-width counts must match.");
expectEqual(run(mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["2", "2"]) }, ["x0", "x1"]), [tensor("x0", "INT32", [1, 2]), tensor("x1", "FLOAT32", [1, 2])]).evidence.status, "fail", "FeatureVectorizer variadic inputs must share the schema type variable.");
expectEqual(run(mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["2", "2"]) }, ["x0", "x1"]), [tensor("x0", "INT32", [1, 2]), tensor("x1", "INT32", [2, 2])]).evidence.status, "fail", "FeatureVectorizer exact batch mismatch must fail.");
expectEqual(run(mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["-1"]) }, ["x0"]), [tensor("x0", "INT32", [1, 2])]).evidence.status, "fail", "Negative FeatureVectorizer dimensions must fail before allocation.");

const arrayFeature = run(
  mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]),
  [tensor("x", "INT32", [2, 3, 4]), exactIntegerTensor("indices", [2, 2], ["0", "3", "3", "1"])],
);
const arrayFeatureRow = arrayFeature.evidence.ml_value_inference.rows[0];
expectEqual(arrayFeature.evidence.status, "assessed", "Exact in-range ArrayFeatureExtractor indices should close the runtime contract.");
expectEqual(JSON.stringify(arrayFeature.tensorMap.get("probabilities").shape), "[2,3,4]", "ArrayFeatureExtractor should replace the final axis with flattened index cardinality.");
expectEqual(arrayFeatureRow.exact_index_count, 4, "Index tensor cardinality should be the exact product of its dimensions.");
expectEqual(arrayFeatureRow.duplicate_index_count, 1, "Repeated extraction indices should be inventoried exactly without being mislabeled invalid.");
expectEqual(arrayFeatureRow.index_bounds_status, "assessed_pass", "Every exact index should be checked against the final data axis.");
const arrayRankOne = run(
  mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]),
  [tensor("x", "FLOAT32", [5]), exactIntegerTensor("indices", [3], ["4", "0", "2"])],
);
expectEqual(JSON.stringify(arrayRankOne.tensorMap.get("probabilities").shape), "[1,3]", "Pinned ORT rank-1 compatibility output must be [1,num_indices].");
const arrayRuntimeIndices = run(
  mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]),
  [tensor("x", "FLOAT32", [2, 4]), tensor("indices", "INT64", [3])],
);
expectEqual(arrayRuntimeIndices.evidence.ml_value_inference.status, "partial", "Runtime index values must leave bounds explicitly unresolved.");
expectEqual(JSON.stringify(arrayRuntimeIndices.tensorMap.get("probabilities").shape), "[2,3]", "Runtime index values do not prevent shape derivation from index tensor cardinality.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", [2, 4]), exactIntegerTensor("indices", [1], ["4"])]).evidence.status, "fail", "An index equal to final-axis size must fail.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", [2, 4]), exactIntegerTensor("indices", [1], ["-1"])]).evidence.status, "fail", "Negative extraction indices must fail.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", [2, 4]), exactIntegerTensor("indices", [1], ["9007199254740993"])]).evidence.status, "fail", "INT64 indices beyond JavaScript safe integers must still fail exact bounds.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", [2, 4]), exactIntegerTensor("indices", [0], [])]).evidence.status, "fail", "Empty extraction indices must fail under pinned ORT behavior.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", []), exactIntegerTensor("indices", [1], ["0"])]).evidence.status, "fail", "Scalar ArrayFeatureExtractor data must fail.");
expectEqual(run(mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"]), [tensor("x", "INT32", [2, 4]), tensor("indices", "INT32", [1])]).evidence.status, "fail", "ArrayFeatureExtractor index dtype must be INT64.");

const customFeature = run(
  mlVariadicOp("FeatureVectorizer", { inputdimensions: intsAttribute(["2"]) }, ["x"], "com.example"),
  [tensor("x", "FLOAT32", [1, 2])],
  [{ domain: "com.example", version: 1 }],
);
expectEqual(customFeature.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain FeatureVectorizer name must not inherit ai.onnx.ml semantics.");
const customArray = run(
  mlVariadicOp("ArrayFeatureExtractor", {}, ["x", "indices"], "com.example"),
  [tensor("x", "FLOAT32", [1, 2]), exactIntegerTensor("indices", [1], ["0"])],
  [{ domain: "com.example", version: 1 }],
);
expectEqual(customArray.evidence.ml_value_inference.assessed_node_count, 0, "A custom-domain ArrayFeatureExtractor name must not inherit ai.onnx.ml semantics.");

const serializedFeature = analyzeOnnxModel(serializedFeatureVectorizerModel(), "serialized_feature_vectorizer.onnx");
const serializedFeatureRow = serializedFeature.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedFeature.onnx_shape_inference.status, "assessed", "Serialized FeatureVectorizer should preserve exact variadic width arithmetic.");
expectEqual(JSON.stringify(serializedFeatureRow.exact_output_shape), "[2,5]", "Serialized FeatureVectorizer output shape should be exact.");
expectEqual(serializedFeatureRow.exact_truncated_feature_count_per_batch, 1, "Serialized FeatureVectorizer should retain exact truncation.");
expectEqual(serializedFeatureRow.exact_padded_feature_count_per_batch, 1, "Serialized FeatureVectorizer should retain exact padding.");
const serializedFeatureFinding = buildFindingsRegister(serializedFeature).find((item) => item.finding_id === "EA-ONX-0017");
expectEqual(serializedFeatureFinding?.technical_priority, "Medium", "Exact FeatureVectorizer truncation should enter the Medium action queue.");
const serializedFeatureBundle = bundle(serializedFeature);
expectEqual(serializedFeatureBundle.evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported FeatureVectorizer truncation risk should pass export conformance.");
expect(serializedFeatureBundle.report.includes("1 total / 1 input(s)") && serializedFeatureBundle.report.includes("pinned_ort_cpu_feature_dimension_allocation"), "Engineering Report should expose exact FeatureVectorizer pad/truncate arithmetic and basis.");
expectCompactMlBom(serializedFeatureBundle.mlBom, ["deepbom:model:onnxMlExactTruncatedFeaturesPerBatch"], "FeatureVectorizer compact ML-BOM");

const serializedArray = analyzeOnnxModel(serializedArrayFeatureExtractorModel([0n, 3n, 3n]), "serialized_array_feature_extractor.onnx");
const serializedArrayRow = serializedArray.onnx_shape_inference.ml_value_inference.rows[0];
const serializedArrayIndexTensor = serializedArray.tensors.find((tensor) => tensor.name === "indices");
expectEqual(serializedArray.onnx_shape_inference.status, "assessed", "Serialized ArrayFeatureExtractor should consume exact initializer indices.");
expectEqual(JSON.stringify(serializedArrayRow.exact_output_shape), "[2,3]", "Serialized ArrayFeatureExtractor shape should replace the final axis exactly.");
expectEqual(JSON.stringify(serializedArrayRow.exact_index_values), '["0","3","3"]', "Exact initializer indices should survive protobuf parsing as decimal strings.");
expectEqual(JSON.stringify(serializedArrayIndexTensor.initializer_integer_values_exact_decimals), '["0","3","3"]', "Public tensor evidence must preserve the exact initializer integers used by ArrayFeatureExtractor inference.");
expectEqual(serializedArrayRow.index_bounds_status, "assessed_pass", "Serialized exact indices should close bounds.");
const serializedArrayBundle = bundle(serializedArray);
expectEqual(serializedArrayBundle.evidence.evidence?.conformance_report?.status, "pass", "A valid serialized ArrayFeatureExtractor report should pass export conformance.");
expect(serializedArrayBundle.report.includes("bounds assessed_pass") && serializedArrayBundle.report.includes("`0` / `3` / `3`"), "Engineering Report should expose exact ArrayFeatureExtractor index evidence.");
expectCompactMlBom(serializedArrayBundle.mlBom, ["deepbom:model:onnxMlExactArrayFeatureIndices"], "ArrayFeatureExtractor compact ML-BOM");

const serializedArrayUnsafe = analyzeOnnxModel(serializedArrayFeatureExtractorModel([9007199254740993n]), "serialized_array_feature_extractor_unsafe.onnx");
const serializedArrayUnsafeRow = serializedArrayUnsafe.onnx_shape_inference.ml_value_inference.rows[0];
expectEqual(serializedArrayUnsafeRow.exact_index_values[0], "9007199254740993", "Serialized unsafe INT64 indices must never round through Number.");
expectEqual(serializedArrayUnsafe.onnx_shape_inference.status, "fail", "An exact unsafe INT64 index outside the final axis must fail.");
expectEqual(buildFindingsRegister(serializedArrayUnsafe).find((item) => item.finding_id === "EA-ONX-0013")?.technical_priority, "High", "Exact ArrayFeatureExtractor bounds failure should enter the High action queue.");
expectEqual(bundle(serializedArrayUnsafe).evidence.evidence?.conformance_report?.status, "pass", "A faithfully reported unsafe INT64 bounds failure should pass export conformance.");

const tamperedCount = patchMlEvidence(serialized, (evidence) => ({ ...evidence, exact_class_key_count: evidence.exact_class_key_count + 1 }));
expectThrows(() => bundle(tamperedCount), "CF-SHAPE-001", "Independent conformance arithmetic should reject a tampered ZipMap key total.");
const tamperedSource = patchMlEvidence(serialized, (evidence) => ({
  ...evidence,
  source_documents: evidence.source_documents.map((source, index) => index ? source : { ...source, sha256: "0".repeat(64) }),
}));
expectThrows(() => bundle(tamperedSource), "CF-SHAPE-ML-SOURCE-001", "Conformance should identify and reject a tampered ONNX-ML schema source digest through its dedicated provenance invariant.");
const tamperedRisk = patchMlEvidence(serializedDuplicate, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => ({ ...row, risk_codes: [] })),
}));
expectThrows(() => bundle(tamperedRisk), "CF-SHAPE-ML-ROW-002", "Conformance should reject a duplicate-key row with its deterministic risk code removed.");
const tamperedDenseShape = patchMlEvidence(serializedCast, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => ({ ...row, exact_output_shape: [6], exact_dense_output_element_count: 6 })),
}));
expectThrows(() => bundle(tamperedDenseShape), "CF-SHAPE-ML-ROW-002", "Conformance should reject a self-consistent CastMap shape/element tamper that disagrees with max_map.");
const tamperedVocabulary = patchMlEvidence(serializedDict, (evidence) => ({
  ...evidence,
  exact_vocabulary_entry_count: evidence.exact_vocabulary_entry_count + 1,
}));
expectThrows(() => bundle(tamperedVocabulary), "CF-SHAPE-001", "Conformance should reject a tampered DictVectorizer vocabulary total.");
const tamperedVocabularyRisk = patchMlEvidence(serializedDictDuplicate, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => ({ ...row, risk_codes: [] })),
}));
expectThrows(() => bundle(tamperedVocabularyRisk), "CF-SHAPE-ML-ROW-002", "Conformance should reject duplicate DictVectorizer columns with their deterministic risk code removed.");
const tamperedArrayExactValues = patchMlEvidence(serializedArray, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => ({
    ...row,
    exact_index_values: ["0", "2", "2"],
    exact_index_preview: ["0", "2", "2"],
    duplicate_index_count: 1,
  })),
}));
expectThrows(() => bundle(tamperedArrayExactValues), "CF-SHAPE-ML-ROW-002", "Conformance must reject self-consistent ArrayFeatureExtractor index evidence that disagrees with the parsed initializer.");
const tamperedCategoryTotal = patchMlEvidence(serializedCategory, (evidence) => ({
  ...evidence, exact_category_pair_count: evidence.exact_category_pair_count + 1,
}));
expectThrows(() => bundle(tamperedCategoryTotal), "CF-SHAPE-001", "Conformance should reject a tampered CategoryMapper pair total.");
const tamperedCategoryDirection = patchMlEvidence(serializedCategory, (evidence) => ({
  ...evidence, rows: evidence.rows.map((row) => ({ ...row, mapping_direction: "INT64_TO_STRING" })),
}));
expectThrows(() => bundle(tamperedCategoryDirection), "CF-SHAPE-ML-ROW-002", "Conformance should reject CategoryMapper direction that contradicts input dtype.");
const tamperedCategoryRisk = patchMlEvidence(serializedCategoryDuplicate, (evidence) => ({
  ...evidence, rows: evidence.rows.map((row) => ({ ...row, risk_codes: [] })),
}));
expectThrows(() => bundle(tamperedCategoryRisk), "CF-SHAPE-ML-ROW-002", "Conformance should reject active duplicate categories with their source-backed risk removed.");
const tamperedFeatureTruncation = patchMlEvidence(serializedFeature, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "FeatureVectorizer" ? { ...row, exact_truncated_feature_count_per_batch: 0, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedFeatureTruncation), "CF-SHAPE-ML-ROW-002", "Conformance should reject tampered FeatureVectorizer truncation arithmetic.");
const tamperedArrayBounds = patchMlEvidence(serializedArray, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "ArrayFeatureExtractor" ? { ...row, index_bounds_status: "not_assessed_runtime_values" } : row),
}));
expectThrows(() => bundle(tamperedArrayBounds), "CF-SHAPE-ML-ROW-002", "Conformance should reject an exact ArrayFeatureExtractor row relabeled runtime-unknown.");

const tamperedBinarizerArithmetic = patchMlEvidence(serializedBinarizer, (evidence) => ({
  ...evidence,
  exact_binarizer_above_threshold_count: evidence.exact_binarizer_above_threshold_count + 1,
  rows: evidence.rows.map((row) => row.op_name === "Binarizer" ? { ...row, exact_above_threshold_count: row.exact_above_threshold_count + 1, exact_output_one_count: row.exact_output_one_count + 1 } : row),
}));
expectThrows(() => bundle(tamperedBinarizerArithmetic), "CF-SHAPE-ML-ROW-002", "Conformance should reject self-consistent Binarizer aggregate and row arithmetic that disagrees with parsed initializer values.");
const tamperedBinarizerOutput = {
  ...serializedBinarizer,
  tensors: serializedBinarizer.tensors.map((tensor) => tensor.name === "probabilities" ? { ...tensor, static_values: [1, 1, 1, 1] } : tensor),
};
expectThrows(() => bundle(tamperedBinarizerOutput), "CF-SHAPE-ML-ROW-002", "Conformance should reject propagated Binarizer outputs that disagree with the source threshold comparison.");
const tamperedBinarizerDtypeRisk = patchMlEvidence(serializedBinarizerFloat64, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "Binarizer" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedBinarizerDtypeRisk), "CF-SHAPE-ML-ROW-002", "Conformance should reject removal of the pinned ORT CPU dtype gap from a schema-valid Binarizer row.");

const tamperedNormalizerArithmetic = patchMlEvidence(serializedNormalizer, (evidence) => ({
  ...evidence,
  exact_normalizer_negative_max_divisor_row_count: 0,
  rows: evidence.rows.map((row) => row.op_name === "Normalizer" ? {
    ...row, normalizer_negative_max_divisor_row_count: 0, risk_codes: [],
  } : row),
}));
expectThrows(() => bundle(tamperedNormalizerArithmetic), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject self-consistent Normalizer aggregate/row arithmetic that disagrees with initializer values.");
const tamperedNormalizerOutput = {
  ...serializedNormalizer,
  tensors: serializedNormalizer.tensors.map((tensor) => tensor.name === "probabilities" ? { ...tensor, static_values: [1, 1, 0, 0] } : tensor),
};
expectThrows(() => bundle(tamperedNormalizerOutput), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject propagated Normalizer output values that disagree with pinned source-order arithmetic.");
const tamperedNormalizerSignedZeroLedger = {
  ...serializedNormalizerSignedZero,
  tensors: serializedNormalizerSignedZero.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values_negative_zero_count: 0, static_values_negative_zero_indices: [],
  } : tensor),
};
expectThrows(() => bundle(tamperedNormalizerSignedZeroLedger), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of a signed-zero position even when canonical JSON values remain unchanged.");
const malformedNormalizerSignedZeroLedger = {
  ...serializedNormalizerSignedZero,
  tensors: serializedNormalizerSignedZero.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values_negative_zero_indices: [2],
  } : tensor),
};
expectThrows(() => bundle(malformedNormalizerSignedZeroLedger), "CF-STATIC-001", "Conformance must reject an out-of-bounds signed-zero index ledger.");
const tamperedNormalizerOverflowRisk = patchMlEvidence(serializedNormalizerOverflow, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "Normalizer" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedNormalizerOverflowRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of a proven Normalizer signed-overflow risk.");
const tamperedNormalizerNonfinite = patchMlEvidence(serializedNormalizerNonfinite, (evidence) => ({
  ...evidence,
  exact_normalizer_non_finite_output_count: 0,
  rows: evidence.rows.map((row) => row.op_name === "Normalizer" ? {
    ...row, normalizer_non_finite_output_count: 0, normalizer_output_preview: ["0"], risk_codes: [],
  } : row),
}));
expectThrows(() => bundle(tamperedNormalizerNonfinite), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject a self-consistent finite relabeling of a proven NaN Normalizer output.");

const tamperedScalerArithmetic = patchMlEvidence(serializedScaler, (evidence) => ({
  ...evidence,
  exact_scaler_input_value_count: evidence.exact_scaler_input_value_count + 1,
  rows: evidence.rows.map((row) => row.op_name === "Scaler" ? {
    ...row, scaler_exact_input_value_count: row.scaler_exact_input_value_count + 1,
  } : row),
}));
expectThrows(() => bundle(tamperedScalerArithmetic), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject self-consistent Scaler aggregate/row arithmetic that disagrees with initializer cardinality.");
const tamperedScalerOutput = {
  ...serializedScaler,
  tensors: serializedScaler.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values: [0, 0, 0, 0, 0, 0],
  } : tensor),
};
expectThrows(() => bundle(tamperedScalerOutput), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject propagated Scaler values that disagree with source-order affine arithmetic.");
const tamperedScalerAttribute = {
  ...serializedScaler,
  ops: serializedScaler.ops.map((op) => op.name === "Scaler" ? {
    ...op,
    onnx_attributes: op.onnx_attributes.map((attribute) => attribute.name === "scale" ? {
      ...attribute, float_values: [2, -3, 1], float_values_text: ["2", "-3", "1"],
    } : attribute),
  } : op),
};
expectThrows(() => bundle(tamperedScalerAttribute), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject Scaler attribute evidence changed independently of the inferred row and output.");
const tamperedScalerInvalidRisk = patchMlEvidence(serializedScalerInvalid, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "Scaler" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedScalerInvalidRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of a deterministic Scaler pinned-runtime contract risk.");
const tamperedScalerSignedZeroLedger = {
  ...serializedScalerSignedZero,
  tensors: serializedScalerSignedZero.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values_negative_zero_count: 0, static_values_negative_zero_indices: [],
  } : tensor),
};
expectThrows(() => bundle(tamperedScalerSignedZeroLedger), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of a Scaler signed-zero output position.");
const tamperedScalerNonfiniteRisk = patchMlEvidence(serializedScalerNonfinite, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "Scaler" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedScalerNonfiniteRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of a proven Scaler non-finite risk.");

const tamperedImputerArithmetic = patchMlEvidence(serializedImputer, (evidence) => ({
  ...evidence,
  exact_imputer_replacement_count: evidence.exact_imputer_replacement_count + 1,
  rows: evidence.rows.map((row) => row.op_name === "Imputer" ? {
    ...row, imputer_exact_replacement_count: row.imputer_exact_replacement_count + 1,
  } : row),
}));
expectThrows(() => bundle(tamperedImputerArithmetic), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject self-consistent Imputer replacement totals that disagree with initializer values.");
const tamperedImputerOutput = {
  ...serializedImputer,
  tensors: serializedImputer.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values: [0, 0, 0, 0, 0, 0],
  } : tensor),
};
expectThrows(() => bundle(tamperedImputerOutput), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject propagated Imputer values that disagree with source-pinned replacement arithmetic.");
const tamperedImputerAttribute = {
  ...serializedImputer,
  ops: serializedImputer.ops.map((op) => op.name === "Imputer" ? {
    ...op,
    onnx_attributes: op.onnx_attributes.map((attribute) => attribute.name === "imputed_value_floats" ? {
      ...attribute, float_values: [99, 20, 30], float_values_text: ["99", "20", "30"],
    } : attribute),
  } : op),
};
expectThrows(() => bundle(tamperedImputerAttribute), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject Imputer attributes changed independently of the inferred row and output.");
const tamperedImputerFallbackRisk = patchMlEvidence(serializedImputerFallback, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "Imputer" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedImputerFallbackRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of the pinned Imputer scalar-first fallback risk.");

const tamperedOneHotArithmetic = patchMlEvidence(serializedOneHotInt64, (evidence) => ({
  ...evidence,
  exact_onehot_matched_input_count: evidence.exact_onehot_matched_input_count + 1,
  rows: evidence.rows.map((row) => row.op_name === "OneHotEncoder" ? {
    ...row, onehot_exact_matched_input_count: row.onehot_exact_matched_input_count + 1,
  } : row),
}));
expectThrows(() => bundle(tamperedOneHotArithmetic), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject self-consistent OneHotEncoder counts that disagree with initializer values.");
const tamperedOneHotOutput = {
  ...serializedOneHotInt64,
  tensors: serializedOneHotInt64.tensors.map((tensor) => tensor.name === "probabilities" ? {
    ...tensor, static_values: [0, 1, 0, 0, 0, 1, 0, 0, 0],
  } : tensor),
};
expectThrows(() => bundle(tamperedOneHotOutput), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject propagated OneHotEncoder values that disagree with source-pinned lookup arithmetic.");
const tamperedOneHotAttribute = {
  ...serializedOneHotInt64,
  ops: serializedOneHotInt64.ops.map((op) => op.name === "OneHotEncoder" ? {
    ...op,
    onnx_attributes: op.onnx_attributes.map((attribute) => attribute.name === "cats_int64s" ? {
      ...attribute, int_values: [9, 2, 3], int_values_exact_decimal: ["9", "2", "3"],
    } : attribute),
  } : op),
};
expectThrows(() => bundle(tamperedOneHotAttribute), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject OneHotEncoder vocabulary evidence changed independently of the inferred row and output.");
const tamperedOneHotDuplicateRisk = patchMlEvidence(serializedOneHotDuplicate, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "OneHotEncoder" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedOneHotDuplicateRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of the pinned OneHotEncoder duplicate-vocabulary risk.");
const tamperedOneHotFailureRisk = patchMlEvidence(serializedOneHotFailure, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "OneHotEncoder" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedOneHotFailureRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of the exact unknown-category runtime-failure risk.");

const tamperedLinearCoefficientUse = patchMlEvidence(serializedLinearClassifier, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "LinearClassifier" ? { ...row, linear_used_coefficient_count: 3 } : row),
  exact_linear_used_coefficient_count: 3,
  exact_linear_unresolved_coefficient_use_count: 1,
}));
expectThrows(() => bundle(tamperedLinearCoefficientUse), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject tampered linear coefficient-use accounting even when aggregate conservation is forged.");

const tamperedLinearRuntimeReason = patchMlEvidence(serializedRegressorDefault, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "LinearRegressor" ? {
    ...row, linear_pinned_ort_contract_reason: "linear_regressor_pinned_ort_requires_explicit_targets",
  } : row),
}));
expectThrows(() => bundle(tamperedLinearRuntimeReason), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject a forged LinearRegressor pinned-runtime reason.");

const tamperedLinearRisk = patchMlEvidence(serializedLinearIntentRisks, (evidence) => ({
  ...evidence,
  rows: evidence.rows.map((row) => row.op_name === "LinearClassifier" ? { ...row, risk_codes: [] } : row),
}));
expectThrows(() => bundle(tamperedLinearRisk), "CF-SHAPE-ML-ROW-002", "Independent conformance must reject removal of source-backed linear intent and numerical risks.");

done("source-pinned Binarizer, Normalizer, Scaler, Imputer, OneHotEncoder, LabelEncoder, LinearClassifier, LinearRegressor, ZipMap, CastMap, DictVectorizer, CategoryMapper, FeatureVectorizer, and ArrayFeatureExtractor type, shape, exact-value/cardinality, malformed-model, native ORT, report, finding, and tamper contracts");

function run(node, input, opsets = [{ domain: "ai.onnx.ml", version: 1 }]) {
  const output = { name: "probabilities", dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  const inputs = Array.isArray(input) ? input : [input];
  const tensorMap = new Map([...inputs.map((value) => [value.name, value]), [output.name, output]]);
  const evidence = inferOnnxShapes(
    { nodes: [node], inputs, outputs: [output], valueInfo: [], initializers: [], sparseInitializers: [] },
    tensorMap,
    opsets,
    typeName,
  );
  return { evidence, tensorMap };
}

function runLinearClassifier(node, input, opsets = [{ domain: "ai.onnx.ml", version: 1 }]) {
  const labelOutput = { name: "labels", dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  const scoreOutput = { name: "probabilities", dtype: "UNKNOWN", shape: [], shapeDeclared: false };
  const inputs = Array.isArray(input) ? input : [input];
  const tensorMap = new Map([...inputs.map((value) => [value.name, value]), [labelOutput.name, labelOutput], [scoreOutput.name, scoreOutput]]);
  const evidence = inferOnnxShapes(
    { nodes: [node], inputs, outputs: [labelOutput, scoreOutput], valueInfo: [], initializers: [], sparseInitializers: [] },
    tensorMap,
    opsets,
    typeName,
  );
  return { evidence, tensorMap };
}

function tensor(name, dtype, shape) {
  return { name, dtype, shape: [...shape], shapeDeclared: true, valueKind: "tensor", typeProto: makeOnnxTensorType(dtype, shape, true) };
}

function staticNumericTensor(name, dtype, shape, values) {
  return {
    ...tensor(name, dtype, shape),
    role: "initializer",
    staticValuesStatus: "complete",
    staticValuesComplete: true,
    staticValues: [...values],
    staticValuesSource: "initializer",
  };
}

function staticStringTensor(name, shape, values) {
  return {
    ...tensor(name, "STRING", shape),
    role: "initializer",
    staticValuesStatus: "complete",
    staticValuesComplete: true,
    staticValues: [...values],
    staticValuesSource: "initializer_string_data",
  };
}

function exactIntegerTensor(name, shape, values) {
  return {
    ...tensor(name, "INT64", shape),
    role: "initializer",
    initializerIntegerValuesExactComplete: true,
    initializerIntegerValuesExactDecimals: [...values],
  };
}

function mapValue(name, keyType, valueDtype, mapKeyCount = null, mapKeysExact = null) {
  const descriptor = onnxValueDescriptorFromType(makeOnnxMapType(keyType, makeOnnxTensorType(valueDtype, [], true)));
  return {
    name, ...descriptor,
    ...(mapKeyCount == null ? {} : { mapKeyCount }),
    ...(mapKeysExact == null ? {} : { mapKeysExact: [...mapKeysExact] }),
  };
}

function mlNode(attributes, domain = "ai.onnx.ml") {
  return mlOp("ZipMap", attributes, domain);
}

function mlOp(opType, attributes, domain = "ai.onnx.ml") {
  return {
    name: `${opType.toLowerCase()}_fixture`, opType, domain, overload: "",
    inputs: ["scores"], outputs: ["probabilities"],
    attributes: new Map(Object.entries(attributes)), duplicateAttributeNames: [],
  };
}

function linearClassifierOp(attributes, domain = "ai.onnx.ml") {
  return {
    name: "linear_classifier_fixture", opType: "LinearClassifier", domain, overload: "",
    inputs: ["scores"], outputs: ["labels", "probabilities"],
    attributes: new Map(Object.entries(attributes)), duplicateAttributeNames: [],
  };
}

function mlVariadicOp(opType, attributes, inputs, domain = "ai.onnx.ml") {
  return {
    name: `${opType.toLowerCase()}_fixture`, opType, domain, overload: "",
    inputs: [...inputs], outputs: ["probabilities"],
    attributes: new Map(Object.entries(attributes)), duplicateAttributeNames: [],
  };
}

function stringAttribute(value) {
  return { type: 3, s: value, strings: [], ints: [], intExactDecimals: [], valueTypesPresent: [3], duplicateValueTypes: [] };
}

function floatAttribute(value) {
  return { type: 1, f: Number(value), strings: [], ints: [], intExactDecimals: [], valueTypesPresent: [1], duplicateValueTypes: [] };
}

function floatsAttribute(values) {
  return {
    type: 6, floats: values.map((value) => Math.fround(value)), strings: [], ints: [], intExactDecimals: [],
    valueTypesPresent: [6], duplicateValueTypes: [],
  };
}

function intAttribute(value) {
  const parsed = Number(value);
  return {
    type: 2, i: Number.isSafeInteger(parsed) ? parsed : 0, iExactDecimal: String(value),
    strings: [], ints: [], intExactDecimals: [], valueTypesPresent: [2], duplicateValueTypes: [],
  };
}

function stringsAttribute(values) {
  return { type: 8, strings: [...values], ints: [], intExactDecimals: [], valueTypesPresent: [8], duplicateValueTypes: [] };
}

function intsAttribute(values) {
  return {
    type: 7,
    ints: values.map((value) => {
      const parsed = Number(value);
      return Number.isSafeInteger(parsed) ? parsed : 0;
    }),
    intExactDecimals: [...values], strings: [], valueTypesPresent: [7], duplicateValueTypes: [],
  };
}

function typeName(id) {
  return ["UNDEFINED", "FLOAT32", "UINT8", "INT8", "UINT16", "INT16", "INT32", "INT64", "STRING", "BOOL", "FLOAT16", "FLOAT64"][id] || `TYPE_${id}`;
}

function bundle(analysis) {
  const mlBom = buildMlBomDocument(analysis, { hash: "" });
  const files = buildEngineeringBundleArtifactFiles(analysis, {
    reportContext: { identity: { filename: analysis.filename, format: "onnx" }, generatedAt: "2026-07-22T00:00:00.000Z" },
    rawEvidenceContext: { identity: { filename: analysis.filename, format: "onnx" } },
    mlBomDocument: mlBom,
  });
  return {
    report: files.find((file) => file.name === "engineering_report.md")?.data || "",
    evidence: JSON.parse(files.find((file) => file.name === "engineering_evidence.json")?.data || "{}"),
    mlBom,
  };
}

function patchMlEvidence(analysis, patch) {
  return {
    ...analysis,
    onnx_shape_inference: {
      ...analysis.onnx_shape_inference,
      ml_value_inference: patch(analysis.onnx_shape_inference.ml_value_inference),
    },
  };
}

function serializedZipMapModel(labels, inputShape) {
  const inputType = tensorTypeProto(1, inputShape, true);
  const outputType = sequenceTypeProto(mapTypeProto(8, tensorTypeProto(1, [], true)));
  const node = protoMessage([
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "zip_map_fixture"), protoString(4, "ZipMap"),
    protoBytes(5, stringListAttributeProto("classlabels_strings", labels)), protoString(7, "ai.onnx.ml"),
  ]);
  const graph = protoMessage([
    protoBytes(1, node), protoString(2, "deepbom_zip_map_graph"),
    protoBytes(11, valueInfoProto("scores", inputType)), protoBytes(12, valueInfoProto("probabilities", outputType)),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_zip_map_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedCastMapModel() {
  const inputType = mapTypeProto(7, tensorTypeProto(1, [], true));
  const outputType = tensorTypeProto(8, [5], true);
  const node = protoMessage([
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "cast_map_fixture"), protoString(4, "CastMap"),
    protoBytes(5, stringAttributeProto("cast_to", "TO_STRING")),
    protoBytes(5, stringAttributeProto("map_form", "SPARSE")),
    protoBytes(5, intAttributeProto("max_map", 5)),
    protoString(7, "ai.onnx.ml"),
  ]);
  return serializedMlModel("deepbom_cast_map_graph", "deepbom_cast_map_fixture", node, inputType, outputType);
}

function serializedDictVectorizerModel(vocabulary) {
  const inputType = mapTypeProto(8, tensorTypeProto(11, [], true));
  const outputType = tensorTypeProto(11, [1, vocabulary.length], true);
  const node = protoMessage([
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "dict_vectorizer_fixture"), protoString(4, "DictVectorizer"),
    protoBytes(5, stringListAttributeProto("string_vocabulary", vocabulary)), protoString(7, "ai.onnx.ml"),
  ]);
  return serializedMlModel("deepbom_dict_vectorizer_graph", "deepbom_dict_vectorizer_fixture", node, inputType, outputType);
}

function serializedCategoryMapperModel(strings, integers) {
  const inputType = tensorTypeProto(8, [2, 2], true);
  const outputType = tensorTypeProto(7, [2, 2], true);
  const node = protoMessage([
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "category_mapper_fixture"), protoString(4, "CategoryMapper"),
    protoBytes(5, stringListAttributeProto("cats_strings", strings)),
    protoBytes(5, intListAttributeProto("cats_int64s", integers)),
    protoBytes(5, stringAttributeProto("default_string", "unknown")),
    protoBytes(5, intAttributeProto("default_int64", 99)),
    protoString(7, "ai.onnx.ml"),
  ]);
  return serializedMlModel("deepbom_category_mapper_graph", "deepbom_category_mapper_fixture", node, inputType, outputType);
}

function serializedBinarizerModel(values, threshold, dtype = 1) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "binarizer_fixture"), protoString(4, "Binarizer"),
    protoString(7, "ai.onnx.ml"),
  ];
  if (threshold != null) nodeFields.splice(4, 0, protoBytes(5, floatAttributeProto("threshold", threshold)));
  const initializer = tensorProto("scores", dtype, [values.length], dtype === 11 ? float64Bytes(values) : float32Bytes(values));
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_binarizer_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(dtype, [values.length], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_binarizer_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedNormalizerModel(values, norm = null, dtype = 1, shape = [values.length]) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "normalizer_fixture"), protoString(4, "Normalizer"),
    protoString(7, "ai.onnx.ml"),
  ];
  if (norm != null) nodeFields.splice(4, 0, protoBytes(5, stringAttributeProto("norm", norm)));
  const raw = dtype === 11 ? float64Bytes(values)
    : dtype === 7 ? int64Bytes(values)
      : dtype === 6 ? int32Bytes(values) : float32Bytes(values);
  const initializer = tensorProto("scores", dtype, shape, raw);
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_normalizer_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, shape, true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_normalizer_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedScalerModel(values, scales, offsets, dtype = 1, shape = [values.length]) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "scaler_fixture"), protoString(4, "Scaler"),
    protoString(7, "ai.onnx.ml"),
  ];
  if (offsets != null) nodeFields.splice(4, 0, protoBytes(5, floatListAttributeProto("offset", offsets)));
  if (scales != null) nodeFields.splice(4, 0, protoBytes(5, floatListAttributeProto("scale", scales)));
  const raw = dtype === 11 ? float64Bytes(values)
    : dtype === 7 ? int64Bytes(values)
      : dtype === 6 ? int32Bytes(values) : float32Bytes(values);
  const initializer = tensorProto("scores", dtype, shape, raw);
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_scaler_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, shape, true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_scaler_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedImputerModel(values, imputedValues, replacedValue = null, dtype = 1, shape = [values.length]) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "imputer_fixture"), protoString(4, "Imputer"),
    protoString(7, "ai.onnx.ml"),
  ];
  if (imputedValues != null) {
    nodeFields.splice(4, 0, protoBytes(5, [1, 11].includes(dtype)
      ? floatListAttributeProto("imputed_value_floats", imputedValues)
      : intListAttributeProto("imputed_value_int64s", imputedValues)));
  }
  if (replacedValue != null) {
    nodeFields.splice(4, 0, protoBytes(5, [1, 11].includes(dtype)
      ? floatAttributeProto("replaced_value_float", replacedValue)
      : intAttributeProto("replaced_value_int64", replacedValue)));
  }
  const raw = dtype === 11 ? float64Bytes(values)
    : dtype === 7 ? int64Bytes(values)
      : dtype === 6 ? int32Bytes(values) : float32Bytes(values);
  const initializer = tensorProto("scores", dtype, shape, raw);
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_imputer_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(dtype, shape, true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_imputer_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedOneHotEncoderModel(values, intCategories = null, stringCategories = null, zeros = null, dtype = 7, shape = [values.length]) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"), protoString(3, "one_hot_encoder_fixture"), protoString(4, "OneHotEncoder"),
    protoString(7, "ai.onnx.ml"),
  ];
  if (intCategories != null) nodeFields.splice(4, 0, protoBytes(5, intListAttributeProto("cats_int64s", intCategories)));
  if (stringCategories != null) nodeFields.splice(4, 0, protoBytes(5, stringListAttributeProto("cats_strings", stringCategories)));
  if (zeros != null) nodeFields.splice(4, 0, protoBytes(5, intAttributeProto("zeros", zeros)));
  const initializer = dtype === 8 ? stringTensorProto("scores", shape, values)
    : tensorProto("scores", dtype, shape, dtype === 11 ? float64Bytes(values)
      : dtype === 7 ? int64Bytes(values)
        : dtype === 6 ? int32Bytes(values) : float32Bytes(values));
  const categoryCount = intCategories?.length || stringCategories?.length || 0;
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_one_hot_encoder_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, [...shape, categoryCount], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_one_hot_encoder_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedLinearClassifierModel(values, shape, coefficients, intercepts, labels, {
  stringLabels = false, multiClass = null, postTransform = null, dtype = 1,
} = {}) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "labels"), protoString(2, "probabilities"),
    protoString(3, "linear_classifier_fixture"), protoString(4, "LinearClassifier"),
    protoBytes(5, floatListAttributeProto("coefficients", coefficients)),
    protoBytes(5, floatListAttributeProto("intercepts", intercepts)),
    protoBytes(5, stringLabels ? stringListAttributeProto("classlabels_strings", labels)
      : intListAttributeProto("classlabels_ints", labels)),
    protoString(7, "ai.onnx.ml"),
  ];
  if (multiClass != null) nodeFields.splice(nodeFields.length - 1, 0, protoBytes(5, intAttributeProto("multi_class", multiClass)));
  if (postTransform != null) nodeFields.splice(nodeFields.length - 1, 0, protoBytes(5, stringAttributeProto("post_transform", postTransform)));
  const raw = dtype === 11 ? float64Bytes(values)
    : dtype === 7 ? int64Bytes(values)
      : dtype === 6 ? int32Bytes(values) : float32Bytes(values);
  const initializer = tensorProto("scores", dtype, shape, raw);
  const classCount = intercepts.length;
  const scoreClasses = classCount === 1 && labels.length === 2 ? 2 : classCount;
  const batch = shape.length === 1 ? 1 : shape[0];
  const labelDtype = stringLabels ? 8 : 7;
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_linear_classifier_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("labels", tensorTypeProto(labelDtype, [batch], true))),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, [batch, scoreClasses], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_linear_classifier_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedLinearRegressorModel(values, shape, coefficients, intercepts, {
  targets = 1, includeTargets = true, postTransform = null, dtype = 1,
} = {}) {
  const nodeFields = [
    protoString(1, "scores"), protoString(2, "probabilities"),
    protoString(3, "linear_regressor_fixture"), protoString(4, "LinearRegressor"),
    protoBytes(5, floatListAttributeProto("coefficients", coefficients)),
    protoBytes(5, floatListAttributeProto("intercepts", intercepts)),
    protoString(7, "ai.onnx.ml"),
  ];
  if (includeTargets) nodeFields.splice(nodeFields.length - 1, 0, protoBytes(5, intAttributeProto("targets", targets)));
  if (postTransform != null) nodeFields.splice(nodeFields.length - 1, 0, protoBytes(5, stringAttributeProto("post_transform", postTransform)));
  const raw = dtype === 11 ? float64Bytes(values)
    : dtype === 7 ? int64Bytes(values)
      : dtype === 6 ? int32Bytes(values) : float32Bytes(values);
  const initializer = tensorProto("scores", dtype, shape, raw);
  const batch = shape.length === 1 ? 1 : shape[0];
  const graph = protoMessage([
    protoBytes(1, protoMessage(nodeFields)), protoString(2, "deepbom_linear_regressor_graph"), protoBytes(5, initializer),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, [batch, targets], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_linear_regressor_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedFeatureVectorizerModel() {
  const node = protoMessage([
    protoString(1, "x0"), protoString(1, "x1"), protoString(2, "probabilities"),
    protoString(3, "feature_vectorizer_fixture"), protoString(4, "FeatureVectorizer"),
    protoBytes(5, intListAttributeProto("inputdimensions", [2, 3])), protoString(7, "ai.onnx.ml"),
  ]);
  const graph = protoMessage([
    protoBytes(1, node), protoString(2, "deepbom_feature_vectorizer_graph"),
    protoBytes(11, valueInfoProto("x0", tensorTypeProto(6, [2, 3], true))),
    protoBytes(11, valueInfoProto("x1", tensorTypeProto(6, [2, 2], true))),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(1, [2, 5], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_feature_vectorizer_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedArrayFeatureExtractorModel(indices) {
  const node = protoMessage([
    protoString(1, "x"), protoString(1, "indices"), protoString(2, "probabilities"),
    protoString(3, "array_feature_extractor_fixture"), protoString(4, "ArrayFeatureExtractor"), protoString(7, "ai.onnx.ml"),
  ]);
  const indexTensor = tensorProto("indices", 7, [indices.length], int64Bytes(indices));
  const graph = protoMessage([
    protoBytes(1, node), protoString(2, "deepbom_array_feature_extractor_graph"),
    protoBytes(5, indexTensor),
    protoBytes(11, valueInfoProto("x", tensorTypeProto(6, [2, 4], true))),
    protoBytes(12, valueInfoProto("probabilities", tensorTypeProto(6, [2, indices.length], true))),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, "deepbom_array_feature_extractor_fixture"), protoBytes(7, graph), protoBytes(8, opset)]);
}

function serializedMlModel(graphName, producerName, node, inputType, outputType) {
  const graph = protoMessage([
    protoBytes(1, node), protoString(2, graphName),
    protoBytes(11, valueInfoProto("scores", inputType)), protoBytes(12, valueInfoProto("probabilities", outputType)),
  ]);
  const opset = protoMessage([protoString(1, "ai.onnx.ml"), protoVarintField(2, 1)]);
  return protoMessage([protoVarintField(1, 8), protoString(2, producerName), protoBytes(7, graph), protoBytes(8, opset)]);
}

function valueInfoProto(name, type) { return protoMessage([protoString(1, name), protoBytes(2, type)]); }
function tensorTypeProto(dtype, dims, declareShape) {
  const fields = [protoVarintField(1, dtype)];
  if (declareShape) {
    const shape = protoMessage(dims.map((dim) => protoBytes(1, dim >= 0 ? protoMessage([protoVarintField(1, dim)]) : protoMessage([]))));
    fields.push(protoBytes(2, shape));
  }
  return protoMessage([protoBytes(1, protoMessage(fields))]);
}
function sequenceTypeProto(elementType) { return protoMessage([protoBytes(4, protoMessage([protoBytes(1, elementType)]))]); }
function mapTypeProto(keyType, valueType) { return protoMessage([protoBytes(5, protoMessage([protoVarintField(1, keyType), protoBytes(2, valueType)]))]); }
function stringListAttributeProto(name, values) {
  return protoMessage([protoString(1, name), ...values.map((value) => protoString(9, value)), protoVarintField(20, 8)]);
}
function intListAttributeProto(name, values) {
  return protoMessage([protoString(1, name), ...values.map((value) => protoVarintField(8, value)), protoVarintField(20, 7)]);
}
function tensorProto(name, dtype, dims, raw) {
  return protoMessage([...dims.map((dim) => protoVarintField(1, dim)), protoVarintField(2, dtype), protoString(8, name), protoBytes(9, raw)]);
}
function stringTensorProto(name, dims, values) {
  return protoMessage([...dims.map((dim) => protoVarintField(1, dim)), protoVarintField(2, 8), ...values.map((value) => protoString(6, value)), protoString(8, name)]);
}
function int64Bytes(values) {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setBigInt64(index * 8, BigInt(value), true));
  return bytes;
}
function int32Bytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, Number(value), true));
  return bytes;
}
function float32Bytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, Number(value), true));
  return bytes;
}
function float64Bytes(values) {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat64(index * 8, Number(value), true));
  return bytes;
}
function stringAttributeProto(name, value) {
  return protoMessage([protoString(1, name), protoString(4, value), protoVarintField(20, 3)]);
}
function floatAttributeProto(name, value) {
  return protoMessage([protoString(1, name), protoFloatField(2, value), protoVarintField(20, 1)]);
}
function floatListAttributeProto(name, values) {
  return protoMessage([protoString(1, name), ...values.map((value) => protoFloatField(7, value)), protoVarintField(20, 6)]);
}
function intAttributeProto(name, value) {
  return protoMessage([protoString(1, name), protoVarintField(3, value), protoVarintField(20, 2)]);
}
function protoString(field, value) { return protoBytes(field, new TextEncoder().encode(value)); }
function protoBytes(field, value) {
  const bytes = value instanceof Uint8Array ? value : new Uint8Array(value);
  return protoMessage([protoVarint((field << 3) | 2), protoVarint(bytes.length), bytes]);
}
function protoVarintField(field, value) { return protoMessage([protoVarint(field << 3), protoVarint(value)]); }
function protoFloatField(field, value) {
  const payload = new Uint8Array(5);
  payload[0] = (field << 3) | 5;
  new DataView(payload.buffer).setFloat32(1, Number(value), true);
  return payload;
}
function protoVarint(value) {
  let remaining = BigInt(value);
  if (remaining < 0n) remaining = BigInt.asUintN(64, remaining);
  const bytes = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining) byte |= 0x80;
    bytes.push(byte);
  } while (remaining);
  return new Uint8Array(bytes);
}
function protoMessage(parts) {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
}
