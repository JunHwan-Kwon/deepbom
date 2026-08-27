import { createHash } from "node:crypto";
import { ONNX_ML_VALUE_SOURCE } from "../web/lib/onnx-ml-value-inference.js";
import { fetchPinnedBytes } from "./fetch-pinned-source.mjs";

const commit = ONNX_ML_VALUE_SOURCE.commit;
if (!/^[a-f0-9]{40}$/.test(commit || "")) throw new Error("ONNX-ML value source commit is not pinned to a full commit hash.");
if (ONNX_ML_VALUE_SOURCE.release !== "v1.21.0") throw new Error("ONNX-ML value source release drifted from v1.21.0.");

const expectedRoles = new Set([
  "traditional_ml_operator_schema",
  "traditional_ml_schema_history",
  "traditional_ml_historical_operator_schema",
  "tree_ensemble_reference",
  "tree_ensemble_classifier_reference",
  "tree_ensemble_regressor_reference",
  "tree_ensemble_legacy_reference_helper",
]);
const documents = ONNX_ML_VALUE_SOURCE.documents || [];
if (documents.length !== expectedRoles.size || new Set(documents.map((source) => source.role)).size !== documents.length
  || documents.some((source) => !expectedRoles.has(source.role))) {
  throw new Error("ONNX-ML value source ledger must contain the schema and TreeEnsemble reference documents with unique roles.");
}

const runtimeCommit = ONNX_ML_VALUE_SOURCE.runtime_reference_commit;
if (!/^[a-f0-9]{40}$/.test(runtimeCommit || "")) throw new Error("ORT ONNX-ML reference commit is not pinned to a full commit hash.");
const expectedRuntimeRoles = new Set([
  "ort_cpu_binarizer_kernel",
  "ort_cpu_binarizer_contract",
  "ort_cpu_binarizer_tests",
  "ort_cpu_normalizer_kernel",
  "ort_cpu_normalizer_contract",
  "ort_cpu_normalizer_tests",
  "ort_cpu_scaler_kernel",
  "ort_cpu_scaler_contract",
  "ort_cpu_scaler_tests",
  "ort_cpu_imputer_kernel",
  "ort_cpu_imputer_contract",
  "ort_cpu_imputer_tests",
  "ort_cpu_one_hot_encoder_kernel",
  "ort_cpu_one_hot_encoder_contract",
  "ort_cpu_one_hot_encoder_tests",
  "ort_cpu_linear_classifier_kernel",
  "ort_cpu_linear_classifier_contract",
  "ort_cpu_linear_classifier_tests",
  "ort_cpu_linear_regressor_kernel",
  "ort_cpu_linear_regressor_contract",
  "ort_cpu_linear_regressor_tests",
  "ort_cpu_ml_post_transform_contract",
  "ort_cpu_svm_classifier_kernel",
  "ort_cpu_svm_classifier_contract",
  "ort_cpu_svm_classifier_tests",
  "ort_cpu_svm_regressor_kernel",
  "ort_cpu_svm_regressor_contract",
  "ort_cpu_svm_regressor_tests",
  "ort_cpu_label_encoder_kernel",
  "ort_cpu_label_encoder_contract",
  "ort_cpu_label_encoder_tests",
  "ort_cpu_zipmap_kernel",
  "ort_cpu_zipmap_tests",
  "ort_cpu_dict_vectorizer_kernel",
  "ort_cpu_dict_vectorizer_contract",
  "ort_cpu_dict_vectorizer_tests",
  "ort_cpu_category_mapper_kernel",
  "ort_cpu_category_mapper_contract",
  "ort_cpu_category_mapper_tests",
  "ort_cpu_feature_vectorizer_kernel",
  "ort_cpu_feature_vectorizer_contract",
  "ort_cpu_feature_vectorizer_tests",
  "ort_cpu_array_feature_extractor_kernel",
  "ort_cpu_array_feature_extractor_contract",
  "ort_cpu_array_feature_extractor_tests",
  "ort_cpu_tree_ensemble_kernel",
  "ort_cpu_tree_ensemble_contract",
  "ort_cpu_tree_classifier_kernel",
  "ort_cpu_tree_classifier_contract",
  "ort_cpu_tree_regressor_kernel",
  "ort_cpu_tree_regressor_contract",
  "ort_cpu_tree_common_contract",
  "ort_cpu_tree_attribute_contract",
  "ort_cpu_tree_aggregator_contract",
  "ort_cpu_tree_helper_kernel",
  "ort_cpu_tree_helper_contract",
  "ort_cpu_tree_classifier_tests",
  "ort_cpu_tree_regressor_and_v5_tests",
]);
const runtimeDocuments = ONNX_ML_VALUE_SOURCE.runtime_reference_documents || [];
if (runtimeDocuments.length !== expectedRuntimeRoles.size
  || new Set(runtimeDocuments.map((source) => source.role)).size !== runtimeDocuments.length
  || runtimeDocuments.some((source) => !expectedRuntimeRoles.has(source.role))) {
  throw new Error("ORT ONNX-ML reference ledger must contain every implemented operator kernel, contract, semantic helper, and test source.");
}

const verified = [];
for (const source of documents) {
  const expectedPrefix = `https://raw.githubusercontent.com/onnx/onnx/${commit}/`;
  if (!String(source.source_ref || "").startsWith(expectedPrefix)) throw new Error(`${source.role} is not bound to the pinned ONNX commit.`);
  if (!/^[a-f0-9]{64}$/.test(source.sha256 || "")) throw new Error(`${source.role} has an invalid SHA-256.`);
  const bytes = await fetchPinnedBytes(source.source_ref, { label: "ONNX" });
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== source.sha256) throw new Error(`${source.role} SHA-256 mismatch: ${sha256} !== ${source.sha256}`);
  verified.push(`${source.role} ${bytes.byteLength} B ${sha256}`);
}

for (const source of runtimeDocuments) {
  const expectedPrefix = `https://raw.githubusercontent.com/microsoft/onnxruntime/${runtimeCommit}/`;
  if (!String(source.source_ref || "").startsWith(expectedPrefix)) throw new Error(`${source.role} is not bound to the pinned ORT commit.`);
  if (!/^[a-f0-9]{64}$/.test(source.sha256 || "")) throw new Error(`${source.role} has an invalid SHA-256.`);
  const bytes = await fetchPinnedBytes(source.source_ref, { label: "ORT" });
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== source.sha256) throw new Error(`${source.role} SHA-256 mismatch: ${sha256} !== ${source.sha256}`);
  verified.push(`${source.role} ${bytes.byteLength} B ${sha256}`);
}

console.log(`Pinned ONNX-ML value-contract source verification passed at onnx/onnx@${commit} and microsoft/onnxruntime@${runtimeCommit}:\n${verified.map((item) => `  - ${item}`).join("\n")}`);
