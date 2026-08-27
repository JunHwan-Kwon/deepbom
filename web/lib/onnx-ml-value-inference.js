import { assessOnnxAttributeProto } from "./onnx-schema-legality.js";
import {
  canonicalOnnxTypeProto,
  cloneOnnxTypeProto,
  makeOnnxMapType,
  makeOnnxSequenceType,
  makeOnnxTensorType,
  onnxTypeProtoFromValue,
  onnxValueDescriptorFromType,
} from "./onnx-type-proto.js";
import { inferOnnxMlNormalizer } from "./onnx-ml-normalizer.js";
import { inferOnnxMlScaler } from "./onnx-ml-scaler.js";
import { inferOnnxMlImputer } from "./onnx-ml-imputer.js";
import { inferOnnxMlOneHotEncoder } from "./onnx-ml-one-hot-encoder.js";
import { inferOnnxMlLinearClassifier, inferOnnxMlLinearRegressor } from "./onnx-ml-linear-model.js";
import { inferOnnxMlLabelEncoder, resolveOnnxMlLabelEncoderVersion } from "./onnx-ml-label-encoder.js";
import { inferOnnxMlSvmClassifier, inferOnnxMlSvmRegressor } from "./onnx-ml-svm.js";
import {
  inferOnnxMlTreeEnsemble,
  inferOnnxMlTreeEnsembleClassifier,
  inferOnnxMlTreeEnsembleRegressor,
  resolveOnnxMlTreeEnsembleVersion,
} from "./onnx-ml-tree-ensemble.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const ORT_SOURCE_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
const MAX_EXACT_SEQUENCE_INVENTORY = 4_096;
const LABEL_ENCODER_SCHEMAS = new Map([
  [1, new Map([["classes_strings", 8], ["default_int64", 2], ["default_string", 3]])],
  [2, new Map([
    ["keys_strings", 8], ["keys_int64s", 7], ["keys_floats", 6],
    ["values_strings", 8], ["values_int64s", 7], ["values_floats", 6],
    ["default_string", 3], ["default_int64", 2], ["default_float", 1],
  ])],
  [4, new Map([
    ["keys_tensor", 4], ["keys_strings", 8], ["keys_int64s", 7], ["keys_floats", 6],
    ["values_tensor", 4], ["values_strings", 8], ["values_int64s", 7], ["values_floats", 6],
    ["default_string", 3], ["default_int64", 2], ["default_float", 1], ["default_tensor", 4],
  ])],
]);
const TREE_CLASSIFIER_V1_SCHEMA = new Map([
  ["nodes_treeids", 7], ["nodes_nodeids", 7], ["nodes_featureids", 7], ["nodes_values", 6],
  ["nodes_hitrates", 6], ["nodes_modes", 8], ["nodes_truenodeids", 7], ["nodes_falsenodeids", 7],
  ["nodes_missing_value_tracks_true", 7], ["class_treeids", 7], ["class_nodeids", 7], ["class_ids", 7],
  ["class_weights", 6], ["classlabels_strings", 8], ["classlabels_int64s", 7], ["post_transform", 3],
  ["base_values", 6],
]);
const TREE_CLASSIFIER_V3_SCHEMA = new Map([
  ...TREE_CLASSIFIER_V1_SCHEMA,
  ["nodes_values_as_tensor", 4], ["nodes_hitrates_as_tensor", 4],
  ["class_weights_as_tensor", 4], ["base_values_as_tensor", 4],
]);
const TREE_REGRESSOR_V1_SCHEMA = new Map([
  ["nodes_treeids", 7], ["nodes_nodeids", 7], ["nodes_featureids", 7], ["nodes_values", 6],
  ["nodes_hitrates", 6], ["nodes_modes", 8], ["nodes_truenodeids", 7], ["nodes_falsenodeids", 7],
  ["nodes_missing_value_tracks_true", 7], ["target_treeids", 7], ["target_nodeids", 7], ["target_ids", 7],
  ["target_weights", 6], ["n_targets", 2], ["post_transform", 3], ["aggregate_function", 3], ["base_values", 6],
]);
const TREE_REGRESSOR_V3_SCHEMA = new Map([
  ...TREE_REGRESSOR_V1_SCHEMA,
  ["nodes_values_as_tensor", 4], ["nodes_hitrates_as_tensor", 4],
  ["target_weights_as_tensor", 4], ["base_values_as_tensor", 4],
]);
const TREE_ENSEMBLE_V5_SCHEMA = new Map([
  ["nodes_featureids", 7], ["nodes_splits", 4], ["nodes_hitrates", 4], ["nodes_modes", 4],
  ["nodes_truenodeids", 7], ["nodes_falsenodeids", 7], ["nodes_trueleafs", 7], ["nodes_falseleafs", 7],
  ["nodes_missing_value_tracks_true", 7], ["tree_roots", 7], ["membership_values", 4],
  ["leaf_targetids", 7], ["leaf_weights", 4], ["n_targets", 2], ["post_transform", 2], ["aggregate_function", 2],
]);
const ML_SCHEMAS = new Map([
  ["Binarizer", new Map([["threshold", 1]])],
  ["Normalizer", new Map([["norm", 3]])],
  ["Scaler", new Map([["scale", 6], ["offset", 6]])],
  ["Imputer", new Map([
    ["imputed_value_floats", 6], ["imputed_value_int64s", 7],
    ["replaced_value_float", 1], ["replaced_value_int64", 2],
  ])],
  ["OneHotEncoder", new Map([["cats_int64s", 7], ["cats_strings", 8], ["zeros", 2]])],
  ["LinearClassifier", new Map([
    ["classlabels_ints", 7], ["classlabels_strings", 8], ["coefficients", 6],
    ["intercepts", 6], ["multi_class", 2], ["post_transform", 3],
  ])],
  ["LinearRegressor", new Map([
    ["coefficients", 6], ["intercepts", 6], ["post_transform", 3], ["targets", 2],
  ])],
  ["LabelEncoder", LABEL_ENCODER_SCHEMAS.get(4)],
  ["SVMClassifier", new Map([
    ["kernel_type", 3], ["kernel_params", 6], ["vectors_per_class", 7], ["support_vectors", 6],
    ["coefficients", 6], ["prob_a", 6], ["prob_b", 6], ["rho", 6], ["post_transform", 3],
    ["classlabels_strings", 8], ["classlabels_ints", 7],
  ])],
  ["SVMRegressor", new Map([
    ["kernel_type", 3], ["kernel_params", 6], ["support_vectors", 6], ["one_class", 2],
    ["coefficients", 6], ["n_supports", 2], ["post_transform", 3], ["rho", 6],
  ])],
  ["TreeEnsemble", TREE_ENSEMBLE_V5_SCHEMA],
  ["TreeEnsembleClassifier", TREE_CLASSIFIER_V3_SCHEMA],
  ["TreeEnsembleRegressor", TREE_REGRESSOR_V3_SCHEMA],
  ["ZipMap", new Map([["classlabels_strings", 8], ["classlabels_int64s", 7]])],
  ["CastMap", new Map([["cast_to", 3], ["map_form", 3], ["max_map", 2]])],
  ["DictVectorizer", new Map([["string_vocabulary", 8], ["int64_vocabulary", 7]])],
  ["CategoryMapper", new Map([["cats_strings", 8], ["cats_int64s", 7], ["default_string", 3], ["default_int64", 2]])],
  ["FeatureVectorizer", new Map([["inputdimensions", 7]])],
  ["ArrayFeatureExtractor", new Map()],
]);

export const ONNX_ML_VALUE_OPS = new Set(ML_SCHEMAS.keys());

export const ONNX_ML_VALUE_SOURCE = Object.freeze({
  release: "v1.21.0",
  commit: SOURCE_COMMIT,
  documents: Object.freeze([
    Object.freeze({
      role: "traditional_ml_operator_schema",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/traditionalml/defs.cc`,
      sha256: "4588b9efe493ea820d54c5b65b6af6ad8a7860f625f97f5d6edcfd5bf06125e6",
    }),
    Object.freeze({
      role: "traditional_ml_schema_history",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/operator_sets_ml.h`,
      sha256: "fa3d663df091a0cadc85d902a12d84d7465bfec8cf7433861f82b99f921278a4",
    }),
    Object.freeze({
      role: "traditional_ml_historical_operator_schema",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/defs/traditionalml/old.cc`,
      sha256: "6cb50c9e803a7295e5581ee3416e8098115806627a7248791eb3730750a8a94f",
    }),
    Object.freeze({
      role: "tree_ensemble_reference",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/reference/ops/aionnxml/op_tree_ensemble.py`,
      sha256: "eaae9d894a32acab1a1e3a0a847ec87dc1f17b57837af0b98d925747b276d47d",
    }),
    Object.freeze({
      role: "tree_ensemble_classifier_reference",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/reference/ops/aionnxml/op_tree_ensemble_classifier.py`,
      sha256: "987217215e7ef6db4855e1d8bdbf15e81fff86aa4de5d99d4aee64398cfa2294",
    }),
    Object.freeze({
      role: "tree_ensemble_regressor_reference",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/reference/ops/aionnxml/op_tree_ensemble_regressor.py`,
      sha256: "01ee6cae9715d9aef6a1e25a63004f1cf5120f83f7b314af8035ee7e5090ba29",
    }),
    Object.freeze({
      role: "tree_ensemble_legacy_reference_helper",
      source_ref: `https://raw.githubusercontent.com/onnx/onnx/${SOURCE_COMMIT}/onnx/reference/ops/aionnxml/op_tree_ensemble_helper.py`,
      sha256: "feee50e93820f12f949175d3b878a63ce54bdf9e9a9744fe94cac5ad70f5f619",
    }),
  ]),
  runtime_reference_commit: ORT_SOURCE_COMMIT,
  runtime_reference_documents: Object.freeze([
    Object.freeze({
      role: "ort_cpu_binarizer_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/binarizer.cc`,
      sha256: "5253671998fd3c5493d2c8acfcca34ba5747b575f05c635ad7533696a53a43bf",
    }),
    Object.freeze({
      role: "ort_cpu_binarizer_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/binarizer.h`,
      sha256: "cbd925dc84bc46d13bf4193bbe3e00a28d485bf1666fa2ebfe51476b1ae653c6",
    }),
    Object.freeze({
      role: "ort_cpu_binarizer_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/binarizer_test.cc`,
      sha256: "b39121bd0431449e6e2d99b9c2cc41085eb0ea70f21930807bdc80df9d2cf1f0",
    }),
    Object.freeze({
      role: "ort_cpu_normalizer_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/normalizer.cc`,
      sha256: "50b0a8eb826fd730b3c895f5493d36a8c12e477e9b91337b10170413b73af20c",
    }),
    Object.freeze({
      role: "ort_cpu_normalizer_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/normalizer.h`,
      sha256: "c8742d10e18154d83e20482c7e57b1263d427e0a5b167f3c4a710bfaf5d4310c",
    }),
    Object.freeze({
      role: "ort_cpu_normalizer_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/normalizer_test.cc`,
      sha256: "74b0c6f57b8c04a549b9969b1feef378d0bb0dd9acc3e2a3109f35b9613ba80b",
    }),
    Object.freeze({
      role: "ort_cpu_scaler_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/scaler.cc`,
      sha256: "08ee63f5e1b4a2341f190537198761d648528262752e4cf24b083cbee1fdaee3",
    }),
    Object.freeze({
      role: "ort_cpu_scaler_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/scaler.h`,
      sha256: "1c3cf3fd8063b892dc46d49c6391dd340442e26410a1fc2efff48d27c611b13c",
    }),
    Object.freeze({
      role: "ort_cpu_scaler_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/scaler_test.cc`,
      sha256: "0aa327aeec7543785e0b8c903500c84c884c9f8de5753906658cb07dd6e1c1d2",
    }),
    Object.freeze({
      role: "ort_cpu_imputer_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/imputer.cc`,
      sha256: "10de709f7625306815ef374afdf3fd6dd930b4c8a6bd65e8f0fc348cda5f4dea",
    }),
    Object.freeze({
      role: "ort_cpu_imputer_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/imputer.h`,
      sha256: "c2497b44ad5346190c3ea2f627e82aa6d61b8a4fae2e61508b63efd6c238b019",
    }),
    Object.freeze({
      role: "ort_cpu_imputer_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/imputer_test.cc`,
      sha256: "1b65b1904c8cea3ba3a1ed3c20fe668a6b54bb1a9a22cb7c2a492a5ab11e7a16",
    }),
    Object.freeze({
      role: "ort_cpu_one_hot_encoder_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/onehotencoder.cc`,
      sha256: "8b5bd9bcdf8455326ec857b540743465cf57bdb50f7f26820f734a179c3431ef",
    }),
    Object.freeze({
      role: "ort_cpu_one_hot_encoder_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/onehotencoder.h`,
      sha256: "f9a8522b075b1c8e33b2f4031c719db69bbab9bc5280f8b532c468286b8c4c95",
    }),
    Object.freeze({
      role: "ort_cpu_one_hot_encoder_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/onehotencoder_test.cc`,
      sha256: "644a6caffd60f1a1721bb3dd282a0c9d372bbae2bc949b85c52bfc68b9815c10",
    }),
    Object.freeze({
      role: "ort_cpu_linear_classifier_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/linearclassifier.cc`,
      sha256: "8fcef175e9db50c017a94ac4db1b3c118294dea7eab93315d58dabfdae95d052",
    }),
    Object.freeze({
      role: "ort_cpu_linear_classifier_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/linearclassifier.h`,
      sha256: "ec8288d8b9f01115c26f9d993d586d53cb6c7936ca4b312f96b2b395aa417344",
    }),
    Object.freeze({
      role: "ort_cpu_linear_classifier_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/linearclassifer_test.cc`,
      sha256: "2e55ce60ffd7f2a9a5b1c5059b9ec0fab5a237ded10ea27908957d8005c51f6e",
    }),
    Object.freeze({
      role: "ort_cpu_linear_regressor_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/linearregressor.cc`,
      sha256: "615259243fec59bd088299bb4778f6f484b84af6f1ea8985497e73bc58ee11e2",
    }),
    Object.freeze({
      role: "ort_cpu_linear_regressor_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/linearregressor.h`,
      sha256: "af47c85ed17ea6c633b16936b6bcc296a390814391ac12acce9b9b92910154b5",
    }),
    Object.freeze({
      role: "ort_cpu_linear_regressor_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/linearregressor_test.cc`,
      sha256: "5e30730fe8925ae3d8044872b0542f477164108a1c89dec5764d14d272b3b360",
    }),
    Object.freeze({
      role: "ort_cpu_ml_post_transform_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/ml_common.h`,
      sha256: "fabd40f04a61f02a882c87d4056a4fb94a7f923cba6290a46ca84bcec8c0493f",
    }),
    Object.freeze({
      role: "ort_cpu_svm_classifier_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/svmclassifier.cc`,
      sha256: "36e2d1d5d69cfbadc7eefea0363d5d5bbaf6ee52b069bb6e9cf52863e6035488",
    }),
    Object.freeze({
      role: "ort_cpu_svm_classifier_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/svmclassifier.h`,
      sha256: "affd924534a267c8fb723beff1bc2072431d27442e98348fb2f7e19abcfb03b7",
    }),
    Object.freeze({
      role: "ort_cpu_svm_classifier_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/svmclassifier_test.cc`,
      sha256: "5ca65fcd3ebe1874c560d1f72471d328d94768871e21bf453e039979328c9428",
    }),
    Object.freeze({
      role: "ort_cpu_svm_regressor_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/svmregressor.cc`,
      sha256: "79badf47a76df5d78005604247e9150f7d63b0604b5668fca1a00655bd8dc5fb",
    }),
    Object.freeze({
      role: "ort_cpu_svm_regressor_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/svmregressor.h`,
      sha256: "9ef36f2efd61ba1f2a13315cce44c2bfe845b45809a8671258cf53213d98a462",
    }),
    Object.freeze({
      role: "ort_cpu_svm_regressor_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/svmregressor_test.cc`,
      sha256: "293d3412fa6b4fa411ea3e9f9f79a880a557d307402ab919cb8f20582566100d",
    }),
    Object.freeze({
      role: "ort_cpu_label_encoder_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/label_encoder.cc`,
      sha256: "477280d9b83624831f4959f4e279c5c8dd8213b787570e6a4cc084eec0214edc",
    }),
    Object.freeze({
      role: "ort_cpu_label_encoder_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/label_encoder.h`,
      sha256: "52accbc66502dedc8babb3c469d2cb969b0869e8ced3a8618de56080dd62729b",
    }),
    Object.freeze({
      role: "ort_cpu_label_encoder_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/label_encoder_test.cc`,
      sha256: "e8f8118f08e40f6b272ee57ba81353837807c3d69834dba8b2720b420406a9e2",
    }),
    Object.freeze({
      role: "ort_cpu_zipmap_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/zipmap.cc`,
      sha256: "f9205124962c59fcaf2f56aee5e0f47af05a7f21b2bb897e0e08dc39ef7f481a",
    }),
    Object.freeze({
      role: "ort_cpu_zipmap_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/zipmap_test.cc`,
      sha256: "e81aa0042a93681ebec50037e5effdca5161076dfa5929991d5ebb77c05351c1",
    }),
    Object.freeze({
      role: "ort_cpu_dict_vectorizer_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/dictvectorizer.cc`,
      sha256: "cbde6289fc3b17a518c6bd9d07404e798a546374be33557ce0dbabbe5779ac38",
    }),
    Object.freeze({
      role: "ort_cpu_dict_vectorizer_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/dictvectorizer.h`,
      sha256: "b372ee05451ff430e7a6f112addabde1da826792411061b456dbca0ce78b69b8",
    }),
    Object.freeze({
      role: "ort_cpu_dict_vectorizer_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/dictvectorizer_test.cc`,
      sha256: "421dde93972699d765938fa82ca28848f3cfef399367f2f053c4167669004097",
    }),
    Object.freeze({
      role: "ort_cpu_category_mapper_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/category_mapper.cc`,
      sha256: "b957901eb947300fff754cb7c9538c5dbdf3f2fa38a54cfae0e0dfda3c94ddfb",
    }),
    Object.freeze({
      role: "ort_cpu_category_mapper_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/category_mapper.h`,
      sha256: "0ed0df8a1616d08c291d8fd41d6c6bada42d489bf801ff86071187314f12d248",
    }),
    Object.freeze({
      role: "ort_cpu_category_mapper_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/category_mapper_test.cc`,
      sha256: "7c619f23f2cc7945ab57f759789014ca71a9cffe998664e85b28671a1b97fead",
    }),
    Object.freeze({
      role: "ort_cpu_feature_vectorizer_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/feature_vectorizer.cc`,
      sha256: "fa429c30a643bbdc5694bc868c5a8355222df8e809581cbf0e3e685990de00c0",
    }),
    Object.freeze({
      role: "ort_cpu_feature_vectorizer_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/feature_vectorizer.h`,
      sha256: "dc335549227277c0a9a38c4448322ca6538114e6453695ad36f6eeb6782f4f8d",
    }),
    Object.freeze({
      role: "ort_cpu_feature_vectorizer_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/feature_vectorizer_test.cc`,
      sha256: "cbe443bd3aea11d02ecd005b6ecb7e46c2ed6a08c68b942b302cc44e01b0cab1",
    }),
    Object.freeze({
      role: "ort_cpu_array_feature_extractor_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc`,
      sha256: "d06602c071cb1a8ab7d1d84b50c6df3edac6c551a289366d01bf5f84276a5975",
    }),
    Object.freeze({
      role: "ort_cpu_array_feature_extractor_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/array_feature_extractor.h`,
      sha256: "2f0eebb9eca913d3211fcd70ca3a8c2b49c5bce5bd8b265ce16fae89891d9279",
    }),
    Object.freeze({
      role: "ort_cpu_array_feature_extractor_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/array_feature_extractor_test.cc`,
      sha256: "f7f7e24bcedde18377e0b261b907d2fc473516bf5cc11bac2c804fb84922a33c",
    }),
    Object.freeze({
      role: "ort_cpu_tree_ensemble_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble.cc`,
      sha256: "bbe851002cdf367cb73dbb2a8fe135759d23b047a32fe120fc53ed643e306f0c",
    }),
    Object.freeze({
      role: "ort_cpu_tree_ensemble_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble.h`,
      sha256: "e8190efe63c7ba697120f268b7dd413b3afe6ffec899fe810bad464b28a49bdf",
    }),
    Object.freeze({
      role: "ort_cpu_tree_classifier_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.cc`,
      sha256: "72247de06649e6d63d9f45b7c4e4a1177422f4578971917160e5da7157e70c8f",
    }),
    Object.freeze({
      role: "ort_cpu_tree_classifier_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_classifier.h`,
      sha256: "c03b4f72bd360bf13efb84b66e4f05646124b142a3c11f4e8fb1ce5093624bb1",
    }),
    Object.freeze({
      role: "ort_cpu_tree_regressor_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/treeregressor.cc`,
      sha256: "2c206015893789228a24981c8a2d8ada0512750d996633be0dd54e6d8c0d7791",
    }),
    Object.freeze({
      role: "ort_cpu_tree_regressor_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/treeregressor.h`,
      sha256: "c829863e6ed10c417c0c740cf8a4fc899399babbc54041294c34c07690d99342",
    }),
    Object.freeze({
      role: "ort_cpu_tree_common_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_common.h`,
      sha256: "57ca7bd8296c3b3d754e653d5e4b7f5876a7e6f3b1354a74d50f3535b47956b3",
    }),
    Object.freeze({
      role: "ort_cpu_tree_attribute_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_attribute.h`,
      sha256: "16501a65083fefdeb3938c62052213590396b4b9db14b081b466234f00955f81",
    }),
    Object.freeze({
      role: "ort_cpu_tree_aggregator_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_aggregator.h`,
      sha256: "f83f53ad0cbcc33636b5cd22c07b15bcc635fd2691119451c5c9ed62fcf1e478",
    }),
    Object.freeze({
      role: "ort_cpu_tree_helper_kernel",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_helper.cc`,
      sha256: "93dd51c289e622e6ba8f910234e234cfb42f294253ce7f3037dbfe0203bc4aec",
    }),
    Object.freeze({
      role: "ort_cpu_tree_helper_contract",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/core/providers/cpu/ml/tree_ensemble_helper.h`,
      sha256: "f8daa5d4bf7cc19f0273540d2dcd19ed4412e2657655bcf64e616d6cd77a51da",
    }),
    Object.freeze({
      role: "ort_cpu_tree_classifier_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/tree_ensembler_classifier_test.cc`,
      sha256: "402fe3a4279a78c05312ca0456f3e3c68bd654ac9350cb620523a157748d0282",
    }),
    Object.freeze({
      role: "ort_cpu_tree_regressor_and_v5_tests",
      source_ref: `https://raw.githubusercontent.com/microsoft/onnxruntime/${ORT_SOURCE_COMMIT}/onnxruntime/test/providers/cpu/ml/tree_ensembler_test.cc`,
      sha256: "5b11601f71a89323cfd77601106de63adaa01c869005249be51d4b8fb7171a2c",
    }),
  ]),
});

export function canInferOnnxMlNode(node) {
  return normalizeDomain(node?.domain) === "ai.onnx.ml" && ONNX_ML_VALUE_OPS.has(node?.opType);
}

export function assessOnnxMlNodeSchemaForm(node, importedOpset) {
  const reasons = [];
  const schemaSinceVersion = node?.opType === "LabelEncoder" ? resolveOnnxMlLabelEncoderVersion(importedOpset)
    : ["TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor"].includes(node?.opType)
      ? resolveOnnxMlTreeEnsembleVersion(node?.opType, importedOpset) : 1;
  const attributes = schemaAttributes(node?.opType, schemaSinceVersion);
  const inputCount = (node?.inputs || []).length;
  const inputRange = node?.opType === "FeatureVectorizer" ? [1, null]
    : node?.opType === "ArrayFeatureExtractor" ? [2, 2] : [1, 1];
  const inputOptions = node?.opType === "FeatureVectorizer" ? "V" : "R";
  const outputRange = ["LinearClassifier", "SVMClassifier", "TreeEnsembleClassifier"].includes(node?.opType) ? [2, 2] : [1, 1];
  const requiredAttributes = node?.opType === "LinearClassifier" ? ["coefficients"]
    : node?.opType === "TreeEnsemble" ? [
      "nodes_featureids", "nodes_splits", "nodes_modes", "nodes_truenodeids", "nodes_falsenodeids",
      "nodes_trueleafs", "nodes_falseleafs", "tree_roots", "leaf_targetids", "leaf_weights",
    ] : [];
  if (!Number.isSafeInteger(importedOpset) || importedOpset < 1 || schemaSinceVersion == null) reasons.push("operator_not_defined_at_imported_opset");
  if (inputCount < inputRange[0] || inputRange[1] != null && inputCount > inputRange[1]) {
    reasons.push(`input_count_mismatch:${inputCount}:${inputRange[0]}:${inputRange[1] ?? "variadic"}`);
  }
  if ((node?.outputs || []).length < outputRange[0] || (node?.outputs || []).length > outputRange[1]) {
    reasons.push(`output_count_mismatch:${(node?.outputs || []).length}:${outputRange[0]}:${outputRange[1]}`);
  }
  for (let index = 0; index < inputCount; index += 1) {
    if (!node?.inputs?.[index]) reasons.push(`required_input_omitted:${index}`);
  }
  for (let index = 0; index < outputRange[0]; index += 1) {
    if (!node?.outputs?.[index]) reasons.push(`required_output_omitted:${index}`);
  }
  for (const name of new Set(node?.duplicateAttributeNames || [])) reasons.push(`duplicate_attribute_name:${name}`);
  for (const [name, attribute] of node?.attributes || []) {
    const expectedType = attributes.get(name);
    if (!expectedType) {
      reasons.push(`attribute_not_defined:${name}`);
      continue;
    }
    const actual = assessOnnxAttributeProto(attribute);
    if (actual.status !== "pass") reasons.push(`${actual.reason}:${name}`);
    else if (actual.type !== expectedType) reasons.push(`attribute_type_mismatch:${name}:${expectedType}:${actual.type}`);
  }
  for (const name of requiredAttributes) {
    if (!node?.attributes?.has(name)) reasons.push(`required_attribute_missing:${name}`);
  }
  return {
    op_name: node?.opType || "UNKNOWN",
    domain: "ai.onnx.ml",
    imported_opset: importedOpset ?? null,
    schema_since_version: schemaSinceVersion,
    input_count: (node?.inputs || []).length,
    output_count: (node?.outputs || []).length,
    explicit_attributes: [...(node?.attributes?.keys?.() || [])].sort(),
    schema_input_range: inputRange,
    schema_output_range: outputRange,
    schema_input_options: inputOptions,
    schema_output_options: "R",
    schema_attribute_count: attributes.size,
    required_attributes: requiredAttributes,
    status: reasons.length ? "fail" : "pass",
    reason_codes: [...new Set(reasons)],
    detail: reasons.length
      ? `The NodeProto violates the pinned ai.onnx.ml ${node?.opType || "UNKNOWN"}-${schemaSinceVersion || "unresolved"} formal schema.`
      : `The NodeProto matches the pinned ai.onnx.ml ${node?.opType || "UNKNOWN"}-${schemaSinceVersion} formal schema.`,
  };
}

export function inferOnnxMlNode({ node, tensorMap, nodeIndex, importedOpset, scope = "main_graph" }) {
  if (!canInferOnnxMlNode(node)) return null;
  if (node.opType === "Binarizer") return inferBinarizer(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "Normalizer") return inferOnnxMlNormalizer({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "Scaler") return inferOnnxMlScaler({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "Imputer") return inferOnnxMlImputer({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "OneHotEncoder") return inferOnnxMlOneHotEncoder({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "LinearClassifier") return inferOnnxMlLinearClassifier({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "LinearRegressor") return inferOnnxMlLinearRegressor({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "LabelEncoder") return inferOnnxMlLabelEncoder({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "SVMClassifier") return inferOnnxMlSvmClassifier({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "SVMRegressor") return inferOnnxMlSvmRegressor({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "TreeEnsemble") return inferOnnxMlTreeEnsemble({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "TreeEnsembleClassifier") return inferOnnxMlTreeEnsembleClassifier({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "TreeEnsembleRegressor") return inferOnnxMlTreeEnsembleRegressor({ node, tensorMap, nodeIndex, importedOpset, scope });
  if (node.opType === "ZipMap") return inferZipMap(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "CastMap") return inferCastMap(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "DictVectorizer") return inferDictVectorizer(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "CategoryMapper") return inferCategoryMapper(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "FeatureVectorizer") return inferFeatureVectorizer(node, tensorMap, nodeIndex, importedOpset, scope);
  if (node.opType === "ArrayFeatureExtractor") return inferArrayFeatureExtractor(node, tensorMap, nodeIndex, importedOpset, scope);
  return null;
}

function schemaAttributes(opName, schemaSinceVersion) {
  if (opName === "LabelEncoder") return LABEL_ENCODER_SCHEMAS.get(schemaSinceVersion) || new Map();
  if (opName === "TreeEnsembleClassifier") return schemaSinceVersion === 1 ? TREE_CLASSIFIER_V1_SCHEMA : TREE_CLASSIFIER_V3_SCHEMA;
  if (opName === "TreeEnsembleRegressor") return schemaSinceVersion === 1 ? TREE_REGRESSOR_V1_SCHEMA : TREE_REGRESSOR_V3_SCHEMA;
  return ML_SCHEMAS.get(opName) || new Map();
}

function inferBinarizer(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("binarizer_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`binarizer_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  const allowedDtypes = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64"]);
  if (inputDtype === "UNKNOWN") reasons.push("binarizer_input_dtype_unresolved");
  else if (!allowedDtypes.has(inputDtype)) failures.push(`binarizer_input_dtype_not_supported:${inputDtype}`);

  const thresholdAttribute = node.attributes?.get("threshold");
  const explicitThreshold = thresholdAttribute ? floatScalarAttribute(thresholdAttribute) : null;
  if (thresholdAttribute && explicitThreshold == null) failures.push("binarizer_threshold_not_float_scalar");
  const threshold = explicitThreshold ?? 0;
  const thresholdSource = thresholdAttribute ? "explicit_attribute" : "onnx_schema_default_0";
  const inputShape = inputType?.kind === "tensor" && inputType.shapeDeclared === true ? [...inputType.shape] : [];
  const outputShapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  if (!outputShapeDeclared) reasons.push("binarizer_input_shape_unresolved");
  else if (inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("binarizer_dynamic_shape_preserved");
  const outputType = inputDtype === "UNKNOWN" ? null : makeOnnxTensorType(inputDtype, inputShape, outputShapeDeclared);
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = outputShapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  if (outputShapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("binarizer_output_element_count_overflow");

  const staticResult = exactBinarizerResult(input, inputDtype, threshold);
  if (staticResult.values && patch) {
    patch.staticValuesStatus = "complete";
    patch.staticValuesComplete = true;
    patch.staticValues = staticResult.values;
    patch.staticValuesSource = "binarizer_exact_initializer_evaluation";
  }
  const riskCodes = [];
  if (!Number.isFinite(threshold)) riskCodes.push("binarizer_non_finite_threshold");
  if (inputDtype !== "UNKNOWN" && allowedDtypes.has(inputDtype) && inputDtype !== "FLOAT32") {
    riskCodes.push("binarizer_dtype_unsupported_by_pinned_ort_cpu");
  }
  if (input?.staticValuesStatus === "not_assessed_non_finite_or_unsafe_value" && ["FLOAT32", "FLOAT64"].includes(inputDtype)) {
    riskCodes.push("binarizer_static_input_contains_non_finite_or_unsafe_value");
  }
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "Binarizer", contract_kind: "tensor_threshold",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: outputShapeDeclared ? inputShape.length : null, input_shape: inputShape,
    exact_batch_count: null, exact_feature_count: null,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: inputDtype,
    exact_output_rank: outputShapeDeclared ? inputShape.length : null, exact_output_shape: inputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_same_type_same_shape_propagation",
    runtime_reference_status: "pinned_ort_cpu_float32_kernel_only",
    attribute_mode: "threshold",
    threshold_value: Number.isFinite(threshold) ? threshold : null,
    threshold_value_text: canonicalFloatText(threshold),
    threshold_source: thresholdSource,
    threshold_finite: Number.isFinite(threshold),
    static_value_assessment_status: staticResult.status,
    exact_static_input_value_count: staticResult.values?.length ?? null,
    exact_above_threshold_count: staticResult.aboveCount,
    exact_at_or_below_threshold_count: staticResult.atOrBelowCount,
    exact_equal_threshold_count: staticResult.equalCount,
    exact_output_zero_count: staticResult.atOrBelowCount,
    exact_output_one_count: staticResult.aboveCount,
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null,
    exact_truncated_feature_count_per_batch: null, padded_input_count: 0, truncated_input_count: 0,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_values: [], exact_index_preview: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function exactBinarizerResult(input, dtype, threshold) {
  let source = null;
  if (dtype === "INT64") source = exactIntegerTensorValues(input);
  else if (["FLOAT32", "FLOAT64", "INT32"].includes(dtype)
    && input?.staticValuesComplete === true && Array.isArray(input.staticValues)) source = input.staticValues;
  if (!source) {
    return {
      status: input?.role === "initializer" ? input.staticValuesStatus || "not_assessed_initializer_values" : "not_assessed_runtime_values",
      values: null, aboveCount: null, atOrBelowCount: null, equalCount: null,
    };
  }
  let aboveCount = 0;
  let equalCount = 0;
  const values = source.map((value) => {
    const above = typeof value === "bigint" ? integerGreaterThanFloat(value, threshold) : value > threshold;
    const equal = typeof value === "bigint" ? integerEqualsFloat(value, threshold) : value === threshold;
    if (above) aboveCount += 1;
    if (equal) equalCount += 1;
    return above ? 1 : 0;
  });
  return {
    status: "assessed_exact",
    values,
    aboveCount,
    atOrBelowCount: values.length - aboveCount,
    equalCount,
  };
}

function integerGreaterThanFloat(value, threshold) {
  if (Number.isNaN(threshold) || threshold === Number.POSITIVE_INFINITY) return false;
  if (threshold === Number.NEGATIVE_INFINITY) return true;
  return value > BigInt(Math.floor(threshold));
}

function integerEqualsFloat(value, threshold) {
  return Number.isFinite(threshold) && Number.isInteger(threshold) && value === BigInt(threshold);
}

function inferZipMap(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (inputType?.kind !== "tensor") failures.push("zip_map_input_not_tensor");
  const inputDtype = inputType?.dtype || inputType?.elementTypeName || "UNKNOWN";
  if (inputDtype !== "UNKNOWN" && inputDtype !== "FLOAT32") failures.push(`zip_map_input_dtype_not_float:${inputDtype}`);
  if (inputDtype === "UNKNOWN") reasons.push("zip_map_input_dtype_unresolved");

  const stringAttribute = node.attributes?.get("classlabels_strings");
  const integerAttribute = node.attributes?.get("classlabels_int64s");
  const stringKeys = stringAttribute ? [...(stringAttribute.strings || [])] : null;
  const integerKeys = integerAttribute ? exactIntegerAttributeValues(integerAttribute) : null;
  const stringKeysPresent = Array.isArray(stringKeys) && stringKeys.length > 0;
  const integerKeysPresent = Array.isArray(integerKeys) && integerKeys.length > 0;
  if (stringKeysPresent === integerKeysPresent) failures.push("zip_map_requires_exactly_one_nonempty_classlabels_attribute");

  const keyType = stringKeysPresent ? "STRING" : integerKeysPresent ? "INT64" : "UNDEFINED";
  const keys = stringKeysPresent ? stringKeys : integerKeysPresent ? integerKeys : [];
  const duplicateKeyCount = keys.length - new Set(keys).size;
  const riskCodes = duplicateKeyCount > 0 ? ["zip_map_duplicate_class_keys_information_loss_risk"] : [];

  let inputRank = null;
  let batchCount = null;
  let featureCount = null;
  if (inputType?.shapeDeclared === true) {
    inputRank = inputType.shape.length;
    if (![1, 2].includes(inputRank)) failures.push(`zip_map_input_rank_not_one_or_two:${inputRank}`);
    if (inputRank === 1) {
      batchCount = 1;
      featureCount = knownDimension(inputType.shape[0]) ? Number(inputType.shape[0]) : null;
    } else if (inputRank === 2) {
      batchCount = knownDimension(inputType.shape[0]) ? Number(inputType.shape[0]) : null;
      featureCount = knownDimension(inputType.shape[1]) ? Number(inputType.shape[1]) : null;
    }
  } else reasons.push("zip_map_input_rank_unresolved");
  if ([1, 2].includes(inputRank) && batchCount == null) reasons.push("zip_map_batch_dimension_runtime_unknown");
  if ([1, 2].includes(inputRank) && featureCount == null) reasons.push("zip_map_feature_dimension_runtime_unknown");
  if (featureCount != null && keys.length && featureCount !== keys.length) {
    failures.push(`zip_map_feature_count_key_count_mismatch:${featureCount}:${keys.length}`);
  }

  // The pinned TypeAndShapeInferenceFunction initializes an empty
  // TensorShapeProto, which is an explicit rank-0 scalar rather than unknown rank.
  const valueType = makeOnnxTensorType("FLOAT32", [], true);
  const mapType = makeOnnxMapType(keyType, valueType);
  const outputType = makeOnnxSequenceType(mapType);
  const inventory = batchCount != null && batchCount <= MAX_EXACT_SEQUENCE_INVENTORY
    ? Array.from({ length: batchCount }, () => cloneOnnxTypeProto(mapType)) : null;
  if (batchCount != null && !inventory) reasons.push("zip_map_sequence_inventory_not_materialized_over_limit");
  const state = {
    sequenceLengthStatus: batchCount == null ? "not_assessed_runtime_length" : "assessed_exact",
    sequenceLength: batchCount,
    sequenceElementInventoryStatus: inventory ? "assessed_exact" : "not_assessed",
    sequenceElementTypes: inventory || [],
    mapKeyType: keyType,
    mapKeyCount: keys.length,
    mapDuplicateKeyCount: duplicateKeyCount,
  };
  const patch = keyType === "UNDEFINED" ? null : onnxValueDescriptorFromType(outputType, state);
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope,
    node_index: nodeIndex,
    op_name: "ZipMap",
    contract_kind: "map_producer",
    imported_opset: importedOpset,
    status,
    input_name: node.inputs?.[0] || "",
    output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype,
    input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null,
    input_map_value_dtype: null,
    exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: inputRank,
    input_shape: inputType?.shapeDeclared === true ? [...inputType.shape] : [],
    exact_batch_count: batchCount,
    exact_feature_count: featureCount,
    class_key_type: keyType,
    class_key_count: keys.length,
    duplicate_key_count: duplicateKeyCount,
    class_key_preview: keys.slice(0, 8),
    exact_output_sequence_length: batchCount,
    canonical_output_type: keyType === "UNDEFINED" ? "unresolved" : canonicalOnnxTypeProto(outputType),
    output_kind: "sequence",
    output_dtype: "UNKNOWN",
    exact_output_rank: null,
    exact_output_shape: [],
    exact_dense_output_element_count: null,
    output_shape_basis: "pinned_onnx_schema_and_ort_cpu_batch_semantics",
    runtime_reference_status: "pinned_ort_cpu_implementation",
    attribute_mode: "class_labels",
    vocabulary_type: "UNDEFINED",
    vocabulary_count: 0,
    duplicate_vocabulary_count: 0,
    vocabulary_preview: [],
    reason_codes: [...new Set([...failures, ...reasons])],
    risk_codes: riskCodes,
  };
  return {
    status,
    reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] },
    row,
  };
}

function inferCastMap(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("cast_map_input_type_unresolved");
  else if (inputType.kind !== "map") failures.push(`cast_map_input_not_map:${inputType.kind}`);
  const mapKeyType = inputType?.kind === "map" ? inputType.keyTypeName || "UNDEFINED" : "UNDEFINED";
  const mapValueDtype = inputType?.kind === "map" ? tensorElementDtype(inputType.valueType) : "UNKNOWN";
  if (mapKeyType !== "UNDEFINED" && mapKeyType !== "INT64") failures.push(`cast_map_key_type_not_int64:${mapKeyType}`);
  if (mapValueDtype !== "UNKNOWN" && !["FLOAT32", "STRING"].includes(mapValueDtype)) failures.push(`cast_map_value_type_not_float_or_string:${mapValueDtype}`);
  if (mapKeyType === "UNDEFINED") reasons.push("cast_map_key_type_unresolved");
  if (mapValueDtype === "UNKNOWN") reasons.push("cast_map_value_type_unresolved");

  const castTo = stringScalarAttribute(node.attributes?.get("cast_to")) ?? "TO_FLOAT";
  const mapForm = stringScalarAttribute(node.attributes?.get("map_form")) ?? "DENSE";
  const outputDtype = { TO_FLOAT: "FLOAT32", TO_STRING: "STRING", TO_INT64: "INT64" }[castTo] || "UNKNOWN";
  if (outputDtype === "UNKNOWN") failures.push(`cast_map_cast_to_invalid:${castTo}`);
  if (!["DENSE", "SPARSE"].includes(mapForm)) failures.push(`cast_map_form_invalid:${mapForm}`);

  const explicitMaxMap = node.attributes?.has("max_map") ? exactScalarIntegerAttribute(node.attributes.get("max_map")) : 1;
  if (mapForm === "SPARSE" && explicitMaxMap == null) reasons.push("cast_map_max_map_not_safe_integer");
  if (mapForm === "SPARSE" && explicitMaxMap != null && explicitMaxMap < 0) failures.push(`cast_map_max_map_negative:${explicitMaxMap}`);
  const exactMapKeys = exactIntegerMapKeys(input?.mapKeysExact);
  const denseLength = Number.isSafeInteger(input?.mapKeyCount) && input.mapKeyCount >= 0
    ? input.mapKeyCount : exactMapKeys ? exactMapKeys.length : null;
  let sparseKeyBoundsStatus = "not_applicable";
  if (mapForm === "SPARSE" && explicitMaxMap != null && explicitMaxMap >= 0) {
    if (!exactMapKeys) {
      sparseKeyBoundsStatus = "not_assessed_runtime_keys";
      reasons.push("cast_map_sparse_key_bounds_runtime_unknown");
    } else {
      const upperBound = BigInt(explicitMaxMap);
      const invalidKeys = exactMapKeys.filter((value) => value < 0n || value >= upperBound);
      sparseKeyBoundsStatus = invalidKeys.length ? "fail" : "assessed_pass";
      if (invalidKeys.length) failures.push(`cast_map_sparse_key_out_of_bounds:${invalidKeys.length}`);
    }
  } else if (mapForm === "SPARSE") {
    sparseKeyBoundsStatus = "not_assessed_invalid_or_unresolved_max_map";
  }
  const outputLength = mapForm === "SPARSE" ? explicitMaxMap : denseLength;
  if (mapForm === "DENSE" && outputLength == null) reasons.push("cast_map_dense_length_runtime_unknown");
  const outputShape = [Number.isSafeInteger(outputLength) && outputLength >= 0 ? outputLength : -1];
  const outputType = outputDtype === "UNKNOWN" ? null : makeOnnxTensorType(outputDtype, outputShape, true);
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "CastMap", contract_kind: "map_consumer",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: "UNKNOWN", input_kind: inputType?.kind || "unresolved",
    input_map_key_type: mapKeyType, input_map_value_dtype: mapValueDtype,
    exact_input_map_key_count: denseLength,
    sparse_key_bounds_status: sparseKeyBoundsStatus,
    input_rank: null, input_shape: [], exact_batch_count: null, exact_feature_count: null,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: 1, exact_output_shape: outputShape,
    exact_dense_output_element_count: Number.isSafeInteger(outputLength) && outputLength >= 0 ? outputLength : null,
    output_shape_basis: mapForm === "SPARSE"
      ? "pinned_onnx_schema_sparse_max_map"
      : denseLength == null ? "rank_only_length_runtime_unknown" : "artifact_exact_input_map_cardinality",
    runtime_reference_status: "onnx_schema_only_no_pinned_ort_cpu_kernel",
    attribute_mode: mapForm, cast_to: castTo, map_form: mapForm,
    max_map: explicitMaxMap,
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: [],
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function inferDictVectorizer(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("dict_vectorizer_input_type_unresolved");
  else if (inputType.kind !== "map") failures.push(`dict_vectorizer_input_not_map:${inputType.kind}`);
  const mapKeyType = inputType?.kind === "map" ? inputType.keyTypeName || "UNDEFINED" : "UNDEFINED";
  const mapValueDtype = inputType?.kind === "map" ? tensorElementDtype(inputType.valueType) : "UNKNOWN";

  const stringAttribute = node.attributes?.get("string_vocabulary");
  const integerAttribute = node.attributes?.get("int64_vocabulary");
  const stringVocabularyPresent = Boolean(stringAttribute);
  const integerVocabularyPresent = Boolean(integerAttribute);
  if (stringVocabularyPresent === integerVocabularyPresent) failures.push("dict_vectorizer_requires_exactly_one_vocabulary_attribute");
  const stringVocabulary = stringAttribute ? [...(stringAttribute.strings || [])] : [];
  const integerVocabulary = integerAttribute ? exactIntegerAttributeValues(integerAttribute) : [];
  if (integerVocabularyPresent && !Array.isArray(integerVocabulary)) failures.push("dict_vectorizer_int64_vocabulary_not_exactly_decoded");
  const vocabularyType = stringVocabularyPresent ? "STRING" : integerVocabularyPresent ? "INT64" : "UNDEFINED";
  const vocabulary = stringVocabularyPresent ? stringVocabulary : Array.isArray(integerVocabulary) ? integerVocabulary : [];
  const duplicateVocabularyCount = vocabulary.length - new Set(vocabulary).size;
  const riskCodes = duplicateVocabularyCount > 0 ? ["dict_vectorizer_duplicate_vocabulary_columns"] : [];

  if (mapKeyType === "UNDEFINED") reasons.push("dict_vectorizer_map_key_type_unresolved");
  else if (vocabularyType !== "UNDEFINED" && mapKeyType !== vocabularyType) failures.push(`dict_vectorizer_map_key_vocabulary_type_mismatch:${mapKeyType}:${vocabularyType}`);
  if (mapValueDtype === "UNKNOWN") reasons.push("dict_vectorizer_map_value_type_unresolved");
  const allowedMapType = mapKeyType === "STRING" && ["INT64", "FLOAT32", "FLOAT64"].includes(mapValueDtype)
    || mapKeyType === "INT64" && ["STRING", "FLOAT32", "FLOAT64"].includes(mapValueDtype);
  if (mapKeyType !== "UNDEFINED" && mapValueDtype !== "UNKNOWN" && !allowedMapType) {
    failures.push(`dict_vectorizer_map_type_not_allowed:${mapKeyType}:${mapValueDtype}`);
  }

  const outputType = mapValueDtype === "UNKNOWN" ? null : makeOnnxTensorType(mapValueDtype, [1, vocabulary.length], true);
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "DictVectorizer", contract_kind: "map_consumer",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: "UNKNOWN", input_kind: inputType?.kind || "unresolved",
    input_map_key_type: mapKeyType, input_map_value_dtype: mapValueDtype,
    exact_input_map_key_count: Number.isSafeInteger(input?.mapKeyCount) && input.mapKeyCount >= 0 ? input.mapKeyCount : null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: null, input_shape: [], exact_batch_count: null, exact_feature_count: vocabulary.length,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: mapValueDtype,
    exact_output_rank: 2, exact_output_shape: [1, vocabulary.length],
    exact_dense_output_element_count: vocabulary.length,
    output_shape_basis: "pinned_onnx_type_constraint_and_ort_cpu_vocabulary_size_allocation",
    runtime_reference_status: "pinned_ort_cpu_implementation",
    attribute_mode: "vocabulary",
    vocabulary_type: vocabularyType, vocabulary_count: vocabulary.length,
    duplicate_vocabulary_count: duplicateVocabularyCount, vocabulary_preview: vocabulary.slice(0, 8),
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function inferCategoryMapper(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const input = tensorMap.get(node.inputs?.[0]);
  const inputType = onnxTypeProtoFromValue(input);
  if (!inputType) reasons.push("category_mapper_input_type_unresolved");
  else if (inputType.kind !== "tensor") failures.push(`category_mapper_input_not_tensor:${inputType.kind}`);
  const inputDtype = inputType?.kind === "tensor" ? inputType.dtype || inputType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (inputDtype !== "UNKNOWN" && !["STRING", "INT64"].includes(inputDtype)) failures.push(`category_mapper_input_dtype_not_string_or_int64:${inputDtype}`);
  if (inputDtype === "UNKNOWN") reasons.push("category_mapper_input_dtype_unresolved");

  const stringAttribute = node.attributes?.get("cats_strings");
  const integerAttribute = node.attributes?.get("cats_int64s");
  if (!stringAttribute) failures.push("category_mapper_cats_strings_required");
  if (!integerAttribute) failures.push("category_mapper_cats_int64s_required");
  const stringCategories = stringAttribute ? [...(stringAttribute.strings || [])] : [];
  const integerCategories = integerAttribute ? exactIntegerAttributeValues(integerAttribute) : [];
  if (integerAttribute && !Array.isArray(integerCategories)) failures.push("category_mapper_cats_int64s_not_exactly_decoded");
  const exactIntegerCategories = Array.isArray(integerCategories) ? integerCategories : [];
  const categoryArraysAligned = Boolean(stringAttribute && integerAttribute && Array.isArray(integerCategories)
    && stringCategories.length === exactIntegerCategories.length);
  if (stringAttribute && integerAttribute && stringCategories.length !== exactIntegerCategories.length) {
    failures.push(`category_mapper_category_count_mismatch:${stringCategories.length}:${exactIntegerCategories.length}`);
  }

  const duplicateStringKeyCount = stringCategories.length - new Set(stringCategories).size;
  const duplicateInt64KeyCount = exactIntegerCategories.length - new Set(exactIntegerCategories).size;
  const mappingDirection = inputDtype === "STRING" ? "STRING_TO_INT64" : inputDtype === "INT64" ? "INT64_TO_STRING" : "UNRESOLVED";
  const activeDuplicateKeyCount = !categoryArraysAligned ? 0 : mappingDirection === "STRING_TO_INT64" ? duplicateStringKeyCount
    : mappingDirection === "INT64_TO_STRING" ? duplicateInt64KeyCount : 0;
  const riskCodes = activeDuplicateKeyCount > 0 ? ["category_mapper_duplicate_active_keys_last_write_wins"] : [];
  const defaultString = stringScalarAttribute(node.attributes?.get("default_string")) ?? "_Unused";
  const defaultInt64 = exactScalarIntegerDecimal(node.attributes?.get("default_int64")) ?? "-1";
  const activeDefaultType = mappingDirection === "STRING_TO_INT64" ? "INT64"
    : mappingDirection === "INT64_TO_STRING" ? "STRING" : "UNDEFINED";
  const activeDefaultValue = mappingDirection === "STRING_TO_INT64" ? defaultInt64
    : mappingDirection === "INT64_TO_STRING" ? defaultString : "";

  const outputDtype = mappingDirection === "STRING_TO_INT64" ? "INT64"
    : mappingDirection === "INT64_TO_STRING" ? "STRING" : "UNKNOWN";
  const inputShape = inputType?.kind === "tensor" && inputType.shapeDeclared === true ? [...inputType.shape] : [];
  const outputShapeDeclared = inputType?.kind === "tensor" && inputType.shapeDeclared === true;
  if (!outputShapeDeclared) reasons.push("category_mapper_input_shape_unresolved");
  else if (inputShape.some((dimension) => !knownDimension(dimension))) reasons.push("category_mapper_dynamic_shape_preserved");
  const outputType = outputDtype === "UNKNOWN" ? null : makeOnnxTensorType(outputDtype, inputShape, outputShapeDeclared);
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = outputShapeDeclared && inputShape.every(knownDimension) ? safeShapeElementCount(inputShape) : null;
  if (outputShapeDeclared && inputShape.every(knownDimension) && exactOutputElements == null) reasons.push("category_mapper_output_element_count_overflow");
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "CategoryMapper", contract_kind: "tensor_mapper",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_kind: inputType?.kind || "unresolved",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: outputShapeDeclared ? inputShape.length : null, input_shape: inputShape,
    exact_batch_count: null, exact_feature_count: null,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: outputDtype,
    exact_output_rank: outputShapeDeclared ? inputShape.length : null, exact_output_shape: inputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_shape_propagation_and_ort_cpu_same_shape_allocation",
    runtime_reference_status: "pinned_ort_cpu_implementation",
    attribute_mode: "bidirectional_categories",
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: mappingDirection,
    category_pair_count: categoryArraysAligned ? stringCategories.length : 0,
    category_string_count: stringCategories.length,
    category_int64_count: exactIntegerCategories.length,
    duplicate_string_key_count: duplicateStringKeyCount,
    duplicate_int64_key_count: duplicateInt64KeyCount,
    active_duplicate_key_count: activeDuplicateKeyCount,
    active_default_type: activeDefaultType,
    active_default_value: activeDefaultValue,
    category_string_preview: stringCategories.slice(0, 8),
    category_int64_preview: exactIntegerCategories.slice(0, 8),
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function inferFeatureVectorizer(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const inputNames = [...(node.inputs || [])];
  const inputs = inputNames.map((name) => tensorMap.get(name));
  const inputTypes = inputs.map(onnxTypeProtoFromValue);
  const allowedDtypes = new Set(["INT32", "INT64", "FLOAT32", "FLOAT64"]);
  const inputDtypes = inputTypes.map((type) => type?.kind === "tensor" ? type.dtype || type.elementTypeName || "UNKNOWN" : "UNKNOWN");
  const inputShapes = inputTypes.map((type) => type?.kind === "tensor" && type.shapeDeclared === true ? [...type.shape] : []);
  const inputRanks = inputTypes.map((type) => type?.kind === "tensor" && type.shapeDeclared === true ? type.shape.length : null);
  if (!inputs.length) failures.push("feature_vectorizer_requires_at_least_one_input");
  inputTypes.forEach((type, index) => {
    if (!type) reasons.push(`feature_vectorizer_input_type_unresolved:${index}`);
    else if (type.kind !== "tensor") failures.push(`feature_vectorizer_input_not_tensor:${index}:${type.kind}`);
    const dtype = inputDtypes[index];
    if (dtype === "UNKNOWN") reasons.push(`feature_vectorizer_input_dtype_unresolved:${index}`);
    else if (!allowedDtypes.has(dtype)) failures.push(`feature_vectorizer_input_dtype_not_supported:${index}:${dtype}`);
    const rank = inputRanks[index];
    if (rank == null) reasons.push(`feature_vectorizer_input_rank_unresolved:${index}`);
    else if (rank === 0) failures.push(`feature_vectorizer_scalar_input_not_supported:${index}`);
  });
  const knownDtypes = [...new Set(inputDtypes.filter((dtype) => dtype !== "UNKNOWN"))];
  if (knownDtypes.length > 1) failures.push(`feature_vectorizer_input_dtype_mismatch:${knownDtypes.join(":")}`);

  const dimensionsAttribute = node.attributes?.get("inputdimensions");
  if (!dimensionsAttribute) failures.push("feature_vectorizer_inputdimensions_required_by_pinned_ort");
  const configuredDimensionDecimals = dimensionsAttribute ? exactIntegerAttributeValues(dimensionsAttribute) : [];
  if (dimensionsAttribute && !Array.isArray(configuredDimensionDecimals)) failures.push("feature_vectorizer_inputdimensions_not_exactly_decoded");
  const exactConfiguredDimensions = Array.isArray(configuredDimensionDecimals) ? configuredDimensionDecimals : [];
  if (dimensionsAttribute && exactConfiguredDimensions.length === 0) failures.push("feature_vectorizer_inputdimensions_must_be_nonempty");
  if (dimensionsAttribute && exactConfiguredDimensions.length !== inputs.length) {
    failures.push(`feature_vectorizer_input_count_dimension_count_mismatch:${inputs.length}:${exactConfiguredDimensions.length}`);
  }
  const configuredBigInts = [];
  for (const value of exactConfiguredDimensions) {
    try {
      const parsed = BigInt(value);
      configuredBigInts.push(parsed);
      if (parsed < 0n) failures.push(`feature_vectorizer_negative_inputdimension:${value}`);
    } catch {
      failures.push("feature_vectorizer_inputdimensions_not_exactly_decoded");
      break;
    }
  }
  const configuredDimensions = configuredBigInts.length === exactConfiguredDimensions.length
    ? configuredBigInts.map(safeNonnegativeBigIntNumber) : [];
  if (configuredDimensions.some((value) => value == null) && !failures.some((reason) => reason.startsWith("feature_vectorizer_negative_inputdimension:"))) {
    reasons.push("feature_vectorizer_inputdimension_exceeds_safe_static_arithmetic");
  }
  const totalConfiguredBigInt = configuredBigInts.length === exactConfiguredDimensions.length
    ? configuredBigInts.reduce((sum, value) => sum + value, 0n) : null;
  const totalConfiguredFeatures = totalConfiguredBigInt == null ? null : safeNonnegativeBigIntNumber(totalConfiguredBigInt);
  if (totalConfiguredBigInt != null && totalConfiguredFeatures == null && totalConfiguredBigInt >= 0n) {
    reasons.push("feature_vectorizer_total_dimension_exceeds_safe_static_arithmetic");
  }

  const batchCounts = inputRanks.map((rank, index) => {
    if (rank == null || rank === 0) return null;
    if (rank === 1) return 1;
    return knownDimension(inputShapes[index][0]) ? Number(inputShapes[index][0]) : null;
  });
  const exactBatchCount = batchCounts[0] ?? null;
  if (batchCounts.some((value) => value == null)) reasons.push("feature_vectorizer_batch_consistency_runtime_unknown");
  const knownBatchCounts = [...new Set(batchCounts.filter((value) => value != null))];
  if (knownBatchCounts.length > 1) failures.push(`feature_vectorizer_batch_size_mismatch:${knownBatchCounts.join(":")}`);

  const rowFeatureCounts = inputRanks.map((rank, index) => {
    if (rank == null || rank === 0) return null;
    const dimensions = rank === 1 ? inputShapes[index] : inputShapes[index].slice(1);
    return dimensions.every(knownDimension) ? safeShapeElementCount(dimensions) : null;
  });
  if (rowFeatureCounts.some((value) => value == null)) reasons.push("feature_vectorizer_input_row_width_runtime_unknown");
  const copiedByInput = rowFeatureCounts.map((actual, index) => actual == null || configuredDimensions[index] == null
    ? null : Math.min(actual, configuredDimensions[index]));
  const paddedByInput = rowFeatureCounts.map((actual, index) => actual == null || configuredDimensions[index] == null
    ? null : Math.max(0, configuredDimensions[index] - actual));
  const truncatedByInput = rowFeatureCounts.map((actual, index) => actual == null || configuredDimensions[index] == null
    ? null : Math.max(0, actual - configuredDimensions[index]));
  const exactCopiedFeaturesPerBatch = sumExactNonnegative(copiedByInput);
  const exactPaddedFeaturesPerBatch = sumExactNonnegative(paddedByInput);
  const exactTruncatedFeaturesPerBatch = sumExactNonnegative(truncatedByInput);
  const riskCodes = exactTruncatedFeaturesPerBatch > 0 ? ["feature_vectorizer_truncates_input_features"] : [];

  const outputShape = [exactBatchCount ?? -1, totalConfiguredFeatures ?? -1];
  const outputType = makeOnnxTensorType("FLOAT32", outputShape, true);
  const patch = onnxValueDescriptorFromType(outputType);
  const exactOutputElements = outputShape.every(knownDimension) ? safeShapeElementCount(outputShape) : null;
  if (outputShape.every(knownDimension) && exactOutputElements == null) reasons.push("feature_vectorizer_output_element_count_overflow");
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const inputDtype = knownDtypes.length === 1 ? knownDtypes[0] : knownDtypes.length > 1 ? "MIXED" : "UNKNOWN";
  const row = {
    scope, node_index: nodeIndex, op_name: "FeatureVectorizer", contract_kind: "tensor_aggregator",
    imported_opset: importedOpset, status,
    input_name: inputNames[0] || "", input_names: inputNames, output_name: node.outputs?.[0] || "",
    input_dtype: inputDtype, input_dtypes: inputDtypes, input_kind: "tensor_list",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: inputRanks[0] ?? null, input_ranks: inputRanks,
    input_shape: inputShapes[0] || [], input_shapes: inputShapes,
    exact_batch_count: exactBatchCount, exact_batch_counts: batchCounts,
    exact_feature_count: totalConfiguredFeatures,
    exact_input_row_feature_counts: rowFeatureCounts,
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: canonicalOnnxTypeProto(outputType),
    output_kind: "tensor", output_dtype: "FLOAT32", exact_output_rank: 2, exact_output_shape: outputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_ort_cpu_feature_dimension_allocation",
    runtime_reference_status: "pinned_ort_cpu_implementation",
    attribute_mode: "input_dimensions",
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: exactConfiguredDimensions,
    configured_feature_dimension_count: exactConfiguredDimensions.length,
    total_configured_feature_count: totalConfiguredFeatures,
    copied_feature_counts_per_input: copiedByInput,
    padded_feature_counts_per_input: paddedByInput,
    truncated_feature_counts_per_input: truncatedByInput,
    exact_copied_feature_count_per_batch: exactCopiedFeaturesPerBatch,
    exact_padded_feature_count_per_batch: exactPaddedFeaturesPerBatch,
    exact_truncated_feature_count_per_batch: exactTruncatedFeaturesPerBatch,
    padded_input_count: paddedByInput.filter((value) => value > 0).length,
    truncated_input_count: truncatedByInput.filter((value) => value > 0).length,
    index_input_name: "", index_input_dtype: "UNKNOWN", index_input_rank: null, index_input_shape: [],
    exact_index_count: null, exact_index_values_status: "not_applicable", exact_index_preview: [],
    exact_index_values: [],
    duplicate_index_count: 0, index_bounds_status: "not_applicable", out_of_bounds_index_count: 0,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: riskCodes,
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function inferArrayFeatureExtractor(node, tensorMap, nodeIndex, importedOpset, scope) {
  const reasons = [];
  const failures = [];
  const dataInput = tensorMap.get(node.inputs?.[0]);
  const indexInput = tensorMap.get(node.inputs?.[1]);
  const dataType = onnxTypeProtoFromValue(dataInput);
  const indexType = onnxTypeProtoFromValue(indexInput);
  const allowedDataDtypes = new Set(["FLOAT32", "FLOAT64", "INT32", "INT64", "STRING"]);
  if (!dataType) reasons.push("array_feature_extractor_data_type_unresolved");
  else if (dataType.kind !== "tensor") failures.push(`array_feature_extractor_data_not_tensor:${dataType.kind}`);
  if (!indexType) reasons.push("array_feature_extractor_index_type_unresolved");
  else if (indexType.kind !== "tensor") failures.push(`array_feature_extractor_indices_not_tensor:${indexType.kind}`);
  const dataDtype = dataType?.kind === "tensor" ? dataType.dtype || dataType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  const indexDtype = indexType?.kind === "tensor" ? indexType.dtype || indexType.elementTypeName || "UNKNOWN" : "UNKNOWN";
  if (dataDtype === "UNKNOWN") reasons.push("array_feature_extractor_data_dtype_unresolved");
  else if (!allowedDataDtypes.has(dataDtype)) failures.push(`array_feature_extractor_data_dtype_not_supported:${dataDtype}`);
  if (indexDtype === "UNKNOWN") reasons.push("array_feature_extractor_index_dtype_unresolved");
  else if (indexDtype !== "INT64") failures.push(`array_feature_extractor_index_dtype_not_int64:${indexDtype}`);

  const dataShapeDeclared = dataType?.kind === "tensor" && dataType.shapeDeclared === true;
  const indexShapeDeclared = indexType?.kind === "tensor" && indexType.shapeDeclared === true;
  const dataShape = dataShapeDeclared ? [...dataType.shape] : [];
  const indexShape = indexShapeDeclared ? [...indexType.shape] : [];
  const dataRank = dataShapeDeclared ? dataShape.length : null;
  const indexRank = indexShapeDeclared ? indexShape.length : null;
  if (dataRank == null) reasons.push("array_feature_extractor_data_rank_unresolved");
  else if (dataRank === 0) failures.push("array_feature_extractor_scalar_data_not_supported");
  const exactIndexCount = indexShapeDeclared && indexShape.every(knownDimension) ? safeShapeElementCount(indexShape) : null;
  if (!indexShapeDeclared || indexShape.some((dimension) => !knownDimension(dimension))) {
    reasons.push("array_feature_extractor_index_count_runtime_unknown");
  } else if (exactIndexCount == null) reasons.push("array_feature_extractor_index_count_overflow");
  else if (exactIndexCount === 0) failures.push("array_feature_extractor_empty_indices_not_supported");

  const exactIndexValues = exactIntegerTensorValues(indexInput);
  if (exactIndexValues && exactIndexCount != null && exactIndexValues.length !== exactIndexCount) {
    failures.push(`array_feature_extractor_index_value_count_mismatch:${exactIndexValues.length}:${exactIndexCount}`);
  }
  const exactIndexPreview = exactIndexValues ? exactIndexValues.slice(0, 16).map((value) => value.toString()) : [];
  const duplicateIndexCount = exactIndexValues ? exactIndexValues.length - new Set(exactIndexValues.map(String)).size : 0;
  const lastAxisSize = dataRank != null && dataRank > 0 && knownDimension(dataShape.at(-1)) ? Number(dataShape.at(-1)) : null;
  let outOfBoundsIndexCount = 0;
  let indexBoundsStatus = "not_assessed_runtime_values";
  if (exactIndexValues) {
    const upperBound = lastAxisSize == null ? null : BigInt(lastAxisSize);
    const invalid = exactIndexValues.filter((value) => value < 0n || upperBound != null && value >= upperBound);
    outOfBoundsIndexCount = invalid.length;
    if (invalid.length) {
      failures.push(`array_feature_extractor_index_out_of_bounds:${invalid.length}`);
      indexBoundsStatus = "fail";
    } else if (upperBound == null) {
      reasons.push("array_feature_extractor_last_axis_bound_runtime_unknown");
      indexBoundsStatus = "not_assessed_dynamic_axis";
    } else indexBoundsStatus = "assessed_pass";
  } else {
    if (exactIndexCount != null && exactIndexCount > 0 && lastAxisSize === 0) {
      failures.push(`array_feature_extractor_index_out_of_bounds:${exactIndexCount}`);
      outOfBoundsIndexCount = exactIndexCount;
      indexBoundsStatus = "fail";
    } else if (exactIndexCount == null) indexBoundsStatus = "not_assessed_dynamic_cardinality";
    if (!failures.some((reason) => reason.startsWith("array_feature_extractor_index_out_of_bounds:"))) {
      reasons.push("array_feature_extractor_index_values_runtime_unknown");
    }
  }

  let outputShapeDeclared = false;
  let outputShape = [];
  if (dataRank != null && dataRank > 0) {
    outputShapeDeclared = true;
    outputShape = dataRank === 1
      ? [1, exactIndexCount ?? -1]
      : [...dataShape.slice(0, -1), exactIndexCount ?? -1];
  }
  const outputType = dataDtype === "UNKNOWN" ? null : makeOnnxTensorType(dataDtype, outputShape, outputShapeDeclared);
  const patch = outputType ? onnxValueDescriptorFromType(outputType) : null;
  const exactOutputElements = outputShapeDeclared && outputShape.every(knownDimension) ? safeShapeElementCount(outputShape) : null;
  if (outputShapeDeclared && outputShape.every(knownDimension) && exactOutputElements == null) {
    reasons.push("array_feature_extractor_output_element_count_overflow");
  }
  const status = failures.length ? "fail" : reasons.length ? "partial" : "pass";
  const row = {
    scope, node_index: nodeIndex, op_name: "ArrayFeatureExtractor", contract_kind: "tensor_selector",
    imported_opset: importedOpset, status,
    input_name: node.inputs?.[0] || "", input_names: [...(node.inputs || [])], output_name: node.outputs?.[0] || "",
    input_dtype: dataDtype, input_dtypes: [dataDtype, indexDtype], input_kind: "tensor",
    input_map_key_type: null, input_map_value_dtype: null, exact_input_map_key_count: null,
    sparse_key_bounds_status: "not_applicable",
    input_rank: dataRank, input_ranks: [dataRank, indexRank], input_shape: dataShape, input_shapes: [dataShape, indexShape],
    exact_batch_count: null, exact_batch_counts: [], exact_feature_count: exactIndexCount,
    exact_input_row_feature_counts: [],
    class_key_type: "UNDEFINED", class_key_count: 0, duplicate_key_count: 0, class_key_preview: [],
    exact_output_sequence_length: null,
    canonical_output_type: outputType ? canonicalOnnxTypeProto(outputType) : "unresolved",
    output_kind: "tensor", output_dtype: dataDtype,
    exact_output_rank: outputShapeDeclared ? outputShape.length : null, exact_output_shape: outputShape,
    exact_dense_output_element_count: exactOutputElements,
    output_shape_basis: "pinned_onnx_last_axis_shape_rule_and_ort_cpu_rank1_compatibility",
    runtime_reference_status: "pinned_ort_cpu_implementation",
    attribute_mode: "last_axis_indices",
    vocabulary_type: "UNDEFINED", vocabulary_count: 0, duplicate_vocabulary_count: 0, vocabulary_preview: [],
    mapping_direction: "UNRESOLVED", category_pair_count: 0, category_string_count: 0, category_int64_count: 0,
    duplicate_string_key_count: 0, duplicate_int64_key_count: 0, active_duplicate_key_count: 0,
    active_default_type: "UNDEFINED", active_default_value: "", category_string_preview: [], category_int64_preview: [],
    configured_feature_dimensions: [], configured_feature_dimension_count: 0, total_configured_feature_count: null,
    copied_feature_counts_per_input: [], padded_feature_counts_per_input: [], truncated_feature_counts_per_input: [],
    exact_copied_feature_count_per_batch: null, exact_padded_feature_count_per_batch: null,
    exact_truncated_feature_count_per_batch: null, padded_input_count: 0, truncated_input_count: 0,
    index_input_name: node.inputs?.[1] || "", index_input_dtype: indexDtype,
    index_input_rank: indexRank, index_input_shape: indexShape,
    exact_index_count: exactIndexCount,
    exact_index_values_status: exactIndexValues ? "assessed_exact" : "not_assessed_runtime_values",
    exact_index_values: exactIndexValues ? exactIndexValues.map((value) => value.toString()) : [],
    exact_index_preview: exactIndexPreview,
    duplicate_index_count: duplicateIndexCount,
    index_bounds_status: indexBoundsStatus,
    out_of_bounds_index_count: outOfBoundsIndexCount,
    reason_codes: [...new Set([...failures, ...reasons])], risk_codes: [],
  };
  return {
    status, reason: row.reason_codes[0] || "",
    result: { outputs: status === "fail" || !patch || !node.outputs?.[0] ? [] : [[node.outputs[0], patch]] }, row,
  };
}

function exactIntegerAttributeValues(attribute) {
  const exact = Array.isArray(attribute?.intExactDecimals) ? attribute.intExactDecimals : [];
  if (exact.length === (attribute?.ints || []).length && exact.length > 0) {
    try {
      return exact.map((value) => BigInt(value).toString());
    } catch {
      return null;
    }
  }
  return (attribute?.ints || []).every(Number.isSafeInteger) ? attribute.ints.map((value) => String(value)) : null;
}

function exactScalarIntegerAttribute(attribute) {
  const exact = String(attribute?.iExactDecimal || "");
  if (exact) {
    try {
      const value = BigInt(exact);
      return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
    } catch {
      return null;
    }
  }
  return Number.isSafeInteger(attribute?.i) ? attribute.i : null;
}

function exactScalarIntegerDecimal(attribute) {
  if (!attribute) return null;
  const exact = String(attribute.iExactDecimal || "");
  if (exact) {
    try {
      return BigInt(exact).toString();
    } catch {
      return null;
    }
  }
  return Number.isSafeInteger(attribute.i) ? String(attribute.i) : null;
}

function safeShapeElementCount(shape) {
  let total = 1;
  for (const dimension of shape) {
    if (!knownDimension(dimension) || total > Math.floor(Number.MAX_SAFE_INTEGER / Math.max(1, dimension))) return null;
    total *= dimension;
  }
  return total;
}

function exactIntegerMapKeys(values) {
  if (!Array.isArray(values)) return null;
  try {
    return values.map((value) => {
      if (typeof value === "number" && !Number.isSafeInteger(value)) throw new Error("unsafe integer");
      return BigInt(value);
    });
  } catch {
    return null;
  }
}

function exactIntegerTensorValues(tensor) {
  const exact = tensor?.initializerIntegerValuesExactComplete === true
    && Array.isArray(tensor.initializerIntegerValuesExactDecimals)
    ? tensor.initializerIntegerValuesExactDecimals : null;
  const fallback = tensor?.initializerIntegerValuesComplete === true && Array.isArray(tensor.initializerIntegerValues)
    ? tensor.initializerIntegerValues
    : tensor?.staticValuesComplete === true && Array.isArray(tensor.staticValues)
      && tensor.staticValues.every(Number.isSafeInteger) ? tensor.staticValues : null;
  const source = exact || fallback;
  if (!source) return null;
  try {
    return source.map((value) => BigInt(value));
  } catch {
    return null;
  }
}

function safeNonnegativeBigIntNumber(value) {
  return value >= 0n && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
}

function sumExactNonnegative(values) {
  if (!Array.isArray(values) || values.some((value) => !Number.isSafeInteger(value) || value < 0)) return null;
  let total = 0;
  for (const value of values) {
    if (total > Number.MAX_SAFE_INTEGER - value) return null;
    total += value;
  }
  return total;
}

function stringScalarAttribute(attribute) {
  return attribute && typeof attribute.s === "string" ? attribute.s : null;
}

function floatScalarAttribute(attribute) {
  return attribute && typeof attribute.f === "number" ? attribute.f : null;
}

function canonicalFloatText(value) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0";
  return String(value);
}

function tensorElementDtype(type) {
  return type?.kind === "tensor" ? type.dtype || type.elementTypeName || "UNKNOWN" : "UNKNOWN";
}

function knownDimension(value) {
  return Number.isSafeInteger(Number(value)) && Number(value) >= 0;
}

function normalizeDomain(domain) {
  return !domain || domain === "ai.onnx" ? "ai.onnx" : String(domain);
}
