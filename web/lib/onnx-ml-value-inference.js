import { onnxMlSemantic, onnxSemantic, onnxNodeRecord } from './onnx-portable.js';
import "./onnx-static-value-evidence.js";
import "./onnx-schema-legality.js";
import "./onnx-type-proto.js";
import { inferOnnxMlNormalizer } from "./onnx-ml-normalizer.js";
import { inferOnnxMlScaler } from "./onnx-ml-scaler.js";
import { inferOnnxMlImputer } from "./onnx-ml-imputer.js";
import { inferOnnxMlOneHotEncoder } from "./onnx-ml-one-hot-encoder.js";
import { inferOnnxMlLinearClassifier, inferOnnxMlLinearRegressor } from "./onnx-ml-linear-model.js";
import { inferOnnxMlLabelEncoder } from "./onnx-ml-label-encoder.js";
import { inferOnnxMlSvmClassifier, inferOnnxMlSvmRegressor } from "./onnx-ml-svm.js";
import { inferOnnxMlTreeEnsemble, inferOnnxMlTreeEnsembleClassifier, inferOnnxMlTreeEnsembleRegressor } from "./onnx-ml-tree-ensemble.js";

const SOURCE_COMMIT = "be2b5fde82d9c8874f3d19328bdfe3b6962dc67b";
const ORT_SOURCE_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277";
const MAX_EXACT_SEQUENCE_INVENTORY = 4_096;
export const ONNX_ML_VALUE_OPS = new Set(["Binarizer", "Normalizer", "Scaler", "Imputer", "OneHotEncoder", "LinearClassifier", "LinearRegressor", "LabelEncoder", "SVMClassifier", "SVMRegressor", "TreeEnsemble", "TreeEnsembleClassifier", "TreeEnsembleRegressor", "ZipMap", "CastMap", "DictVectorizer", "CategoryMapper", "FeatureVectorizer", "ArrayFeatureExtractor"]);

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

export function canInferOnnxMlNode(node){return onnxSemantic('ml_can',{node:onnxNodeRecord(node)});}

export function assessOnnxMlNodeSchemaForm(node,importedOpset){return onnxSemantic('ml_schema',{node:onnxNodeRecord(node),imported_opset:importedOpset});}

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

function inferBinarizer(node, tensorMap, nodeIndex, importedOpset, scope) {
 return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});
}

function inferZipMap(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}

function inferCastMap(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}

function inferDictVectorizer(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}

function inferCategoryMapper(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}

function inferFeatureVectorizer(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}

function inferArrayFeatureExtractor(node,tensorMap,nodeIndex,importedOpset,scope){return onnxMlSemantic({node,tensorMap,nodeIndex,importedOpset,scope});}
