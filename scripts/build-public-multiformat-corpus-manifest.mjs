import { gunzipSync } from "node:zlib";
import { readFile, writeFile } from "node:fs/promises";

const OUTPUT = "corpus/public-multiformat-corpus.v1.json";
const HF_SNAPSHOT = "corpus/huggingface-community-corpus.v1.json.gz";
const GGUF_ANCHORS = "corpus/gguf-architecture-encoding-corpus.v1.json";
const MAX_BOUND_ONNX_SIDECAR_BYTES = 100_000_000;

const onnxSelections = [
  ["onnx-community/perth-watermarker-ONNX", "onnx/implicit_watermarker_16000.onnx"],
  ["onnx-community/perth-watermarker-ONNX", "onnx/implicit_watermarker_24000.onnx"],
  ["onnx-community/perth-watermarker-ONNX", "onnx/implicit_watermarker_44100.onnx"],
  ["onnx-community/perth-watermarker-ONNX", "onnx/implicit_watermarker_48000.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model_bnb4.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model_fp16.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model_int8.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model_q4.onnx"],
  ["onnx-community/whisper-tiny-ONNX", "onnx/encoder_model_q4f16.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model_uint8.onnx"],
  ["onnx-community/whisper-tiny", "onnx/encoder_model.onnx"],
  ["onnx-community/Kokoro-82M-ONNX", "onnx/model_quantized.onnx"],
  ["onnx-community/mdbr-leaf-ir-ONNX", "onnx/model_fp16.onnx"],
  ["onnx-community/ettin-decoder-32m-ONNX", "onnx/model_q4.onnx"],
  ["onnx-community/gpt2-ONNX-GQA", "onnx/model.onnx"],
  ["onnx-community/granite-embedding-30m-english-ONNX", "onnx/model_q4f16.onnx"],
  ["onnx-community/dinov3-convnext-tiny-pretrain-lvd1689m-ONNX", "onnx/model_quantized.onnx"],
  ["onnx-community/granite-docling-258M-ONNX", "onnx/vision_encoder_bnb4.onnx"],
  ["onnx-community/Voxtral-Mini-3B-2507-ONNX", "onnx/audio_encoder_uint8.onnx"],
  ["onnx-community/bert-base-cased-ONNX", "onnx/model.onnx"],
  ["onnx-community/depth-anything-v2-small-ONNX", "onnx/model.onnx"],
  ["onnx-community/LFM2-350M-ENJP-MT-ONNX", "onnx/model_fp16.onnx"],
  ["onnx-community/gemma-4-E2B-it-ONNX", "onnx/vision_encoder_fp16.onnx"],
  ["onnx-community/cohere-transcribe-03-2026-ONNX", "onnx/decoder_model_merged.onnx"],
  ["onnx-community/chatterbox-multilingual-ONNX", "onnx/language_model.onnx"],
  ["onnx-community/mediapipe_selfie_segmentation_landscape", "onnx/model_int8.onnx"],
  ["onnx-community/eu-pii-safeguard-ONNX", "onnx/model.onnx"],
  ["onnx-community/metaclip-2-worldwide-huge-378-ONNX", "onnx/vision_model.onnx"],
  ["onnx-community/vitpose-plus-huge-ONNX", "onnx/model.onnx"],
  ["onnx-community/yolov10n", "onnx/model_int8.onnx"],
  ["onnx-community/sam-vit-base-ONNX", "onnx/prompt_encoder_mask_decoder_q4f16.onnx"],
  ["onnx-community/mobilenetv4s-webnn", "onnx/model_int8.onnx"],
  ["onnx-community/BERT-tiny-RAID-ONNX", "onnx/model_int8.onnx"],
  ["onnx-community/smart-turn-v3-ONNX", "onnx/model_q4f16.onnx"],
  ["onnx-community/swin2SR-realworld-sr-x4-64-bsrgan-psnr-ONNX", "onnx/model_q4f16.onnx"],
  ["onnx-community/granite-timeseries-patchtsmixer", "onnx/model_bnb4.onnx"],
  ["onnx-community/xlm-roberta-large-qa-multilingual-finedtuned-ru-ONNX", "onnx/model.onnx"],
  ["onnx-community/Supertonic-TTS-2-ONNX", "onnx/text_encoder.onnx"],
  ["onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX", "cuda/cuda-int4-rtn-block-32/model.onnx"],
  ["onnx-community/ettin-encoder-32m-ONNX", "onnx/model_q4.onnx"],
  ["onnx-community/jina-embeddings-v5-omni-nano-ONNX", "onnx/text_model_fp16.onnx"],
  ["onnx-community/depth-anything-v2-base-ONNX", "onnx/model_fp16.onnx"],
  ["onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX", "onnx/model.onnx"],
  ["onnx-community/LFM2-VL-450M-ONNX", "onnx/decoder_model_merged_fp16.onnx"],
  ["onnx-community/Arch-Function-1.5B", "onnx/model.onnx"],
  ["onnx-community/Qwen3.5-0.8B-ONNX", "onnx/vision_encoder.onnx"],
  ["onnx-community/KittenTTS-Nano-v0.8-ONNX", "onnx/model.onnx"],
  ["onnx-community/EdgeTAM-ONNX", "onnx/vision_encoder_fp16.onnx"],
];

const safeTensorsSources = [
  safe("yujiepan/llama-2-tiny-random", "74bb065d6381fdf9fcde5896cb32267ba2c0ee0a", "llama", 647, "eac7ea3c1c3c0e3347f6b1fcbf9e7fa28e6e07cb6b447f4ea8d4afa5e422f1e6", 1027424, "5b933502925c364d8961d8eabc9b2992c6f131df36c972654c90682bc167d828"),
  safe("yujiepan/mistral-tiny-random", "cfcb3f89e224fddb33367793a307b4dee7339518", "mistral", 647, "e9b463e108ddb07232e50b1d59946473e15ac9674ee926476a4b222561af1280", 1030080, "6f69953dc8b7ca873d282f6e1687df3462145d899fd561a0716cfbae9b84a2f3"),
  safe("yujiepan/qwen1.5-tiny-random", "e8c0ac76247de71d5f1b4163b8cd4886a0ade766", "qwen2", 696, "ba98b241089fc874fa71274717b8a8c55416602947d6aeb51e50e7908e965515", 2436328, "c7e84dace652a2f283e69247a4c8ca5338f0793fae2ff7876f1e57cdf21a2f56"),
  safe("yujiepan/qwen3-tiny-random", "c5aa03d329380306bf4890b3715fecf0982efd52", "qwen3", 725, "42b104013fba6355bd032b92ac86d4a2c1b33f3942444015ad8c1fefa4a78bda", 19598752, "795fc013f2c5f6082d5c1719eb68c78e40d7e61cc660924802e926744232d5f4"),
  safe("yujiepan/gemma-tiny-random", "f76bdd6124cfa8b854979ee851bed8f708e1fee6", "gemma", 653, "9a594ee635e9da0b2ca3341da99cdbe22430b2043ba8c9a4469124f567322066", 4100464, "799285f7411c820e40eca071a2991a2a553c9931ebd854f35283fffbdf5c97c7"),
  safe("yujiepan/gemma-2-tiny-random", "1cb969aa15c4f64185596f3410c5aa55381b21f2", "gemma2", 907, "b1528b9df453a457cdb77cf17574f30c2d3392b8744b59e76284a0ed3ca808d4", 4129424, "b822fb4bd70ba7c075fd4ce0c4c16fa31946a1f52f95df3e6868c70fbf915f30"),
  safe("optimum-intel-internal-testing/tiny-random-olmo2", "bcbd58e6547eda039f5afd2426de37e33193a443", "olmo2", 617, "7967187d402640137d7914975a9c4914c6484e00eb311de09eeebd694b0879a5", 25775856, "217faebc00ab0271b2d7cb545dac16d7095c290534c9bf5fc957802d678a2d14"),
  safe("katuni4ka/tiny-random-granite", "705ce2cc1b5acc4a57f24bb6cea5cbec0089388c", "granite", 838, "a142199d402e01fd48924ab1918095efa039590dcd23d8664974b9045b7fbcad", 6668040, "3c416f19943f4d760680cf4921cb6c5a76016e48596cb4c911033bce010ac176"),
  safe("hf-tiny-v2/tiny-random-CohereForCausalLM", "eede22648ac3ab862cdef94b2880ebc7f40477ac", "cohere", 833, "a718b3a7fc33b5b9c1de78ce26d03a31a84882e47bc6ec4d5ffa6a29ed08e02e", 32707200, "f210308b3ffdda9d7a2930bb042251447cb256dcd8924cc82a48010270f7a52e"),
  safe("hf-tiny-v2/tiny-random-Cohere2ForCausalLM", "709c2ca452b7db08191229c9a6da993405f0b252", "cohere2", 926, "735071f9fc9bdd757f617aea60da64396408354f48cf1f4b2de434afbbd79f33", 32707200, "f2fb23e3df9e5f18bea2c43d86722a4e5626c63a1a552d07bd1fce4a710e8e9f"),
  safe("onnx-internal-testing/tiny-random-NemotronForCausalLM", "60d8a5d2d5e7e2b5a915d7b8f7babc0f256a5598", "nemotron", 685, "e094329ea0c83780610976c97f7fe22073332f374c1b862792e6307886af23c5", 65949416, "d0daa5424cea4fed98ed363f95f0ab7a6309a5db6ad889d17a772e81b444deb1"),
  safe("hf-tiny-v2/tiny-random-MinistralForTokenClassification", "b3977454729f79e079717534b29dc5a893db76f7", "ministral", 764, "664c00244cb1682eaa25f181018c6a48b9bd1c2781b545c412a0eac007f22e09", 73088, "1e60986313be1d308edad0f538d874eb0f542e904ec6138cce9729d62f234745"),
  safe("yujiepan/smollm3-tiny-random", "7397769fa1b4fb9d787bf51fcbc35946c0794324", "smollm3", 892, "ef529527f47ffda0c2c60ef2bdb1b5016d5304eea47c067498c1b276ccac0b5b", 16567032, "3b9eeff558ffba042b2780e6fb4926c2aec2769b9d918458f1f5d7f20aaeafad"),
  safe("optimum-intel-internal-testing/tiny-random-exaone4", "d0bcb89ba2b3d1a80d8ade88c9bc0ffa6a28f90c", "exaone4", 923, "35351080dd2f34d3e7e3d1adb62452b16b4af96dddb4b594fc38d0484d1a10cc", 27167128, "09d3dccba2ed5de1e820caafc839ed78f1cbbd02336ed74ef61e604759dca454"),
  safe("katuni4ka/tiny-random-olmo-hf", "8f05f41b70736efb2a6e2b15d051eea02a34f052", "olmo", 707, "bde488dd0fa9d09fac11f1b3eca02958537a316b9d8aef66e6043d38d3e15e5b", 26152632, "49a5bd767851581f9b87b36527418f03697daa0e6a7855f9afe2263c7ec04e0f"),
  safe("yujiepan/mixtral-tiny-random", "c2faa23d97931c5481999c382d79163b02c4793c", "mixtral", 764, "b915b004443c4eb1aff3b1c9c1ae9f5e0e98e88ffd7e2407c42e14346ebe5bec", 522688, "cac5af0aa32d322c4dbd8e2fa6460e74749ca8b6d58632f794f7f3af1192b271"),
  safe("yujiepan/mamba-tiny-random", "6891e834f002b45f3c7f69cf6a8ce12b88e0707f", "mamba", 904, "ee16621b8161736f75fdc916c564979d88da704ed0ed231b092406334707968b", 855760, "dda3cbb122b916339e098328b1f09506d276e2be8d65f01a3555c44e40ab5a13"),
  safe("yujiepan/phi-3-tiny-random", "45275d5d614ca60500c2312985d9a365f46361ad", "phi3", 1353, "32168c23fb069e474b6bec6d170fff1748e88b17213145b6896023ca9c0dfb4c", 2064072, "739beff3f8b771d06fdc4d8a7e01dc230ef9e73e4f3f8e9e082aabaee983cd57"),
];

const ggufExpansion = [
  gguf("phi3-small", "tensorblock/tiny-random-Phi3ForCausalLM-GGUF", "642c29131c8fb7594571b5a110c2fe912c37e26b", "tiny-random-Phi3ForCausalLM-Q2_K.gguf", 2406528, "23571e23676807915c6f46725efe32ebbbbcfb988fb7cca12505bca579165694", null),
  gguf("llama3", "tensorblock/tiny-random-Llama-3-GGUF", "e9033fa865c48c7277cd384b2cf5152400e9a556", "tiny-random-Llama-3-Q2_K.gguf", 14118400, "56447375eaf9b972706d7d4712021b0528d1ea265f43e210f21ca96c2a8dfe9d", "apache-2.0"),
  gguf("falcon40b", "tensorblock/tiny-random-falcon-40b-GGUF", "391ccaf3b3d221811908a7f6158b2d4baf286fa1", "tiny-random-falcon-40b-Q2_K.gguf", 11322816, "0a59856150bee29bd533ecd924f6ee3cb7e01ea04065b297a5e0d498b01b1d95", null),
  gguf("mistral", "tensorblock/tiny-random-mistral-GGUF", "6d261f12f947716905af57eafe119507ea835a44", "tiny-random-mistral-Q2_K.gguf", 22336992, "7e9a4a0c28be6bfd2aec15050954f3476674ca763e5aa6381a9fda91f1ccaf09", null),
  gguf("falcon", "tensorblock/tiny_random_falcon-GGUF", "446328422aa16c25a4b3aa3cee45bae6c27069e6", "tiny_random_falcon-Q2_K.gguf", 6758304, "469001c1f53b82b5cc8b78e888029721d165fb3eabd9ebcc257e29087289b132", "unknown"),
  gguf("llama2-kv", "tensorblock/llama-2-tiny-4kv-heads-4layers-random-GGUF", "6b5bada4acb618950e4951542a0989917e1ca17c", "llama-2-tiny-4kv-heads-4layers-random-Q2_K.gguf", 7581728, "17b638445eb0272abd5c524b69c8cf84dcf23b20142db309595218b93a4424e7", null),
  gguf("mpt", "tensorblock/tiny-mpt-random-remote-code-GGUF", "c151eb3f349485ee8ae72841d7d90b9121d5baa2", "tiny-mpt-random-remote-code-Q2_K.gguf", 8734304, "5627dcb0ff18f6f7200f83c0aed2056a6a7c86b5f2d865833e1b5f00b00e4daa", "apache-2.0"),
  gguf("gptneox", "tensorblock/tiny-random-gpt_neox-GGUF", "b2324868bcb057be9c464614682321ecc725c0b1", "tiny-random-gpt_neox-Q2_K.gguf", 35119936, "44c628e7345dc30df17ca4a9e5be7cf1b2e22baf9e252367b8acd29d90275305", null),
  gguf("qwen3", "tensorblock/optimum-internal-testing_tiny-random-qwen3-GGUF", "ee3df48d1cfc942bf1ab147042713ca158245547", "tiny-random-qwen3-Q2_K.gguf", 16309120, "ea33aa937989b0bd230157c89116d8356ea37f3ed039490ca4e340b650e47a78", null),
  gguf("internlm2", "tensorblock/optimum-internal-testing_tiny-random-internlm2-GGUF", "b7fd1b38d1486897ec1649c1051862e1125068e0", "tiny-random-internlm2-Q2_K.gguf", 21584640, "9e64f76c0994b8940a54d15fdb546219172fe96e7ee88161def9da2cf4dd3244", "apache-2.0"),
  gguf("llama-onnx-origin", "tensorblock/tiny-random-LlamaForCausalLM-ONNX-GGUF", "f90ef2d2ee0901bc246869e8cab456adf18cd19d", "tiny-random-LlamaForCausalLM-ONNX-Q2_K.gguf", 12189792, "26f0172dd7e3fc4b3b7c68771665931343c41b6e596d3b26b784e72415542a85", null),
  gguf("mistral-small", "tensorblock/illuin_tiny-random-MistralForCausalLM-GGUF", "a2852b499771f7da42d1f6fe0bc228aca295b332", "tiny-random-MistralForCausalLM-Q2_K.gguf", 2391648, "4454d69be2ad0dee002b2b29eba254ee6e865a1bf4b0cd78b228644af88f5987", null),
];

const coreMlSources = [
  coreml("bert-squad-fp16", "bert_squad", "question_answering", "float16", "BERTSQUADFP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Text/QuestionAnswering/BERT_SQUAD/BERTSQUADFP16.mlmodel", 217828474, "471c12bb311a0a00d26d89bd55efa3ab4ee865b6664a2f851ccb185ceab55ad5"),
  coreml("deeplabv3-f32", "deeplabv3", "image_segmentation", "float32", "DeepLabV3.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageSegmentation/DeepLabV3/DeepLabV3.mlmodel", 8636475, "4ce025917bcdd9f99eb776d3cb964c78622d4551e031d024f564bc997ad5ffc5"),
  coreml("deeplabv3-fp16", "deeplabv3", "image_segmentation", "float16", "DeepLabV3FP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageSegmentation/DeepLabV3/DeepLabV3FP16.mlmodel", 4342971, "6c2c8cbe1c2d765f231408458b8c273976039ae04a05b112d748fc21cb3bfe95"),
  coreml("deeplabv3-int8-lut", "deeplabv3", "image_segmentation", "int8_lut_weight_storage", "DeepLabV3Int8LUT.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageSegmentation/DeepLabV3/DeepLabV3Int8LUT.mlmodel", 2252685, "935d8c71cdf1e6c2f7941d5ee8c5f4cc4e93363e785d2da7f0ca83cc44981ae2"),
  coreml("depth-anything-v2-small-fp16", "depth_anything_v2", "depth_estimation", "mlprogram_float16", "DepthAnythingV2SmallF16.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/DepthEstimation/DepthAnything/DepthAnythingV2SmallF16.mlpackage.zip", 45828566, "8e875979ec82fa46f292468a7567a520b2aa07c41008c32141f3396da05a4067"),
  coreml("depth-anything-v2-small-fp16-p6", "depth_anything_v2", "depth_estimation", "mlprogram_float16_palettized_6bit", "DepthAnythingV2SmallF16P6.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/DepthEstimation/DepthAnything/DepthAnythingV2SmallF16P6.mlpackage.zip", 18183346, "0268456a2c388d73bc36ef8e8e2d7ecffbbb6afeb1d73e79030c84cb5e7f3594"),
  coreml("detr-resnet50-semantic-segmentation-fp16", "detr_resnet50", "image_segmentation", "mlprogram_float16", "DETRResnet50SemanticSegmentationF16.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/Segmentation/DETR/DETRResnet50SemanticSegmentationF16.mlpackage.zip", 78605496, "4060ed589dd7d7f6f42418aab818846d2c1fc61a9336623c01a2f49ff8ee62a7"),
  coreml("detr-resnet50-semantic-segmentation-fp16-p8", "detr_resnet50", "image_segmentation", "mlprogram_float16_palettized_8bit", "DETRResnet50SemanticSegmentationF16P8.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/Segmentation/DETR/DETRResnet50SemanticSegmentationF16P8.mlpackage.zip", 40257844, "9436f344d7e935ef8f54025a43e760888a058ab9192f18248c3745d25fced4a7"),
  coreml("fastvit-ma36-fp16", "fastvit_ma36", "image_classification", "mlprogram_float16", "FastViTMA36F16.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/FastViT/FastViTMA36F16.mlpackage.zip", 81552218, "b898f6ebe3c734883b05a38258bd3855aaa5e74f96e98c88d312899c8ee925cb"),
  coreml("fastvit-ma36-fp16-headless", "fastvit_ma36", "image_feature_extraction", "mlprogram_float16_headless", "FastViTMA36F16Headless.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/FastViT/FastViTMA36F16Headless.mlpackage.zip", 79288819, "5d1da33e9ea069c773b78920ae4d050c053fbbc354209f5a773459b7253a5797"),
  coreml("fastvit-t8-fp16", "fastvit_t8", "image_classification", "mlprogram_float16", "FastViTT8F16.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/FastViT/FastViTT8F16.mlpackage.zip", 7467627, "156df7e15dd384af30727f1eee4982602924b7e8ae37a65cefa1cfb325261508"),
  coreml("fastvit-t8-fp16-headless", "fastvit_t8", "image_feature_extraction", "mlprogram_float16_headless", "FastViTT8F16Headless.mlpackage.zip", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/FastViT/FastViTT8F16Headless.mlpackage.zip", 6032923, "cd669710c737dab9749a7eecdd0567abd0fef5f4f37f3dccd30c718b5bc0bb87"),
  coreml("mnist-classifier", "mnist_classifier", "drawing_classification", "float32", "MNISTClassifier.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/DrawingClassification/MNISTClassifier/MNISTClassifier.mlmodel", 395695, "816d1a222d5272109166ffc819d7ce44aff923a673884e34d982aa74985ba587"),
  coreml("mobilenetv2-f32", "mobilenetv2", "image_classification", "float32", "MobileNetV2.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/MobileNetV2/MobileNetV2.mlmodel", 24716685, "cb5a35f593582232140556bbfa4618e66b37b8ff2fc33ba17db909e1050fd144"),
  coreml("mobilenetv2-fp16", "mobilenetv2", "image_classification", "float16", "MobileNetV2FP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/MobileNetV2/MobileNetV2FP16.mlmodel", 12393551, "c76832208ff4c936365f0f2609f7b77f7f1a6caf62b0b429056d5ad7e48635ad"),
  coreml("mobilenetv2-int8-lut", "mobilenetv2", "image_classification", "int8_lut_weight_storage", "MobileNetV2Int8LUT.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/MobileNetV2/MobileNetV2Int8LUT.mlmodel", 6257335, "ee53bca4c5042d77aed5c093cba31c2ef229d1cc4c027300af9cc8bf9e240409"),
  coreml("resnet50-f32", "resnet50", "image_classification", "float32", "Resnet50.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/Resnet50/Resnet50.mlmodel", 102586812, "39432f8a20fd16ff311278b95a0953e345c18c907f856ade91cbcadac81bcc07"),
  coreml("resnet50-fp16", "resnet50", "image_classification", "float16", "Resnet50FP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/Resnet50/Resnet50FP16.mlmodel", 51313342, "0ece88983f014e40a108285c0852b626443cb18e1302d525321f3d564e72725b"),
  coreml("resnet50-headless", "resnet50", "image_feature_extraction", "float32_headless", "Resnet50Headless.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/Resnet50/Resnet50Headless.mlmodel", 94367931, "67c0252e458e27e18fdf47d9f71c39f5f207d1e3f92540fb7b4240f7051e7494"),
  coreml("resnet50-int8-lut", "resnet50", "image_classification", "int8_lut_weight_storage", "Resnet50Int8LUT.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ImageClassification/Resnet50/Resnet50Int8LUT.mlmodel", 25677037, "5de1d906095f67c4ca85f778ac04a70830b34e90ff86ab33897cfe3b4edff4c0"),
  coreml("updatable-drawing-classifier", "updatable_drawing_classifier", "drawing_classification", "updatable_pipeline", "UpdatableDrawingClassifier.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/DrawingClassification/UpdatableDrawingClassifier/UpdatableDrawingClassifier.mlmodel", 391495, "d2310a43f60dee043dda4b1367365dfb4735611688089d72b7cc083015e28925"),
  coreml("yolov3-f32", "yolov3", "object_detection", "float32", "YOLOv3.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3/YOLOv3.mlmodel", 248381690, "7fea1f2b83b303e7fc6acf08c05a062793229de474587da336d8724e5bf16e38"),
  coreml("yolov3-fp16", "yolov3", "object_detection", "float16", "YOLOv3FP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3/YOLOv3FP16.mlmodel", 124207764, "276afae53da362e25ad21c827f8a929a8512254b1343433d970e725a0de57d8f"),
  coreml("yolov3-int8-lut", "yolov3", "object_detection", "int8_lut_weight_storage", "YOLOv3Int8LUT.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3/YOLOv3Int8LUT.mlmodel", 62200325, "0e32f297ad9cfc0ea8e67276867956488c7ee10dc87c057da4debc939e20b76d"),
  coreml("yolov3-tiny-f32", "yolov3_tiny", "object_detection", "float32", "YOLOv3Tiny.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3Tiny/YOLOv3Tiny.mlmodel", 35527626, "ed41d7f0c1652994944b4e3578b907f8634a98c2f3feaf33eab8f1c67df44e3f"),
  coreml("yolov3-tiny-fp16", "yolov3_tiny", "object_detection", "float16", "YOLOv3TinyFP16.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3Tiny/YOLOv3TinyFP16.mlmodel", 17769580, "73406178d0f5793d0d5d1e38274acd146a744c2245c9b63a11998a5015925dda"),
  coreml("yolov3-tiny-int8-lut", "yolov3_tiny", "object_detection", "int8_lut_weight_storage", "YOLOv3TinyInt8LUT.mlmodel", "https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/YOLOv3Tiny/YOLOv3TinyInt8LUT.mlmodel", 8913366, "cde8af2528d6eca1d1580fdd0f0147cb6613d40ba962656b5f683c65f571870e"),
];

const hf = JSON.parse(gunzipSync(await readFile(HF_SNAPSHOT)));
const repositories = new Map(hf.repositories.map((row) => [row.id, row]));
const onnxArtifacts = onnxSelections.map(([repositoryId, filePath]) => onnx(repositories.get(repositoryId), filePath));
const oldGguf = JSON.parse(await readFile(GGUF_ANCHORS, "utf8")).artifacts.map((row) => ({
  id: `gguf-${row.id}`,
  format: "gguf",
  stratum: { architecture_class: row.model_family, task: "text_generation_storage", precision: row.declared_quantization_label },
  source: hfSource(row.repository, row.revision, row.license_metadata_status === "declared_mit" ? "mit" : null),
  files: [row.file],
}));
const artifacts = [...onnxArtifacts, ...oldGguf, ...ggufExpansion, ...safeTensorsSources, ...coreMlSources]
  .sort((left, right) => left.format.localeCompare(right.format) || left.id.localeCompare(right.id));
const uniqueSha256 = new Set(artifacts.flatMap((artifact) => artifact.files.filter((file) => isModelFile(file.path)).map((file) => file.sha256)));
const manifest = {
  schema: "deepbom.public_multiformat_corpus.v1",
  generated_at: "2026-08-18T00:00:00.000Z",
  population_boundary: "A declared mixed-frame validation population of immutable public artifacts: purposeful size-bounded ONNX/GGUF/SafeTensors strata plus every asset linked from the Apple Developer Core ML catalog snapshot on 2026-08-18. Counts describe only their declared frame and are not ecosystem prevalence estimates.",
  selection_protocol: {
    status: "purposeful_stratified_non_probability_sample",
    onnx: "Preserve the four still-public Perth baseline path records, replace prior sources that now require authentication, then add public revision-bound graph files across task, architecture, exporter, and precision filename strata from the tracked Hugging Face snapshot. A matching external-data sidecar is included when it is no larger than 100,000,000 bytes; larger payloads remain explicitly unbound.",
    gguf: "Preserve the prior eight real GGUF anchors and add public revision-bound small checkpoints from additional serialized architecture families. File labels are candidates only; encodings are measured from tensor headers.",
    safetensors: "Select one public revision-bound small checkpoint for each architecture family currently decoded by the DeepBOM on-device LLM contract adapter. Random initialization is retained as a task-quality limitation, not hidden.",
    coreml: "Enumerate every model asset linked from the public Apple Developer Core ML model catalog snapshot on 2026-08-18. The frame spans legacy NeuralNetwork and ML Program packages, task families, headless variants, weight precision variants, palettization, and an updatable pipeline. Apple URLs are content-bound because the catalog exposes no source revision.",
    exclusions: "Gated files, unresolved byte identities, unsupported package layouts, and transfers outside the declared sidecar bound are excluded from payload-complete strata. Exclusion is not evidence that a format or feature is absent from the ecosystem.",
  },
  source_policy: {
    revision_binding: "Hugging Face sources use immutable 40-hex repository revisions. Apple Developer assets expose stable publisher URLs but no public source revision, so their downloaded bytes are bound by size and SHA-256 and the missing revision remains explicit.",
    license_boundary: "License metadata is recorded as provenance only. Null means no license was declared in the selected repository metadata; it does not grant reuse rights.",
    byte_retention: "Original model bytes remain in the user-local DeepBOM corpus cache and are not committed or deployed.",
    deduplication: "Population-level artifact denominators use the primary model/package SHA-256. Path aliases remain visible but do not inflate unique-byte counts.",
    onnx_external_data: `Matching sidecars no larger than ${MAX_BOUND_ONNX_SIDECAR_BYTES} bytes are downloaded and hash-bound. Larger or absent sidecars remain explicit unbound external dependencies and are never counted as decoded weight payload.`,
  },
  summary: {
    path_record_count: artifacts.length,
    unique_primary_artifact_count: uniqueSha256.size,
    format_path_counts: countBy(artifacts, (row) => row.format),
    format_unique_counts: Object.fromEntries(["onnx", "gguf", "safetensors", "coreml"].map((format) => [format, new Set(artifacts.filter((row) => row.format === format).map(primarySha)).size])),
    source_kind_counts: countBy(artifacts, (row) => row.source.kind),
    declared_license_record_count: artifacts.filter((row) => row.source.license_metadata).length,
    total_declared_download_bytes: artifacts.flatMap((row) => row.files).reduce((sum, file) => sum + file.size_bytes, 0),
    source_file_count: artifacts.flatMap((row) => row.files).length,
    bound_onnx_sidecar_record_count: artifacts.filter((row) => row.format === "onnx" && row.files.some((file) => file.role === "external_data")).length,
  },
  artifacts,
};
await writeFile(OUTPUT, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`Wrote ${OUTPUT}: ${manifest.summary.path_record_count} path records / ${manifest.summary.unique_primary_artifact_count} unique primary artifacts.`);

function onnx(repository, filePath) {
  if (!repository) throw new Error(`ONNX repository is absent from ${HF_SNAPSHOT}.`);
  const file = repository.files.find((row) => row.path === filePath);
  if (!file?.lfs_sha256 || !Number.isSafeInteger(file.size_bytes)) throw new Error(`${repository.id}/${filePath}: pinned LFS identity is missing.`);
  const sidecar = [
    `${filePath}_data`,
    `${filePath}.data`,
    filePath.replace(/\.onnx$/i, ".onnx_data"),
  ].map((candidate) => repository.files.find((row) => row.path === candidate))
    .find((candidate) => candidate?.lfs_sha256 && Number.isSafeInteger(candidate.size_bytes)
      && candidate.size_bytes <= MAX_BOUND_ONNX_SIDECAR_BYTES);
  return {
    id: slug(`onnx-${repository.id}-${filePath}`),
    format: "onnx",
    stratum: {
      architecture_class: repository.metadata?.architectures?.[0] || repository.metadata?.model_type || "unclassified",
      task: repository.pipeline_tag || "unclassified",
      precision: precisionFromPath(filePath),
    },
    source: hfSource(repository.id, repository.revision, normalizeLicense(repository.metadata?.license)),
    files: [
      { path: file.path, role: "model", size_bytes: file.size_bytes, sha256: file.lfs_sha256 },
      ...(sidecar ? [{ path: sidecar.path, role: "external_data", size_bytes: sidecar.size_bytes, sha256: sidecar.lfs_sha256 }] : []),
    ],
  };
}

function safe(repository, revision, architecture, configBytes, configSha, modelBytes, modelSha) {
  return {
    id: slug(`safetensors-${repository}`), format: "safetensors",
    stratum: { architecture_class: architecture, task: "on_device_llm_weight_contract", precision: "serialized_tensor_dtypes" },
    source: hfSource(repository, revision, null),
    files: [
      { path: "config.json", size_bytes: configBytes, sha256: configSha },
      { path: "model.safetensors", size_bytes: modelBytes, sha256: modelSha },
    ],
  };
}

function gguf(id, repository, revision, filePath, bytes, sha256, license) {
  return {
    id: `gguf-${id}`, format: "gguf",
    stratum: { architecture_class: id, task: "text_generation_storage", precision: "Q2_K_filename_candidate" },
    source: hfSource(repository, revision, license), files: [{ path: filePath, size_bytes: bytes, sha256 }],
  };
}

function coreml(id, architecture, task, precision, filePath, sourceUrl, bytes, sha256) {
  return {
    id: `coreml-${id}`, format: "coreml",
    stratum: { architecture_class: architecture, task, precision },
    source: {
      kind: "apple_developer_asset", publisher: "Apple", repository: null, revision: null,
      repository_visibility: "public_ungated", license_metadata: null,
      license_evidence: "No artifact license is asserted by this corpus; publisher page provenance is recorded separately.",
      catalog_url: "https://developer.apple.com/machine-learning/models/",
    },
    files: [{ path: filePath, source_url: sourceUrl, size_bytes: bytes, sha256 }],
  };
}

function hfSource(repository, revision, license) {
  return {
    kind: "huggingface_repository", publisher: repository.split("/")[0], repository, revision,
    repository_visibility: "public_ungated", license_metadata: license || null, license_evidence: "repository_card_metadata",
  };
}

function precisionFromPath(value) {
  const path = value.toLowerCase();
  if (path.includes("bnb4")) return "bnb4";
  if (path.includes("q4f16")) return "q4f16";
  if (/(?:^|[_/])q4(?:[_.]|$)/.test(path)) return "q4";
  if (path.includes("uint8")) return "uint8";
  if (path.includes("int8") || path.includes("quantized")) return "int8_or_quantized";
  if (path.includes("fp16")) return "fp16";
  if (path.includes("int4")) return "int4";
  return "unspecified_in_filename";
}

function normalizeLicense(value) { return Array.isArray(value) ? value.join(",") : value || null; }
function slug(value) { return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 180); }
function primarySha(artifact) { return artifact.files.find((file) => isModelFile(file.path))?.sha256 || artifact.files[0].sha256; }
function isModelFile(value) { return /\.(?:onnx|gguf|safetensors|mlmodel|zip)$/i.test(value); }
function countBy(rows, selector) {
  const counts = new Map();
  for (const row of rows) { const key = selector(row) || "unknown"; counts.set(key, (counts.get(key) || 0) + 1); }
  return Object.fromEntries([...counts].sort(([left], [right]) => left.localeCompare(right)));
}
