# 외부 검증 결과 트리아지

## Verified correctness addendum (2026-09-09)

The machine-readable authority for status remains
`corpus/external-review/external-review-cases.v1.json`. The following cases
have advanced to `verified` with hash-bound fixtures and cross-surface checks:

- `ER-A1`, `ER-A2`, `ER-I2`: a serialized NaN is an `artifact_defect`, is
  retained in SARIF and MCP, and makes `--gate defects` exit with code 2.
- `ER-B1`: source-backed `SPACE_TO_BATCH_ND` shape reconciliation preserves
  the batch multiplier and the downstream Conv total of 589,824 MACs.
- `ER-B2`: 16x8 classification now requires INT16 per-tensor activation and
  output contracts, an INT8 symmetric constant weight, and a matching
  INT32/INT64 symmetric constant bias when present. No unsupported 16x8
  target speedup is inferred.
- `ER-B3`: a symbolic or partial TFLite compute ledger emits null numeric MAC
  totals, preserves its exact symbolic expression, and fails closed in
  deployment-delta and redesign projections.
- `ER-D1`: a TensorFlow-generated abnormal-scale fixture promotes the maximum
  per-op quantization risk, op identity, and detail into the review summary.
- `ER-D2`: `mac_confidence` is normalized to `exact`, `symbolic`, `partial`, or
  `not_applicable` across executable formats and weight containers.

These changes are verified source state, not a published release. Cases still
marked `reported`, `reproduced`, or `deferred` in the machine ledger remain
open and must not be described as fixed.

외부 리뷰어가 6회에 걸쳐 직접 모델을 생성해 돌린 검증 결과를 항목화한 것입니다.
기준 시점 2026-09-08, 로컬 HEAD `5b474ae`, 게시본 `1.96.10`.

각 항목의 **상태** 표기:

- `reported` — 리뷰어 보고이며 이 저장소에서는 아직 재현하지 않았습니다.
- `reproduced` — 이 저장소 HEAD에서 독립적으로 재현했습니다.
- `fixed` — 재현 fixture를 통과하도록 수정했지만 전체 surface 검증 전입니다.
- `verified` — Web, CLI, MCP 및 관련 export 회귀를 통과했습니다.
- `deferred` — 증거 또는 실행 환경이 없어 의도적으로 보류했습니다.

상태는 반드시 `reported -> reproduced -> fixed -> verified` 순서로만 승격합니다.
각 항목의 기계 판독 상태, 재현 명령, 기대값과 증거 hash는
`corpus/external-review/external-review-cases.v1.json`에서 관리합니다.

리뷰어의 총평은 일관됩니다. **엔진 내부(파서, 수치 계산, 입력 방어, 재현성)는 상용
수준이고, 결과를 사람·CI·모델이 소비하는 층이 엔진을 따라오지 못한다.** 아래 항목의
분포가 그 판단을 그대로 반영합니다.

---

## A. 결함 분류와 게이트 — 최우선

### ER-A1. NaN/Inf 가중치가 `artifact_defect`가 아니라 `caution`으로 분류됨

**상태: verified**

NaN 하나를 심은 safetensors를 만들어 돌린 결과입니다.

```
findings: 3
  caution      | EA-SER-0001 | Serialized tensor payload contains non-finite ...
  evidence_gap | EA-LIN-0001 | Source checkpoint and conversion lineage were not provided
  evidence_gap | EA-LIM-0001 | Representative-data validation not assessable

$ deepbom audit nan.safetensors --gate defects ; echo $?
0
```

`--gate defects`는 "artifact_defect 소견이 있을 때만 exit 2"인데, 명백한 가중치 결함이
있는 모델이 통과합니다. 게이트의 존재 이유가 무너집니다.

**근본 원인 특정.** `web/lib/report-findings.js:213` `inferFindingKind()`는 category가
`integrity`, `numerical_integrity`, `input_contract`, `output_contract`,
`integration_verification` 중 하나이고 evidence가 OBSERVED 계열일 때만
`artifact_defect`를 반환합니다. 그런데 `EA-SER-0001`은 `web/lib/report-findings.js:351`
에서 `category: "numerical_structure_review"`로 선언되어 목록에 없고, 따라서 마지막
줄의 `return "caution"`으로 떨어집니다.

**같은 결함 클래스가 포맷별로 다르게 분류됩니다.** Core ML 경로의 `EA-CML-0001`
(`report-findings.js:300`)은 `category: "integrity"`라서 올바르게 `artifact_defect`가
됩니다. 즉 동일한 "직렬화된 가중치에 NaN이 있다"는 사실이 Core ML에서는 결함,
safetensors/GGUF에서는 주의로 나갑니다.

**해결.** `EA-SER-0001`을 `numerical_integrity`와 명시적
`findingKind: artifact_defect`에 바인딩했습니다. SHA-256
`e65661c71d739ca23dfe9e6f1f83af86323c14d072a62895a5762e2d8efeaaef`인 NaN fixture로
canonical finding, 사람용 summary, SARIF, defect gate exit 2, MCP를 함께 검증합니다.
all-zero tensor는 계속 caution이며 defect gate를 차단하지 않습니다.

### ER-A2. NaN이 SARIF 출력에 실리지 않음

**상태: verified**

ONNX/safetensors에서 JSON 분석에는 `nan_tensors: 1`이 있는데 SARIF에는 없었다는
보고였습니다. 현재 SARIF 투영은 canonical finding 전체를 보존하며, A1 수정 후
`EA-SER-0001`과 `deepbomFindingKind: artifact_defect`가 유지됨을
`node scripts/check-external-review-defect-gate.mjs`로 고정했습니다.

### ER-A3. 출처 메타데이터 부재가 실제 결함보다 앞에 나옴

**상태: reported**

`--fail-on high`로 빌드가 막혔는데, 막은 이유가 NaN이 아니라 "소스 체크포인트 해시와
변환기 버전 정보 없음"이었습니다. 규제 문맥에서는 타당한 설계지만, 엔지니어가
"이 모델 괜찮아?"라고 물었을 때 첫 답이 "출처 문서가 없습니다"이면 도구를 안 씁니다.

**제안.** `--policy engineering|regulatory` 분기. 기본 `engineering`에서는 lineage
부재를 NOTE로 낮추고 NaN·scale 이상을 ERROR로 올립니다. `scan_policy`에 이미 분기
구조가 있으므로 새 개념이 아닙니다.

---

## B. 수치 정확성

### ER-B1. `SPACE_TO_BATCH_ND`의 배치 차원 전파 누락 — dilated conv MAC 4배 과소

**상태: verified**

TFLite 컨버터는 dilation conv를 `SPACE_TO_BATCH_ND` + `CONV_2D` +
`BATCH_TO_SPACE_ND`로 낮춥니다. deepbom은 `SPACE_TO_BATCH_ND` 출력 shape를
`[1,10,10,16]`으로 추론했으나 TF 인터프리터 실측은 `[4,10,10,16]`입니다. block size
2x2가 배치를 4배로 늘리는 것을 반영하지 않았습니다. 결과적으로 뒤따르는 conv의 MAC이
147,456으로 나왔고 정답은 589,824입니다.

**영향 범위.** dilated conv를 쓰는 세그멘테이션 모델군 전체.

**해결.** TensorFlow 2.16.1 생성기를 사용한 SHA-256
`ff64f16a7741c4e517460982db47826bfbac8e0082aa13962d539a5559766877` fixture에서
직렬화된 stale shape와 source-derived effective shape를 모두 보존합니다. 핀 고정된
TFLite Prepare 계약으로 batch `1→4`와 후속 Conv MAC `589,824`를 검증하며, block과
padding을 확정할 수 없으면 숫자를 추측하지 않습니다.

### ER-B2. 16x8 양자화(int16 활성화 + int8 가중치) 오분류

**상태: verified**

`dynamic_range_or_weight_only`로 분류하고 "양자화된 compute MAC 0%"라고 보고했습니다.
실제로는 int16 텐서 13개에 모든 compute op이 정수 도메인에서 돕니다. JSON 안에 int16이
158번 언급되는데도 분류 로직이 8-bit 활성화만 정수로 인정하는 것으로 보입니다.

**영향 범위.** 16x8은 오디오·의료 신호 처리에서 정밀도 때문에 실제로 쓰입니다. 의료
AI를 표방하는 도구에서 특히 아픈 구멍입니다. `full_integer_16x8` 같은 별도 분류가
필요합니다.

**해결.** SHA-256
`f245d5a6435efb81edc0a15cfa24867978b0588ad260a4f4cc0b0d3933977e9f`인
TensorFlow 2.16.1 16x8 fixture를 `full_integer_16x8`로 분류하고, INT16 activation과
INT8 weight의 MAC `112,896`을 quantized compute로 집계합니다. 타깃 프로파일에 16x8
성능 계약이 없으므로 INT8-vs-FP32 speedup은 적용하지 않습니다.

### ER-B3. TFLite 파서가 자리표시자 shape를 확정값처럼 계산 — 문서화된 계약 위반

**상태: verified**

Transformer를 Keras 경로로 변환하면 FFN Dense가 런타임 RESHAPE를 거치고, 파일에
직렬화된 정적 shape는 자리표시자 `[1,1]`입니다. deepbom은 이를 그대로 읽어 FC 출력을
`[1,128]`로 보고 8,192 MAC으로 계산했습니다. 실제는 `[16,128]`에 131,072 MAC이고,
총합은 622,912 대 정답 1,114,432로 56%만 나왔습니다.

**이것은 이미 문서화된 계약을 TFLite 파서만 안 지키는 것입니다.**
`docs/KNOWN_ISSUES.md`는 1.96.0에서 "`total_macs`는 모든 후보 compute 연산이 평가된
경우에만 값을 갖고, 아니면 `null`"이라고 선언했습니다. 리뷰어가 확인한 대로 ONNX
파서는 이 계약을 지켜서 shape를 모르면 `total_macs: None` + `partially_assessed` +
op 단위 사유를 냅니다. TFLite 파서는 안 지킵니다.

**참고.** deepbom이 완전히 놓친 것은 아닙니다. `dynamic_tensor_count` 97개와 기호식
MAC(`65536*D1 + 8192*D32 + ...`)은 정상적으로 냈습니다. 문제는 요약 필드
`total_macs`가 그 불확실성을 숨긴다는 점입니다.

**우회책(문서화 가치 있음).** 변환 시 batch를 고정하면 TFLite 두 경로 모두 정확히
1,114,432가 나옵니다. ONNX는 `onnxsim`으로 상수 접기까지 하면 정확해집니다. 이건 배포
아티팩트를 만들 때 원래 해야 하는 정리라, 도구가 요구하는 게 부당하지 않습니다.

**해결.** SHA-256
`303d997682abd998c803282c3d47c10b3d25e88e292a6931fa1e9a28e4b9f9f9` fixture에서
shape signature의 음수 차원을 런타임 미바인딩 상태로 유지합니다. top-level
`total_macs`와 `total_macs_decimal`은 명시적 `null`, `mac_confidence`는 `symbolic`,
평가 subtotal은 `0`, 정확한 기호식은 `4096*D2`로 보존됩니다.

### ER-B4. `TRANSPOSE_CONV` 집계 관례 미문서화

**상태: reported**

2,359,296(출력 기준)으로 보고했는데, 실제 커널 곱셈 수인 입력 기준은 589,824입니다.
stride² 만큼 차이가 납니다. 어느 관례든 문서화만 되면 틀렸다고 할 수 없지만, B1의
과소집계와 합쳐지면 한 모델 안에서 오차가 양방향으로 생깁니다.

### ER-B5. F16 GGUF를 양자화 계열로 라벨링

**상태: reported**

양자화 텐서 0개라고 스스로 세면서 라벨은 `block_or_tensor_encoded_weights`입니다.

### ER-B6. saturation 비율 정의 미문서화

**상태: reported**

deepbom 보고값 최대 4.86% 대 리뷰어가 `|w| == 127`로 직접 센 값 11.1%(depthwise).
정의가 다를 수 있으나 문서가 없으면 검증이 불가능합니다. `grid_utilization`도 같습니다.

---

## C. 포맷별 성숙도

### ER-C1. Core ML mlprogram: blob 참조 미해석 → MAC 전부 계산 실패

**상태: reported**

conv 4개와 matmul 1개의 출력 shape는 다 읽어놓고 MAC을 하나도 계산하지 못해 total이
None입니다. 가중치가 `weights/weight.bin` 블롭으로 분리되어 있는데 그 참조를 따라가
커널 shape를 얻는 단계가 빠졌습니다. 블롭 참조를 445번 인식해놓고 활용을 못 하는
상태입니다. 레거시 `.mlmodel`(neuralnetwork)은 5,833,344로 정확히 일치합니다.

**영향 범위.** mlprogram은 현재 Apple의 기본 포맷입니다. iOS 팀에게는 사실상 미지원.

### ER-C2. 양자화 mlpackage를 손상 파일로 오판

**상태: reported**

coremltools로 int8 per-channel 양자화한 mlpackage를 "constexpr_affine_dequantize에
quantized_data, zero_point, scale 또는 axis가 없다"며 파싱 거부했습니다. 리뷰어가 MIL을
직접 열어보니 셋은 모두 있고 `axis`만 없었습니다. linear_symmetric per-channel에서
axis 표현이 파서 기대와 다른 것으로 보입니다. **정상 파일을 손상으로 오판하는 것은
지원 목록에 올려둔 것 대비 가장 나쁜 실패 모드입니다.**

### ER-C3. ExecuTorch(.pte), TensorRT 미검증

**상태: deferred** — 리뷰어 환경 제약(디스크, GPU 부재)으로 검증 자체가 불가했습니다.

### ER-C4. 포맷 성숙도 표시 부재

지원 확장자 7개가 동등하게 광고되지만 실제 신뢰 수준은 TFLite / ONNX(정적 shape 필요)
/ GGUF / safetensors 4개입니다. **성숙도 표를 문서와 MCP 도구 설명문 양쪽에 넣어야
합니다.** 미지원이라고 적는 편이 지원한다고 적고 정상 파일을 거부하는 것보다 낫습니다.

---

## D. 출력 소비 계층

### ER-D1. 요약 필드가 정상 모델과 이상 모델을 구별하지 못함

**상태: verified**

per-channel scale 비율을 1012배로 벌린 모델과 정상 모델의 top-level 요약 필드
(`saturated`, `low_grid` 등)가 완전히 같은 값입니다. SARIF와 op 단위
`quantization_risk`에는 `scale ratio 1.01e+3, CV 7.46, warn`이 정확히 떴는데, 요약만
보면 차이를 못 느낍니다.

**해결.** `quantization_risk`의 최댓값과 해당 op identity/detail을 최상위
`max_quantization_risk` 계약으로 승격했습니다. 정상 fixture와 scale 이상 fixture가
동일한 요약으로 축약되지 않는지 `check-external-review-tflite-correctness.mjs`에서
검증합니다.

### ER-D2. `total_macs`에 확실성 필드 없음

**상태: verified**

`mac_confidence`를 `exact` / `symbolic` / `partial` / `not_applicable` 중 하나로
정규화해 Web, CLI, MCP envelope와 report가 함께 소비합니다. 숫자 합계를 닫지 못한
경우에는 `total_macs: null`과 평가 subtotal 또는 기호식을 별도로 보존합니다.

### ER-D3. 출력 크기

**상태: verified**

21KB 모델에 219KB JSON, MobileNetV2에 2MB. 요약 / 소견 / 전체 증거 3층으로 나누고
기본은 요약이어야 합니다.

MCP와 CLI의 기본 audit 출력은 같은 bounded human summary를 사용합니다. 전체 원장은
`--output-format json-compact` 또는 `--output-format envelope`처럼 명시적으로
요청할 때만 반환합니다.

### ER-D4. safetensors markdown 요약 누락

포맷별 출력 완성도가 들쭉날쭉합니다.

---

## E. diff — 사실상 비어 있음

### ER-E1. `graph_delta` 전부 0, `change_impact`는 자명한 문장만

**상태: reported**

scale이 1000배 벌어진 것이 diff에 안 잡혔고, change_impact는 "바이트가 바뀌었으니
성능 재검증 필요"라는 말만 합니다.

### ER-E2. TFLite 전용

**기회.** FDA PCCP(사전 변경 관리 계획)는 정확히 "어떤 변경이 재검증을 요구하는가"를
다룹니다. `change_impact`가 제대로 되면 이 도구만의 차별점이 됩니다. 지금은 그 자리가
비어 있습니다.

---

## F. 규칙 카탈로그

### ER-F1. `explain-rule`에 규칙이 3개뿐

**상태: reproduced**

```
$ deepbom explain-rule --list --json
count: 3
ids: onnx.conv.output-shape, tflite.conv2d.macs, finding.gate.defects
```

SARIF에 나오는 `EA-QNT-0001`, `EA-SER-0001`, `EA-CML-0001`, `EA-LIN-0001` 등 실제 규칙
ID는 설명을 받을 수 없습니다. **CI에서 `--fail-on`을 걸려면 각 규칙의 설명, 심각도
근거, 오탐 가능성이 문서화되어야 합니다.** MCP에 `deepbom_explain_rule` 도구가
추가됐지만 카탈로그가 비어 있어 그 도구도 반쪽입니다.

---

## G. CycloneDX

### ER-G1. CycloneDX 일반 소비자에서 root component를 놓칠 수 있음

**상태: reproduced interoperability risk**

```
components: 0 | metadata.component: machine-learning-model | properties: 111
```

`metadata.component`는 유효한 root component이고 스키마 검증도 통과하며 modelCard와
속성도 제대로 들어갑니다. 따라서 빈 `components` 배열 자체는 CycloneDX 결함이
아닙니다. 다만 `components`만 순회하는 일반 SBOM 소비자에서는 빈 BOM처럼 보일 수
있으므로 실제 소비자 호환성 검사와 명시적인 export 문서가 필요합니다.

---

## H. 제품/API 방향

Python·npm 라이브러리 API 제안은 검증 결함이 아니라 공개 제품 표면에 관한 결정입니다.
외부 리뷰의 원문과 후속 후보는 `docs/PRODUCT_DIRECTION.md`에서 별도로 관리합니다.

---

## I. 품질·릴리스 운영

### ER-I1. 외부 리뷰 경계를 포괄하는 정확성 골든 코퍼스 부족

**상태: reproduced**

저장소에는 expected-output, 공개 다중 포맷, 양자화, Core ML, GGUF, SafeTensors 등
여러 회귀 코퍼스가 이미 있습니다. 다만 리뷰어가 하루 만에 찾은 ER-B1, ER-B2,
ER-A1과 같은 경계 조건을 하나의 hash-bound 기대값 행렬로 묶는 장치가 부족합니다.

**제안.** 아키텍처별 참조 모델 30~50개를 저장소에 넣고 정답(MAC, 파라미터 수, 양자화
분류, 기대 소견 ID)을 JSON으로 고정해 모든 커밋에서 대조합니다. 목록: MobileNet,
ResNet, U-Net, LSTM, Transformer(batch 고정/미고정 양쪽), dilated conv, transposed
conv, 16x8, fp16, Q4/Q8 GGUF, 그리고 NaN·죽은 채널·scale 이상·잘린 파일 결함 주입 모델.
**Keras 몇 줄로 생성되므로 저작권·용량 문제가 없습니다.**

### ER-I2. 결함 주입 회귀 테스트 부재

**상태: verified**

"NaN 심은 모델은 `--gate defects`에서 반드시 exit 2"라는 테스트 하나가 A1을 막았을
것입니다. 현재 이 계약은 `node scripts/check-external-review-defect-gate.mjs`에서
canonical finding, summary, SARIF, CLI gate, MCP와 함께 검증됩니다.

### ER-I3. 스키마 안정성 신호 없음

0.1.0에서 1.94.2로 뛴 이력, 거의 매일 패치, 내부 스키마 버전 수십 개. 소비자가 어느
필드를 믿고 코드를 짤지 알 수 없습니다. **요약 층, SARIF 규칙 ID, CLI 플래그만이라도
semver 대상으로 선언하고, 나머지 내부 스키마는 명시적으로 불안정 표시.**

### ER-I4. stable / nightly 채널 분리

매일 나오는 건 pre-release로, 2~4주 단위로 골든 코퍼스를 통과한 빌드만 stable로 승격.

---

## J. 방향과 포지셔닝

사용자군, 포맷 집중도, 공개·비공개 경계, 진입 경험과 지속 가능성은 코드 결함과 별도의
의사결정입니다. 외부 리뷰의 원문과 후속 후보는 `docs/PRODUCT_DIRECTION.md`에서 관리하며,
이 트리아지의 release blocker 수에는 포함하지 않습니다.

---

## 이미 해결된 항목 (재현 시도 결과)

리뷰어가 테스트한 빌드 이후 고쳐진 것들입니다. **게시본 `1.96.10`에서 직접 실행해
확인했습니다.**

| 리뷰어 보고 | 현재 상태 |
| --- | --- |
| MCP 모든 `tools/call`이 `Unexpected positional argument`로 실패 | **해결됨.** `npx deepbom@1.96.10 mcp`로 `deepbom_capabilities`, `deepbom_audit` 모두 정상 응답 |
| MCP 응답이 컨텍스트에 안 들어감 | **해결됨.** `deepbom_audit` 기본 출력이 `summary`(910바이트 사람이 읽는 요약) |
| MCP 도구가 4개인데 설명 조회 불가 | **부분 해결.** `deepbom_explain_rule` 도구 추가. 다만 F1(규칙 3개)이 남아 반쪽 |
| 경로 제한·취소·동시성 | **해결됨.** `DEEPBOM_MCP_ALLOWED_ROOTS`, 취소 처리, `DEEPBOM_MCP_MAX_CONCURRENT` |

리뷰어의 "우선순위를 하나만 고르라면 MCP argv 버그"와 A1 NaN 분류는 모두 닫혔습니다.
남은 항목의 우선순위는 기계 원장의 상태와 fixture 확보 여부를 기준으로 정합니다.

---

## 권장 처리 순서

**검증 완료**

1. **A1·A2·I2** NaN defect gate, SARIF, MCP, 정상/all-zero 음성 대조
2. **B1·B2·B3** TFLite shape reconciliation, 16x8, nullable numeric MAC total
3. **D1·D2·D3** quant risk 승격, 공통 MAC confidence, bounded 기본 summary

**다음 correctness 순서**

4. **C2** 정상 mlpackage 오판을 독립 fixture로 재현
5. **B4·B6** TRANSPOSE_CONV와 saturation/grid-utilization 정의를 source-pin과 함께 고정
6. **C1** ML Program blob operand 연결을 공개 fixture로 검증
7. **B5·D4** GGUF F16 및 SafeTensors 사람용 문구를 실제 fixture로 대조
8. **F1** 규칙 카탈로그 완성 (SARIF에 나오는 모든 ID)
9. **A3** 정책별 차단 조건을 finding identity와 분리
10. **E1·E2** Artifact IR 기반 semantic diff 범위 확대
11. **H1/H2** Python·npm 얇은 래퍼

**구조 작업**

13. **I1** 골든 코퍼스 — 위 전부의 재발을 막는 유일한 장치
14. **I3/I4** 스키마 semver 선언과 stable 채널
15. **C4/B4/B6** 문서 세 가지: 포맷 성숙도 표, MAC 집계 관례, saturation·grid_utilization 정의
