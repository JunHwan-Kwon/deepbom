# DEEPBOM 계산 및 공통 IR 정합성 재검토

검토일: 2026-09-28. 기준 소스: `02e4b76` (DEEPBOM 1.109.0).
수정 작업: `codex/evidence-correctness-review`.

## 범위와 판정 원칙

이번 작업은 공통 IR, 계산, 형식별 어댑터, BOM 대조, 실행·외부 증거 연결 및 Web/CLI/MCP 소비 경로를 대상으로 기존 전체 검사와 추가 경계 사례를 실행한 정확성 검토다. 코드의 모든 줄과 가능한 모든 모델을 수학적으로 증명한 작업은 아니다.

검증을 세 단계로 구분했다.

1. **원본 규칙**: 고정된 공식 저장소 커밋과 파일 해시를 확인하고, 자료형·차원·포트·저장 형식의 의미를 대조한다.
2. **계산과 내부 정합성**: 독립적인 정수/유리수 기준값, 실제 직렬화 파일, 해시를 다시 계산한 모순 문서를 사용한다. 잘못된 문서가 단순 해시 불일치 때문에만 거부되는 검사는 충분하지 않다.
3. **출처 대조**: `validateModelIrAgainstSource` 등에서 별도로 제공된 원본 IR로 재생성한 결과와 대조한다. 내부 정합성과 SHA-256 일치만으로 출처의 진실성을 입증하지 않는다.

## 재현하고 수정한 문제

| 구분 | 수정 전 | 수정 후 | 회귀 근거 |
| --- | --- | --- | --- |
| 공통 논리 바이트 계산 | ONNX 파서는 FP8, FP4, INT2/UINT2를 인식하지만 공통 IR과 텐서 목록은 일부 자료형의 크기를 `null`로 출력 | 고정 폭 스칼라 비트 수를 공통 함수로 통합. `ceil(element_count × bits / 8)`를 BigInt로 계산 | ONNX의 고정 폭 자료형 25개 × 형상 6개, SafeTensors 별칭 9개, 안전 정수 범위 초과·빈 텐서·스칼라 |
| 논리 텐서 목록과 그래프 | 목록의 이름·dtype·shape·graph reference를 변경하고 재해시해도 일부 모순을 허용 | 동일 ID의 그래프 값과 이름·dtype·shape·type contract·storage refs 및 목록 크기를 대조. 그래프 참조를 null로 지워 검사를 건너뛰지 못하게 함 | Artifact IR 및 Model IR의 재해시 변조 10개 |
| 스칼라 Weight IR 연결 | 원본 `shape=[]`를 `[1]`로 바꿔도 원소 수가 같아서 통과 | 알려진 rank의 스칼라도 Model IR의 저장 형상과 정확히 대조 | 실제 SafeTensors 스칼라에 `[1]`, `[1,1]`, `null`을 각각 대입 |
| 차원 상수 검증 | `kind=constant, value=null`이 유효한 차원처럼 통과 | 상수는 실제 정수 계약이어야 하며 decimal과 숫자 표현이 일치해야 함 | null, 빈 객체, 잘못된 숫자 표현, 비정규 decimal |
| Model IR 내부 연결 | 존재하는 다른 연산·포트를 참조하면 잘못된 소유 관계·방향·가중치 연결도 통과 가능 | 연산↔포트↔값의 상호 참조, 영역, 방향, 위치, 누락을 검사. 가중치 연결을 포트·저장 객체에서 재생성해 대조. 데이터·저장 관계도 연결된 값과 대조 | 다른 포트 소유자·방향, 포트 누락, 가중치 연결 변경·누락, 관계 변경 |
| 계산 값과 평가 범위 | Model IR의 수치와 metric contract가 다르거나, 완전한 값이 부분 coverage를 주장해도 통과 가능 | 값·단위·원본 연산 참조를 대조하고, non-null 완전 값의 평가 범위가 전체인지 검사 | MAC 숫자만 변경, exact 값에 assessed=0/eligible=1 부여 |

예를 들어 원소 3개의 `FLOAT4E2M1`은 논리적으로 **2 bytes**, 원소 3개의 `INT2`는 **1 byte**다. 이 값은 고정 폭 논리 표현의 크기이며, protobuf 파일 크기·희소 저장량·메모리 할당량과는 별개다.

SafeTensors `F6`의 알려진 비트 수로 논리 크기를 계산할 수 있다는 사실은 해당 인코딩의 값 복호화가 가능하다는 뜻이 아니다. 패킹 해석이 구현되지 않은 수치 분석 상태는 그대로 유지한다. GGUF `Q4_0`·`Q4_K`와 같은 블록 인코딩도 단순 4비트 스칼라로 취급하지 않는다.

`staticTensorPayloadBytes`는 명시적인 unknown rank를 스칼라로 계산하지 않는다. 기존의 동적 shape signature, 미지원 인코딩, 안전 숫자 범위 초과 처리도 유지한다.

## 참조 규칙과 대응

| 참조 | 확인한 규칙 | DEEPBOM 적용 경계 |
| --- | --- | --- |
| [ONNX TensorProto](https://github.com/onnx/onnx/blob/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/onnx.proto), [4비트 정수](https://onnx.ai/onnx/technical/int4.html), [4비트 실수](https://onnx.ai/onnx/technical/float4.html) | dtype의 비트 폭과 바이트 패킹 | 원소 수에서 논리 바이트 산출. 양자화·정확도·실행 비용의 자동 추론 근거로 확대하지 않음 |
| [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html), [차원 의미](https://onnx.ai/onnx/repo-docs/ShapeAnnotationSemantics.html) | 스칼라와 unknown rank 구별, 모델 전역의 symbolic dimension | 숫자처럼 보이는 이름도 symbol로 보존. 실제 실행 입력에서 동일 symbol의 차원 충돌 검사 |
| [TensorFlow Lite schema](https://github.com/tensorflow/tensorflow/blob/87bbf65b8d23d3f06912b1b2183587e1884bc45c/tensorflow/compiler/mlir/lite/schema/schema.fbs) | `has_rank`, `shape_signature=-1`, subgraph-local tensor indices | 실제 동적 WHILE fixture의 중첩 조건 텐서가 unknown dimension으로 유지되는지 추가 확인 |
| [SafeTensors](https://github.com/huggingface/safetensors/tree/6eb4dc9a28ebce297606e0f4836bbf28839cacef) | 빈 tensor, rank-zero scalar, dtype와 payload 범위 | 빈 논리 텐서를 누락하지 않으며 스칼라를 1차원 weight로 재해석하지 않음 |
| [GGUF/ggml](https://github.com/ggml-org/llama.cpp/tree/7bd8282c37fcd9c4d7236106d664761a23318f18/ggml) | 블록 단위 타입 특성과 패킹 | 블록 저장 형식과 고정 폭 스칼라 크기 계산을 구별 |
| [CycloneDX 1.7](https://cyclonedx.org/docs/1.7/json/), [properties](https://cyclonedx.org/use-cases/cyclonedx-properties/) | 스키마 유효성과 실제 모델에 대한 사실 대조는 다른 검사 | 공식 스키마, BOM-artifact 비교, vendor property coverage 및 companion 파일 해시 연결 검사 |
| [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) | 결정론적 JSON 정규화 | 각 IR 및 연결 파일의 해시 생성. 수정된 사실·method가 있으면 관련 digest도 갱신 |

원본 커밋 목록은 `config/model-ir-upstream-sources.v1.json`, 지원 범위와 필수 검사는 `config/ir-support-gate.v1.json`에 유지한다. 최신 상위 저장소 HEAD가 존재한다는 사실만으로 그 전체 기능을 지원한다고 선언하지 않는다.

## 호환성과 재현성

- Artifact IR method: **2.3.0 → 2.3.1**.
- Model IR method: **1.1.0 → 1.1.1**.
- schema identity는 `deepbom.artifact_ir.v2`, `deepbom.model_ir.v1`을 유지한다.
- 이전 method 문서는 읽을 수 있고, 읽는 과정에서 기존 digest를 덮어쓰지 않는다. 모순 문서에 대한 검사는 강화된다.
- 같은 아티팩트라도 method 변경에 따라 새 IR digest는 달라진다. 모델 파일 SHA-256은 변하지 않는다.
- 공개 MobileNet 예제의 IR, CycloneDX 및 연결 파일 해시를 함께 갱신한다. 검토된 예제 차이는 method와 이에 종속되는 해시이며, 모델 수치의 임의 변경이 아니다.
- source 재생성 검사는 현재 method에 대해 수행한다. 이전 method를 현재 알고리즘으로 조용히 재생성해 같은 결과라고 주장하지 않는다.
- 브라우저 캐시는 `v595 → v596`으로 갱신한다. 배포 시 기존 공통 IR 코드가 캐시에 남는 것을 방지하기 위한 변경이다.

## 검증 기록

최종 결과: 전체 runner의 **255개 검사 모두 통과**. 구간별 실행 로그에서 `completed`로 확인된 검사 번호를 합산했으며, 중단·실패한 시도는 통과로 세지 않았다. 수정된 기대 method·reference digest·캐시 버전을 포함해 각 실패 항목을 다시 실행하여 통과를 확인했다.

- 공식 직렬화 소스 83개 파일 및 생성된 GGUF codebook 7개: 원본 해시 검사 통과.
- Core ML 공식 소스 33개 파일, 추가 형식 4개 계열의 고정 소스 17개 파일: 원본 검사 통과. 후자는 manifest 계약 확인이며 형식 전체 구현을 입증하지 않는다.
- `cargo test --locked` (Rust stable): **129 passed, 0 failed**.
- 공통 IR 회귀: **27/27개 검사 그룹 통과**. 여기에는 저비트 자료형·scalar·zero dimension·unsafe integer·재해시 모순 문서·이전 method 호환 검사가 포함된다.
- 독립 Python Fraction/Decimal 기준값: **2,040개 moment 대조**, **510개 aggregate mean**, **756개 비교 수치**, **126개 분할 합산 불변성** 검사 통과. 안전 정수 범위 및 symbolic binding 검사도 통과했다.
- 추가 독립 MatMul 검사: rank 1–4, 각 축 크기 0/1/2의 **14,400개 형상 쌍**을 NumPy의 shape 판정 및 스칼라 곱셈 항 직접 열거와 대조해 모두 일치했다. 유효한 3,558개와 호환되지 않는 10,842개를 함께 검사했다. 모델 추론 성능 측정이 아니라 shape/명목 MAC 규칙 대조다.
- 추가 ConvTranspose 검사: 입력 길이·커널·stride·dilation·양쪽 padding·output padding을 조합한 **18,220개 계약**에 대해, 출력 범위 안에 기여하는 입력/커널 곱을 배치·그룹·채널별로 직접 열거한 정수 결과가 모두 일치했다.
- 전체 runner는 형식별 계산, BOM 대조, Evidence Link/OMOP, 변환 receipt, Web/CLI 의미 해시 일치, MCP/widget, export 및 브라우저 동작을 포함한다.

실행 로그와 `check-manifest.json`은 작업 공간의 `.local-validation/correctness-review/`에 보관한다. 검사 도중 method/cache 갱신 시점에 발생한 Web/CLI의 신·구 module 혼용 및 브라우저 페이지 전환 실패는 파일을 고정한 상태에서 다시 실행해 통과했다. 공개 reference의 이전 digest와 method 기대값 실패는 diff를 검토해 reference를 갱신한 뒤 재검사했다. 이 실패들을 계산식 오류나 통과 결과로 섞어 집계하지 않았다.

주요 재현 명령:

```sh
node scripts/check-all.mjs
RUSTUP_TOOLCHAIN=stable cargo test --locked
node scripts/check-ir-hardening.mjs
node scripts/check-numerical-ir.mjs
node scripts/check-numerical-precision.mjs
node scripts/check-calculation-boundaries.mjs
node scripts/verify-serialized-source-pins.mjs
node scripts/verify-coreml-source-pins.mjs
node scripts/verify-model-ir-preview-source-pins.mjs
```

이 검토 환경은 Node 24.12.0, Rust stable 1.98.1, Python 3.13.13을 사용했다. 추가 MatMul oracle은 NumPy 2.4.6으로 실행했다. 소스 pin 검사는 네트워크 접근이 필요하고, 전체 runner의 브라우저 검사는 프로젝트의 Playwright 환경을 사용한다.

## 남는 경계

이번 검토는 미지원 연산, 동적 차원, 다중 파일 누락, 외부 런타임 증거의 진실성, 실제 장치 실행 성능을 대신 검증하지 않는다. 이 항목들은 해당 상태와 근거를 보고해야 한다. 특히 ONNX/IR 문서의 SHA-256을 다시 계산할 수 있다고 해서 그 안에 기재한 모든 사실이 원본에서 유래했다는 뜻은 아니다. 독립된 원본과의 재대조가 계속 필요하다.

모든 테스트 통과는 검사한 입력과 규칙에 대한 결과다. 모든 미래 형식·연산·모델에서 버그가 없다는 보증으로 해석하지 않는다.
