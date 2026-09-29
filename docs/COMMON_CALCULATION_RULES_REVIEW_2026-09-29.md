# 공통 계산 규칙과 중복 구현 검토

검토일: 2026-09-29. 기준 커밋: `026b328a6e01a30d72df9bde4d4ddf9f829a82fc`.

## 결론과 범위

자료형 폭, 정확한 정수, 텐서 원소 수와 논리 바이트, 기본 수치 변환, 정수 저장 범위, 일부 ONNX 계약 읽기, TFLite 명목 MAC 계산의 소유 모듈을 통합했다. JS와 Rust가 각각 유지하던 스칼라 폭 표는 하나의 선언에서 생성한다. 포맷별 실제 저장 구조와 계산 가능성 판정은 공통 산술 위에 남긴다.

`web`, `src`, `bin`, `worker`, `native`, `channels`, `protected`의 애플리케이션 소스 507개를 목록화했다. 목록에는 파일 SHA-256, 언어, 역할, 함수 및 계산 후보 수가 있다. JS/MJS는 AST로 함수와 수치 표를 조사하고, Rust에는 자료형별 숫자 분기 검사도 적용했다. Rust/Python/C/C++는 파일 목록과 후보 탐색 및 관련 구현의 수동 검토 범위다. 생성물·vendor·WASM 바이너리·빌드 디렉터리는 중복 탐색 대상에서 제외하고, 생성물 일치와 빌드 검사로 별도 확인한다.

이 조사는 모든 가능한 모델·입력·컴파일러에서 오류가 없다는 증명이 아니다. 특히 AST의 동일 본문 탐색은 이름을 바꾸거나 다른 알고리즘으로 표현한 의미상 중복까지 증명하지 않는다. 검사한 범위, 남긴 예외, 재현 가능한 회귀 검사를 기록하는 것이 이 문서의 보장 범위다.

- 기계 판독 목록: [common-calculation-inventory.json](reviews/common-calculation-inventory.json)
- 유지한 중복의 이유와 본문 해시: [common-rule-exceptions.v1.json](../config/common-rule-exceptions.v1.json)
- 검사 구현: [audit-common-rules.mjs](../scripts/audit-common-rules.mjs)

## 공통 소유권

| 규칙 | 공통 소유 모듈 | 적용 경로 |
| --- | --- | --- |
| 고정 폭 스칼라의 비트 수·정수 범주 | `config/scalar-types.v1.json` → JS/Rust 생성물 | Artifact/Model IR, 포맷 어댑터, 텐서 무결성, WASM, 선택적 분석 모듈 |
| 정확한 비음수 정수와 안전한 숫자 미러 | `exact-integer.js`, `tensor_math.rs` | IR, LLM 메모리·배치·토큰 예산, 실행 증거, 수집기 |
| 형상 원소 수, 패킹 바이트, 정적 텐서 크기 | `tensor-size.js`, `tensor_math.rs` | 그래프, 인벤토리, 배치 경계, 보고서, 런타임 형상 바인딩 |
| binary16/BF16 해석, 정수→binary32 반올림, 축 정규화, 연속 비트 읽기 | `scalar-numeric.js` | ONNX ML, Core ML, GGUF/SafeTensors 페이로드 분석 |
| 정수 코드 범위와 8비트 per-tensor 전제 | `tensor-size.js`, `quantization-math.js`, Rust `quantization_math.rs` | 양자화 계약, 재양자화, 누산기·잔차 분석 |
| ONNX enum 대응 | `onnx-tensor-types.js` | ONNX 파서, TypeProto 해석 |
| ONNX 정적 값·sequence·optional 증거 읽기 | `onnx-static-value-evidence.js`, `onnx-type-proto.js` | ML 연산, 컨테이너, 확장 형상 추론 |
| GGUF 블록당 원소·실제 저장 바이트 | `gguf-storage-types.js` | GGUF 메타데이터와 수치 디코더 |
| TFLite 명목 MAC 및 형상 전제 | `tflite_subgraphs.rs` | 최초 분석과 Redesign 재계산 |
| 형상·참조 배열의 원소 동일성 | `array-contract.js` | Core ML, ExecuTorch, ONNX 및 형상 소비자 |

`scalar-types.v1.json`의 54개 이름은 이름별 논리 폭을 기술한다. 같은 비트 수를 갖는 FP8 인코딩을 같은 수치 의미라고 합치지 않는다. 이름이 등록되었다고 해당 포맷·디코더·연산자·가속기가 지원한다고 판단하지 않는다.

JS와 Rust의 실행 함수는 언어 및 정수 범위가 다르므로 서로 호출하지 않는다. 대신 동일한 원본 규칙표를 생성기에 통과시키고, JS는 BigInt, Rust는 checked u128 계산을 사용한다. JS 안전 정수 범위를 넘으면 decimal은 보존하고 숫자 미러는 null이다. Rust의 u128 범위를 넘으면 계산 불가로 남긴다.

## 공통 계산이 지켜야 할 계약

1. **알 수 없음은 0이 아니다.** 형상이 없거나 자료형 폭을 모르면 값과 적용 상태를 함께 돌려준다. 알 수 없는 그래프 간선 크기를 0바이트로 표시하지 않는다.
2. **스칼라와 미상 rank를 구분한다.** 알려진 rank의 `[]`는 원소 하나다. rank가 선언되지 않은 빈 배열은 스칼라의 근거가 아니다. SafeTensors/GGUF의 필수 형상은 해당 어댑터 계약으로 해석한다.
3. **모든 차원을 검증한 뒤 곱한다.** 0차원을 포함하는 유효한 형상은 앞의 큰 곱 때문에 조기에 실패하지 않는다. `[0, -1]` 같은 잘못되거나 미상인 차원은 0으로 숨기지 않는다. JS의 원소가 비어 있는 sparse array도 누락된 차원으로 거부한다.
4. **논리 바이트는 `ceil(elements × bits / 8)`이다.** 원소당 바이트를 먼저 반올림하지 않고 중간 비트 곱의 overflow도 피한다. 실제 블록·행 패딩·희소 저장은 별도 저장 계약으로 계산한다.
5. **동적 batch를 암묵적으로 1로 만들지 않는다.** 명시적으로 허용한 소비자만 직렬화된 batch=1 투영을 사용하고, 결과에 `serialized_batch1_projection` 바인딩을 남긴다.
6. **폭과 양자화 의미는 다르다.** INT8 코드 범위, scale·zero-point의 유효성, per-axis 여부, 디코딩 방법은 각각 검증한다. 8비트 분석 함수에 INT16이 들어오면 폭을 알고 있어도 거절한다.
7. **통합은 공개 계약을 바꾸는 지름길이 아니다.** 기존 `{value, decimal}` 형식과 IR의 `{number, decimal}` 형식은 같은 산술을 사용하되 각 wire 형식을 유지한다. 서로 모순되는 숫자 미러는 거부한다.

## 재현하여 수정한 문제

| 문제 | 수정과 확인 |
| --- | --- |
| 자료형 크기표가 IR·그래프·배치·디코더·WASM에서 서로 달라질 수 있음 | 하나의 선언에서 생성. 생성 파일 수동 수정과 새 로컬 크기표를 검사에서 탐지 |
| 미지원 자료형을 4바이트로 추정하거나, 미상 간선 크기를 0으로 표기 | 근거 없는 기본값 제거, 미상 상태 전달 |
| 큰 형상의 뒤에 0차원이 있을 때 조기 overflow; packed 비트 곱의 불필요한 정밀도 손실 | 모든 차원 검증 후 정확한 곱, 최종 바이트 단위에서 범위 판정 |
| 빈 형상을 무조건 스칼라로 처리 | rank 근거를 공통 판정에 포함; 네이티브 TFLite도 미상 rank 구분 |
| Core ML broadcast에서 0과 1의 결과를 `max`로 계산하여 1로 만듦 | 일반 broadcast 규칙으로 통합하여 0을 보존 |
| Redesign이 MAC을 바꾼 뒤 원본의 보고용 숫자·decimal·상태를 유지 | 공통 MAC 평가로 재계산하고 관련 미러·바이트 상태·연산 집약도 갱신 |
| 잔차 분석의 per-tensor 검사가 비유한·소수·범위 밖 zero-point를 통과시킬 수 있음 | 공통 계약에서 엄격한 숫자·코드 범위·실제 scale 개수 확인 |
| Core ML의 3/6비트 immediate와 LUT histogram이 바이트마다 비트 오프셋을 초기화 | 연속 비트 읽기 함수 공유. UINT3/UINT6 경계 횡단 fixture와 실제 MIL 분석으로 검증 |
| public reference 생성기가 Model IR 파일을 갱신하지 않음 | 생성 대상에 Model IR 추가; BOM의 sibling 파일 해시와 함께 재생성 |
| 오프라인 캐시 검사에서 re-export와 상위 디렉터리 모듈 연결을 누락 | AST로 실제 의존성을 추적. 새 공통 모듈 6개와 기존 누락 모듈 3개를 캐시 가능 목록에 추가 |

Core ML 비트 패킹은 Apple의 `pack_elements_into_bits` / `restore_elements_from_packed_bits` 구현과 대조했다. 연속 LSB 순서로 패킹되므로 UINT3/UINT6 값은 바이트 사이에 걸칠 수 있다. [Apple 구현](https://github.com/apple/coremltools/blob/main/coremltools/optimize/_utils.py)

## 의도적으로 통합하지 않은 것

- **독립 conformance 계산:** production 함수를 그대로 재호출하면 같은 오류를 정답으로 인정할 수 있다. IEEE 변환, 일부 ONNX ML 연산과 기호식 재구성은 독립 검증 경로를 유지한다.
- **포맷 고유 레이아웃:** GGUF 블록 메타데이터, Core ML blob 헤더·padding, SafeTensors F4의 두 값 패킹, ONNX raw/typed payload의 전제는 폭 표로 대체하지 않는다.
- **추정과 관측의 차이:** 모델링된 cache row·latency, 직렬화된 MAC, 실행 trace의 바이트는 같은 이름의 숫자라고 합치지 않는다.
- **표시 전용 코드:** 격리된 ChatGPT/Claude 위젯의 바이트 문자열 포매팅은 각각의 전달 경계를 유지한다. 실제 바이트 계산은 위젯에서 하지 않는다.
- **연구용 합성 입력 진폭:** `research.rs`의 자료형별 진폭 값은 dtype 크기표가 아니다. 본문 해시를 고정한 별도 예외로 기록했다.

남은 JS 동일 본문 계산 후보 12개 그룹은 이유·정확한 본문 해시·발생 위치를 기록했다. 독립 폭 oracle 1개도 본문 해시로 고정했다. 파일명만으로 임의의 새 계산 복사본을 허용하지 않는다. 크기만 같은 포맷 계약이나 서로 다른 evidence class를 억지로 통합하지 않는다.

## 호환성과 기준 산출물

- Artifact IR method는 `2.3.2`, Model IR method는 `1.1.2`로 구분했다. schema는 유지하고 이전 method 읽기 허용을 보존했다.
- 기존 method와 새 method가 같은 해시라고 주장하지 않는다. 공개 예제의 sibling 해시와 consumer fixture 해시를 함께 갱신했다.
- 포맷 어댑터에서 공통 파일로 이동한 계산도 `RULEPACK_SHA256` 입력에 포함한다. 전체 번들 해시뿐 아니라 계산 규칙 자체의 식별자가 새 소유 파일의 변경을 반영하는지 검사한다.
- MobileNet 공개 예제의 의미상 변화는 INT32 양자화 저장 객체 53개의 코드 범위가 미상에서 정확한 범위로 채워진 것이다. 각 범위는 −2147483648…2147483647이다. 모델 구조·MAC·바이트 총량을 임의로 변경한 golden 갱신은 없다.
- 서비스 워커 캐시 식별자를 갱신했다. 배포 파일 작성·게시 여부는 검증 결과와 별개로 확인해야 한다.

## 검증

주요 실행 명령:

```sh
node scripts/generate-scalar-types.mjs --check
node scripts/audit-common-rules.mjs --write
node scripts/check-common-calculation-rules.mjs
node scripts/check-ir-hardening.mjs
node scripts/check-coreml-mlprogram-analysis.mjs
cargo test --lib
cargo test --manifest-path protected/deepbom_wasm/Cargo.toml --lib
node scripts/check-all.mjs
```

공통 계산 검사는 521개 크기·소비자 대조, binary16/BF16의 전체 131,072개 비트 패턴, 독립 Python Fraction oracle로 반올림한 정수 709개, 1~8비트 경계 횡단 읽기 520개를 포함한다. Python oracle은 DEEPBOM 계산을 import하지 않는다. malformed rank·미상/0·동적 binding·broadcast·잘못된 zero-point도 별도로 거절하는지 확인한다.

검증 결과:

| 검증 | 결과 |
| --- | --- |
| 전체 로컬 검사 목록 | 257개 모두 통과. 실패 지점을 수정한 뒤 재개한 세 로그의 완료 항목을 합산하고 누락 여부 확인 |
| Rust 단위 검사 | 주 엔진 130개, 보호된 WASM 엔진 50개 통과 |
| WASM | 주 엔진과 보호된 엔진의 release 빌드·hardening 완료 |
| 공통 수치·IR | 위의 독립 수치 대조, IR hardening 27개 그룹, 공통 IR 의미 15개 그룹 통과 |
| CLI | 6개 포맷, BOM, 비교, 정책 종료, 외부 데이터, 결정론적 출력 검사 통과 |
| 패키지 | 공개 소스로 npm·MCPB·Linux 독립 실행 파일·Python wheel 빌드. CLI·독립 실행 파일·설치한 wheel의 자체 검사 통과 |
| MCP | 패키지에 포함된 서버의 실제 transport, 구조화 출력, 취소·ping·경로 제한 검사 통과. MCPB는 로컬 manifest와 npm 파일 해시 대조 통과 |
| 웹 연결 | 426개 파일의 import/export 대조 통과. 오프라인 그래프 333개 의존성과 서비스 워커 생명주기 검사 통과 |
| 공개 소스 경계 | 공개 소스 1,263개 파일의 구성·해시·라이선스·실행 가능한 import 검사 통과 |

후반부에 변경한 공통 배열·형상 검증과 ONNX ML·Core ML·ExecuTorch 소비자는 별도 표적 검사로 재확인했다. 규칙 해시 생성기와 중복 규칙 검사도 다시 실행했다.

검증의 제한: 외부 exact-channel 및 dynamic-range 회귀 모델이 없어 해당 선택 검사는 실행되지 않았다. 패키지 실행 검증은 Linux에서 수행했다. Windows/macOS 실행이나 실제 가속기 배치·성능을 이번 검사로 검증했다고 주장하지 않는다. MCPB의 외부 공식 검증기를 새로 실행한 것이 아니라 로컬 manifest 계약을 확인했다.

로컬 상세 로그와 합산 결과는 `.local-validation/common-rules/`에 둔다. 임시 빌드 환경은 공개 산출물에 포함하지 않는다. 이번 문서는 수정·검증 결과이며, 원격 푸시나 운영 배포의 증명은 아니다.

## 유지보수 규칙

새 dtype은 원본 규칙표, 네이티브 enum 소속, 저장 레이아웃, 디코더 지원을 각각 검토한다. 새 계산 소비자는 기존 공통 함수의 rank·overflow·applicability 계약을 먼저 확인한다. 전제가 다르면 옵션·반환 상태 또는 네이티브 어댑터로 표현하며, 계산식 복사로 해결하지 않는다. 예외 추가에는 독립 검증 또는 다른 의미라는 구체적 이유가 필요하다.

추가로 대조한 기준: [ONNX int4 패킹](https://onnx.ai/onnx/technical/int4.html), [ONNX 형상 의미](https://onnx.ai/onnx/repo-docs/ShapeAnnotationSemantics.html), [고정 ONNX proto](https://github.com/onnx/onnx/blob/be2b5fde82d9c8874f3d19328bdfe3b6962dc67b/onnx/onnx.proto), [Core ML MIL proto](https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/mlmodel/format/MIL.proto), [SafeTensors 고정 소스](https://github.com/huggingface/safetensors/tree/6eb4dc9a28ebce297606e0f4836bbf28839cacef).
