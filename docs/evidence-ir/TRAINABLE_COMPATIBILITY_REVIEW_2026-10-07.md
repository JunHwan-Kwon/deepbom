# 모델 최적화·Training 근거 확장: 호환성 재검토

검토일: 2026-10-07. 대상은 현재 작업 폴더의 구현과 Training 설계 초안 0.5다.
배포본에 대한 인증이나 모든 입력에서의 무결함 보장이 아니다.

## 1. 판단

**모델 최적화, 사용자 학습, 선택적 관찰, 근거 표시를 분리하는 설계는 합리적이다.**
사용자는 후보를 비교·선택하고 그 모델 상태로 새 optimizer를 구성한다. 수집기는
원본 모델에도 독립적으로 연결하며 학습을 시작하거나 재개하지 않는다. DeepBoard와
외부 로거는 공통 근거의 소비자이며 별도의 분석 엔진이 아니다.

다만 현재 artifact 중심 IR에 live tensor와 학습 이벤트를 바로 넣을 수는 없다.
native 구조/상태 adapter, 명시적인 source 전환, Training schema와 실제 프레임워크
검증이 선행돼야 한다. 기존 schema의 필수 식별자를 생략하는 우회는 허용하지 않는다.
이번 변경은 이 조건을 설계에 보완했으며 새 최적화·Training 런타임을 구현하지 않았다.

## 2. 현재 구현에서 확인한 경계

| 공통 owner | 실제 계약 | 확장 시 지킬 조건 |
| --- | --- | --- |
| [Evidence IR family](../../web/lib/evidence-ir.js) | Artifact, Model, Weight, Activation, Provenance의 5종 | Training은 아직 등록되지 않음; 임의 문서를 기존 member로 표시하지 않음 |
| [근거 source](../../web/lib/ir-evidence-contract.js), [수치 source 검증](../../web/lib/numerical-ir/common.js) | 검증된 Model IR와 실제 artifact/set 식별자 | RAM snapshot 직접 결합은 새로운 계약 필요 |
| [Weight IR](../../web/lib/weight-ir.js) | 분석하는 바이트의 전체 해시가 source와 일치해야 함 | 바뀐 live weight를 이전 checkpoint의 수치로 기록하지 않음 |
| [Activation IR](../../web/lib/activation-ir.js) | 입력·run·capture·subject·coverage 검증, mapped value 중복 거부 | 반복 호출/학습 시점을 한 capture에 합치지 않음 |
| [수치 상세 비교](../../web/lib/numerical-details.js) | 같은 Model IR에 결합된 capture끼리 비교, 맥락 일치 여부 별도 반환 | 다른 checkpoint의 비교는 상태·subject 대응부터 필요 |
| [Weight 분석](../../web/lib/weight-analysis.js) | 일대일 tensor 대응과 지원되는 축 정렬, 불완전 비교 표시 | fusion의 다대일 관계를 원소별 동등성으로 오인하지 않음 |
| [Provenance IR](../../web/lib/provenance-ir.js) | 출력의 주대상은 Model IR와 artifact에 결합 | run-only Training 근거에 가짜 artifact를 만들지 않음 |
| [Node SDK](../../sdk/index.mjs), [Python SDK](../../channels/python/src/deepbom/api.py) | 현재는 파일 기반 공통 CLI 엔진 호출 | RAM 모델 API는 별도 선택적 native 모듈; 기존 SDK가 임의 코드를 자동 실행하지 않음 |

공통 수치 owner는 기존 `numerical-ir/statistics.js`, `exact-moments.js`,
`distribution.js`, `tensor-size.js`를 재사용한다. 프레임워크 adapter는 값·dtype·
shape·역할·관찰 범위를 전달한다. 지원되지 않는 인코딩이나 큰 정수를 Python/JSON/
JavaScript 경계에서 몰래 cast하지 않는다. 정수 정확도·nonfinite·표본 수·분모·
계산 방법을 보존해야 같은 엔진을 쓴다는 의미가 성립한다.

## 3. 보완한 설계 문제

| 위험 또는 모호함 | 보완한 규칙 | 실제 구현의 합격 조건 |
| --- | --- | --- |
| fusion이라는 말로 구조 변경·eval folding·delegate를 혼합 | 세 변환 종류와 전제 조건 분리 | 학습/추론 모드와 BN buffer 조건을 틀리게 주면 적용 거부 |
| module 구성, 실행 trace, export 그래프를 같은 구조로 취급 | 구성·호출·배포 그래프 및 관찰 범위 분리 | 공유 모듈·반복 호출·동적 분기·export 변경을 명시적으로 대응 |
| 모델 복제만으로 state 보존을 가정 | parameter/buffer/extra state/trainability/alias를 대응표에 기록 | 원본과 후보 storage 격리, 후보 내부 공유 관계 보존 또는 변경 표시 |
| Keras 파일 로딩이 이전 학습 설정을 이어받음 | 제안 로더는 `compile=False`, 새 optimizer는 사용자 소유 | 모델 state 복원과 compile/optimizer 복원 분리, 변경된 variable 이름 대응 |
| 검증 forward/backward가 후보를 변경 | 검증은 격리한 작업 복사본에서 실행 | BN buffer·gradient·RNG·optimizer 변화가 확정 후보에 남지 않음 |
| `result.model` 학습 후 과거 보고서를 현재 근거로 사용 | mutable 작업 객체와 immutable 후보 패키지 분리 | 기존 export는 기존 snapshot 유지; 변경 상태는 새 식별자로 기록 |
| RAM 상태를 v1 artifact source에 억지로 삽입 | source 종류·필수 식별자·정규화 방법을 명시적으로 개정 | 구형 소비자는 미지원 계약 거부; 변환 시 손실과 새 digest 보존 |
| 반복 activation/다른 step을 같은 value로 덮어씀 | 현재 capture 제한 유지, 호출/시점별 참조 필요 | 중복 capture 거부와 native 호출 대응 검증 |
| 같은 이름·shape를 의미상 같은 tensor로 취급 | 대응 근거와 비교 가능성을 독립적으로 보고 | 일대다/다대일/미대응 tensor에 거짓 수치 diff 없음 |
| 부분 관찰을 완전한 Training snapshot으로 해석 | 수집 범위·누락·sampling·queue 탈락·state 결합 상태 유지 | 불완전 기록도 조회 가능하지만 전체 상태 결합을 발명하지 않음 |
| 지연 도착·부분 파일·중복 이벤트를 정상 완료로 계산 | chunk 확정 순서와 event 충돌 검사 | 파일 쓰기 중단/동일 ID 상이 내용/지연 도착을 구분 |
| exporter 재시도를 새 관찰로 계산 | canonical 관찰과 전송 ledger 분리 | 응답 유실에 따른 remote 중복을 관찰 증가와 구분 |
| MAC·논리 메모리 감소를 실제 성능 향상으로 표시 | 계산량·저장량·실측 memory/latency/cache/power를 분리 | 동일 측정 조건에서만 성능 비교; 미측정은 0 아님 |
| 형식 지원을 모든 framework/backend 지원으로 확대 | 지원 조합과 각 검증 단계별 상태 공개 | 코드 생성·로드·forward·backward·gradient coverage·갱신·품질을 구분 |
| Python에서 공통 엔진으로 전송하며 정밀도·layout 의미가 변함 | native bridge의 dtype/shape/storage/encoding 및 부분 범위 계약 | 기존 수치 oracle와 대조; 전송·chunk 경계로 값/분모/coverage가 변하지 않음 |

정확한 규칙은 [입력·변환·출력 계약](../../examples/integrations/TRAINABLE_ENTRY_POINTS.ko.md)과
[Training lifecycle 초안](TRAINING_LIFECYCLE.md)에 반영했다. 위 표의 합격 조건은
후속 구현을 위한 조건이며 현재 통과한 native framework 테스트 목록이 아니다.

## 4. 공식 자료로 확인한 이유

PyTorch eval Conv–BN folding은 eval mode와 계산된 running buffer를 전제한다.
Keras BatchNormalization 역시 training 여부에 따라 사용하는 통계가 다르다.
따라서 추론 등가 변환을 학습 동작의 동등성으로 확대할 수 없다.
[PyTorch eval folding](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.fuse_conv_bn_eval.html),
[Keras BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/)

PyTorch `state_dict`는 얕은 복사로 tensor 참조를 포함한다. Keras `clone_model`은
새 weights를 생성하며 모든 shared object의 유일성을 보존하는 기능도 아니다.
따라서 독립 snapshot, 저장된 state 로딩, alias 대응을 따로 검증해야 한다.
[PyTorch Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html),
[Keras clone_model](https://keras.io/api/models/model_saving_apis/model_config_serialization/)

Keras 파일은 optimizer 상태를 포함할 수 있고 `load_model`의 compile 기본값은
true다. 모델 state에서 새 학습을 시작하는 계약에 맞춰 로더를 명시적으로 설계해야
한다. 같은 문서는 재로딩 후 variable 이름이 바뀔 수 있음도 설명한다.
[Keras 저장·로딩](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/)

MLflow의 artifacts, TensorBoard의 summaries, W&B의 Artifacts는 서로 다른 소비
형식이다. 원문 IR 파일·digest와 표시 투영을 분리하는 것이 연결 가능한 공통 경계다.
이들 서비스가 DEEPBOM IR를 native로 해석하거나 같은 전송 보장을 제공한다는
뜻은 아니다. [소비자별 연결 계약과 공식 자료](TRAINING_LIFECYCLE.md#85-하나의-근거-선택-가능한-표시저장-대상)

## 5. 실행한 검증과 한계

아래는 이번 재검토 중 현재 checkout에서 실행한 검사다. 새 API의 실행 성공과
현재 공통 엔진의 회귀 통과는 구분한다.

| 검사 명령 | 결과와 범위 |
| --- | --- |
| `node scripts/check-evidence-ir-family.mjs` | 통과: 5 member schema, 식별·digest·참조 및 폐기된 이름 거부 |
| `node scripts/check-numerical-ir.mjs` | 통과: 수치 통계/정수·저장값 decoding·capture/coverage·변조 검증 |
| `node scripts/check-numerical-details.mjs` | 통과: 186 percentile oracle, 분포/flow/정체성/누락 및 CLI·Node·Python 경로 |
| `node scripts/check-evidence-compatibility.mjs` | 통과: catalog 0.3.0, 40 endpoints, 48 mappings, 기존 snapshot 및 owner digest |
| `node scripts/check-sdk.mjs` | 통과: 공통 엔진 결과·오류·타입·프로세스 경계와 SDK 일치 |
| `python3 scripts/check-python-api.py` | 통과: Python facade 계약; native 모델 학습 검증 아님 |
| `node scripts/build-public-source-export.mjs` | 통과: 공개 소스 allowlist 1,331개; 새 검토 문서 포함 |

관련 문서 6개의 상대 링크/앵커 76개와 Python 예시 11개의 문법도 확인했다.
문법 검증은 예시의 미구현 API를 실행한 검증이 아니다.

추가로 합성 ONNX `Add` fixture에서 10개 경계를 확인했다. source 누락, 임의
snapshot source, 잘못된 artifact identity, 임의 training step, 중복 mapped capture,
graph input 누락, 다른 weight 바이트, Model IR 없는 Provenance, 미등록 Training
member의 9개 입력을 거부했고, 선택적 분석 전후 Model IR는 동일했다. 실제 추론이나
학습을 수행한 fixture가 아니다. 로컬 재현 자료는
`.local-validation/training-design-review-2026-10-07/boundaries.mjs`와 `boundaries.json`이다.
이 로컬 자료는 공개 배포 파일에 포함하지 않는다.

확인한 기본 Python과 SDK 검증 가상환경에는 torch/tensorflow/keras가 없었다.
이번에 두 프레임워크의 실제 후보 round-trip·학습·collector 비간섭 검증을 했다고
주장할 수 없다. 설계만 보완했으므로 기존 runtime/schema/catalog는 변경하지 않았다.
일반적인 임의 Python 모델의 모든 부작용·모든 변환·모든 학습 경로를 보장하지 않는다.

## 6. 구현 순서와 완료 조건

두 경로는 독립적으로 진행하되 source와 계산 owner를 공유한다.

1. **공통 계약:** Training 관찰 schema/validator와 native Model State Snapshot
   정규화·source 전환을 정의한다. 관찰과 모델 상태를 동일한 snapshot으로 혼동하지
   않는다. 부분 로그 import에는 완전한 모델 저장을 요구하지 않는다.
2. **모델 경로:** PyTorch eager·TensorFlow backend Keras에서 명시한 작은 지원
   조합부터 입력/복원/subject 대응을 검증한다. 지원 변환별 baseline·후보·상태 이식·
   격리 검증·새 프로세스 재로딩·전후 diff가 끝난 뒤 공개한다.
3. **관찰 경로:** 기존 기록 import와 artifact에 결합된 진단을 먼저 연결한다.
   live state source를 지원한 뒤 `fit`, eager `GradientTape`, PyTorch 사용자 루프를
   각각 실제 학습과 비교한다. compile/distributed/AMP는 검증된 조합만 추가한다.
4. **소비자 경로:** 같은 저장 근거를 DeepBoard와 MLflow/TensorBoard/W&B에 표시한다.
   원문 보존·투영 손실·중복/순서·오프라인 조회를 검증하고 각 adapter를 선택적으로 설치한다.

새 기능의 공개 기준은 [모델 경로 수용 검사](../../examples/integrations/TRAINABLE_ENTRY_POINTS.ko.md#8-구현-완료-기준)와
[Training 경로 수용 검사](TRAINING_LIFECYCLE.md#수용-검증-항목)다. 문서/예시 완료,
schema 검증 완료, 실제 프레임워크 실행 완료, 채널 배포 완료를 별도로 기록한다.

## 7. 추가 변경의 하위 호환성 확인

같은 날짜에 직전 2.1.0 소스 `b161cd4c383e1133e97200a04c8459935f4c7d97`과
현재 작업 폴더를 직접 비교했다. 변경된 소스만 보는 것에 더해 기존 호출을 양쪽에서
실행했다. 검증 범위에서 기존 계약을 깨는 변경은 발견하지 못했다.

| 비교 범위 | 결과 |
| --- | --- |
| ONNX·TFLite·GGUF·SafeTensors 각 summary/envelope/json-compact | 12개 호출의 stdout·stderr·종료 코드 동일 |
| ONNX CycloneDX/SARIF, 기존 수치 section, self diff, contract capture, 해시 불일치, 빈 파일 오류 | 7개 호출 동일; 오류 종료 코드 4/1 유지 |
| 기준 commit에 있는 공개 JSON Schema 10개 | 파일 바이트 동일; 새 numerical-details schema는 별도 추가 |
| 기존 호환성 snapshot 3개 | 파일 바이트 동일; 새 0.3.0 snapshot과 index 추가/갱신 |
| Node 공개 export / Python 기존 공개 함수 인자 | 기존 항목 유지; `numericalEvidence` / `numerical_evidence`만 추가 |
| 로컬 stdio MCP 도구 4개 | 기존 input/output schema·annotations 유지; `activation_baseline` 선택 인자 추가 |
| SDK·IR·호환성·수치 상세·Python·MCP·Redesign 계약 검사 | 통과 |
| Chromium 수치 분석·Redesign 화면 검사 | 통과: light/dark/mobile, opt-in, capture import, 상태 초기화, 코드/시각화 내보내기 포함 |

선택적 수치 분석의 전체 JSON에는 `numerical_details`가 추가된다. 기존 수치 section을
명시해 요청한 결과는 위 비교에서 동일했다. 기존 JSON을 exact-key 전용으로 읽는
소비자는 추가 필드 허용이라는 SDK 계약을 따라야 한다. 새 API·section이 구형
패키지에도 존재한다는 뜻은 아니다.

MCP에서 `activation_evidence`만 지정하고 출력 형식을 생략한 호출은 기존에
summary 기본값 때문에 CLI가 거부했다. 현재는 JSON을 선택해 실행되도록 보완됐다.
기존에 성공하던 명시적 JSON 호출의 계약을 바꾼 것이 아니며, 명시적 미지원 출력
조합은 계속 오류로 처리한다.

비교 자료는 `.local-validation/backward-compatibility-2026-10-07/`의 `compare.py`,
`result.json`, `mcp-contract.py`, `mcp-contract.json`에 보관했다. 기준 source는
분리해 추출했으며 양쪽에서 동일한 WASM·의존성과 생성된 build metadata를 사용했다.
따라서 이는 source 변경의 회귀 검사이며, 별도 배포 바이너리·모든 OS·모든 모델·
현재 원격 ChatGPT MCP 서버에 대한 검증으로 확대하지 않는다.

현재 package 표시는 여전히 2.1.0이다. 새 SDK 메서드는 minor 추가 기능이므로 실제
배포 때는 [SDK 버전 정책](../SDK_CONTRACT.md#version-policy-and-checks)에 따라
새 minor 버전으로 게시해야 한다. 기존 2.1.0을 새 기능이 포함된 버전처럼 안내하거나
같은 버전의 배포 파일을 덮어쓰지 않는다. Training/native snapshot 확장은 여전히
설계 단계이며 기존 IR에 미리 활성화하지 않았다.
