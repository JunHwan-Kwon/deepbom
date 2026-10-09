# DEEPBOM 전체 IR·공통 흐름 검토

후속 구현: [공통 흐름 통합 기록](COMMON_FLOW_CONSOLIDATION_2026-10-09.md). 아래는 수정 전 검토 결과를 보존한 기록이다.

검토일: 2026-10-09. 기준 HEAD: `b161cd4c383e1133e97200a04c8459935f4c7d97`와
현재 미커밋 작업 트리. HEAD만으로 이번 검토 내용을 재현할 수는 없다.
이번 작업은 코드 추적·기존 검사 재실행·반례 재현이며, 런타임 수정이나 배포 기록이 아니다.

## 판정

주요 경로는 Evidence IR 중심으로 연결되어 있다. 그러나 **전체 기능이 같은 의미 검증과
계산 소유권을 따른다는 판정은 아직 내릴 수 없다.** 두 경로에서 실제 반례가 재현됐다.

1. 저장된 최적화 보고서를 읽을 때 텐서 비교 목록의 완전성·참조·payload 동일성 판정을
   원본/후보 저장 인벤토리와 대조하지 않는다.
2. Explorer의 TFLite Weight Distributions는 공통 Weight IR와 별개로 Rust/WASM
   통계를 계산하며, 모든 값이 0인 UINT8 텐서의 중앙값을 0.498046875로 반환한다.

IR 중심이라는 원칙은 모든 기능이 모든 IR를 순서대로 생성해야 한다는 뜻이 아니다.
파서는 네이티브 사실을 만들고, 모델 변경기는 프레임워크 객체를 변경하며, 실행기는 실제
관측을 만든다. 이들의 결과를 소비할 때 식별·참조·근거 종류·수량·적용 범위를 공통
계약으로 확인하는 것이 핵심이다. 서로 다른 출처나 추정/관측을 억지로 합쳐서는 안 된다.

## 현재 흐름

```text
배포 파일 / 지원되는 묶음
  → 네이티브 파서 및 형식별 사실·계산
  → Artifact IR context
      ├─ Artifact IR: 파일·저장 구조·직렬화 그래프·별도 overlay
      ├─ Model IR v1: 공통 program·계약·저장 바인딩
      └─ Model summary / 호환 소비 view
  → 지원 경로별 감사·비교·시각화·BOM·Web/CLI/MCP

선택적 원본 payload + Model IR
  → Weight IR / 공통 수치 분석
명시적 실행 capture + Model IR
  → Activation IR
외부 메타데이터·파일 + Model IR
  → Provenance IR → 지원되는 외부 표준 투영

로컬 PyTorch / TensorFlow 객체
  → 프레임워크 어댑터 → Model State Snapshot
  → native Model IR v2 / Weight IR v2 / 선택적 Activation IR v2
  → 원본·후보 비교 → optimization_report / optimization_diff
  → Web 보고서·CLI·로컬 MCP·Python PDF

사용자가 실행하는 학습
  → 선택적 수집 / 명시적 logger import
  → Training IR: 시간 좌표·이벤트·고정된 상태와 근거의 참조
  → DeepBoard / 명시적 MLflow·TensorBoard·W&B 투영
```

`optimization_report`, `optimization_diff`, snapshot, event chunk, 결과 manifest는
역할이 다른 지원 문서다. 전부 새로운 분석 IR로 승격하지 않는다. `Trained IR`는 없다.
native v2와 artifact v1 Model/Weight/Activation 계약은 서로 다른 출처를 식별한다.
가족 v2의 union이 기존 소비자에게 새 문서의 자동 호환성을 제공하지는 않는다.

## 경로별 확인

| 경로 | 소유권·연결 | 판정 |
| --- | --- | --- |
| 배포 파일 분석 | `artifact-ir-context.js`에서 Artifact/Model/summary/graph를 연결 | 주요 경로 확인 |
| Web 감사 | `web/app.js`가 분석·파일 해시를 확정하고 context의 소비 view를 전달 | 연결 확인; 아래 호환 facade 한계 존재 |
| npm CLI | `bin/deepbom.mjs`에서 공통 파서·context·출력 빌더 호출 | 연결 확인; 원시 전체 분석 JSON도 별도 출력 계약으로 유지 |
| TypeScript / Python static SDK | 패키지 CLI/검증된 엔진의 호출·오류 매핑 | 별도 분석 엔진 없음 |
| 로컬 stdio MCP | 같은 CLI 위에 인자·경로·실행 한도·결과 전달 | 연결 확인 |
| ChatGPT / Claude | 브라우저 위젯이 같은 worker/context를 사용하고 원격 도구는 제한된 결과를 검증 | 서버가 모델을 재분석하는 구조는 아님 |
| 공통 구조 시각화 | 검증된 Model IR에서 SVG/PNG/manifest 투영 | 연결 확인 |
| Weight workspace / native Weight | 공통 통계·분포·행렬·희소성 모듈 사용 | 연결 확인 |
| Explorer의 레이어별 Weight Distributions | 별도 `compute_weight_histogram` → `weight-hist.js` | **공통화 미완료·수치 반례 확인** |
| 실행 증거 / delegate | 형식별 capture를 바인딩한 runtime sidecar·overlay로 전달 | 추정·지원 후보·실제 관측을 구분하는 검사 통과 |
| TFLite Redesign | WASM의 구조 투영·구현 계획·source-bound scenario ledger | 별도 계획 계약; native 후보 모델과 동일한 결과 형식이 아님 |
| native 모델 최적화 | Python 변환 → 공통 snapshot/비교 → 보고서 | 생성 경로의 소유권 확인; 입력 모델 실행은 명시적 로컬 기능 |
| 저장 보고서 재사용 | 공통 schema/digest/수량 검증 → query/export | **텐서 비교 의미 검증 누락** |
| Training / 외부 logger | 사용자 학습과 기록을 분리; 공통 엔진으로 근거 생성·검증 | 현재 CPU 지원 범위에서 확인 |
| OMOP / Provenance / BOM | artifact Model IR에 결합된 메타데이터를 지원 범위 안에서 투영 | native Training 전체로 연결된 것은 아님 |

## 수정 우선순위와 재현

### F1. 저장 보고서의 텐서 비교 검증 누락 — 우선 수정

대상:

- `web/lib/optimization-report-access.js`: schema, digest, diff, 수량 검사 호출.
- `web/lib/native-optimization-diff.js`: 모듈 비교는 `compareNativeNodes`로 재검산.
- `web/lib/native-optimization-report.js`: 상태 총량·probe 기반 비용·delta 재검산.

그러나 `comparison.tensors`의 ID 집합·행 개수·정렬/상태·`payload_equal`을
`baseline.storage`와 `candidate.storage`에서 재검산하지 않는다.
정상 보고서를 복제하고 다음 변경 후 comparison/report digest를 다시 계산하면 모두 수용된다.

| 반례 | 기대 | 실제 |
| --- | --- | --- |
| 텐서 비교 행 전체 삭제 | 저장 인벤토리와 목록 불일치 거부 | 수용 |
| 비교 대상 ID를 존재하지 않는 이름으로 변경 | 미해결 참조 거부 | 수용 |
| 같은 상태·같은 payload에 `payload_equal=false` 기록 | 동일성 판정 모순 거부 | 수용 |

실제 PyTorch 보고서의 15개 비교 행에서도 확인했고, 하나의 텐서로 만든 독립 fixture로
다시 재현했다. 새 문서 digest를 만드는 것은 작성자 인증 문제가 아니다. 문서 안에 이미
존재하는 근거끼리 모순되는데도 허용한다는 **내부 정합성** 문제다. 원래 digest를
외부에서 고정한 `expected_sha256` 검사는 정상적으로 변경을 차단한다.

영향: 같은 importer를 사용하는 Web·npm CLI·로컬 MCP와 저장 JSON 기반 PDF.
원본 후보 패키지에서 보고서를 생성하는 Python 경로는 전체 비교 결과와 대조하므로
이 문제와 구분해야 한다. 원본 모델 없이 분포 변화의 진위를 다시 계산할 수 있다는
주장도 하지 않는다.

권장 수정: 저장 identity에서 비교할 수 있는 공통 tensor correspondence 함수를 분리해
생성기와 importer 양쪽에서 호출한다. 목록 완전성·중복·foreign ID·shape/dtype 정렬·
payload 동일성은 검증하고, 원래 통계가 없어 재계산할 수 없는 distribution은 별도의
관측값/검증 한계로 남긴다. 해시 확인을 의미 검증으로 대체하지 않는다.

### F2. 기존 Explorer 통계가 공통 Weight 경로를 우회 — 우선 수정

실제 호출:

```text
app-graph-workspace.js: renderOpDetail
  → tflite-worker-rpc / static-audit-worker
  → src/lib.rs: compute_weight_histogram_for_tensor
  → weight-hist.js
```

공통 Weight 경로는 `numerical-ir/statistics.js`, `distribution.js`, `weight-math.js`를
사용한다. 기존 Explorer 경로는 Welford moments, 별도 histogram과 percentile 계산을
Rust 안에서 따로 유지한다. 언어를 달리한 동일 함수 본문 검사는 이 차이를 잡지 못한다.

재현: 저장소의 MobileNetV2 sample을 **메모리로 복사**하고 UINT8 tensor #2의 864개
payload 값을 모두 0으로 변경했다. 원본 파일은 수정하지 않았다.

| 값 | 정확한 결과 | 기존 Explorer 계산 |
| --- | --- | --- |
| 최소·최대 | 0 / 0 | 0 / 0 |
| P5·중앙값·P95 | 모두 0 | 모두 0.498046875 |

기존 코드는 실제 순서통계량 대신 bin 중심을 반환한다. UI는 이를 `median`으로 표시한다.
공통 분포 계산은 bin으로부터 알 수 있는 구간을 반환하며, 이 상수 fixture에서는 정확히 0이다.

함께 확인한 표시·의미 불일치:

- WASM은 `val_min`/`val_max`를 반환하지만 통계 grid는 `min_val`/`max_val`를 읽는다.
  같은 모듈의 box plot은 두 철자를 처리하지만 grid는 그렇지 않다.
- `eff_rank`는 특이값 기반 rank가 아니라 필터 norm 에너지 90%를 차지하는 필터 개수다.
  UI의 `Eff. Rank`와 공통 SVD 분석의 rank를 같은 수치로 해석하면 안 된다.
- tensor 원소 수를 일괄 `params`로 표시한다. 저장 값과 학습 가능한 parameter의 구분이 없다.
- depthwise 여부를 tensor shape로 추정한다. 실제 operator/weight binding을 사용하는
  공통 축 계약으로 옮기는 것이 맞다.

권장 수정: Explorer도 공통 Weight 결과를 선택해 렌더링하도록 연결한다. 단순히 중앙값
공식 하나만 고치면 두 통계 체계가 계속 남는다. 원소·저장 코드·복원 실수·채널 축·rank·
미평가 상태를 공통 계약으로 전달하고 UI별 독자 해석을 제거해야 한다.

### F3. 소비 경계 검사는 유용하지만 완전한 강제 장치가 아니다

`artifact-ir-consumer-policy.v1.json` 기준 60개 direct reader가 분류돼 있고,
UI/report direct surface reader는 예산 0개를 지킨다. 다만 다음 한계가 있다.

- AST 검사는 `analysis.ops`/`analysis.tensors` 모양을 중심으로 탐지한다.
  다른 이름·구조분해·worker의 별도 분석 함수까지 전부 의미 추적하는 검사는 아니다.
- `artifactIrOperators`/`artifactIrValues`는 raw analysis와 IR 소비 view를 모두 허용한다.
  호출 경로의 올바른 전달에 의존한다.
- `primary_view`는 원본 분석의 다른 필드를 getter/setter로 전달한다. 고정된 IR 자체와
  동일한 불변 객체라고 표현하면 안 된다.
- context cache는 analysis 객체·artifact·runtime signature를 기준으로 하며,
  네이티브 내용을 수정하는 경로는 명시적인 invalidation을 지켜야 한다.

이는 이번에 별도 사용자 시나리오의 stale-result 오류까지 재현했다는 뜻은 아니다.
그러나 “정책 검사 통과 = 전체 기능의 IR 우회 없음”이라는 결론을 허용하지 않는다.
UI/export 전용 엄격 selector와 producer/test용 compatibility selector를 구분하고,
예외 worker 경로까지 포함하는 소유권 검사를 보강할 필요가 있다.

### F4. 계약 확장과 채널 지원 범위가 같지는 않다

native v2·Training IR는 추가된 계약이며 기존 artifact 전용 consumer에 자동 전달되지 않는다.
현재의 Provenance source helper는 artifact Model IR 검증기를 사용한다. native snapshot과
Training 전체를 기존 OMOP/CycloneDX/SPDX 경로에 연결한 구현으로 설명하면 안 된다.

또한 새 보고서 접근은 Web·npm CLI·Python·로컬 MCP에서 제공되지만 공개 TypeScript SDK의
전용 report API와 hosted ChatGPT/Claude tool 확장은 별도다. 지원하지 않는 연결을
전체 호환성으로 광고하지 말고 compatibility mapping에서 명시적으로 관리해야 한다.

### F5. 문서와 유지보수 경계

`docs/evidence-ir/README.md`의 Proposed training extension은 수집기와 직접 snapshot
binding이 미구현이라고 설명하지만, 같은 문서 마지막에서는 구현된 native 확장을 설명한다.
현재 상태 안내와 역사적 설계 기록을 분리해야 한다.

Python HTML/PDF와 Web HTML은 공통 report/diff 근거를 쓰지만 표현 코드는 따로 존재한다.
새 필드가 한쪽에서만 누락되지 않도록 내용 포함 여부의 parity 검사가 필요하다.
공통 코어가 `web/lib` 아래 있다는 사실 자체는 계산 중복이 아니지만, DOM 없는 core와
채널 adapter의 의존성 방향을 명시적으로 검사하는 편이 장기 유지보수에 적합하다.
이 문제를 해결하려고 전체 디렉터리를 급하게 이동할 필요는 없다.

## 검사와 한계

이번 기존 검사 재실행에서 확인한 범위:

- 공통 계산 inventory: 애플리케이션 소스 547개, 명시적으로 유지한 폭 oracle 1개,
  이유와 본문 hash를 기록한 동일 본문 후보 12개 그룹. 의미상 중복 전부의 부재 증명은 아니다.
- Artifact/Model import 경계, consumer 정책, family/compatibility 계약.
- 실제 artifact 분석의 결정성·타겟 간 구조 불변·수량 보존.
- Weight/Activation/native 근거·Provenance·runtime profile/배치 계약.
- Model IR 시각화, BOM 대조, SPDX, Redesign 계획 계약.
- SDK·로컬 MCP·원격 MCP 계약, 브라우저/Node/Python diff 동일성.
- 저장 보고서 Web·CLI·로컬 MCP query/export와 실제 PDF 생성.
- PyTorch 2.8.0 CPU·TensorFlow 2.20.0·Keras 3.11.3의 실제 실행: 후보 변환·패키지
  round trip, 수집 전후 학습 비간섭, GradientTape·fit callback, 희소 gradient,
  MLflow local store·TensorBoard 실제 round trip·W&B offline artifact 등 10개 시나리오.

올바른 실행 설정으로 재실행한 기존 JavaScript/채널 검사 30개가 통과했고,
별도의 native framework 검사에서 위 10개 시나리오도 통과했다. 설치된 Python 패키지로
실행했으며, 관련 소스 Python/JS/CSS 22개가 작업 트리와 일치하는 것도 확인했다.

기존 검사가 통과하더라도 F1/F2의 새 반례는 남는다. 일반 회귀검사와 문제 재현 결과를
하나의 PASS로 합치지 않는다. 체크 실행 receipt와 재현 산출물은
`.local-validation/system-flow-review-2026-10-09/`에 저장했다. 이 디렉터리는 Git 제외 대상이다.
최종 집계는 `verification-summary.json`, 실제 CPU framework 결과는
`check-native-training-installed.log`에 기록했다.

검사 runner 설정 중 세 번의 잘못된 실행을 따로 기록했다: diff parity의 HTML 인자 누락,
소스 Python 경로의 패키지 엔진 manifest 부재, venv interpreter symlink를 resolve하여
시스템 Python이 선택된 경우다. 각각 올바른 fixture·설치 패키지·venv 경로로 재실행해 통과했다.
이 세 가지를 제품 회귀로 분류하지 않는다.

수치 반례 재현:

```sh
node .local-validation/system-flow-review-2026-10-09/reproduce-flow-gaps.mjs
```

모든 포맷·모든 입력·모든 GPU/운영체제·모든 학습 구성을 검증한 것은 아니다. 운영 배포를
재검증하거나 성능·정확도를 새로 측정하지 않았다. 프레임워크 모델의 arbitrary code를
정적 artifact plugin으로 실행할 수 있게 한 것도 아니다.

## 완료 조건

1. F1의 모순 세 가지를 모든 report importer가 거부하고, 정상 report는 기존과 동일하게 읽는다.
2. Explorer와 Weight workspace가 같은 대상·표현·계산 방법의 수치를 공유한다.
   통계가 다르면 다른 범위/표현이라는 근거를 표시하고, 빈 값이나 미평가를 0으로 바꾸지 않는다.
3. 채널 adapter가 독자 계산으로 되돌아갈 때 회귀검사가 실패한다.
4. 출처 계약과 optional evidence는 유지하며 native 연결이 없는 exporter를 지원으로 표시하지 않는다.
5. 문서·capabilities·compatibility catalog가 동일한 구현 범위를 설명한다.

이번 검토에서는 위 수정 완료를 주장하지 않는다.

## 후속 수정

위 내용은 수정 전 검토 기록이다. 확인된 우회 경로와 비교 검증의 수정 및 검증 결과는
[Common flow consolidation](COMMON_FLOW_CONSOLIDATION_2026-10-09.md)에 기록했다.
