# 스킬 호출과 선택 (Skill Invocation and Routing)

> 호출은 권한에 대한 판단이 먼저 있고 그다음에 관련성에 대한 판단이 따르는 일이다. 좋은 설명은 모델이 고르는 것을 돕고, 좋은 정책은 그 선택을 허용할지 결정한다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## 학습 목표 (Learning Objectives)

- 사용자가 직접 부르는 호출과 모델이 알아서 고르는 호출, 애플리케이션이 거는 호출, 스킬이 다른 스킬을 부르는 호출을 구분한다.
- 사람에게 보이는지와 모델이 고를 수 있는지를 서로 독립적인 정책 축으로 모델링한다.
- 발동해야 하는 경우와 아슬아슬하게 아닌 경우의 경계를 함께 담은 선택용 설명을 쓴다.
- 자격과 선택, 활성화, 인자 결합, 실행을 실행 기록과 테스트에서 각각 나눈다.
- 런타임마다 다른 호출 관련 필드를, 이식 가능한 프론트매터인 것처럼 내세우지 않으면서 다룬다.

## 문제 (The Problem)

`database-migration` 스킬을 설치했다고 하자. 사용자가 이름을 대고 실행할 수 있지만, 모델도 그 설명을 보고 있다가 누군가 데이터베이스에 대해 일반적인 질문을 하면 그 스킬을 고른다. 그러면 설명만 필요했던 요청에 대고 스킬이 스키마 변경을 제안한다.

사람이 손으로 실행하는 것을 막으려고 `user-invocable: false`를 넣는다. 그런데 다른 런타임에서는 그 필드가 무시된다. 스킬을 아예 사라지게 하려고 `disable-model-invocation: true`를 넣는다. 그 필드를 이해하는 런타임에서도 사용자는 여전히 이름을 대고 부를 수 있다.

필드 이름이 잘못된 것이 아니다. 머릿속 모델이 잘못된 것이다. "사용자가 볼 수 있다", "모델이 고를 수 있다", "애플리케이션이 미리 올릴 수 있다", "그 안의 도구가 실행될 수 있다"는 서로 다른 사실이다. `invocable`이라는 참·거짓 값 하나로는 그것을 표현할 수 없다.

선택에는 또 다른 실패 양상이 있다. 설명이 모호하면 여러 스킬이 그럴듯해 보인다. 설명에 키워드를 잔뜩 채워 넣으면 관련 없는 작업에서도 발동한다. 카탈로그는 확률적인 인터페이스다. 들어갈 만큼 간결하면서도 고를 수 있을 만큼 구체적이어야 한다.

## 개념 (The Concept)

### 수명 주기를 시작시키는 통로는 다섯 가지다

| 주체 | 호출 형태 | 흔한 쓰임 | 주된 위험 |
|---|---|---|---|
| 사람 사용자 | UI나 프롬프트에서 스킬 이름을 댄다 | 의도적인 작업 선택 | 호스트가 주지 않은 접근성이나 권한을 사용자가 기대한다 |
| 모델이나 자율 에이전트 | 작업 맥락을 보고 카탈로그 항목을 고른다 | 자동으로 전문 절차를 꺼내 쓰기 | 엉뚱한 스킬을 고른다 |
| 애플리케이션 | 런타임 코드로 스킬을 활성화하거나 미리 올린다 | 정해진 제품 업무 흐름 | 특정 호스트에 드러나지 않게 묶인다 |
| 다른 스킬이나 하위 에이전트 | 업무 흐름의 의존 대상으로 특정 스킬을 요청한다 | 조합 | 순환, 의존 대상 누락, 맥락이 섞임 |
| 평가 하네스 | 고정된 시나리오에서 특정 스킬을 활성화한다 | 되풀이할 수 있는 측정 | 스킬은 시험하면서 정작 살펴보려던 실무 정책을 건너뛴다 |

이식 가능한 Agent Skills 명세는 패키지를 정의한다. 슬래시 명령 UI 하나나 암묵적 선택 플래그, 애플리케이션 API, 하위 에이전트 수명 주기를 표준으로 정하지는 않는다.

### 호출의 다섯 단계

```figure
skill-invocation-stages
```

다음 낱말을 정확히 쓰라.

- **자격 있음**은 정책이 그 주체에게 그 스킬을 요청할 수 있게 허용한다는 뜻이다.
- **선택됨**은 사용자가 이름을 댔거나 선택기가 관련 있다고 판단했다는 뜻이다.
- **활성화됨**은 그 지시문이 작업 맥락에 들어갔다는 뜻이다.
- **실행 중**은 그 지시문 아래에서 에이전트가 모델 작업이나 도구 작업을 시작했다는 뜻이다.
- **완료됨**은 그 결과물이 독립적인 성공 검사를 통과했다는 뜻이다.

`skill_used=true`만 남기는 기록은 어느 경계에서 실패가 일어났는지 가린다.

### 사람과 모델의 호출은 2×2 표를 이룬다

| 사람이 부를 수 있나 | 모델이 부를 수 있나 | 방식 | 알맞은 예 |
|:---:|:---:|---|---|
| 예 | 예 | 공유 | 코드 설명, 테스트 계획, 문서 검토 |
| 예 | 아니오 | 사람 전용 | 배포 준비, 청구 내역 내보내기, 되돌릴 수 없는 정리 계획 |
| 아니오 | 예 | 모델 전용 | 내부 문체 지침, 분야 참고 자료, 자동 지원 절차 |
| 아니오 | 아니오 | 비활성 또는 애플리케이션 전용 | 단계적 출시, 폐기된 패키지, 코드로 미리 올리는 경우 |

이 표는 정책 모델이지 표준 YAML이 아니다.

어떤 호스트는 사람 전용 행에 `disable-model-invocation: true`를, 모델 전용 행에 `user-invocable: false`를 쓴다. 기본값은 둘 다 허용이다. 다른 호스트는 `agents/openai.yaml`에 `allow_implicit_invocation: false`를 써서, 이름을 댄 호출은 남기고 암묵적 선택만 끈다. 이것들은 런타임 어댑터다. 모르는 호스트는 그냥 무시할 수 있다.

헷갈리는 대목이 중요하다. `user-invocable: false`는 "모델이 이것을 쓸 수 없다"는 뜻이 아니다. 그 필드를 정의한 호스트에서 사용자가 직접 부르는 길을 없앨 뿐이다. `disable-model-invocation: true`도 "스킬이 꺼졌다"는 뜻이 아니다. 모델이 먼저 고르는 길만 없애고 사용자가 이름을 대는 길은 남긴다.

### 직접 호출은 신원이 먼저다

직접 호출은 신원을 곧바로 준다.

```text
/release-readiness v2.4.0
```

또는 이렇게도 쓴다.

```text
release-readiness check v2.4.0 without publishing
```

현재 Codex 인터페이스는 고를 때 `/skills`를 쓰고, 직접 호출할 때는 요청 문장에 스킬 이름을 그대로 적는다고 문서에 적어 두었다. Claude Code는 `/스킬-이름`과 호스트마다 다른 인자 전개를 문서에 적어 두었다. 정확한 문법과 메뉴 노출 여부, 따옴표 규칙, 변수 전개는 호스트의 몫이다.

이름을 댄 요청도 정책을 거친다. 스킬 이름을 댔다고 해서 빠진 권한이나 작업 공간 제약, 승인 관문, 실행 격리를 건너뛰어서는 안 된다.

### 암묵적 호출은 설명이 먼저다

암묵적 선택에서 모델이 처음 보는 것은 본문 전체가 아니라 카탈로그 메타데이터다. 그래서 설명이 곧 그 스킬의 선택용 인터페이스다.

약한 설명은 이렇다.

```yaml
description: Helps with releases.
```

지나치게 넓은 설명은 이렇다.

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

경계가 잡힌 설명은 이렇다.

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

경계가 잡힌 판본에는 다음이 들어 있다.

1. **역량:** 이미 준비된 후보를 살펴본다.
2. **결과물:** 준비 상태 보고서다.
3. **발동하는 경계:** 배포 대상이 준비되었는지 묻는 경우다.
4. **발동하지 않는 경계:** 평범한 빌드와 기능 개발은 범위 밖이다.

발동하지 않는 경계는 인접한 두 스킬이 어휘를 공유할 때 쓸모가 있다. 다만 아슬아슬하게 아닌 요청으로 하는 평가를 대신하지는 못한다.

### 선택은 기권할 수 있는 분류다

스킬 `s`와 요청 `x`에 대해 선택 점수를 이렇게 생각해 보자.

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

실제 점수 매김은 산술이 아니라 LLM의 판단일 수 있다. 그래도 엔지니어링 원칙은 그대로다. 선택은 기준값도 넘어야 하고 경쟁하는 스킬도 이겨야 한다. 근거가 약하면 기권하라.

```figure
skill-routing-abstention
```

결과가 무거운 스킬이라면, 설명이 아무리 좋아도 암묵적 선택이 알맞지 않을 수 있다. 잘못 골랐을 때의 비용이 자동으로 골라 주는 편리함보다 크다면 사람 전용 정책을 쓰라.

### 자격 확인이 순위 매김보다 먼저여야 한다

탐색된 스킬 전부에 점수를 매기고, 가장 높은 것을 고른 다음, 그 하나의 정책만 확인해서는 안 된다. 그러면 막혀 있는 1위 때문에, 자격을 갖춘 아래 순위 후보가 검토조차 되지 않는다.

암묵적 선택은 다음 순서로 하라.

1. 요청한 주체와 지금 쓰는 호스트 어댑터를 기준으로 탐색된 스킬을 걸러 낸다.
2. 자격을 갖춘 후보에만 점수를 매긴다.
3. 기준값과 모호성 규칙을 통과한, 자격 있는 최고 점수를 고른다.
4. 자격을 갖춘 후보가 없거나 그중 어느 점수도 충분히 높지 않으면 기권한다.

`incident-triage`가 `0.80`을 받았는데 그 호스트 확장이 모델 호출을 막아 두었다고 하자. `incident-review`는 `0.55`를 받았고 모델 호출을 허용한다. 선택기는 `incident-review`를 자격 있는 최고 후보로 판단해야 한다. `incident-triage`를 골랐다가 막히고 거기에서 멈춰서는 안 된다.

이 순서는 정책 변경이 관련성 점수의 의미를 바꾸는 것도 막아 준다. 자격은 선택 대상 집합을 정하고, 관련성은 그 집합의 순위를 매긴다.

### 선택 평가에는 아슬아슬하게 아닌 요청이 필요하다

발동해야 하는 경우는 재현율을 증명한다.

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

명백한 부정 사례는 기본적인 정밀도를 증명한다.

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

아슬아슬하게 아닌 요청은 경계의 품질을 드러낸다.

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

이 요청은 배포 스킬과 `package`, `build`라는 단어를 공유하지만 다른 스킬의 몫이다. 뻔한 긍정 사례와 전혀 관련 없는 부정 사례만으로 만든 평가 집합은 품질을 실제보다 좋게 보이게 한다.

### 인자는 세 가지 모습으로 나타난다

호출 인자는 여러 경계를 넘는다.

```figure
skill-argument-boundaries
```

경계마다 의도는 보존하되 텍스트를 코드처럼 다루지 마라.

- 호스트 파서가 명령 문법과 따옴표 규칙을 정한다.
- 스킬은 호스트 규칙에 따라 결합된 텍스트나 변수를 받는다.
- 지시문은 필수 값과 기본값을 검증한다.
- 도구 호출은 그 값을 타입이 정해진 스키마로 바꾸고 다시 검증한다.

날것의 인자를 셸 명령에 끼워 넣지 마라. 인자 배열로 호출하는 스크립트나, 타입이 정해진 MCP 도구를 쓰는 편이 낫다.

### 애플리케이션 호출은 명시적인 조율이다

제품이 업무 흐름상 이미 그 작업 유형을 알고 있다면 스킬을 직접 활성화할 수 있다. 예를 들어 풀 리퀘스트 검토 서비스는 사용자가 검토 버튼을 누른 뒤에 `pull-request-risk-review`를 미리 올릴 수 있다.

이렇게 하면 선택의 불확실성은 사라지지만 런타임 API에 대한 의존이 생긴다. 그 어댑터는 이식 가능한 본문 바깥에 두어라.

```figure
skill-host-adapter
```

다른 호환 클라이언트가 열어 봐도 그 스킬의 내용은 그대로 읽혀야 한다.

### 스킬이 스킬을 부르는 것은 도구 호출에 가까운 간선이다

의존성 파일이 바뀌었을 때 `release-readiness`가 `security-change-review`를 요청한다고 하자.

부르는 쪽은 다음을 제공해야 한다.

- 대상 스킬의 신원.
- 범위가 정해진 작업과 산출물 경로.
- 기대하는 응답 계약.
- 왜 부르는지에 대한 이유.
- 쓸 수 없을 때의 대안.
- 최대 깊이나 순환 방지 규칙.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

두 번째 스킬을 첫 번째 스킬에 그대로 붙여 넣는 것이 아니다. 그것을 어떻게 활성화할지, 맥락을 공유할지, 갈라진 곳에서 돌릴지, 도구 실행 결과로 돌려줄지는 호스트가 정한다.

### 맥락의 수명 주기는 호스트마다 다르다

활성화된 뒤 스킬 본문은 대화에 그대로 남을 수도 있고, 맥락을 압축할 때 요약될 수도 있고, 위임된 별도 맥락에서 돌 수도 있다. 도구 사용 허가는 한 턴만 유지되는데 지시문은 더 오래 남을 수도 있다. 하위 에이전트는 부모의 전체 대화 없이 그 스킬만 받을 수도 있다.

눈에 보이지 않는 수명을 전제로 하는 스킬은 쓰지 마라. 오래 남겨야 할 결과는 파일이나 타입이 정해진 상태에 두고, 다시 들어와도 안전하게 만들고, 중단된 뒤에 무엇을 다시 읽어야 하는지 명시하라.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 직접 만들기 (Build It)

`code/main.py`에서 정책과 선택을 서로 다른 어댑터로 구현한다.

모델 쪽에는 다음이 들어간다.

- `Actor`. 사람과 모델, 자율 에이전트, 애플리케이션, 스킬, 평가 하네스를 가리킨다.
- `SkillMetadata`. 선택에 쓰이는 신원이다.
- `InvocationPolicy`. 사람과 모델의 2×2 표를 담는다.
- `InvocationRequest`와 `InvocationDecision`. 추적할 수 있는 입력과 결과다.
- `CorePolicyAdapter`. 호스트 확장 없이 이식 가능한 동작만 다룬다.
- `ExtensionPolicyAdapter`. 인식하는 런타임 필드를 다룬다.
- `build_invocation_matrix(policy)`. 2×2 표를 만든다.
- `route_request(skills, request, adapter)`. 관련성 순위를 매기기 전에 자격으로 거르고, 선택과 거부를 수행한다.

다음으로 실행한다.

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

데모는 표 하나와 함께, 사람이 이름을 댄 경우와 모델이 고른 경우, 자율 에이전트와 애플리케이션, 스킬 조합, 평가 하네스 각각의 판단을 출력한다. 확장 어댑터 결과에서는, 어휘상 1위였던 후보가 막혀서 걸러진 뒤에 자격 있는 대안이 순위에 오르는 모습을 보여 준다. 이름을 정확히 적은 허용 목록도 함께 들어 있다. 모델 API는 필요하지 않다. 이 결정적 선택기는 정책 경계를 들여다볼 수 있게 하려고 만든 것이지, 어휘 일치가 실무 모델의 선택을 재현한다고 주장하려는 것이 아니다.

### 핵심 어댑터와 확장 어댑터를 나누는 이유

파서 하나가 눈에 보이는 모든 프론트매터 필드에 의미를 붙이면, 런타임의 관행이 조용히 가짜 표준으로 승격된다. 어댑터를 나누면 부르는 쪽이 지금 어느 호스트의 의미 체계를 쓰는지 밝히게 된다.

`CorePolicyAdapter`는 애플리케이션이 준 정책만 쓴다. `ExtensionPolicyAdapter`는 명시된 호스트 필드 집합을 인식하고, 어떤 필드가 판단을 바꿨는지 기록한다.

## 실제로 써 보기 (Use It)

스킬을 공개하기 전에 호출 계약을 써 두라.

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

이 계약은 어댑터와 테스트를 위한 설계 문서다. 어떤 표준이 그것을 명시적으로 채택하지 않는 한, 이식 가능한 `SKILL.md` 프론트매터가 아니다.

## 결과물로 남기기 (Ship It)

이 레슨은 `skill-invocation-router` 묶음을 만들어 낸다. 호출 모델 참고 문서와 호스트 정책 예시, 그리고 아무것도 실행하지 않는 명령줄 도구가 들어 있다. 그 도구는 사람과 모델, 자율 에이전트, 애플리케이션, 스킬 조합, 평가 하네스 가운데 하나의 요청을 평가해서, 통로와 어댑터, 점수, 이유를 담은 JSON 판단을 돌려준다.

요청 하나를 평가하는 이 도구는 정책을 찔러 보는 탐침이지, 발동 여부를 온전히 평가하는 도구가 아니다. 혼동 행렬 개수와 정밀도, 재현율, 되풀이 실행의 안정성을 계산하려면 27번 레슨의 레이블된 긍정 사례와 아슬아슬하게 아닌 사례 설계를 쓰라.

## 연습 문제 (Exercises)

1. 사람과 모델의 2×2 표에 해당하는 네 가지 경우를 모두 만들고, 각각에 대해 정당한 쓰임을 하나씩 써라.
2. `CorePolicyAdapter`에 애플리케이션 전용 활성화를 추가하라. 사람과 모델의 호출이 여전히 막히는지 증명하라.
3. 배포 스킬에 대해 아슬아슬하게 아닌 요청 열 개를 써라. 각 요청은 그 스킬과 어휘를 공유하면서도 다른 업무 흐름에 속해야 한다.
4. 상위 두 선택 점수 사이에 모호성 여유를 두어라. 그 여유가 너무 작으면 `ask`를 돌려주게 하라.
5. 스킬이 스킬을 부를 때의 최대 조합 깊이를 추가하고, 스킬 두 개가 서로를 부르는 순환을 잡아내라.
6. 같은 레이블 집합을 핵심 어댑터와 확장 어댑터에 각각 돌려라. 달라진 판단을 하나하나 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 쓰는 뜻 | 정확한 뜻 |
|---|---|---|
| Explicit invocation | "슬래시 명령" | 어떤 주체가 스킬의 신원을 직접 대며 호출하는 것이며 정책을 거친다 |
| Implicit invocation | "모델이 고른다" | 선택기가 작업 맥락을 보고, 자격을 갖춘 카탈로그 메타데이터 가운데 고르는 것 |
| User-invocable | "사람이 쓸 수 있다" | 호스트마다 다른 메뉴나 직접 호출에 대한 성질이며 핵심 표준 필드가 아니다 |
| Model-invocable | "에이전트가 쓸 수 있다" | 호스트 정책 아래에서 모델이 암묵적으로 고를 수 있는 자격 |
| Invocation adapter | "프론트매터 파서" | 호스트의 필드와 API를 명시한 정책 모델로 옮기는 코드 |
| Near miss | "어려운 부정 사례" | 그 스킬이 받을 법한 입력과 닮았지만 발동해서는 안 되는 요청 |
| Abstention | "고른 스킬이 없음" | 근거가 없거나 모호할 때 의도적으로 내리는 선택 결과 |

## 더 읽을거리 (Further Reading)

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions). 발동 조건과 구체성, 평가를 다룬다.
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills). 발동 평가와 출력 평가의 설계를 다룬다.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills). 현재 Codex의 직접 호출과 암묵적 호출 제어를 다룬다.
- [Claude Code skills](https://code.claude.com/docs/en/skills). 한 호스트의 `user-invocable`과 `disable-model-invocation`, 인자, 위임된 맥락을 다룬다.
