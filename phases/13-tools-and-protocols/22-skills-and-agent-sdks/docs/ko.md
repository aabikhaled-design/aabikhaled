# 에이전트 스킬: 이식 가능한 계약과 런타임 경계 (Agent Skills: Portable Contract and Runtime Boundary)

> 스킬은 파일 이름을 잘 붙인 긴 프롬프트가 아니다. 런타임 계약을 거쳐 에이전트의 맥락으로 들어오는, 찾아낼 수 있는 지시와 자원과 실행 가능한 도우미의 꾸러미다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- 에이전트 스킬을 프롬프트, 저장소 지침, 도구, 훅, 서브에이전트, 플러그인과 헷갈리지 않고 정의한다.
- 이식 가능한 `SKILL.md` 계약을 읽고 런타임별 확장과 분리한다.
- 탐색, 선택, 활성화, 자원 적재, 도구 사용, 검증을 각기 다른 생애주기 단계로 설명한다.
- 런타임이 에이전트 카탈로그에 올리기 전에 스킬 꾸러미를 검증한다.
- 구체적인 과제 앞에서 스킬, MCP 도구, 훅, 서브에이전트, 평범한 코드 중 무엇을 쓸지 고른다.

## 10분 만에 첫 성공 (Ten-Minute First Success)

긴 설명을 읽기 전에 이것부터 해 보라. 작은 스킬을 하나 만들고, 검토용 묶음
전체를 진짜 에이전트 호스트에 설치하고, 불러 쓰고, 결과를 확인하고, 지운다.
이 과정이 관측 가능한 결과로 생애주기를 증명해 준다.

### 실제 호스트 실습을 위한 사전 점검 (Preflight for the real-host lab)

실제 호스트 확인 지점에는 Node.js, `npx`, Python 3, 스킬을 다룰 수 있는
호스트 하나, 그리고 설치 프로그램에서 고르는 프로젝트 범위나 사용자 범위에
대한 쓰기 권한이 필요하다. 지역 명령부터 확인하라.

```bash
node --version
npx --version
python3 --version
```

설치하기 전에 어떤 호스트와 범위를 쓸지 정하라. 준비물 중 하나라도 없다면
이 레슨을 웹사이트에서 읽거나 아래의 수동 꾸러미 연습으로 이어 가라. 그
대안은 계약을 가르쳐 주지만, 호스트 탐색과 호출, 함께 담긴 스크립트
실행, 제거 동작을 증명해 주지는 않는다. 그 관찰 항목들은 미확인으로
표시해 두라.

### 1. 빈 작업 디렉터리에서 시작한다 (1. Start in an empty working directory)

학습용 작업을 모아 두는 아무 상위 디렉터리에서 이 명령을 실행하라.

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

마지막 명령은 아무것도 찍지 않아야 한다. 파일이 찍힌다면 검토 경계가
분명해지도록 다른 빈 디렉터리를 고르라.

첫 스킬을 담을 디렉터리를 만든다.

```bash
mkdir -p my-first-skill
```

`my-first-skill/SKILL.md`를 이 내용으로 만든다.

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

의도한 디렉터리에 파일을 만들었는지 확인한다.

```bash
test -f my-first-skill/SKILL.md
```

출력이 없고 종료 코드가 0이면 파일이 있는 것이다.

### 2. 검토용 묶음 전체를 설치한다 (2. Install the complete reviewer bundle)

`agent-skills-first-run`에 머문 채로 실행한다.

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

여러분이 쓰는 에이전트 호스트와 범위를 고르라. 설치 프로그램은
`skill-contract-reviewer`와 그것을 쓴 목적지를 알려 줘야 한다. 이 레슨의
스킬은 참고 문서와 스크립트, 자산을 품은 중첩 묶음이므로 `--full-depth`가
필요하다.

`SKILL_ROOT`를 설치 프로그램이 알려 준 절대 디렉터리로 맞추라. 그것은 설치된
`SKILL.md`가 있는 디렉터리여야 하며, 레슨 원본 디렉터리도 현재 작업 공간도
아니다.

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

에이전트 세션이 이미 열려 있었다면 새 세션을 시작하거나 그 호스트의 스킬
재검색 명령을 쓰라. 모든 호스트가 카탈로그를 즉시 다시 읽는다고 가정하지 마라.

### 3. 명시적으로 불러 쓴다 (3. Invoke it explicitly)

설치된 에이전트에서 `agent-skills-first-run`을 작업 디렉터리로 삼고, 그
호스트가 지원하는 문법을 쓰라.

| 호스트 | 명시적 호출 |
|---|---|
| Codex | `skill-contract-reviewer`, 또는 `/skills`에서 고른 다음 검토 요청을 적는다 |
| Claude Code | `/skill-contract-reviewer`에 이어 검토 요청을 적는다 |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

요청에는 `SKILL_ROOT`와 `TARGET_ROOT`로 찍힌 절대 경로 값을 쓰라. 호스트가
실행 전에 그것을 펼쳐 보이게 하고, 프로세스 작업 디렉터리에 기대는 명령이
아니라 정확히 풀어낸 명령을 보여 달라고 하라.

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

풀어낸 명령은 자리표시자가 하나도 남지 않은 이런 모양이어야 한다.

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

성공한 결과에는 세 성질이 모두 있다.

1. 호스트가 `skill-contract-reviewer`를 이름으로 찾아낸다.
2. 검토기가 꾸러미 계약을 읽고 함께 담긴 검증기를 실행한다.
3. 응답에 이 예제에 대한 구조적 오류가 없는 검증 보고서와, 근거를 갖춘 기본
   요소 선택이 담긴다.

실행 증거에는 스크립트 경로, 대상 경로, cwd, 정확한 인자 벡터, 종료 코드도
들어 있어야 한다. 그 항목들이 없는 매끄러운 보고서는 설치된 동반 스크립트가
실제로 돌았다는 증명이 되지 않는다.

호스트가 스킬을 쓸 수 없다고 알리면 설치 목적지를 확인하고, 한 번 재검색하거나
재시작한 다음, 명시적 요청을 다시 보내라. 설치 실패를 감추려고 스킬 설명을
고쳐 쓰지 마라.

### 4. 암묵적 선택을 시험한다 (4. Probe implicit selection)

새 에이전트 턴을 시작해 스킬 이름을 대지 않고 같은 과제를 적어 보라.

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

호스트가 선택된 스킬을 보여 준다면 `skill-contract-reviewer`를 골랐는지
기록하라. 호스트가 라우팅을 드러내지 않는다면 암묵적 선택은 미확인으로
표시하라. 명시적 호출이 이식 가능한 대안이다.

### 5. 정리한다 (5. Clean up)

설치한 검토 묶음만 지운다.

```bash
npx skills remove skill-contract-reviewer
```

설치할 때 쓴 것과 같은 호스트와 범위를 고르라. 재검색하거나 새 세션을 연 뒤
`skill-contract-reviewer`를 명시적으로 요청하면 쓸 수 없다고 나와야 한다.
이후 레슨을 위해 `my-first-skill`은 남겨 두거나, 이 과정을 마친 뒤에 실습
디렉터리를 지우라.

## 문제 (The Problem)

여러분의 팀에 믿을 만한 릴리스 작업 흐름이 있다고 하자. 병합된 변경을 찾고, 마이그레이션 메모를 확인하고, 변경 기록을 갱신하고, 패키징 명령을 돌리고, 검토 점검표를 만들어 낸다.

그 작업 흐름을 프롬프트 하나에 담으면 붙여넣기는 쉬워지지만 운영하기는 어려워진다. 그 프롬프트에는 안정적인 신원도, 탐색 규칙도, 자원 경계도, 시험할 수 있는 꾸러미 모양도 없고, 기본적인 질문에 답도 없다. 누가 불러 쓸 수 있는가? 모델은 언제 그것을 골라야 하는가? 어떤 스크립트를 돌려도 되는가? 어떤 파일이 신뢰할 만한가? 맥락이 압축될 때 무엇이 살아남는가?

반대쪽 실수는 재사용 가능한 지시를 전부 스킬로 취급하는 것이다. 저장소 관례, 결정적인 자동화, 외부 도구, 이벤트 훅, 위임된 에이전트는 서로 다른 문제를 푼다. 그것들을 모두 `SKILL.md`에 밀어 넣으면 이식 가능해 보이면서도 실제로는 특정 호스트의 문서화되지 않은 동작에 기대는 디렉터리가 나온다.

첫 공학 과제는 분류다. 어떻게 꾸릴지 정하기 전에 그것이 무엇인지부터 정하라.

## 개념 (The Concept)

### 스킬은 절차적 지식을 담는다 (Skills encode procedural knowledge)

에이전트 스킬은 `SKILL.md`를 진입점으로 삼는 디렉터리다. 그 진입 파일에는 YAML 프런트매터가 있고 그 뒤에 마크다운 지시가 온다. 디렉터리에는 참고 문서와 스크립트, 자산도 담을 수 있다.

```figure
skill-package-anatomy
```

배포 단위는 마크다운 파일 하나가 아니라 디렉터리다. 참고 문서가 빠진 채 복사된 `SKILL.md`는 프런트매터가 파싱되더라도 망가진 꾸러미다.

### 이웃한 추상 (The neighboring abstractions)

| 산출물 | 주된 역할 | 언제 적재되거나 실행되나 | 흉내 내서는 안 되는 것 |
|---|---|---|---|
| Prompt | 모델과의 상호작용 하나를 다듬는다 | 애플리케이션이나 사용자가 끼워 넣을 때 | 자원을 갖춘 버전 있는 꾸러미 |
| Repository instructions | 코드베이스 하나의 상시 규칙을 설명한다 | 코딩 런타임이 그 범위에 들어갈 때 | 재사용 가능한 과제 작업 흐름 |
| Agent skill | 재사용 가능한 절차적 지식을 공급한다 | 명시적 또는 암묵적 활성화 때 | 단단한 인가 경계 |
| MCP tool | 타입이 붙은 원격 기능을 내놓는다 | 모델이나 애플리케이션이 호출할 때 | 상세한 운영 절차 |
| Hook | 이벤트에 결정적인 로직을 돌린다 | 선언한 이벤트가 일어날 때 | 확률적인 모델 라우팅 |
| Subagent | 별도 맥락과 상태로 일을 위임한다 | 조율자가 만들거나 호출할 때 | 정적인 지시 묶음 |
| Plugin | 더 큰 런타임 확장을 배포한다 | 호스트가 설치하거나 켤 때 | 이식 가능한 스킬 계약 자체 |
| Learned skill library | 경험으로 발견한 행동을 저장한다 | 정책이 이전 프로그램이나 궤적을 꺼낼 때 | 표준에 기반한 `SKILL.md` 꾸러미 |

릴리스 스킬은 릴리스를 어떻게 살펴볼지 에이전트에게 알려 줄 수 있다. MCP 서버는 릴리스 레지스트리를 내놓을 수 있다. 훅은 직접 푸시를 금지할 수 있다. 서브에이전트는 후보를 독립적으로 감사할 수 있다. 이 조각들은 책임을 서로 다르게 지고 있어서 맞물려 돌아간다.

### "스킬"이라는 말은 서로 다른 두 가지를 가리킨다 (The word "skill" names two different ideas)

연구 시스템에서는 학습된 프로그램이나 성공한 궤적, 환경에 특화된 정책 조각을 스킬이라고 부르기도 한다. 에이전트는 탐색 도중 이런 산출물을 만들고, 과제 유사도로 꺼내 오고, 실행하고, 피드백을 받아 라이브러리를 고칠 수 있다. 페이즈 14 · 10이 그런 평생 학습 라이브러리를 만든다.

이 작은 과정에서 말하는 에이전트 스킬은 다르다. 선언된 파일 시스템 계약, 카탈로그 메타데이터, 점진적 공개, 런타임이 중재하는 호출, 호스트가 통제하는 도구를 갖춘, 사람이 저술한 꾸러미다. 에이전트가 만들거나 개선할 수는 있지만, 이 형식에 학습이 필수인 것은 아니다.

| 차원 | 에이전트 스킬 꾸러미 | 학습된 스킬 라이브러리 |
|---|---|---|
| 기본 단위 | `SKILL.md` 디렉터리 | 프로그램, 정책, 궤적, 기억 레코드 |
| 생성 | 저술하거나 생성하거나 골라 담는다 | 대개 환경 경험에서 발견한다 |
| 선택 | 카탈로그 설명에 런타임 정책을 더한다 | 과제 상태에 대한 검색이나 정책 |
| 실행 | 모델이 지시를 따르고 호스트 도구를 부른다 | 환경이 저장된 행동이나 코드 산출물을 돌린다 |
| 이식성 | 꾸러미 계약이 호환되는 호스트를 넘나든다 | 대개 환경 하나와 행동 공간에 묶인다 |
| 평가 | 라우팅, 산출물, 안전, 호스트 호환성 | 보상, 성공률, 전이, 라이브러리 성장 |

두 생각 모두 재사용 가능한 역량을 꾸린다. 이름이 같다는 이유만으로 구현까지 같다고 주장해서는 안 된다.

### 이식 가능한 핵심 (The portable core)

에이전트 스킬 명세는 프런트매터 필드 두 개를 요구한다.

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name`은 안정적인 식별자다. 명세의 이름 규칙을 지켜야 하고 상위 디렉터리 이름과 같아야 한다. `description`은 문서이면서 동시에 라우팅 메타데이터다. 무엇을 하는지와 언제 해당하는지를 말해야 한다.

이식 가능한 선택 필드는 이렇다.

| 필드 | 목적 | 이식성 참고 |
|---|---|---|
| `license` | 꾸러미의 이용 조건을 밝힌다 | 핵심 명세 |
| `compatibility` | 환경 요구사항을 밝힌다 | 핵심 명세 |
| `metadata` | 문자열 값 확장 데이터를 싣는다 | 핵심 명세 |
| `allowed-tools` | 미리 승인된 도구를 제안한다 | 실험적이며 호스트마다 지원이 다르다 |

마크다운 본문에는 운영 지시가 담긴다. 작업 흐름, 판단 지점, 실패 동작, 그리고 뒷받침하는 자원으로 가는 직접 경로를 정의해야 한다.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### 런타임 확장은 두 번째 계층이다 (Runtime extensions are a second layer)

어떤 호스트는 추가 프런트매터나 동반 설정을 받아들인다. 그 필드들이 쓸모 있을 수는 있지만 자동으로 이식 가능해지는 것은 아니다.

| 동작 | 호스트 확장의 예 | 이식 가능한 핵심인가 |
|---|---|:---:|
| 사용자 직접 호출은 남기고 모델 라우팅에서 감춘다 | `disable-model-invocation` | 아니오 |
| 모델 라우팅은 허용하되 사용자 명령 메뉴에서 감춘다 | `user-invocable` | 아니오 |
| 명령 메뉴에 인자 도움말을 보여 준다 | `argument-hint` | 아니오 |
| 위임된 맥락에서 스킬을 돌린다 | `context`, `agent` | 아니오 |
| 모델이나 추론 설정을 고정한다 | `model`, `effort` | 아니오 |
| 생애주기 자동화를 등록한다 | `hooks` | 아니오 |
| Codex에서 암묵적 호출을 끈다 | `agents/openai.yaml` 정책 | 아니오 |

확장은 하나하나를 어댑터로 다루라. 그것이 없어도 핵심 작업 흐름이 유효하게 두고, 대안을 문서로 남기고, 그것을 소비하는 호스트를 시험하라. 런타임은 모르는 필드를 무시할 수도, 거부할 수도, 동작을 구현하지 않은 채 보존만 할 수도 있다.

### 프런트매터는 실행되는 메타데이터다 (Frontmatter is executable metadata)

메타데이터는 스킬 본문을 읽기도 전에 시스템 동작을 바꾼다.

- 형식이 어긋난 `name`은 탐색을 실패하게 만들 수 있다.
- 두루뭉술한 `description`은 엉뚱한 요청을 끌어올 수 있다.
- 사람 전용 플래그는 모델의 카탈로그에서 스킬을 지워 버릴 수 있다.
- 도구 허용 설정은 호스트가 권한을 물어볼지 말지를 바꿀 수 있다.
- 맥락 설정은 실행을 별도 에이전트 세션으로 옮겨 버릴 수 있다.

프런트매터를 설정 코드처럼 검토하라. 검증하고, 버전을 매기고, 그 동작을 평가에 포함하라.

### 스킬의 생애주기 (The skill lifecycle)

```figure
skill-runtime-lifecycle
```

화살표 하나하나가 자기만의 실패 양상을 가진 경계다.

1. **탐색**은 설정된 위치에서 후보 꾸러미를 찾아낸다.
2. **검증**은 카탈로그에 올리기 전에 형식이 어긋났거나 안전하지 않은 꾸러미를 거부한다.
3. **카탈로그 등재**는 꾸러미 전체가 아니라 간결한 `name`과 `description`을 내놓는다.
4. **선택**은 그 스킬이 해당하는지 판단한다.
5. **활성화**는 본문을 모델이 보는 맥락으로 적재한다.
6. **공개**는 어떤 분기가 필요로 할 때만 참고 문서나 자산을 읽는다.
7. **실행**은 호스트의 권한과 격리 규칙 아래에서 호스트 도구를 쓴다.
8. **검증**은 모델의 주장과 별개로 만들어진 산출물을 확인한다.

이 단계들을 뭉개면 잘못된 머릿속 모형이 생긴다. 탐색된 스킬은 활성화된 것이 아니다. 활성화된 스킬이 자기가 설명하는 모든 일을 할 권한을 가진 것도 아니다. 허용된 도구 호출이 결과가 옳다는 증명도 아니다.

### 스킬과 도구는 서로 직교한다 (Skills and tools are orthogonal)

MCP는 "이 애플리케이션이 어떤 기능을 부를 수 있고 그 스키마는 무엇인가"에 답한다. 스킬은 "에이전트가 이런 종류의 과제에 어떻게 접근해야 하는가"에 답한다.

```figure
skill-tool-orthogonality
```

스킬이 도구 이름을 댈 수는 있지만, 실제 기능 등록부는 호스트가 쥐고 있다. 도구가 없다면 스킬은 대안을 밝히거나 분명하게 실패해야 한다. 기능 이름을 댄다고 그것이 생겨나는 것처럼 비쳐서는 절대 안 된다.

### 스킬과 저장소 지침은 범위가 다르다 (Skills and repository instructions are different scopes)

저장소 지침은 이미 들어와 있는 환경을 설명한다. 명령, 관례, 생성된 파일, 경계 같은 것들이다. 스킬은 여러 저장소에 걸쳐 나타날 수 있는 과제에 재사용 가능한 절차를 준다.

둘 다 해당할 때는 현재 사용자 요청과 저장소 규칙이 스킬을 제약한다. 범용 리팩터링 스킬이 생성된 파일을 고치지 말라는 저장소 규칙을 덮어써서는 안 된다.

### 스킬은 서로를 가져오지 않는다 (Skills do not import one another)

한 스킬이 다른 스킬을 불러 쓰라고 에이전트에게 지시할 수는 있지만, 이것은 언어 수준의 가져오기가 아니다. 두 번째 스킬도 여전히 런타임 탐색, 자격, 활성화, 권한, 맥락 처리를 거친다.

스킬 사이의 의존 관계는 관측 가능한 작업 흐름의 간선으로 적으라.

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

이렇게 하면 의존 관계를 시험할 수 있고 호스트가 정책을 강제할 기회도 생긴다.

## 만들어 보기 (Build It)

`code/main.py`는 표준을 지향하는 작은 검증기와 산출물 선택기를 구현한다. 규칙 하나하나가 눈에 보이도록 표준 라이브러리만 쓴다.

검증기는 이런 것을 내놓는다.

- 메타데이터와 본문을 갈라내는 `parse_frontmatter(text)`.
- 필수 필드, 이름 규칙, 모르는 확장, 본문 존재 여부, 이식 가능한 한계를 확인하는 `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`.
- 불투명한 참 거짓 하나 대신 구조화된 증거를 돌려주는 `ValidationIssue`와 `SkillReport`.
- 안전하게 해석할 수 없는 입력을 위한 `FrontmatterSyntaxError`.

선택기는 `TaskShape`와 `select_primitives(task)`를 내놓는다. 과제가 필요로 하는 것을 평범한 코드, 저장소 지침, 스킬, 훅, 서브에이전트, MCP 도구에 대응시킨다.

실습은 이렇게 돌린다.

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

이 명령 묶음은 지역 클론이 있어야 하고, `git rev-parse --show-toplevel`이
저장소 루트를 찾을 수 있도록 그 클론 안 어딘가에서 시작해야 한다.

예제는 유효한 이식 가능 스킬 하나, 호스트 확장을 쓴 스킬 하나, 유효하지 않은 꾸러미 하나, 그리고 여러 과제 형태 판단에 대한 JSON을 찍는다. 문제 코드를 들여다보라. 꾸러미 검증기는 저자를 대신해 짐작하지 않으면서 산출물을 어떻게 고칠지 설명해 줘야 한다.

### 검증 순서가 중요하다 (Validation order matters)

깊은 내용 규칙보다 값싼 구조적 사실을 먼저 검증하라.

```figure
skill-validation-order
```

이 순서가 있어야 뒤따르는 오류가 처음 깨진 불변식을 가리지 않는다.

## 직접 해 보기 (Use It)

스킬을 쓰기 전에 이 판단 카드를 채워 보라.

| 질문 | 그렇다면 | 어울리는 기본 요소 |
|---|---|---|
| 여러 단계에 걸쳐 재사용 가능한 모델 판단이 필요한가? | 절차는 안정적인데 판단은 매번 다르다 | Skill |
| 이벤트가 일어날 때마다 반드시 일어나야 하는가? | 한 번 놓치는 것도 용납되지 않는다 | Hook 또는 애플리케이션 코드 |
| 모델에게 타입이 붙은 입력을 가진 외부 기능이 필요한가? | 그 작업은 모델 맥락 바깥에 산다 | Tool 또는 MCP 서버 |
| 격리된 맥락, 상태, 소유권이 필요한가? | 별도 작업자가 한정된 결과를 돌려준다 | Subagent |
| 저장소 하나에만 해당하는 안내인가? | 지역 명령과 제약을 설명한다 | Repository instructions |
| 상호작용 한 번으로 충분한가? | 꾸러미 생애주기가 필요 없다 | Prompt |

실제 운영 작업 흐름은 이 표의 여러 줄을 함께 쓰는 일이 많다. 이 카드가 산출물 하나에 모든 성질을 담은 척하는 일을 막아 준다.

## 결과물 (Ship It)

이 레슨은 `outputs/` 아래에 `skill-contract-reviewer` 묶음을 만든다. 안에는 이런 것이 담긴다.

- 제안된 스킬 꾸러미를 검토하는, 이식 가능한 `SKILL.md`,
- 이식 가능한 계약과 기본 요소 선택을 위한 참고 점검표,
- 결정적인 검증 스크립트,
- 프롬프트, 스킬, 도구, 훅, 평범한 코드, 서브에이전트를 아우르는 과제 형태 표본.

진입 파일만이 아니라 묶음 전체를 설치하라.

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

과정용 설치 프로그램은 복사한 페이즈 13 스킬을 하나씩 알려 주고
`/tmp/aiefs-skills/manifest.json`을 쓴다. 이 깨끗한 목적지는 꾸러미의 모양을
확인해 주고, 위의 첫 성공 과정은 실제 호스트에서 탐색과 호출을 확인해 준다.

이어지는 레슨들이 생애주기 단계를 하나씩 깊이 판다. 레슨 24는 탐색과 점진적 공개를 만든다. 레슨 25는 호출 정책과 라우팅을 만든다. 레슨 26은 권한과 모래상자를 갈라놓는다. 레슨 27은 꾸러미 전체를 평가받는 릴리스 산출물로 바꾼다.

## 연습 문제 (Exercises)

1. 여러분 팀의 작업 흐름 다섯 개를 `TaskShape`로 분류하라. 기본 요소를 둘 이상 고른 경우마다 그 이유를 대라.
2. `compatibility` 값이 500자면 통과하고 501자면 명세 오류로 떨어지는 것을 증명하는 경계 테스트를 추가하라.
3. 허용 목록에 런타임 확장을 하나 추가하라. 그 파일이 여전히 이식 가능한 스킬과 구별되는지 증명하는 테스트를 쓰라.
4. 400줄짜리 프롬프트를 `SKILL.md` 하나, 참고 문서 하나, 스크립트 계약 하나, 출력 템플릿 하나로 쪼개라. 파일마다 한 종류의 정보만 맡게 하라.
5. 쓸 수 없는 MCP 도구를 참조하는 스킬의 실패 응답을 설계하라. 권한이 더 넓은 도구로 조용히 갈아 끼우지 마라.
6. 기존 스킬 하나를 검토해 문장마다 라우팅, 절차, 정책, 참고 문서 포인터, 출력 계약 중 무엇인지 이름표를 붙이라. 어디에도 속하지 않는 것은 옮기라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제로 뜻하는 것 |
|---|---|---|
| Agent skill | "저장해 둔 프롬프트" | 절차적 지시와 선택적 자원을 담은, 찾아낼 수 있는 디렉터리 |
| Portable core | "모든 런타임이 공유하는 필드" | 에이전트 스킬 명세가 정의한 계약 |
| Runtime extension | "추가 프런트매터" | 호환되는 어댑터가 있어야 동작하는 호스트별 설정 |
| Activation | "스킬이 돌았다" | 스킬 본문이 모델이 보는 맥락에 들어왔다는 뜻이며 실행은 나중일 수 있다 |
| Skill dependency | "다른 스킬을 가져온다" | 가용성과 정책 검사를 거치는, 런타임이 중재하는 호출 간선 |
| Tool contract | "함수 스키마" | 어떤 기능의 입력, 출력, 권한, 부수 효과, 오류, 증거 |

## 더 읽을거리 (Further Reading)

- [Agent Skills specification](https://agentskills.io/specification) 이식 가능한 디렉터리와 프런트매터 계약.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices) 범위, 지시, 자원 구성.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills) 현행 Codex의 탐색과 호출 동작.
- [Claude Code skills](https://code.claude.com/docs/en/skills) 한 런타임의 호출, 인자, 도구, 위임 맥락 확장.
