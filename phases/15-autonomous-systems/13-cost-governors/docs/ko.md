# 액션 예산, 반복 상한, 그리고 비용 거버너(Cost Governor)

> 어느 중간 규모 전자상거래 에이전트(agent)의 월간 LLM 비용은 팀이 "주문 추적" 스킬을 활성화한 뒤 1,200달러에서 4,800달러로 뛰었다. 가격 책정 버그가 아니다. 새로운 루프를 찾아 그 안에서 계속 지출한 에이전트다. Microsoft의 Agent Governance Toolkit(2026년 4월 2일)은 이 부류에 대한 방어를 성문화한다: 요청당 `max_tokens`, 과제당 토큰(token) 및 달러 예산, 일/월 단위 상한, 반복 상한, 계층형 모델 라우팅, 프롬프트 캐싱(prompt caching), 컨텍스트 윈도잉(context windowing), 비싼 액션에 대한 HITL 체크포인트, 예산 초과 시 킬 스위치(kill switch). Anthropic의 Claude Code Agent SDK는 같은 기본 요소를 다른 이름으로 출하한다. 금융 속도 제한(financial velocity limit)(예를 들어 10분에 50달러 초과 시 접근 차단)은 월간 상한보다 루프를 더 빨리 잡는다.

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60분

## 문제 (The Problem)

자율 에이전트는 매 턴마다 실제 돈을 쓴다. 챗봇의 나쁜 출력은 나쁜 응답이지만, 에이전트의 나쁜 루프는 청구서다. 이 실패 양상을 가리키는, 업계에 문서화된 용어는 "지갑 거부(Denial of Wallet)"다. 에이전트가 계속 추론하고, 계속 도구를 호출하고, 계속 청구하며, 무엇도 그것을 멈추지 않는다. 멈추도록 설계된 것이 없기 때문이다.

해결책은 하나의 숫자가 아니다. 서로 다른 시간 척도와 세분도(granularity)에서의 제한 스택이다: 요청당, 과제당, 시간당, 일당, 월당. 잘 설계된 스택은 폭주 루프를 몇 분 내에, 느린 누수를 몇 시간 내에, 나쁜 릴리스를 하루 내에 잡는다. 에이전트가 장기 지평(long-horizon)이고 자율적일 때도, 같은 스택이 예산을 애초부터 유지시킨다.

이것은 엔지니어링 레슨이다. 수학은 사소하고, 규율이 팀들이 실패하는 지점이다. 아래의 제한 목록은 Microsoft Agent Governance Toolkit이나 Anthropic Claude Code Agent SDK 문서 어느 쪽에든 명명되어 있다.

## 개념 (The Concept)

### 비용 거버너 스택

1. **요청당 `max_tokens`.** 단순하다. 어떤 단일 호출도 무한정한 완성(completion)을 내보내지 못하게 막는다.
2. **과제당 토큰 예산.** 실행 전체에 걸쳐, N 토큰을 초과하지 않는다. 상한에서 하드 스톱.
3. **과제당 달러 예산.** 토큰과 같지만 통화 단위. Claude Code의 `max_budget_usd`.
4. **도구별 호출 상한.** `WebFetch` 호출 N회, `shell_exec` 호출 N회 등 이하.
5. **반복 상한(`max_turns`).** 총 에이전트 루프 반복 횟수; 무한 추론 루프를 방지한다.
6. **분당 / 시간당 / 일당 / 월당 상한.** 롤링 윈도(rolling window). 서로 다른 시간 척도에서 누수를 잡는다.
7. **금융 속도 제한.** 예를 들어 "10분에 지출이 50달러를 초과하면 접근 차단." 월간 상한이 발사되기 전에 루프 기반 소진을 잡는다.
8. **계층형 모델 라우팅.** 기본적으로 더 작은 모델로; 분류기(classifier)가 과제가 정당하다고 판단할 때만 더 큰 모델로 격상한다.
9. **프롬프트 캐싱.** 시스템 프롬프트와 안정적 컨텍스트를 제공자 캐시에 저장; 재전송의 토큰 비용이 거의 0이다.
10. **컨텍스트 윈도잉.** 활성 컨텍스트를 임계값 아래로 유지하기 위한 압축(compaction)/요약; 직접적인 토큰 비용 절감.
11. **비싼 액션에 대한 HITL 체크포인트.** 비싸다고 알려진 액션(긴 도구 호출, 대용량 다운로드, 비용이 큰 모델 업그레이드) 전에, 사람의 탭(tap)을 요구한다.
12. **예산 초과 시 킬 스위치.** 어떤 상한이든 발사되면 세션이 중단된다. 상한이 기록되며; 별도의 재활성화 경로를 요구한다.

### 하나의 상한이 아니라 스택인 이유

단일 월간 상한은 지갑이 사라진 뒤에야 폭주 에이전트를 잡는다. 단일 요청당 상한은 세션 수준에서 아무것도 잡지 못한다. 서로 다른 실패 양상은 서로 다른 시간 척도를 요구한다:

- **폭주 루프**(5초 재시도에 갇힌 에이전트): 속도 제한이 잡는다.
- **느린 누수**(과제당 예상의 약 2배 작업을 하는 에이전트): 일당 상한이 잡는다.
- **나쁜 릴리스**(새 버전이 5배 토큰을 쓰는 경우): 주간/월간 상한이 잡는다.
- **합법적 급증**(버그가 아니라 실제 수요): 명확한 로그와 함께 시간/일당 상한이 잡는다.

### Claude Code의 예산 표면

Claude Code Agent SDK는 (공개 문서 기준) 다음을 노출한다:

- `max_turns`: 반복 상한.
- `max_budget_usd`: 달러 상한; 초과 시 세션 중단.
- `allowed_tools` / `disallowed_tools`: 도구 허용 목록(allowlist)과 거부 목록(denylist).
- 사용자 정의 비용 회계를 위한 도구 사용 전 훅(hook) 지점.

권한 모드 사다리(Lesson 10)와 결합하라. `max_budget_usd` 없는 `autoMode` 세션은 거버넌스 없는 자율성이다. Anthropic은 Auto Mode에 예산 통제가 반드시 필요하다고 명시한다. 분류기는 비용과 직교(orthogonal)다.

### EU AI Act, OWASP Agentic Top 10

Microsoft의 Agent Governance Toolkit은 OWASP Agentic Top 10과 EU AI Act 제14조(사람의 감독) 요구사항을 다룬다. EU에서의 프로덕션을 위해서는 로깅과 상한 강제가 선택이 아니다.

### 관찰된 $1,200 → $4,800 사례

Microsoft 문서의 실제 사례: 새 도구가 추가된 뒤 월간 비용이 세 배가 된 전자상거래 에이전트. 그 도구는 에이전트가 매 세션마다 주문 상태를 폴링(poll)하게 했다. 루프 탐지 없음. 도구별 상한 없음. 주간 대비 증가에 대한 경보 없음. 해결책은 도구별 상한 + 일간 증가 경보였다. 이것은 하나의 템플릿이다. 모든 새 도구 표면은 새로운 잠재적 루프이며; 모든 새 도구는 자기 자신의 상한과 자기 자신의 경보를 필요로 한다.

```figure
cost-governor-stack
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 계층형 비용 거버너 스택이 있을 때와 없을 때의 에이전트 실행을 시뮬레이션한다. 시뮬레이션된 에이전트는 몇 턴 뒤 폴링 루프로 표류한다. 계층형 스택은 그것을 속도 윈도 안에서 잡는 반면, 단일 월간 상한은 며칠 뒤에야 발사된다.

## 산출물 (Ship It)

`outputs/skill-agent-budget-audit.md`는 제안된 에이전트 배포(deployment)의 비용 거버너 스택을 감사하고 누락된 층을 표시한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 폴링 루프 궤적(trajectory)에서 속도 제한이 반복 상한보다 먼저 발사됨을 확인하라. 이제 속도 제한을 비활성화하고, 반복 상한이 그것을 잡기 전까지 에이전트가 얼마나 "지출"하는지 측정하라.

2. 브라우저 에이전트(Lesson 11)를 위한 도구별 상한 집합을 설계하라. 어느 도구가 가장 빡빡한 상한이 필요한가? 어느 도구가 위험 없이 무한정 돌 수 있는가?

3. Microsoft Agent Governance Toolkit 문서를 읽어라. 툴킷이 명명하는 모든 상한 유형을 나열하라. 각각을 실패 양상(폭주 루프, 느린 누수, 나쁜 릴리스, 급증) 중 하나에 매핑하라.

4. 현실적인 과제(예: "저장소의 50개 이슈 분류")에 대한 밤샘 무인 실행의 가격을 매겨라. `max_budget_usd`를 당신의 점추정치의 2배로 설정하라. 그 2배를 정당화하라.

5. Claude Code의 `max_budget_usd`는 세션 누적 비용에서 발사된다. 외부에서 강제할 보완적 속도 제한을 설계하라. 무엇이 차단을 촉발하며, 재활성화는 어떤 모습인가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 지갑 거부 (Denial of Wallet) | "폭주 청구서" | 멈출 상한 없이 지출을 생성하는 에이전트 루프 |
| max_tokens | "요청당 상한" | 단일 완성 크기에 대한 천장 |
| max_turns | "반복 상한" | 한 세션 내 에이전트 루프 반복에 대한 천장 |
| max_budget_usd | "달러 킬 스위치" | 세션 비용 상한; 초과 시 중단 |
| 속도 제한 (Velocity limit) | "비율 상한" | 짧은 윈도당 지출에 대한 제한(예: $50 / 10분) |
| 계층형 라우팅 (Tiered routing) | "작은 모델 먼저" | 싼 모델 기본; 분류기가 정당화할 때만 격상 |
| 프롬프트 캐싱 (Prompt caching) | "캐시된 시스템 프롬프트" | 제공자 측 캐시가 재전송 토큰 비용을 거의 0으로 줄임 |
| HITL 체크포인트 | "사람 승인 게이트" | 비싼 액션 전에 사람의 탭이 요구됨 |

## 더 읽을거리 (Further Reading)

- [Anthropic Claude Code Agent SDK: agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) `max_turns`, `max_budget_usd`, 도구 허용 목록.
- [Microsoft Agent Framework: human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) 비용 거버너 체크포인트.
- [Anthropic: Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) 제공자 측 비용 통제.
- [Anthropic: Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) 캐싱 메커니즘.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평 에이전트의 비용 프로파일.
