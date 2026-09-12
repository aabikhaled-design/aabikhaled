# 핸드오프와 루틴(무상태 오케스트레이션 (Handoffs and Routines) Stateless Orchestration)

> OpenAI의 Swarm(2024년 10월)은 멀티 에이전트 오케스트레이션(orchestration)을 두 가지 프리미티브로 증류했다. **루틴(routines)**(시스템 프롬프트로서의 명령어 + 도구)과 **핸드오프(handoffs)**(다른 Agent를 반환하는 도구)다. 상태 기계(state machine)도, 분기 DSL도 없다. LLM이 올바른 핸드오프 도구를 호출함으로써 라우팅한다. OpenAI Agents SDK(2025년 3월)는 프로덕션(production) 후속작이다. Swarm 자체는 가장 깔끔한 개념적 레퍼런스로 남아 있다. 전체 소스가 수백 줄에 들어간다. 이 패턴이 바이럴(viral)인 이유는 API 표면이 대략 "에이전트 = 프롬프트 + 도구, 핸드오프 = 에이전트를 반환하는 함수"이기 때문이다. 한계는 무상태(stateless)라는 것이고, 그래서 메모리는 호출자의 몫이다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60분

## 문제 (Problem)

모든 멀티 에이전트 프레임워크는 자신의 DSL을 배우라고 요구한다. LangGraph의 노드와 간선, CrewAI의 크루(crew)와 작업(task), AutoGen의 GroupChat와 매니저. DSL은 진짜 추상화지만, 필요 이상으로 무겁게 느껴진다.

Swarm는 반대 방향으로 밀어붙인다. 모델이 이미 가진 도구 호출(tool-calling) 능력을 사용하라는 것이다. 핸드오프가 곧 도구 호출이다. 오케스트레이터는 현재 대화를 보유한 에이전트다. 상태 기계는 에이전트들의 시스템 프롬프트 속에 암묵적으로 들어 있다.

## 개념 (Concept)

### 두 가지 프리미티브

**루틴(Routine).** 에이전트의 역할과 사용 가능한 도구를 정의하는 시스템 프롬프트다. 범위가 좁혀진 명령어 묶음으로 생각하라. "너는 분류(triage) 에이전트다. 사용자가 환불에 관해 물으면 환불 에이전트에게 핸드오프하라."

**핸드오프(Handoff).** 에이전트가 호출할 수 있고 새로운 Agent 객체를 반환하는 도구다. Swarm 런타임은 Agent 반환값을 감지하여 다음 턴의 활성 에이전트(active agent)를 전환한다.

그것이 추상화의 전부다.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

분류 에이전트의 시스템 프롬프트는 사용자 메시지에 따라 올바른 핸드오프를 고르게 만든다. LLM의 도구 호출이 라우팅을 수행한다.

### 왜 바이럴인가

- **작은 API.** 배울 개념이 둘뿐이다.
- **모델이 이미 하는 것을 사용한다.** 도구 호출은 이미 모든 제공자에 걸쳐 프로덕션 수준이다.
- **상태 기계 부담이 없다.** 그래프를 묘사하지 않는다. 에이전트들의 프롬프트가 누구에게 핸드오프하는지 묘사한다.

### 무상태의 트레이드오프(trade-off)

Swarm는 실행과 실행 사이에 명시적으로 무상태다. 프레임워크는 한 번의 실행 동안에는 메시지 이력을 유지하지만 아무것도 영속화(persist)하지 않는다. 메모리도, 연속성도, 장기 실행 작업도 모두 호출자의 몫이다.

프로덕션(OpenAI Agents SDK, 2025년 3월)에서는 이것이 바뀐 주요 사항 중 하나였다. SDK는 핸드오프 프리미티브를 유지하면서 내장 세션 관리, 가드레일(guardrail), 추적(tracing)을 추가한다.

### Swarm/핸드오프가 잘 맞을 때

- **분류 패턴.** 최전선 에이전트가 사용자를 전문가에게 라우팅한다.
- **스킬 기반 핸드오프.** "작업에 코드가 필요하면 코더를 호출하고, 연구가 필요하면 연구자를 호출하라."
- **짧고 한정된 대화.** 고객 지원, FAQ-to-ticket, 단순 워크플로.

### Swarm가 어려워할 때

- **공유 메모리가 있는 긴 세션.** 핸드오프는 대화 상태를 새 에이전트의 프롬프트와 이력으로 재설정한다. 호출자가 메모리를 직접 관리하지 않으면 에이전트 사이에 영속 상태가 남지 않는다.
- **병렬 실행.** 핸드오프는 한 번에 하나씩이다. 활성 에이전트가 전환될 뿐이다. 병렬성을 위해서는 호출자가 여러 Swarm 실행을 오케스트레이션해야 한다.
- **감사와 재생(replay).** 무상태 실행은 정확히 재생하기 어렵다. LLM의 핸드오프 선택은 결정론적이지 않다.

### OpenAI Agents SDK (2025년 3월)

프로덕션 후속작은 다음을 추가한다.

- **세션 상태.** 실행 간 영속되는 스레드.
- **가드레일.** 입력/출력 검증 훅(hook).
- **추적.** 모든 도구 호출과 핸드오프가 기록된다.
- **핸드오프 필터.** 핸드오프 시 어떤 맥락이 전달되는지 제어한다.

핸드오프 프리미티브는 살아남고, 프로덕션 편의 기능이 그 주위에 추가된다.

### Swarm vs GroupChat

둘 다 LLM 주도 라우팅을 사용하지만, **누가 다음을 고르는가**에서 다르다.

- GroupChat: 선택자(함수 또는 LLM)가 외부에서 다음 발화자를 고른다.
- Swarm: 현재 에이전트가 핸드오프 도구를 호출하여 자신의 후임을 고른다.

Swarm는 "에이전트가 다음을 결정"하고, GroupChat는 "매니저가 다음을 결정"한다. Swarm의 결정은 활성 에이전트의 도구 호출 안에 있고, GroupChat의 결정은 `GroupChatManager` 안에 있다.

```figure
sw-handoff-routing
```

## 직접 만들기 (Build It)

`code/main.py`는 Swarm를 밑바닥부터 구현한다. Agent 데이터클래스, 핸드오프 메커니즘(도구가 Agent를 반환), 그리고 에이전트 전환을 감지하는 실행 루프다.

데모: 분류 에이전트가 환불, 영업, 또는 지원 전문가에게 라우팅한다. 각 전문가는 자신만의 도구를 가진다. 실행 루프는 각 핸드오프를 출력한다.

실행:

```
python3 code/main.py
```

## 라이브러리로 써보기 (Use It)

`outputs/skill-handoff-designer.md`는 주어진 작업에 대한 핸드오프 토폴로지(topology)를 설계한다. 어떤 에이전트가 존재하는지, 그들이 어떤 핸드오프를 호출할 수 있는지, 어떤 맥락이 전달되는지를 다룬다.

## 산출물 (Ship It)

체크리스트:

- **핸드오프 로깅.** 모든 핸드오프는 출발 에이전트, 도착 에이전트, 맥락 스냅샷을 담은 추적 이벤트를 기록한다.
- **맥락 전달 규칙.** 핸드오프 시 무엇이 넘어가는지 결정하라. 전체 이력(비쌈), 마지막 N개 메시지, 또는 요약.
- **핸드오프 가드레일.** 도구 권한이 다른 전문가로의 핸드오프는 인증되어야 한다. 그렇지 않으면 프롬프트 인젝션(prompt injection)이 원치 않는 핸드오프를 강제할 수 있다.
- **루프 감지.** 두 에이전트가 서로 주고받는 것은 흔한 실패다. 단순한 마지막 K개 링(ring) 검사로 감지하라.
- **폴백 에이전트.** 핸드오프 대상이 존재하지 않으면 안전한 기본값으로 폴백하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하여 환불 에이전트로 분류하라. 두 번째 턴의 활성 에이전트가 환불 에이전트인지 확인하라.
2. 루프 감지 규칙을 추가하라. 같은 두 에이전트가 연속으로 3번 핸드오프하면 강제로 종료한다. 폴백을 설계하라.
3. 핸드오프 필터에 관한 OpenAI Agents SDK 문서를 읽어라. "핸드오프 시 요약" 버전을 구현하라. 나가는 에이전트가 들어오는 에이전트가 인계받기 전에 맥락을 불릿 요약으로 압축한다.
4. Swarm 핸드오프를 GroupChatManager 선택자와 비교하라. 어느 패턴이 프롬프트 인젝션을 더 악화시키며, 그 이유는 무엇인가?
5. Swarm 쿡북(https://developers.openai.com/cookbook/examples/orchestrating_agents)을 읽어라. OpenAI Agents SDK가 변경했거나 유지한 Swarm의 명시적 설계 결정 하나를 식별하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 루틴(Routine) | "에이전트 프롬프트" | 시스템 프롬프트 + 도구 목록. 역할과 사용 가능한 핸드오프를 정의한다. |
| 핸드오프(Handoff) | "다른 에이전트로 이관" | 활성 에이전트가 호출할 수 있고 새 Agent를 반환하는 도구. 런타임이 활성 에이전트를 전환한다. |
| 무상태(Stateless) | "실행 간 메모리 없음" | Swarm는 아무것도 영속화하지 않는다. 메모리는 호출자의 책임이다. |
| 활성 에이전트(Active agent) | "지금 말하는 사람" | 현재 대화를 보유한 에이전트. 핸드오프가 이를 바꾼다. |
| 맥락 전달(Context transfer) | "핸드오프 시 넘어가는 것" | 들어오는 에이전트가 보는 이력에 대한 정책: 전체, 마지막 N개, 또는 요약. |
| 핸드오프 루프(Handoff loop) | "에이전트 핑퐁" | 두 에이전트가 서로 계속 주고받는 실패 모드. |
| OpenAI Agents SDK | "프로덕션 Swarm" | 2025년 3월 후속작. 핸드오프 프리미티브 위에 세션, 가드레일, 추적을 추가한다. |
| 핸드오프 필터(Handoff filter) | "이관 시 게이트" | 핸드오프 경계에서 맥락을 검사하고 수정하는 SDK 기능. |

## 더 읽을거리 (Further Reading)

- [OpenAI cookbook: Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) 레퍼런스 정식 설명
- [OpenAI Swarm repo](https://github.com/openai/swarm): 원본 구현, 개념적 레퍼런스로 유지됨
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/): 세션과 추적을 갖춘 프로덕션 후속작
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code): Claude Code 서브에이전트가 `Task`를 통해 핸드오프 유사 패턴을 사용하는 방식
