# 멀티 에이전트 프리미티브 모델 (The Multi-Agent Primitive Model)

> 2026년에 출시되는 모든 멀티 에이전트(multi-agent) 프레임워크(AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework)는 4차원 설계 공간 속의 한 점이다. 네 개의 프리미티브(primitive), 그뿐이다: 에이전트(agent), 핸드오프(handoff), 공유 상태(shared state), 오케스트레이터(orchestrator). 이 레슨은 이들을 밑바닥부터 만들고, 네 가지 모두로 장난감 시스템을 돌린 다음, 모든 주요 프레임워크를 같은 축에 매핑하여 어떤 새 릴리스든 한 문단으로 읽을 수 있게 한다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60분

## 문제 (Problem)

6개월마다 새 멀티 에이전트 프레임워크가 나온다. 2023년 AutoGen. 2024년 CrewAI. 2024년 LangGraph와 OpenAI Swarm. 2025년 4월 Google ADK. 2026년 2월 Microsoft Agent Framework RC. 각 보도자료는 스스로를 "올바른 추상화"라고 주장한다.

이들을 하나씩 배우려 하면 번아웃된다. API들이 달라 보인다. 문서들은 "에이전트"가 무엇인지를 두고 서로 의견이 다르다. 한 프레임워크는 공유 메모리를 "블랙보드(blackboard)"라 부르고, 다른 것은 "메시지 풀(message pool)"이라 부르고, 세 번째는 "StateGraph"라 부른다. 이 분야가 그저 헛돌고 있다고 의심하기 시작한다.

그렇지 않다. 마케팅 아래에서, 네 개의 프리미티브는 안정적이다. 한 번 배우면 모든 새 프레임워크를 한 문단으로 읽는다.

## 개념 (Concept)

### 네 개의 프리미티브

1. **에이전트(Agent)**: 시스템 프롬프트(system prompt)에 도구 목록을 더한 것. 무상태(stateless)다. 모든 실행은 그 시스템 프롬프트와 현재 메시지 이력에서 시작한다.
2. **핸드오프(Handoff)**: 한 에이전트에서 다른 에이전트로의 구조화된 제어 이전. 기계적으로는, 새 에이전트를 반환하는 도구 호출이거나 조건을 따르는 그래프 간선(edge)이다.
3. **공유 상태(Shared state)**: 둘 이상의 에이전트가 읽을(때로는 쓸) 수 있는 임의의 데이터 구조. 메시지 풀, 블랙보드, 키-값 저장소, 벡터 메모리.
4. **오케스트레이터(Orchestrator)**: 다음에 누가 말할지 결정하는 주체. 선택지: 명시적 그래프(결정적), LLM 화자 선택기(soft), 마지막 화자의 핸드오프 호출(OpenAI Swarm), 또는 큐 위의 스케줄러(swarm 아키텍처).

그것이 전체 설계 공간이다. 모든 프레임워크는 각 축마다 기본값을 고른다. 나머지는 표면 문법이다.

### 모든 2026년 프레임워크가 이에 매핑되는 방식

| 프레임워크 | 에이전트 | 핸드오프 | 공유 상태 | 오케스트레이터 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 도구가 Agent를 반환 | 호출자의 문제 | LLM의 다음 핸드오프 호출 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat의 화자 선택기 | 메시지 풀 | 선택기 함수(LLM 또는 라운드로빈) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task 출력이 연쇄됨 | 매니저 LLM 또는 정적 순서 |
| LangGraph | 노드 함수 | 그래프 간선 + 조건 | `StateGraph` 리듀서 | 그래프, 결정적 |
| Microsoft Agent Framework | 에이전트 + 오케스트레이션 패턴 | 패턴별 | 스레드 / 컨텍스트 | 패턴별 |
| Google ADK | 에이전트 + A2A 카드 | A2A 작업 | A2A 아티팩트 | 호스트가 결정 |

표면 차이는 거대해 보인다. 아래에는: 같은 네 개의 손잡이.

### 왜 이것이 중요한가

프리미티브를 보고 나면, 프레임워크 비교는 짧은 체크리스트가 된다.

- 오케스트레이터는 LLM이 라우팅하도록 신뢰하는가(Swarm), 아니면 라우팅을 코드에 고정하는가(LangGraph)?
- 공유 상태는 전체 이력(GroupChat)인가, 아니면 투영된 것(StateGraph 리듀서)인가?
- 에이전트가 서로의 프롬프트를 수정할 수 있는가(CrewAI 매니저), 아니면 오직 핸드오프만 하는가(Swarm)?

그 세 질문이 주어진 문제에 어떤 프레임워크가 맞는지 80%를 답한다. "최고의 멀티 에이전트 프레임워크"를 쇼핑하기를 멈추고 정말로 신경 쓰는 축에 맞춰 설계하기 시작한다.

### 무상태 통찰 (The stateless insight)

공유 상태를 제외한 모든 프리미티브는 무상태다. 에이전트는 (프롬프트, 도구)의 함수다. 핸드오프는 함수 호출이다. 오케스트레이터는 스케줄러다. **시스템에서 유일하게 상태를 갖는 것은 공유 상태다.** 그곳에 모든 흥미로운 버그가 산다: 메모리 오염(memory poisoning)(Lesson 15), 메시지 순서, 버전 관리, 쓰기 경합(write contention).

공유 상태를 숨기는 프레임워크(Swarm)는 문제를 호출자에게 떠넘긴다. 이를 중앙화하는 프레임워크(LangGraph 체크포인트, AutoGen 풀)는 이를 검사 가능하게 만들지만 조율 비용을 공유 상태 구현으로 옮긴다.

### 단일 프리미티브의 해부

#### 에이전트 (Agent)

```
Agent = (system_prompt, tools, model, optional_name)
```

메모리 없음. 상태 없음. 같은 시스템 프롬프트와 도구를 가진 두 에이전트는 교체 가능하다. 에이전트별 상태처럼 보이는 모든 것은 실제로는 공유 상태나 핸드오프 프로토콜 안에 있다.

#### 핸드오프 (Handoff)

```
Handoff = (from_agent, to_agent, reason, payload)
```

세 가지 구현이 지배적이다.

- **함수 반환**: 도구가 다음 에이전트를 반환한다. 이것이 OpenAI Swarm 패턴이다. 에이전트가 자신의 도구 스키마에 라우팅을 담는다.
- **그래프 간선**: LangGraph. 간선은 선언적이다. LLM이 값을 생성하고, 조건이 다음 노드를 선택한다.
- **화자 선택**: AutoGen GroupChat. 선택기 함수(때로는 그 자체가 LLM 호출)가 풀을 읽고 다음에 누가 말할지 고른다.

#### 공유 상태 (Shared state)

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

최소한, 메시지의 리스트. 흔히 그 이상: 구조화된 아티팩트(CrewAI Task 출력), 타입이 지정된 컨텍스트(LangGraph 리듀서), 외부 메모리(MCP, 벡터 DB).

두 가지 토폴로지: **전체 풀(full pool)**(모든 에이전트가 모든 메시지를 봄)과 **투영(projected)**(에이전트가 역할 범위로 한정된 뷰를 봄). 전체 풀은 단순하고 확장이 나쁘다. 투영 풀은 확장되지만 사전 스키마 설계가 필요하다.

#### 오케스트레이터 (Orchestrator)

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

네 가지 종류:

- **정적(Static)**: 그래프가 빌드 시점에 고정됨(LangGraph 결정적, CrewAI Sequential).
- **LLM 선택(LLM-selected)**: LLM이 풀을 읽고 다음 화자를 고름(AutoGen, CrewAI Hierarchical).
- **핸드오프 주도(Handoff-driven)**: 현재 에이전트가 핸드오프 도구를 호출하여 결정함(Swarm).
- **큐 주도(Queue-driven)**: 워커가 공유 큐에서 가져옴. 명시적 다음 화자 없음(swarm 아키텍처, Matrix).

### 프레임워크 간에 무엇이 달라지는가

프리미티브가 고정되면, 남은 설계 결정은 다음과 같다.

- **메모리 전략**: 일시적(ephemeral) 대 지속적 체크포인팅(LangGraph 체크포인터).
- **안전 경계**: 누가 핸드오프를 승인할 수 있는가(human-in-the-loop).
- **비용 회계**: 에이전트별 토큰 예산.
- **관측 가능성(observability)**: 핸드오프 추적, 재생(replay)을 위한 상태 지속.

모두 프리미티브 위에 구현 가능하다. 그중 어느 것도 새 프리미티브가 아니다.

```figure
a5-primitive-radar
```

## 직접 만들기 (Build It)

`code/main.py`는 약 150줄의 stdlib Python으로 네 프리미티브를 구현한다. 실제 LLM은 없다. 각 에이전트는 스크립트된 정책(policy)이라서 초점이 조율 구조에 머무른다.

이 파일은 다음을 내보낸다.

- `Agent`: 이름, 시스템 프롬프트, 도구, 정책 함수의 데이터클래스.
- `Handoff`: 새 에이전트를 반환하는 함수.
- `SharedState`: 스레드 안전(thread-safe) 메시지 풀.
- `Orchestrator`: 세 변형: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator`(시뮬레이션).

데모는 동일한 세 에이전트 파이프라인(research → write → review)을 세 가지 오케스트레이터 타입 모두로 돌리고 끝에 메시지 풀을 출력한다. 출력이 오직 *누가 다음을 고르는가*에서만 다르다는 것을 볼 수 있다. 에이전트와 공유 상태는 실행 전반에 걸쳐 동일하다.

실행:

```
python3 code/main.py
```

예상 출력: 세 번의 오케스트레이터 실행, 패턴당 하나씩. 각각은 최종 메시지 풀을 출력한다. 연구자가 일찍 끝났다고 결정하면 핸드오프 주도 실행은 더 적은 에이전트에 도달한다. 그것이 LLM 라우팅 트레이드오프(tradeoff)의 축소판이다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-primitive-mapper.md`는 임의의 멀티 에이전트 코드베이스나 프레임워크 문서를 읽고 네 프리미티브 매핑을 반환하는 스킬이다. 새 프레임워크 릴리스에 이를 돌려서 문서를 깊이 읽기 전에 한 문단으로 개요를 잡아라.

## 산출물 (Ship It)

새 프레임워크를 채택하기 전에, 그것의 프리미티브 매핑을 작성하라. 작성할 수 없다면, 문서가 불완전하거나 프레임워크가 다섯 번째 프리미티브를 발명하고 있는 것이다(드물다. 본 적 없는 공유 상태 종류가 있는지 확인하라).

매핑을 아키텍처 문서에 고정하라. 새 팀원이 합류하면 API 문서보다 먼저 매핑을 보내라. 프레임워크 버전이 바뀌면 체인지로그가 아니라 매핑을 비교하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 다른 에이전트 정책으로 세 번 실행하라. 오케스트레이터 선택이 어떤 에이전트가 실행되는지를 어떻게 바꾸는지 관찰하라.
2. 네 번째 오케스트레이터 타입을 구현하라: 에이전트가 작업을 위해 공유 상태를 폴링하는 큐 주도 타입. 어떤 교착(deadlock)이 발생할 수 있고, 어떻게 탐지하는가?
3. LangGraph 퀵스타트(https://docs.langchain.com/oss/python/langgraph/workflows-agents)를 가져와 네 프리미티브로 다시 작성하라. LangGraph의 추상화 중 어느 것이 1:1로 매핑되고 어느 것이 편의 래퍼인가?
4. OpenAI Swarm 쿡북(https://developers.openai.com/cookbook/examples/orchestrating_agents)을 읽어라. Swarm이 네 프리미티브 중 어느 것을 가장 사용하기 좋게 만들고, 어느 것을 호출자에게 떠넘기는지 식별하라.
5. 이 표에서 공유 상태를 완전히 숨기는 프레임워크 하나를 찾아라. 에이전트가 이력을 다시 읽지 않고 핸드오프를 가로질러 조율해야 할 때 무엇이 깨지는지 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 에이전트 (Agent) | "도구를 가진 LLM" | `(system_prompt, tools, model)` 삼중쌍. 무상태. |
| 핸드오프 (Handoff) | "제어의 이전" | 다음 에이전트와 선택적 페이로드를 명명하는 구조화된 호출. 세 구현: 함수 반환, 그래프 간선, 화자 선택. |
| 공유 상태 (Shared state) | "메모리" / "컨텍스트" | 멀티 에이전트 시스템에서 유일하게 상태를 갖는 부분. 메시지 풀 또는 블랙보드. |
| 오케스트레이터 (Orchestrator) | "코디네이터" | 다음에 누가 실행될지 결정하는 주체. 정적 그래프, LLM 선택기, 핸드오프 주도, 또는 큐 주도. |
| 프리미티브 (Primitive) | "추상화" | 모든 프레임워크가 매개변수화하는 네 축 중 하나. 프레임워크 기능이 아니다. |
| 메시지 풀 (Message pool) | "공유 채팅 이력" | 전체 이력 공유 상태. 추론하기 쉽고, 확장이 나쁘다. |
| 투영 상태 (Projected state) | "범위가 한정된 뷰" | 공유 상태로의 역할별 뷰. 확장되고, 스키마 설계가 필요하다. |
| 화자 선택 (Speaker selection) | "다음에 누가 말하는가" | 함수(흔히 LLM)가 그룹에서 다음 에이전트를 고르는 오케스트레이터 패턴. |

## 더 읽을거리 (Further Reading)

- [OpenAI cookbook: Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) 핸드오프 주도 오케스트레이션의 가장 명료한 설명
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/): GroupChat + 화자 선택은 LLM 선택 오케스트레이션의 레퍼런스다
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents): 그래프 간선 오케스트레이션과 리듀서 기반 공유 상태
- [CrewAI introduction](https://docs.crewai.com/en/introduction): 역할-목표-배경(role-goal-backstory) 에이전트, Sequential / Hierarchical 프로세스
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2): 마이크로소프트가 v0.4를 유지보수로 옮긴 후의 살아있는 AutoGen v0.2 라인
