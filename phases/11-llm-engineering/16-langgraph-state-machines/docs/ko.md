# LangGraph: 에이전트를 위한 상태 기계 (State Machines for Agents)

> 손으로 작성한 ReAct 루프는 `while True`다. LangGraph로 작성한 ReAct 루프는 체크포인트하고, 중단하고, 분기하고, 시간 여행할 수 있는 그래프다. 에이전트(agent)는 바뀌지 않았다. 그 주변의 하니스가 바뀌었다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75분

## 문제 (The Problem)

함수 호출(function calling) 에이전트를 출시한다고 하자. 세 턴 동안 동작하다가 무언가 잘못된다. 모델이 500을 반환하는 도구를 시도하거나, 사용자가 작업 도중에 마음을 바꾸거나, 에이전트가 사람의 승인 없이 주문을 환불하기로 결정한다. `while True:` 루프에는 후크가 없다. 일시 정지할 수도, 되감을 수도, "모델이 다른 도구를 골랐다면 어땠을까"로 분기할 수도 없다. 데모 너머로 출시하는 순간, 에이전트는 동작했거나 안 했거나인 블랙박스가 된다.

다음 단계는 보고 나면 명백하다. 에이전트는 이미 상태 기계(state machine)다. 시스템 프롬프트(prompt) 더하기 메시지 기록 더하기 보류 중인 도구 호출 더하기 다음 액션. 상태 기계를 명시적으로 만들라: "모델이 생각한다," "도구가 실행된다," "사람이 승인한다"를 위한 노드, 그리고 그들 사이의 조건부 전이를 위한 간선. 그래프가 명시적이 되면, 하니스는 네 가지를 공짜로 얻는다: 체크포인팅(단계 사이 상태 저장), 인터럽트(사람을 위해 일시 정지), 스트리밍(토큰과 중간 이벤트 스트리밍), 시간 여행(이전 상태로 되감아 다른 분기 시도).

LangGraph는 이 추상화를 출시하는 라이브러리다. LangChain적 의미의 에이전트 프레임워크("여기 AgentExecutor가 있으니, 행운을 빈다")가 아니라, 일급(first-class) 상태와 일급 영속성, 일급 인터럽트를 가진 그래프 런타임이다. 에이전트 루프는 손으로 작성하는 것이 아니라 그리는 것이다.

## 개념 (The Concept)

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph`에는 세 가지가 있다.

1. **상태(State).** 그래프를 통해 흐르는 타입 지정된 dict(TypedDict 또는 Pydantic 모델). 모든 노드는 전체 상태를 받고 부분 업데이트를 반환하며, LangGraph는 필드별 *리듀서(reducer)*를 사용해 이를 병합한다. 누적되어야 하는 리스트에는 `operator.add`, 기본적으로는 덮어쓰기.
2. **노드(Nodes).** Python 함수 `state -> partial_state`. 각각은 별개의 단계다: "모델 호출," "도구 실행," "요약."
3. **간선(Edges).** 노드 사이의 전이. 정적 간선은 한 곳으로 간다. 조건부 간선은 라우터 함수 `state -> next_node_name`을 받아 그래프가 모델 출력에 따라 분기할 수 있게 한다.

그래프를 컴파일한다. 컴파일은 토폴로지를 바인딩하고 체크포인터(선택적이지만 프로덕션에는 필수)를 부착해 실행 가능한 것을 반환한다. 이를 초기 상태와 `thread_id`로 호출한다. 실행의 모든 단계는 `(thread_id, checkpoint_id)`로 키가 매겨진 체크포인트를 영속화한다.

### 네 가지 초능력 (The four superpowers)

**체크포인팅.** 모든 노드 전이는 새 상태를 저장소(테스트는 인메모리, 프로덕션은 Postgres/Redis/SQLite)에 쓴다. 같은 `thread_id`로 그래프를 다시 호출하여 재개한다. 그래프는 일시 정지한 곳에서 이어간다.

**인터럽트.** 노드를 `interrupt_before=["human_review"]`로 표시하면 그 노드가 실행되기 전에 실행이 멈춘다. 상태가 영속화된다. API는 "승인 대기 중"으로 사용자에게 응답한다. 같은 `thread_id`에 대한 나중 요청이 `Command(resume=...)`로 실행을 재개한다.

**스트리밍.** `graph.stream(state, mode="updates")`는 상태 델타가 발생하는 대로 산출한다. `mode="messages"`는 모델 노드 내부의 LLM 토큰을 스트리밍한다. `mode="values"`는 전체 스냅샷을 산출한다. UI에 무엇을 표시할지는 골라서 정한다.

**시간 여행.** `graph.get_state_history(thread_id)`는 전체 체크포인트 로그를 반환한다. 어떤 이전 `checkpoint_id`든 `graph.invoke`에 전달하면 그 지점에서 포크한다. 디버깅("모델이 도구 B를 골랐다면?")과 프로덕션 트레이스를 재생하는 회귀 테스트에 훌륭하다.

### 리듀서가 핵심이다 (Reducers are the point)

모든 상태 필드는 리듀서를 가진다. 대부분의 기본값은 괜찮다. 새 값이 옛 값을 덮어쓰기 때문이다. 하지만 메시지 리스트는 새 메시지가 교체 대신 추가되도록 `operator.add`가 필요하다. 병렬 간선은 리듀서로 업데이트를 병합한다. 두 노드가 모두 `messages`를 업데이트하는데 `Annotated[list, add_messages]`를 잊었다면, 두 번째가 조용히 이기고 턴의 절반을 잃는다. 리듀서는 이 라이브러리에서 유일하게 미묘한 부분이다. 이것만 제대로 하면 나머지는 조합된다.

### 네 노드로 만드는 ReAct 그래프 (The ReAct graph in four nodes)

프로덕션 ReAct 에이전트는 네 노드와 두 간선이다:

1. `agent`: 현재 메시지 기록으로 LLM을 호출한다. 어시스턴트 메시지(tool_calls를 포함할 수 있음)를 반환한다.
2. `tools`: 마지막 어시스턴트 메시지의 모든 tool_calls를 실행하고, 도구 결과를 도구 메시지로 추가한다.
3. `agent`로부터의 조건부 간선으로, 마지막 메시지에 tool_calls가 있으면 `tools`로, 아니면 `END`로 라우팅한다.
4. `tools`에서 `agent`로 돌아가는 정적 간선.

그게 전부다. 약 40줄의 코드로 체크포인팅과 인터럽트, 스트리밍을 갖춘 전체 ReAct 루프(Thought → Action → Observation → Thought → …)를 얻는다.

### StateGraph 대 Send (팬아웃)

`Send(node_name, state)`는 노드가 병렬 서브그래프를 디스패치하게 한다. 예를 들어 에이전트가 세 개의 검색기를 동시에 쿼리하기로 결정한다. 각 `Send`는 대상 노드의 병렬 실행을 생성하고, 그 출력은 상태 리듀서로 병합된다. LangGraph는 이렇게 스레딩 프리미티브 없이 오케스트레이터-워커 패턴을 표현한다.

### 서브그래프 (Subgraphs)

컴파일된 그래프는 다른 그래프의 노드가 될 수 있다. 외부 그래프는 단일 노드를 보고, 내부 그래프는 자체 상태와 자체 체크포인트를 가진다. 슈퍼바이저-워커 에이전트는 이렇게 구축한다. 슈퍼바이저 그래프가 사용자 의도를 도메인별 워커 서브그래프로 라우팅한다.

```figure
l5-state-graph-ledger
```

## 직접 만들기 (Build It)

### 1단계: 상태와 노드

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`는 메시지 리스트가 덮어쓰기 대신 누적되게 하는 리듀서다. 그것을 잊는 것이 가장 흔한 LangGraph 버그다.

### 2단계: 스레드로 실행하기

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

모든 업데이트는 dict `{node_name: state_delta}`다. 프런트엔드는 이것들을 UI로 스트리밍해 사용자가 "에이전트가 생각 중… search_web 호출 중… 결과 받음… 답변 중"을 보게 할 수 있다.

### 3단계: 인간 참여 인터럽트 추가하기

노드를 표시하여 실행이 그것이 실행되기 전에 일시 정지하게 한다.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

상태, 체크포인트, 스레드 모두 인터럽트 전반에 걸쳐 영속화된다. 실행 중을 제외하고는 아무것도 메모리에 있지 않다.

### 4단계: 디버깅을 위한 시간 여행

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

입력으로 `None`을 전달하면 주어진 체크포인트에서 재생하고, 값을 전달하면 재개 전에 그 체크포인트 상태에 대한 업데이트로 추가한다. 전체 대화를 다시 실행하지 않고 나쁜 에이전트 실행을 재현하는 방식이다.

### 5단계: 프로덕션을 위해 체크포인터 교체하기

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, Postgres가 출시되어 있다. `MemorySaver`는 테스트용이다. 재시작 전반에 걸쳐 영속화하는 것은 무엇이든 실제 저장소를 원한다.

## 스킬 (The Skill)

> 에이전트는 `while True` 루프가 아니라 그래프로 구축한다.

LangGraph에 손을 뻗기 전에 60초 설계를 하라.

1. **노드 이름 짓기.** 별개의 결정이나 부수 효과(side effect)가 있는 액션은 모두 노드다. "에이전트가 생각한다," "도구가 실행된다," "리뷰어가 승인한다," "응답이 스트리밍된다." 이것들을 나열할 수 없다면 작업이 아직 에이전트 형태가 아니다.
2. **상태 선언하기.** 모든 리스트 필드에 리듀서를 가진 최소 TypedDict를 둔다. 모든 것을 `messages`에 쑤셔넣지 말고, 작업별 필드(작업 중인 `plan`, `budget` 카운터, `retrieved_docs` 리스트)를 최상위로 끌어올려라.
3. **간선 그리기.** 다음 단계가 모델 출력에 의존하지 않는 한 정적. 모든 조건부 간선은 이름 붙은 분기를 가진 라우터 함수가 필요하다.
4. **체크포인터를 미리 선택하기.** 테스트는 `MemorySaver`, 그 외 모든 것은 Postgres/Redis/SQLite. 하나 없이 출시하지 마라. 체크포인터가 없으면 재개도, 인터럽트도, 시간 여행도 없다.
5. **인터럽트를 도구 실행 후가 아니라 전에 결정하기.** 승인은 해를 끼치기 전에 취소할 수 있도록 부수 효과가 있는 노드로 들어가는 간선에 두고, 검증은 나쁜 호출을 싸게 거부할 수 있도록 모델에서 나가는 간선에 둔다.
6. **기본적으로 스트리밍하기.** UI에는 `mode="updates"`, 모델 노드 내부의 토큰 수준 스트리밍에는 `mode="messages"`, 평가 중 전체 스냅샷에는 `mode="values"`.

체크포인터가 없는 LangGraph 에이전트를 출시하기를 거부하라. 부수 효과 *후에* 인터럽트하는 것을 출시하기를 거부하라. `add_messages`를 리듀서로 갖지 않은 `messages` 필드를 출시하기를 거부하라.

## 연습 문제 (Exercises)

1. **쉬움.** 위의 네 노드 ReAct 그래프를 계산기 도구와 웹 검색 도구로 구현하라. 두 턴 대화에 대해 `list(app.get_state_history(config))`가 최소 네 개의 체크포인트를 반환하는지 검증하라.
2. **중간.** `agent` 전에 실행되어 구조화된 `plan: list[str]`을 상태에 쓰는 `planner` 노드를 추가하라. `agent`가 plan 단계를 완료로 표시하게 하라. `plan`이 체크포인트 재개 전반에 걸쳐 손실되면(잘못된 리듀서) 테스트를 실패시켜라.
3. **어려움.** `Send`를 사용해 세 서브그래프(`researcher`, `writer`, `reviewer`) 사이를 라우팅하는 슈퍼바이저 그래프를 구축하라. 각 서브그래프는 자체 상태와 체크포인터를 가진다. 사람이 연구 브리프를 승인할 수 있도록 외부 그래프에 `interrupt_before=["writer"]`를 추가하라. 이전 체크포인트에서의 시간 여행이 포크된 분기만 다시 실행하는지 확인하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 그래프" | 컴파일 전에 노드와 간선을 추가하는 빌더 객체 |
| 리듀서(Reducer) | "필드가 병합되는 방식" | 노드가 그 필드에 대한 업데이트를 반환할 때 적용되는 함수 `(old, new) -> merged`; 기본은 덮어쓰기, `add_messages`는 추가 |
| 스레드(Thread) | "대화 ID" | 하나의 세션에 대한 모든 체크포인트를 스코프하는 `thread_id` 문자열 |
| 체크포인트(Checkpoint) | "일시 정지된 상태" | 노드 전이 후 전체 그래프 상태의 영속화된 스냅샷으로, `(thread_id, checkpoint_id)`로 키가 매겨짐 |
| 인터럽트(Interrupt) | "사람을 위한 일시 정지" | `interrupt_before` / `interrupt_after`가 노드 경계에서 실행을 멈춘다; `Command(resume=...)`로 재개 |
| 시간 여행(Time-travel) | "이전 단계에서 포크" | `graph.invoke(None, config_with_old_checkpoint_id)`가 그 체크포인트에서 앞으로 재생한다 |
| Send | "병렬 서브그래프 디스패치" | 노드가 대상 노드의 N개 병렬 실행을 생성하기 위해 반환할 수 있는 생성자 |
| 서브그래프(Subgraph) | "노드로서의 컴파일된 그래프" | 다른 그래프에서 노드로 사용되는 컴파일된 StateGraph; 자체 상태 스코프를 보존한다 |

## 더 읽을거리 (Further Reading)

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/): StateGraph, 리듀서, 체크포인터, 인터럽트에 관한 표준 레퍼런스
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/): 이 레슨이 사용하는 멘탈 모델, 출처에서 직접
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/): Postgres/SQLite/Redis 저장소, 체크포인트 네임스페이스, 스레드 ID에 관한 세부사항
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): `interrupt_before`, `interrupt_after`, `Command(resume=...)`, 그리고 상태 편집 패턴
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629): 모든 LangGraph 에이전트가 구현하는 패턴; 추론 트레이스 근거를 위해 읽으라
- [Anthropic: Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) 어떤 그래프 형태(체인, 라우터, 오케스트레이터-워커, 평가자-옵티마이저)를 언제 선호할지
- Phase 11 · 09 (Function Calling): 모든 LangGraph 에이전트 노드가 재사용하는 도구 호출 프리미티브
- Phase 11 · 14 (Model Context Protocol): MCP 어댑터를 통해 LangGraph `ToolNode`에 플러그인되는 외부 도구 발견
- Phase 11 · 17 (Agent framework tradeoffs): CrewAI, AutoGen, Agno 대신 LangGraph를 언제 선택할지
