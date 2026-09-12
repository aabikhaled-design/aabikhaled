# LangGraph: 상태 보존 그래프와 내구성 있는 실행

> LangGraph는 저수준 상태 보존 오케스트레이션(low-level stateful orchestration)의 2026년 레퍼런스다. 에이전트(agent)는 상태 기계(state machine)이고, 노드(node)는 함수이며, 간선(edge)은 전이(transition)이고, 상태는 불변(immutable)이며 매 단계 이후 체크포인트(checkpoint)된다. 어떤 실패에서든 멈춘 바로 그 지점에서 재개한다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- LangGraph의 핵심 모델을 기술하기: 불변 상태, 함수 노드, 조건부 간선(conditional edge), 단계 후 체크포인트를 갖춘 상태 기계.
- 문서가 강조하는 네 가지 능력의 이름을 대기: 내구성 있는 실행(durable execution), 스트리밍(streaming), 휴먼 인 더 루프(human-in-the-loop), 포괄적 메모리.
- LangGraph가 지원하는 세 가지 오케스트레이션 토폴로지(topology)를 설명하기: 슈퍼바이저(supervisor), 피어 투 피어(peer-to-peer, swarm), 계층적(hierarchical, 중첩 서브그래프).
- 불변 상태, 조건부 간선, 체크포인트/재개 주기를 갖춘 stdlib 상태 그래프를 구현하기.

## 문제 (The Problem)

에이전트와 워크플로(workflow)는 한 가지 문제를 공유한다. 40단계 실행이 38단계에서 실패하면, 처음부터 다시 시작하는 것이 아니라 38단계에서 재개하고 싶다. 이류(second-class) 상태 모델에서는 새 실행을 가정하는 라이브러리에 맞춰 재시도를 임시방편으로 끼워 넣게 된다.

LangGraph의 설계상 답: 상태는 일급(first-class) 타입 객체이고, 변경은 명시적이며, 체크포인트는 매 노드 이후 영속된다. 재개는 `load_state(session_id)` 호출 하나다.

## 개념 (The Concept)

### 그래프

그래프는 다음으로 정의된다.

- **상태 타입(State type).** 모든 노드가 읽고 변경하는 타입 딕셔너리(typed dict, 또는 Pydantic 모델).
- **노드(Nodes).** 순수 함수 `(state) -> state_update`. 업데이트는 반환 후 상태에 병합된다.
- **간선(Edges).** 노드 사이의 조건부 또는 직접 전이.
- **진입과 종료(Entry and exit).** `START`와 `END` 센티널(sentinel) 노드가 경계를 표시한다.

예: `classify`, `refund`, `bug`, `sales`, `done` 노드를 가진 에이전트: 그래프로서의 라우팅 워크플로.

### 내구성 있는 실행

각 노드가 반환된 후, 런타임(runtime)은 상태를 직렬화(serialize)하여 체크포인터(checkpointer, SQLite, Postgres, Redis, 커스텀)에 쓴다. N단계에서 실패하면, 런타임은 `resume(session_id)`로 정확한 상태를 가지고 N+1단계부터 이어갈 수 있다.

LangGraph 문서는 중요한 프로덕션 사용자로 Klarna, Uber, J.P. Morgan을 명시적으로 든다. 주장은 그래프 형태가 아니라, 그래프 형태에 체크포인팅을 더하면 복구가 저렴해진다는 것이다.

### 스트리밍

모든 노드는 부분 출력을 yield할 수 있다. 그래프는 노드별 델타(delta) 이벤트를 호출자에게 스트리밍하여, 그래프가 실행되는 동안 UI가 업데이트되게 한다.

### 휴먼 인 더 루프

노드 사이에서 상태를 검사하고 수정한다. 구현은 이렇다. 임계 노드 전에 일시 정지해 상태를 사람에게 드러내고, 수정을 받아 재개한다. 상태가 이미 직렬화되어 있으므로 체크포인터가 이를 쉽게 만든다.

### 메모리

단기(short-term, 한 실행 안에서(상태 안의 대화 기록)와 장기(long-term, 실행에 걸쳐) 체크포인터와 별도의 장기 저장소를 통해 영속). LangGraph는 외부 메모리 시스템(Mem0, 커스텀)과 도구로 연동한다.

### 세 가지 토폴로지

1. **슈퍼바이저(Supervisor).** 중앙 라우터 LLM이 전문가 서브에이전트(subagent)에게 디스패치한다. `langgraph-supervisor`의 `create_supervisor()`(단, 2026년 LangChain 팀은 컨텍스트를 더 잘 제어하려면 도구 호출로 직접 하라고 권장한다).
2. **스웜(Swarm) / 피어 투 피어.** 에이전트들이 공유 도구 표면(shared tool surface)을 통해 직접 핸드오프(handoff)한다. 중앙 라우터가 없다.
3. **계층적(Hierarchical).** 슈퍼바이저가 하위 슈퍼바이저를 관리하며, 중첩 서브그래프(nested subgraph)로 구현된다.

### 이 패턴이 잘못되는 지점

- **체크포인트가 너무 작음.** 대화 턴(turn)만 체크포인트하면 도구 상태와 메모리 쓰기가 복구 불가능해진다. 전체 상태가 직렬화되어야 한다.
- **비결정론적 노드.** 재개는 노드 입력이 동일한 상태 업데이트를 만든다고 가정한다. 랜덤 시드, 벽시계 시간(wall-clock), 외부 API는 포착되어야 한다.
- **조건부 간선의 과용.** 모든 간선이 조건부인 그래프는 추론할 수 없는 상태 기계다. 가끔의 분기를 갖춘 선형 체인을 선호하라.

```figure
langgraph-state
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 상태 보존 그래프를 구현한다.

- `State`: `messages`, `step`, `route`, `output`, `human_approval`을 가진 타입 딕셔너리.
- `Node`: 상태를 받아 업데이트 딕셔너리를 반환하는 호출 가능 객체(callable).
- `StateGraph`: 노드 + 간선 + 조건부 간선 + 실행 + 재개.
- `SQLiteCheckpointer`(인메모리 가짜): 매 노드 이후 상태를 직렬화하고, `load(session_id)`로 복원한다.
- 데모 그래프: classify -> branch(refund / bug / sales) -> human gate -> send.

실행:

```
python3 code/main.py
```

트레이스는 첫 실행이 휴먼 게이트(human gate)에서 실패하고, 영속되며, 이후 재개하여 최종 출력을 만드는 것을 보여준다.

## 라이브러리로 써보기 (Use It)

- **LangGraph**: 레퍼런스, 프로덕션 준비 완료. `create_react_agent`, `create_supervisor`를 사용하거나 자신의 그래프를 만들어라.
- **AutoGen v0.4**(Lesson 14): 고동시성 시나리오를 위한 액터 모델 대안.
- **Claude Agent SDK**(Lesson 17): 내장 세션 저장소를 갖춘 관리형 하니스(managed harness).
- **커스텀**: 상태 형태나 체크포인터 백엔드에 대한 정확한 제어가 필요할 때.

## 산출물 (Ship It)

`outputs/skill-state-graph.md`는 체크포인팅과 재개가 배선된 LangGraph 형태의 상태 그래프를 임의의 대상 런타임에 생성한다.

## 연습 문제 (Exercises)

1. 분류 신뢰도가 임계값 미만일 때 `classify`에서 `end`로 가는 조건부 간선을 추가하라. 사람이 `route`를 수동으로 설정한 뒤 실행을 재개하라.
2. SQLite 유사 가짜를 진짜 SQLite 체크포인터로 교체하라. 단계별 직렬화 오버헤드를 측정하라.
3. 병렬 간선을 구현하라: 두 노드가 동시에 실행되고, 커스텀 리듀서(reducer)로 병합된다. 여기서 불변 상태가 무엇을 사주는가?
4. `langgraph-supervisor` 레퍼런스를 읽어라. 토이를 `create_supervisor`로 포팅하라. 트레이스 형태를 비교하라.
5. 스트리밍을 추가하라: 각 노드가 실행되는 동안 부분 상태를 yield한다. 도착하는 대로 델타를 출력하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 상태 그래프(State graph) | "상태 기계로서의 에이전트" | 타입 상태 + 노드 + 간선 + 리듀서 |
| 체크포인터(Checkpointer) | "영속 백엔드" | 매 노드 이후 상태를 직렬화; 재개를 가능하게 함 |
| 리듀서(Reducer) | "상태 병합자" | 현재 상태와 노드의 업데이트를 결합하는 함수 |
| 조건부 간선(Conditional edge) | "분기" | 상태의 함수에 의해 선택되는 간선 |
| 서브그래프(Subgraph) | "중첩 그래프" | 다른 그래프 안에서 노드로 사용되는 그래프 |
| 내구성 있는 실행(Durable execution) | "실패에서 재개" | 마지막으로 성공한 노드에서 정확한 상태로 재시작 |
| 슈퍼바이저(Supervisor) | "라우터 LLM" | 전문가 서브에이전트를 위한 중앙 디스패처 |
| 스웜(Swarm) | "P2P 에이전트" | 에이전트가 공유 도구로 핸드오프; 중앙 라우터 없음 |

## 더 읽을거리 (Further Reading)

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 레퍼런스 문서
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/): 슈퍼바이저 패턴 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/): 액터 모델 대안
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview): 세션 저장소와 서브에이전트
