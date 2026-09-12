# 에이전트 프레임워크 트레이드오프: LangGraph vs CrewAI vs AutoGen vs Agno (Agent Framework Tradeoffs)

> 모든 프레임워크는 같은 데모(연구 에이전트가 리포트를 작성한다)를 팔고 같은 버그(상태 스키마가 오케스트레이션 계층과 싸운다)를 숨긴다. 추상화가 내 문제의 형태와 맞아떨어지는 프레임워크를 골라라. 나머지는 모두 두 번 작성하게 되는 글루(glue)다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45분

## 문제 (The Problem)

LLM 호출이 하나 이상 필요한 작업이 있다고 하자. 연구 워크플로(계획, 검색, 요약, 인용)일 수도 있고, 코드 리뷰 파이프라인(diff 파싱, 비평, 패치, 검증)일 수도 있다. 항공권을 예약하고, 이메일을 쓰고, 경비 보고서를 제출하는 다중 턴 어시스턴트일 수도 있다. 그래서 프레임워크를 하나 고른다.

3일이 지나면 프레임워크의 추상화가 새는 것이 보인다. CrewAI는 역할을 주지만 "연구자"가 구조화된 계획을 "작성자"에게 넘겨야 할 때 발목을 잡는다. AutoGen은 에이전트(agent) 간 채팅을 주지만 일급(first-class) 상태가 없어서 체크포인트가 대화 로그의 pickle로 남는다. LangGraph는 상태 그래프를 주지만 에이전트가 무엇을 할지 알기도 전에 모든 전이의 이름을 짓도록 강요한다. Agno는 단일 에이전트 추상화를 주는데, 세 개의 동시 워커로 팬아웃하려 하면 비명을 지른다.

해결책은 "가장 좋은 프레임워크를 골라라"가 아니다. 프레임워크의 핵심 추상화를 내 문제의 형태와 맞추는 일이다. 이 레슨은 그 지도를 그린다.

## 개념 (The Concept)

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

네 개의 프레임워크가 2026년 풍경을 지배한다. 핵심 추상화는 서로 같지 않다.

| 프레임워크 | 핵심 추상화 | 최적 적합 | 최악 적합 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`: 타입 지정된 상태, 노드, 조건부 간선, 체크포인터. | 명시적 상태와 인간 참여 인터럽트를 가진 워크플로; 시간 여행 디버깅이 필요한 프로덕션 에이전트. | 토폴로지가 알려지지 않은 느슨한, 역할 주도 브레인스토밍. |
| **CrewAI** | `Crew`: 역할(목표, 배경 이야기), 작업, 프로세스(순차 또는 계층). | 짧은 선형/계층 계획을 가진 롤플레잉 또는 페르소나 주도 워크플로. | crew의 턴 기록을 넘어서는 상태가 있는 모든 것; 복잡한 분기. |
| **AutoGen** | `ConversableAgent` 쌍: 종료 조건까지 턴을 주고받으며 말하는 둘 이상의 에이전트. | 사고가 채팅에서 창발하는 다중 에이전트 *대화*(교사-학생, 제안자-비평가, 행위자-리뷰어). | 알려진 DAG를 가진 결정론적 워크플로; 재시작 전반에 걸친 영속 상태가 필요한 모든 것. |
| **Agno** | `Agent`: 단일 LLM + 도구 + 메모리, 팀으로 조합 가능. | 빨리 만드는 단일 에이전트와 경량 팀; 강력한 다중 모달리티와 내장 저장소 드라이버. | 커스텀 리듀서(reducer)를 가진 깊고 명시적으로 분기된 그래프. |

### "추상화"가 실제로 의미하는 것 (What "abstraction" actually means)

프레임워크의 핵심 추상화는 아키텍처를 피칭할 때 화이트보드에 그리는 그림이다.

- **LangGraph** → 그래프를 그린다. 노드는 단계, 간선은 전이이고, 모든 지점의 상태 객체는 타입이 지정되어 있다. 멘탈 모델은 상태 기계(state machine)다.
- **CrewAI** → 조직도를 그린다. 각 역할은 직무 기술서를 가지고 매니저가 작업을 라우팅한다. 멘탈 모델은 작은 전문가 팀이다.
- **AutoGen** → Slack DM을 그린다. 두 에이전트가 서로 메시지를 주고받고, 진행자가 필요하면 세 번째가 합류한다. 멘탈 모델은 채팅이다.
- **Agno** → 도구가 매달린 박스 하나를 그린다. 팀을 만들려면 박스들을 나란히 둔다. 멘탈 모델은 "배터리 포함 에이전트"다.

### 상태 질문 (The state question)

대부분의 프레임워크 선택이 프로덕션에서 무너지는 지점이 바로 상태다.

- **LangGraph.** 타입 지정된 상태(`TypedDict` 또는 Pydantic 모델), 필드별 리듀서, 일급 체크포인터(SQLite/Postgres/Redis). 재개, 인터럽트, 시간 여행이 공짜다. *(Phase 11 · 16을 보라.)*
- **CrewAI.** 상태는 `context` 필드를 통해 작업 사이에 문자열로 흐르거나, `output_pydantic`을 통해 구조화된다. 기본으로는 crew별 영속 저장소가 없다. crew가 재시작을 견뎌야 한다면 직접 덧붙여야 한다.
- **AutoGen.** 상태는 채팅 기록과 사용자 정의 `context`다. 대화 트랜스크립트는 영속화되지만, 임의의 워크플로 상태는 어댑터를 직접 작성하지 않는 한 영속화되지 않는다.
- **Agno.** `storage=`를 통해 `Agent`에 부착된 내장 저장소 드라이버(SQLite, Postgres, Mongo, Redis, DynamoDB): 대화 세션과 사용자 메모리가 자동으로 영속화된다. 전체 그래프 체크포인터가 아니라 세션 저장소다.

### 분기 질문 (The branching question)

모든 비자명한 에이전트는 분기한다. 누가 분기를 결정하는지가 중요하다.

- **LangGraph**: 개발자가 조건부 간선을 통해 결정한다. 라우팅은 이름 붙은 분기를 가진 Python 함수다. 분기는 컴파일된 그래프에서 일급이고, 체크포인터가 어느 분기를 택했는지 기록한다.
- **CrewAI**: 계층 모드에서는 매니저가 결정하고, 순차 모드에서는 빌드 시점에 개발자가 결정한다. 라우팅은 작업 목록에 암시적이다. 매니저의 프롬프트(prompt) 밖에는 일급 "if"가 없다.
- **AutoGen**: 에이전트들이 채팅을 통해 결정한다. 분기는 다음에 누가 말하는지에서 창발한다. `GroupChatManager`가 다음 화자를 선택한다. `speaker_selection_method`를 손으로 작성할 수 있지만 기본은 LLM 주도다.
- **Agno**: 에이전트가 다음에 어느 도구를 호출할지로 결정한다. 팀은 코디네이터/라우터/협력자 모드를 가지며, 그 너머의 분기는 개발자의 책임이다.

### 관측성 질문 (The observability question)

- **LangGraph**: LangSmith 또는 어떤 OTel 익스포터를 통한 OpenTelemetry. 모든 노드 전이가 트레이스 스팬이고, 체크포인트는 재생 가능한 트레이스를 겸한다. LangSmith가 일급 옵션이며 Langfuse/Phoenix도 어댑터를 가진다.
- **CrewAI**: 2025년 후반 이후 일급 OpenTelemetry; Langfuse, Phoenix, Opik, AgentOps와의 통합.
- **AutoGen**: `autogen-core`를 통한 OpenTelemetry 통합; AgentOps와 Opik이 커넥터를 가진다. 트레이싱 입도는 노드별이 아니라 에이전트 메시지별이다.
- **Agno**: 내장 `monitoring=True` 플래그에 OpenTelemetry 익스포터를 더한다. 세션 트레이스를 위해 Langfuse와 긴밀하게 통합된다.

### 비용과 지연 시간 (Cost and latency)

네 프레임워크 모두 호출당 오버헤드(프레임워크 로직, 검증, 직렬화)를 추가한다. 오버헤드 증가의 대략적인 순서는 Agno ≈ LangGraph < CrewAI ≈ AutoGen이다. 차이는 프레임워크가 추가 LLM 라우팅을 얼마나 많이 하느냐가 좌우한다. CrewAI의 계층 매니저는 다음에 누가 갈지 결정하는 데 토큰(token)을 쓰고, AutoGen의 `GroupChatManager`도 마찬가지다. LangGraph는 직접 `llm.invoke`를 작성하는 곳에서만 토큰을 쓴다. Agno의 단일 에이전트 경로는 얇다.

실행당 비용이 중요할 때는 LLM 선택 라우팅보다 명시적 라우팅(LangGraph 간선, AutoGen `speaker_selection_method`)을 택하라.

### 상호운용성 (Interoperability)

- **LangGraph** ↔ **LangChain** 도구, 검색기, LLM. 일급 MCP 어댑터(도구가 MCP 서버로 임포트됨).
- **CrewAI** ↔ 도구는 `BaseTool`을 상속한다. LangChain 도구, LlamaIndex 도구, MCP 도구가 모두 적응되어 들어온다. `allow_delegation=True`를 통한 crew 간 위임.
- **AutoGen** → `FunctionTool`이 어떤 Python 호출 가능 객체든 감싼다. MCP 어댑터를 쓸 수 있다. 에이전트 간 패턴을 위해 AG2 생태계와 긴밀하게 결합된다.
- **Agno** → `@tool` 데코레이터 또는 BaseTool 서브클래스; MCP 어댑터. 도구는 에이전트와 팀 전반에 공유될 수 있다.

## 스킬 (The Skill)

> 주어진 프레임워크가 주어진 에이전트 문제에 왜 옳은지 한 문장으로 설명할 수 있다.

빌드 전 체크리스트:

1. **형태 그리기.** 이것은 그래프(타입 지정된 상태, 이름 붙은 전이)인가? 롤플레이(전문가들이 작업을 넘긴다)인가? 채팅(에이전트들이 끝날 때까지 말한다)인가? 도구를 가진 단일 에이전트인가?
2. **누가 분기하는지 결정하기.** 개발자가 결정하는 분기 → LangGraph. 매니저 에이전트가 결정 → CrewAI 계층. 채팅 창발 → AutoGen. 도구 호출이 결정 → Agno.
3. **상태 예산 확인하기.** 체크포인트에서 재개가 필요한가? 시간 여행? 실행 중 인간 인터럽트? 그렇다면 LangGraph가 기본이다. Agno 세션이 대화 범위 상태를 다룬다.
4. **비용 예산 확인하기.** LLM 선택 라우팅은 턴당 추가 토큰이 든다. 에이전트가 하루에 수천 번 실행된다면 명시적 라우팅을 택하라.
5. **프레임워크 오버헤드 예산 잡기.** 모든 프레임워크는 또 하나의 의존성이다. 작업이 두 번의 LLM 호출과 하나의 도구뿐이라면 30줄의 순수 Python을 작성하라. 프레임워크 없음보다 싼 프레임워크는 없다.

그래프, 조직도, 채팅, 또는 에이전트 박스를 그릴 수 있기 전에는 프레임워크에 손대지 마라. 실제로 필요한 것에 대해 상태 모델과 싸우도록 강요하는 프레임워크를 고르지 마라.

## 결정 행렬 (The Decision Matrix)

| 문제 형태 | 선호 프레임워크 | 이유 |
|---------------|---------------------|-----|
| 타입 지정된 상태, 인간 승인, 장시간 실행을 가진 워크플로 DAG | LangGraph | 일급 상태, 체크포인터, 인터럽트, 시간 여행. |
| 별개의 역할을 가진 연구 / 작성 파이프라인 | CrewAI (순차) 또는 LangGraph 서브그래프 | 작업별 역할은 CrewAI에서 표현하기 싸다; 분기가 복잡해지면 LangGraph로 확장하라. |
| 제안자-비평가 또는 교사-학생 대화 | AutoGen | 두 에이전트 채팅이 그것의 네이티브 형태다. |
| 도구, 세션, 메모리를 가진 단일 에이전트 | Agno | 가장 얇은 설정, 내장 저장소와 메모리. |
| 리듀서를 가진 수천 개의 병렬 팬아웃 | LangGraph + `Send` | 일급 병렬 디스패치 API를 가진 유일한 것. |
| 빠른 프로토타입, 프레임워크 약속 없음 | 순수 Python + 프로바이더 SDK | 프레임워크 없음이 가장 빠른 프레임워크다. |

```figure
l5-framework-fit
```

## 연습 문제 (Exercises)

1. **쉬움.** 같은 작업("Anthropic의 본사를 연구하고, 200단어 브리프를 쓰고, 출처를 인용하라")을 가져와 LangGraph(네 노드: plan, search, write, cite)와 CrewAI(세 역할: researcher, writer, editor)에서 구현하라. 실행당 토큰 비용과 코드 줄 수를 보고하라.
2. **중간.** 같은 작업을 AutoGen(researcher ↔ writer 채팅, editor가 `GroupChat`을 통해 합류)과 Agno(`search_tools`와 `write_tools`를 가진 단일 에이전트, 더하기 세션 저장소)에서 구축하라. 네 구현을 (a) 실행당 비용, (b) 크래시 후 재개 능력, (c) write 단계 전에 인간 승인을 주입하는 능력에 대해 순위 매겨라.
3. **어려움.** 짧은 문제 설명(JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`)을 받아 한 문장 정당화와 함께 추천을 반환하는 결정 트리 스크립트 `pick_framework.py`를 구축하라. 직접 설계한 여섯 가지 사례에서 검증하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 오케스트레이션(Orchestration) | "에이전트들이 어떻게 조율하는지" | 다음에 어느 노드/역할/에이전트가 실행될지 결정하는 계층 |
| 영속 상태(Durable state) | "재시작 후 재개" | 체크포인트나 세션 저장소에 부착되어 프로세스 죽음을 견디는 상태 |
| LLM 선택 라우팅(LLM-selected routing) | "모델이 결정하게 하라" | 플래너 LLM이 매 턴 다음 단계를 고른다; 유연하지만 모든 결정에 토큰을 지불한다 |
| 명시적 라우팅(Explicit routing) | "개발자가 결정한다" | Python 함수나 정적 간선이 다음 단계를 고른다; 싸고 감사 가능 |
| Crew | "CrewAI 팀" | 역할 + 작업 + 프로세스(순차 또는 계층)가 단일 실행 가능한 것으로 묶임 |
| GroupChat | "AutoGen의 다중 에이전트 채팅" | 화자 선택기를 가진 N개 에이전트 간의 관리된 대화 |
| Team (Agno) | "다중 에이전트 Agno" | 에이전트 집합에 대한 라우트 / 코디네이트 / 협력 모드 |
| StateGraph | "LangGraph의 그래프" | 타입 지정된 상태, 노드, 조건부 간선, 체크포인터 추상화 |

## 더 읽을거리 (Further Reading)

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/): StateGraph, 체크포인터, 인터럽트, 시간 여행
- [CrewAI documentation](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes
- [AutoGen documentation](https://microsoft.github.io/autogen/): ConversableAgent, GroupChat, 팀, 도구
- [Agno documentation](https://docs.agno.com/): Agent, Team, Workflow, 저장소, 메모리
- [Anthropic: Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) 프레임워크 비종속적 패턴 라이브러리(프롬프트 체이닝, 라우팅, 병렬화, 오케스트레이터-워커, 평가자-옵티마이저)
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629): 모든 프레임워크가 꾸며내는 루프
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155): AutoGen의 설계 논문
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442): CrewAI 스타일 페르소나 스택이 기반하는 롤플레이 토대
- Phase 11 · 16 (LangGraph): 이 레슨이 벤치마킹하는 프레임워크
- Phase 11 · 19 (Reflexion): LangGraph에는 깔끔하게 매핑되지만 CrewAI에는 어색하게 매핑되는 패턴
- Phase 11 · 22 (Production observability): 당신이 고른 프레임워크가 무엇이든 계측하는 방법
