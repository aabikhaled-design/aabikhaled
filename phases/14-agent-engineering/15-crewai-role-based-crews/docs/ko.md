# CrewAI: 역할 기반 크루와 플로우

> CrewAI는 2026년 역할 기반(role-based) 멀티 에이전트(multi-agent) 프레임워크다. 네 가지 원시 요소(primitive): Agent, Task, Crew, Process. 두 가지 최상위 형태: Crews(자율적, 역할 기반 협업)와 Flows(이벤트 기반, 결정론적). 문서는 직설적이다: "프로덕션 준비 애플리케이션이라면, Flow로 시작하라."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- CrewAI의 네 가지 원시 요소(Agent, Task, Crew, Process)의 이름을 대고 각각이 무엇을 소유하는지 설명하기.
- Sequential, Hierarchical, 그리고 계획 중인 Consensus 프로세스를 구분하기; 워크로드마다 하나를 고르기.
- Crews(자율 역할 기반)와 Flows(이벤트 기반 결정론적)를 구분하고, 문서의 프로덕션 권장을 설명하기.
- `@tool` 데코레이터와 `BaseTool` 서브클래스로 도구를 배선하기; 구조화 출력(structured output)과 자유 텍스트를 비교 추론하기.
- 네 가지 CrewAI 메모리 타입의 이름을 대고 각각이 언제 보상을 주는지 설명하기.
- 브리프(brief)를 생성하는 stdlib 세 에이전트 크루(연구자, 작가, 편집자)를 구현하기.
- 세 가지 CrewAI 실패 모드를 포착하기: 프롬프트 비대화(prompt-bloat), 매니저 LLM 세금(manager-LLM tax), 깨지기 쉬운 핸드오프(brittle handoff).

## 문제 (The Problem)

멀티 에이전트 프레임워크를 채택하는 팀들은 같은 벽에 부딪힌다. "자율 협업"은 데모에서는 훌륭하게 들린다. 그러다 고객이 버그를 제기하면 결정론적 재현(deterministic replay)이 필요해진다. 또는 재무팀이 LLM 라우팅 크루가 실행당 얼마인지 묻는다. 또는 온콜(on-call) 담당자가 새벽 3시에 어느 에이전트가 멈췄는지 알아야 한다.

자유 형식 LLM 라우팅 크루는 그중 무엇에도 깔끔하게 답하지 못한다. 순수 DAG는 그 모두에 답하지만 브레인스토밍 에이전트가 필요로 하는 탐색적 형태를 잃는다.

CrewAI의 분할은 그 거래(trade)를 정직하게 다룬다. Crews는 협업적이고 역할 기반이며 탐색적인 작업을 위한 것이다. Flows는 이벤트 기반이고 코드가 소유하며 감사 가능한 프로덕션을 위한 것이다. 같은 프레임워크, 두 형태이며 표면(surface)마다 골라 쓴다.

## 개념 (The Concept)

### 네 가지 원시 요소

CrewAI의 표면은 작다. 이것을 외우면 나머지는 설정(config)이다.

- **Agent.** `role + goal + backstory + tools + (선택) llm`. 백스토리(backstory)는 하중을 지탱한다(load-bearing). 그것이 톤, 판단, 에이전트가 멈추는 시점을 형성한다. 도구(tools)는 에이전트가 호출할 수 있는 함수다(아래에 더 있음).
- **Task.** `description + expected_output + agent + (선택) context + (선택) output_pydantic`. 재사용 가능한 작업 단위. `expected_output`은 계약(contract)이다. `context`는 출력이 전달되는 상류(upstream) 작업들을 나열한다. `output_pydantic`은 구조화된 형태를 강제한다.
- **Crew.** 컨테이너. `agents` 목록, `tasks` 목록, `process`, 그리고 선택적 `memory` + `verbose` + `manager_llm` 설정을 소유한다.
- **Process.** 실행 전략. Sequential, Hierarchical, Consensus(계획 중). 실행의 형태를 고른다.

에이전트는 서로를 직접 보지 않는다. 작업이 에이전트를 참조한다. Crew가 작업을 순서화한다. Process가 누가 다음 작업을 고를지 결정한다. 그것이 전체 멘탈 모델이다.

> **검증 기준** CrewAI 0.86 (2026-05). 더 새로운 버전은 프로세스 타입을 이름 변경하거나 병합할 수 있다. 특정 형태에 의존하기 전에 [CrewAI Processes 문서](https://docs.crewai.com/concepts/processes)를 확인하라.

### Sequential 대 Hierarchical 대 Consensus

- **Sequential.** 작업이 선언 순서대로 실행된다. 작업 N의 출력은 작업 N+1에 `context`로 제공된다. 가장 낮은 비용. 가장 예측 가능. 순서가 고정되어 있을 때 사용한다.
- **Hierarchical.** 매니저(manager) Agent(별도의 LLM 호출)가 전문가들 사이를 라우팅한다. CrewAI는 `manager_llm` 설정이나 기본값으로 매니저를 생성한다. 매니저는 매 라운드 다음 작업을 고르고 거부하거나 재라우팅할 수 있다. 네 명 이상의 전문가가 있고 순서가 진정으로 이전 출력에 의존할 때 사용한다.
- **Consensus.** 계획 중이며, 현재 공개 API에 구현되어 있지 않다. 문서는 미래의 투표 기반(voting-based) 프로세스를 위해 그 이름을 예약해 둔다. 오늘은 그것에 의존하지 마라.

Hierarchical은 모든 전문가 호출 위에 라운드당 LLM 호출(매니저)을 더한다. 5단계 실행에서 토큰 비용이 세 배가 될 수 있다. 라우팅이 필요할 때만 그 값을 치러라.

### Crews 대 Flows

이것이 2026년 문서가 앞세우는 프레이밍(framing)이다.

- **Crew.** LLM 주도 자율성. 프레임워크가 런타임에 형태를 고른다. 적합: 연구, 브레인스토밍, 초안, 경로 자체가 답의 일부인 모든 곳. 재현이 어렵다. 테스트가 어렵다. 프로토타이핑이 저렴하다.
- **Flow.** 개발자가 직접 소유하는 이벤트 기반 그래프. `@start`가 진입점을 표시한다. `@listen(topic)`은 다른 단계가 그 토픽을 방출할 때 발화하는 단계를 표시한다. 각 단계는 평범한 Python이다(내부적으로 Crew를 호출할 수 있다). 적합: 프로덕션. 관측 가능. 테스트 가능. 결정론적.

문서의 2026년 프로덕션 권장: Flow로 시작하라. 자율성이 그 비용을 정당화할 때 Flow 단계 안에서 `Crew.kickoff()` 호출로 Crews를 접어 넣어라(fold in). Flow는 감사 추적(audit trail)을 주고, Crew는 탐색을 준다. 고르지 말고, 조합하라.

### 도구 통합

Agent에 도구를 주는 세 가지 방식. 들어맞는 가장 단순한 것을 골라라.

1. **`@tool` 데코레이터.** 순수 함수가 도구가 된다. 시그니처(signature)가 스키마이고, 독스트링(docstring)이 LLM이 보는 설명이다. 일회성 헬퍼에 가장 좋다.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 서브클래스.** 명시적 인자 스키마, 비동기 지원, 재시도를 가진 클래스 기반 도구. 도구가 상태(클라이언트, 캐시)를 가지거나 구조화된 인자가 필요할 때 사용한다.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **내장 툴킷.** CrewAI는 일급(first-party) 어댑터를 제공한다: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. 한 번의 임포트로 배선된다.

구조화 출력은 Pydantic을 사용한다. Task에 `output_pydantic=MyModel`을 전달하라. CrewAI는 LLM 응답을 모델에 대해 검증하고, 강제 변환(coerce)하거나 재시도한다. 이를 빡빡한 `expected_output` 문자열과 짝지어라. 자유 텍스트 출력은 초안에는 괜찮다. 구조화 출력은 하류(downstream) Flows가 소비할 수 있는 것이다.

### 메모리 후크 (Memory hooks)

CrewAI는 네 가지 메모리 타입을 기본 제공한다. 이들은 조합된다: Crew는 네 가지를 한 번에 모두 활성화할 수 있다.

> **검증 기준** CrewAI 0.86 (2026-05). 최근 릴리스는 이 네 저장소를 감싸는 통합 `Memory` 시스템을 통해 모든 것을 라우팅한다. 아래 개념적 모델은 여전히 유효하지만, 더 새로운 버전에서는 공개 클래스 표면이 단일 `Memory` 진입점으로 무너질 수 있다. 현재 API는 [CrewAI 메모리 문서](https://docs.crewai.com/concepts/memory)를 확인하라.

- **Short-term.** 단일 실행 안의 대화 버퍼. 끝나면 지워진다.
- **Long-term.** 실행에 걸쳐 영속됨. 벡터 DB(기본 Chroma, 교체 가능)에 저장됨. 현재 작업과의 유사도로 검색됨.
- **Entity.** 개체별(per-entity) 사실. "고객 X는 엔터프라이즈 플랜이다." 유사도가 아니라 개체로 키가 매겨짐. 실행에 걸쳐 살아남음.
- **Contextual.** 조립 시점(assembly-time) 검색. 미리 로드되는 것이 아니라 Agent가 필요로 하는 순간에 관련 메모리를 끌어옴.

Crew에서 `memory=True` 또는 타입별 설정으로 활성화한다. 설정한 임베딩(embedding) 제공자가 이를 뒷받침한다(기본 OpenAI, 로컬로 교체 가능). 메모리는 CrewAI가 더 얇은 프레임워크에 맞서 값을 하는 곳 중 하나다. 순수 LangGraph는 이 각각을 직접 배선하도록 요구한다.

### CrewAI가 들어맞을 때

- 이름 붙은 역할과 협업 워크플로를 가진 세 명에서 여섯 명의 에이전트. 초안 작성, 리뷰, 계획, 브레인스토밍.
- 다음 단계에 대한 LLM의 판단이 가치의 일부인 라우팅(Hierarchical).
- 팀이 그래프 정의를 읽는 것보다 `role + goal + backstory`를 읽는 것을 더 행복해하는 모든 곳.

### CrewAI가 들어맞지 않을 때

- 엄격한 순서를 가진 결정론적 DAG. LangGraph(Lesson 13)를 사용하라. 그래프 형태가 올바른 추상화다. CrewAI의 역할 프레이밍은 마찰이다.
- 1초 미만 지연 시간 예산. Hierarchical은 왕복(round trip)을 더한다. Sequential조차 백스토리와 이전 출력을 포함하는 프롬프트를 직렬화한다.
- 단일 에이전트 루프. 프레임워크를 건너뛰어라. 에이전트 루프(Lesson 1)에 도구 레지스트리(registry)를 더하는 것이 더 짧다.

Lesson 17(Agent Framework Tradeoffs)이 이를 행렬로 펼쳐 놓는다. 짧은 버전: CrewAI는 "협업적 역할 기반" 모서리에 앉아 있다.

### 의존성 형태

LangChain과 독립적. Python 3.10에서 3.13. `uv`를 사용. 스타 수: [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) 참조(2026-05 기준 스냅샷). AWS Bedrock 통합이 문서화되어 있다. 벤더 벤치마크(benchmark)는 QA 워크로드에서 LangGraph 대비 상당한 속도 향상을 보고하지만, 방법론(데이터셋, 하드웨어, 평가 지표)이 공개되지 않았으므로 프레임워크 벤더의 수치는 방향성 참고용으로만 취급하라.

### 이 패턴이 잘못되는 지점

- **백스토리로 인한 프롬프트 비대화.** 에이전트당 2000단어 백스토리에 다섯 에이전트 크루는 첫 도구 호출 전에 컨텍스트 예산을 태운다. 백스토리를 200단어 미만으로 유지하라. 에이전트 간에 어구를 재사용하라. 하우스 스타일을 다섯 번 반복하지 마라.
- **매니저 LLM 토큰 세금.** Hierarchical 프로세스는 모든 전문가 호출 전에 매니저 LLM 호출을 더한다. 다섯 작업 크루에서는 다섯이 아니라 여섯 LLM 호출이고, 매니저 호출은 전체 작업 목록에 이전 출력을 더해 운반한다. 라우팅이 출력에 의존하지 않는 한 Sequential로 전환하라.
- **깨지기 쉬운 핸드오프.** 작업 N의 `expected_output`이 "개요(outline)"다. 작업 N+1이 그것을 `context`로 읽고 세 개 섹션을 파싱하려 한다. LLM은 네 개를 만들었다. 하류 Agent가 즉흥적으로 처리한다. 작업 N에 `output_pydantic`을 두어 작업 N+1이 자유 텍스트가 아니라 타입 객체를 읽게 하여 고쳐라.
- **프로덕션으로서의 Crew.** Flow 래퍼 없이 프로덕션에 출시된 자유 형식 Crew. 출력 변동성이 높다. 재현이 불가능하다. 온콜이 나쁜 실행과 좋은 실행을 diff할 수 없다. Flow로 감싸라.

```figure
ae-crew-vs-flow
```

## 직접 만들기 (Build It)

`code/main.py`는 두 형태의 stdlib 버전과 세 에이전트 크루를 구현한다.

형태:

- CrewAI의 표면에 맞는 `Agent`, `Task` 데이터클래스(dataclass).
- `SequentialCrew.kickoff(inputs)`는 작업을 선언 순서대로 실행하며 출력을 `context`로 엮는다.
- `HierarchicalCrew.kickoff(topic)`는 매 라운드 다음 전문가를 고르는 매니저 Agent를 더하고, "done"에서 멈춘다.
- `@start`와 `@listen(topic)` 데코레이터, 작은 이벤트 루프, 트레이스를 가진 `Flow`.
- CrewAI의 `@tool` 형태를 본뜬 `tool(name)` 데코레이터.
- `short_term`, `long_term`, `entity` 저장소를 가진 `Memory`. 모의(mocked) 유사도는 numpy를 사용한다.
- 모의 LLM 응답은 역할과 입력 접두사를 키로 하는 하드코딩된 문자열이다. 네트워크 없음. 결정론적.

구체적 데모: "agent engineering 2026"에 대한 브리프를 생성하는 연구자, 작가, 편집자 크루. 연구자가 (모의) 출처를 끌어온다. 작가가 초안을 쓴다. 편집자가 다듬는다. 같은 크루가 Flow를 통해 실행되어 결정론적 형태를 보여준다.

실행:

```bash
python3 code/main.py
```

트레이스는 다음을 다룬다: `context`를 통해 출력을 엮는 sequential 크루, 매니저 선택(연구자, 작가, 편집자, 그다음 "done")을 가진 hierarchical 크루, 명시적 토픽(`researched`, `drafted`, `edited`)으로 같은 세 단계를 실행하는 flow, `@tool`을 통해 라우팅되는 도구 호출, 두 번의 kickoff에 걸쳐 살아남는 long-term 메모리.

Crew 트레이스는 유동적이다. 매니저가 원칙적으로 재정렬할 수 있다. Flow 트레이스는 고정되어 있다. 그 선택이 이 레슨이다.

## 라이브러리로 써보기 (Use It)

- **CrewAI Flow**: 프로덕션용. Flow가 `Crew.kickoff()`를 호출하는 한 단계일지라도. Flow가 감사 경계(audit boundary)를 준다.
- **CrewAI Crew (Sequential)**: 명확한 순서의 협업 작업, 특히 초안과 리뷰 루프용.
- **CrewAI Crew (Hierarchical)**: 라우팅이 출력에 의존하고 네 명 이상의 전문가가 있을 때.
- **LangGraph**(Lesson 13): 명시적 상태 기계, 내구성 있는 재개, 엄격한 순서용.
- **AutoGen v0.4**(Lesson 14): 액터 모델 동시성과 결함 격리용.
- **OpenAI Agents SDK**(Lesson 16): 핸드오프와 가드레일(guardrail)을 갖춘 OpenAI 우선 제품용.
- **Claude Agent SDK**(Lesson 17): 서브에이전트와 세션 저장소를 갖춘 Claude 우선 제품용.

## 산출물 (Ship It)

`outputs/skill-crew-or-flow.md`는 작업에 대해 Crew 대 Flow를 고르고 최소 구현을 스캐폴딩한다. 백스토리 없는 Crew, 명시적 토픽 없는 Flow, 세 명 미만 전문가의 Hierarchical은 단호히 거부한다.

## 함정 (Pitfalls)

- **풍미로서의 백스토리.** 그것은 출력을 형성한다. 에이전트당 세 변형을 테스트하라. 변동성은 실재한다. 하나를 골라 고정하라.
- **`expected_output` 건너뛰기.** 작업당 계약이 없으면 하류 작업은 LLM이 만든 무엇이든 집어 든다. 크루는 실행되고, 감사는 실패한다.
- **항상 켜진 메모리.** Long-term이 매 실행마다 쓴다. 벡터 DB가 자란다. 검색이 시끄러워진다. 사실이 영속적인 작업으로 쓰기를 한정하라.
- **매니저 프롬프트 표류.** Hierarchical의 매니저 프롬프트는 암묵적이다. 라우팅이 이상해지면 verbose 모드로 덤프하고 읽어라.
- **Crews 안의 도구 부작용.** Crew는 도구를 예상보다 많이 호출할 수 있다. POST, DELETE, 결제는 Flow 단계에 속하지, 절대 Crew 도구에 속하지 않는다.

## 연습 문제 (Exercises)

1. Sequential 크루를 Flow로 변환하라. 변동성이 떨어지는 접점(touchpoint)을 세어라. 가독성이 떨어진 곳을 기록하라.
2. 크루에 entity 메모리를 추가하라: 고객에 대한 사실이 kickoff에 걸쳐 영속한다. 검색이 올바른 개체를 끌어오는지 검증하라.
3. 매니저가 작가의 출력이 최소 세 문단을 가질 때까지 편집자로 라우팅하기를 거부하는 Hierarchical 프로세스를 구현하라. 재시도를 추적하라.
4. (모의) 웹 검색을 위한 `BaseTool` 서브클래스를 배선하라. `@tool` 데코레이터 버전 대비 트레이스 형태를 비교하라.
5. 편집자 작업에 `output_pydantic=Brief`를 추가하라. 여기서 `Brief`는 `title`, `summary`, `sections`를 가진다. 작가 작업이 한 번 잘못된 JSON을 출력하게 하라; 트레이스에서 CrewAI의 재시도 동작을 검증하라.
6. CrewAI의 문서 소개를 읽어라. 토이를 진짜 `crewai` API로 포팅하라. stdlib 버전이 어느 보장을 건너뛰었는가?
7. AgentOps나 Langfuse(Lesson 24)를 진짜 실행에 배선하라. stdlib 버전에서 어느 트레이스를 놓쳤는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Agent | "페르소나" | 역할 + 목표 + 백스토리 + 도구 |
| Task | "작업 단위" | 설명 + 기대 출력 + 담당자 + 선택적 구조화 출력 |
| Crew | "에이전트 팀" | Agents + Tasks + Process를 담는 컨테이너 |
| Process | "실행 전략" | Sequential / Hierarchical / Consensus (계획 중) |
| Flow | "결정론적 워크플로" | 이벤트 기반, 코드 소유, 테스트 가능 |
| 백스토리(Backstory) | "페르소나 프롬프트" | Agent의 톤과 판단을 형성하는 것 |
| `@tool` | "함수 도구" | 함수를 Agent가 호출할 수 있는 도구로 바꾸는 데코레이터 |
| `BaseTool` | "클래스 도구" | 인자 스키마, 재시도, 비동기 지원을 가진 클래스 기반 도구 |
| Entity 메모리 | "개체별 사실" | 고객 / 계정 / 이슈로 한정된 메모리 |
| Long-term 메모리 | "실행 간 메모리" | kickoff 사이에 살아남는 벡터 기반 메모리 |
| Contextual 메모리 | "적시(just-in-time) 검색" | Agent가 필요로 하는 순간에 끌어오는 메모리 |
| 매니저 LLM(Manager LLM) | "라우터 에이전트" | Hierarchical 프로세스에서 다음 작업을 고르는 추가 LLM |
| `expected_output` | "작업 계약" | Agent(와 감사)에게 어떤 형태를 반환할지 알려주는 문자열 |

## 더 읽을거리 (Further Reading)

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): 개념과 권장 프로덕션 경로
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): 이벤트 기반 형태, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, 내장 툴킷
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): short-term, long-term, entity, contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 멀티 에이전트가 도움이 될 때와 그렇지 않을 때
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 상태 기계 대안
