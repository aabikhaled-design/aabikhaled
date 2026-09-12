# 오케스트레이션 패턴: 슈퍼바이저, 스웜, 계층형

> 2026년 프레임워크 전반에서 네 가지 오케스트레이션(orchestration) 패턴이 반복적으로 등장한다: 슈퍼바이저-워커(supervisor-worker), 스웜/피어 투 피어(swarm / peer-to-peer), 계층형(hierarchical), 토론(debate). Anthropic의 지침은 이렇다: "당신의 필요에 맞는 올바른 시스템을 만드는 것이다." 단순하게 시작하라. 단일 에이전트(agent)와 다섯 가지 워크플로 패턴으로 충분하지 않을 때만 토폴로지(topology)를 추가하라.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 반복적으로 등장하는 네 가지 오케스트레이션 패턴의 이름을 대고 각각이 어떤 상황에 맞는지 설명하기.
- 2026년 LangChain 권장 사항을 설명하기: 도구 호출(tool-call) 기반 슈퍼비전 vs 슈퍼바이저 라이브러리.
- Anthropic의 "올바른 시스템을 만들라" 규칙과 이것이 어떻게 토폴로지 선택을 통제하는지 설명하기.
- 공통의 스크립트화된 LLM을 대상으로 네 가지 패턴을 모두 stdlib로 구현하기.

## 문제 (The Problem)

팀은 필요하지도 않은데 "멀티 에이전트(multi-agent)"부터 손을 댄다. 프레임워크 전반에서 네 가지 패턴이 반복된다. 일단 이름을 댈 수 있게 되면, 올바른 것을 고르거나 토폴로지를 아예 건너뛸 수 있다.

## 개념 (The Concept)

### 슈퍼바이저-워커(Supervisor-worker)

- 중앙의 라우팅 LLM이 전문가 에이전트들에게 작업을 분배한다.
- 결정한다: 자기 자신으로 루프백, 전문가에게 핸드오프(handoff), 종료.
- 전문가들끼리는 서로 대화하지 않는다. 모든 라우팅은 슈퍼바이저를 거친다.

프레임워크: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026년 LangChain 권장 사항:** `create_supervisor`보다는 직접 도구 호출을 통해 슈퍼비전을 수행하라. 더 세밀한 컨텍스트 엔지니어링(context engineering) 제어를 제공한다. 각 전문가가 정확히 무엇을 보는지 직접 결정하기 때문이다.

### 스웜/피어 투 피어(Swarm / peer-to-peer)

- 에이전트들이 공유된 도구 표면(shared tool surface)을 통해 서로에게 직접 핸드오프한다.
- 중앙 라우터가 없다.
- 슈퍼바이저보다 지연 시간(latency)이 낮다(홉(hop)이 더 적다).
- 추론하기는 더 어렵다(단일 제어 지점이 없음).

프레임워크: LangGraph swarm 토폴로지, OpenAI Agents SDK handoffs(모든 에이전트가 다른 모든 에이전트에게 핸드오프할 수 있을 때).

### 계층형(Hierarchical)

- 워커를 관리하는 서브 슈퍼바이저를 관리하는 슈퍼바이저.
- LangGraph에서는 중첩된 서브그래프(nested subgraph)로 구현하고, CrewAI에서는 중첩된 크루(nested crew)로 구현한다.
- 운영 복잡성을 대가로 대규모 에이전트 집단까지 확장된다.

필요한 시점: 단일 슈퍼바이저의 컨텍스트 예산(context budget)이 모든 전문가의 설명을 담을 수 없을 때.

### 토론(Debate)

- 병렬 제안자(proposer) + 반복적 상호 비평(cross-critique)(Lesson 25).
- 사실 오케스트레이션이라기보다는 검증(verification)에 가깝지만, 프레임워크에서 토폴로지 선택지로 등장한다.

### 자율적인 크루와 결정적인 플로우

CrewAI는 두 가지 배포(deployment) 모드를 공식화한다:

- 결정론적(deterministic) 이벤트 기반 자동화를 위한 **Flow**(프로덕션(production)을 위한 권장 출발점).
- 자율적인 역할 기반 협업을 위한 **Crew**.

이것은 위의 네 가지 패턴과 직교(orthogonal)하지만 토폴로지에 매핑된다: Flow는 보통 슈퍼바이저 또는 계층형이고, Crew는 보통 LLM 라우터를 가진 슈퍼바이저다.

### Anthropic의 지침

"LLM 영역에서의 성공은 가장 정교한 시스템을 만드는 것에 관한 것이 아니다. 당신의 필요에 맞는 올바른 시스템을 만드는 것에 관한 것이다."

결정 순서:

1. 단일 에이전트 + 워크플로 패턴(Lesson 12): 여기서 시작하라.
2. 슈퍼바이저-워커: 전문가가 2~4명일 때.
3. 스웜: 추론의 명확성보다 지연 시간이 더 중요할 때.
4. 계층형: 슈퍼바이저 컨텍스트 예산이 실패할 때만.
5. 토론: 비용보다 정확도가 더 중요할 때.

### 이 패턴이 잘못되는 지점

- **토폴로지 우선 사고.** 멀티 에이전트가 어떤 문제를 푸는지 식별하기도 전에 "우리는 멀티 에이전트가 필요해"라고 말하는 것.
- **스웜에서 핸드오프가 튀는(bouncing) 현상.** A -> B -> A -> B. 홉 카운터(hop counter)를 사용하라.
- **가짜 계층 구조.** "엔터프라이즈"라서 세 개의 계층을 두지만 실제 팀은 둘뿐인 경우. 무너뜨려라(collapse).

```figure
orchestration-pattern
```

## 직접 만들기 (Build It)

`code/main.py`는 스크립트화된 LLM을 대상으로 네 가지 패턴을 모두 stdlib로 구현한다:

- `Supervisor`: 중앙 라우터.
- `Swarm`: 직접 핸드오프를 사용하는 피어 투 피어.
- `Hierarchical`: 슈퍼바이저의 슈퍼바이저.
- `Debate`: 병렬 제안자 + 비평.

각 패턴은 동일한 세 가지 의도(intent) 작업(환불 / 버그 / 영업)을 처리한다. 트레이스(trace)의 모양이 다르다.

실행:

```
python3 code/main.py
```

출력: 패턴별 트레이스 + 연산 횟수(op count). 슈퍼바이저가 가장 깔끔하고, 스웜이 가장 짧으며, 계층형이 가장 깊고, 토론이 가장 비싸다.

## 라이브러리로 써보기 (Use It)

- 슈퍼바이저와 계층형(중첩된 서브그래프)에는 **LangGraph**.
- 도구로서의 핸드오프(handoffs-as-tools)(슈퍼바이저 모양)에는 **OpenAI Agents SDK**.
- 프로덕션 결정론적 처리에는 **CrewAI Flow**.
- 토론이나 정확한 제어를 원할 때는 **Custom**.

## 산출물 (Ship It)

`outputs/skill-orchestration-picker.md`는 토폴로지를 고르고 그것을 구현한다.

## 연습 문제 (Exercises)

1. 라우터를 제거하여 슈퍼바이저-워커를 스웜으로 변환하라. 무엇이 깨지는가? 무엇이 개선되는가?
2. 스웜에 홉 카운터를 추가하라: 핸드오프 3회 이후에는 거부하라. A->B->A 튀는 현상을 잡아내는가?
3. 12명의 전문가 도메인을 위한 2단계 계층형 시스템을 만들어라. 중첩 없이는 어디서 컨텍스트 예산이 실패하는가?
4. 프로덕션 형태의 워크로드에서 네 가지 패턴을 프로파일링하라. 어느 지표(지연 시간, 비용, 정확도, 디버깅 용이성)에서 어느 것이 이기는가?
5. Anthropic의 "Building Effective Agents" 글을 읽어라. 자신의 프로덕션 흐름 각각을 네 가지 중 하나에 매핑하라. 깔끔하게 매핑되지 않는 것이 있는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 슈퍼바이저-워커(Supervisor-worker) | "라우터 + 전문가" | 중앙 LLM이 전문가들에게 작업을 분배하며, 전문가들끼리는 서로 대화하지 않는다 |
| 스웜(Swarm) | "피어 투 피어" | 공유 도구를 통한 직접 핸드오프; 중앙 라우터 없음 |
| 계층형(Hierarchical) | "슈퍼바이저의 슈퍼바이저" | 대규모 집단을 위한 중첩 서브그래프 |
| 토론(Debate) | "제안자 + 비평" | 병렬 제안자, 상호 비평(Lesson 25) |
| 도구 호출 기반 슈퍼비전(Tool-call-based supervision) | "라이브러리 없는 슈퍼바이저" | 컨텍스트 제어를 위해 슈퍼바이저를 직접 도구 호출로 구현 |
| 크루(Crew) | "자율 팀" | CrewAI의 역할 기반 협업 모드 |
| 플로(Flow) | "결정론적 워크플로" | CrewAI의 이벤트 기반 프로덕션 모드 |

## 더 읽을거리 (Further Reading)

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 다섯 가지 패턴 + 에이전트 vs 워크플로
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 슈퍼바이저, 스웜, 계층형
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325): 토론 패턴
