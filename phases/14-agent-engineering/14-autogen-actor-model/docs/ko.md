# AutoGen v0.4: 액터 모델과 에이전트 프레임워크

> AutoGen v0.4(Microsoft Research, 2025년 1월)는 에이전트(agent) 오케스트레이션을 액터 모델(actor model)을 중심으로 재설계했다. 비동기 메시지 교환, 이벤트 기반(event-driven) 에이전트, 결함 격리(fault isolation), 자연스러운 동시성. 이 프레임워크는 이제 유지보수 모드(maintenance mode)에 있고, 그 자리는 Microsoft Agent Framework(2025년 10월 퍼블릭 프리뷰)가 후계자로 이어받고 있다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 액터 모델을 기술하기: 액터(actor)로서의 에이전트, 유일한 IPC로서의 메시지, 액터별 실패 격리.
- AutoGen v0.4의 세 API 계층(Core, AgentChat, Extensions)의 이름을 대고 각각이 무엇을 위한 것인지 설명하기.
- 메시지 전달을 처리로부터 분리(decoupling)하는 것이 왜 결함 격리와 자연스러운 동시성을 주는지 설명하기.
- Python으로 stdlib 액터 런타임(runtime)을 구현하고 두 에이전트 코드 리뷰 흐름을 그 위로 포팅하기.

## 문제 (The Problem)

대부분의 에이전트 프레임워크는 동기적이다. 한 에이전트가 생산하고, 한 에이전트가 소비하며, 콜 스택(call stack) 안에서 이루어진다. 실패는 스택을 무너뜨린다. 동시성은 나중에 덧붙인다. 분산(distribution)을 하려면 코드를 다시 짜야 한다.

AutoGen v0.4의 답: 액터 모델. 각 에이전트는 사설 받은편지함(private inbox)을 가진 액터다. 메시지가 유일한 상호작용이다. 런타임은 전달을 처리로부터 분리한다. 실패는 한 액터로 격리된다. 동시성은 기본 제공된다. 분산은 그저 다른 전송(transport)일 뿐이다.

## 개념 (The Concept)

### 액터 (Actors)

액터는 다음을 가진다.

- 사설 상태(외부에서 절대 직접 건드리지 않는다).
- 받은편지함(inbox, 메시지 큐).
- 핸들러: `receive(message) -> effects`로, 여기서 효과(effect)는 "답장", "다른 액터에게 전송", "새 액터 생성", "상태 업데이트", "자기 중지"가 될 수 있다.

두 액터는 메모리를 공유할 수 없다. 오직 메시지만 보낼 수 있다.

### 세 가지 API 계층

AutoGen v0.4는 자신의 표면을 셋으로 나눈다.

1. **Core.** 저수준 액터 프레임워크. `AgentRuntime`, `Agent`, `Message`, `Topic`. 비동기 메시지 교환, 이벤트 기반.
2. **AgentChat.** 작업 주도(task-driven) 고수준 API(v0.2의 ConversableAgent를 대체). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** 통합: OpenAI, Anthropic, Azure, 도구, 메모리.

### 분리가 왜 중요한가

v0.2 모델에서 `agent_a.chat(agent_b)`를 호출하면 agent_b가 반환할 때까지 agent_a를 동기적으로 블로킹(block)한다. v0.4에서 `send(agent_b, msg)`는 메시지를 agent_b의 받은편지함에 넣고 반환한다. 런타임이 나중에 전달한다. 그래서 세 가지 결과가 나온다.

- **결함 격리.** Agent B의 크래시(crash)가 Agent A를 무너뜨리지 않는다. 런타임이 B의 핸들러에서 실패를 잡고 무엇을 할지(로깅, 재시도, 데드레터(dead-letter)) 결정한다.
- **자연스러운 동시성.** 한 번에 여러 메시지가 진행 중이다. 액터들이 자신의 받은편지함을 동시에 처리한다.
- **분산 준비 완료.** 받은편지함 + 전송은 액터가 프로세스 내에 있든 다른 호스트에 있든 동일한 추상화다.

### 토폴로지 (Topologies)

- **RoundRobinGroupChat.** 에이전트들이 고정된 순환(rotation)으로 차례를 가진다.
- **SelectorGroupChat.** 셀렉터(selector) 에이전트가 대화 맥락을 바탕으로 다음에 누가 갈지 고른다.
- **Magentic-One.** 웹 브라우징, 코드 실행, 파일 처리를 위한 레퍼런스 멀티 에이전트 팀. AgentChat 위에 구축됨.

### 관측 가능성 (Observability)

OpenTelemetry 지원이 내장되어 있다. 모든 메시지가 스팬(span)을 방출하고, 도구 호출은 2026년 OTel GenAI 시맨틱 컨벤션(semantic convention, Lesson 23)에 따른 `gen_ai.*` 속성을 운반한다.

### 상태: 유지보수 모드

2026년 초: AutoGen v0.7.x는 연구와 프로토타이핑에 안정적이다. Microsoft는 활발한 개발을 Microsoft Agent Framework(2025년 10월 1일 퍼블릭 프리뷰; 2026년 1분기 말 1.0 GA 목표)로 옮겼다. AutoGen 패턴은 앞으로 깔끔하게 포팅된다. 액터 모델이 그만큼 오래가는 아이디어이기 때문이다.

```figure
actor-mailbox
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 액터 런타임을 구현한다.

- `Message`: `sender`, `recipient`, `topic`, `body`를 가진 타입 페이로드(payload).
- `Actor`: `receive(message, runtime)`을 가진 추상(abstract) 클래스.
- `Runtime`: 공유 큐, 전달, 실패 격리를 갖춘 이벤트 루프.
- 두 액터 데모: `ReviewerAgent`가 코드를 리뷰하고, `ChecklistAgent`가 체크리스트를 실행한다. 합의(consensus)에 이를 때까지 메시지를 교환한다.

실행:

```
python3 code/main.py
```

트레이스는 메시지 전달, 다른 액터를 무너뜨리지 않는 한 액터의 시뮬레이션된 실패, 공유 판정(verdict)으로의 수렴을 보여준다.

## 라이브러리로 써보기 (Use It)

- **AutoGen v0.4/v0.7**(유지보수): 연구, 프로토타이핑, 멀티 에이전트 패턴에 안정적.
- **Microsoft Agent Framework**(퍼블릭 프리뷰): 나아갈 길; 새로워진 API에 담긴 동일한 액터 모델 아이디어.
- **LangGraph 스웜 토폴로지**(Lesson 13): 공유 도구 핸드오프를 통한 유사 패턴.
- **커스텀 액터 런타임**: 특정 전송(NATS, RabbitMQ, gRPC)이 필요할 때.

## 산출물 (Ship It)

`outputs/skill-actor-runtime.md`는 최소 액터 런타임과 더불어 주어진 멀티 에이전트 작업을 위한 팀 템플릿(RoundRobin 또는 Selector)을 생성한다.

## 연습 문제 (Exercises)

1. 데드레터 큐(dead-letter queue)를 추가하라. 핸들러가 예외를 일으키면 실패한 메시지를 사람이 검사하도록 보관(park)하라. 이 토이에서 DLQ는 얼마나 자주 적중하는가?
2. `SelectorGroupChat`을 구현하라: 셀렉터 액터가 대화 상태를 바탕으로 다음 메시지를 누가 처리할지 고른다.
3. 분산 전송을 추가하라: 프로세스 내 큐를 JSON-over-HTTP 서버로 교체하여 액터들이 별도 프로세스에서 실행될 수 있게 하라.
4. 메시지마다 OTel 스팬(또는 no-op 대역)을 배선하라. Lesson 23에 따라 `gen_ai.agent.name`, `gen_ai.operation.name`을 방출하라.
5. AutoGen v0.4의 아키텍처 글을 읽어라. 토이를 진짜 `autogen_core` API로 포팅하라. 프로덕션에서 중요한 무엇을 건너뛰었는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 액터(Actor) | "에이전트" | 사설 상태 + 받은편지함 + 핸들러; 공유 메모리 없음 |
| 메시지(Message) | "이벤트" | 타입 페이로드; 액터가 상호작용하는 유일한 방식 |
| 받은편지함(Inbox) | "메일박스" | 대기 중인 메시지의 액터별 큐 |
| 런타임(Runtime) | "에이전트 호스트" | 메시지를 라우팅하고 실패를 격리하는 이벤트 루프 |
| 토픽(Topic) | "채널" | 액터 사이의 이름 붙은 발행-구독(publish-subscribe) 경로 |
| 결함 격리(Fault isolation) | "그냥 크래시하게 둬라(let it crash)" | 한 액터의 실패가 다른 액터를 무너뜨리지 않음 |
| RoundRobinGroupChat | "고정 순환 팀" | 에이전트들이 순서대로 차례를 가짐 |
| SelectorGroupChat | "맥락 라우팅 팀" | 셀렉터가 다음에 누가 갈지 고름 |
| Magentic-One | "레퍼런스 팀" | 웹 + 코드 + 파일을 위한 멀티 에이전트 분대 |

## 더 읽을거리 (Further Reading)

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/): 재설계 글
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 그래프 형태의 대안
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): AutoGen이 기본으로 방출하는 스팬
