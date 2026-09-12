# 프로덕션 런타임: 큐, 이벤트, 크론

> 프로덕션(production) 에이전트(agent)는 여섯 가지 런타임(runtime) 형태로 실행된다: 요청-응답(request-response), 스트리밍(streaming), 지속적 실행(durable execution), 큐 기반 백그라운드(queue-based background), 이벤트 기반(event-driven), 스케줄(scheduled). 프레임워크를 고르기 전에 형태를 골라라. 관찰 가능성(observability)은 모든 형태에서 하중을 지탱하는(load-bearing) 요소다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 여섯 가지 프로덕션 런타임 형태의 이름을 대고 각각을 프레임워크/제품 패턴에 매칭하기.
- 장기 지평(long-horizon) 작업에서 지속적 실행(LangGraph)이 왜 중요한지 설명하기.
- 이벤트 기반 런타임을 설명하고 Claude Managed Agents가 언제 적합한지 설명하기.
- 다단계(multi-step) 에이전트에 대한 "관찰 가능성은 하중을 지탱한다"는 주장을 설명하기.

## 문제 (The Problem)

프로덕션 에이전트는 Jupyter 노트북에서는 드러나지 않는 방식으로 실패한다: 37번째 스텝에서의 네트워크 타임아웃, 음성 통화 중간에 사용자가 끊어버림, 머신 재부팅으로 죽는 크론 작업, 메모리가 부족해지는 백그라운드 워커. 어떤 실패를 견뎌낼 수 있는지는 런타임 형태가 결정한다.

## 개념 (The Concept)

### 요청-응답(Request-response)

- 동기식 HTTP. 사용자가 완료될 때까지 기다린다.
- 짧은 작업(<30초)에만 실행 가능하다.
- 스택: Agno(Python + FastAPI), Mastra(TypeScript + Express/Hono/Fastify/Koa).
- 관찰 가능성: 표준 HTTP 액세스 로그 + OTel 스팬(span).

### 스트리밍(Streaming)

- 점진적 출력을 위한 SSE 또는 WebSocket.
- LiveKit은 이것을 음성/영상을 위한 WebRTC로 확장한다(Lesson 22).
- 스택: 스트리밍을 지원하는 모든 프레임워크 + SSE/WS를 처리하는 프런트엔드.
- 관찰 가능성: 청크(chunk)별 타이밍, 첫 토큰 지연 시간(first-token latency), 테일 지연 시간(tail latency).

### 지속적 실행(Durable execution)

- 모든 스텝 이후 상태가 체크포인트(checkpoint)된다. 실패 시 자동으로 재개된다.
- AutoGen v0.4 액터 모델(actor model)은 실패를 하나의 에이전트로 격리한다(Lesson 14).
- LangGraph의 핵심 차별화 요소(Lesson 13).
- 스텝 수가 알려지지 않았고 복구 비용이 높을 때 필수적이다.

### 큐 기반/백그라운드(Queue-based / background)

- 작업이 큐(queue)에 들어가고, 워커가 집어가며, 결과는 웹훅(webhook)이나 pub/sub를 통해 돌아온다.
- 장기 지평 에이전트(작업당 수십~수백 스텝, Anthropic의 컴퓨터 사용(computer use) 발표 기준)에 필수적이다.
- 스택: Celery(Python), BullMQ(Node), SQS + Lambda(AWS), 커스텀.
- 관찰 가능성: 큐 깊이(queue depth), 작업별 지연 시간 분포, DLQ 크기.

### 이벤트 기반(Event-driven)

- 에이전트가 트리거(trigger)를 구독한다: 새 이메일, PR 열림, 크론 발동.
- Claude Managed Agents가 이것을 기본으로 제공한다(Lesson 17).
- CrewAI Flows(Lesson 15)는 이벤트 기반 결정론적(deterministic) 워크플로를 구조화한다.
- 관찰 가능성: 트리거 소스, 이벤트-시작 지연 시간, 에이전트 지연 시간.

### 스케줄(Scheduled)

- 주기적으로 실행되는 크론(cron) 형태의 에이전트.
- 지속적 실행과 결합하여, 실패한 야간 실행이 다음 틱(tick)에 재개되도록 하라.
- 스택: Kubernetes CronJob + 지속적 프레임워크; 호스팅형(Render cron, Vercel cron).

### 2026년 배포 패턴

- 이벤트 기반 프로덕션에는 **CrewAI Flows**.
- Python 마이크로서비스에는 **Agno** 무상태(stateless) FastAPI.
- 임베딩(embedding)에는 **Mastra** 서버 어댑터(Express, Hono, Fastify, Koa).
- 관리형 음성에는 **Pipecat Cloud / LiveKit Cloud**(Lesson 22).
- 호스팅형 장기 실행 비동기에는 **Claude Managed Agents**.

### 관찰 가능성은 하중을 지탱한다

OpenTelemetry GenAI 스팬(Lesson 23)과 Langfuse/Phoenix/Opik 백엔드(Lesson 24)가 없으면 40번째 스텝에서 실패한 다단계 에이전트를 디버깅할 수 없다. 이것은 프로덕션에서 선택 사항이 아니다. "우리는 빠르게 디버깅한다"와 "우리는 더 많은 로깅을 붙여서 처음부터 다시 재현한다" 사이의 차이다.

### 프로덕션 런타임이 실패하는 지점

- **잘못된 형태 선택.** 5분짜리 작업에 요청-응답을 고르는 것. 사용자가 끊고, 워커가 쌓이며, 재시도가 누적된다.
- **DLQ 없음.** 데드 레터(dead-letter)가 없는 큐 워커. 실패한 작업이 사라진다.
- **불투명한 백그라운드 작업.** 트레이스(trace) 내보내기 없이 실행되는 백그라운드 에이전트. 사용자가 보고할 때까지 실패가 보이지 않는다.
- **지속적 상태 생략.** 재시작을 감당할 수 없는 30초 이상의 실행은 지속적 실행이 필요하다.

```figure
wb-runtime-shapes
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib로 만든 다중 형태(multi-shape) 데모다:

- 요청-응답 엔드포인트(일반 함수).
- 스트리밍 핸들러(제너레이터).
- DLQ가 있는 큐 기반 워커.
- 이벤트 트리거 레지스트리.
- 크론 형태의 스케줄러.

실행:

```bash
python3 code/main.py
```

출력: 동일한 작업에 대해 각 형태의 동작을 보여주는 다섯 개의 트레이스. 동일한 에이전트 로직, 다른 외부 껍데기. 지속적 실행(여섯 번째 형태)은 의도적으로 Lesson 13에서 LangGraph 체크포인팅으로 다룬다.

## 라이브러리로 써보기 (Use It)

- 챗 스타일 UX에는 **요청-응답**.
- 점진적 응답에는 **스트리밍**.
- 장기 지평 작업에는 **지속적**.
- 배치(batch)/비동기/장기 실행에는 **큐**.
- 에이전트 반응성에는 **이벤트**.
- 하우스키핑(메모리 통합, 평가(eval), 비용 리포트)에는 **크론**.

## 산출물 (Ship It)

`outputs/skill-runtime-shape.md`는 작업에 대한 런타임 형태를 고르고 관찰 가능성 요구 사항을 연결한다.

## 연습 문제 (Exercises)

1. Lesson 01의 ReAct 루프를 당신 스택의 여섯 가지 형태 모두로 포팅하라. 어느 형태가 어느 제품 표면에 맞는가?
2. 큐 기반 데모에 DLQ를 추가하라. 10% 작업 실패를 시뮬레이션하고 DLQ 크기를 드러내라.
3. 하루치 상위 20개 트레이스를 대상으로 야간에 실행되는 크론 트리거 평가 에이전트를 작성하라.
4. 백프레셔(backpressure)가 있는 스트리밍을 구현하라: 클라이언트가 느리면 에이전트를 일시 정지하라. 이것이 턴 예산(turn budget)과 어떻게 상호작용하는가?
5. Claude Managed Agents 문서를 읽어라. 자체 호스팅 장기 지평 에이전트를 언제 관리형으로 옮기겠는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 요청-응답(Request-response) | "동기식" | 사용자가 기다림; 짧은 작업에만 |
| 스트리밍(Streaming) | "SSE / WS" | 점진적 출력; 더 나은 UX; 청크별로 지연 시간 관찰 가능 |
| 지속적 실행(Durable execution) | "실패에서 재개" | 체크포인트된 상태; 마지막 스텝에서 재시작 |
| 큐 기반(Queue-based) | "백그라운드 작업" | 생산자 / 워커 풀 / DLQ |
| 이벤트 기반(Event-driven) | "트리거 기반" | 에이전트가 외부 이벤트에 반응 |
| DLQ | "데드 레터 큐" | 실패한 작업을 위한 주차장 |
| Claude Managed Agents | "호스팅형 하니스" | 캐싱 + 압축(compaction)을 갖춘 Anthropic 호스팅 장기 실행 비동기 |

## 더 읽을거리 (Further Reading)

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 지속적 실행 세부 사항
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview): 호스팅형 장기 실행 비동기
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use): "작업당 수십~수백 스텝"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/): 액터 모델 결함 격리
