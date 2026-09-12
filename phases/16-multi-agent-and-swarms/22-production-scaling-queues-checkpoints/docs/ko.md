# 프로덕션 스케일링(큐, 체크포인트, 내구성 (Production Scaling) Queues, Checkpoints, Durability)

> 다중 에이전트(multi-agent) 시스템을 수천 개의 동시 실행으로 확장하려면 **내구성 실행(durable execution)**이 필요하다. LangGraph의 런타임(runtime)은 각 슈퍼 스텝(super-step) 이후 `thread_id`를 키로 하는 체크포인트(checkpoint)를 기록한다(기본값은 Postgres). 워커(worker)가 충돌하면 리스(lease)가 해제되고 다른 워커가 재개한다. 에이전트(agent)는 인간 입력을 기다리며 무한정 잠들 수 있다. **MegaAgent** (arXiv:2408.09955)는 세 가지 상태(Idle / Processing / Response)와 2계층 협응(그룹 내 채팅 + 그룹 간 관리자 채팅)을 갖춘 에이전트별 생산자-소비자 큐(queue)를 실행했다. **파이버(fiber)/비동기**는 LLM 스트리밍에서 작업당 스레드(thread-per-job)를 능가한다. 스레드는 토큰(token)을 기다리며 99%의 시간 동안 유휴 상태이지만, 파이버는 I/O에서 협력적으로 양보한다. 반론: Ashpreet Bedi의 "Scaling Agentic Software"는 부하가 그렇지 않음을 증명하기 전까지는 **FastAPI + Postgres + 그 외 아무것도 없음**을 주장한다. 단순한 아키텍처가 생각보다 멀리 간다는 것이다. 이 레슨은 내구성 체크포인트 로그, 상태 전이를 갖춘 에이전트별 작업 큐, 비동기 대 스레드 데모를 만들고, 실용적인 "단순하게 시작하라" 규칙을 익힌다.

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75분

## 문제 (Problem)

프로토타입 다중 에이전트 시스템은 인메모리 이벤트 루프에서 세 에이전트로 노트북 한 대에서 작동한다. 이제 프로덕션(production)으로 옮긴다.

- 에이전트가 때때로 몇 시간 동안 실행된다(긴 연구, 인간 개입(human-in-the-loop) 대기).
- 워커 프로세스가 충돌한다. 재시작하면 상태를 잃는다.
- 피크 부하는 평균의 10배다. 수평 확장이 필요하다.
- 사용자는 에이전트 실행당 지불한다. 과금을 위해 정확히 한 번(exactly-once) 의미론이 필요하다.

인메모리 이벤트 루프는 이 중 어느 것도 하지 못한다. 그 아래에 내구성 실행 계층이 필요하다. 2026년의 정석 선택지는 다음과 같다.

1. 체크포인트를 갖춘 워크플로 엔진(Temporal, LangGraph 런타임).
2. 상태 저장소를 갖춘 메시지 큐(Postgres + SQS/RabbitMQ).
3. 액터 모델 프레임워크(MegaAgent의 에이전트별 생산자-소비자).
4. 직접 만든 FastAPI + Postgres(Bedi의 주장).

이 레슨은 각각의 축소판을 만든다.

## 개념 (Concept)

### 내구성 실행, 그 패턴

내구성 실행 엔진은 각 "스텝"(LangGraph 용어로는 슈퍼 스텝) 이후 전체 프로그램 상태를 영속화한다. 충돌 시:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

이것이 작동하기 위한 요구사항:

- **직렬화 가능한 상태.** 모든 에이전트 상태가 영속화 가능해야 한다. 살아 있는 데이터베이스 연결을 가진 함수 클로저는 살아남지 못한다.
- **결정론적 재개.** 같은 상태와 같은 입력이 주어지면, 에이전트는 같은 행동을 생성한다(또는 LLM 호출에 대해 외부 결정론적 오라클에 위임한다).
- **멱등(idempotent) 부작용.** 외부 호출(도구 호출, 지불)은 멱등이거나 중복 제거 키를 사용해야 한다.

LangGraph는 각 슈퍼 스텝 이후 체크포인트를 기록한다. Temporal은 각 액티비티 이후 기록한다. Restate는 이벤트 소싱된 저널을 사용한다. 세 가지 모두 같은 패턴을 구현한다.

### LangGraph의 런타임

각 에이전트는 `thread_id`를 가진다. 상태는 타입이 지정된 딕셔너리다. 각 슈퍼 스텝은 체크포인트 테이블에 한 행을 기록한다. 재개 시, 런타임은 처음부터가 아니라 마지막 체크포인트부터 재생한다. 에이전트는 인간 입력을 기다리며 `interrupt()`할 수 있다. 런타임은 영속화하고 워커를 해제한다. 입력이 도착하면, 어떤 워커든 재개할 수 있다.

이것이 2026년 4월의 참조 프로덕션 설계다.

### MegaAgent의 에이전트별 큐

arXiv:2408.09955은 확장 실험을 기술한다: 한 클러스터 안의 수천 개 동시 에이전트. 아키텍처:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

2계층 협응은 그룹 내 대화는 밀집하게 일어나게 하면서 그룹 간은 희소하게 유지한다. 수천 개 에이전트에서 비용을 선형으로 유지하는 데 쓰이는 패턴이다.

### 비동기 대 작업당 스레드

LLM 호출은 I/O 제약적이다. 다음 토큰을 기다리는 스레드는 99%의 시간 동안 유휴 상태다. 스레드는 각각 약 1MB RAM이 든다. 10,000개의 동시 호출이면 스택만으로 10GB다.

파이버(Python `asyncio`, Go 고루틴, Rust `tokio`)는 I/O에서 협력적으로 양보한다. 같은 10,000개 호출이 한 프로세스 안에 넉넉히 들어간다. LLM 에이전트 규모에서 비동기는 최적화가 아니라 아키텍처 그 자체다.

예외: CPU 제약적 후처리(임베딩(embedding), 토크나이저(tokenizer) 기법)는 여전히 스레드나 프로세스를 원한다. I/O 계층을 CPU 계층에서 분리하라.

### Bedi의 반론

"Scaling Agentic Software"(Ashpreet Bedi, 2026)는 대부분의 팀이 부하를 측정하기 전에 과잉 설계한다고 주장한다. 실용적 기본값:

- FastAPI + Postgres.
- 각 에이전트 실행은 한 행이다. 상태는 낙관적 동시성으로 제자리에서 갱신된다.
- `pg_notify` 또는 간단한 Celery 워커를 통한 백그라운드 작업.
- 애플리케이션 코드의 재시도 정책.

감당 가능한 과제에서 약 100개 미만의 동시 에이전트 실행 부하라면, 대개 이것으로 충분하다. 이 방식이 한계에 부딪히는 지점을 측정한 뒤 업그레이드하라.

규칙: 단순한 아키텍처가 풀 수 없는 구체적 문제에 부딪혔을 때 내구성 실행 프레임워크를 채택하라. 너무 일찍 채택하면 성과 없는 형식에 시간만 태운다.

### 정확히 한 번 의미론

유료 에이전트 실행의 경우, "정확히 한 번 효과적"(적어도 한 번 전달 + 멱등 소비자)이 필요하다. 엔지니어링 수:

- **실행당 중복 제거 키.** 모든 부작용 호출에 포함하라.
- **아웃박스(outbox) 패턴.** 부작용이 먼저 테이블에 기록되고, 그다음 별도 프로세스가 실행한다. 두 단계 모두 멱등.
- **보상 트랜잭션.** 부작용은 성공했지만 그 추적 기록이 실패하면, 보상을 예약하라.

이것들은 LLM에 특화된 것이 아니라 데이터베이스 엔지니어링 패턴이다. LLM이 추가로 무는 비용은 호출이 느리다는 것뿐이고, 나머지는 모두 표준 분산 시스템이다.

### 레인보우 배포

Anthropic의 다중 에이전트 연구 시스템은 "레인보우 배포(rainbow deployments)"를 사용한다: 에이전트 런타임의 여러 버전이 동시에 실행되어, 장기 실행 에이전트가 코드 배포(deployment)마다 종료될 필요가 없게 한다. 트래픽의 일부에 새 버전을 카나리(canary)하고, 에이전트가 끝나면 옛 버전을 폐기한다.

이것은 장기 실행 상태 보존 시스템의 표준이다. 2026년의 적응은 에이전트가 몇 시간 동안 살 수 있으므로 배포 주기가 이를 수용해야 한다는 것이다.

### 정석 프로덕션 체크리스트

- 내구성 상태(체크포인트, 스냅샷, 또는 아웃박스 + 재생 가능 로그).
- 멱등 부작용.
- LLM 호출을 위한 비동기 I/O 계층.
- 중복 제거를 갖춘 적어도 한 번 전달.
- 상태 보존 워크로드를 위한 레인보우/카나리 배포.
- 관측 가능성: 에이전트별 추적, 슈퍼 스텝 감사, 재시도 카운터.

```figure
sw-checkpoint-replay
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다.

- `CheckpointStore`: thread-id 키를 갖춘 SQLite 기반 체크포인트 로그. 각 슈퍼 스텝이 한 행을 추가한다.
- `run_with_checkpoint(agent, thread_id)`: 실행 도중 충돌을 시뮬레이션한다. 두 번째 워커가 마지막 체크포인트부터 재개한다.
- `AgentQueue`: 작은 작업 큐를 갖춘 에이전트별 Idle / Processing / Response 상태 머신.
- `demo_async_vs_threads()`: asyncio를 통해, 그리고 스레드를 통해 500개의 동시 시뮬레이션 "LLM 호출"을 실행한다. 벽시계 시간과 피크 메모리(근사)를 보고한다.

실행:

```
python3 code/main.py
```

예상 출력: 시뮬레이션된 충돌 이후 체크포인트 재개 성공. 비동기 버전은 500개 동시 호출을 < 1s에 처리. 스레드 버전은 수 초가 걸리고 동시 단위당 메모리를 수십 배 더 사용.

## 라이브러리로 써보기 (Use It)

`outputs/skill-scaling-advisor.md`는 내구성 실행 선택에 대해 조언한다: FastAPI + Postgres, LangGraph 런타임, Temporal, 또는 커스텀. 부하, 상태 보존 필요, 배포 빈도로 보정된다.

## 산출물 (Ship It)

정석 프로덕션 강화:

- **단순하게 시작하라(Bedi의 규칙).** 이 방식이 한계에 부딪히는 지점을 측정하기 전까지는 FastAPI + Postgres.
- **최적화 전에 모든 것을 계측하라.** 실행당 지연 시간(latency) 히스토그램, 스텝당 시간, 재시도 횟수, 실패 분류.
- **부작용을 위한 아웃박스 패턴.** 특히 지불과 외부 API 호출.
- **레인보우 배포.** 배포 중에 진행 중인 에이전트 실행을 절대 종료하지 말라.
- **다음일 때 내구성 실행 엔진(Temporal / LangGraph / Restate)을 채택하라**: 한 시간짜리 인간 개입 대기, 교차 지역 협응, 복잡한 재시도/보상 정책 같은 특정 문제에 부딪힐 때.
- **I/O 계층을 위한 비동기.** CPU 제약적 후처리에만 스레드.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 체크포인트 재개가 작동하는지 확인하라. 비동기 대 스레드 동시성 차이를 측정하라.
2. **아웃박스** 테이블을 구현하라: 모든 도구 호출이 먼저 아웃박스에 기록되고, 그다음 별도 고루틴/태스크가 실행한다. 도구 호출을 두 번 실행하여 멱등성을 검증하라.
3. **레인보우 배포**를 시뮬레이션하라: 두 개의 동시 런타임 버전. 새 thread_id의 절반을 각각으로 라우팅하라. 옛 버전의 진행 중 스레드가 중단되지 않음을 확인하라.
4. LangGraph의 런타임 문서(아래 링크)를 읽어라. 런타임의 어떤 기능이 직접 만든 FastAPI + Postgres 버전에서 복제하는 데 가장 오래 걸릴지 식별하라. 그 기능이 채택할 이유가 되는가, 아니면 미뤄도 되는가?
5. MegaAgent (arXiv:2408.09955) 3절을 읽어라. 2계층 협응(그룹 내 + 그룹 간 관리자 채팅)이 명시적이다. 이것을 두 개의 큐 패밀리를 갖춘 메시지 큐에 어떻게 매핑할지 스케치하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 내구성 실행(Durable execution) | "프로그램 상태를 영속화" | 엔진이 각 슈퍼 스텝 이후 상태를 기록. 충돌 복구가 결정론적. |
| 슈퍼 스텝(Super-step) | "트랜잭션 경계" | 체크포인트 사이의 작업 단위. LangGraph 용어. |
| thread_id | "에이전트 실행 식별자" | 체크포인트와 재개 로직을 묶는 키. |
| 멱등성(Idempotency) | "재시도해도 안전" | 부작용을 반복해도 한 번 시도와 같은 결과를 낸다. |
| 아웃박스 패턴(Outbox pattern) | "부작용 분리" | 의도를 테이블에 기록. 별도 실행기가 수행하고 완료 표시. |
| 적어도 한 번 전달(At-least-once delivery) | "중복 가능성" | 메시지 큐 의미론. 중복 제거 키가 소비자를 효과적으로 한 번으로 만든다. |
| 레인보우 배포(Rainbow deploy) | "겹치는 버전" | 장기 실행 워크로드 동안 여러 런타임 버전 동시 실행. |
| 비동기 파이버(Async fiber) | "협력적 양보" | 사용자 모드 동시성. I/O 제약 부하에 대해 스레드보다 저렴. |
| 체크포인트(Checkpoint) | "상태 스냅샷" | 슈퍼 스텝 경계의 직렬화된 상태. 재개의 키. |

## 더 읽을거리 (Further Reading)

- [LangChain: The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) LangGraph 런타임 설계
- [MegaAgent](https://arxiv.org/abs/2408.09955): 에이전트별 생산자-소비자 큐. 수천 개 동시 에이전트에서의 2계층 협응
- [Matrix](https://arxiv.org/abs/2511.21686): 메시지 큐를 협응 기반으로 하는 분산 프레임워크
- [Temporal docs](https://docs.temporal.io/): 내구성 실행을 위한 참조 워크플로 엔진
- [Anthropic: Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) 레인보우 배포를 포함한 프로덕션 교훈
