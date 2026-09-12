# 장시간 실행 백그라운드 에이전트: 지속 실행(Durable Execution)

> 프로덕션(production) 장기 지평(long-horizon) 에이전트(agent)는 `while True`로 돌지 않는다. 모든 LLM 호출은 체크포인트(checkpoint), 재시도(retry), 재생(replay)을 갖춘 액티비티(activity)가 된다. Temporal의 OpenAI Agents SDK 통합은 2026년 3월 정식 출시(GA)되었다. Claude Code Routines(Anthropic)는 지속적인 로컬 프로세스 없이 예약된 Claude Code 호출을 실행한다. 세션은 사람 입력(human-input)에서 일시 정지하고, 배포(deploy)를 견디며, `thread_id`로 키가 지정된 최신 체크포인트에서 재개한다. 새로운 사용성 뒤에는 오래된 패턴(워크플로 오케스트레이션(workflow orchestration))이 자리하며, 한 가지 새로운 입력이 있다: 복구 시 결정론적으로 재생되어야 하는 비결정론적 액티비티로서의 LLM 호출.

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60분

## 문제 (The Problem)

네 시간 동안 도는 에이전트를 생각해 보라. 세 개의 도구를 호출하고, 사용자에게 두 번 프롬프트(prompt)하며, 마흔 번의 LLM 호출을 한다. 절반쯤 진행됐을 때, 그것이 도는 호스트가 재부팅된다. 무슨 일이 일어나는가?

- 순진한 `while True` 루프에서는: 모든 것이 사라진다. 실행이 처음부터 재시작한다. (실제 부작용이 있는) 세 도구 호출이 다시 실행된다. 사용자는 이미 승인한 것들에 대해 다시 프롬프트를 받는다. 마흔 번의 LLM 호출이 다시 청구된다.
- 지속 실행(durable execution)에서는: 실행이 가장 최근 체크포인트에서 재개한다. 이미 완료된 액티비티는 다시 실행되지 않는다. 그 결과는 지속 로그(durable log)에서 재생된다. 사용자는 이미 승인한 것들을 다시 승인하지 않는다. 이미 이뤄진 LLM 호출은 다시 청구되지 않는다.

이것은 워크플로 엔진들이 십 년간 출하해 온 같은 패턴이다(Temporal, Cadence, Uber의 Cherami). 새로운 점은 이제 LLM 호출이 일종의 액티비티(비결정론적이고, 비싸며, 부작용이 있는) 이며, 이 패턴에 깔끔하게 들어맞는다는 것이다.

레슨의 관통하는 주제: 장기 지평 신뢰성은 쇠퇴한다(METR은 "35분 열화"를 관찰한다. 성공률이 지평에 따라 대략 이차적으로 떨어진다). 지속 실행은 신뢰성 프로파일이 지지하는 것보다 더 긴 실행을 가능하게 하며, 이는 설계가 옳으면 안전하게, 설계가 틀리면 위험하게 실패하는 새로운 방식이다.

## 개념 (The Concept)

### 액티비티, 워크플로, 그리고 재생

- **워크플로(Workflow)**: 결정론적 오케스트레이션 코드. 액티비티의 순서, 분기, 대기를 정의한다. 예상치 못한 발산 없이 이벤트 로그에서 재생될 수 있도록 결정론적이어야 한다.
- **액티비티(Activity)**: 비결정론적이고 잠재적으로 실패할 수 있는 작업 단위. LLM 호출, 도구 호출, 파일 쓰기, HTTP 요청. 각 액티비티는 입력과 (완료되면) 출력이 함께 로깅된다.
- **이벤트 로그(Event log)**: 지속 백킹 스토어(durable backing store). 모든 액티비티의 시작, 완료, 실패, 재시도, 그리고 모든 워크플로 결정이 기록된다.
- **재생(Replay)**: 복구 시 워크플로 코드가 처음부터 재실행된다. 이미 완료된 모든 액티비티는 재실행 없이 로깅된 결과를 반환한다. 완료되지 않았던 액티비티만 실제로 실행된다.

이것은 React가 가상 DOM에 대해 재렌더링하는 것, 혹은 Git이 커밋들로부터 작업 트리를 재구축하는 것과 같은 형태다. 오케스트레이터의 결정론이 지속성을 싸게 만든다.

### LLM 호출이 이 패턴에 맞는 이유

LLM 호출은:
- 비결정론적이다(temperature > 0; temperature 0이라도 모델 버전 간에 표류한다).
- 비싸다(돈과 지연 시간(latency)).
- 잠재적으로 실패한다(속도 제한, 타임아웃).
- 부작용이 있다(도구를 호출하는 경우).

이것이 정확히 액티비티 프로파일이다. 모든 LLM 호출을 액티비티로 감싸면 지수 백오프(exponential backoff)를 갖춘 재시도, 재시작 전반에 걸친 체크포인팅, 디버깅을 위한 재생 가능한 트레이스를 얻는다.

### `thread_id`로 키가 지정된 체크포인트

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects, 그리고 Claude Code Routines는 모두 같은 API 형태로 수렴했다. `thread_id`(또는 동등물)가 세션을 식별하고; 각 상태 전이가 백엔드(기본 PostgreSQL, 개발용 SQLite, 캐시용 Redis)에 지속되며; 재개는 최신 체크포인트를 읽는다.

백엔드 선택은 중요하다:

- **PostgreSQL**: 지속적이고, 조회 가능하며, 배포를 견딘다. LangGraph의 기본값.
- **SQLite**: 로컬 개발 전용; 호스트 간에 데이터를 잃는다.
- **Redis**: 빠르지만 AOF/스냅샷이 구성되지 않으면 일시적이다.
- **Cloudflare Durable Objects**: 투명하게 분산됨; 고유 키로 범위가 지정됨; 수 시간에서 수 주간 생존한다.

### 일급(first-class) 상태로서의 사람 입력

제안-후-커밋(propose-then-commit, Lesson 15)은 지속적인 "사람 대기" 상태를 요구한다. 워크플로가 일시 정지하고, 외부 큐가 대기 중인 요청을 보유하며, 승인이 정확히 그 지점에서 재개한다. 지속성이 없으면 이것은 최선 노력(best-effort)이다. 지속성이 있으면 밤새 도착한 승인에 워크플로가 아침에 이어받는다.

### 35분 열화

METR은 측정된 모든 에이전트 부류가 약 35분의 연속 작동을 넘어서면 신뢰성 쇠퇴를 보인다는 것을 관찰했다. 과제 지속 시간을 두 배로 하면 실패율이 대략 네 배가 된다. 지속 실행은 이를 고치지 않는다. 신뢰성 프로파일이 지지하는 것보다 더 길게 돌게 할 뿐이다. 안전한 패턴은 지속성을, 재진입 시 새로운 HITL(human-in-the-loop)을 요구하는 체크포인트, 그리고 벽시계 시간과 무관하게 총 연산을 상한하는 예산 킬 스위치(kill switch, Lesson 13)와 결합한다.

### 지속 실행이 틀린 답일 때

- 사람 입력 없이 몇 분보다 짧은 실행. 오버헤드 > 이득.
- 엄격히 읽기 전용인 정보 검색.
- 정확성이 단일 컨텍스트 윈도우(context window) 안에서 종단 간(end-to-end)으로 이뤄져야 하는 과제(일부 추론 과제; 일부 원샷 생성).

```figure
memory-consolidation
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 stdlib Python으로 최소한의 지속 실행 엔진을 구현한다. 다음을 지원한다:

- 입력과 출력을 JSON 이벤트 로그에 로깅하는 `@activity` 데코레이터.
- 액티비티들을 순서 짓는 워크플로 함수.
- 완료된 액티비티를 재실행 없이 재생하는 `run_or_replay(workflow, event_log)` 함수.

드라이버는 세 액티비티 워크플로를 시뮬레이션하고, 절반쯤에서 크래시(crash)시키고, (a) 모든 것을 재실행하는 순진한 재시도 대 (b) 누락된 액티비티만 실행하는 재생을 보여준다.

## 산출물 (Ship It)

`outputs/skill-durable-execution-review.md`는 제안된 장시간 실행 에이전트 배포가 올바른 지속 실행 형태를 갖췄는지 검토한다: 액티비티, 결정론, 체크포인트 백엔드, 사람 입력 상태, 재개 시 HITL 정책.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 순진한 재시도와 재생 간 액티비티 실행 횟수의 차이를 관찰하라. 크래시 지점을 바꾸고 재생 횟수가 그에 따라 변하는 것을 보여라.

2. 장난감 엔진을 `thread_id`를 명시적으로 쓰도록 변환하라. 엔진을 공유하는 두 개의 동시 세션을 시뮬레이션하고 그들의 이벤트 로그가 충돌하지 않음을 확인하라.

3. 장난감 엔진의 액티비티 하나를 골라라. 비결정론(워크플로 결정 안의 벽시계 타임스탬프)을 도입하라. 재생 시 발산을 시연하라. 실제 엔진들이 이를 어떻게 다루는지 설명하라(부작용 등록, `Workflow.now()` API).

4. LangChain의 "Runtime behind production deep agents" 글을 읽어라. 런타임이 지속하는 모든 상태를 나열하고, 각각이 어떤 실패 양상을 덮는지 명명하라.

5. 6시간짜리 자율 코딩 과제를 위한 체크포인트 정책을 설계하라. 어디에서 체크포인트하는가? 크래시 시 재개는 어떤 모습인가? 무엇이 새로운 HITL을 요구하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 워크플로 (Workflow) | "에이전트의 스크립트" | 결정론적 오케스트레이션 코드; 이벤트 로그에서 재생 가능 |
| 액티비티 (Activity) | "하나의 스텝" | 비결정론적 단위(LLM 호출, 도구 호출); 전후로 로깅됨 |
| 이벤트 로그 (Event log) | "백킹 스토어" | 모든 상태 전이의 지속 기록 |
| 재생 (Replay) | "재개" | 워크플로 재실행; 완료된 액티비티는 재실행 없이 로깅된 결과 반환 |
| 체크포인트 (Checkpoint) | "저장 지점" | thread_id로 키가 지정된 지속 상태; 재개 시 최신값 우선 |
| thread_id | "세션 키" | 지속 상태의 범위를 정하는 식별자 |
| 35분 열화 (35-minute degradation) | "신뢰성 쇠퇴" | METR: 성공률이 지평에 따라 대략 이차적으로 떨어짐 |
| 비결정론 (Non-determinism) | "재생 시 표류" | 벽시계, 난수, LLM 출력; 부작용으로 등록되어야 함 |

## 더 읽을거리 (Further Reading)

- [Anthropic: Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) 예산, 턴, 재개 시맨틱.
- [Microsoft: Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) RequestInfoEvent 형태.
- [LangChain: The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) 구체적 런타임 요구사항.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev): LLM 호출을 위한 액티비티 형태.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 35분 열화 참조.
