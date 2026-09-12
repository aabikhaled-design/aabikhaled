# OpenAI Agents SDK: 핸드오프, 가드레일, 트레이싱

> OpenAI Agents SDK는 Responses API 위에 구축된 경량 멀티 에이전트(multi-agent) 프레임워크다. 다섯 가지 원시 요소(primitive): Agent, Handoff, Guardrail, Session, Tracing. 핸드오프(handoff)는 `transfer_to_<agent>`라는 이름의 도구다. 가드레일(guardrail)은 입력 또는 출력에서 발동(trip)된다. 트레이싱(tracing)은 기본으로 켜져 있다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- OpenAI Agents SDK의 다섯 가지 원시 요소의 이름을 대기.
- 핸드오프를 설명하기: 왜 도구로 모델링되는지, 모델이 보는 이름 형태가 무엇인지, 컨텍스트가 어떻게 이전되는지.
- 입력 가드레일, 출력 가드레일, 도구 가드레일을 구분하기; `run_in_parallel` 대 블로킹(blocking) 모드를 설명하기.
- 핸드오프 + 가드레일 + 스팬(span) 스타일 트레이싱을 갖춘 stdlib 런타임(runtime)을 구현하기.

## 문제 (The Problem)

깔끔하게 위임(delegate)하지 못하는 에이전트(agent)는 결국 모든 것을 하나의 프롬프트에 욱여넣는다. 가드레일 없는 에이전트는 PII나 정책 위반 출력을 내보내거나 끝없이 루프를 돈다. OpenAI의 SDK는 멀티 에이전트 작업을 다룰 수 있게 만드는 세 가지 원시 요소를 성문화한다.

## 개념 (The Concept)

### 다섯 가지 원시 요소

1. **Agent.** LLM + 지시문(instructions) + 도구 + 핸드오프.
2. **Handoff.** 다른 에이전트로의 위임. 모델에게는 `transfer_to_<agent_name>`이라는 이름의 도구로 표현된다.
3. **Guardrail.** 입력(첫 에이전트만), 출력(마지막 에이전트만), 또는 도구 호출(함수 도구별)에 대한 검증.
4. **Session.** 턴(turn)에 걸친 자동 대화 기록.
5. **Tracing.** LLM 생성, 도구 호출, 핸드오프, 가드레일에 대한 내장 스팬.

### 도구로서의 핸드오프

모델은 도구 목록에서 `transfer_to_billing_agent`를 본다. 이를 호출하면 런타임에 다음을 알린다.

1. 대화 컨텍스트를 복사한다(또는 `nest_handoff_history` 베타로 압축한다).
2. 대상 에이전트를 그 지시문으로 초기화한다.
3. 대상 에이전트로 실행을 이어간다.

이것은 제품화된 슈퍼바이저 패턴(supervisor pattern, Lesson 13 / Lesson 28)이다.

### 가드레일

세 가지 종류:

- **입력 가드레일.** 첫 에이전트의 입력에서 실행된다. 어떤 LLM 호출 전에든 안전하지 않거나 범위를 벗어난 요청을 거부한다.
- **출력 가드레일.** 마지막 에이전트의 출력에서 실행된다. PII 누출, 정책 위반, 잘못된 형식의 응답을 잡는다.
- **도구 가드레일.** 함수 도구별로 실행된다. 인자를 검증하고, 권한을 확인하며, 실행을 감사한다.

모드:

- **병렬(Parallel)**(기본). 가드레일 LLM이 메인 LLM과 나란히 실행된다. 낮은 꼬리 지연 시간(tail latency). 발동되면 메인 LLM의 작업이 폐기된다(토큰 낭비).
- **블로킹(Blocking)**(`run_in_parallel=False`). 가드레일 LLM이 먼저 실행된다. 발동되면 메인 호출에 토큰이 낭비되지 않는다.

트립와이어(tripwire)는 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`를 일으킨다.

### 트레이싱

기본으로 켜져 있다. 모든 LLM 생성, 도구 호출, 핸드오프, 가드레일이 스팬을 방출한다. `OPENAI_AGENTS_DISABLE_TRACING=1`로 옵트아웃(opt out)한다. `add_trace_processor(processor)`는 OpenAI의 것과 나란히 직접 운영하는 백엔드로 스팬을 팬아웃(fan out)한다.

### 세션

`Session`은 대화 기록을 백엔드(SQLite, Redis, 커스텀)에 저장한다. `Runner.run(agent, input, session=session)`은 자동으로 로드하고 추가한다.

### 이 패턴이 잘못되는 지점

- **핸드오프 표류(drift).** Agent A가 Agent B로 핸드오프하고 B가 다시 A로 핸드오프한다. 홉(hop) 카운터를 추가하라.
- **가드레일 우회.** 도구 가드레일은 함수 도구에서만 발화한다. 내장 도구(파일 리더, 웹 페치)는 별도의 정책이 필요하다.
- **과잉 트레이싱.** 스팬 안의 민감한 콘텐츠. OTel GenAI 콘텐츠 캡처 규칙(Lesson 23)과 짝지어라. 외부에 저장하고 ID로 참조하라.

```figure
ae-agent-handoff
```

## 직접 만들기 (Build It)

`code/main.py`는 SDK 형태를 stdlib로 구현한다.

- `Agent`, `FunctionTool`, `Handoff`(전송 의미를 가진 함수 도구로).
- 입력/출력/도구 가드레일, 핸드오프 디스패치, 홉 카운터를 가진 `Runner`.
- 트레이스 형태를 보여주는 간단한 스팬 방출기(emitter).
- 사용자 질의에 따라 billing 또는 support로 핸드오프하는 트리아지(triage) 에이전트; 한 입력에서 가드레일이 발동된다.

실행:

```
python3 code/main.py
```

트레이스는 두 번의 성공적인 핸드오프, 한 번의 입력 가드레일 발동, 그리고 진짜 SDK가 방출하는 것을 본뜬 스팬 트리(span tree)를 보여준다.

## 라이브러리로 써보기 (Use It)

- **OpenAI Agents SDK**: OpenAI 우선 제품용.
- **Claude Agent SDK**(Lesson 17): Claude 우선 제품용.
- **LangGraph**(Lesson 13): 명시적 상태와 내구성 있는 재개를 원할 때.
- **커스텀**: 정확한 제어(음성, 멀티 프로바이더, 연합 배포)가 필요할 때.

## 산출물 (Ship It)

`outputs/skill-agents-sdk-scaffold.md`는 트리아지 에이전트, 핸드오프, 입력/출력/도구 가드레일, 세션 저장소, 트레이스 프로세서를 갖춘 Agents SDK 앱을 스캐폴딩한다.

## 연습 문제 (Exercises)

1. 핸드오프 홉 카운터를 추가하라: N번 전송 후 거부한다. 동작을 추적하라.
2. `nest_handoff_history`를 옵션으로 구현하라. 전송 전에 이전 메시지들을 하나의 요약으로 압축한다.
3. 블로킹 출력 가드레일을 작성하라. 그것을 발동시킬 프롬프트와 통과하는 프롬프트에서 지연 시간을 비교하라.
4. `add_trace_processor`를 JSON 로거에 배선하라. 스팬당 어떤 형태를 방출하는가?
5. SDK 문서를 읽어라. stdlib 토이를 `openai-agents-python`으로 포팅하라. 무엇을 잘못 모델링했는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Agent | "LLM + 지시문" | SDK의 Agent 타입; 도구와 핸드오프를 소유 |
| Handoff | "전송" | 모델이 다른 에이전트에 위임하기 위해 호출하는 도구 |
| Guardrail | "정책 검사" | 입력 / 출력 / 도구 호출에 대한 검증 |
| Tripwire | "가드레일 발동" | 가드레일이 거부할 때 일으키는 예외 |
| Session | "기록 저장소" | 실행 사이에 영속되는 대화 메모리 |
| Tracing | "스팬" | LLM + 도구 + 핸드오프 + 가드레일에 대한 내장 관측 가능성 |
| 블로킹 가드레일(Blocking guardrail) | "순차 검사" | 가드레일이 먼저 실행; 발동 시 토큰 낭비 없음 |
| 병렬 가드레일(Parallel guardrail) | "동시 검사" | 가드레일이 나란히 실행; 낮은 지연 시간, 발동 시 토큰 낭비 |

## 더 읽을거리 (Further Reading)

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/): 원시 요소, 핸드오프, 가드레일, 트레이싱
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview): Claude 풍의 대응물
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 애초에 언제 핸드오프에 손을 뻗을 것인가
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): Agents SDK 스팬이 매핑되는 표준
