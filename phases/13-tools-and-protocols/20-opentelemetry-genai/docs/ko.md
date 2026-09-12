# OpenTelemetry GenAI: 툴 호출을 종단 간(End-to-End)으로 추적하기

> 에이전트가 다섯 개의 툴, 세 개의 MCP 서버, 두 개의 하위 에이전트를 호출한다. 그 모든 것을 가로지르는 하나의 트레이스(trace)가 필요하다. OpenTelemetry GenAI 시맨틱 컨벤션(semantic conventions)(v1.37 이상에서 안정화된 속성)은 2026년 표준이며, Datadog, Langfuse, Arize Phoenix, OpenLLMetry, AgentOps가 네이티브로 지원한다. 이 레슨은 필수 속성을 명명하고, 스팬(span) 계층(에이전트 → LLM → 툴)을 살펴보며, 어떤 OTel 익스포터(exporter)에든 연결할 수 있는 stdlib 스팬 방출기를 출하한다.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- LLM 스팬과 툴 실행 스팬에 대한 필수 OTel GenAI 속성을 명명하기.
- 에이전트 루프, LLM 호출, 툴 호출, MCP 클라이언트 디스패치를 포괄하는 트레이스 계층 구축하기.
- 무슨 콘텐츠를 캡처할지(옵트인) vs 마스킹할지(기본값) 결정하기.
- 툴 코드를 다시 쓰지 않고 스팬을 로컬 컬렉터(Jaeger, Langfuse)로 방출하기.

## 문제 (The Problem)

2026년 2월의 한 디버그: 사용자가 "내 에이전트가 어떨 때는 응답에 30초 걸리고; 다른 때는 3초 걸린다"고 보고한다. 트레이스가 없다. 로그는 LLM 호출을 보여주지만, 툴 디스패치도, MCP 서버 왕복도, 하위 에이전트도 아니다. 그래서 추측에 의존하게 된다. 결국 알아낸다. 한 MCP 서버가 가끔 콜드 스타트(cold-start)에서 멈춘다는 것이다.

종단 간 추적이 없으면 이것을 찾을 수 없다. OTel GenAI가 이를 고친다.

이 컨벤션은 2025-2026년 OpenTelemetry 시맨틱 컨벤션 그룹 아래에서 정착되었다. 안정적인 속성 이름을 정의해 Datadog, Langfuse, Phoenix, OpenLLMetry, AgentOps가 모두 동일한 스팬을 파싱하게 한다. 한 번 계측하면 어떤 백엔드로든 출하한다.

## 개념 (The Concept)

### 스팬 계층

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

전체가 하나의 트레이스 id 아래 중첩된다. 스팬 id가 부모-자식 관계를 연결한다.

### 필수 속성

2025-2026 semconv에 따라:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model`: 요청된 모델 문자열(예: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model`: 실제로 서비스된 모델.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id`: 상관관계를 위한 제공자 응답 id.

툴 스팬의 경우:

- `gen_ai.tool.name`: 툴 식별자.
- `gen_ai.tool.call.id`: 특정 호출 id.
- `gen_ai.tool.description`: 툴 설명(선택).

에이전트 스팬의 경우:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### 스팬 종류 (Span kinds)

- 프로세스 경계를 가로지르는 호출(LLM 제공자, MCP 서버)에는 `SpanKind.CLIENT`.
- 에이전트 자신의 루프 단계와 툴 실행에는 `SpanKind.INTERNAL`.

### 옵트인 콘텐츠 캡처

기본적으로 스팬은 메트릭(metric)과 타이밍을 운반한다. 프롬프트나 컴플리션(completion)은 운반하지 않는다. 큰 페이로드와 PII는 기본적으로 꺼져 있다. 콘텐츠를 포함하려면 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`과 특정 콘텐츠 캡처 환경 변수를 설정한다. 프로덕션에서 활성화하기 전에 신중히 검토한다.

### 스팬의 이벤트

토큰 수준 이벤트를 스팬 이벤트로 추가할 수 있다:

- `gen_ai.content.prompt`: 입력 메시지.
- `gen_ai.content.completion`: 출력 메시지.
- `gen_ai.content.tool_call`: 기록된 툴 호출.

이벤트는 상세한 재생을 위해 스팬 내에서 시간 순서로 정렬된다.

### 익스포터 (Exporters)

OTel 스팬은 다음으로 익스포트된다:

- **Jaeger / Tempo.** OSS, 온프레미스.
- **Langfuse.** LLM 관찰성 특화; 토큰 사용량을 시각화한다.
- **Arize Phoenix.** 평가(eval) + 추적 결합.
- **Datadog.** 상업용; `gen_ai.*` 속성을 네이티브로 파싱한다.
- **Honeycomb.** 컬럼 지향; 쿼리 친화적.

모두 와이어 포맷인 OTLP를 말한다. 코드는 어느 백엔드인지 신경 쓰지 않는다.

### MCP를 가로지르는 전파

MCP 클라이언트가 서버를 호출할 때 W3C traceparent 헤더를 요청에 주입한다. Streamable HTTP는 표준 헤더를 지원한다. Stdio는 HTTP 헤더를 네이티브로 운반하지 않으므로, 스펙의 2026년 로드맵은 JSON-RPC 호출에 `_meta.traceparent` 필드를 추가하는 방안을 논의한다.

그것이 출하될 때까지는 모든 요청의 `_meta`에 수동으로 traceparent를 포함한다. 서버가 트레이스 id를 로깅한다.

### 메트릭 (Metrics)

스팬과 함께, GenAI semconv는 메트릭을 정의한다:

- `gen_ai.client.token.usage`: 히스토그램(histogram).
- `gen_ai.client.operation.duration`: 히스토그램.
- `gen_ai.tool.execution.duration`: 히스토그램.

호출별 세부가 필요하지 않은 대시보드에는 이것들을 사용한다.

### AgentOps 계층

AgentOps(2024년 설립)는 GenAI 관찰성에 특화한다. 인기 프레임워크(LangGraph, Pydantic AI, CrewAI)를 감싸 OTel 스팬을 자동으로 방출한다. 스택이 지원되는 프레임워크를 사용하면 유용하고, 그렇지 않으면 수동 계측을 사용한다.

```figure
t3-span-waterfall
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 LLM을 호출하고, 두 개의 툴을 디스패치하고, 한 번의 MCP 왕복을 하는 에이전트에 대해 OTel 형태의 스팬을 (OTLP-JSON 유사 포맷으로) stdout으로 방출한다. 실제 익스포터는 없다. 이 레슨은 스팬 형태와 속성 집합에 집중한다. 출력을 OTLP 호환 뷰어에 붙여넣거나 그냥 읽으면 된다.

볼 것:

- 트레이스 id가 모든 스팬에 걸쳐 공유된다.
- 부모-자식 링크가 `parentSpanId`로 인코딩된다.
- 필수 `gen_ai.*` 속성이 채워진다.
- 콘텐츠 캡처는 기본적으로 꺼져 있다. 한 시나리오가 환경 변수로 그것을 켠다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-otel-genai-instrumentation.md`를 만든다. 에이전트 코드베이스가 주어지면, 이 스킬은 계측 계획을 만들어낸다: 어디에 스팬을 추가할지, 어떤 속성을 채울지, 어떤 익스포터를 목표로 할지.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 스팬을 세고 어느 것이 CLIENT vs INTERNAL인지 식별하라.

2. 콘텐츠 캡처(환경 변수)를 켜고 `gen_ai.content.prompt`와 `gen_ai.content.completion` 이벤트가 나타나는지 확인하라. PII에 대한 함의를 기록하라.

3. 툴 실행 메트릭 `gen_ai.tool.execution.duration`을 추가하고 호출당 히스토그램 샘플로 방출하라.

4. 부모 에이전트 스팬에서 MCP 요청의 `_meta.traceparent` 필드로 traceparent를 전파하라. MCP 서버가 동일한 트레이스 id를 보는지 검증하라.

5. OTel GenAI semconv 스펙을 읽어라. semconv에 나열되어 있지만 이 레슨의 코드가 방출하지 않는 속성 하나를 식별하라. 추가하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 트레이스, 메트릭, 로그를 위한 오픈 표준 |
| GenAI semconv | "GenAI 시맨틱 컨벤션" | LLM / 툴 / 에이전트 스팬을 위한 안정적인 속성 이름 |
| `gen_ai.*` | "속성 네임스페이스" | 모든 GenAI 속성이 이 접두어를 공유한다 |
| 스팬(Span) | "시간이 측정된 연산" | 시작, 끝, 속성을 가진 작업 단위 |
| 트레이스(Trace) | "스팬 간 계보" | 트레이스 id를 공유하는 스팬 트리 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 스팬 방향에 대한 힌트 |
| OTLP | "OpenTelemetry Line Protocol" | 익스포터를 위한 와이어 포맷 |
| 옵트인 콘텐츠(Opt-in content) | "프롬프트 / 컴플리션 캡처" | 기본적으로 꺼짐; 환경 변수로 활성화 |
| traceparent | "W3C 헤더" | 서비스 간 트레이스 컨텍스트를 전파한다 |
| 익스포터(Exporter) | "백엔드별 발송기" | 스팬을 Jaeger / Datadog / 기타로 보내는 컴포넌트 |

## 더 읽을거리 (Further Reading)

- [OpenTelemetry: GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) GenAI 스팬, 메트릭, 이벤트를 위한 표준 컨벤션
- [OpenTelemetry: GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) LLM 및 툴 실행 스팬 속성 목록
- [OpenTelemetry: GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) 에이전트 수준 `invoke_agent` 스팬
- [open-telemetry/semantic-conventions: GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) GitHub 호스팅 진실 소스
- [Datadog: LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) 프로덕션 통합 안내
