# OpenTelemetry GenAI 시맨틱 컨벤션

> OpenTelemetry의 GenAI SIG(2024년 4월 출범)는 에이전트(agent) 텔레메트리(telemetry)의 표준 스키마를 정의한다. 스팬(span) 이름, 속성(attribute), 내용 포착(content-capture) 규칙이 벤더 전반에서 수렴하여, 에이전트 트레이스(trace)가 Datadog, Grafana, Jaeger, Honeycomb에서 동일한 것을 의미하게 한다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- GenAI 스팬 범주를 거명하기: 모델/클라이언트, 에이전트, 도구.
- `invoke_agent` CLIENT 스팬과 INTERNAL 스팬을 구별하고 각각이 언제 적용되는지 설명하기.
- 최상위 GenAI 속성을 나열하기: 제공자 이름, 요청 모델, 데이터 소스 ID.
- 내용 포착 계약(contract)을 설명하기: 옵트인(opt-in), `OTEL_SEMCONV_STABILITY_OPT_IN`, 외부 참조 권장.

## 문제 (The Problem)

모든 벤더가 자신만의 스팬 이름을 발명한다. 운영(ops) 팀은 결국 프레임워크별 대시보드를 만들게 된다. OpenTelemetry의 GenAI SIG는 생태계 전체가 목표로 삼는 하나의 표준을 정의하여 이를 바로잡는다.

## 개념 (The Concept)

### 스팬 범주

1. **모델 / 클라이언트 스팬.** 원시(raw) LLM 호출을 다룬다. 제공자 SDK(Anthropic, OpenAI, Bedrock)와 프레임워크 모델 어댑터가 방출한다.
2. **에이전트 스팬.** `create_agent`(에이전트가 구성될 때)와 `invoke_agent`(에이전트가 실행될 때).
3. **도구 스팬.** 도구 호출당 하나; 부모-자식 관계로 에이전트 스팬에 연결됨.

### 에이전트 스팬 명명

- 스팬 이름: 이름이 있으면 `invoke_agent {gen_ai.agent.name}`; 없으면 `invoke_agent`로 폴백.
- 스팬 종류(span kind):
  - **CLIENT**: 원격 에이전트 서비스용(OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL**: 인프로세스(in-process) 에이전트 프레임워크용(LangChain, CrewAI, 로컬 ReAct).

### 핵심 속성

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model`: 모델 ID.
- `gen_ai.response.model`: 해소된(resolved) 모델(라우팅으로 인해 요청과 다를 수 있음).
- `gen_ai.agent.name`: 에이전트 식별자.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id`: RAG의 경우: 어느 말뭉치(corpus)나 저장소를 조회했는지.

Anthropic, Azure AI Inference, AWS Bedrock, OpenAI를 위한 기술별 컨벤션이 존재한다.

### 내용 포착

기본 규칙: 계측(instrumentation)은 기본적으로 입력/출력을 포착하지 말아야 한다(SHOULD NOT). 포착은 다음을 통해 옵트인된다:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

권장 프로덕션 패턴: 내용을 외부(S3, 로그 저장소)에 저장하고, 스팬에는 참조(포인터 ID, 산문이 아님)를 기록하라. 이것이 관측성에 배선된 Lesson 27의 내용 오염(content-poisoning) 방어다.

### 안정성

대부분의 컨벤션은 2026년 3월 기준 실험적(experimental)이다. 다음으로 안정 프리뷰에 옵트인하라:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+는 GenAI 속성을 자신의 LLM Observability 스키마로 네이티브 매핑한다. 다른 백엔드(Grafana, Honeycomb, Jaeger)는 원시 속성을 지원한다.

### 이 패턴이 잘못되는 지점

- **스팬에 전체 프롬프트(prompt) 포착.** 운영자가 읽을 수 있는 트레이스에 PII, 비밀, 고객 데이터가 들어감. 외부에 저장하라.
- **`gen_ai.provider.name` 없음.** 귀속(attribution)이 누락되면 다중 제공자 대시보드가 깨진다.
- **부모 링크 없는 스팬.** 고아가 된 도구 스팬. 항상 컨텍스트를 전파하라.
- **안정성 옵트인 미설정.** 백엔드 업그레이드 시 속성 이름이 바뀔 수 있다.

```figure
ae-genai-span-tree
```

## 직접 만들기 (Build It)

`code/main.py`는 GenAI 컨벤션에 부합하는 stdlib 스팬 방출기를 구현한다:

- GenAI 속성 스키마를 갖춘 `Span`.
- `start_span`과 중첩 컨텍스트를 갖춘 `Tracer`.
- 다음을 방출하는 스크립트화된 에이전트 실행: `create_agent`, `invoke_agent`(INTERNAL), 도구별 스팬, LLM 호출을 위한 `chat` 스팬.
- 프롬프트를 외부에 저장하고 스팬에 ID를 기록하는 내용 포착 모드.

실행:

```
python3 code/main.py
```

출력: 필요한 모든 GenAI 속성을 갖춘 스팬 트리와, 옵트인된 내용 참조를 보여주는 "외부 저장소".

## 라이브러리로 써보기 (Use It)

- **Datadog LLM Observability**(v1.37+)는 속성을 네이티브 매핑한다.
- **Langfuse / Phoenix / Opik**(Lesson 24): 생태계를 자동 계측한다.
- **Jaeger / Honeycomb / Grafana Tempo**: 원시 OTel 트레이스; GenAI 속성으로 대시보드를 구축하라.
- **자체 호스팅**: GenAI 프로세서를 갖춘 OTel Collector를 실행하라.

## 산출물 (Ship It)

`outputs/skill-otel-genai.md`는 내용 포착 기본값과 외부 참조 저장소를 갖춘 OTel GenAI 스팬을 기존 에이전트에 배선한다.

## 연습 문제 (Exercises)

1. Lesson 01의 ReAct 루프를 `invoke_agent`(INTERNAL) + 도구별 스팬으로 계측하라. Jaeger 인스턴스로 전송하라.
2. "참조 전용" 모드로 내용 포착을 추가하라: 프롬프트는 SQLite로, 스팬 속성은 행 ID만 운반.
3. `gen_ai.data_source.id` 명세를 읽어라. Lesson 09의 Mem0 검색에 배선하라.
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`을 설정하고 컬렉터에 의해 속성 이름이 바뀌지 않는지 검증하라.
5. 대시보드를 구축하라: GenAI 속성만으로 "어떤 도구 오류가 어떤 모델과 상관관계가 있는지".

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 그룹" | 스키마를 정의하는 OTel 워킹 그룹 |
| invoke_agent | "에이전트 스팬" | 에이전트 실행을 나타내는 스팬의 이름 |
| CLIENT span | "원격 호출" | 원격 에이전트 서비스 호출을 위한 스팬 |
| INTERNAL span | "인프로세스" | 인프로세스 에이전트 실행을 위한 스팬 |
| gen_ai.provider.name | "제공자" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 소스" | 검색이 적중한 말뭉치/저장소 |
| Content capture | "프롬프트 로깅" | 메시지의 옵트인 포착; 프로덕션에서는 외부 저장 |
| Stability opt-in | "프리뷰 모드" | 실험적 컨벤션을 고정하는 환경 변수 |

## 더 읽을거리 (Further Reading)

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): 명세
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): 기본적으로 GenAI 스팬
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/): OTel 스팬 내장
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview): W3C 트레이스 컨텍스트 전파
