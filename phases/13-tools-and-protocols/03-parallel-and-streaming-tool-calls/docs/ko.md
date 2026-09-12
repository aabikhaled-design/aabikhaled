# 병렬 도구 호출과 도구를 사용한 스트리밍

> 세 개의 독립적인 날씨 조회를 직렬화하면 세 번의 왕복(round trip)이 든다. 이것들을 병렬로 돌리면 총 시간은 가장 느린 단일 호출 하나로 수렴한다. 이제 모든 프런티어 제공자(frontier provider)는 한 턴(turn)에 여러 도구 호출을 내보낸다. 보상은 분명하지만 배관 작업(plumbing)은 까다롭다. 이 레슨은 양쪽 절반을 모두 다루며, 병렬 팬아웃(fan-out)과 스트리밍된 인자 재조립, 그리고 id 상관(correlation) 함정에 중점을 둔다.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- `parallel_tool_calls: true`가 존재하는 이유와 이를 비활성화할 시점을 설명하기.
- 병렬 팬아웃 동안 스트리밍된 인자 청크(chunk)를 올바른 도구 호출 id에 상관시키기.
- 부분 `arguments` 문자열을 일찍 파싱하지 않고 완전한 JSON으로 재조립하기.
- 직렬 대 병렬 지연 시간(latency)을 보여주는 세 도시 날씨 벤치마크를 돌리기.

## 문제 (The Problem)

병렬 호출이 없으면 "벵갈루루, 도쿄, 취리히의 날씨는?"에 답하는 에이전트는 이렇게 한다.

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

세 번의 LLM 왕복이 일어나고, 각각이 실행기(executor) 지연 시간도 치른다. 대략 이상적인 벽시계 시간(wall-clock time)의 4배다.

병렬 호출로:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

한 번의 LLM 왕복으로 끝난다. 실행기 시간은 셋의 합이 아니라 최댓값이다. OpenAI, Anthropic, Gemini의 프로덕션 벤치마크는 팬아웃 워크로드에서 벽시계 60~70퍼센트 감소를 보여준다.

대가는 상관 복잡성이다. 세 호출이 순서가 뒤바뀐 채 완료되면, 결과는 모델이 정렬할 수 있도록 일치하는 `tool_call_id`를 운반해야 한다. 결과가 스트리밍될 때는 실행 전에 부분 인자 조각을 완전한 JSON으로 조립해야 한다. Gemini 3가 고유 id를 추가한 데에는, 같은 도구를 향한 두 병렬 호출이 구별 불가능했던 실세계 문제를 풀려는 목적이 일부 있었다.

## 개념 (The Concept)

### 병렬 활성화

- **OpenAI.** `parallel_tool_calls: true`가 기본적으로 켜져 있다. 직렬을 강제하려면 `false`로 설정한다.
- **Anthropic.** `disable_parallel_tool_use: false`(Claude 3.5 이상에서 기본값)를 통한 병렬. 직렬을 위해 `true`로 설정한다.
- **Gemini.** 항상 병렬 가능. `tool_config.function_calling_config.mode = "AUTO"`가 모델이 결정하게 한다.

도구에 순서 의존성이 있을 때(`create_file` 다음 `write_file`), 한 호출의 출력이 다른 호출의 입력에 정보를 줄 때, 또는 속도 제한기(rate limiter)가 팬아웃을 감당할 수 없을 때 병렬을 비활성화하라.

### Id 상관

모델이 내보내는 모든 호출은 `id`를 가진다. 호스트가 반환하는 모든 결과는 동일한 id를 포함해야 한다. 이것 없이는 결과가 모호해진다.

- **OpenAI.** 각 tool-역할 메시지의 `tool_call_id`.
- **Anthropic.** 각 `tool_result` 블록의 `tool_use_id`.
- **Gemini.** 각 `functionResponse`의 `id`(Gemini 3 이상. Gemini 2는 이름으로 매칭했고, 이는 같은 이름의 병렬 호출에서 깨졌다).

### 호출을 동시에 실행하기

호스트는 각 호출의 실행기를 자신의 스레드, 코루틴(coroutine), 또는 원격 워커(worker)에서 돌린다. 가장 간단한 하니스(harness)는 스레드 풀(thread pool)을 쓰고, 프로덕션은 `asyncio.gather`나 구조화된 동시성(structured concurrency)을 가진 asyncio를 쓴다. 완료 순서는 예측 불가능하다. id가 식별자다.

흔한 버그가 하나 있다. 완료 순서가 아니라 호출 목록 순서로 결과를 답하는 것이다. 모델은 `tool_call_id`만 신경 쓰므로 보통은 동작하지만, 결과가 누락되거나 중복되면 순서가 뒤바뀐 제출이 디버깅을 더 어렵게 만든다. 명시적인 id와 함께 완료 순서로 답하기를 권한다.

### 도구 호출 스트리밍

모델이 스트리밍할 때 `arguments`는 조각으로 도착한다. 세 병렬 호출에 대한 세 개의 별개 청크 스트림이 와이어(wire)에서 인터리브(interleave)된다. id당 하나의 누적기(accumulator)가 필요하다.

제공자별 형태:

- **OpenAI.** 각 청크는 `choices[0].delta.tool_calls[i].function.arguments`(부분 문자열)다. 청크는 `index`(호출 목록에서의 위치)를 운반한다. 인덱스별로 누적하고, `id`가 처음 나타날 때 읽고, `finish_reason = "tool_calls"`일 때 JSON을 파싱한다.
- **Anthropic.** 스트림 이벤트는 `message_start`, 그다음 블록당 하나의 `content_block_start`(타입 `tool_use`, id, name, 빈 input 포함)다. `content_block_delta` 이벤트가 `input_json_delta` 청크를 운반한다. `content_block_stop`이 각 블록을 닫는다.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3 이상)는 `functionCallId`를 가진 청크를 내보내어 호출이 깔끔하게 인터리브된다. Gemini 3 이전에는 스트리밍이 한 번에 하나의 완전한 호출을 반환했다.

### 부분 JSON과 일찍-파싱 함정

`arguments`가 완전해지기 전에는 파싱할 수 없다. `{"city": "Beng` 같은 부분 JSON은 유효하지 않으며 예외를 일으킨다. 올바른 게이트는 제공자의 호출 종료 신호다. OpenAI의 `finish_reason = "tool_calls"`, Anthropic의 `content_block_stop`, 또는 Gemini의 스트림 종료 이벤트. 그제서야 `json.loads`를 시도한다. 더 견고한 접근법은 구조가 완성될 때 이벤트를 산출하는 증분 JSON 파서(incremental JSON parser)를 쓰는 것이다. OpenAI의 스트리밍 가이드는 라이브 "생각 중" 표시기를 보여주는 UX를 위해 이를 권장한다. 중괄호 세기(brace-counting)는 완전성 테스트로 신뢰할 수 없으며(따옴표로 묶인 문자열이나 이스케이프된 콘텐츠 안의 중괄호가 거짓 양성을 일으킨다), 비공식적인 디버그 휴리스틱으로만 써야 한다.

### 순서가 뒤바뀐 완료

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

호스트 응답은 여전히 id를 인용해야 한다.

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

응답에서의 순서는 OpenAI나 Anthropic의 정확성에 영향을 주지 않는다. Gemini는 id가 일치하는 한 어떤 순서든 받아들인다.

### 벤치마크: 직렬 대 병렬

`code/main.py`의 하니스는 400, 600, 800ms 지연 시간을 가진 세 실행기를 시뮬레이션한다. 직렬은 총 1800ms에 돌린다. 병렬은 max(400, 600, 800) = 800ms에 돌린다. 차이는 비례적이 아니라 일정하므로, 절감은 도구 개수와 함께 커진다.

실세계 주의 사항: 병렬 호출은 다운스트림 API에 부담을 준다. 속도 제한된 서비스로의 10방향 팬아웃은 실패한다. Phase 13 · 17은 게이트웨이 수준의 배압(backpressure)을 다룬다. 재시도 의미론은 미래 phase로 계획되어 있다.

### 스트리밍 팬아웃 벽시계

모델 자체가 스트리밍하면, 모든 호출이 확정되기를 기다리는 대신 한 호출의 인자가 완전해지자마자 실행을 시작할 수 있다. 이것은 OpenAI가 문서화하지만 모든 SDK가 노출하지는 않는 최적화다. 이 레슨의 하니스는 그것을 한다. 시뮬레이션된 스트림이 완전한 인자 객체를 산출하자마자, 호스트는 그 호출을 시작한다.

```figure
tp-parallel-fanout
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 두 절반을 가진다. 첫 번째는 `concurrent.futures.ThreadPoolExecutor`를 써서 세 시뮬레이션된 날씨 호출을 직렬과 병렬로 돌리고 벽시계 시간을 출력한다. 두 번째 절반은 가짜 스트리밍 응답(한 스트림에 인터리브된 세 병렬 호출의 `arguments` 청크)을 재생하고 `StreamAccumulator`로 id별로 재조립한다. LLM 없음, 네트워크 없음, 오직 재조립 로직만.

볼 것:

- 직렬 타이머는 1.8초에 도달한다. 병렬 타이머는 동일한 가짜 지연 시간에서 0.8초에 도달한다.
- 누적기는 id별로 버퍼링하고 각 호출의 JSON이 완전할 때만 파싱하여 순서가 뒤바뀐 채 도착하는 청크를 처리한다.
- 실행기는 모든 스트림이 끝난 후가 아니라 한 id의 인자가 확정되자마자 시작한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-parallel-call-safety-check.md`를 만든다. 도구 레지스트리(registry)가 주어지면, 이 스킬은 어떤 도구가 병렬화하기 안전한지, 어떤 것이 순서 의존성을 가지는지, 어떤 것이 다운스트림 속도 제한을 압도할지 감사하여, 도구별 `parallel_safe` 플래그를 가진 수정된 레지스트리를 반환한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 돌리고 시뮬레이션된 지연 시간을 바꿔라. 병렬-대-직렬 비율이 대략 `max/sum`임을 확인하라(실제 실행은 스레드 스케줄링, 직렬화, 하니스 오버헤드 때문에 이상에서 약간 벗어난다). 어떤 지연 시간 분포에서 병렬이 의미 없어지는가?

2. "호출이 스트림 도중에 취소됨" 사례를 처리하도록 누적기를 확장하여, 그 버퍼를 버리고 `cancelled` 이벤트를 내보내라. 어떤 제공자가 이 사례를 명시적으로 문서화하는가? Anthropic의 `content_block_stop` 의미론과 OpenAI의 `finish_reason: "length"` 동작을 확인하라.

3. 스레드 풀을 `asyncio.gather`로 교체하라. 둘 다 벤치마크하라. async에서 더 낮은 컨텍스트 전환 비용 때문에 작은 이득을 보겠지만, 이는 실행기가 실제 I/O를 할 때만 그렇다.

4. 병렬화해서는 안 되는 두 도구를 골라라(예: `create_file` 다음 `write_file`). 레지스트리에 `ordering_dependency` 그래프를 추가하고 그 그래프에 따라 병렬 팬아웃을 게이트하라. 이것이 의존성 인식 스케줄링(dependency-aware scheduling)을 위한 최소한의 장치이며, 미래의 에이전트 엔지니어링 phase가 이를 형식화한다.

5. OpenAI의 병렬 함수 호출 섹션과 Anthropic의 `disable_parallel_tool_use` 문서를 읽어라. Anthropic이 병렬성 비활성화를 권장하는 실세계 도구 타입 하나를 식별하라. (힌트: 같은 리소스에 대한 결과 초래형(consequential) 변경.)

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 병렬 도구 호출(Parallel tool calls) | "한 턴에 팬아웃" | 모델이 하나의 assistant 메시지에 여러 도구 호출을 내보낸다 |
| `parallel_tool_calls` | "OpenAI의 플래그" | 다중 호출 방출을 활성화하거나 비활성화 |
| `disable_parallel_tool_use` | "Anthropic의 역(inverse)" | 옵트아웃 플래그. 기본값은 병렬 활성화 |
| 도구 호출 id(Tool call id) | "상관 핸들(handle)" | 결과 메시지가 되돌려줘야 하는 호출별 식별자 |
| 누적기(Accumulator) | "스트림 버퍼" | 부분 `arguments` 청크를 위한 id별 문자열 버퍼 |
| 순서가 뒤바뀐 완료(Out-of-order completion) | "빠른 것 먼저" | 병렬 호출은 예측 불가능한 순서로 끝난다. id가 접착제다 |
| 의존성 그래프(Dependency graph) | "순서 제약" | 출력이 다른 도구의 입력으로 들어가는 도구. 병렬화 불가 |
| 일찍-파싱 함정(Parse-early trap) | "JSON.parse가 폭발" | 불완전한 `arguments` 문자열을 파싱하려는 시도 |
| `streamFunctionCallArguments` | "Gemini 3 기능" | 호출당 고유 id를 가진 스트리밍된 인자 청크 |
| 완료 순서 응답(Completion-order reply) | "모두를 기다리지 마라" | 결과가 도착하는 대로 id를 키로 하여 답한다 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) 기본 동작과 옵트아웃 플래그
- [Anthropic: Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) `disable_parallel_tool_use`와 결과 배치(batching)
- [Google: Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) Gemini 3부터의 id 상관 병렬 호출
- [OpenAI: Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) OpenAI 스트림을 위한 청크 인자 재조립
- [Anthropic: Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) `input_json_delta`를 가진 `content_block_delta`
