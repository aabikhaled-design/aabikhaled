# 함수 호출 심층 분석: OpenAI, Anthropic, Gemini

> 세 프런티어 제공자(frontier provider)는 2024년에 동일한 도구 호출 루프로 수렴했지만, 그 외의 모든 것에서는 갈라졌다. OpenAI는 `tools`와 `tool_calls`를 쓴다. Anthropic은 `tool_use`와 `tool_result` 블록을 쓴다. Gemini는 `functionDeclarations`와 고유 id 상관(correlation)을 쓴다. 이 레슨은 세 가지를 나란히 비교(diff)해, 한 제공자에서 출시한 코드를 다른 제공자로 포팅할 때 깨지지 않게 한다.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- OpenAI, Anthropic, Gemini 함수 호출 페이로드 사이의 세 가지 형태 차이(선언, 호출, 결과)를 진술하기.
- 하나의 도구 선언을 세 제공자 형식 모두로 번역하고, 엄격 모드(strict-mode) 제약이 어디서 달라질지 예측하기.
- 각 제공자에서 `tool_choice`를 사용해 도구 호출을 강제, 금지, 또는 자동 선택하기.
- 제공자별 하드 한도(도구 개수, 스키마 깊이, 인자 길이)와 한도 위반 시 각 제공자가 내보내는 오류 서명(error signature)을 알기.

## 문제 (The Problem)

함수 호출 요청의 형태는 제공자마다 다르다. 2026년 프로덕션 스택에서 가져온 세 가지 구체적 예시다.

**OpenAI Chat Completions / Responses API.** `tools: [{type: "function", function: {name, description, parameters, strict}}]`를 전달한다. 모델의 응답은 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`를 담으며, 여기서 `arguments`는 직접 파싱해야 하는 JSON 문자열이다. 엄격 모드(`strict: true`)는 제약 디코딩(constrained decoding)으로 스키마 준수를 강제한다.

**Anthropic Messages API.** `tools: [{name, description, input_schema}]`를 전달한다. 응답은 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`로 돌아온다. `input`은 이미 파싱되어 있다(문자열이 아니라 객체). `{type: "tool_result", tool_use_id, content}` 블록을 담은 새로운 `user` 메시지로 답한다.

**Google Gemini API.** `tools: [{functionDeclarations: [{name, description, parameters}]}]`를 전달한다(`functionDeclarations` 아래에 중첩됨). 응답은 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`로 도착하며, 여기서 `id`는 병렬 호출 상관을 위해 Gemini 3 이상에서 고유하다. `{functionResponse: {name, id, response}}`로 답한다.

같은 루프다. 다른 필드 이름, 다른 중첩, 다른 문자열-대-객체 관례, 다른 상관 메커니즘. OpenAI에서 날씨 에이전트를 작성한 팀은 단지 배관 작업(plumbing)을 위해 Anthropic으로 이틀, Gemini로 다시 하루의 포팅 비용을 치른다.

이 레슨은 세 형식을 하나의 표준(canonical) 도구 선언으로 통합하고 가장자리(edge)에서 라우팅하는 번역기를 만든다. Phase 13 · 17은 같은 패턴을 LLM 게이트웨이로 일반화한다.

## 개념 (The Concept)

### 공통 구조 (The common structure)

모든 제공자에 다섯 가지가 필요하다.

1. **도구 목록.** 도구별 이름, 설명, 입력 스키마.
2. **도구 선택(Tool choice).** 특정 도구를 강제하거나, 도구를 금지하거나, 모델이 결정하게 한다.
3. **호출 방출(Call emission).** 도구와 인자를 명명하는 구조화된 출력.
4. **호출 id.** 응답을 올바른 호출과 상관시킨다(병렬에서 중요).
5. **결과 주입(Result injection).** 결과를 호출과 다시 묶는 메시지 또는 블록.

### 형태 차이, 필드별로

| 측면 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 선언 봉투(envelope) | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 스키마 필드 | `parameters` | `input_schema` | `parameters` |
| 응답 컨테이너 | assistant 메시지의 `tool_calls[]` | `tool_use` 타입의 `content[]` | `functionCall` 타입의 `parts[]` |
| 인자 타입 | 문자열화된 JSON | 파싱된 객체 | 파싱된 객체 |
| Id 형식 | `call_...` (OpenAI가 생성) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| 결과 블록 | `tool` 역할, `tool_call_id` | `tool_result`를 가진 `user`, `tool_use_id` | 일치하는 `id`를 가진 `functionResponse` |
| 특정 도구 강제 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 도구 금지 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 엄격 스키마 | `strict: true` | 스키마-이즈-스키마(항상 강제됨) | 요청 수준의 `responseSchema` |

### 실제로 부딪힐 한도들

- **OpenAI.** 요청당 도구 128개. 스키마 깊이 5. 인자 문자열 <= 8192바이트. 엄격 모드는 `$ref` 없음, 겹치는 `oneOf`/`anyOf`/`allOf` 없음, 모든 속성이 `required`에 나열될 것을 요구한다.
- **Anthropic.** 요청당 도구 64개. 스키마 깊이는 사실상 무제한이나 실용 한도는 10이다. 엄격 모드 플래그가 없다. 스키마는 계약이고 모델은 준수하는 경향이 있다.
- **Gemini.** 요청당 함수 64개. 스키마 타입은 OpenAPI 3.0 부분집합(JSON Schema 2020-12에서 약간 분기됨). 병렬 호출은 Gemini 3부터 고유 id를 쓴다.

### `tool_choice` 동작

모두가 지원하는 세 모드인데, 이름은 다르다.

- **Auto.** 모델이 도구나 텍스트를 고른다. 기본값.
- **Required / Any.** 모델이 최소 하나의 도구를 호출해야 한다.
- **None.** 모델이 도구를 호출해서는 안 된다.

여기에 각 제공자 고유의 한 모드가 더 있다.

- **OpenAI.** 이름으로 특정 도구를 강제한다.
- **Anthropic.** 이름으로 특정 도구를 강제한다. `disable_parallel_tool_use` 플래그가 단일 대 다중을 분리한다.
- **Gemini.** `mode: "VALIDATED"`는 모델 의도와 무관하게 모든 응답을 스키마 검증기(validator)로 라우팅한다.

### 병렬 호출 (Parallel calls)

OpenAI의 `parallel_tool_calls: true`(기본값)는 하나의 assistant 메시지에 여러 호출을 내보낸다. 모두 실행하고 `tool_call_id`당 하나의 항목을 담은 배치(batched) tool-역할 메시지로 답한다. Anthropic은 과거에 단일 호출을 했고, `disable_parallel_tool_use: false`(Claude 3.5 시점 기본값)가 다중을 활성화한다. Gemini 2는 병렬 호출을 허용했지만 안정적인 id를 주지 않았고, Gemini 3는 UUID를 추가해 순서가 뒤바뀐 응답도 깔끔하게 상관된다.

### 스트리밍 (Streaming)

셋 다 스트리밍된 도구 호출을 지원한다. 와이어 형식(wire format)은 다르다.

- **OpenAI.** `tool_calls[i].function.arguments`의 델타 청크(delta chunk)가 점진적으로 도착한다. `finish_reason: "tool_calls"`까지 누적한다.
- **Anthropic.** block-start / block-delta / block-stop 이벤트. `input_json_delta` 청크가 부분 인자를 운반한다.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3에서 신규)는 `functionCallId`를 가진 청크를 내보내어 여러 병렬 호출이 인터리브(interleave)될 수 있다.

Phase 13 · 03은 병렬 + 스트리밍 재조립을 깊이 다룬다. 이 레슨은 선언과 단일 호출 형태에 초점을 둔다.

### 오류와 수리 (Errors and repair)

잘못된 인자 오류도 다르게 보인다.

- **OpenAI (비엄격).** 모델이 `arguments: "{bad json}"`을 반환하고 JSON 파싱이 실패하면, 오류 메시지를 주입하고 다시 호출한다.
- **OpenAI (엄격).** 검증이 디코딩 중에 일어난다. 잘못된 JSON은 불가능하지만 `refusal`이 나타날 수 있다.
- **Anthropic.** `input`이 예상치 못한 필드를 담을 수 있다. 스키마는 권고적(advisory)이다. 서버 측에서 검증하라.
- **Gemini.** OpenAPI 3.0 특이점. 객체 필드의 `enum`이 조용히 무시된다. 직접 검증하라.

### 번역기 패턴 (The translator pattern)

코드에서 표준 도구 선언은 다음과 같이 생겼다(형태는 직접 고른다).

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

세 개의 작은 함수가 이것을 세 제공자 형태로 번역한다. `code/main.py`의 하니스(harness)가 정확히 이것을 한 뒤, 가짜 도구 호출을 각 제공자의 응답 형태로 왕복(round-trip)시킨다. 네트워크는 필요 없다. 이 레슨은 HTTP가 아니라 형태를 가르친다.

프로덕션 팀은 이 번역기를 `AbstractToolset`(Pydantic AI), `UniversalToolNode`(LangGraph), 또는 `BaseTool`(LlamaIndex)로 감싼다. Phase 13 · 17은 세 가지 중 어느 것 앞에서든 OpenAI 형태 API를 노출하는 게이트웨이를 출시한다.

```figure
function-call-args
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 하나의 표준 `Tool` 데이터클래스(dataclass)와 OpenAI, Anthropic, Gemini 선언 JSON을 내보내는 세 개의 번역기를 정의한다. 그다음 각 형태의 손으로 만든 제공자 응답을 동일한 표준 호출 객체로 파싱해, 의미론이 껍질 아래에서는 동일함을 보여준다. 이를 돌리고 세 선언을 나란히 비교하라.

볼 것:

- 세 선언 블록은 봉투와 필드 이름에서만 다르다.
- 세 응답 블록은 호출이 사는 곳에서 다르다(최상위 `tool_calls`, `content[]` 블록, `parts[]` 항목).
- 하나의 `canonical_call()` 함수가 세 응답 형태 모두에서 `{id, name, args}`를 추출한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-provider-portability-audit.md`를 만든다. 한 제공자를 대상으로 한 함수 호출 통합이 주어지면, 이 스킬은 이식성 감사(portability audit)를 만든다. 어떤 제공자 한도에 의존하는가, 어떤 필드가 이름을 바꿔야 하는가, 각 다른 제공자로 포팅할 때 무엇이 깨지는가.

## 연습 문제 (Exercises)

1. `code/main.py`를 돌려서 세 제공자 선언 JSON이 모두 동일한 기저 `Tool` 객체를 직렬화하는지 확인하라. 표준 도구를 수정해 enum 파라미터를 추가하고, Gemini 번역기만 OpenAPI 특이점을 처리해야 함을 확인하라.

2. 모델이 `list_tools` 또는 탐색(discovery) 호출 후 반환하는 도구 목록을 추출하는 `ListToolsResponse` 파서를 각 제공자에 대해 추가하라. OpenAI는 네이티브로 그것을 갖지 않는다. 이 비대칭성을 기록하라.

3. `tool_choice` 변환을 구현하라. 표준 `ToolChoice(mode="force", tool_name="x")`를 세 제공자 형태 모두로 매핑하라. 그다음 `mode="any"`와 `mode="none"`을 매핑하라. 레슨의 비교 표를 확인하라.

4. 세 제공자 중 하나를 골라 그 함수 호출 가이드를 처음부터 끝까지 읽어라. 다른 둘이 지원하지 않는 스키마 명세의 필드 하나를 찾아라. 후보: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. 테스트 벡터를 작성하라. 인자가 선언된 스키마를 위반하는 도구 호출. 각 제공자의 검증기(레슨 01의 stdlib 것을 대용으로 써도 된다)를 통해 돌리고 어떤 오류가 발화하는지 기록하라. 엄격성을 위해 프로덕션에서 어느 제공자를 쓸지 문서화하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 함수 호출(Function calling) | "도구 사용" | 구조화된 도구 호출 방출을 위한 제공자 수준 API |
| 도구 선언(Tool declaration) | "도구 명세" | 이름 + 설명 + JSON Schema 입력 페이로드 |
| `tool_choice` | "강제 / 금지" | Auto / required / none / 특정-이름 모드 |
| 엄격 모드(Strict mode) | "스키마 강제" | 디코딩을 스키마에 맞도록 제약하는 OpenAI 플래그 |
| `tool_use` 블록 | "Anthropic의 호출 형태" | id, name, input을 가진 인라인 콘텐츠 블록 |
| `functionCall` 파트 | "Gemini의 호출 형태" | name, args, id를 담은 `parts[]` 항목 |
| 문자열로서의 인자(Arguments-as-string) | "문자열화된 JSON" | OpenAI는 args를 객체가 아니라 JSON 문자열로 반환한다 |
| 병렬 도구 호출(Parallel tool calls) | "한 턴에 팬아웃(fan-out)" | 하나의 assistant 메시지 안의 여러 도구 호출 |
| 거부(Refusal) | "모델이 거절" | 호출 대신 나오는 엄격 모드 전용 거부 블록 |
| OpenAPI 3.0 부분집합 | "Gemini 스키마 특이점" | Gemini는 사소한 차이가 있는 JSON Schema 유사 방언을 사용한다 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling) 엄격 모드와 병렬 호출을 포함한 표준 레퍼런스
- [Anthropic: Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) `tool_use`와 `tool_result` 블록 의미론
- [Google: Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) 병렬 호출, 고유 id, OpenAPI 부분집합
- [Vertex AI: Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) Gemini의 엔터프라이즈 표면
- [OpenAI: Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) 엄격 모드 스키마 강제 세부 사항
