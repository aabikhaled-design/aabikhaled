# 도구 인터페이스(Tool Interface): 에이전트에게 구조화된 입출력이 필요한 이유

> 언어 모델은 토큰(token)을 생성한다. 프로그램은 행동을 취한다. 이 둘 사이의 간극이 바로 도구 인터페이스(tool interface)다. 모델이 행동을 요청하고 호스트(host)가 그것을 실행하게 해주는 계약이다. 2026년의 모든 스택(OpenAI, Anthropic, Gemini의 함수 호출(function calling); MCP의 `tools/call`; A2A의 태스크 파트(task parts))은 동일한 4단계 루프를 서로 다르게 인코딩한 것이다. 이 레슨은 그 루프에 이름을 붙이고, 그것을 돌리는 데 필요한 최소한의 장치를 보여준다.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- 텍스트만 생성할 수 있는 LLM이 왜 스스로는 현실 세계에 대해 행동을 취할 수 없는지 설명하기.
- 4단계 도구 호출 루프(describe → decide → execute → observe)를 그리고, 각 단계를 누가 소유하는지 명명하기.
- 도구 설명을 세 부분(이름, JSON Schema 입력, 결정론적 실행기(executor) 함수)로 작성하기.
- 순수(pure) 도구와 부수 효과(side-effecting) 도구를 구분하고, 이 구분이 안전성에 중요한 이유를 설명하기.

## 문제 (The Problem)

LLM은 다음 토큰의 확률 분포(probability distribution)를 내보낸다. 그것이 출력 표면(output surface)의 전부다. 채팅 모델에게 "지금 벵갈루루의 날씨는 어때?"라고 물으면, 그럴듯한 문장을 쓸 수는 있지만 날씨 API에 직접 접속할 수는 없다. 그 문장은 우연히 맞을 수도 있고, 사흘 지난 정보일 수도 있다.

그 간극을 메우는 것이 도구 인터페이스의 목적이다. 호스트 프로그램(에이전트(agent) 런타임, Claude Desktop, ChatGPT, Cursor, 혹은 커스텀 스크립트)은 호출 가능한 도구 목록을 모델에게 광고한다. 모델은 행동이 필요하다고 판단하면 도구 이름과 그 인자(arguments)를 담은 구조화된 페이로드(payload)를 내보낸다. 호스트는 그 페이로드를 파싱하고 도구를 실제로 실행한 뒤 결과를 다시 모델에게 전달한다. 이 루프는 모델이 더 이상 호출이 필요 없다고 판단할 때까지 계속된다.

이 계약의 첫 버전은 2023년 6월 OpenAI의 "functions" 파라미터로 출시되었다. Anthropic은 Claude 2.1에서 `tool_use` 블록으로 그 뒤를 이었다. Gemini는 몇 달 뒤 `functionDeclarations`를 추가했다. 이제 모든 제공자(provider)가 동일한 형태를 노출한다. JSON Schema로 타입이 지정된 도구 목록이 들어가고, JSON 페이로드 도구 호출이 나온다. Model Context Protocol(2024년 11월)은 이 계약을 일반화하여 하나의 도구 레지스트리(registry)가 모든 모델을 섬기도록 했다. A2A(2026년 4월, v1.0)는 에이전트 간 위임(agent-to-agent delegation)을 위해 동일한 기본 요소(primitive)를 한 겹 더 쌓았다.

4단계 루프는 이 모든 것 아래에 깔린 불변항(invariant)이다. Phase 13의 나머지 모든 것은 이것의 정교화다.

## 개념 (The Concept)

### 1단계: describe (기술하기)

호스트는 각 도구를 세 가지 필드로 선언한다.

- **이름(Name).** 안정적이고 기계가 읽을 수 있는 식별자. "weather thing"이 아니라 `get_weather`.
- **설명(Description).** 한 문단짜리 자연어 브리프. "특정 도시의 현재 상태를 사용자가 물을 때 사용하라. 과거 데이터에는 사용하지 마라."
- **입력 스키마(Input schema).** 도구의 인자를 기술하는 JSON Schema 객체(draft 2020-12).

모델은 이 목록을 받는다. 현대의 제공자들은 이 선언들을 제공자별 템플릿으로 시스템 프롬프트(system prompt)에 직렬화하므로, 호출자는 구조화된 형태만 다루면 된다.

### 2단계: decide (결정하기)

사용자의 메시지와 사용 가능한 도구가 주어지면, 모델은 세 가지 행동 중 하나를 선택한다.

1. **텍스트로 직접 답한다.** 도구 호출 없음.
2. **하나 이상의 도구를 호출한다.** 구조화된 호출 객체를 내보낸다. `parallel_tool_calls: true`(OpenAI와 Gemini에서는 기본 활성화, Anthropic에서는 옵트인) 아래에서 모델은 한 턴(turn)에 여러 호출을 내보낼 수 있다.
3. **거부한다.** 엄격 모드(strict-mode) 구조화된 출력은 호출 대신 타입이 지정된 `refusal` 블록을 만들어낼 수 있다.

도구 호출 페이로드는 세 개의 안정적인 필드를 가진다. 호출 `id`, 도구 `name`, 그리고 JSON `arguments` 객체다. id가 있는 이유는 호스트가 나중의 결과를 특정 호출과 연결하기 위함이며, 병렬 호출이 순서가 뒤바뀐 채 돌아올 때 이것이 중요하다.

### 3단계: execute (실행하기)

호스트는 호출을 받아 인자를 선언된 스키마에 대해 검증하고 실행기를 돌린다. 잘못된 인자는 모델이 필드를 환각(hallucinate)했거나 잘못된 타입을 사용했다는 뜻이며, 약한 모델에서 매우 흔한 실패 양상이다. 프로덕션 호스트는 잘못된 인자에 대해 세 가지 중 하나를 한다. 빠르게 실패하고 오류를 모델에게 드러내거나, 제약 파서(constrained parser)로 JSON을 수리하거나, 검증 오류를 프롬프트에 포함시켜 모델을 재시도시킨다.

실행기 자체는 평범한 코드다. Python, TypeScript, 셸 명령, 데이터베이스 쿼리. 그것은 결과를 만들어내며, 그 결과는 보통 문자열이지만 어떤 JSON 값일 수도 있고 구조화된 콘텐츠 블록(MCP에서는 텍스트, 이미지, 또는 리소스 참조)일 수도 있다. 결과는 반드시 직렬화 가능(serializable)해야 한다.

### 4단계: observe (관찰하기)

호스트는 도구 결과를 대화에 추가하고(일치하는 `id`를 가진 `tool` 역할 메시지로), 모델을 다시 호출한다. 이제 모델은 컨텍스트 안에 도구 출력을 가지고 있으며, 최종 답을 만들거나 추가 호출을 요청할 수 있다. 이는 모델이 호출을 멈추거나 호스트가 반복 횟수의 안전 한도에 도달할 때까지 계속된다.

### 신뢰 구분 (The trust split)

도구는 안전성 측면에서 중요한 두 가지 종류로 나뉜다.

- **순수(Pure).** 읽기 전용, 결정론적, 부수 효과 없음. `get_weather`, `search_docs`, `get_current_time`. 추측적으로 호출해도 안전하다.
- **결과 초래형(Consequential).** 상태를 변경하고, 돈을 쓰고, 사용자 데이터를 건드린다. `send_email`, `delete_file`, `execute_trade`. 반드시 게이트(gate)를 거쳐야 한다.

Meta의 2026년 에이전트 보안 "둘의 규칙(Rule of Two)"은 단일 턴이 다음 셋 중 최대 두 개만 결합할 수 있다고 말한다. 신뢰할 수 없는 입력, 민감한 데이터, 결과 초래형 행동. 도구 인터페이스는(호출을 거부하거나, 사용자 확인을 요구하거나, 스코프(scope)를 에스컬레이션함으로써) 그 규칙을 강제하는 지점이다. 전체 보안 챕터는 Phase 13 · 15를, 에이전트 수준 권한 정책은 Phase 14 · 09를 참고하라.

### 루프가 사는 곳 (Where the loop lives)

| 맥락 | 누가 기술하는가 | 누가 결정하는가 | 누가 실행하는가 |
|---------|---------------|-------------|--------------|
| 단일 턴 함수 호출 (OpenAI/Anthropic/Gemini) | 앱 개발자 | LLM | 앱 개발자 |
| MCP | MCP 서버 | MCP 클라이언트를 통한 LLM | MCP 서버 |
| A2A | Agent Card 발행자 | 호출하는 에이전트 | 호출받는 에이전트 |
| 웹 브라우저 (함수 호출 에이전트) | 브라우저 확장 / WebMCP | LLM | 브라우저 런타임 |

어디서나 같은 네 단계다. 열 이름은 바뀌지만, 구조는 바뀌지 않는다.

### 그냥 모델에게 JSON을 내보내라고 프롬프트하면 안 되나?

"모델에게 JSON으로 답하라고 요청하기"는 함수 호출 이전의 패턴이었다. 이것은 프런티어 모델(frontier model)에서 약 5~15퍼센트 실패하고, 더 작은 모델에서는 훨씬 더 자주 실패한다. 실패 양상에는 빠진 중괄호, 끝에 붙은 쉼표(trailing comma), 환각된 필드, 잘못된 타입이 포함된다. 그러면 JSON 수리 패스, 재시도, 혹은 제약 디코더(constrained decoder)가 필요하다.

네이티브 함수 호출이 더 나은 이유는 세 가지다. 첫째, 제공자는 정확한 호출 형태로 모델을 종단 간(end-to-end)으로 학습시키므로, 엄격 모드에서 유효 JSON 비율이 98~99퍼센트까지 올라간다. 둘째, 호출 페이로드는 자유 텍스트(free-text) 안이 아니라 자신만의 프로토콜 슬롯에 들어가므로: 도구 호출이 사용자에게 보이는 응답으로 새어 나가지 않는다. 셋째, 제공자들은 제약 디코딩(constrained decoding)으로 스키마 준수를 강제한다(OpenAI의 엄격 모드, Anthropic의 `tool_use`, Gemini의 `responseSchema`). 출력은 검증을 통과하도록 보장된다.

Phase 13 · 02는 세 제공자 API를 나란히 살펴본다. Phase 13 · 04는 구조화된 출력을 깊이 다룬다.

### 회로 차단기 (Circuit breakers)

루프는 모델이 호출을 멈추거나 호스트가 최대 턴 횟수에 도달할 때 종료된다. 프로덕션 호스트는 이를 5에서 20턴 사이로 설정한다. 그 이상이면, 모델이 빠져나올 수 없는 루프에 거의 확실히 갇혀 있는 것이다. Claude Code는 기본값 20, OpenAI Assistants는 10, Cursor의 에이전트 모드는 25다.

대안(무한 루프)은 6개월마다 "에이전트가 밤새 API 호출에 400달러를 썼다"는 사후 분석(post-mortem)으로 나타난다. 한도 없이는 출시하지 마라.

Phase 14 · 12는 오류 복구와 자가 치유(self-healing)를 깊이 다루고, Phase 17은 프로덕션 속도 제한(rate limit)을 다룬다.

### Phase 13이 여기서 나아가는 방향

- 레슨 02부터 05까지는 제공자 수준의 도구 호출 표면을 다듬는다.
- 레슨 06부터 14까지는 루프를 MCP로 일반화한다.
- 레슨 15부터 18까지는 적대적인 서버, 적대적인 사용자, 인증되지 않은 원격 인증 표면으로부터 루프를 방어한다.
- 레슨 19부터 22까지는 패턴을 에이전트 간 협업, 관측 가능성(observability), 라우팅(routing), 패키징으로 확장한다.
- 레슨 23은 모든 기본 요소를 사용해 완전한 생태계를 출시한다.

남은 모든 레슨은 이 4단계 루프의 정교화다. 이것을 불변항으로 마음에 새겨라.

```figure
tp-tool-loop
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 LLM 없이 4단계 루프를 돌린다. 가짜 "decider" 함수가 사용자 메시지를 패턴 매칭하여 모델을 시뮬레이션하고, 실행기, 스키마 검증기(validator), observe 단계 하니스(harness)는 진짜다. 이를 돌려서 출력 가능한 중간 상태와 함께 전체 요청/응답 안무(choreography)를 확인한 뒤, 나중 레슨에서 가짜 decider를 실제 제공자로 교체하라.

볼 것:

- 도구 레지스트리는 도구당 네 가지 필드를 가진다. 이름, 설명, 스키마, 그리고 실행기 참조.
- 검증기는 stdlib만으로 작성된 최소한의 JSON Schema 부분집합(타입, required, enum, min/max)이다. Phase 13 · 04는 더 완전한 것을 제공한다.
- 루프는 반복 횟수를 다섯으로 제한한다. 프로덕션 에이전트는 정확히 이런 종류의 회로 차단기가 필요하다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-tool-interface-reviewer.md`를 만든다. 초안 도구 정의(이름 + 설명 + 스키마 + 실행기 개요)가 주어지면, 이 스킬은 루프 적합성(loop fitness)을 감사한다. 이름이 기계적으로 안정적인가, 설명이 완전한 사용 브리프인가, 스키마가 JSON Schema 2020-12를 올바르게 사용하는가, 순수-대-결과초래형 분류가 명시적인가.

## 연습 문제 (Exercises)

1. `code/main.py`에 `get_stock_price(ticker)`라는 네 번째 도구를 추가하라. 설명을 "사용자가 티커(ticker)로 현재 주가를 물을 때 사용하라. 과거 주가나 시장 요약에는 사용하지 마라."로 작성하라. 하니스를 돌려 가짜 decider가 티커를 언급하는 쿼리를 새 도구로 라우팅하는지 확인하라.

2. 스키마 검증기를 깨뜨려라. `arguments` 객체에 required 필드가 빠진 호출을 전달하고, 호스트가 실행 전에 그것을 거부하는지 확인하라. 그다음 알 수 없는 추가 필드를 가진 호출을 전달하라. 결정하라. 호스트는 거부해야 하는가 무시해야 하는가? 안전성 논거로 당신의 선택을 정당화하라.

3. 하니스의 각 도구를 순수 또는 결과 초래형으로 분류하라. 필요한 레지스트리 항목에 `consequential: true` 플래그를 추가하고, 결과 초래형 도구가 선택될 때마다 루프가 "would confirm with user" 줄을 출력하도록 바꿔라. 이것이 모든 프로덕션 호스트에게 필요한 확인 게이트(confirmation gate)의 형태다.

4. 4단계 루프를 종이에 그리고, 위의 제공자 열 표를 당신이 가장 좋아하는 클라이언트(Claude Desktop, Cursor, ChatGPT, 또는 커스텀 스택)에 맞게 채워라. Phase 13 · 06의 MCP 전용 변형과 교차 참조하라.

5. OpenAI의 함수 호출 가이드를 처음부터 끝까지 읽어라. 요청에는 있지만 여기 제시된 4단계 루프에는 없는 필드 하나를 식별하라. 그것이 무엇을 더하는지, 그리고 왜 필수적이라기보다 편리한 것인지 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 도구(Tool) | "모델이 호출할 수 있는 것" | 이름 + JSON Schema 타입 입력 + 실행기 함수의 세 쌍(triple) |
| 함수 호출(Function calling) | "네이티브 도구 사용" | 산문 대신 구조화된 도구 호출을 내보내기 위한 제공자 수준 API 지원 |
| 도구 호출(Tool call) | "모델의 행동 요청" | 모델이 내보내는 `id`, `name`, `arguments`를 가진 JSON 페이로드 |
| 도구 결과(Tool result) | "도구가 반환한 것" | 실행기의 출력, 일치하는 id를 가진 `tool` 역할 메시지로 감싸진 것 |
| 병렬 도구 호출(Parallel tool calls) | "한 번에 여러 호출" | 한 모델 턴 안의 여러 호출 객체, 독립적이고 id로 순서를 매길 수 있음 |
| 엄격 모드(Strict mode) | "보장된 JSON" | 모델의 출력이 선언된 스키마에 대해 검증을 통과하도록 강제하는 제약 디코딩 |
| 순수 도구(Pure tool) | "읽기 전용 도구" | 부수 효과 없음, 재실행해도 안전함 |
| 결과 초래형 도구(Consequential tool) | "행동 도구" | 외부 상태를 변경함, 게이트·감사(audit)·사용자 확인이 필요함 |
| 4단계 루프(Four-step loop) | "도구 호출 사이클" | describe → decide → execute → observe |
| 호스트(Host) | "에이전트 런타임" | 도구 레지스트리를 보유하고, 모델을 호출하고, 실행기를 돌리는 프로그램 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Function calling guide](https://platform.openai.com/docs/guides/function-calling) OpenAI 스타일 도구 선언과 호출 형태에 대한 표준 레퍼런스
- [Anthropic: Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) Claude의 `tool_use` / `tool_result` 블록 형식
- [Google: Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) Gemini의 `functionDeclarations`와 병렬 호출 의미론
- [Model Context Protocol: Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) 도구 인터페이스를 무상태로, 제공자에 매이지 않게 일반화한 현행 명세
- [JSON Schema: 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) 모든 현대 도구 API가 사용하는 스키마 방언(dialect)
