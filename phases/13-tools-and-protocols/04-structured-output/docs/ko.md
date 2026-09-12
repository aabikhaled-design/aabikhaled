# 구조화된 출력: JSON Schema, Pydantic, Zod, 제약 디코딩

> "모델에게 JSON을 반환해 달라고 정중히 부탁하기"는 프런티어 모델(frontier model)에서도 5~15퍼센트 실패한다. 구조화된 출력(structured output)은 제약 디코딩(constrained decoding)으로 그 간극을 메운다. 스키마를 위반하는 토큰은 애초에 모델이 내보내지 못하도록 막힌다. OpenAI의 엄격 모드(strict mode), Anthropic의 스키마 타입 도구 사용, Gemini의 `responseSchema`, Pydantic AI의 `output_type`, Zod의 `.parse`는 같은 아이디어의 다섯 가지 표면 형태다. 이 레슨에서는 모든 프로덕션 추출 파이프라인(extraction pipeline)에 쓸 스키마 검증기(validator)와 엄격 모드 계약을 직접 만든다.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 올바른 제약(enum, min/max, required, pattern)을 사용해 추출 대상의 JSON Schema 2020-12를 작성하기.
- 엄격 모드와 제약 디코딩이 "생성 후 검증"과 다른 보장을 주는 이유를 설명하기.
- 세 가지 실패 양상(파싱 오류, 스키마 위반, 모델 거부(refusal))을 구분하기.
- 타입이 지정된 수리와 타입이 지정된 거부 처리를 갖춘 추출 파이프라인을 출시하기.

## 문제 (The Problem)

발주 이메일을 읽는 에이전트는 자유 텍스트를 `{customer, line_items, total_usd}`로 바꿔야 한다. 세 가지 접근법이 있다.

**접근법 1: JSON을 프롬프트한다.** "customer, line_items, total_usd 필드를 가진 JSON으로 답하라." 프런티어 모델에서 85~95퍼센트 동작한다. 실패하는 방식은 여섯 가지다. 빠진 중괄호, 끝에 붙은 쉼표(trailing comma), 잘못된 타입, 환각된 필드, 토큰 한도에서 잘림, "Here is your JSON:" 같은 새어 나온 산문.

**접근법 2: 생성 후 검증한다.** 자유롭게 생성하고 파싱한 뒤 스키마에 대해 검증하고, 실패하면 재시도한다. 신뢰할 수 있지만 비싸다. 재시도마다 비용을 치르고, 잘림(truncation) 버그는 발생할 때마다 한 턴(turn)씩 더 잡아먹는다.

**접근법 3: 제약 디코딩.** 제공자가 디코드 시점에 스키마를 강제한다. 유효하지 않은 토큰은 샘플링 분포(sampling distribution)에서 마스킹된다. 출력은 파싱과 검증이 보장된다. 실패는 한 양상으로 좁혀진다. 입력이 스키마에 맞지 않는다고 모델이 판단하는 거부(refusal)다.

2026년의 모든 프런티어 제공자는 접근법 3의 어떤 형태를 제공한다.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`에 더해, 모델이 거절하면 응답에 `refusal`이 붙는다.
- **Anthropic.** `tool_use` 입력에 대한 스키마 강제. `stop_reason: "refusal"`은 없지만, 도구 호출 없는 `end_turn`이 그 신호다.
- **Gemini.** 요청 수준의 `responseSchema`. 2026년 Gemini는 선택된 타입에 토큰 수준 문법 제약(grammar constraint)을 제공한다.
- **Pydantic AI.** `output_type=InvoiceModel`은 `InvoiceModel`로 타입이 지정된 구조화된 `RunResult`를 내보낸다.
- **Zod (TypeScript).** 제공자 출력을 Zod 스키마에 대해 검증하는 런타임 파서. OpenAI의 `beta.chat.completions.parse`와 짝을 이룬다.

공통된 맥락은 이렇다. 스키마를 한 번 선언하고, 종단 간(end to end)으로 강제하라.

## 개념 (The Concept)

### JSON Schema 2020-12: 공통어(lingua franca)

모든 제공자가 JSON Schema 2020-12를 받아들인다. 가장 많이 쓰는 구문은 다음과 같다.

- `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` 중 하나.
- `properties`: 필드 이름에서 하위 스키마(subschema)로의 맵.
- `required`: 반드시 나타나야 하는 필드 이름의 목록.
- `enum`: 허용되는 값의 닫힌 집합.
- `minimum` / `maximum`(숫자), `minLength` / `maxLength` / `pattern`(문자열).
- `items`: 모든 배열 요소에 적용되는 하위 스키마.
- `additionalProperties`: `false`는 추가 필드를 금지한다(기본값은 모드에 따라 다르다).

OpenAI 엄격 모드는 여기에 세 가지 요구 사항을 더한다. 모든 속성이 `required`에 나열되어야 하고, 어디서나 `additionalProperties: false`여야 하며, 해결되지 않은 `$ref`가 없어야 한다. 이를 어기면 API가 요청 시점에 400을 반환한다.

### Pydantic, Python 바인딩

Pydantic v2는 `model_json_schema()`로 데이터클래스 형태 모델에서 JSON Schema를 생성한다. Pydantic AI가 이를 감싸므로 다음과 같이 작성하면:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

에이전트 프레임워크가 가장자리(edge)에서 스키마를 OpenAI 엄격 모드, Anthropic `input_schema`, 또는 Gemini `responseSchema`로 번역한다. 모델의 출력은 타입이 지정된 `Invoice` 인스턴스로 돌아온다. 검증 오류는 타입이 지정된 오류 경로와 함께 `ValidationError`를 일으킨다.

### Zod, TypeScript 바인딩

Zod(`z.object({customer: z.string(), ...})`)는 TS 등가물이다. OpenAI의 Node SDK는 API의 JSON Schema 페이로드로 번역되는 `zodResponseFormat(Invoice)`를 노출한다.

### 거부 (Refusals)

엄격 모드라고 해서 모델이 답하도록 강제하지는 못한다. 입력이 스키마에 맞지 않으면("이메일은 송장이 아니라 시였다"), 모델은 이유를 담은 `refusal` 필드를 내보낸다. 코드는 이를 실패가 아니라 일급(first-class) 결과로 처리해야 한다. 거부는 안전 신호로도 유용하다. 보호된 콘텐츠 이메일에서 신용카드 번호를 추출하라는 요청을 받은 모델은 안전 이유가 첨부된 거부를 반환한다.

### 공개 환경에서의 제약 디코딩

오픈 웨이트(open-weights) 구현은 세 가지 기법을 쓴다.

1. **문법 기반 디코딩**(`outlines`, `guidance`, `lm-format-enforcer`): 스키마에서 결정론적 유한 오토마톤(deterministic finite automaton)을 만든다. 매 단계마다 FSM을 위반할 토큰의 로짓(logit)을 마스킹한다.
2. **JSON 파서를 사용한 로짓 마스킹**: 모델과 보조를 맞춰 스트리밍 JSON 파서를 돌린다. 매 단계마다 유효한 다음 토큰 집합을 계산한다.
3. **검증기를 사용한 추측 디코딩(speculative decoding)**: 저렴한 초안 모델이 토큰을 제안하고, 검증기가 스키마를 강제한다.

상업적 제공자는 이 중 하나를 막후에서 고른다. 2026년의 최첨단은 짧은 구조화된 출력에서는 평범한 생성보다 빠르고, 긴 출력에서는 대략 같은 속도다.

### 세 가지 실패 양상

1. **파싱 오류(Parse error).** 출력이 유효한 JSON이 아니다. 엄격 모드에서는 일어날 수 없다. 비엄격 제공자에서는 여전히 일어난다.
2. **스키마 위반(Schema violation).** 출력은 파싱되지만 스키마를 위반한다. 엄격 모드에서는 일어날 수 없다. 그 밖에서는 흔하다.
3. **거부(Refusal).** 모델이 거절한다. 타입이 지정된 결과로 처리해야 한다.

### 재시도 전략 (Retry strategy)

엄격 모드 밖에 있을 때(Anthropic 도구 사용, 비엄격 OpenAI, 구버전 Gemini), 복구 패턴은 다음과 같다.

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

한 번의 재시도면 보통 충분하다. 세 번의 재시도는 약한 모델의 변덕(flake)을 잡아낸다. 셋을 넘으면 스키마가 나쁘다는 신호다. 모델이 일부 입력에 대해 스키마를 만족시키지 못하니, 프롬프트나 스키마를 고쳐야 한다.

### 소형 모델 지원

제약 디코딩은 소형 모델에서도 동작한다. 문법 강제를 갖춘 30억 파라미터 오픈 모델은 구조화된 작업에서 원시 프롬프팅(raw prompting)을 쓴 700억 파라미터 모델을 능가한다. 구조화된 출력이 프로덕션에 중요한 주된 이유가 이것이다. 신뢰성을 모델 크기에서 떼어내기 때문이다.

```figure
constrained-decoding
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 stdlib로 최소한의 JSON Schema 2020-12 검증기(타입, required, enum, min/max, pattern, items, additionalProperties)를 제공한다. `Invoice` 스키마를 감싸고 가짜 LLM 출력을 검증기에 통과시켜, 파싱 오류, 스키마 위반, 거부 경로를 보여준다. 프로덕션에서는 가짜 출력을 어떤 제공자의 실제 응답으로 교체하라.

볼 것:

- 검증기는 경로와 메시지를 가진 타입 지정 `[ValidationError]` 목록을 반환한다. 재시도 프롬프트에 드러내고 싶은 형태가 바로 그것이다.
- 거부 분기는 재시도하지 않는다. 로깅하고 타입이 지정된 거부를 반환한다. Phase 14 · 09는 거부를 안전 신호로 사용한다.
- `additionalProperties: false` 검사는 적대적 테스트 입력에서 발화하여, 엄격 모드가 왜 환각된 필드에 문을 닫는지 보여준다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-structured-output-designer.md`를 만든다. 자유 텍스트 추출 대상(송장, 지원 티켓, 이력서 등)이 주어지면, 이 스킬은 엄격 모드 호환 JSON Schema 2020-12와 그것을 반영하는 Pydantic 모델을, 타입이 지정된 거부 및 재시도 처리가 스텁(stub)으로 들어간 채로 만든다.

## 연습 문제 (Exercises)

1. `code/main.py`를 돌려라. `total_usd`가 음수인 네 번째 테스트 케이스를 추가하라. 검증기가 `minimum` 제약 경로로 그것을 거부하는지 확인하라.

2. 판별자(discriminator)를 가진 `oneOf`를 지원하도록 검증기를 확장하라. 흔한 사례: `line_item`은 `kind`로 태그된 제품 또는 서비스 중 하나다. 엄격 모드는 여기서 미묘한 규칙을 가진다. OpenAI의 구조화된 출력 가이드를 확인하라.

3. 같은 Invoice 스키마를 Pydantic BaseModel로 작성하고 `model_json_schema()` 출력을 손으로 만든 스키마와 비교하라. Pydantic이 기본적으로 설정하지만 손으로 만든 버전이 생략하는 필드 하나를 식별하라.

4. 거부율을 측정하라. 추출 불가능해야 하는 입력 열 개(노래 가사, 수학 증명, 빈 이메일)를 구성하고 엄격 모드를 갖춘 실제 제공자를 통해 돌려라. 거부와 환각된 출력을 세어라. 이것이 거부 인식 재시도(refusal-aware retry)를 위한 정답(ground truth)이다.

5. OpenAI의 구조화된 출력 가이드를 처음부터 끝까지 읽어라. 평범한 JSON Schema는 허용하지만 엄격 모드가 명시적으로 금지하는 구문 하나를 식별하라. 그다음 금지된 구문을 비필수적으로 사용하는 스키마를 설계하고 엄격 호환이 되도록 리팩터링하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "스키마 명세" | 모든 현대 제공자가 사용하는 IETF 초안 스키마 방언(dialect) |
| 엄격 모드(Strict mode) | "보장된 스키마" | 제약 디코딩으로 스키마를 강제하는 OpenAI 플래그 |
| 제약 디코딩(Constrained decoding) | "로짓 마스킹" | 유효하지 않은 다음 토큰을 마스킹하는 디코드 시점 강제 |
| 거부(Refusal) | "모델이 거절" | 입력이 스키마에 맞지 않을 때의 타입 지정 결과 |
| 파싱 오류(Parse error) | "유효하지 않은 JSON" | 출력이 JSON으로 파싱되지 않음. 엄격 모드에서는 불가능 |
| 스키마 위반(Schema violation) | "잘못된 형태" | 파싱은 되지만 타입 / required / enum / 범위를 위반 |
| `additionalProperties: false` | "추가 금지" | 알 수 없는 필드를 금지. OpenAI 엄격 모드에서 필수 |
| Pydantic BaseModel | "타입 지정 출력" | JSON Schema를 내보내고 검증하는 Python 클래스 |
| Zod 스키마 | "TypeScript 출력 타입" | 제공자 출력 검증을 위한 TS 런타임 스키마 |
| 문법 강제(Grammar enforcement) | "오픈 웨이트 제약 디코드" | outlines / guidance에서처럼 FSM 기반 로짓 마스킹 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) 엄격 모드, 거부, 스키마 요구 사항
- [OpenAI: Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) 디코딩 보장을 설명하는 2024년 8월 출시 게시물
- [Pydantic AI: Output](https://ai.pydantic.dev/output/) 각 제공자로 직렬화되는 타입 지정 output_type 바인딩
- [JSON Schema: 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) 표준 명세
- [Microsoft: Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) 엔터프라이즈 배포 노트와 엄격 모드 주의 사항
