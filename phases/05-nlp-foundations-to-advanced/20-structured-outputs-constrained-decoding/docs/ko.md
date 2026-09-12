# Structured Outputs & Constrained Decoding

> LLM에게 JSON을 요청하라. 대부분의 경우 JSON을 얻는다. 프로덕션(production)에서는 "대부분"이 문제다. 제약 디코딩(constrained decoding)은 샘플링 전에 로짓(logit)을 편집하여 "대부분"을 "항상"으로 바꾼다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60분

## 문제 (The Problem)

분류기(classifier)가 LLM에 프롬프트(prompt)한다: "Return one of {positive, negative, neutral}." 모델은 "The sentiment is positive: this review is overwhelmingly favorable because the customer explicitly states that they ..."를 반환한다. 파서(parser)가 충돌한다. 분류기의 F1이 0.0이다.

자유 형식 생성은 계약이 아니다. 제안이다. 프로덕션 시스템은 계약이 필요하다.

2026년에는 세 개의 계층이 존재한다.

1. **프롬프팅(Prompting).** 정중하게 요청한다. "Return only the JSON object." 프런티어 모델에서 약 80% 작동하고, 더 작은 모델에서는 덜 작동한다.
2. **네이티브 구조화 출력 API(Native structured output APIs).** OpenAI `response_format`, Anthropic 도구 사용, Gemini JSON 모드. 지원되는 스키마(schema)에서 신뢰할 만하다. 벤더 종속적이다.
3. **제약 디코딩(Constrained decoding).** 모든 생성 단계에서 로짓을 수정하여 모델이 유효하지 않은 토큰(token)을 *내보낼 수 없게* 한다. 구조상 100% 유효하다. 어떤 로컬 모델에서도 작동한다.

이 레슨은 셋 모두에 대한 직관을 쌓고 언제 어느 것에 손을 뻗을지 명명한다.

## 개념 (The Concept)

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**제약 디코딩이 작동하는 방식.** 각 생성 단계에서 LLM은 전체 어휘(~100k 토큰)에 대한 로짓 벡터(vector)를 만든다. *로짓 프로세서(logit processor)*가 모델과 샘플러(sampler) 사이에 자리한다. 로짓 프로세서는 대상 문법(JSON Schema, 정규식, 문맥 자유 문법) 내 현재 위치에서 어떤 토큰이 유효한지 계산하고, 유효하지 않은 모든 토큰의 로짓을 음의 무한대로 설정한다. 남은 로짓에 대한 소프트맥스(softmax)는 유효한 연속(continuation)에만 확률 질량을 둔다.

2026년의 구현:

- **Outlines.** JSON Schema나 정규식을 유한 상태 기계(finite-state machine)로 컴파일한다. 모든 토큰이 O(1) 유효 다음 토큰 조회를 받는다. FSM 기반이므로 재귀 스키마는 평탄화(flattening)가 필요하다.
- **XGrammar / llguidance.** 문맥 자유 문법(context-free grammar) 엔진. 재귀 JSON Schema를 다룬다. 거의 0에 가까운 디코딩 오버헤드. OpenAI는 2025년 구조화 출력 구현에서 llguidance의 공을 인정했다.
- **vLLM 가이드 디코딩.** Outlines, XGrammar, 또는 lm-format-enforcer 백엔드를 통한 내장 `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`.
- **Instructor.** 임의의 LLM에 대한 Pydantic 기반 래퍼(wrapper). 검증 실패 시 재시도한다. 교차 프로바이더지만, 로짓을 수정하지 않는다. 재시도 + 구조화 출력 인식 프롬프트에 의존한다.

### 직관에 반하는 결과

제약 디코딩은 종종 제약 없는 생성보다 *더 빠르다*. 두 가지 이유. 첫째, 다음 토큰 탐색 공간을 줄인다. 둘째, 영리한 구현은 강제된 토큰의 경우 토큰 생성을 완전히 건너뛴다(`{"name": "` 같은 골격(scaffolding): 모든 바이트가 결정되어 있다).

### 당신에게 비용을 치르게 하는 함정

필드 순서가 중요하다. `answer`를 `reasoning` 앞에 두면, 모델은 생각하기 전에 답에 전념한다. JSON은 유효하다. 답은 틀리다. 어떤 검증도 이를 잡지 못한다.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

스키마 필드 순서는 서식이 아니라 논리다.

```figure
constrained-decoder
```

## 직접 만들기 (Build It)

### Step 1: 밑바닥부터 만드는 정규식 제약 생성

독립 실행형 FSM 구현은 `code/main.py`를 보라. 30줄로 표현한 핵심 아이디어:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM은 지금까지 우리가 문법의 어떤 부분을 충족했는지 추적한다. `valid_tokens(state, tokenizer)`는 수용(accepting) 경로를 벗어나지 않으면서 FSM을 전진시킬 수 있는 어휘 토큰을 계산한다.

### Step 2: JSON Schema를 위한 Outlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

검증 오류 0개. 언제나. FSM이 유효하지 않은 출력을 도달 불가능하게 만든다.

### Step 3: 프로바이더 무관 Pydantic을 위한 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

다른 메커니즘이다. Instructor는 로짓을 건드리지 않는다. 스키마를 프롬프트로 서식화하고, 출력을 파싱하며, 검증 실패 시 재시도한다(기본 3회). 어떤 프로바이더와도 작동한다. 재시도는 지연 시간(latency)과 비용을 더한다. 교차 프로바이더 이식성이 장점이다.

### Step 4: 네이티브 벤더 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

서버 측 제약 디코딩이다. 지원되는 스키마에 대해 Outlines와 동등한 신뢰성. 로컬 모델 관리가 없다. 벤더에 종속된다.

## 함정 (Pitfalls)

- **재귀 스키마(Recursive schemas).** Outlines는 재귀를 고정 깊이로 평탄화한다. 트리 구조 출력(중첩 댓글, AST)에는 XGrammar나 llguidance(CFG 기반)가 필요하다.
- **거대한 enum.** 10,000개 옵션 enum은 느리게 컴파일되거나 타임아웃된다. 검색기로 전환하라: 먼저 상위 k개 후보를 예측하고, 그것들로 제약하라.
- **너무 엄격한 문법.** `date: "YYYY-MM-DD"` 정규식을 강제하면 모델이 누락된 날짜에 대해 `"unknown"`을 출력할 수 없다. 그러면 모델은 날짜를 지어내어 보상한다. `null`이나 센티넬(sentinel)을 허용하라.
- **성급한 전념(Premature commitment).** 위의 필드 순서 함정을 보라. 항상 추론을 먼저 두라.
- **스키마 없는 벤더 JSON 모드.** 순수 JSON 모드는 유효한 JSON만 보장하지, *사용 사례에 유효함*을 보장하지 않는다. 항상 전체 스키마를 제공하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| OpenAI/Anthropic/Google 모델, 간단한 스키마 | 네이티브 벤더 구조화 출력 |
| 임의의 프로바이더, Pydantic 워크플로, 재시도 허용 | Instructor |
| 로컬 모델, 100% 유효성 필요, 평탄한 스키마 | Outlines (FSM) |
| 로컬 모델, 재귀 스키마 | XGrammar 또는 llguidance |
| 자체 호스팅 추론 서버 | vLLM 가이드 디코딩 |
| 재시도가 허용되는 배치 처리 | Instructor + 가장 저렴한 모델 |

## 산출물 (Ship It)

`outputs/skill-structured-output-picker.md`로 저장하라:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 연습 문제 (Exercises)

1. **Easy.** 작은 오픈 가중치(open-weights) 모델(예: Llama-3.2-3B)을 제약 디코딩 없이 `Review(sentiment, confidence, evidence_span)`에 프롬프트하라. 100개 리뷰에서 유효한 JSON으로 파싱되는 비율을 측정하라.
2. **Medium.** 같은 코퍼스(corpus)를 Outlines JSON 모드로 처리하라. 준수율, 지연 시간, 의미적 정확도를 비교하라.
3. **Hard.** 전화번호(`\d{3}-\d{3}-\d{4}`)를 위한 정규식 제약 디코더를 밑바닥부터 구현하라. 1000개 샘플에서 유효하지 않은 출력이 0개임을 검증하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Constrained decoding | 유효한 출력 강제 | 모든 생성 단계에서 유효하지 않은 토큰의 로짓을 마스킹. |
| Logit processor | 제약을 거는 것 | 함수: `(logits, state) -> masked_logits`. |
| FSM | 유한 상태 기계 | 컴파일된 문법 표현. O(1) 유효 다음 토큰 조회. |
| CFG | 문맥 자유 문법 | 재귀를 다루는 문법. FSM보다 느리지만 더 표현력이 있다. |
| Schema field order | 중요한가? | 그렇다. 첫 필드가 전념한다. 항상 추론을 답 앞에 두라. |
| Guided decoding | vLLM이 부르는 이름 | 같은 개념, 추론 서버에 통합됨. |
| JSON mode | OpenAI의 초기 버전 | JSON 구문을 보장. 스키마 일치는 보장하지 *않음*. |

## 더 읽을거리 (Further Reading)

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702): Outlines 논문.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100): 빠른 CFG 기반 제약 디코딩.
- [vLLM: Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) 추론 서버 통합.
- [OpenAI: Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) API 레퍼런스 + 주의점.
- [Instructor library](https://python.useinstructor.com/): 프로바이더 전반에 걸친 Pydantic + 재시도.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868): 6개 제약 디코딩 프레임워크 벤치마킹.
