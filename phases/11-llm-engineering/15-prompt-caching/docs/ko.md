# 프롬프트 캐싱과 컨텍스트 캐싱 (Prompt Caching and Context Caching)

> 시스템 프롬프트(prompt)는 4,000토큰(token), RAG 컨텍스트는 20,000토큰이다. 모든 요청에 둘 다 보내고, 매번 둘 다에 비용을 낸다. 프롬프트 캐싱(prompt caching)은 프로바이더가 그 프리픽스를 자기 쪽에서 따뜻하게 유지하다가 재사용 시 정상 요율의 10%만 청구하게 해준다. 제대로 쓰면 추론(inference) 비용은 50–90%, 첫 토큰 지연 시간(latency)은 40–85% 줄어든다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60분

## 문제 (The Problem)

코딩 에이전트(agent)가 대화의 모든 턴마다 동일한 15,000토큰 시스템 프롬프트를 Claude에 보낸다. 입력 토큰 100만당 $3로 20턴이면, 사용자의 실제 메시지가 하나도 들어가기 전에 입력 비용만 $0.90다. 하루 10,000개의 대화를 곱하면, 절대 변하지 않는 텍스트에 대해 청구서가 하루 $9,000에 도달한다.

품질을 해치지 않고는 프롬프트를 줄일 수 없다. 모델은 모든 턴에 그것이 필요하므로 보내지 않을 수도 없다. 유일한 수는 프로바이더가 이미 본 프리픽스에 정가를 매기는 것을 멈추게 하는 것이다.

그 수가 프롬프트 캐싱이다. Anthropic이 2024년 8월에 출시했고(2025년에 1시간 확장 TTL 변형을 추가했다), OpenAI가 그해 후반에 자동화했으며, Google이 Gemini 1.5와 함께 명시적 컨텍스트 캐싱을 내놓았다. 이제 세 곳 모두 자신들의 프런티어 모델에서 이를 일급(first-class) 기능으로 제공한다.

## 개념 (The Concept)

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**메커니즘.** 요청의 프리픽스가 최근 요청의 것과 일치하면, 프로바이더는 토큰을 다시 인코딩하는 대신 이전 실행의 KV 캐시를 제공한다. 첫 번째에 작은 쓰기 프리미엄을 내고, 그 이후 매번 큰 읽기 할인을 받는다.

**2026년의 세 가지 프로바이더 방식.**

| 프로바이더 | API 스타일 | 히트 할인 | 쓰기 프리미엄 | 기본 TTL | 최소 캐시 가능 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 콘텐츠 블록의 명시적 `cache_control` 마커 | 입력 90% 할인 | 25% 추가 요금 | 5분 (1시간까지 확장 가능) | 1,024 토큰 (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | 자동 프리픽스 탐지 | 입력 50% 할인 | 없음 | 최대 1시간 (최선 노력) | 1,024 토큰 |
| Google (Gemini) | 명시적 `CachedContent` API | 저장소 과금; 정상의 ~25%로 읽기 | 토큰·시간당 저장 요금 | 사용자 설정 (기본 1시간) | 4,096 토큰 (Flash), 32,768 (Pro) |

**불변 규칙.** 셋 모두 프리픽스만 캐시한다. 요청 간에 토큰이 하나라도 다르면, 처음 달라진 토큰 이후의 모든 것이 미스다. *안정적인* 부분을 위에, *가변적인* 부분을 아래에 두라.

### 캐시 친화적 레이아웃 (The cache-friendly layout)

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

순서를 어기면(사용자 메시지를 시스템 프롬프트 위에 두거나, 동적 검색을 few-shot 사이에 끼워넣으면) 캐시가 절대 히트하지 않는다.

### 손익분기 계산 (The break-even calculation)

Anthropic의 25% 쓰기 프리미엄은, 캐시된 블록이 순(net) 절약에 이르려면 최소 두 번은 읽혀야 함을 뜻한다. 쓰기 1 + 읽기 1은 요청당 평균 0.675배 비용이다(32% 절약). 쓰기 1 + 읽기 10은 평균 0.205배다(80% 절약). 경험칙은 이렇다. TTL 내에 최소 3회 재사용할 것으로 예상되는 것은 무엇이든 캐시하라.

```figure
prompt-cache-hit
```

## 직접 만들기 (Build It)

### 1단계: 명시적 마커를 사용한 Anthropic 프롬프트 캐싱

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 마커는 Anthropic에게 그 블록을 5분간 저장하라고 알린다. 그 윈도우 안의 재사용은 히트하고, 만료 후의 재사용은 다시 쓴다.

**응답 usage 필드:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

CI에서 두 필드를 모두 확인하라. `cache_read_input_tokens`가 요청 전반에서 0에 머무르면, 캐시 키가 드리프트하고 있는 것이다.

### 2단계: 1시간 확장 TTL

장시간 실행되는 배치(batch) 작업에서는 5분 기본값이 작업 사이에 만료된다. `ttl`을 설정하라:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1시간 TTL은 쓰기 프리미엄이 2배(25% 대신 베이스라인 대비 50%) 들지만, 프리픽스를 5회 이상 재사용하는 배치라면 금세 본전을 뽑는다.

### 3단계: OpenAI 자동 캐싱

OpenAI는 설정할 거리를 주지 않는다. 최근 요청과 일치하는 1,024토큰 이상의 프리픽스는 무엇이든 자동으로 50% 할인을 받는다.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

동일한 캐시 친화적 레이아웃 규칙이 적용된다. Anthropic의 캐시는 죽이지 않지만 OpenAI의 캐시는 죽이는 것이 두 가지다. `user` 필드 변경(캐시 키 구성 요소로 쓰인다)과 도구 재정렬이다.

### 4단계: Gemini 명시적 컨텍스트 캐싱

Gemini는 캐시를 직접 생성하고 이름 붙이는 일급 객체로 다룬다:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini는 캐시가 살아 있는 동안 토큰·시간당 저장소 비용을 청구하고, 정상 입력 요율의 ~25%로 읽는다. 여러 날에 걸쳐 많은 세션에서 같은 거대한 프롬프트를 재사용할 때 맞는 형태다.

### 5단계: 프로덕션에서 히트율 측정하기

쓰기·읽기·미스 횟수를 추적하고 1K 요청당 혼합 비용을 계산하는, 시뮬레이션된 3-프로바이더 회계기는 `code/main.py`를 보라. 목표 히트율로 배포를 게이팅하라. 대부분의 프로덕션 Anthropic 설정은 워밍업 후 80%를 넘는 읽기 비율을 봐야 한다.

## 2026년에도 여전히 출시되는 함정 (Pitfalls that still ship in 2026)

- **상단의 동적 타임스탬프.** 시스템 프롬프트 상단의 `"Current time: 2026-04-22 15:30:02"`. 모든 요청이 미스다. 타임스탬프를 캐시 브레이크포인트 아래로 옮겨라.
- **도구 재정렬.** 도구를 안정적인 순서로 직렬화하라. 배포 사이의 딕셔너리 재배치는 모든 히트를 깨뜨린다.
- **자유 텍스트 근접 중복.** "You are helpful." 대 "You are a helpful assistant.": 1바이트 차이가 완전한 미스다.
- **너무 작은 블록.** Anthropic은 1,024토큰 하한(Haiku는 2,048)을 강제한다. 더 작은 블록은 조용히 캐시되지 않는다.
- **눈먼 비용 대시보드.** "입력 토큰"을 캐시됨과 캐시 안 됨으로 분리하라. 그러지 않으면 트래픽 감소가 캐시 승리처럼 보인다.

## 라이브러리로 써보기 (Use It)

2026 캐싱 스택:

| 상황 | 선택 |
|-----------|------|
| 안정적인 10k+ 시스템 프롬프트와 많은 턴을 가진 에이전트 | 5분 TTL을 가진 Anthropic `cache_control` |
| 프리픽스를 30분 이상 재사용하는 배치 작업 | `ttl: "1h"`를 가진 Anthropic |
| 커스텀 인프라 없는 GPT-5의 서버리스 엔드포인트 | OpenAI 자동 (그냥 프리픽스를 안정적이고 길게 만들라) |
| 거대한 코드/문서 코퍼스의 여러 날 재사용 | Gemini 명시적 `CachedContent` |
| 크로스 프로바이더 폴백 | 어떤 히트든 동작하도록 프로바이더 전반에 캐시 가능한 프리픽스 레이아웃을 동일하게 유지하라 |

사용자 메시지 계층에는 시맨틱 캐싱(Phase 11 · 11)을 결합하라. 프롬프트 캐싱은 *토큰 동일* 재사용을, 시맨틱 캐싱은 *의미 동일* 재사용을 처리한다.

## 산출물 (Ship It)

`outputs/skill-prompt-caching-planner.md`를 저장한다:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 연습 문제 (Exercises)

1. **쉬움.** 5,000토큰 시스템 프롬프트를 가진 10턴 대화를 Claude에 대해 가져오라. `cache_control` 없이 실행한 다음 그것과 함께 실행하라. 각각의 입력 토큰 청구서를 보고하라.
2. **중간.** 프롬프트 템플릿과 요청 로그가 주어지면, 프로바이더별(Anthropic 5분, Anthropic 1시간, OpenAI 자동, Gemini 명시적) 예상 히트율과 달러 절감을 계산하는 테스트 하니스를 작성하라.
3. **어려움.** 레이아웃 옵티마이저를 구축하라: 프롬프트와 `stable=True/False`로 표시된 필드 목록이 주어지면, 정보를 잃지 않으면서 단일 캐시 브레이크포인트를 최대로 캐시 친화적인 위치에 두도록 프롬프트를 재작성하라. 실제 Anthropic 엔드포인트에서 검증하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 프롬프트 캐싱(Prompt caching) | "긴 프롬프트를 싸게 만든다" | 일치하는 프리픽스에 대해 프로바이더 측 KV 캐시를 재사용하는 것; 반복되는 입력 토큰에 50-90% 할인 |
| `cache_control` | "Anthropic 마커" | "여기까지의 모든 것은 캐시 가능하다"를 선언하는 콘텐츠 블록 속성; `{"type": "ephemeral"}` |
| 캐시 쓰기(Cache write) | "프리미엄 지불" | 캐시를 채우는 첫 요청; Anthropic에서 ~1.25배 입력 요율로 과금, OpenAI에서 무료 |
| 캐시 읽기(Cache read) | "할인" | 프리픽스와 일치하는 후속 요청; 10%(Anthropic), 50%(OpenAI), ~25%(Gemini)로 과금 |
| TTL | "얼마나 오래 사는지" | 캐시가 따뜻하게 유지되는 초; Anthropic 5분 기본(1시간 확장 가능), OpenAI 최선 노력으로 최대 1시간, Gemini 사용자 설정 |
| 확장 TTL(Extended TTL) | "1시간 Anthropic 캐시" | `{"type": "ephemeral", "ttl": "1h"}`; 쓰기 프리미엄 2배지만 배치 재사용에는 가치 있음 |
| 프리픽스 매치(Prefix match) | "내 캐시가 미스한 이유" | 시작부터 브레이크포인트까지의 모든 토큰이 바이트 동일할 때만 캐시가 히트한다 |
| 컨텍스트 캐싱 (Gemini) (Context caching) | "명시적인 것" | Google의 이름 붙은, 저장소 과금 캐시 객체; 큰 코퍼스의 여러 날 재사용에 가장 적합 |

## 더 읽을거리 (Further Reading)

- [Anthropic: Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) `cache_control`, 1시간 TTL, 손익분기 표
- [OpenAI: Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) 자동 프리픽스 매칭
- [Google: Context caching](https://ai.google.dev/gemini-api/docs/caching) `CachedContent` API와 저장소 가격
- [Anthropic engineering: Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) 지연 시간 수치가 담긴 원래 출시 게시물
- Phase 11 · 05 (Context Engineering): 캐시가 안착할 수 있도록 프롬프트를 어디서 자를지
- Phase 11 · 11 (Caching and Cost): 프롬프트 캐싱을 사용자 메시지에 대한 시맨틱 캐시와 짝지으라
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102): 프롬프트 캐싱이 사용자에게 노출하는 KV 캐시 메모리 모델; 캐시된 프리픽스를 다시 읽는 것이 다시 계산하는 것보다 ~10배 싼 이유를 설명한다
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369): 프리필은 프롬프트 캐싱이 단축하는 단계다; 이 논문은 캐시 히트 시 TTFT가 극적으로 떨어지는 반면 TPOT는 영향받지 않는 이유를 설명한다
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192): 프롬프트 캐싱은 추론 비용 곡선을 굽히는 레버로서 추측적 디코딩, Flash Attention, MQA/GQA와 나란히 위치한다; 나머지 셋은 이것을 읽으라
