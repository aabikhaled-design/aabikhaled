# 프롬프트 캐싱과 시맨틱 캐싱 경제학(Prompt Caching and Semantic Caching Economics)

> **2026-04 기준 가격 스냅샷.** 아래의 수치 주장들은 이 레슨 발행 시점에 포착한 벤더 요금표를 반영한다. 하류로 인용하기 전에 링크된 문서와 대조하여 검증하라.

> 캐싱은 두 계층에서 일어난다. L2(프로바이더 수준) 프롬프트/프리픽스 캐싱은 반복되는 프리픽스(prefix)에 대해 어텐션(attention) KV를 재사용한다. Anthropic의 프롬프트 캐싱 문서는 긴 프롬프트에서 최대 90% 비용 절감과 85% 지연 시간(latency) 절감을 광고한다. Claude 3.5 Sonnet의 경우 캐시 읽기는 신규 $3.00/M 대비 $0.30/M이며, 5분 TTL과 1시간 TTL 옵션의 2배 쓰기 프리미엄이 적용된다(docs.anthropic.com, 2026-04). OpenAI 프롬프트 캐싱은 ≥1024 토큰 프롬프트에 자동으로 적용되며 캐시된 입력을 신규 대비 대략 90% 할인으로 가격 책정한다(platform.openai.com, 2026-04). 정확한 모델별 캐시 요율은 라이브 요금표에 따라 다르다. L1(앱 수준) 시맨틱 캐싱은 임베딩(embedding) 유사도 적중 시 LLM을 전적으로 건너뛴다. 벤더의 "95% 정확도"는 적중률이 아니라 매치 정확성을 가리킨다. 보고된 프로덕션 적중률은 10%(개방형 채팅)부터 70%(구조화된 FAQ)까지 분포한다. 어느 프로바이더도 공식 베이스라인을 공개하지 않으므로, 이것들을 보장이 아니라 커뮤니티 텔레메트리로 취급하라. 프로덕션 함정들: 병렬화(parallelization)는 캐싱을 죽이고(첫 캐시 쓰기 전에 발행된 N개 병렬 요청이 지출을 몇 배로 부풀릴 수 있음), 프리픽스 내부의 동적 콘텐츠는 캐시 적중을 전적으로 막는다. ProjectDiscovery는 동적 텍스트를 캐시 가능 프리픽스 밖으로 옮겨 7%에서 74% 적중률로 이동했다고 보고했다(2025-11).

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- L2 프롬프트/프리픽스 캐싱(프로바이더에서의 KV 재사용)과 L1 시맨틱 캐싱(유사 프롬프트에서의 LLM 우회)을 구별하기.
- Anthropic의 `cache_control` 명시적 마킹과 두 TTL 옵션(5분 대 1시간), 그리고 그 가격 배수를 설명하기.
- 적중률, 프롬프트/응답 구성, 토큰 가격이 주어졌을 때 예상 월간 절감액을 계산하기.
- 청구액을 5-10배 부풀리는 병렬화 안티패턴과 적중률을 무너뜨리는 동적 콘텐츠 안티패턴을 짚기.

## 문제 (The Problem)

RAG 서비스에 프롬프트 캐싱을 추가했다고 하자. 청구액은 그대로다. 적중률을 측정하니 7%다. 프롬프트가 정적으로 보이지만 실제로는 그렇지 않다. 시스템 프롬프트에 분 단위로 포맷된 현재 날짜, 요청 ID, 다양성을 위해 무작위로 재배열된 예시가 들어 있기 때문이다. 모든 요청이 새 캐시 엔트리를 쓰고, 읽기는 0이다.

별개로, 에이전트가 사용자 질문당 열 개의 병렬 도구 호출을 돌린다. 첫 캐시 쓰기가 완료되기 전에 열 개 모두 프로바이더에 도착한다. 열 번의 쓰기, 0번의 읽기. 청구액은 "캐싱 적용" 시 들었어야 할 비용의 5-10배가 된다.

캐싱은 플래그가 아니라 프로토콜이다. 두 계층, 두 가지 다른 실패 모드.

## 개념 (The Concept)

### L2: 프로바이더 프롬프트/프리픽스 캐싱

프로바이더가 캐시 가능 프리픽스에 대한 어텐션 KV를 저장하고, 프리픽스가 일치하는 다음 요청에서 재사용한다. 쓰기 비용을 한 번 지불하고, 읽기는 거의 무료다.

**Anthropic (Claude 3.5 / 3.7 / 4 시리즈)**: 요청 내 명시적 `cache_control` 마커. 어느 블록이 캐시 가능한지 태그한다. TTL: 5분(쓰기 비용 기본의 1.25배) 또는 1시간(쓰기 비용 기본의 2배). 캐시 읽기: Claude 3.5 Sonnet에서 신규 $3.00/M 대비 $0.30/M: 10배 저렴(docs.anthropic.com, 2026-04 기준). 요율은 모델별로 다르다(Opus/Haiku는 별도 공개). 항상 라이브 가격 페이지와 교차 확인하라.

**OpenAI**: ≥1024 토큰 프롬프트에 대한 자동 캐싱(platform.openai.com, 2026-04). 명시적 플래그 없음. 캐시된 입력은 현재 gpt-4o/gpt-5 요금표에서 신규보다 대략 10배 저렴하다. 문서도 릴리스 노트도 공식 적중률 베이스라인을 공개하지 않는다. 커뮤니티 보고는 신중한 프롬프트 설계로 30–60% 부근에 모인다. 자체 측정을 위해 `usage.cached_tokens`를 모니터링하라.

**Google (Gemini)**: 명시적 API를 통한 컨텍스트 캐싱. 1M 토큰 컨텍스트는 캐싱이 더욱 이득임을 뜻한다.

**셀프 호스트(vLLM, SGLang)**: Phase 17 · 06이 RadixAttention을 다룬다. 자체 컴퓨트에서의 같은 패턴.

### L1: 앱 수준 시맨틱 캐싱

LLM을 호출하기 전에 아예, 프롬프트를 해시하고, 임베딩하고, 유사한 캐시된 요청을 찾는다(임계값 이상의 코사인 유사도, 보통 0.95+). 적중 시 캐시된 응답을 반환한다. 미스 시 LLM을 호출하고 결과를 캐시한다.

오픈소스: Redis Vector Similarity, GPTCache, Qdrant. 상용: Portkey Cache, Helicone Cache.

벤더 정확도 주장은 반환된 캐시 응답이 얼마나 자주 의미적으로 적절했는지를 가리킨다. 얼마나 자주 적중했는지가 아니다. 프로덕션 적중률:

- 개방형 채팅: 10-15%.
- 구조화된 FAQ / 지원: 40-70%.
- 코드 질문: 20-30%(작은 변형이 적중을 죽임).
- 프롬프트를 반복하는 음성 에이전트: 50-80%(음성 정규화로 집합이 고정됨).

### 병렬화 안티패턴

에이전트가 10개의 도구 호출을 병렬로 한다. 10개 모두 같은 4K 토큰 시스템 프롬프트를 갖는다. Anthropic 캐시 쓰기는 요청별이다. 첫 캐시 쓰기는 프로바이더가 프롬프트를 본 뒤 약 300ms 후에 완료된다. 요청 2-10은 같은 밀리초 윈도우에 도착해 각각 캐시 미스를 본다. 결국 10개의 쓰기 프리미엄을 지불하고, 읽기 할인은 하나도 받지 못한다.

해결책은 순차 우선 배치다. 요청 1을 단독으로 보낸 뒤, 1의 캐시가 채워지면 2-10을 발사한다. 첫 도구 호출에 300ms가 더 걸리지만, 청구액은 5-10배 절약한다.

### 동적 콘텐츠 안티패턴

시스템 프롬프트가 이렇게 생겼다고 하자:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

모든 요청이 고유하다. 모든 요청이 쓴다. 적중 0.

해결책은 진정으로 정적인 부분을 모두 캐시 가능 프리픽스로 옮기고, 동적 콘텐츠는 캐시 경계 뒤에 덧붙이는 것이다:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery는 이 방식으로 7%에서 74% 캐시 적중률로 이동했고 그 해부도를 공개했다.

### 야간 워크로드를 위해 배치 + 캐시 쌓기

배치 API(Phase 17 · 15)는 24시간 처리 시간에 50% 할인을 준다. 그 위에 캐시된 입력을 얹으면 추가로 ~10배를 얻는다. 야간 분류, 레이블링, 리포트 생성 워크로드는 이를 쌓아서 동기·비캐시 비용의 ~10%로 떨어질 수 있다.

### 기억해야 할 숫자들

가격 지점들은 링크된 벤더 문서로부터 2026-04에 포착된 것이며 몇 달마다 바뀐다. 의존하기 전에 다시 확인하라.

- Anthropic 캐시 읽기: Claude 3.5 Sonnet에서 $0.30/M, 신규 입력보다 대략 10배 저렴(docs.anthropic.com).
- Anthropic 캐시 쓰기 프리미엄: 1.25배(5분 TTL) 또는 2배(1시간 TTL).
- OpenAI 자동 캐시: ≥1024 토큰 프롬프트에 적용. 캐시된 입력은 현재 요금표에서 신규 입력의 대략 10%로 가격 책정(platform.openai.com).
- 시맨틱 캐시 적중률(커뮤니티 보고): 개방형 채팅 ~10%; 구조화된 FAQ 최대 ~70%. 벤더 문서화 베이스라인 아님.
- ProjectDiscovery: 동적을 프리픽스 밖으로 옮겨 7% → 74% 적중률(프로젝트 블로그, 2025-11).
- 병렬화 안티패턴: N개 병렬 요청이 첫 캐시 쓰기를 놓칠 때 5–10배 청구액 부풀림의 전형적 보고.

```figure
semantic-cache-hit
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 혼합 워크로드에서 L1 + L2 캐싱을 시뮬레이션한다. 적중률, 청구액을 보고하고 병렬화 페널티를 보여준다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-cache-auditor.md`를 생성한다. 프롬프트 템플릿과 트래픽이 주어지면 캐시 가능성을 감사하고 재구조화를 권고한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 병렬화 플래그를 토글하라. 청구액이 얼마나 바뀌는가?
2. 시스템 프롬프트에 날짜가 있다. 이를 밖으로 옮겨라. 전후 적중률 산수를 보여라.
3. 요청 도착률이 주어졌을 때, 1시간 TTL(2배 쓰기)과 5분 TTL(1.25배 쓰기)의 손익분기점을 계산하라.
4. 0.95 임계값의 시맨틱 캐시가 20% 적중한다. 0.85에서는 50% 적중하지만 잘못된 캐시 응답이 보인다. 올바른 임계값을 고르고 정당화하라.
5. 사용자 질문당 10개의 병렬 서브 쿼리를 배치한다. 종단 간 지연 시간을 늘리지 않으면서 캐시 친화적으로 다시 작성하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| L2 프롬프트 캐시 (L2 prompt cache) | "프리픽스 캐시" | 프로바이더가 반복 프리픽스에 대한 KV를 저장 |
| `cache_control` | "Anthropic 캐시 마커" | 캐시 가능 블록을 표시하는 명시적 속성 |
| 캐시 쓰기 프리미엄 (Cache write premium) | "쓰기 세금" | 첫 미스-투-캐시에 대한 추가 비용(1.25배 또는 2배) |
| L1 시맨틱 캐시 (L1 semantic cache) | "임베딩 캐시" | LLM 호출 전 앱 수준의 해시-그리고-임베딩 |
| GPTCache | "LLM 캐싱 라이브러리" | 인기 있는 OSS L1 캐시 라이브러리 |
| 캐시 적중률 (Cache hit rate) | "적중 / 전체" | 캐시에서 처리된 요청의 비율 |
| 병렬화 안티패턴 (Parallelization anti-pattern) | "N-쓰기 함정" | N개 병렬 요청이 캐시를 N번 미스 |
| 동적 콘텐츠 함정 (Dynamic content trap) | "프롬프트 속 시간 함정" | 프리픽스의 동적 바이트가 적중률을 죽임 |
| RadixAttention | "레플리카 내부 캐시" | SGLang의 프리픽스 캐시 구현 |

## 더 읽을거리 (Further Reading)

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching): 공식 `cache_control` 시맨틱과 TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching): 자동 캐싱 동작과 적격성.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
