# LLM 라우팅 계층: LiteLLM, OpenRouter, Portkey

> 제공자 종속(provider lock-in)은 비싸다. 툴 호출 워크로드마다 적합한 모델이 다르다. 라우팅 게이트웨이(routing gateway)는 하나의 API 표면, 재시도, 페일오버(failover), 비용 추적, 가드레일(guardrail)을 제공한다. 2026년에는 세 가지 원형(archetype)이 지배한다: LiteLLM(오픈소스 셀프 호스팅), OpenRouter(매니지드 SaaS), Portkey(프로덕션 등급, 2026년 3월 오픈소스화). 이 레슨은 결정 기준을 명명하고 stdlib 라우팅 게이트웨이를 살펴본다.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- 셀프 호스팅, 매니지드, 프로덕션 등급 라우팅 옵션을 구별하기.
- 정의된 우선순위로 제공자 실패 시 재시도하는 폴백 체인(fallback chain) 구현하기.
- 제공자에 걸쳐 요청별 비용과 토큰 사용량을 추적하기.
- 주어진 프로덕션 제약에 대해 LiteLLM, OpenRouter, Portkey 중에서 결정하기.

## 문제 (The Problem)

제공자 라우팅이 중요한 시나리오:

1. **비용.** Claude Sonnet은 Haiku의 3배 비용이 든다. 분류(triage) 작업에는 Haiku로 충분하고, 종합(synthesis) 작업에는 Sonnet이 그 값을 한다. 요청별로 라우팅하라.

2. **페일오버.** OpenAI가 한 시간 동안 좋지 않다. 모든 요청이 실패한다. 재배포 없이 Anthropic으로 자동 폴백하기를 원한다.

3. **지연 시간(latency).** 라이브 채팅 UI는 빠른 첫 토큰 도달 시간(time-to-first-token)이 필요하다. 배치 요약기는 그렇지 않다. 지연 시간 SLA로 라우팅하라.

4. **컴플라이언스.** EU 사용자는 EU 리전에 머물러야 한다. 리전으로 라우팅하라.

5. **실험.** 동일 워크로드에서 두 모델을 A/B 테스트한다. 테스트 버킷으로 라우팅하라.

이 모두를 통합마다 손으로 코딩하는 것은 반복적이다. 라우팅 게이트웨이는 하나의 OpenAI 호환 API를 제공하고 나머지를 처리한다.

## 개념 (The Concept)

### OpenAI 호환 프록시 형태

모두가 OpenAI 형태를 말한다. 라우팅 게이트웨이는 `/v1/chat/completions`를 노출하고, OpenAI 스키마를 받아들이며, 내부적으로 Anthropic / Gemini / Cohere / Ollama / 무엇이든으로 프록시한다. 클라이언트는 신경 쓰지 않는다.

### 모델 별칭 (Model aliases)

`claude-3-5-sonnet-20251022` 대신, 코드는 `our_smart_model`이라고 말한다. 게이트웨이가 별칭을 실제 모델에 매핑한다. Anthropic이 Claude 4를 출하하면 서버 측에서 별칭을 바꾸면 되고, 코드는 아무것도 건드리지 않는다.

### 폴백 체인 (Fallback chains)

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

게이트웨이는 이를 구성(config)에 정의한다. 재시도는 예산에 카운트되므로 폴백 캐스케이드가 비용을 폭발시키지 않는다.

### 시맨틱 캐싱 (Semantic caching)

동일하거나 거의 동일한 프롬프트는 제공자 대신 캐시에 적중한다. 반복되는 에이전트 루프에서의 절감은 30~60퍼센트에 이른다. 키는 임베딩 기반이고, 거의 동일한 프롬프트는 캐시 슬롯을 공유한다.

### 가드레일 (Guardrails)

게이트웨이 수준:

- **PII 마스킹.** 프롬프트를 보내기 전 정규식 또는 ML 기반 패스.
- **정책 위반.** 금지된 콘텐츠를 가진 프롬프트를 거부한다.
- **출력 필터.** 컴플리션에서 누출을 정리한다.

Portkey와 Kong은 둘 다 의견이 반영된 가드레일을 출하한다. LiteLLM은 그것들을 선택사항으로 둔다.

### 키별 속도 제한

API 키 하나 = 팀 하나. 키별 예산은 한 팀이 공유 쿼터를 소비하는 것을 막는다. 대부분의 게이트웨이가 이를 지원한다.

### 셀프 호스팅 vs 매니지드 트레이드오프

| 요인 | LiteLLM (셀프 호스팅) | OpenRouter (매니지드) | Portkey (프로덕션) |
|--------|----------------------|----------------------|----------------------|
| 코드 | 오픈소스, Python | 매니지드 SaaS | 오픈소스 (2026년 3월) + 매니지드 |
| 설정 | 프록시 배포 | 가입 | 둘 다 |
| 제공자 | 100+ | 300+ | 100+ |
| 청구 | 자체 키 | OpenRouter 크레딧 | 자체 키 |
| 관찰성 | OpenTelemetry | 대시보드 | 전체 OTel + PII 마스킹 |
| 적합 대상 | 완전한 제어를 원하는 팀 | 빠른 프로토타이핑 | 컴플라이언스가 있는 프로덕션 |

LiteLLM은 SRE 팀이 있고 데이터 주권(data sovereignty)을 원할 때 이긴다. OpenRouter는 단일 구독과 인프라 없음을 원할 때 이긴다. Portkey는 가드레일과 컴플라이언스가 즉시 필요할 때 이긴다.

### 비용 추적 (Cost tracking)

모든 요청은 `provider`, `model`, `input_tokens`, `output_tokens`를 운반한다. 모델별 토큰당 가격(게이트웨이가 유지하는 가격표에서 가져온다)을 곱한다. 사용자별 / 팀별 / 프로젝트별 집계.

### MCP 더하기 라우팅

게이트웨이는 LLM 호출과 MCP 샘플링(sampling) 요청을 둘 다 라우팅할 수 있다. 샘플링 요청의 modelPreferences가 특정 모델을 선호할 때, 게이트웨이는 올바른 백엔드로 변환한다. 여기서 Phase 13 · 17(MCP 게이트웨이)과 이 레슨의 라우팅 게이트웨이가 때때로 하나의 서비스로 병합된다.

### 라우팅 전략

- **정적 우선순위.** 목록의 첫 번째; 오류 시 폴백.
- **로드 밸런싱.** 라운드 로빈 또는 가중치.
- **비용 인식.** 지연 시간 / 품질을 만족하는 가장 저렴한 모델을 고른다.
- **지연 시간 인식.** 지난 N분간 가장 빠른 모델을 고른다.
- **태스크 인식.** 프롬프트 분류기가 코딩을 한 모델로, 요약을 다른 모델로 라우팅한다.

```figure
tp-router-failover
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 약 150줄짜리 라우팅 게이트웨이를 구현한다: OpenAI 형태의 요청을 받아, 제공자별 스텁(stub)으로 변환하고, 우선순위 폴백 체인을 실행하고, 요청별 비용을 추적하고, 입력에 PII 마스킹 패스를 적용한다. 세 시나리오로 실행하라: 정상 요청, 폴백을 촉발하는 기본 제공자 장애, 마스킹에 걸린 PII 누출.

볼 것:

- `ROUTES` 딕셔너리: 별칭 -> 구체적 제공자의 우선순위 정렬 목록.
- 폴백 루프가 5xx에서 재시도한다.
- 비용 추적기가 토큰 사용량에 모델별 요율을 곱한다.
- PII 마스커가 전달 전 SSN 형태 패턴을 정리한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-routing-config-designer.md`를 만든다. 워크로드 프로파일(지연 시간, 비용, 컴플라이언스)이 주어지면, 이 스킬은 LiteLLM / OpenRouter / Portkey를 고르고 라우팅 구성을 만들어낸다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 장애 시나리오를 촉발하라; 폴백이 두 번째 제공자에 안착하고 비용이 올바르게 귀속되는지 확인하라.

2. 시맨틱 캐싱을 추가하라: 프롬프트의 SHA256이 조회 키다; 캐시 적중은 즉시 반환한다. 반복 호출에서 비용 절감을 측정하라.

3. "code ..." 프롬프트를 지능을 우대하는 별칭으로, "summarize ..." 프롬프트를 속도를 우대하는 별칭으로 라우팅하는 프롬프트 분류기를 추가하라.

4. 팀별 예산을 설계하라: 각 팀에 월간 지출 상한이 있다; 상한에 도달하면 게이트웨이가 요청을 거부한다. 강제 입도(요청별 또는 윈도우 단위)를 고르라.

5. LiteLLM, OpenRouter, Portkey 문서를 나란히 읽어라. 각각이 출하하지만 나머지 둘은 출하하지 않는 기능 하나를 명명하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 라우팅 게이트웨이(Routing gateway) | "LLM 프록시" | 많은 제공자 앞단의 단일 API 표면 계층 |
| OpenAI 호환(OpenAI-compatible) | "OpenAI 스키마를 말한다" | `/v1/chat/completions` 형태를 받아 어떤 백엔드로든 변환한다 |
| 모델 별칭(Model alias) | "our_smart_model" | 게이트웨이가 구체적 모델로 매핑하는 코드 내 이름 |
| 폴백 체인(Fallback chain) | "재시도 목록" | 실패 시 시도되는 제공자의 정렬된 목록 |
| 시맨틱 캐싱(Semantic caching) | "프롬프트 임베딩 캐시" | 키가 프롬프트의 임베딩이다; 거의 중복은 캐시 적중을 공유한다 |
| 가드레일(Guardrails) | "입출력 필터" | PII 마스킹, 정책 위반 거부 |
| 키별 속도 제한(Per-key rate limit) | "팀 예산" | API 키로 스코프된 쿼터 |
| 비용 추적(Cost tracking) | "요청별 지출" | 토큰 사용량 x 모델별 가격 집계 |
| LiteLLM | "오픈 프록시" | 셀프 호스팅 가능한 OSS 라우팅 게이트웨이 |
| OpenRouter | "매니지드 SaaS" | 크레딧 기반 청구의 호스팅 게이트웨이 |
| Portkey | "프로덕션 옵션" | 가드레일이 내장된 오픈소스 + 매니지드 |

## 더 읽을거리 (Further Reading)

- [LiteLLM: docs](https://docs.litellm.ai/) 셀프 호스팅 라우팅 게이트웨이
- [OpenRouter: quickstart](https://openrouter.ai/docs/quickstart) 매니지드 라우팅 SaaS
- [Portkey: docs](https://portkey.ai/docs) 가드레일이 있는 프로덕션 라우팅
- [TrueFoundry: LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) 결정 가이드
- [Relayplane: LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) 벤더 개관
