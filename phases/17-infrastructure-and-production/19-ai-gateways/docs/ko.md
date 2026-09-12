# AI 게이트웨이(AI Gateways): LiteLLM, Portkey, Kong AI Gateway, Bifrost

> 게이트웨이(gateway)는 앱과 모델 프로바이더 사이에 위치한다. 핵심 기능은 프로바이더 라우팅, 폴백(fallback), 재시도, 속도 제한(rate limiting), 시크릿 참조, 관측성(observability), 가드레일(guardrails)이다. 2026년 시장 분할: **LiteLLM**은 100+ 프로바이더, OpenAI 호환의 MIT OSS이지만 ~2000 RPS 부근에서 무너진다(8 GB 메모리, 발표된 벤치마크에서 연쇄 장애). Python, <500 RPS, 개발/프로토타이핑에 최적. **Portkey**는 컨트롤 플레인(control-plane)으로 포지셔닝되어 있고(가드레일, PII 마스킹, 탈옥(jailbreak) 탐지, 감사 추적), 2026년 3월 Apache 2.0 오픈소스가 되었으며, 20-40ms 지연 시간(latency) 오버헤드, 월 $49 프로덕션 티어. **Kong AI Gateway**는 Kong Gateway 위에 구축됨: Kong 자체 벤치마크에서 같은 12 CPU 기준: Portkey보다 228% 빠르고, LiteLLM보다 859% 빠름. 모델당 월 $100 가격(Plus 티어에서 최대 5개). 이미 Kong을 쓴다면 엔터프라이즈 적합. **Bifrost**(Maxim AI): 설정 가능한 백오프(backoff)가 있는 자동 재시도, OpenAI 429에서 Anthropic으로 폴백. **Cloudflare / Vercel AI Gateways**: 관리형, 제로 운영(zero-ops), 기본 재시도. 셀프 호스트 여부는 데이터 거주성이 좌우한다. Portkey와 Kong은 OSS + 선택적 관리형으로 중간에 위치한다.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 여섯 가지 핵심 게이트웨이 기능(라우팅, 폴백, 재시도, 속도 제한, 시크릿, 관측성, 가드레일)을 열거하기.
- 네 가지 2026년 게이트웨이(LiteLLM, Portkey, Kong AI, Bifrost)를 스케일 상한과 사용 사례에 매핑하기.
- Kong 벤치마크(Portkey 대비 228%, LiteLLM 대비 859%)를 인용하고 왜 >500 RPS에서 중요한지 설명하기.
- 데이터 거주성과 운영 예산이 주어졌을 때 셀프 호스트 대 관리형을 선택하기.

## 문제 (The Problem)

제품이 OpenAI, Anthropic, 셀프 호스트 Llama를 호출한다고 하자. 프로바이더마다 SDK, 오류 모델, 속도 제한, 인증 스킴이 제각각이다. 여기서 필요한 것은 페일오버(OpenAI가 429를 내면 Anthropic 시도), 단일 자격 증명 저장소, 통합 관측성, 테넌트별 속도 제한이다.

이것을 앱 계층에서 재발명하면 모든 서비스가 모든 프로바이더에 결합된다. 게이트웨이 계층은 프로바이더로 팬아웃(fan out)하는 하나의 API(보통 OpenAI 호환)를 갖춘 단일 프로세스로 이를 통합한다.

## 개념 (The Concept)

### 여섯 가지 핵심 기능

1. **프로바이더 라우팅**: OpenAI, Anthropic, Gemini, 셀프 호스트 등을 하나의 API 뒤에.
2. **폴백**: 429, 5xx, 또는 품질 실패 시 다른 곳에서 재시도.
3. **재시도**: 지수 백오프, 시도 횟수 제한.
4. **속도 제한**: 테넌트별, 키별, 모델별.
5. **시크릿 참조**: 런타임에 볼트(vault)에서 자격 증명을 가져옴(앱 안에는 절대 없음).
6. **관측성**: OTel + GenAI 속성(Phase 17 · 13) + 비용 귀속.
7. **가드레일**: PII 마스킹, 탈옥 탐지, 허용 주제 필터.

### LiteLLM — MIT OSS, Python

- 100+ 프로바이더, OpenAI 호환, 라우터 설정, 폴백, 기본 관측성.
- Kong 벤치마크에서 2000 RPS 부근에 무너짐. 8 GB 메모리 풋프린트, 지속 부하 하에서 연쇄 장애.
- 최적: Python 앱, <500 RPS, 개발/스테이징 게이트웨이, 실험적 라우팅.
- 비용: OSS는 $0. 클라우드 무료 티어 존재.

### Portkey: 컨트롤 플레인 포지셔닝

- 2026년 3월 기준 Apache 2.0 OSS. 가드레일, PII 마스킹, 탈옥 탐지, 감사 추적.
- 요청당 20-40ms 지연 시간 오버헤드.
- 보존 + SLA가 있는 프로덕션 티어 월 $49.
- 최적: 가드레일 + 관측성이 묶인 것이 필요한 규제 산업.

### Kong AI Gateway: 스케일 플레이

- Kong Gateway(성숙한 API 게이트웨이 제품, lua+OpenResty) 위에 구축.
- 12 CPU 등가에서의 Kong 자체 벤치마크: Portkey보다 228% 빠르고, LiteLLM보다 859% 빠름.
- 가격: 모델당 월 $100, Plus 티어에서 최대 5개.
- 최적: 이미 Kong을 씀; >1000 RPS; 라이선스 의향 있음.

### Bifrost (Maxim AI)

- 설정 가능한 백오프가 있는 자동 재시도.
- OpenAI 429에서 Anthropic으로 폴백이 정식 레시피.
- 신규 진입자; 상용.

### Cloudflare AI Gateway / Vercel AI Gateway

- 관리형, 제로 운영. 기본 재시도와 관측성.
- 최적: Cloudflare/Vercel의 엣지 서빙 JavaScript 앱.
- 가드레일과 속도 제한 면에서 Kong/Portkey 대비 제한적.

### 셀프 호스트 대 관리형

데이터 거주성이 강제 함수다. 헬스케어와 금융은 셀프 호스트가 기본(LiteLLM 또는 Portkey OSS 또는 Kong). 소비자 제품은 관리형(Cloudflare AI Gateway) 또는 중간 티어(Portkey 관리형)가 기본. 하이브리드: 규제 테넌트는 셀프 호스트, 나머지는 관리형.

### 지연 시간 예산

- LiteLLM: 전형적으로 5-15ms 오버헤드.
- Portkey: 20-40ms 오버헤드.
- Kong: 3-8ms 오버헤드.
- Cloudflare/Vercel: 1-3ms 오버헤드(엣지 이점).

게이트웨이 지연 시간은 TTFT에 직접 더해진다. TTFT P99 < 100ms SLA에는 Kong 또는 Cloudflare. P99 < 500ms에는 어느 것이든.

### 속도 제한 의미론이 중요하다

단순한 토큰 버킷(token-bucket)은 중간 스케일까지 동작한다. 멀티 테넌트는 슬라이딩 윈도우(sliding-window) + 버스트 허용 + 테넌트별 티어링을 요구한다. LiteLLM은 토큰 버킷을 출시한다. Kong은 슬라이딩 윈도우를 출시한다. Portkey는 티어드를 출시한다.

### 게이트웨이 + 관측성 + 라우팅은 조합된다

Phase 17 · 13(관측성) + 16(모델 라우팅) + 19(게이트웨이)는 프로덕션에서 같은 계층이다. 셋 다 커버하는 하나의 도구를 고르거나 신중하게 연결하라: 대부분의 2026년 배포는 역할 분할을 위해 Helicone(관측성) 또는 Portkey(가드레일)를 Kong(스케일)과 결합한다.

### 기억해야 할 숫자들

- LiteLLM: ~2000 RPS에서 무너짐, 8 GB 메모리.
- Portkey: 20-40ms 오버헤드; 2026년 3월 이후 Apache 2.0.
- Kong: Portkey보다 228% 빠르고, LiteLLM보다 859% 빠름.
- Kong 가격: 모델당 월 $100, Plus 티어에서 최대 5개.
- Cloudflare/Vercel: 엣지에서 1-3ms 오버헤드.

```figure
mx-gateway-fallback
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 429/5xx 주입 하에서 3개 프로바이더에 걸친 폴백이 있는 게이트웨이 라우팅을 시뮬레이션한다. 지연 시간, 재시도율, 폴백 적중률을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-gateway-picker.md`를 생성한다. 스케일, 운영 자세, 컴플라이언스, 지연 시간 예산이 주어지면 게이트웨이를 고른다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. OpenAI→Anthropic→셀프 호스트로 폴백을 설정하라. 5% 프로바이더 오류율에서 예상 적중률은 무엇인가?
2. SLA가 300ms 베이스라인에서 TTFT P99 < 200ms라고 하자. 어느 게이트웨이가 예산 안에 머무는가?
3. 한 헬스케어 고객이 셀프 호스트 + PII 마스킹 + 감사를 요구한다. Portkey OSS 또는 Kong을 고르라.
4. LiteLLM 대 Kong을 비교하라: 어느 RPS 상한에서 팀이 마이그레이션해야 하는가?
5. 멀티 테넌트 SaaS를 위한 속도 제한 정책을 설계하라: 무료 티어, 트라이얼 티어, 유료 티어. 토큰 버킷인가 슬라이딩 윈도우인가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 게이트웨이 (Gateway) | "API 브로커" | 앱과 프로바이더 사이에 위치하는 프로세스 |
| LiteLLM | "MIT인 것" | Python OSS, 100+ 프로바이더, 2K RPS에서 무너짐 |
| Portkey | "가드레일 게이트웨이" | 컨트롤 플레인 + 관측성, Apache 2.0 |
| Kong AI Gateway | "스케일인 것" | Kong Gateway 위에 구축, 벤치마크 선두 |
| Bifrost | "Maxim의 게이트웨이" | 재시도 + Anthropic 폴백 레시피 |
| Cloudflare AI Gateway | "엣지 관리형" | 엣지 배포 관리형 게이트웨이, 제로 운영 |
| PII 마스킹 (PII redaction) | "데이터 스크럽" | 모델로 보내기 전 정규식 + NER 마스킹 |
| 탈옥 탐지 (Jailbreak detection) | "프롬프트 주입 가드" | 사용자 입력에 대한 분류기 |
| 감사 추적 (Audit trail) | "규제 로그" | 모든 LLM 호출의 불변 기록 |
| 토큰 버킷 (Token-bucket) | "단순 속도 제한" | 리필 기반 속도 제한기 |
| 슬라이딩 윈도우 (Sliding-window) | "정밀 속도 제한" | 시간 윈도우 속도 제한기; 더 나은 공정성 |

## 더 읽을거리 (Further Reading)

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
