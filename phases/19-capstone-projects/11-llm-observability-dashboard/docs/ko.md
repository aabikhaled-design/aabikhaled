# Capstone 11: LLM 관측성 및 평가 대시보드 (LLM Observability & Eval Dashboard)

> Langfuse는 오픈 코어(open-core)로 전환했다. Arize Phoenix는 2026 GenAI semconv 매핑을 발표했다. Helicone과 Braintrust는 둘 다 사용자별 비용 귀속(per-user cost attribution)에 더욱 집중했다. Traceloop의 OpenLLMetry는 사실상의 SDK 계측(instrumentation) 표준이 되었다. 프로덕션(production)의 형태는 트레이스(trace)용 ClickHouse, 메타데이터용 Postgres, UI용 Next.js, 그리고 샘플링된 트레이스 위에서 돌아가는 작은 평가 작업(eval job) 군단(DeepEval, RAGAS, LLM-judge)이다. 셀프 호스팅(self-hosted)으로 하나를 만들고, 최소 네 개의 SDK 계열에서 데이터를 수집(ingest)하며, 주입된 회귀(regression)를 5분 이내에 잡아내는 것을 시연한다.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P17 · P18
**Time:** 25시간

## 문제 (Problem)

2026년에 프로덕션 트래픽을 운영하는 모든 AI 팀은 모델 옆에 관측성(observability) 평면을 둔다. 비용 귀속. 환각(hallucination) 탐지. 드리프트(drift) 모니터링. 탈옥(jailbreak) 신호. SLO 대시보드. PII 유출 경보. 오픈소스 레퍼런스들(Langfuse, Phoenix, OpenLLMetry)은 수집 스키마로 OpenTelemetry GenAI 시맨틱 컨벤션(semantic conventions)에 수렴했다. 이제 OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM을 하나의 SDK로 계측하고 호환되는 스팬(span)을 보낼 수 있다.

여기서는 최소 네 개의 SDK 계열에서 데이터를 수집하고, 샘플링된 트레이스 위에서 소규모 평가 작업 집합을 돌리며, 드리프트를 탐지하고 경보를 보내는 셀프 호스팅 대시보드를 만든다. 측정 기준은 이렇다. 의도적으로 주입된 회귀(PII를 만들어내기 시작하는 프롬프트)가 주어졌을 때, 대시보드가 이를 잡아내고 5분 이내에 경보를 발생시킨다.

## 개념 (Concept)

수집은 OTLP HTTP다. SDK는 GenAI-semconv 스팬을 생성한다: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. 스팬은 컬럼형(columnar) 분석을 위해 ClickHouse에 들어가고, 메타데이터(사용자, 세션, 앱)는 Postgres에 들어간다.

평가는 샘플링된 트레이스 위에서 배치(batch) 작업으로 실행된다. DeepEval은 충실성(faithfulness), 독성(toxicity), 답변 관련성(answer relevance)을 채점한다. RAGAS는 트레이스가 검색 컨텍스트를 담고 있을 때 검색(retrieval) 지표를 채점한다. 커스텀 LLM-judge는 도메인 특화 검사(PII 유출, 정책 위반 응답)를 실행한다. 평가 실행은 부모 트레이스에 연결된 평가 스팬으로서 동일한 ClickHouse에 다시 기록한다.

드리프트 탐지는 시간에 따른 임베딩 공간(embedding space) 분포(프롬프트 임베딩에 대한 PSI 또는 KL 발산)와 평가 점수 추세를 함께 감시한다. 경보는 Prometheus Alertmanager로 들어간 뒤 Slack / PagerDuty로 전달된다. UI는 Recharts를 쓰는 Next.js 15다.

## 아키텍처 (Architecture)

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 스택 (Stack)

- 수집(Ingest): OpenTelemetry SDK + GenAI 시맨틱 컨벤션; OTLP HTTP 전송
- 컬렉터(Collector): tail-sampling 프로세서를 갖춘 OpenTelemetry Collector (비용 제어용)
- 저장소(Storage): 스팬용 ClickHouse, 메타데이터용 Postgres, 원본 이벤트 아카이브용 S3
- 평가(Evals): DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, 커스텀 LLM-judge
- 드리프트(Drift): 풀링된 프롬프트 임베딩에 대한 PSI / KL (sentence-transformers), 주간 단위
- 경보(Alerting): Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- 기본 지원 SDK: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## 직접 만들기 (Build It)

1. **컬렉터 설정.** OTLP HTTP 리시버, 오류 발생 트레이스는 100%, 성공 트레이스는 10%를 유지하는 tail-sampler, 그리고 ClickHouse와 S3로 보내는 익스포터를 갖춘 OpenTelemetry Collector.

2. **ClickHouse 스키마.** GenAI semconv를 반영하는 컬럼을 가진 `spans` 테이블: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, 그리고 긴 페이로드를 담는 JSON bag. user_id와 app_id 기준의 보조 인덱스를 추가한다.

3. **SDK 커버리지 테스트.** OpenLLMetry 자동 계측을 사용해 각 SDK(OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM)별로 작은 클라이언트 앱을 작성한다. 각각이 ClickHouse에 들어가는 표준 GenAI 스팬을 생성하는지 검증한다.

4. **평가 작업.** 스케줄링된 작업이 최근 15분의 샘플링된 트레이스를 읽고 DeepEval 충실성, 독성, 답변 관련성을 실행한다. 출력은 부모 트레이스에 연결된 평가 스팬이다.

5. **커스텀 LLM-judge.** PII 유출 judge: 응답이 주어지면 가드(guard) LLM을 호출해 PII 유출 가능성을 채점한다. 점수가 높은 응답은 분류(triage) 큐에 들어간다.

6. **드리프트 탐지.** 주간 작업이 이번 주의 풀링된 프롬프트 임베딩과 직전 4주 베이스라인(baseline) 사이의 PSI를 계산한다. PSI가 임계값을 넘으면 경보를 보낸다.

7. **대시보드.** 다음 페이지들을 가진 Next.js 15: 개요(초당 스팬 수, 사용자별 비용, p95 지연 시간), 트레이스(검색 + 워터폴), 평가(충실성 추세, 독성), 드리프트(시간에 따른 PSI), 경보.

8. **경보 체인.** Prometheus 익스포터가 평가 점수 집계와 지연 시간 백분위수를 읽고, Alertmanager가 경고는 Slack으로, 치명적 위반은 PagerDuty로 라우팅한다.

9. **회귀 프로브.** 버그를 주입한다: 평가 대상 챗봇이 1%의 확률로 가짜 SSN을 유출하기 시작한다. MTTR을 측정한다: 버그 배포부터 Slack 경보까지.

## 라이브러리로 써보기 (Use It)

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 산출물 (Ship It)

`outputs/skill-llm-observability.md`가 결과물이다. LLM 애플리케이션이 주어지면, 대시보드는 그 트레이스를 수집하고, 평가를 실행하며, 드리프트에 경보를 보내고, 사용자별 비용 분석을 Next.js로 노출한다.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 트레이스 스키마 커버리지 | 표준 GenAI 스팬을 생성하는 SDK 계열 수 (목표: 6개 이상) |
| 20 | 평가 정확성 | 수작업 레이블 세트 대비 DeepEval / RAGAS 점수 |
| 20 | 대시보드 UX | 주입된 회귀에 대한 MTTR (5분 미만 목표) |
| 20 | 비용 / 확장성 | 백로그 없이 초당 1k 스팬 지속 수집 |
| 15 | 경보 + 드리프트 탐지 | Prometheus/Alertmanager 체인을 끝에서 끝까지 동작시킴 |
| **100** | | |

## 연습 문제 (Exercises)

1. Haystack 프레임워크에 대한 커스텀 계측을 추가한다. 충실한 `gen_ai.*` 속성과 함께 표준 스팬이 ClickHouse에 들어가는지 검증한다.

2. 동일한 트레이스에 대해 DeepEval을 Phoenix evaluator로 교체한다. 두 평가 엔진 사이의 점수 드리프트를 측정한다.

3. 드리프트 탐지기를 정교화한다: PSI를 전역이 아니라 app-id별로 계산한다. 앱별 드리프트 궤적을 보여준다.

4. "사용자 영향(user impact)" 페이지를 추가한다: 스파크라인(sparkline)과 함께 사용자별 비용과 사용자별 실패율.

5. 독성 > 0.5인 트레이스 100%에 더해 나머지의 10% 계층화 샘플을 유지하는 tail-sampling 정책을 만든다. 도입된 샘플링 편향(sampling bias)을 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM 속성" | LLM 스팬 속성(시스템, 모델, 토큰)에 대한 2025 OpenTelemetry 명세 |
| Tail sampling | "사후 트레이스 샘플링" | 컬렉터가 트레이스 완료 후에 유지할지 버릴지 결정함 (오류를 들여다볼 수 있음) |
| PSI | "모집단 안정성 지수(population stability index)" | 두 분포를 비교하는 드리프트 지표; 보통 > 0.2면 유의미한 드리프트 신호 |
| LLM-judge | "모델로서의 평가" | 다른 LLM의 출력을 루브릭(충실성, 독성, PII)으로 채점하는 LLM |
| Tail-sampling policy | "유지 규칙(keep-rule)" | 어떤 트레이스를 보존하고 어떤 것을 버릴지 결정하는 규칙; 오류 + 샘플 비율 |
| Eval span | "연결된 평가 트레이스" | 원본 LLM 호출 스팬에 연결되어 평가 점수를 담는 자식 스팬 |
| Cost per user | "단위 경제성(unit economics)" | 기간 동안 user_id에 귀속된 달러 비용; 핵심 제품 지표 |

## 더 읽을거리 (Further Reading)

- [Langfuse](https://github.com/langfuse/langfuse): 레퍼런스 오픈 코어 관측성 플랫폼
- [Arize Phoenix](https://github.com/Arize-ai/phoenix): 강력한 드리프트 지원을 갖춘 대안 레퍼런스
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry): 자동 계측 SDK 계열
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): 수집 스키마
- [Helicone](https://www.helicone.ai): 대안 호스팅형 관측성
- [Braintrust](https://www.braintrust.dev): 대안 평가 우선(eval-first) 플랫폼
- [ClickHouse documentation](https://clickhouse.com/docs): 컬럼형 스팬 저장소
- [DeepEval](https://github.com/confident-ai/deepeval): evaluator 라이브러리
