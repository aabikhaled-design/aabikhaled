# LLM 관측성 스택 선택(LLM Observability Stack Selection)

> 2026년 관측성(observability) 시장은 두 범주로 나뉜다. 개발 플랫폼(LangSmith, Langfuse, Comet Opik)은 모니터링을 평가(eval), 프롬프트 관리, 세션 리플레이와 묶는다. 게이트웨이/계측(instrumentation) 도구(Helicone, SigNoz, OpenLLMetry, Phoenix)는 텔레메트리(telemetry)에 집중한다. Langfuse는 MIT 라이선스 코어에 강력한 OSS 균형을 갖췄다(월 50K 이벤트 무료 클라우드). Phoenix는 Elastic License 2.0 하의 OpenTelemetry 네이티브로, 드리프트(drift)/RAG 시각화에 탁월하지만 지속적인 프로덕션 백엔드는 아니다. Arize AX는 제로 카피(zero-copy) Iceberg/Parquet 통합을 사용해 모놀리식(monolithic) 관측성보다 100배 저렴하다고 주장한다. LangSmith는 LangChain/LangGraph에서 선두이며, 사용자당 월 $39, 셀프 호스트는 Enterprise에서만 가능하다. Helicone은 프록시 기반으로 15-30분 설정, 월 100K 요청 무료지만, 에이전트 트레이스(agent trace)에 대한 깊이는 덜하다. 흔한 프로덕션 패턴: OpenTelemetry로 접착된 게이트웨이(Helicone/Portkey) + 평가 플랫폼(Phoenix/TruLens).

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 개발 플랫폼(번들: 평가 + 프롬프트 + 세션)과 게이트웨이/텔레메트리 도구(트레이스 + 메트릭만)를 구별하기.
- 6개 주요 도구(Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik)를 라이선스, 가격, 최적 사용 사례에 매핑하기.
- 게이트웨이 도구와 별도의 평가 플랫폼을 결합하게 해주는 OpenTelemetry 접착 패턴을 설명하기.
- 2026년 비용 차별 요소(Arize AX의 제로 카피 방식 대 모놀리식 수집)를 짚고 대략 100배 배수를 진술하기.

## 문제 (The Problem)

LLM 기능을 하나 출시했다. 동작한다. 그런데 프롬프트 실패, 도구 루프(tool loop), 지연 시간 회귀(regression), 비용 급증, 프롬프트 캐시 적중률은 전혀 들여다볼 수 없다. "LLM observability"를 구글링하면, 세 가지 다른 가격대에서 같은 문제를 푼다고 주장하는 여덟 개의 도구가 나온다.

그것들은 같은 문제를 풀지 않는다. LangSmith는 "이 LangGraph 실행이 왜 실패했는가?"에 답한다. Phoenix는 "내 RAG 파이프라인이 드리프트하고 있는가?"에 답한다. Helicone은 "어느 앱이 토큰을 태우고 있는가?"에 답한다. Langfuse는 "전체를 셀프 호스트할 수 있는가?"에 답한다. 다른 도구, 다른 청중.

선택을 가르는 축은 네 가지다: 스택(LangChain? 순수 SDK? 멀티 벤더?), 라이선스 허용도(MIT만? Elastic OK? 상용 괜찮음?), 예산(무료 티어? 월 $100? 월 $1000?), 셀프 호스트(필수? 있으면 좋음? 절대 안 함?).

## 개념 (The Concept)

### 두 범주

**개발 플랫폼**은 관측성을 평가, 프롬프트 관리, 데이터셋 버전 관리, 세션 리플레이와 묶는다. 실험을 돌리고, 어느 프롬프트가 통했는지 보고, 새 프롬프트를 이전 우승자들에 대해 데이터셋 회귀 테스트한다. LangSmith, Langfuse, Comet Opik.

**게이트웨이/텔레메트리 도구**는 추론 호출을 계측한다. 프롬프트, 응답, 토큰, 지연 시간, 모델, 비용. Helicone, SigNoz, OpenLLMetry, Phoenix. 미니멀리스트. OpenTelemetry를 통해 별도의 평가 도구와 결합 가능.

### Langfuse: OSS 균형

- 코어는 Apache / MIT 라이선스. Docker로 셀프 호스트.
- 클라우드 무료 티어: 월 50K 이벤트. 유료: 팀에 월 $29.
- 평가, 프롬프트 관리, 트레이스, 데이터셋. 네 가지 개발 플랫폼 기능 모두를 합리적으로 커버.
- 최적 지점: LangSmith급 기능을 원하지만 셀프 호스트해야 하거나 OSS 라이선스에 머물러야 함.

### Phoenix (Arize): 텔레메트리 우선, OpenTelemetry 네이티브

- Elastic License 2.0. 셀프 호스트가 간단함.
- RAG 및 드리프트 시각화에 탁월. 임베딩 공간 산점도(scatter plot)가 일급으로 출시됨.
- 지속적 프로덕션 백엔드로 설계되지 않음: 주로 개발 시점 관측성.
- 최적 지점: RAG 파이프라인 개발, 드리프트 디버깅, 프로덕션을 위한 별도 게이트웨이와 페어링.

### Arize AX: 스케일 플레이

- 상용. Iceberg/Parquet를 통한 제로 카피 데이터 레이크 통합.
- 스케일에서 모놀리식 관측성(Datadog급)보다 ~100배 저렴하다고 주장. 계산은 이렇다: 트레이스를 S3의 자체 Parquet에 저장하고 Arize가 직접 읽음.
- 최적 지점: 하루 >1000만 트레이스, 기존 데이터 레이크, Datadog 가격 없이 LLM 특화 대시보드를 원함.

### LangSmith: LangChain/LangGraph 우선

- 상용, 사용자당 월 $39. 셀프 호스트는 Enterprise에서만.
- LangChain과 LangGraph 스택에 동급 최고. 둘 다 아니면 매력이 덜함.
- 최적 지점: LangChain에 전념하고 지불 의향이 있는 팀.

### Helicone: 프록시 기반 최소 실행 가능

- `OPENAI_API_BASE`를 Helicone 프록시로 바꿔 15-30분 설정.
- MIT 라이선스. 월 100K 요청 무료, 유료 월 $20+.
- 페일오버, 캐싱, 속도 제한 포함: 게이트웨이 역할도 함.
- 에이전트 / 다단계 트레이스에 대한 깊이는 덜함.
- 최적 지점: 빠른 시작, 단일 스택 앱, 게이트웨이 + 관측성을 하나로 필요.

### Opik (Comet): OSS 개발 플랫폼

- Apache 2.0, 완전 OSS.
- Comet 혈통에 Langfuse와 유사한 기능 세트.
- 최적 지점: 이미 Comet을 쓰는 ML 팀, 같은 화면에서 LLM 관측성을 원함.

### SigNoz: OpenTelemetry 우선 풀 APM

- Apache 2.0. 일반 APM에 더해 OpenTelemetry를 통한 LLM도 처리.
- 최적 지점: 서비스와 LLM 호출 전반의 통합 관측성.

### 접착제: OpenTelemetry + GenAI 시맨틱 컨벤션

OpenTelemetry는 2025년 말에 GenAI 시맨틱 컨벤션을 발표했다(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). OTel을 소비하는 도구들은 상호 운용 가능하다. 떠오르는 프로덕션 패턴:

1. 모든 LLM 호출에서 GenAI 컨벤션으로 OTel 방출.
2. 일상 운영을 위해 게이트웨이(Helicone / Portkey)로 라우팅.
3. 회귀를 위해 평가 플랫폼(Phoenix / Langfuse)으로 이중 전송.
4. Arize AX 또는 DuckDB를 통한 장기 분석을 위해 데이터 레이크(Iceberg)에 아카이브.

### 함정: 잘못된 계층에서 계측

에이전트 프레임워크 내부에서 계측하면(예: LangSmith 트레이스 추가) 그 프레임워크에 결합된다. HTTP/OpenAI-SDK 계층에서 계측하면(OpenLLMetry 또는 게이트웨이를 통해) 이식 가능하다.

### 샘플링: 모든 것을 보관할 수는 없다

하루 >100만 요청에서, 전체 트레이스 보존은 LLM 호출보다 비싸다. 규칙으로 샘플링하라: 오류 100%, 고비용 100%, 성공 5%. 집계는 항상 유지하고, 롱테일을 위해 원본을 유지하라.

### 기억해야 할 숫자들

- Langfuse 무료 클라우드: 월 50K 이벤트.
- LangSmith: 사용자당 월 $39.
- Helicone 무료: 월 100K 요청.
- Arize AX 주장: 스케일에서 모놀리식보다 ~100배 저렴.
- OpenTelemetry GenAI 컨벤션: 2025년 출시, 2026년 널리 채택.

```figure
i4-otel-glue
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 보존 전략(100% 수집, 샘플링, 샘플링 + 오류) 전반에서 하루 100만 트레이스를 시뮬레이션한다. 각각에서 저장 비용과 손실되는 것을 보고한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-observability-stack.md`를 생성한다. 스택, 스케일, 예산, 라이선스 자세가 주어지면 도구를 고른다.

## 연습 문제 (Exercises)

1. LangChain을 쓰는 팀이 OSS 셀프 호스트 관측성을 원한다. Langfuse 또는 Opik을 고르고 정당화하라.
2. 하루 500만 트레이스에서 Datadog이 월 $150K를 견적하는 상황에서, Arize AX의 손익분기점을 계산하라.
3. 조직의 가이드라인이 모든 LLM 호출에 의무화해야 할 OpenTelemetry GenAI 속성 집합을 설계하라.
4. Phoenix 단독으로 프로덕션에 충분한지 논하라. 언제 충분하지 않은가?
5. Helicone은 프록시 오버헤드 20ms다. P99 TTFT 300ms에서 이것은 허용 가능한가? SLA가 100ms라면 어떤가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| OpenLLMetry | "LLM용 OTel" | LLM을 위한 오픈소스 OpenTelemetry 계측 |
| GenAI 컨벤션 (GenAI conventions) | "OTel 속성" | LLM 호출을 위한 표준 OTel 속성 이름 |
| LangSmith | "LangChain 관측성" | LangChain 생태계와 묶인 상용 플랫폼 |
| Langfuse | "OSS LangSmith" | 유사한 기능 세트의 MIT OSS |
| Phoenix | "Arize 개발 도구" | OpenTelemetry 네이티브 개발/평가 플랫폼 |
| Arize AX | "스케일 관측성" | 상용 제로 카피 Iceberg/Parquet 관측성 |
| Helicone | "프록시 관측성" | LLM 텔레메트리 + 게이트웨이 기능을 수집하는 HTTP 프록시 |
| Opik | "Comet LLM" | Comet의 Apache 2.0 OSS 개발 플랫폼 |
| 세션 리플레이 (Session replay) | "트레이스 재실행" | 도구 호출을 포함한 전체 에이전트 세션 재생 |
| 평가 (Eval) | "오프라인 테스트" | 레이블된 데이터셋에 대해 후보 모델/프롬프트를 실행 |

## 더 읽을거리 (Further Reading)

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
