# 에이전트 관측성: Langfuse, Phoenix, Opik

> 세 가지 오픈소스 에이전트(agent) 관측성(observability) 플랫폼이 2026년을 지배한다. Langfuse(MIT): 월 600만+ 설치, 추적(tracing) + 프롬프트(prompt) 관리 + 평가 + 세션 리플레이. Arize Phoenix(Elastic 2.0): 깊은 에이전트 특화 평가, RAG 관련성, OpenInference 자동 계측. Comet Opik(Apache 2.0): 자동 프롬프트 최적화, 가드레일(guardrail), LLM 심판 환각 탐지.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- 상위 세 오픈소스 에이전트 관측성 플랫폼과 그들의 라이선스를 거명하기.
- 각각이 무엇에 가장 강한지 구별하기: Langfuse(프롬프트 관리 + 세션), Phoenix(RAG + 자동 계측), Opik(최적화 + 가드레일).
- 왜 2026년까지 조직의 89%가 에이전트 관측성을 갖추었다고 보고하는지 설명하기.
- LLM 심판 평가를 갖춘 stdlib 트레이스-투-대시보드 파이프라인을 구현하기.

## 문제 (The Problem)

OTel GenAI(Lesson 23)는 스키마를 제공한다. 그래도 스팬(span)을 수집(ingest)하고 평가를 실행하며 프롬프트 버전을 저장하고 회귀(regression)를 드러내는 플랫폼은 여전히 필요하다. 세 경쟁자는 각각 라이프사이클의 서로 다른 부분을 강조한다.

## 개념 (The Concept)

### Langfuse (MIT)

- 월 600만+ SDK 설치, 1만 9천+ GitHub 스타.
- 기능: 추적, 버전 관리 + 플레이그라운드를 갖춘 프롬프트 관리, 평가(LLM-as-judge, 사용자 피드백, 맞춤), 세션 리플레이.
- 2025년 6월: 이전에 상업용이던 모듈(LLM-as-a-judge, 주석 큐, 프롬프트 실험, Playground)이 MIT로 오픈소스화됨.
- 가장 강한 부분: 긴밀한 프롬프트 관리 루프를 갖춘 종단 간(end-to-end) 관측성.

### Arize Phoenix (Elastic License 2.0)

- 더 깊은 에이전트 특화 평가: 트레이스 클러스터링, 이상 탐지, RAG를 위한 검색 관련성.
- 네이티브 OpenInference 자동 계측.
- 프로덕션을 위해 관리형 Arize AX와 짝을 이룸.
- 프롬프트 버전 관리 없음: 더 넓은 플랫폼과 함께 쓰는 드리프트(drift)/행동 회귀 도구로 자리매김.
- 가장 강한 부분: RAG 관련성, 행동 드리프트, 이상 탐지.

### Comet Opik (Apache 2.0)

- A/B 실험을 통한 자동 프롬프트 최적화.
- 가드레일(PII 편집, 주제 제약).
- LLM 심판 환각 탐지.
- Comet 자체 측정에 따른 벤치마크: Opik 로그 + 평가가 23.44초인 반면 Langfuse는 327.15초(~14배 격차): 벤더 벤치마크는 방향성으로만 받아들여라.
- 가장 강한 부분: 최적화 루프, 자동 실험, 가드레일 강제.

### 업계 데이터

Maxim(2026 현장 분석)에 따르면: 조직의 89%가 에이전트 관측성을 갖추고 있다; 품질 문제가 최상위 프로덕션 장벽이다(응답자의 32%가 이를 언급).

### 하나 고르기

| 필요 | 선택 |
|------|------|
| 프롬프트 관리를 갖춘 올인원 | Langfuse |
| 깊은 RAG 평가 + 드리프트 | Phoenix |
| 자동 최적화 + 가드레일 | Opik |
| 개방형 라이선스, ELv2 아님 | Langfuse(MIT) 또는 Opik(Apache 2.0) |
| Datadog / New Relic 통합 | 아무거나: 모두 OTel을 내보냄 |

### 이 패턴이 잘못되는 지점

- **평가 전략 없음.** 평가 없는 추적은 그저 비싼 로깅일 뿐이다.
- **그라운딩 없이 자체 제작한 LLM 심판.** CRITIC 패턴(Lesson 05)이 적용된다. 심판은 사실 검증을 위해 외부 도구가 필요하다.
- **프롬프트 버전이 트레이스에 묶여 있지 않음.** 프로덕션이 회귀할 때 원인이 된 프롬프트로 이분 탐색(bisect)할 수 없다.

```figure
wb-trace-ingest
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 트레이스 수집기 + LLM 심판 평가자를 구현한다:

- GenAI 형태의 스팬을 수집.
- 세션별로 그룹화하고, 실패한 실행(가드레일 발동, 낮은 신뢰도 평가)을 태그.
- 루브릭(rubric)으로 에이전트 응답을 채점하는 스크립트화된 LLM 심판.
- 대시보드 같은 요약: 실패율, 최상위 실패 이유, 평가 점수 분포.

실행:

```
python3 code/main.py
```

출력: Langfuse/Phoenix/Opik가 보여줄 만한 세션별 평가 점수와 실패 범주화.

## 라이브러리로 써보기 (Use It)

- **Langfuse** 자체 호스팅 또는 클라우드; OTel 또는 그들의 SDK로 배선.
- **Arize Phoenix** 자체 호스팅; OpenInference 자동 계측.
- **Comet Opik** 자체 호스팅 또는 클라우드; 자동 최적화 루프.
- 이미 Datadog를 운영하는 혼합 운영+ML 팀을 위한 **Datadog LLM Observability**.

## 산출물 (Ship It)

`outputs/skill-obs-platform-wiring.md`는 플랫폼을 골라 트레이스 + 평가 + 프롬프트 버전을 기존 에이전트에 배선한다.

## 연습 문제 (Exercises)

1. 일주일 치 OTel 트레이스를 Langfuse 클라우드(무료 티어)로 내보내라. 어떤 세션이 실패했는가? 왜?
2. 자기 도메인에 맞는 LLM 심판 루브릭(사실 정확성, 어조, 범위 준수)을 작성하라. 50개 트레이스에서 테스트하라.
3. Langfuse 프롬프트 버전 관리를 Phoenix의 트레이스 클러스터링과 비교하라. 어느 것이 무엇이 깨졌는지 더 빨리 알려주는가?
4. Opik의 가드레일 문서를 읽어라. 자기 에이전트 실행 중 하나에 PII 편집 가드레일을 배선하라.
5. 자기 말뭉치에서 셋을 벤치마크하라. 벤더가 발표한 수치는 무시하고, 직접 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Tracing | "스팬 수집기" | OTel / SDK 스팬 수집; 세션별 색인 |
| Prompt management | "프롬프트 CMS" | 트레이스에 묶인 버전 관리 프롬프트 |
| LLM-as-judge | "자동 평가" | 별도 LLM이 루브릭에 따라 에이전트 출력을 채점 |
| Session replay | "트레이스 재생" | 디버깅을 위해 과거 실행을 단계별로 진행 |
| RAG relevancy | "검색 품질" | 검색된 컨텍스트가 쿼리와 맞는지 |
| Trace clustering | "행동 그룹화" | 드리프트 탐지를 위해 유사한 실행을 클러스터링 |
| Guardrail enforcement | "로그 시점 정책" | 로깅된 내용에 대한 PII/유해성/범위 검사 |

## 더 읽을거리 (Further Reading)

- [Langfuse docs](https://langfuse.com/): 추적, 평가, 프롬프트 관리
- [Arize Phoenix docs](https://docs.arize.com/phoenix): 자동 계측, 드리프트
- [Comet Opik](https://www.comet.com/site/products/opik/): 최적화 + 가드레일
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): 셋 모두가 소비하는 스키마
