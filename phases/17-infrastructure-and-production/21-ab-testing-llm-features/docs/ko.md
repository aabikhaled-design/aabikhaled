# LLM 기능 A/B 테스트: GrowthBook, Statsig, 그리고 감(vibes) 문제

> 전통적인 A/B 테스트는 비결정적(non-deterministic) LLM을 위해 만들어지지 않았다. 결정적 구분은 이렇다. 평가(eval)는 "모델이 그 일을 할 수 있는가?"에 답하고, A/B 테스트는 "사용자가 신경 쓰는가?"에 답한다. 둘 다 필요하다. 감(vibe check)으로 배포하는 시대는 끝났다. 2026년에 무엇을 테스트할 것인가: 프롬프트 엔지니어링(prompt engineering)(표현), 모델 선택(GPT-4 vs GPT-3.5 vs OSS; 정확도 vs 비용 vs 지연 시간(latency)), 생성 파라미터(generation parameter)(temperature, top-p). 실제 사례: 한 챗봇의 보상 모델(reward-model) 변형은 대화 길이 +70%, 리텐션(retention) +30%를 달성했다. Nextdoor의 AI 제목 줄(subject-line) 실험은 보상 함수(reward-function) 정제 후 CTR +1%를 달성했다. Khan Academy의 Khanmigo는 지연 시간 대 수학 정확도 축에서 반복 개선했다. 플랫폼 양분: **Statsig**(2025년 9월 OpenAI에 11억 달러에 인수됨)(순차 테스트(sequential testing), CUPED, 올인원. **GrowthBook**) 오픈소스, 웨어하우스 네이티브(warehouse-native), 베이지안(Bayesian) + 빈도주의(Frequentist) + 순차 엔진, CUPED, SRM 점검, Benjamini-Hochberg + Bonferroni 보정. 웨어하우스-SQL 선호와 "OpenAI에 인수됨"이 조직에 중요한지에 따라 고른다.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 평가("모델이 그 일을 할 수 있는가")와 A/B 테스트("사용자가 신경 쓰는가")를 구별하기.
- 테스트 가능한 세 가지 축(프롬프트, 모델, 파라미터)을 열거하고 각각의 지표를 고르기.
- CUPED, 순차 테스트, Benjamini-Hochberg 다중 비교(multiple-comparison) 보정을 설명하기.
- 웨어하우스-SQL 태세와 기업 인수 입장에 따라 Statsig 또는 GrowthBook 고르기.

## 문제 (The Problem)

시스템 프롬프트를 손으로 조정했다. 더 나아 보인다. 배포한다. 전환율(conversion)이 잡음 수준으로 바뀐다. 지표 탓을 한다. 또는 새 모델을 배포했는데 전환율이 움직이지 않았다. 모델이 나빠진 것인가, 아니면 변화가 너무 작아 감지되지 않은 것인가? 모른다. A/B 없이 배포했기 때문이다.

평가는 모델이 레이블(label)이 달린 셋에서 어떤 과제를 할 수 있는지에 답한다. 사용자가 그 출력을 선호하는지에는 답하지 않는다. 오직 통제된 온라인 실험만이 그것에 답하며, 그것도 실험이 충분한 검정력(power)을 갖추고, 비결정성을 통제하며, 다중 비교를 보정할 때만 그렇다.

## 개념 (The Concept)

### 평가 vs A/B 테스트

**평가(Evals)**: 오프라인, 레이블 셋, 심판(루브릭(rubric) 또는 LLM-as-judge 또는 사람). 답: "이 고정된 분포에서 출력이 정확한가 / 도움이 되는가 / 안전한가?"

**A/B 테스트**: 온라인, 실 사용자, 무작위화. 답: "새 변형이 중요한 사용자 수준 지표를 움직이는가?"

둘 다 필요하다. 평가는 노출 전에 회귀(regression)를 잡고, A/B는 그 이후 제품 영향을 확인한다.

### 무엇을 테스트할 것인가

1. **프롬프트 엔지니어링**: 표현, 시스템 프롬프트 구조, 예시. 지표: 과제 성공, 사용자 리텐션, 요청당 비용.
2. **모델 선택**: GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. 지표: 정확도(과제) + 요청당 비용 + 지연 시간 P99. 다목적(multi-objective).
3. **생성 파라미터**: temperature, top-p, max_tokens. 지표: 과제 특화(출력 다양성 vs 결정성).

### CUPED: 분산 감소

Controlled-experiments Using Pre-Experiment Data. 사후 기간(post-period)을 비교하기 전에 사전 기간(pre-period) 분산을 회귀로 제거한다. 전형적 분산 감소: 30~70%. 유효 표본 크기(effective sample size)가 공짜로 올라간다.

구현: Statsig와 GrowthBook 둘 다 구현한다.

### 순차 테스트 (Sequential testing)

고전적 A/B는 고정 표본 크기를 가정한다. 순차 테스트("엿보고 결정하기")는 반복 관찰 하에서 거짓 양성률(false-positive rate)을 통제한다. 항상 유효한 순차 절차(mSPRT, Howard의 신뢰 시퀀스(confidence sequence))는 명확한 승자에 대해 일찍 멈추게 해준다.

### 다중 비교 보정 (Multiple-comparison corrections)

95% 신뢰도로 20개의 A/B 테스트를 돌리면 우연히 거짓 양성 하나가 나온다. Bonferroni 보정은 테스트당 α를 더 빡빡하게 한다. Benjamini-Hochberg는 거짓 발견율(false-discovery rate)을 통제한다. GrowthBook은 둘 다 구현한다.

### SRM: 표본 비율 불일치 (sample ratio mismatch)

할당 해시(assignment hash)가 사용자를 변형들에 무작위로 배정한다. 50/50 분할이 47/53을 내놓는다면 무언가 망가진 것이다. SRM 점검이 그것을 표시한다. 두 플랫폼 모두 구현한다.

### Statsig vs GrowthBook

**Statsig**:
- OpenAI에 11억 달러에 인수됨(2025년 9월). 호스팅형, SaaS.
- 순차 테스트, CUPED, 홀드아웃(held-out) 모집단.
- 올인원: 기능 플래그(feature flag) + 실험 + 관측성(observability).
- 가장 잘 맞는 경우: 이미 번들 제품을 원하고 OpenAI 소유 여부를 신경 쓰지 않는 팀.

**GrowthBook**:
- 오픈소스(MIT); 웨어하우스 네이티브(Snowflake/BigQuery/Redshift에서 직접 읽음).
- 다중 엔진: 베이지안, 빈도주의, 순차.
- CUPED, SRM, Bonferroni, BH 보정.
- 셀프 호스트 또는 매니지드 클라우드.
- 가장 잘 맞는 경우: 웨어하우스-SQL 환경, 데이터 팀이 지표 계층을 통제하고 OSS를 원하는 곳.

### 비결정성이 검정력을 복잡하게 만든다

같은 프롬프트가 변하는 출력을 만든다. 전통적 검정력 계산은 IID 관측을 가정한다. LLM 비결정성에서는 유효 표본 크기가 명목값보다 낮다. 안전 여유로 필요한 표본 크기에 약 1.3~1.5배를 곱하라.

### 실제 사례 결과

- 챗봇 보상 모델 변형: 대화 길이 +70%, 리텐션 +30%.
- Nextdoor 제목 줄: 보상 함수 정제 후 CTR +1%.
- Khan Academy Khanmigo: 지연 시간 대 수학 정확도의 반복적 트레이드오프(trade-off).

### 안티패턴: 감으로 배포하기

모든 시니어 엔지니어는 A/B 없이 "더 나아 보여서" 배포된 기능 하나를 댈 수 있다. 그중 대부분은 팀이 몇 달 동안 알아채지 못한 채 제품 지표를 회귀시켰다. A/B는 강제 함수(forcing function)다.

### 기억해 둘 숫자들

- Statsig OpenAI 인수: 11억 달러, 2025년 9월.
- GrowthBook: 오픈소스 MIT; 베이지안 + 빈도주의 + 순차.
- CUPED 분산 감소: 30~70%.
- LLM 비결정성 → +30~50% 표본 크기 버퍼.

```figure
mx-sequential-test
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 고정 경계와 순차 경계를 가진 순차 A/B 테스트를 시뮬레이션한다. 순차 방식이 어떻게 일찍 멈출 수 있게 해주는지 보여준다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-ab-plan.md`를 만든다. 기능 변경, 워크로드(workload), 베이스라인이 주어지면 플랫폼, 게이트, 표본 크기를 고른다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 베이스라인 전환율 3%에서 예상 5% 향상에 대해, 80% 검정력을 위한 표본 크기는 얼마인가?
2. 의료 규제 온프레미스(on-prem) 고객을 위해 Statsig 또는 GrowthBook을 고르라.
3. 해결된 티켓당 비용(cost-per-resolved-ticket)에서 GPT-4 vs GPT-3.5를 테스트하는 A/B를 설계하라. 주 지표(primary metric), 가드레일 지표(guardrail metric), 부 지표(secondary)는 무엇인가?
4. 카나리는 통과했지만 A/B가 전환율 -1.2%를 보인다. 배포할 것인가? 에스컬레이션 기준을 작성하라.
5. 사후 기간 분산의 60%를 가진 사전 기간에 CUPED를 적용하라. 유효 표본 크기 향상을 계산하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 평가 (Eval) | "오프라인 테스트" | 모델 역량에 대한 레이블 셋 평가 |
| A/B 테스트 (A/B test) | "실험" | 사용자에 대한 실시간 무작위 비교 |
| CUPED | "분산 감소" | 분산을 줄이기 위한 사전 기간 회귀 |
| 순차 테스트 (Sequential test) | "엿봐도 되는 테스트" | 조기 중단을 허용하는 항상 유효한 절차 |
| 다중 비교 (Multiple comparison) | "패밀리 오류" | 많은 테스트를 돌리면 거짓 양성이 부풀려짐 |
| Bonferroni | "빡빡한 보정" | α를 테스트 수로 나눔 |
| Benjamini-Hochberg | "BH FDR" | 거짓 발견율 통제, 덜 보수적 |
| SRM | "잘못된 분할" | 표본 비율 불일치; 할당 버그 |
| Statsig | "OpenAI 소유" | 상용 올인원, 2025년 인수됨 |
| GrowthBook | "그 OSS" | MIT 웨어하우스 네이티브 플랫폼 |
| mSPRT | "순차 확률비 검정(sequential probability ratio test)" | 고전적 순차 절차 |

## 더 읽을거리 (Further Reading)

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
