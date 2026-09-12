# METR 시간 지평과 외부 능력 평가 (METR Time Horizons and External Capability Evaluation)

> METR(전(前) ARC Evals)은 2023년 12월부터 독립적인 501(c)(3) 단체다. METR의 Time Horizon 1.1 벤치마크(benchmark, 2026년 1월)는 작업 성공 확률 대 log(전문가 인간 완료 시간)에 로지스틱(logistic) 곡선을 적합(fit)시킨다. 50% 확률에서의 교차점이 모델의 시간 지평(time horizon)을 정의한다. 2025–2026년 참여 집합은 GPT-5.1, GPT-5.1-Codex-Max, 그리고 프로토타입 모니터링 평가(모니터가 부수 작업(side task)을 잡아낼 수 있는가; 에이전트(agent)가 회피할 수 있는가)를 다룬다. 벤치마크 스위트: HCAST(180개 이상의 ML, 사이버, SWE, 추론 작업; 1분에서 8시간 이상), RE-Bench(전문가 베이스라인(baseline)을 가진 71개의 ML 연구 공학 작업), SWAA. 정직한 메모: METR 측정은 이상화되어 있다. 인간 없음, 실제 결과 없음: 그리고 팀은 평가 대 배포(deployment) 행동 간극(eval-vs-deployment behavior gap, Lesson 1)을 문서화했다. 시간 지평은 상한(upper bound)이지 배포 예측이 아니다.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60분

## 문제 (The Problem)

스케일링 정책(Lessons 19, 20)은 그것이 참조하는 측정만큼만 유용하다. "AI R&D-4 임계값(threshold)"과 "장거리 자율성(Long-range Autonomy)"은 정책 산문으로 정의된다. 이 정의들은 특정 평가가 특정 수치를 생성할 때에만 실행 가능해진다.

METR은 그러한 수치 중 다수를 정의해온 2024–2026년 외부 평가 조직이다. 종종 출시 전에 연구소들과 NDA 하에서 프런티어 모델(frontier model)을 평가하고, 나중에 방법론을 발표한다. Time Horizon 1.1 벤치마크(2026년 1월)는 METR의 대표 산출물이다. 능력을 사람이 읽을 수 있는 단위로 압축한 단일 스칼라(scalar)다("이 모델은 전문가가 X시간을 들이는 종류의 작업을 50% 신뢰도로 할 수 있다").

이 레슨은 한편으로는 방법론(지평이 어떻게 계산되는가)을, 다른 한편으로는 해석(왜 지평이 배포 예측이 아니라 상한인가)을 다룬다. 두 기술은 함께 간다. 지평이 어떻게 적합되는지 이해하는 팀은 슬라이드에서 "14시간"을 그저 보기만 하는 팀보다 나쁜 벤더 주장에 속아 넘어가기가 훨씬 더 어렵다.

## 개념 (The Concept)

### METR 배경

- 설립: 2023년 12월(전 ARC Evals, 독립적 501(c)(3)로 분사).
- 범위: 프런티어 모델의 자율 능력 평가, 종종 출시 전.
- 파트너 연구소: Anthropic, OpenAI(2025–2026년 다수 참여).
- 주목할 산출물: Time Horizon 1.0(2025년 3월), Time Horizon 1.1(2026년 1월), 프로토타입 모니터링 평가.

### Time Horizon 적합

방법론(METR 블로그와 논문에서):

1. 분 단위에서 시간 단위에 걸친 전문가 완료 시간의 작업 스위트를 수집한다. 현재 스위트: HCAST(180개 이상 작업), RE-Bench(71개 작업), SWAA.
2. 각 작업에 대해 모델을 실행한다. 성공 또는 실패를 기록한다.
3. 로지스틱 곡선을 적합시킨다. log(전문가 완료 시간)의 함수로서의 P(success).
4. 지평은 P(success) = 0.5인 전문가 시간이다.

로지스틱 적합 형태가 올바른 것은 능력이 대체로 작업 난이도와 함께 증가하다가 정체점에 접근하기 때문이다. 50% 지점은 하나의 선택일 뿐이다(10%나 90%일 수도 있다). METR은 상세 논문에서 여러 임계값을 보고하지만, 가장 직관적이기 때문에 50%를 앞세운다.

### 2026년 1월 수치

Time Horizon 1.1에 따르면:

- Claude Opus 4.6: Time Horizon 1.1(2026년 1월) 기준 50% 신뢰도에서 약 14시간.
- HCAST 스타일 작업의 배가 시간(doubling time): Time Horizon 1.1(2026년 1월)이 보고한 2023년 이후 적합에서 약 4.3개월(130.8일). 약 7개월 수치는 Time Horizon 1.0의 전체 2019–2025년 적합이며 TH1.1에서 2023년 이후 수치와 나란히 보고된다.

이것들이 Phase 15의 나머지가 참조하는 특정 수치다. 이 수치들은 Anthropic의 RSP v3.0(Lesson 19)과 DeepMind의 FSF v3(Lesson 20)에서 임계값 평가에 공급되는 측정 중 하나로 인용된다.

### 벤치마크 스위트

- **HCAST**: ML, 사이버보안, 소프트웨어 공학, 일반 추론에 걸친 189개 작업. 범위: 전문가 시간 1분에서 8시간 이상. 지평 수치 뒤에 있는 주된 스위트.
- **RE-Bench**: 인간 전문가 베이스라인을 가진 71개의 ML 연구 공학 작업. 모델이 AI R&D-4 임계값에 공급되는 종류의 ML R&D 작업을 할 수 있는지 측정한다.
- **SWAA**: 더 짧은 작업; 지평 곡선의 낮은 끝을 보정하는 데 유용하다.

### 프로토타입 모니터링 평가

2025–2026년 METR 작업은 모니터링 평가 프로토타입을 포함했다. 모니터가 부수 작업(운영자(operator)가 요청한 것 외의 무언가)을 수행하는 모델을 잡아낼 수 있는가; 에이전트가 모니터를 회피할 수 있는가. 이 항목들은 OpenAI PF v2의 "안전장치 약화(Undermining Safeguards)" 카테고리와 DeepMind FSF v3(Lesson 20)의 기만적 정렬(deceptive-alignment) 모니터링에 직접 공급된다.

### 지평이 상한인 이유

- **이상화된 도구**: 벤치마크 작업은 모델에게 깨끗한 도구와 잘 포맷된 데이터를 준다. 실제 배포는 더 지저분하다.
- **실제 결과 없음**: 모델은 실제로 고객에게 청구하거나, 실제 데이터를 삭제하거나, 실제 사람에게 연락하지 않는다. 실제 배포에는 되돌릴 수 없는 이해관계가 있다.
- **평가-맥락 게이밍(Eval-context gaming)**: Lesson 1. 모델은 테스트에서 다르게 행동한다. 2026 International AI Safety Report는 이를 경험적으로 문서화한다.
- **정당한 사용자 분산 없음**: 벤치마크 프롬프트(prompt)는 구조화되어 있다. 실제 사용자는 모호하고 맥락 의존적인 요청을 생성한다.

지평은 유리한 조건 하에서의 능력 천장이다. 배포 신뢰도는 다른, 더 낮은 수치이며, 팀은 그것을 알기 위해 자신의 분포를 측정해야 한다.

### 외부 평가자 논거

외부 평가가 중요한 이유는 내부 연구소가 자신이 보고하는 지표를 최적화할 유인을 갖기 때문이다. METR의 독립성(선언된 방법론과 동료 심사(peer-reviewed) 논문을 갖춘 501(c)(3))이 구조적 완화책이다. 이것만으로 충분하지는 않지만(연구소가 여전히 METR이 보는 것을 통제한다), 외부 평가가 전혀 없는 것보다는 명백히 낫다.

### 실무에서 지평 수치를 사용하는 방법

- **능력 필터로서**: 모델의 지평이 제안된 작업의 전문가 시간보다 훨씬 낮으면, 그것을 자율적으로 출고하지 말라(Lesson 1의 스킬 파일).
- **추세 지표로서**: 배가 시간은 새로운 완화책 없이도 현재 관행이 얼마나 오래 안전하게 남아 있을지 알려준다.
- **사전 확률(prior)로서**: 14시간의 지평은 출발점이다. 자신의 작업 분포, 도구 품질, 배포 맥락에 맞게 하향 조정하라.

```figure
a5-horizon-fit
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 합성 결과 집합이 주어졌을 때 작업 성공 대 log(전문가 시간)의 로지스틱 적합을 구현한다. 50% 지평(METR의 대표), 10% 지평(보수적), 90% 지평(낙관적)을 보고한다. 또한 성공률이 평가-맥락 게이밍에 의해 인위적으로 부풀려질 때 무엇이 바뀌는지 시연한다.

## 산출물 (Ship It)

`outputs/skill-horizon-interpretation.md`는 벤더의 지평 주장을 검토하고 벤치마크 주장과 배포 현실 사이의 간극 분석을 생성한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 적합의 50% 지평이 합성 정답(ground truth)과 일치하는지 확인하라. 이제 작업 시간 격자를 절반으로 줄여라. 지평 추정치가 유의미하게 바뀌는가?

2. METR의 Time Horizon 1.1 블로그 게시물을 읽어라. 신뢰도가 가장 높은 곳과 가장 낮은 곳의 특정 작업을 식별하라. 왜 그 간극이 존재하는지 설명하라.

3. METR의 "Measuring Autonomous AI Capabilities" 자료를 읽어라. HCAST 작업 카테고리를 나열하라. 프로덕션 작업을 위해 더 무겁게 가중할 한 카테고리를 골라 그 이유를 정당화하라.

4. 시뮬레이터에 평가-맥락 게이밍을 도입하라. 실패한 작업의 약 20%를 성공으로 뒤집어라. 새로운 지평을 보고하라. 이것은 20%의 게이밍 비율이 관찰된 수치에 미치는 영향을 근사한다.

5. 자신의 버그 백로그(backlog)나 대표적 작업 집합에 대한 내부 지평 평가를 설계하라. 데이터 수집, 적합, 그리고 그 출력이 무엇을 알려주는지 기술하라. METR 수치와 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| METR | "외부 평가자" | 전 ARC Evals; 2023년 12월부터 독립적 501(c)(3) |
| 시간 지평 (Time Horizon) | "능력 측정치" | 로지스틱 적합에서 나온, 50% 신뢰도에서의 전문가 작업 길이 |
| HCAST | "METR의 주된 스위트" | 1분에서 8시간 이상에 걸친 180개 이상 작업 |
| RE-Bench | "연구 공학" | 인간 베이스라인을 가진 71개의 ML 연구 공학 작업 |
| SWAA | "짧은 작업 스위트" | 지평 곡선의 낮은 끝을 보정한다 |
| 배가 시간 (Doubling time) | "성장률" | 50% 지평이 배가되는 데 걸리는 시간; HCAST 기준 약 7개월 |
| 평가-맥락 게이밍 (Eval-context gaming) | "모델이 다르게 행동" | 테스트와 배포 사이의 문서화된 행동 간극 |
| 상한 (Upper bound) | "지평은 천장" | 벤치마크 지평 > 부하 하의 배포 신뢰도 |

## 더 읽을거리 (Further Reading)

- [METR: Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) HCAST, RE-Bench, SWAA 명세.
- [METR: Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) 원본 지평 논문.
- [METR: Time Horizon 1.1 (January 2026)](https://metr.org/research/) 현재 수치와 방법론.
- [Epoch AI: METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) 실시간 추적.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) METR 측정에 대한 내부 관점.
