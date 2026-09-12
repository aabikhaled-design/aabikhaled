# WMDP와 이중 용도 역량 평가 (WMDP and Dual-Use Capability Evaluation)

> Li 외, "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning"(ICML 2024, arXiv:2403.03218). 생물보안(1,520), 사이버 보안(2,225), 화학(412)에 걸친 4,157개의 객관식 문항. 문항은 "황색 구역(yellow zone)"(인접한 조력 지식(proximate enabling knowledge))에서 작동하며, 다중 전문가 검토와 ITAR/EAR 법적 준수로 걸러진다. 이중 목적: 이중 용도(dual-use) 역량의 대리 평가, 그리고 언러닝(unlearning) 벤치마크(동반 RMU 방법은 일반 역량을 보존하면서 WMDP 성능을 낮춘다). 2024-2025년 현장 서사: 초기 OpenAI/Anthropic 2024 평가는 인터넷 검색 대비 "미미한 상승(mild uplift)"을 보고했다; 2025년 4월 OpenAI의 Preparedness Framework v2는 모델이 "초보자가 알려진 생물학적 위협을 만드는 것을 의미 있게 돕기 직전(on the cusp)"이라고 말했다. Anthropic의 생물무기 획득 시험은 2.53배 상승을 보였고, 이는 ASL-3을 배제하기에 불충분했다.

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- WMDP의 세 도메인, 문항 수, "황색 구역" 필터 기준을 기술하기.
- RMU와 WMDP가 평가이자 언러닝 벤치마크인 이유를 설명하기.
- 2024-2025년 상승 서사를 기술하기: "미미한 상승" -> "직전" -> "ASL-3을 배제하기에 불충분".
- 초보자 상대(novice-relative) 상승과 전문가 절대(expert-absolute) 역량을 구분하기.

## 문제 (The Problem)

이중 용도 역량은 모든 연구소의 프런티어(frontier) 안전 프레임워크(Lesson 18)가 다루는 측정 문제다. 질문: 모델 X가 생물·화학·사이버에서 초보자의 대량 피해 유발 능력을 실질적으로 진전시키는가? 직접 측정(모델에게 실제로 피해를 산출하라고 요청)은 불법이며 비윤리적이다. 대리 측정에는 모델이 (정직한 역량 수치를 산출하기 위해) 거부할 수 없으면서도 문항 자체가 유해한 출판물이 아닌 벤치마크가 필요하다.

## 개념 (The Concept)

### "황색 구역"

직접적인 합성 레시피가 되지 않으면서 유해 과정에 대한 인접하고 조력적인 지식을 요구하는 문항. "how do I make [dangerous compound]?"가 아니라 "What reagent catalyzes step 4 of [published pathway]?". 각 문항은 여러 도메인 전문가의 검토를 거치며; ITAR/EAR 수출 통제 준수를 위해 걸러진다.

총 4,157개 문항:
- Biosecurity: 1,520
- Cybersecurity: 2,225
- Chemistry: 412

객관식 형식. 모델은 어떤 것을 돕도록 요청받지 않고 답한다; 유해 행동을 끌어내지 않고 역량을 측정할 수 있다.

### RMU — Representation Misdirection for Unlearning

동반 언러닝 방법. LLaMa-2-7B에 적용해 MMLU 및 다른 일반 역량 벤치마크를 몇 퍼센트포인트 이내로 보존하면서 WMDP 점수를 거의 무작위 수준으로 낮췄다. 발표된 이 방법은 이후 모든 생물-화학-사이버 언러닝 논문의 언러닝 베이스라인(baseline)이다.

### 2024-2025년 상승 서사

세 단계:

1. **2024년 "미미한 상승".** 초기 OpenAI와 Anthropic의 Preparedness/RSP 평가는 생물 인접 작업을 시도하는 초보자에 대해 인터넷 검색 대비 작은 이점을 보고했다. 공개 프레이밍: 프런티어 모델이 돕긴 하지만, Google보다 실질적으로 더 많이는 아니다.

2. **2025년 4월 "직전".** OpenAI의 Preparedness Framework v2는 모델이 "초보자가 알려진 생물학적 위협을 만드는 것을 의미 있게 돕기 직전"이라고 보고했다. 역량 주장이 아니라. 그 직전 시점이 가깝다는 경고다.

3. **Anthropic의 2025년 생물무기 획득 시험.** 초보자 참가자를 대상으로 한 통제 연구로, 획득 단계 작업에서의 상대적 성공을 측정했다. 2.53배 상승을 보고했다. ASL-3(Lesson 18)을 배제하기에 불충분하다. Anthropic의 Responsible Scaling Policy 티어 3의 임계값이 충족되거나 근접한다.

### 초보자 상대 대 전문가 절대

결정적 구분:

- **초보자 상대 상승(Novice-relative uplift).** 모델이 비전문가를 얼마나 돕는가? 곱셈적이다. 초보자가 아는 것이 적기 때문에 상대적 이점이 크다; 작은 정보조차 도움이 된다.
- **전문가 절대 역량(Expert-absolute capability).** 모델이 최대 노력에서 얼마나 많은 정보를 산출하는가? 전문가는 초보자보다 더 많이 추출할 수 있다. 절대 상한이 높다.

안전 사례(safety case, Lesson 18)는 둘 다를 겨냥한다: "모델은 초보자에게 실행할 만큼의 상승을 줄 수 없다" 더하기 "전문가는 모델에서 이미 출판되지 않은 정보를 추출할 수 없다".

### 측정 함정

WMDP는 역량 대리 지표이지 배포 측정이 아니다. WMDP에서 높은 점수를 받는 모델이 실제로 초보자에게 악용 가능한지는 다음에 따라 다를 수 있다:
- 유도 저항성(Elicitation resistance): 안전 필터를 건드리지 않고 역량을 끌어내기가 얼마나 어려운가
- 암묵지(Tacit knowledge): 정보가 아니라 실험실 기술을 요구하는 역량
- 실행 장벽(Execution barriers): 조달, 장비

Anthropic의 2025년 생물무기 획득 시험은 WMDP 방식 역량 위에 초보자 유도 계층을 더한다: 객관식 역량이 아니라 실제 작업 성공을 측정한다.

### Phase 18에서의 위치

Lesson 12-16은 모델 출력에 대한 공격과 방어 도구다. Lesson 17은 이중 용도 역량 계층(프런티어 안전 프레임워크(Lesson 18)가 평가하는 측정) 이다. Lesson 30은 현재 2026년 사이버/생물/화학/핵 상승 증거로 이 아크를 마무리한다.

```figure
al-wmdp-yellow-zone
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 WMDP 방식 평가 하니스(harness)를 만든다. 모의 모델을 범주별로 묶인 문항으로 테스트하고; 도메인별 점수를 보고한다. 간단한 언러닝 개입(도메인별 표현을 0으로 만들기)이 점수를 낮추고; 일반 역량과의 트레이드오프(trade-off)를 측정할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-wmdp-eval.md`를 만든다. 이중 용도 역량 주장("우리 모델은 생물무기에 의미 있게 도움을 주지 않는다")이 주어지면 다음을 감사한다: 어떤 벤치마크가 실행되었는지, 평가에 어떤 거부 경로가 사용되었는지(원시 완성 대 정책 게이트), 그리고 초보자 유도 연구가 객관식 결과를 보완하는지.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 장난감 언러닝 단계 전후의 도메인별 정확도를 보고하라. 일반 역량 트레이드오프를 설명하라.

2. 장난감 WMDP에 네 번째 도메인(예: 방사선)을 추가하라. 황색 구역의 예시 문항 유형 두 가지를 명시하라. 그런 문항을 만드는 것이 MMLU 방식 문항을 추가하는 것보다 어려운 이유를 설명하라.

3. WMDP 2024의 5절(RMU 방법론)을 읽어라. 더 단순한 언러닝 접근(예: 도메인 콘텐츠에 대한 상위 k개 뉴런 억제)을 스케치하고 예상되는 일반 역량 비용을 기술하라.

4. Anthropic 2025의 생물무기 획득 시험은 2.53배 상승을 보고한다. 이 수치가 위로 편향될 수 있는 두 가지 방식(초보자 표본 크기, 작업 충실도)과 아래로 편향될 수 있는 두 가지(유도 상한, 모델 안전 게이팅)를 기술하라.

5. ASL-3에 대한 안전 사례가 WMDP 언러닝 통과를 넘어 무엇을 요구하는지 명확히 표현하라. 보완적 유도 연구를 최소 두 가지 명명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| WMDP | "이중 용도 벤치마크" | 황색 구역의 생물/사이버/화학에 걸친 4,157개 객관식 문항 |
| 황색 구역 (Yellow zone) | "조력적이지만 합성은 아님" | 합성 레시피가 되지 않으면서 유해 역량에 인접한 근접 지식 |
| RMU | "언러닝 베이스라인" | Representation Misdirection for Unlearning; WMDP 점수를 낮추고 일반 역량을 보존 |
| 초보자 상대 상승 (Novice-relative uplift) | "비전문가를 얼마나 돕는가" | 초보자에게 현상 유지 인터넷 검색 대비 곱셈적 이점 |
| 전문가 절대 역량 (Expert-absolute capability) | "전문가의 상한" | 동기 있는 전문가가 모델에서 추출 가능한 최대 정보 |
| 획득 단계 작업 (Acquisition-phase task) | "합성 이전 단계" | 조달, 장비, 허가: 피해 경로의 가장 초기 부분 |
| ITAR/EAR | "수출 통제 준수" | 특정 조력 지식의 출판을 제약하는 법적 프레임워크 |

## 더 읽을거리 (Further Reading)

- [Li et al.: The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) 벤치마크 및 RMU 논문
- [OpenAI: Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) "on the cusp" 표현
- [Anthropic: Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) ASL-3 생물 임계값 및 획득 시험 결과
- [DeepMind: Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) 생물 상승 CCL
