# 공정성 기준: 집단, 개인, 반사실

> 세 가지 계열이 공정성(fairness) 문헌을 구조화한다. 집단 공정성(group fairness): 인구통계학적 동등성(demographic parity), 균등화된 승산(equalized odds), 조건부 사용 정확도 동등성(conditional use accuracy equality): 보호 집단(protected group) 간에 평균적으로 동일한 비율. 개인 공정성(individual fairness)(Dwork et al. 2012): 비슷한 개인은 비슷한 결정을 받는다. 결정 사상(decision map)에 대한 립시츠 조건(Lipschitz condition). 반사실 공정성(counterfactual fairness)(Kusner et al. 2017): 민감 속성(sensitive attribute)을 반사실적으로 변경했을 때 결정이 바뀌지 않으면, 그 결정은 해당 개인에게 공정하다. 2024년 이론적 결과(NeurIPS 2024): 반사실 공정성(CF)과 정확도 사이에는 본질적인 트레이드오프(trade-off)가 존재한다. 모델 비의존적(model-agnostic) 방법은 최적이지만 불공정한 예측기를, 정확도 손실이 유계인 상태로 반사실 공정 예측기로 변환한다. 백트래킹 반사실(backtracking counterfactuals)(arXiv:2401.13935, 2024년 1월): 법적으로 보호되는 속성에 대한 개입(intervention)을 요구하지 않는 새로운 패러다임. 철학적 화해(ICLR Blogposts 2024): 인과 그래프(causal graph)가 주어지면, 특정 집단 공정성 척도를 만족하는 것이 반사실 공정성을 함의한다.

**Type:** Learn
**Languages:** Python (stdlib, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 세 가지 집단 공정성 기준(인구통계학적 동등성, 균등화된 승산, 조건부 사용 정확도 동등성)과 한 가지 불가능성 결과(impossibility result)를 진술하기.
- Dwork et al. 2012의 립시츠 정식화(Lipschitz formulation)를 통해 개인 공정성을 기술하기.
- 반사실 공정성과 그것의 인과 그래프 의존성을 기술하기.
- 백트래킹 반사실이 무엇인지, 그리고 그것이 보호 속성에 대한 개입 문제를 어떻게 우회하는지 설명하기.

## 문제 (The Problem)

레슨 20은 편향(bias)을 측정했다. 레슨 21은 그 측정이 따라야 할 공정성 기준을 정의한다. 세 가지 계열은 구조적으로 서로 다른 기준을 제시한다. 어떤 모델은 집단 공정하면서 개인 불공정할 수 있고, 반사실 공정하면서 집단 불공정할 수 있다. 기준 선택은 정책 결정이다. 어떤 기준도 보편적으로 최적이지 않다.

## 개념 (The Concept)

### 집단 공정성

- **인구통계학적 동등성(demographic parity).** 모든 집단에 대해 P(Y=1 | A=a) = P(Y=1 | A=a'). 동일한 합격률.
- **균등화된 승산(equalized odds).** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). 집단 간 동일한 TPR과 FPR.
- **조건부 사용 정확도 동등성(conditional use accuracy equality).** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). 집단 간 동일한 예측값(predictive value).

불가능성(Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): 기저율(base rate)이 동일하지 않은 경우, 이 세 가지는 동시에 만족될 수 없다.

### 개인 공정성

Dwork et al. 2012. 결정 사상 f는, 어떤 립시츠 상수(Lipschitz constant) L에 대해 |f(x) - f(x')| <= L * d(x, x')가 성립하면, 과업별 유사도 척도(similarity metric) d에 관하여 개인 공정하다. 비슷한 개인은 비슷한 결정을 받는다.

d를 정의해야 한다. 이는 통계적 문제가 아니라 정책 문제다.

### 반사실 공정성

Kusner et al. 2017. 개인 i의 민감 속성을 반사실적으로 변경했을 때, 모집단의 인과 모델(causal model) 하에서 결정이 바뀌지 않으면, 그 결정은 개인 i에게 반사실 공정하다.

인과 DAG가 필요하다. 이 DAG는 모델링 선택이다. 반사실 공정성은 그 DAG가 정당화되는 만큼만 정당화된다.

### 반사실 공정성 대 정확도 트레이드오프

NeurIPS 2024 이론적 결과: 반사실 공정성과 예측 정확도 사이에는 본질적인 트레이드오프가 존재한다. 모델 비의존적 방법은 최적이지만 불공정한 예측기를, 유계 정확도 비용으로 반사실 공정 예측기로 변환할 수 있다. 그 정확도 비용은 최적의 불공정 예측기에서 민감 속성 계수(coefficient)의 크기에 따라 달라진다.

### 백트래킹 반사실

arXiv:2401.13935 (2024년 1월). 전통적인 반사실은 민감 속성에 대한 개입을 요구한다. "이 사람의 성별이 달랐다면 결정이 바뀌었을까." 법적으로 이는 문제가 된다. 분류 관련 법에서 보호 속성에는 개입할 수 없다.

백트래킹 반사실은 방향을 뒤집는다. 속성에 개입하는 대신, 그 개인의 실제 특성들의 어떤 조합이 반사실적 결과를 만들어냈을지를 묻는다. 이는 법적 반론을 우회한다.

### 철학적 화해

ICLR Blogposts 2024. 인과 그래프가 손에 있으면, 특정 집단 공정성 척도를 만족하는 것이 반사실 공정성을 함의한다. 세 가지 계열은 직교하지 않는다. 그것들은 동일한 기저 인과 구조의 서로 다른 단면이다.

이것이 불가능성 정리를 해소하지는 않는다(동일하지 않은 기저율은 여전히 집단 공정성의 동시 충족을 막는다). 그러나 이는 "집단"과 "개인 / 반사실" 공정성 사이의 외견상 대립이, 부분적으로는 인과 모델을 명시하지 않은 데서 비롯된 인공물(artifact)임을 보여준다.

### Phase 18에서 이 레슨의 위치

레슨 20은 편향 측정이다. 레슨 21은 공정성 정의다. 레슨 22는 프라이버시(차등 프라이버시)다. 레슨 23은 워터마킹이다. 이들은 기만(deception) 인접 레슨인 레슨 7-11을 보완하는, 배분(allocation) 인접 레슨이다.

```figure
an-fairness-trilemma
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 민감 속성과 동일하지 않은 기저율을 가진 장난감 수준의 이진 분류(binary classification) 데이터셋을 구성한다. 단순한 분류기(classifier)에 대해 인구통계학적 동등성, 균등화된 승산, 조건부 사용 정확도 동등성을 계산한다. 세 척도가 서로 불일치함을 관찰한다. 인구통계학적 동등성을 위한 재가중(re-weighting)을 적용하고, 그것이 나머지 두 척도에 끼치는 비용을 관찰한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-fairness-criterion.md`를 생성한다. 공정성 주장이나 정책이 주어지면, 어떤 기준이 주장되고 있는지, 주장된 동일하지 않은 기저율 하에서 모델이 나머지 기준들을 만족할 수 있는지, 그리고 그 주장이 어떤 인과 DAG에 의존하는지를 식별한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 기본 데이터에 대한 세 가지 집단 척도를 보고하라. 인구통계학적 동등성을 목표로 한 재가중을 적용하고 다시 보고하라.

2. 비민감 특성에 L2를 사용하여 Dwork et al. 2012의 개인 공정성 척도를 구현하라. 상수 L=1로 립시츠 조건을 위반하는 쌍이 몇 개인지 보고하라.

3. Kusner et al. 2017을 읽어라. 이력서 채점을 위한 단순한 두 특성 인과 DAG를 구성하고, 그것이 함의하는 반사실 공정성 조건을 식별하라.

4. 2024년 백트래킹 반사실 논문은 보호 속성에 대한 개입을 피한다. 이것이 법적 준수에 중요해지는 시나리오를 기술하라.

5. ICLR 2024 화해 논문은 집단 공정성과 반사실 공정성이 동일한 구조의 단면이라고 주장한다. `code/main.py`의 세 기준 중 둘을 고르고, 그것들을 동등하게 만드는 인과적 가정을 진술하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 인구통계학적 동등성(Demographic parity) | "동일한 비율" | 집단 간 P(Y=1 | A=a)가 동일함 |
| 균등화된 승산(Equalized odds) | "동일한 TPR/FPR" | 집단 간 동일한 참 양성률과 거짓 양성률 |
| 조건부 사용 정확도(Conditional use accuracy) | "동일한 PPV/NPV" | 집단 간 동일한 예측값 |
| 개인 공정성(Individual fairness) | "립시츠 조건" | 비슷한 개인은 비슷한 결정을 받음 |
| 반사실 공정성(Counterfactual fairness) | "인과적 변경 불변성" | 반사실적 속성 변경 하에서 결정이 바뀌지 않음 |
| 백트래킹 반사실(Backtracking counterfactual) | "실제값으로 설명" | 속성에서 앞으로가 아니라 결과에서 거꾸로 추론한 반사실 |
| 불가능성 정리(Impossibility theorem) | "세 가지가 충돌한다" | Chouldechova / KMR 2017: 동일하지 않은 기저율 하에서 집단 기준들은 상호 배타적 |

## 더 읽을거리 (Further Reading)

- [Dwork et al.: Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) 개인 공정성
- [Kusner, Loftus, Russell, Silva: Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) 반사실 공정성
- [Chouldechova: Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) 불가능성
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935): 보호 속성 개입을 위한 새로운 패러다임
