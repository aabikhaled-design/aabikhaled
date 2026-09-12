# 직접 선호 최적화 계열(The Direct Preference Optimization Family)

> Rafailov et al. (2023)은 RLHF의 최적해가 선호 데이터에 관한 닫힌 형태(closed form)를 가짐을 보였고, 따라서 명시적 보상 모델(reward model)을 건너뛰고 정책(policy)을 직접 최적화할 수 있다. 그 통찰은 하나의 계열을 낳았다. IPO, KTO, SimPO, ORPO, BPO: 각각 DPO의 실패 모드를 하나씩 고친다. 2026년에는 직접 정렬 알고리즘(direct alignment algorithm)이 PPO보다 더 많은 프론티어 사후 학습(post-training) 실행을 출시한다. 하지만 레슨 2의 과최적화 곡선은 여전히 적용된다: DAA는 굿하트(Goodhart)를 벗어나지 못하며, 단지 물어뜯는 자리를 옮길 뿐이다.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- KL을 둔 RLHF 최적해로부터 DPO 닫힌 형태를 유도하기.
- IPO, KTO, SimPO, ORPO, BPO 각각이 DPO에서 고치는 실패 모드를 진술하기.
- "암묵적 보상 간극(implicit reward gap)"을 "선호 강도(preference strength)"와 구별하고, IPO의 항등 사상(identity mapping)이 왜 중요한지 설명하기.
- 명시적 RM이 없는데도 Rafailov et al. (NeurIPS 2024)이 DAA가 과최적화함을 증명하는 이유를 설명하기.

## 문제 (The Problem)

RLHF 목적함수(레슨 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

는 알려진 최적해를 가진다:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

따라서 보상은 최적 정책 대 참조의 비율로 암묵적으로 정의된다:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

이것을 브래들리-테리(Bradley-Terry) 선호 가능도에 대입하면, 분배 함수(partition function) `Z(x)`는 오직 `x`에만 의존하므로 소거된다. 남는 것은 정책 파라미터(parameter)만으로 이뤄진 손실(loss)이다. 보상 모델이 필요 없다. 그것이 DPO다.

함정: 그 유도는 최적해가 도달 가능하고, 선호 데이터가 분포 내(in-distribution)이며, 참조 정책이 참된 모드 앵커임을 가정한다. 이 중 어느 것도 정확히 성립하지 않는다. 각 계열 구성원은 위반된 가정을 서로 다르게 고친다.

## 개념 (The Concept)

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

무엇이 잘못될 수 있는가:

- 암묵적 보상 간극 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`은 무계(unbounded)다. 아주 작은 선호가 임의로 큰 간극을 만들 수 있다.
- 손실은 선택된(chosen) 로그 확률과 거부된(rejected) 로그 확률을 반대 방향으로 몰아간다. 거부된 것이 더 빨리 떨어지기만 하면 선택된 절대 로그 확률을 끌어내릴 수 있다. 이것이 선택 응답 저하(Degraded Chosen Response) 현상이다.
- 분포 밖(out-of-distribution) 선호(드문-드문 쌍 대 드문-드문 쌍)는 임의의 암묵적 보상을 만든다.

### IPO (Azar et al., 2024)

항등 선호 최적화(Identity Preference Optimization)는 log-sigmoid를 선호 확률에 대한 항등 사상으로 대체한다. 손실은 유계 목표에 대한 제곱 오차가 된다:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

마진(margin)은 `1/(2 beta)`로 유계다. 선호 강도와 암묵적 보상 간극은 비례한다. 폭주가 없다.

### KTO (Ethayarajh et al., 2024)

카너먼-트버스키 최적화(Kahneman-Tversky Optimization)는 쌍별 구조를 통째로 버린다. 단일 레이블된 출력과 이진 "바람직함" 또는 "바람직하지 않음" 신호가 주어지면, 이를 전망 이론(prospect-theory) 효용으로 사상한다:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

이득과 손실에 서로 다른 가중치(손실 회피, loss aversion)를 둔다. 이점: 쌍을 이루지 않은 데이터를 쓸 수 있으며, 이런 데이터가 훨씬 풍부하다.

### SimPO (Meng et al., 2024)

단순 선호 최적화(Simple Preference Optimization)는 학습 신호를 생성과 정렬시킨다. 참조 정책을 통째로 제거하고 로그 가능도를 길이로 정규화한다:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

안정화를 위한 마진 `gamma`를 둔다. 길이 정규화는 DPO의 길이 편향 실패 모드(구성상 더 긴 `y_w`가 더 큰 로그 확률 간극을 준다)를 이용할 유인을 제거한다.

### ORPO (Hong et al., 2024)

오즈비 선호 최적화(Odds-Ratio Preference Optimization)는 표준 SFT 음의 로그 가능도에 선호 항을 더한다:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

참조 정책 없음: SFT 항이 정규화 항이다. 베이스 모델에서 정렬된 모델까지 단일 단계로 학습한다. 별도의 SFT 체크포인트가 없다.

### BPO (ICLR 2026 제출, OpenReview id=b97EwMUWu7)

선택 응답 저하(Degraded Chosen Responses) 문제를 식별한다: DPO는 순위 `y_w > y_l`을 보존하지만 `y_w`의 절대 로그 확률은 떨어질 수 있다. BPO는 선택된 응답의 하향 이동을 벌하는 한 줄짜리 보정을 더한다. Llama-3.1-8B-Instruct에서 수학 추론에서 DPO보다 +10.1% 정확도를 보고한다.

### 보편적 결과: DAA는 여전히 과최적화한다

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024)은 여러 데이터셋과 여러 KL 예산 전반에서 DPO, IPO, SLiC로 정책을 학습했다. 골드 보상 대 KL 곡선은 Gao et al.과 동일한 정점-후-붕괴 형태를 가진다. 암묵적 보상은 학습 중 분포 밖 표본을 질의하며, KL 정규화는 이를 안정화하지 못한다.

DAA는 굿하트를 벗어나지 못한다. 물어뜯는 표면을 "보상 모델이 과최적화됨"에서 "참조 정책 비율이 과최적화됨"으로 바꿀 뿐이다. 보편적 해법(더 나은 데이터, 앙상블, 조기 종료)은 둘 다에 적용된다.

### 그것들 중 선택하기 (2026)

- 큰 쌍별 선호 데이터가 있다면: 보수적 beta를 둔 DPO, 길이 편향이 분명하면 SimPO.
- 쌍을 이루지 않은 이진 피드백이 있다면: KTO.
- 베이스 모델에서 단일 단계 파이프라인을 원한다면: ORPO.
- DPO 로그에서 선택된 로그 확률 저하가 보이면: BPO.
- 선호 강도가 폭넓게 변하고 DPO가 포화(saturation)하고 있다면: IPO.

모든 연구소는 다섯 가지 모두를 배터리(battery)로 실행하고 작업별로 우승자를 고른다. 최적해가 수학 추론과 안전성에서 같아야 할 이유는 없다.

```figure
dpo-margin
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 참 선호 강도가 쌍마다 다른 장난감 선호 데이터셋에서 여섯 가지 손실(DPO, IPO, KTO, SimPO, ORPO, BPO)을 비교한다. 각 손실은 작은 소프트맥스(softmax) 정책으로 같은 500쌍 표본에 대해 최적화된다. 방법별로 최종 승률, 선택된 로그 확률 표류, 암묵적 보상 분산을 그린다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-preference-loss-selector.md`를 생성한다. 데이터셋 통계(쌍별 대 비쌍별, 가변 대 균일 선호 강도, 길이 분포)와 목표(단일 단계 또는 SFT-그다음-선호)가 주어지면, 선호 손실을 권고하고 그것이 막아주는 실패 모드를 보고한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. DPO와 BPO에 대해 최종 선택된 로그 확률 하락을 보고하라. BPO는 더 높은 선택된 절대 확률을 유지해야 한다. 이를 검증하라.

2. 모든 쌍이 동일한 강도를 갖도록 선호 데이터를 수정하라. 여섯 방법 중 어느 것이 가장 견고한가? 어느 것이 저하되는가? 여기서 IPO의 이점을 설명하라.

3. 거부된 응답이 평균적으로 선택된 것보다 2배 길도록 만들어라. 다른 것을 바꾸지 않고, DPO의 길이 악용을 수치적으로 보이고 SimPO의 수정도 보여라.

4. Rafailov et al. (NeurIPS 2024)은 DAA가 과최적화한다고 주장한다. 단일 지점 버전을 재현하라: 선택-빼기-거부 KL 발산을 그리고 큰 beta에서 DPO의 과최적화를 관찰하라.

5. BPO 논문 초록(OpenReview b97EwMUWu7)을 읽어라. BPO가 DPO에 더하는 한 줄짜리 보정을 적어라. `code/main.py`의 구현과 대조하여 확인하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| DPO | "보상 모델 없는 RLHF" | 닫힌 형태 RLHF 최적해에서 유도된 손실; 정책 파라미터만 |
| 암묵적 보상(Implicit reward) | "로그 비율" | `beta * log(pi(y\|x) / pi_ref(y\|x))`: DPO가 암시하는 보상 |
| IPO | "유계 DPO" | log-sigmoid를 항등 사상으로 대체; 암묵적 보상 간극이 `1/(2 beta)`로 상한 |
| KTO | "비쌍별 DPO" | 손실 회피를 둔 단일 레이블에 대한 전망 이론 효용 |
| SimPO | "참조 없는 DPO" | 길이 정규화된 로그 가능도 + 마진; 참조 정책 없음 |
| ORPO | "단일 단계 DPO" | NLL + 오즈비 선호 항; 베이스 모델에서 한 번에 학습 |
| BPO | "선택 보존 DPO" | DPO에 선택된 응답의 절대 로그 확률 감소에 대한 페널티를 더함 |
| 선택 저하(Degraded Chosen) | "선택된 것이 내려감" | 거부된 것이 더 빨리 떨어지는 한 DPO는 선택된 로그 확률을 감소시킨다 |
| DAA | "직접 정렬 알고리즘" | 명시적 RM을 건너뛰는 모든 선호 손실 방법 |

## 더 읽을거리 (Further Reading)

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
