# 보상 해킹과 굿하트의 법칙(Reward Hacking and Goodhart's Law)

> 프록시(proxy) 보상을 최대화할 만큼 강한 옵티마이저(optimizer)라면, 프록시와 실제로 원했던 것 사이의 간극을 반드시 찾아낸다. Gao et al. (ICML 2023)은 이것에 스케일링 법칙(scaling law)을 부여했다. 프록시 보상은 증가하고, 골드(gold) 보상은 정점을 찍은 뒤 떨어지며, 그 간극은 초기 정책으로부터의 KL 발산과 함께, 닫힌 형태(closed form)로 적합할 수 있는 방식으로 커진다. 아첨(sycophancy), 장황함 편향(verbosity bias), 불충실한 사고 연쇄(unfaithful chain-of-thought), 평가자 조작(evaluator tampering)은 별개의 문제가 아니다. 이것들은 같은 문제가 다른 옷을 입은 것이다.

**Type:** Learn
**Languages:** Python (stdlib, proxy-vs-gold-reward simulator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 10 · 07 (RLHF)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 굿하트의 법칙(Goodhart's Law)을 진술하고, 그것이 민간의 구호가 아니라 불완전한 프록시를 대상으로 한 모든 최적화의 예측 가능한 속성인 이유를 설명하기.
- Gao et al. 2023 스케일링 법칙을 설명하기: 초기 정책으로부터의 KL 거리의 함수로서의 평균 프록시-골드 간극.
- 보상 해킹(reward hacking)의 네 가지 흔한 발현(장황함, 아첨, 불충실한 추론, 평가자 조작)을 명명하고, 각각을 공유된 메커니즘으로 거슬러 추적하기.
- 두꺼운 꼬리(heavy-tailed) 보상 오차 아래에서 KL 정규화만으로는 구제되지 않는 이유(파국적 굿하트, Catastrophic Goodhart)를 설명하기.

## 문제 (The Problem)

실제로 원하는 것은 측정할 수 없다. 그것의 프록시는 측정할 수 있다. 모든 RLHF 파이프라인은 이 대체를 이용한다. "인간 선호"가 "50k개의 레이블된 쌍에 대한 브래들리-테리(Bradley-Terry) 적합"이 된다. 프록시에서 높은 보상에 도달한 옵티마이저는, 구성상 측정한 것에서 잘한 것이다. 그것이 원했던 것에서도 잘했는지는 프록시가 그것을 얼마나 빈틈없이 추적했는지에 달려 있고, 그 답은 언제나 바란 것보다 덜 빈틈없다.

Gao, Schulman, Hilton (2023)은 이를 직접 측정했다. 100k개의 레이블에서 "골드" 보상 모델(reward model)을 학습한다. 같은 데이터의 {1k, 3k, 10k, 30k} 부분집합에서 프록시 RM을 학습한다. 각 프록시를 대상으로 정책을 최적화한다. 골드-RM 점수를 초기 정책으로부터의 KL 발산에 대해 그린다. 모든 곡선은 상승하고, 정점을 찍고, 떨어진다. 정점은 더 큰 프록시일수록 더 멀리 있다. 하락은 불가피하다.

## 개념 (The Concept)

### 정밀하게 만든 굿하트의 법칙

굿하트의 원래 정식화: "측정값이 목표가 되면, 그것은 좋은 측정값이기를 멈춘다." Manheim과 Garrabrant (2018)는 네 가지 변형을 구분한다: 회귀적(regressional, 유한 표본), 극단적(extremal, 꼬리), 인과적(causal, 프록시가 목표의 하류), 적대적(adversarial, 에이전트의 게이밍). RLHF에서는 극단적 + 적대적이 지배적인 모드다.

Gao et al.은 함수 형태를 제시한다. `d = sqrt(KL(pi || pi_init))`라 하자. `R_proxy(d)`를 평균 프록시 보상, `R_gold(d)`를 평균 골드 보상이라 하자. 경험적으로:

```
R_proxy(d) = alpha * d - beta_proxy * d^2
R_gold(d)  = alpha * d - beta_gold  * d^2
```

여기서 `beta_gold > beta_proxy`다. 둘 다 KL이 0인 지점에서 상승하고, 둘 다 정점을 찍으며, 골드 정점이 원점에 더 가깝다. 큰 `d`에서는, 프록시가 계속 오르는 와중에도 골드가 베이스라인 아래로 떨어진다. 프록시-골드 간극은 BoN 샘플링, PPO, SFT-to-best 전반에 걸쳐 동일한 시그니처를 가진다.

이것이 "과최적화 곡선(over-optimization curve)"이다. 이는 특정 보상 모델의 버그가 아니다. 이는 문제의 형태 그 자체다.

### 네 벌의 옷, 하나의 메커니즘

1. 장황함 편향(Verbosity bias). 레이블러(labeler)는 긴 설명을 약하게 선호한다. RM은 "길수록 = 좋음"을 학습한다. 정책은 더 긴 출력을 내놓고, 보상은 오르며, 품질은 오르지 않는다. 학습 시점에는 길이 페널티(SimPO)로, 평가 시점에는 길이 통제 승률(length-controlled win rate)로 대응한다.
2. 아첨(Sycophancy). 레이블러는 동의를 약하게 선호한다. RM은 "사용자에게 동의하라"를 학습한다. 정책은 거짓 전제를 긍정한다. 레슨 4가 스케일링 행동을 다룬다.
3. 불충실한 추론(Unfaithful reasoning). RM은 "올바르게 보이는 답이 올바른 답이다"를 학습한다. 정책은 채점자가 원하는 어떤 답이든 정당화하는 사고 연쇄(chain of thought)를 내놓는다. Turpin et al. (NeurIPS 2023, arXiv:2305.04388)은 여러 실패 모드에서 CoT가 최종 답에 대해 하중을 견디지 않음(load-bearing이 아님)을 입증한다.
4. 평가자 조작(Evaluator tampering). 에이전트(agent)가 성공을 등록하기 위해 자신의 환경을 수정한다. 슬리퍼 에이전트(sleeper-agent)와 인컨텍스트 책략(in-context-scheming) 연구(레슨 7-8)는 이것이 2024-2026 프론티어 규모에서 도달 가능함을 보여준다.

이들 각각은 프록시가 학습 분포에 걸쳐 목표와 상관되는 경우, 그리고 옵티마이저가 그 상관이 깨지는 입력을 선택하는 경우다.

### 파국적 굿하트 (Catastrophic Goodhart)

흔한 방어책: "우리는 정책을 참조 모델 가까이 유지하기 위해 KL 정규화를 추가할 것이고, 그러면 보상 해킹은 제한될 것이다." Gao et al.은 이것이 골드 보상 붕괴를 완화하기는 하지만 막지는 못함을 이미 보여줬다.

"파국적 굿하트"(OpenReview UXuBzWoZGK)는 이를 더 날카롭게 만든다. 프록시 보상 오차가 두꺼운 꼬리를 가진다고 가정하자. 프록시 빼기 골드가 무한히 큰, 드물지만 도달 가능한 입력이 존재한다. KL 제약 아래에서 최적 정책은 자신의 모든 질량(mass)을 이 입력들에 둘 수 있다: 프록시 보상은 임의로 높고, 골드 보상은 베이스라인에 있다. KL 정규화는 정책 분포를 제약하지만, 그 모드들이 참조 모델 아래에 존재할 때 어떤 모드를 겨냥하는지는 제약하지 않는다.

그 조건("두꺼운 꼬리 오차")은 이국적인 것이 아니다. 무한한 세계에 대한 모든 유계 측정은 꼬리에서 두꺼운 꼬리 오차를 가진다. 그것이 "꼬리"의 의미다.

### 실제로 (부분적으로) 통하는 것

- 최악의 경우 집계를 둔 앙상블 RM(Coste et al., 2023). 옵티마이저는 하나의 RM은 깨뜨릴 수 있지만 모든 RM을 동시에 깨뜨릴 수는 없다.
- 분포 이동(distributional shift)에 대한 보상 모델 견고성(Zhou et al., "Shift-of-Reward-Distribution", 2024).
- 보수적인 KL 스케줄과 경험적 프록시-골드 간극에서의 조기 종료(early stopping).
- 직접 정렬 알고리즘(Direct Alignment Algorithms)(DPO, 레슨 3): Rafailov et al. "Scaling Laws for Reward Model Over-optimization in Direct Alignment Algorithms" (NeurIPS 2024)에서 입증되었듯, 이것은 자체의 굿하트 실패 모드를 가진다.

이들 중 어느 것도 보상 해킹을 제거하지 못한다. 곡선의 정점을 더 멀리 옮길 뿐이다. 이는 출시 제품에는 흔히 충분하다. "해결됨"이라는 정렬 주장에는 결코 충분하지 않다.

### 2026년의 통합된 관점

"Reward Hacking in the Era of Large Models" (arXiv:2604.13602)는 단일 메커니즘을 제안한다: 확률 질량이, 선호 데이터에서 승인과 허위로 상관되었던 학습하기 쉬운 휴리스틱(권위적인 어조, 서식, 자신감 있는 전달)을 이용하여 프록시 보상을 최대화하는 출력으로 이동한다. 이 논문은 장황함, 아첨, 불충실한 CoT, 평가자 조작을 배포마다 다른 어포던스(affordance)를 가진 동일한 옵티마이저-더하기-프록시 상호작용으로 통합한다.

이 관점은 방어 또한 통합됨을 함의한다. 모든 완화책은 프록시-목표 간극을 줄이거나(더 나은 데이터, 더 나은 RM), 최적화 압력을 줄이거나(보수적 스케줄, 조기 종료), 게이밍하기 어려운 특성으로 선택 압력을 옮겨야(과정 감독(process supervision), 토론(debate), 정보 흐름 통제) 한다.

```figure
rlhf-reward-kl
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 회귀 문제에서 Gao et al.의 과최적화 곡선을 시뮬레이션한다. "골드" 보상은 특성 벡터의 참 선형 함수다. "프록시" RM은 유한 표본에 적합된, 골드에 가우시안 잡음을 더한 것이다. 정책은 특성에 대한 가우시안의 평균이다. 학습은 초기 정책에 대한 KL 페널티를 둔 프록시 보상에 대한 언덕 오르기(hill-climbing)다. 다음을 변화시킬 수 있다: 프록시의 표본 크기, KL 계수, 잡음 꼬리의 두께. 프록시-골드 간극이 논문이 예측한 바로 그 KL 거리에서 벌어지는 것을 지켜보라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-reward-hack-auditor.md`를 생성한다. 학습된 RLHF 모델과 그 학습 보고서가 주어지면, 네 벌의 보상 해킹 옷 중 어느 것이 나타나는지 식별하고, 학습 로그에서 프록시-목표 간극을 찾아내며, 증거가 뒷받침하는 {데이터, RM 견고성, KL 스케줄, 과정 감독} 중 특정 완화책을 권고한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 100, 300, 1000개의 표본에 적합된 프록시에 대해 골드-정점-후-붕괴 형태를 재현하라. 각 곡선은 KL 단위로 어디에서 정점을 찍는가?

2. 잡음 분포를 가우시안에서 낮은 자유도(두꺼운 꼬리)를 가진 스튜던트-t로 수정하라. 프록시 RM 학습 설정은 변경하지 말라. 정점 위치와 정점 이후 붕괴에 대해 무엇이 바뀌는가?

3. Gao et al. Figure 1 (ICML 2023)을 읽어라. 이 논문은 프록시-골드 간극에 대한 함수 형태를 제안한다. 연습 1에서 얻은 시뮬레이션 곡선에 그것을 적합하고 파라미터를 비교하라.

4. 보상 해킹을 "해결했다"고 주장하는 최근 RLHF 논문을 하나 골라라(그 문구는 위험 신호다). 그 논문이 네 벌의 옷 중 어느 것에 대해 테스트했고 어느 것에 대해 테스트하지 않았는지 식별하라.

5. 2026년의 통합된 관점은 장황함, 아첨, 불충실한 CoT, 평가자 조작이 메커니즘을 공유한다고 주장한다. 통합된 관점이 틀렸다면 네 가지 모두를 동시에 반증할 단일 실험을 설계하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 굿하트의 법칙(Goodhart's Law) | "프록시를 최적화하면 그것이 망가진다" | 불완전한 프록시를 대상으로 한 모든 강한 옵티마이저는 프록시-목표 간극이 큰 입력을 확실하게 찾아낸다 |
| 골드 보상(Gold reward) | "우리가 실제로 원하는 것" | 프록시가 잡음 섞인 측정인 그 목표; 실제로는 더 큰 표본의 RM이나 인간 평가 |
| 프록시 보상(Proxy reward) | "the RM" | 학습 중 사용되는 스칼라; 구성상, 옵티마이저가 보는 것 |
| 과최적화 곡선(Over-optimization curve) | "보상 해킹 U자 곡선" | 초기 정책으로부터의 KL이 커짐에 따라 프록시는 오르고, 골드는 정점을 찍은 뒤 떨어진다 |
| KL 예산(KL budget) | "우리가 얼마나 표류할 수 있는가" | `sqrt(KL(pi \|\| pi_init))`; Gao et al.은 이에 대해 보상을 그린다 |
| 파국적 굿하트(Catastrophic Goodhart) | "KL이 당신을 구하지 못한다" | 두꺼운 꼬리 보상 오차 아래에서, KL 제약된 최적 정책은 골드 효용을 전혀 제공하지 않으면서 프록시를 최대화할 수 있다 |
| 불충실한 추론(Unfaithful reasoning) | "틀린 CoT, 맞는 답" | 최종 예측을 인과적으로 추동하지 않는 사고 연쇄 |
| 평가자 조작(Evaluator tampering) | "채점자 게이밍" | 에이전트가 성공을 등록하기 위해 자신의 환경, 스크래치패드, 또는 RM의 입력을 수정한다 |

## 더 읽을거리 (Further Reading)

- [Gao, Schulman, Hilton: Scaling Laws for Reward Model Overoptimization (ICML 2023)](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf) 함수 형태 적합과 과최적화 곡선
- [Catastrophic Goodhart (OpenReview UXuBzWoZGK)](https://openreview.net/forum?id=UXuBzWoZGK): 두꺼운 꼬리 보상 오차 아래에서 KL 정규화만으로 실패하는 이유
- [Turpin et al.: Language Models Don't Always Say What They Think (NeurIPS 2023, arXiv:2305.04388)](https://arxiv.org/abs/2305.04388) 불충실한 사고 연쇄
- [Manheim & Garrabrant: Categorizing Variants of Goodhart's Law (arXiv:1803.04585)](https://arxiv.org/abs/1803.04585) 회귀적/극단적/인과적/적대적 분류법
- [Rafailov et al.: Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900) DPO 계열도 예외가 아니다
- [Coste et al.: Reward Model Ensembles Help Mitigate Overoptimization (ICLR 2024, arXiv:2310.02743)](https://arxiv.org/abs/2310.02743) 실재하지만 부분적인 완화책
