# 스케일링 법칙(Scaling Laws)

> 2020년 Kaplan 논문은 말했다: 더 큰 모델, 더 낮은 손실(loss). 2022년 Hoffmann 논문은 말했다: 학습(training)을 덜 시키고 있었다. 연산은 두 양동이로 들어간다. 파라미터(parameter)와 토큰(token): 그리고 그 분배는 자명하지 않다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45분

## 문제 (The Problem)

학습 연산 C FLOPs를 가지고 최고의 모델(model)을 원할 때, 두 개의 손잡이를 마주한다:

1. **파라미터(N)는 몇 개?** 더 큰 모델, 더 높은 용량(capacity).
2. **학습 토큰(D)은 몇 개?** 더 많은 데이터, 용량을 더 잘 사용.

FLOPs는 대략 `6 × N × D`로 스케일한다. N을 올리고 D를 내릴 수도, D를 올리고 N을 내릴 수도 있다. 어느 쪽이 나은가?

2022년 이전에는 답이 "N을 세게 밀어라"였다. GPT-3(2020)은 ~300B 토큰으로 학습된 175B 파라미터였다. 파라미터당 약 1.7 토큰의 비율이다. Kaplan 스케일링 법칙이 이를 뒷받침했다.

Hoffmann et al. (2022)는 Chinchilla라 불리는 작은 모델군을 학습하면서 다른 것을 발견했다: 최적 비율은 **파라미터당 20 토큰**에 가깝다. GPT-3은 10배 학습이 부족했다. Chinchilla(70B 파라미터, 1.4T 토큰)는 추론(inference) 비용 2.5배 절감과 함께 모든 벤치마크(benchmark)에서 GPT-3(175B, 300B 토큰)을 이겼다.

2026년은 Chinchilla의 세계다. 단, 한 가지 중요한 반전이 있다. Llama 3 8B는 15조 토큰으로 학습되었는데, 파라미터당 1,875 토큰의 비율이다. Chinchilla-최적의 94배를 넘어선다. 대규모로 사용될 모델의 경우 추론 비용이 학습 비용보다 더 중요하므로, 더 작은 배포 가능 풋프린트(footprint)를 위한 과학습(over-training, Chinchilla 너머)이 2026년의 기본값이다.

## 개념 (The Concept)

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann 법칙

Chinchilla 논문에 따르면, 손실은 다음을 따른다:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 파라미터(임베딩(embedding) 제외).
- `D` = 학습 토큰.
- `α ≈ 0.34`, `β ≈ 0.28`(대략 대칭(symmetry)).
- `E ≈ 1.69`, 줄일 수 없는 손실 상한.
- `A ≈ 406`, `B ≈ 411`.

스케일을 키울수록 두 항이 서로 트레이드오프(trade-off)한다. 고정 연산(C = 6ND)에서 `N`에 대한 도함수(derivative)를 취하고 풀면:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

연산 최적(compute-optimal): 파라미터당 20 토큰.

### 그래도 왜 과학습하는가

Chinchilla-최적은 학습 FLOP당 학습 손실을 최소화한다. 하지만 학습 비용은 한 번 치르고, 추론 비용은 영원히 치른다.

월 1조 토큰을 서빙(serving)하는 챗봇의 경우, 추론이 총비용을 지배한다. Llama의 접근: 더 작게, 더 길게 학습한다. 15T 토큰의 8B는 깊이 추론 최적화되어 있다:

- 소비자용 GPU에 들어간다.
- 지연 시간(latency)이 70B Chinchilla-최적의 일부에 불과하다.
- 대부분의 작업에 충분히 가까운 품질이다.

DeepMind의 2024년 논문("Over-training is the new optimal")이 이를 정식화했다. 추론이 지배하는 워크로드의 경우, 올바른 비율은 서빙 볼륨에 따라 파라미터당 100~500 토큰에 가깝다.

### 출현(emergence) vs 매끄러움(smoothness)

주장: 특정 능력(산술, 다단계 추론, 사고 사슬(chain-of-thought) 따르기)이 어떤 규모에서 갑자기 "출현"한다.

Schaeffer et al. (2023)는 이것이 측정 인공물(measurement artifact)이라고 주장했다: 출현 지표는 불연속적 채점(정확 일치(exact match), 임계값 정확도)을 사용하는데, 이는 기저 로짓(logit)의 매끄러운 개선을 가린다. 연속적 지표(교차 엔트로피(cross-entropy))는 매끄러운 곡선을 보인다.

2026년의 합의는: 연속 손실을 통한 예측은 신뢰할 수 있다. 벤치마크 점프는 종종 채점기 인공물이다. 예산은 연속 지표에 맞춰 계획하라.

### 2026년 그림

스케일링 법칙은 여전히 작동하지만:

| 요인 | 어떻게 바뀌었나 |
|--------|-------------|
| 데이터 품질 | "좋은" 토큰을 큐레이션(Phi 스타일)하면 유효 연산이 >2배 이동한다 |
| MoE | 전체 파라미터가 활성 FLOPs와 분리된다; 활성 FLOP당 스케일링 법칙 |
| 사후 학습(post-training) | 일부 능력(명령어 따르기, 코드)은 사전 학습(pretraining)보다 SFT+RLHF로 더 이동한다 |
| 멀티모달리티(multimodality) | 이미지 + 텍스트 토큰이 함께 스케일한다; 모달리티(modality)별로 별도 곡선 |
| 합성 데이터(synthetic data) | 모델이 학습 데이터를 생성한다; 유효 연산이 복리로 늘 수 있다 |

Muon 옵티마이저(optimizer)(Kimi Moonlight, 2024)는 동일 데이터에서 AdamW 대비 ~2배의 유효 연산 이득을 보였다. 일부 2026년 학습 실행은 기본적으로 Muon을 쓴다. 스케일링 법칙의 모양이 아니라 절대 상수를 바꾼다.

```figure
scaling-laws
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. Chinchilla 손실 방정식을 구현하고, 여러 연산 예산 각각에서 연산 최적 `(N, D)`를 푼다.

### 1단계: Chinchilla 손실

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

고정 `C = 6ND`에서 `L`을 `(N, D)`에 대한 등고선(contour)으로 그린다. 최솟값을 찾는다.

### 2단계: 연산 최적 프런티어(frontier)

`1e17`부터 `1e25` FLOPs까지의 연산 예산에 대해, `6ND = C` 제약하에서 손실을 최소화하는 `(N, D)`를 찾는다. 비율 `D/N ≈ 20`을 확인한다.

### 3단계: 과학습 비용

10배 작은 모델(최적 N의 1/10, 최적 D의 10배)을 학습하기 위해 치르는 추가 손실을 계산한다. 그 대가로 얻는 추론 FLOP 절감(N에 비례)을 보고한다.

### 4단계: 실제 모델과 비교

GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3(활성 파라미터)에 대해 알려진 `(N, D)` 쌍을 넣고, 예측 손실과 보고된 손실을 비교한다.

## 라이브러리로 써보기 (Use It)

직접 프런티어 모델을 학습할 가능성은 낮다. 하지만 스케일링 법칙은 다음을 알려 준다:

1. **파인튜닝(fine-tuning)에 충분한 데이터가 있는가.** 작업 전용 데이터가 베이스 모델 파라미터당 20 토큰 미만이라면, 어떤 손실 바닥에서의 포화(saturation)를 예상하라.
2. **더 큰 베이스 모델을 골라야 하는가.** 예산을 전부 추론에 쓰고 있다면, 더 작고 더 길게 학습된 모델을 선호하라.
3. **수익이 어디서 감소하는가.** Chinchilla-최적의 1000배를 넘어서면, 로그 손실 변화는 잡음이 된다.

**2026년 연구 궤적:**

- **데이터 제약 영역.** 웹에는 유한한 수의 고품질 토큰(필터링 후 영어 ~5~10조)만 있다. 프런티어 사전 학습은 이 상한에 접근하고 있다. 합성 데이터, 다국어, 멀티모달, RLHF로 스케일된 파인튜닝이 다음 손잡이다.
- **연산 배수(compute-multiplier) 트릭.** Muon 옵티마이저, MoE, 더 나은 데이터 큐레이션: 각각 절대 상수를 이동시킬 뿐 점근선(asymptote)을 바꾸지는 않는다.
- **RL을 위한 스케일링 법칙.** 미해결 문제. 초기 증거는 RL 샘플에서 거듭제곱 법칙(power-law)을 시사하지만, 사전 학습과는 매우 다른 지수(exponent)를 보인다.

## 산출물 (Ship It)

`outputs/skill-training-budget-estimator.md`를 참고하라. 이 스킬은 연산 예산, 배포(deployment) 제약, 목표 손실이 주어진 새 학습 실행에 대해 `(N, D, hours, GPU)`를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 연산 예산 `1e20`, `1e22`, `1e24`에 대한 Chinchilla-최적 `(N, D)`를 출력하라. 실제 모델 표와 비교하라.
2. **중간.** Hoffmann 손실-연산-함수 곡선을 구현하라. 연산 최적 프런티어에 대해 손실 vs `log10(C)`를 그려라. 다음 0.1의 교차 엔트로피 감소에 `>10^28` FLOPs가 필요하다고 법칙이 예측하는 지점을 식별하라.
3. **어려움.** 같은 데이터셋(dataset)으로 학습한 5개의 작은 모델(100K~10M 파라미터)에 대해 자신만의 스케일링 법칙을 적합(fit)하라. `α`와 `E`를 추정하라. 적합한 지수가 발표된 값과 얼마나 잘 맞는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 파라미터(Parameters, N) | "모델 크기" | 임베딩 제외 가중치(weight) 수; 용량을 결정한다. |
| 토큰(Tokens, D) | "학습 데이터" | 본 학습 토큰 수; 파라미터가 얼마나 잘 사용되는지를 결정한다. |
| 연산(Compute, C) | "쓴 FLOPs" | 표준 트랜스포머(transformer)의 경우 대략 `6 × N × D`. |
| Chinchilla-최적(Chinchilla-optimal) | "D/N ≈ 20" | 사전 학습 FLOP당 손실을 최소화하는 비율. |
| 과학습(Over-training) | "Chinchilla 너머" | 추론 FLOPs를 아끼기 위해 추가 학습 FLOPs를 쓰는 것; D/N >> 20. |
| 줄일 수 없는 손실(Irreducible loss) | "바닥" | 스케일링 법칙의 `E` 항; 데이터 자체의 엔트로피. |
| 출현 능력(Emergent capability) | "규모에서의 갑작스러운 점프" | 종종 채점기 인공물; 연속 손실은 매끄럽다. |
| 유효 연산(Effective compute) | "학습 효율 배수" | 더 나은 데이터 / 옵티마이저 / 아키텍처가 FLOP의 도달 거리를 배가한다. |

## 더 읽을거리 (Further Reading)

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361): 최초의 스케일링 법칙 논문; 학습 부족.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004): 측정 인공물로서의 출현.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448): Llama의 과학습이 그 워크로드에 옳은 이유.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/): 2배 연산 배수.
