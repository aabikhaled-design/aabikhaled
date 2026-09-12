# 근접 정책 최적화 (Proximal Policy Optimization, PPO)

> A2C는 각 롤아웃을 한 번 갱신한 뒤 버린다. PPO는 정책 그래디언트(policy gradient)를 클리핑된 중요도 비율(importance ratio)로 감싸서 정책이 폭발하지 않으면서 같은 데이터로 10번 이상의 에폭(epoch)을 돌릴 수 있게 한다. Schulman et al. (2017). 2026년에도 여전히 기본 정책 그래디언트 알고리즘이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75분

## 문제 (The Problem)

A2C(Lesson 07)는 온-폴리시(on-policy)다: 그래디언트 `E_{π_θ}[A · ∇ log π_θ]`는 *현재* `π_θ`로부터 샘플링된 데이터를 요구한다. 한 번 갱신하면 `π_θ`가 바뀌고, 방금 쓴 데이터는 이제 오프-폴리시(off-policy)다. 이 데이터를 재사용하면 그래디언트가 편향된다.

롤아웃은 비싸다. Atari에서 8개 환경 × 128스텝에 걸친 한 롤아웃 = 1024 전이와 십수 초의 환경 시간이다. 그래디언트 스텝 한 번 뒤에 이를 버리는 것은 낭비다.

신뢰 영역 정책 최적화(Trust Region Policy Optimization, TRPO, Schulman 2015)가 첫 번째 해법이었다: 옛 정책과 새 정책 사이의 KL 발산(divergence)이 `δ` 아래에 머물도록 각 갱신을 제약한다. 이론적으로는 깔끔하지만 갱신마다 켤레 그래디언트(conjugate-gradient) 풀이가 필요하다. 2026년에 아무도 TRPO를 실행하지 않는다.

PPO(Schulman et al. 2017)는 단단한 신뢰 영역 제약을 단순한 클리핑된 목적함수로 교체한다. 코드 한 줄 추가. 롤아웃당 10 에폭. 켤레 그래디언트 없음. 충분히 좋은 이론적 보장. 9년 뒤에도 여전히 MuJoCo에서 RLHF까지 모든 것의 기본 정책 그래디언트 알고리즘이다.

## 개념 (The Concept)

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**중요도 비율.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

이것은 데이터를 수집한 정책 대비 새 정책의 우도비(likelihood ratio)다. `r_t = 1`은 변화 없음을 뜻한다. `r_t = 2`는 새 정책이 옛 정책보다 `a_t`를 취할 확률이 두 배라는 뜻이다.

**클리핑된 대리 목적함수(clipped surrogate).**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

두 항:

- 어드밴티지(advantage) `A_t > 0`이고 비율이 `1 + ε`를 넘어 커지려 하면, 클립이 그래디언트를 평평하게 만든다. 좋은 행동을 옛 확률보다 `+ε` 이상으로 밀지 마라.
- 어드밴티지 `A_t < 0`이고 비율이 `1 - ε`를 넘어 커지려 하면(즉, 나쁜 행동을 그 클리핑된 감소 대비 더 가능성 있게 만들려 하면), 클립이 그래디언트를 제한한다. 나쁜 행동을 `-ε` 아래로 밀지 마라.

`min`은 다른 방향을 처리한다: 비율이 *유익한* 방향으로 움직였다면 여전히 그래디언트를 얻는다(해가 될 쪽에만 클리핑이 없다).

전형적 `ε = 0.2`. 목적함수를 `r_t`의 함수로 그려보라: "좋은 쪽"에 평평한 지붕, "나쁜 쪽"에 평평한 바닥을 가진 조각별 선형 함수다.

**전체 PPO 손실(loss).**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2C와 같은 액터-크리틱(actor-critic) 구조. 세 계수, 보통 `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**학습 루프.**

1. `N`개의 병렬 환경에 걸쳐 각각 `T`스텝씩 `N × T` 전이를 수집한다.
2. 어드밴티지(GAE)를 계산하고, 상수로 동결한다.
3. `π_{θ_old}`를 현재 `π_θ`의 스냅샷으로 동결한다.
4. `K` 에폭 동안, `(s, a, A, V_target, log π_old(a|s))`의 각 미니배치(mini-batch)에 대해:
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`를 계산한다.
   - `L^{CLIP}` + 가치 손실 + 엔트로피를 적용한다.
   - 그래디언트 스텝.
5. 롤아웃을 폐기한다. 1단계로 돌아간다.

`K = 10`과 64개의 미니배치가 표준 하이퍼파라미터(hyperparameter) 세트다. PPO는 견고하다: 정확한 숫자는 ±50% 이내에서 거의 중요하지 않다.

**KL-페널티 변형.** 원조 논문은 적응적 KL 페널티를 사용하는 대안을 제안했다: 관측된 KL에 기반해 `β`를 조정하는 `L = L^{PG} - β · KL(π_θ || π_old)`. 클리핑 버전이 지배적이 되었다; KL 변형은 RLHF에서 살아남는다(거기서 참조 정책에 대한 KL은 어차피 항상 원하는 별도의 제약이다).

```figure
ppo-clip
```

## 직접 만들기 (Build It)

### 1단계: 롤아웃 시점에 `log π_old(a | s)`를 포착

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

스냅샷은 롤아웃 시점에 한 번 찍힌다. 갱신 에폭 동안 바뀌지 않는다.

### 2단계: GAE 어드밴티지 계산 (Lesson 07)

A2C와 동일. 배치 전체에 걸쳐 정규화(normalization)한다.

### 3단계: 클리핑된 대리 목적함수 갱신

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"클리핑됨 → 그래디언트 0" 패턴이 PPO의 심장이다. 새 정책이 이미 유익한 방향으로 너무 멀리 표류했다면 갱신이 멈춘다.

### 4단계: 가치와 엔트로피

A2C와 동일하게, 크리틱 타깃에 표준 MSE를, 액터에 엔트로피 보너스를 추가한다.

### 5단계: 진단

매 갱신마다 지켜볼 세 가지:

- **평균 KL** `E[log π_old - log π_θ]`. `[0, 0.02]`에 머물러야 한다. `0.1`을 넘어 폭발하면 `K_EPOCHS`나 `LR`을 줄여라.
- **클립 비율(clip fraction)**: 비율이 `[1-ε, 1+ε]` 밖에 있는 샘플의 비율. `~0.1-0.3`이어야 한다. `~0`이면 클립이 결코 발동하지 않는 것이니 `LR`이나 `K_EPOCHS`를 올려라. `~0.5+`이면 롤아웃에 과적합(overfitting)하는 것이니 둘 다 낮춰라.
- **설명된 분산(explained variance)** `1 - Var(V_target - V_pred) / Var(V_target)`. 크리틱 품질 지표. 크리틱이 학습하면서 1을 향해 올라가야 한다.

## 함정 (Pitfalls)

- **클립 계수 오조정.** `ε = 0.2`가 사실상의 표준이다. `0.1`로 가면 갱신이 너무 소심해지고; `0.3+`는 불안정을 부른다.
- **너무 많은 에폭.** `K > 20`이면 정책이 `π_old`에서 멀리 표류하기 때문에 흔히 불안정해진다. 특히 큰 네트워크에서는 에폭에 한계를 두어라.
- **보상 정규화 없음.** 큰 보상 스케일은 클립 범위를 잠식한다. 어드밴티지를 계산하기 전에 보상을 정규화하라(실행 표준편차).
- **어드밴티지 정규화 잊기.** 배치당 평균 0/단위 표준편차 정규화가 표준이다. 그것을 건너뛰면 대부분의 벤치마크(benchmark)에서 PPO가 망가진다.
- **학습률을 감쇠하지 않음.** PPO는 0으로의 선형 LR 감쇠에서 이득을 본다. 상수 LR은 종종 더 나쁘다.
- **중요도 비율 수학 오류.** 수치 안정성을 위해 항상 `new / old`가 아니라 `exp(log_new - log_old)`를 쓰라.
- **잘못된 그래디언트 부호.** 대리 목적함수 최대화 = `-L^{CLIP}` *최소화*. 뒤집힌 부호가 가장 흔한 PPO 버그다.

## 라이브러리로 써보기 (Use It)

PPO는 놀라울 정도로 많은 도메인에서 2026년의 기본 강화 학습(reinforcement learning) 알고리즘이다:

| 사용 사례 | PPO 변형 |
|----------|-------------|
| MuJoCo / 로보틱스 제어 | 가우시안 정책, GAE(0.95)를 가진 PPO |
| Atari / 이산 게임 | 범주형 정책, 롤링 128-스텝 롤아웃을 가진 PPO |
| LLM용 RLHF | 참조 모델에 대한 KL 페널티, 응답 끝의 RM 보상을 가진 PPO |
| 대규모 게임 에이전트 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 추론 LLM | GRPO (Lesson 12): 크리틱 없는 PPO 변형 |
| 선호도만 있는 데이터 | DPO: PPO+KL의 닫힌 형식 압축, 온라인 샘플링 없음 |

PPO *손실 형태*(클리핑된 대리 목적함수 + 가치 + 엔트로피)는 DPO, GRPO, 나아가 거의 모든 RLHF 파이프라인(pipeline)의 골격이다.

## 산출물 (Ship It)

`outputs/skill-ppo-trainer.md`로 저장하라:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 연습 문제 (Exercises)

1. **쉬움.** `ε=0.2, K=4`로 4×4 GridWorld에서 PPO를 실행하라. 일치하는 환경 스텝에서 A2C(롤아웃당 한 에폭)와 표본 효율을 비교하라.
2. **중간.** `K ∈ {1, 4, 10, 30}`을 스윕(sweep)하라. 환경 스텝 대 리턴을 그리고 갱신당 평균 KL을 추적하라. 이 작업에서 어떤 `K`에서 KL이 폭발하는가?
3. **어려움.** 클리핑된 대리 목적함수를 적응적 KL 페널티(`KL > 2·target`이면 `β`를 두 배, `KL < target/2`이면 절반)로 교체하라. 최종 리턴, 안정성, 클립-없음을 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 중요도 비율 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; 데이터를 수집한 정책으로부터의 이탈. |
| 클리핑된 대리 목적함수 | "PPO의 주요 트릭" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; 유익한 쪽에서 클립을 넘으면 평평한 그래디언트. |
| 신뢰 영역 | "TRPO / PPO 의도" | 단조 개선을 보장하기 위해 각 갱신의 KL을 제한. |
| KL 페널티 | "소프트 신뢰 영역" | 대안 PPO: `L - β · KL(π_θ \|\| π_old)`. 적응적 `β`. |
| 클립 비율 | "클리핑이 얼마나 자주 발동하는지" | 진단: 0.1-0.3이어야 함; 벗어나면 오조정. |
| 다중 에폭 학습 | "데이터 재사용" | 각 롤아웃에 K 에폭; 표본 효율을 위해 분산 비용을 거래. |
| 거의-온-폴리시 | "대체로 온-폴리시" | PPO는 명목상 온-폴리시지만 K>1 에폭은 약간 오프-폴리시인 데이터를 안전하게 사용. |
| PPO-KL | "다른 PPO" | KL-페널티 변형; KL-대-참조가 이미 제약인 RLHF에서 사용. |

## 더 읽을거리 (Further Reading)

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): 그 논문.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477): TRPO, PPO의 선구자.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990): 모든 PPO 하이퍼파라미터를 절제(ablation)함.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155): InstructGPT; RLHF에서의 PPO 레시피.
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) PyTorch를 곁들인 명료한 현대적 설명.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl): 많은 논문이 사용하는 참조 단일 파일 PPO.
- [Hugging Face TRL: PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) 언어 모델에서 PPO의 프로덕션 레시피; Lesson 09 (RLHF)와 함께 읽어라.
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729): "37가지 코드 수준 최적화" 논문; 어떤 PPO 트릭이 하중을 견디고 어떤 것이 민간 전승인지.
