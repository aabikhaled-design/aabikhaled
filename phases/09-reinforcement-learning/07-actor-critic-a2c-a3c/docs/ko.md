# 액터-크리틱(A2C와 A3C (Actor-Critic) A2C and A3C)

> REINFORCE는 노이즈가 심하다. `V̂(s)`를 학습하는 크리틱(critic)을 추가하고 리턴에서 이 값을 빼면, 기댓값은 그대로지만 분산은 훨씬 낮은 어드밴티지(advantage)를 얻는다. 이것이 액터-크리틱(actor-critic)이다. A2C는 동기적으로 실행하고, A3C는 스레드에 걸쳐 실행한다. 둘 다 모든 현대 심층 강화 학습(reinforcement learning) 방법의 사고 모델이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75분

## 문제 (The Problem)

바닐라 REINFORCE는 작동하지만 분산이 끔찍하다. 몬테카를로(Monte Carlo) 리턴 `G_t`는 에피소드마다 10배까지 출렁인다. 이 노이즈에 `∇ log π`를 곱하고 평균을 내면, DQN 갱신 몇 번이면 끝낼 정책 이동을 수천 에피소드에 걸쳐서야 해내는 그래디언트(gradient) 추정기가 나온다.

분산은 원본 리턴을 사용하는 데서 온다. 베이스라인(baseline) `b(s_t)`(학습된 가치를 포함한 임의의 상태 함수)를 빼면, 기댓값은 변하지 않고 분산은 떨어진다. 가장 다루기 쉬운 최선의 베이스라인은 `V̂(s_t)`다. 이제 `∇ log π`에 곱해지는 양은 *어드밴티지*다:

`A(s, a) = G - V̂(s)`

행동이 평균 이상의 리턴을 냈다면 좋고, 이하라면 나쁘다. 학습된 크리틱을 더한 REINFORCE가 *액터-크리틱*이다. 크리틱은 액터(actor)에게 분산이 낮은 교사 신호 역할을 한다. 이것이 2015년 이후의 모든 심층 정책 방법(A2C, A3C, PPO, SAC, IMPALA)이다.

## 개념 (The Concept)

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**두 네트워크, 하나의 공유 손실:**

- **액터** `π_θ(a | s)`: 정책. 행동하기 위해 샘플링됨. 정책 그래디언트(policy gradient)로 학습.
- **크리틱** `V_φ(s)`: 상태로부터의 기대 리턴을 추정. `(V_φ(s) - target)²`를 최소화하도록 학습.

**어드밴티지.** 두 가지 표준 형태:

- *MC 어드밴티지:* `A_t = G_t - V_φ(s_t)`. 불편(unbiased), 더 높은 분산.
- *TD 어드밴티지:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. 편향(`V_φ`를 사용), 훨씬 낮은 분산. *TD 잔차(TD residual)* `δ_t`라고도 불린다.

**n-스텝 어드밴티지.** 둘 사이를 보간한다:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`은 순수 TD다. `n = ∞`는 MC다. 대부분의 구현은 Atari에 `n = 5`를, MuJoCo의 PPO에 `n = 2048`을 사용한다.

**일반화 어드밴티지 추정(Generalized Advantage Estimation, GAE).** Schulman et al. (2016)은 모든 n-스텝 어드밴티지에 대한 지수 가중 평균을 제안했다:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

`λ ∈ [0, 1]`로. `λ = 0`은 TD(저분산, 고편향)다. `λ = 1`은 MC(고분산, 불편)다. `λ = 0.95`가 2026년 기본값이다. 편향/분산 다이얼이 원하는 곳에 올 때까지 조정하라.

**A2C: 동기 어드밴티지 액터-크리틱.** `N`개의 병렬 환경에 걸쳐 `T`스텝을 수집한다. 각 스텝에 대해 어드밴티지를 계산한다. 결합된 배치(batch)에서 액터와 크리틱을 갱신한다. 반복한다. A3C의 더 단순하고 더 확장 가능한 형제다.

**A3C: 비동기 어드밴티지 액터-크리틱.** Mnih et al. (2016). `N`개의 워커(worker) 스레드를 생성하고, 각각 환경을 실행한다. 각 워커는 자신의 롤아웃에서 국소적으로 그래디언트를 계산한 뒤, 공유 파라미터 서버에 비동기적으로 적용한다. 재현 버퍼(replay buffer)가 필요 없다. 워커들은 서로 다른 궤적을 실행함으로써 비상관화된다. A3C는 CPU에서 대규모로 학습할 수 있음을 증명했다. 2026년에는 GPU 기반 A2C(배치된 병렬 환경)가 지배적인데, GPU가 큰 배치를 원하기 때문이다.

**결합 손실.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

세 항: 정책 그래디언트 손실, 가치 회귀, 엔트로피 보너스. `c_v ~ 0.5`, `c_e ~ 0.01`이 정전적 시작점이다.

```figure
actor-critic
```

## 직접 만들기 (Build It)

### 1단계: 크리틱

MSE로 갱신되는 선형 크리틱 `V_φ(s) = w · features(s)`:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

표(tabular) 환경에서 크리틱은 수백 에피소드 만에 수렴한다. Atari에서는 선형 크리틱을 공유 CNN 트렁크 + 가치 헤드로 교체한다.

### 2단계: n-스텝 어드밴티지

길이 `T`의 롤아웃과 부트스트랩된 최종 `V(s_T)`가 주어졌을 때:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`는 크리틱 타깃이다. `advantages`는 `∇ log π`에 곱해지는 것이다.

### 3단계: 결합 갱신

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

온-폴리시(on-policy), 갱신당 하나의 롤아웃, 액터와 크리틱에 별도의 학습률(learning rate).

### 4단계: 병렬화 (A3C 대 A2C)

- **A3C:** `N`개의 스레드를 띄운다. 각각 자신의 환경과 자신의 순방향 패스(forward pass)를 실행한다. 주기적으로 그래디언트 갱신을 공유 마스터에 푸시한다. 마스터에 락이 없다. 경쟁(race)은 괜찮으며, 그저 노이즈를 더할 뿐이다.
- **A2C:** 단일 프로세스에서 `N`개의 환경 인스턴스를 실행하고, 관측을 `[N, obs_dim]` 배치로 쌓고, 배치된 순방향 패스, 배치된 역방향 패스(backward pass). 더 높은 GPU 활용, 결정론적, 추론하기 더 쉬움. 2026년의 기본값.

우리 장난감 코드는 명료성을 위해 단일 스레드다; 배치된 A2C로 다시 쓰는 것은 numpy 세 줄이다.

## 함정 (Pitfalls)

- **액터 그래디언트 이전의 크리틱 편향.** 크리틱이 무작위라면 그 베이스라인은 아무 정보도 주지 못하고, 결국 순수 노이즈로 학습하는 셈이다. 정책 그래디언트를 켜기 전에 크리틱을 수백 스텝 워밍업하거나, 느린 액터 학습률을 사용하라.
- **어드밴티지 정규화.** 어드밴티지를 배치당 평균 0/단위 표준편차로 정규화(normalization)하라. 거의 0의 비용으로 학습을 막대하게 안정화한다.
- **공유 트렁크.** 이미지 입력에서 액터와 크리틱에 공유 특성 추출기를 사용하라. 헤드는 분리한다. 공유 특성은 두 손실 모두에 무임승차한다.
- **온-폴리시 계약.** A2C는 정확히 한 번의 갱신만 데이터를 재사용한다. 그 이상이면 그래디언트가 편향된다(중요도 샘플링 보정이 PPO가 추가하는 것이다).
- **엔트로피 붕괴.** `c_e > 0`이 없으면, 정책이 수백 갱신 만에 거의 결정론적이 되고 탐험을 멈춘다.
- **보상 스케일.** 어드밴티지 크기는 보상 스케일에 의존한다. 작업 간 일관된 그래디언트 크기를 위해 보상을 정규화하라(예: 실행-표준편차로 나누기).

## 라이브러리로 써보기 (Use It)

A2C/A3C가 2026년에 최종 선택이 되는 경우는 드물지만, 이후의 방법들이 모두 다듬어 발전시킨 토대 아키텍처다:

| 방법 | A2C와의 관계 |
|--------|----------------|
| PPO | A2C + 다중 에폭(epoch) 갱신을 위한 클리핑된 중요도 비율 |
| IMPALA | A3C + V-trace 오프-폴리시 보정 |
| SAC (Phase 9 · 07) | 소프트-가치 크리틱을 가진 오프-폴리시 A2C (다음 레슨) |
| GRPO (Phase 9 · 12) | 크리틱 없는 A2C: 그룹 상대 어드밴티지 |
| DPO | 선호도-순위 손실로 압축된 A2C, 샘플링 없음 |
| AlphaStar / OpenAI Five | 리그 학습 + 모방 사전 학습(pretraining)을 가진 A2C |

2026년 논문에서 "어드밴티지"를 보면, 액터-크리틱을 떠올려라.

## 산출물 (Ship It)

`outputs/skill-actor-critic-trainer.md`로 저장하라:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 연습 문제 (Exercises)

1. **쉬움.** 4×4 GridWorld에서 MC 어드밴티지(`G_t - V(s_t)`)로 액터-크리틱을 학습하라. Lesson 06의 실행-평균-베이스라인을 가진 REINFORCE와 표본 효율을 비교하라.
2. **중간.** TD-잔차 어드밴티지(`r + γ V(s') - V(s)`)로 전환하라. 어드밴티지 배치의 분산을 측정하라. 얼마나 떨어지는가?
3. **어려움.** GAE(λ)를 구현하라. `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`을 스윕(sweep)하라. 최종 리턴 대 표본 효율을 그려라. 이 작업에서 편향/분산 최적점은 어디인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 액터 | "정책 망" | `π_θ(a\|s)`, 정책 그래디언트로 갱신. |
| 크리틱 | "가치 망" | `V_φ(s)`, 리턴 / TD 타깃에 대한 MSE 회귀로 갱신. |
| 어드밴티지 | "평균보다 얼마나 나은지" | `A(s, a) = Q(s, a) - V(s)` 또는 그 추정기들. `∇ log π`의 승수. |
| TD 잔차 | "δ" | `δ_t = r + γ V(s') - V(s)`; 한-스텝 어드밴티지 추정. |
| GAE | "보간 손잡이" | `λ`로 파라미터화된, n-스텝 어드밴티지의 지수 가중 합. |
| A2C | "동기 액터-크리틱" | 환경에 걸쳐 배치됨; 롤아웃당 한 번의 그래디언트 스텝. |
| A3C | "비동기 액터-크리틱" | 워커 스레드가 공유 파라미터 서버에 그래디언트를 푸시. 원조 논문; 2026년에는 덜 흔함. |
| 부트스트랩 | "지평에서 V를 사용" | 롤아웃을 절단하고, 합을 닫기 위해 `γ^n V(s_{t+n})`를 더함. |

## 더 읽을거리 (Further Reading)

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783): A3C, 원조 비동기 액터-크리틱 논문.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13: Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) 기초; 크리틱이 신경망일 때 함수 근사에 관한 9장과 짝지어 읽어라.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561): V-trace 오프-폴리시 보정을 가진 확장 가능한 분산 액터-크리틱.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/): 읽을 가치가 있는 프로덕션 A2C/PPO 구현.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms): 두-시간척도 액터-크리틱 분해의 기초적 수렴 결과.
