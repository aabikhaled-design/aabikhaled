# 정책 그래디언트(밑바닥부터 만드는 REINFORCE (Policy Gradient) REINFORCE from Scratch)

> 가치 추정을 멈춰라. 정책을 직접 파라미터화(parameterize)하고, 기대 리턴의 그래디언트(gradient)를 계산하고, 오르막으로 걸음을 옮겨라. Williams (1992)는 이를 정리 하나로 썼다. 이것이 PPO, GRPO, 그리고 모든 LLM 강화 학습(reinforcement learning) 루프가 존재하는 이유다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75분

## 문제 (The Problem)

Q-러닝(Q-learning)과 DQN은 *가치* 함수를 파라미터화한다. `argmax Q`로 행동을 고른다. 이산 행동과 이산 상태에는 괜찮다. 행동이 연속이거나(10차원 토크에 대한 `argmax`는 무엇인가?) 확률적 정책을 원할 때(`argmax`는 구성상 결정론적이다) 깨진다.

정책 그래디언트(policy gradient)는 대신 *정책*을 파라미터화한다. `π_θ(a | s)`는 행동에 대한 분포를 출력하는 신경망(neural network)이다. 여기서 샘플링하여 행동한다. `θ`에 대한 기대 리턴의 그래디언트를 계산한다. 오르막으로 걸음을 옮긴다. `argmax` 없음. 벨만 재귀 없음. 그저 `J(θ) = E_{π_θ}[G]`에 대한 경사 상승(gradient ascent)이다.

REINFORCE 정리(Williams 1992)는 이 그래디언트가 계산 가능함을 알려준다: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. 에피소드를 돌린다. 리턴을 계산한다. 매 스텝에서 `∇ log π_θ(a | s)`를 곱한다. 평균을 낸다. 경사 상승. 끝.

2026년 모든 LLM-강화 학습 알고리즘(PPO, DPO, GRPO)은 REINFORCE의 정제다. 이를 손에 익히는 것이 이 단계의 나머지, 그리고 Phase 10 · 07 (RLHF 구현)과 Phase 10 · 08 (DPO)의 전제조건이다.

## 개념 (The Concept)

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**정책 그래디언트 정리.** `θ`로 파라미터화된 모든 정책 `π_θ`에 대해:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

여기서 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`는 스텝 `t`로부터의 할인 리턴이다. 기댓값은 `π_θ`로부터 샘플링된 전체 궤적(trajectory) `τ`에 대한 것이다.

**증명은 짧다.** 기댓값 하에서 `J(θ) = Σ_τ P(τ; θ) G(τ)`를 미분한다. `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`(로그-도함수 트릭, log-derivative trick)를 사용한다. `log P(τ; θ) = Σ log π_θ(a_t | s_t) + θ에 의존하지 않는 환경 항`으로 인수분해한다. 환경 항은 사라진다. 두 줄의 대수가 정리를 준다.

**분산 감소 트릭.** 바닐라 REINFORCE는 살인적인 분산을 가진다. 리턴은 노이즈가 있고, `∇ log π`도 노이즈가 있으며, 그 곱은 매우 노이즈가 심하다. 두 가지 표준 해법:

1. **베이스라인 차감(baseline subtraction).** `a_t`에 의존하지 않는 임의의 베이스라인(baseline) `b(s_t)`에 대해 `G_t`를 `G_t - b(s_t)`로 교체한다. `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`이므로 불편(unbiased)하다. 전형적 선택: 크리틱(critic)이 학습한 `b(s_t) = V̂(s_t)` → 액터-크리틱(actor-critic, Lesson 07).
2. **리워드-투-고(reward-to-go).** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`를 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`로 교체한다. 주어진 행동에는 미래 리턴만 중요하다. 과거 보상은 평균 0의 노이즈를 기여한다.

둘을 합치면:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

이것이 베이스라인을 가진 REINFORCE이며: A2C(Lesson 07)와 PPO(Lesson 08)의 직접적 조상이다.

**소프트맥스 정책 파라미터화.** 이산 행동에 대한 표준 선택:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

여기서 `f_θ`는 행동별 점수를 출력하는 임의의 신경망이다. 그래디언트는 깔끔한 형태를 가진다:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

즉, 취한 행동의 점수에서 정책 하의 기댓값을 뺀 것이다.

**연속 행동을 위한 가우시안 정책.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)`는 닫힌 형식(closed form)을 가진다. Phase 9 · 07의 SAC에 필요한 것은 그게 전부다.

```figure
policy-gradient-landscape
```

## 직접 만들기 (Build It)

### 1단계: 소프트맥스 정책 네트워크

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

표(tabular) 환경에는 선형 정책(행동당 가중치 벡터 하나)을 사용하라. Atari에는 CNN으로 바꾸고 소프트맥스 헤드를 유지하라.

### 2단계: 샘플링과 로그 확률

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 3단계: 로그 확률을 포착하는 롤아웃

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 4단계: REINFORCE 갱신

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

그래디언트 `∇ log π(a|s) = e_a - π(·|s)`(`a`의 원-핫에서 확률을 뺀 것)는 소프트맥스 정책 그래디언트의 심장이다. 근육 기억에 새겨라.

### 5단계: 베이스라인

최근 에피소드들에 대한 `G`의 실행 평균만으로도 4×4 GridWorld를 돌리기에 충분한 분산 감소가 된다; 수렴에 약 500 에피소드가 걸린다. 베이스라인을 학습된 `V̂(s)`로 업그레이드하면 액터-크리틱이 된다.

## 함정 (Pitfalls)

- **그래디언트 폭발(exploding gradients).** 리턴은 거대할 수 있다. `∇ log π`를 곱하기 전에 항상 배치(batch) 전체에 걸쳐 `G`를 `~N(0, 1)`로 정규화(normalization)하라.
- **엔트로피 붕괴(entropy collapse).** 정책이 너무 일찍 거의 결정론적인 행동으로 수렴하고, 탐험을 멈추고, 갇힌다. 해법: 목적함수에 엔트로피 보너스 `β · H(π(·|s))`를 추가하라.
- **높은 분산.** 바닐라 REINFORCE는 수천 에피소드가 필요하다. 크리틱 베이스라인(Lesson 07)이나 TRPO/PPO의 신뢰 영역(trust region, Lesson 08)이 표준 해법이다.
- **표본 비효율성.** 온-폴리시(on-policy)는 한 번 갱신 후 모든 전이를 버린다는 뜻이다. 중요도 샘플링(importance sampling)을 통한 오프-폴리시 보정은 분산을 대가로 데이터를 되살린다(PPO의 비율은 클리핑된 IS 가중치다).
- **비정상 그래디언트.** 100 에피소드 전의 같은 그래디언트는 오래된 `π`를 사용한다. 온-폴리시 방법이 몇 롤아웃마다 갱신하는 이유다.
- **신용 할당(credit assignment).** 리워드-투-고가 없으면 과거 보상이 노이즈를 기여한다. 항상 리워드-투-고를 사용하라.

## 라이브러리로 써보기 (Use It)

2026년에 REINFORCE는 직접 실행되는 일은 드물지만 그 그래디언트 공식은 어디에나 있다:

| 사용 사례 | 파생 방법 |
|----------|---------------|
| 연속 제어 | 가우시안 정책을 가진 PPO / SAC |
| LLM RLHF | KL 페널티를 가진 PPO, 토큰 수준 정책에서 실행 |
| LLM 추론 (DeepSeek) | GRPO: 그룹 상대 베이스라인을 가진 REINFORCE, 크리틱 없음 |
| 멀티 에이전트 | 중앙집중식 크리틱 REINFORCE (MADDPG, COMA) |
| 이산 행동 로보틱스 | A2C, A3C, PPO |
| 선호도만 있는 설정 | DPO: 선호도-우도 손실(loss)로 다시 쓴 REINFORCE, 샘플링 없음 |

2026년 학습 스크립트에서 `loss = -advantage * log_prob`를 읽으면, 바로 베이스라인을 가진 REINFORCE다. 논문 전체(DPO, GRPO, RLOO)가 이 한 줄 위의 분산 감소 트릭이다.

## 산출물 (Ship It)

`outputs/skill-policy-gradient-trainer.md`로 저장하라:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 연습 문제 (Exercises)

1. **쉬움.** 선형 소프트맥스 정책으로 4×4 GridWorld에서 REINFORCE를 구현하라. 베이스라인 없이 1,000 에피소드 동안 학습하라. 학습 곡선을 그리고, 분산(리턴의 표준편차)을 측정하라.
2. **중간.** 실행 평균 베이스라인을 추가하라. 다시 학습하라. 바닐라 실행과 표본 효율 및 분산을 비교하라. 베이스라인이 수렴까지의 스텝을 얼마나 줄이는가?
3. **어려움.** 엔트로피 보너스 `β · H(π)`를 추가하라. `β ∈ {0, 0.01, 0.1, 1.0}`을 스윕(sweep)하라. 최종 리턴과 정책 엔트로피를 그려라. 이 작업에서 최적점은 어디인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 정책 그래디언트 | "정책을 직접 학습" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; 로그-도함수 트릭에서 유도됨. |
| REINFORCE | "원조 PG 알고리즘" | Williams (1992); 몬테카를로 리턴에 로그-정책 그래디언트를 곱함. |
| 로그-도함수 트릭 | "스코어 함수 추정기" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; 기댓값의 그래디언트를 다루기 쉽게 만듦. |
| 베이스라인 | "분산 감소" | `G`에서 빼는 임의의 `b(s)`; `E[b · ∇ log π] = 0`이므로 불편. |
| 리워드-투-고 | "미래 리턴만 센다" | 전체 `G_0` 대신 `G_t^{from t}`; 정확하고 분산이 낮음. |
| 엔트로피 보너스 | "탐험을 장려" | `+β · H(π(·\|s))` 항이 정책이 붕괴하는 것을 막음. |
| 온-폴리시 | "방금 본 것으로 학습" | 그래디언트 기댓값이 현재 정책에 대한 것: 오래된 데이터를 직접 재사용 불가. |
| 어드밴티지 | "평균보다 얼마나 나은지" | `A(s, a) = G(s, a) - V(s)`; 베이스라인을 가진 REINFORCE가 곱하는 부호 있는 양. |

## 더 읽을거리 (Further Reading)

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696): 원조 REINFORCE 논문.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html): 함수 근사를 가진 현대적 정책 그래디언트 정리.
- [Sutton & Barto (2018). Ch. 13: Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) 교과서적 제시.
- [OpenAI Spinning Up: VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) PyTorch 코드를 곁들인 명료한 교육적 설명.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf): 분산 감소와 REINFORCE를 신뢰 영역 계열(TRPO, PPO)에 연결하는 자연 그래디언트 관점.
