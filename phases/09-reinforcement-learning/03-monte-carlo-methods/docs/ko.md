# 몬테카를로 방법(완전한 에피소드로부터 학습하기 (Monte Carlo Methods) Learning from Complete Episodes)

> 동적 계획법(dynamic programming)은 모델이 필요하다. 몬테카를로(Monte Carlo)는 에피소드 외에는 아무것도 필요하지 않다. 정책을 돌리고, 리턴을 지켜보고, 평균을 낸다. 강화 학습(reinforcement learning)에서 가장 단순한 아이디어이자. 그 아래의 모든 것을 여는 아이디어다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75분

## 문제 (The Problem)

동적 계획법은 우아하지만, 모든 상태와 행동에 대해 `P(s' | s, a)`를 질의할 수 있다고 가정한다. 현실 세계의 거의 모든 것은 그렇게 작동하지 않는다. 로봇은 관절 토크 이후 카메라 픽셀에 대한 분포를 해석적으로 계산할 수 없다. 가격 책정 알고리즘은 모든 가능한 고객 반응을 적분할 수 없다. LLM은 토큰 이후 가능한 모든 연속을 열거할 수 없다.

필요한 것은 환경으로부터 *샘플링*하는 능력만 요구하는 방법이다. 정책을 돌린다. 궤적(trajectory) `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`를 얻는다. 이것으로 가치를 추정한다. 이것이 몬테카를로다.

DP에서 MC로의 전환은 철학적으로 중요하다: *알려진 모델 + 정확한 백업*에서 *샘플링된 롤아웃(rollout) + 평균 리턴*으로 옮겨간다. 분산은 치솟지만 적용 가능성은 폭발한다. 이 레슨 이후의 모든 강화 학습 알고리즘(TD, Q-러닝, REINFORCE, PPO, GRPO)은 본질적으로 몬테카를로 추정기이며, 때때로 그 위에 부트스트래핑(bootstrapping)이 얹혀 있다.

## 개념 (The Concept)

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**핵심 아이디어, 한 줄로:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`이며, 여기서 `G^{(i)}(s)`는 정책 `π` 하에서 `s` 방문 이후 관측된 리턴이다.

**첫-방문(first-visit) 대 모든-방문(every-visit) MC.** 상태 `s`를 여러 번 방문하는 에피소드가 주어졌을 때, 첫-방문 MC는 첫 방문으로부터의 리턴만 센다. 모든-방문 MC는 모든 방문을 센다. 둘 다 극한에서 불편(unbiased)하다. 첫-방문은 분석하기 더 단순하다(iid 샘플). 모든-방문은 에피소드당 더 많은 데이터를 사용하며 실전에서 보통 더 빨리 수렴한다.

**증분 평균(incremental mean).** 모든 리턴을 저장하는 대신, 실행 중인 평균을 갱신한다:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

재정리하면: `α = 1/n`인 `V_new = V_old + α · (target - V_old)`. `1/n`을 상수 스텝 크기 `α ∈ (0, 1)`로 바꾸면, `π`의 변화를 추적하는 비정상(non-stationary) MC 추정기를 얻는다. 그 한 수가 MC에서 TD로, 그리고 모든 현대 강화 학습 알고리즘으로 가는 도약 전체다.

**이제 탐험이 문제다.** DP는 열거로 모든 상태를 건드렸다. MC는 정책이 방문하는 상태만 본다. `π`가 결정론적이면, 상태 공간의 전체 영역이 결코 샘플링되지 않고, 그 가치 추정값은 영원히 0으로 남는다. 역사적 순서로 세 가지 해법:

1. **탐험적 시작(exploring starts).** 각 에피소드를 무작위 (s, a) 쌍에서 시작한다. 커버리지를 보장하지만 실전에서는 비현실적이다(로봇을 임의의 상태로 "리셋"할 수 없다).
2. **ε-탐욕(ε-greedy).** 현재 Q에 대해 탐욕적으로 행동하되, 확률 `ε`로 무작위 행동을 고른다. 점근적으로 모든 상태-행동 쌍이 샘플링된다.
3. **오프-폴리시(off-policy) MC.** 행동 정책(behavior policy) `μ` 하에서 데이터를 수집하고, 중요도 샘플링(importance sampling)으로 목표 정책 `π`를 학습한다. 분산이 높지만 DQN 같은 리플레이 버퍼(replay-buffer) 방법으로 가는 다리다.

**몬테카를로 제어(Monte Carlo Control).** 평가 → 개선 → 평가, 정책 반복과 똑같지만, 평가가 샘플링 기반이다:

1. `π`를 돌려 에피소드를 얻는다.
2. 관측된 리턴으로 `Q(s, a)`를 갱신한다.
3. `Q`에 대해 `π`를 ε-탐욕으로 만든다.
4. 반복한다.

온화한 조건 하에서 확률 1로 `Q*`와 `π*`로 수렴한다(모든 쌍이 무한히 자주 방문되고, `α`가 로빈스-먼로(Robbins-Monro)를 만족).

```figure
epsilon-greedy
```

## 직접 만들기 (Build It)

### 1단계: 롤아웃 → (s, a, r) 리스트

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

모델은 없고, `env.reset()`과 `env.step(s, a)`만 있다. gym 환경과 같은 인터페이스지만 군더더기를 뺀 것이다.

### 2단계: 리턴 계산 (역방향 스윕)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

한 번의 패스, `O(T)`. 역방향 점화식 `G_t = r_{t+1} + γ G_{t+1}`은 재합산을 피한다.

### 3단계: 첫-방문 MC 평가

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

세 줄이 일을 한다: 첫 방문 시 상태를 본 것으로 표시, 카운트 증가, 실행 평균 갱신.

### 4단계: ε-탐욕 MC 제어 (온-폴리시)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 5단계: DP 황금 표준과 비교

`V^π`에 대한 MC 추정값은 에피소드 → ∞에 따라 Lesson 02의 DP 결과와 일치해야 한다. 실전에서: 4×4 GridWorld에서 50,000 에피소드면 DP 답의 `~0.1` 이내에 든다.

## 함정 (Pitfalls)

- **무한 에피소드.** MC는 에피소드가 *종료*되어야 한다. 정책이 영원히 루프를 돌 수 있다면, `max_steps`로 한계를 두고 그 한계를 암묵적 실패로 취급하라. 무작위 정책의 GridWorld는 일상적으로 타임아웃이 난다. 정상이니 제대로 세고 있는지만 확인하라.
- **분산.** MC는 전체 리턴을 사용한다. 긴 에피소드에서는 분산이 거대하다. 끝에서 운 나쁜 보상 하나가 `V(s_0)`를 같은 양만큼 이동시킨다. TD 방법(Lesson 04)은 부트스트래핑으로 이를 줄인다.
- **상태 커버리지.** 동점이 있는 신선한 Q에 대한 탐욕 MC는 오직 한 행동만 시도할 것이다. 반드시 탐험해야 한다(ε-탐욕, 탐험적 시작, UCB).
- **비정상 정책.** `π`가 바뀌면(MC 제어에서처럼), 이전 리턴은 다른 정책에서 나온 것이다. 상수-α MC는 이를 처리하지만, 표본 평균 MC는 그렇지 못하다.
- **오프-폴리시 중요도 샘플링.** 가중치 `π(a|s)/μ(a|s)`는 궤적 전체에 걸쳐 곱해진다. 지평이 길어지면 분산이 폭발한다. 결정별 가중 IS로 한계를 두거나 TD로 전환하라.

## 라이브러리로 써보기 (Use It)

2026년 몬테카를로 방법의 역할:

| 사용 사례 | MC를 쓰는 이유 |
|----------|--------|
| 단기 지평 게임 (블랙잭, 포커) | 에피소드가 자연스럽게 종료; 리턴이 깔끔하다. |
| 로깅된 정책의 오프라인 평가 | 저장된 궤적에 대한 평균 할인 리턴. |
| 몬테카를로 트리 탐색 (AlphaZero) | 트리 잎(leaf)에서의 MC 롤아웃이 선택을 안내한다. |
| LLM 강화 학습 평가 | 주어진 정책에 대해 샘플링된 완성에 대한 평균 보상 계산. |
| PPO에서의 베이스라인 추정 | 어드밴티지 타깃 `A_t = G_t - V(s_t)`가 MC `G_t`를 사용한다. |
| 강화 학습 교육 | 실제로 작동하는 가장 단순한 알고리즘: 부트스트래핑을 떼어내 핵심을 본다. |

현대 심층 강화 학습 알고리즘(PPO, SAC)은 `n`-스텝 리턴이나 GAE로 순수 MC(전체 리턴)와 순수 TD(한-스텝 부트스트랩) 사이를 보간한다. 두 극단 모두 같은 추정기의 사례다.

## 산출물 (Ship It)

`outputs/skill-mc-evaluator.md`로 저장하라:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 연습 문제 (Exercises)

1. **쉬움.** 4×4 GridWorld에서 균일 무작위 정책의 첫-방문 MC 평가를 구현하라. 10,000 에피소드를 돌려라. 에피소드 수의 함수로서 `V(0,0)`을 DP 답과 비교하여 그려라.
2. **중간.** `ε ∈ {0.01, 0.1, 0.3}`로 ε-탐욕 MC 제어를 구현하라. 20,000 에피소드 후 평균 리턴을 비교하라. 곡선은 어떻게 생겼는가? 편향-분산 트레이드오프(trade-off)는 어디에 있는가?
3. **어려움.** 중요도 샘플링을 사용한 *오프-폴리시* MC를 구현하라: 균일 무작위 정책 `μ` 하에서 데이터를 수집하고, 결정론적 최적 정책 `π`에 대한 `V^π`를 추정하라. 단순 IS 대 결정별 IS 대 가중 IS를 비교하라. 어느 것이 분산이 가장 낮은가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 몬테카를로 | "무작위 샘플링" | 분포로부터의 iid 샘플에 대한 평균으로 기댓값을 추정. |
| 리턴 `G_t` | "미래 보상" | 스텝 `t`부터 에피소드 끝까지의 할인 보상 합: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| 첫-방문 MC | "각 상태를 한 번만 센다" | 에피소드 내 첫 방문만 가치 추정에 기여. |
| 모든-방문 MC | "모든 방문을 쓴다" | 모든 방문이 기여; 약간 편향되지만 표본 효율이 더 높다. |
| ε-탐욕 | "탐험 노이즈" | 확률 `1-ε`로 탐욕 행동; 확률 `ε`로 무작위 행동. |
| 중요도 샘플링 | "잘못된 분포에서 샘플링한 것을 교정" | `μ` 데이터로 `V^π`를 추정하기 위해 `π(a\|s)/μ(a\|s)` 곱으로 리턴을 재가중. |
| 온-폴리시 | "내 자신의 데이터로 학습" | 목표 정책 = 행동 정책. 바닐라 MC, PPO, SARSA. |
| 오프-폴리시 | "남의 데이터로 학습" | 목표 정책 ≠ 행동 정책. 중요도 샘플링 MC, Q-러닝, DQN. |

## 더 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Ch. 5: Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) 정전적 다룸.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726): 첫-방문 대 모든-방문 분석.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf): 오프-폴리시 MC와 분산 제어.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362): 현대의 저분산 IS 추정기.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343): MC/TD 자기 대국(self-play)이 초인적 플레이로 수렴함을 처음으로 대규모로 실증한 사례; 이 단계 후반부 모든 레슨의 개념적 선구자.
