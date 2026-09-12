# 시간차(Q-러닝과 SARSA (Temporal Difference) Q-Learning & SARSA)

> 몬테카를로(Monte Carlo)는 에피소드가 끝날 때까지 기다린다. TD는 다음 가치 추정값을 부트스트랩(bootstrap)하여 매 스텝마다 갱신한다. Q-러닝(Q-learning)은 오프-폴리시(off-policy)이며 낙관적이고, SARSA는 온-폴리시(on-policy)이며 신중하다. 둘 다 코드 한 줄이다. 둘 다 이 단계의 모든 심층 강화 학습 방법을 떠받친다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75분

## 문제 (The Problem)

몬테카를로는 작동하지만 두 가지를 비싸게 요구한다. 종료되는 에피소드가 필요하고, 최종 리턴이 들어온 뒤에만 갱신한다. 에피소드가 1,000스텝이면, MC는 무엇이든 갱신하기 위해 1,000스텝을 기다린다. 고분산, 저편향이며, 실전에서 느리다.

동적 계획법(dynamic programming)은 정반대다. 분산이 0인 부트스트랩 백업을 쓰지만, 알려진 모델이 필요하다.

시간차(temporal difference, TD) 학습은 그 차이를 절충한다. 단일 전이 `(s, a, r, s')`로부터, 한-스텝 타깃 `r + γ V(s')`를 만들고 `V(s)`를 그쪽으로 밀어준다. 모델 없음. 완전한 에피소드 없음. 우변에서 근사 `V`를 사용하는 데서 오는 편향이 있지만, 분산은 MC보다 극적으로 낮고 첫 스텝부터 온라인으로 갱신한다.

이것이 DQN, A2C, PPO, SAC 등 모든 현대 강화 학습이 돌아가는 중심축이다. Phase 9의 나머지는 이 레슨에서 작성할 한-스텝 TD 갱신 위에 쌓아 올린 함수 근사(function approximation)와 트릭의 층들이다.

## 개념 (The Concept)

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V에 대한 TD(0) 갱신:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

대괄호 안의 양이 TD 오차 `δ = r + γ V(s') - V(s)`다. 이는 MC의 `G_t - V(s_t)`에 대한 온라인 대응물이다. 수렴에는 로빈스-먼로(Robbins-Monro)를 만족하는 `α`(`Σ α = ∞`, `Σ α² < ∞`)와 모든 상태의 무한 반복 방문이 필요하다.

**Q-러닝.** 제어를 위한 오프-폴리시 TD 방법:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`는 에이전트가 실제로 어떤 행동을 취하든 상관없이 `s'` 이후로 *탐욕(greedy)* 정책을 따른다고 가정한다. 그 분리(decoupling) 덕분에 에이전트가 ε-탐욕으로 탐험하는 동안 Q-러닝은 `Q*`를 학습한다. Mnih et al. (2015)은 이것을 Atari에서의 심층 Q-러닝으로 전환했다(Lesson 05).

**SARSA.** 온-폴리시 TD 방법:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

이름은 튜플 `(s, a, r, s', a')`다. SARSA는 탐욕 `argmax`가 아니라 에이전트가 *실제로* 다음에 취하는 행동 `a'`를 사용한다. 실행 중인 ε-탐욕 `π`가 무엇이든 그에 대한 `Q^π`로 수렴하며, 극한 `ε → 0`에서 `Q*`가 된다.

**절벽 걷기(cliff-walking)의 차이.** 고전적인 절벽 걷기 작업(절벽으로 떨어짐 = 보상 -100)에서, Q-러닝은 절벽 가장자리를 따르는 최적 경로를 학습하지만 탐험 중에 가끔 페널티를 받는다. SARSA는 탐험 노이즈를 자신의 Q-값에 반영하기 때문에 절벽에서 한 스텝 떨어진 더 안전한 경로를 학습한다. 학습이 진행되면, `ε → 0`에서 둘 다 최적에 도달한다. 이 점이 실전에서 중요하다. 배포 시 실제로 탐험이 일어나면 SARSA의 행동이 더 보수적이다.

**기대 SARSA(Expected SARSA).** `Q(s', a')`를 `π` 하의 기댓값으로 교체한다:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA보다 낮은 분산(`a'`의 표본 없음), 같은 온-폴리시 타깃. 현대 교과서에서 흔히 기본값이다.

**n-스텝 TD와 TD(λ).** 부트스트랩 전에 `n`스텝을 기다림으로써 TD(0)와 MC 사이를 보간한다. `n=1`은 TD, `n=∞`는 MC다. TD(λ)는 기하 가중치 `(1-λ)λ^{n-1}`로 모든 `n`에 대해 평균을 낸다. 대부분의 심층 강화 학습은 3에서 20 사이의 `n`을 사용한다.

```figure
qlearning-gridworld
```

## 직접 만들기 (Build It)

### 1단계: ε-탐욕 정책에서의 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

여덟 줄. Q-러닝과의 *유일한* 차이는 타깃 줄이다.

### 2단계: Q-러닝

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`가 타깃을 행동에서 분리한다. 그 기호 하나가 온-폴리시와 오프-폴리시의 차이다.

### 3단계: 학습 곡선

100 에피소드당 평균 리턴을 추적한다. Q-러닝은 단순한 결정론적 GridWorld에서 더 빨리 수렴하고, SARSA는 절벽 걷기에서 더 보수적이다. `code/main.py`의 4×4 GridWorld에서, `α=0.1, ε=0.1`로 약 2,000 에피소드 후 둘 다 거의 최적이다.

### 4단계: DP 진실과 비교

가치 반복(Lesson 02)을 실행해 `Q*`를 얻는다. `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`를 확인한다. 건강한 표(tabular) TD 에이전트는 4×4 GridWorld에서 10,000 에피소드 후 `~0.5` 이내에 든다.

## 함정 (Pitfalls)

- **초기 Q 값이 중요하다.** 낙관적 초기화(음의 보상 작업에서 `Q = 0`)는 탐험을 장려한다. 비관적 초기화는 탐욕 정책을 영원히 가둘 수 있다.
- **α 스케줄.** 상수 `α`는 비정상 문제에 적합하다. 감쇠 `α_n = 1/n`은 이론적으로 수렴을 주지만 실전에서 너무 느리다. `α`를 `[0.05, 0.3]`에 고정하고 학습 곡선을 모니터링하라.
- **ε 스케줄.** 높게 시작(`ε=1.0`), `ε=0.05`까지 감쇠. "GLIE"(무한 탐험 하에서 극한에서 탐욕적)가 수렴 조건이다.
- **Q-러닝의 최대 편향.** `max` 연산자는 `Q`가 노이즈가 있을 때 위쪽으로 편향된다. 과대추정으로 이어진다. Hasselt의 이중 Q-러닝(Double Q-learning, Lesson 05에서 DDQN이 사용)이 두 개의 Q 표로 이를 고친다.
- **종료되지 않는 에피소드.** TD는 종료 상태 없이 학습할 수 있지만, 스텝에 한계를 두거나 그 한계에서 부트스트랩을 올바르게 처리해야 한다. 표준: 한계를 비종료로 취급하고 부트스트랩을 계속한다.
- **상태 해싱.** 상태가 튜플/텐서라면, 해시 가능한 키를 사용하라(리스트가 아닌 튜플; 원본이 아닌 반올림된 float의 튜플).

## 라이브러리로 써보기 (Use It)

2026년 TD 지형:

| 작업 | 방법 | 이유 |
|------|--------|--------|
| 작은 표 환경 | Q-러닝 | 최적 정책을 직접 학습. |
| 온-폴리시 안전 중요 | SARSA / 기대 SARSA | 탐험 중 보수적. |
| 고차원 상태 | DQN (Phase 9 · 05) | 리플레이와 타깃 망(target net)을 갖춘 신경망 Q-함수. |
| 연속 행동 | SAC / TD3 (Phase 9 · 07) | Q-망에 대한 TD 갱신; 정책 망이 행동을 내보냄. |
| LLM 강화 학습 (보상 모델 기반) | PPO / GRPO (Phase 9 · 08, 12) | GAE를 통한 TD 스타일 어드밴티지를 가진 액터-크리틱. |
| 오프라인 강화 학습 | CQL / IQL (Phase 9 · 08) | 보수적 정규화를 가진 Q-러닝. |

2026년 논문에서 읽는 "강화 학습"의 90퍼센트는 Q-러닝이나 SARSA의 어떤 정교화다. 더 깊이 읽기 전에 표 갱신을 손에 익혀라.

## 산출물 (Ship It)

`outputs/skill-td-agent.md`로 저장하라:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 연습 문제 (Exercises)

1. **쉬움.** 4×4 GridWorld에서 Q-러닝과 SARSA를 구현하라. 2,000 에피소드에 대한 학습 곡선(100 에피소드당 평균 리턴)을 그려라. 누가 더 빨리 수렴하는가?
2. **중간.** 절벽 걷기 환경(4×12, 마지막 행은 보상 -100과 시작으로 리셋되는 절벽)을 만들어라. Q-러닝과 SARSA의 최종 정책을 비교하라. 각각이 취하는 경로를 스크린샷으로 남겨라. 어느 쪽이 절벽에 더 가까운가?
3. **어려움.** 이중 Q-러닝을 구현하라. 노이즈 보상 GridWorld(스텝당 보상에 가우시안 노이즈 σ=5 추가)에서, Q-러닝이 `V*(0,0)`을 유의미한 양만큼 과대추정하는 반면 이중 Q-러닝은 그렇지 않음을 보여라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| TD 오차 | "갱신 신호" | `δ = r + γ V(s') - V(s)`, 부트스트랩된 잔차. |
| TD(0) | "한-스텝 TD" | 다음 상태의 추정값만 사용해 매 전이 후 갱신. |
| Q-러닝 | "오프-폴리시 강화 학습 입문" | 다음 상태 행동에 대한 `max`를 가진 TD 갱신; 행동 정책과 무관하게 `Q*`를 학습. |
| SARSA | "온-폴리시 Q-러닝" | 실제 다음 행동을 사용한 TD 갱신; 현재 ε-탐욕 π에 대한 `Q^π`를 학습. |
| 기대 SARSA | "저분산 SARSA" | 샘플된 `a'`를 π 하의 기댓값으로 교체. |
| GLIE | "올바른 탐험 스케줄" | 무한 탐험 하에서 극한에서 탐욕적; Q-러닝 수렴에 필요. |
| 부트스트래핑 | "타깃에 현재 추정값을 사용" | TD를 MC와 구별하는 것. 편향의 원천이지만 막대한 분산 감소. |
| 최대화 편향 | "Q-러닝이 과대추정한다" | 노이즈 추정값에 대한 `max`는 위쪽으로 편향됨; 이중 Q-러닝이 고침. |

## 더 읽을거리 (Further Reading)

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698): 원조 논문과 수렴 증명.
- [Sutton & Barto (2018). Ch. 6: Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) TD(0), SARSA, Q-러닝, 기대 SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html): 최대화 편향에 대한 해법.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542): 기대 SARSA의 동기.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems): SARSA(당시에는 "수정 연결주의 Q-러닝"으로 불림)를 만든 논문.
- [Sutton & Barto (2018). Ch. 6: Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) TD(0)를 TD(n)으로 일반화, Q-러닝에서 적격성 흔적(eligibility traces)으로, 그리고 나중에 PPO의 GAE로 가는 경로.
