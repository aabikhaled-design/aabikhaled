# 동적 계획법(정책 반복과 가치 반복 (Dynamic Programming) Policy Iteration & Value Iteration)

> 동적 계획법(dynamic programming)은 커닝하는 강화 학습이다. 이미 전이 함수와 보상 함수를 알고 있으니, 그저 `V` 또는 `π`가 더 이상 움직이지 않을 때까지 벨만 방정식을 반복할 뿐이다. 이것은 모든 샘플링 기반 방법이 따라잡으려 애쓰는 벤치마크(benchmark)다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75분

## 문제 (The Problem)

모델이 알려진 MDP가 있다고 하자: 임의의 상태-행동 쌍에 대해 `P(s' | s, a)`와 `R(s, a, s')`를 질의할 수 있다. 재고 관리자는 수요 분포를 안다. 보드 게임은 전이가 결정론적이다. 그리드월드는 파이썬 네 줄이면 된다. 즉 *모델*이 있는 것이다.

모델-프리(model-free) 강화 학습(Q-러닝, PPO, REINFORCE)은 모델이 없는 경우(환경으로부터 샘플링만 할 수 있는 경우)를 위해 발명되었다. 하지만 모델이 있다면 더 빠르고 더 나은 방법이 있다: 동적 계획법이다. 벨만(Bellman)이 1957년에 설계했다. 이 방법들은 여전히 정확성을 정의한다: "이 MDP에 대한 최적 정책"이라고 말할 때 그 정책은 DP가 반환할 정책을 뜻한다.

2026년에 이것이 필요한 이유는 세 가지다. 첫째, 강화 학습 연구의 모든 표(tabular) 환경(GridWorld, FrozenLake, CliffWalking)은 황금 표준(gold-standard) 정책을 만들기 위해 DP로 풀린다. 둘째, 정확한 가치는 샘플링 방법을 *디버깅*하게 해준다: Q-러닝의 `V*(s_0)` 추정값이 DP 답과 30% 차이가 난다면 Q-러닝에 버그가 있다는 뜻이다. 셋째, 현대의 오프라인 강화 학습과 계획 방법(MCTS, AlphaZero의 탐색, Phase 9 · 10의 모델 기반 강화 학습)은 모두 학습된 혹은 주어진 모델 위에서 벨만 백업(Bellman backup)을 반복한다.

## 개념 (The Concept)

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**두 알고리즘, 둘 다 벨만에 대한 고정점 반복.**

**정책 반복(Policy iteration).** 정책이 더 이상 바뀌지 않을 때까지 두 단계를 번갈아 수행한다.

1. *평가:* 정책 `π`가 주어지면, `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`를 수렴할 때까지 반복 적용하여 `V^π`를 계산한다.
2. *개선:* `V^π`가 주어지면, `π`를 `V^π`에 대해 탐욕적(greedy)으로 만든다: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

수렴이 보장되는 이유는 (a) 각 개선 단계가 `π`를 그대로 유지하거나 어떤 상태에 대해 `V^π`를 엄밀히 증가시키고, (b) 결정론적 정책의 공간이 유한하기 때문이다. 큰 상태 공간에서도 보통 약 5–20번의 외부 반복으로 수렴한다.

**가치 반복(Value iteration).** 평가와 개선을 하나의 스윕(sweep)으로 합친다. 벨만 *최적성(optimality)* 방정식을 적용한다:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`가 될 때까지 반복한다. 마지막에 탐욕 행동을 취해 정책을 추출한다. 내부 평가 루프가 없으므로 반복당으로는 엄밀히 더 빠르지만, 보통 수렴에는 더 많은 반복이 필요하다.

**일반화 정책 반복(generalized policy iteration, GPI).** 통합적 틀이다. 가치 함수와 정책은 양방향 개선 루프에 묶여 있다. 둘을 상호 일관성으로 몰아가는 모든 방법(비동기 가치 반복, 수정 정책 반복, Q-러닝, 액터-크리틱, PPO)은 GPI의 한 사례다.

**`γ < 1`이 중요한 이유.** 벨만 연산자(Bellman operator)는 상한 노름(sup-norm)에서 `γ`-축약(contraction)이다: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. 축약은 유일한 고정점과 기하적 수렴을 함의한다. `γ < 1`을 버리면 보장을 잃는다. 유한 지평이나 흡수 종료 상태가 필요하다.

```figure
value-iteration-gamma
```

## 직접 만들기 (Build It)

### 1단계: GridWorld MDP 모델 만들기

Lesson 01과 같은 4×4 GridWorld를 사용한다. 확률적 변형을 추가한다: 확률 `0.1`로 에이전트가 무작위 수직 방향으로 미끄러진다.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`는 `(s', r, p)`의 리스트를 반환한다. 이것이 모델 전체다.

### 2단계: 정책 평가

정책 `π(s) = {action: prob}`가 주어지면, `V`가 더 이상 움직이지 않을 때까지 벨만 방정식을 반복한다:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 3단계: 정책 개선

`π`를 `V`에 대한 탐욕 정책으로 교체한다. `π`가 바뀌지 않았다면 반환한다. 우리는 최적점에 있다.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 4단계: 둘을 엮기

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4에서의 전형적 수렴: 4–6번의 외부 반복. `V*(0,0) ≈ -6`과 스텝 수를 엄밀히 줄이는 정책을 출력한다.

### 5단계: 가치 반복 (단일 루프 버전)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

같은 고정점, 더 적은 코드 줄.

## 함정 (Pitfalls)

- **종료 상태 처리를 잊기.** 흡수 상태에 벨만을 적용하면, 여전히 아무것도 바꾸지 않는 "최선의 행동"을 골라낸다. `if s == terminal: V[s] = 0`으로 막아라.
- **상한 노름 대 L2 수렴.** 평균이 아니라 `max |V_new - V|`를 사용하라. 이론적 보장은 상한 노름에 대한 것이다.
- **제자리(in-place) 대 동기(synchronous) 갱신.** `V[s]`를 제자리에서 갱신하면(가우스-자이델, Gauss-Seidel) 별도의 `V_new` 딕셔너리(야코비, Jacobi)보다 더 빨리 수렴한다. 프로덕션(production) 코드는 제자리 방식을 쓴다.
- **정책 동점.** 두 행동의 Q-값이 같으면, `argmax`가 반복마다 동점을 다르게 깨뜨려 "정책 안정" 검사가 진동할 수 있다. 안정적인 동점 깨기(고정된 순서의 첫 번째 행동)를 사용하라.
- **상태 공간 폭발.** DP는 스윕당 `O(|S| · |A|)`이다. 약 10⁷ 상태까지 작동한다. 그 이상은 함수 근사(function approximation)가 필요하다(Phase 9 · 05부터).

## 라이브러리로 써보기 (Use It)

2026년에 DP는 정확성 베이스라인(baseline)이자 계획기의 내부 루프다:

| 사용 사례 | 방법 |
|----------|--------|
| 작은 표 MDP를 정확히 풀기 | 가치 반복(더 단순) 또는 정책 반복(외부 스텝이 더 적음) |
| Q-러닝 / PPO 구현 검증 | 장난감 환경에서 DP-최적 V*와 비교 |
| 모델 기반 강화 학습 (Phase 9 · 10) | 학습된 전이 모델 위의 벨만 백업 |
| AlphaZero / MuZero에서의 계획 | 몬테카를로 트리 탐색 = 비동기 벨만 백업 |
| 오프라인 강화 학습 (CQL, IQL) | 보수적 Q-반복: OOD 행동에 페널티를 둔 DP |

누군가 "최적 가치 함수"라고 말할 때마다 그것은 "DP 고정점"을 뜻한다. 논문에서 `V*`나 `Q*`를 보면 이 루프를 떠올려라.

## 산출물 (Ship It)

`outputs/skill-dp-solver.md`로 저장하라:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 연습 문제 (Exercises)

1. **쉬움.** `γ ∈ {0.9, 0.99}`로 4×4 GridWorld에서 가치 반복을 실행하라. `max |ΔV| < 1e-6`이 될 때까지 몇 번의 스윕이 걸리는가? `V*`를 4×4 그리드로 출력하라.
2. **중간.** *확률적* GridWorld(미끄럼 확률 `0.1`)에서 정책 반복과 가치 반복을 비교하라. 다음을 세어라: 스윕 수, 벽시계(wall-clock) 시간, 최종 `V*(0,0)`. 반복으로는 어느 쪽이 더 빨리 수렴하는가? 벽시계로는?
3. **어려움.** 수정 정책 반복(modified policy iteration)을 만들어라: 평가 단계에서 수렴까지가 아니라 `k`번의 스윕만 실행한다. `k ∈ {1, 2, 5, 10, 50}`에 대해 `V*(0,0)` 오차 대 `k`를 그려라. 그 곡선은 평가/개선 트레이드오프(trade-off)에 대해 무엇을 말해주는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 정책 반복 | "DP 알고리즘" | 정책이 더 이상 바뀌지 않을 때까지 평가(`V^π`)와 개선(`V^π`에 대한 탐욕 `π`)을 번갈아 수행. |
| 가치 반복 | "더 빠른 DP" | 한 스윕에 적용되는 벨만 최적성 백업; `V*`로 기하적으로 수렴. |
| 벨만 연산자 | "그 재귀" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; 상한 노름에서의 `γ`-축약. |
| 축약 | "DP가 수렴하는 이유" | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`를 만족하는 모든 연산자 `T`는 유일한 고정점을 가진다. |
| GPI | "모든 것이 DP다" | 일반화 정책 반복: `V`와 `π`를 상호 일관성으로 몰아가는 모든 방법. |
| 동기 갱신 | "야코비 방식" | 스윕 전체에서 이전 `V`를 사용; 깔끔하게 분석 가능하지만 더 느림. |
| 제자리 갱신 | "가우스-자이델 방식" | 갱신되는 중인 `V`를 사용; 실전에서 더 빨리 수렴. |

## 더 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Ch. 4: Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) 정책 반복과 가치 반복의 정전(正典)적 제시.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html): 축약 사상(contraction-mapping) 논증의 엄밀한 다룸.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887): 수정 정책 반복과 그 수렴 분석.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/): 원조 정책 반복 논문.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html): 이후 모든 레슨이 사용하는 DP에서 근사-DP / 심층 강화 학습으로의 다리.
