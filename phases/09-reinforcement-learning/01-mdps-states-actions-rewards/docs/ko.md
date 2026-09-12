# MDP, 상태, 행동, 보상 (MDPs, States, Actions & Rewards)

> 마르코프 결정 과정(Markov Decision Process)은 다섯 가지로 이루어진다. 상태, 행동, 전이, 보상, 할인. 강화 학습(reinforcement learning)의 모든 것(Q-러닝, PPO, DPO, GRPO)이 형태 위에서 최적화한다. 한 번 익혀두면, 나머지 강화 학습을 공짜로 읽을 수 있다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45분

## 문제 (The Problem)

체스 봇을 짠다고 하자. 혹은 재고 계획기를. 혹은 트레이딩 에이전트(agent)를. 혹은 추론 모델(reasoning model)을 학습시키는 PPO 루프를. 네 가지 서로 다른 도메인이지만, 한 가지 놀라운 사실이 있다. 넷 모두 같은 수학적 대상으로 환원된다.

지도 학습(supervised learning)은 `(x, y)` 쌍을 주고 함수를 맞추라고 한다. 강화 학습은 레이블(label)을 주지 않는다. 오직 상태의 흐름, 취한 행동, 그리고 스칼라(scalar) 보상만 준다. 그 수가 게임을 이겼는가? 그 재입고 결정이 돈을 아꼈는가? 그 거래가 이익을 냈는가? LLM이 방금 생성한 토큰(token)이 심판으로부터 더 높은 보상을 끌어냈는가?

이 흐름을 형식화하기 전까지는 흐름으로부터 학습할 수 없다. "내가 본 것", "내가 한 것", "그다음 일어난 일", "그것이 얼마나 좋았는지". 각각이 추론할 수 있는 대상이 되어야 한다. 그 형식화가 바로 마르코프 결정 과정(Markov Decision Process)이다. 이 단계의 모든 강화 학습 알고리즘은, 끝부분의 RLHF와 GRPO 루프까지 포함해, 이 형태 위에서 최적화한다.

## 개념 (The Concept)

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**다섯 가지 대상.**

- **상태(States)** `S`. 에이전트가 결정을 내리는 데 필요한 모든 것. GridWorld에서는 셀(cell). 체스에서는 보드. LLM에서는 컨텍스트 윈도우(context window)에 더해 임의의 메모리.
- **행동(Actions)** `A`. 선택지. 위/아래/왼쪽/오른쪽 이동. 수를 두기. 토큰을 내보내기.
- **전이(Transitions)** `P(s' | s, a)`. 상태 `s`와 행동 `a`가 주어졌을 때, 다음 상태에 대한 분포. 체스에서는 결정론적, 재고에서는 확률적, LLM 디코딩에서는 거의 결정론적이다.
- **보상(Rewards)** `R(s, a, s')`. 스칼라 신호. 승리 = +1, 패배 = -1. 매출에서 비용을 뺀 값. GRPO의 로그 우도비(log-likelihood ratio) 항.
- **할인(Discount)** `γ ∈ [0, 1)`. 미래 보상이 현재 대비 얼마나 중요한지. `γ = 0.99`는 약 100스텝의 지평(horizon)을 사고, `γ = 0.9`는 약 10스텝을 산다.

**마르코프 성질(Markov property)** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. 미래는 오직 현재 상태에만 의존한다. 그렇지 않다면 상태 표현이 불완전하다. 방법론의 실패가 아니라 상태의 실패다.

**정책과 리턴(Policies and returns).** 정책 `π(a | s)`는 상태를 행동 분포로 매핑한다. 리턴 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`은 미래 보상의 할인 합이다. 가치 `V^π(s) = E[G_t | s_t = s]`는 정책 `π` 하에서 `s`에서 시작했을 때의 기대 리턴이다. Q-값 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`는 특정 행동으로 시작했을 때의 기대 리턴이다. 모든 강화 학습 알고리즘은 이 둘 중 하나를 추정한 뒤, 그에 따라 `π`를 개선한다.

**벨만 방정식(Bellman equations).** 이 단계의 모든 것이 사용하는 고정점(fixed-point) 방정식이다.

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

이 방정식들은 기대 리턴을 "이번 스텝의 보상" 더하기 "착지하는 곳의 할인된 가치"로 쪼갠다. 재귀적이다. Phase 9의 모든 알고리즘은 이 방정식을 수렴(convergence)할 때까지 반복하거나(동적 계획법), 그것으로부터 샘플링하거나(몬테카를로), 한 스텝만 부트스트랩(bootstrap)한다(시간차).

```figure
discount-horizon
```

## 직접 만들기 (Build It)

### 1단계: 아주 작은 결정론적 MDP

4×4 GridWorld. 에이전트는 좌측 상단에서 시작하고, 종료 지점은 우측 하단이며, 스텝당 보상은 -1, 행동은 `{up, down, left, right}`이다. `code/main.py`를 보라.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

다섯 줄. 그것이 환경 전체다. 결정론적 전이, 일정한 스텝 페널티, 흡수(absorbing) 종료 상태.

### 2단계: 정책 롤아웃(rollout)

정책은 상태에서 행동 분포로 가는 함수다. 가장 단순한 것: 균일 무작위(uniform random).

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

무작위 정책을 1000번 돌려보라. 이 4×4 보드에서 평균 리턴은 -60에서 -80 사이다. 최적 리턴은 -6이다(우측 하단으로 직선 경로). 그 격차를 메우는 것이 Phase 9의 전부다.

### 3단계: 벨만 방정식으로 `V^π`를 정확히 계산하기

작은 MDP에서 벨만 방정식은 선형 시스템이다. 상태를 열거하고 기댓값을 적용하며 값이 변하지 않을 때까지 반복한다.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

이것이 반복적 정책 평가(iterative policy evaluation)다. Sutton & Barto의 첫 번째 알고리즘이며, 뒤따르는 모든 강화 학습 방법의 이론적 토대다.

### 4단계: `γ`는 물리적 의미를 가진 하이퍼파라미터(hyperparameter)다

유효 지평은 대략 `1 / (1 - γ)`이다. `γ = 0.9` → 10스텝. `γ = 0.99` → 100스텝. `γ = 0.999` → 1000스텝.

너무 낮으면 에이전트가 근시안적으로 행동한다. 너무 높으면, 많은 초기 스텝이 먼 미래의 보상에 대한 책임을 공유하기 때문에 신용 할당(credit assignment)이 노이즈에 묻힌다. LLM RLHF는 보통 에피소드가 짧고 유한하기 때문에 `γ = 1`을 사용한다. 제어 작업은 `0.95–0.99`를 사용한다. 장기 지평 전략 게임은 `0.999`를 사용한다.

## 함정 (Pitfalls)

- **비마르코프 상태.** 결정을 내리는 데 마지막 세 개의 관측이 필요하다면, "상태"는 현재 관측만이 아니다. 해법: 프레임을 쌓거나(Atari에서 DQN은 4개를 쌓는다) 순환 상태(LSTM/GRU)를 사용한다.
- **희소 보상(Sparse rewards).** 승리에만 보상을 주면 큰 상태 공간에서 학습이 거의 불가능해진다. 보상을 셰이핑(shaping)하거나(중간 신호) 모방으로 부트스트랩한다(Phase 9 · 09).
- **보상 해킹(Reward hacking).** 프록시(proxy) 보상을 최적화하면 종종 병적인 행동이 나온다. OpenAI의 보트 레이싱 에이전트는 경주를 끝내는 대신 파워업을 모으며 영원히 제자리를 빙빙 돌았다. 보상은 항상 프록시가 아니라 목표 결과로 정의하라.
- **할인 잘못 명세.** 무한 지평 작업에서 `γ = 1`은 모든 가치를 무한대로 만든다. 항상 유한 지평이나 `γ < 1`로 한계를 둔다.
- **보상 스케일.** {+100, -100} 대 {+1, -1} 보상은 동일한 최적 정책을 주지만 그래디언트(gradient) 크기는 크게 다르다. PPO/DQN에 넣기 전에 `[-1, 1]` 정도로 정규화(normalization)한다.

## 라이브러리로 써보기 (Use It)

2026년 스택은 코드를 건드리기 전에 모든 강화 학습 파이프라인(pipeline)을 MDP로 환원한다.

| 상황 | 상태 | 행동 | 보상 | γ |
|-----------|-------|--------|--------|---|
| 제어 (이동, 조작) | 관절 각도 + 속도 | 연속 토크 | 작업별 셰이핑 | 0.99 |
| 게임 (체스, 바둑, 포커) | 보드 + 이력 | 합법 수 | 승리=+1 / 패배=-1 | 1.0 (유한) |
| 재고 / 가격 책정 | 재고 + 수요 | 주문 수량 | 매출 - 비용 | 0.95 |
| LLM용 RLHF | 컨텍스트 토큰 | 다음 토큰 | 마지막에 보상 모델 점수 | 1.0 (에피소드 ~200 토큰) |
| 추론용 GRPO | 프롬프트 + 부분 응답 | 다음 토큰 | 마지막에 검증기 0/1 | 1.0 |

어떤 학습 루프를 작성하기 전에 다섯 개의 튜플을 먼저 써라. "강화 학습이 작동하지 않는다"는 대부분의 버그 보고는 종이 위에서 이미 깨져 있던 MDP 정식화로 거슬러 올라간다.

## 산출물 (Ship It)

`outputs/skill-mdp-modeler.md`로 저장하라:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에 4×4 GridWorld와 무작위 정책 롤아웃을 구현하라. 10,000 에피소드를 돌려라. 리턴의 평균과 표준편차를 보고하라. 최적 리턴(-6)과 비교하라.
2. **중간.** 균일 무작위 정책에 대해 `γ ∈ {0.5, 0.9, 0.99}`로 `policy_evaluation`을 실행하라. 각각에 대해 `V`를 4×4 그리드로 출력하라. 더 큰 `γ`에서 종료 지점 근처의 상태 가치가 왜 더 빨리 커지는지 설명하라.
3. **어려움.** GridWorld를 확률적으로 바꿔라: 각 행동이 확률 `p = 0.1`로 인접한 방향으로 미끄러진다. 균일 정책을 다시 평가하라. `V[start]`가 더 좋아지는가 나빠지는가? 왜 그런가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| MDP | "강화 학습 설정" | 마르코프 성질을 만족하는 튜플 `(S, A, P, R, γ)`. |
| 상태 | "에이전트가 보는 것" | 선택한 정책 클래스 하에서 미래 동역학에 대한 충분 통계량. |
| 정책 | "에이전트의 행동" | 조건부 분포 `π(a \| s)` 또는 결정론적 맵 `s → a`. |
| 리턴 | "전체 보상" | 현재 스텝부터의 할인 합 `Σ γ^t r_t`. |
| 가치 | "상태가 얼마나 좋은지" | `s`에서 시작하는 `π` 하의 기대 리턴. |
| Q-값 | "행동이 얼마나 좋은지" | 첫 행동 `a`로 `s`에서 시작하는 `π` 하의 기대 리턴. |
| 벨만 방정식 | "동적 계획법 재귀" | 가치/Q를 한 스텝 보상 더하기 할인된 후속 가치로 분해하는 고정점 분해. |
| 할인 `γ` | "미래 대 현재" | 먼 미래 보상에 대한 기하 가중치; 유효 지평 `~1/(1-γ)`. |

## 더 읽을거리 (Further Reading)

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf): 교과서. 3장이 MDP와 벨만 방정식을 다루고, 1장이 이후 모든 레슨의 바탕이 되는 보상 가설(reward hypothesis)에 동기를 부여한다.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming): 벨만 방정식의 기원.
- [OpenAI Spinning Up: Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) 심층 강화 학습 관점에서의 간결한 MDP 입문.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887): MDP와 정확 해법에 관한 운용 과학(operations-research) 레퍼런스.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf): MDP를 동적 계획법의 특수화로 보는 가장 깔끔한 유도.
