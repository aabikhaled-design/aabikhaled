# 멀티 에이전트 RL (Multi-Agent RL)

> 단일 에이전트(agent) 강화 학습(reinforcement learning)은 환경이 정상(stationary)이라고 가정한다. 같은 세계에 학습하는 두 에이전트를 두면 그 가정이 깨진다: 각 에이전트는 상대의 환경의 일부이며, 둘 다 변화하고 있다. 멀티 에이전트 RL은 마르코프(Markov) 가정이 더 이상 성립하지 않을 때 학습을 수렴(convergence)시키는 트릭의 집합이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45분

## 문제 (The Problem)

방을 탐색하는 법을 배우는 로봇은 단일 에이전트 강화 학습 문제다. 축구 팀은 아니다. AlphaStar 대 StarCraft 상대는 아니다. 입찰 에이전트들의 시장은 아니다. 4방향 정지에서 협상하는 두 자동차는 아니다. 다대다 실세계 문제는 아니다.

모든 멀티 에이전트 설정에서, 어느 한 에이전트의 관점에서 다른 에이전트들은 환경의 일부*이다*. 그들이 학습하고 행동을 바꾸면서 환경은 비정상(non-stationary)이 된다. 마르코프 성질("다음 상태는 오직 현재 상태와 나의 행동에만 의존한다")이 위반되는데, 다음 상태가 *다른* 에이전트들이 무엇을 선택했는지에도 의존하고 그들의 정책은 움직이는 표적이기 때문이다.

이는 표(tabular) 수렴 증명을 깨뜨린다(Q-러닝의 보장은 정상 환경을 가정한다). 순진한 심층 강화 학습도 깨뜨린다: 에이전트들이 루프 속에서 서로를 쫓으며 결코 안정적인 정책으로 수렴하지 않는다. 멀티 에이전트 특화 기법이 필요하다: 중앙집중 학습 / 분산 실행, 반사실(counterfactual) 베이스라인(baseline), 리그 플레이(league play), 자기 대국(self-play).

2026년 응용: 로봇 군집, 교통 라우팅, 자율주행 차량 함대, 시장 시뮬레이터, 멀티 에이전트 LLM 시스템(Phase 16), 그리고 둘 이상의 지능적 플레이어가 있는 모든 게임.

## 개념 (The Concept)

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**형식론: 마르코프 게임(Markov Game).** MDP의 일반화: 상태 `S`, 결합 행동 `a = (a_1, …, a_n)`, 전이 `P(s' | s, a)`, 그리고 에이전트별 보상 `R_i(s, a, s')`. 각 에이전트 `i`는 자신의 정책 `π_i` 하에서 자신의 리턴을 최대화한다. 보상이 동일하면, **완전 협력(fully cooperative)**이다. 영합(zero-sum)이면, **적대적(adversarial)**이다. 혼합이면, **일반합(general-sum)**이다.

**핵심 도전:**

- **비정상성.** 에이전트 `i`의 관점에서 `P(s' | s, a_i)`는 변화하는 `π_{-i}`에 의존한다.
- **신용 할당(credit assignment).** 공유 보상에서, 어느 에이전트가 그것을 일으켰는가?
- **탐험 조율.** 에이전트들은 같은 상태를 중복 탐험하는 것이 아니라 상호 보완적 전략을 탐험해야 한다.
- **확장성.** 결합 행동 공간이 `n`에 대해 지수적으로 커진다.
- **부분 관측성.** 각 에이전트는 자신의 관측만 본다; 전역 상태는 숨겨져 있다.

**네 가지 지배적 체제:**

**1. 독립 Q-러닝 / 독립 PPO (IQL, IPPO).** 각 에이전트가 다른 에이전트들을 환경의 일부로 취급하며 자신의 Q나 정책을 학습한다. 단순하고 때때로 작동한다(특히 경험 재현(experience replay)이 평활화 에이전트-모델링 트릭으로 작용할 때). 이론적 수렴: 없음. 실전에서: 느슨하게 결합된 작업에는 괜찮고, 단단하게 결합된 작업에는 나쁘다.

**2. 중앙집중 학습, 분산 실행 (centralized training, decentralized execution, CTDE).** 가장 흔한 현대 패러다임. 각 에이전트는 국소 관측 `o_i`에 조건화된 자신의 *정책* `π_i`를 가진다. 배포 시 표준 분산 실행. *학습* 중에는, 중앙집중 크리틱(critic) `Q(s, a_1, …, a_n)`이 전체 전역 상태와 결합 행동에 조건화된다. 예:
- **MADDPG** (Lowe et al. 2017): 에이전트별 중앙집중 크리틱을 가진 DDPG.
- **COMA** (Foerster et al. 2017): 반사실 베이스라인("내가 대신 행동 `a'`을 취했다면 내 보상이 어땠을까?"를 물어) 나의 기여를 분리한다.
- **MAPPO** / 공유 크리틱을 가진 **IPPO** (Yu et al. 2022): 중앙집중 가치 함수를 가진 PPO. 2026년 협력 MARL에 지배적.
- **QMIX** (Rashid et al. 2018): 가치 분해: 단조 혼합을 가진 `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`.

**3. 자기 대국.** 같은 에이전트의 두 복사본이 서로를 상대한다. 상대의 정책은 과거 스냅샷에서 온 내 정책*이다*. AlphaGo / AlphaZero / MuZero. OpenAI Five. 영합 게임에 가장 잘 작동한다; 학습 신호가 대칭적이다.

**4. 리그 플레이.** 일반합 / 적대적 환경으로의 자기 대국 확장: 과거와 현재 정책의 집단을 유지하고, 리그에서 상대를 샘플링하고, 그들에 대해 학습한다. 익스플로이터(exploiter, 현재 최선을 이기는 데 특화)와 메인 익스플로이터(main exploiter, 익스플로이터를 이기는 데 특화)를 추가한다. AlphaStar (StarCraft II). 게임이 "가위바위보" 전략 순환을 허용할 때 필요하다.

**통신.** 에이전트들이 학습된 메시지 `m_i`를 서로에게 보내도록 허용한다. 협력 설정에서 작동한다. Foerster et al. (2016)은 미분 가능한 에이전트 간 통신이 종단간(end-to-end)으로 학습될 수 있음을 보였다. 오늘날의 LLM 기반 멀티 에이전트 시스템(Phase 16)은 본질적으로 자연어로 통신한다.

```figure
f3-marl-orbit
```

## 직접 만들기 (Build It)

이 레슨은 두 협력 에이전트를 가진 6×6 GridWorld를 사용한다. 그들은 반대편 모서리에서 시작해 공유 목표에 도달해야 한다. 공유 보상: 어느 에이전트든 여전히 움직이는 동안 스텝당 `-1`, 둘 다 도착하면 `+10`. `code/main.py`를 보라.

### 1단계: 멀티 에이전트 환경

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*결합* 행동 공간은 `|A|² = 16`이다. 전역 상태는 두 위치다.

### 2단계: 독립 Q-러닝

각 에이전트는 결합 상태로 키가 지정된 자신의 Q-표를 실행한다. 각 스텝에서: 둘 다 ε-탐욕 행동을 고르고, 결합 전이를 수집하고, 각각 공유 보상으로 자신의 Q를 갱신한다.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

보상이 조밀하고 정렬되어 있기 때문에 이 작업에서 작동한다. 단단하게 결합된 작업(예: 한 에이전트가 다른 에이전트를 *기다려야* 하는 경우)에서는 실패한다.

### 3단계: 분해된-가치 갱신을 가진 중앙집중 Q

결합 행동에 대한 하나의 Q `Q(s, a_1, a_2)`를 사용한다. 공유 보상으로 갱신한다. 실행 시 주변화(marginalizing)로 분산화한다: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. 지수적 결합 행동 공간을 *올바른* 전역 관점과 맞바꾼다.

### 4단계: 단순 자기 대국 (적대적 2-에이전트)

같은 에이전트, 두 역할. 에이전트 A를 에이전트 B에 대해 학습한다; `K` 에피소드 후, A의 가중치(weight)를 B에 복사한다. 대칭 학습, 일관된 진전. 축소판 AlphaZero 레시피다.

## 함정 (Pitfalls)

- **비정상 재현.** 독립 에이전트를 가진 경험 재현은 단일 에이전트보다 나쁜데, 오래된 전이가 이제는 쓸모없는 상대에 의해 생성되었기 때문이다. 해법: 재레이블하거나 최신성으로 가중한다.
- **신용 할당 모호성.** 긴 에피소드 후의 공유 보상; 어느 에이전트가 기여했는지 말할 명확한 방법이 없다. 해법: 반사실 베이스라인(COMA), 또는 에이전트별 보상 셰이핑(shaping).
- **정책 표류 / 쫓기.** 각 에이전트의 최선 응답이 서로의 갱신에 따라 변한다. 해법: 중앙집중 크리틱, 느린 학습률(learning rate), 또는 한-번에-하나-동결.
- **조율을 통한 보상 해킹.** 에이전트들이 설계자가 예상하지 못한 조율된 악용을 찾는다. 경매 에이전트가 입찰 0으로 수렴한다. 해법: 신중한 보상 설계, 행동 제약.
- **탐험 중복.** 두 에이전트가 같은 상태-행동 쌍을 탐험한다. 해법: 에이전트별 엔트로피 보너스, 또는 역할 조건화.
- **리그 순환.** 순수 자기 대국은 지배 순환에 갇힐 수 있다. 해법: 다양한 상대를 가진 리그 플레이.
- **표본 폭발.** `n` 에이전트 × 상태 공간 × 결합 행동. 함수 근사(function approximation)로 근사한다; 인수분해된 행동 공간(에이전트당 하나의 정책 출력 헤드).

## 라이브러리로 써보기 (Use It)

2026년 MARL 응용 지도:

| 도메인 | 방법 | 비고 |
|--------|--------|-------|
| 협력 탐색 / 조작 | MAPPO / QMIX | CTDE; 공유 크리틱 + 분산 액터(actor). |
| 2인 게임 (체스, 바둑, 포커) | MCTS를 가진 자기 대국 (AlphaZero) | 영합; 대칭 학습. |
| 복잡한 멀티플레이어 (Dota, StarCraft) | 리그 플레이 + 모방 사전 학습(pretraining) | OpenAI Five, AlphaStar. |
| 자율주행 차량 함대 | 어텐션(attention)을 가진 CTDE MAPPO / PPO | 부분 관측; 가변 팀 크기. |
| 경매 시장 | 게임 이론적 균형 + RL | `n` → ∞일 때 평균장(mean-field) RL. |
| LLM 멀티 에이전트 시스템 (Phase 16) | 자연어 통신 + 역할 조건화 | 에이전트-계획 계층에서의 RL 루프. |

2026년에 MARL의 가장 큰 성장 영역은 LLM 기반이다: 협상하고, 토론하고, 소프트웨어를 만드는 언어 모델 에이전트의 군집. RL은 토큰 수준이 아니라 *궤적 수준(trajectory-level)* 출력에 대한 선호 최적화로 나타난다(Phase 16 · 03).

## 산출물 (Ship It)

`outputs/skill-marl-architect.md`로 저장하라:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 연습 문제 (Exercises)

1. **쉬움.** 2-에이전트 협력 GridWorld에서 독립 Q-러닝을 학습하라. 평균 리턴 > 0이 될 때까지 몇 에피소드가 걸리는가? 결합 학습 곡선을 그려라.
2. **중간.** "조율" 작업을 추가하라: 두 에이전트가 같은 턴에 목표에 올라설 때만 목표에 도달한다. 독립 Q가 여전히 수렴하는가? 무엇이 깨지는가?
3. **어려움.** MAPPO 스타일 학습을 위한 중앙집중 크리틱을 구현하고, 조율 작업에서 독립 PPO와 수렴 속도를 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| 마르코프 게임 | "멀티 에이전트 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; 각 에이전트가 자신의 보상을 가짐. |
| CTDE | "중앙집중 학습, 분산 실행" | 학습 시 결합 크리틱; 각 에이전트의 정책은 국소 관측만 사용. |
| IPPO | "독립 PPO" | 각 에이전트가 PPO를 별도로 실행. 단순 베이스라인; 종종 과소평가됨. |
| MAPPO | "멀티 에이전트 PPO" | 전역 상태에 조건화된 중앙집중 가치 함수를 가진 PPO. |
| QMIX | "단조 가치 분해" | `Q_tot = f_monotone(Q_1, …, Q_n)`이 분산 argmax를 허용. |
| COMA | "반사실 멀티 에이전트" | 어드밴티지(advantage) = 내 Q 빼기 내 행동에 대해 주변화한 기대 Q. |
| 자기 대국 | "에이전트 대 과거 자신" | 단일 에이전트, 두 역할; 영합 게임의 표준. |
| 리그 플레이 | "집단 학습" | 과거 정책을 캐시하고, 풀에서 상대를 샘플; 전략 순환을 처리. |

## 더 읽을거리 (Further Reading)

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275): 중앙집중 크리틱을 가진 CTDE.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926): 신용 할당을 위한 반사실 베이스라인.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485): 단조성을 가진 가치 분해.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955): PPO가 MARL에 놀랍도록 강하다.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z): 대규모 리그 플레이.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270): 영합 게임에서의 순수 자기 대국.
- [Sutton & Barto (2018). Ch. 15: Neuroscience & Ch. 17: Frontiers](http://incompleteideas.net/book/RLbook2020.pdf): 멀티 에이전트 설정과 CTDE가 풀도록 설계된 비정상성 문제에 대한 교과서의 짧은 다룸을 포함.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635): 수렴 결과와 함께 협력, 경쟁, 혼합 MARL을 다루는 개관.
