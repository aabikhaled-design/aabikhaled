# 심층 Q-네트워크 (Deep Q-Networks, DQN)

> 2013년: Mnih은 원본 픽셀로 하나의 Q-러닝 네트워크를 학습시켜, 일곱 개 Atari 게임에서 모든 고전적 강화 학습(reinforcement learning) 에이전트를 이겼다. 2015년: 49개 게임으로 확장해 Nature에 발표하며 심층 강화 학습 시대를 촉발했다. DQN은 Q-러닝(Q-learning)에 함수 근사(function approximation)를 안정화하는 세 가지 트릭을 더한 것이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75분

## 문제 (The Problem)

표(tabular) Q-러닝은 모든 (상태, 행동) 쌍에 대해 별도의 Q-값이 필요하다. 체스 보드는 약 10⁴³개의 상태를 가진다. Atari 프레임은 210×160×3 = 100,800개의 특성(feature)이다. 표 강화 학습은 수십억은커녕 수천 개의 상태에서 죽는다.

해법은 돌이켜보면 명백하다: Q-표를 신경망(neural network) `Q(s, a; θ)`로 교체한다. 하지만 돌이켜보면-명백한 것에 수십 년이 걸렸다. Q-러닝과 함께하는 순진한 함수 근사는 "치명적 삼중주(deadly triad)"(함수 근사 + 부트스트래핑(bootstrapping) + 오프-폴리시(off-policy) 학습) 하에서 발산한다. Mnih et al. (2013, 2015)은 학습을 안정화하는 세 가지 공학적 트릭을 찾아냈다:

1. **경험 재현(experience replay)**은 전이를 비상관화한다.
2. **타깃 네트워크(target network)**는 부트스트랩 타깃을 고정한다.
3. **보상 클리핑(reward clipping)**은 그래디언트(gradient) 크기를 정규화한다.

Atari에서의 DQN은 단일 하이퍼파라미터(hyperparameter) 세트를 가진 단일 아키텍처가 원본 픽셀로부터 수십 개의 제어 문제를 푼 최초의 사례였다. 그 이후 만들어진 모든 "심층 강화 학습"(DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57)은 이 세-트릭 기반 위에 쌓여 있다.

## 개념 (The Concept)

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**목적함수.** DQN은 신경 Q-함수에 대한 한-스텝 TD 손실(loss)을 최소화한다:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 온라인 네트워크, 경사 하강법(gradient descent)으로 매 스텝 갱신. `θ^-` = 타깃 네트워크, 주기적으로 `θ`에서 복사(약 10,000스텝마다). `D` = 과거 전이의 재현 버퍼(replay buffer).

**세 가지 트릭, 중요도 순으로:**

**경험 재현.** 약 10⁶개 전이의 링 버퍼(ring buffer). 각 학습 스텝은 미니배치(mini-batch)를 균일 무작위로 샘플링한다. 이는 시간적 상관(연속 프레임은 거의 동일하다)을 깨고 네트워크가 드문 보상 전이로부터 여러 번 학습하게 하며 연속된 그래디언트 갱신을 비상관화한다. 이것 없이는, 신경망을 가진 온-폴리시 TD가 Atari에서 발산한다.

**타깃 네트워크.** 벨만 방정식(Bellman equation)의 양변에 같은 네트워크 `Q(·; θ)`를 사용하면 타깃이 매 갱신마다 움직인다. "자기 꼬리 쫓기." 해법: 동결된 가중치(weight)를 가진 두 번째 네트워크 `Q(·; θ^-)`를 유지한다. `C`스텝마다 `θ → θ^-`로 복사한다. 이는 한 번에 수천 번의 그래디언트 스텝 동안 회귀 타깃을 안정화한다. 소프트 갱신 `θ^- ← τ θ + (1-τ) θ^-`(DDPG, SAC에서 사용)는 더 매끄러운 변형이다.

**보상 클리핑.** Atari 보상 크기는 1에서 1000+까지 다양하다. `{-1, 0, +1}`로 클리핑하면 어떤 단일 게임도 그래디언트를 지배하지 못한다. 보상 크기가 중요할 때는 잘못이지만 부호만 중요한 Atari에는 적합하다.

**더블 DQN(Double DQN).** Hasselt (2016)은 최대화 편향(maximization bias)을 고친다: 온라인 망을 행동을 *선택*하는 데, 타깃 망을 그것을 *평가*하는 데 사용한다.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

드롭인 교체이며, 일관되게 더 낫다. 기본으로 사용하라.

**기타 개선(Rainbow, 2017):** 우선순위 재현(prioritized replay, 높은 TD-오차 전이를 더 많이 샘플), 듀얼링 아키텍처(dueling architecture, `V(s)`와 어드밴티지 헤드 분리), 노이지 네트워크(noisy networks, 학습된 탐험), n-스텝 리턴, 분포적 Q(distributional Q, C51/QR-DQN), 다중 스텝 부트스트래핑. 각각 몇 퍼센트를 더하며, 이득은 대략 가산적이다.

```figure
f3-dqn-stability
```

## 직접 만들기 (Build It)

여기 코드는 표준 라이브러리(stdlib)만 쓰고 numpy를 쓰지 않는다. 작은 연속 GridWorld에서 손으로 만든 단일 은닉층(hidden layer) MLP를 사용하므로, 모든 학습 스텝이 마이크로초 안에 실행된다. 알고리즘은 대규모의 Atari DQN과 동일하다.

### 1단계: 재현 버퍼

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari에는 약 50,000 용량; 우리 장난감 환경에는 5,000이면 충분하다.

### 2단계: 작은 Q-네트워크 (수동 MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

순방향 패스(forward pass): 선형 → ReLU → 선형. 그것이 네트워크 전체다.

### 3단계: DQN 갱신

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

형태는 Lesson 04의 Q-러닝이며 두 가지 차이가 있다: (a) 표를 인덱싱하는 대신 미분 가능한 `Q(·; θ)`를 통해 역전파(backpropagation)한다, (b) 타깃이 `Q(·; θ^-)`를 사용한다.

### 4단계: 외부 루프

각 에피소드마다 `Q(·; θ)`에 대해 ε-탐욕으로 행동하고 전이를 버퍼에 넣고 미니배치를 샘플링하고 그래디언트 스텝을 밟은 뒤 주기적으로 `θ^- ← θ`로 동기화한다. 패턴:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16차원 원-핫(one-hot) 상태를 가진 우리의 작은 GridWorld에서, 에이전트는 약 500 에피소드 만에 거의 최적인 정책을 학습한다. Atari에서는 이를 2억 프레임으로 확장하고 CNN 특성 추출기를 추가한다.

## 함정 (Pitfalls)

- **치명적 삼중주.** 함수 근사 + 오프-폴리시 + 부트스트래핑은 발산할 수 있다. DQN은 타깃 망 + 재현으로 완화한다; 둘 중 어느 것도 제거하지 마라.
- **탐험.** ε는 감쇠해야 하며, 보통 학습의 첫 약 10% 동안 1.0에서 0.01로 감쇠한다. 충분한 초기 탐험이 없으면 Q-망이 국소 분지(local basin)로 수렴한다.
- **과대추정.** 노이즈가 있는 Q에 대한 `max`는 위쪽으로 편향된다. 프로덕션(production)에서는 항상 더블 DQN을 사용하라.
- **보상 스케일.** 보상을 클리핑하거나 정규화(normalization)하라; 그래디언트 크기는 보상 크기에 비례한다.
- **재현 버퍼 콜드스타트.** 버퍼에 수천 개의 전이가 쌓일 때까지 학습하지 마라. 약 20개 샘플에 대한 초기 그래디언트는 과적합(overfitting)한다.
- **타깃 동기화 빈도.** 너무 잦으면 ≈ 타깃 망 없음; 너무 드물면 ≈ 오래된 타깃. Atari DQN은 10,000 환경 스텝을 사용한다. 경험칙: 학습 지평의 약 1/100마다 동기화하라.
- **관측 전처리.** Atari DQN은 상태를 마르코프(Markov)로 만들기 위해 4개 프레임을 쌓는다. 속도 정보가 있는 모든 환경은 프레임 쌓기나 순환 상태가 필요하다.

## 라이브러리로 써보기 (Use It)

2026년에 DQN은 거의 최첨단은 아니지만 여전히 기준 오프-폴리시 알고리즘이다:

| 작업 | 선택 방법 | 왜 DQN이 아닌가? |
|------|------------------|--------------|
| 이산 행동 Atari류 | Rainbow DQN 또는 Muesli | 같은 틀, 더 많은 트릭. |
| 연속 제어 | SAC / TD3 (Phase 9 · 07) | DQN에는 정책 네트워크가 없다. |
| 온-폴리시 / 고처리량 | PPO (Phase 9 · 08) | 재현 버퍼 없음; 확장이 더 쉽다. |
| 오프라인 강화 학습 | CQL / IQL / Decision Transformer | 보수적 Q 타깃, 부트스트래핑 폭발 없음. |
| 큰 이산 행동 공간 (추천기) | 행동 임베딩을 가진 DQN, 또는 IMPALA | 적합; 장식이 중요하다. |
| LLM 강화 학습 | PPO / GRPO | 스텝 수준이 아닌 시퀀스 수준; 다른 손실. |

레슨은 여전히 전해진다. 재현과 타깃 네트워크는 SAC, TD3, DDPG, SAC-X, AlphaZero의 자기 대국 버퍼, 그리고 모든 오프라인 강화 학습 방법에 나타난다. 보상 클리핑은 PPO에서 어드밴티지 정규화로 살아남는다. 아키텍처는 청사진이다.

## 산출물 (Ship It)

`outputs/skill-dqn-trainer.md`로 저장하라:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 에피소드별 리턴 곡선을 그려라. 실행 평균이 -10을 초과할 때까지 몇 에피소드가 걸리는가?
2. **중간.** 타깃 네트워크를 비활성화하라(벨만 타깃의 양변에 온라인 망을 사용). 학습 불안정성을 측정하라. 리턴이 진동하는가 발산하는가?
3. **어려움.** 더블 DQN을 추가하라: 온라인 망으로 `argmax a'`를 고르고, 타깃 망으로 평가하라. 노이즈 보상 GridWorld에서 더블 DQN을 쓸 때와 안 쓸 때 1,000 에피소드 후 `Q(s_0, best_a)`의 편향을 참 `V*(s_0)`와 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| DQN | "심층 Q-러닝" | 신경 Q-함수, 재현 버퍼, 타깃 네트워크를 가진 Q-러닝. |
| 경험 재현 | "섞인 전이" | 각 그래디언트 스텝마다 균일하게 샘플되는 링 버퍼; 데이터를 비상관화. |
| 타깃 네트워크 | "동결된 부트스트랩" | 벨만 타깃에 사용되는 Q의 주기적 복사본; 학습을 안정화. |
| 치명적 삼중주 | "강화 학습이 발산하는 이유" | 함수 근사 + 부트스트래핑 + 오프-폴리시 = 수렴 보장 없음. |
| 더블 DQN | "최대화 편향에 대한 해법" | 온라인 망이 행동을 선택, 타깃 망이 평가. |
| 듀얼링 DQN | "V와 A 헤드" | Q = V + A - mean(A)로 분해; 같은 출력, 더 나은 그래디언트 흐름. |
| Rainbow | "모든 트릭" | DDQN + PER + 듀얼링 + n-스텝 + 노이지 + 분포적을 하나로. |
| PER | "우선순위 재현" | TD-오차 크기에 비례해 전이를 샘플. |

## 더 읽을거리 (Further Reading)

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602): 심층 강화 학습을 촉발한 2013 NeurIPS 워크숍 논문.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236): Nature 논문, 49-게임 DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581): 듀얼링 DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298): 트릭을 쌓은 논문.
- [OpenAI Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) 명료한 현대적 설명.
- [Sutton & Barto (2018). Ch. 9: On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) DQN의 타깃 네트워크와 재현 버퍼가 길들이도록 설계된 "치명적 삼중주"(함수 근사 + 부트스트래핑 + 오프-폴리시)의 교과서적 다룸.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/): 절제 연구(ablation studies)에서 사용되는 참조 단일 파일 DQN; 이 레슨의 밑바닥 버전과 함께 읽으면 좋다.
