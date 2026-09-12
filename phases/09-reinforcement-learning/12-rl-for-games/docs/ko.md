# 게임을 위한 RL(AlphaZero, MuZero, 그리고 LLM 추론 시대 (RL for Games) AlphaZero, MuZero, and the LLM-Reasoning Era)

> 1992년: TD-Gammon이 순수 TD로 백개먼(backgammon)에서 인간 챔피언을 이겼다. 2016년: AlphaGo가 이세돌을 이겼다. 2017년: AlphaZero가 밑바닥부터 체스, 쇼기, 바둑을 지배했다. 2024년: DeepSeek-R1이 PPO를 GRPO로 대체한 같은 레시피가 추론에 작동함을 증명했다. 게임은 이 단계의 모든 돌파구를 이끄는 벤치마크(benchmark)다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120분

## 문제 (The Problem)

게임은 강화 학습(reinforcement learning)이 원하는 모든 것을 갖추고 있다. 깔끔한 보상(승/패). 무한한 에피소드(자기 대국(self-play) 리셋). 완벽한 시뮬레이션(게임이 곧 시뮬레이터다). 이산 또는 작은 연속 행동 공간. 적대적 견고성을 강제하는 멀티 에이전트(multi-agent) 구조.

그리고 게임은 모든 주요 강화 학습 돌파구가 검증된 방식이다. TD-Gammon(백개먼, 1992). Atari-DQN(2013). AlphaGo(2016). AlphaZero(2017). OpenAI Five(Dota 2, 2019). AlphaStar(StarCraft II, 2019). MuZero(학습된 모델, 2019). AlphaTensor(행렬 곱셈, 2022). AlphaDev(정렬 알고리즘, 2023). DeepSeek-R1(수학 추론, 2025): 게임-RL 기법이 텍스트에 작동한다는 최신 증명.

이 캡스톤(capstone)은 세 가지 이정표적 아키텍처(AlphaZero, MuZero, GRPO)를 하나의 통합 렌즈로 살펴본다: **자기 대국 + 탐색 + 정책 개선**. 각각은 이전 것을 일반화한다. 특히 GRPO는 토큰(token)을 행동으로, 수학적 검증을 승리 신호로 삼아 LLM 추론에 적용된 AlphaZero의 레시피다.

## 개념 (The Concept)

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**통합 루프.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. 알려진 규칙을 가진 게임(체스, 쇼기, 바둑)이 주어졌을 때:

- 정책-가치 네트워크: 하나의 타워 `f_θ(s) → (p, v)`. `p`는 합법 수에 대한 사전 분포(prior)다. `v`는 기대 게임 결과다.
- 몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS): 각 수에서, 가능한 연속의 트리를 확장한다. `(p, v)`를 사전 분포 + 부트스트랩(bootstrap)으로 사용한다. UCB(PUCT)로 노드를 선택한다: `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- 자기 대국: 에이전트(agent) 대 에이전트로 게임을 한다. 수 `t`에서, MCTS 방문 분포 `π_t`가 정책 학습 목표가 된다.
- 손실(loss): `L = (v - z)² - π · log p + c · ||θ||²`. `z`는 게임 결과(+1 / 0 / -1)다.

인간 지식 제로. 손으로 만든 휴리스틱(heuristic) 제로. 각각 수천만 번의 자기 대국 후 체스, 쇼기, 바둑을 정복한 단일 레시피.

**MuZero (2019).** Schrittwieser et al. 규칙이 알려져 있어야 한다는 요구를 제거한다.

- 고정된 환경 대신, *잠재 동역학 모델(latent dynamics model)* `(h, g, f)`를 학습한다:
  - `h(s)`: 관측을 잠재 상태로 인코딩.
  - `g(s_latent, a)`: 다음 잠재 상태 + 보상 예측.
  - `f(s_latent)`: 정책 사전 분포 + 가치 예측.
- MCTS가 *학습된 잠재 공간*에서 실행된다. 같은 탐색, 같은 학습 루프.
- 바둑, 체스, 쇼기 *그리고* Atari에서 작동한다. 하나의 알고리즘, 규칙 지식 없음.

**확률적 MuZero(Stochastic MuZero) (2022).** 확률적 동역학과 우연 노드(chance node)를 추가한다; 백개먼급 게임으로 확장한다.

**Muesli, Gumbel MuZero (2022-2024).** 표본 효율과 결정론적 탐색에 대한 개선.

**GRPO (2024-2025).** DeepSeek-R1 레시피. 언어 모델 추론에 적용된 같은 AlphaZero 형태의 루프:

- "게임": 수학 / 코딩 / 추론 문제에 답하기. "승리" = 검증기(verifier, 테스트 케이스 통과, 수치 답 일치)가 1을 반환.
- 정책: LLM. 행동: 토큰. 상태: 프롬프트(prompt) + 지금까지의 응답.
- 크리틱(critic, PPO 스타일 V_φ) 없음. 대신, 각 프롬프트에 대해 정책에서 `G`개의 완성을 샘플링한다. 각각에 대해 보상을 계산한다. REINFORCE 스타일 갱신의 신호로 **그룹 상대 어드밴티지(group-relative advantage)** `A_i = (r_i - mean_r) / std_r`를 사용한다.
- 표류를 막기 위한 참조 정책에 대한 KL 페널티(RLHF처럼).
- 전체 손실:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

보상 모델 없음, 크리틱 없음, MCTS 없음. 그룹 상대 베이스라인(baseline)이 셋 모두를 대체한다. 계산의 일부로 추론 벤치마크에서 PPO-RLHF 품질과 일치하거나 능가한다.

**R1 레시피 전체.** DeepSeek-R1(DeepSeek 2025)은 한 논문 안의 두 모델이다:

- **R1-Zero.** DeepSeek-V3 베이스 모델에서 시작. SFT 없음. 두 보상 구성요소로 GRPO를 직접 적용한다: *정확도 보상*(규칙 기반: 최종 답이 올바른 숫자로 파싱되었는가 / 코드가 단위 테스트를 통과했는가)과 *형식 보상*(완성이 사고 사슬(chain-of-thought)을 `<think>…</think>` 태그로 감쌌는가). 수천 스텝에 걸쳐, 평균 응답 길이가 약 100에서 약 10,000 토큰으로 자라고 수학 벤치마크 점수가 거의 o1-preview 수준으로 오른다. 모델이 밑바닥부터 추론하는 법을 배운다. 단점: 그 사고 사슬이 종종 읽을 수 없고, 언어를 섞고, 문체적 다듬음이 부족하다.
- **R1.** 4단계 파이프라인(pipeline)으로 R1-Zero의 가독성 문제를 고친다:
  1. **콜드스타트 SFT.** 깔끔한 형식을 가진 수천 개의 긴-CoT 시연을 수집한다. 그것들에 베이스 모델을 지도 파인튜닝(fine-tuning)한다. 이것이 읽을 수 있는 시작점을 준다.
  2. **추론 지향 GRPO.** 코드 전환(code-switching)을 막기 위한 *언어 일관성* 보상에 더해 정확도+형식 보상으로 GRPO를 적용한다.
  3. **거절 샘플링(rejection sampling) + SFT 2라운드.** RL 체크포인트에서 약 60만 개의 추론 궤적을 샘플링하고, 올바른 최종 답과 읽을 수 있는 CoT를 가진 것만 유지하고, 약 20만 개의 비추론 SFT 예제(글쓰기, QA, 자기 인지)와 결합한다. 베이스를 다시 파인튜닝한다.
  4. **전 스펙트럼 GRPO.** 추론(규칙 기반 보상)과 일반 정렬(유용성/무해성 선호 기반 보상) 모두를 다루는 한 번 더의 RL 라운드.

결과는 오픈 가중치(weight)로 AIME와 MATH-500에서 o1과 일치하며, 증류(distill)하기에 충분히 작다. 같은 논문은 R1의 추론 흔적에 SFT함으로써 여섯 개의 증류된 밀집 모델(Qwen-1.5B에서 Llama-70B까지)도 공개한다. 학생에는 RL 없음. 강한 RL 교사의 증류는 학생 규모에서 밑바닥부터의 RL을 일관되게 이긴다.

**추론에 PPO 대신 GRPO인 이유.** DeepSeekMath 논문(2024년 2월)의 세 가지 이유: (1) 학습할 가치 네트워크가 없어 메모리가 절반; (2) 그룹 베이스라인이 추론 작업이 만드는 희소한 궤적 끝 보상을 자연스럽게 처리; (3) 프롬프트별 정규화(normalization)가 매우 다른 난이도의 문제에 걸쳐 어드밴티지를 비교 가능하게 만드는데, PPO의 단일 크리틱은 그럴 수 없다.

**탐색 없음 대 탐색 기반.** 게임은 갈라졌다:

- *긴 지평을 가진 완전 정보 게임*(바둑, 체스): 여전히 탐색 기반. AlphaZero / MuZero가 지배한다.
- *LLM 추론*: 아직 프로덕션(production)에 MCTS 없음; 전체 롤아웃(rollout)에 GRPO, 추론 계산에 best-of-N. 과정 보상 모델(process reward models, PRMs)이 스텝 수준 탐색이 다시 추가될 것임을 암시한다.

```figure
f3-selfplay-ladder
```

## 직접 만들기 (Build It)

`code/main.py`의 코드는 **축소판 GRPO**(여러 샘플 그룹을 가진 밴딧(bandit))를 구현한다. 알고리즘은 LLM에서와 같다; 정책과 환경만 더 단순하다. 2025년 혁신인 *손실*과 *그룹 상대 어드밴티지*를 가르친다.

### 1단계: 작은 검증기 환경

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

실제 GRPO에서 검증기는 단위 테스트를 실행하거나 수학 동등성을 확인한다.

### 2단계: 정책: 프롬프트당 K개의 답 토큰에 대한 소프트맥스

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

프롬프트에 조건화된 LLM의 최종 층(layer) 출력과 동등하다.

### 3단계: 그룹 샘플링과 그룹 상대 어드밴티지

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

그룹 상대 어드밴티지가 2024년 DeepSeek 트릭이다. 크리틱이 필요 없다. "베이스라인"은 그룹 평균이고, 정규화는 그룹 표준편차를 사용한다.

### 4단계: REINFORCE 베이스라인(가치 없음)과 비교

같은 설정, 같은 계산, 평범한 REINFORCE. GRPO가 더 빠르고 더 안정적으로 수렴한다.

### 5단계: 엔트로피와 KL 관찰

RLHF와 같은 진단: 참조에 대한 평균 KL, 정책 엔트로피, 시간에 따른 보상. 이것들이 안정되면, 학습이 끝난 것이다.

## 함정 (Pitfalls)

- **검증기 농락을 통한 보상 해킹(reward hacking).** GRPO는 RLHF의 위험을 물려받는다: 검증기가 틀렸거나 악용 가능하면, LLM이 그 악용을 찾을 것이다. 견고한 검증기(다수의 테스트 케이스, 형식 증명)가 중요하다.
- **너무 작은 그룹 크기.** 그룹 베이스라인의 분산은 `1/√G`처럼 간다. `G = 4` 아래에서는 어드밴티지 신호가 노이즈가 심하다; 표준 선택은 `G = 8`에서 `64`다.
- **길이 편향.** 다른 길이의 LLM 완성은 다른 로그 확률을 가진다. 토큰 수로 정규화하거나, 시퀀스 수준 로그 확률을 사용하거나, 최대 길이로 절단하라.
- **순수 자기 대국 순환.** AlphaZero 스타일 학습은 일반합(general-sum) 게임에서 지배 루프에 갇힐 수 있다. 다양한 상대 풀(league play, Lesson 10)로 완화된다.
- **탐색-정책 불일치.** AlphaZero는 정책이 탐색 출력을 모방하도록 학습한다. 정책 망이 탐색의 분포를 표현하기에 너무 작으면, 학습이 멈춘다.
- **계산 하한.** MuZero / AlphaZero는 막대한 계산이 필요하다. 단일 절제(ablation)가 종종 수백 GPU-시간이다. 학습을 위한 축소판 데모(예: Connect Four에서의 AlphaZero)가 존재한다.
- **검증기 커버리지.** 버그 있는 해에 대해 통과하는 단위 테스트는 버그를 강화한다. 경계 사례를 잡는 검증기를 설계하라.

## 라이브러리로 써보기 (Use It)

2026년 게임-RL 지형, 도메인별:

| 도메인 | 지배적 방법 |
|--------|-----------------|
| 2인 영합 보드 게임 (바둑, 체스, 쇼기) | AlphaZero / MuZero / KataGo |
| 불완전 정보 카드 게임 (포커) | CFR + 딥러닝 (DeepStack, Libratus, Pluribus) |
| Atari / 픽셀 게임 | Muesli / MuZero / IMPALA-PPO |
| 대규모 멀티플레이어 전략 (Dota, StarCraft) | PPO + 자기 대국 + 리그 (OpenAI Five, AlphaStar) |
| LLM 수학/코드 추론 | GRPO (DeepSeek-R1, Qwen-RL, 오픈 재현) |
| LLM 정렬 | DPO / RLHF-PPO (GRPO 아님; 검증기가 검증 가능이 아니라 선호) |
| 로보틱스 | PPO + DR (게임-RL은 아니지만 같은 정책 그래디언트 도구 사용) |
| 조합 문제 | AlphaZero 변형 (AlphaTensor, AlphaDev) |

*레시피*(자기 대국, 탐색 증강 개선, 정책 증류)는 텍스트, 픽셀, 물리 제어를 아우른다. GRPO는 가장 어린 사례다; 더 많은 것이 오고 있다.

## 산출물 (Ship It)

`outputs/skill-game-rl-designer.md`로 저장하라:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에 GRPO 밴딧을 구현하라. 각 4개의 답 토큰을 가진 2개의 프롬프트로 학습하라. `G=8`로 1,000 갱신 미만에서 수렴하라.
2. **중간.** PPO(클리핑됨)와 바닐라 REINFORCE를 끼워 넣어라. 같은 밴딧에서 GRPO와 표본 효율 및 보상 분산을 비교하라.
3. **어려움.** 길이-2 "추론 사슬"로 확장하라: 에이전트가 두 토큰을 내보내고 검증기가 그 쌍을 보상한다. GRPO가 2-스텝 시퀀스에 걸친 신용 할당(credit assignment)을 어떻게 처리하는지 측정하라. (힌트: *전체 시퀀스*당 그룹 어드밴티지를 계산하고, 두 토큰 위치 모두에 전파하라.)

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| MCTS | "학습된 망을 가진 트리 탐색" | 몬테카를로 트리 탐색; 학습된 `(p, v)` 사전 분포를 가진 UCB1/PUCT 선택. |
| AlphaZero | "자기 대국 + MCTS" | MCTS 방문과 게임 결과에 일치하도록 학습된 정책-가치 망. |
| MuZero | "학습된-모델 AlphaZero" | 같은 루프지만 학습된 동역학을 통해 잠재 공간에서. |
| GRPO | "크리틱 없는 PPO" | 그룹 상대 정책 최적화(Group Relative Policy Optimization); 그룹-평균 베이스라인 + KL을 가진 REINFORCE. |
| PUCT | "AlphaZero의 UCB" | `Q + c · p · √N / (1 + N_a)`: 가치 추정과 사전 분포를 균형. |
| 자기 대국 | "에이전트 대 과거 자신" | 영합의 표준; 대칭 학습 신호. |
| 리그 플레이 | "집단 기반 자기 대국" | 과거 + 현재 + 익스플로이터를 상대로 샘플. |
| 검증기 보상 | "검증 가능 RL" | 보상이 결정론적 검사기(테스트 통과, 답 일치)에서 옴. |
| 과정 보상 | "PRM" | 최종 답만이 아니라 각 추론 스텝을 채점. |

## 더 읽을거리 (Further Reading)

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300): GRPO와 그룹 상대 베이스라인을 도입한 논문.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): 전체 4단계 R1 레시피와 R1-Zero 절제.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400): 대규모 CFR + 딥러닝.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343): 모든 것을 시작한 논문.
- [Hugging Face TRL: GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) 커스텀 보상 함수로 GRPO를 적용하는 프로덕션 레퍼런스.
- [Qwen Team (2024). Qwen2.5-Math: GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) 여러 규모에서 R1 레시피의 오픈 재현.
- [Sutton & Barto (2018). Ch. 17: Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) R1이 LLM 규모에서 구현하는 자기 대국, 탐색, "설계된 보상"에 대한 교과서적 틀.
