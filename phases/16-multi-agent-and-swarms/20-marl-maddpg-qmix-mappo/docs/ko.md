# MARL — MADDPG, QMIX, MAPPO

> 2026년에도 LLM 에이전트(agent) 시스템에 여전히 영향을 주는, 다중 에이전트(multi-agent) 협응의 강화 학습(reinforcement learning) 유산이다. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275)는 중앙 집중 학습, 분산 실행(Centralized Training, Decentralized Execution, CTDE)을 도입했다. 각 비평자(critic)는 학습 중 모든 에이전트의 상태와 행동을 본다. 테스트 시에는 지역 행위자(actor)만 실행된다. 협력, 경쟁, 혼합 환경에서 작동한다. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485)는 단조 혼합 신경망(monotonic mixing network)을 갖춘 가치 분해(value-decomposition)다. 에이전트별 Q가 결합되어 결합 Q(joint Q)가 되므로 `argmax`가 깔끔하게 분배된다. StarCraft Multi-Agent Challenge(SMAC)에서 지배적이다. **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955)는 중앙 집중 가치 함수를 갖춘 PPO다. 최소한의 튜닝으로 입자 세계(particle-world), SMAC, Google Research Football, Hanabi에서 "놀랍도록 효과적"이다. 이 세 알고리즘은 분산적으로 행동해야 하는 에이전트 팀이 정책을 학습하는 토대다. MAPPO는 **2026년 협력 MARL 기본 베이스라인(baseline)**이다. 이 레슨은 작은 격자 세계(grid-world) 장난감에서 각각을 만들고, LLM 에이전트 학습에 손대기 전에 세 아이디어를 근육 기억에 새긴다.

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90분

## 문제 (Problem)

LLM 에이전트 시스템은 점점 더 에이전트 간 협응을 위한 정책을 학습한다. 언제 양보할지, 언제 행동할지, 어떤 동료를 호출할지. 그런 정책을 어떻게 학습하는지 알려주는 문헌은 다중 에이전트 강화 학습(Multi-Agent Reinforcement Learning, MARL)이며, LLM 물결보다 앞서 등장했고, 지배적인 알고리즘은 몇 가지로 좁혀진다.

패턴 어휘 없이 MARL 논문을 읽는 것은 고통스럽다. 중앙 집중 학습과 분산 실행(CTDE), 가치 분해, 중앙 집중 비평자는 유행어가 아니다. 특정 문제에 대한 특정 답이다.

- 독립 RL(각 에이전트가 홀로 학습)은 각 에이전트의 관점에서 비정상적(non-stationary)이다. 나쁘다.
- 중앙 집중 RL(한 에이전트가 모두를 제어)은 확장되지 않고 실행 제약을 위반한다.
- CTDE는 둘의 장점을 모두 취한다. 전역 정보로 학습하고, 지역 정책으로 배포한다.

## 개념 (Concept)

### 논문들이 사용하는 세 가지 환경

- **입자 세계(multi-agent particle env).** 협력/경쟁 과제를 가진 단순한 2D 물리. MADDPG의 원래 시험대.
- **StarCraft Multi-Agent Challenge (SMAC).** 협력적 미세 관리, 부분 관찰. QMIX의 시험대. 이산 행동, 연속 상태.
- **Google Research Football, Hanabi, MPE.** MAPPO 베이스라인.

환경마다 행동/관찰 유형이 다르므로, 알고리즘은 그에 맞게 고른다.

### MADDPG (2017): CTDE 패턴

각 에이전트 `i`에는 자신의 관찰을 행동으로 매핑하는 행위자 `mu_i(o_i)`가 있다. 또한 각 에이전트에는 학습 중 모든 관찰과 모든 행동을 보는 비평자 `Q_i(x, a_1, ..., a_n)`가 있다. 행위자는 비평자의 평가에 대한 정책 그래디언트(policy gradient)로 갱신된다.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

CTDE인 이유: 학습 시에는 모두의 행동을 알므로, 그것을 이용해 각 비평자의 분산을 줄인다. 배포 시에는 각 에이전트가 `o_i`만 보고 `mu_i(o_i)`를 호출한다.

실패 모드: 비평자가 N개의 에이전트에 따라 커진다(입력이 모든 행동을 포함). 근사 없이는 약 10개 에이전트를 넘어 확장되지 않는다.

### QMIX (2018): 가치 분해

협력 전용. 전역 보상은 에이전트별 Q값의 단조 함수의 합이다.

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

단조성은 `argmax_a Q_tot`가 각 에이전트가 독립적으로 `argmax_{a_i} Q_i`를 선택함으로써 계산될 수 있음을 보장한다. 이것이 바로 분산 실행에 필요한 속성이다. 학습 시에는 혼합 신경망이 에이전트별 Q로부터 `Q_tot`을 만든다.

QMIX가 SMAC에서 이기는 이유: 협력적 StarCraft 미세 관리는 동질적 에이전트, 지역 관찰, 전역 보상을 가진다. 가치 분해에 완벽하게 들어맞는다.

실패 모드: 단조성 제약은 제한적이다. 일부 과제는 단조 분해 불가능한 보상 구조를 가진다(한 에이전트가 팀을 위해 희생). 확장(QTRAN, QPLEX)이 이를 완화한다.

### MAPPO (2022): 간과된 기본값

다중 에이전트 PPO: 중앙 집중 가치 함수를 갖춘 PPO. 각 에이전트는 자신의 정책을 둔다. 모든 에이전트는 전체 상태를 보는 가치 함수를 공유한다(또는 에이전트별로 둔다). Yu et al. 2022는 MAPPO를 다섯 개 벤치마크에서 MADDPG, QMIX, 그리고 그 확장들과 비교 평가했고 다음을 발견했다.

- MAPPO는 입자 세계, SMAC, Google Research Football, Hanabi, MPE에서 오프 폴리시(off-policy) MARL 방법들과 대등하거나 이긴다.
- 최소한의 하이퍼파라미터(hyperparameter) 튜닝이 필요하다.
- 안정적인 학습. 시드(seed) 전반에 재현 가능하다.

이 논문 이전까지 커뮤니티는 온 폴리시(on-policy) MARL을 과소평가했다. 2026년에 MAPPO는 협력 MARL의 기본 베이스라인이다. 어떤 새 방법이든 이를 이겨야 한다.

### LLM 에이전트 엔지니어가 신경 써야 하는 이유

세 가지 직접적 용도:

1. **라우터(router) 학습.** 메타 에이전트가 어떤 하위 에이전트가 과제를 처리할지 선택한다. 이것은 N개의 분산 하위 에이전트와 하나의 중앙 집중 라우터를 가진 MARL 문제다. MAPPO가 들어맞는다.
2. **역할 창발.** 생성적 에이전트(generative-agent) 시뮬레이션에서, 에이전트가 시간이 지나면서 상보적 역할을 채택하도록 학습시키는 것은 사실 MARL 문제다. QMIX 스타일 가치 분해는 구성상 상보성을 강제한다.
3. **다중 에이전트 도구 사용.** 에이전트들이 도구를 공유하고 예산을 두고 경쟁할 때, CTDE를 통해 학습시키면 자원 제약을 존중하는 배포 가능한 지역 정책이 만들어진다.

실용적 주의사항: 2026년에 대부분의 프로덕션(production) LLM 에이전트 시스템은 정책을 학습시키기보다 프롬프트(prompt)로 지시한다. MARL은 (a) 많은 상호작용 데이터, (b) 명확한 보상 신호, (c) 학습 인프라에 투자할 의향이 모두 갖춰졌을 때 비로소 쓸 만하다.

### RL을 넘어선 설계 패턴으로서의 CTDE

학습 없이도, CTDE는 유용한 아키텍처 패턴이다.

- *설계* 중에는 전체 팀 가시성을 가정한다.
- *런타임*에는 분산 실행을 강제한다. 각 에이전트는 `o_i`만 본다.

이 패턴은 에이전트별 상태를 명시적으로 유지하고 부분 관찰 가능성을 미리 생각하도록 강제한다. 많은 프로덕션 다중 에이전트 시스템은 어디서나 공유 상태를 암묵적으로 가정한다. CTDE 규율이 그것을 방지한다.

### 비정상성 문제

여러 에이전트가 동시에 학습할 때, 각 에이전트의 환경(여기에는 다른 에이전트의 정책이 포함됨)은 비정상적이다. 고전적 단일 에이전트 RL 증명은 깨진다. 이 레슨의 MARL 알고리즘들은 모두 이를 다룬다.

- MADDPG: 전역 비평자가 모든 행동을 보므로, 그 가치 추정이 정상적이다.
- QMIX: 가치 분해는 학습을 최적성이 잘 정의된 결합 Q 공간으로 옮긴다.
- MAPPO: 중앙 집중 가치 함수가 다른 에이전트의 정책 변화로 인한 분산을 완화한다.

LLM 에이전트 시스템에서 비정상성은 "내 에이전트가 지난달에 작동했는데, 이제 상류의 다른 에이전트가 바뀌니 내 것이 오작동한다"로 나타난다. CTDE로 MARL을 학습시키는 것이 원칙에 입각한 해결책이다. 프롬프트 수준 해결책은 더 빠르지만 덜 오래간다.

### 이 레슨이 다루지 *않는* 것

실제 신경망(neural network) 학습은 Phase 09 주제다. 이 레슨은 그래디언트 갱신 없이 CTDE, 가치 분해, 중앙 집중 가치 패턴을 보여주는 스크립트 정책 버전을 만든다. 목표는 전체 MARL 라이브러리(PyMARL, MARLlib, RLlib multi-agent)를 집어 들기 전에 패턴을 내면화하는 것이다.

```figure
sw-ctde
```

## 직접 만들기 (Build It)

`code/main.py`는 세 가지 패턴 시연을 구현하며, 모두 작은 2 에이전트 협력 격자 세계에서 이루어진다.

- 환경: 4x4 격자 위의 2개 에이전트, 하나의 보상 알갱이. 어느 에이전트든 알갱이에 도달하면 보상 = 1이고 과제가 끝난다.
- `IndependentAgents`: 각 에이전트가 다른 에이전트를 환경으로 취급한다. 베이스라인.
- `MADDPGStyle`: 중앙 집중 비평자가 결합 가치를 계산한다. 행위자 정책이 그것으로부터 갱신된다. 스크립트화된 정책 개선.
- `QMIXStyle`: 단조 혼합기를 갖춘 가치 분해.
- `MAPPOStyle`: 중앙 집중 가치 함수. 정책이 공유 베이스라인에 대해 갱신된다.

네 가지 모두 같은 에피소드를 실행하고 목표까지의 평균 스텝을 보고한다. CTDE 변형들은 독립 베이스라인보다 짧은 경로로 수렴(convergence)한다.

실행:

```
python3 code/main.py
```

예상 출력: 독립 에이전트는 평균 약 6스텝이 걸린다. CTDE 변형들은 약 3.5스텝 쪽으로 수렴한다(4x4 격자의 최적은 3). 스크립트 정책에도 불구하고 패턴 차이가 나타난다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-marl-picker.md`는 주어진 다중 에이전트 과제에 대해 MARL 알고리즘을 고르는 스킬(skill)이다: 협력 대 경쟁, 동질 대 이질, 행동 공간 유형, 규모, 보상 신호.

## 산출물 (Ship It)

프로덕션에서 MARL은 드물다. 사용할 때는:

- **MAPPO로 시작하라.** 2022년 논문이 이를 베이스라인으로 확립했다. 먼저 재현하면 더 화려한 방법을 쫓는 데 드는 몇 주를 절약한다.
- **모든 에이전트의 관찰과 행동 스트림을 로깅하라.** 에이전트별 추적 없이 MARL을 디버깅하는 것은 가망이 없다.
- **학습 코드를 실행 코드에서 분리하라.** CTDE는 규율이다. 실행 경로가 정말로 `o_i`만 보게 하라.
- **보상 형성 경고.** MARL은 보상 설계에 극도로 민감하다. 형성에 협응 버그 하나만 있어도 에이전트가 그것을 악용하도록 학습한다. 적대적 테스트를 실행하라.
- **LLM 에이전트의 경우**, 먼저 프롬프트 수준 정책을 고려하라. 상호작용 데이터 + 보상 신호 + 인프라가 모두 있을 때만 MARL 학습에 투자하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 독립 에이전트와 MAPPO 스타일 에이전트 간의 목표까지 스텝 격차를 측정하라. 6x6 격자에서 격차가 커지는가, 줄어드는가?
2. 경쟁 변형을 구현하라: 두 에이전트, 하나의 알갱이, 먼저 도달한 자만 보상을 받는다. 어떤 패턴이 경쟁을 깔끔하게 다루는가? 역사적으로 MADDPG.
3. MADDPG (arXiv:1706.02275) 3절을 읽어라. 정확한 비평자 갱신 규칙을 자신의 말로 의사 코드에서 기호적으로 구현하라.
4. MAPPO (arXiv:2103.01955)을 읽어라. 저자들은 왜 중앙 집중 가치 + PPO가 그들의 벤치마크에서 오프 폴리시 MARL을 이긴다고 주장하는가? 가장 강력한 세 가지 주장을 나열하라.
5. CTDE를 설계 패턴으로 가상의 LLM 에이전트 시스템(예: 연구 에이전트 + 요약기 + 코더)에 적용하라. 설계 시에는 사용 가능하지만 런타임에는 사용 불가능한 결합 정보는 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| MARL | "다중 에이전트 RL" | 다중 에이전트 시스템을 위한 강화 학습. |
| CTDE | "중앙 집중 학습, 분산 실행" | 전역 정보로 학습하고, 지역 정책으로 배포. |
| MADDPG | "다중 에이전트 DDPG" | 모든 관찰 + 행동을 보는 에이전트별 비평자를 갖춘 CTDE. |
| QMIX | "가치 분해" | 에이전트별 Q의 단조 혼합. 협력. |
| MAPPO | "다중 에이전트 PPO" | 중앙 집중 가치 함수를 갖춘 PPO. 2026년 기본 베이스라인. |
| 가치 분해(Value decomposition) | "개별 Q의 합" | 에이전트별 Q의 단조 함수로 표현된 결합 Q. |
| 비정상성(Non-stationarity) | "움직이는 표적" | 다른 에이전트가 학습함에 따라 각 에이전트의 환경이 바뀜. MARL의 핵심 문제. |
| 온 폴리시 / 오프 폴리시(On-policy / off-policy) | "현재에서 학습 / 리플레이" | PPO는 온 폴리시(MAPPO). DDPG와 Q 학습은 오프 폴리시. |
| SMAC | "StarCraft Multi-Agent Challenge" | 협력적 미세 관리 벤치마크. QMIX의 자생적 기반. |

## 더 읽을거리 (Further Reading)

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/): MAPPO 결과의 읽기 쉬운 정리
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
