# 마음 이론과 창발적 협응 (Theory of Mind and Emergent Coordination)

> Li et al. (arXiv:2310.10701)은 협력적 텍스트 게임에서 LLM 에이전트(agent)가 **창발적 고차 마음 이론(emergent high-order Theory of Mind, ToM)**(다른 에이전트가 제3의 에이전트의 믿음에 대해 무엇을 믿는지에 관한 추론)을 보이지만, 컨텍스트 관리와 환각(hallucination)으로 인해 긴 시야의 계획 수립에는 실패함을 보였다. Riedl (arXiv:2510.05174)은 한 집단 전체에서 고차 시너지(higher-order synergy)를 측정했고, **오직** ToM 프롬프트(prompt) 조건만이 정체성과 연결된 차별화(identity-linked differentiation)와 목표 지향적 상보성(goal-directed complementarity)을 만들어 내며, 저용량 LLM은 허위 창발만을 보임을 발견했다. 즉, 협응의 창발은 프롬프트에 조건적이고 모델 의존적이며, 공짜로 얻어지지 않는다. 이 레슨은 최소한의 ToM 인식 에이전트를 구현하고, ToM 프롬프트가 있을 때와 없을 때 협력 과제를 실행하며, Riedl 2025 프로토콜에 대비하여 협응 차이(coordination delta)를 측정한다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75분

## 문제 (Problem)

다중 에이전트(multi-agent) 협응은 종종 마법처럼 보인다. 에이전트들이 노동을 분담하고, 서로를 예측하며, 중복을 피한다. 보통 이 "창발(emergence)"은 프롬프트 엔지니어링의 산물이다. 누군가 에이전트들에게 "협응하라"고 말한 것이다. 그 프롬프트를 제거하면 협응도 사라진다.

Riedl의 2025년 발견은 더 엄격하다. 통제된 조건에서 협응은 에이전트들이 **다른 에이전트의 마음**(ToM)에 대해 추론하도록 프롬프트될 때만 창발한다. ToM 프롬프트가 없으면 강력한 모델조차 통계적 통제를 견디지 못하는 협응 패턴을 보인다. 이것이 프로덕션(production)에서 중요한 이유는 프롬프트에 의존하고 깨지기 쉬운 "다중 에이전트 협응" 기능을 출시하는 팀이 많기 때문이다.

이 레슨은 ToM을 특정한 능력(믿음에 대한 믿음을 추론하는 능력)으로 다루고, 최소한의 ToM 인식 에이전트를 만들며, 진짜 협응이 어떤 모습인지와 프롬프트 치장이 어떤 모습인지를 측정한다.

## 개념 (Concept)

### ToM의 의미

발달 심리학: 3세 아동은 누구의 내면 세계든 자신의 것과 일치한다고 생각한다. 5세 아동은 타인이 다른 믿음을 가진다는 것을 이해한다. 7세 아동은 믿음에 대한 믿음을 추론한다("그녀는 내가 공이 컵 아래에 있다고 생각한다고 믿는다"). 이것이 각각 0차, 1차, 2차 ToM이다.

LLM 에이전트의 경우, ToM 차수는 다음에 대응한다.

- **0차(Zeroth-order):** 타인에 대한 모델이 없다. 에이전트는 자신의 관찰만으로 행동한다.
- **1차(First-order):** 에이전트가 다른 각 에이전트의 믿음에 대한 모델을 가진다. "Alice는 X를 믿는다."
- **2차(Second-order):** 에이전트가 재귀적 믿음을 모델링한다. "Alice는 Bob이 X를 믿는다고 믿는다."

Li et al. 2023은 1차 및 2차 ToM이 협력 게임에서 LLM 에이전트에 창발하지만, 긴 시야와 신뢰할 수 없는 통신에서는 저하됨을 발견했다.

### 샐리-앤 검사, 간략히

1985년의 거짓 믿음 검사(false-belief test): 샐리(Sally)가 바구니 A에 구슬을 넣고 떠난다. 앤(Anne)이 그것을 바구니 B로 옮긴다. 샐리가 돌아오면 어디를 볼까? 1차 ToM을 가진 아이는 바구니 A라고 답한다(샐리의 믿음은 현실과 다르다). ToM이 없는 아이는 바구니 B라고 답한다.

GPT-4 시대의 LLM은 평이하게 제시되면 샐리-앤 유형의 검사를 통과한다. 서사가 길거나, 장면이 여러 번 바뀌거나, 질문이 간접적으로 표현되면 실패한다. 이것이 프로덕션 LLM에서 ToM의 2026년 실제 상태다.

### Riedl의 협응 측정

Riedl (arXiv:2510.05174)은 집단 규모의 검사를 구축했다. N개의 에이전트, 협력 목표, 가변적 프롬프트 조건. 측정 항목은 다음과 같다.

1. **정체성과 연결된 차별화(Identity-linked differentiation).** 에이전트들이 시간이 지나면서 안정적인 역할 구분을 발전시키는가?
2. **목표 지향적 상보성(Goal-directed complementarity).** 에이전트들의 행동이 중복되지 않고(서로 다른 하위 과제) 상보적인가?
3. **고차 시너지(Higher-order synergy).** 어떤 부분 집합도 달성할 수 없는 것을 그룹이 달성하는지에 대한 통계적 측정치.

결과: ToM 프롬프트 조건에서만 세 지표 모두 베이스라인(baseline) 위의 신호를 만들어 낸다. ToM 프롬프트가 없으면, 중간 용량 모델의 경우 지표들이 우연 수준 근처에 머무른다. 대형 모델은 명시적 ToM 프롬프트 없이도 일부 협응을 보이지만, 그 효과는 명시적 프롬프트가 있을 때보다 작다.

### 협응 착시

통계적 통제가 없으면, 데모에서의 "창발적 협응"은 종종 다음을 반영한다.

- 협응을 내장하는 프롬프트 엔지니어링("함께 일하라"고 말하는 시스템 프롬프트).
- 관찰자 편향(우리가 기대하는 패턴을 본다).
- 성공한 실행만 사후 선택하기.

측정 가능한 신호 없이 "창발적 협응"을 마케팅하는 프로덕션 시스템은 마케팅으로 취급해야 한다. 주장하기 전에 측정하라.

### 최소한의 ToM 인식 에이전트

구조:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 속성이 ToM 상태다. 1차 ToM은 한 수준만 유지한다. 2차는 `other_models[i][other_models_of_j]`를 추가한다. 내가 에이전트 i가 에이전트 j에 대해 믿는다고 생각하는 것.

### 긴 시야가 해가 되는 이유

Li et al.이 기록한 바: 컨텍스트 한계로 인해 에이전트는 어떤 믿음이 누구의 것인지 잊어버린다. 환각은 다른 에이전트 모델에 거짓 믿음을 추가한다. 둘 다 시간이 지나면서 누적되는 "나는 그가 X를 생각한다고 생각했다" 오류를 만든다.

논문과 2024-2026년 후속 연구에서 기록된 완화책:

- **프롬프트에 명시적 ToM 상태.** 구조화된 형식: `{agent_id: belief_list}`. 정체성-믿음 결합을 보존하도록 검색을 강제한다.
- **더 짧은 추론 사슬.** 턴당 ToM 갱신을 줄이면 누적되는 환각이 감소한다.
- **외부 ToM 저장소.** LLM 컨텍스트 밖에 모델을 유지하고, 턴마다 관련 부분만 주입한다.

### 프로덕션에서 ToM이 실패하는 지점

- **적대적 환경.** 좋은 ToM을 가진 에이전트는 조작하기 더 쉽다(그 에이전트가 상대를 어떻게 모델링하는지를 다시 모델링한 뒤 이를 악용하면 된다).
- **이질적인 팀.** 모델들이 서로 다를 때, 한 상대에게 작동하는 ToM 모델은 일반화되지 않는다.
- **정답 의존적 과제.** ToM은 믿음에 관한 것이다. 정확성이 사실에 의존한다면, ToM은 방해 요소가 될 수 있다.

### 실제로 측정할 수 있는 협응

팀의 협응이 프롬프트로 치장된 것이 아니라 진짜임을 보여주는 세 가지 실용적 신호:

1. **시간에 따른 상보성.** 다중 턴 과제에서 에이전트들의 행동이 서로 겹치지 않는 하위 과제를 다루는가?
2. **예측.** 턴 T+1에서 에이전트 A의 행동이, 정확하게 들어맞은 턴 T+2의 B의 행동에 대한 예측에 의존하는가?
3. **교정.** A가 턴 T에서 B의 믿음을 잘못 읽었을 때, A가 턴 T+2까지 교정하는가?

이것들은 로그가 기록되는 다중 에이전트 시스템에서 측정 가능하다. 이것이 "협응" 서사의 실질적 버전이다.

```figure
sw-theory-of-mind
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다.

- `ToMAgent`: 자신의 믿음과 다른 에이전트별 믿음 모델을 추적한다.
- 협력 과제: 세 에이전트가 세 상자에서 세 토큰(token)을 수집해야 한다. 각 상자는 토큰 하나를 담을 수 있다. 에이전트들은 통신할 수 없고, 서로의 행동에서 의도를 추론한다.
- 두 가지 구성: `zeroth_order`(ToM 없음)와 `first_order`(한 수준 믿음 모델을 가진 ToM).
- 무작위화된 200회 시행에 대한 측정: 완료율, 중복률(두 에이전트가 같은 상자를 목표로 함), 완료까지의 평균 턴 수.

실행:

```
python3 code/main.py
```

예상 출력: 0차 에이전트는 약 35% 비율로 노력을 중복하고 10턴 내에 시행의 약 60%를 완료한다. 1차 ToM 에이전트는 약 5%로 중복하고 약 95%를 완료한다. 이 차이가 측정 가능한 협응 효과다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-tom-auditor.md`는 다중 에이전트 시스템의 "창발적 협응" 주장을 감사하는 스킬(skill)이다. 프롬프트 치장, 통제 대비 통계적 유의성, 측정된 상보성을 점검한다.

## 산출물 (Ship It)

협응 주장 체크리스트:

- **통제 조건.** 협응 프롬프트가 없는 시스템 버전. 둘 다 측정한다.
- **통계 검정.** 시스템과 통제 사이의 차이가 지표상 `p < 0.05`로 유의한가?
- **상보성 측정치.** 최종 성공만이 아니라, 시간에 따른 행동 비중복성.
- **실패 사례 로그.** 에이전트들이 협응에 실패할 때, ToM 상태는 어떤 모습인가?
- **모델 용량 공개.** 효과가 더 작은 모델에서 사라진다면, 그렇게 명시하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 1차 ToM이 중복률을 약 7배 줄이는지 확인하라. 에이전트 5개와 상자 5개로 확장해도 그 격차가 유지되는가?
2. 2차 ToM(에이전트 A가 B가 C에 대해 생각하는 것을 모델링)을 구현하라. 1차보다 개선되는가? 어떤 과제에서 그러한가?
3. ToM 상태에 **환각**을 주입하라: 턴마다 하나의 믿음을 무작위로 뒤집어라. 이것이 1차 성능을 얼마나 저하시키는가?
4. Li et al. (arXiv:2310.10701)을 읽어라. "긴 시야 저하" 발견을 재현하라. 턴이 10에서 30으로 늘어날 때 1차 ToM 성능은 어떻게 변하는가?
5. Riedl 2025 (arXiv:2510.05174)을 읽어라. 시뮬레이션 로그에 고차 시너지 통계량을 구현하라. ToM 프롬프트 조건 없이도 효과가 존재하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 마음 이론(Theory of Mind) | "타인의 마음을 이해함" | 다른 에이전트의 믿음을 모델링하는 능력. 차수(0, 1, 2+)로 등급이 매겨진다. |
| 샐리-앤 검사(Sally-Anne test) | "거짓 믿음 검사" | 1985년 발달 심리학. LLM은 평이한 버전은 통과하고 복잡한 것은 실패한다. |
| 1차 ToM(First-order ToM) | "A는 X를 믿는다" | 한 타인의 사실에 대한 믿음을 모델링함. |
| 2차 ToM(Second-order ToM) | "A는 B가 X를 믿는다고 믿는다" | 한 수준 더 깊은 재귀적 모델링. |
| 정체성과 연결된 차별화(Identity-linked differentiation) | "시간에 따른 안정적 역할" | Riedl의 지표: 역할이 무작위가 아니라 지속됨. |
| 목표 지향적 상보성(Goal-directed complementarity) | "비중복 행동" | 에이전트들이 같은 것이 아니라 서로 다른 하위 과제를 목표로 함. |
| 고차 시너지(Higher-order synergy) | "그룹이 어떤 부분 집합도 능가함" | 진짜 협응에 대한 Riedl의 통계적 측정치. |
| 협응 착시(Coordination illusion) | "협응하는 것처럼 보임" | 측정 가능한 신호 없이 프롬프트로 치장된 협응의 외관. |

## 더 읽을거리 (Further Reading)

- [Li et al.: Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) 협력 게임에서의 창발적 ToM; 긴 시야 실패 모드
- [Riedl: Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) 집단 규모 측정; ToM 프롬프트가 핵심 조건
- [Premack & Woodruff: Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) ToM 개념의 1978년 기원
- [Baron-Cohen, Leslie, Frith: Does the autistic child have a theory of mind?](https://doi.org/10.1016/0010-0277(85)90022-8) 샐리-앤 논문 (1985)
