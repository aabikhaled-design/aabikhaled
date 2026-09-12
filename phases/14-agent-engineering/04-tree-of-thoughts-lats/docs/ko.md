# Tree of Thoughts와 LATS: 의도적 탐색(Deliberate Search)

> 단일 사고 사슬(chain-of-thought) 트래젝토리에는 되돌아갈 여지가 없다. ToT(Yao et al., 2023)는 추론을 각 노드에 자기 평가가 붙은 트리로 바꾼다. LATS(Zhou et al., 2024)는 ToT를 ReAct, Reflexion과 함께 몬테카를로 트리 탐색(Monte Carlo Tree Search) 아래 통합한다. Game of 24는 4%(CoT)에서 74%(ToT)로 올라가고, LATS는 HumanEval에서 92.7% pass@1을 달성한다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 추론을 탐색으로 틀 잡기: 노드는 "사고", 간선은 "확장", 값은 "얼마나 유망한지".
- 자기 평가 채점을 갖춘 stdlib 기반 ToT 스타일 BFS 트리 탐색을 구현하기.
- 선택 / 확장 / 시뮬레이션 / 역전파(select / expand / simulate / backpropagate)를 갖춘 장난감 LATS MCTS 루프로 확장하기.
- 탐색이 토큰 배수를 치를 가치가 있을 때(Game of 24, 코드 생성)와 단일 트래젝토리로 충분할 때(단순 Q&A)를 결정하기.

## 문제 (The Problem)

사고 사슬은 선형 보행이다. 첫 단계가 틀리면, 이후의 모든 단계는 나쁜 전제 위에서 작동한다. Game of 24(네 자릿수를 + − × ÷로 써서 24 만들기)에서 GPT-4 CoT는 4% 정확도를 기록한다. 모델은 일찍 틀린 부분식을 골라 회복하지 못한다.

추론에 필요한 것은 여러 후보를 제안하고, 평가하고, 유망한 것을 고르고, 막다른 길이 나타나면 되돌아갈 능력이다. 그것이 탐색이다. Tree of Thoughts와 LATS는 두 가지 표준 정식화다.

## 개념 (The Concept)

### Tree of Thoughts (Yao et al., NeurIPS 2023)

각 노드는 일관된 중간 단계("하나의 사고")다. 각 노드는 K개의 자식 사고로 확장될 수 있다. LLM은 채점 프롬프트로 각 노드를 자기 평가한다. 탐색은 BFS, DFS, 또는 빔(beam)으로 트리를 훑는다.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

자기 평가가 핵심을 떠받치는 부분이다. 논문은 세 가지 변형을 보여준다: `sure / likely / impossible` 분류, `1..10` 수치 점수, 후보 간 투표. 셋 모두 Game of 24에서 CoT를 상당히 능가한다(GPT-4로 4% -> 74%).

### LATS (Zhou et al., ICML 2024)

LATS는 ToT, ReAct, Reflexion을 MCTS 아래 통합한다. LLM은 세 가지 역할을 한다:

- **정책(Policy)**: 후보 다음 행동을 제안(ReAct 스타일).
- **가치 함수(Value function)**: 부분 트래젝토리를 채점(ToT 스타일 자기 평가).
- **자기 성찰자(Self-reflector)**: 실패 시 자연어 성찰(Reflexion 스타일)을 쓰고, 이를 다음 롤아웃(rollout)의 출발점으로 다시 심는다.

환경 피드백(관찰)이 가치 함수에 섞여 들어가므로, 탐색은 모델의 의견만이 아니라 실제 도구 결과를 근거로 삼는다. 논문 시점 결과: GPT-4로 HumanEval pass@1 92.7%(SOTA), GPT-3.5로 WebShop 평균 75.9(그래디언트 기반 파인튜닝(fine-tuning)에 근접).

### MCTS, 최소한으로

반복당 네 단계:

1. **선택(Select)**: UCT(트리에 대한 상한 신뢰 구간, upper confidence bound for trees)를 사용하여 루트에서 리프(leaf)까지 보행.
2. **확장(Expand)**: 정책을 통해 K개의 자식 생성.
3. **시뮬레이션(Simulate)**: 정책을 사용하여 자식에서 롤아웃하고, 가치 함수(또는 환경 보상)로 리프를 채점.
4. **역전파(Backpropagate)**: 경로를 따라 위로 방문 횟수와 가치 추정치를 갱신.

UCT 공식: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. 첫 항은 활용(exploitation)이고, 둘째는 탐험(exploration)이다. 작업별로 `c`를 조정하라.

### 비용 현실

탐색은 토큰을 폭발시킨다. Game of 24에서 ToT는 CoT 토큰의 100~1000배를 쓰고, LATS도 비슷하다. 결코 공짜가 아니다. 탐색은 다음 경우에 아껴 쓴다:

- 단일 트래젝토리가 명백히 불충분한 작업(Game of 24, 복잡한 코드).
- 벽시계 시간(wall-clock)이 정확성보다 덜 중요한 작업.
- 값싸고 신뢰할 수 있는 가치 함수가 있는 작업(코드에 대한 단위 테스트, 수학에 대한 명시적 목표).

작업에 정답이 하나뿐이고 평가자가 시끄럽다면, 탐색은 오히려 상황을 악화시켜 "점수만 좋은" 틀린 답을 찾아내기 일쑤다.

### 2026년 위치

대부분의 프로덕션 에이전트는 LATS를 돌리지 않는다. 도구에 근거한 검증(CRITIC, Lesson 05)을 갖춘 ReAct를 돌린다. 탐색은 특수한 틈새에서만 쓰인다:

- 가치 함수로 테스트를 돌리는 코딩 에이전트(HumanEval 스타일).
- 여러 쿼리 경로를 탐험하는 심층 연구 에이전트.
- LangGraph 서브그래프 안의 계획 중심 워크플로.

AlphaEvolve(Lesson 11)는 2025년의 극단이다: 코드에 대한 진화적 탐색, 기계 검증 가능한 적합도(fitness), 프런티어 이득(56년 만의 첫 4x4 행렬곱 개선).

```figure
tree-of-thoughts
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다:

- 양식화된 "산술 연산 고르기" 작업에 대한 작은 ToT BFS.
- 같은 작업에 대한 장난감 LATS MCTS 루프(선택 / 확장 / 시뮬레이션 / 역전파), UCT 선택 포함.
- 기호 점수 더하기 자기 평가 점수를 조합하는 가치 함수.

실행:

```
python3 code/main.py
```

트레이스는 ToT가 BFS로 노드당 세 후보를 확장하는 것을, MCTS를 통해 최선의 롤아웃으로 수렴하는 LATS와 비교하여 보여준다. 둘 다 토큰 수가 출력된다.

## 라이브러리로 써보기 (Use It)

LangGraph는 ToT 스타일 탐험을 서브그래프 패턴으로 제공한다. LATS에 관한 LangChain 팀의 블로그(2024년 5월)가 참조 튜토리얼이다. LlamaIndex는 `TreeOfThoughts` 에이전트를 제공한다. 대부분의 2026년 프로덕션 에이전트에서 이 패턴은 `if task_complexity > threshold: use_search()` 게이트 뒤에 산다. Lesson 05의 평가자-최적화기(evaluator-optimizer) 패턴 참고.

## 산출물 (Ship It)

`outputs/skill-search-policy.md`는 작업 형태, 예산, 평가자 충실도가 주어졌을 때 선형 ReAct, ToT, LATS, 진화적 탐색 중에서 선택한다.

## 연습 문제 (Exercises)

1. 장난감 LATS를 UCT c=0.1과 c=2.0으로 돌려라. 트레이스에서 무엇이 바뀌는가?
2. 가치 함수를 더 시끄러운 채점기로 교체하라(무작위 지터 추가). MCTS가 여전히 최선의 리프를 찾는가? 견디는 최소 신호 대 잡음비는 얼마인가?
3. 빔 탐색 ToT(각 레벨에서 top-k 유지)를 구현하고 BFS와 비교하라. 빠듯한 토큰 예산에서 어느 쪽이 더 나은가?
4. LATS 5.1절을 읽어라. HumanEval 트래젝토리 수를 재현하라: 보고된 pass@1을 달성하는 데 롤아웃이 몇 번 걸리는가?
5. "LATS가 덜 도움이 될 때"에 관한 LATS 논문의 논의를 읽어라. 작업 형태를 탐색 전략에 매핑하는 한 문단짜리 결정 규칙을 써라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Tree of Thoughts | "분기하는 CoT" | Yao et al.: 자기 평가가 붙은 사고 노드의 트리 |
| LATS | "LLM을 위한 MCTS" | Zhou et al.: ToT + ReAct + Reflexion을 MCTS 아래 통합 |
| UCT | "상한 신뢰 구간" | 활용(Q)과 탐험(ln N / n)의 균형을 맞추는 선택 공식 |
| 가치 함수(Value function) | "이 상태가 얼마나 좋은가" | 프롬프팅된 LLM 점수 또는 환경 보상; 역전파에 공급 |
| 정책(Policy) | "행동 제안자" | ReAct 스타일 생성기; 후보 다음 사고/행동을 방출 |
| 롤아웃(Rollout) | "시뮬레이션된 트래젝토리" | 정책을 사용하여 노드에서 리프까지 보행, 가치로 채점 |
| 역전파(Backpropagate) | "조상 갱신" | 리프의 보상을 경로 위로 밀어 올려 방문 횟수와 Q를 갱신 |
| 탐색 비용(Search cost) | "토큰 폭발" | Game of 24에서 CoT의 100~1000배; 채택 전에 예산을 잡을 것 |

## 더 읽을거리 (Further Reading)

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601): 표준 논문
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406): Reflexion 피드백을 갖춘 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 탐색을 위한 서브그래프 패턴
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131): 프로그램적 평가자를 갖춘 진화적 탐색
