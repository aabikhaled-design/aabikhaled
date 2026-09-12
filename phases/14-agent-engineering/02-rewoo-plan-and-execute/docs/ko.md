# ReWOO와 Plan-and-Execute: 분리된 계획 수립

> ReAct는 사고와 행동을 하나의 스트림에서 번갈아 배치한다. ReWOO는 이를 분리한다: 앞쪽에 큰 계획 하나, 그다음 실행. 토큰 5배 절감, HotpotQA에서 정확도 +4%, 그리고 계획기(planner)를 7B 모델로 증류(distill)할 수 있다. Plan-and-Execute는 이를 일반화했고, Plan-and-Act는 이를 웹 내비게이션으로 확장했다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- ReWOO의 계획기(Planner) / 작업자(Worker) / 해결기(Solver) 분리가 ReAct의 번갈아 도는 루프에 비해 왜 토큰을 절약하고 견고성을 높이는지 설명하기.
- 계획 DAG, 의존성 순서 실행기(executor), 작업자 출력을 조합하는 해결기를 모두 stdlib로 구현하기.
- 2026년 "다섯 가지 워크플로 패턴(five workflow patterns)" 틀(Anthropic)을 사용하여, 작업을 계획-후-실행(plan-then-execute)으로 돌릴지 번갈아 도는 ReAct로 돌릴지 결정하기.
- 장기 지평(long-horizon) 웹 또는 모바일 작업에 Plan-and-Act의 합성 계획 데이터(synthetic plan data)가 언제 필요한지 인식하기.

## 문제 (The Problem)

ReAct의 번갈아 도는 사고-행동-관찰 루프는 단순하고 유연하지만, 도구를 호출할 때마다 이전의 모든 사고를 포함한 전체 사전 컨텍스트를 함께 실어 보내야 한다. 토큰 사용량은 깊이에 따라 제곱으로 늘어난다. 게다가 도구가 루프 도중 실패하면 모델은 에러 관찰로부터 전체 계획을 다시 유도해야 한다.

ReWOO(Xu et al., arXiv:2305.18323, 2023년 5월)는 이를 간파하고 한 가지에 승부를 걸었다: 전체를 앞쪽에서 계획하고, 증거를 병렬로 가져오고, 마지막에 답을 조합한다. 계획에 LLM 호출 한 번, 증거에 N번의 도구 호출(병렬 가능), 해결에 LLM 호출 한 번. 이렇게 맞바꾸면 유연성은 일부 포기하지만(계획이 정적이다) 훨씬 나은 토큰 효율과 더 명확한 실패 양상을 얻는다.

## 개념 (The Concept)

### 세 가지 역할

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

계획기는 DAG를 생성한다. 각 노드는 도구, 그 인자, 그리고 어느 앞선 노드에 의존하는지(예: `#E1`, `#E2` 같은 참조)를 명시한다. 작업자는 노드를 위상 정렬(topological) 순서로 실행한다. 해결기는 이 모든 조각을 하나로 엮는다.

### 왜 토큰이 5배 적은가

ReAct는 단계 수에 따라 프롬프트 길이가 선형으로 늘어난다. 10단계에서 프롬프트는 사고 1 더하기 행동 1 더하기 관찰 1 더하기 사고 2 더하기 행동 2 더하기 관찰 2 등을 담는다. 각 중간 단계는 원래 프롬프트도 중복으로 포함한다.

ReWOO는 계획기 프롬프트 하나(큼), N개의 작은 작업자 프롬프트(각각 도구 호출만, 체인 없음), 그리고 해결기 프롬프트 하나를 치른다. HotpotQA에서 논문은 토큰이 약 5배 적으면서 정확도 절대치 +4를 기록한다고 측정한다.

### 왜 더 견고한가

ReAct에서 작업자 3이 실패하면 루프는 스트림 도중에 에러를 빠져나오는 추론을 해야 한다. ReWOO에서는 작업자 3이 에러 문자열을 반환하고, 해결기는 그것을 원래 계획과 함께 컨텍스트에서 보고 성능을 단계적으로 낮출 수 있다. 실패 국소화는 단계별이 아니라 노드별이다.

### 계획기 증류 (Planner distillation)

논문의 두 번째 결과: 계획기가 관찰을 보지 않기 때문에, 175B 교사(teacher)의 계획기 출력으로 7B 모델을 파인튜닝(fine-tuning)할 수 있다. 작은 모델이 계획을 처리하고, 추론(inference) 시에는 큰 모델이 필요 없다. 이는 이제 표준이다. 많은 2026년 프로덕션 에이전트가 작은 계획기와 큰 실행기 또는 그 반대를 사용한다.

### Plan-and-Execute (2023)

LangChain 팀의 2023년 8월 게시글은 ReWOO를 패턴 이름으로 일반화했다: Plan-and-Execute. 앞쪽 계획기가 단계 목록을 방출하고, 실행기가 각 단계를 돌리며, 선택적 재계획기(replanner)가 결과를 관찰한 후 수정할 수 있다. 이는 ReWOO보다 ReAct에 가깝지만(재계획기가 관찰을 계획으로 되가져옴) 토큰 절감은 유지한다.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act는 이 패턴을 장기 지평 웹 및 모바일 에이전트로 확장한다. 핵심 기여는 합성 계획 데이터다: 레이블된 트래젝토리 생성기가 계획이 명시적인 학습 데이터를 생성한다. 단일 ReAct 트래젝토리가 일관성을 잃는 WebArena류 작업에서 30~50 단계를 넘어서도 계속 작동하는 계획기 모델을 파인튜닝하는 데 쓰인다.

### 어느 것을 고를지

| 패턴 | 언제 |
|---------|------|
| ReAct | 짧은 작업, 미지의 환경, 반응적 예외 처리 필요 |
| ReWOO | 알려진 도구를 쓰는 구조화된 작업, 토큰 민감, 병렬화 가능한 증거 |
| Plan-and-Execute | ReWOO와 비슷하나 부분 실행 후 재계획 |
| Plan-and-Act | 장기 지평(30단계 초과), 웹/모바일/컴퓨터 사용 |
| Tree of Thoughts | 탐색에 비용을 치를 가치가 있을 때(Lesson 04) |

Anthropic의 2024년 12월 지침: 가장 단순한 것부터 시작하라. 작업이 도구 호출 하나에 요약 하나라면 ReWOO를 만들지 마라. 작업이 40단계 연구 과제라면 ReAct만으로 하지 마라.

```figure
rewoo-plan
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 ReWOO를 구현한다:

- `Planner`: 프롬프트로부터 계획 DAG를 방출하는 스크립트된 정책.
- `Worker`: 레지스트리를 통해 각 노드의 도구 호출을 디스패치.
- `Solver`: 증거를 읽고 최종 답을 생성하는 스크립트된 조합.
- 의존성 해결: `#E1` 같은 참조가 앞선 작업자 출력으로 치환됨.

데모는 두 단계 계획으로 "프랑스 수도의 인구는 백만 단위로 반올림하면 얼마인가?"에 답한다: (1) 수도를 조회, (2) 인구를 조회, 그다음 해결.

실행:

```
python3 code/main.py
```

트레이스는 먼저 전체 계획을, 그다음 작업자 결과를, 그다음 해결기 조합을 보여준다. 토큰 수(대략적인 문자 수를 출력함)를 ReAct 스타일의 번갈아 도는 실행과 비교하라. 이런 종류의 구조화된 작업에서는 ReWOO가 이긴다.

## 라이브러리로 써보기 (Use It)

LangGraph는 Plan-and-Execute를 레시피로 제공한다(ReAct에는 `create_react_agent`, plan-execute에는 커스텀 그래프). CrewAI의 Flows는 이 패턴을 직접 인코딩한다: 작업을 앞쪽에서 정의하면 Flow DAG가 이를 실행한다. Plan-and-Act의 합성 데이터 접근법은 아직 대부분 연구 단계다. 런타임 패턴(명시적 계획 DAG)은 LangGraph와 CrewAI Flows를 통해 프로덕션에서 출하된다.

## 산출물 (Ship It)

`outputs/skill-rewoo-planner.md`는 도구 카탈로그가 주어졌을 때 사용자 요청으로부터 ReWOO 계획 DAG를 생성한다. 실행기로 넘기기 전에 계획을 검증한다(비순환, 모든 참조 해결됨, 모든 도구 존재함).

## 연습 문제 (Exercises)

1. 독립적인 계획 노드들에 대해 작업자 실행을 병렬화하라. 병렬 그룹 2개를 가진 6노드 DAG에서 무엇을 얻는가?
2. 어느 작업자든 에러를 반환하면 발동하는 재계획기 노드를 추가하라. ReWOO를 Plan-and-Execute로 만드는 가장 작은 변경은 무엇인가?
3. `Planner`를 작은 모델(7B급)로 교체하고 `Solver`는 프런티어(frontier) 모델에 유지하라. 종단 간(end-to-end) 품질을 비교하라. 이 분리가 어디서 실패하는가?
4. ReWOO 논문의 계획기 증류에 관한 4절을 읽어라. 175B -> 7B 결과를 개념적으로 재현하라: 어떤 학습 데이터가 필요하며, 계획 품질을 어떻게 채점하는가?
5. 장난감을 Plan-and-Act의 트래젝토리 형태로 이식하라: 계획은 DAG가 아니라 시퀀스다. 어떤 트레이드오프가 바뀌는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ReWOO | "관찰 없는 추론" | 계획하고, 증거를 병렬로 가져오고, 해결: 계획 프롬프트에 관찰 없음 |
| Plan-and-Execute | "LangChain의 plan-execute 패턴" | 실행 후 선택적 재계획기 노드를 가진 ReWOO |
| Plan-and-Act | "확장된 plan-execute" | 장기 지평 작업을 위한 합성 계획 학습 데이터를 갖춘 명시적 계획기/실행기 분리 |
| 증거 참조(Evidence reference) | "#E1, #E2, ..." | 디스패치 시점에 이전 작업자 출력으로 치환되는 계획 노드 자리표시자 |
| 계획기 증류(Planner distillation) | "작은 계획기, 큰 실행기" | 큰 교사의 계획기 트레이스로 작은 모델을 파인튜닝 |
| 토큰 효율(Token efficiency) | "더 적은 왕복" | 논문에서 ReAct 대비 HotpotQA 토큰 5배 절감 |
| DAG 실행기(DAG executor) | "위상 정렬 디스패처" | 계획 노드를 의존성 순서로 실행; 각 레벨에서 병렬 |

## 더 읽을거리 (Further Reading)

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323): 표준 논문
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572): 합성 계획을 갖춘 확장된 계획기-실행기
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview): 프레임워크 레시피
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 작동하는 가장 단순한 패턴을 고르기
