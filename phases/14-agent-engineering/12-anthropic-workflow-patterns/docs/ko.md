# Anthropic의 워크플로 패턴: 복잡함보다 단순함

> Schluntz와 Zhang(Anthropic, 2024년 12월)은 워크플로(workflow, 미리 정의된 경로)와 에이전트(agent, 동적 도구 사용)를 구분한다. 다섯 가지 워크플로 패턴이 대부분의 경우를 다룬다. 직접 API 호출로 시작하라. 단계를 예측할 수 없을 때만 에이전트를 추가하라.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- Anthropic의 다섯 가지 워크플로 패턴: 프롬프트 체이닝(prompt chaining), 라우팅(routing), 병렬화(parallelization), 오케스트레이터-워커(orchestrator-workers), 평가자-최적화자(evaluator-optimizer)의 이름을 대기.
- 에이전트 대 워크플로 구분과 각각의 엔지니어링 비용을 설명하기.
- 에이전트보다 워크플로를 (그리고 그 반대를) 언제 선택할지 식별하기.
- 스크립트된 LLM에 대해 stdlib로 다섯 패턴을 모두 구현하기.

## 문제 (The Problem)

팀들은 단일 함수 호출이면 충분한 문제에 멀티 에이전트(multi-agent) 프레임워크를 가져다 쓴다. 그 비용은 실재한다. 프레임워크는 프롬프트를 가리고 제어 흐름을 숨기며 성급한 복잡성을 불러들이는 계층을 더한다. Schluntz와 Zhang의 2024년 12월 글은 가장 많이 인용되는 업계의 반론이다: 단순하게 시작하고, 비용을 치를 가치가 있을 때만 복잡성을 더하라.

## 개념 (The Concept)

### 워크플로 대 에이전트

- **워크플로.** 미리 정의된 코드 경로를 통해 조율되는 LLM과 도구들. 엔지니어가 그래프를 소유한다.
- **에이전트.** LLM이 자신의 도구를 동적으로 지시하고 자신의 단계를 밟는다. 모델이 그래프를 소유한다.

둘 다 제자리가 있다. 워크플로는 더 싸고 빠르고 디버그하기 쉽다. 에이전트는 개방형(open-ended) 문제를 열어주지만 실패 모드를 추론하기 어렵다.

### 증강된 LLM (The augmented LLM)

다섯 패턴 모두의 토대: 세 가지 능력(검색(search, retrieval), 도구(tools, actions), 메모리(memory, persistence))을 갖춘 하나의 LLM이다. 어떤 API 호출이든 이것들을 사용할 수 있다.

### 다섯 가지 패턴

1. **프롬프트 체이닝(Prompt chaining).** 호출 1의 출력이 호출 2의 입력이 된다. 작업이 깔끔한 선형 분해를 가질 때 사용한다. 단계 사이에 선택적 프로그래밍적 게이트(gate)를 둔다.

2. **라우팅(Routing).** 분류기(classifier) LLM이 어느 하위 LLM이나 도구를 호출할지 고른다. 범주적으로 다른 입력이 다른 처리를 필요로 할 때(1단계 지원 대 환불 대 버그 대 영업) 사용한다.

3. **병렬화(Parallelization).** N개의 LLM 호출을 동시에 실행하고 결과를 집계한다. 두 가지 형태: 분할(sectioning, 서로 다른 청크)과 투표(voting, 같은 프롬프트, N회 실행, 다수결/종합).

4. **오케스트레이터-워커(Orchestrator-workers).** 오케스트레이터 LLM이 어느 워커(역시 LLM)를 실행할지 동적으로 결정하고 그 출력을 종합한다. 에이전트 루프와 유사하지만 오케스트레이터는 무한정 루프하지 않는다.

5. **평가자-최적화자(Evaluator-optimizer).** 한 LLM이 답을 제안하고, 다른 LLM이 그것을 평가한다. 평가자가 통과시킬 때까지 반복한다. 이것은 일반화된 Self-Refine(Lesson 05)이다.

### 워크플로가 에이전트를 이기는 곳

- **예측 가능한 작업.** 단계를 열거할 수 있다면, 그렇게 해야 한다.
- **비용에 묶인 작업.** 워크플로는 유한한 단계 수를 가지며, 에이전트는 폭주할 수 있다.
- **컴플라이언스에 묶인 작업.** 감사자(auditor)는 궤적(trajectory)에서 추론하는 것이 아니라 그래프를 읽기 원한다.

### 에이전트가 워크플로를 이기는 곳

- **개방형 연구.** 다음 단계가 마지막 단계가 반환한 것에 의존할 때.
- **가변 길이 작업.** 단계 수를 알 수 없는, 수 분에서 수 시간의 작업.
- **새로운 도메인.** 올바른 워크플로를 아직 모를 때: 먼저 탐색하고 나중에 성문화하라.

### 컨텍스트 엔지니어링 동반자

"Effective context engineering for AI agents"(Anthropic 2025)는 인접한 분야를 형식화한다: 200k 윈도우는 컨테이너가 아니라 예산(budget)이다. 무엇을 포함할지, 언제 압축(compact)할지, 언제 컨텍스트가 자라도록 둘지. Phase 14의 컨텍스트 압축(context compression) 레슨에서 자세히 다룬다(이 커리큘럼의 재번호 이전 Phase 14의 이전 레슨 06).

```figure
workflow-chain
```

## 직접 만들기 (Build It)

`code/main.py`는 `ScriptedLLM`에 대해 다섯 워크플로 패턴을 모두 구현한다.

- `prompt_chain(input, steps)`: 순차적.
- `route(input, classifier, handlers)`: 분류 + 디스패치.
- `parallel_vote(prompt, n, aggregator)`: N회 실행, 집계.
- `orchestrator_workers(task, workers)`: 오케스트레이터가 워커를 고름.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`: 통과할 때까지 루프.

실행:

```
python3 code/main.py
```

각 패턴이 자신의 트레이스를 출력한다. 패턴당 총 코드 줄 수는 ~10-15줄이다. 프레임워크의 비용은 수천 줄로 측정된다.

## 라이브러리로 써보기 (Use It)

- 대부분의 작업에는 직접 API 호출.
- 패턴이 진정으로 영속 상태(LangGraph), 액터 모델(actor-model) 동시성(AutoGen v0.4), 또는 역할 템플릿화(role templating, CrewAI)를 필요로 할 때만 프레임워크.
- 처음부터 다시 만들지 않고 Claude Code 하니스(harness) 형태를 원할 때는 Claude Agent SDK에 손을 뻗어라.

## 산출물 (Ship It)

`outputs/skill-workflow-picker.md`는 주어진 작업 설명에 대해 올바른 패턴을 고르며, 결정 근거와 워크플로가 부족할 경우 에이전트로 가는 리팩터 경로를 포함한다.

## 연습 문제 (Exercises)

1. 신뢰도 임계값(confidence threshold)을 갖춘 라우팅을 구현하라. 임계값 미만이면 사람에게 에스컬레이션(escalate)한다. 1단계 지원 사용 사례에서 임계값은 어디에 자리하는가?
2. `parallel_vote`에 타임아웃(timeout)을 추가하라. 한 호출이 멈추면 어떻게 되는가? 누락된 표가 있을 때 어떻게 집계하는가?
3. `evaluator_optimizer`를 밴딧(bandit)으로 바꿔라. 반복에 걸쳐 상위 2개 출력을 유지하여, 늦게 나온 좋은 결과가 늦게 나온 나쁜 결과로 덮어쓰이지 않게 하라.
4. 프롬프트 체이닝을 라우팅과 결합하라: 라우터가 세 개의 체인 중 하나를 고른다. 단일 큰 프롬프트 대안 대비 토큰 비용을 측정하라.
5. 프로덕션 기능 하나를 골라라. 워크플로 그래프를 그려라. 단계를 세어라. 여기서 에이전트가 실제로 더 나을 것인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 워크플로(Workflow) | "미리 정의된 흐름" | LLM과 도구 호출로 이루어진, 엔지니어가 소유하는 그래프 |
| 에이전트(Agent) | "자율 AI" | 모델이 소유하는 그래프; 동적 도구 지시 |
| 증강된 LLM(Augmented LLM) | "도구를 가진 LLM" | LLM + 검색 + 도구 + 메모리; 원자적 단위 |
| 프롬프트 체이닝(Prompt chaining) | "순차 호출" | 호출 N의 출력이 호출 N+1의 입력 |
| 라우팅(Routing) | "분류기 디스패치" | 어느 체인/모델이 입력을 처리할지 고름 |
| 병렬화(Parallelization) | "팬 아웃(fan out)" | N개의 동시 호출; 분할 또는 투표로 집계 |
| 오케스트레이터-워커(Orchestrator-workers) | "디스패처 에이전트" | 오케스트레이터 LLM이 전문가 LLM을 동적으로 고름 |
| 평가자-최적화자(Evaluator-optimizer) | "제안자 + 심판" | 평가자가 통과시킬 때까지 반복; 일반화된 Self-Refine |

## 더 읽을거리 (Further Reading)

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents): 다섯 가지 워크플로 패턴
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): 동반 분야
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 상태 보존 그래프가 비용을 치를 가치가 있을 때
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): 제품화된 오케스트레이터-워커 패턴
