# HTN과 진화적 탐색을 활용한 계획 수립

> 기호적 계획 수립(symbolic planning)은 계획이 증명 가능하게 올바른 경우를 다룬다. 진화적 코드 탐색(evolutionary code search)은 적합도 함수(fitness function)가 기계로 검증 가능한 경우를 다룬다. ChatHTN(2025)과 AlphaEvolve(2025)는 각각이 LLM과 짝지어졌을 때 무엇을 열어주는지 보여준다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 계층적 작업 네트워크(Hierarchical Task Network): 작업(task), 방법(method), 연산자(operator), 전제 조건(precondition), 효과(effect)를 설명하기.
- ChatHTN의 하이브리드 루프(LLM 폴백(fallback) 분해를 갖춘 기호적 탐색)를 기술하기.
- AlphaEvolve의 진화적 루프와 그것이 왜 프로그래밍적 평가자(evaluator)가 있어야만 작동하는지 설명하기.
- stdlib로 토이 HTN 플래너와 토이 진화적 탐색을 구현하기.

## 문제 (The Problem)

ReWOO(Lesson 02), Plan-and-Execute, ReAct는 대부분의 에이전트(agent) 계획 수립을 다룬다. 이들이 잘 다루지 못하는 두 가지 경우가 있다.

1. **증명 가능한 정확성을 가진 계획.** 스케줄링, 항공 경로 설정, 컴플라이언스(compliance) 워크플로: 계획은 구성상(by construction) 건전해야 한다. 가끔 단계를 환각(hallucinate)하는 유창한 LLM 계획은 용납될 수 없다.
2. **기계로 검증 가능한 적합도 함수를 가진 최적화.** 행렬 곱셈, 스케줄링 휴리스틱(heuristic), 컴파일러 패스(pass): 목표는 "올바른 계획"이 아니라 "최선의 계획"이다.

HTN 계획 수립과 AlphaEvolve는 서로 다른 두 문제를 해결한다. 둘 다 LLM을 대체물이 아니라 증폭기(amplifier)로 사용한다.

## 개념 (The Concept)

### 계층적 작업 네트워크 (Hierarchical Task Networks)

HTN은 다음으로 구성된다.

- **작업(Tasks)**: 복합(compound, 분해되어야 함) 작업과 원시(primitive, 직접 실행 가능) 작업.
- **방법(Methods)**: 복합 작업을 하위 작업(subtask)으로 분해하는 방식들로, 전제 조건을 가진다.
- **연산자(Operators)**: 전제 조건과 효과를 가진 원시 행동.
- **상태(State)**: 사실(fact)들의 집합.

계획 수립: 목표 작업과 초기 상태가 주어졌을 때, 전제 조건이 순서대로 만족되는 원시 연산자들로의 분해를 찾는 것.

HTN은 LLM보다 오래되었으며, 여전히 증명 가능하게 올바른 계획의 레퍼런스다.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN(arXiv:2505.11814)은 기호적 HTN과 LLM 질의를 교차시킨다.

1. 기존 방법들로 현재 복합 작업을 분해하려 시도한다.
2. 적용 가능한 방법이 없으면 LLM에게 묻는다: "상태 `s`에서 `task`를 어떻게 분해하겠는가?"
3. LLM 응답을 후보 하위 작업으로 번역한다.
4. 연산자 스키마에 대해 검증한다. 유효하지 않은 분해는 거부한다.
5. 재귀한다.

논문의 핵심 주장: 생성되는 모든 계획은 증명 가능하게 건전한데, LLM 제안이 직접적인 계획 편집이 아니라 후보 분해로만 들어오기 때문이다. 기호적 계층이 정확성을 소유하고, LLM은 방법 라이브러리를 확장한다.

온라인 방법 학습(online method learning, OpenReview `gwYEDY9j2x`, 2025 후속 연구)은 LLM이 생성한 분해를 회귀(regression)로 일반화하는 학습자를 추가하여, LLM 질의 빈도를 최대 75%까지 줄인다.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve(arXiv:2506.13131, DeepMind, 2025년 6월)는 다른 종류의 짐승이다. Gemini 2.0 Flash/Pro 앙상블(ensemble)이 조율하는 진화적 코드 탐색이다.

루프:

1. 시드(seed) 프로그램 + 프로그래밍적 평가자(적합도 점수를 반환)로 시작한다.
2. LLM 앙상블이 변이(mutation)를 제안한다.
3. 변이를 평가자에 통과시킨다.
4. 최선의 것을 유지하고, 다시 변이한다.

발표된 성과:

- 56년 만에 처음으로 4x4 복소수 행렬 곱셈에서 Strassen을 능가하는 개선(스칼라 곱셈 48회).
- Borg 스케줄링 휴리스틱을 통해 Google 컴퓨트의 0.7% 회수.
- 프론티어 워크로드에서 FlashAttention 32% 속도 향상.

엄격한 제약: 적합도 함수는 기계로 검증 가능해야 한다. 산문(prose) 답변에 대한 진화적 탐색은 수렴(convergence)하지 않는다.

### 어느 것을 언제 쓸 것인가

| 문제 부류 | 사용 | 이유 |
|---------------|-----|-----|
| 강한 제약을 가진 스케줄링 | HTN + ChatHTN | 증명 가능한 건전성 |
| 컴파일러 최적화 | AlphaEvolve | 기계로 검증 가능한 적합도 |
| 다단계 작업 실행 | ReAct / ReWOO | 루프 안의 LLM, 형식적 보장 없음 |
| 테스트를 갖춘 코드 개선 | AlphaEvolve | 테스트가 평가자 |
| 정책에 묶인 자동화 | HTN | 전제 조건이 정책을 인코딩 |

### 이 패턴이 잘못되는 지점

- **연산자 없는 HTN.** 전제 조건/효과 스키마가 없으면 건전성 주장이 무너진다. ChatHTN의 "LLM이 분해를 제안" 방식은 유효하지 않은 수(move)를 거부하기 위해 스키마를 필요로 한다.
- **진짜 평가자 없는 AlphaEvolve.** "코드가 더 나은지 LLM에게 물어보라"는 적합도 함수가 아니다. 평가자는 결정론적(deterministic)이고 빨라야 한다.
- **과잉 설계(over-engineering).** 대부분의 에이전트 작업은 둘 다 필요로 하지 않는다. 먼저 ReAct나 ReWOO부터 써 보라.

```figure
htn-tree-expand
```

## 직접 만들기 (Build It)

`code/main.py`는 두 개의 토이를 구현한다.

- 연산자, 방법, 전제 조건, 효과, 그리고 복합 작업에 일치하는 방법이 없을 때 작동하는 `LLMFallback`을 갖춘 stdlib HTN 플래너. "LLM"은 스크립트된 분해자라서 플래너가 오프라인으로 실행된다.
- 산술 프로그램에 대한 stdlib 진화적 탐색: 테스트 집합에 걸쳐 출력이 `|f(x) - target|`을 최소화하는 표현식을 키워나간다. 평가자는 결정론적이다.

실행:

```
python3 code/main.py
```

트레이스는 (중간 계획에서 LLM 폴백을 동반하며) 복합 작업을 분해하는 HTN 플래너와, 목표 표현식으로 수렴하는 진화적 루프를 보여준다.

## 라이브러리로 써보기 (Use It)

- **HTN 플래너**: `pyhop`, `SHOP3`, 또는 도메인 특화 정책 집행을 위해 직접 만든 것.
- **ChatHTN**: 연구용 코드. 그 패턴(기호적 + LLM 폴백)은 임의의 HTN 플래너로 깔끔하게 포팅된다.
- **AlphaEvolve**: DeepMind 논문. 그 패턴(앙상블 + 평가자)은 재현 가능하다. OpenEvolve와 유사한 오픈소스 포크(fork)들이 등장하고 있다.
- **에이전트 프레임워크**: 아직 일급(first-class) HTN이나 AlphaEvolve를 제공하는 것은 없다. 서브에이전트나 백그라운드 워커로 직접 만들어라.

## 산출물 (Ship It)

`outputs/skill-hybrid-planner.md`는 LLM의 역할을 명시적으로 한정한 하이브리드 플래너 스캐폴드(HTN 또는 진화적)를 생성한다.

## 연습 문제 (Exercises)

1. HTN 플래너를 역추적(backtracking)으로 확장하라. 연산자의 사후 조건(postcondition)이 런타임에 실패하면, 롤백하고 다음 방법을 시도하라.
2. ChatHTN에 LLM 방법 캐시를 추가하라. LLM이 상태 패턴 `P`에서 작업 `T`를 분해하면 그 결과를 저장하라. 다음 호출 시 방법 라이브러리를 먼저 재확인하라.
3. 진화적 탐색의 평가자를 진짜 테스트 스위트(test suite)로 교체하라. 20개의 테스트 케이스를 통과하는 정렬 함수를 진화시키고, 수렴까지의 세대(generation) 수를 보고하라.
4. AlphaEvolve의 평가자 설계 노트를 읽어라. 관심 있는 도메인(SQL 질의 최적화, 테스트 스위트 최소화, 배포 YAML)을 위한 평가자를 설계하라.
5. 결합하라: HTN으로 복합 작업을 하위 작업으로 분해한 뒤, 각 하위 작업의 원시 연산자에 진화적 탐색을 사용하라. 어디서 빛나고, 어디서 과잉 설계되는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| HTN | "계층적 플래너" | 연산자, 전제 조건, 효과를 갖춘 작업 분해 |
| 방법(Method) | "분해 규칙" | 복합 작업을 하위 작업으로 쪼개는 방식 |
| 연산자(Operator) | "원시 행동" | 전제 조건과 효과를 가진 구체적 단계 |
| ChatHTN | "LLM + HTN" | 기호적 플래너가 일치하는 방법이 없을 때 LLM에게 물음 |
| AlphaEvolve | "진화적 코드 탐색" | 앙상블 LLM이 코드를 변이하고, 결정론적 평가자가 선택함 |
| 적합도 함수(Fitness function) | "평가자" | 출력에 대한 결정론적이고 기계로 검증 가능한 점수 |
| 온라인 방법 학습(Online method learning) | "캐시된 LLM 분해" | LLM 계획을 저장 + 일반화하여 질의 비용을 줄임 |

## 더 읽을거리 (Further Reading)

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814): 기호적 + LLM 하이브리드 플래너
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131): LLM 변이를 활용한 진화적 코드 탐색
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 단순 루프 대신 플래너에 손을 뻗을 때
