# 역할 전문화(플래너, 비평가, 실행자, 검증자 (Role Specialization) Planner, Critic, Executor, Verifier)

> 2026년 가장 흔한 멀티 에이전트(multi-agent) 분해: 한 에이전트(agent)가 계획하고, 하나가 실행하고, 하나가 비평하거나 검증한다. MetaGPT(arXiv:2308.00352)는 이를 역할 프롬프트에 인코딩된 SOP로 형식화한다. Product Manager, Architect, Project Manager, Engineer, QA Engineer: `Code = SOP(Team)`을 따른다. ChatDev(arXiv:2307.07924)는 디자이너, 프로그래머, 리뷰어, 테스터를 "communicative dehallucination"(에이전트가 누락된 세부사항을 명시적으로 요청)이 있는 "채팅 사슬(chat chain)"을 통해 연결한다. 검증자(verifier)는 하중을 지탱한다(load-bearing): Cemri et al.(MAST, arXiv:2503.13657)은 모든 멀티 에이전트 실패가 누락되거나 깨진 검증으로 추적될 수 있음을 보인다. PwC는 CrewAI의 구조화된 검증 루프로부터 7배 정확도 이득(10% → 70%)을 보고했다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60분

## 문제 (Problem)

범용 멀티 에이전트 시스템은 범용 출력을 낳는다. 그룹 채팅에 있는 세 명의 코더는 같은 평범한 코드의 세 가지 변형을 작성한다. 더 많은 에이전트를 추가하고, 더 많은 라운드를 추가해도, 여전히 품질 문턱을 넘지 못한다.

해법은 더 많은 에이전트가 아니라. *다른* 에이전트다. 구별되는 역할을 할당하라. 비평가(critic)에게 플래너(planner)가 갖지 않은 도구를 주어라. 검증자에게 객관적 테스트 스위트를 주어라. 이제 시스템은 단순한 병렬 추측이 아니라, 근거에 기반한 교정을 갖춘 내부 불일치 구조가 된다.

## 개념 (Concept)

### 네 개의 정전적(canonical) 역할

**플래너(Planner).** 목표를 읽고, 단계 목록이나 명세(spec)를 생성한다. 도구: 지식 검색, 문서. 출력: 구조화된 계획.

**실행자(Executor).** 한 번에 한 계획 단계를 읽고, 아티팩트(artifact)를 생성한다. 도구: 실제 작업 도구(코드 컴파일러, 셸, API 클라이언트). 출력: 아티팩트.

**비평가(Critic).** 플래너의 의도에 비추어 실행자의 출력을 읽는다. 도구: 아티팩트에 대한 읽기 전용 접근, 정적 분석. 출력: 이유와 함께 수락/거절.

**검증자(Verifier).** 아티팩트를 읽고 결정적 검사를 실행한다. 도구: 테스트 러너, 타입 체커, 스키마 검증기. 출력: 증거와 함께 통과/실패.

비평가는 주관적이고, 의견이 강하고, 흔히 LLM 기반이다. 검증자는 객관적이고, 결정적이고, 흔히 코드 기반이다. 둘은 같은 역할이 아니다.

### MetaGPT의 SOP 패턴

MetaGPT(arXiv:2308.00352)는 소프트웨어 엔지니어링 SOP를 역할 프롬프트로 인코딩한다.

- **Product Manager**가 PRD를 작성한다.
- **Architect**가 시스템 설계를 생성한다.
- **Project Manager**가 작업을 나눈다.
- **Engineer**가 구현한다.
- **QA Engineer**가 테스트를 실행한다.

각 역할은 엄격한 입력/출력 스키마를 따른다. 역할 프롬프트는 역할이 *무엇인지*와 *무엇을 생산해야 하는지*를 말한다. `Code = SOP(Team)` 정식화: 결정적 SOP가 LLM 팀을 예측 가능한 파이프라인으로 바꾼다.

### ChatDev의 communicative dehallucination

ChatDev는 여기에 핵심 장치를 하나 더한다. 실행자가 계획에 없던 특정 세부사항이 필요하면, 계속하기 전에 디자이너에게 명시적으로 묻는다. 이는 세부사항을 그럴듯하게 지어내는 고전적 LLM 실패를 막는다.

구현: 역할 프롬프트는 "당신이 받지 못한 특정 정보가 필요할 때, 출력을 생산하기 전에 해당 역할을 이름으로 불러 물어라"를 포함한다.

### 왜 검증자가 가장 중요한가

Cemri et al.(MAST)은 1642개의 멀티 에이전트 실행 실패를 추적했다. 21.3%는 검증 격차였다. 시스템이 아무도 확인하지 않은 답을 출하했다. 나머지 79%는 종종 "조용히 실패했거나 결코 실행되지 않은 검사가 있었다"로 추적된다. 검증은 하중을 지탱하는 역할이다.

PwC는 (CrewAI 배포, 2025) 구조화된 검증 루프를 추가하면 정확도가 10%에서 70%로 이동했다고 보고했다. 한 역할로부터 7배 이득.

### 비평가 대 검증자

- 비평가는 아티팩트의 품질을 검토하는 LLM이다. 주관적. 그럴듯한 산문에 속을 수 있다.
- 검증자는 아티팩트에서 실행되는 결정적 프로그램이다. 객관적. 증거와 함께 통과/실패를 준다.

둘 다 사용하라. 비평가는 검증자가 명료화할 수 없는 취향 문제를 잡는다. 검증자는 비평가가 볼 수 없는 버그를 잡는데, 그것이 오직 런타임에만 나타나기 때문이다.

### 안티패턴 (The anti-pattern)

시스템의 모든 역할이 LLM이고 모든 역할의 출력이 "내가 보기엔 좋아 보인다"이다. 고전적 MAST 실패 양식. 통과/실패가 LLM이 아니라 코드로 결정되는 검증자를 최소 하나 추가하라.

### 프레임워크 매핑

- **CrewAI**: `Agent(role, goal, backstory)`가 교과서적 전문화 표면이다.
- **LangGraph**: 노드가 전문화된 프롬프트를 가질 수 있다; 간선이 파이프라인을 강제한다.
- **AutoGen**: GroupChat 안의 한 단어 이름을 가진 역할별 ConversableAgent.
- **OpenAI Agents SDK**: 역할 전문화된 Agent 간의 핸드오프(handoff) 도구.

```figure
swarm-roles
```

## 직접 만들기 (Build It)

`code/main.py`는 간단한 Python 함수를 만드는 4-역할 파이프라인을 구현한다.

- **Planner**가 명세를 생성한다.
- **Executor**가 코드 문자열을 생성한다.
- **Critic**(LLM 시뮬레이션)가 명백한 문제를 표시한다.
- **Verifier**가 생성된 코드를 샌드박스(`exec`)에서 테스트 케이스에 대해 실행한다.

데모는 두 번 실행된다: 한 번은 실행자가 올바른 코드를 생성하는 경우(비평가 + 검증자 모두 통과), 한 번은 실행자가 명세에서 벗어난 코드를 생성하는 경우(비평가는 그럴듯해 보여서 버그를 놓치고, 검증자는 테스트가 실패하므로 그것을 잡는다).

실행:

```
python3 code/main.py
```

## 라이브러리로 써보기 (Use It)

`outputs/skill-role-designer.md`는 작업을 받아 역할 명단(3-5개 역할), 역할당 입력/출력 스키마, 그리고 검증자 검사를 생성한다. 에이전트를 프레임워크에 엮기 전에 이를 사용하라.

## 산출물 (Ship It)

체크리스트:

- **최소 하나의 결정적 검증자.** 결코 전부-LLM이 아니게.
- **역할당 명시적 I/O 스키마.** 플래너는 산문이 아니라 명세를 반환한다; 실행자는 그 스키마를 읽는다.
- **Communicative dehallucination.** 실행자는 정보가 누락되면 플래너에게 물어야 한다; 결코 지어내지 않는다.
- **비평가/검증자 순서.** 비평가를 먼저 실행하고(저렴, 설계 문제를 잡음), 검증자를 두 번째로(느림, 버그를 잡음).
- **루프 예산.** 인간에게 에스컬레이션하기 전 최대 2번의 비평가-실행자 수정 라운드.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하고 검증자가 비평가가 놓친 버그를 어떻게 잡는지 관찰하라. 정적 분석 검사(`return` 출현 횟수 세기)를 추가 검증자로 더하라. 그것은 런타임 테스트가 놓치는 무엇을 잡는가?
2. 5번째 역할을 추가하라: 사용자 소망을 플래너가 쓸 수 있는 명세로 번역하는 "requirements analyst". 어떤 communicative dehallucination 요청이 그것으로 흘러 올라가야 하는가?
3. MetaGPT 섹션 3("Agents")을 읽어라. MetaGPT의 5개 역할 각각의 입력/출력 스키마를 나열하라.
4. ChatDev의 채팅 사슬 다이어그램(arXiv:2307.07924 Figure 3)을 읽어라. communicative dehallucination이 그렇지 않았다면 무한했을 루프를 어디서 깨는지 식별하라.
5. PwC의 7배 정확도 이득은 검증 루프에서 왔다. 검증자를 추가해도 도움이 되지 않을(정확성의 결정적 검사가 불가능하거나 금지될 만큼 비싼) 세 가지 작업을 가설로 세워라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 역할 전문화 (Role specialization) | "다른 에이전트, 다른 일" | 플래너/실행자/비평가/검증자 역할에 맞게 조정된 구별되는 시스템 프롬프트. |
| SOP 패턴 (SOP pattern) | "인코딩된 표준 운영 절차" | MetaGPT의 틀: 역할당 엄격한 I/O 스키마가 팀을 파이프라인으로 바꾼다. |
| Communicative dehallucination | "지어내기 전에 물어라" | ChatDev 패턴: 실행자가 세부사항이 누락되면 지어내지 않고 플래너에게 묻는다. |
| 비평가 (Critic) | "LLM 리뷰어" | 주관적이고 의견이 강한 리뷰어. 취향 문제를 잡는다. 그럴듯한 산문에 속을 수 있다. |
| 검증자 (Verifier) | "결정적 검사" | 코드 기반 통과/실패. 테스트 러너, 타입 체커, 스키마 검증기. 속을 수 없다. |
| 검증 격차 (Verification gap) | "아무도 확인 안 했다" | MAST 실패의 21.3%. 버그를 잡았을 검사 없이 출하된 답. |
| 수정 루프 (Revision loop) | "비평가가 돌려보낸다" | 비평가 거절이 피드백과 함께 실행자 재실행을 촉발한다. 예산이 필요하다. |
| 전부-LLM 안티패턴 (All-LLM anti-pattern) | "내가 보기엔 좋아 보인다" | 모든 역할이 LLM이고, 결정적 검사가 없음. 고전적 MAST 실패. |

## 더 읽을거리 (Further Reading)

- [Hong et al.: MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) SOP-as-역할-프롬프트 레퍼런스 논문
- [Qian et al.: Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) 채팅 사슬 + communicative dehallucination
- [Cemri et al.: Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) MAST 분류 체계; 검증 격차가 실패의 21.3%
- [CrewAI docs: Agent roles](https://docs.crewai.com/en/introduction) 프로덕션 역할 명세 표면
