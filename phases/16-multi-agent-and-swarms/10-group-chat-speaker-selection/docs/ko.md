# 그룹 챗과 발화자 선택 (Group Chat and Speaker Selection)

> AutoGen GroupChat와 AG2 GroupChat는 N개의 에이전트(agent)가 하나의 대화를 공유한다. 선택자(selector) 함수(LLM, 라운드 로빈, 또는 커스텀)가 다음에 누가 말할지 고른다. 이것은 창발적(emergent) 멀티 에이전트 대화의 원형이다. 에이전트는 정적 그래프 안에서 자신의 역할을 알지 못하며, 그저 공유 풀(shared pool)에 반응할 뿐이다. AutoGen v0.2의 GroupChat 시맨틱은 AG2 포크에서 보존되었고, AutoGen v0.4는 이를 이벤트 기반 액터 모델(event-driven actor model)로 다시 작성했다. 마이크로소프트는 2026년 2월 AutoGen을 유지보수 모드로 전환하고 이를 Semantic Kernel과 병합해 Microsoft Agent Framework(2026년 2월 RC)로 통합했다. GroupChat 프리미티브는 AG2와 Microsoft Agent Framework 양쪽에 모두 살아남았다. 한 번 익히면 어디서든 쓸 수 있다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60분

## 문제 (Problem)

정적 그래프(LangGraph)는 워크플로가 알려져 있을 때 훌륭하다. 실제 대화는 정적이지 않다. 어떤 때는 코더가 리뷰어에게 묻고, 어떤 때는 연구자에게, 어떤 때는 작성자에게 묻는다. 가능한 모든 핸드오프(handoff)를 하드코딩하면 간선 폭발(edge explosion)이 일어난다. 필요한 것은 *공유 풀에 반응하는 에이전트들*과, 다음에 누가 말할지 결정하는 함수다.

그것이 바로 AutoGen GroupChat가 하는 일이다.

## 개념 (Concept)

### 형태

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

모든 에이전트는 모든 메시지를 본다. 매 턴(turn)마다 선택자 함수가 호출되어 다음에 누가 말할지 고른다.

### 세 가지 선택자 방식

**라운드 로빈(Round-robin).** 고정된 순환이다. 결정론적이다. N에 대해 선형으로 확장되지만 맥락(context)을 무시한다. 주제가 법률 검토인데도 코더에게 차례가 돌아간다.

**LLM 선택(LLM-selected).** 최근 풀을 읽고 다음으로 가장 적합한 발화자를 반환하는 LLM 호출이다. 맥락을 인식하지만 느리다. 매 턴마다 LLM 호출이 추가된다. AutoGen의 기본값이다.

**커스텀(Custom).** 원하는 어떤 로직이든 담을 수 있는 파이썬 함수다. 전형적인 형태는 폴백 규칙(fallback rule)이 있는 LLM 선택이다(예: "코더 다음에는 항상 검증자에게 차례를 준다").

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager`는 선택자를 보유한다. 한 에이전트가 턴을 완료하면 매니저가 선택자를 호출하고, 선택자는 다음 에이전트를 반환한다. 종료 조건에 도달할 때까지 루프가 계속된다.

### 종료 (Termination)

흔한 세 가지 패턴이 있다.

- **최대 라운드(Max rounds).** 전체 턴 수에 대한 하드 캡이다.
- **"TERMINATE" 토큰.** 에이전트는 센티넬(sentinel) 메시지를 내보낼 수 있다. 그런 메시지가 하나라도 나타나면 매니저가 멈춘다.
- **목표 도달 검사(Goal-reached check).** 가벼운 검증자가 매 턴 실행되어 작업이 끝나면 챗을 멈춘다.

### AutoGen → AG2 분기와 Microsoft Agent Framework 병합

2025년 초, 마이크로소프트는 이벤트 기반 액터 모델을 중심으로 AutoGen(v0.4)을 대대적으로 재작성하기 시작했다. 커뮤니티는 AutoGen v0.2의 GroupChat 시맨틱을 AG2로 포크하여, 초기 도입자들이 통합했던 API를 보존했다.

2026년 2월, 마이크로소프트는 AutoGen을 유지보수 모드로 전환하고 이벤트 기반 액터 모델을 **Microsoft Agent Framework**(2026년 2월 RC, 현재 Semantic Kernel과 병합됨)에 통합한다고 발표했다. GroupChat 개념은 두 갈래 모두에 살아남았으나 구현 세부사항은 다르다. AG2는 v0.2 호환 코드를 위한 선호 업스트림이다.

### GroupChat가 잘 맞을 때

- **창발적 대화.** 가능한 모든 다음 발화자를 미리 배선하고 싶지 않을 때.
- **역할 혼합 작업.** 코더가 연구자에게 묻고, 연구자가 아카이브 담당에게 묻고, 아카이브 담당이 다시 코더에게 묻는다. 흐름이 DAG가 아니다.
- **탐색적 문제 해결.** "조립 라인"이 아니라 "브레인스토밍 회의"를 생각하라.

### GroupChat가 실패할 때

- **엄격한 결정론.** LLM 선택자는 일관성이 없을 수 있다. 같은 프롬프트(prompt)라도 실행마다 다른 다음 발화자가 나온다.
- **아첨 연쇄(Sycophancy cascade).** 에이전트가 가장 자신 있게 말한 쪽에 동조한다. 이를 명시적으로 반박하는 카운터 프롬프트를 써라.
- **컨텍스트 비대화(Context bloat).** 모든 에이전트가 모든 메시지를 읽는다. 10턴이 지나면 컨텍스트가 거대해진다. 뷰(view)의 범위를 좁히려면 프로젝션(projection)(Lesson 15)을 사용하라.
- **핫 발화자(Hot speakers).** 선택자가 한 에이전트의 전문 분야를 선호하는 탓에 그 에이전트가 대화를 지배한다. 선택자 기능에 발화자 균형(speaker balance)을 도입하라.

### 그룹 챗 vs 슈퍼바이저

같은 프리미티브, 다른 기본값이다.

- 슈퍼바이저(Supervisor): 한 에이전트가 계획하고 나머지가 실행한다. 선택자는 "플래너에게 무엇을 할지 묻기"이다.
- 그룹 챗(Group chat): 모든 에이전트가 동등하다. 선택자는 공유 풀에 대한 함수다.

둘 다 Lesson 04의 네 가지 프리미티브를 사용한다. 그룹 챗은 기본적으로 LLM 선택 오케스트레이션(orchestration)과 전체 풀 공유 상태를 사용한다.

```figure
swarm-speaker
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib만으로 GroupChat를 밑바닥부터 구현한다. 세 에이전트(코더, 리뷰어, 매니저), 라운드 로빈과 LLM 선택 변형, 그리고 `TERMINATE` 토큰에 대한 종료를 담는다.

데모는 대화 트랜스크립트(transcript)와 두 변형 각각의 선택자 결정 추적(decision trace)을 출력한다.

실행:

```
python3 code/main.py
```

## 라이브러리로 써보기 (Use It)

`outputs/skill-groupchat-selector.md`는 주어진 작업에 대한 GroupChat 선택자를 구성한다. 라운드 로빈 vs LLM 선택 vs 커스텀, 그리고 어떤 선택자 입력(최근 메시지, 에이전트 전문 분야, 턴 횟수)을 사용할지를 다룬다.

## 산출물 (Ship It)

체크리스트:

- **최대 라운드 캡.** 항상 둔다. 전형적인 작업에는 10~20.
- **발화자 균형 지표.** 에이전트별 턴 수를 추적하고, 불균형이 임계값을 넘으면 알린다.
- **종료 토큰.** `TERMINATE` 또는 전용 검증자 에이전트.
- **프로젝션 또는 범위 제한 메모리.** 메시지가 ~10개를 넘으면, 컨텍스트 비대화를 막기 위해 각 에이전트에게 범위가 좁혀진 뷰만 주는 것을 고려하라.
- **선택자 로깅.** LLM 선택 변형의 경우, 선택자의 입력과 그 선택을 모두 기록하라. 그렇지 않으면 디버깅이 불가능하다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 라운드 로빈 vs LLM 선택에서 대화를 비교하라. 각각에서 어느 에이전트가 지배하는가?
2. 선택자에 "에이전트당 최대 발화 횟수" 규칙을 추가하라. 트랜스크립트에 어떤 영향을 주는가?
3. 목표 도달 종료를 구현하라. 리뷰어가 "approved"를 반환하면 멈춘다. 라운드 캡에 도달하기 전에 얼마나 자주 작동하는가?
4. GroupChat에 관한 AutoGen 안정 문서를 읽어라(https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). `GroupChatManager`가 사용하는 기본 선택자를 식별하라.
5. AG2 저장소(https://github.com/ag2ai/ag2)를 읽고, 그것의 v0.2 GroupChat를 v0.4 이벤트 기반 버전과 비교하라. v0.4가 추가하는 구체적인 속성(처리량(throughput), 결함 허용성(fault-tolerance), 조합성(composability))은 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| GroupChat | "한 채팅방 안의 에이전트들" | 공유 메시지 풀 + 선택자 함수. AutoGen / AG2 프리미티브. |
| 발화자 선택(Speaker selection) | "다음에 누가 말하나" | 다음 에이전트를 고르는 함수. 라운드 로빈, LLM 선택, 또는 커스텀. |
| GroupChatManager | "회의 진행자" | 선택자를 소유하고 턴을 순회하는 AutoGen 컴포넌트. |
| ConversableAgent | "기본 에이전트" | AutoGen 기반 클래스. 메시지를 보내고 받을 수 있는 에이전트. |
| 종료 토큰(Termination token) | "'멈춤' 단어" | 챗을 끝내는 센티넬 문자열(보통 `TERMINATE`). |
| 핫 발화자(Hot speaker) | "한 에이전트가 지배함" | 선택자가 같은 에이전트를 계속 고르는 실패 모드. |
| 컨텍스트 비대화(Context bloat) | "풀이 무한정 커짐" | 각 에이전트가 이전 메시지를 모두 읽어 컨텍스트가 턴과 함께 커진다. |
| 프로젝션(Projection) | "범위 제한 뷰" | 컨텍스트 비대화를 막기 위한, 공유 풀에 대한 역할별 뷰. |

## 더 읽을거리 (Further Reading)

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html): 레퍼런스 구현
- [AG2 repo](https://github.com/ag2ai/ag2): 커뮤니티 AutoGen v0.2 연속판
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/): 병합된 후속작, 2026년 2월 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/): 이벤트 기반 액터 모델 재작성 세부사항
