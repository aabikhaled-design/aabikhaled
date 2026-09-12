# Chatbots — Rule-Based to Neural to LLM Agents

> ELIZA는 패턴 매칭으로 답했다. DialogFlow는 의도(intent)를 매핑했다. GPT는 가중치(weight)로부터 답했다. Claude는 도구를 실행하고 검증한다. 각 시대는 이전 시대의 최악의 실패를 해결했다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75분

## 문제 (The Problem)

사용자가 "I want to change my flight"라고 말한다. 시스템은 사용자가 무엇을 원하는지, 어떤 정보가 빠졌는지, 그 정보를 어떻게 얻을지, 그 행동을 어떻게 완수할지 알아내야 한다. 그러면 사용자가 "wait, what if I cancel instead?"라고 말하고, 시스템은 맥락을 기억하고 작업을 전환하며 상태(state)를 보존해야 한다.

대화는 ML 시스템에게 어렵다. 입력은 개방형이다. 출력은 여러 턴에 걸쳐 일관성이 있어야 한다. 시스템이 세상에 직접 작용해야 할 때도 있다(항공편 변경, 카드 결제). 잘못된 단계는 하나도 빠짐없이 사용자에게 보인다.

챗봇(chatbot) 아키텍처는 네 개의 패러다임을 거쳐 순환해왔으며, 각각은 이전 것이 너무 눈에 띄게 실패해서 도입되었다. 이 레슨은 그것들을 순서대로 훑는다. 2026년 프로덕션(production) 지형은 마지막 둘의 하이브리드(hybrid)다.

## 개념 (The Concept)

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### 대본이 지배한 반세기, 1950-2001

첫 패러다임은 5년이 아니라 50년을 갔다. 그 궤적을 알아야 하는 이유는, 그 안의 모든 시스템이 같은 기계이기 때문이다. 입력을 맞춰 보고, 정해진 응답을 내보내고, 상태를 조금 갱신한다. 그리고 그 기계에 50년 동안 규칙을 더해도 일반적인 경우는 끝내 나오지 않았다. 두 번째부터 네 번째 패러다임이 존재하는 이유가 바로 그 천장이다.

**1950.** 튜링은 "기계가 생각할 수 있는가"를 비켜 가며 실행 가능한 질문으로 바꿔 놓는다. 심문자가 텔레타이프 너머의 기계와 사람을 구별할 수 없다면 그 철학적 물음은 무의미해진다. 분야에 이름이 붙기도 전에 대화가 그 분야의 기준이 된 것이다.

**1956.** 이름이 도착한다. 다트머스의 여름 워크숍이 "인공지능"이라는 말을 만들면서, 지능의 모든 특징을 "원리상 기계가 흉내 낼 수 있을 만큼 정확하게 기술할 수 있다"는 추측을 내건다. 제안서는 상당한 진전에 두 달을 배정했다.

**1966.** ELIZA가 1단계에서 여러분이 만들 되비추기 기법을 내놓는다. 분해 규칙이 입력에서 조각을 뽑아내고, 재조립 규칙이 그것을 질문으로 되돌려 준다. 패턴은 모두 합쳐 200개쯤, 상태도 이해도 전혀 없었는데도 사용자들은 거기에 속내를 털어놓았다. 바이첸바움은 그토록 적은 장치로 그런 일이 벌어진다는 사실에 남은 평생을 두려워하며 보냈다.

**1972.** 편집증을 모형화하려고 스탠퍼드에서 만든 PARRY는 ELIZA에 없던 조각을 더한다. 내부 상태다. 두려움, 분노, 불신을 나타내는 수치 변수가 턴마다 갱신되며 다음에 어떤 대본이 나갈지를 정하므로, 같은 입력도 그때까지의 대화에 따라 다른 응답을 낳는다. 눈가림 기록 시험에서 정신과 의사들은 PARRY와 사람 환자를 우연 수준으로밖에 구별하지 못했다. 이것이 페르소나 조건화의 직계 조상이다. 부동소수점 세 개로 구현한 시스템 프롬프트인 셈이다. 같은 해에 두 봇은 ARPANET 위에서 서로를 향하게 됐다. 치료사 대본이 편집증 상태 기계를 면담한, 네트워크 위 최초의 봇 대 봇 대화였다.

**1995.** ALICE는 패턴과 템플릿 쌍을 위한 XML 방언인 AIML로 ELIZA의 방식을 키운다. 손으로 쓴 범주가 대략 4만 개, 뢰브너상 3회 수상. 이것이 규칙 기반 시스템의 규모 법칙을 증명했다. 규칙을 더하면 적용 범위를 살 수 있지만 일반성은 결코 살 수 없다. 규칙 하나하나가 누군가 유지해야 하는 부채다.

**2001.** SmarterChild는 그 방식을 3천만 명의 메신저 사용자 앞에 올려놓고, 날씨와 주가와 영화 시간표 같은 백엔드 조회를 템플릿에 끼워 넣는다. 눈을 가늘게 뜨고 보면 2001년 옷을 입은 도구 호출이다. 의도를 파싱하고, 서비스를 부르고, 그 결과를 답변으로 그려 낸다.

50년, 하나의 기계 장치, 늘어나는 규칙 수. 이 패러다임이 끝난 것은 누가 그것을 반증해서가 아니다. 손으로 쓴 상태 기계의 유지 비용은 적용 범위에 비례해 늘어나는데, 사용자의 기대는 그들이 지난주에 본 무언가를 따라 늘어나기 때문이다.

```figure
chatbot-lineage
```

**규칙 기반(Rule-based, ELIZA, AIML, DialogFlow).** 손으로 작성한 패턴이 사용자 입력과 매칭되어 응답을 만든다. 의도 분류기(intent classifier)가 미리 정의된 흐름으로 라우팅(routing)한다. 슬롯 채우기(slot-filling) 상태 기계가 필요한 정보를 수집한다. 설계된 좁은 범위 안에서는 훌륭하게 작동한다. 그 밖에서는 즉시 실패한다. 환각(hallucination)이 용납되지 않는 안전이 중요한 도메인(은행 인증, 항공 예약)에서는 여전히 배포된다.

**검색 기반(Retrieval-based).** FAQ 스타일 시스템이다. (발화, 응답)의 모든 쌍을 인코딩한다. 런타임에 사용자 메시지를 인코딩하고 가장 가까운 저장된 응답을 검색한다. Zendesk의 고전적인 "유사 문서" 기능을 생각하라. 규칙보다 패러프레이즈(paraphrase)를 더 잘 다룬다. 생성이 없으므로 환각이 없다.

**신경망(Neural, seq2seq).** 대화 로그로 학습된 인코더-디코더(encoder-decoder)다. 응답을 밑바닥부터 생성한다. 유창하지만 일반적인 출력("I don't know")과 사실 표류에 취약하다. 절대 안정적으로 주제에 머물지 않는다. Google, Facebook, Microsoft 모두 2016-2019년에 실망스러운 챗봇을 내놓은 이유다.

**LLM 에이전트(LLM agents).** 계획하고 도구를 호출하고 결과를 검증하는 루프(loop)로 감싼 언어 모델이다. 긴 프롬프트(prompt)를 가진 챗봇이 아니다. 에이전트(agent) 루프다. 계획 → 도구 호출 → 결과 관찰 → 다음 단계 결정. 검색 우선 근거화(retrieval-first grounding, RAG)가 환각을 막는다. 도구 호출이 실제로 일을 하게 한다. 이것이 2026년 아키텍처다.

네 패러다임은 순차적 대체가 아니다. 2026년 프로덕션 챗봇은 네 가지 모두로 라우팅한다. 인증과 파괴적 행동에는 규칙 기반, FAQ에는 검색, 자연스러운 표현에는 신경망 생성, 모호한 개방형 질의에는 LLM 에이전트다.

## 직접 만들기 (Build It)

### Step 1: 규칙 기반 패턴 매칭

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20줄짜리 ELIZA다. 반영 트릭("I feel sad" → "Why do you feel sad")은 Weizenbaum 1966에서 나온 표준 심리치료사 데모로, 여전히 교훈적이다.

### Step 2: 검색 기반 (FAQ)

이 예시 스니펫은 `pip install sentence-transformers`(torch를 끌어온다)를 필요로 한다. 이 레슨의 실행 가능한 `code/main.py`는 대신 표준 라이브러리(stdlib) Jaccard 유사도(similarity)를 쓰므로, 레슨은 외부 의존성 없이 실행된다.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

임계값 기반 거부가 핵심 설계 선택이다. 최선의 일치가 충분히 가깝지 않으면 `None`을 반환하고 시스템이 에스컬레이션(escalate)하게 둔다.

### Step 3: 신경망 생성 (베이스라인)

작은 인스트럭션 튜닝(instruction-tuned) 인코더-디코더(FLAN-T5)나 파인튜닝(fine-tuning)된 대화 모델을 쓰라. 2026년에 그것만으로는 프로덕션에 쓸 수 없지만(모순, 주제 이탈 표류, 사실적 헛소리), 자연스러운 표현을 위해 하이브리드 시스템 안에 배포된다. DialoGPT 스타일의 디코더 전용(decoder-only) 모델은 일관된 답을 만들려면 명시적인 턴 구분자와 EOS 처리가 필요하다. FLAN-T5 text2text 파이프라인(pipeline)은 교육용 예제로는 바로 작동한다.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Step 4: LLM 에이전트 루프

2026년 프로덕션 형태:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

명명할 세 가지. 도구(tool)는 LLM이 호출할 수 있는 호출 가능한 함수다. 루프는 LLM이 도구 호출 대신 최종 답을 반환할 때 종료된다. 단계 예산(step budget)은 모호한 작업에서 무한 루프를 막는다.

실제 프로덕션은 여기에 다음을 더한다. 검색 우선 근거화(각 LLM 호출 전에 관련 문서 주입), 가드레일(guardrail)(확인 없이 파괴적 행동 거부), 관찰 가능성(observability)(모든 단계 로깅), 평가(evaluation)(에이전트 동작이 명세대로 유지되는지 자동 검사).

### Step 5: 하이브리드 라우팅

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

패턴은 이렇다. 파괴적인 모든 것에는 결정론적 규칙, 정해진 FAQ에는 검색, 나머지 모든 것에는 LLM 에이전트. 이것이 2026년 고객 지원 시스템에 배포되는 형태다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 사용 사례 | 아키텍처 |
|---------|---------------|
| 예약, 결제, 인증 | 규칙 기반 상태 기계 + 슬롯 채우기 |
| 고객 지원 FAQ | 큐레이션된 답에 대한 검색 |
| 개방형 도움말 채팅 | RAG + 도구 호출을 갖춘 LLM 에이전트 |
| 내부 도구 / IDE 어시스턴트 | 도구 호출(검색, 읽기, 쓰기)을 갖춘 LLM 에이전트 |
| 컴패니언 / 캐릭터 챗봇 | 페르소나 시스템 프롬프트를 가진 튜닝된 LLM, 지식에 대한 검색 |

프로덕션에서는 항상 하이브리드 라우팅을 쓰라. 단일 아키텍처는 모든 요청을 잘 다루지 못한다. 라우팅 계층 자체는 보통 작은 의도 분류기다.

## 여전히 배포되는 실패 양상

- **자신만만한 날조(Confident fabrication).** LLM 에이전트가 하지 않은 행동을 완료했다고 주장한다. 완화책: 결과를 검증하고, 도구 호출을 로깅하며, 성공적인 도구 반환 없이 LLM이 무언가를 했다고 주장하게 절대 두지 마라.
- **프롬프트 인젝션(Prompt injection).** 사용자가 시스템 프롬프트를 무시하게 하는 텍스트를 삽입한다. OWASP Top 10 for LLM Applications 2025에서 LLM01로 순위가 매겨졌다. 두 가지 종류로, 직접 인젝션(채팅에 붙여넣음)과 간접 인젝션(에이전트가 읽는 문서, 이메일, 도구 출력에 숨겨짐)이다.

  공격률은 시나리오에 따라 다르다. 측정된 성공률은 일반 도구 사용 및 코딩 벤치마크(benchmark)에서 프런티어 모델 전반에 걸쳐 약 0.5-8.5% 범위다. 특정 고위험 설정(AI 코딩 에이전트에 대한 적응형 공격, 취약한 오케스트레이션)은 약 84%에 도달했다. 프로덕션 CVE에는 EchoLeak(CVE-2025-32711, CVSS 9.3)이 포함된다. 공격자가 제어하는 이메일로 트리거되는 Microsoft 365 Copilot의 제로 클릭 데이터 탈취(zero-click data-exfiltration) 결함이다.

  완화책: 루프 전반에서 사용자 입력을 신뢰할 수 없는 것으로 취급하고, 도구 호출 전에 정화(sanitize)하며, 도구 출력을 주 프롬프트로부터 격리하고, 에이전트가 먼저 계획한 다음 실행 전에 각 행동을 그 계획에 대해 검증하는 Plan-Verify-Execute(PVE) 패턴을 쓰라(이렇게 하면 도구 결과가 계획되지 않은 새 행동을 주입하는 것을 막는다). 파괴적 행동에는 사용자 확인을 요구하고, 도구 범위에 최소 권한(least-privilege)을 적용하라.

  어떤 양의 프롬프트 엔지니어링도 이 위험을 완전히 제거하지 못한다. 외부 런타임 방어 계층(LLM Guard, 허용 목록(allowlist) 검증, 의미적 이상 탐지)이 필요하다.
- **범위 확장(Scope creep).** 도구 호출이 주변적으로 관련된 정보를 반환한 탓에 에이전트가 작업을 벗어난다. 완화책: 좁은 도구 계약(tool contract), 시스템 프롬프트를 집중되게 유지, 작업 이탈률에 대한 평가 추가.
- **무한 루프(Infinite loops).** 에이전트가 같은 도구를 계속 호출한다. 완화책: 단계 예산, 도구 호출 중복 제거, "진전을 이루고 있는가"에 대한 LLM 판정.
- **컨텍스트 윈도우 소진(Context window exhaustion).** 긴 대화가 가장 이른 턴을 컨텍스트(context) 밖으로 밀어낸다. 완화책: 오래된 턴을 요약하거나, 유사도로 관련된 과거 턴을 검색하거나, 긴 컨텍스트 모델을 쓰라.

## 산출물 (Ship It)

`outputs/skill-chatbot-architect.md`로 저장하라:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 연습 문제 (Exercises)

1. **Easy.** 위의 규칙 기반 응답을 커피숍 주문 봇을 위한 10개 패턴으로 구현하라. 엣지 케이스를 테스트하라: 중복 주문, 수정, 취소, 불명확한 의도.
2. **Medium.** 하이브리드 FAQ + LLM 폴백(fallback)을 만들어라. SaaS 제품을 위한 50개의 정해진 FAQ 항목, 문서 사이트에 대한 검색을 갖춘 LLM 폴백. 100개의 실제 지원 질문에서 거부율과 정확도를 측정하라.
3. **Hard.** 위의 에이전트 루프를 세 개의 도구(검색, 사용자 데이터 읽기, 이메일 전송)로 구현하라. 프롬프트 인젝션 시도를 포함한 50개 테스트 시나리오로 평가를 실행하라. 작업 이탈률, 실패한 작업률, 그리고 인젝션 성공 여부를 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Intent | 사용자가 원하는 것 | 범주형 레이블(book_flight, reset_password). 핸들러로 라우팅된다. |
| Slot | 정보 조각 | 봇이 필요로 하는 파라미터(date, destination). 슬롯 채우기는 묻는 순서다. |
| RAG | 검색 더하기 생성 | 관련 문서를 검색한 다음, LLM의 응답을 근거화한다. |
| Tool call | 함수 호출 | LLM이 이름 + 인자를 가진 구조화된 호출을 내보낸다. 런타임이 실행하고 결과를 반환한다. |
| Agent loop | 계획, 실행, 검증 | 작업이 완료될 때까지 도구 호출과 교차하여 LLM 호출을 실행하는 컨트롤러. |
| Prompt injection | 사용자가 프롬프트를 공격함 | 시스템 프롬프트를 무시하려는 악의적 입력. |

## 더 읽을거리 (Further Reading)

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) 대화를 이 분야의 기준으로 만든 논문.
- [Weizenbaum (1966). ELIZA: A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) 원래의 규칙 기반 챗봇 논문.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) 최초의 상태를 가진 챗봇인 PARRY의 감정 변수 구조.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239): LLM 에이전트가 장악하기 직전, Google의 후기 신경망 챗봇 논문.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): 에이전트 루프 패턴을 명명한 논문.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents): 2026년에도 유효한 2024년 프로덕션 가이드.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173): 프롬프트 인젝션 논문.
- [OWASP Top 10 for LLM Applications 2025: LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) 프롬프트 인젝션을 최상위 보안 우려로 만든 순위.
- [AWS: Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) Plan-Verify-Execute와 사용자 확인 흐름을 포함한 실용적인 오케스트레이션 계층 방어.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection): 간접 프롬프트 인젝션으로 인한 표준 제로 클릭 데이터 탈취 CVE. 쓰기 권한 에이전트가 런타임 방어를 필요로 하는 이유에 대한 참조 사례.
