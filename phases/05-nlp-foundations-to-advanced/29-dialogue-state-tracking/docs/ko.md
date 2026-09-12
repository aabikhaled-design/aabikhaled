# 대화 상태 추적(Dialogue State Tracking)

> "북쪽에 있는 저렴한 식당을 원해요... 사실 적당한 가격으로 바꿔주세요... 그리고 이탈리아 음식 추가요." 세 턴(turn), 세 번의 상태 업데이트. DST는 예약이 동작하도록 슬롯-값(slot-value) 딕셔너리를 동기화 상태로 유지한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75분

## 문제 (The Problem)

과제 지향 대화 시스템(task-oriented dialogue system)에서, 사용자의 목표는 슬롯-값 쌍의 집합으로 인코딩된다: `{cuisine: italian, area: north, price: moderate}`. 모든 사용자 턴은 슬롯을 추가하거나, 바꾸거나, 제거할 수 있다. 시스템은 전체 대화를 읽고 현재 상태를 올바르게 출력해야 한다.

슬롯 하나만 틀려도 시스템은 엉뚱한 식당을 예약하거나, 엉뚱한 항공편을 잡거나, 엉뚱한 카드에 청구한다. DST는 사용자가 말한 것과 백엔드가 실행하는 것 사이의 경첩이다.

LLM에도 불구하고 2026년에 여전히 중요한 이유:

- 컴플라이언스에 민감한 도메인(은행, 의료, 항공 예약)은 자유 형식 생성이 아니라 결정론적 슬롯 값을 요구한다.
- 도구 사용(tool-use) 에이전트(agent)는 API를 호출하기 전에 여전히 슬롯 해결이 필요하다.
- 다중 턴 정정은 보기보다 어렵다: "아니 사실은, 목요일로 해주세요."

현대 파이프라인(pipeline): 고전적 DST 개념 + LLM 추출기 + 구조화 출력(structured-output) 가드레일(guardrail).

## 개념 (The Concept)

![DST: 대화 이력 → 슬롯-값 상태](../assets/dst.svg)

**과제 구조.** 스키마(schema)가 도메인(식당, 호텔, 택시)과 그 슬롯(cuisine, area, price, people)을 정의한다. 각 슬롯은 비어 있거나, 닫힌 집합의 값으로 채워지거나(price: {cheap, moderate, expensive}), 자유 형식 값(name: "The Copper Kettle")일 수 있다.

**두 가지 DST 정식화.**

- **분류(Classification).** 각 (슬롯, 후보_값) 쌍에 대해 예/아니오를 예측한다. 닫힌 어휘(closed-vocab) 슬롯에 동작한다. 2020년 이전의 표준.
- **생성(Generation).** 대화가 주어지면, 슬롯 값을 자유 텍스트로 생성한다. 열린 어휘(open-vocab) 슬롯에 동작한다. 현대의 기본값.

**지표.** 결합 목표 정확도(Joint Goal Accuracy, JGA): *모든* 슬롯이 올바른 턴의 비율. 전부 아니면 전무(all-or-nothing). MultiWOZ 2.4 리더보드는 2026년에 약 83%에서 정점을 찍는다.

**아키텍처.**

1. **규칙 기반(슬롯 정규식 + 키워드).** 좁은 도메인에 강력한 베이스라인(baseline). 디버깅 가능.
2. **TripPy / BERT-DST.** BERT 인코딩을 동반한 복사 기반(copy-based) 생성. LLM 이전 표준.
3. **LDST(LLaMA + LoRA).** 도메인-슬롯 프롬프팅을 동반한 명령어 튜닝(instruction-tuned) LLM. MultiWOZ 2.4에서 ChatGPT 수준의 품질에 도달한다.
4. **온톨로지 없음(Ontology-free, 2024–26).** 스키마를 건너뛰고, 슬롯 이름과 값을 직접 생성한다. 열린 도메인을 처리한다.
5. **프롬프트 + 구조화 출력(2024–26).** Pydantic 스키마 + 제약 디코딩(constrained decoding)을 동반한 LLM. 5줄의 코드, 프로덕션(production) 준비 완료.

### 고전적 실패 모드

- **턴 간 상호참조(co-reference).** "첫 번째 옵션으로 합시다." 어느 옵션인지 해결해야 한다.
- **덮어쓰기 vs 추가.** 사용자가 "이탈리아 음식 추가요"라고 말한다. cuisine을 교체할 것인가 추가할 것인가?
- **암묵적 확인.** "그래 좋아": 그것이 제안된 예약을 수락한 것인가?
- **정정.** "사실 오후 7시로 해주세요." 다른 슬롯을 지우지 않고 시간을 업데이트해야 한다.
- **이전 시스템 발화에 대한 상호참조.** "네, 그거요." 어느 "그거"?

```figure
n5-slot-tracker
```

## 직접 만들기 (Build It)

### Step 1: 규칙 기반 슬롯 추출기

`code/main.py`를 보라. 정규식 + 동의어 사전은 좁은 도메인에서 표준 발화의 70%를 포괄한다:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

표준 어휘 밖에서는 부서지기 쉽다. 결정론적 슬롯 확인에는 동작한다.

### Step 2: 상태 업데이트 루프

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

세 가지 불변식(invariant):

- 사용자가 건드리지 않은 슬롯을 절대 리셋하지 마라.
- 명시적 부정("음식 종류는 신경 쓰지 마세요")은 반드시 지워야 한다.
- 사용자 정정("사실...")은 추가가 아니라 덮어써야 한다.

### Step 3: 구조화 출력을 동반한 LLM 주도 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic은 유효한 상태 객체를 보장한다. 정규식 없음, 스키마 불일치 없음, 환각된(hallucinated) 슬롯 없음.

### Step 4: JGA 평가

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

보정하라: 시스템이 모든 슬롯을 올바르게 맞추는 턴의 비율은? MultiWOZ 2.4의 경우 2026년 상위 시스템은 80~83%다. 도메인 한정 시스템이라면 좁은 어휘에서 이 수치를 넘어야 한다. 그러지 못하면 LLM 베이스라인에 진다.

### Step 5: 정정 처리

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

정정이 탐지되면, 추가하는 대신 마지막으로 업데이트된 슬롯을 덮어쓴다. LLM 도움 없이 제대로 하기 어렵다. 현대 패턴: 증분적으로 업데이트하기보다 항상 LLM이 이력에서 전체 상태를 재생성하게 하라. 이렇게 하면 정정이 자연스럽게 처리된다.

## 함정 (Pitfalls)

- **전체 이력 재생성 비용.** LLM이 매 턴마다 상태를 재생성하게 하면 총 O(n²) 토큰(token)이 든다. 이력에 상한을 두거나 오래된 턴을 요약하라.
- **스키마 표류(drift).** 사후에 새 슬롯을 추가하면 오래된 학습 데이터가 깨진다. 스키마에 버전을 매겨라.
- **대소문자 민감성.** "Italian" vs "italian" vs "ITALIAN": 모든 곳에서 정규화(normalization)하라.
- **암묵적 상속.** 사용자가 이전에 "4명이요"라고 지정했다면, 다른 시간을 묻는 새 요청이 people을 지우면 안 된다. 항상 전체 이력을 넘겨라.
- **자유 형식 vs 닫힌 집합.** 이름, 시간, 주소는 자유 형식 슬롯이 필요하다. cuisine과 area는 닫혀 있다. 스키마에서 둘 다 섞어라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 접근법 |
|-----------|----------|
| 좁은 도메인(인텐트 하나 또는 둘) | 규칙 기반 + 정규식 |
| 넓은 도메인, 레이블(label) 데이터 있음 | LDST(MultiWOZ 스타일 데이터에 LLaMA + LoRA) |
| 넓은 도메인, 레이블 없음, 프로덕션 준비 | LLM + Instructor + Pydantic 스키마 |
| 음성(spoken / voice) | ASR + 정규화기 + LLM-DST |
| 다중 도메인 예약 흐름 | 도메인별 Pydantic 모델을 동반한 스키마 유도 LLM |
| 컴플라이언스 민감 | 규칙 기반 주력, 확인 흐름을 동반한 LLM 폴백 |

## 산출물 (Ship It)

`outputs/skill-dst-designer.md`로 저장한다:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 슬롯 3개(cuisine, area, price)에 대한 규칙 기반 상태 추적기를 만들어라. 직접 만든 대화 10개에서 테스트하라. JGA를 측정하라.
2. **보통.** 같은 데이터셋을 Instructor + Pydantic + 작은 LLM으로 처리하라. JGA를 비교하라. 가장 어려운 턴을 검사하라.
3. **어려움.** 둘 다 구현하고 라우팅하라: 규칙 기반 주력, 규칙 기반이 신뢰도와 함께 슬롯을 2개 미만으로 내보낼 때 LLM 폴백. 결합된 JGA와 턴당 추론(inference) 비용을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| DST | 대화 상태 추적 | 대화 턴 전체에 걸쳐 슬롯-값 딕셔너리를 유지한다. |
| 슬롯(Slot) | 사용자 인텐트의 단위 | 백엔드가 필요로 하는 명명된 파라미터(cuisine, date). |
| 도메인(Domain) | 그 과제 영역 | 식당, 호텔, 택시: 슬롯의 집합. |
| JGA | 결합 목표 정확도 | 모든 슬롯이 올바른 턴의 비율. 전부 아니면 전무. |
| MultiWOZ | 그 벤치마크 | 다중 도메인 WOZ 데이터셋. 표준 DST 평가. |
| 온톨로지 없는 DST(Ontology-free DST) | 스키마 없음 | 고정 목록 없이 슬롯 이름과 값을 직접 생성한다. |
| 정정(Correction) | "사실..." | 이전에 채워진 슬롯을 덮어쓰는 턴. |

## 더 읽을거리 (Further Reading)

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — the canonical benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — the copy-based DST workhorse.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results.
