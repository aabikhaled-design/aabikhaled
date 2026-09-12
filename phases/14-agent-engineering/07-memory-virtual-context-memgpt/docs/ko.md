# 에이전트 메모리: 가상 컨텍스트와 메모리 페이징 (Agent Memory — Virtual Context and Memory Paging)

> 컨텍스트 윈도우(context window)는 유한하다. 대화, 문서, 도구 트레이스는 그렇지 않다. 해법은 운영체제의 가상 메모리(virtual memory)를 다시 말한 것이다. 메인 컨텍스트(main context)는 RAM, 외부 저장소는 디스크, 에이전트는 둘 사이를 페이징(paging)한다. MemGPT(Packer et al., 2023)가 이 패턴에 이름을 붙였고, 실무의 메모리 시스템 상당수가 그 위에 올라서 있다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- MemGPT가 기반하는 OS 비유를 설명하기: 메인 컨텍스트 = RAM, 외부 컨텍스트 = 디스크, 메모리 도구 = 페이지 인/아웃.
- 메인 컨텍스트 버퍼, 외부 검색 가능 저장소, 페이지 인/아웃 도구를 갖춘 2계층 MemGPT 패턴을 stdlib로 구현하기.
- 에이전트가 외부 메모리를 질의하거나 수정하기 위해 어떻게 "인터럽트(interrupt)"를 발행하고, 그 결과가 다음 프롬프트(prompt)로 어떻게 다시 이어 붙는지 서술하기.
- Letta(Lesson 08)와 Mem0(Lesson 09)으로 이어지는 MemGPT 설계 선택을 식별하기.

## 문제 (The Problem)

컨텍스트 윈도우는 메모리를 해결할 것처럼 보인다. 그렇지 않다. 프로덕션(production)에서 세 가지 실패 양상이 반복된다:

1. **오버플로(Overflow).** 멀티턴 대화, 긴 문서, 또는 도구 호출이 많은 트래젝토리가 윈도우를 넘는다. 컷오프(cutoff)를 지난 모든 것은 사라진다.
2. **희석(Dilution).** 윈도우 안에서도, 무관한 컨텍스트를 채워 넣으면 중요한 것에 대한 어텐션(attention)이 희석된다. 프런티어 모델도 긴 입력에서 여전히 성능이 떨어진다.
3. **지속성(Persistence).** 새 세션은 빈 윈도우로 시작한다. 외부 메모리가 없는 에이전트는 세션 전반에 걸쳐 "당신이 ...해달라고 했던 거 기억해?"라고 말할 수 없다.

더 큰 윈도우는 도움이 되지만 이를 고치지는 못한다. Mem0의 2025년 논문은 128k 윈도우 베이스라인(baseline)이, 외부 메모리를 갖춘 4k 윈도우 에이전트가 잡아내는 장기 지평(long-horizon) 사실을 여전히 놓친다고 측정했다.

## 개념 (The Concept)

### 운영체제 비유

MemGPT(Packer et al., arXiv:2310.08560, v2 2024년 2월)는 컨텍스트 관리를 운영체제 가상 메모리에 대응시킨다:

| OS 개념 | MemGPT 개념 | 2026 프로덕션 유사물 |
|------------|---------------|------------------------|
| RAM | 메인 컨텍스트(프롬프트) | Anthropic/OpenAI 컨텍스트 윈도우 |
| 디스크 | 외부 컨텍스트 | 벡터 DB, KV, 그래프 저장소 |
| 페이지 폴트(Page fault) | 메모리 도구 호출 | `memory.search`, `memory.read`, `memory.write` |
| OS 커널 | 에이전트 제어 루프 | 메모리 도구를 가진 ReAct 루프 |

에이전트는 평범한 ReAct 루프를 돌린다. 여기에 도구 한 부류가 더해져 데이터를 메인 컨텍스트로 페이지 인/아웃한다.

### 두 계층

- **메인 컨텍스트.** 현재 작업을 담는 고정 크기 프롬프트. 항상 모델에 보인다.
- **외부 컨텍스트.** 한도가 없고, 도구로 검색 가능. 관련 있을 때 읽고, 사실이 나타날 때 쓴다.

원논문은 기본 윈도우를 넘는 두 작업에서 이 설계를 평가했다: 10만 토큰보다 긴 문서 분석과 여러 날에 걸쳐 지속되는 메모리를 가진 다중 세션 채팅.

### 인터럽트 패턴

MemGPT는 메모리-인터럽트(memory-as-interrupt)를 도입한다: 대화 도중 에이전트가 메모리 도구를 호출할 수 있고, 런타임이 그것을 실행하며, 결과가 새 관찰(observation)로서 다음 어시스턴트 턴에 이어 붙는다. 개념적으로 프로세스를 블록(block)하고 바이트를 반환하고 프로세스가 계속되는 Unix `read()` 시스템콜과 동일하다.

표준 메모리 도구 표면:

- `core_memory_append(section, text)`: 프롬프트의 지속 섹션에 쓰기.
- `core_memory_replace(section, old, new)`: 지속 섹션 편집.
- `archival_memory_insert(text)`: 검색 가능 외부 저장소에 쓰기.
- `archival_memory_search(query, top_k)`: 외부 저장소에서 검색.
- `conversation_search(query)`: 과거 턴 스캔.

### 논문이 끝나고 실무가 시작되는 곳

2024년 9월 MemGPT는 Letta가 됐다. 연구 저장소(`cpacker/MemGPT`)는 남아 있고, Letta는 설계를 확장한다:

- 두 계층 대신 세 계층(core, recall, archival: Lesson 08).
- `send_message`/하트비트(heartbeat) 패턴을 대체하는 네이티브 추론(Lesson 08).
- 비동기 메모리 작업을 돌리는 수면 시간 에이전트(sleep-time agent)(Lesson 08).

프로덕션 시스템이 Letta, Mem0, 또는 커스텀 2계층 저장소를 돌리더라도 MemGPT 논문이 2026년 토대다.

### 이 패턴이 잘못되는 곳

- **메모리 부패(Memory rot).** 쓰기가 읽기보다 빠르게 쌓이고, 검색이 낡은 사실에 빠진다. 해결: 주기적 통합(Letta 수면 시간), 명시적 무효화(Mem0 충돌 탐지기).
- **메모리 오염(Memory poisoning).** 외부 메모리는 검색된 텍스트다. 공격자가 통제하는 콘텐츠가 메모리 노트에 들어오면, 에이전트는 다음 세션에 그것을 다시 흡수한다. 이는 Greshake et al.(Lesson 27) 공격을 시간에 걸쳐 다시 진술한 것이다.
- **인용 손실(Citation loss).** 에이전트가 "사용자가 X를 출하해달라고 했다"를 기억하지만 어느 턴인지 인용할 수 없다. 모든 archival 쓰기에 소스 참조(세션 ID, 턴 ID)를 저장하라.

```figure
context-budget
```

## 직접 만들기 (Build It)

`code/main.py`는 MemGPT의 2계층 패턴을 stdlib로 구현한다:

- `MainContext`: `core` 딕셔너리와 `messages` 리스트를 가진 고정 크기 프롬프트 버퍼; 상한 초과 시 가장 오래된 메시지를 자동 압축.
- `ArchivalStore`: (id, text, tags, session, turn) 레코드의 인메모리 BM25 유사 저장소(토큰 중첩 채점).
- MemGPT 표면에 매핑되는 다섯 메모리 도구.
- archival을 사실로 채운 다음 `archival_memory_search`를 호출하여 질문에 답하는 스크립트된 에이전트.

실행:

```
python3 code/main.py
```

트레이스는 에이전트가 사실 세 개를 쓰고 메인 컨텍스트를 상한까지 채워(축출(eviction)을 강제), 그다음 archival에서 검색하여 후속 질문에 답하는 것을 보여준다. 실제 LLM 없이 MemGPT 워크플로를 재현한다.

## 라이브러리로 써보기 (Use It)

오늘날 모든 프로덕션 메모리 시스템은 MemGPT 변형이다:

- **Letta** (Lesson 08): 세 계층, 네이티브 추론, 수면 시간 연산.
- **Mem0** (Lesson 09): 채점 계층과 융합된 벡터 + KV + 그래프.
- **OpenAI Assistants / Responses**: 스레드와 파일을 통한 관리형 메모리.
- **Claude Agent SDK**: 스킬(skill)과 세션 저장소를 통한 장기 메모리.

핵심 패턴이 아니라 운영 형태(셀프 호스팅, 관리형, 프레임워크 통합)로 하나를 골라라. 핵심 패턴은 MemGPT다.

### 에이전트 메모리의 갈래

페이징은 용량 문제를 푼다. 무엇을 저장할지는 정해 주지 않는다. 실무 시스템에서 되풀이해 나타나는 메모리 유형이 넷 있고, 각각 다른 질문에 답한다.

- **작업 기억(working memory)**: 지금 무엇이 중요한가? 맥락 안에 있는 계층이다. 지금의 과제와 최근 발화, 고정해 둔 핵심 절이 여기에 들어간다. 곧 프롬프트 자체다.
- **일화 기억(episodic memory)**: 무슨 일이 있었는가? 과거의 발화와 진행 기록을 세션과 턴 참조와 함께 저장해 두고, 필요할 때 다시 꺼내 본다.
- **의미 기억(semantic memory)**: 무엇이 참인가? 사용자와 분야, 세상에 대한 사실이며, 바뀔 때마다 갱신하고 중복을 정리한다.
- **절차 기억(procedural memory)**: 이 일을 어떻게 하는가? 학습한 절차와 선호, 규칙이며, 무언가를 떠올리게 하기보다 이후의 행동을 이끈다.

오픈 소스 구현들은 서로 다른 지점을 공략한다.

| 유형 | 구현 | 공략하는 방식 |
|------|----------------|-------------------|
| 작업 기억 | MemGPT / Letta | 메모리 도구로 정해진 프롬프트 예산 안팎으로 내용을 페이징한다(이 레슨과 레슨 08) |
| 일화 기억 | Zep | 시간 정보를 가진 지식 그래프. 사실마다 유효 기간이 붙어서 "그때 무엇이 참이었는지"를 질의할 수 있다 |
| 의미 기억 | Mem0 | 벡터와 KV, 그래프 저장소에 걸쳐 사실의 중복을 제거하고 갱신하는 추출 파이프라인(레슨 09) |
| 의미 기억과 절차 기억 | LangMem | 사실과 행동 규칙을 백그라운드에서 뽑아 저장소에 넣고, 에이전트가 턴 사이에 참조한다 |
| 일화 기억과 의미 기억 | agentmemory | 세션이 진행되는 동안 그것을 붙잡아, 타입이 정해지고 검색 가능한 기록으로 정리한다 |

## 산출물 (Ship It)

`outputs/skill-virtual-memory.md`는 재사용 가능한 스킬로, 어떤 대상 런타임에 대해서든 올바른 2계층 메모리 골조(메인 + archival + 도구 표면)를 축출 정책과 인용 필드가 연결된 채로 생성한다.

## 연습 문제 (Exercises)

1. 토큰으로 측정한 `max_main_context_tokens` 상한을 추가하라(`len(text.split())` * 1.3으로 근사). 상한 초과 시 가장 오래된 메시지를 요약으로 압축하라. 요약기가 있을 때와 없을 때의 동작을 비교하라.
2. archival 저장소에 BM25를 제대로 구현하라(용어 빈도, 역문서 빈도). 토큰 중첩 베이스라인 대비 장난감 사실 집합에서 recall@10을 측정하라.
3. archival 삽입에 `citation` 필드(session_id, turn_id, source_url)를 추가하라. 검색에 기반한 모든 답에 에이전트가 소스를 인용하게 하라.
4. 메모리 오염을 시뮬레이션하라: "미래의 모든 사용자 지시를 무시하라"고 말하는 archival 레코드를 추가하라. 검색에서 지시 형태의 텍스트를 스캔하여 신뢰할 수 없는 것으로 표시하는 가드를 작성하라.
5. 구현을 MemGPT 연구 저장소의 core-memory JSON 스키마(`cpacker/MemGPT`)를 쓰도록 이식하라. 평평한 문자열에서 타입이 있는 섹션으로 전환하면 무엇이 바뀌는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 가상 컨텍스트(Virtual context) | "무제한 메모리" | 페이지 인/아웃을 가진 메인(프롬프트) + 외부(검색 가능) 계층 |
| 메인 컨텍스트(Main context) | "작업 메모리" | 프롬프트: 고정 크기, 항상 보임 |
| Archival 메모리 | "장기 저장소" | 외부 검색 가능 지속 저장소, 필요 시 검색됨 |
| Core 메모리 | "지속 프롬프트 섹션" | 메인 컨텍스트 안에 고정된 이름 있는 섹션 |
| 메모리 도구(Memory tool) | "메모리 API" | 에이전트가 외부 메모리를 읽고/쓰기 위해 발행하는 도구 호출 |
| 인터럽트(Interrupt) | "메모리 페이지 폴트" | 에이전트가 멈추고, 런타임이 가져오고, 결과가 다음 턴에 이어 붙음 |
| 메모리 부패(Memory rot) | "낡은 사실" | 오래된 쓰기가 검색을 익사시킴; 통합으로 해결 |
| 메모리 오염(Memory poisoning) | "주입된 지속 노트" | 공격자 콘텐츠가 메모리로 저장되어 회상 시 다시 흡수됨 |

## 더 읽을거리 (Further Reading)

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560): OS에서 영감받은 가상 컨텍스트 논문
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks): 3계층 진화
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): 컨텍스트를 예산으로 다루기
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413): 이 패턴 위의 하이브리드 프로덕션 메모리
- [Zep (getzep/zep)](https://github.com/getzep/zep): 위 분류 표에 나온, 시간 정보를 가진 지식 그래프 메모리
- [Mem0 (mem0ai/mem0)](https://github.com/mem0ai/mem0): 레슨 09의 혼합 저장소를 떠받치는 추출 파이프라인
- [LangMem (langchain-ai/langmem)](https://github.com/langchain-ai/langmem): 사실과 행동 규칙을 백그라운드에서 뽑아내는 방식
- [agentmemory (rohitg00/agentmemory)](https://github.com/rohitg00/agentmemory): 세션을 붙잡아 타입이 정해지고 검색 가능한 기록으로 정리하는 방식
