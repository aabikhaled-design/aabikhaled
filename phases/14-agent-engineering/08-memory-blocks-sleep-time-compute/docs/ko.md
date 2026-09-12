# 메모리 블록과 수면 시간 연산 (Letta)

> MemGPT는 2024년 Letta가 됐다. 2026년 진화는 두 아이디어를 추가한다: 모델이 직접 편집할 수 있는 개별 기능적 메모리 블록(memory block), 그리고 주 에이전트가 유휴 상태일 때 메모리를 비동기로 통합하는 수면 시간 에이전트(sleep-time agent). 이것이 메모리를 하나의 대화 너머로 확장하는 방법이다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- Letta가 사용하는 세 메모리 계층(core, recall, archival)과 각각의 역할을 명명하기.
- 메모리 블록 패턴을 설명하기: Human 블록, Persona 블록, 그리고 일급(first-class) 타입 객체로서의 사용자 정의 블록.
- 수면 시간 연산(sleep-time compute)이 무엇인지, 왜 임계 경로(critical path) 밖에 있는지, 왜 주 에이전트보다 강한 모델을 돌릴 수 있는지 서술하기.
- 주 에이전트가 응답을 제공하고 수면 시간 에이전트가 턴 사이에 블록을 통합하는 스크립트된 2에이전트 루프를 구현하기.

## 문제 (The Problem)

MemGPT(Lesson 07)는 가상 메모리(virtual memory) 제어 흐름을 풀었다. 세 가지 프로덕션(production) 문제가 나타났다:

1. **지연(Latency).** 모든 메모리 연산이 임계 경로 위에 있다. 사용자가 기다리는 동안 에이전트가 가지치기하거나, 요약하거나, 조정해야 한다면 꼬리 지연(tail latency)이 폭발한다.
2. **메모리 부패(Memory rot).** 쓰기가 쌓인다. 반박된 사실이 남는다. 검색이 낡은 콘텐츠에 빠진다.
3. **구조 손실(Structure loss).** 평평한 archival 저장소는 "Human 블록은 항상 프롬프트에 있고; Persona 블록은 항상 프롬프트에 있고; Task 블록은 세션마다 교체된다"를 표현할 수 없다.

Letta(letta.com)는 2026년 재작성이다. 메모리 블록은 구조를 명시적으로 만들고, 수면 시간 연산은 통합을 임계 경로 밖으로 옮긴다.

## 개념 (The Concept)

### 세 계층

| 계층 | 범위 | 어디에 사는가 | 작성자 |
|------|-------|----------------|------------|
| Core | 항상 보임 | 메인 프롬프트 안 | 에이전트 도구 호출 + 수면 시간 재작성 |
| Recall | 대화 이력 | 검색 가능 | 자동 턴 로깅 |
| Archival | 임의 사실 | 벡터 + KV + 그래프 | 에이전트 도구 호출 + 수면 시간 흡수 |

Core는 MemGPT의 core다. Recall은 축출된 꼬리를 가진 대화 버퍼다. Archival은 외부 저장소다. 이 분할은 MemGPT의 2계층 과부하를 정리한다.

### 메모리 블록

블록은 core 계층의 타입이 있고, 지속적이며, 편집 가능한 섹션이다. 원래 MemGPT 논문은 두 가지를 정의했다:

- **Human 블록**: 사용자에 관한 사실(이름, 역할, 선호, 목표).
- **Persona 블록**: 에이전트의 자기 개념(정체성, 어조, 제약).

Letta는 임의의 사용자 정의 블록으로 일반화한다: 현재 목표를 위한 `Task` 블록, 코드베이스 사실을 위한 `Project` 블록, 강한 제약을 위한 `Safety` 블록. 각 블록은 `id`, `label`, `value`, `limit`(문자 상한), `description`(모델이 언제 편집할지 알도록)을 가진다.

블록은 도구 표면을 통해 편집 가능하다:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`: 한계에 가까운 블록을 압축.

### 수면 시간 연산

2025년 Letta 추가: 임계 경로 밖, 백그라운드에서 두 번째 에이전트를 돌린다. 수면 시간 에이전트는 대화 트랜스크립트(transcript)와 코드베이스 컨텍스트를 처리하고, `learned_context`를 공유 블록에 쓰며, archival 레코드를 통합하거나 무효화한다.

여기서 다음 속성이 따라온다:

- **지연 비용 없음.** 주 응답은 메모리 연산을 기다리지 않는다.
- **더 강한 모델 허용.** 수면 시간 에이전트는 지연에 제약되지 않으므로 더 비싸고 느린 모델일 수 있다.
- **자연스러운 통합 창.** 사용자가 기다리지 않을 때 중복 제거하고, 요약하고, 반박된 사실을 무효화한다.

이 형태는 사람이 일하는 방식과 일치한다: 작업을 하고, 하룻밤 재워두면, 장기 메모리가 밤새 자리 잡는다.

### Letta V1과 네이티브 추론

Letta V1(`letta_v1_agent`, 2026)은 `send_message`/하트비트(heartbeat)와 인라인 `Thought:` 토큰을 폐기하고 네이티브 추론을 채택한다. Responses API(OpenAI)와 확장 사고(extended thinking)를 갖춘 Messages API(Anthropic)는 추론을 별도 채널로 방출하고, 이 추론은 턴 전반에 걸쳐 전달된다(프로덕션에서는 제공자 간 암호화됨). 제어 루프는 여전히 ReAct다. 사고 트레이스는 프롬프트 형태가 아니라 구조적이다.

### 이 패턴이 잘못되는 곳

- **블록 비대(Block bloat).** 무한한 `block_append`는 한계에 빠르게 도달한다. 상한을 넘기는 쓰기 전에 블록 요약기를 연결하라.
- **조용한 표류(Silent drift).** 수면 시간 에이전트가 블록을 재작성하는데 주 에이전트가 전혀 알아채지 못한다. 블록을 버전 관리하고 트레이스에 차이(diff)를 드러내라.
- **오염된 통합(Poisoned consolidation).** 수면 시간 에이전트가 공격자도 닿을 수 있는 콘텐츠를 core로 처리한다. Lesson 27은 수면 시간 표면에도 적용된다.

```figure
memory-blocks
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다:

- `Block` — id, label, value, limit, description.
- `BlockStore`: CRUD + `near_limit(label)` 헬퍼.
- 두 스크립트된 에이전트: `PrimaryAgent`는 턴을 제공하고, `SleepTimeAgent`는 턴 사이에 통합.
- 블록 쓰기를 가진 세 턴 대화에, 블록을 요약하고 낡은 사실을 무효화하는 수면 시간 패스를 더한 트레이스.

실행:

```
python3 code/main.py
```

트랜스크립트는 분할을 보여준다: 주 턴은 빠르고 원시 쓰기를 생성하며, 수면 패스가 압축하고 정리한다.

## 라이브러리로 써보기 (Use It)

- 참조 구현을 위한 **Letta** (letta.com). 셀프 호스팅 또는 관리형 클라우드.
- 블록 형태 지식으로서의 **Claude Agent SDK 스킬(skills)**: 스킬은 에이전트가 필요 시 로드하는 이름 있고, 버전 관리되고, 검색 가능한 지시 블록이다.
- 저장소 백엔드를 통제하고 싶은 팀을 위한 **커스텀 빌드**. 나중에 마이그레이션할 수 있도록 Letta API 계약을 사용하라.

## 산출물 (Ship It)

`outputs/skill-memory-blocks.md`는 안전 규칙과 인용 연결을 포함하여, 어떤 런타임에 대해서든 수면 시간 훅(hook)을 가진 Letta 형태 블록 시스템을 생성한다.

## 연습 문제 (Exercises)

1. `near_limit`이 참을 반환할 때 블록 값을 모델 생성 요약으로 대체하는 `block_summarize` 도구를 추가하라. 어떤 트리거 임계값이 요약 호출과 블록 오버플로를 모두 최소화하는가?
2. archival에 대한 수면 시간 중복 제거를 구현하라: 텍스트의 토큰 중첩이 90%를 넘는 두 레코드는 하나로 합쳐진다. 임계 경로가 아니라 수면 패스에서만 하라.
3. 블록을 버전 관리하라. 모든 쓰기마다 이전 값과 차이를 기록하라. 운영자가 "에이전트가 왜 X를 잊었는지"를 디버깅할 수 있도록 `block_history(label)`을 노출하라.
4. 수면 시간 에이전트를 신뢰할 수 없는 작성자로 다뤄라. 그들이 Persona나 Safety 블록을 건드릴 때 커밋(commit) 전에 두 번째 에이전트 검토를 요구하라.
5. 예제를 Letta API(`letta_v1_agent`)를 쓰도록 이식하라. 블록 스키마에서 무엇이 바뀌고, 네이티브 추론이 트레이스 형태를 어떻게 바꾸는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 메모리 블록(Memory block) | "편집 가능한 프롬프트 섹션" | core 메모리의 타입 있고, 지속적이며, LLM이 편집 가능한 세그먼트 |
| Human 블록 | "사용자 메모리" | core에 고정된 사용자에 관한 사실 |
| Persona 블록 | "에이전트 정체성" | core에 고정된 자기 개념, 어조, 제약 |
| 수면 시간 연산(Sleep-time compute) | "비동기 메모리 작업" | 임계 경로 밖에서 통합을 하는 두 번째 에이전트 |
| Core / Recall / Archival | "계층" | 3계층 메모리 분할: 항상 보임 / 대화 / 외부 |
| 블록 한계(Block limit) | "상한" | 블록당 문자 한계; 요약을 강제 |
| 네이티브 추론(Native reasoning) | "사고 채널" | 프롬프트 수준 `Thought:`가 아닌 제공자 수준 추론 출력 |
| 학습된 컨텍스트(Learned context) | "수면 출력" | 수면 시간 에이전트가 공유 블록에 쓰는 사실 |

## 더 읽을거리 (Further Reading)

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks): 블록 패턴
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute): 비동기 통합
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent): 네이티브 추론 재작성
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560): 기원
