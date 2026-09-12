# 스킬 라이브러리와 평생 학습 (Voyager)

> Voyager(Wang et al., TMLR 2024)는 실행 가능한 코드를 하나의 스킬(skill)로 다룬다. 스킬은 이름이 붙고, 검색 가능하며, 조합 가능하고, 환경 피드백으로 정제된다. 이것이 Claude Agent SDK 스킬, skillkit, 그리고 2026년 스킬 라이브러리 패턴의 레퍼런스 아키텍처다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- Voyager의 세 구성 요소(자동 커리큘럼(automatic curriculum), 스킬 라이브러리(skill library), 반복 프롬프팅(iterative prompting))의 이름을 대고 각각의 역할을 설명하기.
- Voyager가 왜 행동 공간(action space)을 원시 명령어(primitive command)가 아닌 코드로 만드는지 설명하기.
- 등록, 검색, 조합, 실패 기반 정제를 갖춘 stdlib 스킬 라이브러리 구현하기.
- Voyager의 패턴을 2026년 Claude Agent SDK 스킬과 skillkit 생태계에 매핑하기.

## 문제 (The Problem)

세션마다 모든 능력을 처음부터 다시 만드는 에이전트(agent)는 세 가지를 잘못한다.

1. **토큰을 낭비한다.** 모든 작업이 동일한 추론을 다시 끌어낸다.
2. **진척을 잃는다.** 세션 A에서 학습한 수정이 세션 B로 이전되지 않는다.
3. **장기 지평(long-horizon) 조합에서 실패한다.** 복잡한 작업은 능력 계층(capability hierarchy)을 필요로 하는데, 일회성 프롬프트(prompt)는 이를 표현할 수 없다.

Voyager의 답은 이렇다. 재사용 가능한 각 능력을 라이브러리에 저장된 이름 붙은 코드 덩어리로 다루며, 유사도로 검색하고, 다른 스킬과 조합하며, 실행 피드백으로 정제한다.

## 개념 (The Concept)

### 세 구성 요소

Voyager(arXiv:2305.16291)는 에이전트를 다음을 중심으로 구조화한다.

1. **자동 커리큘럼.** 호기심 기반(curiosity-driven) 제안자가 에이전트의 현재 스킬 집합과 환경 상태를 바탕으로 다음 작업을 고른다. 탐색은 상향식(bottom-up)이다.
2. **스킬 라이브러리.** 각 스킬은 실행 가능한 코드다. 작업이 성공하면 새 스킬이 추가된다. 스킬은 질의-설명 유사도로 검색된다.
3. **반복 프롬프팅 메커니즘.** 실패 시 에이전트는 실행 오류, 환경 피드백, 자기 검증(self-verification) 출력을 받은 뒤 스킬을 정제한다.

마인크래프트 평가(Wang et al., 2024): 베이스라인(baseline) 대비 고유 아이템 3.3배, 돌 도구 8.5배 빠르게, 철 도구 6.4배 빠르게, 맵 횡단 2.3배 길게. 이 수치는 마인크래프트에 한정되지만 패턴은 그대로 이전된다.

### 행동 공간 = 코드

대부분의 에이전트는 원시 명령어를 내보낸다. Voyager는 JavaScript 함수를 내보낸다. 스킬은 다음과 같다.

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

하위 스킬(sub-skill)들로부터 조합된다. 설명과 임베딩(embedding)을 키로 하여 저장된다. 프롬프트가 아니라 프로그램으로 검색된다.

이것이 2026년 Claude Agent SDK 스킬이다. 에이전트가 필요할 때 로드하는, 이름 붙은 검색 가능한 코드 덩어리에 지시문(instruction)을 더한 것이다.

### 스킬 검색

새 작업 "다이아몬드 곡괭이를 만들어라". 에이전트는:

1. 작업 설명을 임베딩한다.
2. 스킬 라이브러리에서 유사한 상위 k개(top-k) 스킬을 질의한다.
3. `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` 등을 검색한다.
4. 검색된 원시 요소들과 새 로직으로 새 스킬을 조합한다.

이것이 MCP 리소스(Phase 13)와 Agent SDK 스킬이 구현하는 패턴이다. 현재 작업에 한정해 지식/코드 표면(surface)을 훑는 검색이다.

### 반복적 정제

Voyager의 피드백 루프:

1. 에이전트가 스킬을 작성한다.
2. 스킬이 환경에 대해 실행된다.
3. 세 가지 신호 중 하나가 반환된다: `success`, `error`(스택 트레이스 포함), `self-verification failure`.
4. 에이전트가 그 신호를 컨텍스트로 삼아 스킬을 다시 작성한다.
5. 성공하거나 최대 라운드에 도달할 때까지 반복한다.

이것은 Self-Refine(Lesson 05)을 코드 생성에 적용한 것이며, 환경에 근거한(environment-grounded) 검증을 갖춘다. CRITIC(Lesson 05)은 외부 도구를 검증자로 삼는 동일한 패턴이다.

### 커리큘럼과 탐색

Voyager의 커리큘럼 모듈은 에이전트가 가진 것과 아직 하지 않은 것을 바탕으로 "호숫가에 대피소를 지어라" 같은 작업을 제안한다. 제안자는 환경 상태와 스킬 인벤토리를 사용해 현재 능력보다 약간 위에 있는 작업(탐색의 최적 지점(sweet spot))을 고른다.

프로덕션 에이전트에서 이는 "무엇이 빠졌는가" 연산자에 해당한다. 현재 스킬 라이브러리와 도메인이 주어졌을 때, 우리가 아직 다루지 않는 스킬은 무엇인가? 팀들은 보통 이를 커리큘럼 리뷰로 수동 구현한다.

### 이 패턴이 잘못되는 지점

- **스킬 라이브러리 부패(rot).** 같은 스킬이 조금씩 다른 설명으로 10번 추가된다. 쓰기 시점에 중복 제거(deduplication)를 추가하라. 검색은 하나만 반환한다.
- **조합된 스킬의 표류(drift).** 부모 스킬이 정제된 자식에 의존한다. 스킬에 버전을 매겨라. v1에 고정(pin)된 부모는 마법처럼 v3을 가져오지 않는다.
- **검색 품질.** 스킬 설명을 대상으로 하는 벡터(vector) 검색은 라이브러리가 수백 개를 넘어 커지면 성능이 저하된다. 태그 필터와 강한 제약("`category=tooling`인 스킬만")으로 보완하라.

```figure
voyager-skills
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 스킬 라이브러리를 구현한다.

- `Skill`: 이름, 설명, 코드(문자열로), 버전, 태그, 의존성.
- `SkillLibrary`: 등록(register), 검색(search, 토큰 겹침), 조합(compose, 의존성의 위상 정렬(topological sort)), 정제(refine, 업데이트 시 버전 올림).
- 세 개의 원시 스킬을 등록하고, 네 번째를 조합하며, 실패를 마주치고, 정제하는 스크립트된 에이전트.

실행:

```
python3 code/main.py
```

트레이스는 라이브러리 쓰기, 검색, 조합, 실패한 실행, v2 정제(Voyager의 루프 전체)를 보여준다.

## 라이브러리로 써보기 (Use It)

- **Claude Agent SDK 스킬**(Anthropic): 2026년 레퍼런스: 각 스킬은 설명, 코드, 지시문을 갖고, 에이전트 세션 중에 필요할 때 로드된다.
- **skillkit**(npm: skillkit): 32개 이상의 AI 코딩 에이전트를 위한 교차 에이전트 스킬 관리.
- **커스텀 스킬 라이브러리**: 도메인 특화(데이터 에이전트를 위한 SQL 스킬, 인프라 에이전트를 위한 Terraform 스킬). Voyager 패턴은 축소되어도 적용된다.
- **OpenAI Agents SDK `tools`**: 낮은 쪽 끝에서; 각 도구가 가벼운 스킬이다.

## 산출물 (Ship It)

`outputs/skill-skill-library.md`는 등록, 검색, 버전 관리, 정제가 배선된 Voyager 형태의 스킬 라이브러리를 임의의 대상 런타임(runtime)에 대해 생성한다.

## 연습 문제 (Exercises)

1. `compose()`에 의존성 순환(dependency-cycle) 검출기를 추가하라. 스킬 A가 B에 의존하고 B가 다시 A에 의존하면 어떻게 되는가? 오류인가 경고인가?
2. 스킬별 버전 고정(version pinning)을 구현하라. 부모 스킬이 자식 `crafting@1`을 조합할 때, `crafting@2`로의 정제가 부모를 조용히 업그레이드해서는 안 된다.
3. 토큰 겹침 검색을 sentence-transformers 임베딩(또는 BM25 stdlib 구현)으로 교체하라. 50개 스킬 토이 라이브러리에서 retrieval@5를 측정하라.
4. "커리큘럼" 에이전트를 추가하라. 현재 라이브러리와 도메인 설명이 주어졌을 때, 빠진 스킬 5개를 제안하게 하라. 매주 호출하라.
5. Anthropic의 Claude Agent SDK 스킬 문서를 읽어라. 토이 라이브러리를 SDK의 스킬 스키마로 포팅하라. 발견 가능성(discoverability)에 대해 무엇이 바뀌는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 스킬(Skill) | "재사용 가능한 능력" | 이름 붙은 코드 덩어리 + 설명, 유사도로 검색 가능 |
| 스킬 라이브러리(Skill library) | "방법에 대한 에이전트 기억" | 검색·조합 가능한 스킬의 영속 저장소 |
| 커리큘럼(Curriculum) | "작업 제안자" | 현재 능력 격차로 구동되는 상향식 목표 생성기 |
| 조합(Composition) | "스킬 DAG" | 스킬이 스킬을 호출함; 실행 시 위상 정렬됨 |
| 반복적 정제(Iterative refinement) | "자기 교정 루프" | 환경 피드백 + 오류 + 자기 검증이 다음 버전으로 되먹임됨 |
| 코드로서의 행동 공간(Action-space-as-code) | "프로그래밍적 행동" | 시간적으로 확장된 행동을 위해 원시 명령어가 아닌 함수를 내보냄 |
| 쓰기 시점 중복 제거(Dedup on write) | "스킬 붕괴" | 거의 중복인 설명들이 하나의 정규(canonical) 스킬로 합쳐짐 |

## 더 읽을거리 (Further Reading)

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291): 원조 스킬 라이브러리 논문
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview): 2026년 제품화로서의 스킬
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk): 실전에서의 스킬과 서브에이전트
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651): Voyager 밑에 있는 정제 루프
