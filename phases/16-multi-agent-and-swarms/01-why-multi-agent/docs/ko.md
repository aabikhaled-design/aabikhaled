# 왜 멀티 에이전트인가? (Why Multi-Agent?)

> 에이전트(agent) 하나는 벽에 부딪힌다. 영리한 선택은 더 큰 에이전트가 아니라 더 많은 에이전트다.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 단일 에이전트의 한계(컨텍스트 오버플로, 혼재된 전문성, 순차 병목)를 식별하고, 언제 여러 에이전트로 쪼개는 것이 옳은 선택인지 설명하기
- 오케스트레이션(orchestration) 패턴(파이프라인, 병렬 팬아웃, 슈퍼바이저, 계층형)을 비교하고 주어진 작업 구조에 맞는 것을 선택하기
- 명확한 역할 경계, 공유 상태, 통신 계약(communication contract)을 갖춘 멀티 에이전트(multi-agent) 시스템을 설계하기
- 멀티 에이전트의 복잡성(지연 시간, 비용, 디버깅 난이도)과 단일 에이전트의 단순성 사이의 트레이드오프(trade-off)를 분석하기

## 문제 (The Problem)

Phase 14에서 단일 에이전트(single agent)를 만들었다. 잘 동작한다. 파일을 읽고, 명령을 실행하고, API를 호출하고, 결과를 놓고 추론할 수 있다. 그런 다음 실제 코드베이스에 이 에이전트를 들이댄다. 파일 200개, 세 가지 언어, 인프라에 의존하는 테스트들, 그리고 코드를 작성하기 전에 외부 API를 조사해야 한다는 요구사항.

에이전트가 막힌다. LLM이 멍청해서가 아니라, 작업이 하나의 에이전트 루프가 감당할 수 있는 범위를 넘어서기 때문이다. 컨텍스트 윈도우(context window)는 파일 내용으로 가득 찬다. 에이전트는 40번의 도구 호출 전에 읽었던 것을 잊어버린다. 연구자이자 코더이자 리뷰어가 동시에 되려고 하다가, 셋 다 형편없이 해낸다.

이것이 단일 에이전트의 한계다. 작업이 다음을 요구할 때마다 이 한계에 부딪힌다.

- **하나의 윈도우에 들어가는 것보다 많은 컨텍스트** - 파일 50개를 읽으면 200k 토큰(token)을 훌쩍 넘긴다
- **단계마다 다른 전문성** - 조사는 코드 생성과는 다른 프롬프팅(prompting)을 요구한다
- **병렬로 일어날 수 있는 작업** - 파일 세 개를 동시에 읽을 수 있는데 왜 순차적으로 읽는가?

## 개념 (The Concept)

### 단일 에이전트의 한계 (The Single-Agent Ceiling)

단일 에이전트는 하나의 루프, 하나의 컨텍스트 윈도우, 하나의 시스템 프롬프트(system prompt)다. 그림으로 그려 보자.

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

세 가지가 망가진다.

1. **컨텍스트 포화(saturation)** - 도구 결과가 쌓인다. 30번째 턴이 되면 에이전트는 파일 내용, 명령 출력, 이전 추론으로 150k 토큰을 소비한 상태다. 5번째 턴의 중요한 세부사항은 사라진다.

2. **역할 혼란** - "너는 연구자이자 코더이자 리뷰어이자 테스터다"라고 말하는 시스템 프롬프트는, 절반쯤 조사하고, 절반쯤 코딩하고, 리뷰는 끝내지 못하는 에이전트를 만들어낸다.

3. **순차 병목** - 에이전트는 파일 A를 읽고, 그다음 B를, 그다음 C를 읽는다. 세 번의 직렬 LLM 호출. 세 번의 직렬 도구 실행. 병렬성은 전혀 없다.

### 멀티 에이전트 해법 (The Multi-Agent Solution)

작업을 쪼갠다. 각 에이전트에게 하나의 일, 하나의 컨텍스트 윈도우, 그리고 그 일에 맞게 조정된 하나의 시스템 프롬프트를 준다.

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

각 에이전트는 다음을 갖는다.
- 집중된 시스템 프롬프트("당신은 코드 리뷰어다. 당신의 유일한 일은 버그를 찾는 것이다.")
- 자기만의 컨텍스트 윈도우(다른 에이전트의 작업으로 오염되지 않음)
- 명확한 입력/출력 계약(연구 노트를 받아 코드를 출력)

### 실제로 이렇게 하는 시스템들 (Real Systems That Do This)

**Claude Code 서브에이전트(subagent)** - Claude Code가 `Task`로 서브에이전트를 생성하면, 범위가 한정된 작업을 가진 자식 에이전트를 만든다. 부모는 자신의 컨텍스트를 깨끗하게 유지한다. 자식은 집중된 작업을 하고 요약을 반환한다.

**Devin** - 플래너 에이전트, 코더 에이전트, 브라우저 에이전트를 운영한다. 플래너는 작업을 단계로 나눈다. 코더는 코드를 작성한다. 브라우저는 문서를 조사한다. 각각 별도의 컨텍스트를 갖는다.

**멀티 에이전트 코딩 팀(SWE-bench)** - SWE-bench에서 최고 성능을 내는 시스템들은 코드베이스를 읽는 연구자, 수정안을 설계하는 플래너, 그리고 이를 구현하는 코더를 사용한다. 단일 에이전트 시스템은 더 낮은 점수를 받는다.

**ChatGPT Deep Research** - 여러 검색 에이전트를 병렬로 생성하여 각각 다른 각도를 탐색한 뒤, 결과를 종합한다.

### 스펙트럼 (The Spectrum)

멀티 에이전트는 이분법이 아니다. 그것은 스펙트럼이다.

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**단일 에이전트** - 하나의 루프, 하나의 프롬프트. 단순한 작업에 적합하다.

**서브에이전트** - 부모가 집중된 하위 작업을 위해 자식을 생성한다. 부모는 계획을 유지한다. 자식들은 보고한다. 이것이 Claude Code가 하는 방식이다.

**파이프라인(pipeline)** - 에이전트들이 순차적으로 실행된다. 에이전트 A의 출력이 에이전트 B의 입력이 된다. 단계적 워크플로에 적합하다: 조사 -> 코드 -> 리뷰 -> 테스트.

**팀(team)** - 에이전트들이 공유 메시지 버스(message bus)를 통해 병렬로 실행된다. 각자 역할을 갖는다. 오케스트레이터(orchestrator)가 조율한다. 서로 다른 기술이 동시에 필요할 때 적합하다.

**스웜(swarm)** - 공유 상태를 가진 다수의 동일하거나 거의 동일한 에이전트들. 고정된 오케스트레이터가 없다. 에이전트들은 큐(queue)에서 작업을 가져온다. 높은 처리량(throughput)의 병렬 작업에 적합하다.

### 네 가지 멀티 에이전트 패턴 (The Four Multi-Agent Patterns)

#### 패턴 1: 파이프라인 (Pipeline)

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

각 에이전트는 데이터를 변환하여 앞으로 전달한다. 추론하기 쉽다. 한 단계의 실패가 나머지를 막는다.

#### 패턴 2: 팬아웃 / 팬인 (Fan-out / Fan-in)

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

작업을 병렬 에이전트들에게 나눈 다음, 결과를 병합한다. 독립적인 하위 작업으로 분해되는 작업에 적합하다.

#### 패턴 3: 오케스트레이터-워커 (Orchestrator-Worker)

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

똑똑한 오케스트레이터가 무엇을 할지 결정하고, 워커(worker)에게 위임하고, 결과를 종합한다. 오케스트레이터 자체가 워커를 생성하는 도구를 가진 에이전트다.

#### 패턴 4: 피어 스웜 (Peer Swarm)

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

중앙 오케스트레이터가 없다. 에이전트들이 피어 투 피어(peer-to-peer)로 통신한다. 결정은 상호작용에서 창발(emerge)한다. 디버깅하기는 더 어렵지만 많은 에이전트로 확장된다.

### 멀티 에이전트를 쓰지 말아야 할 때 (When NOT to Use Multi-Agent)

멀티 에이전트는 복잡성을 더한다. 에이전트 간의 모든 메시지는 잠재적 실패 지점이다. 디버깅이 "하나의 대화를 읽는 것"에서 "다섯 개의 에이전트에 걸쳐 메시지를 추적하는 것"으로 바뀐다.

**다음의 경우 단일 에이전트를 유지하라:**
- 작업이 하나의 컨텍스트 윈도우에 들어간다(작업 데이터 약 100k 토큰 미만)
- 단계마다 다른 시스템 프롬프트가 필요하지 않다
- 순차 실행이 충분히 빠르다
- 작업이 충분히 단순해서, 쪼개는 것이 가치보다 더 많은 오버헤드를 더한다

**복잡성 비용:**
- 모든 에이전트 경계는 손실 압축(lossy compression) 단계다: 에이전트 A의 전체 컨텍스트가 에이전트 B를 위한 메시지로 요약된다
- 조율 로직(누가 무엇을, 언제, 어떤 순서로 하는지)은 그 자체로 버그의 원천이다
- 지연 시간이 늘어난다: N개의 에이전트는 최소 N번의 직렬 LLM 호출을 의미하고, 서로 주고받아야 한다면 더 많아진다
- 비용이 곱해진다: 각 에이전트가 독립적으로 토큰을 소모한다

경험 법칙: 작업이 도구 호출 20번 미만이고 100k 토큰에 들어간다면, 단일 에이전트로 유지하라.

```figure
swarm-messages
```

## 직접 만들기 (Build It)

### 1단계: 과부하된 단일 에이전트

여기 모든 것을 하려는 단일 에이전트가 있다. 하나의 거대한 시스템 프롬프트와 조사·코드·리뷰를 담은 하나의 컨텍스트 윈도우를 갖는다.

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

이 접근법의 문제점:
- 컨텍스트 윈도우는 단계마다 커진다. 리뷰 단계에 이르면, 연구 노트와 코드와 이전 추론을 모두 담고 있다.
- 시스템 프롬프트가 범용적이다. 각 단계에 맞게 조정할 수 없다.
- 아무것도 병렬로 실행되지 않는다.

### 2단계: 전문가 에이전트

이제 이를 쪼갠다. 각 에이전트는 하나의 일을 맡는다.

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

각 전문가는 집중된 프롬프트를 갖는다. 각각은 필요한 입력만 담은 깨끗한 컨텍스트 윈도우를 받는다.

### 3단계: 메시지를 통한 조율

명시적 메시지 전달(message passing)로 전문가들을 연결한다.

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

각 에이전트는 자신에게 보내진 메시지만 받는다. 컨텍스트 오염이 없다. 연구자가 읽은 50k 토큰의 문서는 리뷰어의 컨텍스트에 결코 들어가지 않는다.

### 4단계: 비교

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

멀티 에이전트 버전은 전체 토큰을 더 많이 사용한다(세 에이전트, 세 번의 별도 LLM 호출). 하지만 각 에이전트의 컨텍스트는 깨끗하게 유지된다. 시스템 프롬프트가 전문화되어 있으므로 각 단계의 품질이 향상된다.

## 라이브러리로 써보기 (Use It)

이 레슨은 언제 멀티 에이전트로 갈지 결정하는 재사용 가능한 프롬프트를 만들어낸다. `outputs/prompt-multi-agent-decision.md`를 참고하라.

## 연습 문제 (Exercises)

1. 네 번째 전문가를 추가하라: 코더로부터 코드를, 리뷰어로부터 리뷰 피드백을 받아 테스트를 작성하는 "테스터" 에이전트
2. 리뷰어가 수정을 위해 코더에게 피드백을 다시 보낼 수 있도록 파이프라인을 수정하라(최대 2라운드)
3. 순차 파이프라인을 팬아웃으로 변환하라: 연구자와 "요구사항 분석가" 에이전트를 병렬로 실행한 다음, 그들의 출력을 병합하여 코더에게 전달하라

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 스웜 (Swarm) | "AI 에이전트들의 집단 지성" | 공유 상태를 갖고 고정된 리더가 없는 피어 에이전트들의 집합. 행동은 국소적 상호작용에서 창발한다. |
| 오케스트레이터 (Orchestrator) | "보스 에이전트" | 다른 에이전트를 생성하고 관리하는 도구를 가진 에이전트. 계획하고 위임하지만 실제 작업은 하지 않을 수도 있다. |
| 코디네이터 (Coordinator) | "교통 경찰" | 규칙에 따라 에이전트 간 메시지를 라우팅하는 비(非)에이전트 구성요소(보통 LLM이 아닌 그냥 코드). |
| 합의 (Consensus) | "에이전트들이 동의한다" | 여러 에이전트가 진행하기 전에 합의에 도달해야 하는 프로토콜. 충돌하는 출력을 해소해야 할 때 사용한다. |
| 창발 행동 (Emergent behavior) | "에이전트들이 스스로 알아냈다" | 에이전트 상호작용에서 발생하지만 명시적으로 프로그래밍되지 않은 시스템 수준의 패턴. 유용할 수도 해로울 수도 있다. |
| 팬아웃 / 팬인 (Fan-out / fan-in) | "에이전트를 위한 맵리듀스(map-reduce)" | 작업을 병렬 에이전트들에게 나누고(팬아웃), 그 결과를 결합하는 것(팬인). |
| 메시지 전달 (Message passing) | "에이전트들이 서로 대화한다" | 에이전트 간 통신 메커니즘: 한 에이전트에서 다른 에이전트로 보내지는 구조화된 데이터로, 공유 컨텍스트 윈도우를 대체한다. |

## 더 읽을거리 (Further Reading)

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 멀티 에이전트 패턴에 대한 서베이
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 마이크로소프트의 멀티 에이전트 대화 프레임워크
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code가 Task로 위임하는 방식
- [CrewAI documentation](https://docs.crewai.com/) - 역할 기반 멀티 에이전트 프레임워크
