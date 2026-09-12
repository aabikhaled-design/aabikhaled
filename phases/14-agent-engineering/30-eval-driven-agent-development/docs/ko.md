# 평가 주도 에이전트 개발(Eval-Driven Agent Development)

> Anthropic의 지침: "단순한 프롬프트(prompt)로 시작하고, 포괄적인 평가(evaluation)로 그것을 최적화하며, 필요할 때만 다단계(multi-step) 에이전트(agent) 시스템을 추가하라." 평가는 마지막 단계가 아니다. 평가는 Phase 14의 다른 모든 선택을 이끄는 외부 루프(outer loop)다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 세 가지 평가 계층(정적 벤치마크(static benchmark), 커스텀 오프라인(custom offline), 온라인 프로덕션(online production))의 이름을 대고 각각이 무엇을 위한 것인지 설명하기.
- 평가자-최적화자(evaluator-optimizer) 긴밀한 루프(tight loop)를 설명하기.
- 2026년 모범 사례를 설명하기: 평가는 코드 옆에 살고, CI에서 실행되며, PR을 통제한다.
- Phase 14의 모든 레슨을 그것이 생성하는 평가 케이스에 연결하기.

## 문제 (The Problem)

에이전트는 데모를 통과한다. 그러나 데모가 예측할 수 없는 방식으로 프로덕션(production)에서 실패한다. 벤치마크(benchmark)는 "이 모델(model)이 전반적으로 유능한가?"에는 답하지만 "이 에이전트가 내 제품에 맞는 패치를 제대로 출하하고 있는가?"에는 답하지 않는다. 답은 이렇다: 세 계층에서 지속적으로 실행되는 평가, 그리고 모든 가드레일(guardrail)과 학습된 규칙이 평가 케이스에 매핑되는 것.

## 개념 (The Concept)

### 세 가지 평가 계층

1. **정적 벤치마크**: 코드에는 SWE-bench Verified(Lesson 19), 브라우징/데스크톱에는 WebArena/OSWorld(Lesson 20), 제너럴리스트에는 GAIA(Lesson 19), 도구 사용에는 BFCL V4(Lesson 06). 모델 간 비교와 회귀(regression) 통제에 사용한다. 오염(contamination)은 실재한다: SWE-bench+는 32.67%의 솔루션 누출(solution leakage)을 발견했다. 항상 Verified / +-감사 점수를 보고하라.

2. **커스텀 오프라인 평가**: 제품의 형태에 맞춘다:
   - LLM-as-judge(Langfuse, Phoenix, Opik — Lesson 24).
   - 실행 기반(execution-based)(패치를 실행하고 테스트를 확인).
   - 궤적 기반(trajectory-based)(액션 시퀀스를 골드(gold)와 비교; OSWorld-Human은 최상위 에이전트가 골드 대비 1.4~2.7배임을 보여준다).

3. **온라인 평가**: 프로덕션:
   - 세션 리플레이(session replay)(Langfuse).
   - 가드레일 발동 경보(Lesson 16, 21).
   - 스텝별 비용/지연 시간 추적(Lesson 23 OTel 스팬).

### 평가자-최적화자(Anthropic)

긴밀한 루프:

1. 제안자(proposer)가 출력을 생성한다.
2. 평가자(evaluator)가 판정한다.
3. 평가자가 통과시킬 때까지 다듬는다.

이것은 Self-Refine(Lesson 05)를 일반화한 것이다. 신경 쓰는 어떤 에이전트 흐름이든 신뢰성을 위해 평가자-최적화자로 감쌀 수 있다.

### 2026년 모범 사례

- 평가는 코드 옆에 산다.
- 모든 PR마다 CI에서 실행된다.
- 평가 점수로 병합(merge)을 통제한다(예: "main 대비 5% 초과 회귀 없음").
- 모든 가드레일이 평가 케이스에 매핑된다.
- 모든 학습된 규칙(Reflexion, pro-workflow learn-rule)이 실패 케이스에 매핑된다.

### Phase 14를 하나로 묶기

Phase 14의 모든 레슨은 평가 케이스를 생성한다:

| 레슨 | 그것이 생성하는 평가 케이스 |
|--------|------------------------|
| 01 Agent Loop | 예산 소진, 무한 루프 가드 |
| 02 ReWOO | 도구가 실패할 때 플래너가 올바르게 재계획 |
| 03 Reflexion | 학습된 성찰(reflection)이 재시도에 적용됨 |
| 05 Self-Refine/CRITIC | 판정자가 다듬어진 출력을 통과시킴 |
| 06 Tool Use | 인자 강제 변환(argument coercion)이 작동; 알 수 없는 도구는 거부됨 |
| 07-10 Memory | 검색 인용(retrieval citation)이 출처와 일치; 오래된 사실은 무효화됨 |
| 12 Workflow Patterns | 각 패턴이 올바른 출력을 생성 |
| 13 LangGraph | 재개(resume)가 상태를 정확히 재현 |
| 14 AutoGen Actors | DLQ가 크래시된 핸들러를 잡아냄 |
| 16 OpenAI Agents SDK | 가드레일이 올바른 입력에서 발동됨 |
| 17 Claude Agent SDK | 서브에이전트(subagent) 결과가 오케스트레이터로 반환됨 |
| 19-20 Benchmarks | SWE-bench Verified 점수, WebArena 성공률, OSWorld 효율성 |
| 21 Computer Use | 스텝별 안전 장치가 주입된 DOM을 잡아냄 |
| 23 OTel | 스팬이 필수 속성(attribute)을 방출 |
| 26 Failure Modes | 탐지기(detector)가 알려진 실패를 태깅 |
| 27 Prompt Injection | PVE가 오염된 검색을 거부 |
| 28 Orchestration | 슈퍼바이저가 올바른 전문가에게 라우팅 |
| 29 Runtime Shapes | DLQ가 N% 실패를 처리 |

평가 스위트(eval suite)에 각각에 대한 케이스가 있다면, Phase 14를 모두 다룬 것이다.

### 평가 주도 개발이 실패하는 지점

- **베이스라인(baseline) 없음.** 마지막으로 알려진 정상값(last-known-good)이 없는 평가는 해석할 수 없다. 베이스라인을 저장하라.
- **그라운딩 없는 LLM 판정.** 판정자도 환각(hallucinate)을 일으킨다. CRITIC 패턴(Lesson 05): 판정자가 외부 도구에 그라운딩(ground)한다.
- **평가에 대한 과적합(over-fitting).** 평가를 위해 최적화하면 프로덕션 유용성에서 멀어진다. 케이스를 교체(rotate)하라.
- **불안정한(flaky) 평가.** 비결정론적 케이스는 거짓 경보를 유발한다. 시드(seed)를 고정하고 상태를 스냅샷하라.

```figure
ae-eval-three-layers
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 평가 하니스(harness)다:

- 카테고리(벤치마크, 커스텀, 온라인)가 있는 케이스 레지스트리.
- 테스트 대상의 스크립트화된 에이전트.
- 평가자-최적화자 루프: 제안, 판정, 통과 또는 최대 라운드까지 다듬기.
- CI 게이트: 집계 통과율 + 베이스라인 대비 회귀.

실행:

```
python3 code/main.py
```

출력: 케이스별 통과/실패, 회귀 플래그, CI 게이트 판정.

## 라이브러리로 써보기 (Use It)

- 에이전트 코드와 동일한 레포(repo)에 평가 케이스를 작성하라.
- CI를 통해 모든 PR마다 실행하라.
- 회귀 시 빌드를 실패시켜라.
- 시간에 따른 통과율을 추적하라.
- 모든 프로덕션 실패를 새 케이스에 연결하라.

## 산출물 (Ship It)

`outputs/skill-eval-suite.md`는 CI 게이트와 회귀 추적을 갖춘 에이전트 제품을 위한 3계층 평가 스위트를 구축한다.

## 연습 문제 (Exercises)

1. 프로덕션 실패 중 하나를 골라라. 그것을 재현하는 평가 케이스를 작성하라. 지금 에이전트가 그것을 통과하는가?
2. 세 가지 차원(사실성, 톤, 범위)을 가진 도메인용 LLM 판정 루브릭(rubric)을 만들어라. 50개 세션을 채점하라.
3. 평가 스위트를 CI에 연결하라. 5% 이상 회귀 시 빌드를 실패시켜라.
4. 궤적 효율성(trajectory-efficiency) 지표를 추가하라: 에이전트가 골드 궤적 대비 몇 스텝을 거쳤는가?
5. Phase 14의 모든 레슨을 스위트의 평가 케이스에 매핑하라. 빠진 것이 있는가? 그것이 메워야 할 공백이다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 정적 벤치마크(Static benchmark) | "기성 평가" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| 커스텀 오프라인 평가(Custom offline eval) | "도메인 평가" | 제품 형태에 맞춘 LLM-as-judge / 실행 / 궤적 |
| 온라인 평가(Online eval) | "프로덕션 평가" | 세션 리플레이, 가드레일 경보, 비용/지연 시간 추적 |
| 평가자-최적화자(Evaluator-optimizer) | "제안-판정-다듬기" | 판정자가 통과시킬 때까지 반복 |
| CI 게이트(CI gate) | "병합 차단기" | 평가 회귀 시 빌드 실패 |
| 베이스라인(Baseline) | "마지막으로 알려진 정상값" | 회귀를 탐지하기 위한 기준 점수 |
| 궤적 효율성(Trajectory efficiency) | "골드 대비 스텝 수" | 에이전트 스텝 수를 인간 전문가 최소값으로 나눈 값 |

## 더 읽을거리 (Further Reading)

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): "단순하게 시작하고, 평가로 최적화하라"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/): 큐레이션된 벤치마크
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html): 도구 사용 벤치마크
- [Langfuse docs](https://langfuse.com/): 실전에서의 평가 + 세션 리플레이
