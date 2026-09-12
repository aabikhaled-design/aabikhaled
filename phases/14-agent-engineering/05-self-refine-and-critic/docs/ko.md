# Self-Refine와 CRITIC: 반복적 출력 개선

> Self-Refine(Madaan et al., 2023)은 하나의 LLM을 생성, 피드백, 정제의 세 역할로 루프에서 사용한다. 평균 이득: 7개 작업에서 절대치 +20. CRITIC(Gou et al., 2023)은 검증을 외부 도구로 라우팅하여 피드백 단계를 견고하게 만든다. 2026년에 이 패턴은 모든 프레임워크에서 "평가자-최적화기(evaluator-optimizer)"(Anthropic) 또는 가드레일(guardrail) 루프(OpenAI Agents SDK)로 출하된다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- Self-Refine의 세 프롬프트(생성, 피드백, 정제)를 진술하고, 정제 프롬프트에 왜 이력(history)이 중요한지 설명하기.
- CRITIC의 핵심 통찰을 설명하기: LLM은 외부 근거 없이 자기 검증에 신뢰할 수 없다.
- 이력과 선택적 외부 검증기(verifier)를 갖춘 stdlib 기반 Self-Refine 루프를 구현하기.
- 이 패턴을 Anthropic의 "평가자-최적화기" 워크플로와 OpenAI Agents SDK의 출력 가드레일에 매핑하기.

## 문제 (The Problem)

에이전트가 거의 맞는 답을 생성한다. 어쩌면 코드 한 줄에 구문 오류가 있을 수 있다. 어쩌면 요약이 너무 길 수 있다. 어쩌면 계획이 경계 사례(edge case)를 놓칠 수 있다. 원하는 결과는 분명하다. 에이전트가 자신의 출력을 비평한 다음 고치는 것이다.

Self-Refine은 이것이 단일 모델로, 학습 데이터 없이, 강화 학습 없이 작동함을 보여준다. 하지만 함정이 있다: LLM은 어려운 사실에 대한 자기 검증에 서투르다. CRITIC은 해결책을 명명한다. 검증 단계를 외부 도구(검색, 코드 인터프리터, 계산기, 테스트 러너)로 라우팅하라.

이 두 논문은 함께 반복적 개선을 위한 2026년 기본값을 정의한다: 생성하고, (가능하면 외부적으로) 검증하고, 정제하고, 검증기가 통과하면 멈춘다.

## 개념 (The Concept)

### Self-Refine (Madaan et al., NeurIPS 2023)

LLM 하나, 세 역할:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

핵심 세부: `refine`은 전체 이력(모든 이전 출력과 비평)을 보므로 실수를 반복하지 않는다. 논문은 이를 절제(ablation)한다: 이력을 빼면 품질이 급격히 떨어진다.

표제: GPT-4를 포함한 7개 작업(수학, 코드, 약어, 대화)에 걸쳐 평균 절대치 +20 개선. 학습 없음, 외부 도구 없음, 단일 모델.

### CRITIC (Gou et al., arXiv:2305.11738, v4 2024년 2월)

Self-Refine의 약점: 피드백 단계가 자기 자신을 채점하는 LLM이다. 사실적 주장에 대해 이는 신뢰할 수 없다(환각(hallucination)은 종종 그것을 생성한 모델에게 설득력 있어 보인다). CRITIC은 `feedback(task, output)`을 `verify(task, output, tools)`로 대체하며, 여기서 `tools`는 다음을 포함한다:

- 사실적 주장을 위한 검색 엔진.
- 코드 정확성을 위한 코드 인터프리터.
- 산술을 위한 계산기.
- 도메인 특화 검증기(단위 테스트, 타입 체커, 린터).

검증기는 도구 결과에 근거한 구조화된 비평을 생성한다. 그러면 정제기가 이 비평에 조건화된다.

표제: CRITIC은 비평이 근거를 가지기 때문에 사실적 작업에서 Self-Refine을 능가한다. 외부 검증기가 없는 작업(창의적 글쓰기, 서식)에서는 CRITIC이 Self-Refine으로 환원된다.

### 정지 조건

흔한 두 형태:

1. **검증기 통과.** 외부 테스트가 성공을 반환. 가능하면 선호됨(단위 테스트, 타입 체커, 가드레일 단언).
2. **피드백 미발행.** 모델이 "출력이 괜찮다"고 말함. 더 싸지만 신뢰할 수 없음; 최대 반복 상한과 짝지을 것.

2026년 기본값: 둘을 결합. "검증기가 통과하거나 OR 모델이 괜찮다고 말하고 AND 반복 >= 2 OR 반복 >= max_iterations이면 멈춤."

### 평가자-최적화기 (Evaluator-Optimizer) (Anthropic, 2024)

Anthropic의 2024년 12월 게시글은 이를 다섯 가지 워크플로 패턴 중 하나로 명명한다. 두 역할:

- 평가자(Evaluator): 출력을 채점하고 비평을 생성.
- 최적화기(Optimizer): 비평이 주어지면 출력을 수정.

평가자가 통과할 때까지 루프. 이것이 Anthropic의 틀에서 본 Self-Refine/CRITIC이다. Anthropic이 추가하는 핵심 공학 세부: 평가자와 최적화기 프롬프트는 모델이 그저 고무도장을 찍지 않도록 상당히 달라야 한다.

### OpenAI Agents SDK 출력 가드레일

OpenAI Agents SDK는 이 패턴을 "출력 가드레일(output guardrails)"로 출하한다. 가드레일은 에이전트의 최종 출력에서 돌아가는 검증기다. 가드레일이 발동하면(`OutputGuardrailTripwireTriggered`를 발생시키면) 출력이 거부되고 에이전트가 재시도할 수 있다. 가드레일은 도구를 호출하거나(CRITIC 스타일) 순수 함수일 수 있다(Self-Refine 스타일).

### 2026년 함정

- **고무도장 루프(Rubber-stamp loops).** 같은 모델이 같은 프롬프트 스타일로 생성과 비평을 하면 "내가 보기엔 좋다"로 수렴한다. 구조적으로 다른 프롬프트를 쓰거나, 비평에는 더 작고 싼 모델을 써라.
- **과도한 정제(Over-refinement).** 각 정제 패스는 지연(latency)과 토큰을 더한다. 1~3패스를 예산으로 잡고, 그 이후에는 사람 검토로 에스컬레이션하라.
- **사소한 작업에 대한 CRITIC.** 외부 검증기가 없으면 CRITIC은 Self-Refine으로 퇴화한다. 스텁(stub) 검증기를 위해 지연을 치르지 마라.

```figure
self-refine
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 작업에 Self-Refine과 CRITIC을 구현한다: 주제가 주어지면 짧은 글머리 기호 목록을 생성하기. 검증기는 서식을 확인한다(글머리 3개, 각각 60자 미만). CRITIC은 알려진 환각을 벌하는 외부 "사실 검증기"를 추가한다.

구성 요소:

- `generate`: 스크립트된 생성기.
- `feedback`: LLM 스타일 자기 비평.
- `verify_external`: CRITIC 스타일 근거 검증기.
- `refine`: 이력이 주어지면 출력을 재작성.
- 정지 조건: 검증기 통과 또는 최대 4회 반복.

실행:

```
python3 code/main.py
```

Self-Refine과 CRITIC 실행을 비교하라. CRITIC은 Self-Refine이 놓친 사실 오류를 잡는데, 외부 검증기가 자기 비평자에게 없는 근거를 가지기 때문이다.

## 라이브러리로 써보기 (Use It)

Anthropic의 평가자-최적화기는 Claude 친화적 언어로 표현한 이 패턴이다. OpenAI Agents SDK의 출력 가드레일은 CRITIC 형태다(가드레일이 도구를 호출할 수 있음). LangGraph는 Self-Refine처럼 읽히는 성찰 노드를 제공한다. Google의 Gemini 2.5 Computer Use는 CRITIC 변형인 단계별 안전 평가자를 추가한다: 모든 행동은 커밋(commit) 전에 검증된다.

## 산출물 (Ship It)

`outputs/skill-refine-loop.md`는 작업 형태, 검증기 가용성, 반복 예산이 주어졌을 때 평가자-최적화기 루프를 구성한다. 생성기, 평가자/검증기, 최적화기에 대한 프롬프트와 정지 정책을 방출한다.

## 연습 문제 (Exercises)

1. 장난감을 max_iterations=1로 돌려라. CRITIC이 여전히 도움이 되는가?
2. 외부 검증기를 시끄러운 것으로 교체하라(무작위 30% 거짓 양성). 루프가 무엇을 하는가? 이것이 대부분의 가드레일 스택의 2026년 현실이다.
3. "서로 다른 모델에서의 생성기-비평자" 변형을 구현하라: 큰 모델이 생성하고, 작은 모델이 비평한다. 같은 모델을 이기는가?
4. CRITIC 3절(arXiv:2305.11738 v4)을 읽어라. 세 가지 검증 도구 범주를 명명하고 각각에 예를 들어라.
5. OpenAI Agents SDK의 `output_guardrails`를 CRITIC의 검증기 역할에 매핑하라. SDK가 무엇을 잘못하고, 무엇을 잘하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Self-Refine | "자기 자신을 고치는 LLM" | 한 모델에서 생성 -> 피드백 -> 정제 루프, 이력 포함 |
| CRITIC | "도구 근거 검증" | 피드백을 외부 검증기(검색, 코드, 계산, 테스트)로 대체 |
| 평가자-최적화기(Evaluator-Optimizer) | "Anthropic 워크플로 패턴" | 두 역할(평가자가 채점, 최적화기가 수정)을 수렴까지 루프 |
| 출력 가드레일(Output guardrail) | "사후 확인" | 에이전트가 출력을 생성한 후 돌아가는 OpenAI Agents SDK 검증기 |
| 검증 단계(Verify step) | "비평 국면" | 핵심을 떠받치는 결정: 근거를 가지거나 자기 평가 |
| 정제 이력(Refine history) | "모델이 이미 시도한 것" | 이전 출력 + 비평이 정제 프롬프트 앞에 붙음; 빼면 품질이 무너짐 |
| 고무도장 루프(Rubber-stamp loop) | "자기 동의 실패" | 같은 프롬프트 비평이 "좋아 보인다"를 반환; 구조적으로 다른 프롬프트로 해결 |
| 정지 조건(Stop condition) | "수렴 테스트" | 검증기 통과 OR 피드백 없음 AND 반복 상한; 절대 단일 조건이 아님 |

## 더 읽을거리 (Further Reading)

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651): 표준 논문
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738): 도구 근거 검증
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 평가자-최적화기 워크플로 패턴
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/): CRITIC 형태 검증기로서의 출력 가드레일
