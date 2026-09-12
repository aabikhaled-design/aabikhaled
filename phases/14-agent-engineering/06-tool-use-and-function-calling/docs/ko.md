# 도구 사용과 함수 호출(Tool Use and Function Calling)

> Toolformer(Schick et al., 2023)는 자기지도(self-supervised) 도구 주석을 처음 선보였다. Berkeley Function Calling Leaderboard V4(Patil et al., 2025)는 2026년 기준선을 세운다: 40% 에이전트형(agentic), 30% 멀티턴(multi-turn), 10% 라이브(live), 10% 논라이브(non-live), 10% 환각(hallucination). 단일 턴은 풀렸지만, 메모리와 동적 의사결정, 장기 지평(long-horizon) 도구 사슬은 아직이다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- Toolformer의 자기지도 학습 신호를 설명하기: 실행이 다음 토큰 손실(loss)을 줄일 때만 도구 주석을 유지한다.
- BFCL V4의 다섯 평가 범주와 각각이 무엇을 측정하는지 명명하기.
- 스키마 검증, 인자 강제 변환(coercion), 실행 샌드박싱(sandboxing)을 갖춘 stdlib 기반 도구 레지스트리(tool registry)를 구현하기.
- 2026년의 세 가지 미해결 문제를 진단하기: 장기 지평 도구 사슬, 동적 의사결정, 메모리.

## 문제 (The Problem)

초기 도구 사용이 던진 질문은 이랬다: 모델이 올바른 함수 호출을 예측할 수 있는가? 현대의 도구 사용은 이렇게 묻는다: 모델이 40단계에 걸쳐 메모리를 유지하고 부분 관측가능성(partial observability) 속에서 도구 실패로부터 회복하면서, 존재하지 않는 도구를 환각하지 않고 도구를 연쇄할 수 있는가?

Toolformer는 베이스라인(baseline)을 확립했다: 모델은 자기지도로 도구를 언제 호출할지 학습할 수 있다. BFCL V4는 2026년 평가 목표를 정의한다. 둘 사이의 간격이 프로덕션(production) 에이전트가 사는 공간이다.

## 개념 (The Concept)

### Toolformer (Schick et al., NeurIPS 2023)

아이디어: 모델이 자신의 사전 학습(pretraining) 말뭉치를 후보 API 호출로 주석 달게 한다. 각 후보에 대해 그것을 실행한다. 도구 결과를 포함하는 것이 다음 토큰에 대한 손실을 줄일 때만 주석을 유지한다. 걸러진 말뭉치에 대해 파인튜닝(fine-tuning)한다.

다룬 도구: 계산기, QA 시스템, 검색 엔진, 번역기, 캘린더. 자기지도 신호는 순전히 도구가 텍스트 예측에 도움이 되는지에 관한 것이다. 사람 레이블(label) 없음.

규모 결과: 도구 사용은 규모에서 창발한다. 작은 모델은 도구 주석에서 손해를 보고 큰 모델은 이득을 본다. 그래서 2026년 프런티어 모델에는 강한 도구 사용이 내장된 반면, 대부분의 7B 모델은 신뢰성을 확보하려면 명시적 도구 사용 파인튜닝이 필요하다.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL은 2026년의 사실상 표준 평가다. V4 구성:

- **에이전트형(Agentic, 40%)**: 전체 에이전트 트래젝토리: 메모리, 멀티턴, 동적 결정.
- **멀티턴(Multi-Turn, 30%)**: 도구 사슬을 가진 대화형 대화.
- **라이브(Live, 10%)**: 사용자가 제출한 실제 프롬프트(prompt)(더 어려운 분포).
- **논라이브(Non-Live, 10%)**: 합성 테스트 케이스.
- **환각(Hallucination, 10%)**: 어떤 도구도 호출하지 말아야 할 때를 탐지.

V3는 상태 기반 평가를 도입했다: 도구 시퀀스 이후, 도구 호출의 AST를 매칭하는 대신 API의 실제 상태(예: "파일이 생성됐는가?")를 확인한다. V4는 웹 검색, 메모리, 형식 민감도 범주를 추가했다.

핵심 2026년 발견: 단일 턴 함수 호출은 거의 풀렸다. 실패는 메모리(턴 전반에 걸쳐 컨텍스트 운반), 동적 의사결정(이전 결과에 기반한 도구 선택), 장기 지평 사슬(20단계 이상 후의 표류), 환각 탐지(맞는 도구가 없을 때 호출 거부)에 집중된다.

### 도구 스키마

모든 제공자는 스키마를 가진다. 세부에서 다르지만 같은 형태를 공유한다:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic은 `input_schema`를 직접 쓰고, OpenAI는 `function.parameters`를 쓴다. 둘 다 JSON Schema를 받는다. 설명(description)이 결정적이다. 모델은 설명을 읽고 올바른 도구를 고른다. 나쁜 도구 설명은 잘못된 도구 선택 실패의 1순위 근본 원인이다.

### 인자 검증

어떤 도구 호출도 신뢰하지 마라. 검증하라:

1. **타입 강제 변환(Type coercion).** 모델이 스키마가 int라고 한 곳에 문자열 "5"를 반환할 수 있다. 모호하지 않으면 강제 변환하고, 그렇지 않으면 거부하라.
2. **열거형 검증(Enum validation).** 스키마가 `status in {"open", "closed"}`라고 했는데 모델이 `"in_progress"`를 방출하면, 서술적 에러로 거부하라.
3. **필수 필드(Required fields).** 필수 필드 누락 -> 크래시가 아니라 즉각적인 에러 관찰을 모델에 돌려보내라.
4. **형식 검증(Format validation).** 날짜, 이메일, URL: 정규식이 아니라 구체적인 파서로 검증하라.

모든 검증 실패는 모델이 올바른 형태로 재시도할 수 있도록 구조화된 관찰을 반환해야 한다.

### 병렬 도구 호출

현대 제공자는 하나의 어시스턴트 턴에서 병렬 도구 호출을 지원한다. 루프:

1. 모델이 서로 다른 `tool_use_id`를 가진 도구 호출 3개를 방출한다.
2. 런타임이 이를 실행한다(독립적이면 병렬로).
3. 각 결과는 `tool_use_id`로 상관된 `tool_result` 블록으로 돌아간다.

공학 규칙: 상관 ID를 결정적인 것으로 다뤄라. 이를 뒤바꾸면 잘못된 도구에 잘못된 결과가 라우팅된다.

### 샌드박싱

도구 실행이 샌드박스 경계다. 자세한 내용은 Lesson 09 참고. 짧은 버전: 모든 도구는 읽기/쓰기 표면, 네트워크 접근, 타임아웃, 메모리 상한을 명시해야 한다. 일반적인 `run_shell(cmd)`은 위험 신호다. 구체적인 `git_status()`가 더 안전하다.

```figure
tool-routing
```

## 직접 만들기 (Build It)

`code/main.py`는 프로덕션 형태의 도구 레지스트리를 구현한다:

- JSON Schema 부분집합 검증기(stdlib만).
- 설명, 입력 스키마, 타임아웃, 실행기를 가진 도구 등록.
- 인자 강제 변환과 열거형 검증.
- 상관 ID를 가진 병렬 도구 디스패치.
- 구조화된 문자열로서의 에러 관찰.

실행:

```
python3 code/main.py
```

트레이스는 미니 에이전트가 한 턴에서 도구 세 개를 호출하는 것을 보여주며, 그중 하나는 의도적으로 잘못된 형식의 호출로, 모델이 대응할 수 있는 서술적 에러로 거부된다.

## 라이브러리로 써보기 (Use It)

모든 제공자는 자체 도구 스키마를 가진다. Anthropic, OpenAI, Gemini, Bedrock. 다중 제공자가 필요하면 변환 계층(OpenAI Agents SDK, Vercel AI SDK, LangChain 도구 어댑터)을 사용하라. BFCL이 참조 벤치마크(benchmark)다. 도구 사용이 제품의 핵심이라면 출하 전에 에이전트에 대해 그것을 돌려라.

## 산출물 (Ship It)

`outputs/skill-tool-registry.md`는 주어진 작업 도메인에 대한 도구 카탈로그, 스키마, 레지스트리를 생성한다. 설명 품질 확인을 포함한다(각 도구의 설명이 모델에게 언제 쓸지 알려주는가?).

## 연습 문제 (Exercises)

1. 모델이 다른 어떤 도구도 명시적으로 쓰기를 거부할 수 있게 하는 "no-op" 도구를 추가하라. BFCL류 환각 테스트에서 측정하라.
2. int-as-string과 float-as-string에 대한 인자 강제 변환을 구현하라. 강제 변환이 실제 버그를 숨기기 시작하는 지점은 어디인가?
3. 도구별 타임아웃과 회로 차단기(circuit breaker)를 추가하라(연속 3회 실패 후 60초간 도구 거부). 이것이 모델이 회복하는 방식에 관해 무엇을 바꾸는가?
4. BFCL V4 설명을 읽어라. 한 범주(예: "멀티턴")를 골라 예제 프롬프트 10개를 에이전트로 돌려라. 통과율을 보고하라.
5. stdlib 검증기를 Pydantic이나 Zod로 이식하라. Pydantic/Zod가 장난감이 놓친 무엇을 잡았는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 함수 호출(Function calling) | "도구 사용" | 검증된 스키마를 가진 구조화 출력 도구 호출 |
| Toolformer | "자기지도 도구 주석" | Schick 2023: 결과가 다음 토큰 손실을 줄이는 도구 호출을 유지 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 벤치마크: 40% 에이전트형, 30% 멀티턴, 10% 라이브, 10% 논라이브, 10% 환각 |
| 도구 스키마(Tool schema) | "모델을 위한 함수 시그니처" | name, description, 인자의 JSON Schema |
| tool_use_id | "상관 ID" | 도구 호출을 그 결과에 묶음; 병렬 디스패치에 필수 |
| 환각 탐지(Hallucination detection) | "호출하지 말아야 할 때를 앎" | V4 범주: 맞는 도구가 없을 때 호출 거부 |
| 인자 강제 변환(Argument coercion) | "문자열-정수 수리" | 예측 가능한 스키마 불일치에 대한 좁은 수정; 모호하면 거부 |
| 샌드박싱(Sandboxing) | "도구 실행 경계" | 도구별 읽기/쓰기 표면, 네트워크, 타임아웃, 메모리 상한 |

## 더 읽을거리 (Further Reading)

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761): 자기지도 도구 주석
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html): 2026 평가 벤치마크
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview): Claude Agent SDK의 프로덕션 도구 스키마
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/): 함수 도구 타입과 Guardrails
