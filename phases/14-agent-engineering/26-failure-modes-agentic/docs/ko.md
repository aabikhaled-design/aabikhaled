# 실패 양상: 에이전트가 망가지는 이유

> MASFT(Berkeley, 2025)는 3개 범주에 걸쳐 14가지 멀티 에이전트 실패 양상(failure mode)을 분류한다. Microsoft의 분류 체계(Taxonomy)는 기존 AI 실패가 에이전트 환경에서 어떻게 증폭되는지 문서화한다. 업계 현장 데이터는 다섯 가지 반복 양상으로 수렴한다: 환각된 동작, 범위 확장, 연쇄 오류, 컨텍스트 손실, 도구 오용.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- MASFT의 세 실패 범주와 각 범주의 구체적 양상을 최소 네 가지씩 거명하기.
- 에이전트 실패가 왜 기존 AI 실패 양상(편향, 환각)을 증폭하는지 설명하기.
- 업계에서 반복되는 다섯 양상과 그 완화책을 기술하기.
- 에이전트 트레이스(trace)에 실패 양상 라벨을 태그하는 stdlib 탐지기를 구현하기.

## 문제 (The Problem)

팀은 트레이스의 90%에서 작동하는 에이전트(agent)를 출하한다. 10%의 실패는 무작위 잡음이 아니다. 소수의 반복되는 범주로 떨어진다. 일단 이름을 붙일 수 있으면, 모니터링하고 고칠 수 있다.

## 개념 (The Concept)

### MASFT (Berkeley, arXiv:2503.13657)

멀티 에이전트 시스템 실패 분류 체계(Multi-Agent System Failure Taxonomy). 14가지 실패 양상이 3개 범주로 군집화된다. 평가자 간 코헨의 카파(Cohen's Kappa) 0.88: 범주는 안정적으로 구별 가능하다.

핵심 주장: 실패는 더 나은 기반 모델(base model)로 고칠 LLM의 한계가 아니라, 멀티 에이전트 시스템의 근본적인 설계 결함이다.

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 기존 AI 실패(편향, 환각, 데이터 누출)가 에이전트 환경에서 증폭된다.
- 자율성에서 새로운 실패가 출현한다: 대규모 의도치 않은 동작, 도구 오용, 임무 표류(mission drift).
- 백서는 에이전트 제품을 위한 리스크 레지스터(risk register)다.

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- 실패는 오케스트레이션(orchestration), 내부 상태 진화, 환경 상호작용에서 발생한다.
- 단지 "나쁜 코드"나 "나쁜 모델 출력"만이 아니다.

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

두 가지 주된 발현:

1. **지시 따르기 이탈(Instruction-following Deviation)**: 에이전트가 시스템 프롬프트(prompt)를 따르지 않는다.
2. **장거리 컨텍스트 오용(Long-range Contextual Misuse)**: 에이전트가 앞선 턴의 컨텍스트를 잊거나 잘못 적용한다.

하위 의도 오류(sub-intention error): 누락(Omission, 빠진 단계), 중복(Redundancy, 반복된 단계), 무질서(Disorder, 순서 어긋난 단계).

### 업계에서 반복되는 다섯 양상

Arize, Galileo, NimbleBrain의 2024-2026 현장 분석은 다음으로 수렴한다:

1. **환각된 동작.** 에이전트가 존재하지 않는 도구를 호출하거나 인자를 날조한다.
2. **범위 확장.** 에이전트가 사용자의 요청을 넘어 과제를 확장한다(추가 PR을 만들고, 추가 이메일을 보낸다).
3. **연쇄 오류.** 한 번의 잘못된 호출이 하류 효과를 촉발한다. 유령 SKU 환각이 네 번의 API 호출을 촉발한다. 다중 시스템 사고.
4. **컨텍스트 손실.** 장기(long-horizon) 과제가 초기 턴의 제약을 잊는다.
5. **도구 오용.** 올바른 도구를 잘못된 인자로 호출하거나, 완전히 잘못된 도구를 호출한다.

연쇄가 치명적이다. 에이전트는 "내가 실패했다"와 "과제가 불가능하다"를 구별하지 못하고, 종종 400 오류에서 성공 메시지를 환각하여 루프를 닫는다.

### 완화: 모든 단계에 게이트

추론 체인의 모든 단계에 자동 검증 게이트를 두어, 환경 상태에 대한 사실 그라운딩(factual grounding)을 검사한다. 구체적으로:

- 스텝마다 안전 분류기(Lesson 21).
- 도구 호출 인자 검증(Lesson 06).
- 검색된 내용을 알려진 사실과 교차 검사(Lesson 05, CRITIC).
- 상태를 재탐침하여 성공 환각 탐지(파일이 실제로 생성되었는가?).

### 실패 모니터링이 잘못되는 지점

- **크래시만 태그.** 대부분의 에이전트 실패는 유효해 보이는 출력을 낸다. 내용 수준 검사가 필요하다.
- **베이스라인(baseline) 없음.** 드리프트(drift) 탐지는 마지막으로 알려진 정상값(last-known-good)이 필요하다; 그것 없이는 "이것이 나빠지고 있다"고 말할 수 없다.
- **과도한 경보.** 모든 실패가 호출을 발생시킨다. 군집화하고 속도 제한하라.

```figure
failure-cascade
```

## 직접 만들기 (Build It)

`code/main.py`는 stdlib 실패 양상 태거를 구현한다:

- 다섯 양상을 다루는 합성 트레이스 데이터셋(dataset).
- 양상별 탐지기 함수(도구 호출, 출력, 반복 동작에 대한 시그니처 패턴).
- 각 트레이스에 라벨을 붙이고 양상 분포를 보고하는 태거.

실행:

```
python3 code/main.py
```

출력: 트레이스별 라벨 + 집계 분포, Phoenix의 트레이스 클러스터링이 드러내는 것의 저렴한 재현.

## 라이브러리로 써보기 (Use It)

- 프로덕션 드리프트 클러스터링에는 **Phoenix**(Lesson 24).
- 세션 리플레이 + 주석에는 **Langfuse**.
- 관측성 플랫폼이 탐지할 수 없는 도메인 특화 시그니처에는 **맞춤(custom)**.

## 산출물 (Ship It)

`outputs/skill-failure-detector.md`는 트레이스 저장소에 배선된, 자기 도메인에 맞춤화된 실패 양상 탐지기를 생성한다.

## 연습 문제 (Exercises)

1. "성공 환각" 탐지기를 추가하라: 에이전트가 성공을 반환하지만 대상 상태가 변하지 않음.
2. 직접 만든 제품에서 실제 트레이스 100개를 태그하라. 어느 양상이 지배적인가? 그것을 고치는 비용은?
3. "연쇄 반경(cascade radius)" 지표를 구현하라: N 단계에서의 실패가 주어졌을 때, 몇 개의 하류 단계에 영향을 미쳤는가?
4. MASFT의 14가지 실패 양상을 읽어라. 자기 제품에 적용되는 세 가지를 골라라. 탐지기를 작성하라.
5. 탐지기 하나를 CI 작업에 배선하라: 트레이스의 5% 이상이 한 양상을 태그하면 빌드를 실패시켜라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| MASFT | "멀티 에이전트 실패 분류 체계" | Berkeley의 14양상 범주화 |
| Cascading error | "파급 실패" | 초기 실수 하나가 N 단계를 통해 전파됨 |
| Context loss | "제약을 잊음" | 장기 턴이 초기 턴 사실을 떨어뜨림 |
| Tool misuse | "잘못된 도구 / 잘못된 인자" | 유효한 호출, 잘못된 호출 방식 |
| Success hallucination | "위조된 완료" | 에이전트가 400에서 성공을 주장; 상태 미변경 |
| Scope creep | "과도한 범위" | 에이전트가 요청보다 더 많이 함 |
| Instruction-following deviation | "불복종" | 시스템 프롬프트나 사용자 제약을 무시 |
| Sub-intention errors | "계획 버그" | 계획 실행에서의 누락, 중복, 무질서 |

## 더 읽을거리 (Further Reading)

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657): 14가지 실패 양상, 3개 범주
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf): 리스크 레지스터
- [Arize Phoenix](https://docs.arize.com/phoenix): 실무에서의 드리프트 클러스터링
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 더 단순한 패턴이 양상을 아예 피하는 경우
