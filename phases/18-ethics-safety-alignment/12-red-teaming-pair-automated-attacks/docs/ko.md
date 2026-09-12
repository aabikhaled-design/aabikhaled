# 레드팀: PAIR와 자동화된 공격 (Red-Teaming: PAIR and Automated Attacks)

> Chao, Robey, Dobriban, Hassani, Pappas, Wong(NeurIPS 2023, arXiv:2310.08419). PAIR(Prompt Automatic Iterative Refinement)는 대표적인 자동화 블랙박스(black-box) 탈옥(jailbreak)이다. 레드팀(red-team) 시스템 프롬프트(system prompt)를 가진 공격자 LLM이 대상 LLM에 대한 탈옥을 반복적으로 제안하며, 시도와 응답을 자신의 채팅 기록에 인컨텍스트(in-context) 피드백으로 누적한다. PAIR는 보통 20번의 쿼리 안에 성공하며, 이는 GCG(Zou 외의 토큰 수준 그래디언트 탐색)보다 수십에서 수백 배 더 효율적이고 화이트박스(white-box) 접근을 요구하지 않는다. PAIR는 이제 GCG, AutoDAN, TAP, Persuasive Adversarial Prompt와 함께 JailbreakBench(arXiv:2404.01318)와 HarmBench의 표준 베이스라인(baseline)이다.

**Type:** Build
**Languages:** Python (stdlib, mock PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- PAIR 알고리즘을 기술하기: 공격자 시스템 프롬프트, 반복적 정제(iterative refinement), 인컨텍스트 피드백.
- 대상이 블랙박스일 때 PAIR가 GCG보다 엄밀하게 더 효율적인 이유를 설명하기.
- 다른 네 가지 자동화 공격 베이스라인(GCG, AutoDAN, TAP, PAP)을 제시하고 각각의 구별되는 특징 하나씩 말하기.
- JailbreakBench와 HarmBench 평가 프로토콜과 각각에서 "공격 성공률(attack success rate)"이 의미하는 바를 기술하기.

## 문제 (The Problem)

레드팀은 한때 수작업 활동이었다. 소수의 전문 테스터가 적대적 프롬프트를 구성하고 어떤 것이 통하는지 추적했다. 이 방식은 규모를 키우기 어렵다. 공격 성공률에는 통계적 표본이 필요하고, 대상은 모델이 출시될 때마다 움직이는 표적이다. PAIR는 레드팀을 블랙박스 대상에 대한 최적화 문제로 바꾼다.

## 개념 (The Concept)

### PAIR 알고리즘

입력:
- 대상 LLM T (우리가 공격하는 모델).
- 심판 LLM J (응답이 탈옥인지 점수를 매긴다).
- 공격자 LLM A (레드팀 옵티마이저(optimizer)).
- 목표 문자열 G: "respond with [harmful instruction]."
- 예산 K (보통 20번의 쿼리).

루프, k in 1..K:
1. A에게 목표 G와 지금까지의 (프롬프트, 응답) 쌍 기록을 프롬프트로 준다.
2. A가 새로운 프롬프트 p_k를 내놓는다.
3. p_k를 T에 제출하고; 응답 r_k를 받는다.
4. J가 목표에 대해 (p_k, r_k)에 점수를 매긴다.
5. 점수가 임계값 이상이면: 중단한다. 탈옥을 찾았다.
6. 아니면 (p_k, r_k)를 A의 기록에 추가하고; 계속한다.

경험적 결과(NeurIPS 2023): GPT-3.5-turbo, Llama-2-7B-chat에 대해 50% 이상의 공격 성공률; 성공까지의 평균 쿼리 수는 10-20 범위.

### PAIR가 효율적인 이유

GCG(Zou 외 2023)는 그래디언트(gradient)로 적대적 토큰 접미사(suffix)를 탐색한다. 화이트박스 모델 접근이 필요하고 읽을 수 없는 접미사를 만든다. PAIR는 블랙박스이며 모델 간에 전이(transfer)되는 자연어 공격을 만든다. PAIR의 인컨텍스트 피드백 덕분에 공격자는 매 거부에서 학습한다. GCG에는 이에 상응하는 장치가 없다(매 토큰 갱신이 이전 진척을 처음부터 다시 발견해야 한다).

### 관련 자동화 공격

- **GCG (Zou 외 2023, arXiv:2307.15043).** 적대적 접미사를 위한 토큰 수준 그래디언트 탐색. 화이트박스, 전이 가능, 읽을 수 없는 문자열을 만든다.
- **AutoDAN (Liu 외 2023).** 계층적 목적 함수로 안내되는 프롬프트에 대한 진화적 탐색.
- **TAP (Mehrotra 외 2024).** 가지치기를 동반한 공격 트리(tree-of-attacks with pruning): 여러 PAIR 방식의 롤아웃(rollout)으로 분기한다.
- **PAP (Zeng 외 2024).** Persuasive Adversarial Prompts: 인간의 설득 기법을 프롬프트 템플릿으로 인코딩한다.

### JailbreakBench와 HarmBench

둘 다(2024) 평가를 표준화한다:

- JailbreakBench(arXiv:2404.01318). OpenAI 정책 10개 범주에 걸친 100개의 유해 행동. 주요 지표로 공격 성공률(ASR). 심판(GPT-4-turbo, Llama Guard, 또는 StrongREJECT)을 요구한다.
- HarmBench(Mazeika 외 2024). 7개 범주에 걸친 510개 행동으로, 의미적·기능적 유해성 테스트를 포함한다. 18개 공격을 33개 모델에 대해 비교한다.

ASR은 보통 고정된 쿼리 예산에서 보고된다. 공격을 비교하려면 예산을 맞춰야 한다; 200 쿼리에서의 90% ASR은 20 쿼리에서의 85% ASR과 비교할 수 없다.

### 2026년 배포에서 중요한 이유

이제 모든 프런티어(frontier) 연구소는 출시 전 프로덕션 모델에 PAIR와 TAP를 실행한다. ASR 궤적은 모델 카드(model card, Lesson 26)와 안전 사례(safety-case) 부록(Lesson 18)에 등장한다. 이 공격은 별난 것이 아니라 표준 인프라다.

### Phase 18에서의 위치

Lesson 12는 자동화 공격의 토대다. Lesson 13(다중샷 탈옥, Many-Shot Jailbreaking)은 상호 보완적인 길이 활용(length-exploit)이다. Lesson 14(아스키 아트 / 시각적)는 인코딩 공격이다. Lesson 15(간접 프롬프트 주입, Indirect Prompt Injection)는 2026년 프로덕션 공격 표면이다. Lesson 16은 방어 도구 대응물(Llama Guard, Garak, PyRIT)을 다룬다.

```figure
al-pair-loop
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 PAIR 루프를 만든다. 대상은 "명백한" 유해 프롬프트를 거부하는 모의 분류기(키워드 필터)다. 공격자는 패러프레이즈(paraphrase), 역할극 프레이밍(roleplay-framing), 인코딩을 시도하는 규칙 기반 정제기다. 심판이 응답에 점수를 매긴다. 공격자가 키워드 필터에 대해 약 5-15회 반복 안에 성공하고 의미 필터에 대해 실패하는 것을 지켜본다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-attack-audit.md`를 만든다. 레드팀 평가 보고서가 주어지면 다음을 감사한다: 어떤 공격이 실행되었는지(PAIR, GCG, TAP, AutoDAN, PAP), 각각 어떤 예산으로, 어떤 심판으로, 어떤 유해 행동 집합(JailbreakBench, HarmBench, 내부)에서.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 내장된 세 가지 공격자 전략에 대해 성공까지의 평균 쿼리 수를 측정하라. 각각이 어떤 대상-방어 가정을 활용하는지 설명하라.

2. 네 번째 공격자 전략(예: 다른 언어로 번역, base64 인코딩)을 구현하라. 키워드 필터 대상과 의미 필터 대상에 대한 새로운 성공까지 평균 쿼리 수를 보고하라.

3. Chao 외 2023의 Figure 5(PAIR 대 GCG 비교)를 읽어라. PAIR의 효율 이점에도 불구하고 GCG가 선호되는 두 가지 시나리오를 기술하라.

4. JailbreakBench는 고정된 목표 집합에 대한 ASR을 보고한다. 공격 다양성(성공한 프롬프트의 분산)을 측정하는 추가 지표를 설계하라. 방어 평가에 다양성이 중요한 이유를 설명하라.

5. TAP(Mehrotra 2024)는 PAIR를 분기 + 가지치기로 확장한다. `code/main.py`에 대한 TAP 방식 확장을 스케치하고 계산 비용 대 성공률 트레이드오프(trade-off)를 기술하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| PAIR | "자동화 탈옥" | Prompt Automatic Iterative Refinement; 공격자-LLM + 심판-LLM 루프 |
| GCG | "그래디언트 탈옥" | 적대적 접미사를 위한 화이트박스 토큰 수준 그래디언트 탐색 |
| 공격 성공률 (ASR) | "k 쿼리에서 탈옥 %" | 주요 지표; 쿼리 예산과 심판 정체성과 함께 보고되어야 함 |
| 심판 LLM (Judge LLM) | "점수 매기는 것" | 응답이 유해 목표를 충족하는지 채점하는 LLM |
| JailbreakBench | "그 평가" | 태그가 붙은 범주를 가진 표준화된 유해 행동 집합 |
| HarmBench | "더 넓은 벤치" | 510개 행동, 기능적 + 의미적 유해성 테스트 |
| TAP | "공격 트리" | 분기 + 가지치기를 동반한 PAIR; 더 높은 연산에서 더 나은 ASR |

## 더 읽을거리 (Further Reading)

- [Chao et al.: Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) PAIR 논문, NeurIPS 2023
- [Zou et al.: Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) GCG 논문
- [Chao et al.: JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) 표준화된 평가
- [Mazeika et al.: HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) 더 넓은 평가
