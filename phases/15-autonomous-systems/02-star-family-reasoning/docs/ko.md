# STaR, V-STaR, Quiet-STaR: 스스로 학습한 추론

> 가능한 한 가장 작은 자기 개선(self-improvement) 루프는 근거(rationale) 안에 자리한다. 모델이 사고 사슬(chain of thought)을 생성하고, 정답에 도달한 것들만 남긴 뒤, 그것들로 파인튜닝(fine-tuning)한다. 이것이 STaR다. V-STaR는 검증기(verifier)를 더해 추론 시점(inference-time) 선택을 더 낫게 만든다. Quiet-STaR는 근거를 모든 토큰(token) 수준까지 밀어 내린다. 셋 다 작동한다. 어느 것도 마법이 아니다. 이 루프는 우연히 정답에 도달한 어떤 지름길이든 보존한다.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60분

## 문제 (The Problem)

모델에게 추론을 가르치는 직설적인 방법은 인간이 작성한 추론 흔적(reasoning trace)을 수집하는 것이다. 이 방법은 비싸고 느린 데다, 인간이 기꺼이 작성할 만한 고품질 사고 사슬의 양에 묶여 있다.

STaR(Self-Taught Reasoner, Zelikman et al., 2022)는 묻는다. 모델이 자기 자신의 근거를 작성하고 알려진 정답에 비추어 채점하면 어떨까? 그 루프는 이렇다:

1. 추론 흔적과 답을 샘플링(sampling)한다.
2. 최종 답이 정답이면, 그 흔적을 남긴다.
3. 남긴 흔적들로 파인튜닝한다.
4. 반복한다.

작동한다. GSM8K와 CommonsenseQA 모두 새로운 인간 주석(annotation) 없이 향상되었다. 하지만 이 루프에는 내재된 편향이 있다. 정답을 만들어낸 근거는 추론 자체가 타당했는지와 무관하게 보존된다. V-STaR(Hosseini et al., 2024)는 학습된 검증기로 이를 보완하고, Quiet-STaR(Zelikman et al., 2024)는 그 아이디어를 토큰별 내부 근거로 일반화한다.

## 개념 (The Concept)

### STaR: 잘 된 것을 부트스트랩한다

약간의 약한 추론 능력을 가진 기반 모델(base model)에서 출발한다. 각 학습 문제에 대해 근거와 답을 샘플링한다. 답이 레이블(label)과 일치하면, (문제, 근거, 답) 삼중항을 남긴다. 남긴 집합으로 모델을 파인튜닝한다. 반복한다.

한 가지 비틀기가 중요하다. 모델이 어떤 문제를 결코 맞히지 못한다면, 루프는 그 문제로 학습할 수 없다. STaR는 **합리화(rationalization)**를 더한다. 모델이 실패한 문제에는 정답을 힌트로 주입하고, 그 답으로 이어지는 근거를 만들어내도록 모델에게 다시 프롬프트한다. 합리화된 근거는 학습 집합에 추가된다.

원 논문의 결과(Zelikman et al., 2022): GPT-J 기반 모델이 합리화를 동반한 반복적 STaR 라운드를 통해 GSM8K에서 5.8%에서 10.7%로 향상되었다. 절대값으로 약 5퍼센트포인트. CommonsenseQA에서는 STaR로 학습한 GPT-J 6B가 72.5%에 도달했는데, 이는 손으로 주석한 근거로 학습한, 약 30배 더 큰 모델인 파인튜닝된 GPT-3 175B(~73%)에 필적한다.

### V-STaR: DPO로 검증기를 학습시킨다

STaR는 틀린 근거를 버린다. Hosseini et al.(2024)은 그것들 역시 데이터임을 관찰했다. (근거, "이것이 옳은가")의 모든 쌍은 검증기를 학습시킬 수 있다. 그들은 옳은 풀이와 틀린 풀이 모두에 대해 Direct Preference Optimization(DPO)을 사용해 랭커(ranker)를 만든다. 추론 시점에 N개의 근거를 샘플링하고 검증기의 최상위 선택을 고른다.

보고된 델타: GSM8K와 MATH에서 이전 자기 개선 베이스라인 대비 +4에서 +17퍼센트포인트이며, 이득의 대부분은 추가적인 생성기 파인튜닝이 아니라 추론 시점 선택에 검증기를 사용한 데서 나왔다.

### Quiet-STaR: 토큰별 내부 근거

Zelikman et al.(2024)은 물었다. 문제와 답 사이에서만이 아니라, 모델이 모든 토큰 위치에서 짧은 내부 근거를 생성하도록 학습하면 어떨까? Quiet-STaR는 모델이 예측되는 각 토큰 앞에 숨겨진 "생각(thought)"을 내보내도록 학습시킨 뒤, 학습된 가중치로 생각을 반영한 예측을 베이스라인 예측과 섞는다.

결과: Mistral 7B는 작업별 파인튜닝 없이 GSM8K에서 5.9%에서 10.9%로, CommonsenseQA에서 36.3%에서 47.2%로 절대 제로샷(zero-shot) 향상을 얻었다. 모델은 "언제 생각할지"를 학습했다. 어려운 토큰은 더 긴 내부 근거를 받고, 쉬운 토큰은 거의 받지 않는다.

### 셋 모두가 공유하는 안전 우려는 무엇인가

세 방법 모두 최종 답을 그래디언트 신호(gradient signal)로 사용한다. 결함 있는 추론을 통해 정답에 도달한 근거(지름길을 악용하거나, 찍거나, 일반화되지 않는 패턴을 사용한 근거)가 긍정적으로 강화된다. 분포 내(in-distribution) 문제에서는 지름길이 작동한다. 분포 밖(out-of-distribution) 문제에서는 조용히 무너진다.

V-STaR의 검증기는 근거에 순위를 매기도록 학습해 이를 완화하지만 검증기 역시 같은 레이블 집합으로 학습된다. 검증기는 정직한 불확실성보다 잘 포맷된 틀린 추론을 선호하도록 학습할 수 있다. 더 안전한 설계는 STaR 방식 데이터를 (a) 과정 지도(process-supervised) 보상 모델(답이 아니라 중간 스텝을 보상하는 것) 및 (b) 단순한 지름길을 깨뜨리는 별도 보관된(held-out) OOD 평가와 결합하는 것이다.

### 비교

| 방법 | 학습 신호 | 추론 비용 | 데이터 낭비 | 알려진 실패 양상 |
|---|---|---|---|---|
| STaR | 정답이면 (근거, 답) 유지 | 1x | 모든 틀린 근거를 폐기 | 지름길 근거 |
| STaR + 합리화 | 위 + 정답 힌트 재시도 | 1x | 더 적음 | 합리화된 근거가 그럴듯하지 않을 수 있음 |
| V-STaR | STaR + 양쪽 클래스에서 나온 DPO 검증기 | Nx (best-of-N) | 최소 | 검증기가 자신만만한 오답을 강화할 수 있음 |
| Quiet-STaR | 토큰별 근거 + 혼합 가중치 | 1.5-3x | 최소 | 여전히 답에 조건화된 그래디언트 |

### 2026년 스택에서 이것이 자리하는 곳

STaR는 오래되었다. 하지만 이 패턴은 2025-2026년 도처에서 다시 나타난다. 검증 가능한 수학 문제를 푸는 RL(DeepSeek-R1, Kimi-k1.5, o1)은 STaR의 답에 조건화된 그래디언트 신호를 규모만 키운 것이다. 과정 보상 모델(process reward model)(Lightman et al., 2023; OpenAI의 "Let's verify step by step")은 과정 지도 대안이다. AlphaEvolve(Lesson 3)는 레이블 대신 프로그램 평가기(evaluator)를 가진, 코드용 STaR다. Darwin Godel Machine(Lesson 4)은 에이전트 스캐폴딩(scaffolding) 자체를 위한 STaR다.

STaR를 이해하면 이 모든 것이 맞아떨어진다. 그것은 최소 실행 가능(minimum-viable) 자기 개선 루프다.

```figure
reflection-loop
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 산술 작업에 대해 시뮬레이션된 STaR 루프를 돌린다. 다음을 지켜볼 수 있다:

- 부트스트랩 라운드에 걸쳐 정확도가 어떻게 오르는지.
- 지름길이 어떻게 슬며시 끼어드는지: 시뮬레이터에는 40%의 확률로 정답을 맞히지만 일반화는 잘 못하는 "게으른(lazy)" 근거 클래스가 포함되어 있다. STaR가 그것들을 남기는지 지켜보라.
- 검증기(V-STaR 방식)가 추론에서는 도움이 되지만 학습 중에 끼어든 지름길은 완전히 가지치기할 수 없다는 점.

## 산출물 (Ship It)

`outputs/skill-star-loop-reviewer.md`는 제안된 자가 학습 추론(self-taught-reasoning) 파이프라인을 학습에 쓰기 전에 감사하는 데 도움을 준다.

## 연습 문제 (Exercises)

1. 시뮬레이터를 실행하라. 지름길 빈도를 0으로 설정한 다음 0.4로 설정하라. 두 실행 모두 학습 분포에서 90% 이상을 기록함에도, 최종 정확도가 두 실행 사이에서 얼마나 벌어지는가?

2. 시뮬레이터에 별도 보관된 OOD 테스트를 추가하라. 다른 분포에서 문제를 뽑아 부트스트랩된 모델을 분포 내 집합과 OOD 집합 모두에서 평가하라. 그 간극을 정량화하라.

3. Quiet-STaR 논문(arXiv:2403.09629) Section 3을 읽어라. "end-of-thought" 토큰과 혼합 가중치 헤드(mixing-weight head)를 각각 세 문장으로 설명하라.

4. STaR의 정답이면 유지(keep-if-correct) 필터를, 각 근거 스텝을 독립적으로 보상하는 과정 지도 대안과 비교하라. 레이블링 비용 차이와 그럴듯한 품질 차이를 식별하라.

5. 배포된 모델에서 지름길 근거를 잡아낼 평가 하나를 설계하라. 완벽할 필요는 없다. STaR 루프가 강화할 가장 단순한 지름길을 깨뜨리기만 하면 된다.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| STaR | "Self-Taught Reasoner" | 정답에 도달한 모델 생성 근거로 파인튜닝하고 반복하는 것 |
| 합리화 (Rationalization) | "힌트 준 재시도" | 정답을 주입하고, 기반 모델이 실패한 문제에 대해 근거를 다시 프롬프트하는 것 |
| V-STaR | "Verifier STaR" | 옳은 근거와 틀린 근거 모두로 검증기를 DPO 학습시키고 추론 시점 선택에 사용하는 것 |
| Quiet-STaR | "토큰별 근거" | 모든 토큰 위치에서 숨겨진 생각을 생성하고 베이스라인 예측과 혼합하는 것 |
| 답에 조건화된 그래디언트 (Answer-conditioned gradient) | "결과 기반 신호" | 추론 스텝이 아니라 최종 답을 보상하는 학습 루프 |
| 과정 보상 모델 (Process reward model) | "스텝 수준 검증기" | 결과가 아니라 스텝별 정확성으로 학습된 보상 모델: STaR와 대조됨 |
| 지름길 근거 (Shortcut rationale) | "정답, 틀린 추론" | 일반화되지 않는 패턴으로 레이블에 도달한 근거. STaR는 이것들을 남긴다 |

## 더 읽을거리 (Further Reading)

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465): 최초의 논문.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457): 추론 시점 선택을 위한 DPO 검증기를 추가한다.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629): 토큰별 내부 근거.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): 과정 보상 모델, 대안적 그래디언트 신호.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948): 검증 가능한 작업에 대한 RL, 프런티어 학습으로 확장된 STaR.
