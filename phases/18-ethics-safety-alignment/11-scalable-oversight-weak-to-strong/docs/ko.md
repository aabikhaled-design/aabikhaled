# 확장 가능한 감독과 약-강 일반화 (Scalable Oversight and Weak-to-Strong Generalization)

> Burns 외(OpenAI Superalignment, "Weak-to-Strong Generalization", 2023)는 초정렬(superalignment) 문제의 대리 지표(proxy)를 제안했다: 더 약한 모델이 만든 레이블(label)을 사용해 강한 모델을 파인튜닝(fine-tuning)하는 것이다. 강한 모델이 불완전한 약한 감독으로부터 올바르게 일반화한다면, 현재의 인간 수준 정렬(alignment) 기법이 초인적 시스템으로 확장될 수 있을지도 모른다. 확장 가능한 감독(scalable oversight)과 W2SG는 상호 보완적이다. 확장 가능한 감독(토론, 재귀적 보상 모델링, 작업 분해)은 감독자의 유효 역량을 높여, 감독 대상 모델을 따라잡을 수 있게 한다. W2SG는 감독자가 제공하는 불완전한 감독이 무엇이든 강한 모델이 그로부터 올바르게 일반화하도록 보장한다. Debate Helps W2SG(arXiv:2501.13124, 2025년 1월)는 이 둘을 결합한다.

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 확장 가능한 감독과 약-강 일반화(weak-to-strong generalization)를 정의하고 둘이 어떻게 상호 보완적인지 설명하기.
- Burns 외 2023의 실험 설정을 기술하기: GPT-2의 레이블을 사용해 GPT-4를 파인튜닝하기.
- 성능 격차 회복(PGR) 지표와 그것이 측정하는 바를 설명하기.
- 세 가지 주요 확장 가능 감독 메커니즘(토론, 재귀적 보상 모델링, 작업 분해)을 제시하고 각각의 강점 하나씩 말하기.

## 문제 (The Problem)

Phase 18에서 지금까지 다룬 모든 정렬 기법은 감독자가 모델의 행동을 평가할 수 있다고 가정한다. 모델이 초인적이라면, 감독자가 약한 고리가 된다. 초정렬 질문은 이렇다: 더 약한 감독자가 더 강하고 정렬된 모델을 신뢰성 있게 만들어낼 수 있는가?

Burns 외는 이를 조작 가능한 경험적 설정으로 환원한다: 약한 것으로 강한 것을 감독하고, 강한 모델의 역량 중 얼마만큼이 약한 감독을 거치고도 살아남는지 측정한다. 이것은 초정렬의 해결책이 아니라 진전을 측정하게 해주는 대리 지표다.

## 개념 (The Concept)

### W2SG: Burns 외의 설정

- 약한 모델: GPT-2 급.
- 강한 모델: GPT-4 급.
- 목표치(target): 해당 작업에서 강한 GPT-4의 상한(ceiling).

절차:
1. 어떤 작업에 대해 약한 모델의 제로샷(zero-shot) 예측을 얻는다.
2. 약한 레이블이 붙은 데이터로 강한 모델을 파인튜닝한다.
3. 파인튜닝된 강한 모델의 정확도를 측정한다.

베이스라인(baseline): 약한 모델 자체의 정확도, 골드 레이블(gold label)에 대한 강한 모델의 지도 학습 상한.

격차 지표: 성능 격차 회복(Performance Gap Recovered, PGR) = (fine-tuned - weak) / (ceiling - weak). PGR이 1.0이면 약한 감독이 격차를 완전히 메운다는 뜻이고, PGR이 0이면 약한 감독이 도움이 되지 않았다는 뜻이다.

### Burns 외의 경험적 발견

자연어 처리(NLP), 체스 퍼즐, 보상 모델링 작업 전반에서 PGR은 일관되게 양수였다: 강한 모델은 약한 감독자의 실수를 넘어 일반화했다. 작업별 PGR은 약 20%에서 약 80%까지 변동했다. 강한 모델은 불완전한 레이블로부터 의도된 작업을 "이해"하며, 사전 학습된 사전 지식(prior)으로 빈틈을 메운다.

Burns 외가 지적한 한계:
- 여기서의 약-강 격차는 정렬이 아니라 역량(capability)의 격차다. 약한 모델은 정렬되어 있다; 진정으로 초인적인 모델은 인간이 정렬했을 수 있지만 훨씬 더 큰 역량 격차가 있다.
- 강한 모델의 일반화는 실제 의도(ground-truth intent)를 회복하는 것보다 작업 사전 지식에 더 기인할 수 있다.
- 강한 모델의 잠재 지식(latent knowledge)을 끌어내는 것이 진짜 문제다; PGR은 특정한 한 가지 조작화(operationalization)를 측정한다.

### 확장 가능한 감독: 세 가지 메커니즘

- **토론(Debate, Irving 외 2018).** U의 두 인스턴스가 반대 입장으로 논쟁하고, 약한 심판이 판정한다. 가정: 설득력 있는 참된 답을 찾는 것이 설득력 있는 거짓 답을 찾는 것보다 쉽다. 최근 2024-2025년 연구(Khan 외, Michael 외)는 토론이 때로는 도움이 되고 때로는 그렇지 않으며, 도움이 되는지 여부는 작업 구조에 달려 있음을 보여준다.
- **재귀적 보상 모델링(Recursive Reward Modeling, Leike 외 2018).** U가 인간을 도와 U+1을 위한 보상 모델을 학습시킨다. 감독자의 유효 역량이 모델과 함께 성장한다.
- **작업 분해(Task Decomposition, Christiano, Shlegeris, Amodei 2018).** 어려운 작업을 인간이 확인할 수 있는 하위 작업으로 재귀적으로 분해한다. 분해 가능성을 가정한다.

각 메커니즘은 작업의 구조나 중간 구성요소의 정렬에 관해 무언가를 가정한다.

### 확장 가능한 감독과 W2SG가 상호 보완적인 이유

확장 가능한 감독은 감독자의 유효 신호 품질을 높인다.
W2SG는 감독자가 제공할 수 있는 불완전한 신호가 무엇이든 그로부터 격차를 메운다.

Lang 외: Debate Helps Weak-to-Strong Generalization(arXiv:2501.13124)는 이 둘을 결합한다: 토론 프로토콜이 더 나은 약한 레이블을 제공하고, 강한 모델은 그 레이블로 학습된다. NLP 작업에서 PGR 향상을 보고했다.

### 조직적 드라마

OpenAI의 Superalignment 팀은 Jan Leike가 Anthropic으로 떠난 후 2024년 5월에 해체되었다. 그 의제(확장 가능한 감독, W2SG, 자동화된 정렬 연구)는 Anthropic과 학술 연구소들(MATS(Lesson 28), Redwood(Lesson 10), Apollo(Lesson 8), METR(Lesson 28))에서 계속되었다. 조직 구조는 바뀌었지만, 연구 질문은 바뀌지 않았다.

### Phase 18에서의 위치

Lesson 6-10은 U가 신뢰할 수 없다는 가정 하에 위협과 방어적 패러다임을 기술한다. Lesson 11은 공격적 패러다임이다: 감독자를 U의 정렬을 검증할 수 있을 만큼 강하게 만드는 것이다. 그다음 Lesson 12-16은 적대적 평가의 실용적 도구로 넘어간다.

```figure
scalable-oversight
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 합성 작업에 대한 W2SG 파인튜닝을 시뮬레이션한다. 약한 레이블러(labeler)는 구조화된 오류와 함께 정확도가 70%이며, 강한 모델은 골드 레이블에서 상한이 95%다. 약한 레이블로 강한 모델을 파인튜닝하고, PGR을 측정하고, 강한-골드(strong-on-gold) 및 약한-단독(weak-alone)과 비교한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-w2sg-pgr.md`를 만든다. 감독 설정 설명이 주어지면, 약한 감독자, 강한 모델, 감독 품질을 식별하고 PGR을 계산(또는 요청)한다. 주장이 "약한 것이 강한 것을 감독할 수 있다"인지 "약한 것 + 감독 메커니즘이 강한 것을 감독할 수 있다"인지 표시한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. weak_accuracy = 0.60, 0.70, 0.80에 대한 PGR을 보고하라. PGR 곡선의 형태를 설명하라.

2. 약한 레이블러가 구조화된 오류(예: 특정 입력 클래스에서 항상 틀림)를 갖도록 수정하라. PGR이 증가하는가, 감소하는가, 동일하게 유지되는가? 설명하라.

3. Burns 외 2023의 4.3절(NLP 작업)을 읽어라. "신뢰도 보조 손실(confidence auxiliary loss)" 직관을 재현하라: 강한 모델이 약한 레이블보다 더 확신할 때, 누가 이기는가?

4. 소프트웨어 엔지니어링 작업을 위해 토론과 작업 분해를 결합한 확장 가능 감독 프로토콜을 설계하라. 각 구성요소의 실패 모드 하나씩을 명명하고, 그 결합이 각각을 어떻게 해결하거나 해결하지 못하는지 설명하라.

5. "약-강 일반화가 초정렬로 가는 실현 가능한 경로다"라는 주장을 반증할 수 있는 것이 무엇인지 명확히 표현하라. 봐야 할 경험적 징후(empirical signature)에 대해 구체적으로 기술하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 확장 가능한 감독 (Scalable oversight) | "감독자를 더 강하게 만들기" | 감독자가 더 역량 있는 모델을 평가하는 능력을 높이는 메커니즘 |
| W2SG | "약한 것이 강한 것을 감독한다" | 강한 모델을 약한 레이블로 파인튜닝하고 회복된 역량을 측정하는 것 |
| PGR | "성능 격차 회복" | (fine-tuned - weak) / (ceiling - weak); 1.0 = 완전히 메움, 0 = 도움 없음 |
| 토론 (Debate) | "두 U 인스턴스가 논쟁한다" | 약한 심판이 두 U 방어자 사이에서 고르는 확장 가능 감독 메커니즘 |
| RRM | "재귀적 보상 모델링" | U가 U+1을 위한 보상 모델 학습을 돕는다; 감독자 역량이 U를 따라간다 |
| 작업 분해 (Task decomposition) | "인간이 확인하는 하위 작업" | 어려운 작업을 인간이 검증할 수 있는 하위 작업으로 재귀적으로 쪼개기 |
| 초정렬 (Superalignment) | "초인적 AI 정렬하기" | 인간이 직접 평가할 수 없는 모델을 정렬하는 데 관련된 연구 의제 |

## 더 읽을거리 (Further Reading)

- [Burns et al.: Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) W2SG 논문
- [Irving, Christiano, Amodei: AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) 토론 메커니즘
- [Leike et al.: Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) 재귀적 보상 모델링
- [Khan et al.: Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) 더 강한 토론자를 사용한 토론에 대한 2024년 경험적 연구
- [Lang et al.: Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) 토론 + W2SG의 2025년 결합
