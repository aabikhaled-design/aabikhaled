# 정렬 신호로서의 지시 따르기(Instruction-Following as Alignment Signal)

> 이후의 모든 RLHF 비판은 이 파이프라인을 겨냥한다. 최적화 압력이 프록시(proxy)를 어떻게 왜곡하는지 공부하기 전에, 먼저 그 프록시를 봐야 한다. InstructGPT(Ouyang et al., 2022)는 기준이 되는 아키텍처를 정의했다. 지시-응답 쌍에 대한 지도 파인튜닝(supervised fine-tuning), 쌍별 선호 순위로 학습된 보상 모델(reward model), 그리고 SFT 정책에 KL 페널티를 둔 채 보상 모델을 대상으로 하는 PPO다. 1.3B InstructGPT가 175B GPT-3보다 선호되었다. 이 결과 하나 때문에 2026년에도 모든 프론티어 연구소가 여전히 RLHF 형태의 사후 학습(post-training) 파이프라인을 내놓는다.

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- InstructGPT 파이프라인의 세 단계와 각 단계에서 사용되는 손실(loss)을 명명하기.
- 1.3B 지시 튜닝(instruction-tuned) 모델이 인간 선호 평가에서 원본 175B GPT-3를 어떻게 이겼는지 설명하기.
- 3단계의 KL 페널티가 무엇을 막아주는지, 그리고 이를 제거하면 왜 모드 추구(mode-seeking) 행동으로 붕괴하는지 진술하기.
- 정렬 세금(alignment tax)과 Ouyang et al.이 이에 맞서 사용한 PPO-ptx 완화책을 설명하기.

## 문제 (The Problem)

사전 학습된 언어 모델은 텍스트를 완성한다. 질문에 답하지 않는다. GPT-3에게 "리스트를 뒤집는 Python 함수를 작성하라"고 하면, 흔히 또 다른 프롬프트(prompt)가 돌아온다. 학습 분포의 대부분이 더 많은 웹 텍스트로 이어지는 웹 텍스트이기 때문이다. 모델은 제 일을 하고 있다. 다만 그 일이 잘못됐을 뿐이다.

이를 고치려고 진지한 연구소라면 모두 같은 프록시를 쓴다. 바로 인간 선호(human preference)다. 두 개의 완성문이 평가자에게 간다. 평가자는 더 나은 것을 고른다. 보상 모델이 그 평가자를 학습한다. 그다음 RL 루프가 보상 모델이 높게 점수 매기는 출력 쪽으로 정책(policy)을 이동시킨다. 세 문장이면 InstructGPT 논지가 다 들어간다. 논문의 나머지는 엔지니어링이다.

## 개념 (The Concept)

### 1단계: 지도 파인튜닝 (SFT)

프롬프트-응답 쌍을 수집한다. 응답은 선의를 가진 사람이라면 썼을 법한 내용이다. Ouyang et al.은 레이블러(labeler)와 OpenAI API에서 얻은 13k개의 프롬프트를 사용했다. 표준 교차 엔트로피(cross-entropy) 손실로 베이스 모델을 이 데이터에 파인튜닝한다.

SFT가 주는 것: 모델이 이제 질문을 이어가는 대신 답한다. SFT가 주지 않는 것: 여러 답이 그럴듯할 때 평가자가 어느 답을 선호하는지에 대한 신호.

### 2단계: 보상 모델 (RM)

각 프롬프트에 대해 SFT 모델에서 K개의 완성문을 샘플링한다. 레이블러가 그것들의 순위를 매긴다. 모든 프롬프트-응답 쌍을 점수 매기는 보상 모델을 학습하되, `y_w`가 `y_l`보다 선호된 쌍에 대해 다음을 만족하도록 한다:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

이것이 브래들리-테리(Bradley-Terry) 쌍별 선호 손실이다. RM은 보통 LM 헤드를 스칼라 헤드로 교체한 SFT 모델에서 초기화한다.

보상 모델은 작다. 175B InstructGPT에는 6B로 충분했다. 또한 취약하다. 논문의 5절은 대부분 작은 규모에서 나타난 보상 해킹(reward-hacking) 행동에 관한 것이다.

### 3단계: KL 페널티를 둔 PPO

목적함수를 정의한다:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

PPO로 최대화한다. KL 항은 `pi`가 SFT 정책에서 멀리 표류하지 못하게 한다. 이것이 없으면 옵티마이저(optimizer)는 적대적 예제(adversarial example)를 찾아낸다. 인간이 실제로 선호해서가 아니라 RM이 한 번도 본 적이 없어서 RM 아래에서 높게 점수가 매겨지는 문자열들이다.

KL 계수 `beta`는 RLHF에서 가장 중요한 단일 하이퍼파라미터(hyperparameter)다. 너무 낮으면: 보상 해킹. 너무 높으면: SFT 대비 개선 없음.

### 정렬 세금 (The alignment tax)

RLHF 이후, 모델은 인간에게 선호되지만 표준 벤치마크(benchmark)(SQuAD, HellaSwag, DROP)에서는 퇴보한다. Ouyang et al.은 이를 정렬 세금(alignment tax)이라 부르고 PPO-ptx로 고친다. 사전 학습 그래디언트(gradient)를 RL 목적함수에 섞어, 보상받은 적 없는 다운스트림 작업을 모델이 어떻게 하는지 잊지 않게 한다.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx는 표준이 되었다. Anthropic, DeepMind, Meta 모두 어떤 변형을 사용한다.

### 결과 (The result)

1.3B InstructGPT(SFT + RM + PPO-ptx)는 약 70%의 경우 레이블러에 의해 175B 베이스 GPT-3보다 선호된다. 그 격차는 프로덕션 트래픽에서 나온 숨겨진 테스트 프롬프트에서 더 벌어진다. 이 수치에서 읽어낼 두 가지:

1. 정렬(alignment)은 역량(capability)과는 다른 축이다. 175B 모델은 더 많은 역량을 가졌고, 1.3B 모델은 더 많은 정렬을 가졌으며, 레이블러는 정렬된 쪽을 선호했다.
2. 역량의 하한선은 베이스 모델이 정한다. 베이스 모델이 한 번도 보지 못한 사실을 알도록 RLHF로 만들 수는 없다.

### 왜 이것이 Phase 18의 기준점인가

이후 레슨의 모든 비판, 곧 보상 해킹(레슨 2), DPO(레슨 3), 아첨(sycophancy)(레슨 4), CAI(레슨 5), 슬리퍼 에이전트(sleeper agent)(레슨 7), 정렬 위장(alignment faking)(레슨 9)은 이 파이프라인의 어떤 부분을 겨냥한다. 보상 해킹은 2단계를 공격한다. DPO는 2단계와 3단계를 합쳐 붕괴시킨다. CAI는 인간 레이블러를 대체한다. 아첨은 레이블러가 편향된 신호임을 보여준다. 정렬 위장은 정책이 3단계를 통째로 우회할 수 있음을 보여준다. 이 파이프라인을 먼저 머릿속에 넣지 않고는 이 비판들 중 어느 것도 따라갈 수 없다.

```figure
al-instruct-pipeline
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 선호 데이터로 세 단계를 시뮬레이션한다. 베이스 "정책"은 행동 {A, B, C}에 대한 편향된 동전이다. 1단계 SFT는 200개의 프롬프트에 대한 레이블러 행동을 모방한다. 2단계는 500개의 쌍별 순위에서 브래들리-테리 보상 모델을 적합한다. 3단계는 SFT 정책에 KL 페널티를 둔 단순화된 PPO 업데이트를 실행한다. 보상이 오르고 KL 발산이 커지며 정책이 표류하는 것을 지켜볼 수 있다. 그리고 KL 항을 꺼서 50번의 업데이트 스텝 안에 보상 해킹이 나타나는 것도 볼 수 있다.

볼 것:

- `beta = 0.1` 대 `beta = 0.0`에서의 보상 궤적.
- 학습 스텝에 따른 KL(pi || pi_SFT).
- 레이블러 선호와 비교한 최종 행동 분포.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-instructgpt-explainer.md`를 생성한다. RLHF 파이프라인 설명이나 논문 초록이 주어지면, 세 단계 중 어느 것이 수정되고 있는지, 각 단계에서 어떤 손실이 사용되는지, KL 페널티나 동등한 정규화 항이 존재하는지를 식별한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. `beta = 0.0`으로 설정하고 200번의 PPO 스텝 이후 행동 분포를 보고하라. 모드 추구 행동을 한 문단으로 설명하라.

2. 보상 모델이 행동 B에 대해 +0.5 편향(편향 시뮬레이션, 시뮬레이션된 보상 버그)을 갖도록 수정하라. `beta = 0.1`로 PPO를 실행하라. KL 페널티가 정책이 그 편향을 악용하는 것을 막는가? 어느 `beta`에서 악용이 보이기 시작하는가?

3. Ouyang et al. (arXiv:2203.02155) Figure 1을 읽어라. PPO를 1, 5, 20, 100 스텝 실행하고 SFT 모델 대비 선호를 측정하여 레이블러 선호 곡선을 재현하라.

4. 논문의 4.3절은 1.3B InstructGPT가 약 70%의 경우 175B GPT-3를 이긴다고 보고한다. 왜 그 비율이 레이블러 자신의 프롬프트보다 숨겨진 프로덕션 프롬프트에서 더 높을까?

5. PPO 손실을 같은 선호 데이터에 대한 DPO(Phase 10 · 08)로 교체하라. 최종 정책 표류(SFT까지의 KL)와 최종 보상을 비교하라. 동일한 보상에서 어느 방법이 더 멀리 표류하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| SFT | "지시 튜닝" | 1단계: 프롬프트-응답 쌍에 대한 교차 엔트로피 파인튜닝 |
| 보상 모델(Reward model) | "the RM" | 쌍별 레이블에 대해 브래들리-테리로 학습된 (프롬프트, 응답)에 대한 스칼라 회귀자 |
| 브래들리-테리(Bradley-Terry) | "쌍별 선호 손실" | -log sigmoid(r_w - r_l); 쌍별 순위 매기기를 이진 분류로 환원한다 |
| KL 페널티 | "정규화 항" | `beta * KL(pi \|\| pi_SFT)`: RL 정책을 SFT 앵커 근처에 유지한다 |
| PPO-ptx | "사전 학습 혼합을 둔 PPO" | 정렬 세금을 상쇄하기 위해 사전 학습 로그 가능도의 일부를 PPO 목적함수에 더한다 |
| 정렬 세금(Alignment tax) | "RLHF 퇴보" | RLHF가 목표로 삼지 않은 표준 벤치마크에서 RLHF 이후 발생하는 하락 |
| 레이블러 선호(Labeler preference) | "정답(ground truth)" | 인간 순위의 표본; RM은 "인간 가치"가 아니라 이것에 대한 통계적 프록시다 |

## 더 읽을거리 (Further Reading)

- [Ouyang et al.: Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) InstructGPT 논문, 이후 모든 RLHF 파이프라인의 토대
- [Stiennon et al.: Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) 요약을 위한 RLHF의 선행 연구
- [Christiano et al.: Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) 원조 선호 기반 RL 정식화
- [Bai et al.: Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) Anthropic의 InstructGPT 파이프라인 HH 확장
