# 보상 모델링과 RLHF (Reward Modeling & RLHF)

> 인간은 "좋은 어시스턴트 응답"에 대한 보상 함수를 쓸 수 없지만, 두 응답을 비교해 더 나은 것을 고를 수는 있다. 그 비교에 보상 모델(reward model)을 맞춘 뒤, 언어 모델을 그것에 대해 강화 학습(RL)한다. Christiano 2017. InstructGPT 2022. GPT-3를 ChatGPT로 바꾼 레시피다. 2026년에는 대부분 DPO로 대체되고 있지만: 사고 모델은 그대로 남는다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45분

## 문제 (The Problem)

다음-토큰-예측 목적함수로 언어 모델을 학습시켰다고 하자. 문법적으로 옳은 영어를 쓴다. 또한 거짓말을 하고, 횡설수설하고, 거절해야 할 때 거절을 거부한다. 더 많은 사전 학습(pretraining)으로는 이를 고칠 수 없다. 웹 텍스트가 문제이지, 해법이 아니기 때문이다.

여기서 필요한 것은 "지시 X에 대해 응답 A가 응답 B보다 낫다"고 말하는 *스칼라 보상*이다. 그 보상 함수를 손으로 쓰는 것은 불가능하다. "유용성(helpfulness)"은 토큰에 대한 닫힌 형식 표현이 아니다. 하지만 인간은 두 출력을 비교해 선호를 표시할 수 있다. 그것은 대규모로 수집하기에 저렴하다.

RLHF(Christiano et al. 2017; Ouyang et al. 2022)는 선호를 보상 모델로 변환한 뒤, 그 보상에 대해 PPO를 통해 LM을 최적화한다. 세 단계로: SFT → RM → PPO. 그것이 ChatGPT, Claude, Gemini, 그리고 2023–2025년의 다른 모든 정렬된(aligned) LLM을 출시한 레시피다.

2026년에 PPO 단계는 대부분 DPO(Phase 10 · 08)로 대체되는데, 더 저렴하고 정렬 튜닝에 거의 그만큼 좋기 때문이다. 하지만 *보상 모델* 부분은 여전히 모든 Best-of-N 샘플러, 모든 검증 가능 보상 기반 RL 파이프라인(pipeline), 그리고 과정 보상 모델(process reward model)을 사용하는 모든 추론 모델의 바탕을 이룬다. RLHF를 이해하면 정렬 스택 전체를 이해하게 된다.

## 개념 (The Concept)

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**1단계: 지도 파인튜닝(Supervised Fine-Tuning, SFT).** 사전 학습된 베이스 모델에서 시작한다. 목표 행동에 대한 인간 작성 시연(지시 따르기 응답, 유용한 답변 등)으로 파인튜닝(fine-tuning)한다. 결과: *좋은 행동 쪽으로 편향된* 그러나 여전히 무한한 행동 공간을 가진 모델 `π_SFT`.

**2단계: 보상 모델 학습.**

- 프롬프트(prompt) `x`에 대한 응답 쌍 `(y_+, y_-)`을 수집하고, 인간이 "y_+가 y_-보다 선호됨"으로 레이블(label)한다.
- 보상 모델 `R_φ(x, y)`가 `y_+`에 더 높은 점수를 부여하도록 학습한다.
- 손실(loss): **브래들리-테리 쌍별 로지스틱(Bradley-Terry pairwise logistic)**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ는 시그모이드다. 보상의 차이가 선호의 로그 오즈(log-odds)를 함의한다. BT는 1952년(Bradley-Terry) 이래 표준이었으며 현대 RLHF에서 지배적 선택이다.

- `R_φ`는 보통 위에 스칼라 헤드를 얹은 SFT 모델에서 초기화된다. 같은 트랜스포머(transformer) 백본; 단일 선형 층(layer)이 보상을 출력한다.

**3단계: KL 페널티를 가진 RM에 대한 PPO.**

- 학습 가능한 정책 `π_θ`를 `π_SFT`에서 초기화한다. 동결된 *참조* `π_ref = π_SFT`를 유지한다.
- 응답 `y`의 끝에서의 보상:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 페널티는 `π_θ`가 `π_SFT`에서 임의로 표류하는 것을 막는다. 단단한 신뢰 영역(trust region)이 아니라 *정규화기(regularizer)*다. `β`는 보통 `0.01`-`0.05`다.
- 이 보상으로 PPO(Lesson 08)를 실행한다. 어드밴티지(advantage)는 토큰 수준 궤적(trajectory)에서 계산되지만, RM은 전체 응답만 채점한다.

**왜 KL인가?** 그것이 없으면 PPO는 기꺼이 보상 해킹(reward-hacking) 전략을 찾는다. RM은 분포 내 완성에서만 학습되었기 때문이다. 분포 밖 응답이 어떤 인간 작성 응답보다 높은 점수를 받을 수 있다. KL은 `π_θ`를 RM이 학습된 다양체(manifold) 근처에 유지한다. RLHF에서 가장 중요한 단일 손잡이다.

**2026년 현황:**

- **DPO** (Rafailov 2023): 닫힌 형식 대수가 2+3단계를 선호 데이터에 대한 단일 지도 손실로 압축한다. RM 없음, PPO 없음. 계산의 일부로 정렬 벤치마크(benchmark)에서 같은 품질. Phase 10 · 08에서 다룬다.
- **GRPO** (DeepSeek 2024–2025): 크리틱(critic) 대신 그룹 상대 베이스라인(baseline)을 가진 PPO, 인간 학습 RM 대신 *검증기(verifier)*(코드 실행 / 수학 답 일치)로부터의 보상. 추론 모델에 지배적이다. Phase 9 · 12에서 다룬다.
- **과정 보상 모델(process reward models, PRMs):** 부분 해(각 추론 스텝)를 채점하며, 추론을 위한 RLHF와 GRPO 변형 모두에서 사용된다.
- **헌법적 AI / RLAIF:** 인간 대신 정렬된 LLM을 사용해 선호를 생성한다. 선호 예산을 확장한다.

```figure
reward-model
```

## 직접 만들기 (Build It)

이 레슨은 문자열로 표현된 작은 합성 "프롬프트"와 "응답"을 사용한다. RM은 토큰 가방(bag-of-tokens) 표현에 대한 선형 채점기다. 실제 LLM 없음: 규모가 아니라 파이프라인의 *형태*가 중요하다. `code/main.py`를 보라.

### 1단계: 합성 선호 데이터

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

실제 RLHF에서는 이것이 인간 레이블러로 대체된다. 형태(`(prompt, preferred_response, rejected_response)`)는 동일하다.

### 2단계: 브래들리-테리 보상 모델

선형 점수: `R(x, y) = w · bag(y)`. BT 쌍별 로그 손실을 최소화하도록 학습한다:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

수백 번의 갱신 후, `w`는 좋은 단어 토큰에 양의 가중치를, 나쁜 단어에 음의 가중치를 부여한다.

### 3단계: RM 위의 PPO류 정책

우리 장난감 정책은 어휘로부터 단일 토큰을 생성한다. RM 하에서 토큰을 채점하고, `log π_θ(token | prompt)`를 계산하고, KL-대-참조 페널티를 더하고, 클리핑된 PPO 대리 목적함수를 적용한다.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 4단계: KL 모니터링

매 갱신마다 평균 `KL(π_θ || π_ref)`를 추적한다. `~5-10`을 넘어 슬금슬금 오르면 정책이 `π_SFT`에서 멀리 표류한 것이다. `β`가 낮아 KL이 상승하거나 보상 해킹이 시작되고 있다는 뜻이다. 이것이 실제 RLHF의 최상위 진단이다.

### 5단계: TRL을 사용한 프로덕션 레시피

장난감 파이프라인을 이해했으면, 여기 실제 라이브러리 사용자가 작성하는 동일한 루프가 있다. Hugging Face의 [TRL](https://huggingface.co/docs/trl)이 참조 구현이다. 2단계에는 `RewardTrainer`, 3단계에는 (KL-대-참조가 내장된) `PPOTrainer`.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

라이브러리가 대신 처리하는 세 가지. `adap_kl_ctrl=True`는 적응적-β 스케줄을 구현한다: 관측된 KL이 `target_kl`을 초과하면 β가 두 배, 절반 이하면 β가 절반이 된다. 참조 모델은 관례상 동결된다. `policy`와 파라미터(parameter)를 실수로 공유하면 안 된다. 그리고 가치 헤드는 정책과 같은 백본에 산다(`AutoModelForCausalLMWithValueHead`가 스칼라 MLP 헤드를 붙인다). 그래서 TRL이 `policy/kl`과 `value/loss`를 별도로 보고하는 것이다.

## 함정 (Pitfalls)

- **과최적화 / 보상 해킹.** RM은 불완전하다; `π_θ`는 높은 점수를 받지만 나쁜 적대적 완성을 찾는다. 증상: 인간 평가 점수가 정체되거나 떨어지는 동안 보상이 무한정 오른다. 해법: 일찍 멈추고, `β`를 올리고, RM 학습 데이터를 넓혀라.
- **길이 해킹.** 유용한 응답에 학습된 RM은 종종 암묵적으로 길이를 보상한다. 정책은 응답을 채우는 법을 배운다. 시정: 길이 정규화 보상, 또는 길이 인식 RM을 가진 RLAIF.
- **너무 작은 RM.** RM은 적어도 정책만큼 커야 한다. 작은 RM은 정책의 출력을 충실히 채점할 수 없다.
- **KL 조정.** β가 너무 낮으면 → 표류와 보상 해킹. β가 너무 높으면 → 정책이 거의 변하지 않음. 표준 트릭은 스텝당 고정 KL을 목표로 하는 *적응적* β다.
- **선호 데이터 노이즈.** 인간 레이블의 약 30%는 노이즈가 있거나 모호하다. 합의 필터링된 데이터로 RM을 학습하거나 BT에 온도(temperature)를 사용해 보정하라.
- **오프-폴리시 문제.** PPO 데이터는 첫 에폭(epoch) 후 약간 오프-폴리시다. Lesson 08처럼 클립 비율(clip fraction)을 모니터링하라.

## 라이브러리로 써보기 (Use It)

2026년의 RLHF는 계층적이다:

| 계층 | 목표 | 방법 |
|-------|--------|--------|
| 지시 따르기, 유용성, 무해성 | 정렬 | DPO (Phase 10 · 08)가 RLHF-PPO보다 선호됨. |
| 추론 정확성 (수학, 코드) | 능력 | 검증기 보상을 가진 GRPO (Phase 9 · 12). |
| 장기 지평 다단계 작업 | 에이전트형 | 스텝에 대한 과정 보상 모델을 가진 PPO / GRPO. |
| 안전 / 거절 행동 | 안전 | 별도 안전 RM을 가진 RLHF-PPO, 또는 헌법적 AI. |
| 추론 시 Best-of-N | 빠른 정렬 | 디코드 시 RM 사용; 정책 학습 불필요. |
| 보상 증류 | 추론 계산 | 동결된 LM 위에 작은 "보상 헤드"를 학습. |

RLHF는 2022–2024년에 *그* 방법이었다. 2026년에는, 프로덕션 정렬 파이프라인이 DPO-우선이며, RM 집약적이거나 안전 중요 단계에만 PPO를 쓴다.

## 산출물 (Ship It)

`outputs/skill-rlhf-architect.md`로 저장하라:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 브래들리-테리 보상 모델을 500개의 합성 선호 쌍에 학습하라. 보류된 100쌍에 대한 쌍별 정확도를 측정하라. 90%를 초과해야 한다.
2. **중간.** `β ∈ {0.0, 0.1, 1.0}`으로 장난감 PPO-RLHF 루프를 실행하라. 각각에 대해, 갱신에 걸친 RM 점수 대 KL-대-참조를 그려라. 어느 것이 보상 해킹을 하는가?
3. **어려움.** 같은 선호 데이터에 DPO(닫힌 형식 선호-우도 손실)를 구현하고, 사용된 계산과 달성된 최종 RM 점수에서 RLHF-PPO 파이프라인과 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제로 의미하는 것 |
|------|-----------------|-----------------------|
| RLHF | "정렬 RL" | 3단계 SFT + RM + PPO 파이프라인 (Christiano 2017, Ouyang 2022). |
| 보상 모델 (RM) | "채점 망" | 브래들리-테리를 통해 쌍별 선호에 맞춰진 학습된 스칼라 함수. |
| 브래들리-테리 | "쌍별 로지스틱 손실" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; 표준 RM 목적함수. |
| KL 페널티 | "참조 근처에 머물기" | 보상 속의 `β · KL(π_θ \|\| π_ref)`; 보상 해킹 방지 정규화기. |
| 보상 해킹 | "굿하트의 법칙" | 정책이 RM 결함을 악용; 증상: 보상은 올라가고 인간 평가는 평평. |
| RLAIF | "AI 레이블 선호" | 레이블이 인간 대신 다른 LM에서 오는 RLHF. |
| PRM | "과정 보상 모델" | 부분 추론 스텝을 채점; 추론 파이프라인에서 사용. |
| 헌법적 AI | "Anthropic의 방법" | 명시적 규칙으로 안내되는 AI 생성 선호. |

## 더 읽을거리 (Further Reading)

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741): RLHF를 시작한 논문.
- [Ouyang et al. (2022). InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) ChatGPT 뒤의 레시피.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325): 요약을 위한 초기 RLHF.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290): DPO; 2026년 RLHF 이후의 기본값.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073): RLAIF와 자기 비판 루프.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862): HH 논문.
- [Hugging Face TRL library](https://huggingface.co/docs/trl): 프로덕션 `RewardTrainer`와 `PPOTrainer`. 적응적-KL과 가치 헤드 세부사항은 트레이너 소스를 읽어라.
- [Hugging Face(Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla) 다이어그램과 함께하는 3단계 파이프라인의 정전적 안내.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl): 그 라이브러리; `examples/`에 Llama, Mistral, Qwen을 위한 종단간(end-to-end) RLHF 스크립트가 있다.
- [Sutton & Barto (2018). Ch. 17.4: Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) 보상 가설 관점; 보상 해킹을 생각하기 위한 필수 전제조건.
