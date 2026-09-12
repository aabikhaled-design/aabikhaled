# GPT: 인과적 언어 모델링(Causal Language Modeling)

> BERT는 양쪽을 본다. GPT는 과거만 본다. 삼각형 마스크(triangle mask)는 현대 AI에서 가장 중대한 단 한 줄의 코드다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75분

## 문제 (The Problem)

언어 모델은 하나의 질문에 답한다. 처음 `t-1`개 토큰(token)이 주어졌을 때, 토큰 `t`에 대한 확률 분포(probability distribution)는 무엇인가? 그 신호(다음 토큰 예측(next-token prediction))로 학습(training)하면, 한 번에 한 토큰씩 임의의 텍스트를 생성할 수 있는 모델을 얻는다.

전체 시퀀스에 대해 병렬로 엔드-투-엔드 학습하려면, 각 위치의 예측이 이전 위치에만 의존해야 한다. 그렇지 않으면 모델이 답을 들여다보며 손쉽게 부정행위를 한다.

인과 마스크(causal mask)가 이를 한다. 소프트맥스(softmax) 전에 어텐션 점수에 더해지는 단 하나의 상삼각(upper-triangular) `-inf` 값 행렬이다. 소프트맥스 후 그 위치들은 0이 된다. 각 위치는 자신과 이전 위치에만 어텐션(attention)할 수 있다. 그리고 전체 시퀀스에 한 번 적용하므로, 한 번의 순방향 패스(forward pass)로 N개의 병렬 다음 토큰 예측을 얻는다.

GPT-1(2018), GPT-2(2019), GPT-3(2020), GPT-4(2023), GPT-5(2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi: 이들은 모두 같은 핵심 루프를 가진 디코더(decoder) 전용 인과 트랜스포머(transformer)다. 그저 더 크고, 더 나은 데이터에, 더 나은 RLHF를 곁들였을 뿐이다.

## 개념 (The Concept)

![인과 마스크가 삼각 어텐션 행렬을 만든다](../assets/causal-attention.svg)

### 마스크

길이 `N`의 시퀀스가 주어지면, `N × N` 행렬을 만든다.

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

소프트맥스 전에 원시 어텐션 점수에 `M`을 더한다. `exp(-inf) = 0`이므로 마스킹된 위치는 가중치 0을 기여한다. 어텐션 행렬의 각 행은 이전 위치들에 대해서만의 확률 분포다.

구현 비용: `torch.tril()` 호출 한 번. 계산 시간: 나노초. 분야에 미친 영향: 모든 것.

### 삼각형은 어디에서 왔나

마스크는 보통 어텐션에 덧대어 붙인 땜질처럼 소개된다. 유도를 반대 방향으로 돌려 보면 신비로울 것이 없어진다. 어텐션은 앞자락 평균을 세 번째로 다듬은 것이고, 삼각형은 그 평균의 반복 범위를 행렬로 적어 놓은 것이다.

**1단계: 앞자락 평균.** 시퀀스를 인과적으로 요약하는 가장 멍청한 방법은 이렇다. 위치 `i`를 위치 `0…i`의 평균으로 삼는다. 반복문으로 쓰면 `out[i] = X[:i+1].mean(0)`이다. 같은 계산을 행렬 곱 한 번으로 할 수 있다. 1로 채운 하삼각 행렬을 가져다 각 행을 그 개수로 나눈 다음 곱한다.

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

`A`의 `i`번째 행은 `[1/(i+1), …, 1/(i+1), 0, …, 0]`이다. 대각선 위의 0들이 곧 인과성이다. 미래를 가려 낸 것이 아니라, 미래가 애초에 합에 들어간 적이 없다.

**2단계: 학습된 가중치.** 고른 평균은 과거의 모든 토큰을 똑같이 중요하게 취급한다. 그 1들을 학습된 점수 행렬 `S`로 바꾼다. 이제 행의 합이 저절로 1이 되지는 않으므로, 개수로 나누는 대신 각 행에 소프트맥스를 씌워 정규화한다. 소프트맥스는 정확한 0을 절대 내놓지 않아 인과성을 깨뜨린다. 미래 점수를 `-inf`로 넣지 않는 한 그렇다. `exp(-inf) = 0`이기 때문이다.

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

같은 삼각형, 같은 행 확률 행렬, 같은 행렬 곱 한 번이다. `-inf` 마스크는 새로운 장치가 아니다. 1단계의 0 항목을 소프트맥스의 입력 영역으로 옮겨 적은 것이다.

**3단계: 내용에 따라 달라지는 가중치.** 2단계에서 `S`는 학습이 끝나면 고정된다. 토큰이 무엇이든 위치 7은 언제나 위치 3을 같은 무게로 본다. 이제 점수가 토큰 자체에 따라 달라지게 하자. `S = Q @ K.T / sqrt(d_k)`다. 나머지는 하나도 바뀌지 않는다. 마스크, 소프트맥스, 행렬 곱이 똑같다.

세 단계를 관통하는 불변식 하나. 하삼각 행 확률 행렬에 시퀀스를 곱하는 것이다. 고른 평균, 학습된 고정 가중치, 내용에 따라 달라지는 가중치로 이어진다. 마스크는 어텐션에 더해진 적이 없다. 평균에서 살아남은 것이다.

```figure
mask-derivation
```

### 병렬 학습, 직렬 추론

학습: 전체 `(N, d_model)` 시퀀스를 한 번 순방향 패스하고, N개의 교차 엔트로피(cross-entropy) 손실(위치당 하나)을 계산하고, 합하고, 역전파한다. 시퀀스를 따라 병렬이다. 이것이 GPT 학습이 확장되는 이유다. 한 GPU 패스에서 배치(batch)당 100만 토큰을 처리한다.

추론(inference): 토큰을 하나씩 생성한다. `[t1, t2, t3]`을 넣어 `t4`를 얻는다. `[t1, t2, t3, t4]`를 넣어 `t5`를 얻는다. `[t1, t2, t3, t4, t5]`를 넣어 `t6`을 얻는다. KV 캐시(cache)(레슨 12)는 `t1…tn`의 은닉 상태를 저장해 매 스텝 재계산하지 않게 한다. 하지만 추론 시 직렬 깊이 = 출력 길이다. 그것이 자기회귀(autoregressive) 세금이며, 디코딩이 모든 LLM의 지연 시간(latency) 병목인 이유다.

### 손실: 한 칸 이동

토큰 `[t1, t2, t3, t4]`가 주어지면:

- 입력: `[t1, t2, t3]`
- 타깃: `[t2, t3, t4]`

모든 위치 `i`에 대해 `-log P(target_i | inputs[:i+1])`을 계산한다. 합한다. 이것이 전체 시퀀스에 대한 교차 엔트로피다.

들어본 모든 트랜스포머 LM이 이 손실로 학습한다. 사전 학습(pretraining), 파인튜닝(fine-tuning), SFT: 같은 손실, 다른 데이터.

### 디코딩 전략

학습 후, 샘플링(sampling) 선택은 사람들이 생각하는 것보다 더 중요하다.

| 방법 | 하는 일 | 언제 쓰는가 |
|--------|--------------|-------------|
| Greedy | 매 스텝 argmax | 결정론적 과제, 코드 완성 |
| Temperature | 로짓을 T로 나누고 샘플 | 창의적 과제, 높은 T = 더 다양 |
| Top-k | 상위 k개 토큰에서만 샘플 | 저확률 꼬리를 죽임 |
| Top-p (nucleus) | 누적 확률 ≥ p인 가장 작은 집합에서 샘플 | 2020+ 기본값. 분포 모양에 적응 |
| Min-p | `p > min_p * max_p`인 토큰 유지 | 2024+. top-p보다 긴 꼬리 거부에 더 나음 |
| Speculative decoding | 드래프트 모델이 N개 토큰 제안, 큰 모델이 검증 | 같은 품질에서 2-3배 지연 시간 감소 |

2026년에 min-p + 온도 0.7이 오픈 웨이트 모델의 합리적 기본값이다. 추측 디코딩(speculative decoding)은 모든 프로덕션(production) 추론 스택의 기본 요구사항이다.

### "GPT 레시피"를 동작하게 만든 것

1. **디코더 전용.** 인코더 오버헤드 없음. 층당 어텐션 + FFN 한 패스.
2. **스케일링.** 124M → 1.5B → 175B → 조 단위. Chinchilla 스케일링 법칙(레슨 13)이 계산을 어떻게 쓸지 알려준다.
3. **맥락 내 학습(In-context learning).** 약 6B-13B에서 출현했다. 모델이 파인튜닝 없이 퓨샷(few-shot) 예시를 따를 수 있다.
4. **RLHF.** 인간 선호에 대한 사후 학습(post-training)이 원시 사전 학습 텍스트를 챗 비서로 바꿨다.
5. **Pre-norm + RoPE + SwiGLU.** 대규모에서의 안정적 학습.

핵심 아키텍처는 GPT-2 이후 크게 바뀌지 않았다. 흥미로운 모든 것은 데이터, 규모, 사후 학습에서 일어났다.

```figure
causal-mask
```

## 직접 만들기 (Build It)

### 1단계: 인과 마스크

`code/main.py`를 참조하라. 한 줄짜리:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

소프트맥스 전에 어텐션 점수에 더한다. 그것이 메커니즘 전부다.

### 2단계: 2층 GPT 비슷한 모델

두 개의 디코더 블록(마스킹된 셀프 어텐션 + FFN, 크로스 어텐션 없음)을 쌓는다. 토큰 임베딩(embedding), 위치 인코딩(positional encoding), 언임베딩(unembedding)(토큰 임베딩 행렬에 묶임: GPT-2 이후의 표준 트릭)을 더한다.

### 3단계: 다음 토큰 예측, 엔드-투-엔드

20 토큰 장난감 어휘에서, 모든 위치에 로짓(logit)을 생성한다. 한 칸 이동 타깃 대비 교차 엔트로피 손실을 계산한다. 그래디언트(gradient) 없음: 이것은 순방향 패스 온전성 검사다.

### 4단계: 샘플링

greedy, temperature, top-k, top-p, min-p를 구현한다. 고정된 프롬프트(prompt)에 각각을 실행하고 출력을 비교한다. 샘플링 함수는 10줄이다.

## 라이브러리로 써보기 (Use It)

PyTorch, 2026년 관용구:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

내부적으로 `generate()`는 순방향 패스를 실행하고, 마지막 위치 로짓을 뽑고, 다음 토큰을 샘플링하고, 덧붙이고, 반복한다. 모든 프로덕션 LLM 추론 스택(vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX)이 같은 루프에 무거운 최적화를 더해 구현한다. 배치 프리필(prefill), 연속 배칭(continuous batching), KV 캐시 페이징, 추측 디코딩.

**GPT vs BERT, 각각 한 줄:** GPT는 `P(x_t | x_{<t})`를 예측한다. BERT는 `P(x_masked | x_unmasked)`를 예측한다. 손실이 모델이 생성할 수 있는지를 결정한다.

## 산출물 (Ship It)

`outputs/skill-sampling-tuner.md`를 참조하라. 이 스킬은 새 생성 과제에 맞는 샘플링 파라미터(parameter)를 고르고 결정론적 디코딩이 필요한 때를 표시한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하고 소프트맥스 후 인과 어텐션 행렬이 하삼각인지 검증한다. 표본 점검: 행 3은 열 0-3에만 가중치를 가져야 한다.
2. **보통.** 폭 4의 빔 서치(beam search)를 구현한다. 10개 짧은 프롬프트에서 빔-4 대 greedy의 퍼플렉서티(perplexity)를 비교한다. 빔이 항상 이기는가? (힌트: 보통 번역에서는 그렇지만, 개방형 챗에서는 아니다.)
3. **어려움.** 추측 디코딩을 구현한다: 작은 2층 모델을 드래프트로, 6층 모델을 검증자로 쓴다. 길이 64의 완성 100개에서 실제 시계 속도 향상을 측정한다. 출력이 검증자의 greedy와 일치하는지 확인한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Causal mask | "삼각형" | 위치 `i`가 위치 `≤ i`만 보도록 어텐션 점수에 더해지는 상삼각 `-inf` 행렬. |
| Next-token prediction | "그 손실" | 모든 위치에서 참 다음 토큰 대비 모델 분포의 교차 엔트로피. |
| Autoregressive | "한 번에 하나 생성" | 출력을 입력으로 되먹임. 병렬성은 학습 중에만, 생성 중에는 아님. |
| Logits | "소프트맥스 전 점수" | 소프트맥스 전 LM 헤드의 원시 출력. 샘플링이 이 위에서 일어남. |
| Temperature | "창의성 손잡이" | 로짓을 T로 나눔. T→0 = greedy, T→∞ = 균등. |
| Top-p | "뉴클리어스 샘플링" | 합이 ≥p인 가장 작은 집합으로 분포를 자름. 남은 것에서 샘플. |
| Min-p | "top-p보다 나음" | `p ≥ min_p × max_p`인 토큰 유지. 분포의 날카로움에 컷오프를 적응. |
| Speculative decoding | "드래프트 + 검증" | 저렴한 모델이 N개 토큰 제안. 큰 모델이 병렬로 검증. |
| Teacher forcing | "학습 트릭" | 학습 중 모델 예측이 아니라 참 이전 토큰을 넣음. 모든 seq2seq LM의 표준. |

## 더 읽을거리 (Further Reading)

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): GPT-3와 맥락 내 학습.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192): 추측 디코딩 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py): 정전적 인과 LM 레퍼런스 코드.
