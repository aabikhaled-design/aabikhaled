# 밑바닥부터 만드는 셀프 어텐션(Self-Attention from Scratch)

> 어텐션(attention)은 모든 단어가 "나에게 누가 중요한가?"를 묻고, 그 답을 학습하는 룩업 테이블이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 Lesson 10 (Sequence-to-Sequence)
**Time:** ~90분

## 학습 목표 (Learning Objectives)

- 쿼리/키/값 투영(projection)과 소프트맥스(softmax) 가중합을 포함하여, NumPy만으로 스케일드 닷-프로덕트 셀프 어텐션(scaled dot-product self-attention)을 밑바닥부터 구현하기
- 헤드를 분할하고, 병렬 어텐션을 계산하고, 결과를 이어 붙이는 멀티헤드 어텐션(multi-head attention) 층(layer) 만들기
- 어텐션 행렬(matrix)이 토큰(token) 관계를 어떻게 포착하는지 추적하고, sqrt(d_k)로 스케일링하는 것이 왜 소프트맥스 포화(saturation)를 막는지 설명하기
- 인과 마스킹(causal masking)을 적용해 양방향 어텐션을 자기회귀(autoregressive)(디코더 스타일) 어텐션으로 변환하기

## 문제 (The Problem)

RNN은 시퀀스를 한 번에 한 토큰씩 처리한다. 토큰 50에 도달할 때쯤이면, 토큰 1의 정보는 50번의 압축 단계를 거쳐 짜내어진다. 장거리 의존성은 고정 크기 은닉 상태(hidden state)로 짓눌린다. 아무리 LSTM 게이팅을 해도 완전히 풀지 못하는 병목(bottleneck)이다.

2014년 Bahdanau 어텐션 논문이 해법을 보였다. 디코더가 모든 인코더 위치를 되돌아보고 현재 스텝에 어느 것이 중요한지 결정하게 하라. 하지만 어텐션은 여전히 RNN에 볼트로 붙어 있었다. 2017년 "Attention Is All You Need" 논문은 더 날카로운 질문을 던졌다. 어텐션이 *유일한* 메커니즘이라면 어떨까? 순환(recurrence)도 없고, 합성곱(convolution)도 없이. 오직 어텐션만.

셀프 어텐션은 시퀀스의 모든 위치가 단일 병렬 스텝에서 다른 모든 위치에 어텐션하게 한다. 그래서 트랜스포머(transformer)는 빠르고 확장 가능하며 지배적이다.

## 개념 (The Concept)

### 데이터베이스 룩업 비유

어텐션을 부드러운 데이터베이스 룩업으로 생각하라.

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

모든 토큰은 세 개의 벡터(vector)를 생성한다.
- **쿼리(Query, Q)**: "나는 무엇을 찾고 있는가?"
- **키(Key, K)**: "나는 무엇을 담고 있는가?"
- **값(Value, V)**: "내가 선택되면 어떤 정보를 제공하는가?"

쿼리와 모든 키 사이의 내적(dot product)이 어텐션 점수를 만든다. 높은 점수는 "이 키가 내 쿼리와 일치한다"는 뜻이다. 그 점수들이 값에 가중치를 준다. 출력은 값들의 가중합이다.

### Q, K, V 계산

각 토큰 임베딩(embedding)은 세 개의 학습된 가중치(weight) 행렬을 통해 투영된다.

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

한 토큰에 대해 시각적으로:

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### 어텐션 행렬

모든 토큰에 대한 Q, K, V를 갖추고 나면, 어텐션 점수가 행렬을 이룬다.

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

쿼리 하나가 키들을 훑어 가는 과정을 눈으로 따라가 보라. 각 행은 모든 토큰에 점수를 매기고, 소프트맥스가 그 점수를 가중치로 바꾸고, 문맥 벡터는 값들을 그 가중치로 섞은 결과다.

```figure
attention-matrix
```

### 왜 스케일링하는가?

내적은 차원 dk와 함께 커진다. dk = 64이면, 내적은 수십 단위 범위에 들 수 있고, 소프트맥스를 그래디언트(gradient)가 소실되는 영역으로 밀어 넣는다. 해법: sqrt(dk)로 나눈다.

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

이렇게 하면 점수가 소프트맥스에서 유용한 그래디언트가 나오는 범위에 머문다.

### 소프트맥스가 점수를 가중치로 바꾼다

소프트맥스는 원시 점수를 각 행에 걸친 확률 분포(probability distribution)로 변환한다.

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

이제 각 토큰은 다른 모든 토큰에 얼마나 어텐션할지를 말하는 가중치 집합을 가진다.

### 값들의 가중합

각 토큰의 최종 출력은 모든 값 벡터의 가중합이다.

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 전체 파이프라인

```mermaid
flowchart LR
  X["X (input)"] --> Q["Q = X · Wq"]
  X --> K["K = X · Wk"]
  X --> V["V = X · Wv"]
  Q --> S["Q · Kᵀ / √dk"]
  K --> S
  S --> SM["softmax"]
  SM --> WS["weighted sum"]
  V --> WS
  WS --> O["output"]
```

한 줄로 된 공식:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

```figure
softmax-attention-scaling
```

## 직접 만들기 (Build It)

### 1단계: 밑바닥부터 만드는 소프트맥스

소프트맥스는 원시 로짓(logit)을 확률로 변환한다. 수치 안정성을 위해 최댓값을 뺀다.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 2단계: 스케일드 닷-프로덕트 어텐션

핵심 함수다. Q, K, V 행렬을 받아 어텐션 출력과 가중치 행렬을 반환한다.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 3단계: 학습된 투영을 갖춘 셀프 어텐션 클래스

Xavier 유사 스케일링으로 초기화된 Wq, Wk, Wv 가중치 행렬을 갖춘 완전한 셀프 어텐션 모듈.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 4단계: 문장에 대해 실행하기

문장에 대한 가짜 임베딩을 만들고 어텐션 가중치를 살펴본다.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 5단계: ASCII 히트맵으로 어텐션 시각화하기

빠른 시각화를 위해 어텐션 가중치를 문자로 매핑한다.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 라이브러리로 써보기 (Use It)

PyTorch의 `nn.MultiheadAttention`은 우리가 만든 것을 정확히 수행하고, 추가로 멀티헤드 분할과 출력 투영을 한다.

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

핵심 차이: 멀티헤드 어텐션은 여러 어텐션 함수를 병렬로 실행하며, 각각 크기 dk = d_model / n_heads의 자체 Q, K, V 투영을 가진 뒤, 결과를 이어 붙인다. 덕분에 모델은 서로 다른 관계 유형에 동시에 어텐션한다.

## 산출물 (Ship It)

이 레슨은 다음을 생성한다.
- `outputs/prompt-attention-explainer.md` - 데이터베이스 룩업 비유를 통해 어텐션을 설명하기 위한 프롬프트(prompt)

## 연습 문제 (Exercises)

1. `scaled_dot_product_attention`을 수정해, 소프트맥스 전에 특정 위치를 음의 무한대로 설정하는 선택적 마스크 행렬을 받게 한다(이것이 인과/디코더 마스킹이 동작하는 방식이다)
2. 멀티헤드 어텐션을 밑바닥부터 구현한다: Q, K, V를 `n_heads`개 청크로 나누고, 각각에 어텐션을 실행하고, 이어 붙이고, 최종 가중치 행렬 Wo를 통해 투영한다
3. 길이가 같은 서로 다른 두 문장을 같은 SelfAttention 인스턴스에 넣고, 어텐션 패턴을 비교한다. 무엇이 바뀌는가? 무엇이 그대로인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Query (Q) | "질문 벡터" | 이 토큰이 어떤 정보를 찾는지 나타내는, 입력의 학습된 투영 |
| Key (K) | "라벨 벡터" | 이 토큰이 어떤 정보를 담는지 나타내며 쿼리와 매칭되는 학습된 투영 |
| Value (V) | "내용 벡터" | 어텐션 점수에 따라 집계되는 실제 정보를 담는 학습된 투영 |
| Scaled dot-product attention | "어텐션 공식" | softmax(QK^T / sqrt(dk)) @ V - 스케일링이 고차원에서 소프트맥스 포화를 막음 |
| Self-attention | "토큰이 자신과 남을 봄" | Q, K, V가 모두 같은 시퀀스에서 나와, 모든 위치가 다른 모든 위치에 어텐션하게 하는 어텐션 |
| Attention weights | "얼마나 집중하는가" | 스케일드 내적에 대한 소프트맥스로 만들어진, 위치에 걸친 확률 분포 |
| Multi-head attention | "병렬 어텐션" | 서로 다른 투영으로 여러 어텐션 함수를 실행한 뒤, 더 풍부한 표현을 위해 결과를 이어 붙임 |

## 더 읽을거리 (Further Reading)

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - 원조 트랜스포머 논문
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 전체 아키텍처에 대한 최고의 시각적 안내
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 설명이 곁들여진 한 줄 한 줄 PyTorch 구현
