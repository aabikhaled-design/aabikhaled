# 멀티헤드 어텐션(Multi-Head Attention)

> 하나의 어텐션(attention) 헤드는 한 번에 하나의 관계를 학습한다. 여덟 개의 헤드는 여덟 개를 학습한다. 헤드는 공짜다. 더 많이 가져라.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75분

## 문제 (The Problem)

단일 셀프 어텐션(self-attention) 헤드는 하나의 어텐션 행렬(matrix)을 계산한다. 그 행렬은 한 종류의 관계를 포착한다. 보통 주어진 학습 신호의 손실(loss)을 최소화하는 관계다. 데이터에 주어-동사 일치, 상호 참조(co-reference), 장거리 담화, 구문 청킹이 모두 뒤엉켜 있다면, 단일 헤드는 그것들을 하나의 소프트맥스(soft-max) 분포로 뭉개버리고 신호의 절반을 잃는다.

2017년 Vaswani 논문의 해법: 여러 어텐션 함수를 병렬로, 각각 자체 Q, K, V 투영(projection)을 가지고 실행한 뒤 출력을 이어 붙인다. 각 헤드는 차원 `d_model / n_heads`의 더 작은 부분 공간에서 동작한다. 전체 파라미터(parameter)는 그대로다. 표현력은 올라간다.

멀티헤드 어텐션은 2026년의 모든 트랜스포머(transformer)가 기본으로 탑재한다. 유일한 논쟁은 헤드를 *몇 개* 둘지, 그리고 키와 값이 투영을 공유하는지(Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention)에 관한 것뿐이다.

## 개념 (The Concept)

![멀티헤드 어텐션이 분할하고, 어텐션하고, 이어 붙인다](../assets/multi-head-attention.svg)

**분할(Split).** `(N, d_model)` 모양의 `X`를 취한다. 각각 `(N, d_model)` 모양인 Q, K, V로 투영한다. `d_head = d_model / n_heads`인 `(N, n_heads, d_head)`로 리셰이프한다. `(n_heads, N, d_head)`로 전치한다.

**병렬 어텐션(Attend in parallel).** 각 헤드 안에서 스케일드 닷-프로덕트 어텐션(scaled dot-product attention)을 실행한다. 각 헤드는 `(N, d_head)`를 만든다. 헤드들은 임베딩(embedding)의 서로 다른 부분 공간에서 동작하며 어텐션 계산 자체 동안에는 결코 대화하지 않는다.

**이어 붙이고 투영(Concatenate and project).** 헤드들을 다시 `(N, d_model)`으로 쌓고 `(d_model, d_model)` 모양의 학습된 출력 행렬 `W_o`를 곱한다. `W_o`가 헤드들이 섞이는 곳이다.

**왜 동작하는가.** 각 헤드는 표현 예산을 두고 다른 헤드와 경쟁하지 않고 전문화할 수 있다. 2019-2024년의 프로빙(probing) 연구는 뚜렷한 헤드 역할을 보여준다. 위치 헤드, 이전 토큰에 어텐션하는 헤드, 복사 헤드, 개체명 헤드, 인덕션 헤드(induction head)(맥락 내 학습(in-context learning)의 기반).

**2026년의 변형 계보:**

| 변형 | Q 헤드 | K/V 헤드 | 사용처 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA는 거의 완전한 품질을 유지하면서 KV 캐시(cache) 메모리를 `N/G`배 줄이기 때문에 현대의 기본값이다. MLA는 한 걸음 더 나아가 K/V를 잠재(latent) 공간으로 압축한 뒤 계산 시점에 되투영한다. FLOPs를 들이고, 훨씬 더 많은 메모리를 아낀다.

```figure
multihead-split
```

## 직접 만들기 (Build It)

### 1단계: 이미 가진 단일 헤드 어텐션에서 헤드 분할하기

레슨 02의 `SelfAttention`을 가져와 분할/연결 한 쌍으로 감싼다. numpy 구현은 `code/main.py`를 참조하라. 로직은 다음과 같다.

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

리셰이프 하나와 전치 하나. 루프 없음. 이것이 `nn.MultiheadAttention` 아래에서 PyTorch가 하는 일 그대로다.

### 2단계: 헤드별로 스케일드 닷-프로덕트 어텐션 실행하기

각 헤드는 Q, K, V의 자체 조각을 받는다. 어텐션은 배치 행렬곱(batched matmul)이 된다.

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

실제 하드웨어에서 `Qh @ Kh.transpose(...)`는 하나의 `bmm`이다. GPU는 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 모양의 단일 배치 행렬곱을 본다. 헤드를 추가하는 것은 공짜다.

### 3단계: Grouped-Query Attention 변형

키와 값 투영만 바뀐다. Q는 `n_heads`개 그룹을 받는다. K와 V는 `n_kv_heads < n_heads`개 그룹을 받고 맞추기 위해 반복된다.

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

추론(inference) 시 KV 캐시에 `n_heads`개가 아닌 `n_kv_heads`개 사본만 살아 있으므로 메모리를 아낀다. Llama 3 70B는 64개 쿼리 헤드와 8개 KV 헤드를 쓴다. 캐시가 8배 축소된다.

### 4단계: 각 헤드가 학습한 것을 탐침하기

4개 헤드로 짧은 문장에 MHA를 실행한다. 각 헤드에 대해 `(N, N)` 어텐션 행렬을 출력한다. 무작위 초기화로도 서로 다른 헤드가 서로 다른 구조를 골라내는 것을 볼 수 있다. 일부는 신호, 일부는 부분 공간의 회전 대칭성(symmetry) 때문이다.

## 라이브러리로 써보기 (Use It)

PyTorch에서 한 줄 버전:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 기준 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**헤드를 몇 개?** 2026년 프로덕션(production) 모델의 경험칙:

| 모델 크기 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`는 거의 항상 64나 128로 떨어진다. 그것은 한 헤드가 얼마나 "볼" 수 있는지의 단위다. 32 아래로 내려가면 헤드들이 스케일링 인자 `sqrt(d_head)`와 싸우기 시작한다. 256 위로 올라가면 "많은 작은 전문가" 이점을 잃는다.

## 산출물 (Ship It)

`outputs/skill-mha-configurator.md`를 참조하라. 이 스킬은 파라미터 예산, 시퀀스 길이, 배포 대상이 주어지면 새 트랜스포머에 맞는 헤드 수, kv 헤드 수, 투영 전략을 추천한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 MHA를 가져와 `d_model=64`를 고정한 채 `n_heads`를 1에서 16으로 바꾼다. 합성 복사 과제에 대한 작은 1층 모델의 손실을 플롯한다. 헤드가 많을수록 도움이 되는가, 정체되는가, 해가 되는가?
2. **보통.** MQA(모든 쿼리 헤드에 걸쳐 공유되는 하나의 KV 헤드)를 구현한다. 완전한 MHA 대비 파라미터 수가 얼마나 줄어드는지 측정한다. N=2048에서 추론 시 KV 캐시 크기가 얼마나 줄어드는지 계산한다.
3. **어려움.** Multi-head Latent Attention의 작은 버전을 구현한다: K, V를 랭크 `r` 잠재로 압축하고, 잠재를 KV 캐시에 저장하고, 어텐션 시점에 압축 해제한다. 품질이 검증 ppl의 1비트 이내를 유지하면서 캐시 메모리가 완전한 MHA의 1/8 아래로 떨어지는 `r`은 얼마인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Head | "단일 어텐션 회로" | 자체 어텐션 행렬을 가진 차원 `d_head = d_model / n_heads`의 Q/K/V 투영 하나. |
| d_head | "헤드 차원" | 헤드당 은닉 폭. 프로덕션에서 거의 항상 64나 128. |
| Split / combine | "리셰이프 트릭" | 어텐션 전후의 `(N, d_model) ↔ (n_heads, N, d_head)` 리셰이프+전치. |
| W_o | "출력 투영" | 헤드를 이어 붙인 뒤 적용하는 `(d_model, d_model)` 행렬. 헤드가 섞이는 곳. |
| MQA | "하나의 KV 헤드" | Multi-Query Attention: 단일 공유 K/V 투영. 가장 작은 KV 캐시, 약간의 품질 손실. |
| GQA | "Llama 2 이후의 기본값" | `n_kv_heads < n_heads`인 Grouped-Query Attention. Q에 맞추려 반복함. |
| MLA | "DeepSeek의 트릭" | Multi-head Latent Attention: K, V를 저랭크 잠재로 압축, 어텐션 시점에 압축 해제. |
| Induction head | "맥락 내 학습 뒤의 회로" | 이전 출현을 감지하고 그 뒤에 온 것을 복사하는 헤드 한 쌍. |

## 더 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762): 원조 멀티헤드 사양.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150): MQA 논문.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): 학습 후 MHA를 GQA로 변환하는 법.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434): MLA와 그것이 캐시 메모리에서 MHA/GQA를 이기는 이유.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): 헤드가 실제로 하는 일에 대한 기계론적 관점.
