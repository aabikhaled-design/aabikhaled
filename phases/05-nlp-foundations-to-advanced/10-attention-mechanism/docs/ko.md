# 어텐션 메커니즘(돌파구 (Attention Mechanism) The Breakthrough)

> 디코더는 압축된 요약을 곁눈질하기를 멈추고 원문 전체를 보기 시작한다. 이 이후의 모든 것은 어텐션에 공학을 더한 것이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45분

## 문제 (The Problem)

레슨 09는 절제된 실패로 끝났다. 장난감 복사 과제로 학습된 GRU 인코더-디코더는 길이 5에서 89% 정확도(accuracy)였다가 길이 80에서 우연 수준에 가까워진다. 그 이유는 학습 버그가 아니라 구조적이다. 인코더(encoder)가 그러모은 모든 정보가 하나의 고정 크기 은닉 상태(hidden state)에 들어가야 하고, 디코더(decoder)는 그 밖의 어떤 것도 보지 못한다.

Bahdanau, Cho, Bengio는 2014년에 세 줄짜리 해결책을 발표했다. 디코더에 마지막 인코더 상태만 주는 대신, 모든 인코더 상태를 보관한다. 각 디코더 스텝에서 인코더 상태의 가중 평균을 계산하는데, 그 가중치는 "지금 디코더가 인코더 위치 `i`를 얼마나 봐야 하는가?"를 말한다. 그 가중 평균이 맥락(context)이고, 디코더 스텝마다 바뀐다.

발상은 그게 전부다. 트랜스포머(transformer)는 이를 확장했다. 셀프 어텐션(self-attention)은 이를 단일 시퀀스에 적용했다. 멀티헤드 어텐션(multi-head attention)은 이를 병렬로 돌렸다. 하지만 2014년 버전이 이미 병목을 깼고, 일단 이것만 손에 넣으면 트랜스포머로의 전환은 개념보다 공학에 가깝다.

## 개념 (The Concept)

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

각 디코더 스텝 `t`에서:

1. 이전 디코더 은닉 상태 `s_{t-1}`을 **쿼리(query)**로 사용한다.
2. 이를 모든 인코더 은닉 상태 `h_1, ..., h_T`에 대해 점수화한다. 인코더 위치당 스칼라 하나.
3. 점수를 소프트맥스(softmax)하여 합이 1이 되는 어텐션 가중치 `α_{t,1}, ..., α_{t,T}`를 얻는다.
4. 맥락 벡터 `c_t = Σ α_{t,i} * h_i`. 인코더 상태의 가중 평균.
5. 디코더가 `c_t`와 이전 출력 토큰(token)을 받아 다음 토큰을 만든다.

가중 평균이 핵심이다. 디코더가 "Je"를 "I"로 번역해야 할 때, "Je"에 해당하는 인코더 상태에 높은 가중치를, 나머지에 낮은 가중치를 둔다. "not"이 필요할 때는 "pas"에 높은 가중치를 둔다. 맥락 벡터는 매 스텝마다 모양을 다시 잡는다.

## 형태 (모두를 무는 그것)

여기가 모든 어텐션 구현이 처음에 잘못되는 곳이다. 천천히 읽어라.

| 대상 | 형태 | 비고 |
|-------|-------|-------|
| 인코더 은닉 상태 `H` | `(T_enc, d_h)` | BiLSTM이면 `d_h = 2 * d_hidden` |
| 디코더 은닉 상태 `s_{t-1}` | `(d_s,)` | 벡터 하나 |
| 어텐션 점수 `e_{t,i}` | 스칼라 | 인코더 위치당 하나 |
| 어텐션 가중치 `α_{t,i}` | 스칼라 | 모든 `i`에 대한 소프트맥스 후 |
| 맥락 벡터 `c_t` | `(d_h,)` | 인코더 상태와 같은 형태 |

**Bahdanau (가산적) 점수.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}`은 형태 `(d_s,)`, `h_i`는 형태 `(d_h,)`다.
- `W_a`는 형태 `(d_attn, d_s)`. `U_a`는 형태 `(d_attn, d_h)`다.
- tanh 안에서 둘의 합은 형태 `(d_attn,)`다.
- `v_α`는 형태 `(d_attn,)`다. `v_α`와의 내적이 스칼라로 무너진다. **`v_α`가 하는 일이 바로 이것이다.** 마법이 아니다. 어텐션 차원 벡터를 스칼라 점수로 바꾸는 투영(projection)이다.

**Luong (곱셈적) 점수.** 세 가지 변형:

- `dot`: `e_{t,i} = s_t^T * h_i`. `d_s == d_h`를 요구한다. 강한 제약. 인코더가 양방향이면 건너뛴다.
- `general`: `e_{t,i} = s_t^T * W * h_i`, `W`는 형태 `(d_s, d_h)`. 동일 차원 제약을 없앤다.
- `concat`: 본질적으로 Bahdanau 형태. 앞의 둘이 더 저렴해서 드물게 쓰인다.

**짚어 둘 만한 Bahdanau / Luong 함정.** Bahdanau는 `s_{t-1}`(현재 단어를 생성하기 *전*의 디코더 상태)을 쓴다. Luong은 `s_t`(*후*의 상태)를 쓴다. 둘을 헷갈리면 디버깅하기 극도로 어려운, 미묘하게 잘못된 그래디언트(gradient)가 나온다. 한 논문을 골라 그 관례를 고수하라.

```figure
attention-heatmap
```

## 직접 만들기 (Build It)

### 1단계: 가산적 (Bahdanau) 어텐션

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

위 표에 대조해 형태를 점검하라. `encoder_states`는 형태 `(T_enc, d_h)`. `projected_enc`는 형태 `(T_enc, d_attn)`. `projected_dec`는 형태 `(d_attn,)`이고 브로드캐스트된다. `combined`는 형태 `(T_enc, d_attn)`. `scores`는 형태 `(T_enc,)`. `weights`는 형태 `(T_enc,)`. `context`는 형태 `(d_h,)`. 출하하라.

### 2단계: Luong dot과 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

각각 세 줄. 이것이 Luong의 논문이 안착한 이유다. 대부분의 과제에서 같은 정확도, 훨씬 적은 코드.

### 3단계: 풀어 본 수치 예제

세 인코더 상태(대략 "cat", "sat", "mat")와 첫 번째와 가장 잘 정렬되는 디코더 상태가 주어지면, 어텐션 분포가 위치 0에 집중된다. 디코더 상태를 마지막과 정렬되도록 옮기면, 어텐션이 위치 2로 이동한다. 맥락 벡터가 그것을 따라간다.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

첫 행이 이긴다. 그다음 디코더 상태를 세 번째 인코더 상태에 더 가깝게 옮기고 가중치가 이동하는 것을 지켜보라. 그게 전부다. 어텐션은 명시적 정렬(alignment)이다.

### 4단계: 이것이 트랜스포머로의 다리인 이유

위의 언어를 Q/K/V로 번역하라:

- **쿼리(Query)** = 디코더 상태 `s_{t-1}`
- **키(Key)** = 인코더 상태(우리가 점수화하는 대상)
- **밸류(Value)** = 인코더 상태(우리가 가중하고 합하는 대상)

고전 어텐션에서는 키와 밸류가 같다. 셀프 어텐션은 이들을 분리한다. 시퀀스를 자기 자신에 대해 쿼리할 수 있고, K와 V에 서로 다른 학습된 투영을 쓴다. 멀티헤드 어텐션은 서로 다른 학습된 투영으로 이를 병렬로 돌린다. 트랜스포머는 그 단계 전체를 여러 번 쌓고 RNN을 버린다.

수학은 같다. 형태도 같다. Bahdanau 어텐션에서 스케일드 닷-프로덕트 어텐션(scaled dot-product attention)으로의 교육적 도약은 대부분 표기법일 뿐이다.

## 라이브러리로 써보기 (Use It)

PyTorch와 TensorFlow는 어텐션을 직접 제공한다.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

이것이 트랜스포머 어텐션 층(layer)이다. 5개 위치의 쿼리 배치(batch), 10개 위치의 키/밸류 배치, 각 128차원, 8개 헤드. `output`은 맥락이 증강된 새 쿼리다. `weights`는 시각화할 수 있는 5x10 정렬 행렬(matrix)이다.

### 고전 어텐션이 여전히 중요할 때

- 교육. 단일 헤드, 단일 층, RNN 기반 버전은 모든 개념을 가시화한다.
- 트랜스포머가 들어가지 않는 온디바이스 시퀀스 과제.
- 2014-2017년의 모든 논문. Bahdanau의 관례를 모르면 잘못 읽게 된다.
- MT에서의 세밀한 정렬 분석. 가공되지 않은 어텐션 가중치는 트랜스포머 모델에서도 해석 가능성(interpretability) 도구이며, 이를 읽으려면 그것이 무엇인지 알아야 한다.

### 어텐션 가중치를 설명으로 보는 함정

어텐션 가중치는 해석 가능해 보인다. 위치들에 걸쳐 합이 1이 되는 가중치다. 그릴 수 있다. 높으면 "이것을 봤다"는 뜻이다. 리뷰어들이 좋아한다.

하지만 보이는 것만큼 해석 가능하지는 않다. Jain과 Wallace (2019)는 일부 과제에서 어텐션 분포를 치환하고 임의의 대안으로 대체해도 모델 예측이 바뀌지 않음을 보였다. 어블레이션(ablation)이나 반사실(counterfactual) 점검 없이 어텐션 가중치를 추론의 증거로 보고하지 마라.

## 산출물 (Ship It)

`outputs/prompt-attention-shapes.md`로 저장한다:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 연습 문제 (Exercises)

1. **쉬움.** 인코더의 패딩 토큰이 어텐션 가중치 0을 받도록 `softmax` 마스킹을 구현하라. 가변 길이 시퀀스 배치에서 테스트하라.
2. **보통.** Luong `general` 형태에 멀티헤드 어텐션을 추가하라. `d_h`를 `n_heads`개의 그룹으로 나누고, 헤드별로 어텐션을 돌려, 연결하라. 단일 헤드 경우가 앞선 구현과 일치하는지 확인하라.
3. **어려움.** 레슨 09의 장난감 복사 과제에서 Bahdanau 어텐션을 가진 GRU 인코더-디코더를 학습하라. 정확도 대 시퀀스 길이를 그려라. 무어텐션 베이스라인(baseline)과 비교하라. 길이가 늘수록 격차가 벌어지는 것을 볼 텐데, 이는 어텐션이 병목을 들어 올림을 확인해 준다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 어텐션(Attention) | 무언가를 봄 | 밸류 시퀀스의 가중 평균. 가중치는 쿼리-키 유사도에서 계산됨. |
| 쿼리, 키, 밸류(Query, Key, Value) | QKV | 세 가지 투영: Q는 묻고, K는 무엇과 매칭할지, V는 무엇을 반환할지. |
| 가산적 어텐션(Additive attention) | Bahdanau | 피드포워드 점수: `v^T tanh(W q + U k)`. |
| 곱셈적 어텐션(Multiplicative attention) | Luong dot / general | 점수가 `q^T k` 또는 `q^T W k`. 더 저렴하고, 대부분의 과제에서 같은 정확도. |
| 정렬 행렬(Alignment matrix) | 예쁜 그림 | `(T_dec, T_enc)` 격자로서의 어텐션 가중치. 모델이 무엇에 주목했는지 보려면 읽는다. |

## 더 읽을거리 (Further Reading)

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473): 그 논문.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025): 세 가지 점수 변형과 그 비교.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186): 해석 가능성에 대한 주의.
- [Dive into Deep Learning: Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) PyTorch로 된 실행 가능한 따라하기.
