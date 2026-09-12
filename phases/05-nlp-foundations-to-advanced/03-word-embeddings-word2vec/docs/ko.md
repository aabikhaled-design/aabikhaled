# 단어 임베딩(밑바닥부터 만드는 Word2Vec (Word Embeddings) Word2Vec from Scratch)

> 한 단어는 그것이 어울리는 무리로 정의된다. 그 발상으로 얕은 신경망을 학습시키면 기하학적 구조가 떨어져 나온다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75분

## 문제 (The Problem)

TF-IDF는 `dog`와 `puppy`가 서로 다른 단어임을 안다. 하지만 둘이 거의 같은 것을 의미한다는 사실은 모른다. `dog`로 학습된 분류기(classifier)는 `puppy`에 관한 리뷰로 일반화하지 못한다. 동의어 목록을 만들어 이를 덮을 수는 있지만, 희귀한 용어, 도메인 전문 용어, 그리고 예상하지 못한 모든 언어에서 실패한다.

우리가 원하는 것은 `dog`와 `puppy`가 공간에서 가까이 자리 잡는 표현이다. `king - man + woman`이 `queen` 근처에 자리 잡는 표현, `dog`로 학습된 모델이 일부 신호를 `puppy`로 공짜로 전이하는 표현이다.

Word2Vec은 그 공간을 주었다. 2계층 신경망(neural network), 조 단위 토큰(token) 학습 실행, 2013년 발표. 아키텍처는 거의 민망할 정도로 단순하다. 그 결과는 10년간 NLP를 재편했다.

## 개념 (The Concept)

**분포 가설(distributional hypothesis)** (Firth, 1957): "한 단어는 그것이 어울리는 무리로 알 수 있다." 두 단어가 비슷한 맥락에 나타난다면, 아마도 비슷한 것을 의미한다.

Word2Vec은 두 가지 변형으로 나뉘며, 둘 다 그 발상을 이용한다.

- **스킵그램(Skip-gram).** 중심 단어가 주어지면 주변 단어들을 예측한다. 윈도우 크기 2에서 `cat -> (the, sat, on)`.
- **CBOW (continuous bag of words).** 주변 단어들이 주어지면 중심 단어를 예측한다. `(the, sat, on) -> cat`.

스킵그램은 학습이 더 느리지만 희귀 단어를 더 잘 다룬다. 이것이 기본값이 되었다.

이 신경망은 비선형성이 없는 은닉층(hidden layer) 하나를 가진다. 입력은 어휘(vocabulary)에 대한 원-핫 벡터(one-hot vector)다. 출력은 어휘에 대한 소프트맥스(softmax)다. 학습 후에는 출력층을 버린다. 은닉층 가중치(weight)가 바로 임베딩(embedding)이다.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

비결: 10만 단어에 대한 소프트맥스는 비용이 감당할 수 없을 만큼 크다. Word2Vec은 **네거티브 샘플링(negative sampling)**을 사용해 이를 이진 분류(binary classification) 과제로 바꾼다. "이 맥락 단어가 이 중심 단어 근처에 나타났는가, 예 또는 아니오"를 예측한다. 전체 어휘에 대한 소프트맥스를 계산하는 대신, 학습 쌍마다 소수의 네거티브(동시 출현하지 않는) 단어를 샘플링한다.

```figure
word-vector-arithmetic
```

## 직접 만들기 (Build It)

### 1단계: 코퍼스에서 학습 쌍 만들기

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

윈도우 안의 모든 (중심, 맥락) 쌍은 양성(positive) 학습 예시다.

### 2단계: 임베딩 테이블

두 개의 행렬(matrix). `W`는 중심 단어 임베딩 테이블(우리가 보관하는 것). `W'`는 맥락 단어 테이블(흔히 버려지고, 때로는 `W`와 평균을 낸다).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

작은 무작위 초기화. 어휘 크기 1만, 차원 100이 현실적이다. 교육용으로는 어휘 50 x 차원 16이면 기하학적 구조를 보기에 충분하다.

### 3단계: 네거티브 샘플링 목적 함수

각 양성 쌍 `(center, context)`에 대해, 어휘에서 무작위 단어 `k`개를 네거티브로 샘플링한다. 내적(dot product) `W[center] · W'[context]`가 양성에 대해 높고 네거티브에 대해 낮아지도록 모델을 학습시킨다.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

마법의 공식: 양성 쌍에 대한 로지스틱 손실(시그모이드가 1에 가깝길 원함)에 더해, 네거티브 쌍에 대한 로지스틱 손실(시그모이드가 0에 가깝길 원함). 그래디언트(gradient)는 두 테이블 모두로 흐른다. 전체 유도는 원본 논문에 있다. 머릿속에 남기고 싶다면 연필과 종이로 한 번 따라가 보라.

### 4단계: 장난감 코퍼스로 학습

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

큰 코퍼스에서 충분한 에폭(epoch)을 돌린 뒤에는, 맥락을 공유하는 단어들이 비슷한 중심 임베딩을 갖는다. 장난감 코퍼스에서는 그 효과가 희미하게 보인다. 수십억 토큰에서는 극적으로 보인다.

### 5단계: 유추 트릭

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

사전 학습된 300차원 Google News 벡터(vector)에서:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. 모델이 왕족이 무엇인지 알아서가 아니다. 벡터 `(king - man)`이 "왕족 같은" 무언가를 포착하고, 이를 `woman`에 더하면 왕족-여성 영역 근처에 자리 잡기 때문이다.

## 라이브러리로 써보기 (Use It)

Word2Vec을 밑바닥부터 작성하는 것은 교육이다. 프로덕션(production) NLP는 `gensim`을 쓴다.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

실제 작업에서는 Word2Vec을 직접 학습시키는 일이 거의 없다. 사전 학습된 벡터를 다운로드한다.

- **GloVe**: 스탠퍼드의 동시 출현 행렬(co-occurrence-matrix) 분해 접근법. 50차원, 100차원, 200차원, 300차원 체크포인트. 좋은 범용 커버리지. 레슨 04에서 GloVe를 구체적으로 다룬다.
- **fastText**: 문자 n-그램을 임베딩하는 페이스북의 Word2Vec 확장. 서브워드(subword)를 조합하여 미등록(out-of-vocabulary) 단어를 처리한다. 레슨 04.
- **Google News로 사전 학습된 Word2Vec**: 300차원, 300만 단어 어휘, 2013년 발표. 여전히 매일 다운로드된다.

### 2026년에도 Word2Vec이 이기는 때

- 가벼운 도메인 특화 검색. 노트북에서 한 시간 만에 의학 초록으로 학습하여, 어떤 범용 모델도 포착하지 못하는 특화 벡터를 얻는다.
- 유추 스타일의 특성 공학(feature engineering). `gender_vector = mean(man - woman pairs)`. 이를 다른 단어에서 빼서 성 중립 축을 얻는다. 여전히 공정성 연구에서 쓰인다.
- 해석 가능성(interpretability). 100차원은 PCA나 t-SNE로 그려서 클러스터가 형성되는 것을 실제로 볼 수 있을 만큼 작다.
- GPU 없이 온디바이스로 추론(inference)이 실행되어야 하는 모든 곳. Word2Vec 룩업은 단일 행 조회다.

### Word2Vec이 실패하는 곳

다의성의 벽(polysemy wall). `bank`는 벡터가 하나다. `river bank`(강둑)와 `financial bank`(은행)가 이를 공유한다. `table`(스프레드시트 vs. 가구)도 공유한다. 하류의 분류기는 그 벡터로부터 의미를 구별할 수 없다.

문맥적 임베딩(contextual embedding, ELMo, BERT, 그 이후의 모든 트랜스포머)은 주변 맥락에 기반해 단어의 각 출현마다 다른 벡터를 만들어 이를 해결했다. 그것이 Word2Vec에서 BERT로의 도약이다. 정적(static)에서 문맥적(contextual)으로. Phase 7이 트랜스포머 부분을 다룬다.

미등록 단어 문제가 또 다른 실패다. Word2Vec은 `Zoomer-approved`가 학습 데이터에 없었다면 본 적이 없다. 폴백이 없다. fastText는 서브워드 조합으로 이를 고친다(레슨 04).

## 산출물 (Ship It)

`outputs/skill-embedding-probe.md`로 저장한다:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 연습 문제 (Exercises)

1. **쉬움.** 작은 코퍼스(고양이와 개에 관한 문장 20개)에서 학습 루프를 돌려라. 200 에폭 후, `nearest(vocab, W, W[vocab["cat"]])`가 상위 3개에 `dog`를 반환하는지 확인하라. 그렇지 않으면 에폭이나 어휘를 늘려라.
2. **보통.** 빈번한 단어의 서브샘플링(subsampling)을 추가하라. 빈도가 `10^-5`보다 높은 단어는 빈도에 비례하는 확률로 학습 쌍에서 제거된다. 희귀 단어 유사도에 미치는 효과를 측정하라.
3. **어려움.** 20 Newsgroups 코퍼스에서 모델을 학습시켜라. 두 개의 편향 축을 계산하라: `he - she`와 `doctor - nurse`. 직업 단어들을 두 축에 투영하라. 어느 직업이 가장 큰 편향 격차를 갖는지 보고하라. 이것이 공정성 연구자들이 사용하는 종류의 프로브다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 단어 임베딩(Word embedding) | 벡터로서의 단어 | 맥락으로부터 학습된 밀집, 저차원(보통 100-300) 표현. |
| 스킵그램(Skip-gram) | Word2Vec 트릭 | 중심 단어로부터 맥락 단어를 예측. CBOW보다 느리지만 희귀 단어에 더 낫다. |
| 네거티브 샘플링(Negative sampling) | 학습 지름길 | 전체 어휘에 대한 소프트맥스를 `k`개의 무작위 단어에 대한 이진 분류로 대체. |
| 정적 임베딩(Static embedding) | 단어당 벡터 하나 | 맥락과 무관하게 같은 벡터. 다의성에서 실패. |
| 문맥적 임베딩(Contextual embedding) | 맥락에 민감한 벡터 | 주변 단어에 기반해 각 출현마다 다른 벡터. 트랜스포머가 만드는 것. |
| OOV | 미등록 단어 | 학습에서 보지 못한 단어. Word2Vec은 이에 대한 벡터를 만들 수 없다. |

## 더 읽을거리 (Further Reading)

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546): 네거티브 샘플링 논문. 짧고 읽기 쉽다.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738): 원본 논문의 수식이 빽빽하게 느껴진다면, 그래디언트에 대한 가장 명료한 유도.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html): 실제로 동작하는 프로덕션 학습 설정.
