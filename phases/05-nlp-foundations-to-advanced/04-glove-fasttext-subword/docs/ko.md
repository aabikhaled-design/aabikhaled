# GloVe, FastText, 서브워드 임베딩 (GloVe, FastText, and Subword Embeddings)

> Word2Vec은 단어당 임베딩 하나를 학습했다. GloVe는 동시 출현 행렬을 분해했다. FastText는 조각들을 임베딩했다. BPE는 트랜스포머로 이어 주었다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45분

## 문제 (The Problem)

Word2Vec은 두 가지 열린 질문을 남겼다.

첫째, 온라인 스킵그램(skip-gram) 업데이트를 하는 대신 동시 출현 행렬(co-occurrence matrix)을 직접 분해하는 병렬 연구 갈래(LSA, HAL)가 있었다. Word2Vec의 반복적 접근법이 근본적으로 더 나은 것이었을까, 아니면 그 차이는 두 방법이 카운트를 다루는 방식의 부산물이었을까? **GloVe**가 답했다. 신중하게 선택한 손실(loss)을 가진 행렬 분해(matrix factorization)는 Word2Vec과 비등하거나 더 낫고, 학습 비용이 더 적다.

둘째, 어느 방법도 처음 보는 단어는 다루지 못했다. `Zoomer-approved`, `dogecoin`, 지난주에 만들어진 어떤 고유명사, 희귀 어근의 모든 굴절형이 그렇다. **FastText**는 문자 n-그램을 임베딩하여 이를 고쳤다. 한 단어는 형태소(morpheme)를 포함한 그 부분들의 합이므로, 미등록(out-of-vocabulary) 단어조차 합리적인 벡터(vector)를 얻는다.

셋째, 트랜스포머(transformer)가 등장하자 질문이 다시 바뀌었다. 단어 단위 어휘(vocabulary)는 약 100만 항목에서 한계에 부딪힌다. 실제 언어는 그보다 더 개방적이다. **바이트 페어 인코딩(byte-pair encoding, BPE)**과 그 친척들은 모든 것을 포괄하는 빈번한 서브워드(subword) 단위의 어휘를 학습하여 이를 해결했다. 모든 현대 LLM의 모든 현대 토크나이저(tokenizer)는 서브워드 토크나이저다.

이 레슨은 셋 모두를 살펴본 다음, 언제 무엇을 집어 들지 설명한다.

## 개념 (The Concept)

**GloVe (Global Vectors).** 단어-단어 동시 출현 행렬 `X`를 만든다. 여기서 `X[i][j]`는 단어 `j`가 단어 `i`의 맥락에 얼마나 자주 나타나는지다. `v_i · v_j + b_i + b_j ≈ log(X[i][j])`가 되도록 벡터를 학습한다. 빈번한 쌍이 지배하지 못하도록 손실에 가중치를 둔다. 끝.

**FastText.** 한 단어는 그 문자 n-그램들과 단어 자체의 합이다. `where`는 `<wh, whe, her, ere, re>, <where>`가 된다. 단어 벡터는 그 구성 요소 벡터들의 합이다. Word2Vec처럼 학습한다. 이점: 본 적 없는 단어(`whereupon`)는 알려진 n-그램들로부터 조합된다.

**BPE (바이트 페어 인코딩).** 개별 바이트(또는 문자)로 이루어진 어휘에서 시작한다. 코퍼스의 모든 인접 쌍을 센다. 가장 빈번한 쌍을 새 토큰(token)으로 병합한다. `k`번 반복한다. 결과: `k + 256`개의 토큰으로 이루어진 어휘로, 빈번한 시퀀스(`ing`, `tion`, `the`)는 단일 토큰이고 희귀 단어는 익숙한 조각들로 쪼개진다. 모든 문장이 무언가로 토큰화된다.

```figure
n5-subword-merge
```

## 직접 만들기 (Build It)

### GloVe: 동시 출현 행렬 분해

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

짚어 둘 만한 구성 요소가 둘 있다. 가중치 함수 `f(x) = (x/x_max)^alpha`는 매우 빈번한 쌍(`(the, and)` 같은)의 가중치를 낮춰 손실을 지배하지 못하게 한다. 최종 임베딩은 `W`(중심)와 `W_tilde`(맥락) 테이블의 합이다. 둘을 합하면 하나만 쓸 때보다 성능이 더 좋게 나오는 경향이 있는데, 이는 발표된 트릭이다.

### FastText: 서브워드 인식 임베딩

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

각 단어는 그 n-그램 집합(보통 3-6 문자)으로 표현된다. 단어 임베딩은 그 n-그램 임베딩들의 합이다. 스킵그램 학습에서는 Word2Vec이 단일 벡터를 쓰던 자리에 이것을 꽂는다.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

본 적 없는 단어라도 그 n-그램 중 일부가 알려져 있으면 벡터를 얻는다. `whereupon`은 `where`와 `<wh`, `her`, `ere`, `<where`를 공유하므로, 둘은 서로 가까이 자리 잡는다.

### BPE: 학습된 서브워드 어휘

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

첫 반복은 가장 흔한 인접 쌍을 병합한다. 충분히 반복하고 나면, 빈번한 부분 문자열(`low`, `est`, `tion`)은 단일 토큰이 되고 희귀 단어는 깔끔하게 쪼개진다.

실제 GPT / BERT / T5 토크나이저는 3만-10만 개의 병합을 학습한다. 그래서 어떤 텍스트든 알려진 ID들의 유계 길이 시퀀스로 토큰화되며, OOV가 결코 없다.

## 라이브러리로 써보기 (Use It)

실제로는 이들 중 어느 것도 직접 학습시키는 일이 드물다. 사전 학습된 체크포인트를 로드한다.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

트랜스포머 시대의 BPE 스타일 서브워드 토큰화는:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 접두사는 단어 경계를 표시한다(GPT-2의 관례). 모든 현대 토크나이저는 BPE 변형, WordPiece(BERT), 또는 SentencePiece(T5, LLaMA)다.

### 무엇을 언제 고를까

| 상황 | 선택 |
|-----------|------|
| 사전 학습된 범용 단어 벡터, OOV 허용이 필요 없음 | GloVe 300차원 |
| 사전 학습된 범용 단어 벡터, 오타 / 신조어 / 형태론적으로 풍부한 언어를 다뤄야 함 | FastText |
| 트랜스포머로 들어가는 모든 것(학습 또는 추론) | 모델이 함께 출하된 토크나이저가 무엇이든 그것. 절대 바꾸지 말 것. |
| 자신의 언어 모델을 밑바닥부터 학습 | 먼저 코퍼스로 BPE나 SentencePiece 토크나이저를 학습 |
| 선형 모델을 쓰는 프로덕션 텍스트 분류 | 여전히 TF-IDF. 레슨 02. |

## 산출물 (Ship It)

`outputs/skill-embeddings-picker.md`로 저장한다:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 연습 문제 (Exercises)

1. **쉬움.** `char_ngrams("playing")`과 `char_ngrams("played")`를 실행하라. 두 n-그램 집합의 자카드(Jaccard) 겹침을 계산하라. 상당히 공유되는 조각(`pla`, `lay`, `play`)이 보일 텐데, 바로 이 때문에 FastText가 형태론적 변형 사이에서 잘 전이된다.
2. **보통.** `learn_bpe`를 확장하여 어휘 증가를 추적하라. 병합 수의 함수로 코퍼스 문자당 토큰 수를 그려라. 처음에는 빠른 압축을, 그다음 토큰당 약 2-3 문자 근처에서 점근하는 것을 볼 것이다.
3. **어려움.** 셰익스피어 전집에서 1천-병합 BPE를 학습하라. 흔한 단어 대 희귀 고유명사의 토큰화를 비교하라. 전후의 단어당 평균 토큰 수를 측정하라. 무엇이 놀라웠는지 정리하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 동시 출현 행렬(Co-occurrence matrix) | 단어-단어 빈도 표 | `X[i][j]` = 단어 `j`가 단어 `i` 주변 윈도우에 얼마나 자주 나타나는가. |
| 서브워드(Subword) | 단어의 한 조각 | 문자 n-그램(FastText) 또는 학습된 토큰(BPE/WordPiece/SentencePiece). |
| BPE | 바이트 페어 인코딩 | 어휘가 목표 크기에 도달할 때까지 가장 빈번한 인접 쌍을 반복적으로 병합. |
| OOV | 미등록 단어 | 모델이 한 번도 본 적 없는 단어. Word2Vec/GloVe는 실패. FastText와 BPE는 처리. |
| 바이트 단위 BPE(Byte-level BPE) | 가공되지 않은 바이트에 대한 BPE | GPT-2의 방식. 어휘가 256개 바이트로 시작하므로 결코 OOV가 없다. |

## 더 읽을거리 (Further Reading)

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf): GloVe 논문, 일곱 쪽, 여전히 손실에 대한 최고의 유도.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): 현대 NLP에 BPE를 도입한 논문.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary): BPE, WordPiece, SentencePiece가 실제로 어떻게 다른지.
