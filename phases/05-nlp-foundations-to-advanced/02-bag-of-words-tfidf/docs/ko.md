# 백 오브 워즈, TF-IDF, 텍스트 표현 (Bag of Words, TF-IDF, and Text Representation)

> 먼저 세고, 생각은 나중에. TF-IDF는 2026년에도 잘 정의된 과제에서는 여전히 임베딩(embedding)을 이긴다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75분

## 문제 (The Problem)

모델에는 숫자가 필요하다. 그런데 가진 것은 문자열이다.

모든 NLP 파이프라인(pipeline)은 동일한 질문에 답해야 한다. 가변 길이의 토큰(token) 흐름을, 분류기(classifier)가 소비할 수 있는 고정 크기 벡터(vector)로 어떻게 바꿀 것인가. 이 분야가 처음 내놓은 답은 작동하는 것 중에 가장 단순무식한 방식이었다. 단어를 센다. 벡터를 만든다.

그 벡터는 그 어떤 임베딩 모델보다 더 많은 프로덕션(production) NLP를 지탱해 왔다. 스팸 필터, 토픽 분류기, 로그 이상 탐지, 검색 랭킹(BM25 이전), 감성 분석(sentiment analysis)의 첫 물결, 학술 NLP 벤치마크(benchmark)의 첫 10년. 2026년의 실무자들도 좁은 분류(classification) 과제에서는 여전히 이것을 먼저 집어 든다. 빠르고, 해석 가능하며, 단어의 존재 여부가 중요한 과제에서는 종종 4억 파라미터(parameter) 임베딩 모델과 구별되지 않는다.

이 레슨은 백 오브 워즈(bag of words)를, 그다음 TF-IDF를 밑바닥부터 만든다. 그다음 scikit-learn이 동일한 일을 세 줄로 해내는 것을 보여 준다. 그다음 임베딩으로 손을 뻗게 만드는 실패 양상을 짚는다.

## 개념 (The Concept)

**백 오브 워즈(Bag of Words, BoW)**는 순서를 버린다. 각 문서에 대해, 어휘(vocabulary)의 각 단어가 몇 번 나타나는지 센다. 벡터 길이는 어휘 크기다. 위치 `i`는 단어 `i`의 카운트다.

**TF-IDF**는 BoW를 재가중한다. 모든 문서에 나타나는 단어는 정보량이 없으므로 그 값을 낮춘다. 코퍼스 전체에서는 드물지만 단일 문서에서 빈번한 단어는 신호이므로 그 값을 높인다.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

여기서 `TF`는 문서 내 단어 빈도, `df`는 문서 빈도(document frequency, 그 단어를 포함하는 문서가 몇 개인지), `N`은 전체 문서 수다. `log`는 어디에나 등장하는 단어의 가중치를 유계로 유지한다.

핵심 성질: 둘 다 해석 가능한 축을 가진 희소 벡터(sparse vector)를 만든다. 학습된 분류기의 가중치(weight)를 들여다보면 어떤 단어가 문서를 각 클래스 쪽으로 미는지 읽을 수 있다. 768차원 BERT 임베딩으로는 이것을 할 수 없다.

```figure
bow-tfidf
```

## 직접 만들기 (Build It)

### 1단계: 어휘 구축

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

입력: 토큰화된 문서의 리스트(어떤 단어 단위 토크나이저든 된다. 이 레슨의 `code/main.py`는 단순화된 소문자 변형을 사용한다). 출력: `{word: index}` 딕셔너리. 삽입 순서가 안정적이므로 단어 인덱스 0은 첫 번째 문서에서 처음 본 단어가 된다. 관례는 다양하다. scikit-learn은 알파벳순으로 정렬한다.

### 2단계: 백 오브 워즈

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

행은 문서다. 열은 어휘 인덱스다. 항목 `[i][j]`는 "단어 `j`가 문서 `i`에 몇 번 나타나는가"이다. 문서 1은 `cat`이 실제로 두 번 나오므로 2다. 문서 0은 `ran`이 한 번도 나오지 않으므로 0이다.

### 3단계: 단어 빈도와 문서 빈도

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

짚어 둘 만한 두 가지 평활화(smoothing) 기법이 있다. `(n+1)/(d+1)`은 `log(x/0)`를 피한다. 끝의 `+1`은 모든 문서에 등장하는 단어도 IDF가 0이 아니라 1이 되도록 보장하여 scikit-learn의 기본값과 맞춘다. 다른 구현은 가공되지 않은 `log(N/df)`를 사용한다. 둘 다 동작한다. 평활화된 버전이 더 친절하다.

### 4단계: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

세 문서, 다섯 어휘 단어(`the`, `cat`, `sat`, `dog`, `ran`). `the`는 셋 모두에 나타나므로 IDF가 낮다. `dog`는 하나에만 나타나므로 IDF가 높다. 벡터는 희소하고(대부분의 항목이 작다) 변별력 있는 단어가 두드러진다.

### 5단계: 행을 L2 정규화

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

정규화(normalization)가 없으면 더 긴 문서가 더 큰 벡터를 갖게 되어 유사도 점수를 지배한다. L2 정규화는 모든 문서를 단위 초구면(unit hypersphere) 위에 올려놓는다. 이제 행들 사이의 코사인 유사도(cosine similarity)는 그저 내적(dot product)이다.

## 라이브러리로 써보기 (Use It)

scikit-learn은 프로덕션 버전을 제공한다.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`는 토큰화, 어휘 구축, BoW를 한 번의 호출로 한다. `TfidfVectorizer`는 IDF 가중과 L2 정규화를 더한다. 둘 다 희소 행렬(sparse matrix)을 반환한다. 10만 문서라면 밀집(dense) 버전은 메모리에 들어가지 않는다. 분류기가 밀집을 요구할 때까지 희소를 유지하라.

모든 것을 바꾸는 손잡이들:

| 인자 | 효과 |
|-----|--------|
| `ngram_range=(1, 2)` | 바이그램(bigram)을 포함한다. 보통 분류 성능을 높인다. |
| `min_df=2` | 2개 미만 문서에 등장하는 단어를 버린다. 노이즈가 많은 데이터에서 어휘를 줄인다. |
| `max_df=0.95` | 95%를 초과하는 문서에 등장하는 단어를 버린다. 하드코딩된 목록 없이 불용어 제거를 근사한다. |
| `stop_words="english"` | scikit-learn의 내장 불용어 목록. 과제 의존적: 감성 분석은 부정어를 버리면 *안 된다*. |
| `sublinear_tf=True` | 가공되지 않은 `tf` 대신 `1 + log(tf)`를 사용한다. 한 문서에서 한 단어가 여러 번 반복될 때 도움이 된다. |

### TF-IDF가 여전히 이기는 때 (2026년 기준)

- 스팸 탐지, 토픽 라벨링, 로그 이상 플래깅. 단어의 존재 여부가 중요하고, 의미적 뉘앙스는 중요하지 않다.
- 데이터가 적은 상황(레이블된 예시 수백 개). TF-IDF에 로지스틱 회귀(logistic regression)를 더하면 사전 학습(pretraining) 비용이 없다.
- 지연 시간(latency)이 중요한 모든 곳. TF-IDF에 선형 모델을 더하면 마이크로초 단위로 답한다. 문서를 트랜스포머로 임베딩하는 데는 10-100ms가 걸린다.
- 예측을 설명해야 하는 시스템. 분류기의 계수를 들여다보라. 가장 큰 양의 단어들이 그 이유다.

### TF-IDF가 실패하는 때

의미적 맹점(semantic blindness) 실패. 다음 두 문서를 보라:

- "The movie was not good at all."
- "The movie was excellent."

하나는 부정적 리뷰다. 하나는 긍정적이다. 둘의 TF-IDF 겹침은 정확히 `{the, movie, was}`뿐이다. 백 오브 워즈 분류기는 `good` 근처의 `not`이 레이블을 뒤집는다는 사실을 외워야 한다. 충분한 데이터로 이를 학습할 수는 있지만, 구문(syntax)을 이해하는 모델만큼 우아하게는 결코 안 된다.

또 다른 실패: 추론(inference) 시점의 미등록 단어(out-of-vocabulary). IMDb 리뷰로 학습된 BoW 모델은 `Zoomer-approved`라는 토큰이 학습에 한 번도 등장하지 않았다면 그것을 어떻게 다뤄야 할지 전혀 모른다. 서브워드(subword) 임베딩(레슨 04)은 이를 처리한다. TF-IDF는 할 수 없다.

### 하이브리드: TF-IDF 가중 임베딩

중간 규모 데이터 분류를 위한 2026년의 실용적 기본값: TF-IDF 가중치를 단어 임베딩에 대한 어텐션(attention)으로 사용한다.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

임베딩으로부터 의미적 수용력을, TF-IDF로부터 희귀 단어 강조를 얻는다. 분류기는 풀링된 벡터로 학습한다. 약 5만 개 미만의 레이블된 예시에서 감성, 토픽, 의도 분류에 대해 이는 둘 중 어느 하나 단독보다 더 나은 성능을 낸다.

## 산출물 (Ship It)

`outputs/prompt-vectorization-picker.md`로 저장한다:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 연습 문제 (Exercises)

1. **쉬움.** L2 정규화된 TF-IDF 출력에 대해 `cosine_similarity(doc_vec_a, doc_vec_b)`를 구현하라. 동일한 문서는 1.0, 어휘가 겹치지 않는 문서는 0.0이 나오는지 확인하라.
2. **보통.** `bag_of_words`에 `n-gram` 지원을 추가하라. 파라미터 `n`은 `n`-그램에 대한 카운트를 만든다. `["the", "cat", "sat"]`에 대해 `n=2`가 `["the cat", "cat sat"]`의 바이그램 카운트를 만드는지 테스트하라.
3. **어려움.** 위의 TF-IDF 가중 임베딩 하이브리드를 GloVe 100차원 벡터(한 번 다운로드하고 캐시)를 사용해 만들어라. 20 Newsgroups 데이터셋(dataset)에서 순수 TF-IDF, 순수 평균 풀링(mean-pooled) 임베딩과 분류 정확도를 비교하라. 어디서 무엇이 이기는지 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BoW | 단어 빈도 벡터 | 한 문서 내 어휘 단어들의 카운트. 순서를 버린다. |
| TF | 단어 빈도 | 문서 내 한 단어의 카운트, 선택적으로 문서 길이로 정규화됨. |
| DF | 문서 빈도 | 그 단어를 최소 한 번 포함하는 문서의 수. |
| IDF | 역문서 빈도 | 평활화된 `log(N / df)`. 어디에나 등장하는 단어의 가중치를 낮춘다. |
| 희소 벡터(Sparse vector) | 대부분이 0 | 어휘는 보통 1만-10만 단어다. 대부분은 주어진 문서에 없다. |
| 코사인 유사도(Cosine similarity) | 벡터 각도 | L2 정규화된 벡터의 내적. 1은 동일, 0은 직교. |

## 더 읽을거리 (Further Reading)

- [scikit-learn: feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) 표준 API 레퍼런스, 모든 손잡이에 대한 설명 포함.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210): TF-IDF를 10년간 기본값으로 만든 논문.
- ["Why TF-IDF Still Beats Embeddings": Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) 옛 방법이 언제, 왜 이기는지에 대한 2026년의 관점.
