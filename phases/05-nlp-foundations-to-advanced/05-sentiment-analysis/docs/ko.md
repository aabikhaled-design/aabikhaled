# 감성 분석 (Sentiment Analysis)

> 대표적인 NLP 과제. 고전 텍스트 분류에 관해 알아야 할 것 대부분이 여기에 나타난다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75분

## 문제 (The Problem)

"The food was not great." 긍정인가 부정인가?

감성(sentiment)은 단순하게 들린다. 리뷰어가 무언가를 좋아했거나 좋아하지 않았다고 말했으니, 그 문장에 레이블(label)을 붙이면 된다. 이것이 대표적인 NLP 과제가 된 이유는 쉬워 보이는 사례마다 어려운 사례를 하나씩 숨기고 있기 때문이다. 부정(negation)은 의미를 뒤집는다. 풍자는 의미를 반전시킨다. "Not bad at all"은 부정으로 코딩된 두 단어가 있음에도 긍정이다. 이모지는 주변 텍스트보다 더 많은 신호를 담는다. 도메인 어휘가 중요하다(음악 리뷰의 `tight` 대 패션 리뷰의 `tight`).

감성은 고전 NLP의 살아 있는 실험실이다. 순진한 베이스라인(baseline)이 저마다 특정한 실패 양상을 왜 갖는지 이해하면, 더 풍부한 모델들이 왜 나왔는지도 이해하게 된다. 이 레슨은 나이브 베이즈(Naive Bayes) 베이스라인을 밑바닥부터 만들고, 로지스틱 회귀(logistic regression)를 더하고, 프로덕션(production) 감성을 규정 준수 등급의 문제로 만드는 함정들을 짚는다.

## 개념 (The Concept)

고전 감성 분석은 두 단계 레시피다.

1. **표현(Represent).** 텍스트를 특성(feature) 벡터(vector)로 바꾼다. BoW, TF-IDF, 또는 n-그램.
2. **분류(Classify).** 레이블된 예시로 선형 모델(나이브 베이즈, 로지스틱 회귀, SVM)을 적합시킨다.

나이브 베이즈는 동작하는 것 중 가장 멍청한 모델이다. 레이블이 주어졌을 때 모든 특성이 독립이라고 가정한다. 카운트로부터 `P(word | positive)`와 `P(word | negative)`를 추정한다. 추론(inference) 시점에 확률들을 곱한다. "나이브"한 독립 가정은 우스울 정도로 틀렸지만 결과는 놀랄 만큼 잘 나온다. 이유는 이렇다. 희소한 텍스트 특성과 적당한 데이터에서 분류기는 각 단어가 얼마나 기우는지보다 어느 쪽으로 기우는지에 관심을 둔다.

로지스틱 회귀는 독립 가정을 고친다. 음의 가중치(weight)를 포함해 특성마다 가중치를 학습한다. `not good`을 바이그램(bigram) 특성으로 보면 음의 가중치를 받는다. 나이브 베이즈는 한 번도 레이블링하지 않은 바이그램에 대해서는 그렇게 할 수 없다.

```figure
sentiment-logits
```

## 직접 만들기 (Build It)

### 1단계: 진짜 미니 데이터셋

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

일부러 작게 했다. 실제 작업은 수만 개의 예시(IMDb, SST-2, Yelp polarity)를 쓴다. 수학은 동일하다.

### 2단계: 밑바닥부터 만드는 다항 나이브 베이즈

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

가산 평활화(additive smoothing, alpha=1.0)는 라플라스 평활화(Laplace smoothing)다. 이것이 없으면 어떤 클래스에서 보지 못한 단어는 확률이 0이 되고 로그가 폭발한다. 실무에서는 `alpha=0.01`이 흔하다. `alpha=1.0`은 교육용 기본값이다.

### 3단계: 밑바닥부터 만드는 로지스틱 회귀

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

여기서는 L2 규제(regularization)가 중요하다. 텍스트 특성은 희소하다. L2가 없으면 모델은 학습 예시를 외운다. `0.01`에서 시작해 튜닝하라.

### 4단계: 부정 처리하기 (실패 양상)

"not good"과 "not bad"를 보라. BoW 분류기는 `{not, good}`과 `{not, bad}`를 보고, 학습 데이터에서 더 자주 나타난 쪽을 따라 학습한다. 바이그램 분류기는 `not_good`과 `not_bad`를 보고 이들을 별개의 특성으로 학습한다. 보통은 그것으로 충분하다.

바이그램이 없을 때 동작하는 더 거친 해결책: **부정 범위 지정(negation scoping)**. 부정 단어 뒤에 오는 토큰(token)에 다음 구두점까지 `NOT_`을 접두로 붙인다.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

이제 `good`과 `NOT_good`은 서로 다른 특성이다. 분류기는 이들에 반대 가중치를 줄 수 있다. 전처리 세 줄로 감성 벤치마크(benchmark)에서 측정 가능한 정확도 향상이 일어난다.

### 5단계: 중요한 평가 지표

클래스가 불균형하면 정확도(accuracy)만으로는 오해를 부른다. 실제 감성 코퍼스는 보통 70-80%가 긍정이거나 70-80%가 부정이다. 항상 다수를 찍는 분류기는 80% 정확도를 얻지만 쓸모가 없다. 다음을 빠짐없이 보고하라:

- **클래스별 정밀도(precision)와 재현율(recall).** 클래스당 한 쌍. 이들을 매크로 평균(macro-average)하여 클래스 균형을 존중하는 단일 숫자를 얻는다.
- **매크로-F1(불균형 데이터의 주요 지표).** 클래스별 F1 점수의, 동등하게 가중된 평균. 클래스가 불균형할 때 정확도 대신 이것을 쓴다.
- **가중-F1(대안).** 매크로와 같되 클래스 빈도로 가중. 불균형 자체가 비즈니스 의미를 가질 때 매크로-F1과 함께 보고한다.
- **혼동 행렬(confusion matrix).** 가공되지 않은 카운트. 어떤 스칼라 지표든 믿기 전에 항상 들여다보라. 모델이 어떤 클래스 쌍을 혼동하는지 드러낸다.
- **클래스별 오류 샘플.** 클래스당 잘못된 예측 5개를 뽑아라. 읽어라. 실제 오류를 읽는 것을 대체할 것은 없다.

심하게 불균형한 데이터(> 95-5 비율)에서는 정확도 대신 **AUROC**와 **AUPRC**를 보고하라. AUPRC는 소수 클래스에 더 민감한데, 보통 신경 쓰는 대상이 바로 그 소수 클래스(스팸, 사기, 희귀 감성)다.

**피해야 할 흔한 버그.** 불균형 데이터에서 매크로-F1 대신 마이크로-F1을 보고하면 다수 클래스에 지배되어 높아 보이는 숫자가 나온다. 매크로-F1은 소수 클래스 성능을 강제로 보게 만든다.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 라이브러리로 써보기 (Use It)

scikit-learn은 이것을 여섯 줄로, 올바르게 해낸다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

주목할 세 가지. `stop_words=None`은 부정어를 유지한다. `ngram_range=(1, 2)`는 바이그램을 더해 `not_good`이 특성이 되게 한다. `sublinear_tf=True`는 반복된 단어를 누그러뜨린다. 이 세 플래그가 SST-2에서 정확도 75% 베이스라인과 85% 베이스라인을 가른다.

### 트랜스포머로 손을 뻗을 때

- 풍자 탐지. 고전 모델은 여기서 실패한다. 끝.
- 감성이 문서 중간에 바뀌는 긴 리뷰.
- 애스펙트 기반 감성. "Camera was great but battery was terrible." 감성을 애스펙트(aspect)에 귀속시켜야 한다. 트랜스포머나 구조화 출력 모델만 가능하다.
- 비영어, 저자원 언어. 다국어 BERT는 제로샷 베이스라인을 공짜로 준다.

위의 어느 것이라도 필요하다면 Phase 7(트랜스포머 심화)로 건너뛰어라. 그렇지 않다면, TF-IDF에 바이그램과 부정 처리를 더한 나이브 베이즈나 로지스틱 회귀가 2026년의 프로덕션 베이스라인이다.

### 재현성의 함정 (다시)

감성 모델의 재학습은 일상이다. 재평가는 그렇지 않다. 논문에 보고된 정확도 수치는 특정 분할, 특정 전처리, 특정 토크나이저(tokenizer)를 쓴다. 동일한 파이프라인(pipeline)을 쓰지 않고 새 모델을 베이스라인과 비교하면 오해를 부르는 차이값이 나온다. 논문의 숫자가 아니라, 같은 파이프라인에서 베이스라인을 항상 다시 생성하라.

## 산출물 (Ship It)

`outputs/prompt-sentiment-baseline.md`로 저장한다:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 연습 문제 (Exercises)

1. **쉬움.** scikit-learn 파이프라인에 `apply_negation`을 전처리 단계로 추가하고 작은 감성 데이터셋에서 F1 차이값을 측정하라.
2. **보통.** 클래스 가중 로지스틱 회귀를 구현하라(scikit-learn에 `class_weight="balanced"`를 넘기거나, 그래디언트를 직접 유도하라). 합성된 90-10 클래스 불균형에서의 효과를 측정하라.
3. **어려움.** 감성 모델의 잔차에 대해 두 번째 분류기를 학습시켜 풍자 탐지기를 만들어라. 실험 설정을 문서화하라. 정확도가 우연 수준 이하일 때 독자에게 경고하라(2-클래스 풍자에서 우연 수준은 약 50%이며, 대부분의 첫 시도가 거기에 떨어진다).

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 극성(Polarity) | 긍정 또는 부정 | 이진 레이블. 때로는 중립이나 세분화(5점 척도)로 확장됨. |
| 애스펙트 기반 감성(Aspect-based sentiment) | 애스펙트별 극성 | 텍스트에 언급된 특정 엔티티나 속성에 감성을 귀속. |
| 부정 범위 지정(Negation scoping) | 인근 토큰 뒤집기 | "not" 뒤의 토큰에 구두점까지 `NOT_`을 접두로. |
| 라플라스 평활화(Laplace smoothing) | 카운트에 1 더하기 | 나이브 베이즈에서 확률 0 특성을 방지. |
| L2 규제(L2 regularization) | 가중치 줄이기 | 손실에 `lambda * sum(w^2)`를 더함. 희소 텍스트 특성에 필수. |

## 더 읽을거리 (Further Reading)

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html): 기초가 되는 서베이. 길지만, 처음 네 절이 모든 고전적 내용을 다룬다.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/): 짧은 텍스트에서 바이그램 + 나이브 베이즈를 이기기 어렵다는 것을 보여 준 논문.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction): `CountVectorizer`, `TfidfVectorizer`, 그리고 튜닝할 모든 손잡이를 다루는 레퍼런스.
