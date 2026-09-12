# Topic Modeling — LDA and BERTopic

> LDA: 문서는 토픽(topic)의 혼합이고, 토픽은 단어에 대한 분포다. BERTopic: 문서는 임베딩 공간에서 군집을 이루고, 군집이 토픽이다. 같은 목표, 다른 분해다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45분

## 문제 (The Problem)

고객 지원 티켓 10,000건, 뉴스 기사 50,000건, 트윗 200,000건이 있다고 하자. 일일이 읽지 않고도 이 모음이 무엇에 관한 것인지 알아야 한다. 레이블링된 범주는 없다. 범주가 몇 개나 존재하는지조차 모른다.

토픽 모델링(topic modeling)은 지도(supervision) 없이 이 질문에 답한다. 코퍼스(corpus)를 넣으면 일관된 토픽의 작은 집합과, 각 문서가 그 토픽들에 어떻게 분포하는지를 돌려받는다.

두 개의 알고리즘 계열이 지배한다. LDA(2003)는 각 문서를 잠재 토픽(latent topic)의 혼합으로, 각 토픽을 단어에 대한 분포로 다룬다. 추론(inference)은 베이지안(Bayesian)이다. 혼합 소속(mixed-membership) 토픽 할당과 설명 가능한 단어 수준 확률 분포(probability distribution)가 필요한 곳에서는 여전히 프로덕션(production)에 배포된다.

BERTopic(2020)은 문서를 BERT로 인코딩하고, UMAP으로 차원을 축소하며, HDBSCAN으로 군집화하고, 클래스 기반 TF-IDF로 토픽 단어를 추출한다. 짧은 텍스트, 소셜 미디어, 그리고 단어 중복보다 의미적 유사도(similarity)가 더 중요한 모든 것에서 이긴다. 한 문서가 토픽 하나만 받는데, 이는 장문 콘텐츠에서는 한계로 작용한다.

이 레슨은 두 계열의 직관을 쌓고, 주어진 코퍼스에 어느 쪽을 고를지 짚는다.

## 개념 (The Concept)

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 생성 이야기.** 각 토픽은 단어에 대한 분포다. 각 문서는 토픽의 혼합이다. 문서 안의 단어를 생성하려면, 문서의 혼합에서 토픽을 샘플링(sampling)한 다음, 그 토픽의 분포에서 단어를 샘플링한다. 추론은 이 과정을 뒤집는다. 관측된 단어로부터 문서당 토픽 분포와 토픽당 단어 분포를 거꾸로 추론한다. 붕괴된 깁스 샘플링(collapsed Gibbs sampling)이나 변분 베이즈(variational Bayes)가 그 계산을 한다.

핵심 LDA 출력:

- `doc_topic`: `(n_docs, n_topics)` 행렬(matrix), 각 행의 합은 1(문서의 토픽 혼합).
- `topic_word`: `(n_topics, vocab_size)` 행렬, 각 행의 합은 1(토픽의 단어 분포).

**BERTopic 파이프라인(pipeline).**

1. 각 문서를 문장 트랜스포머(sentence transformer)(예: `all-MiniLM-L6-v2`)로 인코딩한다. 384차원 벡터(vector).
2. UMAP으로 차원을 약 5차원으로 축소한다. BERT 임베딩은 군집화하기에 너무 고차원이다.
3. HDBSCAN으로 군집화한다. 밀도 기반(density-based)이며, 가변 크기 군집과 "이상치(outlier)" 레이블을 만든다.
4. 각 군집에 대해, 군집의 문서들에 대한 클래스 기반 TF-IDF를 계산하여 상위 단어를 추출한다.

출력은 문서당 하나의 토픽(더하기 -1 이상치 레이블)이다. 선택적으로 HDBSCAN의 확률 벡터를 통한 소프트 소속(soft membership).

```figure
topic-drift
```

## 직접 만들기 (Build It)

### Step 1: scikit-learn을 사용한 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

주목하라: 불용어(stopword)가 제거되고, min_df와 max_df가 희귀하고 어디에나 있는 단어를 걸러내며, LDA가 원시 카운트를 기대하므로 CountVectorizer(TfidfVectorizer가 아님)를 쓴다.

### Step 2: BERTopic (프로덕션)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 필터는 BERTopic의 이상치 버킷(HDBSCAN이 군집화할 수 없었던 문서)을 떨어뜨린다. `min_topic_size`는 HDBSCAN의 최소 군집 크기를 제어한다. BERTopic 라이브러리 기본값은 10이다. 이 예제는 레슨의 규모를 위해 명시적으로 15로 설정한다. 10,000개를 넘는 문서 코퍼스에는 50이나 100으로 늘려라.

### Step 3: 평가

두 방법 모두 토픽 단어를 출력한다. 문제는 그 단어들이 일관성이 있는가다.

- **토픽 일관성(Topic coherence, c_v).** 슬라이딩 윈도우 맥락에 걸친 상위 단어 쌍의 NPMI(정규화된 점별 상호 정보, normalized pointwise mutual information)를 결합하고, 그 점수들을 토픽 벡터로 집계한 뒤, 코사인 유사도로 그 벡터들을 비교한다. 높을수록 좋다. `coherence="c_v"`와 함께 `gensim.models.CoherenceModel`을 사용하라.
- **토픽 다양성(Topic diversity).** 모든 토픽의 상위 단어에 걸친 고유 단어의 비율. 높을수록 좋다(토픽이 겹치지 않음).
- **정성적 검사(Qualitative inspection).** 각 토픽의 상위 단어를 읽어라. 실제 무언가를 명명하는가? 사람의 판단이 여전히 최후의 방어선이다.

## 어느 것을 언제 고를 것인가

| 상황 | 선택 |
|-----------|------|
| 짧은 텍스트(트윗, 리뷰, 헤드라인) | BERTopic |
| 토픽 혼합을 가진 긴 문서 | LDA |
| GPU 없음 / 제한된 컴퓨팅 | LDA 또는 NMF |
| 문서 수준 다중 토픽 분포 필요 | LDA |
| 토픽 레이블링을 위한 LLM 통합 | BERTopic (직접 지원) |
| 자원 제약 엣지 배포 | LDA |
| 최대 의미적 일관성 | BERTopic |

가장 큰 실용적 고려 사항은 문서 길이다. BERT 임베딩은 잘린다(truncate). LDA 카운트는 어떤 길이에서도 작동한다. 임베딩 모델의 컨텍스트(context)보다 긴 문서라면 청크 + 집계(chunk + aggregate)를 하거나 LDA를 쓴다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

- **BERTopic.** 짧은 텍스트와 의미가 중요한 모든 것의 기본값.
- **`gensim.models.LdaModel`.** 프로덕션용 고전 LDA, 성숙하고, 실전 검증됨.
- **`sklearn.decomposition.LatentDirichletAllocation`.** 실험용 손쉬운 LDA.
- **NMF.** 비음수 행렬 분해(Non-negative matrix factorization). LDA에 대한 빠른 대안, 짧은 텍스트에서 비슷한 품질.
- **Top2Vec.** BERTopic과 유사한 설계. 더 작은 커뮤니티지만 일부 벤치마크(benchmark)에서 우수.
- **FASTopic.** 더 새롭고, 매우 큰 코퍼스에서 BERTopic보다 빠름.
- **LLM 기반 레이블링.** 임의의 군집화를 실행한 뒤, 각 군집을 명명하도록 모델에 프롬프트(prompt)한다.

## 산출물 (Ship It)

`outputs/skill-topic-picker.md`로 저장하라:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 연습 문제 (Exercises)

1. **Easy.** 20 Newsgroups 데이터셋(dataset)에 5개 토픽으로 LDA를 적합(fit)시켜라. 토픽당 상위 10개 단어를 출력하라. 각 토픽을 손으로 레이블링하라. 알고리즘이 실제 범주를 찾았는가?
2. **Medium.** 같은 20 Newsgroups 부분집합에 BERTopic을 적합시켜라. 찾은 토픽 수, 상위 단어, 정성적 일관성을 LDA와 비교하라. 어느 것이 실제 범주를 더 깔끔하게 드러내는가?
3. **Hard.** 직접 고른 코퍼스에서 LDA와 BERTopic 양쪽의 c_v 일관성을 계산하라. 각각을 5, 10, 20, 50개 토픽으로 실행하라. 일관성 대 토픽 수를 그래프로 그려라. 어느 방법이 토픽 수에 걸쳐 더 안정적인지 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Topic | 코퍼스가 다루는 무언가 | 단어에 대한 확률 분포(LDA) 또는 유사한 문서의 군집(BERTopic). |
| Mixed membership | 문서가 여러 토픽임 | LDA는 각 문서에 모든 토픽에 대한 분포를 할당한다. |
| UMAP | 차원 축소 | 국소 구조를 보존하는 매니폴드(manifold) 학습. BERTopic에서 사용된다. |
| HDBSCAN | 밀도 군집화 | 가변 크기 군집을 찾는다. 이상치에 "잡음(noise)" 레이블(-1)을 만든다. |
| c_v coherence | 토픽 품질 지표 | 슬라이딩 윈도우 안에서 상위 토픽 단어의 평균 점별 상호 정보. |

## 더 읽을거리 (Further Reading)

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf): LDA 논문.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794): BERTopic 논문.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf): c_v와 그 동료들을 소개한 논문.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/): 프로덕션 레퍼런스. 훌륭한 예제들.
