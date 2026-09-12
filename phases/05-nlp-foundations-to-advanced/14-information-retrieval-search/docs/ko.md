# Information Retrieval and Search

> BM25는 정밀하지만 깨지기 쉽다. 밀집(dense) 방식은 넓은 그물을 던지지만 키워드를 놓친다. 하이브리드(hybrid)가 2026년의 기본값이다. 나머지는 전부 튜닝이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75분

## 문제 (The Problem)

사용자가 "what happens if someone lies to get money"라고 입력하면, 실제로 그 내용을 다루는 법조문을 찾아주기를 기대한다. "Section 420 IPC"이다. 키워드 검색은 이를 완전히 놓친다(공유하는 어휘가 없다). 의미 검색도 임베딩(embedding)을 법률 텍스트로 학습하지 않았다면 이를 놓친다. 실제 검색은 둘 다 다뤄야 한다.

IR(정보 검색)은 모든 RAG 시스템, 모든 검색창, 모든 문서 사이트의 퍼지 조회(fuzzy lookup) 아래에 깔린 파이프라인(pipeline)이다. 프로덕션(production)에서 작동하는 2026년 아키텍처는 단일 방법이 아니다. 서로 보완하는 방법들의 연쇄이며, 각 방법이 앞선 방법의 실패를 잡아낸다.

이 레슨은 각 조각을 만들면서 그것이 어떤 실패를 잡는지 짚는다.

## 개념 (The Concept)

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

네 개의 계층이다. 필요한 것을 골라라.

1. **희소 검색(Sparse retrieval, BM25).** 빠르고, 정확 일치에 정밀하며, 의미에는 형편없다. 역색인(inverted index) 위에서 실행한다. 수백만 문서에 대해 쿼리당 10ms 미만. 법조문 참조, 제품 코드, 오류 메시지, 개체명(named entity)을 정확히 잡아준다.
2. **밀집 검색(Dense retrieval).** 쿼리와 문서를 벡터(vector)로 인코딩한다. 최근접 이웃(nearest neighbor) 검색. 패러프레이즈(paraphrase)와 의미적 유사도(similarity)를 포착한다. 한 글자 차이의 정확 키워드 일치는 놓친다. FAISS나 벡터 DB로 쿼리당 50-200ms.
3. **융합(Fusion).** 희소 방식과 밀집 방식의 순위 목록을 병합한다. Reciprocal Rank Fusion(RRF)이 손쉬운 기본값인데, (서로 다른 척도에 사는) 원시 점수를 무시하고 순위 위치만 쓰기 때문이다. 한 신호가 해당 도메인에서 지배적임을 알 때는 가중 융합(weighted fusion)이 선택지다.
4. **크로스 인코더 리랭크(Cross-encoder rerank).** 융합에서 상위 30개를 가져온다. 크로스 인코더(query + document를 함께 넣어 각 쌍을 채점)를 실행한다. 상위 5개를 유지한다. 크로스 인코더는 쌍당 바이 인코더(bi-encoder)보다 느리지만 훨씬 정확하다. 상위 30개에만 실행해 비용을 분산한다.

3방향 검색(BM25 + 밀집 + SPLADE 같은 학습 기반 희소)은 2026년 벤치마크(benchmark)에서 2방향을 능가하지만, 학습 기반 희소 인덱스를 위한 인프라가 필요하다. 대부분의 팀에게는 2방향에 크로스 인코더 리랭크를 더한 것이 최적점이다.

```figure
gx-hybrid-retrieval
```

## 직접 만들기 (Build It)

### Step 1: 밑바닥부터 만드는 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

알아둘 만한 파라미터(parameter)가 둘 있다. `k1=1.5`는 단어 빈도(term-frequency) 포화(saturation)를 제어한다. 값이 높을수록 단어 반복에 더 큰 가중치를 둔다. `b=0.75`는 길이 정규화(length normalization)를 제어한다. 0은 문서 길이를 무시하고, 1은 완전히 정규화한다. 기본값은 원 논문에 나온 Robertson의 권장값이며 튜닝이 거의 필요 없다.

### Step 2: 바이 인코더를 사용한 밀집 검색

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

임베딩을 L2 정규화해 내적(dot product)이 코사인(cosine)과 같아지도록 한다. `all-MiniLM-L6-v2`는 384차원이고, 빠르며, 대부분의 영어 검색에 충분히 강력하다. 다국어 작업에는 `paraphrase-multilingual-MiniLM-L12-v2`를 쓰라. 최고 정확도에는 `bge-large-en-v1.5`나 `e5-large-v2`를 쓰라.

### Step 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 상수는 원래 RRF 논문에서 나왔다. `k`가 높으면 순위 차이의 기여가 평평해지고, 낮으면 상위 순위가 지배한다. 60은 공개된 기본값이며 튜닝이 거의 필요 없다.

### Step 4: 하이브리드 검색 + 리랭크

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

세 단계가 합쳐진다. BM25가 어휘적 일치를 찾는다. 밀집 방식이 의미적 일치를 찾는다. RRF가 점수 보정 없이 두 순위를 병합한다. 크로스 인코더가 쿼리-문서 쌍을 함께 써서 상위 30개를 재채점하는데, 바이 인코더가 놓친 세밀한 관련성을 포착한다. 상위 5개를 유지한다.

### Step 5: 평가

| 지표 | 의미 |
|--------|---------|
| Recall@k | 정답 문서가 존재하는 쿼리 중, 그것이 상위 k개에 들어가는 비율은? |
| MRR (Mean Reciprocal Rank) | 첫 번째 관련 문서의 1/순위의 평균. |
| nDCG@k | 단순 이진 관련/비관련이 아닌, 관련성의 등급을 고려한다. |

특히 RAG에서는 검색기(retriever)의 **Recall@k**가 가장 중요한 수치다. 올바른 지문이 검색된 집합에 없으면 리더(reader)는 답할 수 없다.

디버깅 팁: 실패하는 쿼리에 대해 희소 순위와 밀집 순위를 비교(diff)하라. 한쪽이 올바른 문서를 찾고 다른 쪽이 못 찾는다면, 어휘 불일치(해결책: 누락된 절반을 추가) 아니면 의미적 모호성(해결책: 더 나은 임베딩이나 리랭커)이 있는 것이다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 규모 | 스택 |
|-------|-------|
| 1k-100k 문서 | 인메모리 BM25 + `all-MiniLM-L6-v2` 임베딩 + RRF. 별도 DB 없음. |
| 100k-10M 문서 | 밀집용 FAISS 또는 pgvector + BM25용 Elasticsearch / OpenSearch. 병렬 실행. |
| 10M+ 문서 | 하이브리드 지원이 있는 Qdrant / Weaviate / Vespa / Milvus. 상위 30개에 크로스 인코더 리랭크. |
| 최고 품질 프런티어 | 3방향(BM25 + 밀집 + SPLADE) + ColBERT 후기 상호작용(late-interaction) 리랭킹 |

무엇을 고르든 평가를 위한 예산을 잡아라. 엔드투엔드 RAG 정확도를 벤치마킹하기 전에 검색 재현율을 벤치마킹하라. 리더는 검색기가 놓친 것을 고칠 수 없다.

### 2026년 프로덕션 RAG에서 얻은 값비싼 교훈

- **RAG 실패의 80%는 모델이 아니라 인제스션(ingestion)과 청킹(chunking)에서 비롯된다.** 팀은 LLM을 갈아치우고 프롬프트를 튜닝하며 몇 주를 보내는데, 그동안 검색은 세 번째 쿼리마다 조용히 잘못된 맥락을 반환한다. 청킹을 먼저 고쳐라.
- **청킹 전략이 청크 크기보다 더 중요하다.** 고정 크기 분할은 표, 코드, 중첩된 헤더를 망가뜨린다. 문장 인식(sentence-aware)이 기본값이다. 의미 기반이나 LLM 기반 청킹은 기술 문서와 제품 매뉴얼에서 성과를 낸다.
- **부모 문서 패턴(Parent-doc pattern).** 정밀도를 위해 작은 "자식" 청크를 검색한다. 같은 부모 섹션에서 여러 자식이 나타나면, 맥락을 보존하려고 부모 블록으로 교체한다. 재학습 없이 답 품질을 꾸준히 끌어올리는 방법이다.
- **k_rerank=3이 보통 최적이다.** 그 이상의 청크는 답 품질을 끌어올리지 않으면서 토큰 비용과 생성 지연 시간(latency)만 더한다. k=8이 여전히 k=3보다 낫다면, 리랭커가 제 성능을 못 내는 것이다.
- **HyDE / 쿼리 확장(query expansion).** 쿼리로부터 가상의 답을 생성하고, 그것을 임베딩한 뒤, 검색한다. 짧은 질문과 긴 문서 사이의 표현 격차를 메운다. 학습 없는 공짜 정밀도 향상.
- **8K 토큰 미만의 컨텍스트 예산.** 그 한계에서 일관되게 적중한다면 리랭커 임계값이 너무 느슨하다는 뜻이다.
- **모든 것을 버전 관리하라.** 프롬프트, 청킹 규칙, 임베딩 모델, 리랭커. 어떤 표류든 조용히 답 품질을 망가뜨린다. 충실성, 맥락 정밀도, 미답변 질문률에 대한 CI 게이트는 사용자가 보기 전에 회귀(regression)를 막는다.
- **3방향 검색(BM25 + 밀집 + SPLADE 같은 학습 기반 희소)은 2026년 벤치마크에서 2방향을 능가한다.** 특히 고유명사와 의미가 섞인 쿼리에서 그렇다. 인프라가 SPLADE 인덱스를 지원하면 배포하라.

적절한 검색 설계는 2026년 업계 측정에 따르면 환각(hallucination)을 70-90% 줄인다. 대부분의 RAG 성능 향상은 모델 파인튜닝(fine-tuning)이 아니라 더 나은 검색에서 나온다.

## 산출물 (Ship It)

`outputs/skill-retrieval-picker.md`로 저장하라:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 연습 문제 (Exercises)

1. **Easy.** 위의 `hybrid_search`를 500개 문서 코퍼스에 구현하라. 쿼리 20개를 테스트하라. BM25 단독, 밀집 단독, 하이브리드 사이의 recall at 5를 비교하라.
2. **Medium.** MRR 계산을 추가하라. 알려진 정답 문서가 있는 각 테스트 쿼리에 대해, BM25, 밀집, 하이브리드 순위에서 정답 문서의 순위를 찾아라. 각각의 MRR을 보고하라.
3. **Hard.** MultipleNegativesRankingLoss(Sentence Transformers)를 사용해 해당 도메인에서 밀집 인코더를 파인튜닝하라. 500개 쿼리-문서 쌍으로 학습 세트를 만들어라. 파인튜닝 전후의 재현율을 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| BM25 | 키워드 검색 | Okapi BM25. 단어 빈도, IDF, 길이로 문서를 채점한다. |
| Dense retrieval | 벡터 검색 | 쿼리 + 문서를 벡터로 인코딩하여 최근접 이웃을 찾는다. |
| Bi-encoder | 임베딩 모델 | 쿼리와 문서를 독립적으로 인코딩한다. 쿼리 시점에 빠르다. |
| Cross-encoder | 리랭커 모델 | 쿼리 + 문서를 함께 인코딩한다. 느리지만 정확하다. |
| RRF | 순위 융합 | `1/(k + rank)`를 합산하여 두 순위를 결합한다. |
| Recall@k | 검색 지표 | 관련 문서가 상위 k개에 있는 쿼리의 비율. |

## 더 읽을거리 (Further Reading)

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf): BM25에 대한 결정판 해설.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906): DPR, 표준 바이 인코더.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720): 밀집 방식과의 격차를 메우는 학습 기반 희소 검색기.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): RRF 논문.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832): 후기 상호작용 검색.
