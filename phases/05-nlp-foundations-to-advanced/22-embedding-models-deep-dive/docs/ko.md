# 임베딩 모델(Embedding Models): 2026년 심층 분석

> Word2Vec은 단어당 하나의 벡터(vector)를 주었다. 현대 임베딩 모델은 단락당 하나의 벡터를 준다. 교차 언어(cross-lingual)이고, 희소(sparse)·밀집(dense)·다중 벡터(multi-vector) 관점을 제공하며, 인덱스에 맞는 크기로 조절된다. 잘못 고르면 RAG가 엉뚱한 것을 검색한다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60분

## 문제 (The Problem)

RAG 시스템은 40%의 경우 엉뚱한 단락(passage)을 검색한다. 범인은 벡터 데이터베이스나 프롬프트(prompt)인 경우가 드물다. 임베딩 모델이다.

2026년에 임베딩을 고른다는 것은 다섯 가지 축에 걸쳐 선택한다는 뜻이다.

1. **밀집 vs 희소 vs 다중 벡터.** 단락당 하나의 벡터, 또는 토큰(token)당 하나, 또는 희소하게 가중된 단어 가방(bag of words).
2. **언어 커버리지.** 영어 전용 과제에서는 여전히 단일 언어 영어 모델이 이긴다. 코퍼스가 섞여 있을 때는 다국어 모델이 이긴다.
3. **컨텍스트 길이.** 512 토큰 vs 8,192 vs 32,768: 그리고 실제 유효 용량은 광고된 최댓값의 60~70%인 경우가 많다.
4. **차원 예산.** 전체 정밀도(full precision)의 3,072개 부동소수 = 벡터당 12 KB. 1억 개 벡터면 저장 비용이 월 $1,300이다. 마트료시카(Matryoshka) 절단으로 이를 4배 줄인다.
5. **오픈 vs 호스팅.** 오픈 웨이트(open-weight)는 스택과 데이터를 직접 통제한다는 뜻이다. 호스팅은 통제권을 항상 최신 상태와 맞바꾼다는 뜻이다.

이 레슨은 트레이드오프(trade-off)에 이름을 붙여, 지난 분기에 인기 있었던 것이 아니라 증거에 따라 고를 수 있게 한다.

## 개념 (The Concept)

![밀집, 희소, 다중 벡터 임베딩](../assets/embedding-modes.svg)

**밀집 임베딩(Dense embeddings).** 단락당 하나의 벡터(보통 384~3,072차원). 코사인 유사도(cosine similarity)가 의미적 근접성으로 단락의 순위를 매긴다. OpenAI `text-embedding-3-large`, BGE-M3 밀집 모드, Voyage-3. 기본 선택지.

**희소 임베딩(Sparse embeddings).** SPLADE 스타일. 트랜스포머(transformer)가 모든 어휘 토큰에 대해 가중치를 예측한 다음, 대부분을 0으로 만든다. 결과는 |vocab| 크기의 희소 벡터다. (BM25처럼) 어휘적 매칭을 포착하되, 학습된 용어 가중치를 사용한다. 키워드 중심 쿼리에 강하다.

**다중 벡터(후기 상호작용, late interaction).** ColBERTv2, Jina-ColBERT. 토큰당 하나의 벡터. MaxSim으로 점수를 매긴다. 각 쿼리 토큰에 대해 가장 유사한 문서 토큰을 찾아 점수를 합산한다. 저장과 채점 비용이 더 크지만, 긴 쿼리와 도메인 특화 코퍼스에서 이긴다.

**BGE-M3: 세 가지를 한 번에.** 단일 모델이 밀집, 희소, 다중 벡터 표현을 동시에 출력한다. 각각을 독립적으로 쿼리할 수 있으며, 점수는 가중합으로 융합된다. 하나의 체크포인트에서 유연성을 원할 때 2026년의 기본값.

**마트료시카 표현 학습(Matryoshka Representation Learning).** 벡터의 처음 N개 차원이 그 자체로 유용한 독립 임베딩을 이루도록 학습된다. 1,536차원 벡터를 256차원으로 절단하고 약 1%의 정확도를 지불해 저장 공간을 6배 절약한다. OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+가 지원한다.

### MTEB 리더보드는 이야기의 일부만 들려준다

대규모 텍스트 임베딩 벤치마크(Massive Text Embedding Benchmark)(출시 당시(2022) 8개 과제 유형에 걸친 56개 과제, MTEB v2에서 100개 이상으로 확장. 2026년 초 기준, Gemini Embedding 2가 검색(retrieval)에서 1위(67.71 MTEB-R)다. Cohere embed-v4가 범용(general)에서 선두(65.2 MTEB)다. BGE-M3가 오픈 웨이트 다국어에서 선두(63.0)다. 리더보드는 필요하지만 충분하지는 않다) 항상 자신의 도메인에서 벤치마크(benchmark)하라.

### 3계층 패턴

| 사용 사례 | 패턴 |
|----------|---------|
| 빠른 1차 통과 | 밀집 바이 인코더(bi-encoder) (BGE-M3, text-3-small) |
| 재현율(recall) 향상 | 희소(SPLADE, BGE-M3 sparse) + RRF 융합 |
| 상위 50개에 대한 정밀도 | 다중 벡터(ColBERTv2) 또는 교차 인코더(cross-encoder) 리랭커(reranker) |

대부분의 프로덕션(production) 스택은 셋 다 사용한다.

```figure
gx-matryoshka
```

## 직접 만들기 (Build It)

### Step 1: 베이스라인: Sentence-BERT 밀집 임베딩

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`는 내적(dot product)을 코사인 유사도와 같게 만든다. 항상 설정하라.

### Step 2: 마트료시카 절단

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

절단 후 다시 정규화(normalization)하라. Nomic v1.5, OpenAI text-3, Voyage-4는 처음 몇 단계까지는 이것이 무손실이 되도록 학습되었다. 마트료시카가 아닌 모델(원본 Sentence-BERT)은 절단 시 급격히 성능이 떨어진다.

### Step 3: BGE-M3 다기능성

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

세 개의 인덱스, 하나의 추론(inference) 호출. 점수 융합:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

가중치는 자신의 도메인에서 튜닝하라.

### Step 4: 커스텀 과제에 대한 MTEB 평가

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

후보 모델을 *대표적인* 부분집합에서 실행하라. 리더보드 순위만 믿지 마라. 자신의 도메인이 중요하다.

### Step 5: 밑바닥부터 만든 코사인

`code/main.py`를 보라. 평균화된 해싱 트릭(Hashing Trick) 임베딩(표준 라이브러리만 사용). 트랜스포머 임베딩과 경쟁할 수준은 아니지만, 형태를 보여준다. 토큰화 → 벡터 → 정규화 → 내적.

## 함정 (Pitfalls)

- **쿼리와 문서에 같은 모델.** 일부 모델(Voyage, Jina-ColBERT)은 비대칭 인코딩(asymmetric encoding)을 사용한다. 쿼리와 문서가 서로 다른 경로를 통과한다. 항상 모델 카드를 확인하라.
- **접두사 누락.** `bge-*` 모델은 쿼리 앞에 `"Represent this sentence for searching relevant passages: "`를 붙여야 한다. 잊으면 재현율이 3~5점 벌어진다.
- **마트료시카 과도한 절단.** 1,536 → 256은 보통 안전하다. 1,536 → 64는 아니다. 자신의 평가 세트에서 검증하라.
- **컨텍스트 절단.** 대부분의 모델은 최대 길이를 초과하는 입력을 조용히 잘라낸다. 긴 문서는 청킹(chunking)이 필요하다(레슨 23 참고).
- **지연 시간 꼬리(latency tail) 무시.** MTEB 점수는 p99 지연 시간(latency)을 숨긴다. 600M 모델이 335M 모델을 2점 앞설 수 있지만 쿼리당 비용이 3배 더 들 수 있다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어 전용, 빠름, API | `text-embedding-3-large` 또는 `voyage-3-large` |
| 오픈 웨이트, 영어 | `BAAI/bge-large-en-v1.5` |
| 오픈 웨이트, 다국어 | `BAAI/bge-m3` 또는 `Qwen3-Embedding-8B` |
| 긴 컨텍스트(32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU 전용 배포 | Nomic Embed v2 (137M 파라미터, MoE) |
| 저장 공간 제약 | 마트료시카 절단 + int8 양자화(quantization) |
| 키워드 중심 쿼리 | SPLADE 희소를 추가하고, 밀집과 RRF 융합 |

2026년 패턴: BGE-M3나 text-3-large로 시작하고, MTEB로 자신의 도메인에서 평가하며, 도메인 특화 모델이 3점 넘게 이기면 교체하라.

## 산출물 (Ship It)

`outputs/skill-embedding-picker.md`로 저장한다:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 연습 문제 (Exercises)

1. **쉬움.** `bge-small-en-v1.5`로 문장 100개를 전체 차원(384)에서 인코딩한 다음, 마트료시카 128에서 인코딩하라. 쿼리 10개에 대한 MRR 하락을 측정하라.
2. **보통.** 자신의 도메인에서 단락 500개에 대해 BGE-M3 밀집, 희소, colbert를 비교하라. recall@10에서 어느 것이 이기는가? RRF 융합이 최고의 단일 모드를 이기는가?
3. **어려움.** 자신의 상위 2개 도메인 과제에 걸쳐 후보 모델 세 개로 MTEB를 실행하라. MTEB 점수, 100개 쿼리 배치에서의 p99 지연 시간, 100만 쿼리당 비용($)을 보고하라. 파레토 최적(Pareto-optimal)인 것을 골라라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 밀집 임베딩(Dense embedding) | 그 벡터 | 텍스트당 하나의 고정 크기 벡터. 순위 매기기에 코사인 유사도. |
| 희소 임베딩(Sparse embedding) | 학습된 BM25 | 어휘 토큰당 하나의 가중치. 대부분 0. 종단 간(end-to-end) 학습됨. |
| 다중 벡터(Multi-vector) | ColBERT 스타일 | 토큰당 하나의 벡터. MaxSim 채점. 더 큰 인덱스, 더 나은 재현율. |
| 마트료시카(Matryoshka) | 러시아 인형 트릭 | 처음 N개 차원이 그 자체로 유효한 더 작은 임베딩이다. |
| MTEB | 그 벤치마크 | 대규모 텍스트 임베딩 벤치마크: 출시 당시 56개 과제, v2에서 100개 이상. |
| BEIR | 그 검색 벤치마크 | 18개의 제로샷(zero-shot) 검색 과제. 교차 도메인 강건성에 자주 인용됨. |
| 비대칭 인코딩(Asymmetric encoding) | 쿼리 ≠ 문서 경로 | 모델이 쿼리와 문서에 서로 다른 투영(projection)을 사용한다. |

## 더 읽을거리 (Further Reading)

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — the bi-encoder paper.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — the leaderboard paper.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — the unified three-mode model.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — the dimension-ladder training objective.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction in production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live rankings.
