# RAG를 위한 청킹 전략(Chunking Strategies)

> 청킹(chunking) 설정은 임베딩 모델(embedding model) 선택만큼이나 검색(retrieval) 품질에 영향을 준다(Vectara NAACL 2025). 청킹을 잘못하면 아무리 리랭킹(reranking)을 해도 구원받지 못한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60분

## 문제 (The Problem)

50쪽짜리 계약서를 RAG 시스템에 넣었다고 하자. 사용자가 묻는다. "해지 조항이 뭔가요?" 검색기는 표지를 반환한다. 왜? 모델이 512 토큰(token) 청크로 학습되었는데, 해지 조항은 20쪽 안쪽에 있고, 페이지 구분선을 가로질러 쪼개져 있으며, 쿼리와 연결되는 국소 키워드가 없기 때문이다.

해결책은 "더 좋은 임베딩 모델을 사라"가 아니다. 해결책은 청킹이다. 얼마나 크게? 중첩(overlap)은? 어디서 나눌까? 주변 컨텍스트는 함께?

2026년 2월 벤치마크(benchmark)는 놀라운 결과를 보여준다.

- Vectara의 2026년 연구: 재귀적(recursive) 512 토큰 청킹이 의미적(semantic) 청킹을 정확도 69% → 54%로 앞섰다.
- Natural Questions에서의 SPLADE + Mistral-8B: 중첩은 측정 가능한 이득을 전혀 제공하지 않았다.
- 컨텍스트 절벽(context cliff): 컨텍스트 약 2,500 토큰 부근에서 응답 품질이 급격히 떨어진다.

"뻔한" 답(의미적 청킹, 20% 중첩, 1000 토큰)이 종종 틀린다. 이 레슨은 여섯 가지 전략에 대한 직관을 쌓고, 언제 어떤 것에 손을 뻗어야 하는지 알려준다.

## 개념 (The Concept)

![하나의 단락에 대해 시각화된 여섯 가지 청킹 전략](../assets/chunking.svg)

**고정 청킹(Fixed chunking).** N개의 문자 또는 토큰마다 나눈다. 가장 단순한 베이스라인(baseline). 문장 중간을 끊는다. 압축은 좋지만 일관성은 나쁘다.

**재귀적(Recursive).** LangChain의 `RecursiveCharacterTextSplitter`. 먼저 `\n\n`에서 나누고, 그다음 `\n`, 그다음 `.`, 그다음 공백에서 나눈다. 깔끔하게 후퇴한다(falls back). 2026년의 기본값.

**의미적(Semantic).** 각 문장을 임베딩한다. 인접한 문장 사이의 코사인 유사도(cosine similarity)를 계산한다. 유사도가 임곗값 아래로 떨어지는 곳에서 나눈다. 주제 일관성을 보존한다. 느리고, 때때로 검색에 해가 되는 40 토큰짜리 작은 파편을 만든다.

**문장(Sentence).** 문장 경계에서 나눈다. 청크당 한 문장 또는 N개 문장의 윈도우. 비용의 일부만으로 약 5k 토큰까지 의미적 청킹과 맞먹는다.

**부모 문서(Parent-document).** 검색을 위한 작은 자식 청크 *그리고* 컨텍스트를 위한 더 큰 부모 청크를 저장한다. 자식으로 검색하고, 부모를 반환한다. 우아하게 성능이 저하된다. 나쁜 자식 청크도 여전히 그럴듯한 부모를 반환한다.

**후기 청킹(Late chunking, 2024).** 먼저 전체 문서를 토큰 수준에서 임베딩한 다음, 토큰 임베딩을 청크 임베딩으로 풀링(pooling)한다. 청크 간 컨텍스트를 보존한다. 긴 컨텍스트 임베더(BGE-M3, Jina v3)와 함께 동작한다. 연산량이 더 크다.

**컨텍스트 검색(Contextual retrieval, Anthropic, 2024).** 각 청크 앞에, 문서 내 위치를 LLM이 생성한 요약을 붙인다("이 청크는 해지 조항의 3.2절입니다..."). Anthropic 자체 벤치마크에서 35~50% 검색 향상. 인덱싱 비용이 크다.

### 모든 기본값을 이기는 규칙

청크 크기를 쿼리 유형에 맞춰라.

| 쿼리 유형 | 청크 크기 |
|------------|-----------|
| 단답형(factoid, "CEO 이름이 뭐지?") | 256~512 토큰 |
| 분석적 / 다중 홉(multi-hop) | 512~1024 토큰 |
| 전체 절(section) 이해 | 1024~2048 토큰 |

NVIDIA의 2026년 벤치마크. 청크는 답변과 국소 컨텍스트를 담을 만큼 충분히 크되, 검색기의 top-K 반환이 컨텍스트 노이즈가 아니라 답변에 집중할 만큼 작아야 한다.

```figure
n5-chunk-cuts
```

## 직접 만들기 (Build It)

### Step 1: 고정 및 재귀적 청킹

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Step 2: 의미적 청킹

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

`threshold`를 도메인에 맞게 튜닝하라. 너무 높으면 → 파편. 너무 낮으면 → 하나의 거대한 청크.

### Step 3: 부모 문서

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

핵심 통찰: 부모를 중복 제거하라. 여러 자식이 같은 부모에 매핑될 수 있다. 전부 반환하면 컨텍스트를 낭비한다.

### Step 4: 컨텍스트 검색(Anthropic 패턴)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

컨텍스트화된 청크를 인덱싱하라. 쿼리 시점에 검색은 추가된 주변 신호로부터 이득을 본다.

### Step 5: 평가

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

항상 벤치마크하라. 코퍼스에 따라 "최고" 전략은 어떤 블로그 글과도 맞지 않을 수 있다.

## 함정 (Pitfalls)

- **단답형 쿼리로만 청킹을 평가.** 다중 홉 쿼리는 매우 다른 승자를 드러낸다. 쿼리 유형으로 계층화된(stratified) 평가 세트를 사용하라.
- **최소 크기 없는 의미적 청킹.** 검색에 해가 되는 40 토큰짜리 파편을 만든다. 항상 `min_tokens`를 강제하라.
- **카고 컬트(cargo cult)식 중첩.** 2026년 연구는 중첩이 종종 이득을 전혀 주지 않으면서 인덱스 비용을 두 배로 만든다고 밝혔다. 가정하지 말고 측정하라.
- **최소/최대 강제 없음.** 5 토큰짜리 청크든 5000 토큰짜리 청크든 둘 다 검색을 망친다. 범위를 제한(clamp)하라.
- **문서 간 청킹.** 절대 청크가 두 문서에 걸치게 하지 마라. 항상 문서별로 청킹한 다음 병합하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 전략 |
|-----------|----------|
| 첫 구축, 알 수 없는 코퍼스 | 재귀적, 512 토큰, 중첩 없음 |
| 단답형 QA | 재귀적, 256~512 토큰 |
| 분석적 / 다중 홉 | 재귀적, 512~1024 토큰 + 부모 문서 |
| 강한 상호 참조(계약서, 논문) | 후기 청킹 또는 컨텍스트 검색 |
| 대화형 / 다이얼로그 코퍼스 | 턴 수준 청크 + 화자 메타데이터 |
| 짧은 발화(트윗, 리뷰) | 한 문서 = 한 청크 |

재귀적 512로 시작하라. 50개 쿼리 평가 세트에서 recall@5를 측정하라. 거기서부터 튜닝하라.

## 산출물 (Ship It)

`outputs/skill-chunker.md`로 저장한다:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 연습 문제 (Exercises)

1. **쉬움.** 20쪽짜리 문서 하나를 fixed(512, 0), recursive(512, 0), recursive(512, 100)으로 청킹하라. 청크 개수와 경계 품질을 비교하라.
2. **보통.** 문서 5개에 대해 30개 쿼리 평가 세트를 만들어라. 재귀적, 의미적, 부모 문서에 대해 recall@5를 측정하라. 어느 것이 이기는가? 블로그 글과 맞는가?
3. **어려움.** 컨텍스트 검색을 구현하라. 베이스라인 재귀적 대비 MRR 향상을 측정하라. 인덱스 비용(LLM 호출)과 정확도 이득을 비교해 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 청크(Chunk) | 문서의 한 조각 | 임베딩되고, 인덱싱되고, 검색되는 하위 문서 단위. |
| 중첩(Overlap) | 안전 여유 | 인접한 청크 간에 공유되는 N개 토큰. 2026년 벤치마크에서 종종 쓸모없다. |
| 의미적 청킹(Semantic chunking) | 똑똑한 청킹 | 인접 문장 임베딩 유사도가 떨어지는 곳에서 나눈다. |
| 부모 문서(Parent-document) | 2단계 검색 | 작은 자식을 검색하고, 더 큰 부모를 반환한다. |
| 후기 청킹(Late chunking) | 임베딩 후 청킹 | 전체 문서를 토큰 수준에서 임베딩하고, 청크 벡터로 풀링한다. |
| 컨텍스트 검색(Contextual retrieval) | Anthropic의 트릭 | 인덱싱 전에 각 청크 앞에 LLM이 생성한 요약을 붙인다. |
| 컨텍스트 절벽(Context cliff) | 2500 토큰 벽 | RAG에서 컨텍스트 약 2.5k 토큰 부근에서 관찰된 품질 하락(2026년 1월). |

## 더 읽을거리 (Further Reading)

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — the default in production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking matters as much as embedding choice.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — the late chunking paper.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 35-50% retrieval improvement with LLM-generated context prefixes.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — chunk size by query type.
