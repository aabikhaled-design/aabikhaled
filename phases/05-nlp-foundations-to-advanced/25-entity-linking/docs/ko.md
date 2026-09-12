# 엔티티 링킹 및 중의성 해소(Entity Linking & Disambiguation)

> NER이 "Paris"를 찾았다. 그게 어느 Paris인지를 가려내는 일이 엔티티 링킹(entity linking)이다. 프랑스의 Paris? Paris Hilton? 텍사스의 Paris? Paris(트로이의 왕자)? 링킹이 없으면 지식 그래프(knowledge graph)는 모호한 채로 남는다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60분

## 문제 (The Problem)

한 문장이 이렇다. "Jordan beat the press." NER이 "Jordan"을 PERSON으로 태깅한다. 좋다. 그런데 *어떤* Jordan인가?

- Michael Jordan(농구)?
- Michael B. Jordan(배우)?
- Michael I. Jordan(버클리 ML 교수: 그렇다, ML 논문에서 이 혼동은 실제로 일어난다)?
- Jordan(국가)?
- Jordan(히브리어 이름)?

엔티티 링킹(EL)은 각 언급(mention)을 지식 베이스(knowledge base)의 고유 항목으로 해결한다. Wikidata, Wikipedia, DBpedia, 또는 도메인 KB. 두 개의 하위 과제:

1. **후보 생성(Candidate generation).** "Jordan"이 주어졌을 때, 어떤 KB 항목들이 그럴듯한가?
2. **중의성 해소(Disambiguation).** 컨텍스트가 주어졌을 때, 어떤 후보가 옳은 것인가?

두 단계 모두 학습 가능하다. 둘 다 벤치마크(benchmark)된다. 결합된 파이프라인(pipeline)은 10년간 안정적이었다. 바뀌는 것은 중의성 해소기의 품질이다.

## 개념 (The Concept)

![엔티티 링킹 파이프라인: 언급 → 후보 → 중의성 해소된 엔티티](../assets/entity-linking.svg)

**후보 생성.** 언급 표층형("Jordan")이 주어지면, 별칭 인덱스(alias index)에서 후보를 조회한다. Wikipedia 별칭 사전은 대부분의 명명된 엔티티를 포괄한다: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK 공항, JFK(영화). 전형적인 인덱스는 언급당 10~30개의 후보를 반환한다.

**중의성 해소: 세 가지 접근법.**

1. **사전 확률 + 컨텍스트(Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. 잘 동작하고, 빠르며, 학습이 필요 없다.
2. **임베딩 기반(ESS / REL / Blink).** 언급 + 컨텍스트를 인코딩한다. 각 후보의 설명을 인코딩한다. 최대 코사인(cosine)을 고른다. 2020~2024년의 기본값.
3. **생성형(GENRE, 2021; LLM 기반, 2023+).** 엔티티의 표준 이름을 토큰(token) 단위로 디코딩한다. 출력이 유효한 KB id임이 보장되도록 유효한 엔티티 이름의 트라이(trie)로 제약한다.

**종단 간(end-to-end) vs 파이프라인.** 현대 모델(ELQ, BLINK, ExtEnD, GENRE)은 NER + 후보 생성 + 중의성 해소를 한 번에 실행한다. 컴포넌트를 교체할 수 있기 때문에 프로덕션(production)에서는 여전히 파이프라인 시스템이 우세하다.

### 두 가지 측정

- **언급 재현율(후보 생성).** 정답(gold) 언급 중 올바른 KB 항목이 후보 목록에 나타나는 비율. 전체 파이프라인의 하한.
- **중의성 해소 정확도 / F1.** 올바른 후보가 주어졌을 때, top-1이 옳은 빈도.

항상 둘 다 보고하라. 후보 재현율 80%에서 중의성 해소 99%인 시스템은 80% 파이프라인이다.

```figure
gx-entity-linking
```

## 직접 만들기 (Build It)

### Step 1: Wikipedia 리다이렉트로부터 별칭 인덱스 구축

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 별칭 데이터: 약 1,800만 개의 (별칭, 엔티티) 쌍. Wikidata 덤프에서 다운로드한다. 역색인(inverted index)으로 저장한다.

### Step 2: 컨텍스트 기반 중의성 해소

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

자카드(Jaccard) 중첩은 장난감이다. 임베딩에 대한 코사인 유사도로 교체하라(트랜스포머 버전은 `code/main.py`의 step-2 참고).

### Step 3: 임베딩 기반(BLINK 스타일)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

인덱싱 시점에 모든 KB 엔티티를 한 번 임베딩한다. 쿼리 시점에 언급 + 컨텍스트를 한 번 임베딩하고, 후보 풀에 대해 내적(dot-product)하여 최댓값을 고른다.

### Step 4: 생성형 엔티티 링킹(개념)

GENRE는 엔티티의 Wikipedia 제목을 문자 단위로 디코딩한다. 제약 디코딩(constrained decoding, 레슨 20 참고)은 유효한 제목만 출력될 수 있도록 보장한다. KB를 기반으로 한 트라이와 긴밀하게 통합된다. 현대의 후예는 REL-GEN과 구조화 출력을 쓴 LLM 프롬프트 EL이다.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

화이트리스트(Outlines `choice`)와 결합하면, 이것이 2026년에 출시하기 가장 단순한 EL 파이프라인이다.

### Step 5: AIDA-CoNLL에서 평가

AIDA-CoNLL은 표준 EL 벤치마크다. 로이터 기사 1,393개, 언급 3만 4천 개, Wikipedia 엔티티. KB 내(in-KB) 정확도(`P@1`)와 KB 외(out-of-KB) NIL 탐지율을 보고한다.

## 함정 (Pitfalls)

- **NIL 처리.** 일부 언급은 KB에 없다(신흥 엔티티, 무명 인물). 시스템은 엉뚱한 엔티티를 추측하는 대신 NIL을 예측해야 한다. 별도로 측정된다.
- **언급 경계 오류.** 상류의 NER이 부분 스팬(span)을 놓친다("Bank of America"가 그냥 "Bank"로 태깅됨). EL 재현율이 떨어진다.
- **인기도 편향(popularity bias).** 학습된 시스템은 빈번한 엔티티를 과도하게 예측한다. ML 논문의 "Michael I. Jordan" 언급이 종종 농구 Jordan으로 링크된다.
- **교차 언어 EL.** 중국어 텍스트의 언급을 영어 Wikipedia 엔티티에 매핑하기. 다국어 인코더나 번역 단계가 필요하다.
- **KB 노후화.** 새 회사, 사건, 인물은 작년 Wikipedia 덤프에 없다. 프로덕션 파이프라인은 갱신 루프가 필요하다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 범용 영어 + Wikipedia | BLINK 또는 REL |
| 교차 언어, KB = Wikipedia | mGENRE |
| LLM 친화적, 하루 소수 언급 | 후보 목록 + 제약된 JSON으로 Claude/GPT-4 프롬프트 |
| 도메인 특화 KB(의료, 법률) | KB 인식 검색 + 도메인 AIDA 스타일 세트로 파인튜닝(fine-tuning)한 커스텀 BERT |
| 극도로 낮은 지연 시간(latency) | 정확 일치 사전 확률만(Milne-Witten 베이스라인) |
| 연구 SOTA | GENRE / ExtEnD / 생성형 LLM-EL |

2026년에 출시되는 프로덕션 패턴: NER → 상호참조(coref) → 각 언급에 대한 EL → 클러스터를 클러스터당 하나의 표준 엔티티로 축소. 출력: 언급당 하나가 아니라 문서 내 엔티티당 하나의 KB id.

## 산출물 (Ship It)

`outputs/skill-entity-linker.md`로 저장한다:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 사전 확률+컨텍스트 중의성 해소기를 모호한 언급 10개(Paris, Jordan, Apple)에 대해 구현하라. 올바른 엔티티를 직접 레이블링하라. 정확도를 측정하라.
2. **보통.** 문장 트랜스포머로 모호한 언급 50개를 인코딩하라. 각 후보의 설명을 임베딩하라. 임베딩 기반 중의성 해소를 자카드 컨텍스트 중첩과 비교하라.
3. **어려움.** 1천 엔티티 규모의 도메인 KB(예: 회사의 직원 + 제품)를 구축하라. NER + EL을 종단 간으로 구현하라. 별도로 분리된 문장 100개에 대해 정밀도와 재현율을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 엔티티 링킹(Entity linking, EL) | Wikipedia에 링크하기 | 언급을 고유한 KB 항목에 매핑한다. |
| 후보 생성(Candidate generation) | 누구일 수 있을까? | 언급에 대해 그럴듯한 KB 항목의 짧은 목록을 반환한다. |
| 중의성 해소(Disambiguation) | 옳은 것 고르기 | 컨텍스트로 후보를 채점해 승자를 고른다. |
| 별칭 인덱스(Alias index) | 그 조회 테이블 | 표층형 → 후보 엔티티로의 매핑. |
| NIL | KB에 없음 | 일치하는 KB 항목이 없다는 명시적 예측. |
| KB | 지식 베이스 | Wikidata, Wikipedia, DBpedia, 또는 도메인 KB. |
| AIDA-CoNLL | 그 벤치마크 | 정답 엔티티 링크가 달린 로이터 기사 1,393개. |

## 더 읽을거리 (Further Reading)

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — the foundational prior+context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — the embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — generative EL with constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — the benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — the open production stack.
