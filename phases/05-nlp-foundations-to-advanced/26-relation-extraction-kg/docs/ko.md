# 관계 추출 및 지식 그래프 구축(Relation Extraction & Knowledge Graph Construction)

> NER이 엔티티(entity)를 찾았다. 엔티티 링킹(entity linking)이 그 엔티티를 고정했다. 관계 추출(relation extraction)은 엔티티 사이의 간선(edge)을 찾는다. 지식 그래프(knowledge graph)는 노드(node), 간선, 그 출처(provenance)의 합이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60분

## 문제 (The Problem)

한 분석가가 읽는다. "Tim Cook became CEO of Apple in 2011." 네 가지 사실:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

관계 추출(RE)은 자유 텍스트를 구조화된 트리플(triple) `(subject, relation, object)`로 바꾼다. 코퍼스 전체에 걸쳐 모으면 지식 그래프가 된다. 이를 모아 쿼리하면 RAG, 분석, 컴플라이언스 감사를 위한 추론 기반이 된다.

2026년의 문제는 이렇다. LLM은 관계를 열정적으로 추출한다. 너무 열정적으로. 원본 텍스트에 근거가 없는 트리플을 환각(hallucination)한다. 출처가 없으면 진짜 트리플과 그럴듯한 허구를 구별하지 못한다. 2026년의 답은 AEVS 스타일의 앵커-검증(anchor-and-verify) 파이프라인이다.

## 개념 (The Concept)

![텍스트 → 트리플 → 지식 그래프](../assets/relation-extraction.svg)

**트리플 형식.** `(subject_entity, relation_type, object_entity)`. 관계는 닫힌 온톨로지(closed ontology, Wikidata 속성, FIBO, UMLS)나 열린 집합(open set, OpenIE 스타일로 무엇이든 가능)에서 온다.

**세 가지 추출 접근법.**

1. **규칙 / 패턴 기반.** Hearst 패턴: "X such as Y" → `(Y, isA, X)`. 여기에 손으로 만든 정규식(regex)을 더한다. 부서지기 쉽고, 정밀하며, 설명 가능하다.
2. **지도 학습 분류기.** 문장 내 두 엔티티 언급(mention)이 주어지면, 고정된 집합에서 관계를 예측한다. TACRED, ACE, KBP로 학습한다. 2015~2022년의 표준이었다.
3. **생성형 LLM.** 모델이 트리플을 내보내도록 프롬프트한다. 바로 동작한다. 출처가 필요하며, 없으면 그럴듯해 보이는 쓰레기를 환각한다.

**AEVS(Anchor-Extraction-Verification-Supplement, 2026).** 현재의 환각 완화 프레임워크다.

- **앵커(Anchor).** 모든 엔티티 스팬(span)과 관계 구문 스팬을 정확한 위치와 함께 식별한다.
- **추출(Extract).** 앵커 스팬에 연결된 트리플을 생성한다.
- **검증(Verify).** 각 트리플 요소를 원본 텍스트로 되돌려 대조한다. 뒷받침되지 않는 것은 거부한다.
- **보완(Supplement).** 커버리지 패스로 앵커된 스팬이 빠지지 않도록 보장한다.

환각이 급격히 줄어든다. 연산량이 더 들지만 감사 가능하다.

**열림 vs 닫힘 트레이드오프(trade-off).**

- **닫힌 온톨로지.** 고정된 속성 목록(예: Wikidata의 11,000개 이상의 속성). 예측 가능하다. 쿼리 가능하다. 발명하기 어렵다.
- **Open IE.** 모든 동사 구문이 관계가 된다. 높은 재현율(recall). 낮은 정밀도. 쿼리하기 지저분하다.

프로덕션(production) KG는 보통 둘을 섞는다. 발견에는 open IE를 쓴 다음, 메인 그래프에 병합하기 전에 관계를 닫힌 온톨로지로 표준화한다.

```figure
relation-triples
```

## 직접 만들기 (Build It)

### Step 1: 패턴 기반 추출

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

전체 장난감 추출기는 `code/main.py`를 보라. Hearst 패턴은 디버깅이 쉬워서 여전히 도메인 특화 파이프라인에 들어간다.

### Step 2: 지도 학습 관계 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL은 seq2seq 관계 추출기다. 텍스트를 입력하면 트리플을 출력하는데, 이미 Wikidata 속성 id 형태다. 원거리 지도(distant-supervision) 데이터로 파인튜닝(fine-tuning)했다. 표준 오픈 웨이트(open-weights) 베이스라인(baseline)이다.

### Step 3: 앵커링을 동반한 LLM 프롬프트 추출

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

반환된 모든 스팬을 원본에 대조해 검증하라. `text[start:end] != triple_entity`인 것은 거부하라. AEVS의 "검증" 단계를 최소한의 형태로 구현한 것이다.

### Step 4: 닫힌 온톨로지로 표준화

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

표준화가 엔지니어링 작업의 60~80%를 차지할 때가 많다. 그만큼 예산을 잡아라.

### Step 5: 작은 그래프를 만들고 쿼리하기

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

이것이 모든 RAG-over-KG 시스템의 원자다. RDF 트리플 저장소(Blazegraph, Virtuoso), 속성 그래프(Neo4j), 벡터 증강 그래프 저장소로 규모를 확장하라.

## 함정 (Pitfalls)

- **RE 이전의 상호참조(coreference).** "He founded Apple": RE는 "he"가 누구인지 알아야 한다. 상호참조를 먼저 실행하라(레슨 24).
- **엔티티 표준화.** "Apple Inc"와 "Apple"은 같은 노드로 해결되어야 한다. 엔티티 링킹을 먼저 하라(레슨 25).
- **환각된 트리플.** LLM은 텍스트가 뒷받침하지 않는 트리플을 내보낸다. 스팬 검증을 강제하라.
- **관계 표준화 표류(drift).** Open IE 관계는 일관성이 없다("was born in", "came from", "is a native of"). 표준 id로 축소하지 않으면 그래프를 쿼리할 수 없다.
- **시간 오류.** "Tim Cook is CEO of Apple": 지금은 참이지만 2005년에는 거짓이었다. 많은 관계가 시간으로 한정된다. 한정어(Wikidata의 `P580` 시작 시간, `P582` 종료 시간)를 사용하라.
- **도메인 불일치.** REBEL은 Wikipedia로 학습했다. 법률, 의료, 과학 텍스트는 도메인 파인튜닝된 RE 모델이 필요할 때가 많다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 빠른 프로덕션, 일반 도메인 | Wikidata 표준화를 동반한 REBEL 또는 LlamaPred |
| 도메인 특화(바이오메드, 법률) | SciREX 스타일 도메인 파인튜닝 + 커스텀 온톨로지 |
| LLM 프롬프트, 감사된 출력 | AEVS 파이프라인: 앵커 → 추출 → 검증 → 보완 |
| 대량 뉴스 IE | 패턴 기반 + 지도 학습 하이브리드 |
| 밑바닥부터 KG 구축 | Open IE + 수동 표준화 패스 |
| 시간 KG | 한정어와 함께 추출(시작/종료 시간, 시점) |

통합 패턴: NER → 상호참조 → 엔티티 링킹 → 관계 추출 → 온톨로지 매핑 → 그래프 적재. 모든 단계가 잠재적 품질 게이트다.

## 산출물 (Ship It)

`outputs/skill-re-designer.md`로 저장한다:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 패턴 추출기를 뉴스 기사 문장 5개에 대해 실행하라. 정밀도를 손으로 확인하라.
2. **보통.** 같은 문장에 REBEL(또는 작은 LLM)을 사용하라. 트리플을 비교하라. 어느 추출기가 정밀도가 더 높은가? 재현율이 더 높은 쪽은?
3. **어려움.** AEVS 파이프라인을 만들어라: LLM으로 추출 + 원본에 대한 스팬 검증. Wikipedia 스타일 문장 50개에서 검증 단계 이전과 이후의 환각률을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 트리플(Triple) | 주어-관계-목적어 | KG의 원자 단위인 `(s, r, o)` 튜플. |
| Open IE | 무엇이든 추출 | 열린 어휘 관계 구문. 높은 재현율, 낮은 정밀도. |
| 닫힌 온톨로지(Closed ontology) | 고정된 스키마 | 한정된 관계 유형 집합(Wikidata, UMLS, FIBO). |
| 표준화(Canonicalization) | 모든 것을 정규화 | 표층 이름 / 관계를 표준 id로 매핑한다. |
| AEVS | 근거 있는 추출 | Anchor-Extraction-Verification-Supplement 파이프라인(2026). |
| 출처(Provenance) | 진실 출처 링크 | 모든 트리플이 출처에 대한 문서 id + 문자 스팬을 지닌다. |
| 원거리 지도(Distant supervision) | 값싼 레이블 | 텍스트를 기존 KG와 정렬해 학습 데이터를 만든다. |

## 더 읽을거리 (Further Reading)

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — the distant-supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries.
