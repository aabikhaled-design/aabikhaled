# 상호참조 해결(Coreference Resolution)

> "그녀가 그에게 전화했다. 그는 받지 않았다. 그 의사는 점심을 먹는 중이었다." 두 사람에 대한 세 번의 지시(reference)이고, 누구도 이름이 불리지 않았다. 상호참조 해결(coreference resolution)은 누가 누구인지 알아낸다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60분

## 문제 (The Problem)

300단어 기사에서 Apple Inc.에 대한 모든 언급(mention)을 추출하라. 기사가 "Apple"이라고 말하면 쉽다. "그 회사", "그들", "쿠퍼티노의 기술 거인", "Jobs의 회사"라고 말하면 어렵다. 이 언급들을 같은 엔티티(entity)로 해결하지 않으면 NER 파이프라인(pipeline)은 언급의 60~80%를 놓친다.

상호참조 해결은 같은 실세계 엔티티를 가리키는 모든 표현을 하나의 클러스터(cluster)로 연결한다. 표층 수준 NLP(NER, 구문 분석)와 다운스트림 의미론(IE, QA, 요약, KG)을 잇는 접착제인 셈이다.

2026년에 중요한 이유:

- 요약: "그 CEO가 발표했다..." vs "Tim Cook이 발표했다...": 요약은 그 CEO의 이름을 불러야 한다.
- 질의응답: "그녀가 누구에게 전화했나?"는 "그녀"를 해결해야 한다.
- 정보 추출: "PER1이 Apple을 창업했다"와 "Jobs가 Apple을 창업했다"를 별도 항목으로 둔 지식 그래프는 틀렸다.
- 다중 문서 IE: 같은 사건에 관한 여러 기사에 걸쳐 언급을 병합하는 것이 교차 문서 상호참조(cross-document coreference)다.

## 개념 (The Concept)

![상호참조 클러스터링: 언급 → 엔티티](../assets/coref.svg)

**과제.** 입력: 문서 하나. 출력: 각 클러스터가 하나의 엔티티를 가리키는, 언급(스팬, span)들의 클러스터링.

**언급 유형.**

- **명명된 엔티티(Named entity).** "Tim Cook"
- **명사적(Nominal).** "그 CEO", "그 회사"
- **대명사적(Pronominal).** "그", "그녀", "그들", "그것"
- **동격(Appositive).** "Tim Cook, Apple의 CEO,"

**아키텍처.**

1. **규칙 기반(Hobbs, 1978).** 문법 규칙을 사용하는 구문 트리 기반 대명사 해결. 좋은 베이스라인(baseline). 대명사에서는 놀랍도록 이기기 어렵다.
2. **언급 쌍 분류기(Mention-pair classifier).** 모든 언급 쌍 (m_i, m_j)에 대해 상호참조 여부를 예측한다. 전이 폐쇄(transitive closure)로 클러스터링한다. 2016년 이전의 표준.
3. **언급 순위 매기기(Mention-ranking).** 각 언급에 대해 후보 선행사("선행사 없음" 포함)의 순위를 매긴다. 최상위를 고른다.
4. **스팬 기반 종단 간(Span-based end-to-end, Lee et al., 2017).** 트랜스포머(transformer) 인코더. 길이 상한까지의 모든 후보 스팬을 열거한다. 언급 점수를 예측한다. 각 스팬에 대해 선행사 확률을 예측한다. 탐욕적으로 클러스터링한다. 현대의 기본값.
5. **생성형(Generative, 2024+).** LLM에 프롬프트한다. "이 텍스트의 모든 대명사와 그 선행사를 나열하라." 쉬운 경우에는 잘 동작하지만, 긴 문서와 드문 지시 대상에서는 고전한다.

**평가 지표.** 단일 지표로는 클러스터링 품질을 포착하지 못하므로 다섯 가지 표준 지표(MUC, B³, CEAF, BLANC, LEA)가 있다. 앞의 세 가지 평균을 CoNLL F1으로 보고한다. 2026년 CoNLL-2012 기준 최신 기술(state-of-the-art): 약 83 F1.

**알려진 어려운 경우.**

- 여러 쪽 앞에서 도입된 엔티티를 가리키는 한정 기술(definite description).
- 가교 조응(bridging anaphora)("그 바퀴들" → 앞서 언급된 자동차).
- 중국어와 일본어 같은 언어에서의 영조응(zero anaphora).
- 후방 조응(cataphora, 지시 대상 이전에 오는 대명사): "**그녀**가 걸어 들어왔을 때, Mary가 미소 지었다."

```figure
coref-links
```

## 직접 만들기 (Build It)

### Step 1: 사전 학습된 신경망 상호참조(AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

더 긴 문서에서는 다음과 같은 것을 얻는다:
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### Step 2: 규칙 기반 대명사 해결기(교육용)

표준 라이브러리(stdlib)만 쓴 구현은 `code/main.py`를 보라:

1. 언급 추출: 명명된 엔티티(대문자 스팬), 대명사(사전 조회), 한정 기술("the X").
2. 각 대명사에 대해 직전 K개의 언급을 보고 다음으로 점수를 매긴다:
   - 성/수 일치(휴리스틱)
   - 최근성(가까운 것이 이긴다)
   - 구문적 역할(주어 선호)
3. 가장 높은 점수의 선행사를 연결한다.

신경망 모델과 경쟁할 수준은 아니다. 하지만 탐색 공간과, 종단 간 모델이 내려야 하는 결정을 보여준다.

### Step 3: 상호참조에 LLM 사용하기

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

주의할 두 가지 실패 모드. 첫째, LLM은 과도하게 병합한다(서로 다른 두 사람을 가리키는 "him"과 "her"). 둘째, LLM은 긴 문서에서 언급을 조용히 빠뜨린다. 항상 스팬 오프셋 검사로 검증하라.

### Step 4: 평가

표준 conll-2012 스크립트는 MUC, B³, CEAF-φ4를 계산하고 그 평균을 보고한다. 사내 평가라면 직접 주석한 테스트 세트에서 스팬 수준 정밀도와 재현율로 시작한 다음 언급 연결 F1을 추가하라.

## 함정 (Pitfalls)

- **단일 언급(singleton) 폭발.** 일부 시스템은 모든 언급을 자체 클러스터로 보고한다. B³는 관대하다. MUC는 이를 벌한다. 항상 세 가지 지표를 모두 확인하라.
- **긴 컨텍스트의 대명사.** 2,000 토큰을 넘는 문서에서 성능이 약 15 F1 떨어진다. 신중하게 청킹(chunking)하라.
- **성별 가정.** 하드코딩된 성별 규칙은 논바이너리 지시 대상, 조직, 동물에서 깨진다. 학습된 모델이나 중립적 채점을 사용하라.
- **긴 문서에서의 LLM 표류(drift).** 단일 API 호출은 50개 이상의 단락에 걸친 언급을 안정적으로 클러스터링할 수 없다. 슬라이딩 윈도우 + 병합을 사용하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 단일 문서 | `en_coreference_web_trf`(spaCy-experimental) 또는 AllenNLP 신경망 상호참조 |
| 다국어 | OntoNotes 또는 Multilingual CoNLL로 학습된 SpanBERT / XLM-R |
| 교차 문서 사건 상호참조 | 특화된 종단 간 모델(2025–26 SOTA) |
| 빠른 LLM 베이스라인 | 구조화 출력 상호참조 프롬프트를 쓴 GPT-4o / Claude |
| 프로덕션 다이얼로그 시스템 | 규칙 기반 폴백 + 신경망 주력 + 핵심 슬롯에 대한 수동 검토 |

2026년에 출시되는 통합 패턴: 먼저 NER을 실행하고, 상호참조를 실행하고, 상호참조 클러스터를 NER 엔티티로 병합한다. 다운스트림 과제는 언급당 하나의 엔티티가 아니라 클러스터당 하나의 엔티티를 본다.

## 산출물 (Ship It)

`outputs/skill-coref-picker.md`로 저장한다:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 규칙 기반 해결기를 직접 만든 단락 5개에 대해 실행하라. 정답(ground truth) 대비 언급 연결 정확도를 측정하라.
2. **보통.** 뉴스 기사에 사전 학습된 신경망 상호참조 모델을 사용하라. 클러스터를 직접 단 수동 주석과 비교하라. 어디서 실패했는가?
3. **어려움.** 상호참조로 강화한 NER 파이프라인을 만들어라: 먼저 NER, 그다음 상호참조 클러스터로 병합. 기사 100개에 대해 NER 전용 대비 엔티티 커버리지 향상을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 언급(Mention) | 하나의 지시 | 엔티티를 가리키는 텍스트 스팬(이름, 대명사, 명사구). |
| 선행사(Antecedent) | "그것"이 가리키는 것 | 나중 언급이 상호참조하는 앞선 언급. |
| 클러스터(Cluster) | 그 엔티티의 언급들 | 모두 같은 실세계 엔티티를 가리키는 언급들의 집합. |
| 조응(Anaphora) | 뒤를 향한 지시 | 나중 언급이 앞을 가리킨다("그" → "John"). |
| 후방 조응(Cataphora) | 앞을 향한 지시 | 앞선 언급이 나중을 가리킨다("그가 도착했을 때, John은..."). |
| 가교(Bridging) | 암묵적 지시 | "나는 차를 샀다. 그 바퀴들이 나빴다."(그 차의 바퀴들.) |
| CoNLL F1 | 리더보드의 그 숫자 | MUC, B³, CEAF-φ4 F1 점수의 평균. |

## 더 읽을거리 (Further Reading)

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — canonical textbook chapter.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining that improves coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — the benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — the rule-based classic.
