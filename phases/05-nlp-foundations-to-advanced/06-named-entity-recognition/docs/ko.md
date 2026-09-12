# 개체명 인식 (Named Entity Recognition)

> 이름들을 뽑아내라. 모호한 경계, 중첩 개체, 도메인 전문 용어를 다루기 전까지는 쉽게 들린다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75분

## 문제 (The Problem)

"Apple sued Google over its iPhone search deal in the US." 다섯 개의 개체(entity): Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (아마도), US (GPE). 좋은 NER 시스템은 이들을 모두 올바른 유형으로 추출한다. 나쁜 시스템은 iPhone을 놓치고, 과일 Apple과 회사 Apple을 혼동하며, "US"를 PERSON으로 레이블링한다.

NER은 모든 구조화 추출 파이프라인(pipeline) 아래의 일꾼이다. 이력서 파싱, 규정 준수 로그 스캔, 의료 기록 익명화, 검색 질의 이해, 챗봇 응답을 위한 근거 부여(grounding), 법률 계약 추출. 좀처럼 눈에 띄지 않지만 어디서나 이 일꾼에 기댄다.

이 레슨은 고전적 경로(규칙 기반, HMM, CRF)를 거쳐 현대적 경로(BiLSTM-CRF, 그다음 트랜스포머)로 나아간다. 각 단계는 앞 단계가 남긴 특정 한계를 해결한다. 바로 그 패턴이 이 레슨의 핵심이다.

## 개념 (The Concept)

**BIO 태깅**(또는 BILOU)은 개체 추출을 시퀀스 레이블링(sequence-labeling) 문제로 바꾼다. 각 토큰(token)을 `B-TYPE`(개체의 시작), `I-TYPE`(개체 내부), 또는 `O`(어떤 개체에도 속하지 않음)로 레이블링한다.

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

여러 토큰 개체는 사슬처럼 이어진다: `New B-GPE`, `York I-GPE`, `City I-GPE`. BIO를 이해하는 모델은 임의의 스팬(span)을 추출할 수 있다.

아키텍처 진행:

- **규칙 기반.** 정규식 + 가제티어(gazetteer) 룩업. 알려진 개체에 대해서는 높은 정밀도, 새로운 개체에 대해서는 커버리지 0.
- **HMM.** 은닉 마르코프 모델(Hidden Markov Model). 태그가 주어졌을 때 토큰의 방출(emission) 확률, 태그-대-태그 전이(transition) 확률. 비터비(Viterbi) 디코드. 레이블된 데이터로 학습.
- **CRF.** 조건부 무작위장(Conditional Random Field). HMM과 비슷하지만 판별적(discriminative)이라, 임의의 특성(단어 형태, 대문자화, 인접 단어)을 섞을 수 있다. 2026년에도 저자원 배포에서는 여전히 고전적 프로덕션 일꾼이다.
- **BiLSTM-CRF.** 수작업 대신 신경망 특성. LSTM이 문장을 양방향으로 읽고, 그 위의 CRF 층(layer)이 일관된 태그 시퀀스를 강제한다.
- **트랜스포머 기반.** 토큰 분류 헤드로 BERT를 파인튜닝(fine-tuning). 최고의 정확도. 가장 많은 연산.

```figure
ner-bio-tagging
```

## 직접 만들기 (Build It)

### 1단계: BIO 태깅 헬퍼

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 2단계: 수작업 특성

고전(비신경망) NER에서는 특성(feature)이 핵심이다. 유용한 것들:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`은 `xXxxxx`를 반환한다. `word_shape("USA-2024")`는 `XXX-dddd`를 반환한다. 대문자화 패턴은 고유명사에 대해 신호가 강하다.

### 3단계: 단순한 규칙 기반 + 사전 베이스라인

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

프로덕션 가제티어에는 위키피디아와 DBpedia에서 긁어모은 수백만 개의 항목이 들어 있다. 커버리지는 좋다. 중의성 해소(`Apple` 회사 대 과일)는 형편없다. 그래서 통계적 모델이 이겼다.

### 4단계: CRF 단계 (전체 구현이 아닌 스케치)

확률 이론의 토대 없이 50줄로 밑바닥부터 만드는 전체 CRF는 깨우침을 주지 못한다. 대신 `sklearn-crfsuite`를 쓴다:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`과 `c2`는 L1과 L2 규제(regularization)다. `all_possible_transitions=True`는 모델이 불법적인 시퀀스(예: `O` 뒤의 `I-ORG`)가 일어날 법하지 않음을 학습하게 한다. 이것이 CRF가 제약을 따로 작성하지 않고도 BIO 일관성을 강제하는 방식이다.

### 5단계: BiLSTM-CRF가 더하는 것

특성이 학습된다. 입력: 토큰 임베딩(embedding, GloVe 또는 fastText). LSTM이 좌-우, 우-좌로 읽는다. 연결된 은닉 상태가 CRF 출력층을 통과한다. CRF는 여전히 태그 시퀀스 일관성을 강제하고, LSTM은 수작업 특성을 학습된 특성으로 대체한다.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 층에는 `torchcrf.CRF`(pip install pytorch-crf)를 쓴다. 수작업 CRF 대비 이득은 측정 가능하지만, 수만 개의 레이블된 문장이 있지 않은 한 예상보다 작다.

## 라이브러리로 써보기 (Use It)

spaCy는 프로덕션 등급 NER을 기본으로 제공한다.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

`iPhone`이 `PRODUCT`가 아니라 `ORG`로 레이블링된 것에 주목하라. spaCy의 작은 모델은 제품 개체 커버리지가 약하다. 큰 모델(`en_core_web_lg`)은 더 낫다. 트랜스포머 모델(`en_core_web_trf`)은 더더욱 낫다.

BERT 기반 NER을 위한 Hugging Face:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`은 연속된 B-X, I-X 토큰을 하나의 스팬으로 병합한다. 이것이 없으면 토큰 단위 레이블을 얻어 직접 병합해야 한다.

### LLM 기반 NER (2026년의 선택지)

제로샷, 퓨샷 LLM NER은 이제 많은 도메인에서 파인튜닝된 모델과 경쟁할 만하고, 레이블된 데이터가 부족할 때는 극적으로 더 낫다.

- **제로샷 프롬프팅.** LLM에 개체 유형 목록과 예시 스키마를 준다. JSON 출력을 요청한다. 기본 상태로 동작한다. 새로운 도메인에서는 정확도가 보통 수준.
- **ZeroTuneBio 스타일 프롬프팅.** 과제를 후보 추출 → 의미 설명 → 판단 → 재확인으로 분해한다. 다단계 프롬프트(원샷이 아님)는 생의학 NER에서 정확도를 상당히 끌어올린다. 같은 패턴이 법률, 금융, 과학 도메인에서도 동작한다.
- **RAG를 이용한 동적 프롬프팅.** 추론 호출마다 작은 주석 시드 집합에서 가장 유사한 레이블된 예시를 검색해 퓨샷 프롬프트를 즉석에서 구성한다. 2026년 벤치마크(benchmark)에서 이 방식은 정적 프롬프팅 대비 GPT-4 생의학 NER F1을 11-12% 끌어올린다.
- **개체 유형별 분해.** 긴 문서에서는 모든 개체 유형을 한 번에 추출하는 단일 호출이 길이가 늘수록 재현율(recall)을 잃는다. 개체 유형마다 한 번의 추출 패스를 실행하라. 추론 비용은 더 높지만 정확도가 상당히 더 높다. 이것이 임상 노트와 법률 계약의 표준 패턴이다.

2026년 기준 프로덕션 권장 사항: 학습 데이터를 모으기 전에 LLM 제로샷 베이스라인으로 시작하라. 종종 F1이 충분히 좋아서 파인튜닝이 전혀 필요 없다.

### 고전 NER이 여전히 이기는 곳

LLM이 있더라도, 고전 NER은 다음과 같을 때 이긴다:

- 지연 시간(latency) 예산이 50ms 미만일 때.
- 수천 개의 레이블된 예시가 있고 98% 이상의 F1이 필요할 때.
- 사전 학습된 CRF나 BiLSTM이 잘 전이되는 안정적 온톨로지를 가진 도메인일 때.
- 규제 제약이 온프레미스 비생성형 모델을 요구할 때.

### 무너지는 곳

- **도메인 시프트.** 법률 계약에 대해 CoNLL로 학습된 NER은 가제티어보다 못한 성능을 낸다. 자신의 도메인으로 파인튜닝하라.
- **중첩 개체.** "Bank of America Tower"는 동시에 ORG이자 FACILITY다. 표준 BIO는 겹치는 스팬을 표현할 수 없다. 중첩 NER(다중 패스 또는 스팬 기반 모델)이 필요하다.
- **긴 개체.** "United States Federal Deposit Insurance Corporation." 토큰 단위 모델은 때때로 이것을 쪼갠다. `aggregation_strategy`를 쓰거나 후처리하라.
- **희소 유형.** DRUG_BRAND, ADVERSE_EVENT, DOSE 같은 의료 NER 레이블. 범용 모델은 전혀 모른다. Scispacy와 BioBERT가 그 출발점이다.

## 산출물 (Ship It)

`outputs/skill-ner-picker.md`로 저장한다:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 연습 문제 (Exercises)

1. **쉬움.** `bio_to_spans`(`spans_to_bio`의 역함수)를 구현하고 10개 문장에서 왕복 일관성을 확인하라.
2. **보통.** 위의 sklearn-crfsuite CRF를 CoNLL-2003 영어 NER 데이터셋(dataset)에서 학습하라. `seqeval`을 사용해 개체별 F1을 보고하라. 전형적 결과: 약 84 F1.
3. **어려움.** 도메인 특화 NER 데이터셋(의료, 법률, 또는 금융)에서 `distilbert-base-cased`를 파인튜닝하라. spaCy 작은 모델과 비교하라. 데이터 누수 검사를 문서화하고 무엇이 놀라웠는지 정리하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NER | 이름 추출 | 토큰 스팬을 유형(PERSON, ORG, GPE, DATE, ...)으로 레이블링. |
| BIO | 태깅 방식 | `B-X`는 시작, `I-X`는 계속, `O`는 외부. |
| BILOU | 더 나은 BIO | 더 깔끔한 경계를 위해 `L-X`(마지막), `U-X`(단일)를 추가. |
| CRF | 구조화 분류기 | 방출만이 아니라 레이블 간 전이를 모델링. 유효한 시퀀스를 강제. |
| 중첩 NER(Nested NER) | 겹치는 개체 | 한 스팬이 그 하위 스팬과 다른 개체. BIO는 이를 표현할 수 없다. |
| 개체 단위 F1(Entity-level F1) | 올바른 NER 지표 | 예측 스팬이 참 스팬과 정확히 일치해야 함. 토큰 단위 F1은 정확도를 과대 평가한다. |

## 더 읽을거리 (Further Reading)

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360): BiLSTM-CRF 논문. 표준.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805): 표준이 된 토큰 분류 패턴을 도입.
- [spaCy linguistic features: named entities](https://spacy.io/usage/linguistic-features#named-entities) `Doc.ents`와 `Span`의 모든 속성에 대한 실용 레퍼런스.
- [seqeval](https://github.com/chakki-works/seqeval): 올바른 지표 라이브러리. 항상 이것을 써라.
