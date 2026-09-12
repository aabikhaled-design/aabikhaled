# 품사 태깅과 구문 파싱 (POS Tagging and Syntactic Parsing)

> 문법은 한동안 인기가 없었다. 그러다 모든 LLM 파이프라인이 구조화 추출을 검증해야 하게 되었고, 문법이 돌아왔다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45분

## 문제 (The Problem)

레슨 01은 표제어 추출(lemmatization)이 품사(part-of-speech) 태그를 필요로 한다고 약속했다. `running`이 동사임을 모르면 표제어 추출기는 이를 `run`으로 환원할 수 없다. `better`가 형용사임을 모르면 `good`으로 환원할 수 없다.

그 약속은 하나의 하위 분야 전체를 숨기고 있었다. 품사 태깅(part-of-speech tagging)은 문법 범주를 부여한다. 구문 파싱(syntactic parsing)은 문장의 트리 구조를 복원한다. 어느 단어가 어느 것을 수식하는지, 어느 동사가 어느 논항을 지배하는지. 고전 NLP는 20년을 두 가지를 다듬는 데 썼다. 그러다 딥러닝(deep learning)이 이 둘을 사전 학습된 트랜스포머(transformer) 위의 토큰 분류 과제로 무너뜨렸고, 연구 공동체는 다른 곳으로 넘어갔다.

응용 공동체는 아니다. 모든 구조화 추출 파이프라인(pipeline)은 여전히 내부에서 품사와 의존 트리(dependency tree)를 쓴다. LLM이 생성한 JSON은 문법 제약에 대해 검증된다. 질의응답 시스템은 의존 파스를 사용해 질의를 분해한다. 기계 번역 품질 평가기는 파스 트리의 정렬을 점검한다.

알아 둘 가치가 있다. 이 레슨은 태그셋(tagset), 베이스라인(baseline), 그리고 밑바닥부터 구현하기를 멈추고 spaCy를 호출하는 지점을 소개한다.

## 개념 (The Concept)

**품사 태깅(POS tagging)**은 각 토큰(token)에 문법 범주를 레이블링한다. **펜 트리뱅크(Penn Treebank, PTB)** 태그셋은 영어의 기본값이다. 36개의 태그가 있고, 일반 독자에게는 까다롭게 느껴지는 구별을 한다: `NN` 단수 명사, `NNS` 복수 명사, `NNP` 단수 고유명사, `VBD` 동사 과거형, `VBZ` 동사 3인칭 단수 현재형 등등. **유니버설 디펜던시스(Universal Dependencies, UD)** 태그셋은 더 거칠고(17개 태그) 언어 독립적이다. 이것이 교차 언어 작업의 기본값이 되었다.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**구문 파싱**은 트리를 만든다. 두 가지 주요 방식:

- **구성성분 파싱(Constituency parsing).** 명사구, 동사구, 전치사구가 서로 안에 중첩된다. 출력은 단어를 잎으로 갖는 비단말 범주(NP, VP, PP)의 트리다.
- **의존 파싱(Dependency parsing).** 각 단어는 그것이 의존하는 단일 핵어(head word)를 가지며, 문법 관계로 레이블링된다. 출력은 모든 간선이 (핵어, 의존소, 관계) 삼중쌍인 트리다.

의존 파싱은 2010년대에 이겼는데, 언어 전반에, 특히 어순이 자유로운 언어에 깔끔하게 일반화되기 때문이다.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 직접 만들기 (Build It)

### 1단계: 최빈 태그 베이스라인

동작하는 것 중 가장 멍청한 품사 태거. 각 단어에 대해, 학습에서 가장 자주 가졌던 태그를 예측한다.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

브라운(Brown) 코퍼스에서 이 베이스라인은 약 85% 정확도(accuracy)에 도달한다. 좋지는 않지만, 진지한 모델이라면 그 아래로 떨어져서는 안 되는 바닥이다.

### 2단계: 바이그램 HMM 태거

시퀀스의 결합 확률을 모델링한다:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

두 개의 테이블: 전이 확률(이전 태그가 주어졌을 때의 태그), 방출 확률(태그가 주어졌을 때의 단어). 둘 다 라플라스 평활화(Laplace smoothing)와 함께 카운트로부터 추정한다. 비터비(Viterbi, 태그 격자에 대한 동적 계획법)로 디코드한다.

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

브라운에서 바이그램 HMM은 약 93% 정확도에 도달한다. 85%에서 93%로 도약한 것은 대부분 전이 확률에서 온다. 모델은 `DET NOUN`이 흔하고 `NOUN DET`이 드물다는 것을 학습한다.

### 3단계: 현대 태거가 이것을 이기는 이유

전이 + 방출 확률은 국소적이다. 이들은 "I bought a saw"에서 `saw`가 명사지만 "I saw the movie"에서는 동사라는 것을 포착할 수 없다. 임의의 특성(접미사, 단어 형태, 앞뒤 단어, 단어 자체)을 가진 CRF는 약 97%에 도달한다. BiLSTM-CRF나 트랜스포머는 약 98% 이상에 도달한다.

이 과제의 상한은 주석자 불일치로 정해진다. 인간 주석자는 펜 트리뱅크에서 약 97% 일치한다. 98%를 넘는 모델은 아마 테스트 세트에 과적합(overfitting)한 것이다.

### 4단계: 의존 파싱 스케치

밑바닥부터 만드는 전체 의존 파싱은 범위 밖이다. 표준 교과서적 다룸은 Jurafsky와 Martin에 있다. 알아 둘 고전적 두 계열:

- **전이 기반(Transition-based)** 파서(arc-eager, arc-standard)는 시프트-리듀스 파서처럼 동작한다. 토큰을 읽어 스택에 시프트하고, 아크를 만드는 리듀스 동작을 적용한다. 탐욕적 디코딩은 빠르다. 고전적 구현은 MaltParser다. 현대 신경망 버전: Chen과 Manning의 전이 기반 파서.
- **그래프 기반(Graph-based)** 파서(Eisner 알고리즘, Dozat-Manning biaffine)는 가능한 모든 핵어-의존소 간선에 점수를 매기고 최대 신장 트리를 고른다. 더 느리지만 더 정확하다.

대부분의 응용 작업에는 spaCy를 호출한다:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep` 열을 아래에서 위로 읽으면 문장의 문법 구조가 떨어져 나온다.

## 라이브러리로 써보기 (Use It)

모든 프로덕션(production) NLP 라이브러리는 품사와 의존 파서를 표준 파이프라인의 일부로 제공한다.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). 빠르고, 정확하며, 토큰화 + NER + 표제어 추출과 통합됨. `token.tag_`(Penn), `token.pos_`(UD), `token.dep_`(의존 관계).
- **Stanford NLP (stanza)**. CoreNLP에 대한 스탠퍼드의 후속작. 60개 이상의 언어에서 최첨단.
- **trankit**. 트랜스포머 기반, 좋은 UD 정확도.
- **NLTK**. `pos_tag`. 쓸 만하고, 느리고, 더 오래되었다. 교육용으로는 괜찮다.

### 2026년에 이것이 여전히 중요한 곳

- **표제어 추출.** 레슨 01은 올바르게 표제어를 추출하려면 품사가 필요하다. 항상.
- **LLM 출력으로부터의 구조화 추출.** 생성된 문장이 문법 제약(예: 주어-동사 일치, 필수 수식어)을 지키는지 검증.
- **애스펙트 기반 감성.** 의존 파스는 어느 형용사가 어느 명사를 수식하는지 알려 준다.
- **질의 이해.** "movies directed by Wes Anderson starring Bill Murray"는 파스를 통해 구조화된 제약으로 분해된다.
- **교차 언어 전이.** UD 태그와 의존 관계는 언어 독립적이라, 새 언어의 제로샷 구조 분석을 가능하게 한다.
- **저연산 파이프라인.** 트랜스포머를 출하할 수 없다면, 품사 + 의존 파스 + 가제티어(gazetteer)만으로도 놀라울 만큼 멀리 갈 수 있다.

## 산출물 (Ship It)

`outputs/skill-grammar-pipeline.md`로 저장한다:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 연습 문제 (Exercises)

1. **쉬움.** 작은 태깅된 코퍼스(예: NLTK의 브라운 부분집합)에서 최빈 태그 베이스라인을 사용해, 떼어 둔 문장에서 정확도를 측정하라. 약 85% 결과를 확인하라.
2. **보통.** 위의 바이그램 HMM을 학습하고 태그별 정밀도(precision)/재현율(recall)을 보고하라. HMM이 가장 많이 혼동하는 태그는 무엇인가?
3. **어려움.** spaCy의 의존 파스를 사용해 1000개 문장 샘플에서 주어-동사-목적어 삼중쌍을 추출하라. 수작업으로 레이블링한 50개 삼중쌍에서 평가하라. 추출이 실패하는 곳(흔히 수동태, 등위 접속, 생략된 주어)을 문서화하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 품사 태그(POS tag) | 단어의 유형 | 문법 범주. PTB는 36개, UD는 17개. |
| 펜 트리뱅크(Penn Treebank) | 표준 태그셋 | 영어 특화. 세분화된 동사 시제와 명사 수. |
| 유니버설 디펜던시스(Universal Dependencies) | 다국어 태그셋 | PTB보다 거침. 언어 중립적. 교차 언어 작업의 기본값. |
| 의존 파스(Dependency parse) | 문장 트리 | 각 단어가 핵어 하나를 가지고, 각 간선이 문법 관계를 가짐. |
| 비터비(Viterbi) | 동적 계획법 | 방출과 전이가 주어졌을 때 최고 확률의 태그 시퀀스를 찾음. |

## 더 읽을거리 (Further Reading)

- [Jurafsky and Martin: Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) 품사와 파싱에 대한 표준 교과서적 다룸.
- [Universal Dependencies project](https://universaldependencies.org/): 모든 다국어 파서가 쓰는 교차 언어 태그셋과 트리뱅크 모음.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features): `Token`에 노출된 모든 속성에 대한 실용 레퍼런스.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf): 신경망 파서를 주류로 들여온 논문.
