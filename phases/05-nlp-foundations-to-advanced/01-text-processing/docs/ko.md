# 텍스트 처리(토큰화, 어간 추출, 표제어 추출 (Text Processing) Tokenization, Stemming, Lemmatization)

> 언어는 연속적이다. 모델은 이산적이다. 전처리(preprocessing)는 그 둘을 잇는 다리다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45분

## 문제 (The Problem)

모델은 "The cats were running."을 읽지 못한다. 모델이 읽는 것은 정수다.

모든 NLP 시스템은 동일한 세 가지 질문으로 시작한다. 단어는 어디에서 시작하는가. 단어의 어근은 무엇인가. "run", "running", "ran"을 도움이 될 때는 같은 것으로, 도움이 되지 않을 때는 다른 것으로 어떻게 다룰 것인가.

토큰화(tokenization)를 잘못하면 모델은 쓰레기로부터 학습한다. 토크나이저(tokenizer)가 `don't`를 하나의 토큰(token)으로 다루면서 `do n't`는 두 개로 다룬다면 학습 분포가 갈라진다. 어간 추출기(stemmer)가 `organization`과 `organ`을 같은 어간(stem)으로 합쳐 버리면 토픽 모델링(topic modeling)이 망가진다. 표제어 추출기(lemmatizer)는 품사(part-of-speech) 맥락이 필요한데 그 맥락을 넘기지 않으면 동사가 명사로 처리된다.

이 레슨에서는 세 가지 전처리 단계를 밑바닥부터 만든 다음, NLTK와 spaCy가 같은 작업을 어떻게 수행하는지 보여 트레이드오프(tradeoff)를 직접 확인하게 한다.

## 개념 (The Concept)

세 가지 연산이 있다. 각각은 하나의 역할과 하나의 실패 양상을 가진다.

**토큰화(Tokenization)**는 문자열을 토큰들로 나눈다. "토큰"이라는 말은 의도적으로 모호하게 쓴 것인데, 올바른 단위(granularity)가 과제에 따라 달라지기 때문이다. 고전 NLP에서는 단어 단위, 트랜스포머(transformer)에서는 서브워드(subword) 단위, 띄어쓰기가 없는 언어에서는 문자 단위가 쓰인다.

**어간 추출(Stemming)**은 규칙으로 접미사를 잘라낸다. 빠르고, 공격적이고, 멍청하다. `running -> run`. `organization -> organ`. 두 번째 예가 바로 실패 양상이다.

**표제어 추출(Lemmatization)**은 문법 지식을 사용해 단어를 사전형(dictionary form)으로 환원한다. 더 느리고, 더 정확하며, 룩업 테이블(lookup table)이나 형태소 분석기(morphological analyzer)를 필요로 한다. `ran -> run`("ran"이 "run"의 과거형임을 알아야 한다). `better -> good`(비교급 형태를 알아야 한다).

경험 법칙. 속도가 중요하고 노이즈를 어느 정도 감수할 수 있을 때(검색 인덱싱, 거친 분류) 어간 추출을 쓴다. 의미가 중요할 때(질의응답, 의미 검색, 사용자가 읽게 될 모든 것) 표제어 추출을 쓴다.

```figure
edit-distance
```

## 직접 만들기 (Build It)

### 1단계: 정규식 단어 토크나이저

가장 단순하지만 쓸 만한 토크나이저는 영숫자가 아닌 문자에서 분할하되 구두점은 그 자체로 하나의 토큰으로 유지한다. 완벽하지도 최종본도 아니지만, 한 줄로 동작한다.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

우선순위 순서대로 세 가지 패턴이 있다. 내부 아포스트로피를 선택적으로 포함하는 단어(`don't`, `it's`). 순수 숫자. 영숫자가 아니면서 공백도 아닌 단일 문자를 독립 토큰(구두점)으로.

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

주목할 실패 양상. `3pm`은 `['3', 'pm']`으로 갈라지는데, 글자 묶음과 숫자 묶음을 번갈아 매칭했기 때문이다. 대부분의 과제에는 충분하다. URL, 이메일, 해시태그는 모두 깨진다. 프로덕션(production)에서는 일반 패턴보다 앞에 별도 패턴을 추가한다.

### 2단계: 포터 어간 추출기 (1a 단계만)

전체 포터(Porter) 알고리즘은 다섯 단계의 규칙으로 이루어진다. 1a 단계 하나만으로도 영어에서 가장 빈번한 접미사를 다루며 그 패턴을 가르쳐 준다.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

규칙을 위에서 아래로 읽어라. `ies -> i` 규칙 때문에 `ponies`가 `pony`가 아니라 `poni`가 된다. 실제 포터에는 이를 바로잡을 1b 단계가 있다. 규칙들은 서로 경쟁한다. 앞선 규칙이 이긴다. 순서가 어떤 개별 규칙보다 더 중요하다.

### 3단계: 룩업 기반 표제어 추출기

제대로 된 표제어 추출은 형태론(morphology)을 필요로 한다. 다루기 쉬운 교육용 버전은 작은 표제어 테이블과 폴백(fallback)을 사용한다.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

마지막 사례가 핵심적인 교육 포인트다. `watched`는 우리 테이블에 없고 우리 폴백은 `ing`만 처리한다. 실제 표제어 추출은 `ed`, 불규칙 동사, 비교급 형용사, 소리가 바뀌는 복수형(`children -> child`)까지 다룬다. 바로 이 때문에 프로덕션 시스템은 WordNet, spaCy의 형태소 분석기(morphologizer), 또는 완전한 형태소 분석기를 사용한다.

### 4단계: 하나로 연결하기

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

빠진 조각은 품사 태거(POS tagger)다. Phase 5 · 07 (POS Tagging)에서 하나를 만든다. 지금은 모든 것을 `NOUN`으로 기본 처리하고 그 한계를 인정한다.

## 라이브러리로 써보기 (Use It)

NLTK와 spaCy는 프로덕션 버전을 제공한다. 각각 몇 줄이면 된다.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`는 축약형, 유니코드, 정규식이 놓치는 엣지 케이스를 처리한다. `PorterStemmer`는 다섯 단계를 모두 돌린다. `WordNetLemmatizer`는 NLTK의 Penn Treebank 체계에서 WordNet의 약어 집합으로 변환된 품사 태그를 필요로 한다. 위의 변환 연결 부분은 대부분의 튜토리얼이 건너뛴다.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy는 전체 파이프라인을 `nlp(text)` 뒤에 감춘다. 토큰화, 품사 태깅, 표제어 추출이 모두 실행된다. 대규모에서는 NLTK보다 빠르다. 기본 상태에서 더 정확하다. 트레이드오프는 개별 구성 요소를 쉽게 교체할 수 없다는 점이다.

### 무엇을 언제 고를까

| 상황 | 선택 |
|-----------|------|
| 교육, 연구, 구성 요소 교체 | NLTK |
| 프로덕션, 다국어, 속도가 중요 | spaCy |
| 트랜스포머 파이프라인(어차피 모델의 토크나이저로 토큰화하게 된다) | `tokenizers` / `transformers`를 쓰고 고전 전처리는 건너뛴다 |

### 아무도 경고해 주지 않는 두 가지 실패 양상

대부분의 튜토리얼은 알고리즘만 가르치고 멈춘다. 실제 전처리 파이프라인을 무는 두 가지가 있는데, 거의 다뤄지지 않는다.

**재현성 표류(reproducibility drift).** NLTK와 spaCy는 버전 사이에서 토큰화와 표제어 추출기의 동작을 바꾼다. spaCy 2.x에서 `['do', "n't"]`를 만들어 내던 것이 3.x에서는 `["don't"]`을 만들어 낼 수 있다. 모델은 하나의 분포로 학습했는데, 이제 추론(inference)은 다른 분포 위에서 돌아간다. 정확도가 조용히 떨어지고 아무도 그 이유를 모른다. `requirements.txt`에 라이브러리 버전을 고정하라. 20개 샘플 문장의 기대 토큰화를 고정하는 전처리 회귀 테스트를 작성하라. 업그레이드할 때마다 그것을 돌려라.

**학습/추론 불일치(training / inference mismatch).** 공격적인 전처리(소문자화, 불용어 제거, 어간 추출)로 학습하고, 가공되지 않은 사용자 입력에 배포한 다음, 성능이 곤두박질치는 것을 지켜보라. 이것은 단일 항목으로 가장 흔한 프로덕션 NLP 실패다. 학습 중에 전처리를 한다면, 추론 중에도 동일한 함수를 실행해야 한다. 전처리를 서빙 팀이 다시 작성하는 노트북 셀이 아니라, 모델 패키지 안의 함수로 출하하라.

## 산출물 (Ship It)

엔지니어가 세 권의 교과서를 읽지 않고도 전처리 전략을 고를 수 있도록 돕는 재사용 가능한 프롬프트.

`outputs/prompt-preprocessing-advisor.md`로 저장한다:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 연습 문제 (Exercises)

1. **쉬움.** `tokenize`를 확장하여 URL을 단일 토큰으로 유지하라. 테스트: `tokenize("Visit https://example.com today.")`는 하나의 URL 토큰을 만들어야 한다.
2. **보통.** 포터 1b 단계를 구현하라. 단어가 모음을 포함하면서 `ed`나 `ing`로 끝나면 그것을 제거하라. 자음 중복 규칙(`hopping -> hop`, `hopp`이 아님)을 처리하라.
3. **어려움.** WordNet을 룩업 테이블로 사용하되 WordNet에 항목이 없을 때 직접 만든 포터 어간 추출기로 폴백하는 표제어 추출기를 만들어라. 태깅된 코퍼스에서 순수 WordNet과 순수 포터에 대비해 정확도를 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 토큰(Token) | 단어 | 모델이 소비하는 단위는 무엇이든. 단어, 서브워드, 문자, 또는 바이트일 수 있다. |
| 어간(Stem) | 단어의 어근 | 규칙 기반 접미사 제거의 결과. 항상 실재하는 단어인 것은 아니다. |
| 표제어(Lemma) | 사전형 | 사전에서 찾아볼 형태. 올바르게 계산하려면 문법적 맥락이 필요하다. |
| 품사 태그(POS tag) | 품사 | NOUN, VERB, ADJ 같은 범주. 정확한 표제어 추출에 필요하다. |
| 형태론(Morphology) | 단어 형태 규칙 | 시제, 수, 격에 따라 단어가 형태를 바꾸는 방식. 표제어 추출이 이에 의존한다. |

## 더 읽을거리 (Further Reading)

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt): 원본 논문, 다섯 쪽, 여전히 가장 명료한 설명.
- [spaCy 101: linguistic features](https://spacy.io/usage/linguistic-features) 실제 파이프라인이 어떻게 연결되는지.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html): 미처 생각지 못한 토큰화 엣지 케이스까지.
