# 자연어 추론(Natural Language Inference): 텍스트 함의(Textual Entailment)

> "t가 h를 함의한다"는 것은 t를 읽은 사람이 h가 참이라고 결론 내린다는 뜻이다. NLI는 함의(entailment) / 모순(contradiction) / 중립(neutral)을 예측하는 과제다. 겉보기엔 지루하지만 프로덕션(production)에서는 핵심을 떠받친다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60분

## 문제 (The Problem)

요약기를 만들었다고 하자. 요약문이 나왔다. 그 요약에 환각(hallucination)이 들어있지 않다는 것을 어떻게 아는가?

챗봇을 만들었다고 하자. 챗봇이 "예"라고 답했다. 그 답이 검색된 단락(passage)으로 뒷받침된다는 것을 어떻게 아는가?

1만 개의 뉴스 기사를 주제별로 분류해야 한다. 학습 레이블(label)은 하나도 없다. 모델을 재사용할 수 있을까?

세 문제 모두 자연어 추론(Natural Language Inference)으로 환원된다. NLI는 이렇게 묻는다. 전제(premise) `t`와 가설(hypothesis) `h`가 주어졌을 때, `h`는 `t`에 의해 함의되는가, 모순되는가, 아니면 중립(무관)인가?

- **환각 검사:** `t` = 원본 문서, `h` = 요약 주장. 함의가 아니면 = 환각.
- **근거 있는 QA:** `t` = 검색된 단락, `h` = 생성된 답변. 함의가 아니면 = 날조.
- **제로샷 분류(zero-shot classification):** `t` = 문서, `h` = 언어화된 레이블("이것은 스포츠에 관한 것이다"). 함의면 = 예측된 레이블.

하나의 과제, 세 가지 프로덕션 용도. 이것이 모든 RAG 평가 프레임워크가 내부적으로 NLI 모델을 탑재하는 이유다.

## 개념 (The Concept)

![NLI: 3방향 분류, 전제 vs 가설](../assets/nli.svg)

**세 가지 레이블.**

- **함의(Entailment).** `t` → `h`. "고양이가 매트 위에 있다"는 "고양이가 있다"를 함의한다.
- **모순(Contradiction).** `t` → ¬`h`. "고양이가 매트 위에 있다"는 "고양이가 없다"와 모순된다.
- **중립(Neutral).** 어느 쪽으로도 추론되지 않음. "고양이가 매트 위에 있다"는 "고양이가 배고프다"에 대해 중립이다.

**논리적 함의가 아니다.** NLI는 *자연어* 추론이다. 엄밀한 논리가 아니라, 일반적인 사람 독자가 추론할 법한 내용이다. NLI에서 "John이 자기 개를 산책시켰다"는 "John은 개를 가지고 있다"를 함의하지만, 엄밀한 1차 논리(first-order logic)에서는 소유를 공리화해야만 이를 인정한다.

**데이터셋(dataset).**

- **SNLI** (2015). 사람이 주석한 쌍 57만 개, 전제로 이미지 캡션을 사용. 좁은 도메인.
- **MultiNLI** (2017). 10개 장르에 걸친 43만 3천 개 쌍. 2026년 기준 표준 학습 코퍼스.
- **ANLI** (2019). 적대적(adversarial) NLI. 기존 모델을 깨뜨리도록 사람이 일부러 설계한 예시. 더 어렵다.
- **DocNLI, ConTRoL** (2020–21). 문서 길이의 전제. 다중 홉(multi-hop) 및 장거리 추론을 시험한다.

**아키텍처.** 트랜스포머(transformer) 인코더(BERT, RoBERTa, DeBERTa)가 `[CLS] premise [SEP] hypothesis [SEP]`를 읽는다. `[CLS]` 표현이 3방향 소프트맥스(softmax)로 들어간다. MNLI로 학습하고, 별도로 분리된 벤치마크(benchmark)로 평가하면, 분포 내(in-distribution) 쌍에 대해 90% 이상의 정확도를 얻는다.

**NLI를 통한 제로샷.** 문서와 후보 레이블이 주어지면, 각 레이블을 가설로 바꾼다("이 텍스트는 스포츠에 관한 것이다"). 각각에 대해 함의 확률을 계산한다. 최댓값을 고른다. 이것이 Hugging Face의 `zero-shot-classification` 파이프라인(pipeline) 뒤에 있는 메커니즘이다.

```figure
nli-router
```

## 직접 만들기 (Build It)

### Step 1: 사전 학습된 NLI 모델 실행

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

프로덕션 NLI에서는 `facebook/bart-large-mnli`와 `microsoft/deberta-v3-large-mnli`가 오픈소스 기본값이다. DeBERTa-v3가 리더보드(leaderboard) 상위를 차지한다.

### Step 2: 제로샷 분류

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

기본 템플릿은 "This example is about {label}."이다. `hypothesis_template`로 커스터마이즈한다. 학습 데이터가 필요 없다. 파인튜닝(fine-tuning)도 필요 없다. 바로 동작한다.

### Step 3: RAG를 위한 충실도(faithfulness) 검사

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

이것이 RAGAS 충실도의 핵심이다. 생성된 답변을 원자적 주장(atomic claim)으로 나눈다. 각 주장을 검색된 컨텍스트와 대조해 검사한다. 함의되는 비율을 보고한다.

### Step 4: 손수 만든 NLI 분류기(개념적)

`code/main.py`에서 표준 라이브러리(stdlib)만 쓴 장난감 버전을 보라. 전제와 가설을 어휘 중첩(lexical overlap) + 부정 탐지로 비교한다. 트랜스포머 모델과 경쟁할 수준은 아니지만, 과제의 형태를 보여준다. 두 텍스트가 입력되고, 3방향 레이블이 출력되며, 손실(loss)은 `{entail, contradict, neutral}`에 대한 교차 엔트로피(cross-entropy)다.

## 함정 (Pitfalls)

- **가설만으로 푸는 지름길.** "not", "nobody", "never"가 모순과 상관관계가 있기 때문에, 모델은 SNLI에서 가설만으로 약 60% 정확도로 레이블을 예측할 수 있다. 레이블 누수(label leakage)를 탐지하는 강력한 베이스라인(baseline)이다.
- **어휘 중첩 휴리스틱.** 부분 시퀀스 휴리스틱("모든 부분 시퀀스는 함의된다")은 SNLI는 통과하지만 HANS/ANLI에서는 실패한다. 적대적 벤치마크를 사용하라.
- **문서 길이에서의 성능 저하.** 단일 문장 NLI 모델은 문서 길이의 전제에서 F1이 20점 이상 떨어진다. 긴 컨텍스트에는 DocNLI로 학습된 모델을 사용하라.
- **제로샷 템플릿 민감성.** "This example is about {label}" vs "{label}" vs "The topic is {label}"은 정확도를 10점 이상 흔들 수 있다. 템플릿을 튜닝하라.
- **도메인 불일치.** MNLI는 일반 영어로 학습한다. 법률, 의료, 과학 텍스트에는 도메인 특화 NLI 모델(예: SciNLI, MedNLI)이 필요하다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 사용 사례 | 모델 |
|---------|-------|
| 범용 NLI | `microsoft/deberta-v3-large-mnli` |
| 빠른 / 엣지(edge) | `cross-encoder/nli-deberta-v3-base` |
| 제로샷 분류(경량) | `facebook/bart-large-mnli` |
| 문서 수준 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 다국어 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG에서의 환각 탐지 | RAGAS / DeepEval 내부의 NLI 레이어 |

2026년 메타 패턴: NLI는 텍스트 이해의 만능 접착 테이프다. "A가 B를 뒷받침하는가?" 또는 "A가 B와 모순되는가?"가 필요할 때마다, 또 다른 LLM 호출에 손을 뻗기 전에 NLI에 손을 뻗어라.

## 산출물 (Ship It)

`outputs/skill-nli-picker.md`로 저장한다:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 연습 문제 (Exercises)

1. **쉬움.** 세 클래스를 모두 포함하는, 직접 만든 (전제, 가설, 레이블) 삼중쌍 20개에 대해 `facebook/bart-large-mnli`를 실행하라. 정확도를 측정하라. 적대적 "부분 시퀀스 휴리스틱" 함정("I did not eat the cake" vs "I ate the cake")을 추가하고 모델이 깨지는지 보라.
2. **보통.** AG News 헤드라인 100개에 대해 제로샷 템플릿 `"This text is about {label}"`를 `"The topic is {label}"` 및 `"{label}"`과 비교하라. 정확도 변동 폭을 보고하라.
3. **어려움.** RAG 충실도 검사기를 만들어라. 원자적 주장 분해 + 주장별 NLI. 골드 컨텍스트가 있는 RAG 생성 답변 50개에 대해 평가하라. 수동 레이블 대비 거짓 양성(false-positive)과 거짓 음성(false-negative) 비율을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NLI | 자연어 추론(Natural Language Inference) | 전제-가설 관계의 3방향 분류. |
| RTE | 텍스트 함의 인식(Recognizing Textual Entailment) | NLI의 옛 이름. 같은 과제. |
| 함의(Entailment) | "t가 h를 함의한다" | t가 주어졌을 때 일반적인 독자가 h를 참이라고 결론짓는다. |
| 모순(Contradiction) | "t가 h를 배제한다" | t가 주어졌을 때 일반적인 독자가 h를 거짓이라고 결론짓는다. |
| 중립(Neutral) | "미결정" | t에서 h로 어느 쪽으로도 추론되지 않음. |
| 제로샷 분류(Zero-shot classification) | 분류기로서의 NLI | 레이블을 가설로 언어화하고, 최대 함의를 고른다. |
| 충실도(Faithfulness) | 답변이 뒷받침되는가? | (검색된 컨텍스트, 생성된 답변)에 대한 NLI. |

## 더 읽을거리 (Further Reading)

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — the ANLI benchmark.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — the 2026 NLI workhorse.
