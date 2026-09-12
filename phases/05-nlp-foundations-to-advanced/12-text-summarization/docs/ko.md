# Text Summarization

> 추출적(extractive) 시스템은 문서가 무엇을 말했는지 알려준다. 추상적(abstractive) 시스템은 저자가 무엇을 의도했는지 알려준다. 다른 작업, 다른 함정이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75분

## 문제 (The Problem)

2,000단어짜리 뉴스 기사가 피드에 들어온다. 그 내용을 담아낼 120단어가 필요하다. 기사에서 가장 중요한 세 문장을 고르거나(추출적), 내용을 직접 다시 쓸 수 있다(추상적). 둘 다 요약(summarization)이라 부르지만, 서로 완전히 다른 문제다.

추출적 요약(extractive summarization)은 순위 매기기 문제다. 모든 문장에 점수를 매기고 상위 `k`개를 반환한다. 출력이 원문을 그대로 들어낸 것이므로 항상 문법적으로 맞다. 위험은 기사 전체에 흩어진 내용을 놓치는 것이다.

추상적 요약(abstractive summarization)은 생성 문제다. 트랜스포머(transformer)가 입력에 조건화된 새 텍스트를 만든다. 출력은 유창하고 압축적이지만 원문에 없던 사실을 환각(hallucination)할 수 있다. 위험은 자신만만한 날조다.

이 레슨은 각자가 떠안는 실패 양상과 함께 둘 다 만든다.

## 개념 (The Concept)

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**추출적(Extractive).** 기사를 노드(node)가 문장이고 간선(edge)이 유사도(similarity)인 그래프로 다룬다. 그래프 위에서 PageRank(또는 비슷한 것)를 실행하여 각 문장이 다른 모든 문장과 얼마나 연결되어 있는지로 점수를 매긴다. 가장 높은 점수의 문장들이 요약이다. 표준 구현은 **TextRank**(Mihalcea and Tarau, 2004)다.

**추상적(Abstractive).** 문서-요약 쌍으로 트랜스포머 인코더-디코더(encoder-decoder)(BART, T5, Pegasus)를 파인튜닝(fine-tuning)한다. 추론(inference) 시 모델은 문서를 읽고 크로스 어텐션(cross-attention)으로 요약을 토큰(token) 단위로 생성한다. 특히 Pegasus는 갭 문장(gap-sentence) 사전 학습(pretraining) 목표를 사용하여, 많은 파인튜닝 없이도 요약에 탁월하다.

평가는 **ROUGE**(Recall-Oriented Understudy for Gisting Evaluation)로 한다. ROUGE-1과 ROUGE-2는 유니그램(unigram)과 바이그램(bigram) 중복을 채점한다. ROUGE-L은 최장 공통 부분 수열(longest common subsequence)을 채점한다. 높을수록 좋지만 ROUGE-L 40은 "좋은" 수준이고 50은 "탁월한" 수준이다. 모든 논문이 세 가지를 모두 보고한다. `rouge-score` 패키지를 사용하라.

```figure
summarize-collapse
```

## 직접 만들기 (Build It)

### Step 1: TextRank (추출적)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

짚어둘 만한 점이 둘 있다. 유사도 함수는 로그 정규화된 단어 중복을 쓰는데, 이것이 원래의 TextRank 변형이다. TF-IDF 벡터의 코사인(cosine)도 통한다. 감쇠 인자(damping factor) 0.85와 반복 횟수는 PageRank 기본값이다.

### Step 2: BART를 사용한 추상적 요약

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN은 CNN/DailyMail 코퍼스(corpus)로 파인튜닝되어 있다. 바로 뉴스 스타일의 요약을 만들어낸다. 다른 도메인(과학 논문, 대화, 법률)에는 해당하는 Pegasus 체크포인트(checkpoint)를 쓰거나 목표 데이터로 직접 파인튜닝하라.

### Step 3: ROUGE 평가

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

항상 어간 추출(stemming)을 사용하라. 어간 추출이 없으면 "running"과 "run"이 다른 단어로 계산되어 ROUGE가 일치를 과소 계산한다.

### ROUGE를 넘어서 (2026 요약 평가)

ROUGE는 20년 동안 지배적인 요약 지표였으나 2026년에는 그것만으로 충분하지 않다. NLG 논문에 대한 대규모 메타 분석은 다음을 보여줬다.

- **BERTScore**(맥락적 임베딩 유사도)는 2023년까지 입지를 넓혔으며 이제 대부분의 요약 논문에서 ROUGE와 나란히 보고된다.
- **BARTScore**는 평가를 생성으로 다룬다. 사전 학습된 BART가 원문이 주어졌을 때 요약에 얼마나 높은 확률을 부여하는지로 요약을 채점한다.
- **MoverScore**(맥락적 임베딩에 대한 Earth Mover's Distance)는 ROUGE보다 의미적 중복을 더 잘 포착하여 2025년 요약 벤치마크(benchmark)에서 최상위에 올랐다.
- **FactCC**와 **QA 기반 충실성(QA-based faithfulness)**은 2021-2023년에 흔했으나, 이제는 종종 **G-Eval**(연쇄적 사고(chain-of-thought) 추론으로 일관성, 정합성, 유창성, 관련성을 채점하는 GPT-4 프롬프트 연쇄)로 대체된다.
- **G-Eval**과 유사한 LLM-judge 접근법은 평가 기준(rubric)이 잘 설계되면 사람의 판단과 약 80% 일치한다.

프로덕션 권장 사항: 레거시 비교를 위한 ROUGE-L, 의미적 중복을 위한 BERTScore, 일관성과 사실성을 위한 G-Eval을 보고하라. 사람이 레이블링한 50-100개 요약에 대해 보정(calibrate)하라.

### Step 4: 사실성 문제

추상적 요약은 환각에 취약하다. 추출적 요약은 출력이 원문에서 그대로 들어낸 것이므로 환각 위험이 훨씬 낮지만, 원문 문장이 맥락에서 분리되거나, 시대에 뒤떨어지거나, 순서가 뒤바뀐 채 인용되면 여전히 오도할 수 있다. 프로덕션 시스템이 컴플라이언스 인접 콘텐츠에 여전히 추출적 방법을 선호하는 가장 큰 이유가 이것이다.

짚어둘 환각 유형:

- **엔티티 교체(Entity swap).** 원문은 "John Smith"라고 한다. 요약은 "John Brown"이라고 한다.
- **숫자 표류(Number drift).** 원문은 "25,000"이라고 한다. 요약은 "25 million"이라고 한다.
- **극성 뒤집힘(Polarity flip).** 원문은 "rejected the offer"라고 한다. 요약은 "accepted the offer"라고 한다.
- **사실 날조(Fact invention).** 원문은 CEO를 언급하지 않는다. 요약은 CEO가 승인했다고 한다.

효과적인 평가 접근법:

- **FactCC.** 원문 문장과 요약 문장 사이의 함의(entailment)로 학습된 이진 분류기(classifier). 사실/비사실을 예측한다.
- **QA 기반 사실성.** 답이 원문에 있는 질문을 QA 모델에 묻는다. 요약이 다른 답을 뒷받침하면 플래그를 단다.
- **엔티티 수준 F1.** 원문과 요약의 개체명(named entity)을 비교한다. 요약에만 존재하는 엔티티는 의심스럽다.

사실성이 중요한 사용자 대면 콘텐츠(뉴스, 의료, 법률, 금융)에는 추출적 방식이 더 안전한 기본값이다. 추상적 방식은 루프 안에 사실성 검사가 필요하다.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 사용 사례 | 권장 |
|---------|-------------|
| 뉴스, 3-5문장 요약, 영어 | `facebook/bart-large-cnn` |
| 과학 논문 | `google/pegasus-pubmed` 또는 튜닝된 T5 |
| 다문서, 장문 | 32k+ 컨텍스트(context)를 가진 임의의 LLM, 프롬프트 사용 |
| 대화 요약 | `philschmid/bart-large-cnn-samsum` |
| 추출적, 구조적으로 낮은 환각 위험 | TextRank 또는 `sumy`의 LSA / LexRank |

2026년에는 컴퓨팅이 제약이 아닐 때 긴 컨텍스트를 가진 LLM이 종종 특화된 모델을 이긴다. 트레이드오프(trade-off)는 비용과 재현성이다. 특화된 모델은 더 일관된 출력을 준다.

## 산출물 (Ship It)

`outputs/skill-summary-picker.md`로 저장하라:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 연습 문제 (Exercises)

1. **Easy.** 뉴스 기사 5개에 TextRank를 실행하라. 상위 3개 문장을 참조 요약과 비교하라. ROUGE-L을 측정하라. CNN/DailyMail 스타일 기사에서 30-45 ROUGE-L을 보게 될 것이다.
2. **Medium.** 엔티티 수준 사실성을 구현하라. 원문과 요약에서 개체명을 추출하고(spaCy), 요약 안에서 원문 엔티티의 재현율(recall)과 원문 대비 요약 엔티티의 정밀도(precision)를 계산하라. 높은 정밀도와 낮은 재현율은 안전하지만 무뚝뚝함을, 낮은 정밀도는 환각된 엔티티를 의미한다.
3. **Hard.** CNN/DailyMail 기사 50개에 대해 BART-large-CNN과 LLM(Claude 또는 GPT-4)을 비교하라. ROUGE-L, 사실성(엔티티 F1 기준), 요약당 비용을 보고하라. 각각이 어디서 이기는지 문서화하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Extractive | 문장 고르기 | 원문에서 문장을 그대로 반환한다. 절대 환각하지 않는다. |
| Abstractive | 다시 쓰기 | 원문에 조건화된 새 텍스트를 생성한다. 환각할 수 있다. |
| ROUGE | 요약 지표 | 시스템 출력과 참조 사이의 n-gram / LCS 중복. |
| TextRank | 그래프 기반 추출적 | 문장 유사도 그래프에 대한 PageRank. |
| Factuality | 맞는가 | 요약의 주장이 원문에 의해 뒷받침되는지 여부. |
| Hallucination | 지어낸 내용 | 원문이 뒷받침하지 않는 요약 속 내용. |

## 더 읽을거리 (Further Reading)

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/): 추출적 표준 논문.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461): BART 논문.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777): Pegasus와 갭 문장 목표.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/): ROUGE 논문.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661): 사실성 지형 논문.
