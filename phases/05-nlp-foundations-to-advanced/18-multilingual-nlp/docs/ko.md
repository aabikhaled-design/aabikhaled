# Multilingual NLP

> 하나의 모델, 100개 이상의 언어, 그중 대부분은 학습 데이터 0개. 교차 언어 전이(cross-lingual transfer)는 2020년대의 실용적 기적이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45분

## 문제 (The Problem)

영어에는 레이블링된 예시가 수십억 개 있다. 우르두어에는 수천 개 있다. 마이틸리어에는 거의 없다. 전 세계 청중을 대상으로 하는 모든 실용적 NLP 시스템은 작업별 학습 데이터가 존재하지 않는 언어의 롱테일(long tail)에서 작동해야 한다.

다국어 모델(multilingual model)은 하나의 모델을 여러 언어로 동시에 학습시켜 이를 해결한다. 공유 표현(shared representation) 덕분에 모델은 고자원(high-resource) 언어에서 배운 기술을 저자원(low-resource) 언어로 전이한다. 영어 감성 분석(sentiment analysis)으로 모델을 파인튜닝(fine-tuning)하면, 별도 작업 없이 우르두어에서 놀랄 만큼 좋은 감성 예측을 만들어낸다. 그것이 제로샷 교차 언어 전이(zero-shot cross-lingual transfer)이며, NLP가 세상에 배포되는 방식을 재편했다.

이 레슨은 트레이드오프(trade-off), 표준 모델, 그리고 다국어 작업에 처음 들어선 팀들을 넘어뜨리는 한 가지 결정을 짚는다. 바로 전이를 위한 소스 언어 선택이다.

## 개념 (The Concept)

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**공유 어휘(Shared vocabulary).** 다국어 모델은 모든 대상 언어의 텍스트로 학습된 SentencePiece 또는 WordPiece 토크나이저(tokenizer)를 사용한다. 어휘가 공유된다. 같은 서브워드(subword) 단위가 관련 언어 전반에서 같은 형태소를 나타낸다. 영어와 이탈리아어의 `anti-`는 같은 토큰(token)을 받는다.

**공유 표현(Shared representation).** 여러 언어에 걸친 마스크 언어 모델링(masked language modeling)으로 사전 학습(pretraining)된 트랜스포머(transformer)는 서로 다른 언어로 된 의미적으로 유사한 문장이 유사한 은닉 상태(hidden state)를 만들어냄을 배운다. mBERT, XLM-R, NLLB 모두 이를 보인다. 영어 "cat"에 대한 임베딩(embedding)은 프랑스어 "chat"과 스페인어 "gato" 근처에 군집을 이루며, 전체 문장 임베딩도 마찬가지다.

**제로샷 전이(Zero-shot transfer).** 한 언어(보통 영어)의 레이블링된 데이터로 모델을 파인튜닝한다. 추론(inference) 시에는 모델이 지원하는 다른 어떤 언어에서든 실행한다. 대상 언어 레이블이 필요 없다. 유형론적으로 관련된 언어에는 결과가 강하고, 먼 언어에는 더 약하다.

**퓨샷 파인튜닝(Few-shot fine-tuning).** 대상 언어로 된 레이블링 예시 100-500개를 추가한다. 분류(classification) 작업에서 정확도가 영어 베이스라인(baseline)의 95-98%로 뛰어오른다. 이것이 다국어 NLP에서 가장 비용 효율적인 단일 레버다.

## 모델들 (The models)

| 모델 | 연도 | 커버리지 | 비고 |
|-------|------|----------|-------|
| mBERT | 2018 | 104개 언어 | 위키피디아로 학습. 첫 실용적 다국어 LM. 저자원에 약함. |
| XLM-R | 2019 | 100개 언어 | CommonCrawl(위키피디아보다 훨씬 큼)로 학습. 교차 언어 베이스라인을 세움. Base 270M, Large 550M. |
| XLM-V | 2023 | 100개 언어 | 100만 토큰 어휘(250k 대비)를 가진 XLM-R. 저자원에 더 우수. |
| mT5 | 2020 | 101개 언어 | 다국어 생성을 위한 T5 아키텍처. |
| NLLB-200 | 2022 | 200개 언어 | Meta의 번역 모델. 55개 저자원 언어 포함. |
| BLOOM | 2022 | 46개 언어 + 13개 프로그래밍 | 다국어로 학습된 개방형 176B LLM. |
| Aya-23 | 2024 | 23개 언어 | Cohere의 다국어 LLM. 아랍어, 힌디어, 스와힐리어에 강함. |

사용 사례에 따라 고르라. 분류는 합리적 기본값인 XLM-R-base로 잘 작동한다. 생성 작업은 번역이냐 개방형 생성이냐에 따라 mT5나 NLLB가 필요하다. LLM 스타일 작업은 Aya-23이나 명시적 다국어 프롬프팅(prompting)을 사용한 Claude와 짝을 이룬다.

## 소스 언어 결정 (2026 연구)

대부분의 팀은 파인튜닝 소스로 영어를 기본값으로 둔다. 최근 연구(2026)는 이것이 종종 잘못되었음을 보여준다.

언어 유사성이 원시 코퍼스(corpus) 크기보다 전이 품질을 더 잘 예측한다. 슬라브어 대상에는 독일어나 러시아어가 종종 영어를 이긴다. 인도어 대상에는 힌디어가 종종 영어를 이긴다. **qWALS** 유사도 지표(2026, World Atlas of Language Structures 특성 기반)가 이를 정량화한다. **LANGRANK**(Lin et al., ACL 2019)는 언어적 유사성, 코퍼스 크기, 계통적 관련성을 조합해 후보 소스 언어의 순위를 매기는 별개의 더 이른 방법이다.

실용 규칙: 대상 언어에 유형론적으로 가까운 고자원 친척이 있다면, 그것으로 먼저 파인튜닝을 시도한 다음 영어 파인튜닝과 비교하라.

```figure
n5-crosslingual-bridge
```

## 직접 만들기 (Build It)

### Step 1: 제로샷 교차 언어 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

하나의 모델, 세 개의 언어, 같은 API. NLI 데이터로 학습된 XLM-R은 함의(entailment) 트릭으로 분류에 잘 전이된다.

### Step 2: 다국어 임베딩 공간

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

번역은 임베딩 공간에서 가까이 자리한다. 다른 영어 문장은 더 멀리 자리한다. 교차 언어 검색, 군집화, 유사도를 작동하게 만드는 것이 바로 이것이다.

### Step 3: 퓨샷 파인튜닝 전략

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

대상 언어 예시 100-500개에는 `num_train_epochs=5`와 `learning_rate=2e-5`가 안전한 기본값이다. 학습률(learning rate)이 더 높으면 다국어 정렬(alignment)이 붕괴해 영어 전용 모델이 되고 만다.

## 실제로 작동하는 평가

- **홀드아웃(held-out) 세트에서의 언어별 정확도.** 집계하지 마라. 집계는 롱테일을 숨긴다.
- **단일 언어 베이스라인과의 벤치마크(benchmark).** 충분한 데이터가 있는 언어라면, 밑바닥부터 학습된 단일 언어 모델이 때때로 다국어 모델을 이긴다. 테스트하라.
- **엔티티 수준 테스트.** 대상 언어의 개체명(named entity). 다국어 모델은 라틴 문자에서 먼 문자 체계에 토큰화가 약할 때가 많다.
- **교차 언어 일관성.** 두 언어로 된 같은 의미는 같은 예측을 만들어야 한다. 그 격차를 측정하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 작업 | 권장 |
|-----|-------------|
| 분류, 100개 언어 | 파인튜닝된 XLM-R-base (~270M) |
| 제로샷 텍스트 분류 | `joeddav/xlm-roberta-large-xnli` |
| 다국어 문장 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 번역, 200개 언어 | `facebook/nllb-200-distilled-600M` (레슨 11 참조) |
| 생성형 다국어 | Claude, GPT-4, Aya-23, mT5-XXL |
| 저자원 언어 NLP | XLM-V 또는 관련 고자원 언어에 대한 도메인별 파인튜닝 |

성능이 중요하다면 항상 대상 언어에서의 파인튜닝을 위한 예산을 잡아라. 제로샷은 출발점이지 최종 답이 아니다.

### 토큰화 세금 (저자원 언어에서 무엇이 잘못되는가)

다국어 모델은 모든 언어에 걸쳐 하나의 토크나이저를 공유한다. 그 어휘는 영어, 프랑스어, 스페인어, 중국어, 독일어가 지배하는 코퍼스로 학습된다. 이 지배적 집합 밖의 언어라면 어떤 언어든, 세 가지 세금이 조용히 복리로 쌓인다.

- **다산성 세금(Fertility tax).** 저자원 언어 텍스트는 영어보다 단어당 훨씬 많은 토큰으로 토큰화된다. 힌디어 문장은 동등한 영어 문장의 3-5배 토큰이 필요하기도 하다. 그 3-5배가 컨텍스트 윈도우(context window), 학습 효율, 지연 시간(latency)을 잡아먹는다.
- **변형 복구 세금(Variant recovery tax).** 모든 오타, 발음 구별 부호 변형, 유니코드 정규화 불일치, 대소문자 변형이 임베딩 공간에서 콜드 스타트(cold-start) 무관 수열이 된다. 모델은 원어민이 당연하게 여기는 철자 대응을 배울 수 없다.
- **용량 유출 세금(Capacity spillover tax).** 세금 1과 2는 컨텍스트 위치, 층(layer) 깊이, 임베딩 차원을 소비한다. 실제 추론을 위해 남는 것은 같은 모델에서 고자원 언어가 얻는 것보다 체계적으로 더 작다.

실용적 증상: 모델이 힌디어에서 정상적으로 학습되고, 손실(loss) 곡선이 맞아 보이고, 평가 펄플렉시티(perplexity)가 합리적으로 보이지만, 프로덕션 출력은 미묘하게 틀린다. 형태론이 문장 중간에 붕괴한다. 희귀한 굴절이 복구 불가능한 채로 남는다. **망가진 토크나이저는 데이터 규모 확장으로 빠져나갈 수 없다.**

완화책: 대상 언어에 대한 좋은 커버리지를 가진 토크나이저를 고르라(XLM-V의 100만 토큰 어휘가 직접적인 해결책이다). 학습 전에 홀드아웃 대상 텍스트에서 토큰화 다산성을 검증하라. 진정으로 롱테일인 문자 체계에는 바이트 수준 폴백(byte-level fallback)(SentencePiece `byte_fallback=True`, GPT-2 스타일 바이트 수준 BPE)을 써서 무엇도 절대 OOV가 되지 않게 하라.

## 산출물 (Ship It)

`outputs/skill-multilingual-picker.md`로 저장하라:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 연습 문제 (Exercises)

1. **Easy.** 영어, 프랑스어, 힌디어, 아랍어에 걸쳐 언어당 10개 문장에 제로샷 분류 파이프라인(pipeline)을 실행하라. 각각의 정확도를 보고하라. 강한 프랑스어, 괜찮은 힌디어, 들쭉날쭉한 아랍어가 나올 것이다.
2. **Medium.** `paraphrase-multilingual-MiniLM-L12-v2`를 사용해 작은 혼합 언어 코퍼스에 대한 교차 언어 검색기를 만들어라. 영어로 질의하고, 어떤 언어로든 문서를 검색하라. recall@5를 측정하라.
3. **Hard.** 힌디어 분류 작업에 대해 영어 소스 파인튜닝과 힌디어 소스 파인튜닝을 비교하라. 두 방식 모두 퓨샷 파인튜닝에 대상 언어 예시 500개를 사용하라. 어느 소스가 더 나은 힌디어 정확도를 만들고 얼마나 차이 나는지 보고하라. 이것이 축소판 LANGRANK 명제다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Multilingual model | 하나의 모델, 여러 언어 | 언어 전반에 걸친 공유 어휘와 파라미터(parameter). |
| Cross-lingual transfer | 한 언어로 학습, 다른 언어로 실행 | 소스에서 파인튜닝하고, 대상 언어 레이블 없이 대상에서 평가. |
| Zero-shot | 대상 언어 레이블 없음 | 대상 언어에서 파인튜닝 없이 전이. |
| Few-shot | 적은 대상 레이블 | 파인튜닝에 사용되는 100-500개의 대상 언어 예시. |
| mBERT | 첫 다국어 LM | 위키피디아로 사전 학습된 104개 언어 BERT. |
| XLM-R | 표준 교차 언어 베이스라인 | CommonCrawl로 사전 학습된 100개 언어 RoBERTa. |
| NLLB | Meta의 200개 언어 MT | No Language Left Behind. 55개 저자원 언어 포함. |

## 더 읽을거리 (Further Reading)

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116): XLM-R 논문.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502): 교차 언어 전이 연구 라인을 시작한 분석 논문.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672): NLLB-200 논문.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827): Aya, Cohere의 다국어 LLM.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65): qWALS / LANGRANK 소스 언어 논문.
