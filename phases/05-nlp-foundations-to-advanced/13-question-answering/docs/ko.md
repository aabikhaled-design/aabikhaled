# Question Answering Systems

> 세 가지 시스템이 현대 QA를 빚어냈다. 추출적(extractive) 방식은 스팬(span)을 찾았다. 검색 증강(retrieval-augmented) 방식은 그 답을 문서에 근거시켰다. 생성적(generative) 방식은 답을 만들어냈다. 모든 현대 AI 어시스턴트는 이 셋의 혼합이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75분

## 문제 (The Problem)

사용자가 "When did the first iPhone launch?"라고 입력하면 "June 29, 2007"을 기대한다. "Apple's history is long and varied"가 아니다. 문장 없이 고립되어 있는 "2007"도 아니다. 직접적이고, 근거 있고, 정확한 답이다.

지난 10년간 세 가지 아키텍처가 QA를 지배했다.

- **추출적 QA(Extractive QA).** 질문과 답을 포함한다고 알려진 지문(passage)이 주어지면, 지문 안 답 스팬의 시작과 끝 인덱스를 찾는다. SQuAD가 표준 벤치마크(benchmark)다.
- **오픈 도메인 QA(Open-domain QA).** 지문이 주어지지 않는다. 먼저 관련 지문을 검색(retrieve)한 다음 답을 추출하거나 생성한다. 이것이 오늘날 모든 RAG 파이프라인(pipeline)의 토대다.
- **생성적 / 클로즈드북 QA(Generative / Closed-book QA).** 대형 언어 모델이 자신의 파라미터(parameter) 기억으로부터 답한다. 검색이 없다. 추론(inference)이 가장 빠르고, 사실에 가장 덜 신뢰할 만하다.

2026년의 추세는 하이브리드다. 가장 좋은 지문 몇 개를 검색한 다음, 그 지문들에 근거하여 답하도록 생성 모델에 프롬프트(prompt)한다. 그것이 RAG이며, 레슨 14가 검색 측면을 깊이 다룬다. 이 레슨은 QA 측면을 만든다.

## 개념 (The Concept)

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**추출적(Extractive).** 질문과 지문을 트랜스포머(transformer)(BERT 계열)로 함께 인코딩한다. 답의 시작과 끝 토큰(token) 인덱스를 예측하는 두 개의 헤드(head)를 학습시킨다. 손실(loss)은 유효한 위치에 대한 교차 엔트로피(cross-entropy)다. 출력은 지문에서 나온 스팬이다. (구조상) 절대 환각(hallucination)하지 않으며, (구조상) 지문이 답할 수 없는 질문은 절대 다루지 못한다.

**검색 증강(Retrieval-augmented, RAG).** 두 단계다. 첫째, 검색기(retriever)가 코퍼스(corpus)에서 상위 `k`개 지문을 찾는다. 둘째, 리더(reader)(추출적 또는 생성적)가 그 지문들을 사용해 답을 만든다. 검색기와 리더를 나누면 각각을 따로 학습하고 평가할 수 있다. 현대 RAG는 종종 그 사이에 리랭커(reranker)를 추가한다.

**생성적(Generative).** 디코더 전용(decoder-only) LLM(GPT, Claude, Llama)이 학습된 가중치(weight)로부터 답한다. 검색 단계가 없다. 일반 지식에는 탁월하고, 희귀하거나 최근 사실에는 치명적이다. 환각률은 사전 학습(pretraining) 데이터에서의 사실 빈도와 반비례한다.

```figure
qa-span
```

## 직접 만들기 (Build It)

### Step 1: 사전 학습된 모델을 사용한 추출적 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`는 답할 수 없는 질문을 포함하는 SQuAD 2.0으로 학습되었다. 기본적으로 `question-answering` 파이프라인은 모델의 null 점수가 이기더라도 가장 높은 점수의 스팬을 반환한다. 자동으로 빈 답을 반환하지 *않는다*. 명시적인 "답 없음" 동작을 얻으려면 파이프라인 호출에 `handle_impossible_answer=True`를 전달하라. 그러면 파이프라인은 null 점수가 모든 스팬 점수를 초과할 때만 빈 답을 반환한다. 어느 경우든 항상 `score` 필드를 확인하라.

### Step 2: 검색 증강 파이프라인 (스케치)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

두 단계 파이프라인이다. 밀집 검색기(dense retriever)(Sentence-BERT)가 의미적 유사도(similarity)로 관련 지문을 찾는다. 추출적 리더(RoBERTa-SQuAD)가 결합된 상위 지문에서 답 스팬을 끌어낸다. 작은 코퍼스에서 작동한다. 백만 문서 코퍼스에는 FAISS나 벡터 데이터베이스를 사용하라.

### Step 3: RAG를 사용한 생성적 방식

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

프롬프트 패턴이 중요하다. 맥락에 근거하고 맥락이 불충분할 때 "I don't know"를 반환하라고 모델에게 명시적으로 말하면, 순진한 프롬프팅에 비해 환각률을 40-60% 줄인다. 더 정교한 패턴은 인용, 신뢰도 점수, 구조화된 추출을 추가한다.

### Step 4: 현실 세계를 반영하는 평가

SQuAD는 **Exact Match (EM)**와 **토큰 수준 F1**을 사용한다. EM은 정규화(소문자화, 구두점 제거, 관사 제거) 후의 엄격한 일치다. 예측이 정확히 일치하거나 0점을 받는다. F1은 예측과 참조(reference) 사이의 토큰 중복으로 계산되며 부분 점수를 준다. 둘 다 패러프레이즈(paraphrase)를 과소 평가한다. "June 29, 2007" 대 "June 29th, 2007"은 보통 EM 0점을 받지만(서수가 정규화를 깨뜨린다) 겹치는 토큰에서 상당한 F1을 여전히 얻는다.

프로덕션(production) QA의 경우:

- **답 정확도(Answer accuracy)**(LLM 판정 또는 사람 판정. 지표가 의미적 동등성을 포착하지 못하기 때문).
- **인용 정확도(Citation accuracy).** 인용된 지문이 실제로 답을 뒷받침하는가? 생성된 인용과 검색된 지문 사이의 문자열 일치로 자동으로 확인하기 쉽다.
- **거부 보정(Refusal calibration).** 답이 검색된 지문에 없을 때, 시스템이 올바르게 "I don't know"라고 말하는가? 잘못된 확신율을 측정하라.
- **검색 재현율(Retrieval recall).** 리더를 평가하기 전에, 검색기가 올바른 지문을 상위 `k`개 안으로 가져오는지 측정하라. 리더는 누락된 지문을 고칠 수 없다.

### RAGAS: 2026 프로덕션 평가 프레임워크

`RAGAS`는 RAG 시스템을 위해 특별히 만들어졌으며 2026년의 배포 기본값이다. 골드 참조(gold reference)를 요구하지 않고 네 가지 차원을 채점한다.

- **충실성(Faithfulness).** 답의 각 주장이 검색된 맥락에서 나오는가? NLI 기반 함의(entailment)로 측정한다. 주요 환각 지표다.
- **답 관련성(Answer relevance).** 답이 질문을 다루는가? 답으로부터 가상의 질문을 생성하고 실제 질문과 비교하여 측정한다.
- **맥락 정밀도(Context precision).** 검색된 청크(chunk) 중 실제로 관련 있던 비율은? 낮은 정밀도 = 프롬프트의 잡음.
- **맥락 재현율(Context recall).** 검색된 집합이 필요한 모든 정보를 담았는가? 낮은 재현율 = 리더가 성공할 수 없음.

참조 없는 채점은 큐레이션된 골드 답 없이 실시간 프로덕션 트래픽에서 평가할 수 있게 해준다. 정확 일치 지표가 쓸모없는 개방형 질문에는 그 위에 LLM-as-judge를 얹어라.

`pip install ragas`. 검색기 + 리더를 연결하라. 쿼리당 네 개의 스칼라를 얻는다. 회귀(regression)에 알림을 설정하라.

## 라이브러리로 써보기 (Use It)

2026년 스택.

| 사용 사례 | 권장 |
|---------|-------------|
| 지문 주어짐, 답 스팬 찾기 | `deepset/roberta-base-squad2` |
| 고정 코퍼스 대상, 클로즈드북 허용 불가 | RAG: 밀집 검색기 + LLM 리더 |
| 문서 저장소에 대한 실시간 | 하이브리드(BM25 + 밀집) 검색기 + 리랭커를 갖춘 RAG (레슨 14) |
| 대화형 QA (후속 질문) | 대화 기록 + 매 턴마다 RAG를 갖춘 LLM |
| 고도로 사실적인, 규제 도메인 | 권위 있는 코퍼스에 대한 추출적 방식. 절대 생성적 방식만으로는 안 됨 |

추출적 QA는 LLM을 사용한 RAG가 더 많은 경우를 처리하기 때문에 2026년에는 유행이 지났다. 그래도 문자 그대로의 인용이 필요한 맥락에서는 여전히 배포된다. 법률 조사, 규제 컴플라이언스, 감사 도구가 그것이다.

## 산출물 (Ship It)

`outputs/skill-qa-architect.md`로 저장하라:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 연습 문제 (Exercises)

1. **Easy.** 위의 SQuAD 추출적 파이프라인을 위키피디아 지문 10개에 설정하라. 질문 10개를 직접 만들어라. 답이 얼마나 자주 맞는지 측정하라. 지문과 질문이 깔끔하다면 7-9개가 맞는 것을 볼 것이다.
2. **Medium.** 거부 분류기(classifier)를 추가하라. 최상위 검색 점수가 임계값(가령 코사인 0.3) 아래일 때, 리더를 호출하는 대신 "I don't know"를 반환하라. 홀드아웃(held-out) 세트에서 임계값을 튜닝하라.
3. **Hard.** 직접 고른 10,000개 문서 코퍼스에 대한 RAG 파이프라인을 만들어라. RRF 융합(fusion)으로 하이브리드 검색(BM25 + 밀집)을 구현하라(레슨 14 참조). 하이브리드 단계가 있을 때와 없을 때의 답 정확도를 측정하라. 어떤 질문 유형이 가장 이득을 보는지 문서화하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Extractive QA | 답 스팬 찾기 | 주어진 지문 안에서 답의 시작과 끝 인덱스를 예측한다. |
| Open-domain QA | 코퍼스에 대한 QA | 지문이 주어지지 않음. 검색한 다음 답해야 한다. |
| RAG | 검색한 다음 생성 | 검색 증강 생성. 검색기 + 리더 파이프라인. |
| SQuAD | 표준 벤치마크 | Stanford Question Answering Dataset. EM + F1 지표. |
| Hallucination | 지어낸 답 | 검색된 맥락이 뒷받침하지 않는 리더 출력. |
| Refusal calibration | 입 다물 때를 안다 | 답할 수 없을 때 시스템이 올바르게 "I don't know"라고 말함. |

## 더 읽을거리 (Further Reading)

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250): 벤치마크 논문.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906): DPR, QA를 위한 표준 밀집 검색기.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401): RAG라는 이름을 붙인 논문.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997): 포괄적인 RAG 서베이.
