# LLM 평가(LLM Evaluation): RAGAS, DeepEval, G-Eval

> 정확 일치(exact-match)와 F1은 의미적 동등성을 놓친다. 사람 검토는 확장되지 않는다. LLM을 심판으로 쓰는 것(LLM-as-judge)이 프로덕션(production)의 답이다. 그 숫자를 신뢰할 만큼 충분한 보정(calibration)과 함께라면.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75분

## 문제 (The Problem)

RAG 시스템이 답한다. "June 29th, 2007."
정답(gold) 참조는 이렇다. "June 29, 2007."
정확 일치는 0점이다. F1은 약 75%다. 사람은 100점을 줄 것이다.

이제 1만 개의 테스트 케이스를 곱하라. 검색기, 청킹(chunking), 프롬프트(prompt), 모델을 바꿀 때마다 다시 곱하라. 의미를 이해하고, 규모에서 저렴하게 실행되며, 회귀(regression)에 대해 거짓말하지 않고, 올바른 실패 모드를 드러내는 평가기가 필요하다.

2026년에는 이 문제를 장악한 세 가지 프레임워크가 있다.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. NLI + LLM 심판 백엔드를 쓰는 네 가지 RAG 지표(충실도faithfulness, 답변 관련성answer-relevance, 컨텍스트 정밀도context-precision, 컨텍스트 재현율context-recall). 연구 기반이며, 경량이다.
- **DeepEval.** LLM을 위한 Pytest. G-Eval, 과제 완수(task-completion), 환각(hallucination), 편향(bias) 지표. CI/CD 네이티브.
- **G-Eval.** 하나의 방법(이자 DeepEval 지표): 사고 연쇄(chain-of-thought), 커스텀 기준, 0~1 점수를 동반한 LLM-as-judge.

세 가지 모두 LLM-as-judge에 기댄다. 이 레슨은 그 방법과 그것을 둘러싼 신뢰 계층에 대한 직관을 쌓는다.

## 개념 (The Concept)

![네 가지 평가 차원, LLM-as-judge 아키텍처](../assets/llm-evaluation.svg)

**LLM-as-judge.** 정적 지표를, 평가 기준표(rubric)가 주어졌을 때 출력을 채점하는 LLM으로 대체한다. `(query, context, answer)`가 주어지면, 심판 LLM에 프롬프트한다. "충실도를 0~1로 채점하라." 점수를 반환한다.

작동하는 이유: LLM은 사람 판단을 아주 적은 비용으로 근사한다. 채점된 케이스당 약 $0.003인 GPT-4o-mini는 $5 미만으로 1000 샘플 회귀 평가 실행을 가능하게 한다.

조용히 실패하는 이유:

1. **심판 편향.** 심판은 더 긴 답변, 자기 모델 계열의 답변, 프롬프트 스타일에 맞는 답변을 선호한다.
2. **JSON 파싱 실패.** 잘못된 JSON → NaN 점수 → 집계에서 조용히 제외됨. RAGAS 사용자는 이 고통을 안다. try/except + 명시적 실패 모드로 게이트하라.
3. **모델 버전에 따른 표류(drift).** 심판을 업그레이드하면 모든 지표가 바뀐다. 심판 모델 + 버전을 고정하라.

**RAG 4종.**

| 지표 | 질문 | 백엔드 |
|--------|----------|---------|
| 충실도(Faithfulness) | 답변의 각 주장이 검색된 컨텍스트에서 오는가? | NLI 기반 함의(entailment) |
| 답변 관련성(Answer relevance) | 답변이 질문을 다루는가? | 답변에서 가상의 질문을 생성하고, 실제 질문과 비교 |
| 컨텍스트 정밀도(Context precision) | 검색된 청크 중 관련 있던 비율은? | LLM 심판 |
| 컨텍스트 재현율(Context recall) | 검색이 필요한 모든 것을 반환했는가? | 정답에 대한 LLM 심판 |

**G-Eval.** 커스텀 기준을 정의한다. "답변이 올바른 출처를 인용했는가?" 프레임워크가 이를 사고 연쇄 평가 단계로 자동 확장한 다음, 0~1로 채점한다. RAGAS가 다루지 않는 도메인 특화 품질 차원에 좋다.

**보정(Calibration).** 사람 레이블(label)에 대한 상관관계를 확보하기 전까지는 원시 심판 점수를 절대 믿지 마라. 손으로 레이블링한 예시 100개를 실행하라. 심판 vs 사람을 그려라. 스피어만 로(Spearman rho)를 계산하라. rho < 0.7이면 심판 기준표를 손봐야 한다.

```figure
n5-judge-gauge
```

## 직접 만들기 (Build It)

### Step 1: NLI로 충실도(RAGAS 스타일)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

답변을 원자적 주장(atomic claim)으로 분해한다. 각 주장을 검색된 컨텍스트와 NLI로 대조 검사한다. 충실도 = 뒷받침되는 비율.

### Step 2: 답변 관련성

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

답변이 질문된 것과 다른 질문을 암시하면, 관련성이 떨어진다.

### Step 3: G-Eval 커스텀 지표

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

평가 단계가 기준표다. 명시적 단계는 암묵적인 "0~1로 채점하라" 프롬프트보다 더 안정적이다.

### Step 4: CI 게이트

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

pytest 파일로 출시하라. 모든 PR에서 실행하라. 회귀가 있으면 머지를 막아라.

### Step 5: 밑바닥부터 만든 장난감 평가

`code/main.py`를 보라. 충실도(답변 주장과 컨텍스트의 중첩)와 관련성(답변 토큰과 질문 토큰의 중첩)의 표준 라이브러리(stdlib) 전용 근사. 프로덕션용이 아니다. 형태를 보여준다.

## 함정 (Pitfalls)

- **보정 없음.** 사람 레이블과 상관관계 0.3인 심판은 노이즈다. 출시 전에 보정 실행을 요구하라.
- **자기 평가.** 같은 LLM으로 생성하고 심판하면 점수가 10~20% 부풀려진다. 심판에는 다른 모델 계열을 쓰라.
- **쌍별 심판에서의 위치 편향.** 심판은 먼저 제시된 옵션을 선호한다. 항상 순서를 무작위화하고 둘 다 실행하라.
- **원시 집계가 실패를 숨긴다.** 평균 점수 0.85는 종종 5%의 치명적 실패를 숨긴다. 항상 하위 분위수를 검사하라.
- **골든 데이터셋 부패.** 시간이 지나며 표류하는 버전 없는 평가 세트는 종단적 비교를 망친다. 모든 변경마다 데이터셋에 태그를 달아라.
- **LLM 비용.** 규모에서는 심판 호출이 비용을 지배한다. 보정 임곗값을 충족하는 가장 저렴한 모델을 쓰라. GPT-4o-mini, Claude Haiku, Mistral-small.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 사용 사례 | 프레임워크 |
|---------|-----------|
| RAG 품질 모니터링 | RAGAS(4개 지표) |
| CI/CD 회귀 게이트 | DeepEval + pytest |
| 커스텀 도메인 기준 | DeepEval 안의 G-Eval |
| 온라인 실시간 트래픽 모니터링 | 참조 없는(reference-free) 모드의 RAGAS |
| 인간 개입(human-in-the-loop) 스폿 체크 | 주석 UI를 갖춘 LangSmith 또는 Phoenix |
| 레드팀 / 안전 평가 | Promptfoo + DeepEval |

전형적 스택: 모니터링에는 RAGAS, CI에는 DeepEval, 새로운 차원에는 G-Eval. 셋 다 실행하라. 그들은 유용하게 서로 의견이 갈린다.

## 산출물 (Ship It)

`outputs/skill-eval-architect.md`로 저장한다:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 연습 문제 (Exercises)

1. **쉬움.** 알려진 환각이 있는 RAG 예시 10개에 RAGAS를 사용하라. 충실도 지표가 각각을 잡아내는지 검증하라.
2. **보통.** QA 답변 50개를 정확성에 대해 0~1로 손으로 레이블링하라. G-Eval로 채점하라. 심판과 사람 사이의 스피어만 로를 측정하라.
3. **어려움.** DeepEval로 pytest CI 게이트를 만들어라. 의도적으로 검색기를 회귀시켜라. 게이트가 실패하는지 검증하라. 최하위 10%에 대한 임곗값 검사를 통해 하위 분위수 알림을 추가하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| LLM-as-judge | LLM으로 채점하기 | 기준표가 주어졌을 때 출력을 0~1로 채점하도록 심판 모델에 프롬프트한다. |
| RAGAS | 그 RAG 지표 라이브러리 | 참조 없는 4개 RAG 지표를 가진 오픈소스 평가 프레임워크. |
| 충실도(Faithfulness) | 답변이 근거가 있는가? | 검색된 컨텍스트에 의해 함의되는 답변 주장의 비율. |
| 컨텍스트 정밀도(Context precision) | 검색된 청크가 관련 있었나? | 실제로 중요했던 top-K 청크의 비율. |
| 컨텍스트 재현율(Context recall) | 검색이 모든 것을 찾았나? | 검색된 청크에 의해 뒷받침되는 정답 주장의 비율. |
| G-Eval | 커스텀 LLM 심판 | 기준표 + 사고 연쇄 평가 단계 + 0~1 점수. |
| 보정(Calibration) | 믿되 검증하라 | 심판 점수와 사람 점수 사이의 스피어만 상관관계. |

## 더 읽을거리 (Further Reading)

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — the RAGAS paper.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — the G-Eval paper.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases, calibration, limits.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — unifying framework that integrates RAGAS, DeepEval, Phoenix.
