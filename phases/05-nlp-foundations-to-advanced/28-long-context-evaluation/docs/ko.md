# 긴 컨텍스트 평가(Long-Context Evaluation): NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro는 1천만 토큰(token)의 컨텍스트를 광고한다. 100만 토큰에서 8-니들(needle) MRCR은 26.3%로 떨어진다. 광고된 것 ≠ 사용 가능한 것. 긴 컨텍스트 평가(long-context evaluation)는 그 위에서 출시하는 모델의 실제 용량을 알려준다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60분

## 문제 (The Problem)

200쪽짜리 계약서가 있다. 모델은 100만 토큰 컨텍스트를 주장한다. 계약서를 붙여 넣고 묻는다. "해지 조항이 뭔가?" 모델이 답한다. 하지만 표지에서 답한다. 해지 조항이 12만 토큰 깊이에, 모델이 실제로 주의를 기울이는 지점 너머에 있기 때문이다.

이것이 2026년 컨텍스트 용량 격차다. 사양표는 100만 또는 1천만이라 말한다. 현실은 그중 60~70%가 사용 가능하며, "사용 가능"은 과제에 따라 다르다고 말한다.

- **검색(건초더미 속 단일 니들):** 프런티어 모델에서 광고된 최댓값까지 거의 완벽.
- **다중 홉(multi-hop) / 집계:** 대부분 모델에서 약 128k를 넘으면 급격히 저하.
- **흩어진 사실에 대한 추론:** 가장 먼저 실패하는 과제.

긴 컨텍스트 평가는 이 축들을 측정한다. 이 레슨은 벤치마크(benchmark)들의 이름, 각각이 실제로 무엇을 측정하는지, 그리고 도메인별 커스텀 니들 테스트를 만드는 법을 짚는다.

## 개념 (The Concept)

![NIAH 베이스라인, RULER 다중 과제, LongBench 종합](../assets/long-context-eval.svg)

**건초더미 속 바늘(Needle-in-a-Haystack, NIAH, 2023).** 긴 컨텍스트의 통제된 깊이에 사실("the magic word is pineapple")을 둔다. 모델에 그것을 검색하라고 요청한다. 깊이 × 길이를 휩쓸어 본다. 원조 긴 컨텍스트 벤치마크. 이제 프런티어 모델들은 이를 포화(saturation)시킨다. 필요하지만 충분하지는 않은 베이스라인(baseline)이다.

**RULER(Nvidia, 2024).** 4개 범주에 걸친 13개 과제 유형: 검색(단일 / 다중 키 / 다중 값), 다중 홉 추적(변수 추적), 집계(공통 단어 빈도), QA. 설정 가능한 컨텍스트 길이(4k부터 128k+까지). NIAH는 포화시키지만 다중 홉에서 실패하는 모델을 드러낸다. 2024년 릴리스에서, 32k+ 컨텍스트를 주장한 17개 모델 중 절반만이 32k에서 품질을 유지했다.

**LongBench v2(2024).** 503개의 객관식 질문, 8k~2M 단어 컨텍스트, 여섯 개 과제 범주: 단일 문서 QA, 다중 문서 QA, 긴 인컨텍스트 학습(in-context learning), 긴 다이얼로그, 코드 저장소, 긴 구조화 데이터. 실세계 긴 컨텍스트 동작을 위한 프로덕션(production) 벤치마크.

**MRCR(Multi-Round Coreference Resolution).** 규모에서의 다중 턴 상호참조(coreference). 8-니들, 24-니들, 100-니들 변형. 어텐션(attention)이 저하되기 전에 모델이 몇 개의 사실을 저글링할 수 있는지 드러낸다.

**NoLiMa.** "비어휘적 니들(Non-lexical needle)." 니들과 쿼리가 문자적 중첩을 전혀 공유하지 않는다. 검색에 의미적 추론 한 단계가 필요하다. NIAH보다 어렵다.

**HELMET.** 많은 문서를 이어 붙이고, 그중 아무 하나에서 질문한다. 선택적 어텐션을 시험한다.

**BABILong.** 무관한 건초더미 안에 bAbI 추론 사슬을 임베딩한다. 단순 검색이 아니라 건초더미 속 추론(reasoning-in-a-haystack)을 시험한다.

### 실제로 보고할 것

- **광고된 컨텍스트 윈도우.** 사양표 숫자.
- **유효 검색 길이.** 어떤 임곗값(예: 90%)에서의 NIAH 통과.
- **유효 추론 길이.** 그 임곗값에서의 다중 홉 또는 집계 통과.
- **저하 곡선.** 컨텍스트 길이 대비 정확도, 과제 유형별로 그린 것.

사양표에 적을 두 숫자: 검색 유효(retrieval-effective)와 추론 유효(reasoning-effective). 보통 추론 유효는 광고된 윈도우의 25~50%다.

```figure
gx-niah-decay
```

## 직접 만들기 (Build It)

### Step 1: 당신의 도메인을 위한 커스텀 NIAH

`code/main.py`를 보라. 골격:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}를 휩쓸어 본다. 히트맵을 그려라. 그것이 대상 모델의 NIAH 카드다.

### Step 2: 다중 니들 변형

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"세 개의 마법 단어는 무엇인가?" 같은 질문은 세 개 모두 검색해야 한다. 단일 니들 성공이 다중 니들 성공을 예측하지 못한다.

### Step 3: 다중 홉 변수 추적(RULER 스타일)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

답은 세 개의 대입을 연쇄해야 한다. 128k의 프런티어 모델은 여기서 종종 정확도 50~70%로 떨어진다.

### Step 4: 당신의 스택에서 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

범주별 정확도를 보고하라. 집계 점수는 큰 과제 수준 차이를 숨긴다.

## 함정 (Pitfalls)

- **NIAH 전용 평가.** 100만 토큰에서 NIAH를 통과한다고 다중 홉에 대해 말해주는 것은 없다. 항상 RULER나 커스텀 다중 홉 테스트를 실행하라.
- **균일 깊이 샘플링.** 많은 구현이 depth=0.5만 테스트한다. depth=0, 0.25, 0.5, 0.75, 1.0을 테스트하라. "중간에서 길을 잃는(lost in the middle)" 효과는 실재한다.
- **필러(filler)와의 어휘 중첩.** 니들이 필러와 키워드를 공유하면 검색이 사소해진다. NoLiMa 스타일의 비중첩 니들을 사용하라.
- **지연 시간(latency) 무시.** 100만 토큰 프롬프트는 프리필(prefill)에 30~120초가 걸린다. 정확도와 함께 첫 토큰까지의 시간(time-to-first-token)을 측정하라.
- **벤더 자체 보고 수치.** OpenAI, Google, Anthropic 모두 자체 점수를 발표한다. 항상 자신의 사용 사례에서 독립적으로 다시 실행하라.

## 라이브러리로 써보기 (Use It)

2026년 스택:

| 상황 | 벤치마크 |
|-----------|-----------|
| 빠른 새너티 체크 | 3개 깊이 × 3개 길이의 커스텀 NIAH |
| 프로덕션용 모델 선택 | 대상 길이에서의 RULER(13개 과제) |
| 실세계 QA 품질 | LongBench v2 단일 문서 QA 부분집합 |
| 다중 홉 추론 | BABILong 또는 커스텀 변수 추적 |
| 대화형 / 다이얼로그 | 대상 길이에서의 MRCR 8-니들 |
| 모델 업그레이드 회귀(regression) | 고정된 사내 NIAH + RULER 하니스, 모든 새 모델에서 실행 |

프로덕션 경험칙: 의도한 길이에서 NIAH + 추론 과제 1개를 확보하기 전까지는 컨텍스트 윈도우를 절대 믿지 마라.

## 산출물 (Ship It)

`outputs/skill-long-context-eval.md`로 저장한다:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 연습 문제 (Exercises)

1. **쉬움.** 3개 깊이(0.25, 0.5, 0.75) × 3개 길이(1k, 4k, 16k)의 NIAH를 만들어라. 아무 모델에서 실행하라. 통과율을 3×3 히트맵으로 그려라.
2. **보통.** 3-니들 변형을 추가하라. 각 길이에서 3개 모두의 검색을 측정하라. 같은 길이에서의 단일 니들 통과율과 비교하라.
3. **어려움.** 64k의 필러 안에 임베딩된 변수 추적 과제(X1 → X2 → X3, 3홉)를 구성하라. 프런티어 모델 3개에 걸쳐 정확도를 측정하라. 모델별 유효 추론 길이를 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NIAH | 건초더미 속 바늘 | 필러에 사실을 심고, 모델에 그것을 검색하라고 요청한다. |
| RULER | 강화된 NIAH | 검색 / 다중 홉 / 집계 / QA에 걸친 13개 과제 유형. |
| 유효 컨텍스트(Effective context) | 진짜 용량 | 정확도가 임곗값 위에서 여전히 유지되는 길이. |
| 중간에서 길을 잃다(Lost in the middle) | 깊이 편향 | 모델이 긴 입력의 중간 내용에 덜 주의를 기울인다. |
| 다중 니들(Multi-needle) | 한 번에 많은 사실 | 여러 개를 심음. 검색만이 아니라 어텐션 저글링을 시험한다. |
| MRCR | 다중 라운드 상호참조 | 8, 24, 또는 100-니들 상호참조. 어텐션 포화를 드러낸다. |
| NoLiMa | 비어휘적 니들 | 니들과 쿼리가 문자적 토큰을 공유하지 않음. 추론이 필요하다. |

## 더 읽을거리 (Further Reading)

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — the multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — real-world long-context eval.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — harder needles.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — the depth-bias paper.
