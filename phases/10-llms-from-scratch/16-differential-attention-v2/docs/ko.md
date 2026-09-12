# 차분 어텐션 (V2) (Differential Attention (V2))

> 소프트맥스 어텐션(softmax attention)은 일치하지 않는 모든 토큰(token)에 소량의 확률을 퍼뜨린다. 10만 토큰에 걸쳐 그 노이즈(noise)가 쌓이면 신호가 파묻힌다. Differential Transformer (Ye et al., ICLR 2025)는 어텐션을 두 소프트맥스의 차분(difference)으로 계산하여, 공유된 노이즈 바닥(noise floor)을 빼서 이를 고친다. DIFF V2 (Microsoft, 2026년 1월)는 프로덕션 스택(production-stack) 재작성이다. 디코드 지연 시간(decode latency)을 베이스라인(baseline) Transformer에 맞추고, 커스텀 커널(custom kernel)이 없으며, FlashAttention 호환이다. 이 레슨은 V1에서 V2까지의 엔드투엔드이며, stdlib Python에서 실행할 수 있는 차분 연산의 작동하는 장난감 구현을 곁들인다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 소프트맥스 어텐션이 왜 노이즈 바닥을 갖는지, 그리고 그것이 왜 컨텍스트 길이(context length)에 따라 커지는지 정확히 진술하기.
- 차분 어텐션(differential attention) 공식을 유도하고, 왜 그 뺄셈이 신호를 보존하면서 공유된 노이즈 성분을 상쇄하는지 설명하기.
- V1에서 V2로의 diff를 따라가기: 무엇이 더 빨라졌고, 무엇이 더 단순해졌고, 무엇이 더 안정되었고, 각 변경이 왜 프로덕션 사전 학습(pre-training)에 필요했는지.
- 순수 Python으로 차분 어텐션을 밑바닥부터 구현하고, 합성 신호 더하기 노이즈 쿼리(query)에서 노이즈 상쇄 속성을 경험적으로 검증하기.

## 문제 (The Problem)

표준 소프트맥스 어텐션에는 규모가 커지면 운영상의 골칫거리로 변하는 수학적 속성이 있다. 쿼리 `q`에 대해, 어텐션 가중치(attention weight)는 `softmax(qK^T / sqrt(d))`다. 소프트맥스는 결코 정확한 0을 생산할 수 없다. 일치하지 않는 모든 토큰이 어느 정도의 양의 질량(positive mass)을 받는다. 그 잔여 질량은 노이즈이며, 컨텍스트 길이에 따라 커진다. 128k 토큰에서, 일치하지 않는 각 토큰이 확률의 0.001%만 받더라도, 그중 127,999개가 합쳐지면 전체의 약 12%를 기여한다. 모델은 컨텍스트에 따라 커지는 노이즈 바닥을 우회하는 법을 배워야 한다.

경험적으로 이것은 어텐션 헤드(attention head) 간섭으로 나타난다. 긴 컨텍스트 RAG에서의 환각된 인용(hallucinated citation), 100k 토큰 검색 작업에서의 중간 분실(lost-in-the-middle) 실패, 그리고 32k를 넘는 건초 더미 속 바늘(needle-in-haystack) 벤치마크(benchmark)에서의 미묘한 정확도 저하. Differential Transformer 논문(arXiv:2410.05258, ICLR 2025)은 그 격차를 측정했다. DIFF Transformer는 같은 크기의 베이스라인보다 더 낮은 퍼플렉시티(perplexity), 더 높은 긴 컨텍스트 정확도, 더 적은 환각에 도달했다.

DIFF V1에는 그것을 프런티어 사전 학습 파이프라인(pipeline)에서 배제한 세 가지 문제가 있었다. 값 캐시(value cache)를 디코드 스텝당 두 번 적재해야 했고, FlashAttention 호환성을 깨는 커스텀 CUDA 커널을 필요로 했으며, 헤드별 RMSNorm이 70B 이상 규모에서 장기 학습을 불안정하게 만들었다. DIFF V2 (Microsoft unilm 블로그, 2026년 1월 20일)는 세 가지 모두를 고쳤다. 이 레슨은 두 버전을 모두 따라가고, 차분 연산자를 만들고, 장난감 쿼리에서 노이즈 상쇄를 벤치마크한다.

## 개념 (The Concept)

### 소프트맥스의 노이즈 바닥 (The noise floor of softmax)

쿼리 `q`와 키 `K = [k_1, ..., k_N]`에 대해, 어텐션 가중치는:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

어떤 `w_i`도 결코 0이 아니다. `k_i`가 `q`와 완전히 무관하면, 점수 `q . k_i`는 0이 아니다. 분산 `||q||^2 / d`로 0 주위에서 요동친다. 소프트맥스 정규화 후, 각 무관한 토큰은 여전히 가중 합에 `O(1/N)`을 기여한다. 무관한 토큰의 총 기여는 `O((N-1)/N) = O(1)`이다. 작은 양이 아니다.

모델이 원하는 것은 하드 top-k 같은 것이다. 일치하는 토큰에 높은 가중치, 그 외 모든 곳에 거의 0인 가중치. 소프트맥스는 그것을 직접 하기에는 너무 매끄럽다.

### 차분 아이디어 (The differential idea)

각 헤드의 Q와 K 투영(projection)을 둘로 나눈다: Q = (Q_1, Q_2)와 K = (K_1, K_2). 두 어텐션 맵을 계산한다:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

출력:

```
DiffAttn = (A_1 - lambda * A_2) V
```

뺄셈은 두 맵이 공유하는 어떤 노이즈 분포(distribution)든 상쇄한다. 두 맵이 127k개의 무관한 토큰에 대략 균등한 가중치를 가지면(무작위 초기화에서 그럴 것이다), 그것들은 상쇄된다. 신호(실제로 관련 있는 소수의 토큰에 뾰족한 가중치)는 두 맵에 같은 크기로 나타날 때만 상쇄되는데, 모델이 학습되고 나면 그렇지 않을 것이다.

`lambda`는 헤드별 학습 가능한 스칼라로, `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`로 매개변수화된다. 음수일 수 있다. `lambda_init`은 0.8 같은 작은 양수로 기본 설정된다.

### 왜 이것이 헤드폰의 노이즈 캔슬링과 일치하는가 (Why this matches headed noise-canceling)

같은 목소리를 녹음하는 두 개의 노이즈가 있는 마이크를 생각해 보라. 둘 다 화자 더하기 상관된 배경 소음을 잡는다. 하나를 다른 하나에서 빼면 공유된 소음이 떨어져 나간다. 두 신호가 위상이나 진폭에서 완전한 상쇄를 막을 만큼 충분히 다르기 때문에 목소리는 살아남는다. 헤드별 `lambda`는 정확히 이 균형을 학습한다.

### V1 vs V2: 그 diff (V1 vs V2: the diff)

V1은 파라미터(parameter) 수를 베이스라인 Transformer와 같게 유지했다. 헤드당 두 개의 쿼리를 얻기 위해 헤드 차원(head dimension)을 절반으로 줄였다. 그것은 헤드 표현력을 희생했고, 더 뼈아프게는 헤드당 값 캐시를 절반으로 줄였다. 디코드는 스텝당 값 캐시를 두 번 적재해야 했다(소프트맥스 분기당 한 번). 결과: 파라미터 수가 일치함에도 디코드가 베이스라인보다 느렸다.

V2는 쿼리 헤드 수를 두 배로 하고 KV 헤드는 같게 유지한다(상향 투영, up-projection에서 파라미터를 빌림). 헤드 차원은 베이스라인과 같게 유지된다. 뺄셈 후, 여분의 차원은 베이스라인 Transformer의 O_W 투영에 맞추기 위해 다시 아래로 투영된다. 세 가지가 동시에 일어난다:

1. 디코드 속도가 베이스라인에 맞는다(KV 캐시가 한 번 적재됨).
2. FlashAttention이 변경 없이 돌아간다(커스텀 커널 없음).
3. 디코드에서의 산술 강도(arithmetic intensity)가 올라간다(HBM에서 적재한 바이트당 더 많은 연산).

V2는 또한 V1이 뺄셈을 안정화하는 데 쓴 헤드별 RMSNorm을 제거한다. 70B급 사전 학습 규모에서, 그 RMSNorm은 후기 학습을 불안정하게 만들었다. V2는 그것을 추가 모듈 없이 학습을 안정적으로 유지하는 더 단순한 초기화 방식으로 대체한다.

### 언제 손을 뻗을까 (When to reach for it)

| 워크로드 | 이점 |
|----------|---------|
| 긴 컨텍스트 RAG (64k+) | 더 깨끗한 어텐션 맵, 더 적은 환각된 인용 |
| 건초 더미 속 바늘 벤치마크 | 32k를 넘어 상당한 정확도 향상 |
| 다중 문서 QA | 더 적은 문서 간 간섭 |
| 8k에서의 코드 완성 | 미미함, 아키텍처 변경의 가치 없음 |
| 짧은 채팅 (< 4k) | 본질적으로 베이스라인과 구별 불가 |

가치는 컨텍스트 길이에 따라 커진다. 4k 토큰에서는 노이즈 바닥이 충분히 작아 표준 어텐션으로 괜찮다. 128k에서는 그 노이즈가 성능을 해친다.

### 다른 2026년 손잡이와 어떻게 쌓이는가 (How it stacks with other 2026 knobs)

| 기능 | DIFF V2와 호환? |
|---------|------------------------|
| GQA | 예 (V2는 KV 헤드가 아니라 Q 헤드를 늘림) |
| MLA (DeepSeek) | 원리상 예, 둘을 결합한 발표된 논문은 없음 |
| MoE | 예 (어텐션은 MLP 블록과 독립적) |
| RoPE | 예 (변경 없음) |
| YaRN / 긴 컨텍스트 스케일링 | 예 (정확히 DIFF가 가장 도움이 되는 곳) |
| FlashAttention | V2에서 예 (V1에서는 아니오였음) |
| 추측 디코딩(Speculative decoding) | 예 (어텐션 변경은 추측 디코딩 루프에 보이지 않음) |

```figure
differential-attention
```

## 직접 만들기 (Build It)

`code/main.py`는 순수 Python으로 차분 어텐션을 구현한다. 알려진 신호 더하기 노이즈 구조를 가진 장난감 쿼리가 노이즈 상쇄 비율을 직접 측정하게 한다.

### 1단계: 표준 소프트맥스 어텐션

stdlib 행렬 연산: 리스트의 리스트, 수동 행렬 곱셈(matmul), 최댓값을 빼는 수치 안정성을 갖춘 소프트맥스.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 2단계: Q, K를 두 절반으로 나누기

V1 스타일: 헤드 차원을 절반으로. V2 스타일: 헤드 차원을 유지하고 헤드 수를 두 배로. 장난감 구현은 교육적 명료성을 위해 V1을 쓴다. 수학은 동일하고, 기록(bookkeeping)만 다르다.

### 3단계: 두 소프트맥스 분기 + 뺄셈

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

참고: 출력 가중치는 음수일 수 있다. 그것은 괜찮다. 값 캐시는 여전히 부호 있는 기여를 처리한다. 후속 V 투영이 부호를 흡수한다.

### 4단계: 노이즈 상쇄 측정

길이 1024의 합성 시퀀스를 만든다. 신호 토큰을 알려진 위치에 두고, 나머지를 노이즈로 채운다. (a) 신호 위치에 대한 표준 소프트맥스 어텐션 가중치와 (b) 차분 어텐션 가중치를 계산한다. 각각의 신호 대 잡음(signal-to-noise) 비율을 측정한다. DIFF 어텐션은 두 분기가 다르도록 학습된 정도에 따라 3배에서 10배의 인자로 더 높은 신호 대 잡음 비율을 안정적으로 생산한다.

### 5단계: V1 vs V2 파라미터 계산

설정(hidden=4096, heads=32, d_head=128)이 주어지면, 출력한다:

- 베이스라인 Transformer: Q, K, V 각각 크기 `hidden * hidden`, MLP는 4 * hidden에서.
- DIFF V1: Q, K 각각 크기 `hidden * hidden`, V 크기 `hidden * hidden`(변경 없음), 헤드 차원이 내부적으로 절반. 헤드별 `lambda` 파라미터(O(heads * d_head))를 추가.
- DIFF V2: Q 크기 `2 * hidden * hidden`, K 크기 `hidden * hidden`, V 크기 `hidden * hidden`. 여분의 차원은 O_W 전에 다시 아래로 투영. 같은 `lambda` 파라미터를 추가.

장난감은 V2에 대한 추가 파라미터 비용(어텐션 블록당 대략 `hidden * hidden` 추가)을 측정하고 출력한다.

## 라이브러리로 써보기 (Use It)

DIFF V2는 2026년 4월 기준으로 아직 모든 프로덕션 추론 서버에 탑재되어 있지는 않지만, vLLM과 SGLang에서 통합이 진행 중이다. 한편 그 패턴은 다음에서 나타난다:

- Microsoft 내부 긴 컨텍스트 프로덕션 모델.
- 256k 이상 컨텍스트를 겨냥한 여러 오픈 모델 학습 실행에서의 연구 복제.
- 차분 어텐션을 교대 층의 슬라이딩 윈도우 어텐션(sliding-window attention)과 결합하는 하이브리드 아키텍처.

2026년에 이것에 손을 뻗을 때:

- 64k 이상의 유효 컨텍스트를 겨냥한 새 모델을 밑바닥부터 학습. 처음부터 차분 어텐션을 추가하라. 나중에 재학습하는 것은 비싸다.
- 중간 분실 실패가 평가를 지배하는 긴 컨텍스트 모델을 파인튜닝. Q 투영에 대한 LoRA가 DIFF 구조를 근사할 수 있다.

손을 뻗지 말아야 할 때:

- 안정적인 긴 컨텍스트 성능을 가진 사전 학습된 밀집(dense) 모델을 서빙하고 있다. 재학습 비용이 기존 가중치에서 거의 보상되지 않는다.
- 컨텍스트가 항상 16k 미만이다. 노이즈 바닥이 무시할 만하다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-diff-attention-integrator.md`를 만든다. 모델 아키텍처, 목표 컨텍스트 길이, 환각 프로필, 학습 예산이 주어지면, 새 사전 학습 실행이나 LoRA 파인튜닝에 차분 어텐션을 추가하기 위한 통합 계획을 생산한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 차분 어텐션에 대해 보고된 신호 대 잡음 비율이 합성 쿼리에서 표준 소프트맥스 어텐션보다 높은지 검증하라. 노이즈 진폭을 변화시키고 표준 어텐션이 쓸 수 없게 되는 교차점(crossover point)을 보여라.

2. 7B급 모델(hidden=4096, heads=32, d_head=128, 32개 층)에 대해 베이스라인에서 DIFF V1로, 그리고 베이스라인에서 DIFF V2로의 파라미터 수 차이를 계산하라. 어떤 구성요소가 파라미터를 얻었고 어떤 것이 그대로인지 보여라.

3. DIFF V1 논문(arXiv:2410.05258)의 3절과 DIFF V2 Hugging Face 블로그의 2절을 읽어라. 두 문장으로, 왜 V1 헤드별 RMSNorm이 필요했는지, 그리고 왜 V2가 학습 발산을 일으키지 않고 그것을 제거할 수 있었는지 설명하라.

4. 절제(ablation)를 구현하라: `lambda = 0`(순수 첫 소프트맥스)과 `lambda = 1`(완전 뺄셈)로 차분 어텐션을 계산하라. 합성 쿼리에서, 스윕(sweep)에 걸쳐 신호 대 잡음이 어떻게 바뀌는지 측정하라. 신호 대 잡음을 최대화하는 `lambda`를 식별하라.

5. 장난감을 GQA + DIFF V2로 확장하라. 8개 KV 헤드와 32개 Q 헤드를 골라라. KV 캐시 크기가 같은 (8, 32) 구성의 베이스라인 GQA 모델과 일치함을 보여라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 차분 어텐션(Differential attention) | "서로 빼는 두 소프트맥스" | Q, K를 두 절반으로 나누고, 두 소프트맥스 맵을 계산하고, 두 번째(lambda로 스케일됨)를 첫 번째에서 뺀 뒤, V를 곱함 |
| 노이즈 바닥(Noise floor) | "소프트맥스의 0이 아닌 꼬리" | 소프트맥스가 모든 무관한 토큰에 두는 O(1/N) 가중치, 긴 컨텍스트에 걸쳐 O(1)로 합산됨 |
| lambda | "뺄셈 스케일" | `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`로 매개변수화된 헤드별 학습 가능 스칼라; 음수일 수 있음 |
| DIFF V1 | "ICLR 2025 버전" | 원조 Differential Transformer; 파라미터 수를 보존하려고 헤드 차원을 절반으로, 커스텀 커널이 필요, 디코드가 더 느림 |
| DIFF V2 | "2026년 1월 수정" | KV 헤드는 유지하며 Q 헤드를 두 배로; 베이스라인 디코드 속도에 맞고 FlashAttention과 작동 |
| 헤드별 RMSNorm(Per-head RMSNorm) | "V1 안정화기" | V1이 차분 후에 적용한 추가 정규화; V2는 후기 학습 불안정을 막기 위해 그것을 제거 |
| 신호 대 잡음 비율(Signal-to-noise ratio) | "얼마나 많은 어텐션이 낭비되는지" | 참 신호 위치에 대한 가중치 대 무관한 위치에 대한 평균 가중치의 비율 |
| 중간 분실(Lost in the middle) | "긴 컨텍스트 실패 모드" | 긴 컨텍스트 중간에 있는 문서에 대해 검색 정확도가 떨어지는 경험적 현상: DIFF 어텐션이 이를 줄임 |
| 산술 강도(Arithmetic intensity) | "적재한 바이트당 FLOPs" | KV 적재당 쿼리를 두 배로 하여 V2가 디코드에서 늘린 비율; 메모리 제약(memory-bound) 디코드에 중요 |

## 더 읽을거리 (Further Reading)

- [Ye et al.: Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) 노이즈 상쇄 이론과 긴 컨텍스트 절제를 담은 원조 논문
- [Microsoft unilm: Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) 프로덕션 스택 재작성, 베이스라인 디코드에 맞고 FlashAttention 호환
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333): 왜 뺄셈이 사전 학습된 어텐션 구조를 복구하는지에 대한 이론적 분석
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900): 파라미터 공유 변형
- [Vaswani et al.: Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) DIFF가 빼는 대상인 베이스라인 Transformer
- [Liu et al.: Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) DIFF 어텐션이 겨냥하는 긴 컨텍스트 벤치마크
