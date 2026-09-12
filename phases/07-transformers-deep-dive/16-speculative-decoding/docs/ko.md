# 추측 디코딩(Speculative Decoding): 드래프트, 검증, 반복

> 자기회귀(autoregressive) 디코딩(decoding)은 직렬이다. 각 토큰(token)은 이전 것을 기다린다. 추측 디코딩(speculative decoding)은 사슬을 끊는다: 저렴한 모델(model)이 N개의 토큰을 드래프트(draft)하고, 비싼 모델이 N개 모두를 한 번의 순방향 패스(forward pass)로 검증한다. 드래프트가 맞으면 N개의 생성에 대해 큰 모델의 순방향 한 번을 치른 셈이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60분

## 문제 (The Problem)

70B LLM이 토큰 하나를 샘플링(sampling)하는 데 H100에서 ~30 ms가 걸린다. 3B 드래프트 모델은 ~3 ms가 걸린다. 3B가 5개 토큰을 앞서 드래프트하게 한 뒤, 70B를 *한 번* 돌려 5개 모두를 검증하면, 최대 5개의 받아들여진 토큰에 대해 총 `5×3 + 30 = 45 ms`다. 직선 생성의 `5×30 = 150 ms` 대비. 이것이 추측 디코딩의 완전한 홍보 문구다: 약간의 추가 GPU 메모리(드래프트 모델)를 2~4배 낮은 디코딩 지연 시간(latency)과 맞바꾼다.

이 트릭은 분포(distribution)를 보존해야 한다. Leviathan et al. (2023)이 처음 제시하고 Chen et al.이 동시에 내놓은 추측 샘플링(speculative sampling)은 출력 시퀀스(sequence)가 큰 모델이 혼자 생성했을 것과 **동일하게 분포(identically distributed)**됨을 보장한다. 품질 트레이드오프(trade-off)가 없다. 그저 더 빠를 뿐이다.

네 가지 부류의 드래프트-검증기(verifier) 쌍이 2026년 추론(inference)을 지배한다:

1. **바닐라 추측(vanilla speculative, Leviathan 2023).** 별도의 드래프트 모델(예: Llama 3 1B) + 검증기(예: Llama 3 70B).
2. **Medusa(Cai 2024).** 검증기 위의 여러 디코딩 헤드(decoding head)가 위치 `t+1..t+k`를 병렬로 예측한다. 별도의 드래프트 모델이 없다.
3. **EAGLE 계열(Li 2024, 2025).** 검증기의 은닉 상태(hidden state)를 재사용하는 경량 드래프트; 바닐라보다 높은 수용률(acceptance rate); 일반적으로 3~4배.
4. **룩어헤드 디코딩(lookahead decoding, Fu 2024).** 야코비(Jacobi) 반복; 드래프트 모델이 전혀 필요 없다. 자기 추측(self-speculation). 틈새지만 의존성이 없다.

2026년 모든 프로덕션(production) 추론 스택(stack)이 기본적으로 추측 디코딩을 탑재한다. vLLM, TensorRT-LLM, SGLang, llama.cpp가 모두 최소한 바닐라 + EAGLE-2를 지원한다.

## 개념 (The Concept)

### 핵심 알고리즘

검증기 `M_q`와 더 저렴한 드래프트 `M_p`가 주어졌을 때:

1. `x_1..x_k`를 이미 디코딩된 접두사(prefix)라 하자.
2. **드래프트**: `M_p`를 사용해 드래프트 확률(probability) `p_1..p_N`과 함께 `d_{k+1}, d_{k+2}, ..., d_{k+N}`을 자기회귀적으로 제안한다.
3. **병렬 검증**: `x_1..x_k, d_{k+1}, ..., d_{k+N}`에 대해 `M_q`를 한 번 돌려, 위치 `k+1..k+N+1`에 대한 검증기 확률 `q_1..q_{N+1}`을 얻는다.
4. **각 드래프트 토큰을 왼쪽에서 오른쪽으로 수용/거부**: 각 `i`에 대해 확률 `min(1, q_i(d_i) / p_i(d_i))`로 수용한다.
5. **위치 `j`에서의 첫 거부 시**: 정규화된(normalized) "잔차(residual)" 분포 `(q_j - p_j)_+`에서 `t_j`를 샘플링한다. `j` 이후의 모든 드래프트는 폐기된다.
6. **`N`개 모두 수용 시**: `q_{N+1}`에서 추가 토큰 `t_{N+1}`을 하나 샘플링한다(공짜 보너스 토큰).

잔차 분포 트릭은 출력이 마치 `M_q`가 처음부터 샘플링한 것처럼 정확히 분포되도록 유지하는 수학적 통찰이다.

### 무엇이 가속을 결정하는가

`α` = 드래프트 토큰당 기대 수용률이라 하자. `c` = 드래프트-대-검증기 비용 비율이라 하자. 스텝당:

- 순진한 생성은 토큰당 큰 모델 호출을 1번 한다.
- 추측은 `α`가 높을 때 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 토큰당 큰 모델 호출을 1번 한다.

`α = 0.75`, `N = 5`에서의 일반적인 경험칙: 큰 모델 호출이 3배 적다. 드래프트 비용은 5배 저렴하다. 총 실측 시간(wall-clock)이 ~2.5배 떨어진다.

**α는 다음에 의존한다:**

- 드래프트가 검증기를 얼마나 잘 근사하는가. 같은 계열 / 같은 학습 데이터가 α를 크게 높인다.
- 디코딩 전략. 그리디(greedy) 드래프트 대 그리디 검증기: 높은 α. 온도 샘플링(temperature sampling): 맞추기 어렵고, 수용이 떨어진다.
- 작업 유형. 코드와 구조화된 출력은 더 많이 수용한다(예측 가능). 자유 형식의 창작 글쓰기는 덜 수용한다.

### Medusa: 드래프트 모델 없는 드래프트

Medusa는 드래프트 모델을 검증기 위의 추가 출력 헤드로 교체한다. 위치 `t`에서:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

각 헤드는 자신의 로짓(logit)을 출력한다. 추론 시 각 헤드에서 샘플링해 후보 시퀀스를 얻은 뒤, 모든 후보 연속(continuation)을 한 번에 고려하는 트리 어텐션(tree-attention) 방식으로 한 번의 순방향 패스로 검증한다.

장점: 두 번째 모델이 없다. 단점: 학습 가능한 파라미터(parameter)를 추가한다; 지도 파인튜닝(supervised fine-tuning) 단계(~1B 토큰)가 필요하다; 수용률이 좋은 드래프트를 쓴 바닐라 추측보다 약간 낮다.

### EAGLE: 은닉 상태 재사용으로 더 나은 드래프트

EAGLE-1/2/3(Li et al., 2024~2025)은 드래프트 모델을, 검증기의 마지막 층(layer) 은닉 상태를 입력으로 받는 작은 트랜스포머(transformer, 보통 1층)로 만든다. 드래프트가 검증기의 특성(feature) 표현을 보기 때문에, 그 예측이 검증기의 출력 분포와 강하게 상관된다. 수용률이 ~0.6(바닐라)에서 0.85+로 올라간다.

EAGLE-3(2025)은 후보 연속에 대한 트리 탐색(tree search)을 추가했다. vLLM과 SGLang은 Llama 3/4와 Qwen 3에 대해 EAGLE-2/3을 기본 추측 경로로 탑재한다.

### KV 캐시 춤사위

검증은 `N`개의 드래프트 토큰을 검증기에 한 번의 순방향 패스로 넣는다. 이는 검증기의 KV 캐시를 `N`개 항목만큼 늘린다. 일부 드래프트가 거부되면, 캐시를 수용된 접두사 길이로 롤백(roll back)해야 한다.

프로덕션 구현(vLLM의 `--speculative-model`, TensorRT-LLM의 LookaheadDecoder)은 스크래치(scratch) KV 버퍼로 이를 처리한다. 먼저 쓰고, 수용 시 커밋(commit)한다. 개념적으로 어렵지는 않지만 까다롭다.

```figure
draft-verify-tokens
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. 다음을 갖춘 핵심 추측 샘플링 알고리즘(거부 단계 + 잔차 분포)을 구현한다:

- 손으로 코딩한 분포에 대한 결정론적-softmax인 "큰 모델"(그래서 수용 수학을 해석적으로 검증할 수 있다).
- 큰 모델의 섭동(perturbation)인 "드래프트 모델".
- 직접 샘플링과 같은 주변 분포(marginal distribution)를 생성하는 수용 / 거부 루프.

### 1단계: 거부 단계

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`는 균등 난수(uniform random number)다. `q_prob`은 드래프트된 토큰에 대한 검증기의 확률이다. `p_prob`은 드래프트 모델의 확률이다. Leviathan 정리는, 거부 시 잔차에서의 샘플링이 뒤따르는 이 베르누이(Bernoulli) 결정이 검증기의 분포를 정확히 보존한다는 것이다.

### 2단계: 잔차 분포

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

`q`에서 `p`를 원소별로 빼고, 음수 값을 0으로 클램프(clamp)하고, 재정규화한다. 어떤 거부에서든 이로부터 샘플링한다.

### 3단계: 하나의 추측 스텝

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5개 수용 → 보너스 1개 → 한 번의 검증기 패스로 6개 토큰 생성.

### 4단계: 수용률 측정

다양한 드래프트 품질 수준에서 10,000번의 추측 스텝을 실행한다. 드래프트와 검증기 분포 간의 KL 발산(KL divergence) 대비 수용률을 그린다. 깨끗한 단조(monotone) 관계가 보여야 한다.

### 5단계: 분포 동등성 검증

경험적으로: 추측 루프가 생성한 토큰의 히스토그램(histogram)이 검증기에서 직접 샘플링해 생성한 히스토그램과 일치해야 한다. 이것이 실제로 작동하는 Leviathan 정리다. 카이제곱(chi-square) 검정이 샘플링 오차 이내에서 이를 확인한다.

## 라이브러리로 써보기 (Use It)

프로덕션:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

2026년 중반 기준 TensorRT-LLM의 Medusa 경로가 가장 빠르다. `faster-whisper`는 작은 드래프트로 Whisper-large에 추측 디코딩을 감싼다.

**드래프트 고르기:**

| 전략 | 언제 고르나 | 가속 |
|----------|--------------|---------|
| 바닐라 드래프트 (1B/3B Llama 계열) | 빠른 프로토타입, 학습 없음 | 1.8~2.3× |
| Medusa 헤드 | 검증기를 파인튜닝할 수 있음 | 2~3× |
| EAGLE-2 / 3 | 프로덕션, 최대 속도 | 3~4× |
| 룩어헤드(Lookahead) | 드래프트 없음, 학습 없음, 추가 파라미터 없음 | 1.3~1.6× |

**추측 디코딩을 하지 말아야 할 때:**

- 1~5개 토큰의 단일 시퀀스 생성. 오버헤드가 지배한다.
- 극도로 창의적 / 고온(high-temperature) 샘플링(α가 떨어진다).
- 메모리 제약 배포(deployment)(드래프트 모델이 VRAM을 더한다).

## 산출물 (Ship It)

`outputs/skill-spec-decode-picker.md`를 참고하라. 이 스킬은 새 추론 워크로드에 대해 추측 디코딩 전략(바닐라 / Medusa / EAGLE / 룩어헤드)과 튜닝 파라미터(N, 드래프트 온도)를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 추측 토큰 분포가 50,000개 토큰에서 카이제곱 p > 0.05 이내로 검증기의 직접-샘플 분포와 일치하는지 확인하라.
2. **중간.** `α = 0.5, 0.7, 0.85`에 대해 `N`의 함수로 가속(큰 모델 순방향당 토큰)을 그려라. 각 α에 대한 최적 `N`을 식별하라. (힌트: 검증 호출당 기대 토큰 = `(1 - α^{N+1}) / (1 - α)`.)
3. **어려움.** 작은 Medusa를 구현하라: Lesson 14의 캡스톤(capstone) GPT를 가져와, 위치 t+2, t+3, t+4를 예측하는 3개의 추가 LM 헤드를 더하라. 공동 다중 헤드 손실(loss)로 tinyshakespeare에 학습하라. 같은 모델을 잘라 만든 바닐라 드래프트 대비 수용률을 비교하라.
4. **어려움.** 롤백을 구현하라: 10토큰 접두사 KV 캐시로 시작해, 5개의 드래프트 토큰을 넣고, 위치 3에서의 거부를 시뮬레이션하라. 다음 반복에서 캐시 읽기가 "접두사 + 처음 수용된 드래프트 2개"와 정확히 일치하는지 확인하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 드래프트 모델(Draft model) | "저렴한 것" | 후보 토큰을 제안하는 더 작은 모델; 보통 검증기보다 10~50배 저렴하다. |
| 검증기(Verifier) | "큰 것" | 우리가 분포를 보존하는 대상 모델; 추측 스텝당 한 번 실행된다. |
| 수용률(Acceptance rate, α) | "드래프트가 얼마나 자주 맞나" | 검증기가 드래프트를 수용하는 토큰당 확률. 0.7~0.9가 일반적. |
| 잔차 분포(Residual distribution) | "거부 대비책" | 정규화된 `(q - p)_+`; 거부 시 이로부터 샘플링하면 검증기의 분포를 보존한다. |
| 보너스 토큰(Bonus token) | "공짜 토큰" | N개 드래프트 모두 수용 시, 검증기의 다음 스텝 분포에서 하나 더 샘플링한다. |
| Medusa | "드래프트 없는 추측" | 검증기 위의 여러 LM 헤드가 위치 t+1..t+k를 병렬로 예측한다. |
| EAGLE | "은닉 상태 드래프트" | 검증기의 마지막 층 은닉 상태에 조건화된 작은 트랜스포머 드래프트. |
| 룩어헤드 디코딩(Lookahead decoding) | "야코비 반복" | 고정점(fixed-point) 반복을 사용하는 자기 추측; 드래프트 모델 없음. |
| 트리 어텐션(Tree attention) | "여러 후보를 한 번에 검증" | 여러 드래프트 연속을 동시에 고려하는 분기(branching) 검증. |
| KV 롤백(KV rollback) | "거부된 드래프트 되돌리기" | 스크래치 KV 버퍼; 수용 시 커밋, 거부 시 폐기. |

## 더 읽을거리 (Further Reading)

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192): 핵심 알고리즘과 동등성 정리.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318): 동시 도입; 깔끔한 베르누이-거부 증명.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774): Medusa 논문; 트리 어텐션 검증.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077): EAGLE-1; 은닉 상태 조건화 드래프트.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858): EAGLE-2; 동적 트리 깊이.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057): 룩어헤드, 드래프트 없는 접근.
- [vLLM docs: Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) 네 가지 전략을 모두 연결한 표준 프로덕션 레퍼런스.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE): EAGLE-1/2/3의 레퍼런스 코드.
