# 추측 디코딩과 EAGLE-3 (Speculative Decoding and EAGLE-3)

> Phase 7 · Lesson 16은 수학을 증명했다. Leviathan 거부 규칙(rejection rule)은 검증자(verifier)의 분포(distribution)를 정확히 보존한다. 이 레슨은 2026년 프로덕션 추측 디코딩(speculative decoding)의 학습 스택(training-stack) 관점이다. EAGLE-3은 드래프트 모델(draft model)을 값싼 근사에서, 검증자 자신의 은닉 상태(hidden state)로 학습된 목적 지향 초소형 신경망으로 바꿨고, 그런 다음 학습과 추론 분포를 정렬하는 학습 시점 테스트 루프를 추가했다. 결과: 3배에서 6.5배의 엔드투엔드 속도 향상, 채팅에서 토큰당 수락률(accepted per-token rate) 0.9 초과, 분포적 트레이드오프(trade-off) 없음. 2026년의 모든 프로덕션 추론 스택이 이를 기본으로 탑재한다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- Leviathan 정리를 한 문장으로 진술하고, 추측 루프가 검증자와 동일하게 분포된 샘플을 생성함을 증명하기.
- 바닐라 추측 디코딩(Leviathan 2023)에서 EAGLE, EAGLE-2, EAGLE-3에 이르는 2년간의 진전을 따라가고, 각 단계가 제거한 정확한 한계를 짚기.
- 수락률 `α`와 드래프트 대 검증자 비용 비율 `c`로부터 기대 속도 향상을 계산하고, 각 영역(regime)에 대한 최적 드래프트 길이 `N`을 고르기.
- 전체 추측 루프를 밑바닥부터 구현하기: 드래프트, 검증, 잔차(residual)로부터 거부 샘플링, 거부 시 KV 캐시(KV cache) 롤백(rollback), 완전 수락 시 보너스 토큰(bonus token) 방출.

## 문제 (The Problem)

70B 모델의 자기회귀(autoregressive) 디코딩은 H100에서 초당 35 토큰쯤으로 돌아간다. GPU는 포화(saturated) 근처에도 못 미친다. 메모리 대역폭(memory bandwidth)이 천장이다. 모든 토큰은 HBM에서 70B의 가중치(weight)를 적재하고, 산술 한 스텝을 하고, 부동소수점 하나를 생산한다. 연산 유닛은 대부분 놀고 있다.

추측 디코딩은 그것을 실제로 풀 수 있는 처리량(throughput) 문제로 바꾼다. 값싼 드래프트가 `N`개의 작은 순방향 패스(forward pass)에서 `N`개의 토큰을 제안한다. 검증자는 접두사(prefix)와 모든 `N`개 드래프트를 묶어 한 번 실행한다. 위치 `i`에서 검증자의 분포가 드래프트와 일치하면(곧 정밀하게 다듬을 통계적 의미에서) 받아들이고, 아니면 거부한 뒤 잔차 분포에서 보정(correction)을 샘플링한다. 대형 모델 순방향 한 번이 토큰 하나가 아니라 최대 `N+1`개의 수락된 토큰을 생산한다.

중요한 정리는 Leviathan, Kalman, Matias (ICML 2023)다. 출력 분포가 검증자에서 직접 샘플링했을 때 생산했을 것과 동일하다. 근사적으로가 아니다. 동일하게. 추측 디코딩이 프로덕션에서 받아들여지는 이유가 바로 이것이다. 품질 트레이드오프 없는 순수한 지연 시간(latency) 최적화다.

Phase 7 · Lesson 16이 준 것은 수학이었고, 이 레슨이 주는 것은 학습 스택이다. 좋은 드래프트는 값싼 드래프트보다 속도 향상에서 2배의 가치가 있다. EAGLE, EAGLE-2, EAGLE-3 (Li et al., 2024–2025)은 "드래프트 = 같은 모델의 더 작은 버전"을 정밀한 엔지니어링 규율로 바꿨다. 2026년 프로덕션 추론 서버는 EAGLE-3을 기본으로 한다.

## 개념 (The Concept)

### 불변항: Leviathan 거부 샘플링 (The invariant: Leviathan rejection sampling)

`p(t)`를 어떤 접두사가 주어졌을 때 다음 토큰에 대한 드래프트의 분포로, `q(t)`를 검증자의 분포로 두자. 드래프트 토큰 `d ~ p`를 샘플링한다. 확률 `min(1, q(d) / p(d))`로 받아들인다. 거부 시, 잔차 분포 `(q - p)_+ / ||(q - p)_+||_1`에서 샘플링한다. 결과 샘플은 `q`에 따라 분포된다. 이는 `p`가 얼마나 나쁘든 참이다. 나쁠수록 더 자주 거부하지만, 출력은 정확하게 남는다.

이런 호출 `N`개를 `prefix + d_1 + ... + d_N`에 대한 하나의 검증자 순방향 패스로 연속해서 쌓는다. 검증자는 `q_1, q_2, ..., q_{N+1}`을 동시에 반환한다. 왼쪽에서 오른쪽으로 훑는다. 위치 `j`에서의 첫 거부에서, `residual(q_j, p_j)`에서 샘플링하고 멈춘다. 완전 수락 시, `q_{N+1}`에서 보너스 토큰 하나를 샘플링한다.

### 무엇이 속도 향상을 결정하는가 (What determines speedup)

`α`를 드래프트된 토큰당 기대 수락률로 두자. `c = cost(draft) / cost(verifier)`를 비용 비율로 두자. 검증자 순방향당 기대 수락 토큰 수는:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

수락 토큰당 기대 총 벽시계 시간은 `(N * c + 1) / E[accepted]`다. 이를 `N`에 대해 최소화하면 최적점(sweet spot)을 얻는다. `α = 0.8, c = 0.05`의 경우: 최적 `N`은 약 5–7, 속도 향상은 3.2배다. `α = 0.95, c = 0.02`의 경우: 최적 `N`은 약 8–10, 속도 향상은 5배에 다다른다.

가장 큰 단일 지렛대는 `α`다. 고정된 `N = 5`에서 `α = 0.6`(바닐라 드래프트)에서 `α = 0.9`(EAGLE-3)로 가면 검증자 순방향당 기대 수락 토큰이 2.2에서 4.1로 늘어난다. 같은 검증자에서 거의 2배 더 많은 처리량이다.

### 2년간의 진전 (The two-year progression)

**바닐라 추측(Vanilla speculative, Leviathan, 2023).** 드래프트 모델은 같은 가족의 독립적으로 학습된 더 작은 LLM이다. 연결하기 쉽고, `α ≈ 0.6`, 속도 향상은 잘해야 약 2배.

**EAGLE-1 (Li et al., 2024).** 드래프트는 작은 트랜스포머다. 보통 한 개나 두 개 층: 검증자의 마지막 층 은닉 상태를 입력으로 받아 다음 토큰을 직접 예측한다. 드래프트가 검증자의 특성 표현(feature representation)을 보기 때문에, 그 분포는 검증자의 분포에 훨씬 가깝다. `α`가 0.7–0.8로 오른다.

**EAGLE-2 (Li et al., 2024).** 동적 드래프트 트리(dynamic draft tree)를 추가한다. `N`개 토큰의 단일 시퀀스를 제안하는 대신, 후보의 작은 트리를 제안하고, 하나의 순방향 패스로 검증자가 각각을 채점하고(트리 어텐션, tree attention), 가장 높은 확률의 경로를 훑는다. 드래프트 길이가 스텝마다 적응적(adaptive)이 된다. 수락된 경로 토큰당 `α`가 0.85 위로 오른다.

**EAGLE-3 (Li et al., 2025, NeurIPS).** 두 가지가 더 바뀐다. 첫째, 특성 예측 손실(feature-prediction loss)을 완전히 버린다. EAGLE-1/2는 드래프트가 검증자의 은닉 상태와 일치하도록 학습시켰는데, 이는 데이터가 얼마나 도움이 되는지에 상한을 둔다. EAGLE-3은 토큰 예측에 직접 학습한다. 둘째, 학습 시점 테스트(training-time test, TTT): 드래프트 학습 중, 추론에서 작동하는 것과 같은 방식으로 드래프트 자신의 이전 예측을 여러 스텝에 걸쳐 입력으로 되먹인다. 이는 학습과 테스트 분포를 정렬하고 오차 누적을 멈춘다. 측정된 속도 향상: 채팅에서 최대 6.5배, H100의 SGLang에서 배치 64에서 38% 처리량 개선.

### KV 캐시 롤백 (KV cache rollback)

검증은 한 패스에서 검증자의 KV 캐시를 `N`개 항목만큼 확장한다. 거부가 위치 `j`에서 일어나면, 위치 `j-1` 이후의 캐시 내용은 이제 틀린다. 두 가지 흔한 구현: 스크래치 버퍼(scratch buffer)에 쓰고 수락 시 커밋(commit)하기(vLLM, TensorRT-LLM), 또는 물리적 KV 캐시 더하기 논리적 길이를 유지하고 거부 시 잘라내기. 어느 쪽이든, 롤백 비용은 층당 헤드당 바이트이며, 순방향 패스 비용 옆에서는 무시할 만하다.

EAGLE-2 트리 탐색에서는 검증자가 트리 토폴로지(topology)를 존중하는 비인과 마스크(non-causal mask)로 어텐션을 실행한다. 엔지니어링은 까다롭지만 계산은 커스텀 마스크를 얹은 표준 flash-attention 호출이다.

### 2026년의 드래프트 아키텍처 (Draft architectures in 2026)

| 전략 | 드래프트 유형 | `α` | 속도 향상 | 학습 비용 |
|----------|-----------|-----|---------|---------------|
| 바닐라(Vanilla) | 별도의 작은 LLM | 0.55-0.70 | 1.8-2.3배 | 없음(기존 작은 모델 재사용) |
| Medusa | 검증자 위의 추가 LM 헤드 | 0.65-0.75 | 2-3배 | ~1B SFT 토큰 |
| EAGLE-1 | 은닉 상태 위의 1층 트랜스포머 | 0.70-0.80 | 2.5-3배 | ~60B 토큰 |
| EAGLE-2 | EAGLE-1 + 동적 드래프트 트리 | 0.80-0.88 | 3-4배 | ~60B 토큰 |
| EAGLE-3 | 다층 특성 융합 + TTT | 0.88-0.92 | 3.5-6.5배 | ~60-200B 토큰 |
| Lookahead | 드래프트 없음(Jacobi 반복) | 해당 없음 | 1.3-1.6배 | 없음 |

2026년 프로덕션에서: vLLM과 SGLang은 가능하면 EAGLE-3을, 아니면 EAGLE-2를 기본으로 한다. TensorRT-LLM은 Meta와 NVIDIA 공개 모델에 대해 가장 빠른 Medusa 경로를 가진다. llama.cpp는 CPU 배포를 위해 바닐라 드래프트를 탑재한다.

```figure
l5-spec-decode-eagle
```

## 직접 만들기 (Build It)

`code/main.py`를 보라. 모든 조각을 갖춘 전체 Leviathan 추측 루프다. N개 드래프트, 검증자 병렬 패스, 위치별 거부, 잔차 샘플링, 보너스 토큰, KV 롤백, 그리고 출력 분포가 `q`로부터의 직접 샘플링과 일치한다는 경험적 검증을 모두 담았다.

### 1단계: 거부 규칙

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 2단계: 잔차 분포

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 3단계: 완전한 추측 스텝

`spec_step` 함수는 `p`에서 `N`개 토큰을 드래프트한 다음, 하나의 병렬 `q` 평가로 그들 모두를 검증한다. 각 드래프트된 토큰에 대해 거부 규칙을 적용하고, 첫 거부에서 잔차로부터 보정을 샘플링한다. 모든 것이 수락되면, `q_{N+1}`에서 보너스 토큰을 방출한다.

### 4단계: KV 롤백 기록

시뮬레이터는 워커(worker)당 논리적 `kv_length`를 추적한다. `k`개 드래프트의 수락 시, `kv_length += k`. 위치 `j`에서의 거부 시, 캐시는 이미 `j`를 지나 쓰여 있지만, 논리적 길이는 `prefix_length + j + 1`로 설정된다. 보정 토큰 하나 너머. 후속 읽기는 논리적 길이로 잘린다.

### 5단계: Leviathan 검사

50,000개의 추측 스텝을 실행한다. 수락된 토큰의 경험적 분포를 센다. `q`로부터의 50,000개 직접 샘플과 비교한다. 카이제곱(chi-square) 통계량은 임계값보다 충분히 낮아야 한다. 정리가 실제로 통과한다.

### 6단계: 속도 향상 vs α

`p`를 다른 진폭으로 `q`에서 멀어지게 섭동(perturbing)하여 드래프트 품질을 스윕(sweep)한다. `α`를 측정한 다음, `α`와 `N`의 함수로 검증자 호출당 기대 토큰을 그린다. 코드는 EAGLE-3급 드래프트 품질(`α ≈ 0.9`)이 어떻게 검증자 호출당 4–5 토큰을 푸는지 보여 주는 표를 출력한다.

## 라이브러리로 써보기 (Use It)

EAGLE-3을 사용한 프로덕션 수준 `vllm serve`:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

H100의 배치 64에서 EAGLE-3을 사용한 SGLang: EAGLE-3 논문에 따르면, 배치 64 바닐라 디코딩보다 대략 1.38배 더 많은 처리량.

추측 디코딩에 손을 뻗을 때:

- p50 지연 시간이 최대 처리량보다 중요한 모든 대화형 채팅 워크로드.
- 코드 생성과 구조화된 출력(JSON, SQL). 타깃 분포가 매우 예측 가능하기 때문에 `α`가 0.9 위다.
- 장문 생성(수천 토큰). 분할 상환된(amortized) 속도 향상이 계속 보상을 준다.

쓰지 말아야 할 때:

- 매우 작은 모델(< 3B). 드래프트가 검증자보다 그리 저렴하지 않다.
- 초소형 배치-1 CPU 배포. 드래프트 모델의 메모리 오버헤드가 가치 없을 수 있다.
- `α`가 붕괴하는 매우 높은 temperature의 창의적 샘플링.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-eagle3-tuner.md`를 만든다. 추론 워크로드(모델, 배치 크기, 목표 지연 시간, 작업 프로필)가 주어지면, 추측 디코딩 전략과 튜닝 파라미터(드래프트 가족, `N`, 트리 깊이, temperature 인식 전환)를 추천한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. Leviathan 분포 검사의 카이제곱 통계량이 50,000개 샘플에서 95% 임계값 아래로 유지되는지 확인하라.

2. `α`를 0.9로, `c`를 0.04로 유지하면서 `N`을 1에서 10까지 스윕하라. 검증자 호출당 기대 토큰과 토큰당 실제 벽시계 시간을 그려라. 벽시계 시간을 최소화하는 `N`을 찾아라. 곡선의 모양을 설명하라.

3. EAGLE-2 트리 탐색을 시뮬레이션하도록 코드를 수정하라. 각 스텝에서, 드래프트는 `[2, 2, 2]` 모양의 트리(여덟 개 후보 경로)를 제안한다. 검증자는 한 번 실행하고, 가장 높은 확률의 수락된 경로가 이긴다. 리프(leaf)당 `α`와 검증자 호출당 총 토큰을 계산하라. 동등한 연산에서의 선형 사슬 추측 디코딩과 비교하라.

4. 두 개의 동시 시퀀스에 대한 배치 KV 롤백 시뮬레이터를 구현하라. 시퀀스 A는 모든 드래프트가 수락되고, 시퀀스 B는 위치 2에서 거부한다. 올바른 `kv_length`가 시퀀스별로 갱신되고 어떤 작업도 낭비되지 않음을 보여라.

5. EAGLE-3 논문의 4절(Training-Time Test)을 읽어라. TTT 없는 순진한 드래프트 학습이 왜 노출 편향(exposure bias)을 겪는지, 그리고 학습 중 드래프트에 자신의 예측을 먹이는 것이 왜 그것을 고치는지 두 문장으로 설명하라. 이를 seq2seq의 예약 샘플링(scheduled-sampling) 문헌과 연결하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Leviathan 규칙 | "min(1, p에 대한 q)" | 확률 `min(1, q(d)/p(d))`로 베르누이(Bernoulli) 수락/거부, 거부 시 잔차에서 샘플링하면 검증자 분포를 정확히 보존 |
| 잔차 분포(Residual distribution) | "(q 빼기 p) 양수, 정규화" | 0에서 클램프(clamp)되고 재정규화된 `(q - p)_+`: 거부 시 샘플링할 올바른 분포 |
| 수락률 α(Acceptance rate α) | "드래프트가 얼마나 자주 맞는지" | 거부 규칙 하에서의 토큰당 기대 베르누이 성공 확률; 모든 속도 향상 수학을 좌우함 |
| EAGLE-1 | "은닉 상태 드래프트" | 검증자의 마지막 층 은닉 상태에 조건화된 작은 트랜스포머 드래프트 (Li et al., 2024) |
| EAGLE-2 | "동적 드래프트 트리" | EAGLE-1 더하기 하나의 검증자 패스에서 트리 어텐션으로 채점되는 후보 연속의 트리 |
| EAGLE-3 | "학습 시점 테스트" | 특성 예측 손실을 버리고, 학습 중 드래프트에 자신의 출력을 먹이며 직접 토큰 예측으로 학습 |
| 학습 시점 테스트(Training-time test, TTT) | "노출 편향 수정" | 학습 중 드래프트를 자기회귀적으로 실행해 학습과 테스트 입력 분포를 일치: 예약 샘플링의 직접 유사물 |
| KV 롤백(KV rollback) | "거부된 드래프트 되돌리기" | 거부 후 검증자의 KV 캐시를 수락된 접두사 길이로 재설정하는 기록 |
| 보너스 토큰(Bonus token) | "공짜로 받는 것" | 모든 `N`개 드래프트가 수락되면, 추가 검증자 비용 없이 `q_{N+1}`에서 하나를 더 샘플링 |
| 트리 어텐션(Tree attention) | "많은 후보를 한 번에 검증" | 드래프트 트리의 토폴로지를 존중하는 비인과 마스크가 있는 어텐션; 하나의 순방향 패스에서 트리의 모든 노드에 대해 `q_i`를 계산 |

## 더 읽을거리 (Further Reading)

- [Leviathan, Kalman, Matias: Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) 기초 논문과 동등성 정리
- [Chen et al.: Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) 깔끔한 증명을 곁들인 동시의 독립적 도입
- [Li et al.: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) EAGLE-1, 은닉 상태 조건화 드래프트
- [Li et al.: EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) 동적 트리 탐색
- [Li et al.: EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) 2026년 프로덕션 기본값
- [Cai et al.: Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) 대안적 드래프트 없는 접근법
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html): 모든 전략이 연결된 표준적인 프로덕션 참조
