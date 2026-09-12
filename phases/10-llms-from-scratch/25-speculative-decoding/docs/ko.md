# 추측 디코딩(Speculative Decoding)과 EAGLE

> 최첨단 LLM이 토큰(token) 하나를 생성하려면 수십억 개의 파라미터(parameter)를 모두 거치는 완전한 순방향 패스(forward pass)가 필요하다. 그 순방향 패스는 엄청나게 과잉 공급되어 있다. 대부분의 경우 훨씬 작은 모델이 다음 3-5개 토큰을 정확히 맞힐 수 있고, 큰 모델은 그 추측을 *검증(verify)*하기만 하면 된다. 추측이 맞으면 하나의 비용으로 5개 토큰을 얻는다. 추측 디코딩(speculative decoding, Leviathan et al. 2023)은 이를 정확하게(exact) 만들었고, EAGLE-3(2025)은 검증당 수용률을 ~4.5 토큰까지 끌어올렸다. 출력 분포를 일치시킨 채로 4-5배 속도 향상이다.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75분

## 문제 (The Problem)

H100에서 70B급 모델의 디코드 처리량(throughput)은 일반적으로 초당 40-80 토큰이다. 각 토큰은 모든 모델 가중치(weight)를 HBM에서 읽어오는 완전한 순방향 패스를 요구한다. 출력을 바꾸지 않고는 모델을 더 작게 만들 수 없다. 메모리를 넘어서 배치(batch) 크기를 늘릴 수도 없다. 막혀 있다. 순방향 패스 한 번에 모델이 토큰을 둘 이상 출력하게 만들지 않는 한 그렇다.

자기회귀(autoregressive) 생성은 언뜻 순차적으로 보인다: `x_{t+1} = sample(p(· | x_{1:t}))`. 하지만 동시에 처리할 여지가 있다. "다음 4개 토큰은 아마 [a, b, c, d]일 것이다"라고 말해주는 저렴한 예측기가 있다면, **큰 모델의 단일 순방향 패스**로 5개 위치 전부를 검증하고 가장 긴 일치 접두사(matching prefix)를 수용할 수 있다.

Leviathan, Kalai, Matias(2023, "Fast Inference from Transformers via Speculative Decoding")는 타깃 모델의 샘플링 분포를 보존하는 영리한 수용/거부(accept/reject) 규칙을 통해 이를 정확하게 만들었다. 동일한 출력 분포, 2-4배 빠르다.

## 개념 (The Concept)

### 두 모델 설정 (The Two-Model Setup)

- **타깃 모델(target model)** `M_p`: 실제로 샘플을 얻고 싶은 크고 느리며 고품질인 모델. 분포: `p(x)`.
- **드래프트 모델(draft model)** `M_q`: 작고 빠르며 품질이 낮은 모델. 분포: `q(x)`. 5-30배 작다.

스텝마다:

1. 드래프트 모델이 `K`개 토큰을 자기회귀적으로 제안한다: `x_1, x_2, ..., x_K ~ q`.
2. 타깃 모델이 모든 `K+1`개 위치에 대해 단 한 번의 순방향 패스를 병렬로 실행하여, 제안된 각 토큰에 대한 `p(x_k)`를 생성한다.
3. 아래의 수정된 거부 샘플링(rejection-sampling) 규칙을 통해 각 토큰을 왼쪽에서 오른쪽으로 수용/거부한다. 가장 긴 일치 접두사를 수용한다.
4. 어떤 토큰이라도 거부되면, 보정된 분포에서 대체값을 샘플링하고 멈춘다. 그렇지 않으면 `p(· | x_1...x_K)`에서 보너스 토큰 하나를 샘플링한다.

드래프트가 타깃과 완벽히 일치하면 타깃 순방향당 K+1개 토큰을 얻는다. 드래프트가 위치 1에서 틀리면 토큰 하나만 얻는다.

### 정확성 규칙 (The Exactness Rule)

추측 디코딩은 **p에서 샘플링하는 것과 분포상 증명 가능하게 동등(provably equivalent)**하다. 거부 규칙:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

여기서 `(p - q)+`는 점별 차이의 양수 부분을 나타낸다. 드래프트와 타깃이 일치하면(`p ≈ q`) 수용 확률은 거의 1이다. 둘이 불일치하면, 전체 샘플이 여전히 정확히 `p`가 되도록 잔차 분포(residual distribution)가 구성된다.

**그리디(greedy) 경우.** temperature=0 샘플링의 경우 단지 `argmax(p) == x_t`를 확인한다. 맞으면 수용하고, 아니면 `argmax(p)`를 출력하고 멈춘다.

### 기대 속도 향상 (Expected Speedup)

드래프트 모델의 토큰 수준 수용률이 `α`라면, 타깃 순방향 패스당 생성되는 기대 토큰 수는:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

`α = 0.8, K = 4`에서: `(1 - 0.8^5)/(1 - 0.8) = 3.36`개 토큰/순방향. 단일 타깃 순방향의 비용은 대략 `cost_q * K + cost_p`이다(K번의 드래프트 스텝과 한 번의 타깃 검증). `cost_p >> cost_q * K`라면 속도 향상 비율은 처리량에서 `3.36× / 1 = 3.36×`이다.

유일한 실질적 파라미터는 `α`이며, 이는 전적으로 드래프트-타깃 정렬에 달려 있다. 좋은 드래프트가 전부다.

### 드래프트 학습: 증류 (Training the Draft: Distillation)

무작위의 작은 모델은 형편없는 드래프트가 된다. 표준 레시피는 타깃으로부터 증류(distill)하는 것이다:

1. 작은 아키텍처를 고른다(70B 타깃에는 ~1B, 7B 타깃에는 ~500M).
2. 큰 텍스트 코퍼스(corpus)에 대해 타깃 모델을 실행하고, 그 다음 토큰 분포를 저장한다.
3. (정답 토큰이 아니라) 타깃의 분포에 대한 KL 발산(KL divergence)으로 드래프트를 학습시킨다.

결과: `α`는 코딩에서 일반적으로 0.6-0.8, 자연어 챗(chat)에서 0.7-0.85이다. 프로덕션(production)에서 2-3배 속도 향상이다.

### EAGLE: 트리 드래프팅 + 특성 재사용 (Tree Drafting + Feature Reuse)

Li, Wei, Zhang, Zhang(2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")은 표준 추측 디코딩에서 두 가지 비효율을 관찰했다:

1. 드래프트는 각각 풀 스택(full-stack)인 K번의 순차 스텝을 한다. 하지만 드래프트는 가장 최근 검증에서 타깃의 특성(은닉 상태, hidden state)을 재사용할 수 있다. 타깃은 이미 풍부한 표현을 계산했는데, 드래프트는 이를 처음부터 다시 유도한다.
2. 드래프트는 선형 체인(linear chain)을 출력한다. 드래프트가 후보들의 *트리(tree)*(각 노드가 여러 추측)를 출력할 수 있다면, 타깃의 단일 순방향 패스가 트리 어텐션 마스크(tree attention mask)를 통해 여러 후보 경로를 병렬로 검증하고 가장 길게 수용된 가지를 고를 수 있다.

EAGLE-1 변경 사항:
- 드래프트 입력 = 원시 토큰이 아니라 위치 t에서의 타깃 최종 은닉 상태.
- 드래프트 아키텍처 = (별도의 작은 모델이 아니라) 트랜스포머 디코더 층(layer) 1개.
- 출력 = 깊이당 K = 4-8개 후보의 트리, 깊이 4-6.

EAGLE-2(2024)는 동적 트리 토폴로지(dynamic tree topology)를 추가한다. 드래프트가 불확실한 곳에서는 트리가 더 넓어지고, 확신하는 곳에서는 좁게 유지된다. 검증 비용을 늘리지 않으면서 `α_effective`를 높인다.

EAGLE-3(Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test")은 고정된 최상위 층 특성 의존성을 제거하고 새로운 "테스트 타임 시뮬레이션(test-time simulation)" 손실로 드래프트를 학습시킨다. 드래프트는 교사 강요(teacher-forced) 학습 분포가 아니라 타깃의 테스트 타임 분포와 일치하는 출력에 대해 학습된다. 수용률이 0.75(EAGLE-2)에서 0.82(EAGLE-3)로 오르고, 검증당 평균 토큰 수가 3.0에서 4.5로 오른다.

### 트리 어텐션 검증 (Tree Attention Verification)

드래프트가 트리를 출력하면, 타깃 모델은 **트리 어텐션 마스크**를 사용해 단일 순방향 패스로 이를 검증한다. 이는 순수한 직선이 아니라 트리 토폴로지를 인코딩하는 인과 마스크(causal mask)다. 각 토큰은 트리에서 자신의 조상(ancestor)에만 어텐션한다. 검증 패스는 여전히 순방향 한 번, 행렬곱(matmul) 한 번이다. 토폴로지 마스크는 단지 약간의 추가 KV 엔트리 비용만 든다.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

`a, b`가 경쟁하는 첫 번째 토큰 후보이고 `c, d, e, f`가 두 번째 토큰 후보라면, 여섯 위치 모두 한 번의 순방향 패스로 검증된다. 출력은 수용된 임의의 경로를 따른 가장 긴 접두사다.

### 언제 이기고 언제 못 이기는가 (When It Wins, When It Doesn't)

**이긴다:**
- 예측 가능한 텍스트(코드, 흔한 영어, 구조화된 출력)의 챗/완성(completion). `α`가 높다.
- 디코드 중 GPU 컴퓨트가 미사용인 환경(메모리 병목 단계). 트리 드래프팅이 가용 FLOPs를 사용한다.

**진다 / 이점 없음:**
- 매우 확률적인 출력(높은 temperature의 창작 글쓰기). `α`가 `1/|vocab|` 쪽으로 떨어진다.
- 동시성이 매우 높은 배치 서빙. 배칭이 이미 FLOPs를 채우고 있어 트리 검증의 여지가 거의 없다.
- 드래프트가 그다지 작지 않은 매우 작은 타깃 모델.

프로덕션 현장은 일반적으로 챗에서 2-3배, 코드 생성에서 3-5배의 실측(wall-clock) 속도 향상, 창작 글쓰기에서는 거의 0의 향상을 보고한다.

```figure
speculative-decoding
```

## 직접 만들기 (Build It)

`code/main.py`:

- 정확한 거부 규칙을 구현하고 타깃의 분포를 보존하는지 검증하는(일반 타깃 샘플링 대비 경험적 KL < 0.01) 참조 구현 `speculative_decode(target, draft, prompt, K, temperature)`.
- top-p 분기로 깊이 K 트리를 만드는 EAGLE 스타일 트리 드래프터.
- 검증기를 위한 올바른 인과 패턴을 생성하는 트리 어텐션 마스크 빌더.
- 작은 LM에서 둘 다 실행하는 수용률 하니스(harness)(GPT-2-medium 타깃에서 GPT-2-small 하나를 증류).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 라이브러리로 써보기 (Use It)

- **vLLM**과 **SGLang**은 일급(first-class) 추측 디코딩을 제공한다. 플래그: `--speculative_model`, `--num_speculative_tokens`. `--spec_decoding_algorithm eagle` 플래그를 통해 EAGLE-2/3을 지원한다.
- **NVIDIA TensorRT-LLM**은 Medusa와 EAGLE 트리를 네이티브로 지원한다.
- **참조 드래프트 모델**: `Qwen/Qwen3-0.6B-spec`(Qwen3-32B용 드래프트), `meta-llama/Llama-3.2-1B-Instruct-spec`(70B용 드래프트).
- **Medusa 헤드**(Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): 드래프트 모델 대신, 타깃 자체에 K개의 병렬 예측 헤드를 추가한다. 배포가 더 간단하고, EAGLE보다 수용률이 약간 낮다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-speculative-tuning.md`를 생성한다. 타깃 모델의 워크로드를 프로파일링하고 드래프트 모델, K(드래프트 길이), 트리 너비, temperature, 그리고 언제 일반 디코드로 폴백(fall back)할지를 선택하는 스킬이다.

## 연습 문제 (Exercises)

1. 정확한 거부 규칙을 구현하고 경험적으로 검증하라. `speculative_decode`를 통해, 그리고 일반 타깃 샘플링을 통해 1만 개 샘플을 실행하고, 두 출력 분포 간의 TV 거리(total variation distance)를 계산하라. < 0.01이어야 한다.

2. 속도 향상 공식을 계산하라. 고정된 `α`와 `K`가 주어졌을 때, 타깃 순방향당 기대 토큰 수를 그려라. α ∈ {0.5, 0.7, 0.9}에 대해 최적 K를 찾아라.

3. 작은 드래프트를 학습시켜라. 124M GPT-2 타깃을 가져와 KL 손실로 1억 토큰에 대해 30M GPT-2 드래프트를 증류하라. 홀드아웃(held-out) 텍스트에서 `α`를 측정하라. 기대값: 0.6-0.7.

4. EAGLE 스타일 트리 드래프팅을 구현하라. 체인 대신, 드래프트가 각 깊이에서 top-3 가지를 출력하게 하라. 트리 어텐션 마스크를 만들어라. 타깃이 가장 긴 올바른 가지를 수용하는지 검증하라.

5. 실패 모드를 측정하라. temperature=1.5(높은 확률성)에서 추측 디코드를 실행하라. α가 붕괴하고 드래프트 오버헤드 때문에 알고리즘이 일반 디코드보다 느려지는 것을 보여라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 타깃 모델(Target model) | "큰 모델" | 샘플을 얻고 싶은 느리고 고품질인 모델(p 분포) |
| 드래프트 모델(Draft model) | "추측기" | 작고 빠른 예측기(q 분포); 5-30배 작음 |
| K / 드래프트 길이(draft length) | "선행 탐색(look-ahead)" | 검증 패스당 추측된 토큰 수 |
| α / 수용률(acceptance rate) | "적중률(hit rate)" | 드래프트의 제안이 수용될 토큰별 확률 |
| 정확한 거부 규칙(Exact rejection rule) | "수용 테스트" | 타깃의 분포를 보존하는 r < p/q 비교 |
| 잔차 분포(Residual distribution) | "보정된 p-q" | (p - q)+ / ||(p - q)+||_1, 거부 시 샘플링할 분포 |
| 트리 드래프팅(Tree drafting) | "분기 추측" | 드래프트가 후보 트리를 출력하고, 트리 구조 어텐션 마스크로 한 번의 패스에 검증 |
| 트리 어텐션 마스크(Tree attention mask) | "토폴로지 마스크" | 각 노드가 자신의 조상에만 어텐션하도록 트리 토폴로지를 인코딩하는 인과 마스크 |
| Medusa 헤드(Medusa heads) | "병렬 헤드" | 타깃 자체에 있는 K개의 추가 예측 헤드; 별도의 드래프트 모델 없음 |
| EAGLE 특성 재사용(feature reuse) | "은닉 상태 드래프트" | 드래프트 입력이 원시 토큰이 아니라 타깃의 마지막 은닉 상태로, 드래프트를 축소 |
| 테스트 타임 시뮬레이션 손실(Test-time simulation loss) | "EAGLE-3 학습" | 교사 강요가 아니라 타깃의 테스트 타임 분포와 일치하는 출력에 대해 드래프트를 학습 |

## 더 읽을거리 (Further Reading)

- [Leviathan, Kalai, Matias, 2023: "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) 정확한 거부 규칙과 이론적 속도 향상 분석
- [Chen, Borgeaud, Irving et al., 2023: "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) DeepMind의 동시기 추측 샘플링 논문
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) 드래프트 모델의 병렬 헤드 대안
- [Li, Wei, Zhang, Zhang, 2024: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) 특성 재사용과 트리 드래프팅
- [Li et al., 2024: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) 동적 트리 토폴로지
- [Li et al., 2025: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) 학습 시점의 테스트 타임 일치
- [Fu, Haotian, Peng et al., 2024: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) Jacobi/lookahead 디코딩, 추측기 없는 대안
