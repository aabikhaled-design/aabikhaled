# 그래디언트 체크포인팅(Gradient Checkpointing)과 활성값 재계산(Activation Recomputation)

> 역전파(backprop)는 모든 중간 활성값(activation)을 보관한다. 70B 파라미터(parameter)와 128K 컨텍스트에서 이는 랭크(rank)당 3 TB의 활성값이다. 체크포인팅(checkpointing)은 FLOPs를 메모리와 맞바꾼다. 저장하는 대신 재계산하는 것이다. 문제는 어느 세그먼트를 버릴지이며, 답은 "전부"가 아니다.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70분

## 문제 (The Problem)

트랜스포머(transformer)를 학습시킬 때, 각 층(layer)마다 역방향(backward)에서 미분되는 모든 연산의 입력을 저장한다. 어텐션(attention) 입력, Q/K/V 투영, 소프트맥스(softmax) 출력, FFN 입력, 정규화(norm) 출력, 그리고 잔차 스트림(residual stream)이다. 은닉 크기 `d`, 시퀀스 길이 `L`, 배치(batch) `B`인 층의 경우 이는 층당 약 `12 * B * L * d`개의 부동소수점이다.

`d=8192, L=8192, B=1`에서 이는 BF16으로 층당 800 MB다. 64층 모델은 51 GB의 활성값이다. 그리고 이것은 마이크로배치(microbatch) 크기를 곱하기 전, 어텐션-소프트맥스 중간값(헤드당 `L^2`)을 더하기 전, 텐서 병렬(tensor-parallel) 부분 복사본을 고려하기 전이다.

양면적 청구서: BF16 가중치(weight)와 옵티마이저(optimizer) 상태는 80GB에 들어갈 수 있지만, 활성값이 그 한계를 넘어서게 만든다. 그래디언트 체크포인팅(gradient checkpointing, 일명 활성값 재계산)이 표준 해결책이다. 대부분의 활성값을 버리고, 역방향 동안 순방향을 다시 수행하여 되찾는다. 비용: 추가 FLOPs. 이득: 체크포인트 세그먼트와 전체 층의 비율만큼 메모리가 감소한다.

순진하게 하면 체크포인팅은 스텝당 순방향 패스(forward pass) FLOPs를 대략 33% 더 쓴다. 잘하면, 즉 Korthikanti et al.의 "스마트 선택(smart selection)"에 따른 선택적 체크포인팅으로는 FLOP 오버헤드 5% 미만으로 메모리를 5배 절약한다. 그리고 FP8 행렬곱, FSDP 오프로드(offload), 전문가 병렬(expert-parallel) MoE에서는 이것이 정말 중요하다. 메모리도 낭비된 컴퓨트도 감당할 수 없기 때문이다.

## 개념 (The Concept)

### 역방향이 실제로 필요로 하는 것 (What Backward Actually Needs)

`output = layer(input)`. 역방향은 `grad_input`과 `grad_params`를 원한다. 이를 계산하려면 다음이 필요하다:

- `input` (선형 층의 경우 `grad_params = input.T @ grad_output`을 계산하기 위해)
- 일부 활성값 도함수 중간값 (ReLU/GELU/softmax의 도함수는 활성값에 의존한다)

순방향 패스는 이것들을 자동 미분(autograd) 그래프에 자동으로 저장한다. 모든 `tensor.retain_grad()`와 자신의 입력이 필요한 모든 연산이 참조를 유지한다.

### 순진한 전체 체크포인팅 (Naive Full Checkpointing)

신경망(network)을 `N`개 세그먼트로 분할한다. 순방향 동안 각 세그먼트의 *입력*만 저장한다. 역방향이 중간값을 필요로 할 때, 세그먼트의 순방향 패스를 다시 실행하여 그것들을 구체화(materialize)한 뒤 미분한다.

예: 32층 트랜스포머를 각 1층짜리 32개 세그먼트로 분할한다.

- 메모리: 32개의 층 입력(작음) 대 32 * (층당 활성값 부피)(엄청남).
- 추가 컴퓨트: 세그먼트당 추가 순방향 1번, 즉 전체적으로 ~33% 더 많은 순방향 FLOPs(역방향이 순방향의 2배이므로, 전체 스텝이 1 + 2 = 3 단위 대신 1 + 1 + 2 = 4 단위가 됨).

이것이 원래의 Chen et al. 2016 레시피다. 메모리와 컴퓨트의 균형을 맞추기 위해 `sqrt(L)`층마다 체크포인트 하나. L=64라면 8개의 체크포인트다.

### 선택적 체크포인팅 (Selective Checkpointing, Korthikanti 2022)

모든 활성값이 같은 비용은 아니다. 어텐션 소프트맥스 출력은 `B*L*L*heads`이며 시퀀스 길이에 대해 *이차적으로(quadratically)* 증가한다. FFN 은닉 활성값은 `B*L*4d`이며 선형으로 증가한다. 긴 시퀀스에서는 소프트맥스가 지배한다.

선택적 체크포인팅은 저장이 저렴한 활성값(선형 투영, 잔차)은 보관하고 비싼 것(어텐션)만 재계산한다. 재계산에 최소한의 FLOPs를 지불하면서 O(L^2) 메모리를 절약한다.

Megatron-Core는 이를 "선택적(selective)" 활성값 재계산으로 구현한다. 대부분의 2024년 이후 최첨단 학습 실행에서 사용된다.

### 오프로드 (Offload)

재계산의 대안: 순방향과 역방향 사이에 활성값을 CPU RAM으로 보낸다. PCIe 대역폭이 필요하며, 유휴 대역폭이 재구체화 비용을 초과할 때 유리하다. 혼합 전략이 흔하다. 일부 층은 체크포인트하고 다른 층은 오프로드한다.

FSDP2는 오프로드를 일급 옵션으로 제공한다. GPU가 메모리에서 병목이지만 CPU-GPU 전송에 여유가 있을 때 오프로드가 빛난다.

### 재계산 비용 모델 (Recompute Cost Model)

`L`개 층 중 `k`층마다 순진한 체크포인팅을 할 때의 스텝당 FLOPs:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

선택적 체크포인팅에서는 전체 층이 아니라 어텐션 커널(kernel)만 재계산한다:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 메모리 절약 모델 (Memory Savings Model)

층당 활성값 부피: `A`. `L`개 층에 대한 총 활성값 메모리: `L * A`.

전체 체크포인트(세그먼트 크기 1): `L * input_volume`만 저장(표준 트랜스포머의 경우 ~`L * 1/10 A`). ~`9 * L * A * 1/10`을 절약한다.

`k`층마다 체크포인트: `L/k * A`에 더해 활성 세그먼트 내의 `k-1`개 층 분량을 저장한다.

`k = sqrt(L)`에서, 메모리와 재계산 비용이 모두 `sqrt(L)`로 스케일링된다. 균일 비용 층에 대한 최적 트레이드오프(trade-off)다.

### 언제 체크포인트하지 않는가 (When Not to Checkpoint)

- 이미 진행 중인 파이프라인 스테이지의 가장 안쪽 층. 어차피 끝내야 한다.
- 첫 층과 마지막 층이 스테이지의 컴퓨트를 지배하는 경우(트랜스포머에서는 드묾).
- 이미 FlashAttention을 사용하는 어텐션 커널. Flash는 이미 소프트맥스를 빠르게 재계산하므로, 추가적인 층 수준 체크포인팅은 그 위에 거의 보태지 않는다.

### 구현 패턴 (Implementation Patterns)

1. **함수 래퍼(function wrapper):** 세그먼트를 `torch.utils.checkpoint.checkpoint(fn, input)`로 감싼다. PyTorch는 `input`만 저장하고, 나머지는 역방향에서 재계산한다.

2. **데코레이터 기반(decorator-based):** 층을 체크포인트 가능으로 표시한다. 트레이너가 설정 시점에 어느 세그먼트를 감쌀지 결정한다.

3. **수동 명시적 재계산(manual explicit recompute):** 역방향 패스를 직접 작성하여, 저장된 입력으로 순방향을 복제하는 커스텀 `recompute_forward`를 호출한다.

세 가지 모두 동일한 기능적 결과를 낸다. 래퍼가 표준 관용구다.

### TP / PP / FP8과의 상호작용 (Interaction with TP / PP / FP8)

- **텐서 병렬(Tensor parallel):** 체크포인트 입력은 재계산 시 수집(gather)하거나 재분산(rescatter)해야 한다. 통신 비용을 다뤄라.
- **파이프라인 병렬(Pipeline parallel):** 일반적인 패턴은 각 파이프라인 스테이지의 순방향을 체크포인트하여, 역순 마이크로배치가 활성값 메모리를 재사용할 수 있게 하는 것이다.
- **FP8 재계산:** 재계산 동안 갱신되는 amax 이력은 원래 순방향의 것과 일치해야 한다. 그렇지 않으면 FP8 스케일이 드리프트한다. 대부분의 프레임워크는 스케일을 스냅샷한다.

```figure
activation-recompute
```

## 직접 만들기 (Build It)

### 1단계: 세그먼트가 있는 장난감 모델 (A Toy Model With Segments)

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 2단계: 모든 활성값이 필요한 순진한 역방향 (Naive Backward Needing All Activations)

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 3단계: k층마다 체크포인트하는 메모리 (Checkpoint-Every-k Memory)

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 4단계: 비용 모델 (Cost Model)

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 5단계: 메모리 추정기 (Memory Estimator)

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 6단계: 최적 세그먼트 크기 (Optimal Segment Size)

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 7단계: 선택적 체크포인트 결정 (Selective Checkpoint Decision)

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 라이브러리로 써보기 (Use It)

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint`: PyTorch의 표준 래퍼. 함수를 감싸고, 입력만 저장하며, 역방향에서 재계산한다.
- **Megatron-Core 활성값 재계산**: `selective`, `full`, `block` 모드를 지원한다. 2024년 이후 최첨단 학습의 표준이다.
- **FSDP2 오프로드**: FSDP2의 `offload_policy`와 함께 `module.to_empty(device="cpu")`로, 재계산 대신 활성값을 CPU로 샤딩(shard)한다.
- **DeepSpeed ZeRO-Offload**: 옵티마이저 상태와 활성값에 대한 CPU 오프로드로, 체크포인팅을 보완한다.

## 산출물 (Ship It)

이 레슨은 `outputs/prompt-activation-recompute-policy.md`를 생성한다. 모델 설정(층, 은닉, 시퀀스, 배치)과 가용 GPU 메모리를 받아 층별 재계산 정책(none / selective / full / offload)을 내보내는 프롬프트(prompt)다.

## 연습 문제 (Exercises)

1. 정확성을 검증하라. `model_forward` + `model_backward`(전체 활성값) 대 `model_forward_checkpointed` + `model_backward_checkpointed`(세그먼트)를 실행하라. 파라미터 그래디언트(gradient)는 머신 정밀도(machine precision)까지 동일해야 한다.

2. 세그먼트 크기 `k`를 1부터 `L`까지 스윕(sweep)하라. FLOP 오버헤드와 메모리를 그려라. 곡선의 변곡점(knee)을 찾아라.

3. 선택적 체크포인팅을 구현하라. 어텐션 모듈의 입력은 저장하되 그 중간값은 저장하지 마라. seq=8192의 32층 모델에 대해 전체 층 체크포인팅 대비 FLOP 오버헤드를 측정하라.

4. 오프로드를 추가하라. 세그먼트 입력을 모의 "CPU 버퍼"(별도의 리스트)에 저장하라. "PCIe 대역폭"을 바이트/시간으로 측정하고 오프로드와 재계산 사이의 손익분기점(breakeven point)을 찾아라.

5. 실제 PyTorch 트랜스포머를 `torch.utils.checkpoint` 유무로 벤치마크(benchmark)하라. 메모리(`torch.cuda.max_memory_allocated` 통해)와 스텝 시간을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 그래디언트 체크포인팅(Gradient checkpointing) | "순방향을 다시 해서 메모리 절약" | 세그먼트 입력만 저장; 역방향 동안 중간값을 재계산하여 그래디언트 계산용 텐서(tensor)를 얻음 |
| 활성값 재계산(Activation recomputation) | "체크포인팅과 동일" | 같은 기법의 HPC 풍 명칭 |
| 세그먼트 크기(Segment size, k) | "체크포인트당 몇 층" | 중간값이 함께 버려지고 재구체화되는 층의 수 |
| 선택적 체크포인팅(Selective checkpointing) | "Korthikanti의 트릭" | 저장이 비싼 활성값(어텐션 소프트맥스)만 재계산; 저렴한 것은 보관 |
| 전체 체크포인팅(Full checkpointing) | "순진한 버전" | 모든 세그먼트에서 모든 층의 중간값을 재계산 |
| 블록 체크포인팅(Block checkpointing) | "거친 입자(coarse-grained)" | 트랜스포머 블록 전체를 체크포인트; 가장 큰 입도 |
| FLOP 오버헤드(FLOP overhead) | "컴퓨트 세금" | 스텝당 추가 FLOPs = (재계산 FLOPs) / (순방향 + 역방향 FLOPs); 순진하게 33%, 선택적으로 5% |
| 활성값 오프로드(Activation offload) | "CPU로 보내기" | 순방향->역방향에 걸쳐 활성값을 CPU RAM으로 이동; 재계산의 대안 |
| sqrt-L 규칙(sqrt-L rule) | "고전적 최적" | 균일 비용 층의 경우 최적 체크포인트 간격은 sqrt(L)층 |
| 어텐션-소프트맥스 부피(Attention-softmax volume) | "O(L^2) 문제" | L^2 * heads * batch개의 부동소수점; 긴 컨텍스트에서 활성값 메모리를 지배 |

## 더 읽을거리 (Further Reading)

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 그래디언트 체크포인팅을 정식화한 원조 논문
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 선택적 활성값 재계산과 정식 비용 분석
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 역방향 모드 재구체화를 통한 대안적 상수 메모리 접근법
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 대규모에서의 활성값 오프로드
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 표준 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective, full, block 모드
