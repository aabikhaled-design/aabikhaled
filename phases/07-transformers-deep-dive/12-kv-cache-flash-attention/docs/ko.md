# KV 캐시, Flash Attention & 추론 최적화

> 학습(training)은 병렬이고 FLOP에 묶여 있다. 추론(inference)은 직렬이고 메모리에 묶여 있다. 다른 병목, 다른 트릭.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75분

## 문제 (The Problem)

순진한 자기회귀(autoregressive) 디코더(decoder)는 `N`개의 토큰(token)을 생성하는 데 `O(N²)`의 일을 한다: 각 스텝에서 전체 접두사(prefix)를 두고 어텐션(attention)을 다시 계산한다. 4K 토큰 응답이라면 1,600만 번의 어텐션 연산이고, 대부분이 중복이다. 접두사 토큰의 모든 은닉 상태(hidden state)는 한 번 계산되면 결정론적이다. 새 토큰의 쿼리(query)를 그 이전 모든 것의 캐시된 키(key)와 값(value)에 대해 돌리기만 하면 된다.

게다가 어텐션 자체가 많은 데이터를 옮긴다. 표준 어텐션은 N×N 점수 행렬(matrix), N×d softmax 출력, N×d 최종 출력을 실체화한다. HBM에 대한 읽기와 쓰기가 너무 많다. N≥2K에서 어텐션은 FLOP에 묶이기 전에 메모리에 묶인다. 고전적 어텐션 커널(kernel)은 현대 GPU를 4~10배 적게 활용한다.

둘 다 Dao et al.에서 나온 두 가지 최적화가 프런티어(frontier) 추론을 "느림"에서 "빠름"으로 밀어 올렸다:

1. **KV 캐시.** 모든 접두사 토큰의 K와 V 벡터(vector)를 저장한다. 새 토큰의 어텐션은 캐시된 키에 대한 쿼리 한 번이다. 추론이 생성 스텝당 `O(N²)`에서 `O(N)`으로 줄어든다.
2. **Flash Attention.** 어텐션 계산을 타일링(tiling)해 전체 N×N 행렬이 절대 HBM에 닿지 않게 한다. softmax + 행렬곱(matmul) 전부가 SRAM에서 일어난다. A100에서 2~4배, FP8을 쓰는 H100에서 5~10배의 실측 시간(wall-clock) 가속.

2026년에 이르러 둘 다 보편적이다. 모든 프로덕션(production) 추론 스택(vLLM, TensorRT-LLM, SGLang, llama.cpp)이 이를 전제한다. 모든 프런티어 모델은 Flash Attention이 켜진 채로 출시된다.

## 개념 (The Concept)

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV 캐시 수학

디코더 층(layer)당, 토큰당, 헤드(head)당:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

32개 층, 32개 헤드, d_head=128, fp16의 7B 모델(model)의 경우:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Llama 3 70B(80개 층, d_head=128, KV 헤드 8개의 GQA)의 경우:

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

그 10 GB가 128K 컨텍스트(context)의 Llama 3 70B가 배치(batch) 크기 1에서도 KV 캐시만으로 40 GB A100의 대부분을 차지하는 이유다.

**GQA가 KV 캐시의 승리다.** 64개 헤드의 MHA였다면 32 GB가 될 것이다. MLA는 더 깊이 압축한다.

차원 값을 끌어당기며 캐시 크기가 어떻게 움직이는지 보라. 시퀀스 길이나 배치를 키우면 GPU 한 장을 얼마나 빨리 넘어서는지 알 수 있다.

```figure
kv-cache-sizer
```

### Flash Attention: 타일링 트릭

표준 어텐션:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

세 번의 HBM 왕복. H100에서 HBM 대역폭은 3 TB/s, SRAM은 30 TB/s다. 모든 HBM 왕복은 모든 것을 칩 위에 두는 것 대비 10배의 둔화다.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

타일당 HBM 왕복 한 번. 전체 메모리 풋프린트(footprint)가 `O(N²)`에서 `O(N)`으로 떨어진다. 역방향 패스(backward pass)는 일부 값을 저장하는 대신 순방향 패스(forward pass)에서 재계산한다. 또 하나의 메모리 승리.

**수치적 트릭.** 러닝 softmax는 타일에 걸쳐 `(max, sum)`을 유지해 최종 정규화(normalization)가 정확하다. 근사가 아니다. Flash Attention은 표준 어텐션과 비트 단위로 동일한 출력을 계산한다(fp16 비결합성(non-associativity)은 제외).

**버전 진화:**

| 버전 | 연도 | 핵심 변화 | 레퍼런스 하드웨어에서의 가속 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 타일링된 SRAM 커널 | A100에서 2× |
| Flash 2 | 2023 | 더 나은 병렬화, 인과(causal) 우선 순서 | A100에서 3× |
| Flash 3 | 2024 | Hopper 비동기성, FP8 | H100에서 1.5~2× (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5단계 파이프라인(pipeline), 소프트웨어 exp2 | 추론 우선(초기에는 순방향만) |

Flash 4는 출시 시점에 순방향 패스 전용이다. 학습은 여전히 Flash 3를 쓴다. Flash 4의 GQA와 가변 길이(varlen) 지원은 대기 중이다(2026년 중반).

### 추측 디코딩(speculative decoding): 또 다른 지연 시간 승리

저렴한 모델이 N개의 토큰을 제안한다. 큰 모델이 N개 모두를 병렬로 검증한다. 검증이 k개의 토큰을 받아들이면, k개의 생성에 대해 큰 모델의 순방향 패스 1번을 치른 셈이다. 코드와 산문에서 일반적으로 k=3~5.

2026년 기본값:
- **EAGLE 2 / Medusa.** 검증기(verifier)의 은닉 상태를 공유하는 통합 드래프트 헤드(draft head). 품질 손실 없이 2~3배 가속.
- **드래프트 모델을 활용한 추측 디코딩.** 소비자용 하드웨어에서 2~4배 가속.
- **룩어헤드 디코딩(lookahead decoding).** 야코비(Jacobi) 반복; 드래프트 모델이 필요 없다. 틈새지만 공짜.

### 연속 배칭(continuous batching)

고전적 배치 추론: 가장 느린 시퀀스(sequence)가 끝날 때까지 기다린 뒤 새 배치를 시작한다. 짧은 응답이 일찍 끝나면 GPU를 낭비한다.

연속 배칭(Orca에서 처음 출시, 현재 vLLM, TensorRT-LLM, SGLang에 탑재): 오래된 시퀀스가 끝나자마자 새 요청을 배치에 끼워 넣는다. 일반적인 채팅 워크로드에서 5~10배 처리량(throughput) 향상.

### PagedAttention: 가상 메모리로서의 KV 캐시

vLLM의 대표 기능이다. KV 캐시는 16토큰 블록 단위로 할당되고, 페이지 테이블(page table)이 논리적 위치를 물리적 블록에 매핑한다. 병렬 샘플(빔 서치(beam search), 병렬 샘플링) 간에 KV를 공유하고, 프롬프트 캐싱(prompt caching)을 위해 접두사를 핫스왑(hot-swap)하며, 메모리를 조각 모음(defragment)할 수 있게 한다. 순진한 연속 할당 대비 4배 처리량 향상.

```figure
flash-attention-memory
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. 다음을 구현한다:

1. 순진한 `O(N²)` 증분(incremental) 디코더.
2. `O(N)` KV 캐시 디코더.
3. Flash Attention의 러닝-맥스 알고리즘을 모사하는 타일링된 softmax.

### 1단계: KV 캐시

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

단순하다: 층별, 헤드별 리스트에 토큰별 K, V 벡터를 계속 늘려 간다.

### 2단계: 타일링된 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

`softmax(qK) V`를 한 번에 한 것과 비트 단위로 동일한 출력이지만, 어느 시점에서든 작업 집합(working set)은 전체 `N × d_head`가 아니라 `tile × d_head` 블록이다.

### 3단계: 100토큰 생성에서 순진한 디코딩 vs 캐시 디코딩 비교

어텐션 연산 수를 센다. 순진: `O(N²)` = 5050. 캐시: `O(N)` = 100. 코드가 둘 다 출력한다.

## 라이브러리로 써보기 (Use It)

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 프로덕션:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

요청 간 접두사 캐싱은 2026년의 큰 승리다. 같은 시스템 프롬프트, 퓨샷(few-shot) 예시, 또는 긴 컨텍스트 문서가 호출 간에 KV를 재사용한다. 반복되는 도구 프롬프트가 있는 에이전트(agent) 워크로드의 경우, 접두사 캐싱은 흔히 5배 처리량 향상을 낸다.

## 산출물 (Ship It)

`outputs/skill-inference-optimizer.md`를 참고하라. 이 스킬은 새 추론 배포(deployment)에 대해 어텐션 구현, KV 캐시 전략, 양자화(quantization), 추측 디코딩을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 순진한 디코더와 캐시 디코더가 같은 출력을 내는지 확인하고, 연산 수의 차이에 주목하라.
2. **중간.** 접두사 캐싱을 구현하라: 프롬프트 P와 여러 완성(completion)이 주어졌을 때, P에 대해 순방향 패스를 한 번 돌려 KV 캐시를 채운 뒤 완성마다 분기(branch)하라. 각 완성에 대해 P를 다시 인코딩하는 것 대비 가속을 측정하라.
3. **어려움.** 장난감 PagedAttention을 구현하라: 자유 리스트(free-list)를 가진 고정 16토큰 블록 단위의 KV 캐시. 시퀀스가 끝나면 그 블록을 풀(pool)로 반환하라. 길이가 다양한 1,000개의 채팅 완성을 시뮬레이션하라. 연속 할당 대비 메모리 단편화(fragmentation)를 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| KV 캐시(KV cache) | "디코딩을 빠르게 만드는 트릭" | 모든 접두사 토큰에서 저장된 K와 V; 새 쿼리가 재계산 대신 이들에 어텐션한다. |
| HBM | "GPU 주 메모리" | High Bandwidth Memory; H100에서 80 GB, B200에서 192 GB. ~3 TB/s 대역폭. |
| SRAM | "온칩 메모리" | SM별 빠른 메모리, H100에서 SM당 ~256 KB. ~30 TB/s 대역폭. |
| Flash Attention | "타일링된 어텐션 커널" | N×N을 HBM에 실체화하지 않고 어텐션을 계산한다. |
| 연속 배칭(Continuous batching) | "대기 없는 배칭" | 배치를 비우지 않고 끝난 시퀀스를 빼고 새 것을 넣는다. |
| PagedAttention | "vLLM의 간판" | 페이지 테이블과 함께 고정 블록으로 할당되는 KV 캐시; 단편화를 제거한다. |
| 접두사 캐싱(Prefix caching) | "긴 프롬프트 재사용" | 요청 간 공유 접두사의 KV를 캐싱; 에이전트의 비용을 크게 절감한다. |
| 추측 디코딩(Speculative decoding) | "드래프트 + 검증" | 저렴한 드래프트 모델이 토큰을 제안; 큰 모델이 한 번에 k개를 검증한다. |

## 더 읽을거리 (Further Reading)

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention): Blackwell 5단계 파이프라인과 소프트웨어-exp2 트릭; 이 레슨이 언급하는 순방향 전용 출시 주의사항은 repo README에서 읽어라.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): vLLM 논문.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192): 추측 디코딩.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077): 레슨이 인용하는 통합 드래프트 접근에 대한 EAGLE-1/2 논문.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774): EAGLE과 함께 언급된 Medusa 접근.
- [vLLM docs: PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) 16토큰 블록과 페이지 테이블 설계에 대한 표준 심층 해설.
