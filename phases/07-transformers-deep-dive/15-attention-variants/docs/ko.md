# 어텐션 변형: 슬라이딩 윈도우, 희소, 차분(Differential)

> 완전 어텐션(full attention)은 원이다. 모든 토큰(token)이 모든 토큰을 보고, 메모리가 그 대가를 치른다. 네 가지 변형이 원의 모양을 구부려 비용의 절반을 되찾는다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60분

## 문제 (The Problem)

완전 어텐션(full attention)은 시퀀스(sequence) 길이에 대해 `O(N²)` 메모리와 `O(N²)` 연산이 든다. 128K 컨텍스트(context)의 Llama 3 70B의 경우 층(layer)당 160억 개의 어텐션 항목이고, 그것이 80개 층이다. Flash Attention(Lesson 12)은 `O(N²)` 활성(activation) 메모리를 숨기지만 산술 비용은 바꾸지 않는다. 모든 토큰이 여전히 다른 모든 토큰에 어텐션한다.

세 부류의 변형이 어텐션 행렬(matrix) 자체의 위상(topology)을 바꾼다:

1. **슬라이딩 윈도우 어텐션(sliding window attention, SWA).** 각 토큰이 전체 접두사(prefix)가 아니라 고정된 이웃 윈도우에 어텐션한다. 메모리와 연산이 `O(N · W)`로 떨어진다(여기서 `W`는 윈도우). Gemma 2/3, Mistral 7B의 초기 층들, Phi-3-Long.
2. **희소(sparse) / 블록 어텐션.** 선택된 쌍 `(i, j)`만 점수가 매겨지고, 나머지는 강제로 가중치 0이 된다. Longformer, BigBird, OpenAI 희소 트랜스포머(transformer).
3. **차분 어텐션(differential attention).** 별도의 Q/K 사영(projection)으로 두 개의 어텐션 맵을 계산해 하나에서 다른 하나를 뺀다. 첫 몇 토큰으로 가중치가 새어 나가는 "어텐션 싱크(attention sink)"를 죽인다. Microsoft의 DIFF Transformer(2024).

이들은 공존한다. 2026년 프런티어(frontier) 모델은 종종 이들을 섞는다: 대부분의 층은 SWA-1024이고, 다섯 번째마다 전역(global) 완전 어텐션이며, 소수는 검색(retrieval)을 정리하는 차분 헤드(head)다. Gemma 3의 5:1 SWA-대-전역 비율이 현재의 교과서 기본값이다.

## 개념 (The Concept)

### 슬라이딩 윈도우 어텐션 (SWA)

위치 `i`의 각 쿼리(query)는 `[i - W, i]`(인과(causal) SWA) 또는 `[i - W/2, i + W/2]`(양방향) 안의 위치에만 어텐션한다. 윈도우 바깥의 토큰은 점수 행렬에서 `-inf`를 받는다.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192`이고 `W = 1024`인 경우, 점수 행렬은 기댓값으로 1024 × 8192개의 0이 아닌 행을 가진다. 8배 감소.

**SWA로 KV 캐시가 줄어든다.** K와 V의 마지막 `W`개 토큰만 층당 유지하면 된다. Gemma-3 스타일 구성(1024 윈도우, 128K 컨텍스트)의 경우, KV 캐시가 128배 줄어든다.

**품질 비용.** SWA만 쓰는 트랜스포머는 장거리 검색에 어려움을 겪는다. 해법: SWA 층을 완전 어텐션 층과 번갈아 배치한다. Gemma 3는 5:1 SWA:전역을 쓴다. Mistral 7B는 정보가 겹치는 윈도우로 "앞으로 흐르는" 인과-SWA 스택(stack)을 썼다. 각 층이 유효 수용 영역(receptive field)을 `W`만큼 늘리고, `L`개 층 후 모델은 `L × W` 토큰 뒤까지 어텐션할 수 있다.

### 희소 / 블록 어텐션

`N × N` 희소성(sparsity) 패턴을 미리 고른다. 세 가지 표준 모양:

- **국소 + 스트라이드(local + strided, OpenAI 희소 트랜스포머).** 마지막 `W`개 토큰에 더해 그 이전 `stride`번째마다의 토큰에 어텐션한다. `O(N · sqrt(N))` 연산으로 국소와 장거리를 모두 포착한다.
- **Longformer / BigBird.** 국소 윈도우 + 모두에게 어텐션하고 모두로부터 어텐션받는 소수의 전역 토큰(예: `[CLS]`) + 무작위-희소 링크. 동일 품질에서 경험적으로 2배 컨텍스트.
- **네이티브 희소 어텐션(Native Sparse Attention, DeepSeek, 2025).** `(Q, K)`의 어떤 블록이 중요한지 학습한다; 커널(kernel) 수준에서 0 블록을 건너뛴다. FlashAttention 호환.

희소 어텐션은 커널 엔지니어링 이야기다. 수학은 단순하다(점수 행렬을 마스킹(mask)). 이득은 0 항목을 절대 SRAM으로 로드하지 않는 데서 나온다. FlashAttention-3와 2026년의 FlexAttention API는 커스텀 희소 패턴을 PyTorch에서 일급(first-class)으로 만든다.

### 차분 어텐션 (DIFF Transformer, 2024)

일반 어텐션에는 "어텐션 싱크" 문제가 있다: softmax는 모든 행이 1로 합산되도록 강제하므로, 특별히 어디에도 어텐션하고 싶지 않은 토큰은 첫 번째 토큰(또는 첫 몇 개)에 가중치를 쏟아붓는다. 이는 실제 내용으로 갔어야 할 용량(capacity)을 훔친다.

차분 어텐션은 **두 개**의 어텐션 맵을 계산해 빼는 방식으로 이를 고친다:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

여기서 `λ`는 학습된 스칼라(보통 0.5~0.8)다. A1은 실제 내용 가중치를 포착하고, A2는 싱크를 포착한다. 뺄셈이 싱크를 상쇄하고 가중치를 관련 토큰으로 재배분한다.

보고된 결과(Microsoft 2024): 5~10% 낮은 퍼플렉시티(perplexity), 같은 학습 길이에서 1.5~2배 긴 유효 컨텍스트, 더 날카로운 건초더미 속 바늘(needle-in-haystack) 검색.

### 변형 비교

| 변형 | 연산 | KV 캐시 | 완전 어텐션 대비 품질 | 프로덕션 사용 |
|---------|---------|----------|-----------------|----------------|
| 완전 어텐션 | O(N²) | 층당 O(N) | 베이스라인(baseline) | 모든 모델의 기본 층 |
| SWA (윈도우 1024) | O(N·W) | 층당 O(W) | -0.1 ppl, 전역 층과 함께면 좋음 | Gemma 2/3, Phi-3-Long |
| 국소 + 스트라이드 희소 | O(N·√N) | 혼합 | SWA와 유사 | OpenAI 희소 트랜스포머, Longformer |
| BigBird (국소 + 전역 + 무작위) | O(N) 근사 | 혼합 | 2배 컨텍스트에서 완전 어텐션에 필적 | 초기 장문 컨텍스트 BERT |
| 네이티브 희소 (DeepSeek-V3.2) | O(N · 활성 비율) | O(N) | 0.05 ppl 이내 | DeepSeek-V3.2, 2025 |
| 차분 | O(2·N²) | O(2N) | -5 ~ -10% ppl | DIFF Transformer, 2026년 초 모델 |

```figure
gqa-kv-sharing
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. 장난감 시퀀스에서 완전, SWA, 국소+스트라이드, 차분 어텐션을 나란히 보여 주는 인과 마스크 비교기(comparator)를 구현한다.

### 1단계: 완전 인과 마스크(베이스라인)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07의 베이스라인. 하삼각(lower triangular); 대각선 위는 가중치 0.

### 2단계: 슬라이딩 윈도우 인과 마스크

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

파라미터(parameter) 하나: `window`. `window >= n`이면 완전 인과 어텐션을 되찾는다. `window = 1`이면 각 토큰이 자기 자신에만 어텐션한다.

### 3단계: 국소 + 스트라이드 희소 마스크

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

밀집(dense) 국소 윈도우에 더해 시퀀스 시작까지 `stride`번째마다의 토큰. 수용 영역이 추가 층과 함께 로그 스텝으로 자란다.

### 4단계: 차분 어텐션

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

두 번의 어텐션 패스, 학습된 혼합 계수로 뺀다. 코드에서는 단일 vs 차분의 어텐션-싱크 히트맵(heatmap)을 비교하고 싱크가 붕괴하는 것을 관찰한다.

### 5단계: KV 캐시 크기

각 변형에 대해 `N = 131072`에서 층당 캐시 크기를 출력한다. SWA와 희소 변형은 10~100배 줄어든다. 차분은 두 배가 된다. 메모리 청구서를 의식적으로 치러라.

## 라이브러리로 써보기 (Use It)

2026년 프로덕션(production) 패턴:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+의 FlexAttention은 마스크 함수를 받는다:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

이것은 커스텀 Triton 커널로 컴파일된다. 일반적인 패턴에 대해 FlashAttention-3 속도의 10% 이내이며, 마스크 함수는 Python 호출 가능 객체(callable)다.

**각각을 선택할 때:**

- **순수 완전 어텐션**: ~16K 컨텍스트까지의 모든 층, 또는 검색 품질이 가장 중요할 때.
- **SWA + 전역 혼합**: 장문 컨텍스트(>32K), 학습과 추론이 메모리에 묶일 때. 32K 이상에서 2026년 기본값.
- **희소 블록 어텐션**: 커스텀 커널, 커스텀 패턴. 특화된 워크로드(검색, 오디오)를 위해 예약.
- **차분 어텐션**: 어텐션-싱크 오염이 해가 되는 모든 워크로드(장문 컨텍스트 RAG, 건초더미 속 바늘).

## 산출물 (Ship It)

`outputs/skill-attention-variant-picker.md`를 참고하라. 이 스킬은 목표 컨텍스트 길이, 검색 요구, 학습/추론 연산 프로파일이 주어진 새 모델에 대해 어텐션 위상을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. `window=4`의 SWA가 행마다 마지막 4개 토큰 바깥의 모든 것을 0으로 만드는지 확인하라. `window=n`이 완전 인과 어텐션을 비트 단위로 동일하게 재현하는지 확인하라.
2. **중간.** Lesson 07 캡스톤 위에 `window=1024`의 인과 SWA를 구현하라. tinyshakespeare에서 1,000 스텝 학습하라. 완전 어텐션 대비 검증 손실(val loss)이 얼마나 후퇴하는가? 최대 메모리는 얼마나 떨어지는가?
3. **어려움.** 캡스톤 모델에 Gemma-3 스타일 5:1 층 혼합(SWA 5개, 전역 1개)을 구현하라. 동일 파라미터에서 순수-SWA 및 순수-전역 베이스라인 대비 손실, 메모리, 생성 품질을 비교하라.
4. **어려움.** 헤드당 학습된 `λ`를 가진 차분 어텐션을 구현하라. 합성 검색 작업(바늘 하나, 방해 요소 2,000개)에 학습하라. 동일 파라미터에서 단일 어텐션 베이스라인 대비 검색 정확도를 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 슬라이딩 윈도우 어텐션(Sliding window attention, SWA) | "국소 어텐션" | 각 쿼리가 마지막 `W`개 토큰에 어텐션한다; KV 캐시가 `O(W)`로 줄어든다. |
| 유효 수용 영역(Effective receptive field) | "모델이 얼마나 뒤까지 보는가" | 윈도우 `W`의 `L`층 SWA 스택에서 최대 `L × W` 토큰. |
| Longformer / BigBird | "국소 + 전역 + 무작위" | 항상 어텐션하는 소수의 전역 토큰을 가진 희소 패턴; 초기 장문 컨텍스트 접근. |
| 네이티브 희소 어텐션(Native Sparse Attention) | "DeepSeek의 커널 트릭" | 블록 수준 희소성을 학습한다; 품질을 유지하며 커널 수준에서 0 블록을 건너뛴다. |
| 차분 어텐션(Differential attention) | "두 맵, 하나를 뺌" | DIFF Transformer: 어텐션 싱크를 상쇄하기 위해 첫 어텐션 맵에서 학습된 `λ` 배의 두 번째 맵을 뺀다. |
| 어텐션 싱크(Attention sink) | "가중치가 토큰 0으로 샘" | softmax 정규화(normalization)가 행을 1로 합산하도록 강제한다; 정보가 없는 쿼리가 위치 0에 가중치를 쏟는다. |
| FlexAttention | "마스크를 Python으로" | 임의의 마스크 함수를 FlashAttention 형태 커널로 컴파일하는 PyTorch 2.5+ API. |
| 층 유형 혼합(Layer type mix) | "5:1 SWA-대-전역" | 더 낮은 메모리로 품질을 유지하기 위해 스택에서 희소 어텐션 층과 완전 어텐션 층을 번갈아 배치. |

## 더 읽을거리 (Further Reading)

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150): 표준 슬라이딩 윈도우 + 전역 토큰 논문.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062): 국소 + 전역 + 무작위.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509): OpenAI의 국소+스트라이드 패턴.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118): 1:1 SWA:전역 혼합.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786): 이제 교과서 기본값인 window=1024의 5:1 혼합.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258): DIFF Transformer 논문.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089): DeepSeek-V3.2의 학습된 희소성 어텐션.
- [PyTorch: FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) Use It의 마스크-호출가능 패턴에 대한 API 레퍼런스.
