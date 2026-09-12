# 위치 인코딩(Positional Encoding): Sinusoidal, RoPE, ALiBi

> 어텐션(attention)은 순열 불변(permutation-invariant)이다. 위치 신호가 없으면 "The cat sat on the mat"와 "mat the on sat cat the"가 같은 출력을 낸다. 세 알고리즘이 이를 고친다. 각각 "위치"가 무엇을 뜻하는지에 대해 다른 베팅을 한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45분

## 문제 (The Problem)

스케일드 닷-프로덕트 어텐션(scaled dot-product attention)은 순서를 보지 못한다. 어텐션 행렬(matrix) `softmax(Q K^T / √d) V`는 쌍별 유사도에서 계산된다. `X`의 행을 섞으면, 출력의 행이 똑같이 섞여 나온다. 어텐션 내부의 그 무엇도 위치에 관심을 두지 않는다.

이는 단어 가방(bag-of-words) 모델에서는 버그가 아니다. 언어, 코드, 오디오, 비디오. 순서가 의미를 담는 모든 것: 에서는 치명적이다.

해법은 어떻게든 위치를 임베딩(embedding)에 주입하는 데 있다. 답의 세 시대:

1. **절대 사인파(Absolute sinusoidal)** (Vaswani 2017). 위치의 `sin/cos`를 임베딩에 더한다. 단순하고, 학습이 필요 없고, 학습된 길이를 넘어서면 외삽이 형편없다.
2. **RoPE: Rotary Position Embeddings** (Su 2021). Q와 K 벡터(vector)를 위치에 비례하는 각도로 회전한다. *상대적* 위치를 내적(dot product)에 직접 인코딩한다. 2026년에 지배적이다.
3. **ALiBi: Attention with Linear Biases** (Press 2022). 임베딩을 완전히 건너뛴다. 거리에 기반한 헤드별 선형 패널티를 어텐션 점수에 더한다. 길이 외삽이 탁월하다.

2026년 기준 사실상 모든 프런티어 오픈 모델이 RoPE를 쓴다. Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. 소수의 장기 컨텍스트 모델은 ALiBi나 그 현대적 변형을 쓴다. 절대 사인파는 이제 역사 속 방식이다.

## 개념 (The Concept)

![사인파 절대 vs RoPE 회전 vs ALiBi 거리 편향](../assets/positional-encoding.svg)

### 절대 사인파

`(max_len, d_model)` 모양의 고정 행렬 `PE`를 미리 계산한다.

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

그다음 어텐션 전에 `X' = X + PE[:N]`. 각 차원은 서로 다른 주파수의 사인파다. 모델은 위상 패턴에서 위치를 읽는 법을 배운다. `max_len`을 넘어서면 실패한다. 모델이 위치 0-2047만 봤다면, 위치 2048에서 무슨 일이 일어나는지 아무도 알려주지 않았다.

### RoPE

Q와 K 벡터(임베딩이 아니라)를 회전한다. 차원 쌍 `(2i, 2i+1)`에 대해:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

위치 `pos_k`로 키에 같은 회전을 적용한다. 내적 `q'_m · k'_n`은 `(m - n)`만의 함수가 된다. 즉, **어텐션 점수는 회전이 절대 위치를 기준으로 했음에도 상대 거리에만 의존한다.** 아름다운 트릭이다.

RoPE 확장: `base`를 스케일링(NTK-aware, YaRN, LongRoPE)해 재학습 없이 더 긴 컨텍스트로 외삽할 수 있다. Llama 3은 이 방식으로 8K에서 128K 컨텍스트로 확장했다.

### ALiBi

임베딩 트릭을 건너뛴다. 어텐션 점수를 직접 편향시킨다.

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

여기서 `m_h`는 헤드별 기울기(예: `1 / 2^(8·h/H)`)다. 가까운 토큰은 부스트되고, 먼 토큰은 패널티를 받는다. 학습 시 비용이 없다. 논문은 길이 외삽이 사인파를 이기고 원래 학습된 길이에서 RoPE와 맞먹음을 보여준다.

### 2026년에 무엇을 고를까

| 변형 | 외삽 | 학습 비용 | 사용처 |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE가 이긴 이유는 아키텍처를 바꾸지 않고 어텐션에 끼워지고, 상대 위치를 인코딩하며, 그 `base` 하이퍼파라미터(hyperparameter)가 장기 컨텍스트 파인튜닝(fine-tuning)을 위한 깔끔한 손잡이를 주기 때문이다.

```figure
rope-explorer
```

## 직접 만들기 (Build It)

### 1단계: 사인파 인코딩

`code/main.py`를 참조하라. 4줄짜리 계산:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

첫 어텐션 층(layer) 전에 이것을 임베딩 행렬에 더한다.

### 2단계: Q, K에 적용된 RoPE

RoPE는 Q와 K에 제자리(in-place)로 동작한다. 각 차원 쌍에 대해:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

결정적으로: 위치 `m`의 Q와 위치 `n`의 K에 같은 함수를 적용한다. 그 내적은 모든 좌표 쌍에서 `cos((m-n)·θ_i)` 인자를 얻는다. 어텐션은 상대 위치를 공짜로 학습한다.

### 3단계: ALiBi 기울기와 편향

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

헤드 `h`의 `(seq_len, seq_len)` 어텐션 점수 행렬에 `bias[h]`를 더한 뒤 소프트맥스한다.

### 4단계: RoPE의 상대 거리 속성 검증하기

무작위 벡터 `a, b` 두 개를 고른다. `(pos_a, pos_b)`로 회전한다. 그다음 `(pos_a + k, pos_b + k)`로 회전한다. 두 내적은 부동소수점 오차 이내에서 일치해야 한다. 그 속성이 RoPE의 핵심이다. 절대 오프셋에 불변이고, 상대 간격만 중요하다.

## 라이브러리로 써보기 (Use It)

PyTorch 2.5+는 `torch.nn.functional`에 RoPE 유틸리티를 제공한다. 대부분의 프로덕션(production) 코드는 RoPE가 어텐션 커널 내부에서 적용되는 `flash_attn`이나 `xformers`를 쓴다.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026년의 장기 컨텍스트 트릭:**

- **NTK-aware 보간.** 4K에서 16K+로 확장할 때 `base`를 `base * (scale_factor)^(d/(d-2))`로 재스케일링한다.
- **YaRN.** 긴 컨텍스트에서 어텐션 엔트로피를 보존하는 더 똑똑한 보간. Llama 3.1 128K가 쓴다.
- **LongRoPE.** 진화 탐색을 사용해 차원별 스케일 인자를 고르는 Microsoft의 2024년 방법. Phi-3-Long이 쓴다.
- **위치 보간 + 파인튜닝.** 그냥 위치를 확장 인자만큼 줄이고 1-5B 토큰 동안 파인튜닝한다. 놀랍도록 효과적이다.

## 산출물 (Ship It)

`outputs/skill-positional-encoding-picker.md`를 참조하라. 이 스킬은 목표 컨텍스트 길이, 외삽 필요성, 학습 예산이 주어지면 새 모델에 맞는 인코딩 전략을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `max_len=512, d=128`에 대해 사인파 `PE` 행렬을 히트맵으로 플롯한다. "차원 인덱스가 커질수록 줄무늬가 넓어지는" 패턴을 확인한다.
2. **보통.** NTK-aware RoPE 스케일링을 구현한다. 길이 256 시퀀스로 작은 LM을 학습한 뒤, 스케일링이 있을 때와 없을 때 길이 1024에서 테스트한다. 퍼플렉서티(perplexity)를 측정한다.
3. **어려움.** 같은 어텐션 모듈에 ALiBi와 RoPE를 구현한다. 길이 512 시퀀스의 복사 과제로 4층 트랜스포머를 학습한다. 테스트 시 2048로 외삽한다. 성능 저하를 비교한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Positional encoding | "어텐션에 순서를 알려줌" | 위치를 인코딩하기 위해 임베딩이나 어텐션에 더해진 모든 신호. |
| Sinusoidal | "원조" | 기하 주파수의 `sin/cos`를 임베딩에 더함. 외삽 안 됨. |
| RoPE | "회전 임베딩" | 위치 의존 각도로 Q, K를 회전. 내적이 상대 거리를 인코딩. |
| ALiBi | "선형 편향 트릭" | 어텐션 점수에 `-m·\|i-j\|`를 더함. 임베딩 불필요, 훌륭한 외삽. |
| base | "RoPE의 손잡이" | RoPE의 주파수 스케일러. 추론 시 컨텍스트 확장을 위해 늘림. |
| NTK-aware | "RoPE 스케일링 트릭" | 컨텍스트 확장 시 고주파 차원이 짓눌리지 않도록 `base`를 재스케일. |
| YaRN | "고급 버전" | 어텐션 엔트로피를 보존하는 차원별 보간+외삽. |
| Extrapolation | "학습 길이를 넘어 동작" | 위치 방식이 학습 시 본 `max_len`을 넘어 올바른 출력을 낼 수 있는가? |

## 더 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762): 원조 사인파.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864): RoPE 논문.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071): 최신 RoPE 스케일링.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595): Meta의 Llama 2 장기 컨텍스트 논문.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753): Phi-3-Long이 쓰고 Use It 섹션에서 인용한 Microsoft 방법.
- [HuggingFace Transformers: `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) 모든 RoPE 스케일링 방식(default, linear, dynamic, YaRN, LongRoPE, Llama-3)의 프로덕션급 구현.
