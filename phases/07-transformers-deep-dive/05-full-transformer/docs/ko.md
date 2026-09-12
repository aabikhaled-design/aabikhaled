# 완전한 트랜스포머(The Full Transformer): 인코더 + 디코더

> 어텐션(attention)이 주연이다. 나머지 모든 것(잔차(residual), 정규화(normalization), 피드포워드(feed-forward), 크로스 어텐션(cross-attention))은 그것을 깊이 쌓을 수 있게 하는 비계(scaffolding)다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75분

## 문제 (The Problem)

단일 어텐션 층(layer)은 모델이 아니라 특성(feature) 추출기다. 층당 하나의 행렬곱(matmul)은 언어를 다루기에 충분한 용량이 아니다. 그래서 깊이가 필요하다. 그런데 깊이는 올바른 배관(plumbing) 없이는 무너진다.

2017년 Vaswani 논문은 하나의 어텐션 층을 쌓을 수 있는 블록으로 바꾼 여섯 가지 설계 결정을 묶었다. 그 이후의 모든 트랜스포머(transformer)(인코더 전용(BERT), 디코더 전용(GPT), 인코더-디코더(T5))가 같은 골격을 물려받는다. 2026년에 블록들은 다듬어졌지만(RMSNorm, SwiGLU, pre-norm, RoPE) 골격은 동일하다.

이 레슨은 그 골격이다. 다음 레슨들이 이를 특화한다. 06은 인코더, 07은 디코더, 08은 인코더-디코더.

## 개념 (The Concept)

![인코더와 디코더 블록 내부, 연결된 모습](../assets/full-transformer.svg)

### 여섯 조각

1. **임베딩 + 위치 신호.** 토큰(token) → 벡터(vector). 위치는 RoPE(현대) 또는 사인파(고전)를 통해 주입된다.
2. **셀프 어텐션(Self-attention).** 모든 위치가 다른 모든 위치에 어텐션한다. 디코더에서는 마스킹된다.
3. **피드포워드 네트워크(FFN).** 위치별 2층 MLP: `W_2 · activation(W_1 · x)`. 기본 확장비 4배.
4. **잔차 연결(Residual connection).** `x + sublayer(x)`. 이것이 없으면 약 6층을 넘어가면 그래디언트(gradient)가 소실된다.
5. **층 정규화(Layer normalization).** `LayerNorm` 또는 `RMSNorm`(현대). 잔차 스트림(residual stream)을 안정화한다.
6. **크로스 어텐션(디코더 전용).** 쿼리는 디코더에서, 키와 값은 인코더 출력에서 온다.

벡터 하나가 블록을 통과하는 과정을 따라가 보라. 어텐션이 위치들을 가로질러 섞고, 잔차가 그것을 앞으로 실어 나르고, FFN이 변환하고, 정규화가 스트림을 안정되게 붙든다.

```figure
transformer-block
```

### 인코더 블록 (BERT, T5 인코더가 사용)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

인코더는 양방향이다. 마스킹 없음. 모든 위치가 모든 위치를 본다.

### 디코더 블록 (GPT, T5 디코더가 사용)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

디코더는 블록당 세 개의 하위 층(sublayer)을 가진다. 가운데 것(크로스 어텐션)이 정보가 인코더에서 디코더로 흐르는 유일한 곳이다. 순수 디코더 전용 아키텍처(GPT)에서는 크로스 어텐션이 생략되고 마스킹된 셀프 어텐션 + FFN만 있다.

### Pre-norm vs post-norm

원 논문: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm은 2019년경 인기가 식었다. 세심한 워밍업 없이 깊게 학습하기가 더 어렵기 때문이다. Pre-norm(하위 층 *전*에 `LN`)이 2026년 기본값이다. Llama, Qwen, GPT-3+, Mistral 모두 이를 쓴다.

### 2026년 현대화된 블록

Vaswani 2017은 LayerNorm + ReLU를 출시했다. 현대 스택은 둘 다 교체했다. 프로덕션(production) 블록이 실제로 어떻게 생겼는지:

| 구성요소 | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm은 LayerNorm의 평균 중심화를 버린다(뺄셈 하나 줄임). 그래서 계산을 아끼면서도 경험적으로 최소한 동등하게 안정적이다. SwiGLU(`Swish(W1 x) ⊙ W3 x`)는 Llama, PaLM, Qwen 논문에서 ReLU/GELU FFN을 일관되게 약 0.5포인트 ppl 능가한다.

### 파라미터 수

`d_model = d`이고 FFN 확장 `r`인 하나의 블록에 대해:

- MHA: `4 · d²` (Q, K, V, O 투영)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norm: 무시할 만함

`d = 4096, r = 2.6, layers = 32`(대략 Llama 3 8B)에서, 총: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(임베딩과 헤드 포함). 발표된 수치와 일치한다.

## 직접 만들기 (Build It)

### 1단계: 빌딩 블록

레슨 03의 작은 `Matrix` 클래스를 사용한다(독립성을 위해 이 파일에 복사됨):

- `layer_norm(x, eps=1e-5)`: 평균을 빼고 표준편차로 나눈다.
- `rms_norm(x, eps=1e-6)`: RMS로 나눈다. 평균 뺄셈 없음.
- `gelu(x)`와 `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)`와 `decoder_block(x, enc_out, params)`.

전체 연결은 `code/main.py`를 참조하라.

### 2단계: 2층 인코더와 2층 디코더 연결하기

이들을 쌓는다. 인코더 출력을 모든 디코더 크로스 어텐션에 넘긴다. 출력 투영 전에 최종 LN을 더한다.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 3단계: 장난감 예제로 순방향 실행하기

6 토큰 소스와 5 토큰 타깃을 통과시킨다. 출력 모양이 `(5, vocab)`인지 검증한다. 학습(training) 없음: 이 레슨은 손실(loss)이 아니라 아키텍처에 관한 것이다.

### 4단계: RMSNorm + SwiGLU로 교체하기

LayerNorm과 ReLU-FFN을 RMSNorm과 SwiGLU로 교체한다. 모양이 여전히 일치하는지 확인한다. 이것이 함수 하나 치환으로 이루어지는 2026년 현대화다.

## 라이브러리로 써보기 (Use It)

PyTorch/TF 레퍼런스 구현: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. 하지만 대부분의 2026년 프로덕션 코드는 자체 블록을 만든다. 이유는:

- Flash Attention은 `nn.MultiheadAttention`이 아니라 어텐션 내부에서 호출된다.
- GQA / MLA는 표준 라이브러리 레퍼런스에 없다.
- RoPE, RMSNorm, SwiGLU는 PyTorch 기본값이 아니다.

HF `transformers`에는 읽어볼 만한 깔끔한 레퍼런스 블록이 있다. `modeling_llama.py`가 정전(canonical)적인 2026년 디코더 전용 블록이다. 약 500줄이고 한 번 훑어볼 가치가 있다.

**인코더 vs 디코더 vs 인코더-디코더: 언제 고를까:**

| 필요 | 선택 | 예시 |
|------|------|---------|
| 분류, 임베딩, 텍스트 QA | 인코더 전용 | BERT, DeBERTa, ModernBERT |
| 텍스트 생성, 챗, 코드, 추론 | 디코더 전용 | GPT, Llama, Claude, Qwen |
| 구조화된 입력 → 구조화된 출력 (번역, 요약) | 인코더-디코더 | T5, BART, Whisper |

디코더 전용이 언어에서 이긴 이유는 가장 깔끔하게 확장되고 이해와 생성을 둘 다 다루기 때문이다. 인코더-디코더는 입력이 명확한 "소스 시퀀스" 정체성을 가질 때(번역, 음성 인식, 구조화된 과제) 여전히 최선이다.

## 산출물 (Ship It)

`outputs/skill-transformer-block-reviewer.md`를 참조하라. 이 스킬은 새 트랜스포머 블록 구현을 2026년 기본값에 비추어 검토하고 빠진 조각(pre-norm, RoPE, RMSNorm, GQA, FFN 확장비)을 표시한다.

## 연습 문제 (Exercises)

1. **쉬움.** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`에서 encoder_block의 파라미터를 센다. 블록을 구현하고 `sum(p.numel() for p in block.parameters())`를 사용해 검증한다.
2. **보통.** post-norm에서 pre-norm으로 전환한다. 둘 다 초기화하고 무작위 입력에 대해 12개 적층 층 후의 활성값 노름(norm)을 측정한다. post-norm의 활성값은 폭발해야 하고, pre-norm의 것은 유계로 유지되어야 한다.
3. **어려움.** 장난감 복사 과제(역순으로 `x` 복사)에 4층 인코더-디코더를 구현한다. 100 스텝 학습한다. 손실을 보고한다. RMSNorm + SwiGLU + RoPE로 교체한다. 손실이 떨어지는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Block | "트랜스포머 층 하나" | 잔차 연결로 감싸인 norm + 어텐션 + norm + FFN의 스택. |
| Residual | "스킵 연결" | `x + f(x)` 출력. 깊은 스택을 통한 그래디언트 흐름을 가능하게 함. |
| Pre-norm | "뒤가 아니라 앞에서 정규화" | 현대: `x + sublayer(LN(x))`. 워밍업 곡예 없이 더 깊게 학습. |
| RMSNorm | "평균 없는 LayerNorm" | RMS로 나눔. 연산 하나 적고, 경험적 안정성 동일. |
| SwiGLU | "모두가 갈아탄 FFN" | `Swish(W1 x) ⊙ W3 x → W2`. LM ppl에서 ReLU/GELU를 이김. |
| Cross-attention | "디코더가 인코더를 보는 법" | Q는 디코더, K/V는 인코더 출력에서 오는 MHA. |
| FFN expansion | "가운데 MLP가 얼마나 넓은가" | 은닉 크기 대 d_model의 비율. 보통 4(LayerNorm) 또는 2.6(SwiGLU). |
| Bias-free | "+b 항을 버림" | 현대 스택은 선형 층에서 편향을 생략함. 약간의 ppl 개선, 더 작은 모델. |

## 더 읽을거리 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762): 원조 블록 사양.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745): pre-norm이 깊게 post-norm을 이기는 이유.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): SwiGLU 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py): 정전적 2026년 디코더 전용 블록.
