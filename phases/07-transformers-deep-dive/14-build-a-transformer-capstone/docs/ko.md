# 밑바닥부터 트랜스포머 만들기: 캡스톤(Capstone)

> 열세 개의 레슨. 하나의 모델. 지름길 없음.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. 건너뛰지 마라.
**Time:** ~120분

## 문제 (The Problem)

논문은 다 읽었다. 어텐션(attention), 멀티헤드(multi-head) 분할, 위치 인코딩(positional encoding), 인코더(encoder)와 디코더(decoder) 블록, BERT와 GPT 손실(loss), MoE, KV 캐시를 구현했다. 이제 이것들을 실제 작업에서 함께 작동시켜라.

캡스톤(capstone): 작은 디코더 전용(decoder-only) 트랜스포머(transformer)를 문자 단위(character-level) 언어 모델링(language modeling) 작업에 대해 종단 간(end-to-end)으로 학습(training)한다. 셰익스피어를 읽는다. 새로운 셰익스피어를 생성한다. 노트북에서 10분 미만으로 학습할 만큼 작다. 더 큰 데이터셋(dataset)과 더 긴 학습으로 바꿔 넣으면 실제 LM이 될 만큼 정확하다.

이것이 이 과정의 "nanoGPT"다. 독창적이지 않다. Karpathy의 2023년 nanoGPT 튜토리얼은 모든 학생이 적어도 한 번은 작성하는 레퍼런스 구현이다. 우리는 그 형태를 가져와 우리가 다룬 내용에 맞춰 재구성한다.

## 개념 (The Concept)

![Transformer-from-scratch block diagram](../assets/capstone.svg)

주석을 단 아키텍처:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 우리가 만드는 것

- `GPTConfig`: 모든 하이퍼파라미터(hyperparameter)를 설정하는 한 곳.
- `MultiHeadAttention`: 인과(causal), 배치(batch) 처리, 선택적 Flash 스타일 경로(PyTorch의 `scaled_dot_product_attention`) 포함.
- `SwiGLUFFN`: 현대적 FFN.
- `Block`: 프리노름(pre-norm), 잔차(residual)로 감싼 어텐션 + FFN.
- `GPT`: 임베딩(embedding), 쌓은 블록, LM 헤드, generate().
- AdamW, 코사인 LR(학습률(learning rate)), 그래디언트 클리핑(gradient clipping)을 갖춘 학습 루프.
- 셰익스피어 텍스트에 대한 문자 단위 토크나이저(tokenizer).

### 우리가 만들지 않는 것

- RoPE: Lesson 04에서 개념적으로 구현했다. 여기서는 단순함을 위해 학습된 위치 임베딩(positional embedding)을 쓴다. 연습 문제에서 RoPE로 바꿔 넣으라고 한다.
- 생성 중 KV 캐시: 각 생성 스텝이 전체 접두사(prefix)에 대한 어텐션을 다시 계산한다. 더 느리지만 더 단순하다. 연습 문제에서 KV 캐시를 추가하라고 한다.
- Flash Attention: PyTorch 2.0+는 입력이 맞으면 자동 디스패치(dispatch)한다; 우리는 `F.scaled_dot_product_attention`을 쓴다.
- MoE: 블록당 단일 FFN. MoE는 Lesson 11에서 봤다.

### 목표 지표

Mac M2 노트북에서, `tinyshakespeare.txt`에 대해 2,000 스텝 학습한 4층, 4헤드, d_model=128 GPT:

- 학습 손실이 약 6분 만에 ~4.2(무작위)에서 ~1.5로 수렴(convergence)한다.
- 샘플링된 출력이 셰익스피어 형태로 보인다: 고풍스러운 단어, 줄바꿈, "ROMEO:" 같은 고유명사가 출현한다.
- 검증 손실(텍스트의 마지막 10%를 떼어 둔 것)이 학습 손실을 가깝게 따라간다; 이 크기/예산에서는 과적합(overfitting)이 없다.

```figure
n5-block-stack
```

## 직접 만들기 (Build It)

이 레슨은 PyTorch를 쓴다. `torch`를 설치하라(CPU 빌드로 충분). `code/main.py`를 참고하라. 스크립트는 다음을 처리한다:

- `tinyshakespeare.txt`가 없으면 다운로드(또는 로컬 사본 읽기).
- 바이트 단위 문자 토크나이저.
- 90/10 학습/검증 분할.
- 지원되는 하드웨어에서 bf16 autocast를 쓰는 학습 루프.
- 학습 완료 후 샘플링.

### 1단계: 데이터

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65개의 고유 문자. 작은 어휘(vocabulary). 4바이트 vocab_size에 들어간다. BPE도, 토크나이저 드라마도 없다.

### 2단계: 모델

`code/main.py`를 참고하라. 블록은 Lesson 05의 교과서다. 프리노름, RMSNorm, SwiGLU, 인과 MHA. 4/4/128의 파라미터(parameter) 수: ~800K.

### 3단계: 학습 루프

길이 256의 토큰 윈도우(window) 무작위 배치를 가져온다. 순방향(Forward). 한 칸 밀기(shift-by-one) 교차 엔트로피(cross-entropy). 역방향(Backward). AdamW 스텝. 로그. 반복.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 4단계: 샘플링

프롬프트가 주어지면, 반복적으로 순방향을 돌리고, top-p 로짓(logit)에서 샘플링(sampling)하고, 이어 붙이고, 계속한다. 500 토큰 후 멈춘다.

### 5단계: 출력 읽기

2,000 스텝 후:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

셰익스피어는 아니다. 하지만 셰익스피어 형태다. 파라미터 ~800K로, 노트북에서 6분 만에 거둔 분명한 성과다.

## 라이브러리로 써보기 (Use It)

이 캡스톤은 레퍼런스 아키텍처다. 이를 실제 무언가로 만들 세 가지 확장:

1. **토크나이저를 교체하라.** BPE를 써라(예: `tiktoken.get_encoding("cl100k_base")`). 어휘 크기가 65에서 ~50,000으로 뛴다. 모델 용량(capacity)을 그만큼 키워 보상해야 한다.
2. **더 큰 코퍼스(corpus)에 학습하라.** `OpenWebText`나 `fineweb-edu`(HuggingFace)를 써라. 단일 A100에서 125M 파라미터 GPT의 경우 10B 토큰에 ~24시간이 걸린다.
3. **RoPE + KV 캐시 + Flash Attention을 추가하라.** 아래 연습 문제가 각각을 안내한다.

이것은 유창한 영어를 생성하는 125M 파라미터 GPT로 마무리된다. 프런티어(frontier) 모델은 아니다. 하지만 Karpathy, EleutherAI, Allen Institute가 2026년에 연구 체크포인트를 학습하는 데 쓰는 코드 경로도 이와 같다. 그저 규모만 더 클 뿐이다.

## 산출물 (Ship It)

`outputs/skill-transformer-review.md`를 참고하라. 이 스킬은 이전 13개 레슨 전반에 걸쳐 정확성을 위해 밑바닥부터 만든 트랜스포머 구현을 검토한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 학습한 모델의 마지막 스텝 검증 손실이 2.0 미만인지 확인하라. `max_steps`를 2,000에서 5,000으로 바꿔라. 검증 손실이 계속 향상되는가?
2. **중간.** 학습된 위치 임베딩을 RoPE로 교체하라. `MultiHeadAttention` 내부에서 Q와 K에 회전(rotation)을 적용하라. 학습하고 검증 손실이 최소한 그만큼 낮은지 확인하라.
3. **중간.** 샘플링 루프에 KV 캐시를 구현하라. 캐시 있을 때와 없을 때 500 토큰을 생성하라. 노트북에서 실측 시간(wall-clock)이 5~20배 향상되어야 한다.
4. **어려움.** 다음의 다음 토큰을 예측하는 두 번째 헤드(MTP: DeepSeek-V3의 다중 토큰 예측(Multi-Token Prediction))를 모델에 추가하라. 공동으로 학습하라. 도움이 되는가?
5. **어려움.** 블록당 단일 FFN을 4전문가 MoE로 교체하라. 라우터(router) + top-2 라우팅. 동일 활성 파라미터에서 검증 손실이 어떻게 바뀌는지 보라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy의 튜토리얼 repo" | 최소한의 디코더 전용 트랜스포머 학습 코드, ~300 LOC; 표준 레퍼런스. |
| tinyshakespeare | "표준 장난감 코퍼스" | ~1.1 MB의 텍스트; 2015년 이후 모든 문자-LM 튜토리얼이 쓴다. |
| 묶인 임베딩(Tied embeddings) | "입력/출력 행렬 공유" | LM 헤드 가중치(weight) = 토큰 임베딩 행렬의 전치(transpose); 파라미터를 아끼고 품질을 개선한다. |
| bf16 autocast | "학습 정밀도 트릭" | 순방향/역방향을 bf16으로 실행하고, 옵티마이저(optimizer) 상태는 fp32로 유지; 2021년 이후 표준. |
| 그래디언트 클리핑(Gradient clipping) | "급등을 멈춤" | 전역 그래디언트(gradient) 노름(norm)을 1.0으로 제한; 학습 폭발을 방지한다. |
| 코사인 LR 스케줄(Cosine LR schedule) | "2020년 이후 기본값" | LR이 선형으로 올라간 뒤(워밍업(warmup)) 코사인 형태로 정점의 10%까지 감쇠한다. |
| MFU | "Model FLOP Utilization" | 달성 FLOPs / 이론적 정점; 2026년에는 밀집(dense) 40%, MoE 30%면 강하다. |
| 검증 손실(Val loss) | "떼어 둔 손실" | 모델이 본 적 없는 데이터에 대한 교차 엔트로피; 과적합 탐지기. |

## 더 읽을거리 (Further Reading)

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/): 고전적인 주석 달린 구현.
