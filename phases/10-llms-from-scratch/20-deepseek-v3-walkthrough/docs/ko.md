# DeepSeek-V3 Architecture Walkthrough

> Phase 10 · 레슨 14는 모든 오픈 모델이 돌리는 여섯 개의 아키텍처 손잡이를 명명했다. DeepSeek-V3(2024년 12월, 총 671B 파라미터(parameter), 활성 37B)는 여섯 개를 모두 돌리고 네 개를 더한다: Multi-Head Latent Attention, 보조 손실 없는(auxiliary-loss-free) 부하 균형, Multi-Token Prediction, DualPipe 학습이다. 이 레슨은 DeepSeek-V3의 아키텍처를 위에서 아래로 읽고 공개된 config에서 모든 파라미터 개수를 유도한다. 끝날 무렵이면 671B/37B 비율이 올바른 베팅인 이유와, MLA와 MoE를 함께 쓸 때 프런티어에서 둘 중 하나만 쓸 때보다 나은 이유를 설명할 수 있다.

**Type:** Learn
**Languages:** Python (stdlib, 파라미터 계산기)
**Prerequisites:** Phase 10 · 14 (오픈 모델 안내), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- DeepSeek-V3 config를 위에서 아래로 읽고 여섯 개의 GPT-2 손잡이 더하기 네 개의 DeepSeek 고유 추가의 관점에서 각 필드를 설명한다.
- 총 파라미터 개수(671B), 활성 파라미터 개수(37B), 그리고 각각에 기여하는 구성 요소를 유도한다.
- 128k 컨텍스트에서 MLA의 KV 캐시 풋프린트를 계산하고, GQA를 가진 동일 활성 파라미터 밀집 모델이 지불할 비용과 비교한다.
- 네 개의 DeepSeek 고유 혁신(MLA, MTP, 보조 손실 없는 라우팅, DualPipe)을 진술하고, 각각이 아키텍처/학습 스택의 어느 부분을 겨냥하는지 명명한다.

## 문제 (The Problem)

DeepSeek-V3는 그 아키텍처가 Llama 계열과 의미 있게 다른 첫 프런티어 오픈 모델이다. Llama 3 405B는 "여섯 손잡이를 돌린 GPT-2"다. DeepSeek-V3는 여섯 손잡이를 모두 돌리고 네 개를 더한 GPT-2다. Llama 3 config를 읽는 것은 DeepSeek config를 읽기 위한 준비운동이지만, 깊은 구조(어텐션(attention) 블록의 형태, 라우팅 로직, 학습 시점 목표)는 별도의 안내가 필요할 만큼 다르다.

이를 배우는 보상은 이렇다. DeepSeek-V3의 오픈 가중치(weight) 공개는 오픈 모델에서 "프런티어 능력"이 의미하는 바를 바꿔놓았다. 그 아키텍처는 2026년의 여러 학습 실행이 베끼고 있는 청사진이다. 이를 이해하는 것은 프런티어 LLM 학습이나 추론을 다루는 어떤 역할에서든 기본 자격이다.

## 개념 (The Concept)

### 불변의 핵심, 다시

DeepSeek-V3는 여전히 자기회귀(autoregressive)다. 여전히 디코더(decoder) 블록을 쌓는다. 각 블록은 여전히 어텐션 더하기 MLP 더하기 두 개의 RMSNorm을 가진다. 여전히 MLP에서 SwiGLU를 쓴다. 여전히 RoPE를 쓴다. Pre-norm. 가중치가 묶인(weight-tied) 임베딩(embedding). 모든 Llama나 Mistral과 동일한 베이스라인(baseline)이다.

### 반전: GQA 대신 MLA

Phase 10 · 14에서 GQA가 K와 V를 Q 헤드 그룹에 걸쳐 공유하여 KV 캐시를 줄임을 알았다. Multi-Head Latent Attention(MLA)은 더 나아간다: K와 V를 공유된 저랭크 잠재 표현(`kv_lora_rank`)으로 압축한 다음, 즉석에서 헤드별로 압축 해제한다. KV 캐시는 잠재 표현만 저장한다. 보통 8 x 128 = 1024 부동소수점이 아니라 토큰당 층(layer)당 512 부동소수점이다.

128k 컨텍스트에서 MLA를 가진 DeepSeek-V3(토큰당 층당 하나의 공유 잠재 `c^{KV}`; K와 V는 둘 다 후속 행렬곱에 흡수될 수 있는 업투영(up-projection)을 통해 이 잠재 표현에서 유도된다):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

가상의 GQA 베이스라인(Llama 3 70B 형태, 8 KV 헤드, 헤드 차원 128)은 다음을 지불한다:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA는 128k 컨텍스트에서 Llama-3-70B 스타일 GQA 캐시보다 4배 작다.

트레이드오프(trade-off): MLA는 어텐션 연산당(헤드당) 압축 해제 단계를 추가한다. 추가 연산은 절약된 대역폭에 비해 작다. 장문 컨텍스트(long-context) 추론에 순이득이다.

### 라우팅: 보조 손실 없는 부하 균형

MoE 라우터는 어느 top-k 전문가(expert)가 각 토큰을 처리할지 결정한다. 순진한 라우터는 너무 많은 일을 소수의 전문가에 집중시켜 나머지를 유휴 상태로 둔다. 표준 해법: 부하 불균형에 페널티를 주는 보조 손실(loss) 항을 추가한다. 이는 동작하지만 본 과제 성능을 약간 저하시킨다.

DeepSeek-V3는 보조 손실 없는 방식을 도입한다. 전문가별 편향(bias) 항이 라우터 로짓(logit)에 추가되고, 학습 중 단순한 규칙으로 조정된다: 전문가 `e`가 과부하면 `bias_e`를 줄이고, 부하가 적으면 늘린다. 추가 손실 항 없음. 학습은 깨끗하게 유지된다. 전문가 부하는 균형 잡힌 채로 유지된다.

본 손실에 대한 효과: 측정 가능한 것 없음. MoE 아키텍처에 대한 효과: 더 깔끔함, 튜닝할 보조 손실 하이퍼파라미터(hyperparameter) 없음.

### MTP: 더 밀집된 학습 + 공짜 드래프트

Phase 10 · 18에서 DeepSeek-V3가 두 위치 앞의 토큰을 예측하는 D=1 MTP 모듈을 추가함을 알았다. 추론 시, 학습된 모듈은 80% 이상의 수락률(acceptance)을 가진 추측 디코딩(speculative-decoding) 드래프트로 재활용된다. 학습 시, 각 은닉 상태(hidden state)는 D+1 = 2개의 타겟에서 감독되어 더 밀집된 신호를 제공한다.

파라미터: 671B 본 모델 위에 14B. 오버헤드: 2.1%.

### 학습: DualPipe

Phase 10 · 19에서 DualPipe가 순방향과 역방향 청크(chunk)를 노드 간 all-to-all 통신과 겹치는 양방향 파이프라인(pipeline)임을 알았다. DeepSeek-V3의 2,048-H800 규모에서, 1F1B였다면 파이프라인 버블(bubble)에 잃었을 대략 245k GPU-시간을 회복한다.

### config, 필드별로

여기 DeepSeek-V3 config(단순화)가 있다:

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

이를 파싱하면:

- `hidden_size=7168`: 임베딩 차원.
- `num_hidden_layers=61`: 총 블록 깊이.
- `first_k_dense_layers=3`: 첫 3개 블록은 크기 18432의 밀집 MLP를 쓴다. 나머지 58개는 MoE를 쓴다.
- `num_attention_heads=128`: 128개 쿼리(query) 헤드.
- `kv_lora_rank=512`: K와 V가 이 잠재 차원으로 압축되고 헤드별로 압축 해제된다.
- `num_experts=256, num_experts_per_tok=8`: 각 MoE 블록은 256개 전문가를 가지며, top-8을 라우팅한다.
- `shared_experts=1`: 256개의 라우팅된 전문가 위에, 1개의 항상 켜진 전문가가 모든 토큰에 기여한다. 모든 토큰이 신뢰할 만한 무언가를 받게 보장하는 "밀집 바닥"으로 생각하면 된다.
- `moe_intermediate_size=2048`: 각 전문가의 MLP 은닉 크기. 256개가 있으므로 밀집 MLP보다 작다.

### 파라미터 계산

전체 계산은 `code/main.py`에 있다. 헤드라인:

- 임베딩: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 첫 3개 밀집 블록: MLA를 가진 어텐션(블록당 ~144M) + 밀집 MLP(블록당 ~260M) + norm. 총 약 1.2B.
- 58개 MoE 블록: MLA를 가진 어텐션(~144M) + 각각 256개 전문가(개당 30M) + 1개 공유 전문가(30M) + norm. 모든 전문가를 포함해 블록당 총 ~7.95B. 58개 MoE 블록에 대해 총 461B.
- MTP 모듈: 14B.

총합: 핵심 아키텍처에 대해 ~476B + MTP 14B다. 공개된 671B 수치는 여기에 더해 추가적인 구조적 파라미터(편향 텐서(tensor), 전문가별 구성 요소, 공유 전문가 스케일링 등)를 설명한다. 계산기에서 우리가 재현하는 수치는 공개된 것의 3~5% 이내다. 차이는 DeepSeek 보고서가 그 2절 부록에 문서화한 세밀한 계산에서 온다.

순방향당 활성 파라미터:

- 어텐션: 층당 144M * 61 = 8.8B (모든 층이 발화).
- 활성 MLP: 첫 3개 층 밀집(3 * 260M = 780M), 58개 MoE 층 각각 8개 라우팅 + 1개 공유 + 라우팅 오버헤드로 활성. 층당 활성 MLP: ~260M. 총: 3 * 260M + 58 * 260M = ~15.9B.
- 임베딩 + norm: 1.2B.
- 총 활성: 대략 핵심 26B + MTP 14B(학습되지만 추론에서 항상 실행되지는 않음) ≈ 37B.

### 671B / 37B 비율

18배 희소성 비율(활성 파라미터가 총합의 5.5%). DeepSeek-V3는 오픈 가중치를 출시한 가장 희소한 프런티어 MoE 모델이다. 비율 13/47(28%)의 Mixtral 8x7B는 훨씬 밀집하다. 비율 17B/400B(4.25%)의 Llama 4 Maverick은 비슷하다. DeepSeek의 베팅: 프런티어 규모에서, 더 많은 전문가와 더 낮은 활성 비율이 활성 FLOP당 더 나은 품질을 낸다.

### DeepSeek-V3의 위치

| 모델 | 총합 | 활성 | 비율 | 어텐션 | 새로운 아이디어 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 확장 |

### 후속: R1, V4

DeepSeek-R1(2025)은 V3 백본(backbone)에 대한 추론 학습 실행이다. R1은 동일한 아키텍처를 쓴다. 바뀐 것은 사후 학습(post-training) 레시피(검증 가능한 과제에 대한 대규모 RL)이지 사전 학습(pretraining) 아키텍처가 아니다.

DeepSeek-V4(만약 출시된다면)는 MLA + MoE + MTP를 유지하고 Phase 10 · 17의 NSA를 잇는 DSA(DeepSeek Sparse Attention)를 추가할 것으로 예상된다. 계보는 안정적이다. 아키텍처 수준의 혁신이 누적되고, 각 버전이 손잡이를 하나씩 더 돌린다.

```figure
moe-routing
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 DeepSeek-V3의 형태에 특화된 파라미터 계산기다. 실행하여 그 출력을 논문의 수치와 비교하고, 가상의 변형(전문가 256 대 512, top-8 대 top-16, MLA 랭크 512 대 1024)에 사용한다.

볼 것:

- 공개된 671B 대비 총 파라미터 개수.
- 공개된 37B 대비 활성 파라미터 개수.
- 128k 컨텍스트에서의 KV 캐시: MLA 대 GQA 비교.
- 파라미터 예산이 실제로 어디로 가는지 보기 위한 층별 분해.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-deepseek-v3-reader.md`를 산출한다. DeepSeek 계열 모델(V3, R1, 또는 미래의 어떤 변형)이 주어지면, config의 각 필드를 명명하고 구성 요소별로 파라미터 개수를 유도하며 모델이 네 개의 DeepSeek 고유 혁신 중 어느 것을 쓰는지 식별하는 구성 요소별 아키텍처 독해를 산출한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행한다. 계산기의 총 파라미터 추정치를 공개된 671B와 비교하고 차이가 어디서 오는지 식별한다. 논문의 2절에 전체 항목별 명세가 있다.

2. config를 512 대신 MLA 랭크 256을 쓰도록 수정한다. 128k 컨텍스트에서 결과 KV 캐시 크기를 계산한다. 어떤 비율의 감소를 사들이며, 헤드별 표현력에 어떤 비용이 드는가?

3. DeepSeek-V3의 (256 전문가, top-8) 라우팅을 가상의 (512 전문가, top-8) 변형과 비교한다. 총 파라미터는 늘고; 활성 파라미터는 그대로다. 추가 전문가 용량은 이론상 무엇을 사들이며, 추론에서 무엇을 비용으로 치르는가?

4. DeepSeek-V3 기술 보고서(arXiv:2412.19437)의 2.1절 MLA를 읽는다. K와 V 압축 해제 행렬이 추론 시점 효율을 위해 후속 행렬곱에 "흡수"될 수 있는 이유를 세 문장으로 설명한다.

5. DeepSeek-V3는 대부분의 연산에 FP8 학습을 쓴다. 671B 가중치를 저장할 때 BF16 대비 FP8의 메모리 절감을 계산한다. 이것이 14.8T 토큰 학습 예산과 어떻게 교차하는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | K와 V를 공유된 저랭크 잠재(kv_lora_rank, 보통 512)로 압축하고 헤드별로 즉석에서 압축 해제; KV 캐시는 잠재 표현만 저장 |
| kv_lora_rank | "MLA 압축 차원" | K와 V를 위한 공유 잠재의 크기; DeepSeek-V3는 512를 쓴다 |
| 첫 k개 밀집 층(First k dense layers) | "초기 층은 밀집을 유지" | 첫 몇 개 MoE 모델 층이 MoE 라우터를 건너뛰고 안정성을 위해 밀집 MLP를 실행한다 |
| num_experts_per_tok | "Top-k 라우팅" | 토큰당 몇 개의 라우팅된 전문가가 발화하는지; DeepSeek-V3는 8을 쓴다 |
| 공유 전문가(Shared experts) | "항상 켜진 전문가" | 라우팅과 무관하게 모든 토큰을 처리하는 전문가; DeepSeek-V3는 1을 쓴다 |
| 보조 손실 없는 라우팅(Auxiliary-loss-free routing) | "편향 조정 부하 균형" | 손실 항을 추가하지 않고 전문가 부하를 균형 잡기 위해 학습 중 조정되는 전문가별 편향 항 |
| MTP 모듈(MTP module) | "추가 예측 헤드" | h^(1)와 E(t+1)로부터 t+2를 예측하는 트랜스포머(transformer) 블록; 더 밀집된 학습, 공짜 추측 디코딩 드래프트 |
| DualPipe | "양방향 파이프라인" | 순방향/역방향 연산을 노드 간 all-to-all과 겹치는 학습 스케줄 |
| 활성 파라미터 비율(Active parameter ratio) | "희소성" | active_params / total_params; DeepSeek-V3는 5.5%에 도달 |
| FP8 학습(FP8 training) | "8비트 학습" | FP8에서의 학습 저장 및 많은 연산 작업; 작은 품질 비용으로 BF16 대비 메모리를 대략 절반으로 줄인다 |

## 더 읽을거리 (Further Reading)

- [DeepSeek-AI: DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) 전체 아키텍처, 학습, 결과 문서
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3): config 파일과 배포(deployment) 메모
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434): MLA를 도입한 전신
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948): V3의 아키텍처에 대한 추론 학습 후속
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089): DeepSeek 계열 어텐션의 미래 방향
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe): 학습 스케줄 참조
