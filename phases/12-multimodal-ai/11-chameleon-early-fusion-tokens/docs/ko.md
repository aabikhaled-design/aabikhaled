# Chameleon과 초기 융합(Early-Fusion) 토큰 전용 멀티모달 모델

> 지금까지 본 모든 VLM은 이미지와 텍스트를 분리해 둔다. 시각 토큰은 비전 인코더(vision encoder)에서 나와 프로젝터(projector)로 흘러간 뒤 LLM 안에서 텍스트와 만난다. 비전 어휘와 텍스트 어휘는 절대 겹치지 않는다. Chameleon(Meta, 2024년 5월)은 물었다. 만약 겹친다면? 이미지를 공유 어휘(shared vocabulary)에서 나온 이산 토큰(discrete token)의 시퀀스로 바꾸는 VQ-VAE를 학습시킨다. 이제 모든 멀티모달 문서는 하나의 시퀀스다. 텍스트 토큰과 이미지 토큰이 인터리브(interleave)되고, 단일 자기회귀 손실(autoregressive loss)을 쓴다. 부수 효과로, 모델은 혼합 모달리티 출력을 생성한다. 단일 추론(inference) 호출에서 텍스트와 이미지 토큰을 번갈아 낸다. 이 레슨은 초기 융합 논제를 읽고 장난감 버전을 처음부터 끝까지 만든다.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 공유 어휘 + 단일 손실이 모델이 할 수 있는 일을 왜 바꾸는지 설명하기.
- VQ-VAE가 이미지를 트랜스포머(transformer)의 다음 토큰 목표와 호환되는 이산 시퀀스로 어떻게 토큰화하는지 기술하기.
- Chameleon의 학습 안정화 기법인 QK-Norm, 드롭아웃(dropout) 배치, LayerNorm 순서를 대기.
- Chameleon과 BLIP-2의 Q-Former 접근법을 비교하고 각각이 옳은 선택인 경우를 기술하기.

## 문제 (The Problem)

어댑터 기반 VLM(LLaVA, BLIP-2, Qwen-VL)은 텍스트와 이미지를 서로 다른 두 가지로 취급한다. 텍스트 토큰은 `embed(text_token)`을 거치고, 이미지는 `visual_encoder(image) → projector → ... pseudo_tokens`를 거친다. 모델에는 도중에 합쳐지는 두 개의 입력 경로가 있다.

세 가지 결과가 나온다.

1. LLM은 이미지를 소비만 할 수 있고 내보내지는 못한다. 출력은 텍스트뿐이다.
2. 혼합 모달리티 문서(기사처럼 문단과 이미지가 번갈아 나오는 것)는 다루기 어색하다. 멀티모달 입력을 모델 밖에서 파싱하거나 생성을 연쇄해야 한다.
3. 분포 불일치. 시각 토큰과 텍스트 토큰은 은닉 공간(hidden space)의 서로 다른 영역에 존재하며, 미묘한 정렬 문제를 만든다.

Chameleon은 그 전제를 거부한다. 이미지는 그저 공유 어휘에서 나온 이산 토큰의 시퀀스일 뿐이다. 인터리브된 문서로 모델을 학습시키면, 하나의 손실, 하나의 자기회귀 디코더(decoder)로 혼합 모달리티 생성을 공짜로 얻는다.

## 개념 (The Concept)

### 이미지 토크나이저로서의 VQ-VAE

토크나이저(tokenizer)는 벡터 양자화 변분 오토인코더(vector-quantized variational autoencoder)다. 아키텍처는 다음과 같다.

- 인코더(encoder): 이미지를 공간 특징 맵(spatial feature map)으로 매핑하는 CNN + ViT. 예를 들어 차원 256짜리 32x32 특징.
- 코드북(codebook): 학습된 K개 벡터의 어휘(Chameleon은 8192개 사용), 역시 차원 256.
- 양자화(quantization): 각 공간 특징에 대해 L2 거리로 가장 가까운 코드북 항목을 찾는다. 연속 특징을 정수 인덱스로 대체한다.
- 디코더(decoder): 양자화된 특징을 다시 픽셀로 되돌리는 CNN.

학습: VAE 재구성 손실 + 커밋먼트 손실(commitment loss) + 코드북 손실. 코드북 인덱스는 이미지를 위한 이산 알파벳을 이룬다.

Chameleon의 경우: 이미지 하나가 8192 어휘에서 뽑힌 32*32 = 1024개 토큰이 된다. 텍스트 토큰(LLM의 BPE 어휘, 예를 들어 32000개)과 이어 붙인다. 최종 어휘: 40192. 트랜스포머는 하나의 시퀀스, 하나의 손실을 본다.

### 공유 어휘

Chameleon의 어휘는 텍스트 토큰, 이미지 토큰, 모달리티 구분자를 결합한다. 각 토큰에는 단일 ID가 있다. 입력 임베딩(embedding) 층(layer)은 모든 ID를 D차원 은닉 벡터로 매핑한다. 출력 투영은 은닉을 다시 어휘 로짓(logit)으로 매핑한다. 소프트맥스(softmax)가 모달리티에 상관없이 다음 토큰을 고른다.

구분자가 중요하다. `<image>`와 `</image>` 태그가 이미지 토큰 시퀀스를 감싼다. 생성 시점에 모델이 `<image>`를 내면, 다운스트림 소프트웨어는 다음 1024개 토큰이 픽셀 렌더링을 위해 디코더로 보낼 VQ 인덱스임을 안다.

### 혼합 모달리티 생성

추론은 공유 어휘에서의 다음 토큰 예측이다. 예시 프롬프트(prompt): "고양이를 그리고 묘사하라." Chameleon은 다음을 낸다.

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

모델은 순서를 자율적으로 고른다. 이미지를 먼저 낼 수도, 텍스트를 먼저 낼 수도, 인터리브할 수도 있다. 같은 디코더, 같은 손실이다.

생성이 텍스트 전용인 어댑터 VLM과 비교하라. Chameleon은 모델 출력 모달리티에 관한 질문을 다시 연다.

### 학습 안정성: QK-Norm, 드롭아웃, LayerNorm 순서

초기 융합 학습은 대규모에서 불안정하다. Chameleon 논문은 세 가지 기법을 기록한다.

- QK-Norm. 어텐션(attention) 내부에서 내적(dot product) 전에 쿼리와 키 투영에 LayerNorm을 적용한다. 깊은 곳에서 로짓 크기 폭발을 막는다. 2024년 이후 여러 대형 모델이 사용한다.
- 드롭아웃 배치. 어텐션과 MLP 뒤뿐 아니라 모든 잔차 덧셈(residual-add) 뒤에 드롭아웃을 둔다. 이미지 토큰에서 오는 그래디언트(gradient)가 지배할 수 있을 때 더 많은 정규화(regularization)가 필요하다.
- LayerNorm 순서. 잔차 분기에 Pre-LN(표준), 더해 마지막 블록의 스킵 연결(skip connection)에 추가 LN. 최종 층 그래디언트 흐름을 안정화한다.

이 기법들이 없으면 34B 파라미터(parameter) Chameleon 학습은 여러 체크포인트에서 발산했다. 기법들을 넣으면 수렴(convergence)한다. 학습 레시피는 아키텍처만큼이나 큰 기여다.

### 토크나이저의 재구성 한계

VQ-VAE는 손실이 있다. 코드북 항목 8192개와 512x512 이미지당 1024개 토큰에서, 재구성 PSNR은 약 26~28 dB에서 한계에 부딪힌다. 알아볼 수 있는 이미지 생성에는 충분하지만 연속 공간 디퓨전(diffusion)(Stable Diffusion 3은 32 dB 이상 달성)보다는 눈에 띄게 나쁘다.

토크나이저가 병목이다. 더 나은 토크나이저(MAGVIT-v2, IBQ, SBER-MoVQGAN)는 한계를 끌어올린다. Emu3(레슨 12.12)는 더 나은 토크나이저만으로 SDXL 수준의 생성을 달성한다.

### Chameleon 대 BLIP-2 / LLaVA

Chameleon (초기 융합, 공유 어휘):
- 하나의 손실, 하나의 디코더.
- 혼합 모달리티 출력을 생성한다.
- 토크나이저가 품질 한계다.
- 비싸다: 추론 경로에서 생성된 이미지마다 VQ-VAE 디코더가 필요하다.

BLIP-2 / LLaVA (후기 융합, 분리된 타워):
- 비전 입력, 텍스트 출력만.
- 사전 학습된 LLM을 재사용한다.
- 이해에는 토크나이저 병목이 없다.
- 싸다: 단일 순방향 패스(forward pass).

태스크로 선택하라. 이미지 생성이 필요하면 Chameleon 계열. 이해만 필요하면 어댑터 VLM이 더 단순하고 더 많은 사전 학습 연산량을 재사용한다.

### Fuyu와 AnyGPT

Fuyu(Adept, 2023)는 관련 접근법이다. 별도 비전 인코더를 아예 건너뛰고, 원본 이미지 패치(patch)를 토큰인 것처럼 LLM의 입력 투영을 통해 먹인다. 토크나이저가 없다. Chameleon보다 단순하지만 공유 어휘 출력 생성을 잃는다.

AnyGPT(Zhan et al., 2024)는 Chameleon을 네 가지 모달리티로 확장한다. 텍스트, 이미지, 음성, 음악. 각각에 동일한 VQ-VAE 기법, 공유 트랜스포머. 임의-대-임의(any-to-any) 생성. 레슨 12.16에서 더 다룬다.

```figure
vq-codebook
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 수준의 처음부터 끝까지 초기 융합 모델을 만든다.

- 8x8 패치를 코드북 인덱스(K=16)로 매핑하는 작은 VQ-VAE 스타일 양자화기.
- (텍스트 id 0..31) + (이미지 id 32..47) + (구분자 48, 49)로 이루어진 공유 어휘.
- 합성 캡션 + 이미지 토큰 시퀀스로 학습한 장난감 자기회귀 디코더(바이그램 테이블).
- 프롬프트가 주어지면 텍스트 + 이미지 토큰을 번갈아 내는 샘플링(sampling) 루프.

코드는 의도적으로 트랜스포머를 작게(바이그램) 유지하므로 신호 흐름을 처음부터 끝까지 추적할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-tokenizer-vs-adapter-picker.md`를 만든다. 제품 명세(이해만 vs 이해 + 생성, 요구 이미지 품질, 비용 예산)가 주어지면, Chameleon 계열(초기 융합)과 LLaVA 계열(후기 융합) 사이에서 고르고 정량적 경험칙으로 정당화한다.

## 연습 문제 (Exercises)

1. Chameleon은 K=8192 코드북 항목과 512x512 이미지당 1024개 토큰을 사용한다. 24비트 RGB 이미지 대비 압축비를 추정하라. 손실이 있는가? 얼마나 손실이 있는가?

2. 4K 이미지(3840x2160)를 동일한 VQ-VAE 밀도로 처리하면 이미지 토큰이 몇 개 나오는가? Chameleon 스타일 모델이 단일 추론 호출로 4K 이미지를 생성할 수 있는가? 무엇이 먼저 무너지는가: 컨텍스트, 토크나이저 품질, 아니면 KV 캐시?

3. 순수 Python으로 QK-Norm을 구현하라. 64차원 쿼리와 키가 주어졌을 때, LayerNorm 전후의 내적을 보여라. 왜 깊은 곳에서 크기 제어가 중요한가?

4. 학습 안정성에 관한 Chameleon 2.3절을 읽으라. QK-Norm 없이 34B에서 논문이 관찰한 정확한 실패 모드를 기술하라. "norm 폭발" 시그니처는 무엇이었나?

5. 텍스트 전용 프롬프트가 주어졌을 때 혼합 모달리티 응답을 내도록 장난감 디코더를 확장하라. 학습 데이터 분포가 60% 텍스트 먼저 / 40% 이미지 먼저일 때 모델이 얼마나 자주 이미지 먼저 대 텍스트 먼저를 고르는지 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 초기 융합(Early fusion) | "통합 토큰" | 첫 스텝부터 트랜스포머의 어휘를 공유하는 이산 토큰으로 변환된 이미지 |
| VQ-VAE | "이미지 토크나이저" | 이미지를 트랜스포머가 예측할 수 있는 정수 인덱스로 매핑하는 CNN + ViT + 코드북 |
| 공유 어휘(Shared vocabulary) | "하나의 사전" | 텍스트 + 이미지 + 모달리티 구분자를 아우르는 단일 토큰 ID 공간 |
| QK-Norm | "어텐션 안정화 장치" | 쿼리와 키에 내적 전에 적용하는 LayerNorm으로, norm 폭발을 막는다 |
| 혼합 모달리티 생성(Mixed-modality generation) | "텍스트 + 이미지 출력" | 단일 패스에서 인터리브된 텍스트와 이미지 토큰을 자율적으로 생성하는 추론 |
| 코드북 크기(Codebook size) | "K개 항목" | VQ-VAE가 양자화할 수 있는 이산 벡터의 수; 압축과 충실도를 트레이드오프한다 |
| 토크나이저 한계(Tokenizer ceiling) | "재구성 한계" | VQ 토큰을 디코딩해 달성 가능한 최고 PSNR; 모델의 이미지 품질을 제한한다 |

## 더 읽을거리 (Further Reading)

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
