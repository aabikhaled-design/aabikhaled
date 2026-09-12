# 비전 트랜스포머와 패치-토큰 원시 단위(Vision Transformers and the Patch-Token Primitive)

> 멀티모달(multimodal)을 다루기 전에, 이미지는 먼저 트랜스포머(transformer)가 받아들일 수 있는 토큰(token) 시퀀스가 되어야 한다. 2020년 ViT 논문은 이 문제를 16x16 픽셀 패치(patch), 선형 투영(linear projection), 위치 임베딩(position embedding)으로 답했다. 5년이 지난 지금도 모든 2026년 프런티어 모델(네이티브 2576px의 Claude Opus 4.7, Gemini 3.1 Pro, Qwen3.5-Omni)은 여전히 이렇게 시작한다. 인코더(encoder)는 ViT에서 DINOv2로, 다시 SigLIP 2로 바뀌었고, 레지스터 토큰(register token)이 추가되었으며 위치 인코딩 방식은 2D-RoPE가 되었지만 그 원시 단위(primitive)는 그대로 유지되었다. 이 레슨은 패치-토큰 파이프라인(pipeline)을 처음부터 끝까지 읽고, 표준 라이브러리 Python으로 직접 만들어 본다. 그래서 Phase 12의 나머지 부분이 "시각 토큰(visual token)"에 대한 구체적인 멘탈 모델을 갖도록 한다.

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- HxWx3 이미지를 올바른 위치 인코딩과 함께 패치 토큰(patch token) 시퀀스로 변환하기.
- 주어진 (패치 크기, 해상도, 은닉 차원, 깊이)에 대한 ViT의 시퀀스 길이, 파라미터(parameter) 개수, FLOPs 계산하기.
- ViT를 2020년 연구에서 2026년 프로덕션(production)으로 끌어올린 세 가지 업그레이드 이름 대기: 자기 지도 사전 학습(self-supervised pretraining)(DINO / MAE), 레지스터 토큰, 네이티브 해상도 패킹(native-resolution packing).
- 다운스트림 작업에 대해 CLS 풀링(pooling), 평균 풀링(mean pooling), 레지스터 토큰 중에서 선택하기.

## 문제 (The Problem)

트랜스포머는 벡터(vector) 시퀀스 위에서 동작한다. 텍스트는 이미 시퀀스다(바이트 또는 토큰). 이미지는 세 개의 색상 채널을 가진 2D 픽셀 격자이지 시퀀스가 아니다. 모든 픽셀을 평탄화(flatten)하면 224x224 RGB 이미지는 150,528개의 토큰이 되는데, 그 길이에서 셀프 어텐션(self-attention)은 시작도 할 수 없다(시퀀스 길이에 이차(quadratic)로 비례하기 때문이다).

2020년 이전의 접근법은 앞단에 CNN 특성 추출기(feature extractor)를 붙였다. ResNet은 2048차원 벡터로 이루어진 7x7 특성 맵(feature map)을 만들고, 그 49개 토큰을 트랜스포머에 넣는다. 이 방식은 동작하지만 CNN의 편향(translation equivariance, 지역적 수용 영역)을 물려받고 트랜스포머가 가진 규모(scale)에 대한 탐욕은 잃어버린다.

Dosovitskiy et al. (2020)은 노골적인 질문을 던졌다. CNN을 건너뛰면 어떨까? 이미지를 고정 크기 패치(가령 16x16 픽셀)로 쪼개고, 각 패치를 벡터로 선형 투영하고, 위치 임베딩을 더한 다음, 평범한 트랜스포머에 시퀀스를 넣는다. 당시 이것은 이단이었다. 합성곱(convolution) 없는 비전이라니. 충분한 데이터(JFT-300M, 그다음 LAION)가 있을 때 이것은 ImageNet에서 ResNet을 이겼고 계속 개선되었다.

2026년 무렵 ViT 원시 단위는 의심할 여지 없는 토대다. 모든 오픈 웨이트(open-weights) VLM의 비전 타워(vision tower)는 그 후손이다(DINOv2, SigLIP 2, CLIP, EVA, InternViT). 질문은 더 이상 "패치를 써야 하는가?"가 아니라 "어떤 패치 크기, 어떤 해상도 스케줄, 어떤 사전 학습 목적, 어떤 위치 인코딩인가?"이다.

## 개념 (The Concept)

### 토큰으로서의 패치 (Patches as tokens)

형상이 `(H, W, 3)`인 이미지 `x`와 패치 크기 `P`가 주어지면, 이미지를 `(H/P) x (W/P)` 격자의 겹치지 않는 패치들로 잘라낸다. 각 패치는 `P x P x 3`짜리 픽셀 큐브다. 각 큐브를 `3 P^2` 벡터로 평탄화한다. 형상이 `(3 P^2, D)`인 공유 선형 투영 `W_E`를 적용해 각 패치를 모델의 은닉 차원 `D`로 매핑한다.

ViT-B/16 표준 설정의 경우:
- 해상도 224, 패치 크기 16 → 격자 14x14 → 196개 패치 토큰.
- 각 패치는 `16 x 16 x 3 = 768`개의 픽셀 값이며, `D = 768`로 투영된다.
- 학습 가능한 `[CLS]` 토큰을 추가 → 시퀀스 길이 197.

패치 투영은 커널 크기 `P`, 스트라이드 `P`, 출력 채널 `D`인 2D 합성곱과 수학적으로 동일하다. 프로덕션 코드가 실제로 이렇게 구현한다. `nn.Conv2d(3, D, kernel_size=P, stride=P)`. "선형 투영"이라는 표현은 개념을 가리키고 커널이라는 표현은 효율적인 구현을 가리킨다.

### 위치 임베딩 (Positional embeddings)

패치에는 본래의 순서가 없다. 트랜스포머는 패치들을 하나의 가방(bag)으로 본다. 초기 ViT는 학습 가능한 1D 위치 임베딩(위치당 768차원 벡터 하나, 총 197개)을 더했다. 동작은 하지만 모델을 학습 해상도에 묶어 버린다. 추론 시 격자를 바꾸면 위치 테이블을 보간(interpolate)해야 한다.

현대 비전 백본(backbone)은 2D-RoPE(Qwen2-VL의 M-RoPE, SigLIP 2의 기본값) 또는 분해된(factorized) 2D 위치를 쓴다. 2D-RoPE는 패치의 (행, 열) 인덱스를 기반으로 쿼리(query)와 키(key) 벡터를 회전시켜, 모델이 회전 각도로부터 상대적인 2D 위치를 추론하게 한다. 위치 테이블이 없다. 모델은 추론 시 임의의 격자 크기를 처리한다.

### CLS 토큰, 풀링된 출력, 레지스터 토큰 (CLS token, pooled output, and register tokens)

이미지 수준 표현(image-level representation)은 무엇인가? 세 가지 선택지가 공존한다:

1. `[CLS]` 토큰. 학습 가능한 벡터를 패치 시퀀스 앞에 붙인다. 모든 트랜스포머 블록을 거친 뒤, CLS 토큰의 은닉 상태가 이미지 표현이 된다. BERT에서 물려받았다. 원조 ViT와 CLIP이 사용한다.
2. 평균 풀링(mean pool). 패치 토큰들의 출력 은닉 상태를 평균낸다. SigLIP, DINOv2, 대부분의 현대 VLM이 사용한다.
3. 레지스터 토큰. Darcet et al. (2023)은 명시적 싱크 토큰(sink token) 없이 학습된 ViT가 셀프 어텐션을 가로채는 높은 노름(high-norm) "아티팩트(artifact)" 패치를 발달시킨다는 것을 관찰했다. 4~16개의 학습 가능한 레지스터 토큰을 추가하면 이 부하를 흡수하고 밀집 예측(dense-prediction) 품질(분할, 깊이)을 개선한다. DINOv2와 SigLIP 2는 둘 다 레지스터를 탑재한 채 출시된다.

이 선택은 다운스트림 작업에서 중요하다. CLS는 분류(classification)에는 괜찮다. 패치 토큰을 LLM에 넣는 VLM의 경우, 풀링을 아예 건너뛴다. 모든 패치가 LLM 입력 토큰이 된다. 레지스터는 넘겨주기 전에 버려진다(레지스터는 내용물이 아니라 비계(scaffolding)다).

### 사전 학습: 지도, 대조, 마스킹, 자기 증류 (Pretraining: supervised, contrastive, masked, self-distilled)

2020년 ViT는 JFT-300M에서 지도 분류(supervised classification)로 사전 학습되었다. 곧 다음 것들로 대체되었다:

- CLIP (2021): 4억 쌍에 대한 대조적(contrastive) 이미지-텍스트. Lesson 12.02.
- MAE (2021, He et al.): 패치의 75%를 마스킹하고 픽셀을 복원한다. 자기 지도 방식이며, 순수 이미지에서 동작한다.
- DINO (2021) / DINOv2 (2023): 학생-교사(student-teacher) 자기 증류(self-distillation), 레이블(label) 없음, 캡션 없음. 2023년 DINOv2 ViT-g/14는 가장 강력한 순수 시각 백본이며 "밀집 특성(dense features)" 사용 사례의 기본값이다.
- SigLIP / SigLIP 2 (2023, 2025): 시그모이드 손실(sigmoid loss)과 네이티브 종횡비를 위한 NaFlex를 갖춘 CLIP. 2026년 오픈 VLM에서 지배적인 비전 타워다(Qwen, Idefics2, LLaVA-OneVision).

사전 학습의 선택이 백본이 무엇에 능한지를 결정한다. 텍스트와의 의미적 매칭에는 CLIP/SigLIP, 밀집 시각 특성에는 DINOv2, 다운스트림 파인튜닝(fine-tuning)의 출발점으로는 MAE.

### 스케일링 법칙 (Scaling laws)

ViT 스케일링(Zhai et al. 2022)은 ViT의 품질이 모델 크기, 데이터 크기, 연산량에서 예측 가능한 법칙을 따른다는 것을 확립했다. 고정된 연산량에서:
- 더 큰 모델 + 더 많은 데이터 → 더 나은 품질.
- 패치 크기는 시퀀스 길이 대 충실도(fidelity)의 지렛대다. 패치 14(DINOv2/SigLIP SO400m의 전형)는 패치 16보다 이미지당 더 많은 토큰을 준다. OCR과 밀집 작업에는 더 좋고, 속도에는 더 나쁘다.
- 해상도는 또 다른 큰 지렛대다. 224에서 384, 512로 올리는 것은 거의 항상 도움이 되지만, FLOPs는 이차로 늘어난다.

ViT-g/14(파라미터 10억, 패치 14, 해상도 224 → 256토큰)와 SigLIP SO400m/14(파라미터 4억, 패치 14)는 2026년 오픈 VLM의 두 가지 일꾼 인코더다.

### ViT의 파라미터 개수 (Parameter count for a ViT)

전체 계산은 `code/main.py`에 있다. 224에서의 ViT-B/16의 경우:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

체크포인트를 로드하기 전에 모든 ViT를 이런 식으로 어림셈하라. 백본 크기가 어떤 다운스트림 VLM에서든 VRAM 하한선을 정한다.

### 2026년 프로덕션 설정 (2026 production config)

2026년 대부분의 오픈 VLM이 탑재하고 출시하는 인코더는 네이티브 해상도(NaFlex)의 SigLIP 2 SO400m/14다. 이것은 다음을 갖는다:
- 파라미터 4억 개.
- 패치 크기 14, 기본 해상도 384 → 이미지당 729개 패치 토큰.
- 이미지 수준 작업에는 평균 풀링, VQA에는 729개 패치 전체가 LLM으로 흐른다.
- 레지스터 토큰 4개, LLM에 넘기기 전에 버려진다.
- 네이티브 종횡비를 위한 이미지 수준 스케일링을 갖춘 2D-RoPE.

그 설정의 모든 결정은 직접 읽어 볼 수 있는 논문으로 거슬러 올라간다.

```figure
image-patch-tokens
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 패치 토크나이저(tokenizer)이자 기하 계산기다. (이미지 H, W, 패치 P, 은닉 D, 깊이 L)를 받아 다음을 보고한다:

- 패치화 후의 격자 형상과 시퀀스 길이.
- 합성된 8x8 픽셀 장난감 이미지에 대한 토큰 시퀀스(평탄화 + 투영 경로를 따라가 본다).
- 패치 임베딩, 위치 임베딩, 트랜스포머 블록, 헤드(head)로 분해한 파라미터 개수.
- 목표 해상도에서 순방향 패스(forward pass)당 FLOPs.
- ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384에 걸친 비교표.

실행해 보라. 파라미터 개수를 발표된 수치와 맞춰 보라. 패치 크기와 해상도를 가지고 놀면서 토큰 개수 비용을 체감하라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-patch-geometry-reader.md`를 만든다. ViT 설정(패치 크기, 해상도, 은닉 차원, 깊이)이 주어지면, 토큰 개수, 파라미터 개수, VRAM 추정치를 근거와 함께 산출한다. VLM을 위한 비전 백본을 고를 때마다 이 스킬을 사용하라. "토큰이 폭발해서 내 LLM 컨텍스트가 가득 찼다"는 식의 예상치 못한 상황을 막아 준다.

## 연습 문제 (Exercises)

1. 패치 크기 14로 네이티브 1280x720 입력을 받는 Qwen2.5-VL의 패치-토큰 시퀀스 길이를 계산하라. 이것은 CLS만 쓰는 표현과 비교하면 어떤가?

2. 패치 14에서 1080p 프레임(1920x1080)은 토큰을 몇 개 만드는가? 5분짜리 비디오에서 30 FPS로 처리하면 총 시각 토큰은 몇 개인가? 풀링, 프레임 샘플링, 토큰 병합 중 어느 것이 가장 많은 비용을 절약하는가?

3. 순수 Python으로 패치 토큰에 대한 평균 풀링을 구현하라. DINOv2 출력의 196개 토큰에 대한 평균 풀링이, 풀링된 임베딩을 요청했을 때 모델의 `forward`가 반환하는 것과 일치하는지 검증하라.

4. "Vision Transformers Need Registers"(arXiv:2309.16588)의 Section 3을 읽어라. 레지스터가 어떤 아티팩트를 흡수하는지, 그리고 그것이 왜 다운스트림 밀집 예측에 중요한지를 두 문장으로 서술하라.

5. patch-n'-pack을 지원하도록 `code/main.py`를 수정하라. 서로 다른 해상도의 이미지 목록이 주어지면, 단일 패킹된 시퀀스와 블록 대각(block-diagonal) 어텐션 마스크를 산출하라. Lesson 12.06에 도달하면 그것과 대조하여 검증하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 패치(Patch) | "16x16 픽셀 정사각형" | 입력 이미지의 고정 크기 겹치지 않는 영역. 하나의 토큰이 된다 |
| 패치 임베딩(Patch embedding) | "선형 투영" | 평탄화된 패치 픽셀을 D차원 벡터로 매핑하는 공유 학습 행렬(또는 stride=P인 Conv2d) |
| CLS 토큰(CLS token) | "클래스 토큰" | 최종 은닉 상태가 이미지 전체를 표현하는, 앞에 붙이는 학습 가능한 벡터. 2026년에는 선택 사항 |
| 레지스터 토큰(Register token) | "싱크 토큰" | ViT가 사전 학습 중 발달시키는 높은 노름 어텐션 아티팩트를 흡수하는 추가 학습 가능 토큰 |
| 위치 임베딩(Position embedding) | "위치 정보" | 시퀀스를 순서 인식하게 만드는 위치별 벡터 또는 회전. 2D-RoPE가 현대의 기본값 |
| 격자(Grid) | "패치 격자" | 주어진 해상도와 패치 크기에 대한 패치의 (H/P) x (W/P) 2D 배열 |
| NaFlex | "네이티브 유연 해상도" | SigLIP 2 기능. 단일 모델이 재학습 없이 여러 종횡비와 해상도를 처리한다 |
| 백본(Backbone) | "비전 타워" | 패치-토큰 출력이 VLM에서 LLM으로 흐르는, 사전 학습된 이미지 인코더 |
| 풀링(Pooling) | "이미지 수준 요약" | 패치 토큰을 하나의 벡터로 바꾸는 전략: CLS, 평균, 어텐션 풀, 또는 레지스터 기반 |
| 패치 14 대 16(Patch 14 vs 16) | "더 촘촘한 격자 대 더 성긴 격자" | 패치 14는 이미지당 더 많은 토큰을 만들고, OCR 충실도가 더 좋으며, 더 느리다. 패치 16은 고전적 기본값 |

## 더 읽을거리 (Further Reading)

- [Dosovitskiy et al.: An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) 원조 ViT.
- [He et al.: Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) MAE, 자기 지도 사전 학습.
- [Oquab et al.: DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) 대규모 자기 증류, 레이블 없음.
- [Darcet et al.: Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) 레지스터 토큰과 아티팩트 분석.
- [Tschannen et al.: SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) 2026년 기본 비전 타워.
- [Zhai et al.: Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) 경험적 스케일링 법칙.
