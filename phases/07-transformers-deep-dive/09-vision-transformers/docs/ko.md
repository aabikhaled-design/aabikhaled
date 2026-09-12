# 비전 트랜스포머(Vision Transformers, ViT)

> 이미지는 패치(patch)의 격자다. 문장은 토큰(token)의 격자다. 같은 트랜스포머(transformer)가 둘 다 먹어 치운다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45분

## 문제 (The Problem)

2020년 이전에 컴퓨터 비전(computer vision)은 곧 합성곱(convolution)을 의미했다. ImageNet과 COCO, 각종 검출(detection) 벤치마크(benchmark)의 모든 SOTA는 CNN 백본(backbone)을 사용했다. 트랜스포머는 언어를 위한 것이었다.

Dosovitskiy et al. (2020)("An Image is Worth 16x16 Words")는 합성곱을 완전히 버릴 수 있음을 보였다. 이미지를 고정 크기 패치로 잘라, 각 패치를 임베딩(embedding)으로 선형 사영(linear projection)한 뒤, 그 시퀀스(sequence)를 평범한 트랜스포머 인코더(encoder)에 넣는다. 충분한 규모(ImageNet-21k 사전 학습(pretraining) 이상)에서 ViT는 ResNet 기반 모델(model)에 필적하거나 능가한다.

ViT는 2026년의 더 넓은 패턴, 즉 "하나의 아키텍처, 여러 모달리티(modality)"의 출발점이었다. Whisper는 오디오를 토큰화(tokenize)한다. ViT는 이미지를 토큰화한다. 로보틱스를 위한 액션 토큰(action token). 비디오를 위한 픽셀 토큰(pixel token). 트랜스포머는 상관하지 않는다. 시퀀스를 넣어 주면 학습(training)한다.

2026년에 이르러 ViT와 그 후손들(DeiT, Swin, DINOv2, ViT-22B, SAM 3)이 비전의 대부분을 차지한다. CNN은 여전히 엣지 디바이스(edge device)와 지연 시간(latency)에 민감한 작업에서 우위를 점한다. 그 외 거의 모든 곳에는 스택(stack) 어딘가에 ViT가 들어 있다.

## 개념 (The Concept)

![Image → patches → tokens → transformer](../assets/vit.svg)

### 1단계: 패치화(patchify)

`H × W × C` 이미지를 `N × (P·P·C)`개의 평탄화된 패치 시퀀스로 분할한다. 일반적인 설정: `224 × 224` 이미지, `16 × 16` 패치 → 각각 768개 값을 가진 196개의 패치.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

패치 크기가 핵심 조절 손잡이다. 작은 패치 = 더 많은 토큰, 더 좋은 해상도, 어텐션(attention)의 이차 비용. 큰 패치 = 더 거칠지만 더 저렴하다.

### 2단계: 선형 임베딩(linear embedding)

학습된 행렬(matrix) 하나가 각 평탄화된 패치를 `d_model`로 사영한다. 커널 크기 `P`, 스트라이드(stride) `P`의 합성곱과 동등하다. PyTorch에서는 말 그대로 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`, 즉 2줄짜리 구현이다.

### 3단계: `[CLS]` 토큰 앞에 붙이기, 위치 임베딩(positional embedding) 더하기

- 학습 가능한 `[CLS]` 토큰을 앞에 붙인다. 이 토큰의 최종 은닉 상태(hidden state)가 분류(classification)에 쓰이는 이미지 표현(representation)이다.
- 학습 가능한 위치 임베딩(ViT 원본) 또는 사인파(sinusoidal) 2D(이후 변형들)를 더한다.
- 2024년 이후에는 RoPE가 위치 표현을 위해 2D로 확장되었고, 명시적 임베딩 없이 쓰이기도 한다.

### 4단계: 표준 트랜스포머 인코더

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 블록 L개를 쌓는다. BERT와 동일하다. 비전 전용 층(layer)은 없다. 이것이 이 논문의 교육적 핵심 메시지다.

### 5단계: 헤드(head)

분류의 경우: `[CLS]` 은닉 상태 → 선형 → softmax. DINOv2나 SAM의 경우: `[CLS]`를 버리고 패치 임베딩을 직접 사용한다.

### 중요했던 변형들

| 모델 | 연도 | 변화 |
|-------|------|------|
| ViT | 2020 | 원조. 고정 패치 크기, 완전 전역(global) 어텐션. |
| DeiT | 2021 | 증류(distillation); ImageNet-1k만으로 학습 가능. |
| Swin | 2021 | 시프트 윈도우(shifted window)를 활용한 계층 구조. 비용을 준이차(sub-quadratic)로 고정. |
| DINOv2 | 2023 | 자기 지도 학습(self-supervised, 레이블 없음). 최고의 범용 비전 특성(feature). |
| ViT-22B | 2023 | 파라미터(parameter) 22B; 스케일링 법칙(scaling laws)이 적용된다. |
| SigLIP | 2023 | ViT + 언어 페어, 시그모이드 대조 손실(sigmoid contrastive loss). |
| SAM 3 | 2025 | Segment anything; ViT-Large + 프롬프트 가능(promptable) 마스크 디코더(decoder). |

### 왜 시간이 걸렸나

ViT는 CNN의 귀납 편향(inductive bias, 평행 이동 불변성·국소성)이 전혀 없기 때문에 CNN에 필적하려면 *아주 많은* 데이터가 필요하다. 1억 개 이상의 레이블된 이미지나 강력한 자기 지도 사전 학습이 없으면, 동일한 연산량에서 여전히 CNN이 이긴다. DeiT가 2021년에 증류 기법으로 이 문제를 해결했고, DINOv2가 2023년에 자기 지도 학습으로 영구히 해결했다.

```figure
n5-patch-stream
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. 순수 표준 라이브러리(stdlib)로 만든 패치화 + 선형 임베딩 + 정상성 검사(sanity check)다. 학습은 없다. 현실적인 규모의 ViT는 PyTorch와 수 시간의 GPU 시간이 필요하기 때문이다.

### 1단계: 가짜 이미지

`(R, G, B)` 튜플의 행 리스트로 표현한 24 × 24 RGB 이미지. 6×6 패치를 사용한다 → 16개 패치, 각 패치는 108차원 임베딩 벡터(vector).

### 2단계: 패치화

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

래스터 순서(raster order): 격자를 가로질러 행 우선(row-major)으로. 모든 ViT가 이 순서를 사용한다.

### 3단계: 선형 임베딩

각 평탄화된 패치에 무작위 `(patch_flat_size, d_model)` 행렬을 곱한다. `[CLS]`를 앞에 붙인 후 출력 형태가 `(N_patches + 1, d_model)`인지 확인한다.

### 4단계: 현실적인 ViT의 파라미터 수 세기

ViT-Base의 파라미터 수를 출력한다: 12개 층, 12개 헤드, d=768, 패치=16. ResNet-50(~25M)과 비교하라. ViT-Base는 ~86M에 도달한다. ViT-Large ~307M. ViT-Huge ~632M.

## 라이브러리로 써보기 (Use It)

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 임베딩은 2026년 이미지 특성의 기본값이다.** 백본을 동결(freeze)하고 작은 헤드만 학습시킨다. 분류, 검색(retrieval), 검출, 캡셔닝(captioning)에 모두 통한다. Meta의 DINOv2 체크포인트는 텍스트가 아닌 모든 비전 작업에서 CLIP을 능가한다.

**패치 크기 고르기.** 작은 모델은 16×16(ViT-B/16)을 쓴다. 밀집 예측(dense prediction, 분할(segmentation))은 8×8 또는 14×14(SAM, DINOv2)를 쓴다. 아주 큰 모델은 14×14를 쓴다.

## 산출물 (Ship It)

`outputs/skill-vit-configurator.md`를 참고하라. 이 스킬은 데이터셋(dataset) 크기, 해상도, 연산 예산이 주어진 새 비전 작업에 대해 ViT 변형과 패치 크기를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 패치 개수가 `(H/P) * (W/P)`와 같고, 평탄화된 패치 차원이 `P*P*C`와 같은지 확인하라.
2. **중간.** 2D 사인파 위치 임베딩을 구현하라. 각 패치의 `row`와 `col`에 대해 두 개의 독립적인 사인파 코드를 만들어 이어 붙인다. 이를 작은 PyTorch ViT에 넣어 CIFAR-10에서 학습 가능한 위치 임베딩과 정확도를 비교하라.
3. **어려움.** 3층 ViT(PyTorch)를 만들고, 4×4 패치로 MNIST 이미지 1,000장에 대해 학습시켜라. 테스트 정확도를 측정하라. 이제 같은 1,000장에 대해 DINOv2 사전 학습을 추가하라(단순화: 마스킹된 패치로부터 패치 임베딩을 예측하도록 인코더만 학습). 정확도가 향상되는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 패치(Patch) | "비전 트랜스포머의 토큰" | 이미지의 `P × P × C` 영역에 대한 픽셀 값의 평탄화된 벡터. |
| 패치화(Patchify) | "자르고 + 평탄화" | 이미지를 겹치지 않는 패치로 잘라 각각을 벡터로 평탄화하는 것. |
| `[CLS]` 토큰 | "이미지 요약" | 앞에 붙이는 학습 가능한 토큰; 그 최종 임베딩이 이미지 표현이다. |
| 귀납 편향(Inductive bias) | "모델이 가정하는 것" | ViT는 CNN보다 사전 가정이 적다; 그 격차를 메우려면 더 많은 데이터가 필요하다. |
| DINOv2 | "자기 지도 ViT" | 이미지 증강(augmentation) + 모멘텀 교사(momentum teacher)를 사용해 레이블 없이 학습. 2026년 최고의 범용 이미지 특성. |
| SigLIP | "CLIP의 후계자" | 시그모이드 대조 손실로 학습한 ViT + 텍스트 인코더; 동일 연산량에서 CLIP보다 낫다. |
| Swin | "윈도우 ViT" | 국소 어텐션 + 시프트 윈도우를 가진 계층적 ViT; 준이차. |
| 레지스터 토큰(Register tokens) | "2023년의 트릭" | 어텐션 싱크(attention sink)를 흡수하는 몇 개의 추가 학습 가능 토큰; DINOv2 특성을 개선한다. |

## 더 읽을거리 (Further Reading)

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929): ViT 논문.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): DINOv2를 위한 레지스터 토큰 해결책.
