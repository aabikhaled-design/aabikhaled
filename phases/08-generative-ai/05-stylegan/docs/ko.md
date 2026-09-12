# StyleGAN

> 대부분의 생성기(generator)는 `z`를 모든 층(layer)에 동시에 섞어 넣는다. StyleGAN은 이를 분리했다: 먼저 `z`를 중간(intermediate) `w`로 매핑한 다음, AdaIN을 통해 모든 해상도 수준에서 `w`를 *주입(inject)*한다. 그 단 하나의 변화가 잠재 공간(latent space)을 풀어냈고, 사실적 얼굴 생성을 7년 연속 해결된 문제로 만들었다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45분

## 문제 (The Problem)

DCGAN은 전치 합성곱(transposed convolution) 스택으로 `z`를 이미지로 매핑한다. 문제는 `z`가 포즈, 조명, 신원, 배경을 모두 함께 얽힌(entangled) 채로 제어한다는 점이다. `z`의 한 축을 따라 움직이면 넷 모두가 바뀐다. 표현(representation)이 그렇게 인수분해되지 않으니 모델에게 "같은 사람, 다른 포즈"를 요청할 수 없다.

Karras et al. (2019, NVIDIA)의 제안은 이렇다. `z`를 합성곱 층에 직접 넣지 마라. 상수 `4×4×512` 텐서를 네트워크 입력으로 넣어라. `z ∈ Z → w ∈ W`로 매핑하는 8층 MLP를 학습하라. *적응형 인스턴스 정규화(adaptive instance normalization)*(AdaIN)를 통해 모든 해상도에서 `w`를 주입하라: 각 합성곱 특성 맵(feature map)을 정규화한 다음, `w`의 아핀 투영(affine projection)으로 스케일하고 이동하라. 확률적 세부(피부 모공, 머리카락 가닥)를 위해 층별 잡음(per-layer noise)을 더하라.

결과적으로 `W`는 "고수준 스타일"(포즈, 신원)과 "미세 스타일"(조명, 색)에 대해 거의 직교하는 축을 갖는다. 이미지 A의 `w`를 저해상도 수준에, 이미지 B의 `w`를 고해상도에 쓰면 두 이미지 사이에서 스타일을 교환할 수 있다. 이로써 편집, 도메인 간 양식화, "StyleGAN-역변환(inversion)"이라는 연구 계열 전체가 열렸다.

## 개념 (The Concept)

![StyleGAN: 매핑 네트워크 + AdaIN + 층별 잡음](../assets/stylegan.svg)

**매핑 네트워크(Mapping network).** `f: Z → W`, 8층 MLP다. `Z = N(0, I)^512`. `W`는 가우시안이어야 한다는 제약이 없으며, 데이터에 적응한 형태를 학습한다.

**합성 네트워크(Synthesis network).** 학습된 상수 `4×4×512`에서 시작한다. 각 해상도 블록: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. 해상도가 두 배씩: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

여기서 `y_scale`과 `y_bias`는 `w`의 아핀 투영에서 나온다. 특성 맵별로 정규화한 다음 다시 양식화한다. 여기서 "스타일"은 특성 맵의 1차 및 2차 통계량이다.

**층별 잡음(Per-layer noise).** 각 특성 맵에 더해지는 단일 채널 가우시안 잡음으로, 학습된 채널별 인자로 스케일된다. 전역 구조에 영향을 주지 않고 확률적 세부를 제어한다.

**절단 트릭(Truncation trick).** 추론(inference) 시, `z`를 샘플링하고, `w = mapping(z)`를 계산한 다음, `w' = ŵ + ψ·(w - ŵ)`를 한다. 여기서 `ŵ`는 많은 샘플에 대한 평균 `w`다. `ψ < 1`은 다양성을 품질과 맞바꾼다. 거의 모든 StyleGAN 데모가 `ψ ≈ 0.7`을 사용한다.

## StyleGAN 1 → 2 → 3

| 버전 | 연도 | 혁신 |
|---------|------|------------|
| StyleGAN | 2019 | 매핑 네트워크 + AdaIN + 잡음 + 점진적 성장(progressive growing). |
| StyleGAN2 | 2020 | 가중치 복조(weight demodulation)가 AdaIN을 대체(물방울 아티팩트 해결); 스킵/잔차 아키텍처; 경로 길이 정규화(path-length regularization). |
| StyleGAN3 | 2021 | 앨리어스 없는(alias-free) 합성곱 + 등변(equivariant) 커널; 텍스처가 픽셀 격자에 달라붙는 현상 제거. |
| StyleGAN-XL | 2022 | 클래스 조건부, 1024², ImageNet. |
| R3GAN | 2024 | 더 강한 정규화로 재브랜딩; 20배 적은 파라미터로 FFHQ-1024에서 확산과의 격차를 좁힘. |

2026년에 StyleGAN3은 (a) 높은 FPS의 좁은 도메인 사실적 묘사, (b) 퓨샷(few-shot) 도메인 적응(100장 이미지로 새 데이터셋에 학습, 매핑은 동결), (c) 역변환 기반 편집(실제 사진을 재구성하는 `w`를 찾은 다음 그 `w`를 편집)의 기본값으로 남아 있다. 개방형 도메인 텍스트-이미지에는 적합하지 않으며, 그쪽은 확산(diffusion)의 몫이다.

```figure
gx-stylegan-mapping
```

## 직접 만들기 (Build It)

`code/main.py`는 1차원에서 장난감 "style-GAN lite"를 구현한다: 매핑 MLP, 학습된 상수 벡터를 받아 `w`에서 유도된 스케일/바이어스로 변조하는 합성 함수, 그리고 층별 잡음. 아핀 변조로 `w`를 주입하는 방식이 생성기 입력에 `z`를 연결하는 방식과 같거나 더 낫다는 것을 여기서 확인할 수 있다.

### 1단계: 매핑 네트워크

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 2단계: 적응형 인스턴스 정규화

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

특성 맵별 스케일과 바이어스는 선형 투영을 통해 `w`에서 나온다.

### 3단계: 층별 잡음

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

채널별 시그마는 학습 가능하다.

## 함정 (Pitfalls)

- **물방울 아티팩트(Droplet artifacts).** StyleGAN 1은 AdaIN이 평균을 0으로 만들었기 때문에 특성 맵에 방울 모양의 물방울을 만들었다. StyleGAN 2의 가중치 복조는 대신 합성곱 가중치를 스케일링하여 이를 해결한다.
- **텍스처 달라붙음(Texture sticking).** StyleGAN 1과 2의 텍스처는 객체 좌표가 아니라 픽셀 좌표를 따랐다(보간(interpolating) 시 보인다). StyleGAN 3의 앨리어스 없는 합성곱은 윈도잉된 sinc 필터로 이를 해결한다.
- **모드 커버리지.** 절단 `ψ < 0.7`은 깨끗해 보이지만 좁은 원뿔에서 샘플링한다; 다양성이 필요하면 `ψ = 1.0`을 사용하라.
- **역변환은 손실이 있다.** 실제 사진을 `W`로 역변환하는 것은 보통 최적화나 인코더(e4e, ReStyle, HyperStyle)를 통해 한다. 결과는 많은 반복에 걸쳐 표류한다.

## 라이브러리로 써보기 (Use It)

| 사용 사례 | 접근법 |
|----------|----------|
| 사실적 인간 얼굴(애니메이션, 제품, 좁은) | StyleGAN3 FFHQ / 커스텀 파인튜닝(fine-tune) |
| 사진으로부터의 얼굴 편집 | e4e 역변환 + StyleSpace / InterFaceGAN 방향 |
| 얼굴 교체 / 재연(reenactment) | StyleGAN + 인코더 + 블렌딩 |
| 아바타 파이프라인 | 저데이터 파인튜닝을 위한 ADA 적용 StyleGAN3 |
| 소수 이미지로부터의 도메인 적응 | 매핑 네트워크 동결, 합성 파인튜닝 |
| 멀티모달 또는 텍스트 조건 생성 | 하지 마라. 확산을 써라 |

답이 "사람 얼굴 사진"인 제품급 데모에서, StyleGAN은 추론 비용(단일 순방향 패스(forward pass), 4090에서 10ms 미만)과 동일한 품질 기준에서의 선명도로 확산을 이긴다.

## 산출물 (Ship It)

`outputs/skill-stylegan-inversion.md`로 저장하라. 이 스킬은 실제 사진을 받아 다음을 출력한다: 역변환 방법(e4e / ReStyle / HyperStyle), 예상 잠재 손실, 편집 예산(아티팩트 전까지 `W`에서 얼마나 멀리 움직일 수 있는지), 그리고 검증된 편집 방향 목록(나이, 표정, 포즈).

## 연습 문제 (Exercises)

1. **쉬움.** `adain_on=True`와 `adain_on=False`로 `code/main.py`를 실행하라. 고정 잠재 대 섭동(perturbed) 잠재에 대한 출력의 퍼짐을 비교하라.
2. **보통.** 혼합 정규화(mixing regularization)를 구현하라: 학습 배치에 대해 `w_a`, `w_b`를 계산하고, 합성의 전반부에 `w_a`를, 후반부에 `w_b`를 적용하라. 디코더가 풀린(disentangled) 스타일을 학습하는가?
3. **어려움.** 사전 학습된 StyleGAN3 FFHQ 모델(ffhq-1024.pkl)을 가져와라. 레이블된 샘플에 SVM을 학습시켜 "미소"를 제어하는 `w` 방향을 찾아라; 신원이 표류하기 전까지 얼마나 밀 수 있는지 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 매핑 네트워크(Mapping network) | "그 MLP" | `f: Z → W`, 8층, 잠재 기하를 데이터 통계로부터 분리한다. |
| W 공간(W space) | "스타일 공간" | 매핑 네트워크의 출력; 대체로 풀려 있음. |
| AdaIN | "적응형 인스턴스 정규화" | 특성 맵을 정규화한 다음 `w` 투영으로 스케일 + 이동. |
| 절단 트릭(Truncation trick) | "프사이(Psi)" | `w = mean + ψ·(w - mean)`, ψ<1은 다양성을 품질과 맞바꾼다. |
| 경로 길이 정규화(Path-length regularization) | "PL reg" | `w` 단위 변화당 이미지의 큰 변화를 페널티; `W`를 더 매끄럽게 만든다. |
| 가중치 복조(Weight demodulation) | "StyleGAN2 해결책" | 활성값 대신 합성곱 가중치를 정규화; 물방울 아티팩트를 없앤다. |
| 앨리어스 없음(Alias-free) | "StyleGAN3의 트릭" | 윈도잉된 sinc 필터; 텍스처가 픽셀 격자에 달라붙는 현상을 제거. |
| 역변환(Inversion) | "실제 이미지에 대한 w 찾기" | `G(w) ≈ x`가 되도록 `x → w`를 최적화하거나 인코딩한다. |

## 프로덕션 노트: 왜 StyleGAN이 2026년에도 여전히 출시되는가 (Production note: why StyleGAN still ships in 2026)

4090에서 StyleGAN3은 1024² FFHQ 얼굴을 10 ms 미만에 생성한다. `num_steps = 1`, VAE 디코드 없음, 교차 어텐션(cross-attention) 패스 없음. 프로덕션 관점에서 이것은 모든 이미지 생성기의 바닥(floor) 지연 시간(latency)이다. 같은 해상도에서 50스텝 SDXL + VAE-디코드 파이프라인은 약 3초다. **300배 격차**이며, 좁은 도메인 제품(아바타 서비스, 신분증 문서 파이프라인, 스톡 얼굴 생성)에서는 총소유비용(TCO)으로 앞선다.

두 가지 운영상 결과:

- **스케줄러도, 배처(batcher)도 없음.** 목표 점유율에서의 정적 배치가 최적이다. 연속 배칭(continuous batching)(LLM과 확산에 필수)은 모든 요청이 같은 FLOPs를 차지하므로 아무 이점이 없다.
- **절단 `ψ`가 안전 손잡이다.** `ψ < 0.7`은 매핑 네트워크 범위의 좁은 원뿔에서 샘플링한다. 이것은 서빙 계층이 샘플 분산에 대해 가진 유일한 레버다. 최대 부하에서는 `ψ`를 낮추고, 프리미엄 사용자에게는 높여라.

## 더 읽을거리 (Further Reading)

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766): e4e 역변환.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441): 현대적 최소 GAN 레시피.
