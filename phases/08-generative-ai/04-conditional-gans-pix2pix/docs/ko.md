# 조건부 GAN과 Pix2Pix (Conditional GANs & Pix2Pix)

> 2014-2017년의 첫 번째 큰 해금(unlock)은 GAN이 무엇을 만드는지 제어하는 것이었다. 레이블(label)을, 또는 이미지를, 또는 문장을 붙여라. Pix2Pix는 이미지 버전을 해냈고, 좁은 이미지-이미지 작업에서는 여전히 모든 범용 텍스트-이미지 모델을 능가한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75분

## 문제 (The Problem)

무조건(unconditional) GAN은 임의의 얼굴을 샘플링한다. 데모에는 쓸모 있지만 프로덕션(production)에서는 쓸모없다. 실제로 필요한 것은 *스케치를 사진으로 매핑*, *지도를 항공 사진으로 매핑*, *낮 장면을 밤으로 매핑*, *흑백 이미지를 채색*하는 일이다. 이 모두에서, 입력 이미지 `x`가 주어지고 어떤 의미적 대응(correspondence)을 가진 `y`를 출력해야 한다. `x`마다 그럴듯한 `y`가 여럿 있다. 평균 제곱 오차는 이들을 죽처럼 뭉갠다. 적대적 손실(adversarial loss)은 그렇지 않은데, "진짜처럼 보인다"가 선명하기 때문이다.

조건부 GAN(Conditional GAN, Mirza & Osindero, 2014)은 조건 `c`를 `G`와 `D` 양쪽의 입력으로 추가한다. Pix2Pix(Isola et al., 2017)는 이를 특수화했다: 조건은 전체 입력 이미지이고, 생성기(generator)는 U-Net이며, 판별기(discriminator)는 *패치 기반(patch-based)* 분류기(PatchGAN)이고, 손실은 적대적 + L1이다. 이 레시피는 2026년에도 좁은 이미지-이미지 도메인에서 밑바닥부터 학습한 텍스트-이미지 모델을 능가한다. *짝지어진 데이터(paired data)*로 학습되기 때문이다. 필요한 신호를 정확히 갖고 있다는 뜻이다.

## 개념 (The Concept)

![Pix2Pix: U-Net 생성기, PatchGAN 판별기](../assets/pix2pix.svg)

**조건부 G.** `G(x, z) → y`. Pix2Pix에서 `z`는 G 내부의 드롭아웃(dropout)이다(입력 잡음은 없다. Isola는 명시적 잡음이 무시된다는 것을 발견했다).

**조건부 D.** `D(x, y) → [0, 1]`. 입력은 (조건, 출력)의 *쌍*이다. 여기에 핵심 차이가 있다. D는 `y`가 단지 진짜처럼 보이는지가 아니라 `x`와 일관되는지를 판단해야 한다.

**U-Net 생성기.** 병목(bottleneck)을 가로지르는 스킵 연결(skip connection)을 가진 인코더-디코더. 입력과 출력이 저수준 구조(가장자리, 실루엣)를 공유하는 작업에 결정적이다. 스킵이 없으면 고주파 세부가 사라진다.

**PatchGAN 판별기.** 단일 진짜/가짜 점수를 출력하는 대신, D는 각 셀이 약 70×70 픽셀의 수용 영역(receptive field)을 판단하는 `N×N` 격자를 출력한다. 평균을 낸다. 이것은 사실성이 국소적이라는 마르코프 무작위장(Markov random field) 가정이다. 학습이 훨씬 빠르고, 파라미터(parameter)가 적고, 출력이 선명하다.

**손실.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 항은 학습을 안정화하고 G를 알려진 목표 쪽으로 민다. L1은 L2보다 가장자리를 더 선명하게 만든다(평균이 아니라 중앙값). `λ = 100`이 Pix2Pix의 기본값이었다.

## CycleGAN(짝이 없을 때 (CycleGAN) when you don't have pairs)

Pix2Pix는 짝지어진 `(x, y)` 데이터가 필요하다. CycleGAN(Zhu et al., 2017)은 추가 손실(*순환 일관성(cycle consistency)* 손실)의 대가로 이 요구사항을 없앤다. 두 생성기 `G: X → Y`와 `F: Y → X`. `F(G(x)) ≈ x`와 `G(F(y)) ≈ y`가 되도록 학습시킨다. 이로써 짝지어진 예제 없이 말을 얼룩말로, 여름을 겨울로 변환할 수 있다.

2026년에 짝 없는 이미지-이미지 변환은 대부분 CycleGAN이 아니라 확산(diffusion)(ControlNet, IP-Adapter)으로 이루어지지만, 순환 일관성 아이디어는 거의 모든 짝 없는 도메인 적응 논문에 살아남아 있다.

```figure
gx-patchgan
```

## 직접 만들기 (Build It)

`code/main.py`는 1차원 데이터에서 아주 작은 조건부 GAN을 구현한다. 조건 `c`는 클래스 레이블(0 또는 1)이다. 작업: 주어진 클래스에 대한 조건부 분포에서 샘플을 생성하는 것.

### 1단계: G와 D 입력 양쪽에 조건을 덧붙이기

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

원-핫(one-hot) 인코딩이 가장 단순한 방법이다. 더 큰 모델은 학습된 임베딩(embedding), FiLM 변조, 또는 교차 어텐션(cross-attention)을 사용한다.

### 2단계: 조건부로 학습

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

생성기는 주변(marginal) 분포가 아니라 *주어진 조건에 대한* 진짜 분포를 맞춰야 한다.

### 3단계: 클래스별 출력 검증

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 함정 (Pitfalls)

- **조건이 무시됨.** G가 주변화(marginalize)를 학습하고, 조건 신호가 약해서 D가 결코 페널티를 주지 않는다. 해결책: D를 더 공격적으로 조건화하라(늦은 층뿐 아니라 이른 층에도), 투영 판별기(projection discriminator)(Miyato & Koyama 2018)를 사용하라.
- **L1 가중치가 너무 낮음.** G가 충실한 출력이 아니라 임의의 진짜처럼 보이는 출력으로 표류한다. Pix2Pix 스타일 작업에는 λ≈100에서 시작하라.
- **L1 가중치가 너무 높음.** L1도 여전히 L_p 노름이므로 G가 흐릿한 출력을 만든다. 학습이 안정되면 점진적으로 낮춰라.
- **D에서의 정답 누설(Ground-truth leakage).** D 입력으로 `y`만이 아니라 `(x, y)`를 연결하라. 이것 없이는 D가 일관성을 확인할 수 없다.
- **클래스별 모드 붕괴.** 각 클래스가 독립적으로 붕괴할 수 있다. 클래스 조건부 다양성 점검을 실행하라.

## 라이브러리로 써보기 (Use It)

2026년 이미지-이미지 작업의 현황:

| 작업 | 최선의 접근법 |
|------|---------------|
| 스케치 → 사진, 같은 도메인, 짝지어진 데이터 | Pix2Pix / Pix2PixHD (여전히 빠르고, 여전히 선명함) |
| 스케치 → 사진, 짝 없음 | Scribble 조건화 모델을 사용한 ControlNet |
| 의미 분할(semantic seg) → 사진 | SPADE / GauGAN2 또는 SD + ControlNet-Seg |
| 스타일 전이 | IP-Adapter 또는 LoRA를 사용한 확산; GAN 방법은 레거시(legacy) |
| 깊이(depth) → 사진 | Stable Diffusion 위의 ControlNet-Depth |
| 초해상도(super-resolution) | Real-ESRGAN (GAN), ESRGAN-Plus, 또는 SD-Upscale (확산) |
| 채색(colorization) | ColTran, 확산 기반 채색기, 또는 Pix2Pix-color |
| 낮 → 밤, 계절, 날씨 | CycleGAN 또는 ControlNet 기반 |

Pix2Pix는 (a) 짝지어진 예제가 수천 개 있고, (b) 작업이 좁고 반복 가능하며, (c) 빠른 추론이 필요할 때 여전히 올바른 도구다. 범용 개방형 도메인 작업에서는 확산이 이긴다.

## 산출물 (Ship It)

`outputs/skill-img2img-chooser.md`로 저장하라. 이 스킬은 작업 설명, 데이터 가용성(짝지어짐 대 짝 없음, N개 샘플), 그리고 지연 시간/품질 예산을 받아 다음을 출력한다: 접근법(Pix2Pix, CycleGAN, ControlNet 변형, SDXL + IP-Adapter), 학습 데이터 요구사항, 추론 비용, 그리고 평가 프로토콜(LPIPS, FID, 작업 특화).

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 수정해 세 번째 클래스를 추가하라. G가 여전히 각 클래스의 잡음을 올바른 모드로 매핑하는지 확인하라.
2. **보통.** 1차원 설정에서 L1을 지각(perceptual) 스타일 손실로 교체하라(예: 특성 추출기로 작동하는 작은 동결(frozen) D). 그것이 조건부 분포의 선명도를 바꾸는가?
3. **어려움.** 1차원 설정에서 CycleGAN을 스케치하라: 두 분포, 두 생성기, 순환 손실. 짝지어진 데이터 없이 그것들 사이를 매핑하도록 학습함을 보여라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 조건부 GAN(Conditional GAN) | "레이블을 가진 GAN" | G(z, c), D(x, c). 두 네트워크 모두 조건을 본다. |
| Pix2Pix | "이미지-이미지 GAN" | U-Net G와 PatchGAN D + L1 손실을 가진 짝지어진 cGAN. |
| U-Net | "스킵을 가진 인코더-디코더" | 대칭 합성곱 네트워크; 스킵이 고주파를 보존. |
| PatchGAN | "국소 사실성 분류기" | D가 전역 점수 대신 패치별 점수를 출력. |
| CycleGAN | "짝 없는 이미지 변환" | 두 G + 순환 일관성 손실; 짝지어진 데이터 없음. |
| SPADE | "GauGAN" | 의미 맵으로 중간 활성값을 정규화; 분할-이미지 변환. |
| FiLM | "특성별 선형 변조(feature-wise linear modulation)" | 조건으로부터의 특성별 아핀 변환; 저렴한 조건화. |

## 프로덕션 노트: 지연 시간에 제약된 베이스라인으로서의 Pix2Pix (Production note: Pix2Pix as a latency-bound baseline)

짝지어진 데이터와 좁은 작업(스케치 → 렌더, 의미 맵 → 사진, 낮 → 밤)이 있을 때, Pix2Pix의 단발(one-shot) 추론은 지연 시간(latency)에서 확산을 한 자릿수 차이로 능가한다. 프로덕션 비교는 보통 이렇다.

| 경로 | 스텝 | 단일 L4에서 512²의 일반적 지연 시간 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net 순방향) | 1 | ~30 ms |
| SD-Inpaint 또는 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix는 정적 배치에서 처리량(throughput)으로 이긴다(모든 요청이 같은 FLOPs다). 확산은 품질과 일반화로 이긴다. 현대적 전략은 좁은 작업에는 Pix2Pix 스타일의 증류된 모델을, 꼬리(tail) 입력에는 확산 폴백(fallback)을 함께 출시하는 경우가 많다.

## 더 읽을거리 (Further Reading)

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784): cGAN 논문.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637): 투영 D.
