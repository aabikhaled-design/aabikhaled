# GAN(생성기 대 판별기 (GANs) Generator vs Discriminator)

> 2014년 Goodfellow의 트릭은 밀도(density)를 아예 건너뛰는 것이었다. 두 개의 네트워크. 하나는 가짜를 만든다. 하나는 그것을 잡는다. 가짜가 진짜와 구별할 수 없을 때까지 싸운다. 작동해서는 안 된다. 종종 작동하지 않는다. 작동할 때, 그 샘플은 좁은 도메인에서 여전히 문헌상 가장 선명하다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75분

## 문제 (The Problem)

VAE는 흐릿한 샘플을 만드는데, 이는 MSE 디코더(decoder) 손실(loss)이 *평균* 이미지에 대해 베이즈 최적이기 때문이다. 그리고 그럴듯한 숫자 여럿의 평균은 흐릿한 숫자다. 필요한 것은 어느 한 목표값과의 픽셀 단위 근접성이 아니라 *그럴듯함(plausibility)*에 보상을 주는 손실이다. 그럴듯함에는 닫힌 형태가 없으니, 직접 학습해야 한다.

Goodfellow의 아이디어: 진짜 이미지와 가짜를 구별하도록 분류기(classifier) `D(x)`를 학습시킨다. `D`를 속이도록 생성기(generator) `G(z)`를 학습시킨다. `G`에 대한 손실 신호는, `D`가 지금 무엇이든 진짜처럼 보이게 만든다고 여기는 바로 그것이다. 이 신호는 `G`가 개선됨에 따라 갱신되며, 움직이는 목표를 쫓는다. 두 네트워크가 모두 수렴하면, `G`는 `log p(x)`를 한 번도 적지 않고 데이터 분포를 학습한 것이다.

이것이 적대적 학습(adversarial training)이다. 수학적으로는 미니맥스(minimax) 게임이다.

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026년에 GAN은 더 이상 SOTA 생성기가 아니다(확산(diffusion)과 흐름 매칭(flow matching)이 그 왕관을 차지했다). 하지만 StyleGAN 2/3은 지금까지 출시된 가장 선명한 얼굴 모델로 남아 있고, GAN 판별기(discriminator)는 확산 학습에서 *지각 손실(perceptual loss)*로 사용되며, 적대적 학습은 실시간 확산을 출시할 수 있게 해주는 빠른 1-스텝 증류(distillation)(SDXL-Turbo, SD3-Turbo, LCM)를 구동한다.

## 개념 (The Concept)

![GAN 학습: 미니맥스 안에서의 생성기와 판별기](../assets/gan.svg)

**생성기 `G(z)`.** 잡음 벡터 `z ~ N(0, I)`를 샘플 `x̂`로 매핑한다. 디코더 형태의 네트워크다(밀집 또는 전치 합성곱(transposed conv)).

**판별기 `D(x)`.** 샘플을 스칼라 확률(또는 점수)로 매핑한다. 진짜 → 1, 가짜 → 0.

**손실.** 두 개의 번갈아 하는 갱신:

- **`D` 학습:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. 진짜=1, 가짜=0에 대한 이진 교차 엔트로피.
- **`G` 학습:** `loss_G = -log D(G(z))`. 이것이 Goodfellow가 사용한 *비포화(non-saturating)* 형태다(원래의 `log(1 - D(G(z)))`는 `D`가 확신할 때 포화(saturation)하여 그래디언트(gradient)를 죽인다).

**학습 루프.** `D` 한 스텝, `G` 한 스텝. 반복.

**왜 작동하는가.** `G`가 `p_data`를 완벽하게 맞추면, `D`는 우연보다 잘할 수 없고 모든 곳에서 0.5를 출력한다; `G`는 더 이상 그래디언트를 받지 못한다. 평형(equilibrium).

**왜 무너지는가.** 모드 붕괴(mode collapse)(`G`가 `D`가 분류할 수 없는 한 모드를 찾아 영원히 찍어낸다), 기울기 소실(vanishing gradient)(`D`가 너무 빨리 학습해 `log D`가 포화한다), 학습 불안정성(학습률(learning rate), 배치(batch) 크기, 무엇이든).

## GAN을 작동하게 만든 변형들 (Variants that made GANs work)

| 연도 | 혁신 | 해결한 것 |
|------|------------|-----|
| 2015 | DCGAN | 합성곱/역합성곱, 배치 정규화(batch norm), LeakyReLU: 최초의 안정적인 아키텍처. |
| 2017 | WGAN, WGAN-GP | BCE를 바서슈타인 거리(Wasserstein distance) + 그래디언트 페널티로 교체. 기울기 소실을 해결. |
| 2017 | 스펙트럼 정규화(Spectral normalization) | 판별기를 립시츠 한계(Lipschitz-bound) 지음. 2026년 판별기에서도 여전히 사용됨. |
| 2018 | Progressive GAN | 저해상도부터 학습, 층(layer)을 추가. 최초의 메가픽셀 결과. |
| 2019 | StyleGAN / StyleGAN2 | 매핑 네트워크 + 적응형 인스턴스 정규화(adaptive instance norm). 고정 도메인 사실적 묘사의 최첨단. |
| 2021 | StyleGAN3 | 앨리어스 없음(alias-free), 평행이동 등변(translation-equivariant): 2026년에도 여전히 얼굴의 황금 표준. |
| 2022 | StyleGAN-XL | 조건부, 클래스 인지, 더 큰 규모. |
| 2024 | R3GAN | 더 강한 정규화(regularization)로 재브랜딩; 트릭 없이 1024²에서 작동. |

```figure
gan-minimax
```

## 직접 만들기 (Build It)

`code/main.py`는 1차원 데이터에서 아주 작은 GAN을 학습시킨다: 두 가우시안의 혼합(mixture). 생성기와 판별기는 단일 은닉층(hidden layer) MLP다. 순방향, 역방향, 그리고 미니맥스 루프를 직접 구현한다. 목표는 두 가지 핵심 실패 모드(모드 붕괴 + 기울기 소실)가 일어나는 것을 보는 것이다.

### 1단계: 비포화 손실

기본 Goodfellow 손실 `log(1 - D(G(z)))`는 D가 G의 가짜를 높은 확신으로 가짜라고 분류할 때 0으로 간다. 그 시점에 G에 대한 그래디언트는 기본적으로 0이다. G가 개선될 수 없다. 비포화 형태 `-log D(G(z))`는 반대의 점근선을 가진다: D가 확신할 때 폭발하여, G에 강한 신호를 준다.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 2단계: 생성기 스텝당 판별기 한 스텝

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G에는 새로 만든 가짜를 써야 한다. 그렇지 않으면 그래디언트가 낡아 버린다.

### 3단계: 모드 붕괴를 주시하라

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

전형적인 증상: 두 진짜 모드 중 하나가 더 이상 생성되지 않는다. 판별기는 그것을 가짜로 본 적이 없으므로 더 이상 교정하지 않는다.

## 함정 (Pitfalls)

- **판별기가 너무 강함.** D의 학습률을 2-5배 낮추거나, 인스턴스/층 잡음을 추가하라. D가 95% 이상의 정확도에 도달하면 G는 죽은 것이다.
- **생성기가 한 모드를 암기함.** D 입력에 잡음을 추가하거나, 미니배치 판별기(minibatch-discriminator) 층을 사용하거나, WGAN-GP로 전환하라.
- **배치 정규화가 통계를 누설함.** 진짜 배치 + 가짜 배치가 같은 BN 층을 통과하면 그 통계가 섞인다. 대신 인스턴스 정규화나 스펙트럼 정규화를 사용하라.
- **인셉션 점수(Inception-score) 조작.** FID와 IS는 샘플 수가 적을 때 잡음이 많다. 평가 시 1만 개 이상의 샘플을 사용하라.
- **단발 샘플링은 조건부 작업에서는 통하지 않는다.** 쓸 만한 출력을 얻으려면 여전히 CFG 스케일, 절단(truncation) 트릭, 재샘플링이 필요하다.

## 라이브러리로 써보기 (Use It)

2026년 GAN 스택:

| 상황 | 선택 |
|-----------|------|
| 사실적 인간 얼굴, 고정 포즈 | StyleGAN3 (가장 선명하고, 가장 작음) |
| 애니메이션 / 양식화된 얼굴 | StyleGAN-XL 또는 Stable Diffusion LoRA |
| 이미지-이미지 변환 | Pix2Pix / CycleGAN (Phase 8 · 04) 또는 ControlNet (Phase 8 · 08) |
| 빠른 1-스텝 텍스트-이미지 | 확산의 적대적 증류 (SDXL-Turbo, SD3-Turbo) |
| 확산 트레이너 내부의 지각 손실 | 이미지 크롭(crop)에 대한 작은 GAN 판별기 |
| 멀티모달, 개방형인 무엇이든 | 하지 마라. 확산이나 흐름 매칭을 써라 |

GAN은 선명하지만 좁다. 도메인이 열리면(사진, 임의의 텍스트 프롬프트(prompt), 비디오) 확산으로 전환하라. 적대적 트릭은 단독 생성기가 아니라 구성 요소(지각 손실, 증류)로 살아남는다.

## 산출물 (Ship It)

`outputs/skill-gan-debugger.md`로 저장하라. 이 스킬은 실패하는 GAN 실행(손실 곡선, 샘플 격자, 데이터셋 크기)을 받아 가능성 높은 원인의 순위 목록, 한 줄짜리 해결책, 그리고 재실행 프로토콜을 출력한다.

## 연습 문제 (Exercises)

1. **쉬움.** 기본 설정으로 `code/main.py`를 실행하라. 그다음 `D_LR = 5 * G_LR`로 설정하고 다시 실행하라. G의 손실이 상수로 붕괴하는 데 얼마나 빠른가?
2. **보통.** Goodfellow BCE 손실을 WGAN 손실로 교체하라: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, 그리고 D의 가중치(weight)를 `[-0.01, 0.01]`로 클리핑(clip)하라. 학습이 더 안정적인가? 벽시계 시간 수렴(convergence)을 비교하라.
3. **어려움.** 1차원 예제를 2차원 데이터(원 위의 8개 가우시안 혼합)로 확장하라. 1k, 5k, 10k 스텝에서 생성기가 8개 모드 중 몇 개를 포착하는지 추적하라. 미니배치 판별을 구현하고 다시 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 생성기(Generator) | "G" | 잡음-샘플 네트워크, `G: z → x̂`. |
| 판별기(Discriminator) | "D" | 분류기 `D: x → [0, 1]`, 진짜 대 가짜. |
| 미니맥스(Minimax) | "그 게임" | 결합 목적함수의 `min_G max_D`. |
| 비포화 손실(Non-saturating loss) | "그 해결책" | G에 대해 `log(1 - D(G(z)))` 대신 `-log D(G(z))`를 사용. |
| 모드 붕괴(Mode collapse) | "G가 한 가지를 암기함" | 다양한 데이터에도 불구하고 생성기가 적은 수의 서로 다른 출력만 만든다. |
| WGAN | "바서슈타인" | BCE를 어스무버 거리(Earth-Mover distance) + 그래디언트 페널티로 교체; 더 매끄러운 그래디언트. |
| 스펙트럼 정규화(Spectral norm) | "립시츠 트릭" | D의 가중치 노름을 제약해 기울기를 한계 지음; 학습을 안정화. |
| StyleGAN | "작동하는 그것" | 매핑 네트워크 + AdaIN; 얼굴에 대해 동급 최강, 2026년에도. |

## 프로덕션 노트: 단발 추론은 GAN의 지속되는 이점이다 (Production note: one-shot inference is GAN's lasting advantage)

GAN은 개방형 도메인 생성의 샘플 품질에서는 더 이상 이기지 못하지만, 추론(inference) 비용에서는 여전히 이긴다. 프로덕션 추론 문헌의 어휘로 보면 GAN은 다음을 가진다.

- **프리필(prefill)도, 디코드(decode) 단계도 없음.** 단일 `G(z)` 순방향 패스(forward pass). TTFT ≈ 전체 지연 시간(latency).
- **KV-cache 압박 없음.** 유일한 상태는 가중치다. 배치 크기는 캐시가 아니라 활성값 메모리로 한계 지어진다.
- **사소한 연속 배칭(continuous batching).** 모든 요청이 같은 고정 FLOPs를 차지하므로, 서버의 목표 점유율에서의 정적 배치가 보통 최적이다. 인플라이트(in-flight) 스케줄러가 필요 없다.

이것이 GAN 증류(SDXL-Turbo, SD3-Turbo, ADD, LCM)가 2026년 빠른 텍스트-이미지의 지배적인 기법인 이유다: 확산 기저의 분포를 유지하면서 20-50 스텝의 확산 파이프라인을 1-4개의 GAN 스타일 순방향 패스로 압축한다. 적대적 손실은 느린 생성기를 빠른 것으로 바꾸는 학습 시 조절 손잡이로 살아남는다.

## 더 읽을거리 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661): 원본 GAN 논문.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434): 최초의 안정적인 아키텍처.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
