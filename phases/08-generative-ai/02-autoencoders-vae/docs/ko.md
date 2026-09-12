# 오토인코더와 변분 오토인코더 (Autoencoders & Variational Autoencoders, VAE)

> 평범한 오토인코더(autoencoder)는 압축한 다음 재구성한다. 그것은 암기한다. 생성하지는 않는다. 한 가지 트릭(코드(code)가 가우시안처럼 보이도록 강제하는 것)을 더하면 샘플러(sampler)를 얻는다. `z = μ + σ·ε`의 재매개변수화(reparameterization)라는 그 단 하나의 트릭이, 2026년에 쓰이는 모든 잠재 확산(latent-diffusion)과 흐름 매칭(flow-matching) 이미지 모델이 입력단에 VAE를 두는 이유다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75분

## 문제 (The Problem)

784픽셀의 MNIST 숫자를 16개의 숫자로 된 코드로 압축한 다음 재구성하라. 평범한 오토인코더는 재구성 MSE에서는 만점을 받겠지만, 코드 공간은 울퉁불퉁한 엉망이다. 코드 공간에서 임의의 점을 골라 디코딩하면 잡음이 나온다. 샘플러가 없다. 차려입은 압축 모델일 뿐이다.

실제로 원하는 것은 이렇다. (a) 코드 공간이 샘플링할 수 있는 깨끗하고 매끄러운 분포(이를테면 등방성 가우시안(isotropic Gaussian) `N(0, I)`) 이고, (b) 어떤 샘플을 디코딩해도 그럴듯한 숫자가 나오며, (c) 인코더(encoder)와 디코더(decoder)가 여전히 잘 압축한다. 세 가지 목표, 하나의 아키텍처, 하나의 손실(loss)이다.

Kingma의 2013년 VAE는 이를 다음과 같이 해결한다. 인코더가 *분포* `q(z|x) = N(μ(x), σ(x)²)`를 출력하도록 학습시키고, KL 페널티로 그 분포를 사전 분포(prior) `N(0, I)` 쪽으로 끌어당기며, 디코딩하기 전에 `q(z|x)`에서 `z`를 샘플링한다. 추론(inference) 시에는 인코더를 버리고, `z ~ N(0, I)`를 샘플링한 뒤 디코딩한다. 코드 공간이 구조를 갖도록 강제하는 것이 바로 KL 페널티다.

2026년에 VAE는 단독으로 출시되는 경우가 드물다. 원본 이미지 품질에서는 확산(diffusion)에 압도당했다. 하지만 모든 잠재 확산 모델(SD 1/2/XL/3, Flux, AudioCraft)에서 선택받는 인코더다. VAE를 배우면 흔히 쓰는 모든 이미지 파이프라인의 보이지 않는 첫 번째 층(layer)을 배우는 셈이다.

## 개념 (The Concept)

![오토인코더 대 VAE: 재매개변수화 트릭](../assets/vae.svg)

**오토인코더.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. 코드 공간은 구조가 없다.

**VAE 인코더.** 두 개의 벡터를 출력한다: `μ(x)`와 `log σ²(x)`. 이들이 `q(z|x) = N(μ, diag(σ²))`를 정의한다.

**재매개변수화 트릭(Reparameterization trick).** `q(z|x)`에서 샘플링하는 것은 미분 불가능하다. 샘플을 `z = μ + σ·ε`로 다시 쓰는데, 여기서 `ε ~ N(0, I)`이다. 이제 `z`는 `(μ, σ)`의 결정론적 함수에 비파라미터 잡음을 더한 것이다. 그래디언트(gradient)가 `μ`와 `σ`를 통해 흐른다.

**손실.** 증거 하한(Evidence Lower BOund, ELBO), 두 항이다.

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

재구성 항은 `x̂`를 `x` 쪽으로 민다. KL 항은 `q(z|x)`를 사전 분포 쪽으로 민다. 이 둘은 트레이드오프(trade-off) 관계다. 작은 β (<1) = 더 선명한 샘플, 덜 가우시안적인 코드 공간. 큰 β (>1) = 더 깨끗한 코드 공간, 더 흐릿한 샘플. β-VAE (Higgins 2017)는 이 조절 손잡이를 유명하게 만들었고 풀림(disentanglement) 연구의 물꼬를 텄다.

**샘플링.** 추론 시: `z ~ N(0, I)`를 그려 디코더를 통해 순방향한다. 순방향 패스(forward pass) 한 번: 확산처럼 반복적인 샘플링이 없다.

```figure
vae-latent-grid
```

## 직접 만들기 (Build It)

`code/main.py`는 numpy나 torch 없이 아주 작은 VAE를 구현한다. 입력은 8차원 공간의 2성분 가우시안 혼합(Gaussian mixture)에서 추출된 8차원 합성 데이터다. 인코더와 디코더는 단일 은닉층(hidden layer) MLP다. tanh 활성화(activation), 순방향 패스, 손실, 그리고 직접 작성한 역방향 패스(backward pass)를 구현한다. 프로덕션(production)이 아니라 교육용이다.

### 1단계: 인코더 순방향

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

네트워크 출력을 제약 없이 두기 위해 `σ` 대신 `log σ²`를 쓴다(σ의 softplus는 함정이다. σ ≈ 0에서 그래디언트가 죽는다).

### 2단계: 재매개변수화와 디코딩

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 3단계: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

두 분포가 모두 가우시안이므로 정확한 닫힌 형태의 KL이다. 수치적으로 적분하지 마라. 사람들은 2026년에도 여전히 몬테카를로(monte-carlo) KL 추정값을 쓰는 코드를 출시한다. 아무 이유도 없이 3배 느리다.

### 4단계: 생성

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

그것이 생성 모델이다. 다섯 줄.

## 함정 (Pitfalls)

- **사후 분포 붕괴(Posterior collapse).** KL 항이 `q(z|x) → N(0, I)`를 너무 공격적으로 몰아붙여 `z`가 `x`에 대한 정보를 전혀 담지 못한다. 해결책: β-어닐링(β=0에서 시작해 1까지 점진적으로 올림), 자유 비트(free bits), 또는 비활성 차원에서 KL 생략.
- **흐릿한 샘플.** 가우시안 디코더 가능도(likelihood)는 MSE 재구성을 함의하는데, 이는 L2에 대해 베이즈 최적(평균)이다. 그럴듯한 숫자들의 집합의 평균은 흐릿한 숫자다. 해결책: 이산 디코더(VQ-VAE, NVAE), 또는 VAE를 인코더로만 쓰고 잠재(latent) 위에 확산을 쌓기(이것이 Stable Diffusion이 하는 것이다).
- **β가 너무 크고, 너무 이름.** 사후 분포 붕괴를 보라. β≈0.01에서 시작해 점진적으로 올려라.
- **잠재 차원이 너무 작음.** 16차원은 MNIST에 적합하고, 256차원은 ImageNet 256², 2048차원은 ImageNet 1024²에 적합하다. Stable Diffusion의 VAE는 512×512×3 → 64×64×4로 압축한다(공간 면적에서 32배, 채널에서 32배 다운샘플 인자).

## 라이브러리로 써보기 (Use It)

2026년 VAE 스택:

| 상황 | 선택 |
|-----------|------|
| 확산용 이미지-잠재 인코더 | Stable Diffusion VAE (`sd-vae-ft-ema`) 또는 Flux VAE |
| 오디오-잠재 인코더 | Encodec (Meta), SoundStream, 또는 DAC (Descript) |
| 비디오 잠재 | Sora의 시공간 패치, Latte VAE, WAN VAE |
| 풀린(disentangled) 표현 학습 | β-VAE, FactorVAE, TCVAE |
| 이산 잠재(트랜스포머 모델링용) | VQ-VAE, RVQ (ResidualVQ) |
| 생성을 위한 연속 잠재 | 평범한 VAE, 그다음 그 잠재 공간에서 흐름/확산 모델 조건화 |

잠재 확산 모델은 인코더와 디코더 사이에 확산 모델이 사는 VAE다. VAE는 거친 압축을 하고, 확산 모델은 무거운 일을 한다. 비디오(VAE + 비디오 확산 DiT)와 오디오(Encodec + MusicGen 트랜스포머)도 같은 패턴이다.

## 산출물 (Ship It)

`outputs/skill-vae-trainer.md`로 저장하라.

이 스킬은 데이터셋(dataset) 프로파일 + 잠재 차원 목표 + 하위 작업 용도(재구성, 샘플링, 또는 잠재 확산 입력)를 받아 다음을 출력한다: 아키텍처 선택(plain/β/VQ/RVQ), β 스케줄, 잠재 차원, 디코더 가능도(가우시안 대 범주형), 그리고 평가 계획(재구성 MSE, 차원당 KL, `q(z|x)`와 `N(0, I)` 사이의 프레셰 거리(Fréchet distance)).

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`의 `β`를 `0.01`, `0.1`, `1.0`, `5.0`으로 바꿔라. 최종 재구성 MSE와 KL을 기록하라. 이 합성 데이터에서 어떤 β가 파레토 최적(Pareto-best)인가?
2. **보통.** 가우시안 디코더 가능도를 베르누이(Bernoulli) 가능도(교차 엔트로피 손실)로 교체하라. 같은 합성 데이터의 이진화 버전에서 샘플 품질을 비교하라.
3. **어려움.** `code/main.py`를 미니 VQ-VAE로 확장하라: 연속 `z`를 K=32개 항목의 코드북(codebook)에서의 최근접 이웃 탐색으로 교체하라. 재구성 MSE를 비교하고 코드북 항목이 몇 개나 사용되는지 보고하라(코드북 붕괴는 실재한다).

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 오토인코더(Autoencoder) | 인코딩-디코딩 네트워크 | `x → z → x̂`, MSE를 학습. 생성적이지 않음. |
| VAE | 샘플러를 가진 AE | 인코더가 분포를 출력하고, KL 페널티가 코드 공간을 빚는다. |
| ELBO | 증거 하한 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; `q = p(z\|x)`일 때 빡빡함(tight). |
| 재매개변수화(Reparameterization) | `z = μ + σ·ε` | 확률적 노드를 결정론적 + 순수 잡음으로 다시 쓴다. 샘플링을 통한 역전파를 가능하게 한다. |
| 사전 분포(Prior) | `p(z)` | 잠재의 목표 분포, 보통 `N(0, I)`. |
| 사후 분포 붕괴(Posterior collapse) | "KL 항이 이긴다" | 인코더가 `x`를 무시하고 사전 분포를 출력한다; 디코더가 환각을 일으켜야 한다. |
| β-VAE | 조정 가능한 KL 가중치 | `loss = recon + β·KL`. β가 높을수록 더 풀리지만 더 흐릿하다. |
| VQ-VAE | 이산 잠재 | 연속 `z`를 최근접 코드북 벡터로 교체; 트랜스포머 모델링을 가능하게 한다. |

## 프로덕션 노트: VAE는 확산 서버에서 가장 뜨거운 경로다 (Production note: the VAE is the hottest path in a diffusion server)

Stable Diffusion / Flux / SD3 파이프라인에서 VAE는 요청당 두 번 호출된다. 한 번은 인코딩(img2img / 인페인팅(inpainting)을 한다면), 한 번은 디코딩이다. 1024²에서 디코더 패스는 종종 파이프라인 전체에서 단일 최대 활성값 메모리 정점인데, `128×128×16` 잠재를 다시 `1024×1024×3`으로 업샘플(upsample)하기 때문이다. 두 가지 실용적 결과가 따른다.

- **디코드를 슬라이스하거나 타일링하라.** `diffusers`는 `pipe.vae.enable_slicing()`과 `pipe.vae.enable_tiling()`을 노출한다. 타일링은 작은 이음새 아티팩트를 `O(H·W)` 대신 `O(tile²)` 메모리와 맞바꾼다. 소비자용 GPU에서 1024² 이상에 필수적이다.
- **bf16 디코더, 최종 리사이즈는 fp32 수치 연산.** SD 1.x VAE는 fp32로 출시되었고 1024² 이상에서 fp16으로 캐스팅하면 *조용히 NaN을 만든다*. SDXL은 `madebyollin/sdxl-vae-fp16-fix`를 함께 제공한다. 항상 fp16-fix 변형을 선호하거나 bf16을 써라.

## 더 읽을거리 (Further Reading)

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): VAE 논문.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl): 풀린 β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898): 최첨단 이미지 VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752): Stable Diffusion; 인코더로서의 VAE.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438): Encodec, 오디오 VAE 표준.
