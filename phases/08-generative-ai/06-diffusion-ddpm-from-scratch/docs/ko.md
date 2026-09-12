# 확산 모델(밑바닥부터 만드는 DDPM (Diffusion Models) DDPM from Scratch)

> Ho, Jain, Abbeel (2020)은 이 분야에 두고두고 쓰일 레시피를 남겼다. 천 번의 작은 스텝에 걸쳐 잡음으로 데이터를 파괴하라. 하나의 신경망(neural network)이 그 잡음을 예측하도록 학습시켜라. 추론(inference)에서 그 과정을 역전시켜라. 오늘날 모든 주류 이미지, 비디오, 3D, 음악 모델이 이 루프 위에서 돌아가며, 위에 흐름 매칭(flow matching)이나 일관성(consistency) 트릭이 얹혀 있을 수도 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75분

## 문제 (The Problem)

목표는 `p_data(x)`의 샘플러(sampler)다. GAN은 종종 발산하는 미니맥스(minimax) 게임을 한다. VAE는 가우시안 디코더(decoder)에서 흐릿한 샘플을 만든다. 정작 필요한 것은 (a) 단일하고 안정적인 손실(loss)(안장점도, 미니맥스도 없음)이고, (b) `log p(x)`의 하한(lower bound)이며(그래서 가능도(likelihood)를 가짐), (c) SOTA 품질에 맞는 샘플을 주는 학습 목적함수다.

Sohl-Dickstein et al. (2015)은 이론적 답을 갖고 있었다: 가우시안 잡음을 점진적으로 더하는 마르코프 연쇄(Markov chain) `q(x_t | x_{t-1})`를 정의하고, 잡음을 제거하도록 역방향 연쇄 `p_θ(x_{t-1} | x_t)`를 학습시킨다. Ho, Jain, Abbeel (2020)은 손실이 한 줄로 단순화될 수 있음을 보였고(잡음을 예측하라) 수학을 정리했다. 2020년에 이것은 호기심거리였다. 2021년에는 최첨단 샘플을 만들었다. 2022년에는 Stable Diffusion이 되었다. 2026년에는 토대(substrate)다.

## 개념 (The Concept)

![DDPM: 순방향 잡음, 역방향 잡음 제거](../assets/ddpm.svg)

**순방향 과정 `q`.** `T`개의 작은 스텝에서 가우시안 잡음을 더한다. 닫힌 형태(수학이 다루기 쉬운 이유)는 누적 스텝도 가우시안이라는 점이다.

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

여기서 `β_t`의 스케줄에 대해 `α̅_t = ∏_{s=1..t} (1 - β_s)`이다. `β_t`를 T=1000 스텝에 걸쳐 1e-4에서 0.02까지 선형으로 고르면 `x_T`는 근사적으로 `N(0, I)`다.

**역방향 과정 `p_θ`.** 더해진 잡음을 예측하는 신경망 `ε_θ(x_t, t)`를 학습한다. `x_t`가 주어지면 다음으로 잡음을 제거한다.

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

여기서 `σ_t`는 `sqrt(β_t)`이거나 학습된 분산이다. 이 식은 보기 흉하지만 그저 대수일 뿐이다. 사후 분포(posterior) `q(x_{t-1} | x_t, x_0)`가 주어졌을 때 `x_{t-1}`을 풀고, `x_0`을 그 잡음 예측 추정값으로 대체한 것이다.

**학습 손실.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

데이터에서 `x_0`을 샘플링하고, 임의의 `t`를 고르고, `ε ~ N(0, I)`를 샘플링하고, 닫힌 형태로 잡음 섞인 `x_t`를 단번에 계산하고, 잡음에 대해 회귀(regress)한다. 하나의 손실, 미니맥스 없음, KL 없음, 재매개변수화(reparameterization) 트릭 없음.

**샘플링.** `x_T ~ N(0, I)`에서 시작한다. `t = T`에서 `1`까지 역방향 스텝을 반복한다. 끝.

## 왜 작동하는가 (Why it works)

세 가지 직관:

1. **잡음 제거는 쉽고; 생성은 어렵다.** `t=T`에서 데이터는 순수 잡음이다. 네트워크는 사소한 문제를 풀면 된다. `t=0`에서 네트워크는 픽셀 몇 개만 정리하면 된다. 중간 `t`에서는 문제가 어렵지만, 네트워크는 모든 잡음 수준에서 같은 가중치(weight)를 통해 흐르는 많은 그래디언트(gradient)를 갖는다.

2. **변장한 점수 매칭(score matching).** Vincent (2011)은 잡음을 예측하는 것이 `∇_x log q(x_t | x_0)`, 즉 *점수(score)*를 추정하는 것과 동등함을 증명했다. 역방향 SDE는 이 점수를 사용해 밀도(density) 그래디언트를 따라 올라간다. 고확률 영역을 향한 안내된 무작위 보행이다.

3. **ELBO가 단순 MSE로 줄어든다.** 전체 변분 하한(variational lower bound)에는 타임스텝당 KL 항이 있다. DDPM의 매개변수화로 그 KL 항들은 특정 계수를 가진 잡음 예측에 대한 MSE로 단순화된다; Ho는 그 계수를 버렸고("simple" 손실이라 부름) 품질이 *향상되었다*.

```figure
diffusion-denoise
```

## 직접 만들기 (Build It)

`code/main.py`는 1차원 DDPM을 구현한다. 데이터는 2개 모드 혼합(mixture)이다. "네트워크"는 `(x_t, t)`를 받아 예측 잡음을 출력하는 아주 작은 MLP다. 학습은 한 줄짜리 손실이다. 샘플링은 역방향 연쇄를 반복한다.

### 1단계: 순방향 스케줄(닫힌 형태)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 2단계: `x_t`를 단번에 샘플링

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 3단계: 한 번의 학습 스텝

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 4단계: 역방향 샘플링

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

40개 타임스텝과 24유닛 MLP를 가진 1차원 문제에서, 이것은 약 200 에폭(epoch)에 2개 모드 혼합을 학습한다.

## 시간 조건화 (Time conditioning)

네트워크는 자신이 어느 타임스텝의 잡음을 제거하는지 알아야 한다. 두 가지 표준 옵션:

- **사인파 임베딩(Sinusoidal embedding).** 트랜스포머(Transformer) 위치 인코딩과 비슷하다. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. MLP를 통과시켜 네트워크에 브로드캐스트한다.
- **Film / 그룹 정규화(group-norm) 조건화.** 임베딩을 각 블록에서 채널별 스케일/바이어스(FiLM)로 투영한다.

우리 장난감 코드는 사인파 → 연결을 사용한다. 프로덕션(production) U-Net은 FiLM을 사용한다.

## 함정 (Pitfalls)

- **스케줄이 아주 중요하다.** 선형 `β`가 DDPM 기본값이지만 코사인 스케줄(Nichol & Dhariwal, 2021)이 같은 연산량에서 더 나은 FID를 준다. 품질이 정체되면 스케줄을 바꿔라.
- **타임스텝 임베딩은 깨지기 쉽다.** 원시 `t`를 부동소수점으로 전달하면 장난감 1차원에는 작동하지만 이미지에는 실패한다; 항상 적절한 임베딩을 사용하라.
- **V-예측 대 ε-예측.** 좁은 영역(아주 작거나 아주 큰 t)에서는 `ε`의 신호 대 잡음비가 나쁘다. V-예측(`v = α·ε - σ·x`)이 더 안정적이다; SDXL, SD3, Flux가 이것을 사용한다.
- **분류기 없는 안내(Classifier-free guidance).** 추론 시 조건부와 무조건 `ε`을 모두 계산한 다음, `w ≈ 3-7`로 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`를 한다. Lesson 08에서 다룬다.
- **1000 스텝은 많다.** 프로덕션은 DDIM(20-50 스텝), DPM-Solver(10-20 스텝), 또는 증류(distillation)(1-4 스텝)를 사용한다. Lesson 12 참조.

## 라이브러리로 써보기 (Use It)

| 역할 | 2026년의 일반적 스택 |
|------|-----------------------|
| 이미지 픽셀 공간 확산(작은, 장난감) | DDPM + U-Net |
| 이미지 잠재 확산(latent diffusion) | VAE 인코더 + U-Net 또는 DiT (Lesson 07) |
| 비디오 잠재 확산 | 시공간 DiT (Sora, Veo, WAN) |
| 오디오 잠재 확산 | Encodec + 확산 트랜스포머 |
| 과학(분자, 단백질, 물리) | 등변(equivariant) 확산 (EDM, RFdiffusion, AlphaFold3) |

확산은 보편적인 생성 백본(backbone)이다. 흐름 매칭(Lesson 13)은 2024-2026년의 경쟁자로, 같은 품질에서 추론 속도로 보통 이긴다.

## 산출물 (Ship It)

`outputs/skill-diffusion-trainer.md`로 저장하라. 이 스킬은 데이터셋(dataset) + 연산 예산을 받아 다음을 출력한다: 스케줄(선형/코사인/시그모이드), 예측 목표(ε/v/x), 스텝 수, 안내 스케일, 샘플러 계열, 그리고 평가 프로토콜.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 T를 40에서 10으로 바꿔라. 샘플 품질(출력의 시각적 히스토그램)이 어떻게 저하되는가? 어느 T에서 2개 모드 구조가 붕괴하는가?
2. **보통.** ε-예측에서 v-예측으로 전환하라. 역방향 스텝을 다시 유도하라. 최종 샘플 품질을 비교하라.
3. **어려움.** 분류기 없는 안내를 추가하라. 클래스 레이블(label) `c ∈ {0, 1}`에 조건화하고, 학습 중 10%는 그것을 버리며, 샘플링 시 `ε = (1+w)·ε_cond - w·ε_uncond`를 사용하라. `w = 0, 1, 3, 7`에서 조건부 모드 적중률을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 순방향 과정(Forward process) | "잡음 더하기" | 데이터를 파괴하는 고정된 마르코프 연쇄 `q(x_t \| x_{t-1})`. |
| 역방향 과정(Reverse process) | "잡음 제거" | 데이터를 재구성하는 학습된 연쇄 `p_θ(x_{t-1} \| x_t)`. |
| β 스케줄 | "잡음 사다리" | 스텝별 분산; 선형, 코사인, 또는 시그모이드. |
| α̅ | "알파 바(Alpha bar)" | 누적곱 `∏(1 - β)`; `x_0`에서 닫힌 형태 `x_t`를 준다. |
| 단순 손실(Simple loss) | "잡음에 대한 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`; 모든 변분 유도가 이것으로 붕괴한다. |
| ε-예측 | "잡음 예측" | 출력이 더해진 잡음; 표준 DDPM. |
| V-예측 | "속도(velocity) 예측" | 출력이 `α·ε - σ·x`; t 전반에 걸쳐 더 나은 조건화. |
| DDPM | "그 논문" | Ho et al. 2020; 선형 β, 1000 스텝, U-Net. |
| DDIM | "결정론적 샘플러" | 비마르코프 샘플러, 20-50 스텝, 같은 학습 목적함수. |
| 분류기 없는 안내(Classifier-free guidance) | "CFG" | 조건부와 무조건 잡음 예측을 섞어 조건화를 증폭한다. |

## 프로덕션 노트: 확산 추론은 스텝 수 문제다 (Production note: diffusion inference is a step-count problem)

DDPM 논문은 T=1000 역방향 스텝을 돌린다. 아무도 프로덕션에서 그것을 출시하지 않는다. 모든 실제 추론 스택은 세 전략 중 하나를 고른다. 그리고 각각은 "지연 시간이 어디서 오는가"라는 프로덕션 프레이밍에 깔끔하게 대응된다.

1. **더 빠른 샘플러, 같은 모델.** DDIM(20-50 스텝), DPM-Solver++(10-20), UniPC(8-16). 역방향 루프의 드롭인(drop-in) 교체; 학습된 `ε_θ` 가중치는 건드리지 않는다. 지연 시간을 20-50배 줄인다.
2. **증류(Distillation).** 학생(student)이 더 적은 스텝으로 교사(teacher)를 맞추도록 학습시킨다: 점진적 증류(Progressive Distillation)(2 → 1), 일관성 모델(Consistency Models)(임의 → 1-4), LCM, SDXL-Turbo, SD3-Turbo. 지연 시간을 추가로 5-10배 줄이며, 재학습이 필요하다.
3. **캐싱과 컴파일.** `torch.compile(unet, mode="reduce-overhead")`, TensorRT-LLM의 확산 백엔드, `xformers`/SDPA 어텐션, bf16 가중치. 스텝당 지연 시간을 약 2배 줄인다. (1)과 (2)와 누적된다.

프로덕션 확산 서버의 예산 대화는 프로덕션 문헌이 LLM에 대해 설명하는 것과 같다: 지연 시간은 `num_steps × step_cost + VAE_decode`이고, 처리량(throughput)은 `batch_size × (num_steps × step_cost)^-1`이다. TTFT는 작고(한 스텝); 사용자 관점에서 이미지 생성이 "한꺼번에" 이루어지므로 TPOT에 해당하는 것이 전체 응답 시간이다.

## 더 읽을거리 (Further Reading)

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585): 시대를 앞선 확산 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502): DDIM, 더 적은 스텝.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672): 코사인 스케줄, 학습된 분산.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233): 분류기 안내.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364): 통일된 표기, 가장 깔끔한 레시피.
