# Flow Matching & Rectified Flows

> 디퓨전(diffusion) 모델은 노이즈에서 데이터까지 굽은 경로를 걷기 때문에 20-50번의 샘플링(sampling) 스텝이 필요하다. 플로 매칭(flow matching)(Lipman et al., 2023)과 정류 흐름(rectified flow)(Liu et al., 2022)은 곧은 경로를 학습했다. 더 곧은 경로는 더 적은 스텝을, 더 적은 스텝은 더 빠른 추론(inference)을 뜻한다. Stable Diffusion 3, Flux.1, AudioCraft 2는 모두 2024년에 플로 매칭으로 전환했다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45분

## 문제 (The Problem)

DDPM의 역방향 과정은 `N(0, I)`에서 데이터 분포로 되돌아가는 1000스텝의 확률적 걸음이다. DDIM은 이를 20-50번의 결정론적 스텝으로 줄였다. 목표는 더 적은 스텝, 이상적으로는 한 스텝이다. 걸림돌은 역방향 과정을 푸는 ODE가 뻣뻣하다(stiff)는 데 있다. 경로가 굽어 있다.

노이즈에서 데이터까지의 경로가 *직선*이 되도록 모델을 학습(training)시킬 수 있다면, `t=1`에서 `t=0`까지의 단일 오일러(Euler) 스텝이면 충분하다. 플로 매칭은 이것을 직접 구성한다. `x_1 ∼ N(0, I)`에서 `x_0 ∼ data`로의 직선 보간을 정의하고, 그 시간 도함수(derivative)에 맞도록 벡터장(vector field) `v_θ(x, t)`를 학습시키고, 추론 시 적분한다.

정류 흐름(Liu 2022)은 한 걸음 더 나아간다. 점진적으로 선형에 가까워지는 ODE를 만들어내는 리플로(reflow) 절차로 경로를 반복적으로 곧게 편다. 두 번의 리플로 반복 후, 2스텝 샘플러가 50스텝 DDPM 품질에 맞먹는다.

## 개념 (The Concept)

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 직선 흐름 (Straight-line flow)

다음을 정의한다.

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

여기서 `x_0 ~ data`이고 `x_1 ~ N(0, I)`이다. 이 직선을 따른 시간 도함수는 상수다.

```
dx_t / dt = x_1 - x_0
```

신경 벡터장 `v_θ(x_t, t)`를 정의하고 이 도함수에 맞도록 학습시킨다.

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

이것이 **조건부 플로 매칭(conditional flow matching)** 손실(loss)이다(Lipman 2023). 학습은 시뮬레이션이 필요 없다(simulation-free). ODE를 풀어 전개하지 않는다. 그저 `(x_0, x_1, t)`를 샘플링하고 회귀(regression)한다.

### 샘플링 (Sampling)

추론 시, 학습된 벡터장을 시간에 대해 *거꾸로* 적분한다.

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

`x_1 ~ N(0, I)`에서 시작해 `t=0`까지 오일러 스텝을 내려간다.

### 정류 흐름 (Liu 2022)

직선 흐름은 동작하지만 학습된 경로는 *실제로는 곧지 않다*. 많은 `x_0`이 같은 `x_1`으로 매핑될 수 있기 때문에 경로가 굽는다. 정류 흐름의 리플로 스텝은 다음과 같다.

1. 무작위 짝짓기로 흐름 모델 v_1을 학습한다.
2. `x_1`에서 그 도착점 `x_0`까지 v_1을 적분해 N개의 짝 `(x_1, x_0)`을 샘플링한다.
3. 그 짝지어진 예제로 v_2를 학습한다. 이제 짝이 "ODE로 매칭"되었기 때문에, 짝 사이를 잇는 직선 보간자가 진짜로 더 평평하다.
4. 반복한다.

실제로 2번의 리플로 반복이면 거의 선형에 도달해 2-4스텝 추론을 가능하게 한다. SDXL-Turbo, SD3-Turbo, LCM은 모두 플로 매칭에서 증류(distillation)된 모델이다.

### 왜 이것이 2024년 이미지에서 이겼는가

세 가지 이유다.

1. **시뮬레이션 없는 학습**: 학습 중 ODE 전개가 없어 구현이 자명하다.
2. **더 나은 손실 기하**: 곧은 경로는 일관된 신호 대 잡음비(SNR)를 갖는 반면, DDPM ε-손실은 스케줄 가장자리에서 SNR이 나쁘다.
3. **더 빠른 추론**: SDXL-Turbo 품질에서 4-8스텝, 일관성 증류로 1스텝.

## 플로 매칭 대 DDPM: 정확한 연결

가우시안 조건부 경로를 쓰는 플로 매칭은 *특정 노이즈 스케줄을 가진* 디퓨전이다. `x_t = α(t) x_0 + σ(t) x_1` 스케줄을 고르면 플로 매칭은 `v = α'·x_0 - σ'·x_1`를 가진 스트라토노비치(Stratonovich) 재정식화된 디퓨전을 복원한다. 가우시안 경로에 대해 둘은 대수적으로 동등하다.

플로 매칭이 더한 것: 타깃의 *명료함*(평범한 속도), 더 깨끗한 손실, 그리고 비가우시안 보간자로 실험할 수 있는 자유.

```figure
normalizing-flow
```

## 직접 만들기 (Build It)

`code/main.py`는 두 모드 가우시안 혼합에 대한 1차원 플로 매칭을 구현한다. 벡터장 `v_θ(x, t)`는 직선 타깃으로 학습된 작은 MLP다. 추론 시 1, 2, 4, 20번의 오일러 스텝으로 적분하고 샘플 품질을 비교한다.

### Step 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts

4스텝 샘플러가 이미 20스텝 품질에 맞먹을 것으로 기대하라. 지연 시간(latency)을 크게 좌우하는 차이다.

## 함정 (Pitfalls)

- **시간 매개변수화.** 플로 매칭은 `t=0`이 데이터, `t=1`이 노이즈인 `t ∈ [0, 1]`을 쓴다. DDPM은 `t=0`이 데이터, `t=T`가 노이즈인 `t ∈ [0, T]`를 쓴다. 같은 방향, 다른 스케일이다. 논문들이 이것을 끊임없이 틀린다.
- **스케줄 선택.** 정류 흐름의 직선이 "그" 플로 매칭 스케줄이지만, 더 나은 스케일 커버리지를 위해 코사인 또는 로짓 정규(logit-normal) t-샘플링을 쓸 수 있다(SD3가 이렇게 한다).
- **리플로 비용.** 리플로용 짝지어진 데이터셋 생성은 샘플당 완전한 추론 패스다. 정말로 1-2스텝 추론이 필요할 때만 리플로하라.
- **분류기 없는 가이던스(classifier-free guidance)는 여전히 적용된다.** 선형 결합에서 ε을 v로 바꾸기만 하면 된다. `v_cfg = (1+w) v_cond - w v_uncond`.

## 라이브러리로 써보기 (Use It)

| 사용 사례 | 2026년 스택 |
|----------|-----------|
| 텍스트-투-이미지, 최고 품질 | 플로 매칭: SD3, Flux.1-dev |
| 텍스트-투-이미지, 1-4스텝 | 증류된 플로 매칭: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 실시간 추론 | 플로 매칭된 베이스로부터의 일관성 증류(LCM, PCM) |
| 오디오 생성 | 플로 매칭: Stable Audio 2.5, AudioCraft 2 |
| 영상 생성 | 디퓨전과 혼합된 플로 매칭(Sora, Veo, Stable Video) |
| 과학 / 물리(입자 궤적, 분자) | 플로 매칭 + 등변(equivariant) 벡터장 |

2025-2026년에 어떤 논문이 "디퓨전보다 빠르다"고 말하면, 거의 항상 플로 매칭 + 증류다.

## 산출물 (Ship It)

`outputs/skill-fm-tuner.md`를 저장한다. 이 스킬은 디퓨전 스타일 모델 명세를 받아 플로 매칭 학습 설정으로 변환한다. 스케줄 선택, 시간 샘플링 분포(균등 / 로짓 정규), 옵티마이저(optimizer), 리플로 계획, 목표 스텝 수, 평가 프로토콜.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하고 참 데이터 분포 대비 1스텝 대 20스텝 MSE를 비교하라.
2. **보통.** 균등 `t` 샘플링에서 로짓 정규(중간 t에 샘플링을 집중)로 바꿔라. 모델 품질이 향상되는가?
3. **어려움.** 한 번의 리플로 반복을 구현하라. 첫 모델을 적분해 짝지어진 (x_0, x_1)을 생성하고, 그 짝으로 두 번째 모델을 학습하고, 1스텝 샘플 품질을 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 플로 매칭(Flow matching) | "직선 디퓨전" | 보간자를 따라 `x_1 - x_0`에 맞도록 `v_θ(x, t)`를 학습. |
| 정류 흐름(Rectified flow) | "리플로" | 학습된 흐름을 곧게 펴는 반복 절차. |
| 속도장(Velocity field) | "v_θ" | 모델의 출력. `x_t`를 움직일 방향. |
| 직선 보간자(Straight-line interpolant) | "그 경로" | `x_t = (1-t)·x_0 + t·x_1`. 타깃 도함수가 자명하다. |
| 오일러 샘플러(Euler sampler) | "1차 ODE 솔버" | 가장 단순한 적분기. 경로가 곧을 때 잘 작동한다. |
| 로짓 정규 t(Logit-normal t) | "SD3 샘플링" | 그래디언트가 가장 강한 중간값 쪽으로 `t` 샘플링을 집중. |
| 일관성 증류(Consistency distillation) | "1스텝 샘플러" | 임의의 `x_t`를 `x_0`로 직접 매핑하도록 학생을 학습. |
| 속도를 쓰는 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`. 같은 트릭, 새 변수. |

## 프로덕션 노트: Flux.1-schnell은 가장 빠른 플로 매칭이다

플로 매칭의 프로덕션 승리는 Flux.1-schnell이다. Flux-dev급 품질을 유지하면서 1-4번의 추론 스텝으로 증류된 플로 매칭 DiT다. Niels의 "Run Flux on an 8GB machine" 노트북이 참조 배포(deployment) 레시피다. T5 + CLIP 인코드, 양자화된 MMDiT 디노이즈(schnell은 4스텝, dev는 50스텝), VAE 디코드. 비용 회계는 다음과 같다.

| 변종 | 스텝 | L4에서 1024² 지연 시간 | 총 FLOPs(상대) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15초 | 1.0× |
| Flux.1-schnell | 4 | ~1.2초 | 0.08× (12배 빠름) |
| SDXL-base | 30 | ~4초 | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3초 | 0.03× |

프로덕션 규칙: **플로 매칭된 베이스 + 증류 = 빠른 텍스트-투-이미지를 위한 2026년 기본값.** 모든 주요 벤더가 이 조합을 내보낸다. SD3-Turbo(SD3 + 플로 + 증류), Flux-schnell(Flux-dev + 정류 흐름 곧게 펴기), CogView-4-Flash. 순수 디퓨전 베이스는 레거시 체크포인트에만 존재한다.

## 더 읽을거리 (Further Reading)

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003): 정류 흐름.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): 플로 매칭.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206): SD3, 대규모 정류 흐름.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797): FM + 디퓨전을 포괄하는 일반 프레임워크.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469): 디퓨전 / 흐름의 1스텝 증류.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042): 터보 변종.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/): 프로덕션의 플로 매칭.
