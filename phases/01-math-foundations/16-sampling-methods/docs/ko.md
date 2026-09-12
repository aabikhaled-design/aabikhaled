# 샘플링 방법 (Sampling Methods)

> 샘플링(sampling)은 AI가 가능성의 공간을 탐색하는 방법이다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 균등 난수만 사용해 역 CDF, 기각(rejection), 중요도 샘플링(importance sampling)을 밑바닥부터 구현하기
- 언어 모델 토큰(token) 생성을 위한 온도(temperature), top-k, top-p(뉴클리어스) 샘플링 만들기
- 재매개변수화 트릭(reparameterization trick)이 VAE에서 샘플링을 통한 역전파(backpropagation)를 어떻게 가능하게 하는지 설명하기
- 정규화되지 않은 목표 분포에서 샘플링하기 위해 메트로폴리스-헤이스팅스(Metropolis-Hastings) MCMC 실행하기

## 문제 (The Problem)

언어 모델이 프롬프트(prompt) 처리를 마치고 50,000개의 로짓(logits) 벡터(vector)를 만든다. 어휘(vocabulary)의 모든 토큰에 하나씩. 이제 하나를 골라야 한다. 어떻게?

항상 가장 높은 확률의 토큰을 고르면 모든 응답이 똑같다. 결정적이고 지루하다. 균등하게 무작위로 고르면 출력은 횡설수설이 된다. 답은 이 양극단 사이 어딘가에 있고, 그 어딘가를 샘플링이 제어한다.

샘플링은 텍스트 생성에만 쓰이지 않는다. 강화 학습(reinforcement learning)은 궤적(trajectory)을 샘플링해 정책 그래디언트(policy gradient)를 추정한다. VAE는 학습된 분포(distribution)에서 샘플링하고 그 무작위성을 통해 역전파해 잠재 표현(latent representation)을 학습한다. 확산 모델(diffusion model)은 잡음을 샘플링하고 반복적으로 노이즈를 제거해 이미지를 생성한다. 몬테카를로(Monte Carlo) 방법은 닫힌 형식 해가 없는 적분을 추정한다. MCMC 알고리즘은 열거가 불가능한 고차원 사후 분포(posterior distribution)를 탐색한다.

모든 생성형 AI 시스템은 샘플링 시스템이다. 샘플링 전략이 출력의 품질, 다양성, 제어 가능성을 결정한다. 이 레슨은 모든 주요 샘플링 방법을 밑바닥부터 만든다. 균등 난수에서 시작해 현대 LLM과 생성 모델을 구동하는 기법으로 끝맺는다.

## 개념 (The Concept)

### 샘플링이 중요한 이유

샘플링은 AI와 머신러닝 전반에서 네 가지 근본적인 역할로 등장한다.

**생성(Generation).** 언어 모델, 확산 모델, GAN은 모두 샘플링으로 출력을 만든다. 샘플링 알고리즘이 창의성, 일관성, 다양성을 직접 제어한다. 온도, top-k, 뉴클리어스 샘플링은 엔지니어가 매일 돌리는 손잡이다.

**학습(Training).** 확률적 경사 하강법(stochastic gradient descent)은 미니배치(mini-batch)를 샘플링한다. 드롭아웃(dropout)은 비활성화할 뉴런(neuron)을 샘플링한다. 데이터 증강(data augmentation)은 무작위 변환을 샘플링한다. 중요도 샘플링은 강화 학습(PPO, TRPO)에서 그래디언트 분산을 줄이려고 샘플에 가중치를 재부여한다.

**추정(Estimation).** ML의 많은 양은 닫힌 형식 해가 없다. 데이터 분포에 대한 기대 손실(loss), 에너지 기반 모델의 분배 함수, 베이지안 추론에서의 증거(evidence)가 그렇다. 몬테카를로 추정은 샘플에 대한 평균으로 이 모든 것을 근사한다.

**탐색(Exploration).** MCMC 알고리즘은 베이지안 추론에서 사후 분포를 탐색한다. 진화 전략은 파라미터(parameter) 섭동을 샘플링한다. 톰슨 샘플링(Thompson sampling)은 밴딧에서 탐색과 활용의 균형을 맞춘다.

핵심 과제: 단순한 분포(균등, 정규)에서만 직접 샘플링할 수 있다. 그 외에는 모두 단순한 샘플을 목표 분포의 샘플로 바꾸는 방법이 필요하다.

### 균등 난수 샘플링

모든 샘플링 방법은 여기서 시작한다. 균등 난수 생성기는 [0, 1) 범위의 값을 만들며, 길이가 같은 모든 부분 구간이 같은 확률을 가진다.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

n개 항목의 이산 집합에서 균등하게 샘플링하려면 U를 생성하고 floor(n * U)를 반환한다. 연속 범위 [a, b]에서 샘플링하려면 a + (b - a) * U를 계산한다.

핵심 통찰: 하나의 균등 난수는 어떤 분포에서든 하나의 샘플을 만들기에 정확히 알맞은 양의 무작위성을 담고 있다. 비결은 올바른 변환을 찾는 것이다.

### 역 CDF 방법 (역변환 샘플링, Inverse Transform Sampling)

누적 분포 함수(cumulative distribution function, CDF)는 값을 확률로 매핑한다.

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

역 CDF는 확률을 다시 값으로 매핑한다. U ~ Uniform(0, 1)이면 X = F_inverse(U)는 목표 분포를 따른다.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**지수 분포 예시:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

이 방법은 F_inverse를 닫힌 형식으로 쓸 수 있을 때 완벽하게 동작한다. 정규 분포는 닫힌 형식의 역 CDF가 없으므로 다른 방법(Box-Muller 또는 수치 근사)을 쓴다.

**이산 버전:** 이산 분포는 CDF를 누적합으로 구성하고, U를 생성하고, 누적합이 U를 초과하는 첫 인덱스를 찾는다. Lesson 06의 `sample_categorical`이 바로 이렇게 동작한다.

### 기각 샘플링 (Rejection Sampling)

CDF를 역변환할 수는 없지만 목표 PDF를 상수 배까지 평가할 수 있을 때 기각 샘플링이 동작한다.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

경계 M이 빡빡할수록 수락률이 높다. 저차원(1-3)에서는 기각 샘플링이 잘 동작한다. 고차원에서는 제안 부피의 대부분이 기각되기 때문에 수락률이 지수적으로 떨어진다. 이것이 기각 샘플링의 차원의 저주다.

**예시: 절단 정규 분포에서 샘플링.** 절단된 범위에 대한 균등 제안을 쓴다. 포락선 M은 그 범위에서 정규 PDF의 최댓값이다.

**예시: 반원에서 샘플링.** 경계 사각형 안에서 균등하게 제안한다. 점이 반원 안에 떨어지면 수락한다. 몬테카를로가 pi를 계산하는 방법이 이것이다. 수락률은 면적 비율 pi/4와 같다.

### 중요도 샘플링 (Importance Sampling)

때로는 목표 분포 p(x)의 샘플이 필요하지 않다. p(x) 아래의 기댓값을 추정해야 하는데, 가진 것은 다른 분포 q(x)의 샘플뿐이다.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

이것은 강화 학습에서 결정적으로 중요하다. PPO(Proximal Policy Optimization)에서는 옛 정책 pi_old 아래에서 궤적을 수집하지만 최적화하려는 대상은 새 정책 pi_new다. 중요도 가중치는 pi_new(a|s) / pi_old(a|s)다. PPO는 새 정책이 옛 정책에서 너무 멀리 벗어나지 않도록 이 가중치를 클리핑한다.

중요도 샘플링 추정량의 분산은 q가 p와 얼마나 비슷한지에 달려 있다. q가 p와 매우 다르면 몇몇 샘플이 거대한 가중치를 얻어 추정치를 지배한다. 자기정규화 중요도 샘플링(self-normalized importance sampling)은 가중치의 합으로 나누어 이 문제를 줄인다.

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 몬테카를로 추정 (Monte Carlo Estimation)

몬테카를로 추정은 무작위 샘플을 평균해 적분을 근사한다. 큰 수의 법칙이 수렴(convergence)을 보장한다.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

오차율은 차원과 무관하다. 그래서 격자 기반 적분이 불가능한 고차원에서 몬테카를로 방법이 지배한다.

**pi 추정:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**기댓값 추정:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 마르코프 연쇄 몬테카를로 (MCMC): 메트로폴리스-헤이스팅스

MCMC는 정상 분포(stationary distribution)가 목표 분포 p(x)인 마르코프 연쇄(Markov chain)를 구성한다. 충분한 스텝 후, 연쇄의 샘플은 (근사적으로) p(x)의 샘플이 된다.

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

대칭 제안(q(x'|x) = q(x|x'))의 경우, 비율은 p(x')/p(x)로 단순해진다. 이것이 원래의 메트로폴리스 알고리즘이다.

**동작하는 이유.** 수락 규칙은 세부 균형(detailed balance)을 보장한다. x에 있다가 x'로 이동할 확률이 x'에 있다가 x로 이동할 확률과 같다. 세부 균형은 p(x)가 연쇄의 정상 분포임을 함의한다.

**실용적 고려사항:**
- 번인(burn-in): 연쇄가 평형에 도달하기 전의 초기 샘플을 버린다
- 솎아내기(thinning): 자기상관을 줄이려고 k번째마다 샘플을 유지한다
- 제안 스케일: 너무 작으면 연쇄가 느리게 움직이고(높은 수락률, 느린 탐색), 너무 크면 대부분의 제안이 기각된다(낮은 수락률, 제자리에 갇힘)
- 고차원에서 가우시안 제안의 최적 수락률은 약 0.234다

### 깁스 샘플링 (Gibbs Sampling)

깁스 샘플링은 다변량 분포를 위한 MCMC의 특수한 경우다. 모든 차원에서 한 번에 이동을 제안하는 대신, 한 번에 한 변수를 그 조건부 분포로부터 갱신한다.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

깁스 샘플링은 각 조건부 분포 p(x_i | x_{-i})에서 샘플링할 수 있어야 한다. 이는 많은 모델에서 간단하다.
- 베이지안 네트워크: 조건부가 그래프 구조에서 따라옴
- 가우시안 혼합: 조건부가 가우시안임
- 이징(Ising) 모델: 각 스핀의 조건부가 이웃에만 의존함

수락률은 항상 1이다(모든 제안이 수락됨). 정확한 조건부에서 샘플링하면 세부 균형이 자동으로 만족되기 때문이다.

**한계.** 변수들이 강하게 상관되어 있으면, 한 번에 한 변수를 갱신하는 것으로는 분포를 가로지르는 큰 대각 이동을 할 수 없어서 깁스 샘플링은 느리게 섞인다.

### 온도 샘플링 (LLM에서 사용)

언어 모델은 어휘의 각 토큰에 대해 로짓 z_1, ..., z_V를 출력한다. 소프트맥스(softmax)가 이를 확률로 변환한다. 온도는 소프트맥스 전에 로짓을 재스케일링한다.

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**동작하는 이유.** 로짓을 T < 1로 나누면 로짓 사이의 차이가 증폭된다. z_1 = 2이고 z_2 = 1일 때, T = 0.5로 나누면 z_1/T = 4, z_2/T = 2가 되어 격차가 더 커진다. 소프트맥스 후, 가장 높은 로짓의 토큰이 훨씬 큰 몫을 얻는다.

**실제로:**
- T = 0.0: 그리디 디코딩, 사실 기반 Q&A에 최적
- T = 0.3-0.7: 약간 창의적, 코드 생성에 좋음
- T = 0.7-1.0: 균형 잡힘, 일반 대화에 좋음
- T = 1.0-1.5: 창의적 글쓰기, 브레인스토밍
- T > 1.5: 점점 무작위, 거의 쓸모없음

온도는 어떤 토큰이 가능한지를 바꾸지 않는다. 각 토큰에 할당된 확률 질량을 바꾼다.

### Top-k 샘플링

top-k 샘플링은 후보 집합을 확률이 가장 높은 k개 토큰으로 제한한 뒤, 재정규화하고 그 제한된 집합에서 샘플링한다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

top-k는 어휘 분포의 긴 꼬리에 있는 극히 가능성 낮은 토큰(오타, 무의미)을 모델이 고르지 못하게 막는다. 문제는 k가 문맥과 무관하게 고정되어 있다는 점이다. 모델이 확신할 때(한 토큰이 95% 확률), k = 40은 여전히 39개의 대안을 허용한다. 모델이 불확실할 때(확률이 1000개 토큰에 퍼져 있음), k = 40은 그럴듯한 선택지를 잘라버린다.

### Top-p (뉴클리어스) 샘플링

top-p 샘플링은 후보 집합 크기를 동적으로 조정한다. 고정된 수의 토큰을 유지하는 대신, 누적 확률이 p를 초과하는 가장 작은 토큰 집합을 유지한다.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

모델이 확신할 때 뉴클리어스 샘플링은 적은 토큰(어쩌면 2-3개)을 유지한다. 모델이 불확실할 때는 많은 토큰(어쩌면 200개)을 유지한다. 이 적응적 행동 덕분에 뉴클리어스 샘플링이 일반적으로 top-k보다 더 나은 텍스트를 만든다.

**흔한 조합:**
- 온도 0.7 + top-p 0.9: 좋은 범용 설정
- 온도 0.0(그리디): 결정적 과제에 최적
- 온도 1.0 + top-k 50: Fan et al. (2018) 원 논문 설정

top-k와 top-p는 결합할 수 있다. 먼저 top-k를 적용하고, 남은 집합에 top-p를 적용한다.

### 재매개변수화 트릭 (VAE에서 사용)

변분 오토인코더(variational autoencoder, VAE)는 입력을 잠재 공간(latent space)의 분포로 인코딩하고, 그 분포에서 샘플링하고, 샘플을 다시 디코딩해 학습한다. 문제는 샘플링 연산을 통해 역전파할 수 없다는 것이다.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

재매개변수화 트릭은 무작위성을 파라미터에서 분리한다.

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

이것이 동작하는 이유는 N(mu, sigma^2)가 mu + sigma * N(0, 1)과 같은 분포를 갖기 때문이다. 핵심 통찰: 무작위성을 파라미터 없는 원천(epsilon)으로 옮긴 뒤, 샘플을 파라미터의 미분 가능한 변환으로 표현한다.

**VAE 학습 루프에서:**
1. 인코더(encoder)가 각 입력에 대해 mu와 log(sigma^2)를 출력함
2. epsilon ~ N(0, 1)을 샘플링함
3. z = mu + sigma * epsilon을 계산함
4. z를 디코딩해 입력을 재구성함
5. 4, 3, 2, 1 단계를 통해 역전파함(3단계가 미분 가능하므로 가능함)

재매개변수화 트릭이 없으면 VAE는 표준 역전파로 학습될 수 없다. 이 단 하나의 통찰이 VAE를 실용적으로 만들었다.

### Gumbel-Softmax (미분 가능한 범주형 샘플링)

재매개변수화 트릭은 연속 분포(가우시안)에 대해 동작한다. 이산 범주형 분포는 다른 접근이 필요하다. Gumbel-Softmax는 범주형 샘플링에 대한 미분 가능한 근사를 제공한다.

**Gumbel-Max 트릭(미분 불가능):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax(미분 가능한 근사):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax는 이산 샘플의 연속적 완화(relaxation)를 만든다. 출력은 하드 원핫 대신 확률 벡터(소프트 원핫)다. 그래디언트가 소프트맥스를 통해 흐른다. 학습 중 순방향 패스에서는 "straight-through" 추정량을 쓸 수 있다. 순방향 패스에는 하드 argmax를 쓰되, 역방향 패스에는 소프트 Gumbel-Softmax 그래디언트를 쓴다.

**응용:**
- VAE의 이산 잠재 변수
- 신경망 아키텍처 탐색(이산 연산 선택)
- 하드 어텐션(attention) 메커니즘
- 이산 행동을 가진 강화 학습

### 층화 샘플링 (Stratified Sampling)

표준 몬테카를로 샘플링은 우연히 샘플 공간에 빈틈을 남길 수 있다. 층화 샘플링은 공간을 층(strata)으로 나누고 각 층에서 샘플링해 고른 커버리지를 강제한다.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

층화 샘플링은 항상 표준 몬테카를로보다 분산이 작거나 같다.

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**응용:**
- 수치 적분(준 몬테카를로)
- 학습 데이터 분할(각 폴드에서 클래스 균형 보장)
- 층화를 곁들인 중요도 샘플링(두 기법 결합)
- NeRF(Neural Radiance Fields)는 카메라 광선을 따라 층화 샘플링을 사용함

### 확산 모델과의 연결

확산 모델은 샘플링 과정을 통해 이미지를 생성한다. 순방향 과정은 이미지가 순수 잡음이 될 때까지 T 스텝에 걸쳐 가우시안 잡음을 더한다. 역방향 과정은 노이즈를 제거하는 법을 학습해 원본 이미지를 한 스텝씩 복구한다.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

이 레슨의 방법들과의 연결:
- 각 노이즈 제거 스텝은 재매개변수화 트릭을 사용한다(잡음을 샘플링하고 결정적 변환을 적용)
- 잡음 스케줄 {alpha_t}는 일종의 온도 어닐링을 제어한다
- 학습은 ELBO(증거 하한)를 근사하기 위해 몬테카를로 추정을 사용한다
- 확산 모델의 조상 샘플링(ancestral sampling)은 마르코프 연쇄다(각 스텝이 현재 상태에만 의존)

전체 이미지 생성 과정은 반복적 샘플링이다. 잡음에서 시작해, 각 스텝마다 학습된 노이즈 제거 모델을 조건으로 약간 덜 노이즈한 버전을 샘플링한다.

```figure
monte-carlo-pi
```

## 직접 만들기 (Build It)

### 1단계: 균등 및 역 CDF 샘플링

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000개의 지수 샘플을 생성하고 평균이 1/lambda인지 검증하라.

### 2단계: 기각 샘플링

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

기각 샘플링을 사용해 절단 정규 분포에서 추출하라. 샘플을 히스토그램으로 만들어 형태를 검증하라.

### 3단계: 중요도 샘플링

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

균등 제안을 사용해 정규 분포 아래의 E[X^2]를 추정하라. 알려진 답(mu^2 + sigma^2)과 비교하라.

### 4단계: pi의 몬테카를로 추정

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 5단계: 메트로폴리스-헤이스팅스 MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

이봉형(bimodal) 분포(두 가우시안의 혼합)에서 샘플링하라. 연쇄의 궤적을 시각화하라.

### 6단계: 깁스 샘플링

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 7단계: 온도 샘플링

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

온도가 토큰 로짓 집합에 대한 출력 분포를 어떻게 바꾸는지 보여라.

### 8단계: Top-k 및 top-p 샘플링

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 9단계: 재매개변수화 트릭

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

재매개변수화된 샘플을 통해서는 그래디언트가 흐르지만 직접 샘플링을 통해서는 흐르지 않음을 시연하라.

### 10단계: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

온도를 낮추면 출력이 원핫 벡터에 가까워지는 것을 보여라.

모든 시각화를 포함한 완전한 구현은 `code/sampling.py`에 있다.

## 라이브러리로 써보기 (Use It)

NumPy와 SciPy를 사용한 프로덕션 버전:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

대규모 MCMC는 전용 라이브러리를 써라.
- PyMC: NUTS(적응형 HMC)를 갖춘 완전한 베이지안 모델링
- emcee: 앙상블 MCMC 샘플러
- NumPyro/JAX: GPU 가속 MCMC

이것들을 밑바닥부터 만들어 봤으니, 이제 라이브러리 호출이 무엇을 하는지 안다.

## 연습 문제 (Exercises)

1. 코시(Cauchy) 분포에 대한 역 CDF 샘플링을 구현하라. CDF는 F(x) = 0.5 + arctan(x)/pi다. 10,000개 샘플을 생성하고 참 PDF에 대비해 히스토그램을 플롯하라. 무거운 꼬리(중심에서 멀리 떨어진 극단값)에 주목하라.

2. Uniform(0, 1) 제안을 사용해 기각 샘플링으로 Beta(2, 5) 분포에서 샘플을 생성하라. 수락된 샘플을 참 Beta PDF에 대비해 플롯하라. 이론적 수락률은 얼마인가?

3. 1,000, 10,000, 100,000개 샘플로 몬테카를로를 사용해 0에서 pi까지 sin(x)의 적분을 추정하라. 각 수준에서 오차를 비교하라. 오차가 O(1/sqrt(N))으로 스케일링되는지 검증하라.

4. exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)에 비례하는 2D 분포 p(x, y)에서 샘플링하기 위해 메트로폴리스-헤이스팅스를 구현하라. 샘플과 연쇄 궤적을 플롯하라. 다른 제안 표준편차로 실험하라.

5. 완전한 텍스트 생성 데모를 만들어라. 로짓을 가진 10개 단어의 어휘가 주어지면, (a) 그리디, (b) 온도=0.7, (c) top-k=3, (d) top-p=0.9를 사용해 20개 토큰의 시퀀스를 생성하라. 5번의 실행에 걸쳐 출력의 다양성을 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 샘플링(Sampling) | "무작위 값 추출" | 확률 분포에 따라 값을 생성하는 것. 모든 생성형 AI 뒤에 있는 메커니즘 |
| 균등 분포(Uniform distribution) | "모두 똑같이 가능" | [a, b]의 모든 값이 같은 확률 밀도 1/(b-a)를 가짐. 모든 샘플링 방법의 출발점 |
| 역 CDF(Inverse CDF) | "확률 변환" | F_inverse(U)는 균등 샘플을 알려진 CDF를 가진 어떤 분포의 샘플로 변환. 정확하고 효율적 |
| 기각 샘플링(Rejection sampling) | "제안하고 수락/기각" | 단순한 제안에서 생성하고, 목표/제안 비율에 비례하는 확률로 수락. 정확하지만 샘플을 낭비 |
| 중요도 샘플링(Importance sampling) | "샘플 재가중" | q(x)의 샘플을 각각 p(x)/q(x)로 가중하여 p(x) 아래의 기댓값을 추정. RL의 PPO의 핵심 |
| 몬테카를로(Monte Carlo) | "무작위 샘플 평균" | 적분을 샘플 평균으로 근사. 차원과 무관하게 오차 O(1/sqrt(N)) |
| MCMC | "수렴하는 무작위 보행" | 정상 분포가 목표인 마르코프 연쇄를 구성. 메트로폴리스-헤이스팅스가 기초 알고리즘 |
| 메트로폴리스-헤이스팅스(Metropolis-Hastings) | "오르막은 수락, 때로 내리막도" | 이동을 제안하고 밀도 비율에 기반해 수락. 세부 균형이 목표 분포로의 수렴을 보장 |
| 깁스 샘플링(Gibbs sampling) | "한 번에 한 변수" | 다른 변수를 고정한 채 각 변수를 그 조건부 분포로부터 갱신. 100% 수락률 |
| 온도(Temperature) | "확신 손잡이" | 소프트맥스 전에 로짓을 T로 나눔. T<1은 날카롭게(더 확신), T>1은 평평하게(더 다양) |
| Top-k 샘플링 | "최고 k개 유지" | 확률 상위 k개를 제외한 모두를 0으로 만들고, 재정규화하고, 샘플링. 고정된 후보 집합 크기 |
| 뉴클리어스 샘플링(top-p) | "가능성 높은 것 유지" | 누적 확률이 p를 초과하는 가장 작은 토큰 집합을 유지. 적응적 후보 집합 크기 |
| 재매개변수화 트릭(Reparameterization trick) | "무작위성을 바깥으로 옮김" | epsilon ~ N(0,1)일 때 z = mu + sigma * epsilon로 씀. 샘플링을 미분 가능하게 만듦. VAE 학습에 필수 |
| Gumbel-Softmax | "소프트 범주형 샘플링" | Gumbel 잡음 + 온도를 가진 소프트맥스를 사용한 범주형 샘플링의 미분 가능한 근사 |
| 층화 샘플링(Stratified sampling) | "강제된 커버리지" | 샘플 공간을 층으로 나누고 각각에서 샘플링. 항상 순진한 몬테카를로보다 분산이 작음 |
| 번인(Burn-in) | "워밍업 기간" | 연쇄가 정상 분포에 도달하기 전 버려지는 초기 MCMC 샘플 |
| 세부 균형(Detailed balance) | "가역성 조건" | p(x) * T(x->y) = p(y) * T(y->x). p가 마르코프 연쇄의 정상 분포이기 위한 충분조건 |
| 확산 샘플링(Diffusion sampling) | "반복적 노이즈 제거" | 잡음에서 시작해 학습된 노이즈 제거 스텝을 적용하여 데이터 생성. 각 스텝이 조건부 샘플링 연산 |

## 더 읽을거리 (Further Reading)

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 기초에 대한 상세 튜토리얼
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 원래의 Gumbel-Softmax 논문
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 뉴클리어스(top-p) 샘플링 논문
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 재매개변수화 트릭을 도입한 VAE 논문
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 샘플링을 이미지 생성과 연결한 DDPM
