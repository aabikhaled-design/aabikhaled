# 확률과 분포 (Probability and Distributions)

> 확률(probability)은 AI가 불확실성을 표현하는 데 사용하는 언어다.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 베르누이(Bernoulli), 범주형(categorical), 푸아송(Poisson), 균등(uniform), 정규(normal) 분포에 대한 PMF와 PDF를 밑바닥부터 구현하기
- 기댓값(expected value), 분산(variance)을 계산하고 중심극한정리(Central Limit Theorem)로 왜 가우시안(Gaussian)이 지배적인지 설명하기
- 수치 안정성 트릭(최대 로짓 빼기)을 사용해 소프트맥스(softmax)와 로그-소프트맥스(log-softmax) 함수 만들기
- 로짓(logits)으로부터 교차 엔트로피(cross-entropy) 손실을 계산하고 그것을 음의 로그 우도(negative log-likelihood)와 연결하기

## 문제 (The Problem)

어떤 분류기(classifier)가 `[0.03, 0.91, 0.06]`을 출력한다. 어떤 언어 모델은 50,000개 후보 중에서 다음 단어를 고른다. 어떤 확산 모델(diffusion model)은 학습된 분포에서 샘플링하여 이미지를 생성한다. 이 모든 것이 확률이 작동하는 모습이다.

모델이 하는 모든 예측은 확률 분포(probability distribution)다. 모든 손실 함수(loss function)는 예측된 분포가 참된 분포에서 얼마나 멀리 떨어져 있는지 측정한다. 모든 학습 스텝은 한 분포를 다른 분포에 더 가깝게 보이도록 파라미터를 조정한다. 확률 없이는 ML 논문 한 편도 읽을 수 없고, 모델 하나도 디버깅할 수 없으며, 학습 손실이 왜 NaN인지 이해할 수 없다.

## 개념 (The Concept)

### 사건, 표본 공간, 확률

표본 공간(sample space) S는 가능한 모든 결과의 집합이다. 사건(event)은 표본 공간의 부분집합이다. 확률은 사건을 0과 1 사이의 숫자로 매핑한다.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 가지 공리가 확률 전체를 정의한다:
1. 임의의 사건 A에 대해 P(A) >= 0
2. P(S) = 1 (무언가는 항상 일어난다)
3. A와 B가 동시에 일어날 수 없을 때 P(A or B) = P(A) + P(B)

그 밖의 모든 것(베이즈 정리(Bayes' theorem), 기댓값, 분포)은 이 세 규칙에서 따라 나온다.

### 조건부 확률과 독립

P(A|B)는 B가 일어났다고 할 때 A의 확률이다.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

한쪽을 안다고 해서 다른 쪽에 대해 아무것도 알 수 없을 때 두 사건은 독립(independent)이다:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

동전 던지기는 독립이다. 복원 없이 카드를 뽑는 것은 독립이 아니다.

### 확률 질량 함수 vs 확률 밀도 함수

이산 확률 변수(discrete random variable)는 확률 질량 함수(probability mass function, PMF)를 갖는다. 각 결과는 직접 읽어낼 수 있는 특정 확률을 갖는다.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률 변수(continuous random variable)는 확률 밀도 함수(probability density function, PDF)를 갖는다. 단일 점에서의 밀도는 확률이 아니다. 확률은 밀도를 구간에 걸쳐 적분하는 데서 나온다.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이 구별은 ML에서 중요하다. 분류 출력은 PMF다(이산 선택). VAE 잠재 공간(latent space)은 PDF를 사용한다(연속).

### 흔한 분포들

**베르누이(Bernoulli):** 한 번의 시행, 두 가지 결과. 이진 분류(binary classification)를 모델링한다.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**범주형(Categorical):** 한 번의 시행, k가지 결과. 다중 클래스 분류(소프트맥스 출력)를 모델링한다.

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**균등(Uniform):** 모든 결과가 동일하게 가능하다. 무작위 초기화에 사용된다.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**정규(Normal, Gaussian):** 종 모양 곡선. 평균(mu)과 분산(sigma^2)으로 매개변수화된다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**푸아송(Poisson):** 고정된 구간에서 드문 사건의 횟수. 사건 발생률을 모델링한다.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 기댓값과 분산

기댓값은 가중 평균 결과다.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

분산은 평균 주위의 퍼짐을 측정한다.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)로 나타난다. 분산은 모델 안정성에 대해 알려준다. 그래디언트(gradient)의 분산이 높으면 학습이 노이즈가 많다는 뜻이다.

### 결합 분포와 주변 분포

결합 분포(joint distribution) P(X, Y)는 두 확률 변수를 함께 묘사한다.

결합 PMF 예시 (X = 날씨, Y = 우산):

| | Y=0 (우산 없음) | Y=1 (우산 있음) | 주변 P(X) |
|---|---|---|---|
| X=0 (맑음) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (비) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **주변 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

주변 분포(marginal distribution)는 다른 변수를 더해서 없앤다:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

위 표의 행과 열의 합계가 주변 분포다.

### 정규 분포가 어디에나 나타나는 이유

중심극한정리: 많은 독립 확률 변수의 합(또는 평균)은 원래 분포와 상관없이 정규 분포로 수렴(convergence)한다.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

그 이유는:
- 측정 오차는 근사적으로 정규다 (많은 작은 독립 원천)
- 신경망(neural network)의 가중치(weight) 초기화는 정규 분포를 사용한다
- SGD의 그래디언트 노이즈는 근사적으로 정규다 (많은 샘플 그래디언트의 합)
- 정규 분포는 주어진 평균과 분산에 대해 최대 엔트로피(maximum entropy) 분포다

### 로그 확률

원래 확률은 수치 문제를 일으킨다. 작은 확률을 여럿 곱하면 빠르게 0으로 언더플로(underflow)된다.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

로그 확률이 이를 해결한다. 곱셈이 덧셈이 된다.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0이다 (0 < P <= 1이므로)
- 더 음수일수록 = 덜 가능하다
- 교차 엔트로피 손실은 올바른 클래스의 음의 로그 확률이다

### 확률 분포로서의 소프트맥스

신경망은 원시 점수(로짓)를 출력한다. 소프트맥스는 그것들을 유효한 확률 분포로 변환한다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

소프트맥스 트릭: 오버플로(overflow)를 막기 위해 지수화하기 전에 최대 로짓을 뺀다.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

로그-소프트맥스는 수치 안정성을 위해 소프트맥스와 로그를 결합한다. PyTorch는 교차 엔트로피 손실에 이를 내부적으로 사용한다.

### 샘플링

샘플링(sampling)은 분포에서 무작위 값을 뽑는 것을 뜻한다. ML에서:
- 드롭아웃(dropout)은 어느 뉴런을 0으로 만들지 무작위로 샘플링한다
- 데이터 증강은 무작위 변환을 샘플링한다
- 언어 모델은 예측된 분포에서 다음 토큰(token)을 샘플링한다
- 확산 모델은 노이즈를 샘플링하고 점진적으로 노이즈를 제거한다

임의 분포에서 샘플링하려면 역변환 샘플링(inverse transform sampling), 기각 샘플링(rejection sampling), 또는 재매개변수화 트릭(reparameterization trick, VAE에서 사용) 같은 기법이 필요하다.

```figure
gaussian-pdf
```

## 직접 만들기 (Build It)

### 1단계: 확률 기초

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 2단계: 밑바닥부터 만드는 PMF와 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 3단계: 기댓값과 분산

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 4단계: 분포에서 샘플링하기

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 5단계: 소프트맥스와 로그 확률

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 6단계: 중심극한정리 시연

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 7단계: 시각화

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

모든 시각화를 포함한 전체 구현은 `code/probability.py`에 있다.

## 라이브러리로 써보기 (Use It)

NumPy와 SciPy로 하면 위의 모든 것이 한 줄짜리가 된다:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

이것들을 밑바닥부터 직접 만들어 봤다. 이제 라이브러리 호출이 무엇을 하는지 안다.

## 연습 문제 (Exercises)

1. 지수 분포(exponential distribution)에 대한 역변환 샘플링을 구현하라. 10,000개 값을 샘플링하고 히스토그램을 참 PDF와 비교하여 검증하라.

2. 편향된 두 주사위에 대한 결합 분포 표를 만들어라. 주변 분포를 계산하고 두 주사위가 독립인지 확인하라.

3. 올바른 클래스가 인덱스 3일 때 로짓 `[2.0, 0.5, -1.0, 3.0, 0.1]`을 출력하는 5-클래스 분류기의 교차 엔트로피 손실을 계산하라. 그다음 PyTorch의 `nn.CrossEntropyLoss`로 답을 검증하라.

4. 로그 확률 리스트를 받아 가장 가능성 높은 시퀀스, 전체 로그 확률, 그리고 그에 해당하는 원시 확률을 반환하는 함수를 작성하라. 각 단어의 확률이 0.01인 50개 단어 문장으로 테스트하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 표본 공간 (Sample space) | "모든 가능성" | 실험의 가능한 모든 결과의 집합 S |
| PMF (확률 질량 함수) | "확률 함수" | 각 이산 결과의 정확한 확률을 주며, 합이 1이 되는 함수 |
| PDF (확률 밀도 함수) | "확률 곡선" | 연속 변수에 대한 밀도 함수. 확률을 얻으려면 구간에 걸쳐 적분한다 |
| 조건부 확률 (Conditional probability) | "무언가가 주어진 확률" | P(A\|B) = P(A and B) / P(B). 베이지안 사고와 베이즈 정리의 토대 |
| 독립 (Independence) | "서로 영향을 주지 않는다" | P(A and B) = P(A) * P(B). 한 사건을 알아도 다른 사건에 대해 아무것도 알 수 없다 |
| 기댓값 (Expected value) | "평균" | 모든 결과의 확률 가중 합. 손실 함수는 기댓값이다 |
| 분산 (Variance) | "얼마나 퍼져 있나" | 평균으로부터의 기대 제곱 편차. 높은 분산 = 노이즈 많고 불안정한 추정 |
| 정규 분포 (Normal distribution) | "종 모양 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). CLT로 인해 어디에나 나타난다 |
| 중심 극한 정리 (Central Limit Theorem) | "평균은 정규가 된다" | 많은 독립 샘플의 평균은 원천과 상관없이 정규 분포로 수렴한다 |
| 결합 분포 (Joint distribution) | "두 변수를 함께" | P(X, Y)는 X와 Y 결과의 모든 조합의 확률을 묘사한다 |
| 주변 분포 (Marginal distribution) | "다른 변수를 더해 없앤다" | P(X) = sum_y P(X, Y). 결합 분포에서 한 변수의 분포를 복원한다 |
| 로그 확률 (Log probability) | "확률의 로그" | log P(x). 곱을 합으로 바꿔 긴 시퀀스에서 수치 언더플로를 막는다 |
| 소프트맥스 (Softmax) | "점수를 확률로 바꾼다" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수 값 로짓을 유효한 확률 분포로 매핑한다 |
| 교차 엔트로피 (Cross-entropy) | "손실 함수" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정한다. 낮을수록 좋다 |
| 로짓 (Logits) | "원시 모델 출력" | 소프트맥스 이전의 비정규화 점수. 로지스틱 함수에서 이름을 따왔다 |
| Sampling | "무작위 값 뽑기" | 확률 분포에 따라 값을 생성하는 것. 모델이 출력을 생성하는 방식 |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규가 되는지에 대한 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기서 다룬 모든 것과 그 이상을 다루는 간결한 레퍼런스
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치 안정성이 왜 중요하고 어떻게 달성하는지
