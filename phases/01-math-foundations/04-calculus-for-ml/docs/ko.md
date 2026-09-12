# 머신러닝을 위한 미적분 (Calculus for Machine Learning)

> 도함수(derivative)는 어느 쪽이 내리막인지 알려준다. 신경망(neural network)이 학습하는 데 필요한 것은 그게 전부다.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 흔한 ML 함수(x^2, 시그모이드(sigmoid), 교차 엔트로피(cross-entropy))에 대한 수치적·해석적 도함수 계산하기
- 1D와 2D에서 손실 함수(loss function)를 최소화하기 위한 경사 하강법(gradient descent)을 밑바닥부터 구현하기
- 선형 회귀(linear regression) 모델의 그래디언트(gradient)를 유도하고 수동 가중치 갱신으로 학습시키기
- 헤시안(Hessian) 행렬, 테일러 급수(Taylor series) 근사, 그리고 그것들이 최적화 방법과 갖는 연결을 설명하기

## 문제 (The Problem)

수백만 개의 가중치(weight)를 가진 신경망이 있다. 각 가중치는 하나의 손잡이다. 모델을 아주 조금 덜 틀리게 만들려면 모든 손잡이를 어느 방향으로 돌려야 하는지 알아내야 한다. 미적분이 그 방향을 알려준다.

미적분이 없다면 신경망을 학습시킨다는 것은 무작위로 변화를 시도하며 잘되기를 바라는 일이 될 것이다. 도함수가 있으면 각 가중치가 오차에 어떻게 영향을 주는지 정확히 안다. 모든 손잡이를 매번 올바른 방향으로 돌린다.

## 개념 (The Concept)

### 도함수란 무엇인가?

도함수는 변화율(rate of change)을 측정한다. 함수 y = f(x)에서 도함수 f'(x)는 x를 아주 작은 양만큼 밀었을 때 y가 얼마나 변하는지 알려준다.

기하학적으로, 도함수는 한 점에서의 접선(tangent line)의 기울기다.

**f(x) = x^2:**

| x | f(x) | f'(x) (기울기) |
|---|------|---------------|
| 0 | 0    | 0 (평평함, 바닥) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (이 점에서의 접선 기울기) |
| 3 | 9    | 6 |

x=2에서 기울기는 4다. x를 오른쪽으로 아주 조금 옮기면 y는 그 양의 약 4배만큼 증가한다. x=0에서 기울기는 0이다. 그릇의 바닥에 있는 것이다.

형식적 정의:

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

코드에서는 극한을 건너뛰고 그냥 아주 작은 h를 사용한다. 그것이 수치 도함수(numerical derivative)다.

### 편도함수: 한 번에 하나의 변수

실제 함수는 입력이 많다. 신경망의 손실은 수천 개의 가중치에 의존한다. 편도함수(partial derivative)는 하나를 제외한 모든 변수를 상수로 고정한 뒤 그 하나만 미분한다.

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

각 편도함수는 이 하나의 가중치만 밀었을 때 손실이 어떻게 변하는지 답한다.

### 그래디언트: 모든 편도함수의 벡터

그래디언트는 모든 편도함수를 하나의 벡터로 모은다. 함수 f(x, y, z)의 그래디언트는 다음과 같다.

```
grad f = [ df/dx, df/dy, df/dz ]
```

그래디언트는 가장 가파른 상승(steepest ascent) 방향을 가리킨다. 함수를 최소화하려면 반대 방향으로 간다.

**f(x,y) = x^2 + y^2의 등고선 그림:**

이 함수는 동심원을 등고선으로 하는 그릇 모양을 이룬다. 최솟값은 (0, 0)에 있다.

| 점 | grad f | -grad f (하강 방향) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (오르막을 가리킴, 최솟값에서 멀어짐) | [-2, -2] (내리막을 가리킴, 최솟값을 향함) |
| (0, 0) | [0, 0] (평평함, 최솟값에 있음) | [0, 0] |

이것이 그림으로 본 경사 하강법이다. 그래디언트를 계산하고, 부호를 뒤집고, 한 걸음 내딛는다.

### 최적화와의 연결

신경망을 학습시키는 것은 최적화다. 모델이 얼마나 틀렸는지 측정하는 손실 함수 L(w1, w2, ..., wn)이 있다. 이를 최소화하고 싶다.

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

학습률(learning rate)은 스텝 크기를 제어한다. 너무 크면 지나쳐 버린다. 너무 작으면 기어간다.

**손실 지형 (1D 단면):**

손실 함수 L(w)는 가중치 w가 변함에 따라 봉우리와 골짜기를 가진 곡선을 이룬다.

| 특징 | 설명 |
|---------|-------------|
| 전역 최솟값 (Global minimum) | 전체 곡선에서 가장 낮은 점: 최선의 해 |
| 지역 최솟값 (Local minimum) | 이웃보다는 낮지만 전체적으로 가장 낮지는 않은 골짜기 |
| 기울기 (Slope) | 경사 하강법은 어떤 시작점에서든 기울기를 따라 내리막으로 간다 |

경사 하강법은 기울기를 따라 내리막으로 간다. 지역 최솟값(local minima)에 갇힐 수 있지만, 고차원 공간(수백만 개 가중치)에서는 이것이 실용적 문제인 경우가 드물다.

### 수치 도함수 vs 해석적 도함수

도함수를 계산하는 두 가지 방법이 있다.

해석적(analytical): 미적분 규칙을 손으로 적용한다. f(x) = x^2의 경우 도함수는 f'(x) = 2x다. 정확하다. 빠르다.

수치적(numerical): 정의를 사용해 근사한다. 아주 작은 h에 대해 f(x+h)와 f(x-h)를 계산한 뒤 그 차이를 사용한다.

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

수치 도함수는 더 느리지만 어떤 함수에도 동작한다. 해석적 도함수는 빠르지만 공식을 직접 유도해야 한다. 신경망 프레임워크는 세 번째 접근법인 자동 미분(automatic differentiation)을 사용해 정확한 도함수를 기계적으로 계산한다. 이는 Phase 3에서 다룬다.

### 간단한 함수의 손계산 도함수

이것들이 ML에서 반복해서 보게 될 도함수들이다.

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

f(x) = x^2의 경우:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

x=3, b=1일 때 f(w) = wx + b의 경우:

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 연쇄 법칙 (The chain rule)

함수가 합성될 때, 연쇄 법칙(chain rule)이 어떻게 미분하는지 알려준다.

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

신경망은 함수들의 연쇄다: 입력 -> 선형 -> 활성화 -> 선형 -> 활성화 -> 손실. 역전파(backpropagation)는 출력에서 입력까지 연쇄 법칙을 반복 적용한 것이다. 그것이 알고리즘 전부다.

### 헤시안 행렬 (The Hessian Matrix)

그래디언트는 기울기를 알려준다. 헤시안은 곡률(curvature)을 알려준다.

헤시안은 2차 편도함수들의 행렬이다. 함수 f(x1, x2, ..., xn)에 대해, 헤시안의 (i, j) 항목은:

```
H[i][j] = d^2f / (dx_i * dx_j)
```

2변수 함수 f(x, y)의 경우:

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**임계점(critical point, 그래디언트 = 0인 곳)에서 헤시안이 알려주는 것:**

| 헤시안 성질 | 의미 | 예시 곡면 |
|-----------------|---------|-----------------|
| 양의 정부호 (모든 고윳값 > 0) | 지역 최솟값 | 위로 향한 그릇 |
| 음의 정부호 (모든 고윳값 < 0) | 지역 최댓값 | 아래로 향한 그릇 |
| 부정부호 (고윳값이 섞임) | 안장점 | 말 안장 모양 |

**예시:** f(x, y) = x^2 - y^2 (안장 함수)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

f(x, y) = x^2 + y^2 (그릇)과 비교하라:

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**헤시안이 ML에서 중요한 이유:**

뉴턴 방법(Newton's method)은 헤시안을 사용해 경사 하강법보다 더 나은 최적화 스텝을 밟는다. 단지 기울기만 따라가는 대신, 곡률을 고려한다:

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

뉴턴 방법은 더 빨리 수렴(convergence)한다. 헤시안이 그래디언트를 "재스케일링"해 가파른 방향에는 더 작은 스텝을, 평평한 방향에는 더 큰 스텝을 주기 때문이다.

함정: N개 파라미터를 가진 신경망의 경우 헤시안은 N x N이다. 100만 개 파라미터를 가진 모델은 1조 개 항목짜리 행렬이 필요하다. 그래서 근사를 사용한다.

| 방법 | 사용하는 것 | 비용 | 수렴 |
|--------|-------------|------|-------------|
| 경사 하강법 | 1차 도함수만 | 스텝당 O(N) | 느림 (선형) |
| 뉴턴 방법 | 전체 헤시안 | 스텝당 O(N^3) | 빠름 (이차) |
| L-BFGS | 그래디언트 이력으로 근사한 헤시안 | 스텝당 O(N) | 중간 (초선형) |
| Adam | 파라미터별 적응 학습률 (대각 헤시안 근사) | 스텝당 O(N) | 중간 |
| 자연 그래디언트 | 피셔 정보 행렬 (통계적 헤시안) | 스텝당 O(N^2) | 빠름 |

실무에서 Adam은 딥러닝의 기본 옵티마이저(optimizer)다. 파라미터별 그래디언트의 이동 평균과 분산을 추적하여 2차 정보를 저렴하게 근사한다.

### 테일러 급수 근사 (Taylor Series Approximation)

매끄러운(smooth) 함수는 모두 국소적으로 다항식으로 근사할 수 있다:

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

항을 더 많이 포함할수록 근사가 더 좋아진다. 단, 점 x 근처에서만 그렇다.

**테일러 급수가 ML에서 중요한 이유:**

- **1차 테일러 = 경사 하강법.** f(x + h) ~ f(x) + f'(x)*h를 사용하면 선형 근사를 만드는 것이다. 경사 하강법은 이 선형 모델을 최소화하여 h = -lr * f'(x)를 선택한다.

- **2차 테일러 = 뉴턴 방법.** f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2를 사용하면 2차 모델을 얻는다. 이를 최소화하면 h = -f'(x)/f''(x): 뉴턴 스텝을 얻는다.

- **손실 함수 설계.** MSE와 교차 엔트로피는 매끄럽고, 이는 그 테일러 전개가 잘 동작함을 뜻한다. 이는 우연이 아니다. 매끄러운 손실은 최적화를 예측 가능하게 만든다.

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

핵심 통찰: 모든 그래디언트 기반 최적화는 사실 손실 함수를 국소적으로 근사하고 그 근사의 최솟값으로 스텝을 밟는 일이다.

### ML에서의 적분 (Integrals in ML)

도함수는 변화율을 알려준다. 적분(integral)은 누적량을 계산한다. 곡선 아래의 면적이다.

ML에서 적분을 손으로 계산하는 일은 드물지만, 그 개념은 어디에나 있다:

**확률.** 밀도 p(x)를 가진 연속 확률 변수의 경우:
```
P(a < X < b) = integral from a to b of p(x) dx
```
a와 b 사이의 확률 밀도 곡선 아래 면적이 그 범위에 떨어질 확률이다.

**기댓값.** 확률로 가중된 평균 결과:
```
E[f(X)] = integral of f(x) * p(x) dx
```
데이터 분포에 대한 기대 손실은 적분이다. 학습은 이것의 경험적 근사를 최소화한다.

**KL 발산.** 두 분포가 얼마나 다른지 측정한다:
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
VAE, 지식 증류(knowledge distillation), 베이지안 추론(Bayesian inference)에 사용된다.

**정규화 상수.** 베이지안 추론에서:
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
분모는 가능한 모든 파라미터 값에 대한 적분이다. 이는 흔히 다루기 어려워(intractable), 그래서 MCMC와 변분 추론(variational inference) 같은 근사를 사용한다.

| 적분 개념 | ML에서 나타나는 곳 |
|-----------------|----------------------|
| 곡선 아래 면적 | 밀도 함수로부터의 확률 |
| 기댓값 | 손실 함수, 위험 최소화 |
| KL 발산 | VAE, 정책 최적화, 증류(distillation) |
| 정규화 | 베이지안 사후 분포, 소프트맥스 분모 |
| 주변 우도 | 모델 비교, 증거 하한(ELBO) |

### 계산 그래프에서의 다변수 연쇄 법칙

연쇄 법칙은 한 줄로 늘어선 스칼라 함수에만 적용되는 게 아니다. 신경망에서 변수들은 갈라지고(fan out) 합쳐진다. 다음은 간단한 순방향 패스(forward pass)에서 도함수가 어떻게 흐르는지를 보여준다.

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

역방향 패스(backward pass)는 오른쪽에서 왼쪽으로 그래디언트를 계산한다:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

각 화살표는 국소 도함수(local derivative)를 곱한다. 어떤 파라미터의 그래디언트든, 손실에서 그 파라미터까지의 경로를 따라 있는 모든 국소 도함수의 곱이다. 경로가 갈라지고 합쳐질 때는 기여분을 더한다(다변수 연쇄 법칙).

이것이 역전파의 전부다: 계산 그래프(computation graph)를 통해 출력에서 입력으로 체계적으로 적용된 연쇄 법칙.

### 야코비안 행렬 (The Jacobian matrix)

함수가 벡터를 벡터로 매핑할 때(신경망 층처럼), 그 도함수는 행렬이다. 야코비안(Jacobian)은 모든 출력의 모든 입력에 대한 모든 편도함수를 담는다.

f: R^n -> R^m에 대해, 야코비안 J는 m x n 행렬이다:

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

신경망에 대해 야코비안을 손으로 계산하지는 않는다. PyTorch가 처리한다. 하지만 그것이 존재한다는 걸 알면 역전파에서의 형태(shape)를 이해하는 데 도움이 된다: 어떤 층이 R^n을 R^m으로 매핑하면 그 야코비안은 m x n이다. 그래디언트는 이 행렬의 전치(transpose)로 역방향으로 흐른다.

### 이것이 신경망에서 중요한 이유

신경망의 모든 가중치에는 그래디언트가 주어진다. 그래디언트는 손실을 줄이려면 그 가중치를 어떻게 조정할지 알려준다.

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

각 가중치 갱신:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

순방향 패스는 예측과 손실을 계산한다. 역방향 패스는 모든 가중치에 대한 손실의 그래디언트를 계산한다. 그다음 모든 가중치가 내리막으로 작은 한 걸음을 내딛는다. 수백만 스텝 반복한다. 그것이 딥러닝이다.

```figure
derivative-tangent
```

## 직접 만들기 (Build It)

### 1단계: 밑바닥부터 만드는 수치 도함수

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

수치 도함수는 해석적 도함수와 소수점 여러 자리까지 일치한다.

### 2단계: 편도함수와 그래디언트

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 3단계: f(x) = x^2의 최솟값을 찾는 경사 하강법

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

x=5에서 시작하여, 각 스텝은 x=0(최솟값)에 더 가까이 다가간다.

### 4단계: 2D 함수에 대한 경사 하강법

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 5단계: 수치 도함수와 해석적 도함수 비교

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 6단계: 헤시안을 수치적으로 계산하기

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

안장 함수의 헤시안은 고윳값 2와 -2를 갖는다(부호가 섞임, 안장점(saddle point)임을 확인). 그릇은 고윳값 2와 2를 갖는다(둘 다 양수, 최솟값임을 확인).

### 7단계: 테일러 근사의 실제

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

x0=0 근처에서 sin(x) ~ x이다(1차 테일러). 근사는 작은 h에서는 훌륭하지만 큰 h에서는 무너진다. 이것이 경사 하강법이 작은 학습률에서 가장 잘 동작하는 이유다. 각 스텝은 선형 근사가 정확하다고 가정한다.

### 8단계: 이것이 신경망에서 중요한 이유

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

모든 그래디언트 기반 학습 루프는 이 패턴을 따른다: 예측하고, 손실을 계산하고, 그래디언트를 계산하고, 가중치를 갱신한다.

## 라이브러리로 써보기 (Use It)

NumPy로 하면 같은 연산이 더 빠르고 간결하다:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

방금 경사 하강법을 밑바닥부터 만든 셈이다. PyTorch는 그래디언트 계산을 자동화하지만 갱신 루프는 동일하다.

## 연습 문제 (Exercises)

1. `numerical_derivative`를 두 번 호출하여 `numerical_second_derivative(f, x)`를 구현하라. x=2에서 x^3의 2차 도함수가 12임을 검증하라.
2. 경사 하강법을 사용해 f(x, y) = (x - 3)^2 + (y + 1)^2의 최솟값을 찾아라. (0, 0)에서 시작하라. 답은 (3, -1)로 수렴해야 한다.
3. 경사 하강법 루프에 모멘텀(momentum)을 추가하라: 과거 그래디언트를 누적하는 속도(velocity) 벡터를 유지한다. f(x) = x^4 - 3x^2에서 모멘텀이 있을 때와 없을 때의 수렴 속도를 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 도함수 (Derivative) | "기울기" | 한 점에서 함수의 변화율. 입력의 단위 변화당 출력이 얼마나 변하는지 알려준다. |
| 편도함수 (Partial derivative) | "한 변수의 도함수" | 다른 모든 변수를 고정한 채 한 변수에 대해 취한 도함수. |
| 그래디언트 (Gradient) | "가장 가파른 상승 방향" | 모든 편도함수의 벡터. 함수를 가장 빠르게 증가시키는 방향을 가리킨다. |
| 경사 하강법 (Gradient descent) | "내리막으로 간다" | 손실을 줄이기 위해 파라미터에서 그래디언트(학습률을 곱한)를 뺀다. 신경망 학습의 핵심. |
| 학습률 (Learning rate) | "스텝 크기" | 각 경사 하강 스텝이 얼마나 큰지 제어하는 스칼라. 너무 크면 발산, 너무 작으면 느린 수렴. |
| 연쇄 법칙 (Chain rule) | "도함수를 곱한다" | 합성 함수를 미분하는 규칙: df/dx = df/dg * dg/dx. 역전파의 수학적 토대. |
| 야코비안 (Jacobian) | "도함수의 행렬" | 함수가 벡터를 벡터로 매핑할 때, 야코비안은 출력의 입력에 대한 모든 편도함수의 행렬이다. |
| 수치 미분 (Numerical derivative) | "유한 차분" | 가까운 두 점에서 함수를 평가하고 그 사이의 기울기를 계산하여 도함수를 근사하는 것. |
| 역전파 (Backpropagation) | "역방향 모드 자동 미분" | 연쇄 법칙을 사용해 출력에서 입력으로 층별로 그래디언트를 계산하는 것. 신경망이 학습하는 방식. |
| 헤시안 (Hessian) | "2차 도함수의 행렬" | 모든 2차 편도함수의 행렬. 함수의 곡률을 묘사한다. 임계점에서 양의 정부호(positive definite) 헤시안이면 지역 최솟값을 뜻한다. |
| 테일러 급수 (Taylor series) | "다항식 근사" | 함수를 그 도함수들로 한 점 근처에서 근사하는 것: f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... 경사 하강법과 뉴턴 방법이 왜 동작하는지 이해하는 토대. |
| 적분 (Integral) | "곡선 아래 면적" | 어떤 양을 범위에 걸쳐 누적한 것. ML에서 적분은 확률, 기댓값, KL 발산을 정의한다. |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - 도함수, 적분, 연쇄 법칙에 대한 시각적 직관
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - 그래디언트가 신경망 층을 통해 어떻게 흐르는지
