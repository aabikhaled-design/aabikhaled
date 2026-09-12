# 수치 안정성 (Numerical Stability)

> 부동소수점(floating point)은 새는 추상화(leaky abstraction)다. 학습(training) 도중 발목을 잡는데, 그것도 예고 없이 들이닥친다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 최댓값 빼기 트릭(max-subtraction trick)을 사용해 수치적으로 안정한 소프트맥스(softmax)와 log-sum-exp 구현하기
- 부동소수점 계산에서 오버플로(overflow), 언더플로(underflow), 파국적 상쇄(catastrophic cancellation) 식별하기
- 중심 차분(centered finite differences)을 사용해 해석적 그래디언트(gradient)를 수치적 그래디언트와 대조하여 검증하기
- 학습에서 float16보다 bfloat16이 선호되는 이유와 손실 스케일링(loss scaling)이 그래디언트 언더플로를 막는 방법 설명하기

## 문제 (The Problem)

모델(model)이 세 시간 동안 학습하다가, 손실(loss)이 NaN이 된다. print 문을 추가한다. 스텝 9,000에서 로짓(logits)은 멀쩡하다. 스텝 9,001에서는 `inf`다. 스텝 9,002에 이르면 모든 그래디언트가 `nan`이 되고 학습은 죽는다.

또는: 모델이 끝까지 학습되지만 정확도가 논문이 주장하는 것보다 2% 낮다. 모든 것을 확인한다. 아키텍처가 일치한다. 하이퍼파라미터(hyperparameter)가 일치한다. 데이터가 일치한다. 문제는 논문이 float32를 썼는데 정작 쓴 것은 올바른 스케일링 없는 float16이었다는 데 있다. 32비트로 누적된 반올림 오차가 정확도를 조용히 갉아먹었다.

또는: 교차 엔트로피(cross-entropy) 손실을 밑바닥부터 구현한다. 작은 로짓에서는 동작한다. 로짓이 100을 넘으면 `inf`를 반환한다. `exp(100)`이 float32가 표현할 수 있는 것보다 크기 때문에 소프트맥스가 오버플로된 것이다. 모든 ML 프레임워크는 이를 두 줄짜리 트릭으로 처리한다. 그 트릭이 존재하는지조차 몰랐던 셈이다.

수치 안정성(numerical stability)은 이론적인 관심사가 아니다. 성공하는 학습 실행과 조용히 실패하는 학습 실행을 가르는 차이다. 앞으로 디버깅하게 될 모든 진지한 ML 버그는 결국 부동소수점으로 귀결된다.

## 개념 (The Concept)

### IEEE 754: 컴퓨터가 실수를 저장하는 방법

컴퓨터는 IEEE 754 표준에 따라 실수를 부동소수점 값으로 저장한다. 부동소수점 수는 세 부분으로 이루어진다. 부호 비트(sign bit), 지수(exponent), 가수(mantissa, significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

가수는 정밀도(유효 숫자가 몇 자리인지)를 결정한다. 지수는 범위(수가 얼마나 크거나 작을 수 있는지)를 결정한다.

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32는 약 7자리의 십진 정밀도를 준다. 즉 1.0000001과 1.0000002는 구분할 수 있지만 1.00000001과 1.00000002는 구분하지 못한다. 7자리를 넘어가면 모든 것이 반올림 잡음이다.

float16은 약 3자리를 준다. 표현할 수 있는 가장 큰 수는 65,504다. 로짓, 그래디언트, 활성값(activation)이 일상적으로 이 값을 넘는 ML에서 이는 불안할 만큼 작다.

bfloat16은 float16의 범위 문제에 대한 구글의 답이다. float32와 동일한 8비트 지수(같은 범위, 최대 3.4e38)를 가지지만 가수는 7비트뿐이다(float16보다 정밀도가 낮음). 신경망(neural network) 학습에서는 정밀도보다 범위가 더 중요하므로, bfloat16이 대개 이긴다.

### 왜 0.1 + 0.2 != 0.3 인가

숫자 0.1은 이진 부동소수점에서 정확히 표현될 수 없다. 2진법에서 그것은 순환 소수다.

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32는 이를 가수 23비트로 잘라낸다. 저장된 값은 대략 0.100000001490116이다. 마찬가지로 0.2는 대략 0.200000002980232로 저장된다. 그 합은 0.3이 아니라 0.300000004470348이다.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

이것이 ML에서 중요한 이유:

1. `if loss < threshold` 같은 손실 비교가 잘못된 답을 줄 수 있다
2. 많은 작은 값들(수천 스텝에 걸친 그래디언트 갱신)을 누적하면 참값 합에서 벗어난다
3. `==`로 부동소수점을 비교하면 체크섬과 재현성 테스트가 실패한다

해결책: 절대 `==`로 부동소수점을 비교하지 말라. `abs(a - b) < epsilon`이나 `math.isclose()`를 써라.

### 파국적 상쇄 (Catastrophic Cancellation)

거의 같은 두 부동소수점 수를 뺄 때, 유효 숫자들이 상쇄되고 반올림 잡음이 선두 자릿수로 격상된 채 남는다.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

단 한 번의 뺄셈에서 19%의 상대 오차가 나온 것이다. ML에서 이는 다음과 같은 경우 발생한다.

- 큰 평균을 가진 데이터의 분산을 계산할 때: E[x]가 클 때의 `E[x^2] - E[x]^2`
- 거의 같은 로그 확률(log-probabilities)을 뺄 때
- 너무 작은 epsilon으로 유한 차분(finite-difference) 그래디언트를 계산할 때

해결책: 크고 거의 같은 수를 빼는 것을 피하도록 공식을 재배열하라. 분산의 경우 Welford 알고리즘을 쓰거나 데이터를 먼저 중심화(center)하라. 로그 확률의 경우 시종일관 로그 공간(log-space)에서 작업하라.

### 오버플로와 언더플로

오버플로는 결과가 표현하기에 너무 클 때 발생한다. 언더플로는 결과가 너무 작을 때(표현 가능한 가장 작은 양수보다 0에 더 가까울 때) 발생한다.

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 함수는 ML에서 오버플로의 주된 원천이다.

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 함수는 반대 방향에서 부딪힌다.

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

ML에서 `exp()`는 소프트맥스, 시그모이드(sigmoid), 확률 계산에 등장한다. `log()`는 교차 엔트로피, 로그 가능도(log-likelihood), KL 발산(KL divergence)에 등장한다. `log(exp(x))` 조합은 올바른 트릭 없이는 지뢰밭이다.

### Log-Sum-Exp 트릭

`log(sum(exp(x_i)))`를 직접 계산하는 것은 수치적으로 위험하다. 어떤 `x_i`가 크면 `exp(x_i)`가 오버플로된다. 모든 `x_i`가 매우 음수이면 모든 `exp(x_i)`가 0으로 언더플로되고 `log(0)`은 `-inf`다.

트릭: 지수화하기 전에 최댓값을 뺀다.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

이것이 동작하는 이유: `max(x)`를 뺀 뒤 가장 큰 지수는 `exp(0) = 1`이다. 오버플로가 불가능하다. 합의 항 중 최소 하나는 1이므로 합은 최소 1이고, `log(1) = 0`이다. `-inf`로의 언더플로가 불가능하다.

증명:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`로 설정하면 오버플로가 제거된다.

이 트릭은 ML 어디에나 등장한다.
- 소프트맥스 정규화(normalization)
- 교차 엔트로피 손실 계산
- 시퀀스 모델에서의 로그 확률 합산
- 가우시안 혼합(Mixture of Gaussians)
- 변분 추론(variational inference)

### 소프트맥스에 최댓값 빼기 트릭이 필요한 이유

소프트맥스는 로짓을 확률로 변환한다.

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

트릭 없이는 [100, 101, 102] 로짓이 오버플로를 일으킨다.

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

트릭을 쓰면 max(x) = 102를 뺀다.

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

확률은 동일하다. 계산은 안전하다. 이것은 최적화가 아니다. 정확성을 위한 요구 사항이다.

### NaN과 Inf: 탐지와 예방

`nan`(Not a Number)과 `inf`(무한대)는 계산을 통해 바이러스처럼 전파된다. 그래디언트 갱신에 `nan` 하나가 있으면 가중치(weight)가 `nan`이 되고, 이후의 모든 출력이 `nan`이 된다. 학습은 한 스텝 안에 죽는다.

`inf`가 나타나는 방식:
- 큰 양수의 `exp()`
- 0으로 나누기: `1.0 / 0.0`
- 누적에서의 `float32` 오버플로

`nan`이 나타나는 방식:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수의 `sqrt()`
- 음수의 `log()`
- 기존 `nan`이 관련된 모든 산술

탐지:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

예방 전략:

1. `exp()`의 입력을 클램프하라: `exp(clamp(x, -80, 80))`
2. 분모에 epsilon을 더하라: `x / (y + 1e-8)`
3. `log()` 안에 epsilon을 더하라: `log(x + 1e-8)`
4. 안정한 구현을 써라(log-sum-exp, 안정한 소프트맥스)
5. 가중치 폭발을 막기 위해 그래디언트 클리핑(gradient clipping)을 써라
6. 디버깅 중에는 매 순방향 패스(forward pass)마다 `nan`/`inf`를 확인하라

### 수치적 그래디언트 검사

(역전파에서 나오는) 해석적 그래디언트에는 버그가 있을 수 있다. 수치적 그래디언트 검사(numerical gradient checking)는 유한 차분으로 그래디언트를 계산하여 이를 검증한다.

중심 차분 공식:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

이것은 O(h^2) 정확도이며, O(h)에 불과한 전방 차분 `(f(x+h) - f(x)) / h`보다 훨씬 낫다.

h 선택: 너무 크면 근사가 틀리고, 너무 작으면 파국적 상쇄가 답을 망친다. `h = 1e-5`에서 `1e-7`이 일반적이다.

검사: 해석적 그래디언트와 수치적 그래디언트 사이의 상대 차이를 계산한다.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

경험칙:
- relative_error < 1e-7: 완벽함, 그래디언트가 올바름
- relative_error < 1e-5: 받아들일 만함, 아마 올바름
- relative_error > 1e-3: 무언가 잘못됨
- relative_error > 1: 그래디언트가 완전히 틀림

새로운 층(layer)이나 손실 함수(loss function)를 구현할 때는 항상 그래디언트를 검사하라. PyTorch는 이를 위해 `torch.autograd.gradcheck()`를 제공한다.

### 혼합 정밀도 학습 (Mixed Precision Training)

현대 GPU에는 float16 행렬 곱(matrix multiplication)을 float32보다 2-8배 빠르게 계산하는 특수 하드웨어(Tensor Cores)가 있다. 혼합 정밀도 학습(mixed precision training)은 이를 활용한다.

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

순수 float16 학습의 문제: 그래디언트는 종종 매우 작다(1e-8 이하). float16은 ~6e-8보다 작은 모든 것을 0으로 언더플로한다. 모든 그래디언트 갱신이 0이므로 모델이 학습을 멈춘다.

해결책은 손실 스케일링(loss scaling)이다.

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

동적 손실 스케일링(dynamic loss scaling)은 스케일 인수를 자동으로 조정한다. 큰 값(65536)으로 시작한다. 그래디언트가 `inf`로 오버플로되면 절반으로 줄인다. N 스텝 동안 오버플로 없이 지나가면 두 배로 늘린다.

### bfloat16 vs float16: 학습에서 bfloat16이 이기는 이유

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16은 더 많은 정밀도(가수 10비트 대 7비트)를 가지지만 제한된 범위(최대 ~65,504)를 가진다. bfloat16은 정밀도가 낮지만 float32와 같은 범위(최대 ~3.4e38)를 가진다.

신경망 학습의 경우:

- 활성값과 로짓은 학습 중 급등(spike) 시 정기적으로 65,504를 넘는다. float16은 오버플로되고, bfloat16은 처리한다.
- float16에서는 손실 스케일링이 필요하지만, bfloat16에서는 그 범위가 그래디언트 크기 스펙트럼을 포괄하므로 대개 불필요하다.
- bfloat16은 float32의 단순한 절단이다: 가수의 하위 16비트를 버린다. 변환이 사소하며 지수에서는 무손실이다.

float16은 값이 유계이고 정밀도가 더 중요한 추론(inference)에 선호된다. bfloat16은 범위가 더 중요한 학습에 선호된다. 이것이 TPU와 현대 NVIDIA GPU(A100, H100)가 네이티브 bfloat16 지원을 갖춘 이유다.

### 그래디언트 클리핑 (Gradient Clipping)

그래디언트 폭발(exploding gradient)은 그래디언트가 많은 층을 통과하며 지수적으로 커질 때 발생한다(RNN, 깊은 신경망, 트랜스포머에서 흔함). 단 하나의 큰 그래디언트가 한 스텝 만에 모든 가중치를 망가뜨릴 수 있다.

클리핑의 두 종류:

**값으로 클리핑(Clip by value):** 각 그래디언트 원소를 독립적으로 클램프한다.

```
grad = clamp(grad, -max_val, max_val)
```

단순하지만 그래디언트 벡터(vector)의 방향을 바꿀 수 있다.

**노름으로 클리핑(Clip by norm):** 전체 그래디언트 벡터를 그 노름(norm)이 임곗값을 넘지 않도록 스케일링한다.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

그래디언트의 방향을 보존한다. 이것이 `torch.nn.utils.clip_grad_norm_()`이 하는 일이다. 표준적인 선택이다.

일반적인 값: 트랜스포머에는 `max_norm=1.0`, 강화 학습(RL)에는 `max_norm=0.5`, 더 단순한 신경망에는 `max_norm=5.0`.

그래디언트 클리핑은 꼼수가 아니다. 안전 장치다. 이것이 없으면 단 하나의 이상치(outlier) 배치(batch)가 몇 주간의 학습을 망칠 만큼 큰 그래디언트를 만들 수 있다.

### 수치 안정화 장치로서의 정규화 층

배치 정규화(batch normalization), 층 정규화(layer normalization), RMS 정규화는 보통 학습 수렴(convergence)을 돕는 규제 장치(regularizer)로 소개된다. 이것들은 또한 수치 안정화 장치이기도 하다.

정규화가 없으면 활성값이 층을 통과하며 지수적으로 커지거나 작아질 수 있다.

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

정규화(normalization)는 모든 층에서 활성값을 재중심화하고 재스케일링한다.

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`(보통 1e-5)은 모든 활성값이 동일할 때 0으로 나누는 것을 막는다. 학습된 파라미터(parameter) `gamma`와 `beta`는 신경망이 필요한 어떤 스케일이든 복원할 수 있게 한다.

이는 신경망 전체에서 값을 수치적으로 안전한 범위에 유지하여, 순방향 패스에서의 오버플로와 역방향 패스에서의 그래디언트 폭발을 모두 막는다.

### 흔한 ML 수치 버그

**버그: 몇 에폭(epoch) 후 손실이 NaN이 된다.**
원인: 로짓이 너무 커져서 소프트맥스가 오버플로됨. 또는 학습률(learning rate)이 너무 높아 가중치가 발산함.
해결: 안정한 소프트맥스(최댓값 빼기)를 쓰고, 학습률을 줄이고, 그래디언트 클리핑을 추가하라.

**버그: 손실이 log(num_classes)에 머물러 있다.**
원인: 모델 출력이 거의 균일한 확률에 가까움. 종종 그래디언트가 소실되거나 모델이 전혀 학습하지 않음을 의미함.
해결: 데이터 레이블(label)이 올바른지 확인하고, 손실 함수를 검증하고, 죽은 ReLU를 확인하라.

**버그: 검증 정확도가 예상보다 1-3% 낮다.**
원인: 적절한 손실 스케일링 없는 혼합 정밀도. 그래디언트 언더플로가 작은 갱신을 조용히 0으로 만듦.
해결: 동적 손실 스케일링을 켜거나 bfloat16으로 전환하라.

**버그: 일부 층에서 그래디언트 노름이 0.0이다.**
원인: 죽은 ReLU 뉴런(모든 입력이 음수) 또는 float16 언더플로.
해결: LeakyReLU나 GELU를 쓰고, 그래디언트 스케일링을 쓰고, 가중치 초기화를 확인하라.

**버그: 한 GPU에서는 동작하는데 다른 GPU에서는 다른 결과가 나온다.**
원인: 비결정적 부동소수점 누적 순서. GPU 병렬 리덕션은 하드웨어마다 다른 순서로 합산하며, 부동소수점 덧셈은 결합법칙이 성립하지 않음.
해결: 작은 차이(1e-6)를 받아들이거나, `torch.use_deterministic_algorithms(True)`를 설정하고 속도 손해를 감수하라.

**버그: 손실 계산에서 `exp()`가 `inf`를 반환한다.**
원인: 최댓값 빼기 트릭 없이 원시 로짓을 `exp()`에 전달함.
해결: 내부적으로 log-sum-exp를 구현한 `torch.nn.functional.log_softmax()`를 써라.

**버그: float32에서 float16으로 전환한 뒤 학습이 발산한다.**
원인: float16은 6e-8보다 작은 그래디언트 크기나 65,504보다 큰 활성값을 표현할 수 없음.
해결: 손실 스케일링이 있는 혼합 정밀도(AMP)를 쓰거나, 대신 bfloat16을 써라.

```figure
logsumexp-stability
```

## 직접 만들기 (Build It)

### 1단계: 부동소수점 정밀도 한계 보여주기

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 2단계: 순진한 소프트맥스 vs 안정한 소프트맥스 구현

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 3단계: 안정한 log-sum-exp 구현

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 4단계: 안정한 교차 엔트로피 구현

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 5단계: 그래디언트 검사

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 라이브러리로 써보기 (Use It)

### 혼합 정밀도 시뮬레이션

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 그래디언트 클리핑

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 탐지

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

모든 엣지 케이스가 시연된 완전한 구현은 `code/numerical.py`를 참조하라.

## 산출물 (Ship It)

이 레슨이 만들어내는 것:
- 안정한 소프트맥스, log-sum-exp, 교차 엔트로피, 그래디언트 검사, 혼합 정밀도 시뮬레이션을 갖춘 `code/numerical.py`
- 학습 중 NaN/Inf 및 수치 문제를 진단하기 위한 `outputs/prompt-numerical-debugger.md`

이 안정한 구현들은 학습 루프를 만드는 Phase 3과 어텐션(attention) 메커니즘을 구현하는 Phase 4에서 다시 등장한다.

## 연습 문제 (Exercises)

1. **파국적 상쇄.** float32에서 순진한 공식 `E[x^2] - E[x]^2`를 사용해 [1000000.0, 1000001.0, 1000002.0]의 분산을 계산하라. 그다음 Welford의 온라인 알고리즘을 사용해 계산하라. 참 분산(0.6667)에 대한 오차를 비교하라.

2. **정밀도 사냥.** Python에서 `1.0 + x == 1.0`이 되는 가장 작은 양의 float32 값 `x`를 찾아라. 이것이 머신 엡실론(machine epsilon)이다. 이것이 `numpy.finfo(numpy.float32).eps`와 일치하는지 검증하라.

3. **Log-sum-exp 엣지 케이스.** `logsumexp_stable` 함수를 다음으로 테스트하라. (a) 모든 값이 같음, (b) 한 값이 나머지보다 훨씬 큼, (c) 모든 값이 매우 음수(-1000)임. 순진한 버전이 실패하는 곳에서 올바른 결과를 주는지 검증하라.

4. **신경망 층 그래디언트 검사.** 단일 선형 층 `y = Wx + b`와 그 해석적 역방향 패스를 구현하라. `numerical_gradient`를 사용해 3x2 가중치 행렬에 대한 정확성을 검증하라.

5. **손실 스케일링 실험.** float16으로 학습을 시뮬레이션하라. [1e-9, 1e-3] 범위의 무작위 그래디언트를 만들고, float16으로 변환한 뒤, 얼마나 많은 비율이 0이 되는지 측정하라. 그다음 손실 스케일링(1024를 곱함)을 적용하고, float16으로 변환하고, 다시 스케일을 되돌린 뒤, 0의 비율을 다시 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| IEEE 754 | "부동소수점 표준" | 이진 부동소수점 형식, 반올림 규칙, 특수 값(inf, nan)을 정의하는 국제 표준. 모든 현대 CPU와 GPU가 구현한다. |
| 머신 엡실론(Machine epsilon) | "정밀도 한계" | 주어진 부동소수점 형식에서 1.0 + e != 1.0이 되는 가장 작은 값 e. float32의 경우 약 1.19e-7이다. |
| 파국적 상쇄(Catastrophic cancellation) | "뺄셈으로 인한 정밀도 손실" | 거의 같은 부동소수점 수를 뺄 때 유효 숫자가 상쇄되고 반올림 잡음이 결과를 지배하는 현상. |
| 오버플로(Overflow) | "수가 너무 큼" | 결과가 표현 가능한 최댓값을 초과하여 inf가 됨. exp(89)는 float32를 오버플로한다. |
| 언더플로(Underflow) | "수가 너무 작음" | 결과가 표현 가능한 가장 작은 양수보다 0에 가까워 0.0이 됨. exp(-104)는 float32를 언더플로한다. |
| Log-sum-exp 트릭 | "최댓값을 먼저 뺀다" | exp(max(x))를 인수분해하여 오버플로와 언더플로를 막으며 log(sum(exp(x)))를 계산하는 것. 소프트맥스, 교차 엔트로피, 로그 확률 수학에 쓰인다. |
| 안정한 소프트맥스(Stable softmax) | "폭발하지 않는 소프트맥스" | 지수화 전에 max(logits)를 빼는 것. 수치적으로 동일한 결과, 오버플로 불가능. |
| 그래디언트 검사(Gradient checking) | "역전파를 검증하라" | 구현 버그를 잡기 위해 역전파에서 나온 해석적 그래디언트를 유한 차분에서 나온 수치적 그래디언트와 비교하는 것. |
| 혼합 정밀도(Mixed precision) | "float16 순방향, float32 역방향" | 속도가 중요한 연산에는 더 낮은 정밀도의 부동소수점을, 수치적으로 민감한 연산에는 더 높은 정밀도의 부동소수점을 쓰는 것. 일반적인 속도 향상은 2-3배. |
| 손실 스케일링(Loss scaling) | "그래디언트 언더플로 방지" | 역전파 전에 손실에 큰 상수를 곱해 그래디언트가 float16의 표현 가능 범위에 머물게 한 뒤, 가중치 갱신 전에 같은 상수로 나누는 것. |
| bfloat16 | "Brain floating point" | 지수 8비트(float32와 같은 범위)와 가수 7비트(float16보다 낮은 정밀도)를 가진 구글의 16비트 형식. 학습에 선호된다. |
| 그래디언트 클리핑(Gradient clipping) | "그래디언트 노름을 제한하라" | 그래디언트 벡터를 그 노름이 임곗값을 넘지 않도록 스케일링하는 것. 폭발하는 그래디언트가 가중치를 망치는 것을 막는다. |
| NaN | "Not a Number" | 정의되지 않은 연산(0/0, inf-inf, sqrt(-1))에서 나오는 특수 부동소수점 값. 이후의 모든 산술을 통해 전파된다. |
| Inf | "무한대" | 오버플로나 0으로 나누기에서 나오는 특수 부동소수점 값. 결합되어 NaN을 만들 수 있다(inf - inf, inf * 0). |
| 수치적 그래디언트(Numerical gradient) | "무차별 대입 도함수" | f(x+h)와 f(x-h)를 평가하고 2h로 나누어 도함수를 근사하는 것. 느리지만 검증에 신뢰할 만하다. |

## 더 읽을거리 (Further Reading)

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 결정적인 참고 문헌, 밀도 높지만 완전함
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 학습을 위한 손실 스케일링을 도입한 NVIDIA 논문
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch에서의 혼합 정밀도에 대한 실용 가이드
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- 구글이 TPU에 이 형식을 선택한 이유
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 부동소수점 합에서 반올림 오차를 줄이는 알고리즘
