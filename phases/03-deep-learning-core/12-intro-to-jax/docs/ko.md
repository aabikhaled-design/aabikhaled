# JAX 입문 (Introduction to JAX)

> PyTorch는 텐서(tensor)를 변경한다. TensorFlow는 그래프를 만든다. JAX는 순수 함수(pure function)를 컴파일한다. 마지막 것이 딥러닝(deep learning)에 대한 사고방식을 바꾼다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90분

## 학습 목표 (Learning Objectives)

- JAX의 함수형 API(jax.numpy, jax.grad, jax.jit, jax.vmap)를 사용해 순수 함수 신경망(neural network) 코드 작성하기
- PyTorch의 즉시(eager) 변경과 JAX의 함수형 컴파일 모델 사이의 핵심 설계 차이 설명하기
- 순진한 Python에 비해 학습 루프를 가속하기 위해 jit 컴파일과 vmap 벡터화 적용하기
- JAX에서 단순한 신경망을 학습시키고, 명시적 상태 관리를 PyTorch의 객체 지향 접근법과 대조하기

## 문제 (The Problem)

PyTorch에서 신경망을 만드는 법은 이미 안다. `nn.Module`을 정의하고, `.backward()`를 호출하고, 옵티마이저(optimizer)를 스텝한다. 잘 작동하고, 수백만 명이 쓴다.

하지만 PyTorch에는 DNA에 박힌 제약이 있다: 연산을 한 번에 하나씩, Python에서 즉시 추적한다. 모든 `tensor + tensor`가 별개의 커널 실행이다. 모든 학습 스텝이 같은 Python 코드를 다시 해석한다. 이것은 5,400억 파라미터(parameter) 모델을 2,048개의 TPU에 걸쳐 학습시킬 필요가 생기기 전까지는 괜찮게 작동한다. 그 지점부터는 오버헤드에 발목이 잡힌다.

Google DeepMind는 Gemini를 JAX로 학습시킨다. Anthropic은 Claude를 JAX로 학습시켰다. 이것들은 작은 작업이 아니라 지구상에서 가장 큰 신경망 학습 실행이다. 이들이 JAX를 고른 이유는, JAX가 학습 루프를 Python 호출의 시퀀스가 아니라 컴파일 가능한 프로그램으로 다루기 때문이다.

JAX는 세 가지 초능력을 가진 NumPy다. 자동 미분(automatic differentiation), XLA로의 JIT 컴파일, 그리고 자동 벡터화다. 예제 하나를 처리하는 함수만 작성하면, JAX가 배치(batch)를 처리하고 그래디언트(gradient)를 계산하고 기계어로 컴파일하며 여러 디바이스에 걸쳐 돌아가는 함수로 만들어 준다. 원래 함수는 하나도 바꾸지 않고서.

## 개념 (The Concept)

### JAX 철학

JAX는 함수형 프레임워크다. 클래스도, 변경 가능한 상태도, `.backward()` 메서드도 없다. 대신:

| PyTorch | JAX |
|---------|-----|
| 상태를 가진 `nn.Module` 클래스 | 순수 함수: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 즉시 실행 | XLA를 통한 JIT 컴파일 |
| `for x in batch:` 수동 루프 | `jax.vmap(f)` 자동 벡터화 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 자동 병렬화 |
| 가변 `model.parameters()` | 불변 배열 pytree |

이것은 스타일 취향이 아니라 컴파일러 제약이다. JIT 컴파일은 순수 함수를 요구한다. 같은 입력은 항상 같은 출력을 내고, 부작용이 없어야 한다. 그 제약이 100배 속도 향상을 가능하게 한다.

### jax.numpy: 익숙한 표면

JAX는 가속기에서 NumPy API를 재구현한다.

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

같은 함수 이름. 같은 브로드캐스팅 규칙. 같은 슬라이싱 의미론. 하지만 배열은 GPU/TPU에 살고, 모든 연산은 컴파일러에 의해 추적 가능하다.

한 가지 결정적 차이: JAX 배열은 불변(immutable)이다. `a[0] = 5`는 안 된다. 대신: `a = a.at[0].set(5)`. 일주일쯤은 어색하게 느껴지다가 어느 순간 딸깍 맞아떨어진다. `grad`, `jit`, `vmap` 같은 변환을 조합 가능(composable)하게 만드는 것이 바로 이 불변성이다.

### jax.grad: 함수형 자동 미분

PyTorch는 그래디언트를 텐서에 붙인다(`.grad`). JAX는 그래디언트를 함수에 붙인다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 그래디언트를 계산하는 새 함수를 반환한다. `.backward()` 호출 없음. 텐서에 저장된 계산 그래프(computation graph) 없음. 그래디언트는 그저 호출하고 조합하고 JIT 컴파일할 수 있는 또 다른 함수일 뿐이다.

이것은 임의로 조합된다.

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2차 도함수(derivative), 3차 도함수, 야코비안(Jacobian), 헤시안(Hessian). 모두 `grad`를 조합하여 얻는다. PyTorch도 이것을 할 수 있지만(`torch.autograd.functional.hessian`), 어디까지나 덧붙여진 기능이다. JAX에서는 그것이 토대다.

제약: `grad`는 순수 함수에서만 작동한다. 안에 print 문 없음(추적 중에 돌아가지, 실행 중이 아니다). 외부 상태 변경 없음. 명시적 키 관리 없는 난수 생성 없음.

### jit: XLA로 컴파일

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 호출에서 JAX는 함수를 추적한다. 어떤 연산이 일어나는지를 실행하지 않고 기록하는 것이다. 그다음 그 추적을 XLA(Accelerated Linear Algebra), 곧 TPU와 GPU를 위한 Google의 컴파일러에 넘긴다. XLA는 연산을 융합(fuse)하고, 불필요한 메모리 복사를 제거하고, 최적화된 기계어를 생성한다.

이후의 호출은 Python을 완전히 건너뛴다. 컴파일된 코드가 가속기에서 C++ 속도로 돌아간다.

JIT이 도움이 될 때:
- 학습 스텝(같은 계산이 수천 번 반복됨)
- 추론(inference, 같은 모델, 다른 입력)
- 비슷한 형태의 입력으로 한 번 이상 호출되는 모든 함수

JIT이 해가 될 때:
- 값에 의존하는 Python 제어 흐름을 가진 함수(`if x > 0`, 여기서 x는 추적된 배열)
- 일회성 계산(컴파일 오버헤드가 런타임을 초과)
- 디버깅(추적이 실제 실행을 숨김)

제어 흐름 제약은 실재한다. `jax.lax.cond`가 `if/else`를 대체하고, `jax.lax.scan`이 `for` 루프를 대체한다. 이것들은 선택이 아니라 컴파일의 대가다.

### vmap: 자동 벡터화

예제 하나를 처리하는 함수를 작성한다.

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`은 그것을 배치를 처리하도록 들어 올린다.

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`는: `params`에 대해서는 배치하지 않고(공유), `x`의 축 0에 대해 배치하라는 뜻이다. 수동 `for` 루프 없음. 리셰이핑 없음. 배치 차원 엮기 없음. JAX가 배치 차원을 알아내고 전체 계산을 벡터화한다.

이것은 문법 설탕(syntactic sugar)이 아니다. `vmap`은 Python 루프보다 10-100배 빠르게 돌아가는 융합된 벡터화 코드를 생성한다. 그리고 `jit`, `grad`와 조합된다.

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 그래디언트. 한 줄. 이것은 PyTorch에서 꼼수 없이는 거의 불가능하다.

### pmap: 디바이스 전반의 데이터 병렬화

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`은 함수를 사용 가능한 모든 디바이스(GPU/TPU)에 복제하고 배치를 나눈다. 함수 안에서, `jax.lax.pmean`과 `jax.lax.psum`이 디바이스 전반의 그래디언트를 동기화한다.

Google은 `pmap`(과 그 후속인 `shard_map`)을 써서 Gemini를 수천 개의 TPU v5e 칩에 걸쳐 학습시킨다. 프로그래밍 모델은: 단일 디바이스 버전을 쓰고, `pmap`으로 감싸면, 끝이다.

### Pytree: 보편 데이터 구조

JAX는 "pytree", 곧 리스트, 튜플, 딕셔너리, 배열의 중첩된 조합 위에서 작동한다. 모델 파라미터도 pytree다.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

`grad`, `jit`, `vmap`을 비롯한 모든 JAX 변환은 pytree를 순회하는 법을 안다. `jax.tree.map(f, tree)`는 모든 리프(leaf)에 `f`를 적용한다. 이것이 옵티마이저가 모든 파라미터를 한 번에 갱신하는 방식이다.

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드 없음. 파라미터 등록 없음. 트리 구조가 곧 모델이다.

### 함수형 대 객체 지향

PyTorch는 상태를 객체 안에 저장한다.

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 상태를 가진 순수 함수를 쓴다.

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params가 전달된다. 아무것도 저장되지 않고, 아무것도 변경되지 않는다. 덕분에 모든 함수가 테스트 가능하고 조합 가능하며 컴파일 가능해진다. 대신 params를 직접 관리해야 한다는 뜻이기도 하다. 아니면 Flax나 Equinox 같은 라이브러리를 쓰면 된다.

### JAX 생태계

JAX는 기본 요소를 준다. 라이브러리는 편의성을 준다.

| 라이브러리 | 역할 | 스타일 |
|---------|------|-------|
| **Flax** (Google) | 신경망 층 | 명시적 상태를 가진 `nn.Module` |
| **Equinox** (Patrick Kidger) | 신경망 층 | Pytree 기반, 파이썬다움 |
| **Optax** (DeepMind) | 옵티마이저 + LR 스케줄 | 조합 가능한 그래디언트 변환 |
| **Orbax** (Google) | 체크포인팅 | pytree 저장/복원 |
| **CLU** (Google) | 메트릭 + 로깅 | 학습 루프 유틸리티 |

Optax는 표준 옵티마이저 라이브러리다. 그래디언트 변환(Adam, SGD, 클리핑)을 파라미터 갱신에서 분리하여, 조합을 식은 죽 먹기로 만든다.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 언제 JAX를 쓰고 언제 PyTorch를 쓸까

| 요인 | JAX | PyTorch |
|--------|-----|---------|
| TPU 지원 | 최고 수준 (Google이 둘 다 만듦) | 커뮤니티 유지 관리 (torch_xla) |
| GPU 지원 | 양호 (XLA를 통한 CUDA) | 업계 최고 (네이티브 CUDA) |
| 디버깅 | 어려움 (추적 + 컴파일) | 쉬움 (즉시 실행, 한 줄씩) |
| 생태계 | 연구 중심 (Flax, Equinox) | 방대함 (HuggingFace, torchvision 등) |
| 채용 | 틈새 (Google/DeepMind/Anthropic) | 주류 (어디서나) |
| 대규모 학습 | 우월 (XLA, pmap, mesh) | 양호 (FSDP, DeepSpeed) |
| 프로토타이핑 속도 | 더 느림 (함수형 오버헤드) | 더 빠름 (변경하고 바로 실행) |
| 프로덕션 추론 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 사용 주체 | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

솔직한 답은 이렇다. JAX를 써야 할 구체적인 이유가 없다면 PyTorch를 써라. 그 이유란 TPU 접근, 예제별 그래디언트의 필요, 대규모의 다중 디바이스 학습, 또는 Google/DeepMind/Anthropic에서 일하는 경우다.

### JAX의 난수

JAX에는 전역 난수 상태가 없다. 모든 난수 연산은 명시적 PRNG 키를 요구한다.

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음에는 성가시다. 하지만 이 방식은 디바이스와 컴파일 전반의 재현성을 보장한다. PyTorch의 `torch.manual_seed`가 다중 GPU 환경에서 보장할 수 없는 속성이다.

```figure
batchnorm-effect
```

## 직접 만들기 (Build It)

### 1단계: 준비와 데이터

JAX와 Optax를 써서 MNIST에 3층 MLP를 학습시켜 본다. 입력 784개, 256개와 128개 뉴런(neuron)의 은닉층(hidden layer) 두 개, 출력 클래스 10개다.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 1단계: 파라미터 초기화

클래스 없음. 그저 pytree를 반환하는 함수.

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 초기화를 수동으로 했다. 하나의 시드에서 갈라진 세 개의 PRNG 키. 모든 가중치(weight)는 중첩된 딕셔너리 안의 불변 배열이다.

### 2단계: 순방향 패스

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

순수 함수다. params가 들어가고, 예측이 나온다. `self` 없음, 저장된 상태 없음. `loss_fn`은 교차 엔트로피(cross-entropy)를 밑바닥부터, 즉 소프트맥스(softmax), 로그, 음의 평균으로 계산한다.

### 3단계: JIT 컴파일된 학습 스텝

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`는 손실(loss) 값과 그래디언트를 한 번의 패스로 둘 다 반환한다. `@jax.jit` 데코레이터는 두 함수를 모두 XLA로 컴파일한다. 첫 호출 이후, 각 학습 스텝은 Python을 건드리지 않고 돌아간다.

### 4단계: 학습 루프

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 에폭(epoch). ~97% 테스트 정확도. 첫 에폭은 느리다(JIT 컴파일). 에폭 2-10은 빠르다.

빠진 것을 주목하라. `.zero_grad()` 없음, `.backward()` 없음, `.step()` 없음. 전체 갱신이 하나의 조합된 함수 호출이다. 그래디언트가 계산되고, Adam으로 변환되고, 파라미터에 적용되는 과정이 모두 `train_step` 안에서 일어난다.

## 라이브러리로 써보기 (Use It)

### Flax: Google 표준

Flax는 가장 흔한 JAX 신경망 라이브러리다. `nn.Module`을 되살리지만, 명시적 상태 관리와 함께다.

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch와 같은 구조이지만, `params`가 모델과 분리되어 있다. `model.init()`이 params를 만든다. `model.apply(params, x)`가 순방향 패스를 돌린다. 모델 객체는 상태가 없다.

### Equinox: Python다운 대안

Equinox(Patrick Kidger 제작)는 모델을 pytree로 표현한다.

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree다. `.apply()` 불필요. 파라미터는 그저 모델의 리프다. 이것은 JAX가 생각하는 방식에 더 가깝다.

### Optax: 조합 가능한 옵티마이저

Optax는 그래디언트 변환을 갱신에서 분리한다.

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

그래디언트 클리핑(gradient clipping), 학습률(learning rate) 웜업(warmup), 가중치 감쇠(weight decay)가 모두 변환의 체인(chain)으로 조합된다. 각 변환은 그래디언트를 보고, 수정하고, 다음으로 넘긴다. 단일체 옵티마이저 클래스 없음.

## 산출물 (Ship It)

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원의 경우:

```bash
pip install jax[cuda12]
```

TPU(Google Cloud)의 경우:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 함정:**

- 첫 JIT 호출은 느리다(컴파일). 벤치마킹 전에 워밍업하라.
- JIT 안에서 JAX 배열에 대한 Python 루프를 피하라. `jax.lax.scan`이나 `jax.lax.fori_loop`를 써라.
- `jax.debug.print()`는 JIT 안에서 작동한다. 일반 `print()`는 안 된다.
- `jax.profiler`나 TensorBoard로 프로파일링하라. XLA 컴파일이 병목을 숨길 수 있다.
- JAX는 기본적으로 GPU 메모리의 75%를 미리 할당한다. 끄려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 설정하라.

**체크포인팅:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이 레슨은 다음을 산출한다:**
- `outputs/prompt-jax-optimizer.md` -- 올바른 JAX 옵티마이저 구성을 고르기 위한 프롬프트
- `outputs/skill-jax-patterns.md` -- JAX의 함수형 패턴을 다루는 스킬

## 연습 문제 (Exercises)

1. MLP에 드롭아웃(dropout)을 추가하라. JAX에서 드롭아웃은 PRNG 키를 요구한다. 순방향 패스에 키를 엮고 각 드롭아웃 층마다 갈라라. 있을 때와 없을 때의 테스트 정확도를 비교하라.

2. `jax.vmap`을 써서 32개의 MNIST 이미지 배치에 대한 예제별 그래디언트를 계산하라. 각 예제의 그래디언트 노름(norm)을 계산하라. 어떤 예제가 가장 큰 그래디언트를 가지며, 왜인가?

3. 수동 순방향 함수를, 임의 개수의 층에 대해 작동하는 일반적 `mlp_forward(params, x)`로 바꿔라. `jax.tree.leaves`를 써서 깊이를 자동으로 결정하라.

4. `@jax.jit`이 있을 때와 없을 때의 학습 스텝을 벤치마킹하라. 각각 100 스텝의 시간을 재라. 자신의 하드웨어에서 속도 향상은 얼마나 큰가? 첫 호출의 컴파일 오버헤드는 얼마인가?

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`을 조합하여 그래디언트 클리핑을 구현하라. 클리핑이 있을 때와 없을 때로 학습하라. 효과를 보기 위해 학습 동안 그래디언트 노름을 그려라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만드는 것" | Accelerated Linear Algebra -- 연산을 융합하고 계산 그래프에서 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "적시 컴파일" | JAX가 첫 호출에서 함수를 추적하고, XLA로 컴파일한 뒤, 이후 호출에서 컴파일된 버전을 돌림 |
| 순수 함수(Pure function) | "부작용 없음" | 출력이 입력에만 의존하는 함수 -- 전역 상태 없음, 변경 없음, 명시적 키 없는 무작위성 없음 |
| vmap | "자동 배치" | 예제 하나를 처리하는 함수를, 재작성 없이 배치를 처리하는 함수로 변환함 |
| pmap | "자동 병렬화" | 함수를 여러 디바이스에 복제하고 입력 배치를 나눔 |
| Pytree | "배열의 중첩 딕셔너리" | JAX가 순회하고 변환할 수 있는, 리스트/튜플/딕셔너리/배열의 모든 중첩 구조 |
| 추적(Tracing) | "계산 기록하기" | JAX가 실제 결과를 계산하지 않고, 추상 값으로 함수를 실행하여 계산 그래프를 만드는 것 |
| 함수형 자동 미분(Functional autodiff) | "함수의 grad" | 텐서에 그래디언트 저장소를 붙이는 게 아니라 함수를 변환하여 도함수를 계산하는 것 |
| Optax | "JAX의 옵티마이저 라이브러리" | 함께 연결되는, 그래디언트 변환의 조합 가능한 라이브러리 -- Adam, SGD, 클리핑, 스케줄링 |
| Flax | "JAX의 nn.Module" | JAX를 위한 Google의 신경망 라이브러리로, 상태를 명시적으로 유지하면서 층 추상화를 추가함 |

## 더 읽을거리 (Further Reading)

- JAX documentation: https://jax.readthedocs.io/ -- grad, jit, vmap에 대한 훌륭한 튜토리얼을 담은 공식 문서
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 설계 철학을 설명한 원조 논문
- Flax documentation: https://flax.readthedocs.io/ -- JAX를 위한 Google의 신경망 라이브러리
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax에 대한 Python다운 대안
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 표준 옵티마이저 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 저자 중 한 명이 쓴, JAX 함정과 패턴에 대한 실용 가이드
