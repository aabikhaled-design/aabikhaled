# 벡터, 행렬 & 연산 (Vectors, Matrices & Operations)

> 모든 신경망(neural network)은 그저 단계가 몇 개 추가된 행렬 곱일 뿐이다.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 원소별(element-wise) 연산, 행렬 곱, 전치(transpose), 행렬식(determinant), 역행렬(inverse)을 갖춘 Matrix 클래스 만들기
- 원소별 곱과 행렬 곱을 구별하고 각각이 언제 적용되는지 설명하기
- 밑바닥부터 만든 Matrix 클래스만으로 단일 밀집(dense) 신경망 층(layer)(`relu(W @ x + b)`) 구현하기
- 브로드캐스팅(broadcasting) 규칙과 신경망 프레임워크에서 편향(bias) 덧셈이 동작하는 방식 설명하기

## 문제 (The Problem)

신경망을 만들고 싶다. 코드를 읽다 보면 이런 것을 본다:

```
output = activation(weights @ input + bias)
```

저 `@`가 행렬 곱이다. `weights`는 행렬(matrix)이고 `input`은 벡터(vector)다. 이 연산들이 무엇을 하는지 모르면 이 한 줄은 마법처럼 보인다. 알고 나면 한 층의 전체 순방향 패스(forward pass)를 세 연산으로 표현한 것임을 알 수 있다.

모델이 처리하는 모든 이미지는 픽셀 값의 행렬이다. 단어 임베딩(embedding)은 벡터고, 신경망의 모든 층은 행렬 변환이다. 변수를 이해하지 못하면 코드를 쓸 수 없듯, 행렬 연산에 익숙하지 않으면 AI 시스템을 만들 수 없다.

이 레슨은 그 능숙함을 밑바닥부터 쌓는다.

## 개념 (The Concept)

### 벡터: 순서가 있는 숫자의 목록

벡터는 방향과 크기를 가진 숫자의 목록이다. AI에서 벡터는 데이터 포인트나 특성(feature), 파라미터(parameter)를 나타낸다.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 벡터 `[3, 4]`는 평면 위의 좌표 (3, 4)를 향한다. 그 길이(크기)는 5다 (3-4-5 삼각형).

### 행렬: 숫자의 격자

행렬은 2차원 격자다. 행(row)과 열(column)로 이루어진다. m x n 행렬은 m개의 행과 n개의 열을 갖는다.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

신경망에서 가중치 행렬(weight matrix)은 입력 벡터를 출력 벡터로 변환한다. 입력 784개, 출력 128개인 층은 128x784 가중치 행렬을 사용한다.

### 형태(shape)가 중요한 이유

행렬 곱에는 엄격한 규칙이 있다: `(m x n) @ (n x p) = (m x p)`. 안쪽 차원이 일치해야 한다.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch에서 형태 불일치(shape mismatch) 오류가 나는 이유가 바로 이것이다.

### 연산 지도

| 연산 | 하는 일 | 신경망에서의 용도 |
|-----------|-------------|-------------------|
| 덧셈 (Addition) | 원소별 결합 | 출력에 편향 더하기 |
| 스칼라 곱 (Scalar multiply) | 모든 원소 스케일링 | 학습률 * 그래디언트 |
| 행렬 곱 (Matrix multiply) | 벡터 변환 | 층 순방향 패스 |
| 전치 (Transpose) | 행과 열을 뒤집기 | 역전파(backpropagation) |
| 행렬식 (Determinant) | 단일 숫자 요약 | 가역성(invertibility) 확인 |
| 역행렬 (Inverse) | 변환 되돌리기 | 선형 시스템 풀기 |
| 항등 (Identity) | 아무것도 안 하는 행렬 | 초기화, 잔차 연결(residual connection) |

### 원소별 곱 vs 행렬 곱

이 구별이 초보자를 끊임없이 헷갈리게 한다.

원소별(element-wise): 일치하는 위치끼리 곱한다. 두 행렬은 형태가 같아야 한다.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

행렬 곱(matrix multiplication): 행과 열의 내적(dot product)이다. 안쪽 차원이 일치해야 한다.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

다른 연산, 다른 결과, 다른 규칙이다.

### 브로드캐스팅 (Broadcasting)

출력 행렬에 편향 벡터를 더할 때 형태가 일치하지 않는다. 브로드캐스팅은 작은 배열을 늘려서 맞춘다.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

현대 프레임워크는 모두 이 작업을 자동으로 처리한다. 원리를 이해하면 형태가 안 맞아 보이는데도 코드가 돌아갈 때 당황하지 않는다.

```figure
vector-projection
```

## 직접 만들기 (Build It)

### 1단계: Vector 클래스

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 2단계: 핵심 연산을 갖춘 Matrix 클래스

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 3단계: 동작 확인

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 4단계: 신경망과 연결하기

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

이것이 단일 밀집 층이다: `output = relu(W @ x + b)`. 모든 신경망의 모든 밀집 층은 정확히 이것을 한다.

## 라이브러리로 써보기 (Use It)

NumPy는 위의 모든 것을 더 적은 줄로, 그리고 수십 배 빠르게 해낸다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python의 `@` 연산자는 `__matmul__`을 호출한다. NumPy는 이를 C와 Fortran으로 작성된 최적화된 BLAS 루틴으로 구현한다. 같은 수학, 100배 빠른 속도.

NumPy의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1차원 편향을 두 행에 걸쳐 자동으로 브로드캐스팅한다. 모든 신경망 프레임워크에서 편향 덧셈이 이렇게 동작한다.

## 산출물 (Ship It)

이 레슨은 기하학적 직관을 통해 행렬 연산을 가르치는 프롬프트를 만들어낸다. `outputs/prompt-matrix-operations.md`를 참고하라.

여기서 만든 Matrix 클래스는 Phase 3, Lesson 10에서 만드는 미니 신경망 프레임워크의 토대다.

## 연습 문제 (Exercises)

1. **역행렬을 검증하라.** `A @ A.inverse_2x2()`를 곱해서 항등 행렬(identity matrix)이 나오는지 확인하라. 서로 다른 2x2 행렬 세 개로 시도해 보라. 행렬식이 0일 때는 무슨 일이 일어나는가?

2. **3x3 역행렬을 구현하라.** 수반행렬(adjugate) 방법을 사용해 3x3 행렬의 역행렬을 계산하도록 Matrix 클래스를 확장하라. NumPy의 `np.linalg.inv`와 비교해 테스트하라.

3. **2층 신경망을 만들어라.** Matrix 클래스만 사용해(NumPy 없이) 2층 신경망을 만들어라: 입력 (3) -> 은닉 (4) -> 출력 (2). 무작위 가중치를 초기화하고, 순방향 패스를 실행한 뒤, 모든 형태가 올바른지 검증하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 벡터 (Vector) | "화살표" | 순서가 있는 숫자의 목록. AI에서는: 고차원 공간의 한 점. |
| 행렬 (Matrix) | "숫자의 표" | 선형 변환. 벡터를 한 공간에서 다른 공간으로 매핑한다. |
| 행렬 곱 (Matrix multiply) | "그냥 숫자를 곱한다" | 첫 행렬의 모든 행과 둘째 행렬의 모든 열 사이의 내적. 순서가 중요하다. |
| 전치 (Transpose) | "뒤집는다" | 행과 열을 맞바꾼다. m x n 행렬을 n x m으로 바꾼다. 역전파에서 결정적이다. |
| 행렬식 (Determinant) | "행렬에서 나온 어떤 숫자" | 행렬이 면적(2D)이나 부피(3D)를 얼마나 스케일링하는지 잰다. 0이면 변환이 한 차원을 짓뭉갠다는 뜻이다. |
| 역행렬 (Inverse) | "행렬을 되돌린다" | 변환을 역으로 되돌리는 행렬. 행렬식이 0이 아닐 때만 존재한다. |
| 항등 행렬 (Identity matrix) | "지루한 행렬" | 1을 곱하는 것에 해당하는 행렬. 잔차 연결(ResNet)에 사용된다. |
| 브로드캐스팅 (Broadcasting) | "마법 같은 형태 맞추기" | 작은 배열을 빠진 차원을 따라 반복하여 큰 배열에 맞게 늘리는 것. |
| 원소별 연산 (Element-wise) | "보통의 곱셈" | 일치하는 위치끼리 곱한다. 두 배열은 같은 형태여야 한다(또는 브로드캐스팅 가능해야 한다). |

## 더 읽을거리 (Further Reading)

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 여기서 다룬 모든 연산에 대한 시각적 직관
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 규칙
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML에 특화된 선형대수의 간결한 레퍼런스
