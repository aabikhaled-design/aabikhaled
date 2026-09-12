# 노름과 거리 (Norms and Distances)

> 거리 함수(distance function)가 "유사하다"의 의미를 정의한다. 잘못 고르면 그 아래의 모든 것이 무너진다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90분

## 학습 목표 (Learning Objectives)

- L1, L2, 코사인(cosine), 마할라노비스(Mahalanobis), 자카드(Jaccard), 편집 거리(edit distance) 함수를 밑바닥부터 구현하기
- 주어진 ML 과제에 적절한 거리 척도(distance metric)를 선택하고 대안이 실패하는 이유 설명하기
- L1과 L2 노름(norm)을 LASSO와 Ridge 정규화(regularization) 및 그 기하학적 제약 영역과 연결하기
- 같은 데이터셋(dataset)이 서로 다른 척도 아래에서 어떻게 다른 최근접 이웃(nearest neighbor)을 만드는지 시연하기

## 문제 (The Problem)

두 벡터(vector)가 있다고 하자. 어쩌면 단어 임베딩(embedding)일 수도, 사용자 프로필일 수도, 픽셀 배열일 수도 있다. 이제 알아야 한다. 이들은 얼마나 가까운가?

답은 전적으로 어떤 거리 함수를 고르느냐에 달려 있다. 두 데이터 포인트는 한 척도에서는 최근접 이웃이고 다른 척도에서는 멀리 떨어져 있을 수 있다. KNN 분류기(classifier), 추천 엔진, 벡터 데이터베이스, 클러스터링(clustering) 알고리즘, 손실 함수(loss function): 이들 모두가 이 선택에 의존한다. 잘못 고르면 모델(model)이 잘못된 것을 최적화한다.

보편적으로 최선인 거리는 없다. L2는 공간 데이터에 잘 동작한다. 코사인 유사도(cosine similarity)는 NLP를 지배한다. 자카드는 집합을 다룬다. 편집 거리는 문자열을 다룬다. 마할라노비스는 상관관계를 고려한다. 바서슈타인(Wasserstein)은 확률 질량을 옮긴다. 각각은 "유사하다"가 무엇을 의미하는지에 대한 서로 다른 가정을 부호화한다.

이 레슨은 모든 주요 거리 함수를 밑바닥부터 만들고, 각각이 언제 올바른 도구인지 보여주며, 같은 데이터가 어떤 척도를 쓰느냐에 따라 완전히 다른 최근접 이웃을 만드는 것을 시연한다.

## 개념 (The Concept)

### 노름: 벡터 크기 측정

노름(norm)은 벡터의 "크기"를 측정한다. 두 벡터 사이의 모든 거리 함수는 그 차이의 노름으로 쓸 수 있다. d(a, b) = ||a - b||. 따라서 노름을 이해하는 것은 거리를 이해하는 것이다.

### L1 노름 (맨해튼 거리, Manhattan distance)

L1 노름은 모든 성분의 절댓값을 합한다.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

축을 따라서만 움직일 수 있는 도시 격자에서 얼마나 걸어야 하는지를 측정하기 때문에 맨해튼 거리라고 불린다. 대각선은 없다.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1을 쓸 때:
- 고차원 희소(sparse) 데이터(텍스트 특성, 원핫 인코딩)
- 이상치(outlier)에 대한 강건성을 원할 때(하나의 거대한 차이가 지배하지 않음)
- 특성 선택 문제(L1 정규화는 희소성을 촉진함)

L1 정규화(Lasso)와의 연결: 손실 함수에 ||w||_1을 더하면 가중치(weight) 절댓값의 합에 페널티를 준다. 이는 작은 가중치를 정확히 0으로 밀어붙여 자동 특성 선택을 수행한다. L1 페널티는 가중치 공간에서 다이아몬드 모양의 제약 영역을 만들고, 다이아몬드의 꼭짓점은 일부 가중치가 0인 축 위에 놓인다.

손실 함수와의 연결: 평균 절대 오차(Mean Absolute Error, MAE)는 예측과 목표값 사이의 평균 L1 거리다. 모든 오차에 선형으로 페널티를 주어 MSE에 비해 이상치에 강건하다.

### L2 노름 (유클리드 거리, Euclidean distance)

L2 노름은 직선 거리다. 제곱한 성분들의 합의 제곱근이다.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

기하학 수업에서 배운 바로 그 거리다. n차원에서의 피타고라스.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2를 쓸 때:
- 저차원에서 중차원의 연속 데이터
- 특성 스케일이 비슷할 때
- 물리적 거리(공간 데이터, 센서 측정값)
- 픽셀 수준의 이미지 유사도

L2 정규화(Ridge)와의 연결: 손실 함수에 ||w||_2^2를 더하면 큰 가중치에 페널티를 준다. L1과 달리 가중치를 0으로 밀지 않는다. 모든 가중치를 0을 향해 비례적으로 줄인다. L2 페널티는 원형 제약 영역을 만들어 축 위에 꼭짓점이 없다. 가중치는 작아지지만 정확히 0이 되는 경우는 드물다.

손실 함수와의 연결: 평균 제곱 오차(Mean Squared Error, MSE)는 L2 거리 제곱의 평균이다. 제곱은 작은 오차보다 큰 오차에 더 무겁게 페널티를 준다.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 노름: 일반화된 계열

L1과 L2는 Lp 노름의 특수한 경우다.

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p의 값이 다르면 다른 모양의 "단위 공(unit ball)"(원점으로부터 거리 1에 있는 모든 점의 집합)을 만든다.

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-무한대 노름 (체비쇼프 거리, Chebyshev distance)

p가 무한대에 가까워지면 Lp 노름은 최대 절댓값 성분으로 수렴한다.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

두 점 사이의 거리는 그들이 가장 많이 다른 단일 차원에 의해 결정된다. 다른 모든 차원은 무시된다.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-무한대를 쓸 때:
- 어떤 단일 차원에서든 최악의 편차가 중요할 때
- 게임 보드(체스의 킹은 L-무한대로 움직인다: 어느 방향으로든 한 칸 이동에 비용 1)
- 제조 공차(모든 차원이 규격 안에 있어야 함)

### 코사인 유사도와 코사인 거리

코사인 유사도(cosine similarity)는 두 벡터의 크기를 무시하고 그 사이의 각도를 측정한다.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

-1(반대 방향)에서 +1(같은 방향)까지 분포한다. 수직인 벡터는 코사인 유사도 0을 갖는다.

코사인 거리는 이를 거리로 변환한다: cosine_distance = 1 - cosine_similarity. 이는 0(동일한 방향)에서 2(반대 방향)까지 분포한다.

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

코사인이 NLP와 임베딩을 지배하는 이유: 텍스트에서 문서 길이는 유사도에 영향을 주지 않아야 한다. 고양이에 관한 어떤 문서가 고양이에 관한 다른 문서보다 두 배 길더라도 여전히 "유사"해야 한다. 코사인 유사도는 크기(길이)를 무시하고 방향만 신경 쓴다. 같은 단어 분포를 가지지만 길이가 다른 두 문서는 같은 방향을 가리키며 코사인 유사도 1.0을 얻는다.

코사인 유사도를 쓸 때:
- 텍스트 유사도(TF-IDF 벡터, 단어 임베딩, 문장 임베딩)
- 크기가 잡음이고 방향이 신호인 모든 도메인
- 추천 시스템(사용자 선호 벡터)
- 임베딩 검색(벡터 데이터베이스는 거의 항상 코사인이나 내적을 쓴다)

### 내적 유사도 vs 코사인 유사도

두 벡터의 내적(dot product)은 다음과 같다.

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

코사인 유사도는 두 크기로 정규화된 내적이다. 두 벡터가 이미 단위 정규화(unit-normalized)되어 있으면(크기 = 1), 내적과 코사인 유사도는 동일하다.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

이들이 다를 때: 내적은 크기 정보를 포함한다. 크기가 더 큰 벡터는 더 높은 내적 점수를 얻는다. 이는 "인기 있는" 항목을 더 높게 순위 매기고 싶은 일부 검색 시스템에서 중요하다. 크기는 암묵적인 품질 또는 중요도 신호 역할을 한다.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

실제로:
- 순수한 방향 유사도를 원할 때는 코사인 유사도를 써라
- 크기가 의미 있는 정보를 담을 때는 내적을 써라
- 많은 벡터 데이터베이스(Pinecone, Weaviate, Qdrant)는 둘 중에서 선택하게 한다
- 임베딩이 L2 정규화되어 있다면 선택은 중요하지 않다

### 마할라노비스 거리 (Mahalanobis Distance)

유클리드 거리는 모든 차원을 동등하게 취급한다. 하지만 특성이 상관되어 있거나 다른 스케일을 가지면 L2는 오도하는 결과를 준다.

마할라노비스 거리는 데이터의 공분산(covariance) 구조를 고려한다.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

여기서 S는 데이터의 공분산 행렬(covariance matrix)이다.

직관적으로: 마할라노비스 거리는 먼저 데이터를 탈상관화하고 정규화한(백색화, whitening) 뒤, 그 변환된 공간에서 L2 거리를 계산한다. S가 항등 행렬이면(무상관, 단위 분산 특성), 마할라노비스 거리는 유클리드 거리로 환원된다.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

마할라노비스 거리를 쓸 때:
- 이상치 탐지(평균으로부터 마할라노비스 거리가 큰 점은 이상치)
- 특성이 다른 스케일과 상관관계를 가질 때의 분류
- 신뢰할 만한 공분산 행렬을 추정할 만큼 충분한 데이터가 있을 때
- 제조 품질 관리(다변량 공정 모니터링)

### 자카드 유사도 (집합용)

자카드 유사도(Jaccard similarity)는 두 집합 사이의 겹침을 측정한다.

```
J(A, B) = |A intersect B| / |A union B|
```

0(겹침 없음)에서 1(동일한 집합)까지 분포한다. 자카드 거리 = 1 - 자카드 유사도.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

자카드를 쓸 때:
- 태그, 카테고리, 특성의 집합을 비교
- 단어 존재 여부에 기반한 문서 유사도(빈도가 아님)
- 근사 중복 탐지(자카드의 MinHash 근사)
- 이진 특성 벡터 비교(존재/부재 데이터)
- 분할(segmentation) 모델 평가(Intersection over Union = 자카드)

### 편집 거리 (레벤슈타인 거리, Levenshtein Distance)

편집 거리(edit distance)는 한 문자열을 다른 문자열로 변환하는 데 필요한 단일 문자 연산의 최소 개수를 센다. 연산은 삽입, 삭제, 치환이다.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

동적 계획법(dynamic programming)으로 계산한다. 항목 (i, j)가 문자열 A의 첫 i개 문자와 문자열 B의 첫 j개 문자 사이의 편집 거리인 행렬을 채운다.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

편집 거리를 쓸 때:
- 맞춤법 검사 및 교정
- DNA 서열 정렬(가중 연산 포함)
- 퍼지 문자열 매칭
- 지저분한 텍스트 데이터의 중복 제거

### KL 발산 (거리는 아니지만 거리처럼 쓰임)

KL 발산(KL divergence)은 한 확률 분포(probability distribution)가 다른 확률 분포와 어떻게 다른지 측정한다. Lesson 09에서 다루었지만, 거리가 아님에도 사람들이 "거리"로 쓰기 때문에 이 논의에 속한다.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

결정적인 성질: KL 발산은 대칭이 아니다.

```
D_KL(P || Q) != D_KL(Q || P)
```

이는 거리 척도의 기본 요구 사항을 충족하지 못한다는 뜻이다. 또한 삼각 부등식도 만족하지 않는다. 그것은 발산이지 거리가 아니다.

순방향 KL(D_KL(P || Q))은 "평균 추구(mean-seeking)"다: Q가 P의 모든 모드(mode)를 덮으려 한다.
역방향 KL(D_KL(Q || P))은 "모드 추구(mode-seeking)"다: Q가 P의 단일 모드에 집중한다.

KL 발산을 볼 때:
- VAE(ELBO의 KL 항이 잠재 분포를 사전 분포 쪽으로 민다)
- 지식 증류(knowledge distillation, 학생이 교사의 분포를 맞추려 함)
- RLHF(KL 페널티가 파인튜닝된 모델을 기반 모델 가까이에 유지함)
- 정책 경사(policy gradient) 방법(정책 갱신을 제약함)

### 바서슈타인 거리 (Earth Mover's Distance)

바서슈타인 거리(Wasserstein distance)는 한 확률 분포를 다른 확률 분포로 변환하는 데 필요한 최소 "작업(work)"을 측정한다. 이렇게 생각하라: 한 분포가 흙더미이고 다른 하나가 구덩이라면, 흙을 얼마나 멀리 얼마나 많이 옮겨야 하는가?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

1D 분포의 경우, 누적 분포 함수의 절대 차이의 적분으로 단순화된다.

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

바서슈타인이 중요한 이유:
- 그것은 진짜 척도다(대칭이며 삼각 부등식을 만족함)
- 분포가 겹치지 않을 때조차 그래디언트(gradient)를 제공한다(KL 발산은 무한대로 간다)
- 이 성질이 그것을 바서슈타인 GAN(WGAN)의 핵심으로 만들었으며, 이는 원래 GAN의 학습(training) 불안정성을 해결했다

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

바서슈타인을 쓸 때:
- GAN 학습(WGAN, WGAN-GP)
- 겹치지 않을 수 있는 분포 비교
- 최적 수송(optimal transport) 문제
- 이미지 검색(색상 히스토그램 비교)

### 서로 다른 과제가 서로 다른 거리를 필요로 하는 이유

| 과제 | 최선의 거리 | 이유 |
|------|--------------|-----|
| 텍스트 유사도 | 코사인 | 크기는 잡음, 방향이 의미 |
| 이미지 픽셀 비교 | L2 | 공간 관계가 중요, 특성 스케일이 비슷 |
| 희소 고차원 특성 | L1 | 강건하며 드문 큰 차이를 증폭하지 않음 |
| 집합 겹침(태그, 카테고리) | 자카드 | 데이터가 본질적으로 벡터가 아닌 집합값 |
| 문자열 매칭 | 편집 거리 | 연산이 사람의 편집 직관에 대응 |
| 이상치 탐지 | 마할라노비스 | 특성 상관관계와 스케일을 고려 |
| 분포 비교 | KL 발산 | P 대신 Q를 써서 잃은 정보를 측정 |
| GAN 학습 | 바서슈타인 | 분포가 겹치지 않아도 그래디언트 제공 |
| 임베딩(벡터 DB) | 코사인 또는 내적 | 임베딩은 방향에 의미를 부호화하도록 학습됨 |
| 추천 | 내적 | 크기가 인기도나 신뢰도를 부호화할 수 있음 |
| DNA 서열 | 가중 편집 거리 | 치환 비용이 뉴클레오타이드 쌍에 따라 다름 |
| 제조 QC | L-무한대 | 어떤 차원에서든 최악의 편차가 중요 |

### 손실 함수와의 연결

손실 함수는 예측 대 목표값에 적용된 거리 함수다.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 정규화와의 연결

정규화(regularization)는 손실 함수에 가중치에 대한 노름 페널티를 더한다.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

L1은 희소성을 만들지만 L2는 그렇지 않은 이유: 2D 가중치 공간에서 제약 영역을 그려보라. L1은 다이아몬드, L2는 원이다. 손실 함수의 등고선(타원)은 다이아몬드를 한 가중치가 0인 꼭짓점에서 닿을 가능성이 가장 높다. 원에 대해서는 두 가중치가 모두 0이 아닌 매끄러운 점에서 닿는다.

### 최근접 이웃 탐색

모든 거리 함수는 최근접 이웃 탐색 문제를 함의한다: 질의 점이 주어지면 데이터셋에서 가장 가까운 점들을 찾는다.

정확한 최근접 이웃 탐색은 n개 점과 d차원의 데이터셋에서 질의당 O(n * d)다. 큰 데이터셋에서는 이것이 너무 느리다.

근사 최근접 이웃(Approximate Nearest Neighbor, ANN) 알고리즘은 약간의 정확도를 내주고 막대한 속도 향상을 얻는다.

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW(Hierarchical Navigable Small World)는 현대 벡터 데이터베이스에서 지배적인 알고리즘이다. 각 노드가 근사 최근접 이웃에 연결되는 다층 그래프를 만든다. 탐색은 최상위 층(희소, 긴 도약)에서 시작하여 최하위 층(조밀, 짧은 도약)으로 내려간다.

```figure
norm-unit-balls
```

## 직접 만들기 (Build It)

### 1단계: 모든 노름과 거리 함수

완전한 구현은 `code/distances.py`를 참조하라. 모든 함수는 기본 Python 수학만 사용해 밑바닥부터 만들어진다.

### 2단계: 같은 데이터, 다른 거리, 다른 이웃

`distances.py`의 데모는 데이터셋을 만들고, 질의 점을 고르고, 거리 척도에 따라 최근접 이웃이 어떻게 바뀌는지 보여준다. L1에서 "가장 가까운" 점이 L2나 코사인에서는 가장 가깝지 않을 수 있다.

### 3단계: 임베딩 유사도 탐색

코드에는 코사인 유사도 대 L2 거리를 사용해 질의와 가장 유사한 "문서"를 찾는 모의 임베딩 유사도 탐색이 포함되어 있으며, 순위가 다를 수 있음을 보여준다.

## 라이브러리로 써보기 (Use It)

가장 흔한 실용적 용도: 벡터 데이터베이스에서 유사 항목 찾기.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

`model.encode(text)`를 호출한 뒤 벡터 데이터베이스를 검색할 때, 내부에서 일어나는 일이 바로 이것이다. 임베딩 모델이 텍스트를 벡터로 매핑한다. 벡터 데이터베이스는 ANN 알고리즘을 사용해 모든 벡터를 확인하는 것을 피하면서 질의 벡터와 저장된 모든 벡터 사이의 코사인 유사도(또는 내적)를 계산한다.

## 연습 문제 (Exercises)

1. (1, 2, 3)과 (4, 0, 6) 사이의 L1, L2, L-무한대 거리를 계산하라. 어떤 점 쌍에 대해서도 항상 L-inf <= L2 <= L1이 성립함을 검증하라. 이 순서가 보장되는 이유를 증명하라.

2. 코사인 유사도가 높지만(> 0.9) L2 거리가 큰(> 10) 두 벡터를 만들어라. 기하학적으로 무슨 일이 일어나는지 설명하라. 그다음 코사인 유사도가 낮지만(< 0.3) L2 거리가 작은(< 0.5) 두 벡터를 만들어라.

3. 데이터셋과 질의 점을 받아 L1, L2, 코사인, 마할라노비스 거리 아래의 최근접 이웃을 반환하는 함수를 구현하라. 네 가지 모두가 어느 점이 가장 가까운지에 대해 의견이 갈리는 데이터셋을 찾아라.

4. CDF 방법을 사용해 [0.5, 0.5, 0, 0]과 [0, 0, 0.5, 0.5] 사이의 바서슈타인 거리를 손으로 계산하라. 그다음 [0.25, 0.25, 0.25, 0.25]와 [0, 0, 0.5, 0.5] 사이를 계산하라. 어느 것이 더 크며 왜 그런가?

5. 근사 자카드 유사도를 위한 MinHash를 구현하라. 100개의 무작위 집합을 생성하고, 모든 쌍에 대해 정확한 자카드를 계산하고, 50, 100, 200개의 해시 함수를 사용한 MinHash 근사와 비교하라. 근사 오차를 플롯하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| 노름(Norm) | "벡터의 크기" | 벡터를 음이 아닌 스칼라로 매핑하는 함수로, 삼각 부등식, 절대 동차성을 만족하며 영벡터에 대해서만 0이 됨 |
| L1 노름 | "맨해튼 거리" | 절대 성분 값의 합. 최적화에서 희소성을 만들고 이상치에 강건함 |
| L2 노름 | "유클리드 거리" | 제곱 성분 합의 제곱근. 유클리드 공간에서의 직선 거리 |
| Lp 노름 | "일반화된 노름" | 절대 성분의 p제곱 합의 p제곱근. L1과 L2는 특수한 경우 |
| L-무한대 노름 | "최대 노름" 또는 "체비쇼프 거리" | 최대 절대 성분 값. p가 무한대로 갈 때의 Lp의 극한 |
| 코사인 유사도(Cosine similarity) | "벡터 사이의 각도" | 두 크기로 정규화된 내적. -1에서 +1까지. 벡터 길이를 무시 |
| 코사인 거리(Cosine distance) | "1 빼기 코사인 유사도" | 코사인 유사도를 거리로 변환. 0에서 2까지 |
| 내적(Dot product) | "정규화 안 된 코사인" | 성분별 곱의 합. 코사인 유사도에 두 크기를 곱한 값과 같음 |
| 마할라노비스 거리(Mahalanobis distance) | "상관관계를 인지하는 거리" | 데이터 공분산 행렬을 사용해 백색화된(탈상관화 및 정규화된) 공간에서의 L2 거리 |
| 자카드 유사도(Jaccard similarity) | "집합 겹침" | 교집합 크기를 합집합 크기로 나눈 값. 벡터가 아닌 집합용 |
| 편집 거리(Edit distance) | "레벤슈타인 거리" | 한 문자열을 다른 문자열로 변환하는 최소 삽입, 삭제, 치환 |
| KL 발산(KL divergence) | "분포 사이의 거리" | 진짜 거리가 아님(비대칭). Q로 P를 부호화할 때 드는 추가 비트를 측정 |
| 바서슈타인 거리(Wasserstein distance) | "Earth mover's distance" | 한 분포에서 다른 분포로 질량을 수송하는 최소 작업. 진짜 척도 |
| 근사 최근접 이웃(Approximate nearest neighbor) | "ANN 탐색" | 정확한 탐색보다 훨씬 빠르게 근사적으로 가장 가까운 점을 찾는 알고리즘(HNSW, LSH, IVF) |
| HNSW | "벡터 DB 알고리즘" | Hierarchical Navigable Small World 그래프. 빠른 근사 최근접 이웃 탐색을 위한 다층 그래프 |
| L1 정규화 | "Lasso" | 손실에 가중치의 L1 노름을 더함. 가중치를 0으로 몰아감(희소성) |
| L2 정규화 | "Ridge" 또는 "weight decay" | 손실에 가중치의 제곱 L2 노름을 더함. 희소성 없이 가중치를 0을 향해 줄임 |
| Elastic Net | "L1 + L2" | L1과 L2 정규화를 결합. 둘 중 어느 하나보다 상관된 특성 그룹을 더 잘 다룸 |

## 더 읽을거리 (Further Reading)

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - 십억 규모 ANN 탐색을 위한 Meta의 라이브러리
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - Earth Mover's distance를 GAN에 도입한 논문
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 기초적인 ANN 알고리즘
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, 코사인 유사도가 임베딩의 기본값이 된 곳
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn에서의 거리 척도와 이웃 알고리즘에 대한 실용 가이드
