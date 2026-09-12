# 머신러닝을 위한 통계 (Statistics for Machine Learning)

> 통계는 모델이 실제로 동작하는지 아니면 그저 운이 좋았던 것인지 알게 해주는 방법이다.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 기술 통계(descriptive statistics), 피어슨/스피어만 상관(Pearson/Spearman correlation), 공분산 행렬(covariance matrix)을 밑바닥부터 계산하기
- 가설 검정(hypothesis test, t-검정, 카이제곱)을 수행하고 p값(p-value)과 신뢰 구간(confidence interval)을 올바르게 해석하기
- 분포 가정 없이 어떤 척도에 대해서든 신뢰 구간을 구성하기 위해 부트스트랩 재표본추출(bootstrap resampling) 사용하기
- 효과 크기(effect size) 측정을 사용해 통계적 유의성과 실질적 유의성 구별하기

## 문제 (The Problem)

두 모델(model)을 학습시켰다고 하자. 모델 A는 테스트 세트에서 0.87을 기록한다. 모델 B는 0.89를 기록한다. 그래서 모델 B를 배포한다. 3주 후, 프로덕션(production) 지표가 이전보다 나쁘다. 무슨 일이 일어난 걸까?

모델 B는 실제로 모델 A를 능가하지 않았다. 0.02 차이는 잡음이었다. 테스트 세트가 너무 작았거나, 분산이 너무 컸거나, 둘 다였다. 개선으로 치장된 무작위성을 배포한 것이다.

이런 일은 끊임없이 일어난다. Kaggle 리더보드 순위 뒤집힘. 재현에 실패하는 논문. 수백 개의 표본을 근거로 승자를 선언하는 A/B 테스트. 근본 원인은 항상 같다. 누군가 통계를 건너뛴 것이다.

통계는 신호와 잡음을 구별하는 도구를 준다. 차이가 언제 진짜인지, 얼마나 확신해야 하는지, 결과를 믿기 전에 얼마나 많은 데이터가 필요한지 알려준다. 모든 ML 파이프라인(pipeline), 모든 모델 비교, 모든 실험에는 통계가 필요하다. 통계 없이는 그저 추측하는 셈이다.

## 개념 (The Concept)

### 기술 통계: 데이터 요약

무언가를 모델링하기 전에, 데이터가 어떻게 생겼는지 알아야 한다. 기술 통계(descriptive statistics)는 데이터셋(dataset)을 그 형태를 포착하는 몇 개의 숫자로 압축한다.

**중심 경향(central tendency) 척도**는 "중간이 어디인가?"에 답한다.

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

평균(mean)은 균형점이다. 중앙값(median)은 절반 지점이다. 둘이 갈라지면 분포(distribution)가 치우친 것이다. 소득 분포는 평균 >> 중앙값(억만장자들로 인한 오른쪽 치우침)을 가진다. 학습(training) 중 손실(loss) 분포는 종종 평균 << 중앙값(쉬운 표본들로 인한 왼쪽 치우침)을 가진다.

**산포(spread) 척도**는 "데이터가 얼마나 흩어져 있는가?"에 답한다.

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**백분위수(percentile)**는 정렬된 데이터를 100개의 동일한 부분으로 나눈다. 25번째 백분위수(Q1)는 값의 25%가 이 지점 아래에 있다는 뜻이다. 50번째 백분위수는 중앙값이다. 75번째 백분위수는 Q3다.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

ML에서는 추론(inference) 지연 시간(latency), 예측 신뢰도 분포, 오차 분포 이해를 위해 백분위수가 중요하다. 평균 오차는 낮지만 P99 오차가 끔찍한 모델은 안전이 중요한 응용에서는 쓸모없을 수 있다.

**표본 통계 vs 모집단 통계.** 표본에서 분산을 계산할 때는 n 대신 (n-1)로 나눈다. 이것이 베셀 보정(Bessel's correction)이다. 표본 평균이 참 모집단 평균이 아니라는 사실을 보정한다. 분모에 n을 쓰면 참 분산을 체계적으로 과소추정한다. (n-1)을 쓰면 추정치가 불편(unbiased)해진다.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

실제로: n이 크면(수천 개 표본) 차이는 무시할 만하다. n이 작으면(수십 개 표본) 중요하다.

### 상관: 변수들이 함께 움직이는 방식

상관(correlation)은 두 변수 사이의 선형 관계의 강도와 방향을 측정한다.

**피어슨 상관 계수(Pearson correlation coefficient)**는 선형 연관을 측정한다.

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

피어슨은 관계가 선형이고 두 변수가 대략 정규 분포(normally distributed)를 따른다고 가정한다. 이상치(outlier)에 민감하다. 단 하나의 극단값이 r을 0.1에서 0.9로 끌어당길 수 있다.

**스피어만 순위 상관(Spearman rank correlation)**은 단조(monotonic) 연관을 측정한다.

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**각각을 언제 쓰는가:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**황금률:** 상관이 인과를 함의하지 않는다. 아이스크림 판매량과 익사 사망자 수는 둘 다 여름에 증가하기 때문에 상관되어 있다. 모델의 정확도와 파라미터(parameter) 수는 상관되어 있지만, 파라미터를 추가한다고 자동으로 정확도가 개선되지는 않는다(과적합 참조).

### 공분산 행렬 (Covariance Matrix)

두 변수 사이의 공분산(covariance)은 그들이 함께 변하는 방식을 측정한다.

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

d개의 특성(feature)에 대해 공분산 행렬 C는 C[i][j] = Cov(feature_i, feature_j)인 d x d 행렬이다. 대각 항목 C[i][i]는 각 특성의 분산이다.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**PCA와의 연결.** PCA는 공분산 행렬을 고유분해(eigendecompose)한다. 고유벡터(eigenvector)는 주성분(principal component, 최대 분산의 방향)이다. 고윳값(eigenvalue)은 각 성분이 얼마나 많은 분산을 포착하는지 알려준다. 이것이 정확히 Lesson 10에서 다룬 내용이지만, 이제 공분산 행렬이 분해하기에 올바른 대상인 이유를 알 수 있다. 그것은 데이터의 모든 쌍별 선형 관계를 부호화한다.

**상관과의 연결.** 상관 행렬은 표준화된 변수(각각 표준편차로 나눈 것)의 공분산 행렬이다. 상관은 공분산을 정규화하여 모든 값이 [-1, 1]에 들어가게 한다.

### 가설 검정 (Hypothesis Testing)

가설 검정(hypothesis testing)은 불확실성 아래에서 결정을 내리는 틀이다. 주장으로 시작하고, 데이터를 수집하고, 데이터가 그 주장과 일치하는지 판단한다.

**설정:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p값(p-value)**은 H0이 참이라고 가정했을 때, 관찰한 것만큼 극단적인 데이터를 볼 확률이다. H0이 참일 확률이 아니다. 이것이 통계에서 가장 흔한 단 하나의 오해다.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**신뢰 구간(confidence interval)**은 파라미터에 대해 그럴듯한 값의 범위를 준다.

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

신뢰 구간의 너비는 정밀도에 대해 알려준다. 넓은 구간은 높은 불확실성을 의미한다. 좁은 구간은 추정치가 정밀하다는 뜻이다(단, 데이터가 편향되어 있다면 반드시 정확하지는 않다).

### t-검정

t-검정(t-test)은 평균을 비교한다. 여러 종류가 있다.

**일표본 t-검정(One-sample t-test):** 모집단 평균이 가설값과 다른가?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**이표본 t-검정(독립, Two-sample t-test):** 두 집단의 평균이 다른가?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**대응 t-검정(Paired t-test):** 측정값이 쌍으로 올 때(같은 모델을 같은 데이터 분할에서 평가):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML에서 대응 t-검정은 흔하다. 같은 10개의 교차 검증(cross-validation) 폴드에서 두 모델을 모두 실행하고 그 점수를 쌍별로 비교한다.

### 카이제곱 검정 (Chi-squared Test)

카이제곱 검정(chi-squared test)은 관찰된 빈도가 기대 빈도와 일치하는지 확인한다. 범주형 데이터에 유용하다.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 모델을 위한 A/B 테스트

ML에서의 A/B 테스트는 웹 A/B 테스트와 같지 않다. 모델 비교에는 특유의 어려움이 있다.

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**절차:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 통계적 유의성 vs 실질적 유의성

결과는 통계적으로 유의하지만 실질적으로는 무의미할 수 있다. 데이터가 충분하면 사소한 차이조차 통계적으로 유의해진다.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**효과 크기(effect size)**는 표본 크기와 무관하게 차이가 얼마나 큰지를 정량화한다.

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

항상 p값과 효과 크기를 함께 보고하라. p값은 차이가 진짜인지 알려준다. 효과 크기는 그것이 중요한지 알려준다.

### 다중 비교 문제 (Multiple Comparison Problem)

많은 가설을 검정하면 일부는 우연히 "유의"해진다. alpha = 0.05에서 20개를 검정하면, 아무것도 진짜가 아니어도 1개의 거짓 양성(false positive)을 예상한다.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**본페로니 보정(Bonferroni correction):** alpha를 검정 수로 나눈다.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML에서는 모델을 여러 지표에 걸쳐 비교하거나, 많은 하이퍼파라미터(hyperparameter) 구성을 테스트하거나, 여러 데이터셋에서 평가할 때 이것이 중요하다.

### 부트스트랩 방법 (Bootstrap Methods)

부트스트래핑(bootstrapping)은 데이터를 복원 추출(with replacement)로 재표본추출하여 통계량의 표본 분포(sampling distribution)를 추정한다. 기저 분포에 대한 가정이 필요 없다.

**알고리즘:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**부트스트랩 신뢰 구간(백분위수 방법):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**부트스트랩이 ML에 중요한 이유:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**모델 비교를 위한 부트스트랩:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

이것은 분포 가정을 하지 않기 때문에 대응 t-검정보다 강건하다.

### 모수 검정 vs 비모수 검정

**모수 검정(parametric tests)**은 특정 분포(보통 정규)를 가정한다.

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**비모수 검정(non-parametric tests)**은 분포 가정을 하지 않는다.

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**비모수를 언제 쓰는가:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**모수를 언제 쓰는가:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML 실험에서는 보통 작은 n(5개 또는 10개의 교차 검증 폴드)을 가지므로, Wilcoxon signed-rank 같은 비모수 검정이 t-검정보다 더 적절한 경우가 많다.

### 중심 극한 정리: 실용적 함의

중심 극한 정리(central limit theorem, CLT)는 기저 모집단 분포와 무관하게, n이 커지면 표본 평균의 분포가 정규 분포에 가까워진다고 말한다.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**이것이 ML에 중요한 이유:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT가 하지 않는 것:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 논문에서 흔한 통계적 실수

1. **학습 세트에서 테스트하기.** 과적합(overfitting)을 보장한다. 항상 모델이 학습 중 보지 못하는 데이터를 따로 떼어두라.

2. **신뢰 구간 없음.** 불확실성 없이 단일 정확도 숫자만 보고하면 결과를 재현하거나 검증할 수 없게 된다.

3. **다중 비교 무시.** 50개 구성을 테스트하고 보정 없이 최고의 것을 보고하면 거짓 양성률이 부풀려진다.

4. **통계적 유의성과 실질적 유의성 혼동.** 0.01% 정확도 개선에 대한 p값 0.001은 의미가 없다.

5. **불균형 데이터에서 정확도 사용.** 99% 음성 클래스를 가진 데이터셋에서 99% 정확도는 모델이 아무것도 배우지 못했다는 뜻이다. 정밀도(precision), 재현율(recall), F1, AUC를 써라.

6. **지표 취사선택(cherry-picking).** 모델이 이기는 지표만 보고하기. 정직한 평가는 모든 관련 지표를 보고한다.

7. **학습/테스트 분할 간 정보 누출.** 분할 전에 정규화하거나, 미래 데이터를 써서 과거를 예측하기.

8. **분산 추정 없는 작은 테스트 세트.** 100개 표본에서 평가하고 2% 개선이라고 주장하는 것은 신호가 아니라 잡음이다.

9. **데이터가 독립이 아닐 때 독립이라고 가정.** 같은 환자의 의료 이미지, 같은 문서의 여러 문장. 한 그룹 내의 관찰은 상관되어 있다.

10. **P-해킹.** p < 0.05를 얻을 때까지 다른 검정, 부분집합, 제외 기준을 시도하기. 결과는 탐색의 인공물(artifact)이다.

## 직접 만들기 (Building It)

구현할 것은 다음과 같다.

1. **기술 통계를 밑바닥부터** (평균, 중앙값, 최빈값, 표준편차, 백분위수, IQR)
2. **상관 함수** (피어슨과 스피어만, 공분산 행렬 포함)
3. **가설 검정** (일표본 t-검정, 이표본 t-검정, 카이제곱 검정)
4. **부트스트랩 신뢰 구간** (어떤 통계량에 대해서든, 가정 불필요)
5. **A/B 테스트 시뮬레이터** (데이터 생성, 검정, 제1종 및 제2종 오류 확인)
6. **통계적 vs 실질적 유의성 데모** (큰 n이 모든 것을 "유의"하게 만드는 것을 보여줌)

모두 밑바닥부터, `math`와 `random`만 사용한다. numpy도, scipy도 없다.

```figure
f3-bootstrap-resample
```

## 핵심 용어 (Key Terms)

| 용어 | 정의 |
|---|---|
| 평균(Mean) | 값의 합을 개수로 나눈 것. 이상치에 민감하다. |
| 중앙값(Median) | 정렬된 데이터의 중간 값. 이상치에 강건하다. |
| 표준편차(Standard deviation) | 분산의 제곱근. 원래 단위로 산포를 측정한다. |
| 백분위수(Percentile) | 주어진 비율의 데이터가 그 아래에 있는 값. |
| IQR | 사분위 범위. Q3 빼기 Q1. 가운데 50%의 산포. |
| 피어슨 상관(Pearson correlation) | 두 변수 사이의 선형 연관을 측정. 범위 [-1, 1]. |
| 스피어만 상관(Spearman correlation) | 순위를 사용해 단조 연관을 측정. |
| 공분산 행렬(Covariance matrix) | 모든 특성 사이의 쌍별 공분산 행렬. |
| 귀무가설(Null hypothesis) | 효과나 차이가 없다는 기본 가정. |
| p값(p-value) | 귀무가설이 참일 때 데이터가 이만큼 극단적일 확률. |
| 신뢰 구간(Confidence interval) | 주어진 신뢰 수준에서 파라미터에 대해 그럴듯한 값의 범위. |
| t-검정(t-test) | 평균이 유의하게 다른지 검정. t-분포를 사용한다. |
| 카이제곱 검정(Chi-squared test) | 관찰된 빈도가 기대 빈도와 다른지 검정. |
| 효과 크기(Effect size) | 표본 크기와 무관한 차이의 크기. Cohen's d가 흔하다. |
| 본페로니 보정(Bonferroni correction) | 거짓 양성을 통제하기 위해 유의 임곗값을 검정 수로 나눈다. |
| 부트스트랩(Bootstrap) | 표본 분포를 추정하기 위한 복원 재표본추출. |
| 제1종 오류(Type I error) | 거짓 양성. H0이 참인데 기각하는 것. |
| 제2종 오류(Type II error) | 거짓 음성. H0이 거짓인데 기각하지 못하는 것. |
| 통계적 검정력(Statistical power) | 거짓 H0을 올바르게 기각할 확률. 검정력 = 1 빼기 제2종 오류율. |
| 중심 극한 정리(Central limit theorem) | 표본 크기가 커지면 표본 평균이 정규 분포로 수렴한다. |
| 모수 검정(Parametric test) | 데이터에 대해 특정 분포(보통 정규)를 가정한다. |
| 비모수 검정(Non-parametric test) | 분포 가정을 하지 않는다. 순위나 부호에 작동한다. |
