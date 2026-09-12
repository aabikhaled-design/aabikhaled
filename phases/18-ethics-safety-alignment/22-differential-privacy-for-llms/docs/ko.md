# LLM을 위한 차등 프라이버시

> DP-SGD는 여전히 표준이다. 노이즈를 주입한 그래디언트(gradient) 갱신이 형식적인 (epsilon, delta) 보장을 제공한다. 계산, 메모리, 효용(utility) 측면의 오버헤드는 상당하다. 파라미터 효율적(parameter-efficient) DP 파인튜닝(fine-tuning)(LoRA + DP-SGD)이 2025년의 일반적인 구성이다(ACM 2025). 긴장 관계에 있는 두 가지 증거 묶음이 있다. 카나리(canary) 기반 멤버십 추론(membership inference)(Duan et al., 2024)은 언어 모델에 대해 제한적인 성공만을 보고한다. 학습 데이터 추출(training-data extraction)(Carlini et al., 2021; Nasr et al., 2025)은 상당한 양의 축자적(verbatim) 암기를 복원한다. 해소(arXiv:2503.06808, 2025년 3월): 그 간극은 무엇을 측정하는가에 있다. 삽입된 카나리 대 "가장 추출 가능한" 데이터. 새로운 카나리 설계는 섀도 모델(shadow model) 없이 손실 기반(loss-based) MIA를 가능하게 하며, 현실적인 DP 보장을 가진 실제 데이터로 학습된 LLM에 대한 최초의 비자명한 DP 감사(audit)를 산출한다. 대안들: PMixED(arXiv:2403.15638): 다음 토큰(next-token) 분포에 대한 전문가 혼합(mixture of experts)을 통한 추론 시점(inference time)의 사적 예측(private prediction). DP 합성 데이터(synthetic data) 생성(Google Research 2024). 새로 떠오르는 공격: LLM 피드백을 통한 차등 프라이버시 역전(Differential Privacy Reversal via LLM Feedback): 신뢰도 점수(confidence score) 누출.

**Type:** Build
**Languages:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- (epsilon, delta)-차등 프라이버시(differential privacy)를 정의하고 DP-SGD 레시피를 진술하기.
- 2024-2025년의 긴장 관계를 설명하기: 카나리 MIA 대 학습 데이터 추출이 서로 다른 그림을 제시한다.
- PMixED를 기술하고, 추론 시점의 사적 예측이 왜 DP 학습의 대안인지 설명하기.
- LLM 피드백을 통한 차등 프라이버시 역전 공격을 기술하기.

## 문제 (The Problem)

LLM은 암기한다. Carlini et al. 2021은 프로덕션 언어 모델이 요청에 따라 학습 텍스트를 축자적으로 재현함을 보였다. DP는 형식적 방어책이다. 출력이 어떤 단일 학습 예제에도 증명 가능하게(provably) 둔감하도록 학습시킨다. 2024-2025년의 증거는 DP-SGD가 필요하지만, 배포된 ε 값이 위협 모델(threat model)과 맞지 않을 수 있음을 보여준다.

## 개념 (The Concept)

### (ε, δ)-차등 프라이버시

무작위화 알고리즘 M은, 한 예제만 다른 임의의 두 데이터셋과 임의의 사건 S에 대해 다음이 성립하면 (ε, δ)-DP이다:
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

해석: 출력 분포가 충분히 가까워서(ε로 매개변수화됨), δ 확률을 제외하면 어떤 단일 개인의 기여도 신뢰성 있게 추론될 수 없다.

### DP-SGD

Abadi et al. 2016. 표준 레시피:
1. 미니배치(mini-batch)를 샘플링한다.
2. 예제별 그래디언트를 계산한다.
3. 각 예제별 그래디언트를 임계값 C로 클리핑(clip)한다.
4. 클리핑된 그래디언트를 합산하고 표준편차 σ * C인 가우시안 노이즈(Gaussian noise)를 더한다.
5. 노이즈가 섞인 합을 사용해 파라미터(parameter)를 갱신한다.

프라이버시 비용은 회계기(accountant)(Moments Accountant, Rényi DP accountant)로 추적한다. LLM 문헌에서 보고된 ε 값은 위협 모델, 데이터 민감도, 효용 목표에 따라 크게 다르다. 보편적으로 "안전한" 기본 ε는 없다. 일부 LLM 학습 환경에서 발표된 예시는 대략 ε ≈ 1–10 범위에 걸쳐 있지만, 이들은 예시일 뿐 권장 기본값이 아니다. 낮은 ε는 일반적으로 더 많은 노이즈를 요구하며 효용 손실을 늘릴 수 있다.

### LoRA + DP-SGD

프런티어 모델(frontier model)에 대한 완전한 DP-SGD는 비용이 감당하기 어렵다. LoRA(Hu et al. 2022)는 그래디언트 갱신을 작은 어댑터(adapter)로 제한하여, 예제별 그래디언트 저장량을 줄인다. LoRA + DP-SGD는 2025년의 일반적인 구성이다. DP 보장은 어댑터에 적용되며, 기반 모델(base model)은 고정된다.

### 2024-2025년의 긴장 관계

두 갈래의 증거:

- **카나리 MIA(Duan et al. 2024).** 고유한 카나리를 학습 데이터에 삽입하고, 멤버십 추론 공격자가 그것들을 식별할 수 있는지 측정한다. 언어 모델에 대해 제한적인 성공을 보고한다. MIA가 어렵다는 뜻이다.
- **학습 데이터 추출(Carlini 2021, Nasr et al. 2025).** 모델에 접두사(prefix)를 프롬프트로 주고, 학습으로부터 축자적 텍스트를 복원하는지 측정한다. 상당한 암기를 보고한다. 관련된 의미에서 MIA가 쉽다는 뜻이다.

2025년 3월의 해소(arXiv:2503.06808): 두 가지는 서로 다른 것을 측정한다. MIA는 삽입된 카나리에 대해 "예제 e가 D에 있는가?"를 묻는다. 추출은 "내가 D로부터 무엇을 복원할 수 있는가?"를 묻는다. 프라이버시에 중요한 것은 "가장 추출 가능한" 예제이며, 카나리는 추출 가능하도록 최적화되지 않았기 때문에 이를 과소 보고한다.

새로운 카나리 설계. 섀도 모델 없는 손실 기반 MIA. 현실적인 DP 보장을 가진 실제 데이터에 대한 LLM의 최초의 비자명한 DP 감사.

### DP 학습의 대안들

- **PMixED(arXiv:2403.15638).** 추론 시점의 사적 예측. 다음 토큰 분포에 대한 전문가 혼합. 각 전문가는 학습 데이터의 한 조각(shard)을 본다. 집계(aggregation)는 DP를 위해 노이즈를 더한다. DP 학습을 완전히 피한다.
- **DP 합성 데이터 생성(Google Research 2024).** DP-SGD로 LoRA 파인튜닝하고, 합성 데이터를 샘플링하며, 그 합성 데이터로 다운스트림(downstream) 분류기를 학습시킨다.

둘 다 다른 위협 모델을 대가로 완전한 DP 학습의 효용 비용을 우회한다.

### LLM 피드백을 통한 차등 프라이버시 역전

새로 떠오르는 2025년 공격. DP로 학습된 모델의 신뢰도 점수를 오라클(oracle)로 사용해 개인을 재식별한다. 출력이 누출되지 않을 때조차, 신뢰도 분포는 누출될 수 있다.

방어책: 신뢰도를 노출하지 않거나, 노출 전에 절단(truncate)/양자화(quantize)한다. 이는 (ε, δ)-DP 학습을 넘어서는 추가 요구 사항이다.

### Phase 18에서 이 레슨의 위치

레슨 20-21은 편향/공정성이다. 레슨 22는 프라이버시다. 레슨 23은 워터마킹을 통한 출처(provenance)다. 레슨 27은 규제 데이터 출처 계층을 다룬다.

```figure
an-dp-clip-noise
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 수준의 이진 분류(binary classification) 데이터셋에 대해 DP-SGD를 시뮬레이션한다. 노이즈 승수(noise multiplier) σ와 클리핑 노름(clipping norm) C를 훑으면서 (ε, δ) 예산과 정확도 비용을 추적할 수 있다. "카나리 공격"은 고유한 학습 예제를 삽입하고, DP 적용 전후에 로그 손실(log-loss) 테스트가 그것을 탐지할 수 있는지 측정한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-dp-audit.md`를 생성한다. 언어 모델 배포(deployment)에 대한 DP 주장이 주어지면, 다음을 감사한다: (ε, δ) 값, 사용된 회계기, MIA 평가 프로토콜, 그리고 신뢰도 노출 벡터(confidence-exposure vector)가 평가되었는지 여부.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. σ를 {0.5, 1.0, 2.0}에서 훑고 (ε, δ)-정확도 트레이드오프를 보고하라. 효용이 붕괴하는 지점을 식별하라.

2. 카나리 삽입과 로그 손실 테스트를 구현하라. σ = 1.0에서 DP-SGD 적용 전후의 탐지율을 측정하라.

3. 학습 데이터 추출에 관한 Nasr et al. 2025를 읽어라. 왜 추출 성공이 적당한 ε 하에서 붕괴하지 않는가? 이것이 평가 수단으로서의 MIA에 대해 무엇을 함의하는가?

4. 전적으로 추론 시점에서 작동하는, PMixED(arXiv:2403.15638)를 사용한 배포를 설계하라. DP-SGD는 다루지 못하지만 PMixED가 다루는 위협 모델은 무엇인가?

5. LLM 피드백을 통한 DP 역전 공격을 개략적으로 그려라. 신뢰도 점수 누출을 제한하는 대응책을 설계하고 그 배포 비용을 추정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| DP | "(ε, δ)-차등 프라이버시" | 형식적 프라이버시: 이웃 데이터셋 변경 하에서 출력 분포가 가까움 |
| DP-SGD | "노이즈 주입 SGD" | 그래디언트 클리핑 + 가우시안 노이즈 추가. 표준 DP 학습 |
| LoRA + DP-SGD | "효율적인 사적 파인튜닝" | 저랭크(low-rank) 어댑터에 대한 DP-SGD. 2025년 표준 구성 |
| MIA | "멤버십 추론" | 예제가 학습 데이터에 있었는지 판정하는 공격 |
| 카나리(Canary) | "삽입된 워터마크 예제" | DP 누출을 측정하는 데 쓰이는 고유한 학습 예제 |
| PMixED | "사적 추론 혼합" | 다음 토큰 분포에 대한 전문가 혼합을 통한 추론 시점 DP |
| DP 역전(DP Reversal) | "신뢰도 누출 공격" | 모델의 신뢰도를 재식별 오라클로 사용하는 공격 |

## 더 읽을거리 (Further Reading)

- [Abadi et al.: DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) 표준 DP 학습 알고리즘
- [Carlini et al.: Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) 표준적인 추출 논문
- [Duan et al.: Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) 제한적 성공의 MIA
- [Kowalczyk et al.: Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) 긴장 관계의 해소
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638): 추론 시점의 사적 예측
