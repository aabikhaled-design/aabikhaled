# LLM의 편향과 표현적 해악 (Bias and Representational Harm in LLMs)

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed(Computational Linguistics 2024, arXiv:2309.00770). 표현적 해악(representational harm; 고정관념, 삭제)을 배분적 해악(allocational harm; 불평등한 자원 분배)과 구분하고, 평가 지표를 임베딩 기반, 확률 기반, 생성 텍스트 기반으로 범주화한 기초적인 2024년 서베이. 2024-2025년 경험적 연구: An 외(PNAS Nexus, 2025년 3월)는 20개 입문 직무에 대한 자동 이력서 평가에서 GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B 전반의 교차(intersectional) 성별 x 인종 편향을 측정한다. WinoIdentity(COLM 2025, arXiv:2508.07111)는 교차 정체성을 위한 불확실성 기반 공정성 평가를 도입한다. Yu & Ananiadou 2025는 MLP 층에서 성별 뉴런(gender neurons)을 식별한다; Ahsan & Wallace 2025는 SAE를 사용해 임상적 인종 편향을 드러낸다; Zhou 외 2024(UniBias)는 디바이어싱(debiasing)을 위해 어텐션 헤드(attention head)를 조작한다. 메타 비평(arXiv:2508.11067): 10년간의 문헌이 이진 성별 편향에 불균형적으로 집중되어 있다.

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 표현적 해악 대 배분적 해악을 정의하고 LLM 배포에서 각각의 예를 하나씩 들기.
- Gallegos 외 2024의 세 가지 평가 지표 범주를 명명하고 각각에서 하나의 지표를 기술하기.
- 교차성(intersectionality)과 WinoIdentity의 불확실성 기반 공정성 측정이 단일 축 편향 평가의 빈틈을 다루는 이유를 기술하기.
- 편향에 대한 두 가지 기계론적 해석 가능성(mechanistic-interpretability) 접근(성별 뉴런, SAE 특성, 어텐션 헤드 조작)을 기술하기.

## 문제 (The Problem)

이전 레슨들은 의도적 해악(탈옥, 음모)과 안전 거버넌스를 다룬다. 편향은 의도 없이 발생하는 해악이다. 학습 데이터 분포에서, 프롬프트 프레이밍에서, 누적된 설계 선택에서 생겨난다. 이를 측정하고 줄이는 일은 적대적 견고성과는 별개의 방법론적 과제다.

## 개념 (The Concept)

### 표현적 대 배분적

- **표현적 해악(Representational harm).** 고정관념, 삭제, 비하적 묘사. 간호사를 오로지 여성으로만 묘사하는 LLM은 표현적 해악을 산출한다.
- **배분적 해악(Allocational harm).** 불평등한 물질적 결과. 흑인 지원자의 이력서에 체계적으로 낮은 점수를 매기는 LLM은 배분적 해악을 산출한다.

이 둘은 같지 않다. 모델은 "표현적으로 편향 없으면서"(다양한 묘사를 산출) "배분적으로 편향될"(불평등한 추천을 함) 수 있다. 평가는 둘 다를 측정해야 한다.

### 세 가지 평가 지표 범주 (Gallegos 외 2024)

- **임베딩 기반(Embedding-based).** RLHF 이전 임베딩(embedding)에 대한 WEAT 방식 테스트. 정체성 용어와 속성 용어 사이의 통계적 연관을 측정한다. 한계: 행동이 아니라 표현을 측정한다.
- **확률 기반(Probability-based).** 고정관념 확증 완성 대 고정관념 위반 완성의 로그 가능도(log-likelihood). 디코더 측 측정. 일부 행동 편향을 포착한다.
- **생성 텍스트 기반(Generated-text-based).** 생성된 텍스트에 대한 다운스트림 작업 측정. 이력서 점수 매기기, 추천문 작성, 대화. 가장 생태학적으로 타당하지만 재현하기는 가장 어렵다.

### 교차성

"성별"에 대한 편향 평가는 (성별, 인종) 쌍에서만 발화하는 편향을 놓친다. An 외 2025는 GPT-4o가 이력서 점수 매기기에서 흑인 여성을 흑인 남성보다, 그리고 백인 여성보다도 별개로 더 불리하게 매긴다는 것을 발견한다. 단일 축 평가는 이를 포착할 수 없다.

WinoIdentity(COLM 2025)는 불확실성 기반 교차 공정성을 도입한다. 점 예측만이 아니라, 모델의 결과에 대한 불확실성이 교차 정체성 튜플 전반에서 다른지를 측정한다. 이는 모델이 그룹 전반에서 똑같이 틀리지만 일부에 대해 더 불확실하여 다른 다운스트림 배분 행동을 산출하는 경우를 잡아낸다.

### 기계론적 접근

2024-2025년 해석 가능성 작업은 편향을 기계론적 개입에 열어준다:

- **성별 뉴런(Gender neurons, Yu & Ananiadou 2025).** 특정 MLP 뉴런이 성별 특정 행동과 상관관계가 있다. 이 뉴런들을 절제(ablating)하면 제한된 역량 비용으로 성별 격차 지표를 줄인다.
- **SAE를 통한 임상적 인종 편향(Ahsan & Wallace 2025).** 희소 오토인코더(sparse autoencoder, SAE) 특성이 내부 표현을 해석 가능한 차원으로 분해한다. 인종과 상관된 특성을 식별하고 억제할 수 있다.
- **UniBias(Zhou 외 2024).** 제로샷(zero-shot) 디바이어싱을 위한 어텐션 헤드 조작. 특정 헤드가 정체성 클래스 민감도를 증폭하는데, 이 헤드들을 0으로 만들거나 재가중하면 파인튜닝 없이 편향을 줄인다.

### 메타 비평

10년간의 문헌 검토(arXiv:2508.11067, 2025)는 이 분야가 이진 성별 편향에 불균형적으로 집중함을 발견한다. 다른 축들(장애, 종교, 이주 지위, 다국어 정체성)은 훨씬 적은 주의를 받는다. 메타 비평은 좁은 집중이 방치를 통해 소외된 집단에 해를 끼칠 수 있다고 주장한다. 이진 성별에 대해 잘 디바이어싱된 모델이 아무도 확인하지 않은 차원에서는 심하게 편향될 수 있다는 것이다.

### Phase 18에서의 위치

Lesson 20-21은 편향과 공정성을 공식적으로 다룬다. Lesson 22는 프라이버시를 다룬다. Lesson 23은 워터마킹을 다룬다. 이들은 앞선 기만/안전 계층을 보완하는 사용자 해악 계층이다.

```figure
an-bias-two-harms
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 임베딩 기반 편향 프로브를 만든다. 간단한 동시 출현(co-occurrence) 임베딩에서 정체성 용어와 속성 용어 사이의 WEAT 방식 거리를 측정한다. 편향을 주입해 지표가 발화하는 것을 관찰할 수 있고, 간단한 디바이어싱 연산을 적용해 부분적 회복을 관찰할 수도 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-bias-eval.md`를 만든다. 모델 카드나 공정성 주장이 주어지면, 세 가지 지표 범주(임베딩, 확률, 생성 텍스트) 전반의 평가, 교차성 커버리지, 그리고 모든 디바이어싱 개입의 메커니즘을 감사한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 디바이어싱 단계 전후의 WEAT 방식 편향 점수를 보고하라. 지표가 0으로 떨어지지 않는 이유를 설명하라.

2. 프로브를 교차 테스트로 확장하라: (성별, 인종) x (경력, 가족). 축 간(cross-axis) 편향 점수를 보고하라.

3. An 외 2025(PNAS Nexus)를 읽어라. 단일 축 성별 평가가 놓칠 그들이 보고한 두 가지 교차 효과를 식별하라.

4. Yu & Ananiadou 2025는 성별 뉴런을 식별한다. "이 뉴런들이 성별 편향을 유발한다"를 "이 뉴런들이 성별 편향과 상관된다"와 구별할 반증 실험을 스케치하라.

5. 메타 비평은 이 분야가 이진 성별에 너무 좁게 집중한다고 주장한다. 덜 연구된 축 하나를 골라 그것에 대한 표현적 해악 측정 프로토콜을 기술하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 표현적 해악 (Representational harm) | "고정관념 / 삭제" | 한 집단에 대한 편향된 묘사 |
| 배분적 해악 (Allocational harm) | "불평등한 결정" | 한 집단에 대한 편향된 물질적 결과 |
| WEAT | "임베딩 테스트" | Word Embedding Association Test; 동시 출현 기반 편향 프로브 |
| 교차성 (Intersectionality) | "결합된 정체성 효과" | 다수의 정체성 축의 교차에서 발생하는 편향 |
| 성별 뉴런 (Gender neurons) | "MLP 편향 뉴런" | 활성값이 성별 특정 행동과 상관되는 특정 뉴런 |
| SAE 특성 (SAE feature) | "해석 가능한 차원" | 희소 오토인코더가 식별한 특성; 기계론적 편향 분석에 유용 |
| UniBias | "어텐션 헤드 디바이어싱" | 어텐션 헤드 재가중을 통한 제로샷 디바이어싱 |

## 더 읽을거리 (Further Reading)

- [Gallegos et al.: Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) 대표 서베이
- [An et al.: Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) 다섯 모델 교차 연구
- [WinoIdentity: uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) 새로운 벤치마크
- [UniBias: attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) 제로샷 디바이어싱
