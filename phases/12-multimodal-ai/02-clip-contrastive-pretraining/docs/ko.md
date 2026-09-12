# CLIP와 대조적 비전-언어 사전 학습(CLIP and Contrastive Vision-Language Pretraining)

> OpenAI의 CLIP(2021)은 이후 5년을 떠받칠 만큼 큰 하나의 아이디어를 증명했다. 잡음 섞인 웹 이미지-캡션 쌍과 대조 손실(contrastive loss)만으로 이미지 인코더(encoder)와 텍스트 인코더를 같은 벡터 공간에 정렬하는 것이다. 지도 레이블(supervised label)은 0개. 쌍은 4억 개. 그 결과로 얻은 임베딩(embedding) 공간은 제로샷 분류(zero-shot classification), 이미지-텍스트 검색을 수행하고, 2026년의 모든 VLM에 비전 타워(vision tower)로 꽂힌다. SigLIP 2(2025)는 소프트맥스(softmax)를 시그모이드(sigmoid)로 대체하여 더 낮은 비용으로 CLIP을 넘어 스케일링했다. 이 레슨은 InfoNCE에서 시그모이드 쌍별(pairwise) 손실까지의 수학을 짚어 보고, 표준 라이브러리 Python으로 학습 스텝을 만든다.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 상호 정보량(mutual information)에서 InfoNCE 손실을 유도하고, 수치적으로 안정적인 벡터화 버전을 구현하기.
- 시그모이드 쌍별 손실(SigLIP)이 소프트맥스가 요구하는 all-gather 오버헤드 없이 배치(batch) 32768+로 스케일링되는 이유 설명하기.
- 텍스트 템플릿(`a photo of a {class}`)을 구성하고 코사인 유사도(cosine similarity)에 대해 argmax를 취하여 제로샷 ImageNet 분류 실행하기.
- CLIP / SigLIP 사전 학습이 주는 네 가지 지렛대 이름 대기: 배치 크기, 온도(temperature), 프롬프트 템플릿, 데이터 품질.

## 문제 (The Problem)

CLIP 이전의 비전은 지도 학습이었다. 레이블된 데이터셋(ImageNet: 120만 장 이미지, 1000개 클래스)을 모으고, CNN을 학습시켜 출시한다. 레이블은 비싸고, 레이블러들이 합의할 수 있는 것 쪽으로 편향되며, 파인튜닝(fine-tuning) 없이는 새 작업으로 전이되지 않는다.

이미지-캡션 웹에는 느슨하게 레이블된 10억 개 이상의 쌍이 공짜로 널려 있다. alt 텍스트 "공원에 있는 내 강아지 맥스(my dog Max in the park)"가 달린 골든 리트리버 사진은 지도 신호를 담고 있다. 텍스트가 이미지를 묘사하기 때문이다. 그렇다면 이것을 유용한 학습으로 바꿀 수 있는가?

CLIP의 답은 이미지-캡션 쌍을 매칭 작업으로 다루는 것이다. N개의 이미지와 N개의 캡션으로 이루어진 배치가 주어지면, N-1개의 방해물(distractor)에 맞서 각 이미지를 자신의 캡션에 매칭하는 법을 학습한다. 지도 신호는 "이 둘은 서로 어울린다, 이 N-1개는 아니다"이다. 클래스 레이블 없음. 사람의 주석(annotation) 없음. 오직 대조 손실뿐.

그 결과로 얻은 임베딩 공간은 CLIP이 학습한 것보다 더 많은 일을 한다. "a photo of a cat"이 한 번도 명시적으로 고양이로 레이블되지 않은 고양이 사진 근처에 임베딩되기 때문에 ImageNet 제로샷이 작동한다. 이것이 2026년의 모든 VLM을 낳은 베팅이다.

## 개념 (The Concept)

### 듀얼 인코더 (The dual encoder)

CLIP에는 두 개의 타워가 있다:

- 이미지 인코더 `f`: ViT 또는 ResNet, 이미지당 D차원 벡터를 출력한다.
- 텍스트 인코더 `g`: 작은 트랜스포머(transformer), 캡션당 D차원 벡터를 출력한다.

두 타워 모두 출력을 단위 길이(unit length)로 정규화(normalization)한다. 둘 다 단위 노름(unit-norm)이므로 유사도는 `cos(f(x), g(y)) = f(x)^T g(y)`이다.

N개의 (이미지, 캡션) 쌍으로 이루어진 배치에 대해, 형상이 `(N, N)`인 유사도 행렬(matrix) `S`를 만든다:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

여기서 `tau`는 학습되는 온도(temperature)다(CLIP은 0.07로 초기화하며, 로그 공간에서 학습한다).

### InfoNCE 손실 (InfoNCE loss)

CLIP은 행과 열에 대한 대칭(symmetric) 교차 엔트로피(cross-entropy)를 쓴다:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

이것이 InfoNCE다. CE 안의 소프트맥스는 각 이미지가 배치 안의 다른 모든 캡션보다 자신의 캡션에 더 매칭되도록 강제한다. "음성(negatives)"은 배치의 다른 모든 항목이다. 배치가 클수록 음성이 많아지고 신호도 강해진다. CLIP은 배치 32k에서 학습했다. 규모가 중요하다.

### 온도 (Temperature)

`tau`는 소프트맥스의 날카로움을 제어한다. 낮은 tau는 날카로운 분포를 만들어 어려운 음성 마이닝(hard negative mining) 효과를 낸다. 높은 tau는 분포를 부드럽게 해 모든 샘플이 기여한다. CLIP은 붕괴를 막으려고 클리핑된 log(1/tau)를 학습한다. SigLIP 2는 초기 tau를 고정하고 대신 학습되는 편향(bias)을 쓴다.

### 시그모이드가 더 잘 스케일링되는 이유 (SigLIP) (Why sigmoid scales better)

소프트맥스는 전체 유사도 행렬이 동기화되어야 한다. 분산 학습에서는 모든 임베딩을 모든 복제본(replica)으로 all-gather한 뒤 소프트맥스를 해야 한다. 통신 측면에서 이것은 월드 크기(world size)에 대해 이차(quadratic)다.

SigLIP은 소프트맥스를 원소별(element-wise) 시그모이드로 대체한다. 각 쌍 `(i, j)`에 대해, 손실은 "이것이 매칭되는 쌍인가?"의 이진 분류다. 양성 클래스 레이블은 대각선이고, 나머지 전부는 음성이다. 손실은:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`i == j`이면 `y_ij = 1`, 아니면 0이다. 각 쌍의 손실은 독립적이라 all-gather가 필요 없다. 각 GPU가 자신의 로컬 블록을 계산해 합산한다. SigLIP 2는 CLIP이라면 비례적으로 더 많은 통신이 필요할 곳에서 배치 32k-512k까지 저렴하게 스케일링한다.

### 제로샷 분류 (Zero-shot classification)

N개의 클래스 이름이 주어지면, 각 클래스에 대해 텍스트 템플릿을 만든다:

```
"a photo of a {class}"
```

각 템플릿을 텍스트 인코더로 임베딩하고, 이미지를 이미지 인코더로 임베딩한다. 코사인 유사도의 argmax가 예측 클래스다. 목표 클래스에 대한 학습은 없다.

프롬프트 템플릿이 중요하다. CLIP 원논문은 클래스당 80개 템플릿(plain, artistic, photo, painting 등)을 사용해 임베딩을 평균냈다. ImageNet +3점. 요즘은 보통 한두 개 템플릿을 고른다.

### 선형 프로브와 파인튜닝 (Linear probes and finetuning)

제로샷은 베이스라인(baseline)이다. 선형 프로브(linear probe)(목표 클래스에 대해 동결된 CLIP 특성 위에 선형 층 하나를 학습)는 도메인 내(in-domain) 작업에서 제로샷을 이긴다. 전체 파인튜닝은 도메인 내에서 선형 프로브를 이기지만 제로샷 전이를 해칠 수 있다. 세 가지 영역에 세 가지 트레이드오프(trade-off).

### SigLIP 2: NaFlex와 밀집 특성 (SigLIP 2: NaFlex and dense features)

SigLIP 2(2025)는 다음을 추가한다:
- NaFlex: 단일 모델이 가변 종횡비와 해상도를 처리한다.
- 분할과 깊이 추정을 위한 더 나은 밀집 특성(dense features), VLM에서 동결 백본(frozen backbone)으로 쓰이는 것을 목표로 한다.
- 다국어: CLIP이 영어 전용이었던 것과 달리 100개 이상의 언어로 학습되었다.
- CLIP이 4억에서 멈춘 곳에서 파라미터 10억 규모.

2026년 오픈 VLM에서 SigLIP 2 SO400m/14는 기본 비전 타워다. CLIP은 특정 LAION-2B 학습 분포가 쿼리 패턴과 맞는 순수 이미지-텍스트 검색에서 여전히 기본값으로 남는다.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN(Google, 2021): CLIP과 같은 아이디어, 18억 쌍 규모, 90% 잡음. 잡음 섞인 데이터가 스케일링됨을 증명했다. OpenCLIP(LAION): LAION-400M / 2B에서의 CLIP 오픈 재현, 여러 규모, 대표적인 오픈 체크포인트. EVA-CLIP: 마스킹 이미지 모델링(masked image modeling)에서 초기화한다. VLM을 위한 강력한 백본. BASIC: Google의 CLIP+ALIGN 하이브리드. 모두 같은 계열이고, 데이터와 튜닝만 다르다.

### 제로샷 천장 (The zero-shot ceiling)

CLIP 계열 모델은 ImageNet 제로샷에서 76% 부근에서 멈춘다(CLIP-G, OpenCLIP-G). 그 너머는 훨씬 더 큰 데이터(SigLIP 2는 80%+를 얻는다)나 아키텍처 변경(지도 헤드, 더 많은 파라미터)이 필요하다. 벤치마크(benchmark)는 포화(saturation)되고 있다. 진짜 가치는 다운스트림 VLM이 소비하는 임베딩 공간이다.

```figure
multimodal-fusion
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다음을 구현한다:

1. 장난감 듀얼 인코더(해시 기반 이미지 특성, 텍스트 문자 특성). numpy 없이 InfoNCE 형태를 볼 수 있다.
2. 순수 Python으로 구현한 InfoNCE 손실(log-sum-exp를 통한 수치 안정성).
3. 비교를 위한 시그모이드 쌍별 손실.
4. 제로샷 분류 루틴: 텍스트 프롬프트 집합에 대한 코사인 유사도를 계산하고, 예측을 위해 argmax.

실행하고 손실 곡선을 지켜보라. 절댓값은 장난감 수준이지만, 형태는 실제 CLIP 트레이너가 내보내는 것과 일치한다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-clip-zero-shot.md`를 만든다. 이미지 집합(경로로)과 목표 클래스 목록이 주어지면, CLIP 템플릿으로 텍스트 프롬프트를 만들고, 명시된 체크포인트(예: `openai/clip-vit-large-patch14`)로 양쪽을 임베딩하여, 유사도 점수와 함께 top-1 / top-5 예측을 반환한다. 이 스킬은 프롬프트 목록에 없는 클래스에 대한 주장은 거부한다.

## 연습 문제 (Exercises)

1. 4개 쌍으로 이루어진 배치에 대해 InfoNCE를 손으로 구현하라. 4x4 유사도 행렬을 만들고, 소프트맥스를 돌리고, 대각선을 골라내어 교차 엔트로피를 계산하라. 직접 작성한 Python 구현을 이 손계산과 대조하여 검증하라.

2. SigLIP은 온도에 더해 편향 파라미터 `b`를 쓴다: `S'[i,j] = S[i,j]/tau + b`. 배치에 큰 클래스 불균형(행당 양성보다 음성이 훨씬 많은)이 있을 때 `b`는 어떤 역할을 하는가? SigLIP Section 3(arXiv:2303.15343)을 읽어라.

3. 고양이 대 개 제로샷 분류기를 만들어라. 두 가지 프롬프트 템플릿을 시도하라: `a photo of a {class}`와 `a picture of a {class}`. 100장의 테스트 이미지에서 정확도를 측정하라. 템플릿 앙상블이 단일을 이기는가?

4. 배치 32k에서 512-GPU 실행에 대해 소프트맥스 InfoNCE 대 시그모이드 쌍별의 통신 비용을 계산하라. 어느 것이 O(N)으로, 어느 것이 O(N^2)으로 스케일링되는가? SigLIP Section 4를 인용하라.

5. OpenCLIP 스케일링 법칙 논문(arXiv:2212.07143, Cherti et al.)을 읽어라. 그림으로부터 데이터 스케일링에 대한 그들의 결론을 재현하라: 고정된 모델 크기에서, ImageNet 제로샷 정확도와 학습 데이터 크기 사이의 로그-선형(log-linear) 관계는 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| InfoNCE | "대조 손실" | 배치의 유사도 행렬에 대한 교차 엔트로피. 각 항목의 양성은 그 짝 항목이고, 음성은 나머지 전부 |
| 시그모이드 손실(Sigmoid loss) | "SigLIP 손실" | 쌍별 이진 교차 엔트로피. 소프트맥스 없음, all-gather 없음, 분산 학습에서 저렴하게 스케일링 |
| 온도(Temperature) | "tau" | 소프트맥스/시그모이드 전에 로짓(logit)을 스케일링하는 스칼라. 분포의 날카로움을 제어 |
| 제로샷(Zero-shot) | "파인튜닝 없는 분류" | 텍스트 프롬프트로 클래스 임베딩을 구성하고 코사인 유사도로 분류. 목표 클래스에 대한 학습 없음 |
| 프롬프트 템플릿(Prompt template) | "a photo of a ..." | 클래스 이름을 감싸는 텍스트 비계(scaffold). 제로샷 정확도를 1-5점만큼 좌우 |
| 듀얼 인코더(Dual encoder) | "투 타워(Two-tower)" | 이미지 인코더 하나 + 텍스트 인코더 하나, 공유된 D차원 공간에 출력 |
| 어려운 음성(Hard negative) | "까다로운 방해물" | 모델이 분리해 내려면 애써야 할 만큼 양성과 충분히 비슷한 음성 |
| 선형 프로브(Linear probe) | "동결 + 한 층" | 동결된 특성 위에 선형 분류기만 학습. 특성 품질을 측정 |
| NaFlex | "네이티브 유연 해상도" | 크기 조정 없이 임의의 종횡비와 해상도로 이미지를 받아들이는 SigLIP 2 능력 |
| 온도 스케일링(Temperature scaling) | "로그 파라미터화된 tau" | CLIP은 그래디언트가 잘 동작하도록 `log(1/tau)`를 파라미터화한다. 0에 가까운 tau로의 붕괴를 막기 위해 클리핑한다 |

## 더 읽을거리 (Further Reading)

- [Radford et al.: Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) CLIP 논문.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al.: SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) 다국어 + NaFlex.
- [Jia et al.: ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) 잡음 섞인 웹 데이터로 스케일링.
- [Cherti et al.: Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) OpenCLIP 스케일링 법칙.
