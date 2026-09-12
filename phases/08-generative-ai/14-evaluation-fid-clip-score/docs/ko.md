# Evaluation — FID, CLIP Score, Human Preference

> 모든 생성 모델 리더보드는 FID, CLIP 점수, 그리고 인간 선호 아레나에서 나온 승률을 인용한다. 각 숫자에는 작정한 연구자가 게이밍할 수 있는 실패 양상(failure mode)이 있다. 실패 양상을 모르면 진짜 개선과 게이밍 실행을 구별할 수 없다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45분

## 문제 (The Problem)

생성 모델은 *샘플 품질*과 *조건화 준수*로 판단된다. 둘 다 닫힌 형식(closed-form)의 측도가 없다. 모델은 10,000장의 이미지를 렌더링해야 하고, 그 이미지에 무언가가 숫자를 매겨야 하며, 그 숫자는 모델 계열과 해상도, 아키텍처를 넘나들며 신뢰할 수 있어야 한다. 2014-2026년의 가혹한 시험을 살아남은 세 가지 지표가 있다.

- **FID(Fréchet Inception Distance).** Inception 신경망(neural network)의 특성(feature) 공간에서 두 분포(실제와 생성) 사이의 거리. 낮을수록 좋다.
- **CLIP 점수(CLIP score).** 생성된 이미지의 CLIP 이미지 임베딩(embedding)과 프롬프트(prompt)의 CLIP 텍스트 임베딩 사이의 코사인 유사도. 높을수록 좋다. 프롬프트 준수를 측정한다.
- **인간 선호(Human preference).** 같은 프롬프트로 두 모델을 정면 대결시키고, 인간(또는 GPT-4급 모델)에게 더 나은 것을 고르게 하고, Elo 점수로 집계한다.

다음 지표도 함께 다룬다. IS(inception score, 대체로 퇴역), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. 각각은 이전 것의 한 가지 실패를 교정한다.

## 개념 (The Concept)

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID: 샘플 품질

Heusel et al. (2017). 단계는 다음과 같다.

1. N장의 실제 이미지와 N장의 생성 이미지에 대해 Inception-v3 특성(2048차원)을 추출한다.
2. 각 풀에 가우시안을 맞춘다. 평균 `μ_r, μ_g`와 공분산 `Σ_r, Σ_g`를 계산한다.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

해석: 특성 공간에서 두 다변량 가우시안 사이의 프레셰(Fréchet) 거리. 낮을수록 = 더 비슷한 분포.

실패 양상:
- **작은 N에서 편향됨.** FID는 특성 분포에 대한 평균 제곱이다. 작은 N은 공분산을 과소평가해 거짓으로 낮은 FID를 준다. 항상 N ≥ 10,000을 쓰라.
- **Inception 의존적.** Inception-v3는 ImageNet으로 학습되었다. ImageNet에서 먼 도메인(얼굴, 예술, 텍스트 이미지)은 무의미한 FID를 만든다. 도메인 특화 특성 추출기를 쓰라.
- **게이밍.** Inception 사전 확률(prior)에 과적합(overfitting)하면 시각적 품질 향상 없이 낮은 FID를 얻는다. CMMD(아래)로 이를 이겨라.

### CLIP 점수: 프롬프트 준수

Radford et al. (2021). 생성된 이미지 + 프롬프트에 대해:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30k장의 생성 이미지에 걸쳐 평균 → 모델 간 비교 가능한 스칼라.

실패 양상:
- **CLIP 자체의 맹점.** CLIP은 조합적 추론이 약하다("a red cube on a blue sphere"는 종종 실패한다). 모델은 복잡한 프롬프트를 실제로 따르지 않고도 CLIP 점수에서 좋은 순위를 받을 수 있다.
- **짧은 프롬프트 편향.** 짧은 프롬프트는 실세계에서 CLIP 이미지 매치가 더 많다. 긴 프롬프트는 기계적으로 더 낮은 CLIP 점수를 받는다.
- **프롬프트 게이밍.** 프롬프트에 "high quality, 4k, masterpiece"를 넣으면 이미지-텍스트 결합 개선 없이 CLIP 점수가 부풀려진다.

CMMD(Jayasumana et al., 2024)는 이 중 일부를 고친다. Inception 대신 CLIP 특성을, 프레셰 대신 최대 평균 불일치(maximum-mean discrepancy)를 쓴다. 미묘한 품질 차이를 탐지하는 데 더 낫다.

### 인간 선호: 그라운드 트루스

프롬프트 풀을 고른다. 모델 A와 모델 B로 생성한다. 인간(또는 강한 LLM 판정자)에게 쌍을 보여준다. 승리를 Elo나 브래들리-테리(Bradley-Terry) 점수로 집계한다. 벤치마크(benchmark):

- **PartiPrompts (Google)**: 1,600개의 다양한 프롬프트, 12개 카테고리.
- **HPSv2**: 10만 7천 개의 인간 주석, 자동 프록시로 널리 사용됨.
- **ImageReward**: 13만 7천 개의 프롬프트-이미지 선호 쌍, MIT 라이선스.
- **PickScore**: Pick-a-Pic의 260만 선호로 학습됨.
- **Chatbot-Arena 스타일 이미지 아레나**: https://imagearena.ai/ 외.

실패 양상:
- **판정자 분산.** 비전문가는 전문가와 다른 선호를 가진다. 둘 다 쓰라.
- **프롬프트 분포.** 체리피킹된 프롬프트는 한 계열에 유리하다. 항상 문서화하라.
- **LLM 판정자 보상 해킹.** GPT-4 판정자는 예쁘지만 틀린 출력에 속는다. 인간으로 삼각 측량하라.

## 함께 쓰기

프로덕션(production) 평가 보고서에는 다음이 포함되어야 한다.

1. 보류된 실제 분포에 대한 10-30k 샘플의 FID(샘플 품질).
2. 같은 샘플의 프롬프트 대비 CLIP 점수 / CMMD(준수).
3. 이전 모델 대비 블라인드 아레나의 승률(전반적 선호).
4. 실패 양상 분석: 무작위로 샘플링된 출력 50개, 알려진 문제(손 해부학, 텍스트 렌더링, 일관된 객체 수)에 대해 플래그.

어떤 단일 지표든 거짓말이다. 서로 입증하는 세 지표 + 정성적 검토가 주장이다.

```figure
gx-fid-distributions
```

## 직접 만들기 (Build It)

`code/main.py`는 합성 "특성 벡터"(Inception 특성의 대역으로 4차원 벡터를 쓴다)에 대해 FID, CLIP 점수 유사물, Elo 집계를 구현한다. 다음을 본다.

- 작은 N과 큰 N에서의 FID 계산: 그 편향.
- 특성 풀 간 코사인 유사도로서의 "CLIP 점수".
- 합성 선호 스트림으로부터의 Elo 갱신 규칙.

### Step 1: FID in four lines

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 함정 (Pitfalls)

- **N=1000에서의 FID.** 휴리스틱은 N=10k 미만에서 신뢰할 수 없다. 낮은 N의 FID를 보고하는 논문은 게이밍하는 것이다.
- **해상도를 넘나드는 FID 비교.** Inception의 299×299 리사이즈는 특성 분포를 바꾼다. 일치하는 해상도에서만 비교하라.
- **하나의 시드만 보고하기.** 최소 3개의 시드를 실행하라. 표준편차를 보고하라.
- **부정 프롬프트를 통한 CLIP 점수 부풀리기.** 일부 파이프라인은 프롬프트에 과적합해 CLIP을 끌어올린다. 시각적 포화(saturation)를 확인하라.
- **프롬프트 중첩으로 인한 Elo 편향.** 두 모델이 학습 중 벤치마크 프롬프트를 봤다면 Elo는 무의미하다. 보류된 프롬프트 세트를 쓰라.
- **인간 평가 유료 크라우드 편향.** Prolific, MTurk 주석자는 더 젊고 / 기술 친화적인 쪽으로 치우친다. 모집된 미술/디자인 전문가와 섞어라.

## 라이브러리로 써보기 (Use It)

2026년의 프로덕션 평가 프로토콜:

| 기둥 | 최소 | 권장 |
|--------|---------|-------------|
| 샘플 품질 | 보류된 실제 대비 10k의 FID | + 5k의 CMMD + 카테고리별 부분집합의 FID |
| 프롬프트 준수 | 30k의 CLIP 점수 | + HPSv2 + ImageReward + VQA 스타일 질문 응답 |
| 선호 | 베이스라인 대비 블라인드 쌍 200개 | + 인간 쌍 2000개 + LLM 판정자 + Chatbot Arena |
| 실패 분석 | 수동 플래그 50개 | 수동 플래그 500개 + 자동 안전 분류기 |

한 보고서에 네 기둥 모두 = 주장. 어느 하나만 = 마케팅.

## 산출물 (Ship It)

`outputs/skill-eval-report.md`를 저장한다. 이 스킬은 새 모델 체크포인트 + 베이스라인을 받아 전체 평가 계획을 출력한다. 샘플 크기, 지표, 실패 양상 탐침, 승인 기준.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 같은 합성 분포에서 N=100 대 N=1000의 FID를 비교하라. 편향 크기를 보고하라.
2. **보통.** 합성 CLIP 스타일 특성으로부터 CMMD를 구현하라(공식은 Jayasumana et al., 2024 참조). 품질 차이에 대한 민감도를 FID와 비교하라.
3. **어려움.** HPSv2 설정을 복제하라. Pick-a-Pic 부분집합에서 이미지-프롬프트 쌍 1000개를 가져와, 그 선호로 작은 CLIP 기반 채점기를 파인튜닝(fine-tuning)하고, 보류된 세트와의 일치도를 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 실제 대 생성 Inception 특성에 맞춘 가우시안의 프레셰 거리. |
| CLIP 점수(CLIP score) | "텍스트-이미지 유사도" | CLIP 이미지 임베딩과 텍스트 임베딩 사이의 코사인 유사도. |
| CMMD | "FID의 대체물" | CLIP 특성 MMD. 덜 편향됨, 가우시안 가정 없음. |
| IS | "Inception score" | Exp KL(p(y\|x) \|\| p(y)). 현대 모델에서 상관성이 낮음, 퇴역됨. |
| HPSv2 / ImageReward / PickScore | "학습된 선호 프록시" | 인간 선호로 학습된 작은 모델. 자동 판정자로 사용됨. |
| Elo | "체스 레이팅" | 쌍별 승리의 브래들리-테리 집계. |
| PartiPrompts | "그 벤치마크 프롬프트 세트" | 12개 카테고리에 걸친 Google 큐레이션 프롬프트 1,600개. |
| FD-DINO | "자기지도 대체물" | DINOv2 특성을 쓰는 FD. ImageNet 밖 도메인에 더 적합. |

## 프로덕션 노트: 평가도 추론 작업 부하다

10k 샘플에 FID를 실행한다는 것은 10k장의 이미지를 생성한다는 뜻이다. 단일 L4에서 1024²의 50스텝 SDXL 베이스로는 단일 요청 추론(inference) ~11시간이다. 평가 예산은 실재하며, 그 틀은 정확히 오프라인 추론 시나리오다(처리량(throughput) 극대화, TTFT 무시).

- **세게 배칭하고 지연 시간(latency)을 잊어라.** 오프라인 평가 = 메모리에 들어가는 가장 큰 크기의 정적 배칭(batching). 80GB H100에서 `num_images_per_prompt=8`로 `pipe(...).images`를 쓰면 단일 요청보다 벽시계 시간으로 4-6배 빠르게 실행된다.
- **실제 특성을 캐싱하라.** 실제 참조 세트에 대한 Inception(FID) 또는 CLIP(CLIP 점수, CMMD) 특성 추출은 *한 번* 실행하고 `.npz`로 저장한다. 평가마다 재계산하지 말라.

CI / 회귀 게이트용: PR마다 500 샘플 부분집합에 FID + CLIP 점수를 실행하고(~30분), 야간에 전체 10k FID + HPSv2 + Elo를 실행한다.

## 더 읽을거리 (Further Reading)

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500): FID 논문.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675): 실패 양상 조사.
