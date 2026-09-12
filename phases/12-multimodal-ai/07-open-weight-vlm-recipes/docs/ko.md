# 오픈 웨이트 VLM 레시피: 실제로 중요한 것(Open-Weight VLM Recipes: What Actually Matters)

> 2024-2026년 오픈 웨이트(open-weight) VLM 문헌은 절제(ablation) 표의 숲이다. Apple의 MM1은 이미지 인코더(encoder), 커넥터(connector), 데이터 혼합의 13가지 조합을 시험했다. Allen AI의 Molmo는 상세한 사람 캡션이 GPT-4V 증류(distillation)를 이긴다는 것을 증명했다. Cambrian-1은 20개 이상의 인코더 비교를 돌렸다. Idefics2는 5축 설계 공간을 형식화했다. Prismatic VLMs는 통제된 벤치마크(benchmark)에서 27개 학습 레시피를 비교했다. 그 모든 잡음 가운데, 논문들을 가로질러 유지되는 작은 결과 집합이 있다: 이미지 인코더가 커넥터 아키텍처보다 더 중요하고, 데이터 혼합이 둘 중 어느 것보다 더 중요하며, 상세한 사람 캡션이 증류된 합성 데이터를 이긴다. 이 레슨은 그 표들을 대신 읽어 준다.

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 5축 VLM 설계 공간 이름 대기: 이미지 인코더, 커넥터, LLM, 데이터 혼합, 해상도 스케줄.
- MM1 / Idefics2 / Cambrian-1 절제 표를 읽고 어떤 손잡이가 주어진 벤치마크를 움직이는지 예측하기.
- 연산 예산과 작업 혼합이 주어졌을 때 새 VLM을 위한 레시피(인코더, 커넥터, 데이터, 해상도) 고르기.
- 같은 토큰(token) 개수에서 상세한 사람 캡션이 GPT-4V 증류를 이기는 이유 설명하기.

## 문제 (The Problem)

수백 개의 오픈 웨이트 VLM이 존재한다. "좋은 것"과 "최첨단(state-of-the-art)" 사이 간극의 대부분은 아키텍처가 아니다. 그것은 데이터, 해상도 스케줄, 인코더 선택이다. 모델이 기대에 못 미칠 때 어떤 손잡이를 먼저 돌릴지 아는 것이 500만 GPU 시간짜리 실수를 막아 준다.

2023년 물결(LLaVA-1.5, InstructBLIP, MiniGPT-4)은 캡션 쌍 사전 학습 + LLaVA-Instruct-150k로 돌았다. 좋은 베이스라인(baseline). MMMU 35% 부근에서 멈췄다.

2024년 물결(MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs)은 철저한 절제 연구를 돌렸다. 결과는 놀랍고 실용적이었다.

## 개념 (The Concept)

### 5축 설계 공간 (The five-axis design space)

Idefics2(Laurençon et al., 2024)는 축들에 이름을 붙였다:

1. 이미지 인코더. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. 인코더는 패치(patch) 크기, 해상도, 사전 학습 목적에서 다르다.
2. 커넥터. MLP(2-4층), Q-Former(쿼리 32개 + 교차 어텐션(cross-attention)), 퍼시버 리샘플러(Perceiver Resampler)(쿼리 64개), C-Abstractor(합성곱(convolution) + 쌍선형(bilinear) 풀링).
3. 언어 모델. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. LLM 크기가 지배적인 파라미터(parameter) 비용이다.
4. 학습 데이터. 캡션 쌍(CC3M, LAION), 인터리브(interleaved)(OBELICS, MMC4), 명령어(LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. 해상도 스케줄. 고정 224/336/448, AnyRes, 네이티브 동적(native dynamic). 학습 중 점증하거나 일정함.

모든 프로덕션(production) VLM은 각 축에서 선택을 한다. MMMU 점수 분산의 대부분은 축 1, 4, 5로 설명된다. 어떤 커넥터를 골랐는지가 아니다.

### 축 1: 인코더 > 커넥터 (Axis 1: encoder > connector)

MM1 Section 3.2는 보였다: CLIP ViT-L/14에서 SigLIP SO400m/14로 교체하면 MMMU가 3점 이상 늘었다. 커넥터를 MLP에서 퍼시버 리샘플러로 교체하면 1점 미만 늘었다. Idefics2가 재현했다: SigLIP > CLIP, 같은 토큰 개수에서 Q-Former ≈ MLP ≈ 퍼시버.

Cambrian-1의 "Cambrian Vision Encoders Match-Up"(Tong et al., 2024)은 비전 중심 벤치마크(CV-Bench)에서 20개 이상의 인코더를 돌렸다. 리더보드 상위는 DINOv2와 SigLIP의 혼합이다. CLIP은 중간이고, ImageBind와 ViT-MAE는 더 낮다. CLIP ViT-L에서 DINOv2 ViT-g/14로의 간극은 CV-Bench에서 약 5-7점이다.

2026년 오픈 VLM의 기본 인코더는 의미 + 밀집 특성(dense feature)을 위한 SigLIP 2 SO400m/14이며, 때로는 DINOv2 ViT-g/14 특성과 연결(concatenate)된다(Cambrian의 "Spatial Vision Aggregator"가 이것을 한다).

### 축 2: 커넥터 설계는 도긴개긴 (Axis 2: connector design is a wash)

MM1, Idefics2, Prismatic, MM-Interleaved 모두 같은 결론에 도달했다: 고정된 시각 토큰 개수에서, 커넥터 아키텍처는 거의 중요하지 않다. 평균 풀링된 패치에 대한 2층 MLP는 같은 토큰 예산에서 32쿼리 Q-Former의 1점 이내로 수행한다.

중요한 것은 토큰 개수다. 더 많은 시각 토큰 = 더 많은 LLM 연산 = 어느 지점까지는 더 나은 성능, 그다음은 수확 체감. 이미지당 64토큰은 OCR에 너무 적다. 576-1024토큰이 대부분의 오픈 VLM에 최적 지점이다. 2048+는 문서와 차트에만 도움이 된다.

Q-Former 대 MLP는 품질 문제가 아니라 비용 문제다: Q-Former는 이미지 해상도와 무관하게 토큰을 32-64로 상한 짓고, MLP는 모든 패치 토큰을 내보낸다. 고해상도 입력의 경우 Q-Former가 LLM 컨텍스트(context)를 절약하고, 저해상도의 경우 그 차이는 잡음이다.

### 축 3: LLM 크기가 천장을 정한다 (Axis 3: LLM size sets the ceiling)

LLM을 7B에서 13B로 두 배 늘리면 모든 VLM 논문에서 MMMU가 안정적으로 2-4점 늘어난다. 70B에서는 대부분의 벤치마크가 포화(saturation)된다. VLM의 멀티모달(multimodal) 추론 천장은 LLM의 텍스트 추론 천장이다. 시각 인코더는 그것에 먹일 수만 있지, 대신 추론해 줄 수는 없다.

이것이 Qwen2.5-VL-72B와 Claude Opus 4.7이 MMMU-Pro와 ScreenSpot-Pro를 압도하는 이유다: 언어 뇌가 거대하다. 7B VLM은 영리한 커넥터 설계로 70B VLM을 대체할 수 없다.

### 축 4: 데이터(상세한 사람 캡션이 증류를 이긴다 (Axis 4: data) detailed human captions beat distillation)

Molmo + PixMo(Deitke et al., 2024)는 모두가 읽어야 할 2024년 결과다. Allen AI는 사람 주석자(annotator)에게 1-3분짜리 밀집 음성-텍스트(speech-to-text) 패스로 이미지를 묘사하게 하여, 71만 2천 개의 밀집 캡션 이미지를 산출했다. 학습 데이터 어디에도 GPT-4V 증류는 없다.

Molmo-72B는 11개 벤치마크 중 11개에서 Llama-3.2-90B-Vision을 이겼다. 그 차이는 아키텍처가 아니다. 캡션 품질이다. 상세한 사람 캡션은 짧은 웹 캡션보다 이미지당 5-10배 많은 정보를 담고, GPT-4V 증류가 환각(hallucinate)하는 곳에서 사실에 기반한 채로 머문다.

ShareGPT4V(Chen et al., 2023)와 Cauldron(Idefics2)은 혼합된 사람 + GPT-4V 캡션으로 같은 전략을 따랐다. 추세는 명확하다: 2026년 프런티어에는 캡션 밀도 > 캡션 양 > 증류의 편리함.

### 축 5: 해상도와 그 스케줄 (Axis 5: resolution and its schedule)

Idefics2의 절제 연구: 384 -> 448은 1-2점을 더한다. 이미지 분할(AnyRes)을 쓴 448 -> 980은 OCR 벤치마크에서 3-5점을 더 더한다. 평탄한 해상도 학습은 중간 정확도에서 정체된다. 해상도 점증(start 224, finish 448 또는 네이티브)은 더 빨리 학습하고 더 높이 끝난다.

Cambrian-1은 해상도 대 토큰 트레이드오프(trade-off)를 돌렸다: 고정된 연산에서, 더 낮은 해상도로 더 많은 토큰을 갖거나 더 높은 해상도로 더 적은 토큰을 가질 수 있다. 더 높은 해상도는 OCR에서 이기고, 저해상도-더-많은-토큰은 일반 장면 이해에서 이긴다.

2026년 프로덕션 레시피: 1단계는 384 고정으로 학습하고, 2단계는 OCR 위주 작업을 위해 최대 1280까지 동적 해상도로 학습한다.

### Prismatic 통제 비교 (The Prismatic controlled comparison)

Prismatic VLMs(Karamcheti et al., 2024)는 모든 축을 통제한 논문이다. 같은 13B LLM, 같은 명령어 데이터, 같은 평가: 한 번에 한 축만 변한다. 결과:

- 이미지당 시각 토큰 개수가 분산의 약 60%를 설명한다.
- 인코더 선택이 약 20%를 설명한다.
- 커넥터 아키텍처가 약 5%를 설명한다.
- 나머지 전부(데이터 혼합, 스케줄러, LR)가 남은 약 15%.

이것은 거친 분해이지만, 문헌에서 "무엇을 먼저 절제해야 하는가"에 대한 가장 깔끔한 답이다.

### 2026년을 위한 선택기 (A picker for 2026)

증거에 비추어, 2026년 새 프로젝트를 위한 기본 오픈 VLM 레시피:

- 인코더: NaFlex를 쓰는 네이티브 해상도의 SigLIP 2 SO400m/14, 분할/그라운딩(grounding)이 필요하면 밀집 특성을 위해 DINOv2 ViT-g/14와 연결.
- 커넥터: 패치 토큰에 대한 2층 MLP. 토큰 제약이 없다면 Q-Former를 건너뛴다.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 비용에는 7B, 품질에는 70B, 목표 지연 시간(latency)으로 선택.
- 데이터: PixMo + ShareGPT4V + Cauldron, 작업별 명령어 데이터로 보강.
- 해상도: 동적(긴 변당 최소 256, 최대 1280픽셀).
- 스케줄: 1단계 정렬(투영기(projector)만), 2단계 전체 파인튜닝(fine-tune), 3단계 작업별 파인튜닝.

그 기본값 하나하나가 이 레슨 끝에 인용된 논문의 측정된 절제 연구로 거슬러 올라간다.

```figure
l5-vlm-recipe-knobs
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 절제 표 파서이자 레시피 선택기다. MM1과 Idefics2 절제 표(압축본)를 인코딩하고 다음을 질의하게 해 준다:

- "예산 X와 작업 Y가 주어지면, 어떤 레시피가 이기는가?"
- "7B Llama에서 SigLIP을 CLIP으로 교체하면, 예상 MMMU 차이는 얼마인가?"
- "80% 신뢰 답을 위해 어떤 축을 먼저 절제해야 하는가?"

출력은 예상 벤치마크 차이와 "먼저 절제하라" 추천을 담은 순위 매겨진 레시피 목록이다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-vlm-recipe-picker.md`를 만든다. 목표 작업 혼합, 연산 예산, 지연 시간 목표가 주어지면, 각 선택을 정당화하는 절제 연구에 대한 인용과 함께 전체 레시피(인코더, 커넥터, LLM, 데이터 혼합, 해상도 스케줄)를 내보낸다. 새 VLM 프로젝트가 시작될 때마다 엔지니어가 Idefics2 절제 표를 재발명하는 것을 막아 준다.

## 연습 문제 (Exercises)

1. MM1 Section 3.2를 읽어라. 예산 5000만 이미지에서 고정된 2B LLM의 경우, 어떤 인코더가 이기는가? 13B LLM에서는 답이 뒤집히는가? 왜인가?

2. Cambrian-1은 DINOv2 + SigLIP을 연결하는 것이 비전 중심 벤치마크에서 둘 중 어느 하나보다 뛰어나지만 MMMU에는 신호를 더하지 않는다는 것을 발견한다. 어떤 벤치마크가 이득을 보고 어떤 것이 평탄하게 유지되는지 예측하라.

3. 목표는 2B LLM 위의 모바일 UI 에이전트(agent)다. 인코더, 커넥터, 해상도, 데이터 혼합을 고르라. 각 선택을 특정 절제 표로 정당화하라.

4. Molmo는 4B와 72B 모델을 출시한다. 4B는 닫힌 7B VLM과 경쟁력이 있고, 72B는 11/11 벤치마크에서 Llama-3.2-90B-Vision을 이긴다. 그것이 LLM 크기 정체 가설에 대해 무엇을 말해 주는가?

5. 7B VLM에서 데이터 혼합 품질을 인코더 품질로부터 분리하는 절제 표를 설계하라. 최소 몇 번의 학습 실행이 필요한가? 네 가지 축 설정을 제안하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 절제(Ablation) | "손잡이 하나 돌리기" | 정확히 하나의 설계 공간 축에서만 다르고 나머지 전부를 일정하게 유지하는 여러 실행을 학습하는 것 |
| 커넥터(Connector) | "다리" / "투영기" | 비전 인코더 출력을 LLM의 토큰 공간으로 매핑하는 학습 가능한 모듈(MLP, Q-Former, 퍼시버) |
| 상세한 사람 캡션(Detailed human caption) | "밀집 캡션" | 웹 alt 텍스트보다 풍부한, 사람이 작성한 여러 문장 설명(보통 80-300토큰) |
| 증류(Distillation) | "GPT-4V 캡션" | 더 강력한 비공개 VLM이 생성한 학습 데이터. 편리하지만 물려받은 환각에 취약 |
| AnyRes / 동적 해상도(dynamic res) | "고해상도 경로" | 타일링 또는 M-RoPE를 통해 인코더의 네이티브 해상도보다 큰 이미지를 먹이는 전략 |
| 해상도 점증(Resolution ramp) | "커리큘럼" | 저해상도로 시작해 높이는 학습 스케줄. 정렬 학습을 가속한다 |
| 비전 중심 벤치(Vision-centric bench) | "CV-Bench / BLINK" | 언어 위주 추론이 아니라 세밀한 시각 지각을 강조하는 평가 |
| PixMo | "Molmo의 데이터" | Allen AI의 71만 2천 개 밀집 캡션 이미지 데이터셋. 사람 음성을 밀집 캡션으로 전사 |

## 더 읽을거리 (Further Reading)

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
