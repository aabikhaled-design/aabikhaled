# Emu3: 이미지·비디오 생성을 위한 다음 토큰 예측(Next-Token Prediction)

> BAAI의 Emu3(Wang et al., 2024년 9월)는 디퓨전(diffusion) 대 자기회귀(autoregressive) 논쟁을 끝냈어야 할 2024년의 결과물이다. 단일 Llama 스타일 디코더 전용(decoder-only) 트랜스포머(transformer)를, 텍스트 + VQ 이미지 토큰 + 3D VQ 비디오 토큰의 통합 어휘에 걸쳐 다음 토큰 예측(next-token-prediction) 목표만으로 학습시켜, 이미지 생성에서 SDXL을, 지각(perception)에서 LLaVA-1.6을 이긴다. CLIP 손실(loss)도, 디퓨전 스케줄도 없다. 품질을 위해 추론(inference) 시 분류기 없는 가이던스(classifier-free guidance)를 쓰지만, 핵심 학습 목표는 교사 강요(teacher forcing)를 사용한 다음 토큰 예측이다. Nature에 게재되었다. 이 레슨은 Emu3 논제(왜 더 나은 토크나이저(tokenizer)와 규모만 있으면 되는지)를 읽고 디퓨전 접근법과 대조한다.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 이미지 품질에는 디퓨전이 필요하다는 오랜 가정에도 불구하고, Emu3의 단일 손실 다음 토큰 목표가 왜 작동하는지 설명하기.
- 3D 비디오 토크나이저를 기술하기: 시공간(spatiotemporal) VQ 코드북(codebook)이 어떻게 생겼는지, 왜 패치(patch)가 시간을 가로지르는지.
- (학습 연산량, 추론 비용, 품질 한계) 측면에서 Emu3와 Stable Diffusion XL을 비교하기.
- 동일한 Emu3 모델이 수행하는 세 가지 역할의 이름을 대기: Emu3-Gen(이미지 생성), Emu3-Chat(지각), Emu3-Stage2(비디오 생성).

## 문제 (The Problem)

2024년까지의 통념: 이미지 생성에는 디퓨전이 필요하다. 그 논거는 이렇다. 이산(discrete) 이미지 토큰은 세부를 재구성하기에는 정보를 너무 많이 잃고, 자기회귀 샘플링(sampling)은 수천 개 토큰에 걸쳐 오차를 누적한다. Stable Diffusion, DALL-E 3, Imagen, Midjourney 모두 어떤 형태든 디퓨전을 쓴다. Chameleon(레슨 12.11)은 작은 규모에서 이를 부분적으로 반증했지만 품질에서 SDXL과 대등하지는 못했다.

Emu3는 그 논거를 정면으로 공격했다. 주장은 이렇다. 더 나은 시각 토크나이저 + 충분한 규모 + 다음 토큰 손실 = 같은 모델에서 지각도 하면서 디퓨전을 이기는 이미지 생성.

이 베팅은 발표 당시 논쟁적이었다. 2년이 지난 지금, 오픈소스 통합 생성 계열(Emu3, Show-o, Janus-Pro, Transfusion)은 연구의 기본 경로다. 프로덕션(production) 프런티어 모델들도 어떤 변형을 쓰는 듯하다.

## 개념 (The Concept)

### Emu3 토크나이저

핵심 재료는 시각 토크나이저다. Emu3는 토큰당 8x8 해상도 축소로 맞춤 IBQ 계열 토크나이저(Inverse Bottleneck Quantizer, SBER-MoVQGAN 계열)를 학습시킨다. 512x512 이미지는 코드북 크기 32768에서 64x64 = 4096개 토큰이 된다.

이는 Chameleon의 512x512당 1024 토큰(K=8192)보다 크지만 토큰당으로는 더 싸다(코드북 조회가 작고 코덱이 단순). 핵심 지표는 재구성 PSNR 30.5 dB로, 32 dB인 Stable Diffusion의 연속 잠재 공간(latent space)과 경쟁할 만하다.

비디오의 경우 3D VQ 토크나이저가 시공간 패치(4x4x4 픽셀)를 하나의 정수로 인코딩한다. 8 FPS의 4초 클립은 32프레임이다. 256x256에서 공간 4배·시간 4배 축소를 적용하면 토큰 수는 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768개다.

토크나이저 품질이 한계다. Emu3의 기여 일부는 "우리는 매우 좋은 토크나이저를 학습시켰다"이다.

### 단일 손실 학습

Emu3는 하나의 목표를 쓴다. 텍스트 토큰, 2D 이미지 토큰, 3D 비디오 토큰에 걸친 공유 어휘에서의 다음 토큰 예측이다. 기여도를 맞추려고 학습 중 가중치(weight)에 모달리티별 계수를 곱하지만, 손실 함수는 동일하다.

다음의 혼합으로 학습한다.
- 이미지 생성: `<text caption> <image> image_tokens </image>`
- 이미지 지각: `<image> image_tokens </image> <question> text_tokens`
- 비디오 생성: `<text caption> <video> video_tokens </video>`
- 비디오 지각: 유사하게.
- 텍스트만: 표준 NTP.

모델은 데이터 분포로부터 언제 이미지 토큰 대 텍스트 토큰을 낼지 학습한다. 생성은 모델이 `<image>` 태그 뒤에 이미지 토큰을 예측하는 데서 창발한다.

### 분류기 없는 가이던스와 온도

자기회귀 이미지 생성은 추론 시 분류기 없는 가이던스(classifier-free guidance, CFG)로 훨씬 좋아진다. Emu3는 이를 사용한다. 두 번 생성하되, 한 번은 전체 캡션으로, 한 번은 빈 캡션으로 하고, 가이던스 가중치(보통 3.0~7.0)로 로짓(logit)을 섞는다. 디퓨전이 쓰는 것과 같은 CFG 기법을 자기회귀 환경으로 빌려온 셈이다.

온도(temperature)가 중요하다. 너무 높으면 아티팩트, 너무 낮으면 모드 붕괴(mode collapse)가 일어난다. Emu3의 권장 온도는 지각에 1.0, 이미지 생성에 0.8이다.

### 세 역할, 하나의 모델

Emu3는 기능적으로 구분되는 세 개의 API로 출시되지만 기저 가중치 집합은 하나다.

- Emu3-Gen. 이미지 생성. 텍스트 입력, 이미지 토큰 출력.
- Emu3-Chat. VQA와 캡셔닝. 이미지(토큰) 입력, 텍스트 출력.
- Emu3-Stage2. 비디오 생성과 비디오 VQA. 텍스트 또는 비디오 입력, 텍스트 또는 비디오 출력.

태스크별 헤드는 없다. 그저 다른 프롬프트(prompt) 템플릿일 뿐이다. 같은 체크포인트.

### 벤치마크

Emu3 논문(2024년 9월)에서:

- 이미지 생성: MJHQ-30K FID(5.4 대 5.6), GenEval 종합(0.54 대 0.55: 통계적 동률)에서 SDXL을 이기고, Deep-Eval의 종합에서 대등하다.
- 이미지 지각: VQAv2(75.1 대 72.4)에서 LLaVA-1.6을 이기고, MMMU에서 대략 대등하다.
- 비디오 생성: Sora 시대에 공개적으로 벤치마크된 모델들과 경쟁력 있는 FVD로 4초 클립 품질을 낸다.

수치가 항상 이기는 것은 아니다. Emu3는 여기서 한 점을 내주고 저기서 한 점을 얻는다. 하지만 "다음 토큰 예측만 있으면 된다"는 주장은 여러 모달리티에 걸쳐 방어할 만하다.

### 연산 비용

Emu3는 7B 파라미터(parameter) 모델로 약 3000억 개 멀티모달 토큰에 대해 학습되었다. GPU-시간은 Llama-2-7B 사전 학습(pretraining)과 대략 비슷하다(A100급 실리콘에서 2k~4k GPU-년). Stable Diffusion 3 같은 디퓨전 모델은 비슷한 예산으로 학습하지만 별도의 텍스트 인코더(encoder)와 더 복잡한 파이프라인(pipeline)이 필요하다.

추론 시 Emu3는 이미지당 SDXL보다 느리다. 30 tok/s에서 4096개 이미지 토큰은 512x512 이미지당 약 2분인 반면, SDXL은 2~5초다. 추측 디코딩(speculative decoding)과 KV 캐시 최적화가 격차를 줄이지만 없애지는 못한다. 자기회귀 이미지 생성은 연산이 많이 든다. 늘 따라붙는 트레이드오프(trade-off)다.

### 왜 중요한가

Emu3의 깊은 기여는 개념적이다. 다음 토큰 예측이 이미지 생성에서 디퓨전과 대등할 만큼 확장된다면, 통합 모델 경로(하나의 손실, 하나의 백본(backbone), 임의의 모달리티)는 실현 가능하다. 미래 모델은 별도의 텍스트 인코더, 별도의 디퓨전 스케줄러, 별도의 VAE가 필요 없다. 하나의 트랜스포머, 모달리티당 하나의 토크나이저, 규모면 된다.

Show-o, Janus-Pro, InternVL-U는 모두 이 논제 위에 쌓거나 이에 도전한다. 중국 연구소(BAAI, DeepSeek)는 2025년까지 미국 연구소보다 이 방향으로 더 공격적으로 발표한다.

```figure
l5-emu3-next-token
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 두 가지 장난감 조각을 만든다.

- 2D 대 3D VQ 토크나이저 개수 계산기: (해상도, 패치, 클립 길이, FPS)가 주어지면 이미지 대 비디오의 토큰 수를 계산한다.
- 온도에서 분류기 없는 가이던스를 적용한 자기회귀 이미지 토큰 샘플러.

CFG 구현은 Emu3의 레시피와 일치한다. 조건부 로짓과 무조건부 로짓을 가이던스 가중치로 섞는다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-token-gen-cost-analyzer.md`를 만든다. 생성 제품 명세(이미지 또는 비디오, 목표 해상도, 품질 등급, 지연 시간(latency) 예산)가 주어지면, 토큰 수와 추론 비용을 계산하고 Emu3 계열 대 디퓨전을 고른다.

## 연습 문제 (Exercises)

1. Emu3는 8x8 축소로 512x512 이미지당 4096개 토큰을 만든다. 1024x1024와 2048x2048에 대한 등가값을 계산하라. 추론 지연 시간에는 무슨 일이 생기는가?

2. 비디오 토크나이저에 관한 Emu3 3.3절을 읽으라. 3D VQ 패치 형태를 기술하고 왜 8x8x1이 아니라 4x4x4인지 설명하라.

3. 분류기 없는 가이던스 가중치 5.0 대 3.0: 어떤 시각적 효과가 있는가? `code/main.py`에서 수식을 추적하라.

4. 3000억 토큰에서 Emu3-7B의 학습 FLOPs를 계산하고 Stable Diffusion 3과 비교하라. 어느 쪽이 학습에 더 비쌌는가?

5. Emu3는 FID에서 SDXL을 이기지만 특화된 VLM 대비 VQAv2에서는 그렇지 못하다. 통합 손실 접근법이 서로 다른 벤치마크에서 특화 모델 대비 다른 강점을 보이는 이유를 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 다음 토큰 예측(Next-token prediction) | "NTP" | 표준 자기회귀 손실: token[0..i]가 주어졌을 때 token[i+1] 예측; 토큰화하면 모든 모달리티에 통한다 |
| IBQ 토크나이저(IBQ tokenizer) | "역병목 양자화기" | 더 큰 코드북(32768+)과 Chameleon보다 나은 재구성을 갖는 VQ-VAE의 한 부류 |
| 3D VQ | "시공간 양자화기" | (시간, 행, 열)로 인덱싱되는 코드북; 토큰 하나가 4x4x4 픽셀 큐브를 덮는다 |
| 분류기 없는 가이던스(Classifier-free guidance) | "CFG" | 조건부와 무조건부 로짓을 가중치 gamma로 섞는다; 추론 시 이미지 품질을 높인다 |
| 통합 어휘(Unified vocabulary) | "공유 토큰" | 텍스트 + 이미지 + 비디오가 모두 같은 정수 공간에서 나온다; 모델이 다음에 올 모달리티를 예측한다 |
| MJHQ-30K | "이미지 생성 벤치마크" | 3만 개 프롬프트를 갖춘 Midjourney 품질 벤치마크; Emu3는 여기서 FID를 보고한다 |

## 더 읽을거리 (Further Reading)

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
