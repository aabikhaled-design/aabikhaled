# 생성 모델(분류 체계와 역사 (Generative Models) Taxonomy & History)

> 모든 이미지 모델, 텍스트 모델, 비디오 모델, 3D 모델은 다섯 개의 통(bucket) 중 하나에 들어간다. 잘못된 통을 고르면 몇 주 동안 수학과 씨름하게 된다. 올바른 통을 고르면 지난 12년간의 분야 발전사가 머릿속에 깔끔하게 쌓인다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45분

## 문제 (The Problem)

생성 모델(generative model)은 단 하나의 일을 한다. 어떤 미지의 분포 `p_data(x)`에서 추출된 학습 샘플(training sample)이 주어졌을 때, 같은 분포에서 나온 것처럼 보이는 새로운 샘플을 출력하는 것이다. 얼굴, 문장, MIDI 파일, 단백질 구조: 눈을 가늘게 뜨고 보면 모두 같은 문제다.

문제는 `p_data`가 수백만 차원의 공간에 산다는 점이다(512x512 RGB 이미지는 약 786k 차원이다). 샘플들은 그 공간 안의 얇은 다양체(manifold) 위에 놓여 있고, 가진 예제는 기껏해야 1천만 개 정도다. 밀도(density)를 무차별 대입으로 푸는 것은 가망이 없다. 모든 생성 모델은 하나의 어려운 문제를 약간 덜 어려운 문제로 바꾸는 타협(compromise)이다.

지난 12년간 다섯 개의 계열(family)이 살아남았다. 각 계열이 어떤 타협을 하는지 알면, 왜 어떤 작업에서는 이기고 어떤 작업에서는 무너지는지 알 수 있다.

## 개념 (The Concept)

![생성 모델의 다섯 계열: 무엇을 모델링하는가에 따른 분류 체계](../assets/taxonomy.svg)

**1. 명시적 밀도, 다루기 쉬움(Explicit density, tractable).** `log p(x)`를 실제로 평가할 수 있는 합(sum)으로 쓴다. 자기회귀(autoregressive) 모델(PixelCNN, WaveNet, GPT)은 `p(x) = ∏ p(x_i | x_<i)`로 인수분해한다. 정규화 흐름(normalizing flow)(RealNVP, Glow)은 단순한 기저(base)의 가역(invertible) 변환으로 `p(x)`를 구성한다. 장점: 정확한 가능도(likelihood), 깔끔한 학습 손실(loss). 단점: 자기회귀 추론은 순차적이고(긴 시퀀스에서 느림), 흐름은 가역 아키텍처를 요구한다(아키텍처적으로 제약이 크다).

**2. 명시적 밀도, 근사(Explicit density, approximate).** `log p(x)`를 아래에서 한계 지어(ELBO) 그 한계를 최적화한다. VAE(Kingma 2013)는 변분 사후 분포(variational posterior)를 가진 인코더-디코더(encoder-decoder)를 사용한다. 확산(diffusion) 모델(DDPM, Ho 2020)은 가중 ELBO를 암묵적으로 최적화하는 잡음 제거기(denoiser)를 학습시킨다. 확산은 2026년 현재 이미지, 비디오, 3D의 지배적인 백본(backbone)이다.

**3. 암묵적 밀도(Implicit density).** 밀도를 아예 건너뛴다. 샘플을 생성하는 생성기(generator) `G(z)`와 진짜와 가짜를 구별하는 판별기(discriminator) `D(x)`를 학습한다. GAN(Goodfellow 2014). 추론은 빠르지만(순방향 패스 한 번) 학습 중에는 악명 높게 불안정하다. StyleGAN 1/2/3은 2026년에도 고정 도메인 사실적 묘사(얼굴, 침실)에서 여전히 최첨단(state of the art)이다.

**4. 점수 기반 / 연속 시간(Score-based / continuous-time).** 로그 밀도의 그래디언트(gradient) `∇_x log p(x)`(점수, score)를 직접 학습한다. Song & Ermon (2019)은 점수 매칭(score matching)이 확산을 SDE로 일반화함을 보였다. 흐름 매칭(flow matching)(Lipman 2023)은 2024-2026년의 화두다. 시뮬레이션 없는 학습, 더 곧은 경로, DDPM보다 4-10배 빠른 샘플링(sampling)을 제공한다. Stable Diffusion 3, Flux, AudioCraft 2가 모두 흐름 매칭을 사용한다.

**5. 이산 코드에 대한 토큰 기반 자기회귀(Token-based autoregressive over discrete codes).** VQ-VAE나 잔차 양자화기(residual quantizer)로 고차원 데이터를 짧은 이산 토큰(token) 시퀀스로 압축한 다음, 트랜스포머(Transformer)로 토큰 시퀀스를 모델링한다. Parti, MuseNet, AudioLM, VALL-E, Sora의 패치 토크나이저(patch tokenizer)가 모두 이것을 사용한다. 이것은 통 1에 학습된 토크나이저를 더한 것이다.

## 짧은 역사 (A brief history)

| 연도 | 모델 | 왜 중요했는가 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 사용 가능한 학습 손실을 가진 최초의 딥러닝 생성 모델. |
| 2014 | GAN (Goodfellow) | 암묵적 밀도, 가능도 없음: 충격적으로 선명한 샘플. |
| 2015 | DRAW, PixelCNN | 순차적 이미지 생성. |
| 2017 | Glow, RealNVP | 가역 흐름; 깊이를 갖춘 정확한 가능도. |
| 2017 | Progressive GAN | 최초의 메가픽셀 얼굴. |
| 2019 | StyleGAN / StyleGAN2 | 그 한 도메인에 한해서는 여전히 능가하기 어려운 사실적 얼굴. |
| 2020 | DDPM (Ho) | 확산이 실용화됨. |
| 2021 | CLIP, DALL-E 1, VQGAN | 텍스트-이미지가 주류로. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 잠재 확산(latent diffusion) + 텍스트 조건화 = 일상재. |
| 2022 | ControlNet, LoRA | 사전 학습된 확산에 대한 정밀 제어. |
| 2023 | SDXL, Midjourney v5, Flow matching | 규모 + 더 나은 학습 동역학. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 비디오 확산; 흐름 매칭이 승리. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 프로덕션(production) 급 비디오. |
| 2026 | Consistency + Rectified Flow | 확산 백본에서의 1-스텝 샘플링. |

## 다섯 가지 질문으로 하는 분류 (The five-question triage)

새로운 생성 모델 논문이 나오면, 방법론 섹션을 읽기 전에 이 다섯 가지 질문에 답하라.

1. **무엇이 모델링되고 있는가?** 픽셀, 잠재(latent), 이산 토큰, 3D 가우시안, 메시(mesh), 파형(waveform) 중 무엇인가?
2. **밀도가 명시적인가 암묵적인가?** `log p(x)`를 적어 두는가?
3. **샘플링: 단발(one-shot)인가 반복(iterative)인가?** 반복은 더 느린 추론을 뜻하고, 단발은 보통 적대적(adversarial)이거나 증류(distilled)된 것을 뜻한다.
4. **조건화(Conditioning): 무조건, 클래스, 텍스트, 이미지, 포즈 중 무엇인가?** 이것이 손실과 아키텍처 골격을 결정한다.
5. **평가(Evaluation): FID, CLIP score, IS, 인간 선호, 작업 정확도 중 무엇인가?** 각각 알려진 실패 모드가 있다(Lesson 14 참조).

이 다섯 가지는 이 phase의 모든 레슨에서 다시 답하게 된다. 끝날 무렵이면 반사적으로 답하게 된다.

```figure
autoencoder-bottleneck
```

## 직접 만들기 (Build It)

이 레슨의 코드는 가벼운 시각화다. 세 가지 장난감 접근법(커널 밀도, 이산 히스토그램, 그리고 최근접 샘플 "GAN스러운" 생성기)을 사용해 샘플로부터 1차원 가우시안 혼합(mixture-of-Gaussians)을 적합(fit)시킨다. 그래서 한 화면에 인쇄할 수 있는 문제에서 명시적 밀도와 암묵적 밀도의 차이를 볼 수 있다.

`code/main.py`를 실행하라. 이는 2개 모드의 가우시안 혼합에서 2000개의 샘플을 그린 다음, 다음을 출력한다.

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

주목하라. 앞의 둘은 "이 점이 얼마나 가능한가?"를 물을 수 있게 한다. 세 번째는 그럴 수 없다. 이것이 앞으로 모든 레슨에서 중요해질 *명시적 대 암묵적* 구분이다.

## 라이브러리로 써보기 (Use It)

2026년 현재, 어떤 작업에 어떤 계열을 쓸 것인가?

| 작업 | 최선의 계열 | 이유 |
|------|-------------|-----|
| 사실적 얼굴, 좁은 도메인 | StyleGAN 2/3 | 여전히 가장 선명하고 추론이 가장 빠름. |
| 일반 텍스트-이미지 | 잠재 확산 + 흐름 매칭 | SD3, Flux.1, DALL-E 3. |
| 빠른 텍스트-이미지 | 정류 흐름(rectified flow) + 증류 | SDXL-Turbo, SD3-Turbo, LCM. |
| 텍스트-비디오 | 확산 트랜스포머(Diffusion Transformer) + 흐름 매칭 | Sora, Veo 2, Kling. |
| 음성 + 음악 | 토큰 기반 AR(AudioLM, VALL-E, MusicGen) 또는 흐름 매칭(AudioCraft 2) | 이산 토큰은 저렴하게 확장됨. |
| 3D 장면 | 가우시안 스플래팅(Gaussian Splatting) 적합, 확산 사전(prior) | 재구성에는 3D-GS, 새로운 시점에는 확산. |
| 밀도 추정(샘플링 없음) | 흐름 | 정확한 `log p(x)`를 가진 유일한 계열. |
| 시뮬레이션 / 물리 | 흐름 매칭, score SDE | 직선 경로, 매끄러운 벡터장(vector field). |

## 산출물 (Ship It)

`outputs/skill-model-chooser.md`로 저장하라.

이 스킬은 작업 설명을 받아 다음을 출력한다. (1) 어떤 계열을 사용할지, (2) 오픈 옵션 3개와 호스팅 옵션 3개의 순위 목록, (3) 주의해야 할 가능성 높은 실패 모드, (4) 연산/시간 예산.

## 연습 문제 (Exercises)

1. **쉬움.** 다음 다섯 제품 각각에 대해 계열과 백본을 식별하라: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. 근거는 공개 기술 보고서에서 가져와야 한다.
2. **보통.** 내일 읽으려는 논문이 확산보다 100배 빠른 샘플링을 주장한다. 그 속도 향상이 조건화와 고해상도에서도 살아남는지 확인할 세 가지 질문을 적어라.
3. **어려움.** 관심 있는 한 도메인(예: 단백질 구조, CAD, 분자, 궤적)을 골라라. 그 도메인의 현재 SOTA 모델에 대해 다섯 가지 질문 분류에 답하고, 더 나은 모델이 무엇을 바꿀지 스케치하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 생성 모델(Generative model) | "새로운 것을 만든다" | `p_data(x)`에 대한 샘플러(sampler)를 학습하고, 선택적으로 `log p(x)`를 노출한다. |
| 명시적 밀도(Explicit density) | "평가할 수 있다" | 모델이 닫힌 형태이거나 다루기 쉬운 `log p(x)`를 제공한다. |
| 암묵적 밀도(Implicit density) | "GAN 스타일" | 샘플러만 있을 뿐: 주어진 점의 `p(x)`를 평가할 방법이 없다. |
| ELBO | "증거 하한(evidence lower bound)" | `log p(x)`에 대한 다루기 쉬운 하한; VAE와 확산이 이를 최적화한다. |
| 점수(Score) | "로그 밀도의 그래디언트" | `∇_x log p(x)`; 확산과 SDE 모델이 이 장(field)을 학습한다. |
| 다양체 가설(Manifold hypothesis) | "데이터는 어떤 표면 위에 산다" | 고차원 데이터가 저차원 다양체에 집중된다; 차원 축소가 작동하는 이유. |
| 자기회귀(Autoregressive) | "다음 조각을 예측한다" | 결합 분포를 조건부 분포들의 곱으로 인수분해한다. |
| 잠재(Latent) | "압축된 코드" | 디코더가 입력을 재구성할 수 있는 저차원 표현. |

## 프로덕션 노트: 다섯 계열, 다섯 추론 형태 (Production note: five families, five inference shapes)

각 계열은 서로 다른 추론 서버 비용 곡선에 대응된다. 프로덕션 추론 문헌은 LLM 추론을 프리필(prefill) + 디코드(decode)로 구성한다. 같은 분해가 여기에도 적용된다.

- **자기회귀(통 1과 5).** 순차적 디코드가 지연 시간(latency)을 지배한다. KV-cache, 연속 배칭(continuous batching), 추측 디코딩(speculative decoding)이 모두 직접 적용된다.
- **VAE / 확산 / 흐름 매칭(통 2와 4).** LLM 의미의 디코드는 없다. 비용 = `num_steps × step_cost`이고, `step_cost`는 전체 잠재 해상도에서의 트랜스포머 또는 U-Net 순방향이다. 프로덕션 조절 손잡이는 스텝 수(DDIM / DPM-Solver / 증류), 배치 크기, 정밀도(bf16 / fp8 / int4)다.
- **GAN(통 3).** 순방향 패스 한 번. 스케줄도, KV-cache도 없다. TTFT ≈ 전체 지연 시간. 이것이 StyleGAN이 좁은 도메인 UX에서 여전히 이기는 이유다.

논문 초록에서 "확산보다 빠름"을 보면, 그것을 "더 적은 스텝 × 같은 스텝 비용" 또는 "같은 스텝 × 더 저렴한 스텝 비용"으로 번역하라. 나머지는 전부 마케팅이다.

## 더 읽을거리 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661): GAN 논문.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): VAE 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239): DDPM 논문.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456): SDE로서의 확산.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): 흐름 매칭 논문.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
