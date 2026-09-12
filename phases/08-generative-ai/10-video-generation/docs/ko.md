# Video Generation

> 이미지는 2차원 텐서(tensor)다. 영상은 3차원 텐서다. 이론은 동일하지만 연산은 10-100배 더 어렵다. OpenAI의 Sora(2024년 2월)는 그것이 가능함을 증명했다. 2026년이 되자 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, WAN 2.2가 텍스트로부터 1080p 프로덕션(production) 영상을 내보낸다. 그리고 오픈 가중치(open-weights) 스택은 12개월 뒤처져 있다(CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45분

## 문제 (The Problem)

24fps의 10초짜리 1080p 영상은 1920×1080×3 픽셀의 프레임 240장이다. 클립 하나당 ~1.5GB의 원시 데이터다. 픽셀 공간 디퓨전(diffusion)은 실현 불가능하다. 다음이 필요하다.

1. **시공간 압축(Spatiotemporal compression).** 프레임이 아니라 영상을 시공간 패치(patch)의 시퀀스로 인코딩하는 VAE.
2. **시간적 일관성(Temporal coherence).** 프레임들은 수 초에 걸쳐 콘텐츠, 조명, 객체 정체성을 공유해야 한다. 신경망(neural network)은 움직임을 모델링해야 한다.
3. **연산 예산(Compute budget).** 같은 모델 크기에서 영상 학습(training)은 이미지보다 10-100배 비싸다.
4. **조건화(Conditioning).** 텍스트, 이미지(첫 프레임), 오디오, 또는 다른 영상. 대부분의 프로덕션 모델은 네 가지 모두를 받는다.

이를 해결한 아키텍처는 시공간 패치에 적용된 **디퓨전 트랜스포머(Diffusion Transformer, DiT)**이고, 거대한 (프롬프트, 캡션, 영상) 데이터셋으로 학습된다. Lesson 06과 동일한 디퓨전 손실(loss)을 쓴다.

## 개념 (The Concept)

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### 패치화 (Patchify)

영상을 3D VAE(학습된 시공간 압축)로 인코딩한다. 잠재(latent)는 형상 `[T_latent, H_latent, W_latent, C_latent]`이다. 크기 `[t_p, h_p, w_p]`의 패치로 나눈다. Sora 스타일 모델에서는 `t_p = 1`(프레임별 패치) 또는 `t_p = 2`(두 프레임마다)다. 10초짜리 1080p 영상은 ~20,000-100,000개의 패치로 압축된다.

### 시공간 DiT (Spatiotemporal DiT)

트랜스포머(transformer)가 평탄화된 패치 시퀀스를 처리한다. 각 패치는 3D 위치 임베딩(positional embedding)(시간 + y + x)을 가진다. 어텐션(attention)은 보통 분해(factorize)된다.

- **공간 어텐션(Spatial attention)**: 각 프레임의 패치 내에서.
- **시간 어텐션(Temporal attention)**: 같은 공간 위치에 있는 프레임들 사이에서.
- **완전 3D 어텐션(Full 3D attention)**은 16-100배 더 비싸다. 저해상도나 연구에서만 쓴다.

### 텍스트 조건화 (Text conditioning)

큰 텍스트 인코더(encoder)와의 교차 어텐션(cross-attention)(Sora는 T5-XXL, CogVideoX-5B는 T5-XXL을 쓴다). 긴 프롬프트가 중요하다. Sora의 학습 세트에는 클립당 평균 200토큰짜리 GPT 생성 밀집 재캡션이 있었다.

### 학습 (Training)

시공간 잠재에 대한 표준 디퓨전 손실(ε 또는 v 예측). 데이터: 웹 영상 + ~1억 개의 큐레이션된 클립 + 합성 텍스트 캡션. 연산: 작은 연구 실행에도 10,000+ GPU 시간이 든다. Sora 규모는 100,000+이다.

## 2026년 프로덕션 지형

| 모델 | 날짜 | 최대 길이 | 최대 해상도 | 오픈 가중치? | 특기 사항 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 아니오 | 대규모에서 월드 시뮬레이터 특성을 보인 최초 모델 |
| Sora Turbo | 2024-12 | 20s | 1080p | 아니오 | 5배 빠른 추론의 프로덕션 Sora |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 아니오 | 2025년 최고 품질 + 물리 |
| Veo 3 | 2025 Q3 | 15s | 4K | 아니오 | 네이티브 오디오와 더 강한 카메라 제어 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 아니오 | 2025년 Q1 최고의 인간 움직임 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 아니오 | 위에 얹은 전문가용 영상 도구 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 아니오 | 가장 강한 캐릭터 일관성 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 예 (2B, 5B) | 최초의 오픈 5B 규모 영상 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 예 (13B) | 2024년 말 오픈 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 예 (10B) | 가장 관대한 라이선스 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 예 | 2025년 중반 가장 강한 오픈 모델 |

오픈 가중치는 이미지 공간보다 격차를 더 빠르게 좁히고 있다. HunyuanVideo + WAN 2.2 LoRA는 2026년 중반이면 이미 대부분의 오픈소스 워크플로를 구동한다.

```figure
video-diffusion-denoise
```

## 직접 만들기 (Build It)

`code/main.py`는 핵심 시공간 DiT 아이디어를 시뮬레이션한다. 작은 합성 영상을 패치화하고, 패치별 위치 임베딩을 추가하고, 패치에 대한 트랜스포머 스타일 어텐션으로 시퀀스 전체를 디노이징(denoising)한다. numpy 없이 순수 파이썬이다. 인접 프레임 패치가 디노이저와 위치 임베딩을 공유하면 1차원에서도 시간적 일관성이 발현됨을 보인다.

### Step 1: patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence

각 프레임을 독립적으로 디노이징하는 대신, 우리의 작은 신경망은 모든 프레임 값 + 그들의 위치 임베딩을 이어 붙여 모든 프레임의 노이즈를 한꺼번에 예측한다.

### Step 4: temporal coherence test

학습 후, 영상을 하나 샘플링한다. 프레임 간 델타를 측정한다. 모델이 시간적 구조를 학습했다면, 델타는 각 프레임을 독립적으로 샘플링하는 것보다 작게 유지된다.

## 함정 (Pitfalls)

- **프레임별 독립 샘플링 = 깜빡임(flicker).** 각 프레임에 이미지 디퓨전을 따로 실행하면, 각 프레임의 노이즈가 독립적이기 때문에 출력이 깜빡인다. 영상 디퓨전은 어텐션이나 공유 노이즈로 프레임을 묶어 이를 해결한다.
- **순진한 3D 어텐션 = OOM.** 10초짜리 1080p 잠재에 완전 3D 어텐션을 쓰면 수천억 번의 연산이다. 공간 + 시간으로 분해하라.
- **데이터 캡셔닝이 크기보다 더 중요하다.** Sora의 이전 작업 대비 주요 업그레이드는 ~10배 더 상세한 캡션(GPT-4가 재라벨링한 클립)으로 학습한 것이었다. OpenAI 기술 보고서가 이를 명시한다.
- **첫 프레임 조건화.** 대부분의 프로덕션 모델은 첫 프레임으로 이미지도 받는다. 이것이 "이미지-투-비디오" 모드다. 학습에 이 변종이 포함된다.
- **물리 표류(Physics drift).** 긴 클립(>10초)은 미묘한 비일관성을 누적한다. 슬라이딩 윈도우 생성 + 키프레임 앵커링이 도움이 된다.

## 라이브러리로 써보기 (Use It)

| 사용 사례 | 2026년 선택 |
|----------|-----------|
| 최고 품질 텍스트-투-비디오, 호스팅형 | Veo 3 또는 Sora |
| 카메라 제어 영화적 | 모션 브러시를 쓰는 Runway Gen-3 |
| 클립 간 캐릭터 일관성 | Pika 2.0 또는 Kling 2.1 |
| 오픈 가중치, 빠른 파인튜닝 | WAN 2.2 + LoRA |
| 이미지-투-비디오 | WAN 2.2-I2V, Kling 2.1 I2V, 또는 Runway |
| 오디오-투-비디오 립싱크 | Veo 3(네이티브 오디오) 또는 전용 립싱크 모델 |
| 영상 편집 | Runway Act-Two, Kling Motion Brush, Flux-Kontext(스틸 프레임) |

품질이 동등한 조건에서 영상 1초당 비용은 2024년과 2026년 사이에 20배 떨어졌다.

## 산출물 (Ship It)

`outputs/skill-video-brief.md`를 저장한다. 이 스킬은 영상 브리프(길이, 종횡비, 스타일, 카메라 계획, 피사체 일관성, 오디오)를 받아 다음을 출력한다. 모델 + 호스팅, 프롬프트 골격(카메라 언어, 피사체 설명, 움직임 서술어), 시드 + 재현성 프로토콜, 그리고 프레임 수준 QA 체크리스트.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 (a) 프레임별 독립 샘플링, (b) 결합 시퀀스 샘플링에 대해 프레임 간 델타를 비교하라. 델타의 평균과 분산을 보고하라.
2. **보통.** 첫 프레임 조건을 추가하라. 프레임 0을 주어진 값에 고정하고 나머지를 샘플링하라. 고정된 값이 어떻게 전파되는지 측정하라.
3. **어려움.** HuggingFace diffusers를 사용해 로컬 GPU에서 CogVideoX-2B를 실행하라. 6초짜리 클립에 대해 720p에서 20번의 추론 스텝의 시간을 측정하라. 시공간 어텐션을 프로파일링해 병목을 찾아내라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 영상 VAE(Video VAE) | "3-D VAE" | `(T, H, W, C)` → 시공간 잠재로 압축하는 인코더. |
| 패치(Patches) | "토큰" | 잠재의 고정 크기 3-D 블록. DiT의 입력. |
| 분해 어텐션(Factorized attention) | "공간 + 시간" | 공간에 대해 어텐션을 실행한 다음 시간에 대해 실행한다. 완전 3-D 어텐션은 건너뛴다. |
| 이미지-투-비디오 (I2V)(Image-to-video) | "이 사진을 움직이게 하라" | 모델이 이미지 + 텍스트를 받아 그것에서 시작하는 영상을 출력한다. |
| 키프레임 조건화(Keyframe conditioning) | "앵커 프레임" | 영상의 흐름을 제어하기 위해 특정 프레임을 고정한다. |
| 모션 브러시(Motion brush) | "방향 힌트" | 사용자가 이미지에 움직임 벡터를 그려 넣는 UI 입력. |
| 재캡셔닝(Re-captioning) | "밀집 캡션" | LLM을 사용해 학습 클립을 상세한 프롬프트로 재라벨링하는 것. |
| 깜빡임(Flicker) | "시간적 아티팩트" | 프레임 간 비일관성. 결합 디노이징으로 해결된다. |

## 프로덕션 노트: 영상 잠재는 메모리 대역폭 문제다

24fps의 10초짜리 1080p 클립은 240프레임 × 1920 × 1080 × 3 ≈ 1.5GB의 원시 픽셀이다. 4× 영상 VAE 압축(`2 × 공간 × 2 × 시간`) 후 잠재는 요청당 ~100MB다. 이것을 배치(batch) 1로 30스텝 동안 시공간 DiT에 통과시키면 스텝당 ~3GB를 HBM으로 옮기게 된다. FLOPs가 아니라 메모리 대역폭이 병목이다.

세 가지 프로덕션 노브, 모두 프로덕션 추론(inference) 문헌의 추론 장에서 곧장 가져온 것이다.

- **DiT에 걸친 TP.** 텍스트-투-비디오 모델은 으레 10B+ 파라미터(parameter)다. 4개의 H100에 걸친 TP=4가 표준이고, 405B급 모델에는 PP=2 × TP=2를 쓴다. 스텝당 지연 시간(latency)은 all-reduce 벽에 닿기 전까지 TP에 따라 대략 선형으로 떨어진다.
- **프레임 배칭 = 연속 배칭(continuous batching).** 생성 시점에 영상은 개념적으로 어텐션으로 연결된 프레임의 배치다. 연속 배칭(인플라이트 스케줄링)이 적용된다. 모델 아키텍처가 슬라이딩 윈도우 생성을 허용한다면, 프레임 `t-1`이 반환되는 동안 프레임 `t+1` 렌더링을 시작한다.
- **클립 수준 프리필 캐시.** 이미지-투-비디오에서 첫 프레임 조건화는 LLM의 프롬프트 프리필과 유사하다. 한 번 계산하고 시간적 디코더 패스에 걸쳐 재사용한다. 이것은 사실상 영상용 KV 캐시다.

## 더 읽을거리 (Further Reading)

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/): Sora 기술 보고서.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/): 2025년 중반 오픈 SOTA.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458): 영상 디퓨전의 효시 논문.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818): Stable Video Diffusion의 조상.
