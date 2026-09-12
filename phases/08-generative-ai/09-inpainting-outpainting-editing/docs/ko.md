# Inpainting, Outpainting & Image Editing

> 텍스트-투-이미지(text-to-image)는 새로운 것을 만든다. 인페인팅(inpainting)은 기존의 것을 고친다. 프로덕션(production)에서 과금 대상 이미지 작업의 70%는 편집이다. 배경 교체, 로고 제거, 캔버스 확장, 손 다시 그리기. 인페인팅은 디퓨전(diffusion)이 제값을 하는 지점이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75분

## 문제 (The Problem)

한 클라이언트가 완벽한 제품 사진을 보냈는데 배경에 시선을 분산시키는 표지판이 있다. 그 표지판만 지우고 나머지는 픽셀 단위로 동일하게 남기고 싶다. 텍스트-투-이미지를 처음부터 실행할 수는 없다. 결과물은 색이 다르고, 조명이 다르고, 제품 각도도 달라진다. 원하는 것은 *마스크된 영역만* 다시 생성하되, 그 재생성이 주변 맥락을 존중하는 것이다.

그것이 인페인팅이다. 변종은 다음과 같다.

- **인페인팅(Inpainting).** 마스크 안쪽을 다시 생성하고, 바깥쪽 픽셀은 유지한다.
- **아웃페인팅(Outpainting).** 마스크 바깥쪽(또는 캔버스 너머)을 다시 생성하고, 안쪽은 유지한다.
- **이미지 편집(Image editing).** 이미지 전체를 다시 생성하되 원본에 대한 의미적·구조적 충실도를 유지한다(SDEdit, InstructPix2Pix).

2026년의 모든 디퓨전 파이프라인(pipeline)에는 인페인팅 모드가 탑재되어 있다. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. 이들은 모두 같은 원리로 동작한다.

## 개념 (The Concept)

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 순진한 접근법(그리고 왜 틀렸는가)

표준 텍스트-투-이미지를 마스크와 함께 실행한다. 각 샘플링(sampling) 스텝에서, 노이즈가 낀 잠재(latent)의 마스크되지 않은 영역을 순방향 디퓨전된 깨끗한 이미지로 교체한다. 동작은 한다... 하지만 형편없다. 모델이 마스크된 영역 안에 무엇이 있는지 전혀 모르기 때문에 경계 아티팩트(boundary artifact)가 새어 나온다.

### 제대로 된 인페인팅 모델

4개가 아닌 9개의 입력 채널(channel)을 받는 수정된 U-Net을 학습(training)한다.

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

추가 채널은 VAE로 인코딩된 원본 이미지의 사본과 단일 채널 마스크다. 학습 시에는 이미지의 영역을 무작위로 마스킹하고, 마스크되지 않은 영역을 깨끗한 조건 신호(conditioning signal)로 제공한 채 마스크된 영역만 디노이징(denoising)하도록 모델을 학습시킨다. 추론(inference) 시에는 모델이 마스크된 영역을 둘러싼 것을 "볼" 수 있어 일관성 있는 완성을 만들어낸다.

SD-Inpaint, SDXL-Inpaint, Flux-Fill는 모두 이 9채널(또는 그에 준하는) 입력을 사용한다. Diffusers의 `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022): 무료 편집

원본 이미지에 어떤 중간 시점 `t`까지 노이즈를 추가한 다음, 새 프롬프트(prompt)로 `t`에서 0까지 역방향 체인을 실행한다. 재학습이 필요 없다. 시작 `t`를 어떻게 고르느냐가 충실도와 창의적 자유 사이의 트레이드오프(trade-off)를 결정한다.

- `t/T = 0.3` → 원본과 거의 동일, 작은 양식적 변화
- `t/T = 0.6` → 중간 정도 편집, 거친 구조 보존
- `t/T = 0.9` → 거의 노이즈에서 생성, 원본 보존 최소

### InstructPix2Pix (Brooks et al., 2023)

디퓨전 모델을 `(input_image, instruction, output_image)` 삼중쌍(triple)으로 파인튜닝(fine-tuning)한다. 추론 시에는 입력 이미지와 텍스트 지시("make it sunset", "add a dragon") 양쪽 모두를 조건으로 준다. 두 개의 CFG 스케일을 쓴다. 이미지 스케일과 텍스트 스케일.

### RePaint (Lugmayr et al., 2022)

표준 무조건부(unconditional) 디퓨전 모델을 그대로 유지한다. 각 역방향 스텝에서 재샘플링한다. 가끔 더 노이즈가 낀 상태로 되돌아가서 다시 생성한다. 경계 아티팩트를 피한다. 학습된 인페인팅 모델이 없을 때 사용한다.

```figure
inpaint-mask-reinject
```

## 직접 만들기 (Build It)

`code/main.py`는 5차원 데이터에 대한 장난감 1차원 인페인팅 방식을 구현한다. 각 샘플이 두 클러스터 중 하나에서 나온 5개의 부동소수점 값인 5차원 혼합 데이터에 DDPM을 학습시킨다. 추론 시에는 5차원 중 2차원을 "마스킹"하고, 각 스텝에서 마스크되지 않은 세 차원의 노이즈 순방향 버전을 주입하며, 마스크된 차원만 다시 생성한다.

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims

표준 DDPM이다. 신경망은 5차원 노이즈가 낀 입력에 대해 5차원 노이즈 예측을 출력한다.

### Step 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

이것은 순진한 접근법이고 장난감 1차원 데이터에서는 동작한다. 실제 이미지 인페인팅은 텍스처 일관성이 더 중요하기 때문에 9채널 입력을 사용한다.

### Step 4: outpainting

아웃페인팅은 마스크를 뒤집은 인페인팅이다. 새로운(이전에는 존재하지 않던) 캔버스를 마스킹하고, 나머지는 원본으로 채운다. 학습 목표는 동일하다.

## 함정 (Pitfalls)

- **이음매(Seams).** 순진한 접근법은 그래디언트(gradient) 정보가 마스크를 가로질러 흐르지 않기 때문에 눈에 보이는 경계를 남긴다. 해결책: 마스크를 8-16픽셀 팽창(dilate)시키거나, 제대로 된 인페인팅 모델을 사용한다.
- **마스크 누출(Mask leakage).** 조건 이미지의 마스크되지 않은 영역이 저품질이거나 노이즈가 끼어 있으면, 마스크 안쪽의 생성을 오염시킨다. 살짝 디노이징하거나 블러 처리한다.
- **CFG는 마스크 크기와 상호작용한다.** 작은 마스크에 높은 CFG = 포화(saturation)된 패치. 작은 편집에는 CFG를 낮춘다.
- **SDEdit 충실도 절벽.** `t/T = 0.5`에서 `t/T = 0.6`으로 가면 피사체의 정체성을 잃을 수 있다. 스윕하고 체크포인트를 남긴다.
- **프롬프트 불일치.** 프롬프트는 새 콘텐츠만이 아니라 *전체* 이미지를 묘사해야 한다. "a cat"이 아니라 "A cat sitting on a chair".

## 라이브러리로 써보기 (Use It)

| 작업 | 파이프라인 |
|------|----------|
| 객체 제거, 작은 마스크 | SD-Inpaint 또는 Flux-Fill, 표준 프롬프트 |
| 하늘 교체 | SD-Inpaint + "blue sky at sunset" |
| 캔버스 확장 | SDXL 아웃페인트 모드(8px 페더) 또는 아웃페인트 마스크를 쓰는 Flux-Fill |
| 손 / 얼굴 재생성 | 피사체를 다시 묘사하는 프롬프트 + ControlNet-Openpose를 쓰는 SD-Inpaint |
| 한 영역의 스타일 변경 | 마스크된 영역에 `t/T=0.5`로 SDEdit |
| "Make it sunset" | InstructPix2Pix 또는 Flux-Kontext |
| 배경 교체 | SAM 마스크 → SD-Inpaint |
| 초고충실도 | 가장 어려운 경우에는 Flux-Fill 또는 GPT-Image(호스팅형) |

SAM(Meta의 Segment Anything, 2023) + 디퓨전 인페인트는 2026년의 배경 제거 파이프라인이다. SAM 2(2024)는 영상에서 동작한다.

## 산출물 (Ship It)

`outputs/skill-editing-pipeline.md`를 저장한다. 이 스킬은 원본 이미지 + 편집 설명 + 선택적 마스크(또는 SAM 프롬프트)를 받아 다음을 출력한다. 마스크 생성 방식, 베이스 모델, CFG 스케일(이미지 + 텍스트), SDEdit-t 또는 인페인팅 모드, 그리고 QA 체크리스트.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 마스킹되는 차원의 비율을 0.2에서 0.8까지 변화시켜라. 어느 비율에서 인페인트 품질(마스크된 차원의 잔차)이 무조건부 생성과 같아지는가?
2. **보통.** RePaint를 구현하라. 10번째 역방향 스텝마다 5스텝 되돌아가서(노이즈 추가) 다시 디노이징하라. 그것이 마스크 가장자리의 경계 잔차를 줄이는지 측정하라.
3. **어려움.** Hugging Face diffusers를 사용해 비교하라. SD 1.5 Inpaint + ControlNet-Openpose 대 Flux.1-Fill를 20개의 얼굴 재생성 작업에서 비교하라. 포즈 준수와 정체성 보존을 각각 따로 채점하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 인페인팅(Inpainting) | "구멍을 채운다" | 마스크 안쪽을 다시 생성하고 바깥쪽 픽셀은 유지한다. |
| 아웃페인팅(Outpainting) | "캔버스를 확장한다" | 캔버스 바깥쪽을 다시 생성하고 안쪽은 유지한다. |
| 9채널 U-Net(9-channel U-Net) | "제대로 된 인페인팅 모델" | `noisy \| encoded-source \| mask`를 입력으로 받는 U-Net. |
| SDEdit | "노이즈 레벨을 쓰는 Img2img" | 시점 `t`까지 노이즈를 넣고, 새 프롬프트로 디노이징한다. |
| InstructPix2Pix | "텍스트만으로 편집" | (이미지, 지시, 출력) 삼중쌍으로 파인튜닝한 디퓨전. |
| RePaint | "재학습 불필요" | 이음매를 줄이기 위해 역방향 중 주기적으로 다시 노이즈를 넣는다. |
| SAM | "Segment Anything" | 클릭 또는 박스로 마스크를 만드는 생성기. 인페인트와 짝지어 쓴다. |
| Flux-Kontext | "맥락을 활용한 편집" | 편집을 위해 참조 이미지 + 지시를 받는 Flux 변종. |

## 프로덕션 노트: 편집 파이프라인은 지연 시간에 민감하다

이미지를 편집하는 사용자는 5초 미만의 왕복을 기대한다. 1024²에서 30스텝 SDXL-Inpaint는 L4에서 3-4초이고, 여기에 SAM 마스크 생성(~200ms)과 VAE 인코드/디코드(합쳐서 ~500ms)가 더해진다. 프로덕션 관점에서 이것은 처리량(throughput) 제약이 아니라 TTFT 제약이다. 배치(batch) 1, 낮은 동시성, 모든 단계를 최소화한다.

- **SAM-H가 느린 녀석이다.** 1024²에서 SAM-H는 ~200ms이고, SAM-ViT-B는 약간의 품질 손실로 ~40ms다. SAM 2(영상)는 시간적 오버헤드를 더한다. 단일 이미지 편집에는 쓰지 말라.
- **가능하면 인코드를 건너뛴다.** `pipe.image_processor.preprocess(img)`는 잠재로 인코딩한다. 이전 생성에서 나온 잠재가 있다면(반복 편집 UI에서 전형적), `latents=...`로 직접 전달해 VAE 인코드 한 번을 건너뛴다.
- **마스크 팽창은 처리량에도 중요하다.** 작은 마스크는 U-Net 순방향 패스의 대부분이 낭비된다는 뜻이다(마스크되지 않은 픽셀은 어차피 고정되니까). `diffusers`의 `StableDiffusionInpaintPipeline`은 상관없이 전체 U-Net을 실행한다. 9채널의 제대로 된 인페인트 변종만이 마스크된 연산을 활용한다.
- **Flux-Kontext가 2025년의 답이다.** `(source_image, instruction)`에 대한 단일 순방향 패스. 별도 마스크도, SDEdit 노이즈 스윕도 없다. H100에서 편집을 ~1.5초에 내보낸다. 아키텍처적 교훈: 단계들을 하나로 합쳐라.

## 더 읽을거리 (Further Reading)

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865): 학습 불필요 인페인팅.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800): 텍스트 지시 편집.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643): SAM, 마스크 소스.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714): 영상 SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626): 어텐션 수준 편집.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/): 2024년 도구.
