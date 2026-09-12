# 잠재 확산과 Stable Diffusion (Latent Diffusion & Stable Diffusion)

> 512×512 이미지에 대한 픽셀 공간 확산(diffusion)은 연산상 전쟁 범죄다. Rombach et al. (2022)은 이미지를 생성하는 데 786k 차원 전부가 필요한 것이 아님을 알아챘다. 의미 구조를 포착할 만큼만 필요하고, 나머지는 별도의 디코더(decoder)가 맡는다. VAE의 잠재 공간(latent space) 안에서 확산을 돌려라. 그 하나의 아이디어가 Stable Diffusion이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75분

## 문제 (The Problem)

512²에서의 픽셀 공간 확산은 U-Net이 `[B, 3, 512, 512]` 형태의 텐서(tensor)에서 돌아간다는 뜻이다. 각 샘플링 스텝은 500M 파라미터(parameter) U-Net에 대해 약 100 GFLOPS다. 50스텝이면 이미지당 5 TFLOPS다. 10억 개 이미지로 학습하면 연산 비용이 터무니없다.

그 FLOPs의 대부분은 지각적으로 중요하지 않은 세부를 네트워크에 밀어 넣는 데 쓰인다. 손실 압축 VAE가 압축해 없앨 수 있는 고주파 텍스처다. Rombach의 아이디어: VAE를 한 번 학습시키고(*첫 번째 단계(first stage)*), 동결한 다음, 확산을 전적으로 4채널 64×64 잠재 공간(*두 번째 단계(second stage)*)에서 돌린다. 같은 U-Net. 1/16의 픽셀. 비슷한 품질에 약 64배 적은 FLOPs.

이것이 Stable Diffusion 레시피다. SD 1.x / 2.x는 `64×64×4` 잠재 위에서 860M U-Net을 사용했고, SDXL은 `128×128×4` 위에서 2.6B U-Net을, SD3은 U-Net을 흐름 매칭(flow matching)을 가진 확산 트랜스포머(Diffusion Transformer, DiT)로 교체했다. Flux.1-dev (Black Forest Labs, 2024)는 12B 파라미터 DiT-MMDiT를 함께 제공한다. 모두 같은 2단계 토대(substrate) 위에서 돌아간다.

## 개념 (The Concept)

![잠재 확산: VAE 압축 + 잠재 공간에서의 확산](../assets/latent-diffusion.svg)

**별도로 학습되는 두 단계.**

1. **1단계: VAE.** 인코더 `E(x) → z`, 디코더 `D(z) → x`. 목표 압축: 각 공간 축에서 8배 다운샘플 + 채널 조정으로 전체 잠재 크기를 픽셀 수의 약 1/16로. 손실(loss) = 재구성(L1 + LPIPS 지각) + KL(작은 가중치, `z`가 너무 가우시안이 되도록 강제하지 않음, `z`에서 정확한 샘플링이 필요 없기 때문). 디코딩된 이미지가 선명하도록 종종 적대적 손실(adversarial loss)로 학습된다.

2. **2단계: `z`에 대한 확산.** `z = E(x_real)`을 데이터로 취급한다. `z_t`의 잡음을 제거하도록 U-Net(또는 DiT)을 학습시킨다. 추론(inference) 시: 확산으로 `z_0`을 샘플링한 다음 `x = D(z_0)`.

**텍스트 조건화(Text conditioning).** 두 가지 추가 구성 요소. 동결된 텍스트 인코더(SD 1.x는 CLIP-L, SD 2/XL은 CLIP-L+OpenCLIP-G, SD3과 Flux는 T5-XXL). 교차 어텐션(cross-attention) 주입: 모든 U-Net 블록이 `[Q = 이미지 특성, K = V = 텍스트 토큰]`을 받아 섞는다. 텍스트가 이미지에 영향을 주는 통로는 이 토큰뿐이다.

**손실 함수는 Lesson 06과 동일하다.** 잡음에 대한 같은 DDPM / 흐름 매칭 MSE다. 데이터 도메인만 교체할 뿐이다.

## 아키텍처 변형 (Architecture variants)

| 모델 | 연도 | 백본 | 잠재 형태 | 텍스트 인코더 | 파라미터 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 토큰) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 리파이너 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 증류됨(Distilled) | 128×128×4 | 동일 | 1-4 스텝 샘플링 |
| SD3 | 2024 | MMDiT (멀티모달 DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 증류됨 | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 스텝 |

추세: U-Net을 DiT(잠재 패치에 대한 트랜스포머)로 교체, 텍스트 인코더 확장(프롬프트 준수에서 T5가 CLIP을 능가), 잠재 채널 증가(4 → 16이면 세부를 담을 여유가 늘어난다).

```figure
noise-schedule
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 1차원 "VAE"(시연용 항등 인코더 + 디코더; 실제 VAE라면 합성곱 네트워크일 것이다)를 Lesson 06의 DDPM 위에 쌓고 분류기 없는 안내(classifier-free guidance)를 가진 클래스 조건화를 추가한다. 이는 원시 1차원 값에서 돌리든 인코딩된 값에서 돌리든 같은 확산 손실이 작동함을 보여준다. 핵심 통찰이다.

### 1단계: 인코더/디코더

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

실제 VAE는 학습된 가중치(weight)를 가진다. 교육 목적상, 이 선형 매핑은 확산이 원래 데이터 공간에 신경 쓰지 않고 `z`에서 작동함을 보여주기에 충분하다.

### 2단계: `z`-공간에서의 확산

Lesson 06과 같은 DDPM. 네트워크가 보는 데이터는 `z = E(x)`다. `z_0`을 샘플링한 후 `D(z_0)`으로 디코딩한다.

### 3단계: 분류기 없는 안내

학습 중 10%는 클래스 레이블(label)을 버린다(널 토큰으로 교체). 추론 시 `ε_cond`와 `ε_uncond`를 모두 계산한 다음:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 안내 없음(완전한 다양성), `w = 3` = 기본값, `w = 7+` = 포화(saturation) / 과도하게 선명.

### 4단계: 텍스트 조건화(개념, 코드 아님)

클래스 레이블을 동결된 텍스트 인코더 출력으로 교체한다. 교차 어텐션을 통해 텍스트 임베딩(embedding)을 U-Net에 넣는다.

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

이것이 클래스 조건부 확산 모델과 Stable Diffusion 사이의 유일한 실질적 차이다.

## 함정 (Pitfalls)

- **VAE 스케일 불일치.** SD 1.x VAE는 인코딩 후 적용되는 스케일링 상수(`scaling_factor ≈ 0.18215`)를 가진다. 이를 잊으면 U-Net이 분산이 크게 잘못된 잠재로 학습된다. 모든 체크포인트가 하나씩 함께 제공한다.
- **텍스트 인코더가 조용히 잘못됨.** SD3은 128개 이상의 토큰을 가진 T5-XXL이 필요하고, CLIP만으로의 폴백(fallback)은 손실이 있다. 항상 `use_t5=True`를 확인하라, 그렇지 않으면 프롬프트 충실도가 폭락한다.
- **잠재 공간 섞기.** SDXL, SD3, Flux는 모두 다른 VAE를 사용한다. SDXL 잠재에서 학습된 LoRA는 SD3에서 작동하지 않는다. Hugging Face diffusers 0.30+는 불일치하는 체크포인트의 로드를 거부한다.
- **CFG가 너무 높음.** `w > 10`은 포화되고 기름진 이미지를 만들고 다양성을 희생해 프롬프트에 과적합(overfitting)한다. 최적 지점은 `w = 3-7`이다.
- **부정 프롬프트(negative prompt) 누설.** 빈 부정 프롬프트는 널 토큰이 되고; 채워진 부정 프롬프트는 `ε_uncond`가 된다. 이 둘은 같지 않다; 일부 파이프라인은 조용히 널로 기본 설정된다.

## 라이브러리로 써보기 (Use It)

2026년 프로덕션(production) 스택:

| 목표 | 권장 백본 |
|--------|----------------------|
| 좁은 도메인, 짝지어진 데이터, 밑바닥부터 모델 학습 | SDXL 파인튜닝(fine-tune)(LoRA / 전체): 출시가 가장 빠름 |
| 개방형 도메인 텍스트-이미지, 공개 가중치 | Flux.1-dev (12B, Apache / 비상업) 또는 SD3.5-Large |
| 가장 빠른 추론, 공개 가중치 | Flux.1-schnell (1-4 스텝, Apache) 또는 SDXL-Lightning |
| 최고의 프롬프트 준수, 호스팅 | GPT-Image / DALL-E 3 (여전히), Midjourney v7, Imagen 4 |
| 편집 워크플로 | Flux.1-Kontext (2024년 12월): 네이티브로 이미지 + 텍스트를 받음 |
| 연구, 베이스라인(baseline) | SD 1.5: 오래됐지만 잘 연구됨 |

## 산출물 (Ship It)

`outputs/skill-sd-prompter.md`로 저장하라. 이 스킬은 텍스트 프롬프트 + 목표 스타일을 받아 다음을 출력한다: 모델 + 체크포인트, CFG 스케일, 샘플러, 부정 프롬프트, 해상도, 선택적 ControlNet/IP-Adapter 조합, 그리고 스텝별 QA 체크리스트.

## 연습 문제 (Exercises)

1. **쉬움.** 안내 `w ∈ {0, 1, 3, 7, 15}`로 `code/main.py`를 실행하라. 클래스별 평균 샘플을 기록하라. 어느 `w`에서 클래스 평균이 실제 데이터 평균을 넘어 발산하는가?
2. **보통.** 장난감 선형 인코더를 재구성 손실을 가진 tanh-MLP 인코더/디코더 쌍으로 교체하라. 새 잠재에서 확산을 재학습하라. 샘플 품질이 바뀌는가?
3. **어려움.** diffusers로 실제 Stable Diffusion 추론을 설정하라: `sdxl-base`를 로드하고, CFG=7로 30 Euler 스텝을 돌리고, 시간을 측정하라. 이제 4스텝과 CFG=0으로 `sdxl-turbo`로 전환하라. 같은 주제, 다른 품질: 무엇이 바뀌었고 왜인지 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| 첫 번째 단계(First stage) | "그 VAE" | 학습된 인코더/디코더 쌍; 512²를 64²로 압축. |
| 두 번째 단계(Second stage) | "그 U-Net" | 잠재 공간에 대한 확산 모델. |
| CFG | "안내 스케일" | `(1+w)·ε_cond - w·ε_uncond`; 조건화 강도를 조정. |
| 널 토큰(Null token) | "빈 프롬프트 임베드" | `ε_uncond`에 사용되는 무조건 임베드. |
| 교차 어텐션(Cross-attention) | "텍스트가 들어가는 법" | 각 U-Net 블록이 텍스트 토큰을 K와 V로 어텐션. |
| DiT | "확산 트랜스포머" | U-Net을 잠재 패치에 대한 트랜스포머로 교체; 더 잘 확장됨. |
| MMDiT | "멀티모달 DiT" | SD3의 아키텍처: 결합 어텐션을 가진 텍스트와 이미지 스트림. |
| VAE 스케일링 인자(VAE scaling factor) | "마법의 숫자" | 확산이 단위 분산 공간에서 작동하도록 잠재를 약 5.4로 나눔. |

## 프로덕션 노트: 8GB 소비자용 GPU에서 Flux-12B 돌리기 (Production note: running Flux-12B on an 8GB consumer GPU)

레퍼런스 Flux 통합은 "나에게 소비자용 GPU가 있는데, 이것을 출시할 수 있나?"의 전형적인 레시피다. 핵심은 프로덕션 추론 문헌이 나열하는 바로 그 세 손잡이 레시피를 확산 DiT에 적용하는 데 있다.

1. **시차 로딩(Staggered loading).** Flux는 VRAM에 공존할 필요가 결코 없는 세 네트워크를 가진다: T5-XXL 텍스트 인코더(fp32에서 약 10 GB), CLIP-L(작음), 12B MMDiT, 그리고 VAE. 먼저 프롬프트를 인코딩하고, 인코더를 *삭제*하고, DiT를 로드하고, 잡음을 제거하고, DiT를 *삭제*하고, VAE를 로드하고, 디코딩한다. 소비자용 8GB GPU는 한 번에 한 단계만 들어간다.
2. **bitsandbytes를 통한 4비트 양자화.** T5 인코더와 DiT 양쪽에 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`. 메모리를 8배 줄이고, Aritra의 벤치마크(benchmark)(노트북에 링크됨)에 따르면 텍스트-이미지에서 품질 저하는 감지할 수 없다.
3. **CPU 오프로드(offload).** `pipe.enable_model_cpu_offload()`는 각 순방향 패스(forward pass)가 진행됨에 따라 모듈을 CPU와 GPU 사이에서 자동으로 교체한다. 지연 시간(latency)을 10-20% 추가하지만 파이프라인이 아예 돌아가게 만든다.

메모리 회계는 이렇다: `10 GB T5 / 8 = 1.25 GB` 양자화, `12 B 파라미터 × 0.5 바이트 = 약 6 GB` 양자화된 DiT, 거기에 활성값(activation). stas00의 용어로 이것은 TP=1 추론의 극단(모델 병렬화 없음, 최대 양자화) 이다. 프로덕션에서는 H100에서 TP=2나 TP=4로 돌리겠지만, 단일 개발 노트북에는 이것이 레시피다.

## 더 읽을거리 (Further Reading)

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1: Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) Flux.1 계열.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index): 위 모든 체크포인트의 레퍼런스 구현.
