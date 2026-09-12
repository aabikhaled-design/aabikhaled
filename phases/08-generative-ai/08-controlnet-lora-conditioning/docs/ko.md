# ControlNet, LoRA, 그리고 조건화 (ControlNet, LoRA & Conditioning)

> 텍스트만으로는 서툰 제어 신호다. ControlNet은 사전 학습된 확산(diffusion) 모델을 복제해, 깊이 맵·포즈 골격·끄적임(scribble)·가장자리 이미지로 조종하게 해준다. LoRA는 1천만 개의 파라미터(parameter)만 학습해 2B 파라미터 모델을 파인튜닝(fine-tune)한다. 둘이 합쳐져 Stable Diffusion을 장난감에서 모든 에이전시가 출시하는 2026년 이미지 파이프라인으로 바꿔놓았다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch: LoRA 기반)
**Time:** ~75분

## 문제 (The Problem)

"붐비는 거리에서 개를 산책시키는 빨간 드레스를 입은 여자" 같은 프롬프트(prompt)는 모델에게 개가 *어디에* 있는지, 여자가 *어떤 포즈*인지, 거리의 *원근*이 어떤지에 대한 정보를 전혀 주지 않는다. 텍스트는 이미지를 명시하는 데 필요한 것의 약 10%를 고정한다. 나머지는 시각적이고 말로는 효율적으로 묘사할 수 없다.

모든 신호(포즈, 깊이, canny, 분할)마다 밑바닥부터 새 조건부 모델을 학습하는 것은 엄두를 낼 수 없다. 대신 2.6B 파라미터 SDXL 백본(backbone)은 동결한 채로 두고, 조건화를 읽는 작은 측면 네트워크를 붙여 백본의 중간 특성(feature)을 살짝 밀게 하면 된다. 이것이 ControlNet이다.

전체 모델을 재학습하지 않고 모델에게 새 개념(내 얼굴, 내 제품, 내 스타일)을 가르치고 싶을 때도 있다. 이때 필요한 것은 100배 작은 델타(delta)다. 이것이 LoRA(기존 어텐션(attention) 가중치(weight)에 끼워 넣는 저랭크 어댑터(low-rank adapter)) 다.

ControlNet + LoRA + 텍스트 = 2026년 실무자의 도구상자다. 대부분의 프로덕션(production) 이미지 파이프라인은 SDXL / SD3 / Flux 기저 위에 2-5개의 LoRA, 1-3개의 ControlNet, 그리고 IP-Adapter를 쌓는다.

## 개념 (The Concept)

![ControlNet은 인코더를 복제하고; LoRA는 저랭크 델타를 더한다](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

사전 학습된 SD를 가져온다. U-Net의 인코더 절반을 *복제*한다. 원본을 동결한다. 복제본이 추가 조건화 입력(가장자리, 깊이, 포즈)을 받도록 학습시킨다. *제로 합성곱(zero-convolution)* 스킵 연결(skip connection)(0으로 초기화된 1×1 합성곱: 무연산으로 시작해 델타를 학습)로 복제본을 원본의 디코더 절반에 다시 연결한다.

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

제로 합성곱 초기화는 ControlNet이 항등으로 시작함을 뜻한다. 학습 전에도 해가 없다. 표준 확산 손실(loss)로 100만 개의 (프롬프트, 조건, 이미지) 삼중쌍에서 학습한다.

모달리티별 ControlNet은 작은 측면 모델(SDXL용 약 360M, SD 1.5용 약 70M)로 제공된다. 추론(inference) 시 이들을 조합할 수 있다.

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

모델 안의 어떤 선형 층(layer) `W ∈ R^{d×d}`에 대해, `W`를 동결하고 저랭크 델타를 더한다.

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

여기서 `r << d`다. 랭크 4-16이 어텐션에 표준이고, 랭크 64-128은 무거운 파인튜닝용이다. 새 파라미터 수: `d²` 대신 `2 · d · r`. `d=640`, `r=16`인 SDXL 어텐션의 경우: 어댑터당 410k 대신 20k 파라미터: 20배 감소. 모델 전체에서: LoRA는 보통 기저 5GB 대비 20-200MB다.

추론 시 LoRA를 스케일할 수 있다: `W' = W + α · B @ A`. `α = 0.5-1.5`가 보통이다. 여러 LoRA는 가산적으로 쌓인다(비선형적으로 상호작용한다는 통상적 주의사항과 함께).

### IP-Adapter (Ye et al., 2023)

조건화로 *이미지*를 (텍스트와 함께) 받는 아주 작은 어댑터다. CLIP 이미지 인코더로 이미지 토큰을 생성하고, 텍스트 토큰과 함께 교차 어텐션(cross-attention)에 주입한다. 기저 모델당 약 20MB. LoRA 없이도 "이 레퍼런스의 스타일로 이미지 생성"이 된다.

## 조합성 행렬 (Composability matrix)

| 도구 | 제어하는 것 | 크기 | 사용 시점 |
|------|------------------|------|-------------|
| ControlNet | 공간 구조(포즈, 깊이, 가장자리) | 70-360MB | 정확한 배치, 구성 |
| LoRA | 스타일, 주제, 개념 | 20-200MB | 개인화, 스타일 |
| IP-Adapter | 레퍼런스 이미지로부터의 스타일 또는 주제 | 20MB | 텍스트로 외양을 묘사할 수 없을 때 |
| 텍스트 역변환(Textual Inversion) | 새 토큰으로서의 단일 개념 | 10KB | 레거시, 대부분 LoRA로 대체됨 |
| DreamBooth | 주제에 대한 전체 파인튜닝 | 2-5GB | 강한 신원, 높은 연산 |
| T2I-Adapter | 더 가벼운 ControlNet 대안 | 70MB | 엣지 디바이스, 추론 예산 |

ControlNet ≈ 공간. LoRA ≈ 의미. 둘 다 써라.

```figure
v4-controlnet-zero
```

## 직접 만들기 (Build It)

`code/main.py`는 두 메커니즘을 1차원에서 시뮬레이션한다.

1. **LoRA.** 사전 학습된 선형 층 `W`를 동결한다. `W + BA`가 목표 선형 층과 맞도록 저랭크 `B @ A`를 학습한다. `r = 1`이면 랭크-1 보정을 완벽하게 학습하기에 충분함을 보여준다.

2. **ControlNet-lite.** "동결 기저" 예측기와 추가 신호를 읽는 "측면 네트워크". 측면 네트워크의 출력은 0으로 초기화된 학습 가능한 스칼라(우리 버전의 제로 합성곱)로 게이팅된다. 학습하고 게이트가 올라가는 것을 지켜본다.

### 1단계: LoRA 수학

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 2단계: 0으로 초기화된 측면 네트워크

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

스텝 0에서 출력은 기저와 동일하다. 초기 학습은 `gate`를 천천히 갱신한다. 파국적 표류 없음.

## 함정 (Pitfalls)

- **LoRA 과도 스케일링.** `α = 2`나 `α = 3`은 "더 강하게 만들기"의 흔한 꼼수로, 과도하게 양식화되거나 망가진 출력을 만든다. `α ≤ 1.5`를 유지하라.
- **ControlNet 가중치 충돌.** 가중치 1.0의 포즈 ControlNet과 가중치 1.0의 깊이 ControlNet을 함께 쓰면 보통 과도하게 넘어간다. 가중치 합 ≈ 1.0이 안전한 기본값이다.
- **잘못된 기저의 LoRA.** SDXL LoRA는 어텐션 차원이 맞지 않으므로 SD 1.5에서 조용히 무연산이 된다. Diffusers 0.30+에서 경고한다.
- **텍스트 역변환 표류.** 한 체크포인트에서 학습된 토큰은 다른 체크포인트에서 심하게 표류한다. LoRA가 더 이식성이 좋다.
- **LoRA 가중치 병합과 저장.** 더 빠른 추론을 위해 LoRA를 기저 모델 가중치에 구워 넣을 수 있지만(런타임 덧셈 없음), 런타임에 `α`를 스케일하는 능력을 잃는다. 두 버전을 모두 유지하라.

## 라이브러리로 써보기 (Use It)

| 목표 | 2026 파이프라인 |
|------|---------------|
| 브랜드의 아트 스타일 재현 | 랭크 32로 엄선된 약 30장 이미지에 학습한 LoRA |
| 생성된 이미지에 내 얼굴 넣기 | DreamBooth 또는 LoRA + IP-Adapter-FaceID |
| 특정 포즈 + 프롬프트 | ControlNet-Openpose + SDXL + 텍스트 |
| 깊이 인지 구성 | ControlNet-Depth + SD3 |
| 레퍼런스 + 프롬프트 | IP-Adapter + 텍스트 |
| 정확한 배치 | ControlNet-Scribble 또는 ControlNet-Canny |
| 배경 교체 | ControlNet-Seg + 인페인팅(Inpainting)(Lesson 09) |
| 빠른 1-스텝 스타일 | SDXL-Turbo 위의 LCM-LoRA |

## 산출물 (Ship It)

`outputs/skill-sd-toolkit-composer.md`로 저장하라. 이 스킬은 작업(입력 에셋: 프롬프트, 선택적 레퍼런스 이미지, 선택적 포즈, 선택적 깊이, 선택적 끄적임)을 받아 도구 스택, 가중치, 그리고 재현 가능한 시드(seed) 프로토콜을 출력한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`에서 LoRA 랭크 `r`을 1에서 4까지 변화시켜라. 어느 랭크에서 LoRA가 랭크-2 목표 델타와 정확히 일치하는가?
2. **보통.** 두 목표 변환에 두 개의 별도 LoRA를 학습하라. 둘을 함께 로드해 가산적 상호작용을 보여라. 상호작용이 언제 선형성을 깨는가?
3. **어려움.** diffusers를 사용해 쌓아라: SDXL-base + Canny-ControlNet(가중치 0.8) + 스타일 LoRA(α 0.8) + IP-Adapter(가중치 0.6). 스택 가중치가 변함에 따라 FID 대 프롬프트 준수 트레이드오프(trade-off)를 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| ControlNet | "공간 제어" | 복제된 인코더 + 제로 합성곱 스킵; 조건화 이미지를 읽는다. |
| 제로 합성곱(Zero convolution) | "항등으로 시작" | 0으로 초기화된 1×1 합성곱; ControlNet이 무연산으로 시작한다. |
| LoRA | "저랭크 어댑터" | `W + B @ A`, `r << d`; 전체 파인튜닝보다 100배 적은 파라미터. |
| 랭크 r(rank r) | "그 손잡이" | LoRA 압축; 4-16이 일반적, 무거운 개인화에는 64+. |
| α | "LoRA 강도" | LoRA 델타의 런타임 스케일링. |
| IP-Adapter | "레퍼런스 이미지" | CLIP-이미지 토큰을 통한 작은 이미지 조건화 어댑터. |
| DreamBooth | "전체 주제 파인튜닝" | 주제의 약 30장 이미지에 전체 모델을 학습. |
| 텍스트 역변환(Textual Inversion) | "새 토큰" | 새 단어 임베딩(embedding)만 학습; 레거시, 대부분 대체됨. |

## 프로덕션 노트: LoRA 교체, ControlNet 차선, 멀티테넌트 서빙 (Production note: LoRA swaps, ControlNet lanes, multi-tenant serving)

실제 텍스트-이미지 SaaS는 같은 기저 체크포인트 위에서 수백 개의 LoRA와 십여 개의 ControlNet을 서빙한다. 서빙 문제는 LLM 멀티테넌시(multi-tenancy)와 많이 닮았다(프로덕션 문헌은 연속 배칭(continuous batching)과 LoRAX / S-LoRA 아래에서 LLM 사례를 다룬다).

- **LoRA를 핫스왑(hot-swap)하고, 병합하지 마라.** `W' = W + α·B·A`를 기저에 병합하면 스텝당 추론이 약 3-5% 빨라지지만 `α`와 기저를 동결한다. LoRA를 랭크-r 델타로 VRAM에 핫 상태로 유지하라; diffusers는 요청별 활성화를 위해 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`를 노출한다. 교체 비용은 `2 · d · r · num_layers` 가중치: MB 규모, 1초 미만.
- **두 번째 어텐션 차선으로서의 ControlNet.** 복제된 인코더는 기저와 병렬로 돌아간다. 각각 가중치 1.0인 두 ControlNet = 스텝당 한 번의 병합 패스가 아니라 두 번의 추가 순방향 패스(forward pass). 배치 크기 여유가 이차적으로 떨어진다. 활성 ControlNet당 약 1.5배 스텝 비용을 예산으로 잡아라.
- **양자화된 LoRA도.** 기저를 양자화했다면(Lesson 07, 8GB에서의 Flux 참조), LoRA 델타도 8비트나 4비트로 깔끔하게 양자화된다. QLoRA 스타일 로딩은 메모리를 터뜨리지 않고 4비트 Flux 기저 위에 5-10개의 LoRA를 쌓을 수 있게 한다.

Flux 특화: Niels의 Flux-on-8GB 노트북은 기저를 4비트로 양자화한다; 그 양자화된 기저 위에 스타일 LoRA(`pipe.load_lora_weights("user/style-lora")`)를 `weight_name="pytorch_lora_weights.safetensors"`로 쌓아도 여전히 작동한다. 이것이 2026년 대부분의 SaaS 에이전시가 출시하는 레시피다.

## 더 읽을거리 (Further Reading)

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): LoRA(원래 LLM용; 확산으로 이식됨).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453): ControlNet의 더 가벼운 대안.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers: ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) 레퍼런스 파이프라인.
