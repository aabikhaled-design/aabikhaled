# 3D Generation

> 3D는 2D-투-3D 레버리지가 가장 강한 모달리티다. 2023년의 돌파구는 3D 가우시안 스플래팅(3D Gaussian Splatting)이었다. 2024-2026년 생성 흐름은 그 위에 멀티뷰 디퓨전(multi-view diffusion) + 3D 재구성을 얹어 단일 프롬프트(prompt)나 사진으로부터 객체와 장면을 만들어낸다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45분

## 문제 (The Problem)

3D 콘텐츠는 고통스럽다.

- **표현(Representation).** 메시(mesh), 점 구름(point cloud), 복셀 그리드(voxel grid), 부호 거리 함수(signed distance field, SDF), 신경 방사 필드(neural radiance field, NeRF), 3D 가우시안. 각각 트레이드오프(trade-off)가 있다.
- **데이터 희소성(Data scarcity).** ImageNet에는 1,400만 장의 이미지가 있다. 가장 큰 깨끗한 3D 데이터셋(Objaverse-XL, 2023)에는 ~1,000만 개의 객체가 있는데, 대부분 저품질이다.
- **메모리(Memory).** 512³ 복셀 그리드는 1억 2,800만 개의 복셀이다. 쓸 만한 장면 NeRF는 광선(ray)당 100만 개의 샘플이 필요하다. 생성은 재구성보다 어렵다.
- **지도(Supervision).** 2D 이미지에는 픽셀이 있다. 3D에는 보통 소수의 2D 뷰만 있어서 3D로 끌어올려야(lift) 한다.

2026년 스택은 두 문제를 분리한다. 첫째, 디퓨전 모델로 *2D 멀티뷰 이미지*를 생성한다. 둘째, 그 이미지들에 *3D 표현*(보통 가우시안 스플래팅)을 맞춘다(fit).

## 개념 (The Concept)

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 표현: 3D 가우시안 스플래팅 (Kerbl et al., 2023)

장면을 ~100만 개의 3D 가우시안 구름으로 표현한다. 각각은 59개의 파라미터(parameter)를 가진다. 위치(3), 공분산(6, 또는 쿼터니언 4 + 스케일 3), 불투명도(1), 구면 조화(spherical-harmonics) 색상(차수 3에서 48, 차수 0에서 3).

렌더링 = 투영 + 알파 합성(alpha-compositing). 빠르다(4090에서 1080p로 ~100fps). 미분 가능하다. 실제 사진을 기준으로 경사 하강법(gradient descent)으로 맞춘다. 한 장면은 소비자용 GPU에서 5-30분 만에 맞춰진다.

그 위에 2023-2024년의 두 가지 혁신이 있다.
- **생성형 가우시안 스플랫(Generative Gaussian splats).** LGM, LRM, InstantMesh 같은 모델은 하나 또는 소수의 이미지로부터 가우시안 구름을 직접 예측한다.
- **4D 가우시안 스플래팅(4D Gaussian Splatting).** 동적 장면을 위해 프레임별 오프셋을 가진 가우시안.

### 멀티뷰 디퓨전 (Multi-view diffusion)

사전 학습된 이미지 디퓨전 모델을 파인튜닝(fine-tuning)해 텍스트 프롬프트나 단일 이미지로부터 같은 객체의 일관된 여러 뷰를 생성하게 한다. Zero123(Liu et al., 2023), MVDream(Shi et al., 2023), SV3D(Stability, 2024), CAT3D(Google, 2024). 보통 객체 주위로 4-16개의 뷰를 출력하고, 가우시안 스플래팅이나 NeRF를 거쳐 3D로 끌어올린다.

### 텍스트-투-3D 파이프라인

| 모델 | 입력 | 출력 | 시간 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | SDS를 통한 NeRF | 에셋당 ~1시간 |
| Magic3D | text | mesh + texture | ~40분 |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1분 |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30분 |
| LRM (Meta, 2023) | image | triplane | ~5초 |
| InstantMesh (2024) | image | mesh | ~10초 |
| SV3D (Stability, 2024) | image | novel views | ~2분 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1분 |
| TripoSR (2024) | image | mesh | ~1초 |
| Meshy 4 (2025) | text + image | PBR mesh | ~30초 |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60초 |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30초 |

2025-2026년 방향: 게임 엔진에 적합한 PBR 재질을 가진 직접 텍스트-투-메시 모델. 멀티뷰 디퓨전 중간 단계는 여전히 일반 객체에서 가장 성능 좋은 레시피다.

### NeRF (맥락을 위해)

신경 방사 필드(Neural Radiance Field, Mildenhall et al., 2020). 작은 MLP가 `(x, y, z, view direction)`을 받아 `(color, density)`를 출력한다. 광선을 따라 적분해 렌더링한다. 품질 면에서 메시 기반 새 시점 합성(novel-view synthesis)을 능가하지만 렌더링이 100-1000배 느리다. 대부분의 실시간 용도에서는 가우시안 스플래팅으로 대체되었지만 연구에서는 여전히 지배적이다.

```figure
v4-3d-multiview
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 2D "가우시안 스플래팅" 맞춤을 구현한다. 합성 타깃 이미지(매끄러운 그래디언트)를 2D 가우시안 스플랫의 합으로 표현한다. 타깃과 일치하도록 위치, 색상, 공분산을 경사 하강법으로 최적화한다. 두 가지 핵심 연산을 볼 수 있다. 순방향 렌더(스플랫 + 알파 합성)와 경사 하강법에 의한 맞춤.

### Step 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: render by summing splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

실제 3D 가우시안 스플래팅은 가우시안을 깊이순으로 정렬하고 순서대로 알파 합성한다. 우리의 2D 장난감은 그냥 합한다.

### Step 3: fit by gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 함정 (Pitfalls)

- **뷰 비일관성(View inconsistency).** 4개의 뷰를 독립적으로 생성했는데 객체 구조를 두고 서로 어긋나면, 3D 맞춤이 흐릿해진다. 해결책: 공유 어텐션(attention)을 쓰는 멀티뷰 디퓨전.
- **뒷면 환각(Back-side hallucination).** 단일 이미지 → 3D는 보이지 않는 면을 지어내야 한다. 품질이 천차만별이다.
- **가우시안 스플랫 폭발.** 제약 없는 학습은 1,000만 개의 스플랫으로 자라 과적합(overfitting)한다. (3D-GS 원논문의) 밀집화(densification) + 가지치기 휴리스틱이 필수다.
- **토폴로지 문제.** 암시적 필드(SDF)에서 나온 메시는 종종 구멍이나 자기 교차가 있다. 내보내기 전에 리메셔(예: 블렌더의 복셀 리메시)를 실행하라.
- **학습 데이터의 라이선스.** Objaverse는 라이선스가 혼합되어 있다. 상업적 사용은 모델마다 다르다.

## 라이브러리로 써보기 (Use It)

| 작업 | 2026년 선택 |
|------|-----------|
| 사진으로부터 장면 재구성 | 가우시안 스플래팅(3DGS, Gsplat, Scaniverse) |
| 게임용 텍스트-투-3D 객체 | Meshy 4 또는 Rodin Gen-1.5(PBR 출력) |
| 이미지-투-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 소수 이미지로부터 새 시점 합성 | CAT3D, SV3D |
| 동적 장면 재구성 | 4D 가우시안 스플래팅 |
| 아바타 / 옷 입은 인간 | Gaussian Avatar, HUGS |
| 연구 / SOTA | 지난주에 나온 무엇이든 |

게임이나 이커머스 파이프라인에서 프로덕션 3D를 내보내려면: Meshy 4나 Rodin Gen-1.5가 Unity / Unreal로 바로 들어가는 PBR 메시를 출력한다.

## 산출물 (Ship It)

`outputs/skill-3d-pipeline.md`를 저장한다. 이 스킬은 3D 브리프(입력: 텍스트 / 이미지 한 장 / 이미지 몇 장; 출력: 메시 / 스플랫 / NeRF; 용도: 렌더 / 게임 / VR)를 받아 다음을 출력한다. 파이프라인(멀티뷰 디퓨전 + 맞춤, 또는 직접 메시 모델), 베이스 모델, 반복 예산, 토폴로지 후처리, 필요한 재질 채널.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 가우시안 4, 16, 64개로 실행하라. 타깃 대비 최종 MSE를 보고하라.
2. **보통.** 색상 가우시안(RGB)으로 확장하라. 재구성이 타깃 색상 패턴과 일치하는지 확인하라.
3. **어려움.** gsplat이나 Nerfstudio를 사용해 50장 사진 캡처로부터 실제 객체를 재구성하라. 맞춤 시간과 보류된 뷰의 최종 SSIM을 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 3D 가우시안 스플래팅(3D Gaussian Splatting) | "3DGS" | 3D 가우시안 구름으로서의 장면. 미분 가능한 알파 합성 렌더. |
| NeRF | "신경 방사 필드" | 3D 점에서 색상 + 밀도를 출력하는 MLP. 광선 적분으로 렌더링. |
| 트라이플레인(Triplane) | "3개의 2-D 평면" | 3D를 3개의 2-D 축 정렬 특성(feature) 그리드로 분해. 체적식보다 저렴. |
| SDS | "점수 증류 샘플링" | 2D 디퓨전 점수를 의사 그래디언트로 써서 3D 모델을 학습. |
| 멀티뷰 디퓨전(Multi-view diffusion) | "한 번에 여러 뷰" | 일관된 카메라 뷰의 배치를 출력하는 디퓨전 모델. |
| PBR | "물리 기반 렌더링" | 알베도, 거칠기, 금속성, 노멀 채널을 가진 재질. |
| 밀집화(Densification) | "스플랫을 키운다" | 3DGS 학습 휴리스틱: 그래디언트가 큰 영역에서 스플랫을 분할 / 복제. |

## 프로덕션 노트: 3D에는 아직 공유 기반이 없다

이미지(잠재 디퓨전 + DiT)와 영상(시공간 DiT)과 달리, 3D에는 2026년에 단일 지배적 런타임이 없다. 프로덕션 의사결정 트리는 표현에서 갈린다.

- **NeRF / 트라이플레인.** 추론(inference)은 레이 마칭(ray-marching) + 샘플당 MLP 순방향이다. 512² 렌더는 수백만 번의 MLP 순방향이 필요하다. 광선 샘플을 공격적으로 배치(batch)하라. SDPA/xformers가 적용된다.
- **멀티뷰 디퓨전 + LRM 재구성.** 2단계 파이프라인. 1단계(멀티뷰 DiT)는 Lesson 07과 똑같은 디퓨전 서버다. 2단계(LRM 트랜스포머)는 뷰를 받는 일회성 순방향 패스다. 전체 지연 시간(latency) 프로파일은 "디퓨전 + 일회성"이다. 단계별 서빙 프리미티브를 그에 맞게 골라라.
- **SDS / DreamFusion.** 추론이 아니라 에셋별 최적화다. 요청 핸들러가 아니라 빌드 작업을 만든다.

2026년 대부분의 제품에 맞는 답은 "요청 시 멀티뷰 디퓨전 모델을 실행하고, 비동기로 3DGS로 재구성하고, 실시간 보기를 위해 3DGS를 서빙한다"이다. 이는 작업 부하를 GPU 추론 서버(빠름)와 오프라인 옵티마이저(optimizer)(느림) 사이로 깔끔하게 나눈다.

## 더 읽을거리 (Further Reading)

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512): 멀티뷰 디퓨전.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
