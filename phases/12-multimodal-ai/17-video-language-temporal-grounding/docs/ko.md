# 비디오-언어 모델: 시간 토큰(Temporal Tokens)과 그라운딩(Grounding)

> 비디오는 사진의 더미가 아니다. 5초 클립에는 이미지 모델이 표현할 수 없는 인과 순서, 동작 동사, 사건 타이밍이 담긴다. Video-LLaMA(Zhang et al., 2023년 6월)는 오디오-비주얼 그라운딩(grounding)을 갖춘 최초의 오픈 비디오-LLM을 출시했다. VideoChat과 Video-LLaVA가 그 패턴을 확장했고, 2025년에는 Qwen2.5-VL의 TMRoPE가 프런티어 독점 모델과의 격차를 좁혔다. 각 시스템은 시간 토큰을 저마다 다르게 풀었다. 클립당 Q-former, 프레임당 concat-pool, 토큰당 TMRoPE다. 이 레슨은 그 패턴들을 읽고, 균일 샘플러와 동적 프레임 샘플러를 만들며, 시간 그라운딩 태스크에서 평가한다.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 비전 인코더(vision encoder)와 독립적으로 시간 위치 인코딩이 왜 비디오 VLM 성능을 바꾸는지 설명하기.
- 초당 토큰 수 대 그라운딩 정확도 측면에서 균일, 동적-FPS, 사건 기반 프레임 샘플링을 비교하기.
- 클립당 Q-former(Video-LLaMA), 프레임당 풀링(Video-LLaVA), 토큰당 M-RoPE(Qwen2.5-VL) 설계를 기술하기.
- 네 가지 비디오 벤치마크(benchmark)의 이름을 대기: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## 문제 (The Problem)

30 FPS의 1분 비디오는 1800프레임이다. 프레임당 196개 시각 토큰(224에서 ViT-B)이면 35만 2천 토큰인데, 2024년 시대의 어느 LLM 컨텍스트보다도 크다.

세 가지 축소 전략이 있다.

1. 프레임 서브샘플링(내용에 따라 1~8 FPS).
2. 각 프레임의 패치(patch) 토큰을 공격적으로 풀링(pooling)(3x3 또는 4x4 양선형 풀).
3. 16프레임 클립을 받아 64개 토큰을 내는 Q-former로 압축.

각각의 트레이드오프(trade-off)가 다르다. 서브샘플링은 시간 디테일을 잃는다. 풀링은 공간 디테일을 잃는다. Q-former는 둘 다 조금씩 잃지만 토큰을 절약한다.

시간 위치 인코딩은 다른 축이다. 모델은 프레임 5가 프레임 6 전에 왔다는 것을 어떻게 아는가? 선택지에는 단순한 1D 시간 RoPE(Video-LLaMA), 학습된 시간 임베딩(embedding)(Video-LLaVA), TMRoPE(Qwen2.5-VL, 완전 3D)가 있다.

## 개념 (The Concept)

### Video-LLaMA: 클립당 Q-former + 오디오 분기

Video-LLaMA(2023)는 최초의 오픈 비디오-LLM이었다. 아키텍처:

- 2 FPS의 16프레임 클립(즉 8초).
- 프레임당 ViT 특성 -> 16프레임 전체에 교차 어텐션(attention)하는 Video Q-former -> 32개 학습된 쿼리 -> LLM.
- 병렬 오디오 분기: 파형 -> ImageBind 오디오 인코더 -> Audio Q-former -> 32개 쿼리 -> LLM.

강점: 오디오-비주얼 공동 추론. 약점: 고정 클립 길이, 임의 시간 그라운딩 불가.

### VideoChat과 Video-LLaVA

VideoChat은 Video-LLaMA 아이디어를 유지하되 오디오를 빼고 단순화했다. Video-LLaVA(Lin et al., 2023)는 단일 시각 인코더를 이미지와 비디오 프레임 모두에 대해 학습시켜("투영 전 정렬", alignment before projection) 통합 표현을 주었다. 둘 다 동결-CLIP-인코더 + MLP + LLM이다.

둘 다 긴 비디오를 다루지 못한다. 둘 다 8~16프레임 시스템이다.

### Qwen2.5-VL과 TMRoPE

Qwen2.5-VL은 TMRoPE: Temporal-Modality Rotary Position Embedding을 도입했다. 각 패치 토큰이 (t, h, w) 위치를 담는데, 여기서 t는 (프레임 인덱스가 아니라) 실제 타임스탬프다.

단순 시간 임베딩과의 주요 차이:

- 인덱스가 아니라 절대 시간. 모델은 "프레임 15에서"가 아니라 "4.2초에서"를 본다.
- 클립당이 아니라 토큰당 회전. 각 시각 토큰이 자신의 타임스탬프로 독립적으로 회전한다.
- 동적 FPS와 호환. 여기서는 2 FPS, 저기서는 4 FPS로 샘플링해도 TMRoPE는 고르지 않은 간격을 네이티브하게 처리한다.

TMRoPE 덕분에 "고양이가 몇 초에 뛰는가?" 같은 질의가 가능해진다. 모델은 "4.2초에"라고 출력한다. Video-LLaMA는 "클립 초반에"라고만 답할 수 있었다.

### 프레임 샘플링 전략

균일(Uniform): 지속 시간에 걸쳐 N개 프레임을 고르게 샘플링. 단순하지만 모션 정점을 놓친다.

동적 FPS(Dynamic FPS): 모션 강도에 기반해 적응적으로 샘플링. 옵티컬 플로우나 프레임 차분이 고모션 구간을 골라 더 조밀하게 샘플링한다. Qwen2.5-VL이 이것으로 학습한다.

사건 기반(Event-driven): 가벼운 탐지기를 돌려 동작이 일어나는 곳에서 더 샘플링한다. VideoAgent가 쓴다.

키프레임 + 컨텍스트(Keyframe + context): 샷 경계 + 인접 프레임 몇 개를 샘플링한다. 영화 콘텐츠에 쓴다.

### 프레임당 풀링

1 FPS와 프레임당 576토큰에서, 5분 클립은 172,800토큰이다. Qwen2.5-VL-72B의 128k 컨텍스트로 가능하지만 비싸다.

3x3 양선형 풀은 프레임당 64토큰으로 줄여 -> 5분에 19,200토큰이 된다. 대부분의 태스크에 적당한 지점이다.

공간 디테일이 덜 중요한 에이전트(agent) 워크플로의 경우 더 공격적으로 풀링한다(6x6 -> 프레임당 16토큰).

### 네 가지 비디오 벤치마크

- VideoMME: 종합 비디오 이해, 짧은 + 중간 + 긴 길이.
- TempCompass: 세밀한 시간 추론, "이전" / "이후" 질문.
- EgoSchema: 긴 시야의 1인칭 비디오.
- Video-MMMU: 멀티모달 다분야 비디오 질문.

완전한 비디오-VLM 평가는 넷 모두를 친다. 이들은 서로 다른 축을 압박한다. TempCompass는 순서가 전부이고, EgoSchema는 3분 이상의 추론을 다루며, VideoMME는 여러 지속 시간에 걸친다.

### 그라운딩 출력 형식

시간 그라운딩을 위한 출력 형식:

- 자유 텍스트: "고양이가 4초 지점쯤에서 뛴다." 파싱하기 쉽지만 부정확하다.
- 구조화된 JSON: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL이 이것을 학습한다.
- 토큰 기반: 답과 인터리브(interleave)된 특수 `<time>4.1</time>` 토큰. Qwen2.5-VL의 내부 형식.

토큰 기반이 다운스트림 사용에 가장 정확하다. Qwen2.5-VL의 JSON 출력 형식은 곧바로 파싱된다.

### 2026년 모범 사례

2026년 비디오 VLM의 경우:

- 인코더: M-RoPE 또는 TMRoPE를 갖춘 SigLIP 2(Qwen2.5-VL).
- 프레임 샘플링: 최대 프레임 상한과 함께 동적 FPS(모션에 따라 1~4).
- 프레임당 풀링: 3x3 양선형.
- 출력: 시간 + 사건 필드를 갖춘 구조화된 JSON.
- 벤치마크: 일반용 VideoMME + TempCompass; 긴 시야용 EgoSchema.

```figure
video-temporal-patches
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다음을 포함한다.

- 균일 및 동적-FPS 프레임 샘플러.
- 장난감 시간 그라운딩 평가기: 시간 T의 "정답" 사건과 모델 출력이 주어지면, 허용 오차와 함께 정확도를 채점한다.
- Video-LLaMA(16프레임, Q-former), Video-LLaVA(8프레임, MLP), Qwen2.5-VL(동적 FPS + TMRoPE)에 걸친 비교.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-video-vlm-frame-planner.md`를 만든다. 비디오 태스크(모니터링, 동작 인식, 시간 그라운딩, 요약)가 주어지면, 프레임 샘플러, 풀링 계수, 출력 형식, 예상 정확도 등급을 고른다.

## 연습 문제 (Exercises)

1. 3분짜리 요리 시연에 대해 균일 대 동적 FPS를 고르라. 토큰 수로 정당화하라.

2. TMRoPE는 단순 시간 임베딩 테이블이 할 수 없는 무엇을 구체적으로 더하는가?

3. VLM이 내도록 학습할 수 있는 시간 그라운딩용 JSON 스키마를 작성하라. 오류 사례를 포함하라.

4. Video-LLaVA의 "투영 전 정렬(Alignment Before Projection)"에 관한 3절을 읽으라. 왜 이것이 별도의 이미지·비디오 인코더를 학습시키는 것보다 나은가?

5. VideoMME 리더보드가 주어졌을 때, 2026년 기준 최고 오픈 모델과 최고 독점 모델 사이의 격차는 얼마인가? 그 격차의 얼마가 베이스 LLM 규모 대비 시간 인코딩에 기인하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 시간 그라운딩(Temporal grounding) | "시간 국소화된 답" | VLM이 사건이 일어나는 특정 타임스탬프 범위를 출력한다 |
| TMRoPE | "Time-Multimodal RoPE" | 절대 타임스탬프를 가진 3D 회전 위치, Qwen2.5-VL이 사용 |
| 동적 FPS(Dynamic FPS) | "모션 인식 샘플링" | 고모션 구간에서 더 많은 프레임을, 정적 구간에서 더 적게 샘플링한다 |
| 프레임 풀링(Frame pooling) | "프레임당 공간 압축" | LLM 전에 양선형 보간으로 프레임당 패치를 줄인다 |
| Video Q-former | "클립 압축기" | N개 프레임을 K개 학습된 쿼리로 매핑하는 교차 어텐션 병목 |
| VideoMME | "비디오 벤치" | 종합 짧은/중간/긴 비디오 벤치마크, 2500개 이상 샘플 |

## 더 읽을거리 (Further Reading)

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
