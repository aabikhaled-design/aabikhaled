# Janus-Pro: 통합 멀티모달 모델을 위한 분리 인코더(Decoupled Encoders)

> 통합 멀티모달 모델에는 피할 수 없는 긴장이 있다. 이해(understanding)는 의미 특성(semantic feature)을 원한다. SigLIP이나 DINOv2는 개념 수준 정보가 풍부한 벡터를 낸다. 생성(generation)은 재구성에 친화적인 코드를 원한다. 다시 선명한 픽셀로 합쳐지는 VQ 토큰이 그것이다. 두 목표는 단일 인코더(encoder)에서 양립하지 않는다. Janus(DeepSeek, 2024년 10월)와 Janus-Pro(DeepSeek, 2025년 1월)는 그 해법이 시도를 멈추는 것이라 주장한다. 두 인코더를 분리하라. 태스크 간에 트랜스포머(transformer) 본체를 공유하되, 이해는 SigLIP으로, 생성은 VQ 토크나이저(tokenizer)로 라우팅한다. 7B에서 Janus-Pro는 GenEval에서 DALL-E 3을 이기면서 MMMU에서 LLaVA와 대등하다. 이 레슨은 하나로는 실패하는 곳에서 왜 두 인코더가 작동하는지 읽는다.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 단일 공유 인코더가 왜 이해 또는 생성 품질 중 하나를 타협하는지 설명하기.
- Janus-Pro의 라우팅을 기술하기: 이해를 위해 입력 측의 SigLIP 특성, 생성을 위해 입력과 출력 양쪽의 VQ 토큰.
- Janus가 못한 곳에서 Janus-Pro를 성공시키는 데이터 혼합 스케일링을 추적하기.
- 분리형(Janus-Pro), 결합-연속형(Transfusion), 결합-이산형(Show-o) 아키텍처를 비교하기.

## 문제 (The Problem)

통합 모델은 이해와 생성에 걸쳐 트랜스포머 본체를 공유한다. 이전 시도들(Chameleon, Show-o, Transfusion)은 모두 양방향에 하나의 시각 토크나이저를 쓴다. 그 토크나이저는 타협이다.

- 재구성(생성)에 최적화: VQ-VAE는 세밀한 픽셀 디테일을 잡지만 의미 일관성이 약한 토큰을 만든다.
- 의미(이해)에 최적화: SigLIP 임베딩(embedding)은 "고양이" 이미지를 "고양이" 토큰 가까이에 모으지만 좋은 재구성을 허용하지 않는다.

Show-o와 Transfusion은 한 방향에 눈에 띄는 품질 세금을 내고 이를 감당한다. Janus-Pro는 묻는다. 태스크의 필요가 다른데 왜 하나의 토크나이저를 요구하는가?

## 개념 (The Concept)

### 분리 시각 인코딩

Janus-Pro의 아키텍처는 두 인코더를 분리한다.

- 이해 경로. 입력 이미지 → SigLIP-SO400m → 2층 MLP → 트랜스포머 본체.
- 생성 경로. 입력 이미지(기존 이미지에 조건화할 경우) → VQ 토크나이저 → 토큰 ID → 트랜스포머 본체.
- 출력 생성. 트랜스포머가 예측한 이미지 토큰 → VQ 디코더(decoder) → 픽셀.

트랜스포머 본체는 공유된다. 본체의 업스트림과 다운스트림 모든 것은 태스크별이다.

입력은 프롬프트(prompt) 형식으로 모호함이 해소된다. `<understand>` 태그는 SigLIP으로 라우팅하고, `<generate>`는 VQ로 라우팅한다. 또는 라우팅이 태스크로부터 암묵적으로 정해진다.

### 왜 작동하는가

이해 손실(loss)은 SigLIP 특성을 받는데, 이는 CLIP 스타일 사전 학습(pretraining)이 의미 유사도에 맞춰 조정한 것이다. 입력 특성이 태스크에 더 좋기 때문에 모델의 지각 벤치마크(benchmark)가 Show-o / Transfusion 대비 개선된다.

생성 손실은 VQ 토큰을 받는데, 이는 토크나이저가 재구성에 맞춰 조정한 것이다. VQ 코드가 깔끔하게 픽셀로 합쳐지기 때문에 이미지 품질이 Show-o 대비 개선된다.

공유 트랜스포머 본체는 두 입력 분포(SigLIP과 VQ)를 보고 둘 다와 작동하는 법을 배운다. 주장: 충분한 데이터와 충분한 파라미터(parameter)면 본체가 전환을 흡수한다.

### 데이터 스케일링: Janus 대 Janus-Pro

Janus(원본, arXiv 2410.13848)는 분리를 도입했지만 작은 규모(1.3B 파라미터, 제한된 데이터)였다. Janus-Pro(arXiv 2501.17811)는 확장했다.

- 7B 파라미터(1.3B 대비).
- 1단계(정렬)용 9000만 이미지-텍스트 쌍, 7200만에서 증가.
- 2단계(통합)용 7200만, 2600만에서 증가.
- 3단계용 20만 이미지 생성 인스트럭션 샘플 추가.

요점: Janus-Pro-7B는 MMMU에서 LLaVA와 대등하고(60.3 대 약 58), GenEval에서 DALL-E 3을 이긴다(0.80 대 0.67). 통합 스펙트럼의 양쪽 모두에서 경쟁력 있는 하나의 오픈 모델이다.

### JanusFlow: 정류 플로우 변형

JanusFlow(arXiv 2411.07975)는 VQ 생성 경로를 정류 플로우(rectified-flow) 생성 경로(연속)로 교체한다. 분할이 이해용 SigLIP과 생성용 정류 플로우가 된다. 품질 한계가 더 올라간다. 아키텍처는 분리 인코더-공유 본체로 남는다.

### 공유 본체의 역할

트랜스포머 본체는 두 입력 분포를 가진 통합 시퀀스를 처리한다. 그 역할은:

- 이해를 위해: SigLIP 특성과 텍스트 토큰을 소비해 텍스트를 자기회귀(autoregressive)로 낸다.
- 생성을 위해: 텍스트 토큰과 (선택적 이미지 VQ 토큰)을 소비해 이미지 VQ 토큰을 자기회귀로 낸다.

본체에는 블록별 모달리티 가중치(weight)가 없다. Qwen이나 Llama 안에서 보게 될 텍스트 스타일 트랜스포머에 두 입력 어댑터를 더한 것이다.

흥미롭게도 이는 Janus-Pro의 본체가 사전 학습된 LLM에서 초기화될 수 있음을 뜻한다. Janus-Pro는 실제로 DeepSeek-MoE-7B에서 초기화한다. 그 선택이 중요하다. LLM은 순수하게 밑바닥부터 만든 통합 모델이 도달하기 어려운 추론 능력을 기여한다.

### InternVL-U와 비교

InternVL-U(레슨 12.10)는 2026년 후속작이다. 다음을 결합한다.

- 네이티브 멀티모달 사전 학습(InternVL3 백본(backbone)).
- 분리 인코더 라우팅(SigLIP 입력, VQ + 디퓨전 헤드 출력).
- 통합 이해 + 생성 + 편집.

InternVL-U는 Janus-Pro의 아키텍처 선택을 더 큰 프레임워크 안에 포섭한다. 분리 인코더 아이디어는 이제 대규모 통합 모델의 기본이다.

### 한계

분리 인코더는 아키텍처 복잡도를 더한다. 학습할 토크나이저 둘, 유지할 입력 경로 둘, 실패 모드 집합 둘. 생성이 필요 없는 제품에는 Janus-Pro가 과설계다. LLaVA 계열 이해 모델을 골라라.

이해가 필요 없는 제품에는 Janus-Pro가 과잉 자격이다. Stable Diffusion 3 / Flux 모델을 골라라.

둘 다 필요한 제품에는 Janus-Pro가 이제 참조 오픈 아키텍처다.

```figure
l5-janus-decouple
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 Janus-Pro 라우팅을 시뮬레이션한다.

- 두 개의 가짜 인코더: SigLIP 유사(256차원 의미 벡터 생성)와 VQ 유사(정수 코드 생성).
- 태스크 태그에 기반해 인코더를 고르는 프롬프트 라우터.
- 어느 인코더가 생성했든 토큰 시퀀스를 처리하는 공유 본체(대역).
- 1단계(정렬)에서 3단계(인스트럭션 튜닝)로의 가중 샘플 스케줄 전환.

3가지 예시에 대해 라우팅된 경로를 출력한다: 이미지 QA, T2I, 이미지 편집.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-decoupled-encoder-picker.md`를 만든다. 프런티어급 품질로 통합 생성 + 이해를 원하는 제품이 주어지면, 구체적인 데이터 규모 권고와 함께 Janus-Pro, JanusFlow, 또는 InternVL-U를 고른다.

## 연습 문제 (Exercises)

1. Janus-Pro-7B는 GenEval에서 DALL-E 3을 이긴다. 왜 7B 오픈 모델이 생성에서는 프런티어 독점 모델과 대등할 수 있지만 이해에서는 그렇지 못한지 설명하라.

2. 라우터 함수를 구현하라: 프롬프트 텍스트가 주어지면 `understand` 또는 `generate`로 분류한다. "묘사하고 나서 스케치하라" 같은 모호한 프롬프트는 어떻게 처리하는가?

3. JanusFlow는 VQ 경로를 정류 플로우로 교체한다. 이제 트랜스포머 본체는 무엇을 출력하며, 손실에서는 무엇이 바뀌는가?

4. Janus-Pro 아키텍처가 분리 인코더 하나를 더 두어 다룰 수 있는 네 번째 태스크를 제안하라. 예: 이미지 분할(DINO 스타일), 깊이(MiDaS 스타일).

5. 데이터 스케일링에 관한 Janus-Pro 4.2절을 읽으라. Janus 대비 T2I 품질 향상에 어느 데이터 단계가 가장 크게 기여하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 분리 인코딩(Decoupled encoding) | "두 개의 시각 인코더" | 방향별 별도 토크나이저 또는 인코더: 이해에는 의미용, 생성에는 재구성용 |
| 공유 본체(Shared body) | "하나의 트랜스포머" | 단일 트랜스포머가 어느 인코더의 출력이든 처리한다; 모달리티별 가중치 없음 |
| 이해용 SigLIP(SigLIP for understanding) | "의미 특성" | 풍부한 개념 특성을 제공하지만 재구성은 빈약한 CLIP 계열 비전 타워 |
| 생성용 VQ(VQ for generation) | "재구성 코드" | 깔끔하게 픽셀로 디코딩되는 벡터 양자화 토큰 |
| JanusFlow | "정류 플로우 변형" | VQ 대신 연속 플로우 매칭 생성 헤드를 가진 Janus-Pro |
| 라우팅 태그(Routing tag) | "태스크 태그" | 입력 인코더를 고르는 프롬프트 표시(`<understand>` / `<generate>`) |

## 더 읽을거리 (Further Reading)

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
