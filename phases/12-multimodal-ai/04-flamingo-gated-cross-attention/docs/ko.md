# Flamingo와 퓨샷 VLM을 위한 게이트 교차 어텐션(Flamingo and Gated Cross-Attention for Few-Shot VLMs)

> DeepMind의 Flamingo(2022)는 누구보다 먼저 두 가지를 해냈다. 단일 모델이 이미지, 비디오, 텍스트가 임의로 인터리브(interleave)된 시퀀스를 처리할 수 있음을 보였다. 그리고 VLM이 인컨텍스트(in-context)로 학습할 수 있음을 보였다. 세 개의 예시 (이미지, 캡션) 쌍이 담긴 퓨샷(few-shot) 프롬프트(prompt)를 주면, 모델은 어떤 그래디언트(gradient) 스텝도 없이 새 이미지를 캡션한다. 그 메커니즘은 게이트 교차 어텐션(gated cross-attention) 층으로, 동결된 LLM의 기존 층 사이에 삽입되며, 0에서 시작하는 학습되는 tanh 게이트(gate)를 가져 LLM의 텍스트 능력이 초기화 시점에 보존되도록 한다. 이 레슨은 Flamingo의 퍼시버 리샘플러(Perceiver resampler)와 게이트 교차 어텐션 아키텍처를 짚어 본다. Gemini의 인터리브 입력과 Idefics2의 시각 토큰(visual token)의 조상이다.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120분

## 학습 목표 (Learning Objectives)

- 게이트 교차 어텐션이 tanh(gate) = 0을 통해 초기화 시점에 동결된 LLM의 텍스트 능력을 어떻게 보존하는지 설명하기.
- 퍼시버 리샘플러 따라가기: N개 이미지 패치 → 교차 어텐션을 통해 K개의 고정된 "잠재(latent)" 쿼리(query).
- Flamingo가 이미지 배치를 존중하는 인과(causal) 마스킹으로 인터리브된 이미지-텍스트 시퀀스를 어떻게 처리하는지 서술하기.
- 퓨샷 멀티모달(multimodal) 프롬프트 구조 재현하기(3개의 이미지-캡션 예시 다음 쿼리 이미지).

## 문제 (The Problem)

BLIP-2는 32개의 시각 토큰을 동결된 LLM의 입력 층에 먹인다. 프롬프트당 이미지 하나에는 동작한다. 하지만 "여기 이미지 A가 있다, 캡션하라, 여기 이미지 B가 있다, 캡션하라, 이제 여기 이미지 C가 있다, 캡션하라"처럼 텍스트와 인터리브된 *많은* 이미지를 먹이고 싶다면? LLM의 셀프 어텐션(self-attention)은 단일 스트림에서 이미지 토큰과 텍스트 토큰을 처리해야 하는데, 어떤 위치가 어떤 이미지에 주목할 수 있는지가 까다로운 문제가 된다.

Flamingo의 답: LLM의 입력 스트림을 전혀 바꾸지 말라. 기존 LLM 블록 사이에 추가 교차 어텐션 층을 삽입하라. 텍스트 토큰은 여전히 늘 그렇듯 LLM의 인과 셀프 어텐션을 통해 흐른다. 몇 개의 LLM 블록마다 한 번씩, 텍스트 토큰은 또한 새로운 게이트 층을 통해 이미지 특성에 교차 어텐션을 한다. (0으로 초기화된) 게이트는 스텝 0에서 새 층들이 항등 연산(no-op)임을 의미한다. 모델은 정확히 사전 학습된 LLM처럼 행동한다. 학습이 진행되면서 게이트가 열리고 시각 정보가 흐르기 시작한다.

Flamingo가 답한 두 번째 질문: 프롬프트당 가변 개수의 이미지(0, 1, 또는 다수)를 어떻게 처리하는가? 퍼시버 리샘플러: 패치 개수가 몇 개든 받아서 고정 개수의 시각 잠재 토큰을 만드는 작은 교차 어텐션 모듈이다. LLM 교차 어텐션 층은 프롬프트에 이미지가 몇 개든 상관없이 같은 형상을 본다.

## 개념 (The Concept)

### 동결된 LLM (The frozen LLM)

Flamingo는 동결된 Chinchilla 70B LLM으로 시작한다. 700억 가중치(weight) 전부 건드리지 않는다. 기존 텍스트 셀프 어텐션과 FFN은 정상적으로 동작한다.

### 퍼시버 리샘플러 (Perceiver resampler)

프롬프트의 각 이미지에 대해, ViT는 N개의 패치 토큰을 만든다. 퍼시버 리샘플러는 K개의 고정된 학습 가능한 잠재(latent)를 가진다(Flamingo는 K=64를 쓴다). 각 리샘플러 블록은 두 개의 하위 단계다:

1. 교차 어텐션: K개의 잠재가 N개의 패치 토큰에 주목한다(Q는 잠재에서, K/V는 패치에서).
2. 잠재 내에서의 셀프 어텐션 + FFN.

6개의 리샘플러 블록 이후, 출력은 ViT가 패치를 몇 개 만들었든 상관없이 차원 1024의 K=64개 시각 토큰이다. 224x224 이미지(패치 196개)와 480x480 이미지(패치 900개)는 둘 다 64개의 리샘플러 토큰으로 나온다.

비디오의 경우, 리샘플러는 시간 축으로 적용된다. 각 프레임의 패치가 64개 잠재를 만들고, 시간적 위치 인코딩(temporal positional encoding)이 모델로 하여금 t=0과 t=N을 구분하게 한다. 전체 비디오는 T * 64개의 시각 토큰이 된다.

### 게이트 교차 어텐션 (Gated cross-attention)

동결된 LLM의 M개 층마다 한 번씩(Flamingo는 M=4를 쓴다), 새로운 게이트 교차 어텐션 블록을 삽입한다:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha`는 0으로 초기화된 학습 가능한 스칼라다.
- `tanh(0) = 0`이므로 초기에는 게이트 분기가 0을 기여한다.
- `alpha`가 0에서 멀어지면서, 교차 어텐션 기여가 매끄럽게 커진다.
- 잔차 연결(residual connection)은 완전히 열린 게이트조차 LLM의 텍스트 표현을 덮어쓰지 않음을 의미한다. 그저 그 위에 시각 정보를 더할 뿐이다.

이것이 Flamingo에서 가장 중요한 단 하나의 설계 선택이다. 시각적 조건화는 덧셈적이고, 게이트가 걸려 있으며, 초기화 시점에는 0이다. 스텝 0의 Flamingo는 텍스트 전용 입력에 대해 완벽한 Chinchilla 70B다.

### 인터리브 입력을 위한 마스킹 교차 어텐션 (Masked cross-attention for interleaved inputs)

"<image A> caption A <image B> caption B <image C> ?" 같은 프롬프트에서, 각 텍스트 토큰은 시퀀스에서 자신보다 앞에 온 이미지만 봐야 한다. 교차 어텐션 마스크는 다음을 강제한다: 위치 `t`의 텍스트 토큰은 이미지 인덱스가 `i < i_t`인 이미지 리샘플러 토큰에만 주목하며, 여기서 `i_t`는 위치 `t` 앞의 가장 최근 이미지다. "바로 직전 이미지만 본다" 또는 "앞선 모든 이미지를 본다"는 둘 다 유효한 선택이다. Flamingo는 전자를 택했다.

### 인컨텍스트 퓨샷 학습 (In-context few-shot learning)

Flamingo 프롬프트는 이렇게 생겼다:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

모델은 완성 패턴을 보고 "bird"(또는 image3가 보여 주는 무엇이든)를 출력한다. 그래디언트 스텝 없음. 동결된 LLM의 인컨텍스트 학습 능력이 게이트 교차 어텐션을 통해 이어진다. 이것이 논문의 핵심이며 그것이 중요한 이유다.

### 학습 데이터 (Training data)

Flamingo는 세 가지 데이터셋(dataset)으로 학습했다:

1. MultiModal MassiveWeb (M3W): 읽기 순서를 재구성한, 이미지와 텍스트가 인터리브된 4300만 개의 웹 페이지.
2. Image-Text Pairs (ALIGN + LTIP): 44억 쌍.
3. Video-Text Pairs (VTP): 2700만 개의 짧은 비디오 클립.

OBELICS(2023)는 인터리브된 웹 코퍼스의 오픈 재현으로, Idefics, Idefics2, 그리고 대부분의 오픈 "Flamingo 류" 모델이 이것으로 학습한다.

### OpenFlamingo와 Otter (OpenFlamingo and Otter)

OpenFlamingo(2023)는 오픈 재현이다. 아키텍처는 동일하다(동결된 LLaMA 또는 MPT 위의 퍼시버 리샘플러 + 게이트 교차 어텐션). 3B, 4B, 9B 체크포인트. 더 작은 베이스 LLM과 더 적은 데이터 때문에 품질이 Flamingo에 뒤진다.

Otter(2023)는 MIMIC-IT(멀티모달 명령어 데이터셋)에 대한 명령어 튜닝(instruction tuning)으로 OpenFlamingo 위에 쌓아, 게이트 교차 어텐션이 명령어 따르기에도 동작함을 보였다.

### 후손들 (The descendants)

- Idefics / Idefics2 / Idefics3: Hugging Face의 게이트 교차 어텐션 혈통, 점진적으로 더 단순해졌다(Idefics2는 적응형 풀링(adaptive pooling)을 쓰는 직접 패치 토큰을 위해 리샘플러를 버렸다).
- Flamingo에서 Chameleon으로의 전환: 2024년 무렵 많은 팀이 조기 융합(early-fusion)(Lesson 12.11)으로 옮겨 갔다. Flamingo 스타일 게이트 교차 어텐션은 백본 동결이 필요한 곳에서 프로덕션에 남는다.
- Gemini의 인터리브 입력: 개념적으로 Flamingo의 인터리브 포맷 유연성을 물려받았다. 다만 정확한 메커니즘은 비공개다.

### BLIP-2와의 비교 (Comparison to BLIP-2)

| | BLIP-2 | Flamingo |
|---|---|---|
| 시각 다리 | 입력에서 한 번의 Q-Former | M개 층마다의 게이트 교차 어텐션 |
| 시각 토큰 | 이미지당 32개 | 교차 어텐션 층당, 이미지당 64개 |
| 동결 LLM | 예 | 예 |
| 퓨샷 인컨텍스트 | 약함 | 강함: 논문의 핵심 |
| 인터리브 입력 | 네이티브 지원 없음 | 예, 설계 목표 |
| 학습 데이터 | 1억 3000만 쌍 | 13억 쌍 + 4300만 인터리브 페이지 |
| 파라미터(parameter) 개수 | 학습 1억 8800만 | 학습 약 100억(교차 어텐션 층) |
| 연산 | 8개 A100에서 며칠 | 수천 개 TPUv4에서 몇 주 |

예산이 빠듯한 단일 이미지 VQA에는 BLIP-2를 골라라. 인터리브, 퓨샷, 또는 멀티 이미지 추론에는 Flamingo/Idefics2를 골라라.

```figure
cross-attention-fusion
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다음을 보여 준다:

1. 8개의 학습 가능한 잠재로 36개의 가짜 패치 토큰에 대한 퍼시버 리샘플러(순수 Python 교차 어텐션).
2. `alpha = 0`인 게이트 교차 어텐션 스텝 → 출력이 입력과 같음(LLM 변하지 않음), 그다음 `alpha = 2.0` → 시각 기여가 섞여 들어감.
3. "(image 1) (text 1) (image 2) (text 2)" 시퀀스에 대한 2D 어텐션 마스크를 만드는 인터리브 마스크 빌더.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-gated-bridge-diagnostic.md`를 만든다. 오픈 VLM의 설정(리샘플러 Y/N, 교차 어텐션 빈도, 게이트 방식)이 주어지면, Flamingo 혈통 요소를 식별하고 동결 전략을 설명한다. 파인튜닝(fine-tuning)이 왜 텍스트 성능을 떨어뜨렸는지 디버깅하는 데 유용하다(답: 게이트가 너무 빨리 너무 넓게 열렸다).

## 연습 문제 (Exercises)

1. Flamingo-9B의 시각 파라미터 개수를 계산하라: 9B LLM + 14억 게이트 교차 어텐션 층 + 6400만 리샘플러. 전체 파라미터의 어느 비율이 학습되는가?

2. 게이트 잔차 `y = tanh(alpha) * cross + x`를 PyTorch로 구현하라. `alpha=0`일 때 초기에 `y==x`임을 실험적으로 보여라.

3. 각 프롬프트가 서로 다른 이미지 개수를 가질 때 배치에서 다중 이미지를 어떻게 처리하는지에 대한 OpenFlamingo Section 3.2(arXiv:2308.01390)를 읽어라. 패딩(padding) 전략을 서술하라.

4. Flamingo의 교차 어텐션 마스크는 왜 텍스트 토큰이 앞선 모든 이미지가 아니라 *가장 최근의* 앞선 이미지에만 주목하게 하는가? Flamingo 논문 Section 2.4를 읽고 트레이드오프(trade-off)를 설명하라.

5. 인컨텍스트 퓨샷: 새 Flamingo 변형에 대해 "이미지 → 주요 객체의 색"의 예시 4개로 프롬프트를 구성하라. 예시 개수를 0에서 8까지 바꿀 때 예상되는 정확도 패턴을 서술하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 퍼시버 리샘플러(Perceiver resampler) | "고정 잠재 교차 어텐션" | 가변 개수의 입력 패치로부터 K개의 고정 토큰을 만드는 모듈 |
| 게이트 교차 어텐션(Gated cross-attention) | "Tanh 게이트가 걸린 다리" | 잔차 층 `y = tanh(alpha)*cross + x`, 학습 가능한 alpha, 초기값 0 |
| 인터리브 입력(Interleaved input) | "섞인 시퀀스" | 이미지와 텍스트가 읽기 순서대로 자유롭게 섞인 프롬프트 포맷 |
| 동결 LLM(Frozen LLM) | "LLM 그래디언트 없음" | 텍스트 LLM의 가중치는 갱신되지 않는다. 리샘플러 + 교차 어텐션 층만 학습한다 |
| 퓨샷(Few-shot) | "인컨텍스트 예시" | 프롬프트에 몇 개의 (이미지, 답) 쌍을 준다. 모델은 파인튜닝 없이 일반화한다 |
| OBELICS | "인터리브 웹 코퍼스" | 이미지와 텍스트가 읽기 순서로 담긴 1억 4100만 웹 페이지의 오픈 데이터셋 |
| Chinchilla | "70B 동결 베이스" | Flamingo의 동결된 텍스트 LLM, DeepMind의 Chinchilla 논문에서 나옴 |
| 게이트 스케줄(Gate schedule) | "alpha가 어떻게 움직이는가" | 학습 중 교차 어텐션 게이트가 열리는 속도 |
| 교차 어텐션 빈도(Cross-attn frequency) | "M개 층마다" | 게이트 교차 어텐션 블록이 얼마나 자주 삽입되는가. Flamingo는 M=4를 쓴다 |
| OpenFlamingo | "오픈 재현" | 3-9B의 MosaicML/LAION 오픈 체크포인트. Flamingo와 아키텍처 동일 |

## 더 읽을거리 (Further Reading)

- [Alayrac et al.: Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) 원논문.
- [Awadalla et al.: OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) 오픈 재현.
- [Laurençon et al.: OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) 인터리브 웹 코퍼스.
- [Jaegle et al.: Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) 일반 퍼시버 아키텍처.
- [Li et al.: Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) 명령어 튜닝된 Flamingo 후손.
- [Laurençon et al.: Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) Flamingo 접근법의 현대적 단순화.
