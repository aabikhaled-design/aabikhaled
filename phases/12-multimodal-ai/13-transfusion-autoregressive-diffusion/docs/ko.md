# Transfusion: 하나의 트랜스포머에서 자기회귀 텍스트 + 디퓨전 이미지

> Chameleon과 Emu3는 이산(discrete) 토큰에 모든 것을 걸었다. 작동은 하지만 양자화(quantization) 병목이 눈에 띈다. 이미지 품질이 연속 공간 디퓨전(diffusion) 모델보다 낮은 수준에서 정체되는 것이다. Transfusion(Meta, Zhou et al., 2024년 8월)은 반대로 베팅한다. 이미지를 연속으로 유지하고, VQ-VAE를 완전히 버리며, 두 개의 손실(loss)로 하나의 트랜스포머(transformer)를 학습시킨다. 텍스트 토큰은 다음 토큰 예측(next-token-prediction)을 받고, 이미지 패치(patch)는 플로우 매칭(flow-matching) / 디퓨전 손실을 받는다. 두 목표가 같은 가중치(weight)를 최적화한다. Stable Diffusion 3의 기반 아키텍처(MMDiT)는 가까운 사촌이다. 이 레슨은 Transfusion 논제를 읽고, 장난감 수준의 두 손실 트레이너를 만들며, 하나의 트랜스포머가 두 역할을 하게 만드는 어텐션 마스크(attention mask)를 추적한다.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180분

## 학습 목표 (Learning Objectives)

- 하나의 백본(backbone)에서 두 손실(텍스트 토큰의 NTP, 이미지 패치의 디퓨전 MSE)을 돌리는 트랜스포머 배선하기.
- 이미지 패치 전반의 양방향 어텐션과 텍스트 토큰 전반의 인과(causal) 어텐션이 왜 올바른 마스크 선택인지 설명하기.
- 연산량, 품질, 코드 복잡도 측면에서 Transfusion 스타일(연속 이미지, 디퓨전 손실)을 Chameleon 스타일(이산 이미지, NTP)과 비교하기.
- MMDiT의 기여를 정리하기: 각 블록의 모달리티별 가중치, 잔차 스트림(residual stream)에서의 공동 어텐션.

## 문제 (The Problem)

이산 대 연속 이미지 토큰 논쟁은 LLM보다 오래되었다. 연속 표현(원본 픽셀, VAE 잠재값)은 세부를 보존한다. 이산 토큰(VQ 인덱스)은 트랜스포머의 네이티브 어휘에 들어맞지만 양자화 단계에서 세부를 잃는다.

Chameleon / Emu3는 이산으로 갔다. 하나의 손실, 하나의 아키텍처지만 이미지 충실도가 토크나이저(tokenizer) 품질에 따라 제한된다.

디퓨전 모델은 연속으로 갔다. 뛰어난 이미지 품질이지만 LLM과는 별개의 모델, 복잡한 노이즈 스케줄 엔지니어링, 텍스트 생성과의 깔끔한 통합 부재.

Transfusion은 둘 다 가질 수 있는지 묻는다. 이미지를 연속으로 유지하면서도 하나의 모델을 학습시키고, 하나의 그래디언트(gradient) 스텝에 두 손실을 꿰매어 쓴다.

## 개념 (The Concept)

### 두 손실 아키텍처

단일 디코더 전용(decoder-only) 트랜스포머가 다음을 담은 시퀀스를 처리한다.

- 텍스트 토큰(이산, BPE 어휘에서).
- 이미지 패치(연속, 선형 임베딩(embedding)으로 은닉 차원에 투영된 16x16 픽셀 블록: ViT 인코더의 입력과 동일).
- 연속 패치가 위치하는 곳을 표시하는 `<image>`와 `</image>` 태그.

순방향 패스(forward pass)는 한 번 돈다. 손실은 토큰마다 두 헤드 중 하나를 고른다.

- 텍스트 토큰의 경우: 어휘 로짓(vocab-logits) 헤드에 대한 표준 교차 엔트로피.
- 이미지 패치의 경우: 연속 패치에 대한 디퓨전 손실: 각 패치에 더해진 노이즈를 예측한다.

그래디언트는 공유 트랜스포머 본체를 통해 흐른다. 두 손실이 공유 가중치를 동시에 개선한다.

### 어텐션 마스크: 인과 텍스트 + 양방향 이미지

텍스트 토큰은 인과여야 한다. 텍스트 토큰이 미래 텍스트에 어텐션하게 두면 교사 강요(teacher forcing)가 깨지기 때문이다. 그러나 이미지 패치는 하나의 스냅샷을 나타내므로, 같은 이미지 블록 안에서는 서로 양방향으로 어텐션해야 한다.

마스크:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

학습과 추론(inference) 시 블록 삼각(block-triangular) 마스크로 구현된다.

### 트랜스포머 내부의 디퓨전 손실

디퓨전 손실은 표준이다. 이미지 패치에 노이즈를 더하고, 모델에게 노이즈(또는 동등하게 깨끗한 패치)를 예측하도록 요청한다. Transfusion의 버전은 플로우 매칭(flow matching)을 쓰며, 노이즈에서 깨끗한 상태로의 속도장(velocity field)을 예측한다.

학습 중:
1. 각 이미지 패치 x0에 대해 임의의 타임스텝 t를 샘플링한다.
2. 노이즈 ε를 샘플링하고, xt = (1-t) * x0 + t * ε를 계산한다(플로우 매칭을 위한 선형 보간).
3. 트랜스포머가 v_theta(xt, t)를 예측한다; loss = MSE(v_theta(xt, t), ε - x0).
4. 같은 시퀀스의 텍스트 NTP 손실과 나란히 역전파(backprop)한다.

추론 시 생성은:
- 텍스트 토큰: 표준 자기회귀 샘플링(sampling).
- 이미지 패치: 앞선 텍스트 토큰에 조건화된 디퓨전 샘플링 루프(보통 10~30 스텝).

### MMDiT: Stable Diffusion 3의 변형

Stable Diffusion 3(Esser et al., 2024년 3월)는 Transfusion과 거의 같은 시기에 MMDiT(Multimodal Diffusion Transformer)를 출시했다. 두 아키텍처는 형제다.

MMDiT의 주요 차이점:

- 블록별 모달리티 가중치. 각 트랜스포머 블록은 텍스트 토큰 대 이미지 패치에 대해 별도의 Q, K, V, MLP 가중치를 갖는다. 어텐션은 공동(교차 모달)이고, 나머지는 모두 모달리티별이다.
- 정류 플로우(rectified flow) 학습. 알려진 샘플링과 DDPM보다 단순한 수식을 가진 특정 플로우 매칭 변형.
- 규모. MMDiT는 SD3(2B, 8B 파라미터 변형)의 백본이다. Transfusion 논문은 7B까지 확장한다.

둘 다 동일한 핵심 아이디어로 수렴한다. 하나의 트랜스포머가 텍스트에는 NTP를, 연속 이미지 표현에는 디퓨전을 돌린다.

### 왜 Chameleon 스타일을 이기는가

이미지 생성에서 연속 디퓨전과 이산 NTP 사이의 품질 격차는 측정 가능하다. Transfusion 논문은 다음을 보고한다.

- 7B 파라미터에서, 같은 크기의 Chameleon 스타일 모델을 FID에서 3~5점 이긴다.
- 토크나이저 학습이 필요 없다. 이미지 인코더(encoder)가 더 단순하다(은닉으로의 선형 투영, ViT의 입력 층과 동일).
- 자기회귀 이미지 토큰과 달리, 추론에서 이미지 패치 디노이징을 병렬화할 수 있다.

단점도 있다. Transfusion은 이중 손실 모델이라 학습 동역학이 더 까다롭고, 손실 가중치를 조정해야 한다. NTP와 디퓨전 사이의 스케줄 불일치가 한 헤드를 지배하게 만들기도 한다.

### 다운스트림에 있는 것

Janus-Pro(레슨 12.15)는 이해와 생성을 위해 비전 인코더를 둘(하나는 SigLIP, 다른 하나는 VQ)로 분리하면서도 트랜스포머 본체를 공유하여 Transfusion의 아이디어를 정제한다. Show-o(레슨 12.14)는 디퓨전을 이산 디퓨전(마스킹 예측)으로 교체한다. 통합 생성 계열은 Transfusion 이후 빠르게 갈라진다.

이미지를 내는 2026년 프로덕션(production) VLM, 곧 Gemini 3 Pro, GPT-5, Claude Opus 4.7의 이미지 생성 경로는 거의 확실히 이 계열의 어떤 후손을 쓴다. 세부 사항은 독점이다.

```figure
cfg-guidance-scale
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 아주 작은 MNIST 유사 문제에서 장난감 Transfusion을 만든다.

- 텍스트 캡션은 숫자(0~9)를 묘사하는 짧은 정수 시퀀스다.
- 이미지는 4x4 바이트 격자다.
- 한 쌍의 공유 가중치 선형 투영이 트랜스포머 대역으로 작동한다; 텍스트에는 NTP 손실, 노이즈 패치에는 MSE 손실.
- 학습 루프가 두 손실을 번갈아 한다. 어텐션 마스크는 명시적이다.
- 생성은 단일 순방향 패스에서 텍스트 캡션과 4x4 이미지를 만든다.

트랜스포머는 장난감이다. 두 손실 배관, 어텐션 마스크 구성, 추론 루프가 진짜 산출물이다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-two-loss-trainer-designer.md`를 만든다. 새 멀티모달 학습 태스크(텍스트 + 이미지, 텍스트 + 오디오, 텍스트 + 비디오)가 주어지면, 두 손실 스케줄(손실 가중치, 마스크 형태, 공유 대 모달리티별 블록)을 설계하고 구현 위험을 표시한다.

## 연습 문제 (Exercises)

1. Transfusion 스타일 모델이 텍스트 토큰 70%와 이미지 패치 30%를 학습한다. 이미지 디퓨전 손실은 크기 면에서 텍스트 NTP 손실의 약 10배다. 어떤 손실 가중치가 둘을 균형 맞추는가?

2. 시퀀스 `[T, T, <image>, P, P, P, P, </image>, T]`에 대한 블록 삼각 마스크를 구현하라. 각 항목을 0 또는 1로 표시하라.

3. MMDiT는 모달리티별 QKV 가중치를 갖는다. Transfusion의 완전 공유 트랜스포머 대비 이것이 더하는 파라미터 수 오버헤드는 얼마인가? 7B 파라미터에서 그럴 가치가 있는가?

4. 생성: 텍스트 프롬프트가 주어지면, 모델은 50개 토큰에 대해 NTP를 돌리고, 그다음 `<image>`에 도달하여, 20번의 디노이즈 스텝에 걸쳐 256개 패치에 디퓨전을 돌린다. 총 몇 번의 순방향 패스인가?

5. SD3 논문 3절을 읽으라. 정류 플로우를 기술하고 왜 DDPM보다 더 적은 추론 스텝으로 수렴하는지 설명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| 두 손실 학습(Two-loss training) | "NTP + 디퓨전" | 단일 트랜스포머가 같은 그래디언트 스텝에서 텍스트 토큰의 교차 엔트로피와 연속 이미지 패치의 MSE를 모두 최적화한다 |
| 플로우 매칭(Flow matching) | "정류 플로우" | 노이즈에서 깨끗한 데이터로의 속도장을 예측하는 디퓨전 변형; DDPM보다 단순한 수식 |
| MMDiT | "멀티모달 DiT" | Stable Diffusion 3의 아키텍처: 공동 어텐션, 모달리티별 MLP와 norm |
| 블록 삼각 마스크(Block-triangular mask) | "인과 텍스트 + 양방향 이미지" | 텍스트 전반에는 인과적이지만 이미지 영역 안에서는 양방향인 어텐션 마스크 |
| 연속 이미지 표현(Continuous image representation) | "VQ 없음" | 정수 코드북 인덱스가 아니라 실수값 벡터로서의 이미지 패치 |
| 속도 예측(Velocity prediction) | "v-파라미터화" | 네트워크 출력이 노이즈 자체가 아니라 노이즈와 데이터 사이의 속도장이다 |

## 더 읽을거리 (Further Reading)

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
