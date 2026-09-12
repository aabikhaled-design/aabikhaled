# 시각적 자기회귀 모델링(Visual Autoregressive Modeling, VAR): 다음 스케일 예측

> 확산 모델(diffusion model)은 시간 축에서 반복적으로 샘플링한다(노이즈 제거 스텝). VAR은 스케일(scale) 축에서 반복적으로 샘플링한다. 1x1 토큰(token)을 예측하고, 그다음 2x2, 그다음 4x4, 최종 해상도까지 예측하며, 각 스케일은 이전 스케일에 조건부로 의존한다. 2024년 논문은 VAR이 이미지 생성에서 GPT 스타일의 스케일링 법칙(scaling law)을 따르며, 동일한 컴퓨트 예산에서 DiT를 능가함을 보였다. 이 레슨은 그 핵심 메커니즘을 직접 만든다.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90분

## 문제 (The Problem)

자기회귀(autoregressive) 생성은 언어 모델링을 지배했다. 예측 가능하게 확장되기 때문이다. 컴퓨트가 늘고 파라미터(parameter)가 늘면 퍼플렉서티(perplexity)가 낮아지고 출력이 좋아진다. 이미지 생성은 2024년 이전에 주요 자기회귀 시도가 두 가지 있었다. PixelRNN/PixelCNN(픽셀 단위)과 DALL-E 1 / Parti / MuseGAN(VQ-VAE 코드에 대한 토큰 단위)이다.

둘 다 생성 순서(generation-order) 문제를 겪었다. 픽셀과 토큰은 2D 격자(grid)에 배열되어 있지만, 자기회귀 모델은 이를 1D 래스터 순서(raster order)로 방문해야 한다. 초기 모서리 픽셀은 이미지가 결국 무엇이 될지 전혀 알지 못한다. 생성 품질은 GPT 텍스트만큼 잘 확장되지 못했고, 동일 컴퓨트에서 확산 모델 품질에 도달한 적도 없었다.

VAR은 생성되는 대상을 바꿈으로써 생성 순서 문제를 해결한다. 이미지 토큰을 공간상에서 하나씩 예측하는 대신, VAR은 점점 높아지는 해상도로 전체 이미지를 예측한다. 스텝 1: 1x1 토큰을 예측한다(전체 이미지의 "요약"). 스텝 2: 2x2 토큰 격자를 예측한다(거친 특성). 스텝 3: 4x4 격자를 예측한다. 스텝 K: 최종 (H/8)x(W/8) 격자를 예측한다.

각 스케일은 이전 모든 스케일에 어텐션(attention)하며("스케일 순서"에서 인과적으로), 자신의 스케일 안에서는 병렬로 처리된다. 순서 문제는 사라진다. 스케일 k에서의 전체 이미지는 하나의 트랜스포머(transformer) 패스(pass)로 생성된다.

## 개념 (The Concept)

### VQ-VAE 다중 스케일 토크나이저 (Multi-Scale Tokenizer)

VAR은 **다중 스케일 이산 토크나이저(multi-scale discrete tokenizer)**가 필요하다. 이미지 x에 대해, 점진적으로 해상도가 높아지는 토큰 격자의 시퀀스를 생성한다.

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

각 z_k는 동일한 코드북(codebook)을 사용한다(일반적으로 크기 4096-16384). 각 스케일에서의 토큰화는 독립적이지 않다. 각 스케일의 잔차(residual)를 합산하면 f가 재구성되도록 학습된다.

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

이것은 **잔차 VQ(residual VQ)** 변형이다. 스케일 k는 스케일 1..k-1이 놓친 것을 포착한다. 디코더(decoder)는 모든 스케일 임베딩(embedding)의 합을 받아 이미지를 생성한다.

다중 스케일 VQ 토크나이저는 (VQGAN처럼) 한 번 학습된 뒤 동결(freeze)된다. 모든 생성 작업은 그 위에 있는 자기회귀 모델이 수행한다.

### 다음 스케일 예측 (Next-Scale Prediction)

생성 모델은 이전 모든 스케일의 토큰을 보고 다음 스케일의 토큰을 예측하는 트랜스포머다.

입력 시퀀스 구조:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

위치 임베딩(position embedding)은 스케일 인덱스와 스케일 내 공간 위치를 모두 인코딩한다. 어텐션은 스케일 순서에서 인과적(causal)이다. 스케일 k의 위치 (i, j)에 있는 토큰은 스케일 1..k의 모든 토큰과, 사용된 스케일 내 순서에서 더 앞에 오는 스케일 k 자체의 토큰에 어텐션할 수 있다(VAR은 스케일 내 인과성이 없는 고정 위치 어텐션을 사용한다. 스케일 내 모든 위치는 병렬로 예측된다).

학습 손실(training loss): 각 스케일 k에서, 이전 스케일의 모든 토큰이 주어졌을 때 토큰 z_k를 예측한다. 이산 VQ 코드에 대한 교차 엔트로피 손실(cross-entropy loss)이다. "시퀀스"가 이제 스케일로 구조화되었다는 점만 빼면 GPT와 동일한 구조다.

### 생성 (Generation)

추론(inference) 시:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

K = 10 스케일의 경우, 생성은 10번의 트랜스포머 순방향 패스(forward pass)다. 각 패스는 스케일 전체를 병렬로 생성한다. 스케일 내 토큰 단위 자기회귀가 없다. 256x256 이미지의 경우 이는 대략 10번의 패스이며, DiT의 28-50번과 대비된다.

### 왜 다음 스케일이 다음 토큰을 이기는가

세 가지 구조적 이점:
1. **거친 것에서 세밀한 것으로(coarse-to-fine)의 진행이 자연 이미지 통계와 정렬된다.** 인간의 시각 지각과 이미지 데이터셋(dataset) 모두 스케일 의존적 규칙성을 보인다. 저주파 구조는 안정적이고 예측 가능하며, 고주파 디테일은 저주파 내용에 조건부로 의존한다. 다음 스케일 예측은 이를 활용한다.
2. **스케일 내 병렬 생성.** GPT 스타일 토큰 자기회귀와 달리, VAR은 한 스케일의 모든 토큰을 한 스텝에 생성한다. 유효 생성 길이는 선형이 아니라 로그 스케일이다.
3. **생성 순서 편향이 없다.** 스케일 k의 토큰은 스케일 k-1 전체를 본다. 후반 컨텍스트가 확보되기 전에 초기 토큰이 먼저 결정되도록 강제하는 "왼쪽(left-of)" 또는 "위(above)" 편향이 없다.

### 스케일링 법칙 (Scaling Law)

Tian et al.은 VAR이 ImageNet의 FID에 대해 거듭제곱 법칙(power-law) 스케일링 곡선을 따름을 입증했다. GPT가 퍼플렉서티에서 보이는 것과 같다. 파라미터나 컴퓨트를 두 배로 늘리면 오차가 안정적으로 절반이 된다. 이는 언어 모델만큼 깔끔하게 이러한 스케일링 거동을 보인 최초의 이미지 생성 모델이었다. 그 결과 VAR 스케일 예측은 아키텍처마다 경험적으로 추측하지 않고 컴퓨트로 예측할 수 있게 된다.

### 확산과의 관계 (Relationship to Diffusion)

VAR과 확산은 동일한 데이터 압축 서사를 공유한다. 둘 다 생성 문제를 더 쉬운 하위 문제들의 시퀀스로 분해한다.

- 확산: 점진적으로 노이즈를 더하고, 한 스텝을 되돌리는 법을 학습한다.
- VAR: 점진적으로 해상도를 더하고, 다음 스케일을 예측하는 법을 학습한다.

이들은 문제를 관통하는 서로 다른 축이다. 둘 다 다루기 쉬운 조건부 분포를 산출한다. 경험적으로 VAR은 추론 시 더 빠르며(패스 수가 적고, 스케일 내 전부 병렬), 클래스 조건부 ImageNet에서 DiT와 동등하거나 능가한다. 텍스트 조건부 VAR(VARclip, HART)은 활발한 연구 방향이다.

```figure
gx-var-next-scale
```

## 직접 만들기 (Build It)

`code/main.py`에서 다음을 수행한다:
1. 합성 "이미지" 데이터(2D 가우시안 링)에 대해 작은 **다중 스케일 VQ 토크나이저**를 만든다.
2. 토큰을 다음 스케일 예측하도록 **VAR 스타일 트랜스포머**를 학습시킨다.
3. 트랜스포머를 4번 호출하고(4 스케일) 디코딩하여 샘플링한다.
4. 스케일 순서 학습이 스케일 내 생성을 병렬로 만든다는 것을 검증한다.

이것은 장난감 수준의 구현이다. 핵심은 스케일로 구조화된 어텐션 마스크(attention mask)와 스케일 내 병렬 생성이 실제로 작동하는 것을 보는 것이다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-var-tokenizer-designer.md`를 생성한다. 다중 스케일 토크나이저를 설계하기 위한 스킬로, 스케일 개수, 스케일 비율, 코드북 크기, 잔차 공유, 디코더 아키텍처를 다룬다.

## 연습 문제 (Exercises)

1. **스케일 개수 절제 실험(ablation).** VAR을 4, 6, 8, 10 스케일로 학습시킨다. 재구성 품질 대 자기회귀 패스 수를 측정한다. 스케일이 많을수록 = 잔차가 세밀할수록 = 품질이 좋지만 패스가 더 많다.

2. **코드북 크기.** 코드북 크기 512, 4096, 16384로 토크나이저를 학습시킨다. 큰 코드북은 재구성이 더 좋지만 예측이 더 어렵다. 변곡점(knee)을 찾아라.

3. **스케일 내 병렬 확인.** 학습된 VAR에 대해 어텐션 패턴을 명시적으로 측정한다. 스케일 k 안에서, 모델은 스케일 간(cross-scale) 위치에는 어텐션하지만 스케일 내(intra-scale)에는 어텐션하지 않는가? 마스크 구현을 검증하라.

4. **VAR 대 DiT 스케일링.** 동일한 ImageNet 클래스 조건부 과제에 대해, 일치하는 파라미터 예산(예: 33M, 130M, 458M)으로 VAR과 DiT를 학습시킨다. FID 대 컴퓨트를 그린다. VAR은 각 크기에서 DiT를 앞서야 한다. 작은 규모에서 논문의 결과를 재현하라.

5. **텍스트 조건화.** adaLN을 통해 텍스트 임베딩(CLIP 풀링)을 추가 조건 입력으로 받도록 VAR을 확장한다. 이것이 HART 레시피다. 텍스트 정렬 샘플링에서 FID가 얼마나 개선되는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive(시각적 자기회귀)" | VQ 토큰 격자의 피라미드에 대한 다음 스케일 예측으로 하는 이미지 생성 |
| 다음 스케일 예측(Next-scale prediction) | "거칠게 예측한 뒤 세밀하게" | 모델이 이전 모든 스케일에 조건부로 의존하여 점점 높아지는 해상도 스케일에서 토큰을 예측 |
| 다중 스케일 VQ 토크나이저(Multi-scale VQ tokenizer) | "잔차 VQ" | 해상도가 점점 높아지는 K개의 토큰 격자를 생성하고, 디코더가 모든 스케일을 합산하는 VQ-VAE |
| 스케일 k(Scale k) | "피라미드 레벨 k" | K개의 해상도 레벨 중 하나로, k=1의 1x1부터 k=K의 (H/p)x(W/p)까지 |
| 스케일 내 병렬(Parallel-within-scale) | "스케일당 순방향 한 번" | 스케일 k의 모든 토큰이 자기회귀적이 아니라 하나의 트랜스포머 패스로 예측됨 |
| 스케일 간 인과성(Causal-across-scales) | "스케일 순서 어텐션" | 스케일 k의 토큰은 스케일 1..k 전체에 어텐션할 수 있지만 스케일 k+1..K에는 할 수 없음 |
| 잔차 VQ(Residual VQ) | "가산적 토큰화" | 각 스케일의 토큰은 하위 스케일이 남긴 잔차를 인코딩하며, 디코더가 모든 스케일 임베딩을 합산 |
| VAR 스케일링 법칙(VAR scaling law) | "이미지 GPT 스케일링" | FID가 언어 모델의 퍼플렉서티처럼 컴퓨트에 대해 예측 가능한 거듭제곱 법칙을 따름 |
| HART | "하이브리드 VAR + 텍스트" | MaskGIT 스타일의 반복적 디코딩을 VAR의 스케일 구조와 결합한 텍스트 조건부 VAR 변형 |
| 스케일 위치 임베딩(Scale position embedding) | "(스케일, 행, 열) 삼중쌍" | 스케일 인덱스와 스케일 내 공간 좌표를 모두 담는 위치 인코딩 |

## 더 읽을거리 (Further Reading)

- [Tian et al., 2024: "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) VAR 논문, 표준 참고 문헌
- [Peebles and Xie, 2022: "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT, 확산 비교 베이스라인(baseline)
- [Esser et al., 2021: "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN, VAR의 다중 스케일 토크나이저가 확장하는 토크나이저 계열
- [van den Oord et al., 2017: "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE, 이산 이미지 토큰화의 토대
- [Tang et al., 2024: "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) 텍스트 조건부 VAR
