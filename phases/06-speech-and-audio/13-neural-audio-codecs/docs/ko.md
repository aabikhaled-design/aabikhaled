# 뉴럴 오디오 코덱(Neural Audio Codecs): EnCodec, SNAC, Mimi, DAC와 의미-음향 분리

> 2026년의 오디오 생성은 거의 전부 토큰(token)이다. EnCodec, SNAC, Mimi, DAC는 연속 파형(waveform)을 트랜스포머(transformer)가 예측할 수 있는 이산(discrete) 시퀀스로 바꾼다. 의미(semantic) 대 음향(acoustic) 토큰 분리(첫 번째 코드북을 의미용으로, 나머지를 음향용으로)는 오디오 분야에서 트랜스포머 이후 가장 중요한 아키텍처적 전환이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60분

## 문제 (The Problem)

언어 모델은 이산 토큰 위에서 동작한다. 오디오는 연속적이다. 음성 / 음악을 위한 LLM 스타일 모델(MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus)을 원한다면, 먼저 **뉴럴 오디오 코덱(neural audio codec)**이 필요하다. 오디오를 작은 토큰 어휘(vocabulary)로 이산화하는 학습된 인코더(encoder)와, 파형을 복원하는 짝 디코더(decoder) 말이다.

두 계열이 등장했다.

1. **복원 우선 코덱(Reconstruction-first codecs)**(EnCodec, DAC. 지각적(perceptual) 오디오 품질을 최적화한다. 토큰은 "음향적(acoustic)"이다) 화자 신원, 음색(timbre), 배경 잡음까지 모든 것을 담는다.
2. **의미 우선 코덱(Semantic-first codecs)**: Mimi(Kyutai), SpeechTokenizer. 첫 번째 코드북이 언어적 / 음소적 내용을 인코딩하도록 강제한다(보통 WavLM으로부터 증류(distill)). 이후 코드북들은 음향적 세부사항이다.

2024-2026년의 통찰: **순수 복원 코덱은 텍스트로부터 생성하려 할 때 흐릿한 음성을 준다.** 코덱 토큰 위의 LLM은 같은 코드북에서 언어 구조와 음향 구조를 둘 다 학습해야 하는데, 이는 규모를 키워도 잘 되지 않는다. 둘을 분리하는 것(의미 코드북 0, 음향 코드북 1-N)이 바로 Moshi와 Sesame CSM을 동작하게 만든다.

## 개념 (The Concept)

![네 가지 코덱 지형: EnCodec, DAC, SNAC (멀티스케일), Mimi (의미+음향)](../assets/codec-comparison.svg)

### 핵심 트릭: 잔차 벡터 양자화(Residual Vector Quantization, RVQ)

좋은 품질을 위해 수백만 개의 코드가 필요한 하나의 큰 코드북 대신, 모든 현대 오디오 코덱은 **RVQ**를 쓴다. 작은 코드북들의 캐스케이드(cascade)다. 첫 번째 코드북은 인코더 출력을 양자화한다. 두 번째는 잔차(residual)를 양자화한다. 이런 식이다. 각 코드북은 1024개의 코드를 가진다. 8개의 코드북 = 유효 어휘 1024^8 = 10^24.

추론(inference) 시 디코더는 프레임마다 선택된 모든 코드를 합산하여 복원한다.

### 2026년에 중요한 네 가지 코덱

**EnCodec (Meta, 2022).** 베이스라인(baseline)이다. 파형 위의 인코더-디코더, RVQ 병목(bottleneck). 24kHz, 최대 32개 코드북 가능, 기본값은 4개 코드북 @ 1.5kbps. `1D conv + transformer + 1D conv` 아키텍처를 쓴다. MusicGen이 사용한다.

**DAC (Descript, 2023).** L2 정규화된 코드북, 주기적 활성화 함수(activation function), 개선된 손실(loss)을 갖춘 RVQ. 모든 오픈 코덱 중 최고의 복원 충실도(fidelity): 12개 코드북으로 때로는 원본 음성과 구분 불가능하다. 44.1kHz 풀밴드.

**SNAC (Hubert Siuzdak, 2024).** 멀티스케일 RVQ: 거친(coarse) 코드북은 세밀한(fine) 코드북보다 낮은 프레임 레이트로 동작한다. 사실상 오디오를 계층적으로 모델링한다. 약 12Hz의 거친 "스케치"에 50Hz의 세부를 더한다. 계층 구조가 LM 기반 생성에 잘 매핑되기 때문에 Orpheus-3B가 사용한다.

**Mimi (Kyutai, 2024).** 2026년의 판도를 바꾼 주역. 12.5Hz 프레임 레이트(극히 낮음), 8개 코드북 @ 4.4kbps. 코드북 0은 **WavLM으로부터 증류된다**. WavLM의 음성-내용 특성을 예측하도록 학습된다. 코드북 1-7은 음향 잔차다. 이 분리가 Moshi(레슨 15)와 Sesame CSM을 구동한다.

### 프레임 레이트는 언어 모델링에 중요하다

낮은 프레임 레이트 = 짧은 시퀀스 = 빠른 LM.

| 코덱 | 프레임 레이트 | 1초 = N 프레임 | 적합한 용도 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 음악, 일반 오디오 |
| DAC-44.1k | 86 Hz | 86 | 고충실도 음악 |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM 효율적 |
| Mimi | 12.5 Hz | 12.5 | 스트리밍 음성 |

12.5Hz에서 10초 발화는 단 125개의 코덱 프레임이다. 트랜스포머가 쉽게 예측할 수 있다.

### 의미 vs 음향 토큰

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **의미 토큰(Mimi의 코드북 0).** 무엇을 말했는지를 인코딩한다. 음소(phoneme), 단어, 내용. 보조 예측 손실을 통해 WavLM으로부터 증류된다.
- **음향 토큰(코드북 1-7).** 음색, 화자 신원, 운율(prosody), 배경 잡음, 세밀한 세부를 인코딩한다.

AR LM은 (텍스트를 조건으로) 의미 토큰을 먼저 예측한 뒤, (의미 + 화자 레퍼런스를 조건으로) 음향 토큰을 예측한다. 이 인수분해(factorization) 덕분에 현대 TTS가 음성을 제로샷(zero-shot)으로 복제할 수 있다. 의미 모델은 내용을 다루고, 음향 모델은 음색을 다룬다.

### 2026년 복원 품질 (초당 비트, 비트레이트가 낮을수록 좋음)

| 코덱 | 비트레이트 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus 같은 전통 코덱은 여전히 비트당 지각 품질에서 이긴다. 뉴럴 코덱은 **이산 토큰**(Opus는 생성하지 않는다)과 **생성 모델 품질**(그 토큰으로 LM이 할 수 있는 것)에서 이긴다.

```figure
rvq-codec-cascade
```

## 직접 만들기 (Build It)

### 1단계: EnCodec으로 인코딩

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6kbps에서 `n_codebooks=8`. 각 코드는 0-1023(10비트)이다.

### 2단계: 디코딩하고 복원 측정하기

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 3단계: 의미-음향 분리 (Mimi 스타일)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

의미 코드북 0은 WavLM에 정렬되어 있다. 텍스트-투-의미 트랜스포머를 학습할 수 있다. 오디오로 곧장 가는 것보다 훨씬 작은 어휘다. 그다음 별도의 음향-투-파형 디코더가 화자 레퍼런스를 조건으로 삼는다.

### 4단계: 코덱 토큰 위 AR LM이 동작하는 이유

Mimi의 12.5Hz × 8개 코드북에서 10초 음성 클립의 경우:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000개 토큰은 트랜스포머에게 사소한 컨텍스트다. 256M 파라미터 트랜스포머는 현대 GPU에서 밀리초 단위로 10초 음성을 생성할 수 있다.

## 라이브러리로 써보기 (Use It)

문제 → 코덱 매핑:

| 과제 | 코덱 |
|------|-------|
| 일반 음악 생성 | EnCodec-24k |
| 최고 충실도 복원 | DAC-44.1k |
| 음성 위 AR LM (TTS) | SNAC 또는 Mimi |
| 스트리밍 풀 듀플렉스 음성 | Mimi (12.5 Hz) |
| 텍스트가 있는 음향효과 라이브러리 | EnCodec + T5 condition |
| 세밀한 오디오 편집 | DAC + inpainting |

경험칙: **생성 모델을 만든다면 Mimi나 SNAC으로 시작하라. 압축 파이프라인을 만든다면 Opus를 써라.**

## 함정 (Pitfalls)

- **너무 많은 코드북.** 코드북을 추가하면 충실도는 선형으로 늘지만 LM 시퀀스 길이도 선형으로 는다. 8-12개에서 멈춰라.
- **프레임 레이트 불일치.** 12.5Hz Mimi로 LM을 학습한 뒤 50Hz EnCodec으로 파인튜닝하면 조용히 실패한다.
- **모든 코드북이 같다고 가정하기.** Mimi에서 코드북 0은 내용을 담는다. 이를 잃으면 명료도가 파괴된다. 코드북 7을 잃는 것은 거의 알아챌 수 없다.
- **복원 품질을 유일한 지표로 쓰기.** 코덱이 훌륭한 복원을 가져도 의미 구조가 나쁘면 LM 기반 생성에는 쓸모없을 수 있다.

## 산출물 (Ship It)

`outputs/skill-codec-picker.md`로 저장한다. 주어진 생성 또는 압축 과제에 맞는 코덱을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 장난감 스칼라 + 잔차 양자화기를 구현하고 코드북을 추가할 때 복원 오차를 측정한다.
2. **보통.** `encodec`을 설치하고 홀드아웃(held-out) 음성 클립에서 1, 4, 8, 32개 코드북을 비교한다. 비트레이트 대비 PESQ 또는 MSE를 플롯한다.
3. **어려움.** Mimi를 로드한다. 클립을 인코딩한다. 코드북 0을 무작위 정수로 교체하고 디코딩한다. 그다음 코드북 7을 비슷하게 교체한다. 두 손상을 비교한다. 코드북 0 손상은 명료도를 파괴해야 하고, 코드북 7 손상은 거의 아무것도 바꾸지 않아야 한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| RVQ | 잔차 양자화 | 작은 코드북들의 캐스케이드. 각각이 이전 잔차를 양자화함. |
| Frame rate | 코덱 속도 | 초당 토큰 프레임 수. 낮을수록 빠른 LM. |
| Semantic codebook | 코드북 0 (Mimi) | SSL 특성에서 증류된 코드북. 내용을 인코딩함. |
| Acoustic codebooks | 나머지 전부 | 음색, 운율, 잡음, 세밀한 세부. |
| PESQ / ViSQOL | 지각 품질 | MOS와 상관관계가 있는 객관적 지표. |
| EnCodec | Meta 코덱 | RVQ 베이스라인. MusicGen이 사용. |
| Mimi | Kyutai 코덱 | 12.5Hz 프레임 레이트. 의미-음향 분리. Moshi를 구동. |

## 더 읽을거리 (Further Reading)

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438): RVQ 베이스라인.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546): 최고 충실도 오픈.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411): 멀티스케일 RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer): 의미-음향 분리, WavLM 증류.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143): 2단계 의미/음향 패러다임.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312): 원조 스트리밍 가능 RVQ 코덱.
