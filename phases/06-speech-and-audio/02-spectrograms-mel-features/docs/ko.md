# 스펙트로그램, 멜 스케일 & 오디오 특성

> 신경망(neural network)은 원시 파형(raw waveform)을 잘 소비하지 못한다. 신경망은 스펙트로그램(spectrogram)을 소비한다. 멜 스펙트로그램(mel spectrogram)은 더 잘 소비한다. 2026년의 모든 ASR, TTS, 오디오 분류기(audio classifier)는 이 단 하나의 전처리(preprocessing) 선택으로 흥하거나 망한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45분

## 문제 (The Problem)

10초짜리 16 kHz 클립을 잡아라. 그것은 모두 `[-1, 1]` 범위에 있는 160,000개의 부동소수점이며, "개 짖음"이나 "고양이라는 단어"라는 레이블(label)과는 거의 완벽하게 상관관계가 없다. 원시 파형은 정보를 담고 있지만 모델이 쉽게 추출할 수 없는 형태로 담고 있다. 100 ms 간격으로 발화된 동일한 두 음소(phoneme)는 원시 샘플이 완전히 다르다.

스펙트로그램이 이를 해결한다. 인간의 지각이 무시하는 곳(마이크로초 단위의 지터)에서는 시간적 세부를 뭉개고, 지각이 주의를 기울이는 곳(~10–25 ms 시간 윈도우에 걸쳐 어떤 주파수가 에너지를 띠는가)에서는 구조를 보존한다.

멜 스펙트로그램은 한 걸음 더 나아간다. 인간은 음높이(pitch)를 로그적으로 지각한다. 100 Hz 대 200 Hz는 1000 Hz 대 2000 Hz와 "같은 거리만큼 떨어져" 들린다. 멜 스케일(mel scale)은 이에 맞도록 주파수 축을 휘게 한다. 멜 스케일이 적용된 스펙트로그램은 2010년부터 2026년까지 음성 머신러닝에서 단연 가장 중요한 특성(feature)이다.

## 개념 (The Concept)

![파형에서 STFT, 멜 스펙트로그램, MFCC로 이어지는 사다리](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** 파형을 겹치는 프레임(frame)으로 자른다(전형적으로: 25 ms 윈도우, 10 ms 홉 = 16 kHz에서 400 샘플 / 160 샘플). 각 프레임에 윈도우 함수(window function)를 곱한다(Hann이 기본; Hamming은 약간 다른 트레이드오프(trade-off)). 각 프레임을 FFT한다. 크기 스펙트럼(magnitude spectra)을 `(n_frames, n_freq_bins)` 형태의 행렬(matrix)로 쌓는다. 이것이 스펙트로그램이다.

**로그 크기(Log-magnitude).** 원시 크기는 5~6 자릿수에 걸쳐 있다. 동적 범위(dynamic range)를 압축하려면 `log(|X| + 1e-6)` 또는 `20 * log10(|X|)`를 취한다. 모든 프로덕션 파이프라인(pipeline)은 원시 크기가 아니라 로그 크기를 사용한다.

**멜 스케일(Mel scale).** Hz 단위의 주파수 `f`는 `m = 2595 * log10(1 + f / 700)`으로 멜 `m`에 대응한다. 이 사상(mapping)은 1 kHz 아래에서는 대략 선형이고 그 위에서는 대략 로그적이다. 0–8 kHz를 덮는 80개의 멜 빈(mel bin)이 표준 ASR 입력이다.

**멜 필터뱅크(Mel filterbank).** 멜 스케일 위에 등간격으로 배치된 삼각 필터(triangular filter)들의 집합이다. 각 필터는 인접한 FFT 빈들의 가중합(weighted sum)이다. STFT 크기에 필터뱅크 행렬을 곱하면 단 한 번의 행렬곱(matmul)으로 멜 스펙트로그램이 나온다.

**로그 멜 스펙트로그램(Log-mel spectrogram).** `log(mel_spec + 1e-10)`. Whisper의 입력. Parakeet의 입력. SeamlessM4T의 입력. 2026년의 보편적인 오디오 프론트엔드(frontend)다.

**MFCC.** 로그 멜 스펙트로그램을 취해 DCT(type II)를 적용하고 처음 13개 계수를 유지한다. 특성을 비상관화(decorrelate)하고 더 압축한다. CNN/트랜스포머(Transformer)가 원시 로그 멜에서 따라잡은 2015년경까지 지배적인 특성이었다. 화자 인식(speaker recognition)(x-vectors, ECAPA)에서는 여전히 쓰인다.

**해상도 트레이드(Resolution trade).** FFT가 클수록 = 더 나은 주파수 해상도지만 더 나쁜 시간 해상도다. 25 ms / 10 ms가 오디오 머신러닝의 기본값이고; 음악에는 50 ms / 12.5 ms; 과도 신호 검출(드럼 타격, 파열음)에는 5 ms / 2 ms를 쓴다.

```figure
spectrogram-window
```

## 직접 만들기 (Build It)

### 단계 1: 파형을 프레임화한다

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160`을 가진 10초짜리 16 kHz 클립은 998개의 프레임을 낸다.

### 단계 2: Hann 윈도우

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 전에 원소별로 곱한다. 0이 아닌 끝점에서 잘림으로 인해 생기는 스펙트럼 누설(spectral leakage)을 제거한다.

### 단계 3: STFT 크기

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

프로덕션은 `torch.stft`나 `librosa.stft`(FFT 기반, 벡터화됨)를 사용한다. 여기의 루프는 교육용이며, `code/main.py`에서 짧은 클립에 대해 실행된다.

### 단계 4: 멜 필터뱅크

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400`으로 0–8 kHz를 덮는 80 멜은 `(80, 201)` 행렬을 준다. `(n_frames, 201)` STFT 크기에 그 전치(transpose)를 곱하면 `(n_frames, 80)` 멜 스펙트로그램이 나온다.

### 단계 5: 로그 멜

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

흔한 대안: `librosa.power_to_db`(기준 정규화된 dB), `10 * log10(power + eps)`. Whisper는 더 복잡한 클립 + 정규화 루틴을 사용한다(Whisper의 `log_mel_spectrogram` 참고).

### 단계 6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

각 로그 멜 프레임에 DCT를 적용하고 처음 13개 계수를 유지한다. 이것이 MFCC 행렬이다. 첫 번째 계수는 보통 버린다(전체 에너지를 인코딩하기 때문).

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 작업 | 특성 |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 로그 멜, 10 ms 홉, 25 ms 윈도우 |
| TTS 음향 모델 (VITS, F5-TTS, Kokoro) | 80 멜, 세밀한 시간 제어를 위한 5–12 ms 홉 |
| 오디오 분류 (AST, PANNs, BEATs) | 128 로그 멜, 10 ms 홉 |
| 화자 임베딩 (ECAPA-TDNN, WavLM) | 80 로그 멜 또는 원시 파형 SSL |
| 음악 (MusicGen, Stable Audio 2) | EnCodec 이산 토큰(멜이 아님) |
| 키워드 스포팅(keyword spotting) | 초소형 디바이스를 위한 40 MFCC |

경험 법칙: **음악 작업이 아니라면 80 로그 멜로 시작하라.** 거기서 벗어나는 선택은 모두 그럴 만한 근거를 직접 입증해야 한다.

## 2026년에도 여전히 출시되는 함정들 (Pitfalls that still ship in 2026)

- **멜 개수 불일치.** 80 멜로 학습하고 128 멜로 추론. 조용한 실패. 양쪽 끝에서 특성 형태(shape)를 로깅하라.
- **업스트림 샘플 레이트 불일치.** 22.05 kHz에서 계산된 멜은 16 kHz와 다르게 보인다. 특성화 *전에* SR을 고쳐라.
- **dB 대 log.** Whisper는 dB 멜이 아니라 로그 멜을 기대한다. 일부 HF 파이프라인은 자동 감지하지만, 직접 작성한 커스텀 코드는 그러지 않는다.
- **정규화 드리프트(Normalization drift).** 학습 중에는 발화별(per-utterance) 정규화, 추론 중에는 전역(global) 정규화. WER을 두 배로 만드는 프로덕션 버그.
- **패딩으로 인한 누설.** 클립 끝을 0으로 패딩하면 뒤따르는 프레임에서 평탄한 스펙트럼이 생긴다. 대칭적으로 패딩하거나 복제하라.

## 산출물 (Ship It)

`outputs/skill-feature-extractor.md`로 저장하라. 이 스킬은 주어진 모델 타깃에 대해 특성 유형, 멜 개수, 프레임/홉, 정규화를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 처프(chirp, 주파수가 200 → 4000 Hz로 스윕됨)를 합성하고 프레임별로 argmax 멜 빈을 출력한다. (선택적으로) 그려서 스윕과 일치하는지 확인하라.
2. **중간.** `n_mels`를 `{40, 80, 128}`로, `frame_len`을 `{200, 400, 800}`으로 바꿔 다시 실행하라. 시간 축에 걸쳐 날카로운 피크의 대역폭을 측정하라. 어떤 조합이 처프를 가장 잘 분해하는가?
3. **어려움.** `power_to_db`를 구현하고 AudioMNIST에서 (a) 원시 로그 멜, (b) `ref=max`인 dB 멜, (c) MFCC-13 + 델타 + 델타-델타를 사용해 초소형 CNN 분류기의 ASR 정확도를 비교하라. top-1 정확도를 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 프레임(Frame) | 조각 | 하나의 FFT에 입력되는 25 ms 파형 청크. |
| 홉(Hop) | 보폭(stride) | 연속된 프레임 사이의 샘플; 10 ms가 ASR 기본값. |
| 윈도우(Window) | Hann/Hamming 같은 것 | 프레임 가장자리를 0으로 점점 줄이는 점별 곱셈자. |
| STFT | 스펙트로그램 생성기 | 프레임화 + 윈도우화된 FFT; 시간 × 주파수 행렬을 낸다. |
| 멜(Mel) | 휘어진 주파수 | 로그 지각 스케일; `m = 2595·log10(1 + f/700)`. |
| 필터뱅크(Filterbank) | 그 행렬 | STFT를 멜 빈으로 투영하는 삼각 필터들. |
| 로그 멜(Log-mel) | Whisper의 입력 | `log(mel_spec + eps)`; 2026년에 표준화됨. |
| MFCC | 구식 특성 | 로그 멜의 DCT; 13개 계수, 비상관화됨. |

## 더 읽을거리 (Further Reading)

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420): MFCC 논문.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/): 원조 멜 스케일.
- [OpenAI: Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) 참조 구현을 읽어보라.
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html): `mfcc`, `melspectrogram`, 홉/윈도우 참고 자료.
- [NVIDIA NeMo: audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) Parakeet + Canary 모델을 위한 프로덕션 규모 파이프라인.
