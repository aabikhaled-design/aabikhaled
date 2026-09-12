# 오디오 기초: 파형, 샘플링, 푸리에 변환

> 파형(waveform)은 원시 신호다. 스펙트로그램(spectrogram)은 그 표현이다. 멜 특성(mel feature)은 머신러닝에 친화적인 형태다. 현대의 모든 ASR과 TTS 파이프라인(pipeline)은 이 사다리를 오르며, 그 첫 단은 샘플링(sampling)과 푸리에(Fourier)를 이해하는 일이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45분

## 문제 (The Problem)

마이크는 압력 대 시간(pressure-vs-time) 신호를 만들어낸다. 신경망(neural network)은 텐서(tensor)를 소비한다. 그 사이에는 일련의 관례(convention)가 쌓여 있는데, 이를 위반하면 조용한 버그가 생긴다. 모델은 잘 학습되는데 WER이 두 배가 되거나, TTS가 잡음(hiss)을 내보내거나, 음성 복제(voice cloning) 시스템이 화자(speaker) 대신 마이크를 암기해 버린다.

음성 시스템의 모든 버그는 세 가지 질문 중 하나로 거슬러 올라간다.

1. 데이터가 어떤 샘플 레이트(sample rate)로 녹음되었으며, 모델은 무엇을 기대하는가?
2. 신호가 에일리어싱(aliasing)되었는가?
3. 원시 샘플(raw sample)을 다루고 있는가, 아니면 주파수 표현(frequency representation)을 다루고 있는가?

이것들을 제대로 하면 Phase 6의 나머지는 다룰 만해진다. 잘못하면 Whisper-Large-v4조차도 쓰레기를 만들어낸다.

## 개념 (The Concept)

![파형, 샘플링, DFT, 주파수 빈 시각화](../assets/audio-fundamentals.svg)

**파형(Waveform).** `[-1.0, 1.0]` 범위의 부동소수점 1차원 배열이다. 샘플 번호로 인덱싱한다. 초 단위로 변환하려면 샘플 레이트로 나눈다: `t = n / sr`. 16 kHz에서 10초짜리 클립은 160,000개의 부동소수점 배열이다.

**샘플링 레이트(Sampling rate, sr).** 초당 샘플 수다. 2026년 기준 흔히 쓰이는 레이트는 다음과 같다.

| 레이트 | 용도 |
|------|-----|
| 8 kHz | 전화망(telephony), 레거시 VOIP. 나이퀴스트(Nyquist)가 4 kHz라 자음을 죽인다. ASR에는 피할 것. |
| 16 kHz | ASR 표준. Whisper, Parakeet, SeamlessM4T v2 모두 16 kHz를 소비한다. |
| 22.05 kHz | 구형 모델용 TTS 보코더(vocoder) 학습. |
| 24 kHz | 현대 TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD 오디오, 음악. |
| 48 kHz | 영화, 프로 오디오, 고충실도(high-fidelity) TTS (VALL-E 2, NaturalSpeech 3). |

**나이퀴스트-섀넌(Nyquist-Shannon).** 샘플 레이트 `sr`은 `sr/2`까지의 주파수를 명확하게 표현할 수 있다. `sr/2` 경계가 *나이퀴스트 주파수(Nyquist frequency)*다. 나이퀴스트 위의 에너지는 *에일리어싱(aliasing)*된다. 더 낮은 주파수로 접혀 내려가 신호를 손상시킨다. 다운샘플링(downsampling) 전에는 항상 저역 통과 필터(low-pass filter)를 적용하라.

**비트 심도(Bit depth).** 16비트 PCM(부호 있는 int16, 범위 ±32,767)은 보편적인 교환 형식이다. 음악에는 24비트, 내부 DSP에는 32비트 float를 쓴다. `soundfile` 같은 라이브러리는 int16을 읽지만 `[-1, 1]` 범위의 float32 배열로 노출한다.

**푸리에 변환(Fourier Transform).** 모든 유한 신호는 서로 다른 주파수의 사인파(sinusoid)의 합이다. 이산 푸리에 변환(Discrete Fourier Transform, DFT)은 `N`개의 샘플에 대해 `N`개의 복소 계수(complex coefficient)를 계산한다. 주파수 빈(frequency bin)마다 하나씩. `bin k`는 주파수 `k · sr / N` Hz에 대응한다. 크기(magnitude)는 그 주파수에서의 진폭(amplitude)이고, 각도(angle)는 위상(phase)이다.

**FFT.** 고속 푸리에 변환(Fast Fourier Transform): `N`이 2의 거듭제곱일 때 DFT를 위한 `O(N log N)` 알고리즘이다. 모든 오디오 라이브러리는 내부적으로 FFT를 사용한다. 16 kHz에서 1024 샘플 FFT는 0–8 kHz에 걸쳐 15.6 Hz 해상도로 512개의 사용 가능한 주파수 빈을 준다.

**프레이밍 + 윈도우(Framing + window).** 우리는 클립 전체를 FFT하지 않는다. 겹치는 *프레임(frame)*(보통 25 ms, 10 ms 홉(hop))으로 잘게 나누고, 각 프레임에 윈도우 함수(window function)(Hann, Hamming)를 곱해 가장자리 불연속(edge discontinuity)을 죽인 뒤, 각 프레임을 FFT한다. 이것이 단시간 푸리에 변환(Short-Time Fourier Transform, STFT)이다. Lesson 02가 여기서부터 이어진다.

```figure
mel-scale
```

## 직접 만들기 (Build It)

### 단계 1: 클립을 읽고 파형을 그린다

`code/main.py`는 데모를 의존성 없이 유지하기 위해 표준 라이브러리 `wave` 모듈만 사용한다. 프로덕션에서는 `soundfile`이나 `torchaudio.load`(둘 다 `(waveform, sr)` 튜플을 반환)를 쓰게 된다:

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 단계 2: 제1원리로부터 사인파를 합성한다

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz에서 1초짜리 440 Hz 사인파(콘서트 A)는 16,000개의 부동소수점이다. 16비트 PCM 인코딩을 사용해 `wave.open(..., "wb")`로 쓴다.

### 단계 3: DFT를 손으로 계산한다

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`: `N=256`부터 정확성을 확인하는 데는 괜찮지만 실제 오디오에는 쓸모없다. 실제 코드는 `numpy.fft.rfft`나 `torch.fft.rfft`를 호출한다.

### 단계 4: 지배적 주파수를 찾는다

크기 피크 인덱스 `k_star`는 주파수 `k_star * sr / N`에 대응한다. 440 Hz 사인파에 대해 이를 실행하면 빈 `440 * N / sr`에서 피크가 나와야 한다.

### 단계 5: 에일리어싱을 시연한다

7 kHz 사인파를 10 kHz로 샘플링한다(나이퀴스트 = 5 kHz). 7 kHz 톤은 나이퀴스트 위에 있어 `10 − 7 = 3 kHz`로 접힌다. FFT 피크가 3 kHz에 나타난다. 이것이 고전적인 에일리어싱 데모이며, 모든 DAC/ADC가 브릭월 저역 통과 필터(brick-wall low-pass filter)를 탑재하는 이유다.

## 라이브러리로 써보기 (Use It)

2026년에 실제로 출시에 쓰게 될 스택:

| 작업 | 라이브러리 | 이유 |
|------|---------|-----|
| WAV/FLAC/OGG 읽기/쓰기 | `soundfile` (libsndfile 래퍼) | 가장 빠르고 안정적이며 float32를 반환한다. |
| 리샘플(Resample) | `torchaudio.transforms.Resample` 또는 `librosa.resample` | 올바른 안티에일리어싱(anti-aliasing)이 내장되어 있다. |
| STFT / 멜(Mel) | `torchaudio` 또는 `librosa` | GPU 친화적; PyTorch 생태계. |
| 실시간 스트리밍 | `sounddevice` 또는 `pyaudio` | 크로스 플랫폼 PortAudio 바인딩. |
| 파일 검사 | `ffprobe` 또는 `soxi` | CLI, 빠름, sr/채널/코덱을 보고한다. |

결정 규칙: **다른 무엇을 맞추기 전에 샘플 레이트부터 맞춰라**. Whisper는 16 kHz 모노 float32를 기대한다. 44.1 kHz 스테레오를 넘기면 모델 버그처럼 보이는 쓰레기가 나온다.

## 산출물 (Ship It)

`outputs/skill-audio-loader.md`로 저장하라. 이 스킬은 오디오 입력이 다운스트림(downstream) 모델의 기대에 맞는지 확인하고, 맞지 않을 때 올바르게 리샘플하도록 돕는다.

## 연습 문제 (Exercises)

1. **쉬움.** 16 kHz에서 220 Hz + 440 Hz + 880 Hz를 섞은 1초짜리를 합성하라. DFT를 실행하라. 기대되는 빈에서 세 개의 피크를 확인하라.
2. **중간.** 48 kHz로 자기 목소리를 3초짜리 WAV로 녹음하라. `torchaudio.transforms.Resample`(안티에일리어싱 포함)을 사용해 16 kHz로 다운샘플하고, 그다음 순진한 데시메이션(decimation, 세 번째 샘플마다 취함)으로 16 kHz로 다운샘플하라. 둘 다 FFT하라. 에일리어싱이 어디에 나타나는가?
3. **어려움.** `math`와 단계 3의 DFT만 사용해 STFT를 밑바닥부터 만들어라. 프레임 크기 400, 홉 160, Hann 윈도우. `matplotlib.pyplot.imshow`로 크기를 그려라. 이것이 Lesson 02의 스펙트로그램이다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 샘플 레이트(Sample rate) | 초당 샘플 수 | ADC가 신호를 측정하는 Hz 단위의 주파수. |
| 나이퀴스트(Nyquist) | 표현할 수 있는 최대 주파수 | `sr/2`; 그 위의 에너지는 아래로 에일리어싱된다. |
| 비트 심도(Bit depth) | 각 샘플의 해상도 | `int16` = 65,536개 레벨; `float32` = `[-1, 1]`에서 24비트 정밀도. |
| DFT | 시퀀스에 대한 푸리에 변환 | `N`개 샘플 → `N`개 복소 주파수 계수. |
| FFT | 고속 DFT | `N` = 2의 거듭제곱이 필요한 `O(N log N)` 알고리즘. |
| 빈(Bin) | 주파수 열(column) | `k · sr / N` Hz; 해상도 = `sr / N`. |
| STFT | 내부의 스펙트로그램 | 시간에 걸친 프레임화 + 윈도우화된 FFT. |
| 에일리어싱(Aliasing) | 이상한 주파수 유령 | 나이퀴스트 위의 에너지가 더 낮은 빈으로 거울처럼 내려옴. |

## 더 읽을거리 (Further Reading)

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf): 샘플링 정리의 배경이 된 논문.
- [Smith: The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) 무료의 정전(正典) 격 DSP 교과서.
- [librosa docs: audio primer](https://librosa.org/doc/latest/tutorial.html) 코드와 함께하는 실용적 안내.
- [Heinrich Kuttruff: Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) 실제 오디오가 깨끗한 사인파가 아닌 이유에 대한 참고서.
- [Steve Eddins: FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) 주파수 빈 직관을 10분 만에 정리.
