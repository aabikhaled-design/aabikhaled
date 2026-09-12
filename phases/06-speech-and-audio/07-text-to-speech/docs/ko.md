# 텍스트 음성 변환(TTS): Tacotron에서 F5와 Kokoro까지

> ASR은 음성을 텍스트로 역변환한다; TTS는 텍스트를 음성으로 역변환한다. 2026년의 스택은 세 부분이다: 텍스트 → 토큰(token), 토큰 → 멜(mel), 멜 → 파형(waveform). 각 부분에는 노트북에 들어가는 기본 모델이 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75분

## 문제 (The Problem)

문자열이 있다: "Please remind me to water the plants at 6 pm." 자연스럽게 들리고, 올바른 운율(prosody, 멈춤, 강세)을 갖추고, "plants"를 올바른 모음으로 발음하며, 라이브 음성 비서를 위해 CPU에서 300 ms 이내에 돌아가는 3초짜리 오디오 클립이 필요하다. 게다가 목소리를 바꾸고, 코드 스위칭(code-switched) 입력("remind me at 6 pm, daijoubu?")을 처리하고, 이름에서 망신당하지 않아야 한다.

현대 TTS 파이프라인은 다음과 같다:

1. **텍스트 프론트엔드(Text frontend).** 텍스트(날짜, 숫자, 이메일)를 정규화하고, 음소(phoneme)나 서브워드 토큰으로 변환하고, 운율 특성을 예측한다.
2. **음향 모델(Acoustic model).** 텍스트 → 멜 스펙트로그램(mel spectrogram). Tacotron 2(2017), FastSpeech 2(2020), VITS(2021), F5-TTS(2024), Kokoro(2024).
3. **보코더(Vocoder).** 멜 → 파형. WaveNet(2016), WaveRNN, HiFi-GAN(2020), BigVGAN(2022), 2024년 이후의 신경 코덱(neural codec) 보코더.

2026년에는 종단간(end-to-end) 확산(diffusion) 및 흐름 매칭(flow-matching) 모델로 음향 + 보코더 구분이 흐려진다. 하지만 세 부분이라는 멘탈 모델은 디버깅에 여전히 유효하다.

## 개념 (The Concept)

![Tacotron, FastSpeech, VITS, F5/Kokoro 나란히 비교](../assets/tts.svg)

**Tacotron 2(2017).** Seq2seq: 문자 임베딩(char-embedding) → BiLSTM 인코더 → 위치 민감 어텐션(location-sensitive attention) → 자기회귀(autoregressive) LSTM 디코더가 멜 프레임을 방출한다. 느리고(AR), 긴 텍스트에서 흔들린다. 여전히 베이스라인으로 인용된다.

**FastSpeech 2(2020).** 비자기회귀(non-autoregressive). 길이 예측기(duration predictor)가 각 음소가 몇 개의 멜 프레임을 받는지 출력한다. 1패스, Tacotron보다 10배 빠르다. 일부 자연스러움(단조 정렬(monotonic alignment))을 잃지만 어디서나 출하된다.

**VITS(2021).** 변분 추론(variational inference)으로 인코더 + 흐름 기반 길이 + HiFi-GAN 보코더를 종단간으로 함께 학습한다. 높은 품질, 단일 모델. 2022–2024년 지배적 오픈소스 TTS. 변형: YourTTS(다중 화자 제로샷(zero-shot)), XTTS v2(2024, Coqui).

**F5-TTS(2024).** 흐름 매칭 위의 확산 트랜스포머(diffusion transformer). 자연스러운 운율, 5초의 참조 오디오로 제로샷 음성 복제(voice cloning). 2026년 오픈소스 TTS 리더보드 정상. 335M 파라미터(parameter).

**Kokoro(2024).** 작고(82M), CPU에서 실행 가능하며, 실시간 사용을 위한 최고 수준의 영어 TTS. 닫힌 어휘 영어 전용, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** 상업적 최첨단. ElevenLabs v2.5의 감정 태그("[whispered]", "[laughing]")와 캐릭터 보이스는 2026년 오디오북 제작을 지배한다.

### 보코더 진화

| 시대 | 보코더 | 지연 시간 | 품질 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 오프라인 전용 | 출시 당시 SOTA |
| 2018 | WaveRNN | ~실시간 | 좋음 |
| 2020 | HiFi-GAN | 100× 실시간 | 거의 인간 수준 |
| 2022 | BigVGAN | 50× 실시간 | 화자/언어에 걸쳐 일반화 |
| 2024 | SNAC, DAC (신경 코덱) | AR 모델과 통합 | 이산 토큰, 비트 효율적 |

2026년 기준 대부분의 "TTS" 모델은 텍스트에서 파형까지 종단간이다; 멜 스펙트로그램은 내부 표현이다.

### 평가

- **MOS(Mean Opinion Score).** 1–5 척도, 크라우드소싱. 여전히 표준이지만; 고통스럽게 느리다.
- **CMOS(Comparative MOS).** A-대-B 선호. 주석당 더 좁은 신뢰 구간.
- **UTMOS, DNSMOS.** 참조 없는(reference-free) 신경 MOS 예측기. 리더보드에 쓰인다.
- **ASR을 통한 CER(Character Error Rate).** TTS 출력을 Whisper에 통과시켜 입력 텍스트에 대한 CER을 계산한다. 명료도(intelligibility)의 대용물.
- **SECS(Speaker Embedding Cosine Similarity).** 음성 복제 품질.

LibriTTS test-clean의 2026년 수치:

| 모델 | UTMOS | CER (Whisper 경유) | 크기 |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## 직접 만들기 (Build It)

### 단계 1: 입력을 음소화한다

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

음소는 보편적인 다리다. VITS 수준에 못 미치는 모델에는 원시 텍스트를 그대로 넣지 마라.

### 단계 2: Kokoro 실행(2026 CPU 기본)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

오프라인, 단일 파일, 82M 파라미터로 돌아간다.

### 단계 3: 음성 복제와 함께 F5-TTS 실행

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5초짜리 참조 클립 + 그 전사를 넘긴다; F5가 운율과 음색(timbre)을 복제한다.

### 단계 4: HiFi-GAN 보코더 밑바닥부터

튜토리얼 스크립트에 담기에는 너무 크지만, 그 형태는 이렇다:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

학습: 적대적(adversarial, 짧은 윈도우에 대한 판별기(discriminator)) + 멜 스펙트로그램 재구성 손실(reconstruction loss) + 특성 매칭 손실(feature-matching loss). 상품화됨: `hifi-gan` 저장소나 nvidia-NeMo의 사전 학습된 체크포인트를 사용하라.

### 단계 5: 전체 파이프라인(의사코드)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 선택 |
|-----------|------|
| 실시간 영어 음성 비서 | Kokoro (CPU) 또는 XTTS v2 (GPU) |
| 5초 참조로부터 음성 복제 | F5-TTS |
| 상업적 캐릭터 보이스 | ElevenLabs v2.5 |
| 오디오북 내레이션 | ElevenLabs v2.5 또는 XTTS v2 + 파인튜닝 |
| 저자원 언어 | 5–20시간 타깃 언어 데이터로 VITS 학습 |
| 표현적 / 감정 태그 | ElevenLabs v2.5 또는 StyleTTS 2 파인튜닝 |

2026년 기준 오픈소스 선두: **품질은 F5-TTS, 효율은 Kokoro**. 역사가가 아니라면 Tacotron에 손대지 마라.

## 함정들 (Pitfalls)

- **텍스트 정규화기 없음.** "Dr. Smith"가 "Doctor"로 읽히는가 "Drive"로 읽히는가? "2026"이 "twenty twenty six"인가 "two zero two six"인가? 음소화기(phonemizer) 전에 정규화하라.
- **OOV 고유명사.** "Ghumare" → "ghyu-mair"? 알려지지 않은 토큰을 위한 폴백 자소-음소 변환(grapheme-to-phoneme) 모델을 출하하라.
- **클리핑(Clipping).** 보코더 출력은 거의 클리핑되지 않지만, 추론 시 멜 스케일링 불일치가 ±1.0을 초과할 수 있다. 항상 `np.clip(wav, -1, 1)`.
- **샘플 레이트 불일치.** Kokoro는 24 kHz를 출력한다; 다운스트림 파이프라인이 16 kHz를 기대하면 → 리샘플하거나 에일리어싱(aliasing)이 생긴다.

## 산출물 (Ship It)

`outputs/skill-tts-designer.md`로 저장하라. 주어진 목소리, 지연 시간, 언어 타깃에 대해 TTS 파이프라인을 설계한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 장난감 어휘로부터 음소 사전을 만들고, 음소당 길이를 추정하고, 가짜 "멜" 일정을 출력한다.
2. **중간.** Kokoro를 설치하고, 같은 문장을 `af_bella`와 `am_adam` 목소리로 합성하라. 오디오 길이와 주관적 품질을 비교하라.
3. **어려움.** 자기 목소리로 5초짜리 참조 클립을 녹음하라. F5-TTS로 그 목소리를 복제하라. 참조와 복제 출력 사이의 SECS를 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 음소(Phoneme) | 소리 단위 | 추상적 소리 클래스; 영어에는 39개(ARPABet). |
| 길이 예측기(Duration predictor) | 각 음소가 얼마나 지속되는가 | 비-AR 모델 출력; 음소당 정수 프레임. |
| 보코더(Vocoder) | 멜 → 파형 | 멜 스펙트로그램을 원시 샘플로 사상하는 신경망. |
| HiFi-GAN | 표준 보코더 | GAN 기반; 2020–2024년 지배적. |
| MOS | 주관적 품질 | 인간 평가자의 1–5 평균 의견 점수. |
| SECS | 음성 복제 지표 | 타깃과 출력 화자 임베딩 사이의 코사인 유사도. |
| F5-TTS | 2024년 오픈소스 SOTA | 흐름 매칭 확산; 제로샷 복제. |
| Kokoro | CPU 영어 선두 | 82M 파라미터 모델, Apache 2.0. |

## 더 읽을거리 (Further Reading)

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884): seq2seq 베이스라인.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103): 종단간 흐름 기반.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885): 현재 오픈소스 SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646): 2026년에도 여전히 출하되는 보코더.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M): 2024년 CPU 친화적 영어 TTS.
