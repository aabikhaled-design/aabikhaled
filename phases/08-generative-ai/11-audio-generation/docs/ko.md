# Audio Generation

> 오디오는 16-48kHz의 1차원 신호다. 5초짜리 클립은 8만-24만 개의 샘플이다. 어떤 트랜스포머(transformer)도 그 시퀀스에 직접 어텐션(attention)하지 않는다. 2026년의 모든 프로덕션(production) 오디오 모델의 해법은 동일하다. 신경 코덱(neural codec)(Encodec, SoundStream, DAC)이 오디오를 50-75Hz의 이산 토큰(token)으로 압축하고, 트랜스포머나 디퓨전(diffusion) 모델이 토큰을 생성한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45분

## 문제 (The Problem)

세 가지 오디오 생성 작업이 있다.

1. **텍스트-투-스피치(Text-to-speech).** 텍스트가 주어지면 음성을 만든다. 깨끗한 음성은 협대역이고 음소 구조가 강하다. 토큰에 대한 트랜스포머로 잘 풀린다. VALL-E(Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **음악 생성(Music generation).** 프롬프트(텍스트, 멜로디, 코드 진행, 장르)가 주어지면 음악을 만든다. 훨씬 넓은 분포다. MusicGen(Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **오디오 효과 / 사운드 디자인(Audio effects / sound design).** 프롬프트가 주어지면 환경음이나 폴리(Foley)를 만든다. AudioGen, AudioLDM 2, Stable Audio Open.

세 가지 모두 같은 기반 위에서 동작한다. 신경 오디오 코덱 + 토큰 자기회귀(token-AR) 또는 디퓨전 생성기.

## 개념 (The Concept)

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 신경 오디오 코덱 (Neural audio codecs)

Encodec(Meta, 2022), SoundStream(Google, 2021), Descript Audio Codec(DAC, 2023). 합성곱(convolution) 인코더(encoder)가 파형을 타임스텝별 벡터(vector)로 압축하고, 잔차 벡터 양자화(residual vector quantization, RVQ)가 각 벡터를 K개 코드북 인덱스의 캐스케이드로 변환한다. 디코더(decoder)가 이를 되돌린다. 75Hz에서 8개의 RVQ 코드북을 쓰는 2kbps의 24kHz 오디오 = 초당 600토큰.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 그 위의 두 가지 생성 패러다임

**토큰 자기회귀(Token-autoregressive).** RVQ 토큰을 시퀀스로 평탄화하고, 디코더 전용(decoder-only) 트랜스포머를 실행한다. MusicGen은 "지연 병렬(delayed parallel)"을 사용해 스트림별 오프셋으로 K개의 코드북 스트림을 병렬로 내보낸다. VALL-E는 텍스트 프롬프트 + 3초 음성 샘플로부터 음성 토큰을 생성한다.

**잠재 디퓨전(Latent diffusion).** 코덱 토큰을 연속 잠재(latent)로 묶거나 범주형 디퓨전으로 모델링한다. Stable Audio 2.5는 연속 오디오 잠재에 플로 매칭(flow matching)을 사용한다. AudioLDM 2는 텍스트-투-멜-투-오디오 디퓨전을 사용한다.

2024-2026년 추세: 음악에서는 플로 매칭이 이기고 있고(더 빠른 추론, 더 깨끗한 샘플), 음성에서는 토큰 자기회귀가 여전히 지배적이다. 자연스럽게 인과적(causal)이고 스트리밍이 잘 되기 때문이다.

## 프로덕션 지형

| 시스템 | 작업 | 백본 | 지연 시간 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | 토큰-AR + 신경 보코더 | ~300ms 첫 토큰 |
| OpenAI GPT-4o audio | 전이중 음성 | 종단간 멀티모달 AR | ~200ms |
| NaturalSpeech 3 | TTS | 잠재 플로 매칭 | 비스트리밍 |
| Stable Audio 2.5 | 음악 / SFX | 오디오 잠재에 DiT + 플로 매칭 | 1분 클립에 ~10초 |
| Suno v4 | 풀송 | 비공개; 토큰-AR 추정 | 곡당 ~30초 |
| Udio v1.5 | 풀송 | 비공개 | 곡당 ~30초 |
| MusicGen 3.3B | 음악 | Encodec 32kHz에 토큰-AR | 실시간 |
| AudioCraft 2 | 음악 + SFX | 플로 매칭 | 5초 클립에 ~5초 |
| Riffusion v2 | 음악 | 스펙트로그램 디퓨전 | ~10초 |

```figure
score-matching
```

## 직접 만들기 (Build It)

`code/main.py`는 핵심 아이디어를 시뮬레이션한다. 두 개의 뚜렷한 "스타일"(스타일 A는 낮은 토큰과 높은 토큰을 번갈아, 스타일 B는 단조로운 램프)에서 생성한 합성 "오디오 토큰" 시퀀스에 작은 다음 토큰 트랜스포머를 학습(training)시킨다. 스타일을 조건으로 주고 샘플링한다.

### Step 1: synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor

스타일을 조건으로 하는 바이그램 스타일 예측기다. 핵심은 패턴이다. 코덱 토큰 → 교차 엔트로피(cross-entropy) 학습 → 자기회귀 샘플링.

### Step 3: sample conditionally

스타일 토큰과 시작 토큰이 주어지면, 예측된 분포에서 다음 토큰을 샘플링한다. 20-40토큰까지 계속한다.

## 함정 (Pitfalls)

- **코덱 품질이 출력 품질의 상한이다.** 코덱이 어떤 소리를 충실히 표현할 수 없으면, 아무리 생성기 품질이 좋아도 소용없다. DAC가 현재 오픈 최고다.
- **RVQ 오차 누적.** 각 RVQ 층(layer)은 이전 층의 잔차를 모델링한다. 1층의 오차가 전파된다. 상위 층에서 온도(temperature) 0으로 샘플링하면 도움이 된다.
- **음악적 구조.** 30초의 토큰은 75Hz에서 2만+ 토큰이다. 트랜스포머에게 어렵다. MusicGen은 슬라이딩 윈도우 + 프롬프트 연장을 쓰고, Stable Audio는 더 짧은 클립 + 크로스페이딩을 쓴다.
- **경계의 아티팩트.** 생성된 클립 간 크로스페이딩에는 신중한 오버랩-애드가 필요하다.
- **깨끗한 데이터 식욕.** 음악 생성기는 수만 시간의 라이선스 음악이 필요하다. Suno / Udio RIAA 소송(2024)이 이를 수면 위로 끌어올렸다.
- **음성 복제 윤리.** 3초 샘플 더하기 텍스트 프롬프트면 VALL-E / XTTS / ElevenLabs가 음성을 복제하기에 충분하다. 모든 프로덕션 모델은 남용 탐지 + 옵트아웃 목록이 필요하다.

## 라이브러리로 써보기 (Use It)

| 작업 | 2026년 스택 |
|------|------------|
| 상업용 TTS | ElevenLabs, OpenAI TTS, 또는 Azure Neural |
| 음성 복제(동의 검증됨) | XTTS v2(오픈) 또는 ElevenLabs Pro |
| 배경 음악, 빠름 | Stable Audio 2.5 API, Suno, 또는 Udio |
| 가사가 있는 음악 | Suno v4 또는 Udio v1.5 |
| 효과음 / 폴리 | AudioCraft 2, ElevenLabs SFX, 또는 Stable Audio Open |
| 실시간 음성 에이전트 | GPT-4o realtime 또는 Gemini Live |
| 오픈 가중치 음악 연구 | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 더빙 / 번역 | HeyGen, ElevenLabs Dubbing |

## 산출물 (Ship It)

`outputs/skill-audio-brief.md`를 저장한다. 이 스킬은 오디오 브리프(작업, 길이, 스타일, 음성, 라이선스)를 받아 다음을 출력한다. 모델 + 호스팅, 프롬프트 형식(장르 태그, 스타일 서술어, 구조 마커), 코덱 + 생성기 + 보코더 체인, 시드 프로토콜, 그리고 평가 계획(MOS / CLAP 점수 / TTS용 CER / 사용자 A/B).

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하고 스타일을 명시적으로 설정하라. 생성된 시퀀스가 그 스타일의 패턴과 일치하는지 검증하라.
2. **보통.** 지연 병렬 디코딩을 추가하라. 1스텝씩 오프셋을 유지해야 하는 2개의 토큰 스트림을 시뮬레이션하라. 결합 예측기를 학습시켜라.
3. **어려움.** HuggingFace transformers를 사용해 MusicGen-small을 로컬에서 실행하라. 세 개의 서로 다른 프롬프트로 10초 클립을 생성하고, 스타일 준수에 대해 A/B하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 코덱(Codec) | "신경 압축" | 오디오용 인코더 / 디코더. 전형적인 출력은 50-75Hz 토큰이다. |
| RVQ | "잔차 VQ" | K개 양자화기의 캐스케이드. 각각 이전 것의 잔차를 모델링한다. |
| 토큰(Token) | "하나의 코덱 심볼" | 코드북으로의 이산 인덱스. 1024 또는 2048이 전형적. |
| 지연 병렬(Delayed parallel) | "오프셋 코드북" | 시퀀스 길이를 줄이기 위해 K개 토큰 스트림을 엇갈린 오프셋으로 내보낸다. |
| 플로 매칭(Flow matching) | "2024년 오디오의 승리" | 디퓨전의 더 곧은 경로 대안. 더 빠른 샘플링. |
| 음성 프롬프트(Voice prompt) | "3초 샘플" | 복제된 음성을 조종하는 화자 임베딩(embedding) 또는 토큰 접두부. |
| 멜 스펙트로그램(Mel spectrogram) | "그 시각화" | 로그 크기 지각 스펙트로그램. 많은 TTS 시스템이 사용한다. |
| 보코더(Vocoder) | "멜에서 파형으로" | 멜 스펙트로그램을 다시 오디오로 변환하는 신경 구성요소. |

## 프로덕션 노트: 오디오는 스트리밍 문제다

오디오는 사용자가 한꺼번에가 아니라 *생성되는 대로* 도착하기를 기대하는 유일한 출력 모달리티다. 프로덕션 관점에서 이는 TPOT(Time Per Output Token)가 중요함을 뜻한다. 사용자의 청취 속도가 목표 처리량(throughput)이지 읽기 속도가 아니기 때문이다. ~75토큰/초로 토큰화된 16kHz 오디오(Encodec)의 경우, 서버는 재생이 매끄럽게 유지되도록 사용자당 ≥75토큰/초를 생성해야 한다.

두 가지 아키텍처적 귀결:

- **플로 매칭 오디오 모델은 쉽게 스트리밍할 수 없다.** Stable Audio 2.5와 AudioCraft 2는 고정된 클립 길이를 한 번의 패스로 렌더링한다. 스트리밍하려면 클립을 청크로 나누고 경계를 겹쳐야 한다. 슬라이딩 윈도우 디퓨전을 생각하라. 코덱 AR 모델 대비 100-300ms의 지연 시간 오버헤드가 추가된다.

제품이 "실시간 음성 채팅"이나 "실시간 음악 연장"이라면 코덱 AR 경로를 택하라. "제출 시 30초 클립 렌더링"이라면 플로 매칭이 품질과 총 지연 시간에서 이긴다.

## 더 읽을거리 (Further Reading)

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438): 코덱 표준.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312): 처음으로 널리 쓰인 신경 오디오 코덱.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5): 플로 매칭을 쓰는 2025년 텍스트-투-뮤직.
