# 음악 생성: MusicGen, Stable Audio, Suno, 그리고 라이선싱 지진

> 2026년 음악 생성: Suno v5와 Udio v4가 상업적으로 지배하고; MusicGen, Stable Audio Open, ACE-Step이 오픈소스를 이끈다. 기술적 문제는 대부분 해결되었다. 법적 문제(Warner Music 5억 달러 합의, UMG 합의)가 2025-2026년에 이 분야를 재편했다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75분

## 문제 (The Problem)

텍스트 → 가사, 보컬, 구조를 갖춘 30초에서 4분짜리 음악 클립. 세 가지 하위 문제:

1. **악기 생성(Instrumental generation).** "따뜻한 건반이 들어간 로파이 힙합 드럼" 같은 텍스트 → 오디오. MusicGen, Stable Audio, AudioLDM.
2. **노래 생성(보컬 + 가사 포함).** "비 오는 텍사스 밤에 관한 컨트리 노래" → 완성된 곡. Suno, Udio, YuE, ACE-Step.
3. **조건부 / 제어 가능.** 기존 클립을 확장하거나, 브릿지를 재생성하거나, 장르를 바꾸거나, 스템(stem) 분리하거나, 인페인팅(inpaint)한다. Udio의 인페인팅 + 스템 분리가 2026년에 맞춰야 할 기능이다.

## 개념 (The Concept)

![음악 생성: 토큰-LM 대 확산, 2026년 모델 지도](../assets/music-generation.svg)

### 신경 코덱 토큰에 대한 토큰 LM

Meta의 **MusicGen**(2023, MIT)과 여러 파생물이 여기에 속한다. 텍스트/멜로디 임베딩(embedding)에 조건화해 EnCodec 토큰(token)(32 kHz, 4개 코드북(codebook))을 자기회귀적(autoregressive)으로 예측하고, EnCodec으로 디코드한다. 300M - 3.3B 파라미터(parameter)다. 강한 베이스라인(baseline)이지만 30초를 넘기면 버거워한다.

**ACE-Step**(오픈소스, 2026년 4월 출시된 4B XL)은 이것을 완성곡 가사 조건부 생성으로 확장한다. 오픈 커뮤니티가 Suno에 가장 가까이 다가간 것.

### 멜 또는 잠재 표현에 대한 확산

**Stable Audio(2023)**와 **Stable Audio Open(2024)**: 압축된 오디오에 대한 잠재 확산(latent diffusion). 루프(loop), 사운드 디자인, 앰비언트 텍스처에 탁월하다. 구조화된 완성곡에는 별로다.

**AudioLDM / AudioLDM2**: T2I 스타일 잠재 확산을 통한 텍스트-오디오 변환, 음악, 음향 효과, 음성으로 일반화됨.

### 하이브리드(프로덕션): Suno, Udio, Lyria

닫힌 가중치(closed weights). 아마도 AR 코덱 LM + 전용 보이스 / 드럼 / 멜로디 헤드를 갖춘 확산 기반 보코더(vocoder). Suno v5(2026)는 ELO 1293 품질 선두다. Udio v4는 인페인팅 + 스템 분리(베이스, 드럼, 보컬을 개별 다운로드)를 추가한다.

### 평가

- **FAD(Fréchet Audio Distance).** VGGish나 PANNs 특성으로 잰 생성 오디오와 실제 오디오 분포 사이의 임베딩 수준 거리. 낮을수록 좋다. MusicGen small은 MusicCaps에서 4.5 FAD, SOTA는 ~3.0.
- **음악성(주관적).** 인간 선호도. Suno v5 ELO 1293이 선두.
- **텍스트-오디오 정렬.** 프롬프트(prompt)와 출력 사이의 CLAP 점수.
- **음악성 아티팩트.** 박자에서 벗어난 전환, 보컬 프레이즈 드리프트(drift), 30초 넘어 구조 손실.

## 2026년 모델 지도

| 모델 | 파라미터 | 길이 | 보컬 | 라이선스 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | 없음 | MIT |
| Stable Audio Open | 1.2B | 47 s | 없음 | Stability 비상업 |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | 있음 | Apache-2.0 |
| YuE | 7B | &gt; 2 min | 있음, 다국어 | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | 있음, ELO 1293 | 상업 |
| Udio v4 (closed) | ? | 4 min | 있음 + 스템 | 상업 |
| Google Lyria 3 (closed) | ? | 실시간 | 있음 | 상업 |
| MiniMax Music 2.5 | ? | 4 min | 있음 | 상업 API |

## 법적 지형 (2025-2026)

- **Warner Music 대 Suno 합의.** 5억 달러. 이제 WMG는 Suno에서 AI-유사성(AI-likeness), 음악 권리, 사용자 생성 트랙을 감독한다. Udio를 상대로 한 UMG 합의도 비슷하다.
- **EU AI Act** + **California SB 942**: AI 생성 음악은 공개되어야 한다.
- **Riffusion / MusicGen**은 MIT 하에서 컴플라이언스 부담이 없지만 상업적 보컬도 없다.

출하 안전한 패턴:

1. 악기만 생성한다(MusicGen, Stable Audio Open, MIT/CC0 출력).
2. 생성당 라이선스를 갖춘 상업 API(Suno, Udio, ElevenLabs Music)를 사용한다.
3. 소유하거나 라이선스받은 카탈로그로 학습한다(대부분의 기업이 결국 여기에 도달한다).
4. 생성물을 워터마크(watermark) + 메타데이터로 태그한다.

```figure
sp-codec-tokens
```

## 직접 만들기 (Build It)

### 단계 1: MusicGen으로 생성한다

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

세 가지 크기: `small`(300M, 빠름), `medium`(1.5B), `large`(3.3B). "아이디어가 통하는가"를 보기에는 small로 충분하다.

### 단계 2: 멜로디 조건화

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody는 크로마그램(chromagram)을 받아 음색(timbre)을 바꾸면서 곡조를 보존한다. "이 멜로디를 현악 사중주로 줘"에 유용하다.

### 단계 3: FAD 평가

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish 임베딩 거리를 계산한다. 장르 수준의 회귀(regression) 테스트에는 유용하지만 인간 청취자를 대체하지는 않는다.

### 단계 4: LLM-음악 워크플로에 추가하기

Lesson 7-8의 아이디어와 결합한다:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 라이브러리로 써보기 (Use It)

| 목표 | 스택 |
|------|-------|
| 악기 사운드 디자인 | Stable Audio Open |
| 게임 / 적응형 음악 | Google Lyria RealTime (closed) |
| 보컬이 있는 완성곡(상업) | 명시적 라이선스를 갖춘 Suno v5 또는 Udio v4 |
| 보컬이 있는 완성곡(오픈) | ACE-Step XL 또는 YuE |
| 짧은 광고 징글 | 흥얼거린 참조에 멜로디 조건화된 MusicGen |
| 뮤직비디오 배경 | MusicGen + Stable Video Diffusion |

## 2026년에도 여전히 출시되는 함정들 (Pitfalls that still ship in 2026)

- **저작권 세탁 프롬프트.** "Taylor Swift 스타일의 노래": 상업용 Suno/Udio는 이제 이를 필터링하지만 오픈 모델은 그러지 않는다. 자체 필터 목록을 따로 두라.
- **30초 넘는 반복 / 드리프트.** AR 모델은 루핑한다. 여러 생성물을 크로스페이드(crossfade)하거나, 구조적 일관성을 위해 ACE-Step을 사용하라.
- **템포 드리프트.** 모델이 BPM에서 벗어난다. 프롬프트에 BPM 태그를 쓰고 librosa의 `beat_track`으로 후처리 필터링하라.
- **보컬 명료도.** Suno는 탁월하다; 오픈 모델은 흔히 단어에서 뭉개진다. 가사가 중요하면 상업 API를 쓰거나 파인튜닝(fine-tuning)하라.
- **모노 출력.** 오픈 모델은 모노 또는 가짜 스테레오를 생성한다. 적절한 스테레오 재구성(ezst, Cartesia의 스테레오 확산)으로 업그레이드하라.

## 산출물 (Ship It)

`outputs/skill-music-designer.md`로 저장하라. 음악 생성 배포를 위해 모델, 라이선스 전략, 길이 / 구조 계획, 공개 메타데이터를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. "생성적" 코드 진행 + 드럼 패턴을 ASCII 기호로 만든다. 음악 생성 만화다. 원한다면 아무 MIDI 렌더러로 재생해 보라.
2. **중간.** `audiocraft`를 설치하고, MusicGen-small로 4개 장르 프롬프트에 걸쳐 10초짜리 클립을 생성하고, 참조 장르 세트에 대한 FAD를 측정하라.
3. **어려움.** ACE-Step(또는 MusicGen-melody)을 사용해 서로 다른 음색 프롬프트로 같은 곡조의 세 변형을 생성하라. 정렬을 검증하기 위해 프롬프트에 대한 CLAP 유사도를 계산하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| FAD | 오디오 FID | 실제 대 생성의 임베딩 분포 사이의 프레셰 거리(Fréchet distance). |
| 크로마그램(Chromagram) | 음높이로서의 멜로디 | 12차원 프레임별 벡터; 멜로디 조건화의 입력. |
| 스템(Stems) | 악기 트랙 | 분리된 베이스 / 드럼 / 보컬 / 멜로디를 WAV로. |
| 인페인팅(Inpainting) | 한 구간 재생성 | 시간 윈도우를 마스킹; 모델이 그 부분만 재생성. |
| CLAP | 텍스트-오디오 CLIP | 대조적 오디오-텍스트 임베딩; 텍스트-오디오 정렬을 평가. |
| EnCodec | 음악 코덱 | MusicGen이 사용하는 Meta의 신경 코덱; 32 kHz, 4개 코드북. |

## 더 읽을거리 (Further Reading)

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284): 오픈 자기회귀 벤치마크.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358): 사운드 디자인 기본.
- [ACE-Step](https://github.com/ace-step/ACE-Step): 오픈 4B 완성곡 생성기, 2026년 4월.
- [Suno v5 platform docs](https://suno.com): 상업적 품질 선두.
- [AudioLDM2](https://arxiv.org/abs/2308.05734): 음악 + 음향 효과를 위한 잠재 확산.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/): 2025년 11월 선례.
