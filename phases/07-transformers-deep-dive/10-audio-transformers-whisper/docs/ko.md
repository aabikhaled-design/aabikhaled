# 오디오 트랜스포머: Whisper 아키텍처

> 오디오는 시간에 따른 주파수의 이미지다. Whisper는 멜 스펙트로그램(mel spectrogram)을 먹고 다시 말로 내뱉는 ViT다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45분

## 문제 (The Problem)

Whisper(OpenAI, Radford et al. 2022) 이전에 최첨단 자동 음성 인식(automatic speech recognition, ASR)은 wav2vec 2.0과 HuBERT를 의미했다. 자기 지도 특성 추출기(feature extractor)에 파인튜닝(fine-tuning)된 헤드를 얹은 형태다. 품질은 높지만 데이터 파이프라인(pipeline)이 비싸고 도메인에 취약했다. 다국어 음성 인식은 언어 계열마다 별도의 모델이 필요했다.

Whisper는 세 가지에 베팅했다:

1. **모든 것으로 학습한다.** 인터넷에서 긁어모은 97개 언어, 680,000시간의 약하게 레이블된(weakly-labeled) 오디오. 깨끗한 학술 코퍼스(corpus)도 없고, 음소(phoneme) 레이블도 없다.
2. **멀티태스크 단일 모델.** 하나의 디코더(decoder)를 작업 토큰(task token)을 통해 전사(transcription), 번역(translation), 음성 활동 감지(voice activity detection), 언어 식별(language ID), 타임스탬프(timestamping)에 대해 공동으로 학습한다.
3. **표준 인코더-디코더 트랜스포머.** 인코더는 로그-멜 스펙트로그램(log-mel spectrogram)을 받는다. 디코더는 텍스트 토큰을 자기회귀적으로(autoregressively) 생성한다. 보코더(vocoder)도, CTC도, HMM도 없다.

결과: Whisper large-v3는 억양, 잡음, 그리고 깨끗한 레이블 데이터가 전혀 없는 언어들에 걸쳐 강건하다. 2026년 모든 오픈소스 음성 비서와 대부분의 상용 음성 비서의 기본 음성 프런트엔드(front-end)다.

## 개념 (The Concept)

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 1단계: 리샘플링(resample) + 윈도잉(window)

16 kHz 오디오. 30초로 자르거나 패딩(pad)한다. 로그-멜 스펙트로그램을 계산한다: 80개 멜 빈(mel bin), 10 ms 스트라이드 → ~3,000 프레임(frame) × 80 특성. 이것이 Whisper가 보는 "입력 이미지"다.

### 2단계: 합성곱 스템(convolutional stem)

커널 3, 스트라이드 2의 Conv1D 층 두 개가 3,000 프레임을 1,500으로 줄인다. 파라미터를 크게 늘리지 않으면서 시퀀스 길이를 절반으로 줄인다.

### 3단계: 인코더

1,500개 타임스텝(timestep)에 대한 24층(large 기준) 트랜스포머 인코더. 사인파 위치 인코딩(positional encoding), 셀프 어텐션(self-attention), GELU FFN. 1,500 × 1,280 은닉 상태를 생성한다.

### 4단계: 디코더

24층 트랜스포머 디코더. GPT-2의 상위 집합(superset)에 몇 개의 오디오 전용 특수 토큰을 더한 BPE 어휘(vocabulary)로부터 토큰을 자기회귀적으로 생성한다.

### 5단계: 작업 토큰

디코더 프롬프트(prompt)는 모델에게 무엇을 할지 알려 주는 제어 토큰으로 시작한다:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

또는

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

모델은 이 관례로 학습되었다. 접두사(prefix)로 작업을 제어한다. 2026년의 명령어 튜닝(instruction-tuning)에 해당하지만, 음성에 적용된 형태다.

### 6단계: 출력

로그 확률(log-prob) 임계값과 함께 빔 서치(beam search, 폭 5). `<|notimestamps|>` 토큰이 없으면 오디오 0.02초마다 타임스탬프를 예측한다.

### Whisper 크기

| 모델 | 파라미터 | 층 | d_model | 헤드 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo(2024)는 디코더를 32층에서 4층으로 줄였다. 디코딩(decoding)이 8배 빨라지면서 WER(단어 오류율) 저하는 1점 미만이다. 이 디코딩 속도 향상이 2026년 실시간 음성 에이전트(agent)에서 Whisper-turbo가 기본값인 이유다.

### Whisper가 하지 않는 것

- 화자 분리(diarization, 누가 말하는지)는 안 한다. 이를 위해서는 pyannote와 함께 써라.
- 네이티브 실시간 스트리밍(streaming)은 안 된다. 30초 윈도우가 고정이다. 최신 래퍼(wrapper, `faster-whisper`, `WhisperX`)는 VAD + 겹침(overlap)을 통해 스트리밍을 덧붙인다.
- 외부 청킹(chunking) 없이는 30초를 넘는 장문 컨텍스트(context)도 없다. 인간의 음성은 전사에 장거리 컨텍스트가 거의 필요 없기 때문에 실제로는 잘 작동한다.

### 2026년 지형

| 작업 | 모델 | 비고 |
|------|-------|-------|
| 영어 ASR | Whisper-turbo, Moonshine | Moonshine은 엣지에서 4배 빠르다 |
| 다국어 ASR | Whisper-large-v3 | 97개 언어 |
| 스트리밍 ASR | faster-whisper + VAD | 150 ms 지연 목표 달성 가능 |
| TTS | Piper, XTTS-v2, Kokoro | 인코더-디코더 패턴이지만 Whisper 형태 |
| 오디오 + 언어 | AudioLM, SeamlessM4T | 텍스트 토큰 + 오디오 토큰을 하나의 트랜스포머에 |

```figure
n5-mel-decode
```

## 직접 만들기 (Build It)

`code/main.py`를 참고하라. Whisper를 학습시키지는 않는다. 로그-멜 스펙트로그램 파이프라인 + 작업 토큰 프롬프트 포매터(formatter)를 만든다. 이것들이 프로덕션(production)에서 실제로 다루는 부분이다.

### 1단계: 오디오 합성

16 kHz로 샘플링된 440 Hz 1초 사인파를 생성한다. 16,000개 샘플.

### 2단계: 로그-멜 스펙트로그램(단순화)

완전한 멜 스펙트로그램에는 FFT가 필요하다. `librosa` 없이도 파이프라인을 보여 주는 단순화된 프레이밍(framing) + 프레임별 에너지 버전을 만든다:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

프레임 = 25 ms, 홉(hop) = 10 ms. Whisper의 윈도잉과 일치한다. 교육 목적상 프레임별 에너지가 멜 빈을 대신한다.

### 3단계: 30초로 패딩

Whisper는 항상 30초 청크를 처리한다. 스펙트로그램을 3,000 프레임으로 패딩(또는 자르기)한다.

### 4단계: 프롬프트 토큰 만들기

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

이것이 작업 제어 표면(surface)의 전부다. 4토큰짜리 접두사.

## 라이브러리로 써보기 (Use It)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

더 빠르고 OpenAI 호환:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026년에 Whisper를 선택할 때:**

- 하나의 모델로 다국어 ASR.
- 잡음이 많고 다양한 오디오의 강건한 전사.
- 연구 / 프로토타입 ASR: 가장 빠른 출발점.

**다른 것을 선택할 때:**

- 엣지에서의 초저지연 스트리밍: 동일 품질에서 Moonshine이 Whisper를 이긴다.
- <200 ms가 필요한 실시간 대화형 AI: 전용 스트리밍 ASR.
- 화자 분리: Whisper는 안 한다; pyannote를 덧붙여라.

## 산출물 (Ship It)

`outputs/skill-asr-configurator.md`를 참고하라. 이 스킬은 새 음성 애플리케이션에 대해 ASR 모델, 디코딩 파라미터, 전처리 파이프라인을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 16 kHz에서 10 ms 홉으로 1초 신호의 프레임 수가 ~100 프레임인지 확인하라. 30초의 경우: ~3,000 프레임.
2. **중간.** `numpy.fft`를 사용해 완전한 로그-멜 스펙트로그램을 만들어라. 80개 멜 빈이 수치 오차 이내에서 `librosa.feature.melspectrogram(n_mels=80)`과 일치하는지 확인하라.
3. **어려움.** 스트리밍 추론(inference)을 구현하라: 오디오를 2초 겹침을 두고 10초 윈도우로 청킹하고, 각 청크에서 Whisper를 실행한 뒤, 전사 결과를 병합하라. 5분짜리 팟캐스트 샘플에서 단일 패스 대비 단어 오류율을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 멜 스펙트로그램(Mel spectrogram) | "오디오 이미지" | 2D 표현: 한 축에는 주파수 빈, 다른 축에는 시간 프레임; 셀마다 로그 스케일된 에너지. |
| 로그-멜(Log-mel) | "Whisper가 보는 것" | 로그를 통과시킨 멜 스펙트로그램; 인간의 소리 크기 지각을 근사한다. |
| 프레임(Frame) | "하나의 시간 조각" | 25 ms 샘플 윈도우; 10 ms 스트라이드로 겹친다. |
| 작업 토큰(Task token) | "음성용 프롬프트 접두사" | 디코더 프롬프트의 `<\|transcribe\|>` / `<\|translate\|>` 같은 특수 토큰. |
| 음성 활동 감지(Voice activity detection, VAD) | "음성을 찾기" | ASR 전에 침묵을 제거하는 게이트; 비용을 대폭 절감한다. |
| CTC | "Connectionist Temporal Classification" | 정렬 없는 학습을 위한 고전적 ASR 손실(loss); Whisper는 이를 사용하지 않는다. |
| Whisper-turbo | "작은 디코더, 완전한 인코더" | large-v3 인코더 + 4층 디코더; 디코딩 8배 빠름. |
| Faster-whisper | "프로덕션 래퍼" | CTranslate2 재구현; int8 양자화(quantization); OpenAI 레퍼런스보다 4배 빠름. |

## 더 읽을거리 (Further Reading)

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): Whisper 논문.
- [OpenAI Whisper repo](https://github.com/openai/whisper): 레퍼런스 코드 + 모델 가중치(weight). `whisper/model.py`를 읽으면 Conv1D 스템 + 인코더 + 디코더를 ~400줄 안에서 위에서 아래로 볼 수 있다.
- [OpenAI Whisper: `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) 5~6단계에서 설명한 빔 서치 + 작업 토큰 로직이 여기 있다; 500줄, 완전히 읽을 만하다.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477): 선행 연구; 일부 환경에서는 여전히 SOTA 특성.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper): 프로덕션 래퍼, 레퍼런스보다 4배 빠름.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608): 2024년의 엣지 친화적 ASR, Whisper 형태이지만 더 작다.
- [HuggingFace blog: "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) 멜 스펙트로그램 전처리기와 토큰-타임스탬프 처리를 포함한 표준 파인튜닝 레시피.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py): 이 레슨의 아키텍처 다이어그램을 그대로 따르는 전체 구현(인코더, 디코더, 교차 어텐션(cross-attention), 생성).
