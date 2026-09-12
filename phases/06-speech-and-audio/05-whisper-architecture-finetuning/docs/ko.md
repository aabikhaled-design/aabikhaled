# Whisper: 아키텍처 & 파인튜닝

> Whisper는 30초 윈도우의 트랜스포머(Transformer) 인코더-디코더(encoder-decoder)이며, 68만 시간의 다국어 약한 지도(weakly-supervised) 오디오-텍스트 쌍으로 학습되었다. 하나의 아키텍처, 여러 과제, 99개 언어에 걸친 견고함. 2026년의 레퍼런스 ASR이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75분

## 문제 (The Problem)

OpenAI가 2022년 9월에 출시한 Whisper는 상품(commodity)으로 출하된 최초의 ASR 모델이었다: 오디오를 붙여넣으면 텍스트가 나오고, 99개 언어를 지원하며, 잡음에 견고하고, 노트북에서 돌아간다. 2024년까지 OpenAI는 Large-v3와 Turbo 변형을 출하했고; 2026년 기준 Whisper는 팟캐스트 전사부터 음성 비서, YouTube 자막까지 모든 것의 기본 베이스라인(baseline)이다.

하지만 Whisper는 영원히 블랙박스로 취급할 수 있는 파이프라인(pipeline)이 아니다. 도메인 시프트(domain shift) 앞에서는 무너진다. 기술 전문 용어, 화자 억양, 고유명사, 짧은 클립, 침묵이 그렇다. 다음을 알아야 한다:

1. 내부적으로 그것이 실제로 무엇인지.
2. 청크 단위(chunked), 스트리밍, 또는 장문 오디오를 올바르게 주는 방법.
3. 언제 파인튜닝(fine-tuning)하고 어떻게 하는지.

## 개념 (The Concept)

![Whisper 인코더-디코더, 과제, 청크 단위 추론, 파인튜닝](../assets/whisper.svg)

**아키텍처.** 표준 트랜스포머 인코더-디코더.

- 입력: 30초 로그 멜 스펙트로그램(log-mel spectrogram), 80 멜, 10 ms 홉 → 3000 프레임. 더 짧은 클립은 0으로 패딩되고, 더 긴 클립은 청크화된다.
- 인코더(Encoder): 컨볼루션 다운샘플(conv-downsample, stride 2) + `N`개 트랜스포머 블록. Large-v3의 경우: 32층, 1280차원, 20헤드.
- 디코더(Decoder): 인과적 셀프 어텐션(causal self-attn) + 인코더 출력에 대한 크로스 어텐션(cross-attn)을 갖춘 `N`개 트랜스포머 블록. 인코더와 동일한 크기.
- 출력: 51,865개 토큰 어휘에 대한 BPE 토큰.

Large-v3는 15.5억 파라미터(parameter)를 가진다. Turbo는 (32층에서) 4층 디코더를 사용해, 1% 미만의 WER 손실로 지연 시간(latency)을 8배 줄인다.

**프롬프트 형식.** Whisper는 디코더 프롬프트(prompt)의 특수 토큰으로 조종되는 멀티태스크 모델이다:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>`: 언어 태그; 번역-대-전사 동작을 강제한다.
- `<|transcribe|>` 또는 `<|translate|>`: 임의 언어 입력에서 영어 출력을 번역하거나, 있는 그대로 전사한다.
- `<|notimestamps|>`: 단어 수준 타임스탬프를 건너뛴다(더 빠름).

하나의 모델로 여러 과제를 처리하게 만드는 것이 바로 이 프롬프트다. `<|en|>`을 `<|fr|>`로 바꾸면 프랑스어를 전사한다.

**30초 윈도우.** 모든 것이 30초에 고정된다. 더 긴 클립은 청킹(chunking)이 필요하고; 더 짧은 클립은 패딩된다. 윈도우는 기본적으로 스트리밍되지 않는다. WhisperX, Whisper-Streaming, faster-whisper가 존재하는 이유가 여기에 있다.

**로그 멜 정규화(Log-mel normalization).** `(log_mel - mean) / std`, 여기서 통계는 Whisper 자체의 학습 코퍼스에서 나온다. `librosa.feature.melspectrogram`이 아니라 Whisper의 전처리(`whisper.audio.log_mel_spectrogram`)를 *반드시* 사용해야 한다.

### 2026년의 변형들

| 변형 | 파라미터 | 지연 시간(A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 실시간 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 스트리밍 | 2.0% |

### 파인튜닝

2026년의 정전(正典) 격 워크플로:

1. 정렬된 전사가 있는 타깃 도메인 오디오 10–100시간을 수집한다.
2. `generate_with_loss` 콜백과 함께 `transformers.Seq2SeqTrainer`를 실행한다.
3. 파라미터 효율적: 어텐션 층의 `q_proj`, `k_proj`, `v_proj`에 LoRA를 적용하면 0.3 미만의 WER 비용으로 GPU 메모리를 4배 줄인다.
4. 10시간 미만이면 인코더를 동결한다. 디코더만 튜닝한다.
5. Whisper 자체의 토크나이저(tokenizer)와 프롬프트 형식을 사용하라; 절대 토크나이저를 바꾸지 마라.

커뮤니티 결과: 20시간의 의료 받아쓰기에 Medium을 파인튜닝하면 의료 어휘에 대한 WER이 12%에서 4.5%로 떨어진다. 4시간의 아이슬란드어에 Turbo를 파인튜닝하면 WER이 18%에서 6%로 떨어진다.

```figure
sp-asr-attention
```

## 직접 만들기 (Build It)

### 단계 1: Whisper를 즉시 실행한다

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

항상 재정의해야 할 핵심 기본값: `temperature=0.0`(샘플링은 기본적으로 0.0 → 0.2 → 0.4 … 폴백 체인을 따름), `condition_on_previous_text=False`(연쇄적 환각 문제를 방지), `no_speech_threshold=0.6`(침묵 검출).

### 단계 2: 청크 단위 장문

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX는 (1) Silero VAD 게이팅, (2) wav2vec 2.0을 통한 단어 수준 정렬, (3) `pyannote.audio`를 통한 화자 분리(diarization)를 추가한다. 프로덕션 전사를 위한 2026년의 일꾼이다.

### 단계 3: LoRA로 파인튜닝한다

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

그다음 표준 Trainer 루프. 1000 스텝마다 체크포인트(checkpoint). 홀드아웃(held-out)에서 WER로 평가한다.

### 단계 4: 각 층이 무엇을 학습하는지 검사한다

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

히트맵(heatmap)으로 시각화해 보라. 디코더 스텝이 인코더 프레임을 훑어가면서 대각선 정렬(diagonal alignment)이 나타난다. 그 대각선이 단어 타임스탬프에 대한 Whisper의 개념이다.

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 선택 |
|-----------|------|
| 일반 영어, 오프라인 | `whisperx`를 통한 Large-v3-turbo |
| 모바일 / 엣지 | 양자화된 Whisper-Tiny(int8) 또는 Moonshine |
| 다국어 장문 | `whisperx`를 통한 Large-v3 + 화자 분리 |
| 저자원 언어 | LoRA로 Medium 또는 Turbo 파인튜닝 |
| 스트리밍(2초 지연) | Whisper-Streaming 또는 Parakeet-TDT |
| 단어 수준 타임스탬프 | WhisperX(wav2vec 2.0을 통한 강제 정렬) |

`faster-whisper`(CTranslate2 백엔드)는 2026년 가장 빠른 CPU+GPU 추론 런타임이다. 동일한 출력으로 바닐라보다 4배 빠르다.

## 2026년에도 여전히 출시되는 함정들 (Pitfalls that still ship in 2026)

- **침묵에서의 환각된 텍스트.** 자막으로 학습된 Whisper는 "Thanks for watching!", "Subscribe!", 노래 가사를 포함한다. 호출 전에 항상 VAD로 게이팅하라.
- **`condition_on_previous_text` 연쇄.** 하나의 환각이 후속 윈도우를 오염시킨다. 청크 간 유창함이 필요하지 않다면 `False`로 설정하라.
- **짧은 클립 패딩.** 30초로 패딩된 2초 클립은 뒤따르는 침묵에서 환각할 수 있다. `pad=False`를 쓰거나 VAD로 게이팅하라.
- **잘못된 멜 통계.** Whisper의 멜 대신 librosa의 멜을 쓰면 거의 무작위에 가까운 출력이 나온다. `whisper.audio.log_mel_spectrogram`을 사용하라.

## 산출물 (Ship It)

`outputs/skill-whisper-tuner.md`로 저장하라. 주어진 도메인에 대해 Whisper 파인튜닝 또는 추론 파이프라인을 설계한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. Whisper 스타일 프롬프트를 토큰화하고, 디코드된 형태(shape) 예산을 계산하며, 10분 클립에 대한 청크 일정(chunk schedule)을 출력한다.
2. **중간.** `faster-whisper`를 설치하고, 10분짜리 팟캐스트를 전사하며, 인간 전사본에 대한 WER을 비교하라. `language="auto"` 대 강제 `language="en"`을 시도하라.
3. **어려움.** HF `datasets`를 사용해 Whisper가 어려워하는 언어(예: 우르두어)를 고르고, 2시간 분량에 대해 2 에폭(epoch) 동안 LoRA로 Medium을 파인튜닝하고, WER 변화량을 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 30초 윈도우 | Whisper의 한계 | 강한 입력 상한; 더 긴 오디오는 청크화하라. |
| SOT | 전사 시작(Start-of-transcript) | `<\|startoftranscript\|>`가 디코더 프롬프트를 시작한다. |
| 타임스탬프 토큰 | 시간적 정렬 | 0.02초 오프셋마다 51k 어휘 안의 특수 토큰이다. |
| Turbo | 빠른 변형 | 4층 디코더, 8배 빠름, 1% 미만 WER 퇴보. |
| WhisperX | 장문 래퍼 | VAD + Whisper + wav2vec 정렬 + 화자 분리. |
| LoRA 파인튜닝 | 효율적 튜닝 | 어텐션에 저랭크 어댑터를 추가; 파라미터의 ~0.3%만 학습. |
| 환각(Hallucination) | 조용한 실패 | Whisper가 잡음/침묵에서 유창한 영어를 만들어냄. |

## 더 읽을거리 (Further Reading)

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356): 원조 아키텍처와 학습 레시피.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363): 4층 디코더, 8배 속도 향상.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747): 장문, 단어 정렬, 화자 분리.
- [Systran: faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) CTranslate2 기반, 4배 빠름.
- [HuggingFace: Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) 정전 격 LoRA / 전체 파인튜닝 안내.
