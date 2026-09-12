# 음성 인식(ASR): CTC, RNN-T, 어텐션

> 음성 인식(speech recognition)은 매 타임스텝(timestep)마다 오디오를 분류하는 작업이고, 여기에 영어와 침묵을 아는 시퀀스 모델(sequence model)을 이어 붙인 것이다. 이를 해내는 방법은 CTC, RNN-T, 어텐션(attention) 세 가지다. 하나를 고르고 왜 그런지 이해하라.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45분

## 문제 (The Problem)

10초짜리 16 kHz 클립이 있고, 여기서 "turn on the kitchen lights"라는 문자열을 얻고 싶다. 어려움은 구조에서 온다. 오디오 프레임(frame)은 문자와 일대일로 정렬되지 않는다. "okay"라는 단어는 200 ms가 걸릴 수도 1200 ms가 걸릴 수도 있다. 침묵이 발화(utterance)에 구두점을 찍는다. 어떤 음소(phoneme)는 다른 것보다 길다. 출력 토큰(token)의 개수는 미리 알 수 없다.

세 가지 공식화가 이를 해결한다:

1. **CTC (Connectionist Temporal Classification).** 특수한 *blank*를 포함해 프레임별 토큰 확률을 방출한다. 디코드(decode) 시점에 반복과 blank를 합친다. 비자기회귀적(non-autoregressive)이고 빠르다. wav2vec 2.0, MMS가 사용한다.
2. **RNN-T (Recurrent Neural Network Transducer).** 결합 네트워크(joint network)가 인코더(encoder) 프레임과 이전 토큰이 주어졌을 때 다음 토큰을 예측한다. 스트리밍 가능. 구글의 온디바이스(on-device) ASR, NVIDIA Parakeet가 사용한다.
3. **어텐션 인코더-디코더(Attention encoder-decoder).** 인코더가 오디오를 은닉 상태(hidden state)로 압축하고, 디코더(decoder)가 크로스 어텐션(cross-attend)으로 토큰을 자기회귀적(autoregressive)으로 생성한다. Whisper, SeamlessM4T가 사용한다.

2026년에 LibriSpeech test-clean의 SOTA WER은 1.4%(Parakeet-TDT-1.1B, NVIDIA)와 1.58%(Whisper-Large-v3-turbo)다. 차이는 미미하지만; 배포(deployment) 차이는 막대하다.

## 개념 (The Concept)

![세 가지 ASR 공식화: CTC, RNN-T, 어텐션 인코더-디코더](../assets/asr-formulations.svg)

**CTC 직관.** 인코더가 `V+1`개 토큰(V개 문자 + blank)에 대한 `T`개의 프레임 수준 분포를 출력하게 한다. 길이 `U < T`인 타깃 문자열 `y`에 대해, `y`로 합쳐지는 모든 프레임 정렬(alignment)이 인정된다. CTC 손실(loss)은 그러한 모든 정렬을 합산한다. 추론: 프레임별 argmax, 반복 합치기, blank 제거.

장점: 비자기회귀적, 스트리밍 가능, 룩어헤드(lookahead) 없음. 단점: *조건부 독립성 가정(conditional independence assumption)*: 각 프레임 예측이 서로 독립적이라 내부 언어 모델(language model)이 없다. 빔 서치(beam search)나 얕은 융합(shallow fusion)을 통한 외부 LM으로 해결한다.

**RNN-T 직관.** 토큰 이력을 임베딩하는 *예측기(predictor)* 네트워크와, 예측기 상태를 인코더 프레임과 결합해 `V+1`(여기서 `+1`은 null / 비방출(no-emit))에 대한 결합 분포(joint distribution)로 만드는 *결합기(joiner)*를 추가한다. CTC가 무시한 조건부 의존성을 명시적으로 모델링한다. 각 스텝이 과거 프레임과 과거 토큰에만 조건화되므로 스트리밍 가능하다.

장점: 스트리밍 가능 + 내부 LM. 단점: 학습이 더 복잡하고 메모리를 많이 먹는다(3D 손실 격자(loss lattice)); RNN-T 손실 커널은 그 자체로 하나의 라이브러리 범주다.

**어텐션 인코더-디코더.** 로그 멜(log-mel) 프레임에 대한 인코더(6-32개 트랜스포머(Transformer) 층). 디코더(6-32개 트랜스포머 층)가 인코더 출력에 크로스 어텐션해 토큰을 자기회귀적으로 생성한다. 정렬 제약이 없다. 어텐션은 오디오의 어디든 볼 수 있다. 어텐션을 제한하지 않는 한 스트리밍은 불가능하다(청크 단위 Whisper-Streaming, 2024).

장점: 오프라인 ASR에서 최고 품질, 표준 seq2seq 도구로 학습하기 쉬움. 단점: 자기회귀 지연 시간(latency)이 출력 길이에 비례한다; 엔지니어링 없이는 스트리밍 불가.

### WER: 그 하나의 숫자

**단어 오류율(Word Error Rate)** = `(S + D + I) / N`, 여기서 S=대체(substitution), D=삭제(deletion), I=삽입(insertion), N=참조 단어 수. 단어 수준에서 레벤슈타인 편집 거리(Levenshtein edit distance)와 일치한다. 낮을수록 좋다. 20%를 넘는 WER은 일반적으로 사용 불가; 5% 아래는 읽기 음성에서 인간 수준이다. 표준 벤치마크의 2026년 수치:

| 모델 | LibriSpeech test-clean | LibriSpeech test-other | 크기 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

이들 모두 인코더-디코더 또는 RNN-T 기반이다. 순수 CTC 시스템(wav2vec 2.0)은 test-clean에서 1.8–2.1% 근처에 있다.

```figure
ctc-collapse
```

## 직접 만들기 (Build It)

### 단계 1: 그리디 CTC 디코드

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

규칙은 두 가지다. 연속된 반복을 합치고, blank를 버린다. 예시: `a a _ _ a b b _ c` → `a a b c`.

### 단계 2: 빔 서치 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

프로덕션은 LM 융합을 곁들인 접두사 트리 빔 서치(prefix tree beam search)를 사용한다; 이것은 개념적 골격이다.

### 단계 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 단계 4: Whisper로 추론하기

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026년 가장 강력한 범용 ASR을 한 줄로 돌리는 코드다. 24 GB GPU에서 실시간의 ~20배 속도로 돌아간다.

### 단계 5: Parakeet 또는 wav2vec 2.0로 스트리밍

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

스트리밍 ASR은 청크 단위 인코더 어텐션과 상태 이월(carryover state)이 필요하다; 이를 지원하는 라이브러리를 사용하라(Parakeet용 NeMo, `chunk_length_s`를 가진 `transformers` 파이프라인).

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 오프라인, 최대 품질 | Whisper-large-v3-turbo |
| 다국어, 견고함 | SeamlessM4T v2 |
| 스트리밍, 저지연 | Parakeet-TDT-1.1B 또는 Riva |
| 엣지, 모바일, <500 ms 지연 | 양자화된 Whisper-Tiny 또는 Moonshine (2024) |
| 장문 | VAD 기반 청킹을 쓴 Whisper (WhisperX) |
| 도메인 특화 (의료, 법률) | wav2vec 2.0 파인튜닝 + 도메인 LM 융합 |

## 2026년에도 여전히 출시되는 함정들 (Pitfalls that still ship in 2026)

- **VAD 없음.** 침묵에 Whisper를 돌리면 환각("Thanks for watching!")이 생긴다. 항상 VAD로 게이팅(gate)하라.
- **문자 대 단어 대 서브워드 WER.** 정규화(소문자화, 구두점 제거)를 *마친 뒤*에 단어 수준 WER을 보고하라.
- **언어 식별 드리프트(drift).** Whisper의 자동 LID는 잡음이 많은 클립을 일본어나 웨일스어로 잘못 라우팅한다; 알고 있을 때는 `language="en"`을 강제하라.
- **청킹 없는 긴 클립.** Whisper는 30초 윈도우를 가진다. 그보다 긴 것에는 `chunk_length_s=30, stride=5`를 사용하라.

## 산출물 (Ship It)

`outputs/skill-asr-picker.md`로 저장하라. 주어진 배포 타깃에 대해 모델, 디코딩 전략, 청킹, LM 융합을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 손으로 만든 CTC 출력을 그리디하게 디코드하고 참조에 대한 WER을 계산한다.
2. **중간.** 단계 2의 접두사 트리 빔 서치를 제대로 구현하라(blank 병합 규칙을 고려할 것). 10개 예제 합성 데이터셋에서 그리디와 비교하라.
3. **어려움.** [LibriSpeech test-clean](https://www.openslr.org/12)에서 `whisper-large-v3-turbo`를 사용하라. 처음 100개 발화에 대해 WER을 계산하라. 발표된 수치와 비교하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| CTC | blank 토큰 손실 | 모든 프레임-토큰 정렬에 대한 주변화(marginal); 비자기회귀. |
| RNN-T | 스트리밍 손실 | CTC + 다음 토큰 예측기; 어순을 처리함. |
| 어텐션 enc-dec | Whisper 스타일 | 인코더 + 크로스 어텐션 디코더; 최고의 오프라인 품질. |
| WER | 보고하는 숫자 | 단어 수준의 `(S+D+I)/N`. |
| Blank | 그 공백 | "이 프레임에서 방출 없음"을 신호하는 CTC의 특수 토큰. |
| LM 융합 | 외부 언어 모델 | 빔 서치 중에 가중된 LM 로그 확률을 더함. |
| VAD | 침묵 게이트 | 음성 활동 검출기(voice activity detector); 비음성을 잘라냄. |

## 더 읽을거리 (Further Reading)

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf): CTC 논문.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711): RNN-T 논문.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): 2022년의 정전 격 논문; 2024년 v3-turbo 확장.
- [NVIDIA NeMo: Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) 2026년 Open ASR Leaderboard 선두.
- [Hugging Face: Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 25개 이상 모델에 걸친 라이브 벤치마크.
