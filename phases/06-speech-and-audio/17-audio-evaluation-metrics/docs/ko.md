# 오디오 평가(Audio Evaluation): WER, MOS, UTMOS, MMAU, FAD, 그리고 오픈 리더보드

> 측정할 수 없는 것은 출시할 수 없다. 이 레슨은 모든 오디오 과제에 대한 2026년 지표를 짚는다. ASR(WER, CER, RTFx), TTS(MOS, UTMOS, SECS, ASR 왕복 WER), 오디오-언어(MMAU, LongAudioBench), 음악(FAD, CLAP), 화자(EER). 그리고 비교가 이루어지는 리더보드들까지.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60분

## 문제 (The Problem)

모든 오디오 과제에는 여러 지표가 있고, 각각 다른 축을 측정한다. 잘못된 지표를 쓰면 대시보드에서는 훌륭해 보이지만 프로덕션(production)에서는 형편없는 모델을 출시하게 된다. 2026년 표준(canonical) 목록은 다음과 같다.

| 과제 | 주요 | 보조 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 개념 (The Concept)

![오디오 평가 매트릭스: 지표 vs 과제 vs 2026 리더보드](../assets/eval-landscape.svg)

### ASR 지표

**WER(Word Error Rate, 단어 오류율).** `(S + D + I) / N`. 채점 전에 소문자화하고, 구두점을 제거하고, 숫자를 정규화한다. `jiwer` 또는 OpenAI의 `whisper_normalizer`를 쓴다. &lt; 5% = 인간 수준의 낭독 음성.

**CER(Character Error Rate, 문자 오류율).** 같은 공식을 문자 단위로. 단어 분절이 모호한 성조 언어(만다린, 광둥어)에 쓴다.

**RTFx(역 실시간 계수).** 실제 경과 시간 1초당 처리되는 오디오 길이(초). 높을수록 좋다. Parakeet-TDT는 3380배에 도달한다. Whisper-large-v3은 약 30배다.

**첫 토큰 지연 시간(First-token latency).** 오디오 입력부터 첫 전사 토큰까지 실제 경과한 시간. 스트리밍에 결정적이다. Deepgram Nova-3: 약 150ms.

### TTS 지표

**MOS(Mean Opinion Score, 평균 의견 점수).** 1-5점으로 매기는 인간 평가. 가장 신뢰받는 기준이지만 느리다. 샘플당 청취자 20명 이상, 모델당 샘플 100개 이상을 모은다.

**UTMOS(2022-2026).** 학습된 MOS 예측기. 표준 벤치마크에서 인간 MOS와 약 0.9 상관한다. F5-TTS: UTMOS 3.95. 정답(ground truth): 4.08.

**SECS(Speaker Encoder Cosine Similarity, 화자 인코더 코사인 유사도).** 음성 복제용. 레퍼런스와 복제 출력 간의 ECAPA 임베딩(embedding) 코사인. &gt; 0.75 = 알아볼 수 있는 복제.

**ASR 왕복 WER(WER-on-ASR-round-trip).** TTS 출력에 Whisper를 돌려 입력 텍스트 대비 WER를 계산한다. 명료도 퇴행을 잡는다. 2026년 SOTA: &lt; 2% CER.

**TTFA(time-to-first-audio).** 실제 시계 지연 시간. Kokoro-82M: 약 100ms. F5-TTS: 약 1초.

### 음성 복제 특화

**SECS + MOS + CER** 세 가지를 함께 본다. SECS는 높은데 MOS가 낮은 복제는 음색은 맞지만 부자연스럽다는 뜻이다. 반대라면 음성은 자연스러우나 화자가 틀린 것이다.

### 화자 검증

**EER(Equal Error Rate, 등오류율).** 거짓 수락률(False Accept Rate)이 거짓 거부율(False Reject Rate)과 같아지는 임계값. VoxCeleb1-O에서의 ECAPA: 0.87%.

**minDCF(min Detection Cost).** 선택된 작동점(보통 FAR=0.01)에서의 가중 비용. EER보다 프로덕션에 더 적합하다.

### 화자 분리

**DER(Diarization Error Rate, 화자 분리 오류율).** `(FA + Miss + Confusion) / total_speaker_time`. 놓친 음성 + 거짓 알람 음성 + 화자 혼동, 각각을 비율로. AMI 회의: DER 약 10-20%가 현실적이다. pyannote 3.1 + Precision-2 상용: 잘 녹음된 오디오에서 &lt;10% DER.

**JER(Jaccard Error Rate, 자카드 오류율).** DER의 대안. 짧은 세그먼트 편향에 강인하다.

### 오디오 분류

다중 레이블: 모든 클래스에 대한 **mAP(mean Average Precision)**. AudioSet: BEATs-iter3이 0.548 mAP.

다중 클래스 배타적: **top-1, top-5 정확도**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

불균형: **매크로 F1** + **클래스별 재현율**. 반드시 클래스별로 보고하라. 집계 정확도는 어느 클래스가 실패하는지를 가린다.

### 음악 생성

**FAD(Fréchet Audio Distance, 프레셰 오디오 거리).** 실제 대 생성 오디오의 VGGish 임베딩 분포 간 거리. MusicCaps에서의 MusicGen-small: 4.5. MusicLM: 4.0. 낮을수록 좋다.

**CLAP 점수.** CLAP 임베딩을 사용한 텍스트-오디오 정렬 점수. &gt; 0.3 = 합리적 정렬.

**청취 패널 MOS.** 소비자급 음악에서는 여전히 최종 판단 기준이다. TTS Arena에서 Suno v5의 ELO는 1293이다(쌍별 인간 선호 기반).

### 오디오-언어 벤치마크

**MMAU(Massive Multi-Audio Understanding).** 1만 개의 오디오-QA 쌍.

**MMAU-Pro.** 1800개의 어려운 항목, 네 카테고리: 음성 / 소리 / 음악 / 다중 오디오. 4지선다에서 무작위 확률 25%. Gemini 2.5 Pro 전체 약 60%. 다중 오디오는 모든 모델에서 약 22%.

**LongAudioBench.** 의미 기반 질의를 동반한 수 분 길이 클립. Audio Flamingo Next가 Gemini 2.5 Pro를 능가한다.

**AudioCaps / Clotho.** 캡셔닝 벤치마크. SPICE, CIDEr, FENSE 지표.

### 스트리밍 음성-투-음성

**지연 시간 P50 / P95 / P99.** 사용자 음성이 끝난 뒤 첫 가청 응답까지 실제 경과한 시간. Moshi: 200ms. GPT-4o Realtime: 300ms.

출력에 대한 **WER / MOS**.

**끼어들기 반응성(Barge-in responsiveness).** 사용자 인터럽트부터 비서 음소거까지의 시간. 목표 &lt; 150ms.

### 2026년 리더보드

| 리더보드 | 트랙 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 직접 만들기 (Build It)

### 1단계: 정규화를 동반한 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 2단계: TTS 왕복 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 3단계: 음성 복제용 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 4단계: 음악 생성용 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 5단계: 화자 검증용 EER (레슨 6과 동일한 코드)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 라이브러리로 써보기 (Use It)

모델을 업데이트할 때마다 같은 평가 하니스를 고정해 함께 돌려라. 세 가지 기본 규칙은 다음과 같다.

1. **채점 전에 정규화하라.** 소문자화, 구두점 제거, 숫자 확장. 정규화 규칙을 보고하라.
2. **평균이 아니라 분포를 보고하라.** 지연 시간은 P50/P95/P99. 분류는 클래스별 재현율. MMAU는 카테고리별.
3. **표준 공개 벤치마크를 하나는 돌려라.** 프로덕션 데이터가 다르더라도 Open ASR / TTS Arena / MMAU로 보고하면 리뷰어가 동일 기준으로 비교할 수 있다.

## 함정 (Pitfalls)

- **UTMOS 외삽.** VCTK 스타일의 깨끗한 음성으로 학습됨. 잡음이 있거나 / 복제되거나 / 감정적인 오디오를 잘못 채점한다.
- **MOS 패널 편향.** Amazon Mechanical Turk 작업자 20명 ≠ 목표 사용자 20명. 위험이 크면 도메인 패널에 비용을 지불하라.
- **FAD는 레퍼런스 셋에 의존한다.** 모델 전반에 걸쳐 동일한 레퍼런스 분포와 비교하라.
- **집계 WER.** 전체 5% WER가 억양 있는 음성에서의 30% WER를 가릴 수 있다. 인구통계 슬라이스별로 보고하라.
- **공개 벤치마크 포화(saturation).** 대부분의 프런티어 모델은 표준 벤치마크에서 이미 천장 근처에 있다. 자체 트래픽을 반영하는 인하우스 홀드아웃(held-out) 셋을 따로 만들어라.

## 산출물 (Ship It)

`outputs/skill-audio-evaluator.md`로 저장한다. 어떤 오디오 모델 릴리스에 대해서든 지표, 벤치마크, 보고 형식을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 장난감 입력에 대해 WER / CER / EER / SECS / FAD 유사 / MMAU 유사를 계산한다.
2. **보통.** TTS 왕복 WER 하니스를 만든다. Kokoro 또는 F5-TTS 출력을 Whisper에 통과시킨다. 50개 프롬프트(prompt)에 대해 WER를 계산한다. WER &gt; 10%인 프롬프트를 표시한다.
3. **어려움.** 레슨 10에서 고른 LALM을 MMAU-Pro 음성 + 다중 오디오 서브셋(각 50개)에서 채점한다. 카테고리별 정확도를 보고하고 발표된 수치와 비교한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| WER | ASR 점수 | 정규화 후 단어 단위 `(S+D+I)/N`. |
| CER | 문자 WER | 성조 언어나 문자 단위 시스템용. |
| MOS | 인간 의견 | 1-5점 평가. 청취자 20명 이상 × 샘플 100개. |
| UTMOS | ML MOS 예측기 | 학습된 모델. 인간 MOS와 약 0.9 상관. |
| SECS | 음성 복제 유사도 | 레퍼런스와 복제 간 ECAPA 코사인. |
| EER | 화자 검증 점수 | FAR = FRR인 임계값. |
| DER | 화자 분리 점수 | (FA + Miss + Confusion) / total. |
| FAD | 음악 생성 품질 | VGGish 임베딩에 대한 프레셰 거리. |
| RTFx | 처리량 | 실제 시계 초당 오디오 초. |

## 더 읽을거리 (Further Reading)

- [jiwer](https://github.com/jitsi/jiwer): 정규화 유틸리티를 갖춘 WER/CER 라이브러리.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152): 학습된 MOS 예측기.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466): 음악 생성 표준.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): 2026년 실시간 순위.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena): 인간 투표 TTS 리더보드.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/): LALM 추론 리더보드.
- [HEAR benchmark](https://hearbenchmark.com/): 오디오 SSL 벤치마크.
