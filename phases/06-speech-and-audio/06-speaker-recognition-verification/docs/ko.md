# 화자 인식 & 검증

> ASR은 "그들이 무엇을 말했는가?"를 묻는다. 화자 인식(speaker recognition)은 "누가 말했는가?"를 묻는다. 수학은 같아 보인다. 임베딩(embedding)에 코사인(cosine): 하지만 모든 프로덕션 결정은 단 하나의 EER 숫자에 달려 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45분

## 문제 (The Problem)

사용자가 암호문구(passphrase)를 말한다. 이때 알아내야 할 것은 이렇다. 본인이 주장하는 그 사람이 맞는가(*검증(verification)*, 1:1), 아니면 등록(enrollment) 뱅크의 첫 번째 사람인가(*식별(identification)*, 1:N)? 아니면 둘 다 아니라, 등록되지 않은 화자인가(*오픈셋(open-set)*)?

2018년 이전: GMM-UBM + i-vector. 합리적인 EER이지만 채널 시프트(channel shift)(전화 대 노트북)와 감정에 취약하다. 2018–2022: x-vector(각 마진(angular margin)으로 학습된 TDNN 백본). 2022년 이후: ECAPA-TDNN과 WavLM-large 임베딩. 2026년 기준 이 분야는 세 개의 모델과 하나의 지표가 지배한다.

그 지표는 **EER**: 동등 오류율(Equal Error Rate)이다. 오수락률(False Accept Rate) = 오거부율(False Reject Rate)이 되도록 결정 임계값(threshold)을 설정하라. 그 교차점이 EER이다. 모든 논문, 모든 리더보드, 모든 조달 회의에서 쓰인다.

## 개념 (The Concept)

![임베딩 + 코사인 + EER을 갖춘 등록 + 검증 파이프라인](../assets/speaker-verification.svg)

**파이프라인.** 등록 단계에서는 타깃 화자의 음성 5–30초를 녹음한 뒤 고정 차원 임베딩을 계산한다(ECAPA-TDNN은 192차원, WavLM-large는 256차원). 검증 단계에서는 테스트 발화의 임베딩을 얻어 코사인 유사도를 구하고 임계값과 비교한다.

**ECAPA-TDNN(2020, 2026년에도 여전히 지배적).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. 스퀴즈-여기(squeeze-excitation)를 갖춘 1D 컨볼루션 블록, 멀티헤드 어텐션 풀링(multi-head attention pooling), 그 뒤를 잇는 192차원으로의 선형 층. VoxCeleb 1+2(2,700명 화자, 110만 발화)에서 Additive Angular Margin 손실(AAM-softmax)로 학습되었다.

**WavLM-SV(2022년 이후).** 사전 학습된 WavLM-large SSL 백본을 AAM 손실로 파인튜닝(fine-tuning)한다. 더 높은 품질이지만 더 느리다. 15 MB 대 300+ MB.

**x-vector(베이스라인).** TDNN + 통계 풀링(statistics pooling). 고전적; CPU / 엣지(edge)에서 여전히 유용하다.

**AAM-softmax.** 각 공간(angular space)에서 마진 `m`을 추가한 표준 softmax: 올바른 클래스에 대해 `cos(θ + m)`. 클래스 간 각 분리를 강제한다. 전형적으로 `m=0.2`, 스케일 `s=30`.

### 점수화

- 등록 임베딩과 테스트 임베딩 사이의 **코사인**. 임계값 기반 결정.
- **PLDA(Probabilistic LDA).** 같은 화자 대 다른 화자가 닫힌 형태의 우도비(likelihood ratio)를 가지는 잠재 공간(latent space)으로 임베딩을 투영한다. EER을 10–20% 추가로 줄이기 위해 코사인 위에 더한다. 2020년 이전 표준; 이제는 닫힌 집합(closed-set) 설정에서만 쓰인다.
- **점수 정규화(Score normalization).** `S-norm` 또는 `AS-norm`: 각 점수를 사칭자(imposter) 평균과 표준편차의 코호트(cohort)에 대해 정규화한다. 교차 도메인 평가에 필수적이다.

### 알아야 할 수치 (2026)

| 모델 | VoxCeleb1-O EER | 파라미터 | 처리량(A100) |
|-------|-----------------|--------|-------------------|
| x-vector (고전) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 화자 분리 (Diarization)

다중 화자 클립에서 "누가 언제 말했는가". 파이프라인: VAD → 분절(segment) → 각 분절 임베딩 → 클러스터링(응집(agglomerative) 또는 스펙트럼) → 경계 매끄럽게. 현대 스택: `pyannote.audio` 3.1, 화자 분절 + 임베딩 + 클러스터링을 하나의 호출 뒤에 묶는다. AMI에서 2026년 SOTA DER은 ~15%다(2022년 23%에서 하락).

```figure
sp-eer-crossover
```

## 직접 만들기 (Build It)

### 단계 1: MFCC 통계로부터의 장난감 임베딩

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

SOTA에는 한참 못 미친다. 교육용일 뿐이다. `code/main.py`는 이것을 합성 화자 데이터에 대한 개념 증명(proof-of-concept)으로 사용한다.

### 단계 2: 코사인 유사도 + 임계값

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 단계 3: 유사도 쌍으로부터의 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

(eer, threshold_at_eer)를 반환한다. 둘 다 보고하라.

### 단계 4: SpeechBrain으로 프로덕션

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 단계 5: pyannote로 화자 분리

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 선택 |
|-----------|------|
| 닫힌 집합 1:1 검증, 엣지 | ECAPA-TDNN + 코사인 임계값 |
| 오픈셋 검증, 클라우드 | WavLM-SV + AS-norm |
| 화자 분리(회의, 팟캐스트) | `pyannote/speaker-diarization-3.1` |
| 안티스푸핑(재생 / 딥페이크 검출) | AASIST 또는 RawNet2 |
| 초소형 임베디드(KWS + 등록) | Titanet-Small (NeMo) |

## 함정들 (Pitfalls)

- **채널 불일치.** VoxCeleb(웹 비디오)로 학습된 모델 ≠ 전화 통화 오디오. 항상 타깃 채널에서 평가하라.
- **짧은 발화.** 테스트 오디오가 3초 미만이면 EER이 급격히 나빠진다.
- **잡음이 있는 등록.** 잡음 섞인 등록 하나가 앵커(anchor)를 오염시킨다. 깨끗한 샘플 3개 이상을 사용해 평균을 내라.
- **조건에 걸친 고정 임계값.** 항상 타깃 도메인의 홀드아웃 개발(dev) 세트에서 임계값을 튜닝하라.
- **정규화되지 않은 임베딩에 대한 코사인.** 먼저 L2 정규화하라; 그렇지 않으면 크기(magnitude)가 지배한다.

## 산출물 (Ship It)

`outputs/skill-speaker-verifier.md`로 저장하라. 모델, 등록 프로토콜, 임계값 튜닝 계획, 사기 방지책을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 합성 "화자"(서로 다른 음색 프로파일)를 만들고, 등록하고, 100쌍 시도 목록에서 EER을 계산한다.
2. **중간.** 30개 VoxCeleb1 발화(화자 5명 × 각 6개)에 SpeechBrain ECAPA를 사용하라. 코사인 대 PLDA로 EER을 계산하라.
3. **어려움.** `pyannote.audio`로 전체 등록 → 화자 분리 → 검증 파이프라인을 만들어라. AMI 개발 세트에서 DER을 평가하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| EER | 대표 지표 | 오수락 = 오거부가 되는 임계값. |
| 검증(Verification) | 1:1 | "이게 Alice인가?" |
| 식별(Identification) | 1:N | "누가 말하고 있는가?" |
| 오픈셋(Open-set) | 알려지지 않은 것 가능 | 테스트 세트가 등록되지 않은 화자를 포함할 수 있음. |
| 등록(Enrollment) | 등록하기 | 화자의 참조 임베딩을 계산하는 것. |
| AAM-softmax | 그 손실 | 가산 각 마진을 갖춘 softmax; 클러스터 분리를 강제함. |
| PLDA | 고전적 점수화 | 확률적 LDA; 임베딩 위의 우도비 점수화. |
| DER | 화자 분리 지표 | 화자 분리 오류율(Diarization Error Rate): 누락 + 오경보 + 혼동. |

## 더 읽을거리 (Further Reading)

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf): 고전적인 딥 임베딩 논문.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143): 2020–2026년 지배적 아키텍처.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900): SV와 화자 분리를 위한 SSL 백본.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio): 프로덕션 화자 분리 + 임베딩 스택.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/): 모델에 걸친 현재 EER 순위.
