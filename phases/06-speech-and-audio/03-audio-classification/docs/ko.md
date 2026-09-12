# 오디오 분류: MFCC 기반 k-NN에서 AST와 BEATs까지

> "개 짖음 대 사이렌"부터 "이건 무슨 언어인가"까지 모든 것이 오디오 분류(audio classification)다. 특성(feature)은 멜(mel)이다. 아키텍처는 십 년마다 바뀐다. 평가(evaluation)는 AUC, F1, 클래스별 재현율(recall)로 유지된다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75분

## 문제 (The Problem)

10초짜리 클립을 받았다고 하자. 묻고 싶은 건 하나다. "이것은 무엇인가?" 도시 소리(사이렌, 드릴, 개), 음성 명령(yes/no/stop), 언어 식별(en/es/ar), 화자 감정(angry/neutral), 환경음(실내/실외, 웅성거림). 이 모두가 *오디오 분류*이며, 2026년에는 베이스라인(baseline) 아키텍처가 성숙했다. 로그 멜(log-mel) → CNN 또는 트랜스포머(Transformer) → softmax 흐름이다.

핵심 어려움은 신경망이 아니다. 데이터다. 오디오 데이터셋(dataset)에는 잔혹한 클래스 불균형(class imbalance), 강한 도메인 시프트(domain shift)(깨끗함 대 잡음), 레이블 노이즈(label noise)(누가 "도시 웅성거림" 대 "식당 소음"을 결정했는가?)가 따라붙는다. 문제의 80%는 CNN을 트랜스포머로 바꾸는 것이 아니라 큐레이션(curation), 증강(augmentation), 평가에 있다.

## 개념 (The Concept)

![오디오 분류 사다리: MFCC 기반 k-NN에서 AST, BEATs까지](../assets/audio-classification.svg)

**MFCC 기반 k-NN(1990년대 베이스라인).** 클립별로 MFCC를 평탄화하고, 레이블이 붙은 뱅크(bank)에 대한 코사인 유사도(cosine similarity)를 계산하고, 상위 K개의 다수결을 반환한다. 깨끗하고 작은 데이터셋(Speech Commands, ESC-50)에서 놀라울 만큼 강하다. GPU 없이 돌아간다.

**로그 멜 기반 2D CNN(2015-2019).** `(T, n_mels)` 로그 멜을 이미지로 취급한다. ResNet-18이나 VGG 스타일을 적용한다. 시간 축을 전역 평균 풀링(global mean pool)한다. 클래스에 대해 softmax. 2026년 대부분의 캐글(kaggle) 대회에서 여전히 베이스라인이다.

**오디오 스펙트로그램 트랜스포머(Audio Spectrogram Transformer, AST)(2021-2024).** 로그 멜을 패치화(patchify)하고(예: 16×16 패치), 위치 임베딩(position embedding)을 더하고, ViT에 넣는다. 지도 학습(supervised learning)에서 AudioSet 기준 최첨단(mAP 0.485).

**BEATs와 WavLM-base(2024-2026).** 수백만 시간에 대한 자기 지도 사전 학습(self-supervised pretraining). 원래 필요했을 지도 데이터의 1-10%만으로 과제에 맞게 파인튜닝(fine-tuning)한다. 2026년에는 비음성(non-speech) 오디오에서 이것이 기본 출발점이다. BEATs-iter3는 1/4의 연산량을 쓰면서 AudioSet에서 AST를 1-2 mAP 앞선다.

**고정 백본으로서의 Whisper 인코더(2024).** Whisper의 인코더(encoder)를 취하고, 디코더(decoder)를 버리고, 선형 분류기를 붙인다. 오디오 증강 없이도 언어 식별과 단순 이벤트 분류에서 거의 SOTA에 근접한다. "공짜 점심" 베이스라인.

### 클래스 불균형이 진짜 난제다

ESC-50: 50개 클래스, 각 40개 클립: 균형 잡혀 있고 쉽다. UrbanSound8K: 10개 클래스, 10:1로 불균형. AudioSet: 632개 클래스에 100,000:1의 롱테일(long tail)이 펼쳐진다. 효과 있는 기법:

- 학습 중 균형 샘플링(평가에서는 아님).
- Mixup: 두 클립(과 그 레이블)을 선형 보간하는 증강.
- SpecAugment: 무작위 시간 및 주파수 대역을 마스킹. 단순하지만 결정적이다.

### 평가

- 다중 클래스 배타적(Speech Commands): top-1 정확도, top-5 정확도.
- 다중 클래스 다중 레이블(AudioSet, UrbanSound 스타일): 평균 정밀도 평균(mean average precision, mAP).
- 심하게 불균형: 클래스별 재현율 + 매크로 F1.

알아야 할 2026년 수치:

| 벤치마크 | 베이스라인 | SOTA 2026 | 출처 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## 직접 만들기 (Build It)

### 단계 1: 특성화한다

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 단계 2: 고정 길이 요약

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

단순하지만 강하다: 시간에 걸친 평균 + 분산은 13개 계수 MFCC에 대해 26차원 고정 임베딩(embedding)을 준다. 즉시 실행된다. 2017년만 해도 ESC-50에서 최첨단 신경망 베이스라인을 능가했다.

### 단계 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 단계 4: 로그 멜 기반 CNN으로 업그레이드한다

PyTorch에서:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300만 파라미터(parameter). 단일 RTX 4090에서 ESC-50에 ~10분 만에 학습된다. 80%+ 정확도.

### 단계 5: 2026년의 기본: BEATs 파인튜닝

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs의 경우 `beats` 라이브러리를 통해 `microsoft/BEATs-base`를 사용하라; transformers API는 같은 형태다.

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 시작점 |
|-----------|-----------|
| 초소형 데이터셋 (<1000 클립) | MFCC 평균 기반 k-NN(당신의 베이스라인) + 오디오 증강 |
| 중간 데이터셋 (1K–100K) | BEATs 또는 AST 파인튜닝 |
| 대형 데이터셋 (>100K) | 밑바닥부터 학습하거나 Whisper 인코더 파인튜닝 |
| 실시간, 엣지(edge) | int8로 양자화된 40-MFCC CNN(KWS 스타일) |
| 다중 레이블 (AudioSet) | BCE 손실 + mixup + SpecAugment를 쓴 BEATs-iter3 |
| 언어 식별 | MMS-LID, SpeechBrain VoxLingua107 베이스라인 |

결정 규칙: **새 모델이 아니라 고정 백본으로 시작하라**. BEATs 헤드를 파인튜닝하면 몇 주가 아니라 몇 시간 만에 SOTA의 95%에 도달한다.

## 산출물 (Ship It)

`outputs/skill-classifier-designer.md`로 저장하라. 주어진 오디오 분류 과제에 대해 아키텍처, 증강, 클래스 균형 전략, 평가 지표를 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 4개 클래스 합성 데이터셋(서로 다른 음높이의 순음)에서 k-NN MFCC 베이스라인을 학습한다. 혼동 행렬(confusion matrix)을 보고하라.
2. **중간.** `summarize`를 [평균, 분산, 왜도(skew), 첨도(kurtosis)]로 교체하라. 같은 합성 데이터셋에서 4-모멘트 풀링(pooling)이 평균+분산을 능가하는가?
3. **어려움.** `torchaudio`를 사용해 ESC-50 fold 1에서 2D CNN을 학습하라. 5-폴드 교차 검증(cross-validation) 정확도를 보고하라. SpecAugment(시간 마스크 = 20, 주파수 마스크 = 10)를 추가하고 그 변화량을 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| AudioSet | 오디오의 ImageNet | 구글의 200만 클립, 632개 클래스 약한 레이블(weakly-labeled) YouTube 데이터셋. |
| ESC-50 | 작은 분류 벤치마크 | 환경음 50개 클래스 × 40개 클립. |
| AST | 오디오 스펙트로그램 트랜스포머 | 로그 멜 패치에 적용한 ViT; 2021년 SOTA. |
| BEATs | 자기 지도 오디오 | 마이크로소프트 모델, 2026년 기준 iter3가 AudioSet 선두. |
| Mixup | 쌍 증강 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | 마스크 기반 증강 | 스펙트로그램의 무작위 시간 및 주파수 대역을 0으로 만듦. |
| mAP | 주요 다중 레이블 지표 | 클래스와 임계값에 걸친 평균 정밀도 평균. |

## 더 읽을거리 (Further Reading)

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778): 2021–2024년의 표준 아키텍처.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058): 2024년 이후의 기본.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779): 지배적인 오디오 증강.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50): 명맥을 잇는 50개 클래스 벤치마크.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/): 632개 클래스 YouTube 분류 체계; 여전히 표준.
