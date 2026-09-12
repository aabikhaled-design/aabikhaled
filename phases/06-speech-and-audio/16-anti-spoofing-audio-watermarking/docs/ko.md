# 음성 안티 스푸핑 & 오디오 워터마킹(Voice Anti-Spoofing & Audio Watermarking): ASVspoof 5, AudioSeal, WaveVerify

> 음성 복제(voice cloning)는 방어보다 빠르게 출시되었다. 2026년 프로덕션(production) 음성 시스템에는 두 가지가 필요하다. 진짜 대 가짜 음성을 분류하는 탐지기(AASIST, RawNet2)와, 압축과 편집을 견디는 워터마크(AudioSeal). 둘 다 출시하거나, 음성 복제를 출시하지 마라.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75분

## 문제 (The Problem)

서로 관련된 세 가지 방어:

1. **안티 스푸핑 / 딥페이크 탐지(Anti-spoofing / deepfake detection).** 오디오 클립이 주어졌을 때, 합성인가 진짜인가? ASVspoof 벤치마크(ASVspoof 2019 → 2021 → 5)가 황금 표준이다.
2. **오디오 워터마킹(Audio watermarking).** 생성된 오디오에 탐지기가 나중에 추출할 수 있는, 지각되지 않는 신호를 심는다. AudioSeal(Meta)과 WavMark이 오픈 옵션이다.
3. **인증된 출처(Authenticated provenance).** 오디오 파일 + 메타데이터의 암호학적 서명. C2PA / Content Authenticity Initiative.

탐지는 협조하지 않는 적을 다룬다. 워터마킹은 컴플라이언스를 다룬다. AI 생성 오디오는 그렇다고 식별 가능해야 한다. 2026년에는 둘 다 필요하다.

## 개념 (The Concept)

![안티 스푸핑 vs 워터마킹 vs 출처: 세 가지 방어 계층](../assets/spoofing-watermark.svg)

### ASVspoof 5: 2024-2025년 벤치마크

이전 판들로부터의 가장 큰 변화:

- **크라우드소싱 데이터**(스튜디오 클린이 아님): 현실적 조건.
- **약 2000명의 화자**(이전 약 100명 대비).
- **32개의 공격 알고리즘.** TTS + 음성 변환 + 적대적 섭동.
- **두 트랙.** 대응책(Countermeasure, CM) 단독 탐지. 생체 시스템을 위한 스푸핑 강인 ASV(Spoofing-robust ASV, SASV).

ASVspoof 5에서의 최신 기술(state-of-the-art): 약 7.23% EER. 더 오래된 ASVspoof 2019 LA에서: 0.42% EER. 실세계 배포: 야생 클립에서 5-10% EER를 예상하라.

### AASIST와 RawNet2: 탐지 모델 계열

**AASIST** (2021, 2026년까지 갱신됨). 스펙트럴 특성에 대한 그래프 어텐션(graph-attention). ASVspoof 5 대응책 과제에서 현재 SOTA.

**RawNet2.** 원시 파형(raw waveform)에 대한 합성곱(convolution) 프런트엔드 + TDNN 백본. 더 단순한 베이스라인. 파인튜닝(fine-tuning)으로 여전히 경쟁력 있음.

**NeXt-TDNN + SSL 특성.** 2025년 변형: ECAPA 스타일 + WavLM 특성 + 포컬 손실(focal loss). ASVspoof 2019 LA에서 0.42% EER를 달성한다.

### AudioSeal: 2024년 워터마크 기본값

Meta의 **AudioSeal** (2024년 1월, v0.2 2024년 12월). 핵심 설계:

- **국소화(Localized).** 16kHz 샘플 해상도(1/16000초)로 프레임 단위로 워터마크를 탐지한다.
- **생성기 + 탐지기 공동 학습.** 생성기는 들리지 않는 신호를 심는 법을 배우고, 탐지기는 증강(augmentation)을 통해 그것을 찾는 법을 배운다.
- **강인함.** MP3 / AAC 압축, EQ, 속도 변화 ±10%, +10dB SNR 잡음 혼합을 견딘다.
- **빠름.** 탐지기가 485배속으로 동작한다. WavMark보다 1000배 빠르다.
- **용량.** 발화마다 심을 수 있는 16비트 페이로드(모델 ID, 생성 타임스탬프, 사용자 ID를 인코딩 가능).

### WavMark

AudioSeal 이전의 오픈 베이스라인. 가역 신경망(invertible neural network), 초당 32비트. 문제점:

- 동기화 무차별 대입이 느리다.
- 가우시안 잡음이나 MP3 압축으로 제거될 수 있다.
- 실시간 친화적이지 않다.

### WaveVerify (2025년 7월)

AudioSeal의 약점, 특히 시간적 조작(역재생, 속도)을 다룬다. FiLM 기반 생성기 + 전문가 혼합(Mixture-of-Experts) 탐지기를 쓴다. 표준 공격에서 AudioSeal과 경쟁력 있고, 시간적 편집을 다룬다.

### 적이 악용하는 빈틈

AudioMarkBench에서: "피치 시프트 하에서 모든 워터마크가 비트 복구 정확도(Bit Recovery Accuracy) 0.6 미만을 보이며, 이는 거의 완전한 제거를 가리킨다." **피치 시프트가 보편적 공격이다.** 2026년의 어떤 워터마크도 공격적인 피치 변조에 완전히 강인하지 않다. 그래서 워터마킹에는 탐지(AASIST)가 함께 필요하다.

### C2PA / Content Authenticity Initiative

ML 기법이 아니라 매니페스트(manifest) 형식이다. 오디오 파일이 생성 도구, 저자, 날짜에 대한 암호학적으로 서명된 메타데이터를 담는다. Audobox / Seamless가 이를 쓴다. 출처에는 좋다. 악의적 행위자가 재인코딩해 메타데이터를 벗겨내면 아무것도 못 한다.

```figure
v4-audio-watermark
```

## 직접 만들기 (Build It)

### 1단계: 단순한 스펙트럴 특성 탐지기 (장난감)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

합성 음성은 종종 비정상적으로 평탄한 고주파 에너지를 가진다. 프로덕션 탐지기는 이것이 아니라 AASIST를 쓴다. 하지만 직관은 유효하다.

### 2단계: AudioSeal 삽입 + 탐지

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 3단계: 평가: EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 4단계: 프로덕션 통합

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

모든 생성은 다음을 함께 출시한다: (1) 워터마크, (2) 서명된 매니페스트, (3) 보존 정책을 준수하는 감사 로그.

## 라이브러리로 써보기 (Use It)

| 사용 사례 | 방어 |
|----------|---------|
| TTS / 음성 복제 출시 | 모든 출력에 AudioSeal 삽입 (타협 불가) |
| 생체 음성 잠금 해제 | AASIST + ECAPA 앙상블. 생존성(liveness) 챌린지 |
| 콜센터 사기 탐지 | 수신 통화의 20% 샘플에 AASIST |
| 팟캐스트 진위 | 업로드 시 C2PA 서명, AI 생성이면 AudioSeal |
| 탐지기 연구 / 학습 | ASVspoof 5 train/dev/eval 셋 |

## 함정 (Pitfalls)

- **탐지기를 한 번도 돌리지 않는 워터마크.** 무의미하다. CI에 탐지기를 넣어라.
- **보정 없는 탐지.** ASVspoof LA로 학습한 AASIST는 과적합(overfitting)한다. 실세계 정확도가 떨어진다. 도메인에서 보정하라.
- **피치 시프트 빈틈.** 공격적인 피치 시프트는 대부분의 워터마크를 제거한다. 탐지 폴백을 두어라.
- **메타데이터 제거 후 재호스팅.** C2PA는 재인코딩으로 손쉽게 우회된다. 항상 암호학적 + 지각적(워터마크) 방어를 함께 추가하라.
- **생존성을 탐지로 쓰기.** 사용자에게 무작위 문구를 말하게 한다. 재생(replay) 공격은 막지만 실시간 복제는 막지 못한다.

## 산출물 (Ship It)

`outputs/skill-spoof-defender.md`로 저장한다. 음성 생성 배포에 대해 탐지 모델, 워터마크, 출처 매니페스트, 운영 플레이북을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 합성 오디오에 대한 장난감 탐지기 + 장난감 워터마크 삽입/탐지.
2. **보통.** `audioseal`을 설치하고 TTS 출력에 16비트 페이로드를 심은 뒤 다시 디코딩한다. 잡음으로 오디오를 손상시키고 비트 복구 정확도를 측정한다.
3. **어려움.** ASVspoof 2019 LA에서 RawNet2 또는 AASIST를 파인튜닝한다. EER를 측정한다. F5-TTS로 생성된 클립의 홀드아웃(held-out) 셋에서 테스트한다. OOD 탐지가 어떻게 저하되는지 본다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| ASVspoof | 그 벤치마크 | 격년 챌린지. 2024 = ASVspoof 5. |
| CM (countermeasure) | 탐지기 | 분류기: 진짜 음성 vs 합성 / 변환. |
| SASV | 화자 검증 + CM | 통합 생체 + 스푸핑 탐지. |
| AudioSeal | Meta 워터마크 | 국소화, 16비트 페이로드, WavMark보다 485배 빠름. |
| Bit Recovery Accuracy | 워터마크 생존 | 공격 후 복구된 페이로드 비트의 비율. |
| C2PA | 출처 매니페스트 | 생성 / 저작에 대한 암호학적 메타데이터. |
| AASIST | 탐지기 계열 | 그래프 어텐션 기반 안티 스푸핑 SOTA. |

## 더 읽을거리 (Further Reading)

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825): 현재 벤치마크.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264): 워터마크 기본값.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150): 시간적 공격용 MoE 탐지기.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200): SOTA 탐지 백본.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf): 강인성 평가.
- [C2PA specification](https://c2pa.org/specifications/specifications/): 출처 매니페스트 형식.
