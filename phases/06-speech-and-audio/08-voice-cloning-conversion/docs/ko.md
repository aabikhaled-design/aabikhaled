# 음성 복제 & 음성 변환

> 음성 복제(voice cloning)는 입력한 텍스트를 다른 사람의 목소리로 읽는다. 음성 변환(voice conversion)은 말한 내용은 그대로 둔 채 목소리만 다른 사람의 것으로 바꾼다. 둘 다 같은 분해에 달려 있다. 화자 정체성(speaker identity)을 내용(content)과 분리하는 일이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75분

## 문제 (The Problem)

2026년에는 5초짜리 오디오 클립이면 소비자용 GPU로 누구의 목소리든 고품질 복제를 만들기에 충분하다. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox 모두 제로샷(zero-shot) 또는 퓨샷(few-shot) 복제를 출하한다. 이 기술은 축복(접근성 TTS, 더빙, 보조 음성)이자 무기(사기 전화, 정치적 딥페이크, 지식재산 도용)다.

밀접하게 관련된 두 가지 과제:

- **음성 복제(TTS 측):** 텍스트 + 5초 참조 음성 → 그 목소리로 된 오디오.
- **음성 변환(음성 측):** 소스 오디오(인물 A가 X를 말함) + 인물 B의 참조 음성 → B가 X를 말하는 오디오.

둘 다 파형(waveform)을 (내용, 화자, 운율(prosody))로 분해하고, 한 소스의 내용을 다른 소스의 화자와 재결합한다.

2026년에 출하하려면 지켜야 할 핵심 제약이 있다. **워터마킹(watermarking)과 동의 게이트(consent gate)는 EU(AI Act, 2026년 8월 시행)와 캘리포니아(AB 2905, 2025년 발효)에서 법적으로 요구된다**. 파이프라인은 들리지 않는 워터마크를 넣고 비동의 복제를 거부해야 한다.

## 개념 (The Concept)

![음성 복제 대 변환: 분해, 화자 교체, 재결합](../assets/voice-cloning.svg)

**제로샷 복제.** 수천 명의 화자로 학습된 모델에 5초짜리 클립을 넘긴다. 화자 인코더(speaker encoder)가 클립을 화자 임베딩(speaker embedding)으로 사상하고, TTS 디코더가 그 임베딩에 텍스트를 더해 조건화한다.

사용: F5-TTS(2024), YourTTS(2022), XTTS v2(2024), OpenVoice v2(2024).

**퓨샷 파인튜닝.** 타깃 목소리를 5–30분 녹음한다. 베이스 모델을 한 시간 동안 LoRA로 파인튜닝(fine-tuning)한다. 품질이 "괜찮음"에서 "구별 불가"로 도약한다. Coqui와 ElevenLabs 둘 다 이 패턴을 지원하고; 커뮤니티는 F5-TTS와 함께 사용한다.

**음성 변환(VC).** 두 계열:

- **인식-합성(Recognition-synthesis).** ASR 유사 모델을 돌려 내용 표현(예: 부드러운 음소 사후확률, PPG)을 추출하고, 타깃 화자 임베딩으로 재합성한다. 언어와 억양에 견고하다. KNN-VC(2023), Diff-HierVC(2023)가 사용한다.
- **분리(Disentanglement).** 병목(bottleneck)에서 잠재 공간(latent space)의 내용, 화자, 운율을 분리하는 오토인코더(autoencoder)를 학습한다. 추론 시 화자 임베딩을 교체한다. 더 낮은 품질이지만 더 빠르다. AutoVC(2019), VITS-VC 변형이 사용한다.

**신경 코덱(neural codec) 기반 복제(2024년 이후).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox: 오디오를 SoundStream / EnCodec의 이산 토큰(discrete token)으로 취급하고, 코덱 토큰에 대한 대형 자기회귀(autoregressive) 또는 흐름 매칭(flow-matching) 모델을 학습한다. 짧은 프롬프트(prompt)에서 ElevenLabs에 필적하는 품질.

### 윤리 부분, 부가물이 아니다

**워터마킹.** PerTh(Perth)와 SilentCipher(2024)는 ~16-32비트 ID를 오디오에 지각 불가능하게 삽입한다. 재인코딩, 스트리밍, 일반적 편집을 견딘다. 프로덕션 준비된 오픈소스.

**동의 게이트.** 모든 복제 출력을 검증 가능한 동의 기록과 짝지어야 한다. "나, Rohit은, 2026-04-22에, 이 목소리를 X 목적으로 승인한다." 변조 증거가 남는(tamper-evident) 로그에 저장하라.

**검출.** AASIST, RawNet2, Wav2Vec2-AASIST가 검출기로 출하된다. ASVspoof 2025 챌린지는 ElevenLabs, VALL-E 2, Bark 출력에 대해 최첨단 검출기의 EER을 0.8–2.3%로 발표했다.

### 수치 (2026)

| 모델 | 제로샷? | SECS (타깃 유사도) | WER (명료도) | 파라미터 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 예 | 0.72 | 2.1% | 335M |
| XTTS v2 | 예 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 예 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 예 | 0.77 | 2.4% | 370M |
| VoiceBox | 예 | 0.78 | 2.1% | 330M |

SECS > 0.70은 대부분의 청취자에게 일반적으로 타깃과 구별 불가능하다.

```figure
sp-voice-factorize
```

## 직접 만들기 (Build It)

### 단계 1: 인식-합성으로 분해한다(main.py의 코드 전용 데모)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

개념적으로는 단순하다. 구현의 무게는 `tts_model`과 화자 인코더에 있다.

### 단계 2: F5-TTS로 제로샷 복제

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

참조 전사는 오디오와 정확히 일치해야 한다. 어긋나면 정렬(alignment)이 깨진다.

### 단계 3: KNN-VC로 음성 변환

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC는 WavLM을 돌려 소스와 타깃 풀(pool)에 대한 프레임별 임베딩을 추출한 뒤, 각 소스 프레임을 풀에서의 최근접 이웃(nearest neighbor)으로 교체한다. 비모수적(non-parametric)이며, 1분 분량의 타깃 음성으로 작동한다.

### 단계 4: 워터마크를 삽입한다

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32비트의 페이로드, MP3 재인코딩과 가벼운 잡음 이후에도 검출 가능.

### 단계 5: 동의 게이트

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 상황 | 선택 |
|-----------|------|
| 5초 제로샷 복제, 오픈소스 | F5-TTS 또는 OpenVoice v2 |
| 상업적 프로덕션 복제 | ElevenLabs Instant Voice Clone v2.5 |
| 음성 변환(다시 쓰기) | KNN-VC 또는 Diff-HierVC |
| 다중 화자 파인튜닝 | StyleTTS 2 + 화자 어댑터 |
| 교차 언어 복제 | XTTS v2 또는 VALL-E X |
| 딥페이크 검출 | Wav2Vec2-AASIST |

## 함정들 (Pitfalls)

- **정렬되지 않은 참조 전사.** F5-TTS 등은 참조 텍스트가 구두점까지 포함해 참조 오디오와 정확히 일치하기를 요구한다.
- **잔향이 있는 참조.** 에코가 복제를 죽인다. 마이크에 가깝게 건조하게(dry) 녹음하라.
- **감정 불일치.** "쾌활한" 학습 참조는 모든 것의 쾌활한 복제를 만든다. 참조 감정을 타깃 용도에 맞춰라.
- **언어 누설.** 영어 화자를 복제한 뒤 모델에게 프랑스어를 말하라고 요청하면 흔히 억양을 그대로 가져간다; 교차 언어 모델(XTTS, VALL-E X)을 사용하라.
- **워터마크 없음.** 2026년 8월부터 EU에서 법적으로 출하 불가.

## 산출물 (Ship It)

`outputs/skill-voice-cloner.md`로 저장하라. 동의 게이트 + 워터마크 + 품질 타깃을 갖춘 복제 또는 변환 파이프라인을 설계한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행하라. 교체 전후의 두 "화자" 사이의 코사인을 계산해 화자 임베딩 교체를 시연한다.
2. **중간.** OpenVoice v2로 자기 목소리를 복제하라. 참조와 복제 사이의 SECS를 측정하라. Whisper로 CER을 측정하라.
3. **어려움.** 20개 복제에 SilentCipher 워터마크를 적용하고, 128 kbps MP3 인코드+디코드를 통과시킨 뒤, 페이로드를 검출하라. 비트 정확도를 보고하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| 제로샷 복제 | 5초면 충분 | 사전 학습된 모델 + 화자 임베딩; 학습 없음. |
| PPG | 음성 사후확률도(Phonetic posteriorgram) | 언어 비의존적 내용 표현으로 쓰이는 프레임별 ASR 사후확률. |
| KNN-VC | 최근접 이웃 변환 | 각 소스 프레임을 최근접 타깃 풀 프레임으로 교체. |
| 신경 코덱 TTS | VALL-E 스타일 | EnCodec/SoundStream 토큰에 대한 AR 모델. |
| 워터마크(Watermark) | 들리지 않는 서명 | 오디오에 삽입된 비트, 재인코딩을 견딤. |
| SECS | 복제 충실도 | 타깃과 복제 화자 임베딩 사이의 코사인. |
| AASIST | 딥페이크 검출기 | 안티스푸핑 모델; 합성 음성을 검출. |

## 더 읽을거리 (Further Reading)

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885): 오픈소스 SOTA 제로샷 복제.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 및 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370): 신경 코덱 TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879): 분리 기반 음성 변환.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975): 검색 기반 VC.
- [SilentCipher (2024): Audio Watermarking](https://github.com/sony/silentcipher) 프로덕션 준비된 32비트 오디오 워터마크.
- [ASVspoof 2025 results](https://www.asvspoof.org/): 검출기 대 합성기 군비 경쟁, 2026년 업데이트.
