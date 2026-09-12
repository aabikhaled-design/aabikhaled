# 음성 활동 탐지 & 턴테이킹(Voice Activity Detection & Turn-Taking): Silero, Cobra, 그리고 플러시 트릭

> 모든 음성 에이전트(agent)는 두 가지 판단에 따라 살고 죽는다. 사용자가 지금 말하고 있는가, 그리고 다 말했는가? VAD가 첫 번째에 답한다. 턴 감지(turn-detection)(VAD + 침묵 행오버 + 의미적 엔드포인트 모델)가 두 번째에 답한다. 둘 중 하나라도 틀리면 비서는 사용자 말을 끊거나 끝없이 떠들게 된다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45분

## 문제 (The Problem)

음성 에이전트가 모든 20ms 청크마다 내리는 세 가지 별개의 판단:

1. **이 프레임은 음성인가?**: VAD. 프레임 단위 이진(binary).
2. **사용자가 새 발화를 시작했는가?**: 시작(onset) 감지.
3. **사용자가 끝냈는가?**: 엔드포인팅(end-pointing)(턴 종료).

순진한 답(에너지 임계값)은 교통, 키보드, 군중 웅성거림 같은 잡음에서 모두 실패한다. 2026년의 답: Silero VAD(오픈, 딥러닝) + 턴 감지 모델(의미적 엔드포인팅) + VAD에 보정된 침묵 행오버.

## 개념 (The Concept)

![VAD 캐스케이드: 에너지 → Silero → 턴 감지기 → 플러시 트릭](../assets/vad-turn-taking.svg)

### 3단(three-tier) VAD 캐스케이드

**1단: 에너지 게이트(energy gate).** 가장 저렴하다. RMS를 -40 dBFS에서 임계 처리한다. 명백한 침묵은 걸러내지만 임계값 이상의 어떤 잡음에도 발동한다.

**2단: Silero VAD** (2020-2026, MIT). 100만 파라미터. 6000개 이상의 언어로 학습했다. 단일 CPU 스레드에서 30ms 청크당 약 1ms로 동작한다. 5% FPR에서 87.7% TPR. 오픈소스 기본값이다.

**3단: 의미적 턴 감지기(semantic turn detector).** LiveKit의 턴 감지 모델(2024-2026) 또는 자체 소형 분류기(classifier). "문장 중간의 멈춤"을 "말 끝냄"과 구별한다. 침묵만이 아닌 언어적 맥락(억양 + 최근 단어)을 사용한다.

### 핵심 파라미터와 기본값

- **임계값(Threshold).** Silero는 확률을 출력한다. &gt; 0.5(기본값) 또는 &gt; 0.3(민감)에서 음성으로 분류한다. 낮은 임계값 = 첫 단어 잘림은 줄고, 거짓 양성은 늘어난다.
- **최소 음성 지속 시간.** 250ms보다 짧은 음성은 거부한다. 보통 기침이나 의자 소리다.
- **침묵 행오버(silence hangover, 엔드포인팅).** VAD가 0으로 돌아온 뒤, 턴 종료를 선언하기 전에 500-800ms 기다린다. 너무 짧으면 → 사용자를 끊는다. 너무 길면 → 굼떠 보인다.
- **프리롤 버퍼(pre-roll buffer).** VAD가 발동하기 전 300-500ms의 오디오를 유지한다. "hey"가 잘리는 것을 막는다.

### 플러시 트릭 (Kyutai 2025)

스트리밍 STT 모델에는 룩어헤드(look-ahead) 지연이 있다(Kyutai STT-1B는 500ms, STT-2.6B는 2.5초). 보통은 음성 종료 후 그만큼 전사를 기다려야 한다. 플러시 트릭: VAD가 음성 종료를 발동하면, **STT에 플러시 신호를 보내** 즉시 출력을 강제한다. STT는 약 4배속으로 처리하므로, 500ms 버퍼가 약 125ms에 끝난다.

엔드-투-엔드: 125ms VAD + 플러시 STT = 대화형 지연 시간.

### 2026년 VAD 비교

| VAD | TPR @ 5% FPR | 지연 시간 | 라이선스 |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero가 올바른 기본값이다. Cobra는 컴플라이언스 / 정확도 업그레이드다. 에너지 전용 VAD는 2026년 프로덕션에 설 자리가 없다.

```figure
sp-vad-cascade
```

## 직접 만들기 (Build It)

### 1단계: 에너지 게이트

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 2단계: 파이썬에서의 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 3단계: 턴 종료 상태 기계

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 4단계: 플러시 트릭 골격

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

이것이 동작하려면 STT(Kyutai, Deepgram, AssemblyAI)가 플러시를 지원해야 한다. Whisper 스트리밍은 지원하지 않는다. 블록 기반이고 항상 청크를 기다린다.

## 라이브러리로 써보기 (Use It)

| 상황 | VAD 선택 |
|-----------|-----------|
| 오픈, 빠름, 일반 | Silero VAD |
| 상용 콜센터 | Cobra VAD |
| 온디바이스 (휴대폰) | Silero VAD ONNX |
| 연구 / 화자 분리 | pyannote segmentation |
| 무의존성 폴백 | WebRTC VAD (레거시) |
| 턴 종료 품질 필요 | Silero + LiveKit 턴 감지기 계층화 |

경험칙: 정말 다른 선택지가 없는 게 아니라면 에너지 전용 VAD는 결코 출시하지 마라.

## 함정 (Pitfalls)

- **고정 임계값.** 조용한 곳에선 동작하고 시끄러운 곳에선 실패한다. 온디바이스에서 보정하거나 Silero로 전환하라.
- **너무 짧은 침묵 행오버.** 에이전트가 문장 중간에 끊는다. 대화형 음성에는 500-800ms가 최적점이다.
- **너무 긴 행오버.** 굼떠 보인다. 목표 사용자로 A/B 테스트하라.
- **프리롤 버퍼 없음.** 사용자 오디오의 첫 200-300ms를 잃는다. 항상 롤링 프리롤을 유지하라.
- **의미적 엔드포인팅 무시하기.** "음, 생각 좀 해볼게요..."에는 긴 멈춤이 들어 있다. 사용자는 생각 도중에 끊기는 걸 싫어한다. LiveKit의 턴 감지기나 유사한 것을 써라.

## 산출물 (Ship It)

`outputs/skill-vad-tuner.md`로 저장한다. 워크로드에 맞는 VAD 모델, 임계값, 행오버, 프리롤, 턴 감지 전략을 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 음성 + 침묵 + 음성 + 기침 시퀀스를 시뮬레이션하고 세 VAD 단을 테스트한다.
2. **보통.** `silero-vad`를 설치하고 5분 녹음을 처리한다. 첫 단어 잘림과 거짓 트리거를 둘 다 최소화하도록 임계값을 튜닝한다. 정밀도/재현율을 보고한다.
3. **어려움.** 미니 턴 감지기를 만든다. Silero VAD + 마지막 10단어 임베딩(embedding)에 대한 3층 MLP(sentence-transformers 사용). 직접 라벨링한 턴 종료 데이터셋(dataset)으로 학습한다. Silero 단독을 F1 10% 앞선다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| VAD | 음성 탐지기 | 프레임 단위 이진: 이것은 음성인가? |
| Turn detection | 엔드포인팅 | VAD + 침묵 행오버 + 의미적 엔드포인트. |
| Silence hangover | 음성 후 대기 | 턴 종료를 선언하기 전 대기 시간. 500-800ms. |
| Pre-roll | 발화 전 버퍼 | VAD 발동 전 300-500ms 오디오를 유지. |
| Flush trick | Kyutai 해킹 | VAD → 플러시-STT → 500ms 대신 125ms 지연. |
| Semantic endpoint | "정말 멈추려 한 건가?" | 침묵만이 아닌 단어를 보는 ML 분류기. |
| TPR @ FPR 5% | ROC 지점 | 표준 VAD 벤치마크. Silero 87.7%, WebRTC 50%. |

## 더 읽을거리 (Further Reading)

- [Silero VAD](https://github.com/snakers4/silero-vad): 레퍼런스 오픈 VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/): 상용 정확도 선두주자.
- [Kyutai: Unmute + flush trick](https://kyutai.org/stt) 200ms 미만 엔지니어링 트릭.
- [LiveKit: turn detection](https://docs.livekit.io/agents/logic/turns/) 프로덕션에서의 의미적 엔드포인팅.
- [WebRTC VAD](https://webrtc.googlesource.com/src/): 레거시 베이스라인.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio): 화자 분리급 세분화.
