# 실시간 오디오 처리(Real-Time Audio Processing)

> 배치(batch) 파이프라인은 파일 하나를 처리한다. 실시간 파이프라인은 다음 20밀리초가 도착하기 전에 다음 20밀리초를 처리한다. 모든 대화형 AI, 방송 스튜디오, 통신(telephony) 봇은 이 지연 시간(latency) 예산에 따라 살고 죽는다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75분

## 문제 (The Problem)

살아 있는 듯한 느낌의 음성 비서를 만들고 싶다. 사람의 대화 턴테이킹(turn-taking) 지연 시간은 약 230ms(침묵에서 응답까지)다. 500ms를 넘으면 로봇 같고, 1500ms를 넘으면 고장 난 것처럼 느껴진다. 2026년에 **듣기 → 이해하기 → 응답하기 → 말하기**의 전체 루프에 주어진 예산은 다음과 같다.

| 단계 | 예산 |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi(Kyutai, 2024)는 풀 듀플렉스(full-duplex) 200ms를 기록했다. GPT-4o-realtime(2024)은 약 320ms를 기록한다. 2022년의 캐스케이드(cascaded) 파이프라인은 2500ms로 출시되었다. 이 10배 개선은 세 가지 기법에서 나왔다. (1) 모든 곳에서 스트리밍하기, (2) 부분 결과를 사용하는 비동기 파이프라이닝, (3) 중단 가능한 생성.

## 개념 (The Concept)

![링 버퍼, VAD 게이트, 인터럽션을 갖춘 스트리밍 오디오 파이프라인](../assets/real-time.svg)

**프레임 / 청크 / 윈도우(Frame / chunk / window).** 실시간 오디오는 고정 크기 블록으로 흐른다. 흔한 선택은 20ms(16kHz에서 320 샘플)다. 하류의 모든 것이 이 박자를 따라가야 한다.

**링 버퍼(Ring buffer).** 고정 크기 순환 버퍼다. 프로듀서 스레드는 새 프레임을 쓰고, 컨슈머 스레드는 읽는다. 핫 패스(hot path)에서 할당이 일어나지 않게 막는다. 크기 ≈ 최대 지연 시간 × 샘플 레이트. 2초짜리 16kHz 링은 32,000 샘플이다.

**VAD(Voice Activity Detection, 음성 활동 탐지).** 아무도 말하지 않을 때 하류 작업을 차단한다. Silero VAD 4.0(2024)은 CPU에서 30ms 프레임당 1ms 미만으로 동작한다. `webrtcvad`는 더 오래된 대안이다.

**스트리밍 ASR(Streaming ASR).** 오디오가 도착하는 대로 부분 전사를 내보내는 모델이다. 스트리밍 모드의 Parakeet-CTC-0.6B(NeMo, 2024)는 320ms 지연 시간에서 2-5% WER를 낸다. Whisper-Streaming(Macháček et al., 2023)은 Whisper를 청크 단위로 잘라 약 2초 지연 시간의 준스트리밍을 한다.

**인터럽션(Interruption).** 비서가 말하는 동안 사용자가 말하면, (a) 끼어들기(barge-in)를 감지하고, (b) TTS를 멈추고, (c) 남은 LLM 출력을 버려야 한다. 이 모두를 100ms 이내에 해야 한다. 그렇지 않으면 사용자는 귀먹은 비서를 느낀다.

**WebRTC Opus 전송.** 20ms 프레임, 48kHz, 적응형 비트레이트 8-128kbps. 브라우저와 모바일의 표준이다. LiveKit, Daily.co, Pion이 음성 앱을 만드는 2026년의 스택이다.

**지터 버퍼(Jitter buffer).** 네트워크 패킷은 순서가 뒤바뀌거나 늦게 도착한다. 지터 버퍼는 재정렬하고 매끄럽게 다듬는다. 너무 작으면 들리는 끊김이 생기고, 너무 크면 지연이 생긴다. 보통 60-80ms다.

### 흔한 함정

- **스레드 경합(Thread contention).** 파이썬의 GIL + 무거운 모델은 오디오 스레드를 굶길 수 있다. C-콜백 오디오 라이브러리(sounddevice, PortAudio)를 쓰고 파이썬을 핫 패스에서 빼라.
- **샘플 레이트 변환 지연.** 파이프라인 내부의 리샘플링은 5-20ms를 더한다. 앞단에서 미리 리샘플링하거나 무지연 리샘플러(PolyPhase, `soxr_hq`)를 써라.
- **TTS 프라이밍(priming).** Kokoro 같은 빠른 TTS도 첫 요청에서 100-200ms의 워밍업이 있다. 모델을 캐시하고, 첫 실제 턴 전에 더미 실행으로 워밍업하라.
- **에코 제거(Echo cancellation).** AEC가 없으면 TTS 출력이 마이크로 다시 들어와 봇 자신의 목소리에 대해 ASR을 트리거한다. WebRTC AEC3이 오픈소스 기본값이다.

```figure
nyquist-aliasing
```

## 직접 만들기 (Build It)

### 1단계: 링 버퍼

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

용량이 최대 버퍼링 지연 시간을 결정한다. 16kHz에서 32,000 샘플 = 2초.

### 2단계: VAD 게이트

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

프로덕션(production)에서는 Silero VAD로 교체한다.

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 3단계: 스트리밍 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 4단계: 인터럽션 핸들러

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

비동기 I/O와 취소 가능한 TTS 스트리밍에 달려 있다. 오디오 트랙에 대한 WebRTC peerconnection.stop()이 정석적인 방법이다.

## 라이브러리로 써보기 (Use It)

2026년의 스택:

| 계층 | 선택 |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## 함정 (Pitfalls)

- **안전을 위해 500ms를 버퍼링하기.** 버퍼가 곧 지연 시간의 하한선이다. 줄여라.
- **스레드를 고정하지 않기.** UI보다 낮은 우선순위 스레드의 오디오 콜백 = 부하 시 글리치.
- **너무 작은 TTS 청크.** 200ms 미만의 청크는 보코더(vocoder) 아티팩트를 들리게 한다. 320ms 청크가 최적점이다.
- **지터 버퍼 없음.** 실제 네트워크는 지터가 많다. 매끄럽게 다듬지 않으면 팝(pop) 소리가 난다.
- **단발성 오류 처리.** 오디오 파이프라인은 크래시에 강해야 한다. 예외 하나가 세션을 죽인다.

## 산출물 (Ship It)

`outputs/skill-realtime-designer.md`로 저장한다. 단계별 구체적 지연 시간 예산을 갖춘 실시간 오디오 파이프라인을 설계한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 링 버퍼 + 에너지 VAD를 시뮬레이션하고, 가짜 10초 스트림에 대해 단계별 지연 시간을 출력한다.
2. **보통.** `sounddevice`를 써서 마이크를 20ms 프레임으로 처리하고 각 프레임에서 VAD 상태를 출력하는 패스스루(passthrough) 루프를 만든다.
3. **어려움.** `aiortc`로 풀 듀플렉스 에코 테스트를 만든다. 브라우저 → WebRTC → 파이썬 → WebRTC → 브라우저. 1kHz 펄스로 글래스-투-글래스(glass-to-glass) 지연 시간을 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Ring buffer | 순환 큐 | 오디오 프레임용 고정 크기, 락프리(또는 SPSC-락) FIFO. |
| VAD | 침묵 게이트 | 음성 대 비음성을 표시하는 모델 또는 휴리스틱. |
| Streaming ASR | 실시간 STT | 오디오가 도착하는 대로 부분 텍스트를 내보냄. 룩어헤드 제한. |
| Jitter buffer | 네트워크 평활기 | 순서가 뒤바뀐 패킷을 재정렬하는 큐. 보통 60-80ms. |
| AEC | 에코 제거 | 스피커-마이크 피드백 경로를 빼냄. |
| Barge-in | 사용자 인터럽트 | 시스템이 TTS 도중 사용자 음성을 감지함. 재생을 취소해야 함. |
| Full duplex | 양방향 동시 | 사용자와 봇이 동시에 말할 수 있음. Moshi가 풀 듀플렉스. |

## 더 읽을거리 (Further Reading)

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743): 청크 기반 준스트리밍 Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf): 풀 듀플렉스 200ms 지연 시간.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/): 프로덕션 오디오 에이전트 오케스트레이션.
- [Silero VAD repo](https://github.com/snakers4/silero-vad): 1ms 미만 VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/): 오픈소스 에코 제거.
