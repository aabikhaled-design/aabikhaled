# 음성 비서 파이프라인 만들기(Build a Voice Assistant Pipeline): Phase 6 캡스톤

> 레슨 01-11의 모든 것을 하나로 엮는다. 듣고, 추론하고, 다시 말하는 음성 비서를 만든다. 2026년에 이것은 연구 문제가 아니라 이미 해결된 엔지니어링 문제다. 다만 통합 세부사항이 출시 여부를 가른다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120분

## 문제 (The Problem)

엔드-투-엔드 비서를 만든다.

1. 마이크 입력(16kHz 모노)을 캡처한다.
2. 사용자 음성의 시작/끝을 감지한다.
3. 스트리밍으로 전사한다.
4. 전사 결과를 도구(타이머, 날씨, 캘린더)를 호출할 수 있는 LLM에 넘긴다.
5. LLM 텍스트를 TTS로 스트리밍한다.
6. 오디오를 사용자에게 재생한다.
7. 응답 도중 사용자가 끼어들면 멈춘다.

지연 시간(latency) 목표: 노트북 CPU에서 사용자가 발화를 마친 뒤 800ms 이내에 첫 TTS 오디오 바이트. 품질 목표: 누락된 단어 없음, 정적에서의 환각 자막 없음, 음성 복제(voice cloning) 유출 없음, 프롬프트 주입(prompt injection) 성공 없음.

## 개념 (The Concept)

![음성 비서 파이프라인: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 일곱 가지 구성요소

1. **오디오 캡처(Audio capture).** 마이크 → 16kHz 모노 → 20ms 청크. 보통 파이썬에서는 `sounddevice`, 프로덕션(production)에서는 네이티브 AudioUnit/ALSA/WASAPI.
2. **VAD (레슨 11).** Silero VAD @ 임계값 0.5, 최소 음성 250ms, 침묵 행오버(hang-over) 500ms. "시작"과 "끝"을 신호로 보낸다.
3. **스트리밍 STT (레슨 4-5).** Whisper-streaming, Parakeet-TDT, 또는 Deepgram Nova-3(API). 부분 + 최종 전사.
4. **도구 호출이 가능한 LLM.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. 도구용 JSON 스키마. 토큰을 스트리밍한다.
5. **스트리밍 TTS (레슨 7).** Kokoro-82M(가장 빠른 오픈) 또는 Cartesia Sonic(상용). LLM 토큰 20개 후에 TTS를 시작한다.
6. **재생(Playback).** 스피커 출력. 저대역폭 네트워크용 opus 인코딩.
7. **인터럽션 핸들러.** TTS 재생 중 VAD가 발동하면, 재생을 멈추고, LLM을 취소하고, STT를 재시작한다.

### 마주치게 될 세 가지 실패 모드

1. **첫 단어 잘림(First-word clip).** VAD가 한 박자 늦게 시작한다. 사용자의 "hey"가 빠진다. 시작 임계값을 0.5가 아닌 0.3으로 한다.
2. **응답 중간 인터럽트 혼란.** 사용자가 끼어든 뒤에도 LLM이 계속 생성한다. 비서가 사용자 말 위로 떠든다. VAD → LLM 취소를 연결하라.
3. **정적 환각(Silence hallucination).** Whisper가 조용한 워밍업 프레임에서 "Thanks for watching"을 출력한다. 항상 VAD로 게이팅하라.

### 2026년 프로덕션 레퍼런스 스택

| 스택 | 지연 시간 | 라이선스 | 비고 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026년 업계 기본값 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY 친화적 |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | 단일 모델, 다른 아키텍처, 레슨 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 가장 빠른 출시, 제한적 커스터마이징 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | 프라이버시 / 엣지 |

```figure
v4-voice-latency
```

## 직접 만들기 (Build It)

### 1단계: 청킹을 동반한 마이크 캡처 (의사코드)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 2단계: VAD로 게이팅된 턴 캡처

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 3단계: 스트리밍 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 4단계: LLM 루프 안에서의 도구 호출

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 5단계: 인터럽션 처리

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 라이브러리로 써보기 (Use It)

스텁(stub) 모델로 일곱 구성요소를 모두 엮는 실행 가능한 시뮬레이션은 `code/main.py`를 참고하라. 하드웨어 없이도 파이프라인 형태를 볼 수 있다. 실제 구현에서는 스텁을 다음으로 교체한다.

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 또는 `openai-whisper`
- `openai` (`gpt-4o`) 또는 `anthropic`
- `kokoro` 또는 `cartesia`
- I/O용 `sounddevice`

## 함정 (Pitfalls)

- **PII를 영구 로깅하기.** 전체 턴 오디오는 대부분의 관할권에서 PII다. 30일 보존, 저장 시 암호화.
- **끼어들기(barge-in) 없음.** 사용자는 끼어든다. 비서는 말을 멈춰야 한다.
- **블로킹하는 TTS.** 동기 TTS는 이벤트 루프를 막는다. 비동기나 별도 스레드를 써라.
- **도구 호출 오류 처리 없음.** 도구는 실패한다. LLM은 오류를 돌려받아 한 번 재시도한 뒤, 성능을 단계적으로 낮춰야 한다.
- **지나치게 공격적인 환각 필터.** 과하게 필터링하면 비서가 "그건 도와드릴 수 없습니다"를 반복한다. 덜 필터링하면 아무 말이나 한다. 홀드아웃(held-out) 셋에서 보정하라.
- **웨이크 워드(wake-word) 옵션 없음.** 항상 듣기는 프라이버시 책임 문제다. 웨이크 워드 게이트(Porcupine 또는 openWakeWord)를 추가하라.

## 산출물 (Ship It)

`outputs/skill-voice-assistant-architect.md`로 저장한다. 예산 + 규모 + 언어 + 컴플라이언스 제약이 주어지면 전체 스택 사양을 생성한다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 스텁 모듈로 전체 한 턴을 엔드-투-엔드로 시뮬레이션하고 단계별 지연 시간을 출력한다.
2. **보통.** STT 스텁을 사전 녹음된 `.wav`에 대한 실제 Whisper 모델로 교체한다. WER와 엔드-투-엔드 지연 시간을 측정한다.
3. **어려움.** 도구 호출을 추가한다. `get_weather`(아무 API)와 `set_timer`를 구현한다. LLM을 도구를 통해 라우팅하고, 사용자가 "5분 타이머 맞춰줘"라고 했을 때 올바른 함수가 발동하고 음성 응답이 이를 확인하는지 검증한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Turn | 사용자 + 비서 왕복 | VAD로 경계 지어진 사용자 음성 하나 + LLM-TTS 응답 하나. |
| Barge-in | 인터럽션 | 비서가 말하는 동안 사용자가 말함. 비서가 멈춤. |
| Wake word | "Hey assistant" | 짧은 키워드 탐지기. Porcupine, Snowboy, openWakeWord. |
| End-pointing | 턴 종료 | 사용자가 끝냈다는 VAD + 최소 침묵 판정. |
| Pre-roll | 발화 전 버퍼 | 첫 단어 잘림을 피하려고 VAD 발동 전 200-400ms 오디오를 유지. |
| Tool call | 함수 호출 | LLM이 JSON을 내보냄. 런타임이 디스패치하고 결과가 루프 안으로 되먹임. |

## 더 읽을거리 (Further Reading)

- [LiveKit: voice agent quickstart](https://docs.livekit.io/agents/) 프로덕션급 레퍼런스.
- [Pipecat: voice agent examples](https://github.com/pipecat-ai/pipecat) DIY 친화적 프레임워크.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime): 매니지드 음성 네이티브 경로.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi): 풀 듀플렉스 레퍼런스 (레슨 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/): 웨이크 워드 게이팅.
- [Anthropic: tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) LLM 함수 호출.
