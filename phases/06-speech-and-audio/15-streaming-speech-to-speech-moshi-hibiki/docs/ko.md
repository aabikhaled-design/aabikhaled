# 스트리밍 음성-투-음성(Streaming Speech-to-Speech): Moshi, Hibiki, 그리고 풀 듀플렉스 대화

> 2024-2026년은 음성 AI를 재정의했다. Moshi는 200ms 지연 시간(latency)으로 듣기와 말하기를 동시에 하는 단일 모델을 출시한다. Hibiki는 청크 단위로 음성-투-음성 번역을 한다. 둘 다 ASR → LLM → TTS 파이프라인을 버리고 Mimi 코덱(codec) 토큰(token) 위의 통합 풀 듀플렉스(full-duplex) 아키텍처를 택한다. 이것이 새로운 레퍼런스 설계다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75분

## 문제 (The Problem)

레슨 11 + 12로 만든 모든 음성 에이전트(agent)는 약 300-500ms의 근본적 지연 시간 하한선이 있다. VAD가 발동하고 STT가 처리하고 LLM이 추론하고 TTS가 생성한다. 각 단계마다 자체 최소 지연 시간이 있다. 튜닝하고 병렬화할 수 있지만, 파이프라인 형태가 한계를 정한다.

Moshi(Kyutai, 2024-2026)는 다른 질문을 던진다. 파이프라인이 없다면 어떨까? 한 모델이 오디오를 입력받아 오디오를 직접, 연속적으로 내보내고, 텍스트는 필수 단계가 아니라 중간 "내적 독백(inner monologue)"이라면 어떨까?

답은 **풀 듀플렉스 음성-투-음성(full-duplex speech-to-speech)**이다. 이론적 지연 시간 160ms(80ms Mimi 프레임 + 80ms 음향 지연). 실용적 지연 시간은 단일 L4 GPU에서 200ms다. 최고 수준의 파이프라인 음성 에이전트가 달성하는 것의 절반이다.

## 개념 (The Concept)

![Moshi 아키텍처: 두 개의 병렬 Mimi 스트림 + 내적 독백 텍스트](../assets/moshi-hibiki.svg)

### Moshi 아키텍처

**입력.** 둘 다 12.5Hz × 8개 코드북인 두 개의 Mimi 코덱 스트림:

- 스트림 1: 사용자 오디오(Mimi 인코딩됨, 계속 도착)
- 스트림 2: Moshi 자신의 오디오(Moshi가 생성)

**트랜스포머(transformer).** 7B 파라미터 시간 트랜스포머(Temporal Transformer)가 두 스트림과 텍스트 "내적 독백" 스트림을 처리한다. 매 80ms 스텝마다 다음을 한다.

1. 최신 사용자 Mimi 토큰(8개 코드북)을 소비한다.
2. 가장 최근의 Moshi Mimi 토큰(생성된 대로 8개 코드북)을 소비한다.
3. 다음 Moshi 텍스트 토큰(내적 독백)을 생성한다.
4. 다음 Moshi Mimi 토큰(작은 깊이 트랜스포머(Depth Transformer)를 통해 8개 코드북)을 생성한다.

세 스트림 모두(사용자 오디오, Moshi 오디오, Moshi 텍스트)가 병렬로 실행된다. Moshi는 말하면서 사용자를 들을 수 있고 사용자가 끼어들면 자신을 중단할 수 있으며, 주된 발화를 깨지 않고 백채널("mhm")을 할 수 있다.

**깊이 트랜스포머.** 한 프레임 안에서 8개 코드북은 병렬로 예측되지 않는다. 코드북 간 의존성이 있다. 작은 2층 "깊이 트랜스포머"가 80ms 안에서 이들을 순차적으로 예측한다. 이것이 AR 코덱 LM의 표준 인수분해(factorization)다(VALL-E, VibeVoice도 사용).

### 내적 독백 텍스트가 도움이 되는 이유

명시적 텍스트가 없으면, 모델은 음향 스트림에서 언어를 암묵적으로 모델링해야 한다. Moshi의 통찰: 오디오와 나란히 텍스트 토큰을 내보내도록 강제한다. 텍스트 스트림은 본질적으로 Moshi가 말하는 내용의 전사(transcript)다. 이는 의미적 일관성을 개선하고 언어 모델 헤드를 교체하기 쉽게 하며, 전사를 공짜로 준다.

### Hibiki: 스트리밍 음성-투-음성 번역

같은 아키텍처를, 번역 쌍으로 학습한다. 소스 오디오를 입력받아 대상 언어 오디오를 연속적으로 출력한다. Hibiki-Zero(2026년 2월)는 단어 단위로 정렬된 학습 데이터의 필요성을 없앤다. 문장 단위 데이터 + 지연 시간 최적화를 위한 GRPO 강화 학습(reinforcement learning)을 쓴다.

초기에 네 개의 언어 쌍을 지원한다. 약 1000시간으로 새 언어에 적응시킬 수 있다.

### 더 넓은 Kyutai 스택 (2026)

- **Moshi**: 풀 듀플렉스 대화(프랑스어 우선, 영어 잘 지원)
- **Hibiki / Hibiki-Zero**: 동시(simultaneous) 음성 번역
- **Kyutai STT**: 스트리밍 ASR(500ms 또는 2.5초 룩어헤드)
- **Kyutai Pocket TTS**: CPU에서 동작하는 100M 파라미터 TTS(2026년 1월)
- **Unmute**: 공개 서버에서 이들을 결합한 전체 파이프라인

L40S GPU에서의 처리량(throughput): 3배속으로 64개 동시 세션.

### Sesame CSM: 사촌격

Sesame CSM(2025)은 비슷한 아이디어를 쓴다. Mimi 코덱 헤드를 단 Llama-3 백본이다. 하지만 CSM은 풀 듀플렉스가 아니라 단방향이다(맥락 + 텍스트를 입력받아 음성을 생성). 시장에서 최고의 "음성 존재감(voice presence)" TTS다. Moshi의 풀 듀플렉스 능력과는 다소 다르다.

### 2026년 성능 수치

| 모델 | 지연 시간 | 사용 사례 | 라이선스 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

```figure
sp-fullduplex
```

## 직접 만들기 (Build It)

### 1단계: 인터페이스

Moshi는 80ms Mimi 인코딩 오디오 청크를 받아 80ms Mimi 인코딩 오디오 청크를 반환하는 WebSocket 서버를 노출한다. 양방향으로. 끊임없이.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 2단계: 풀 듀플렉스 루프

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

두 방향이 동시에 실행된다. 파이썬 asyncio 또는 Rust futures가 표준 전송 방식이다.

### 3단계: 학습 목표 (개념적)

매 80ms 프레임 `t`에 대해:

- 입력: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 예측: `moshi_text[t]`, 그다음 `moshi_mimi[t, codebook_0..7]`

텍스트가 오디오보다 먼저 예측된다(내적 독백). 오디오는 깊이 트랜스포머 안에서 코드북 순차적으로 예측된다.

### 4단계: Moshi가 이기는 곳과 그렇지 않은 곳

Moshi가 이긴다:

- 저렴한 하드웨어에서 250ms 미만 엔드-투-엔드.
- 자연스러운 백채널과 인터럽션.
- 파이프라인 글루 코드 없음.

Moshi가 이기지 못한다:

- 도구 호출(이를 위해 학습되지 않음. 별도 LLM 경로가 필요).
- 긴 추론(Moshi는 8B 정도의 대화 모델이지 Claude/GPT-4가 아니다).
- 틈새 주제에 대한 사실 정확성.
- 대부분의 프로덕션(production) 엔터프라이즈 사용 사례(2026년에도 여전히 파이프라인을 쓴다).

## 라이브러리로 써보기 (Use It)

| 상황 | 선택 |
|-----------|------|
| 최저 지연 음성 동반자 | Moshi |
| 실시간 번역 통화 | Hibiki |
| 음성 데모 / 연구 | Moshi, CSM |
| 도구가 있는 엔터프라이즈 에이전트 | 파이프라인 (레슨 12), Moshi 아님 |
| 맥락 내 커스텀 음성 TTS | Sesame CSM |
| 음성-투-음성, 모든 언어 | GPT-4o Realtime 또는 Gemini 2.5 Live (상용) |

## 함정 (Pitfalls)

- **제한적 도구 호출.** Moshi는 에이전트 프레임워크가 아니라 대화 모델이다. 도구를 위해 파이프라인과 결합하라.
- **특정 음성 조건화.** Moshi는 단일 학습된 페르소나를 쓴다. 복제(cloning)는 별도 학습 실행이다.
- **언어 커버리지.** 프랑스어 + 영어는 훌륭하다. 다른 언어는 제한적이다. Hibiki-Zero가 돕지만, 여전히 학습 데이터가 필요하다.
- **자원 비용.** 전체 Moshi 세션은 GPU 슬롯을 점유한다. 저렴한 공유 테넌트 배포 패턴이 아니다.

## 산출물 (Ship It)

`outputs/skill-duplex-pipeline.md`로 저장한다. 음성 에이전트 워크로드에 대해 파이프라인 vs 풀 듀플렉스 아키텍처를 이유와 함께 고른다.

## 연습 문제 (Exercises)

1. **쉬움.** `code/main.py`를 실행한다. 두 스트림 + 내적 독백 아키텍처를 상징적으로 시뮬레이션한다.
2. **보통.** HuggingFace에서 Moshi를 받아 서버를 실행하고 대화 하나를 테스트한다. 사용자 음성 종료부터 Moshi 응답 시작까지의 실제 시계 지연 시간을 측정한다.
3. **어려움.** 레슨 12의 파이프라인 에이전트를 가져와 20개의 매칭된 테스트 발화에서 P50 지연 시간을 Moshi와 비교한다. 그럼에도 파이프라인이 아키텍처적으로 이기는 경우를 정리한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Full-duplex | 듣기-말하기 동시 | 같은 모델에서 두 오디오 스트림이 동시에 활성. |
| Inner monologue | 모델의 텍스트 스트림 | Moshi가 오디오 출력과 나란히 텍스트 토큰을 내보냄. |
| Depth transformer | 코드북 간 예측기 | 한 80ms 프레임 안에서 8개 코드북을 예측하는 작은 트랜스포머. |
| Mimi | Kyutai의 코덱 | 12.5Hz × 8개 코드북. 의미+음향. Moshi를 구동. |
| Streaming S2S | 오디오 → 오디오 실시간 | 청크 단위 번역/대화, 파이프라인 단계 없음. |
| Back-channeling | "Mhm" 반응 | Moshi가 자기 턴을 깨지 않고 작은 맞장구를 낼 수 있음. |

## 더 읽을거리 (Further Reading)

- [Défossez et al. (2024). Moshi: speech-text foundation model](https://arxiv.org/html/2410.00037v2) 그 논문.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345): 정렬 데이터 없는 스트리밍 번역.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice): CSM 사양.
- [Kyutai: Moshi repo](https://github.com/kyutai-labs/moshi) 설치 + 서버.
- [OpenAI: Realtime API](https://platform.openai.com/docs/guides/realtime) 클로즈드 상용 동급.
- [Kyutai: Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) 내부의 STT/TTS 프레임워크.
