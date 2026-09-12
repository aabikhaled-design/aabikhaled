# 음성 에이전트: Pipecat과 LiveKit

> 음성 에이전트(voice agent)는 2026년에 일급 프로덕션 범주다. Pipecat은 Python 프레임(frame) 기반 파이프라인(VAD → STT → LLM → TTS → 트랜스포트)을 제공한다. LiveKit Agents는 WebRTC를 통해 AI 모델(model)을 사용자에게 연결한다. 프리미엄 스택의 프로덕션 지연 시간(latency) 목표는 종단 간(end-to-end) 450~600ms에 안착한다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- Pipecat의 프레임 기반 파이프라인을 기술하기: DOWNSTREAM(소스→싱크)와 UPSTREAM(제어).
- 정규(canonical) 음성 파이프라인 단계와 Pipecat이 지원하는 트랜스포트를 거명하기.
- LiveKit Agents의 두 가지 음성 에이전트 클래스(MultimodalAgent, VoicePipelineAgent)와 각각이 언제 들어맞는지 설명하기.
- 2026년 프로덕션 지연 시간 기대치와 그것이 어떻게 아키텍처 선택을 좌우하는지 요약하기.

## 문제 (The Problem)

음성 에이전트는 TTS를 덧붙인 텍스트 루프가 아니다. 지연 시간 예산이 가혹하고(~600ms), 부분 오디오가 기본값이며, 턴 감지(turn detection)는 하나의 모델이고, 트랜스포트는 전화망 SIP부터 WebRTC까지 다양하다. 선택지는 프레임 기반 파이프라인을 직접 구축하거나(Pipecat) 플랫폼에 기대는(LiveKit) 둘 중 하나다.

## 개념 (The Concept)

### Pipecat (pipecat-ai/pipecat)

- Python 프레임 기반 파이프라인 프레임워크.
- `Frame` → `FrameProcessor` 체인.
- 두 가지 흐름 방향:
  - **DOWNSTREAM**: 소스 → 싱크(오디오 입력, TTS 출력).
  - **UPSTREAM**: 피드백과 제어(취소, 지표, 끼어들기(barge-in)).
- `PipelineTask`는 이벤트(`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`)와 지표/추적/RTVI를 위한 옵저버로 라이프사이클을 관리한다.

전형적인 파이프라인:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

트랜스포트: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows는 구조화된 대화(상태 기계)를 추가한다. Pipecat Cloud는 관리형 런타임(runtime)이다.

### LiveKit Agents (livekit/agents)

- WebRTC를 통해 AI 모델을 사용자에게 연결한다.
- 핵심 개념: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- 두 가지 음성 에이전트 클래스:
  - **MultimodalAgent**: OpenAI Realtime 또는 동등물을 통한 직접 오디오.
  - **VoicePipelineAgent**: STT → LLM → TTS 캐스케이드; 텍스트 수준 제어를 제공.
- 트랜스포머(transformer) 모델을 통한 시맨틱 턴 감지(semantic turn detection).
- 네이티브 MCP 통합.
- SIP를 통한 전화망.
- LiveKit Inference를 통해 API 키 없이 50개 이상의 모델; 플러그인을 통해 200개 이상 추가.

### 상업용 플랫폼

Vapi(최적화된 프리미엄 스택에서 ~450~600ms)와 Retell(180회 테스트 통화에 걸쳐 종단 간 ~600ms)이 이들 위에 구축된다. WebRTC 팀 없이 관리형 음성 스택을 원할 때 플랫폼을 선택하라.

### 이 패턴이 잘못되는 지점

- **끼어들기 처리 없음.** 사용자가 끼어드는데 에이전트가 계속 말한다. Pipecat에서는 UPSTREAM 취소 프레임이, LiveKit에서는 동등물이 필요하다.
- **STT 신뢰도 무시.** 신뢰도가 낮은 전사(transcript)가 사실인 양 그대로 LLM에 들어간다. 신뢰도로 게이트하거나 확인을 요청하라.
- **TTS 문장 중간 끊김.** 파이프라인이 발화 중간에 취소될 때, TTS는 이를 알거나 오디오를 끊어야 한다.
- **지연 시간 예산 무시.** 구성요소마다 50~200ms씩 더해진다. 출시 전에 체인 전체를 합산해 보라.

### 전형적인 2026년 지연 시간

- VAD: 20~60ms
- STT 부분(partial): 100~250ms
- LLM 첫 토큰: 150~400ms
- TTS 첫 오디오: 100~200ms
- 트랜스포트 RTT: 30~80ms

종단 간 450~600ms는 프리미엄이다. 800~1200ms는 흔하다. 1500ms를 넘는 것은 고장난 느낌이다.

```figure
voice-pipeline
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 갖춘 프레임 기반 장난감 파이프라인이다:

- `Frame` 타입(audio, transcript, text, tts_audio, control).
- `process(frame)`을 갖춘 `Processor` 인터페이스.
- 스크립트화된 프로세서로 구성한 5단계 파이프라인(VAD → STT → LLM → TTS → 트랜스포트).
- 끼어들기를 시연하는 UPSTREAM 취소 프레임.

실행:

```
python3 code/main.py
```

트레이스는 정상 흐름과 발화 중간에 TTS를 멈추는 끼어들기 취소를 보여준다.

## 라이브러리로 써보기 (Use It)

- 완전한 제어에는 **Pipecat**: 맞춤 프로세서, Python 우선, 플러그형 제공자.
- WebRTC 우선 배포와 전화망에는 **LiveKit Agents**.
- WebRTC 팀 없는 호스팅형 음성 에이전트에는 **Vapi / Retell**.
- 직접 오디오 입력/오디오 출력(MultimodalAgent)에는 **OpenAI Realtime / Gemini Live**.

## 산출물 (Ship It)

`outputs/skill-voice-pipeline.md`는 VAD + STT + LLM + TTS + 트랜스포트에 끼어들기 처리를 더한 Pipecat 형태의 음성 파이프라인을 스캐폴딩한다.

## 연습 문제 (Exercises)

1. 장난감 파이프라인에 지표 옵저버를 추가하라: 초당 단계별 프레임 수를 세어라. 지연 시간이 어디에 쌓이는가?
2. 신뢰도 게이트 STT를 구현하라: 임계값 미만이면 "다시 말씀해 주시겠어요?"를 요청하라.
3. 시맨틱 턴 감지를 추가하라: 단순 규칙: 전사가 "?"로 끝나면 턴의 끝.
4. Pipecat의 트랜스포트 문서를 읽어라. stdlib 트랜스포트를 SmallWebRTCTransport 설정(스텁)으로 교체하라.
5. 같은 쿼리에 대해 OpenAI Realtime 대 STT+LLM+TTS 캐스케이드를 측정하라. 텍스트 수준 제어가 짊어지는 지연 시간 비용은 얼마인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Frame | "이벤트" | 파이프라인 내 타입이 지정된 데이터 단위(audio, transcript, text, control) |
| Processor | "파이프라인 단계" | process(frame)을 갖춘 핸들러 |
| DOWNSTREAM | "순방향 흐름" | 소스에서 싱크로: 오디오 입력, 음성 출력 |
| UPSTREAM | "피드백 흐름" | 제어: 취소, 지표, 끼어들기 |
| VAD | "음성 활동 감지" | 사용자가 말하고 있는 때를 감지 |
| Semantic turn detection | "스마트 턴 종료" | 사용자가 끝났다는 모델 기반 결정 |
| MultimodalAgent | "직접 오디오 에이전트" | 오디오 입력, 오디오 출력; 중간에 텍스트 없음 |
| VoicePipelineAgent | "캐스케이드 에이전트" | STT + LLM + TTS; 텍스트 수준 제어 |

## 더 읽을거리 (Further Reading)

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction): 프레임 기반 파이프라인, 프로세서, 트랜스포트
- [LiveKit Agents docs](https://docs.livekit.io/agents/): WebRTC + 음성 프리미티브
- [Vapi](https://vapi.ai/): 관리형 음성 플랫폼
- [Retell AI](https://www.retellai.com/): 관리형 음성, 지연 시간 벤치마크
