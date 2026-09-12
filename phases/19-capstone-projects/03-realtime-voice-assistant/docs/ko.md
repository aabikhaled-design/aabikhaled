# Capstone 03: 실시간 음성 어시스턴트 (ASR to LLM to TTS)

> 제대로 느껴지는 음성 에이전트(voice agent)는 종단 간(end-to-end) 지연 시간(latency)이 800ms 미만이고, 사용자가 말을 멈춘 순간을 알고, 끼어들기(barge-in)를 처리하고, 멈칫하지 않고 도구를 호출한다. Retell, Vapi, LiveKit Agents, Pipecat는 모두 2026년에 이 기준을 충족하며, 그 방식도 같다. 스트리밍 ASR, 턴 감지기(turn-detector), 스트리밍 LLM, 스트리밍 TTS를 매 홉(hop)마다 공격적인 지연 시간 예산을 두고 WebRTC로 연결한다. 하나 만들어서 WER과 MOS와 거짓 끊김 비율(false-cutoff rate)을 측정하고, 패킷 손실(packet loss) 하에서 실행하라.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 hours

## 문제 (Problem)

음성은 2025-2026년에 가장 빠르게 움직인 AI UX 범주였다. 기술적 천장은 분기마다 낮아졌다. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, Pipecat 0.0.70은 모두 800ms 미만의 첫 오디오 출력을 손에 닿는 거리에 두었다. 기준은 지연 시간만이 아니라 상호작용의 느낌이다. 사용자의 말을 끊지 않고, 끊기지 않고, 문장 중간의 중단에서 회복하고, 오디오를 멈칫하게 하지 않으면서 대화 중간에 도구를 호출하고, 지터(jitter)가 심한 모바일 네트워크에서 살아남는 것이다.

REST 호출 세 개를 꿰매서는 거기에 도달할 수 없다. 아키텍처는 종단 간 파이프라인 스트리밍이다. 이를 직접 만들면 실패 모드가 보이기 시작한다. 전화 오디오에 맞춰진 VAD가 배경 TV 소리에 발동하고, 결코 오지 않는 문장 부호를 기다리는 턴 감지기, 출력 전에 400ms를 버퍼링하는 TTS. 캡스톤(capstone)은 부하 하에서 이것들을 하나씩 고치고 지연 시간-품질 보고서를 발행하는 것이다.

## 개념 (Concept)

파이프라인에는 다섯 개의 스트리밍 단계가 있다. **오디오 입력**(브라우저 또는 PSTN으로부터의 WebRTC), **ASR**(Deepgram Nova-3 또는 faster-whisper로부터의 스트리밍 부분 전사), **턴 감지**(VAD에 더해, 부분 전사를 읽어 완료 단서를 찾는 작은 턴 감지기 모델), **LLM**(턴이 완료된 것으로 판단되는 즉시 스트리밍되는 토큰(token)), **TTS**(첫 LLM 토큰으로부터 ~200ms 이내에 스트리밍되는 오디오 출력).

세 가지 횡단 관심사가 있다. **끼어들기(Barge-in)**: 에이전트가 말하는 동안 사용자가 말하기 시작하면, TTS가 취소되고 ASR이 즉시 받아 든다. **도구 사용**: 대화 중간의 함수 호출(날씨, 캘린더)은 오디오를 멈칫하게 하지 않으면서 사이드 채널(side channel)에서 실행되어야 한다. 지연 시간이 300ms를 초과하면 에이전트는 확인 토큰("잠시만요...")을 미리 채운다. **백프레셔(Backpressure)**: 패킷 손실 하에서 부분 전사는 보류되고, VAD는 음성 게이트(speech-gate) 임계값을 높이고, 에이전트는 확인되지 않은 메시지 위에 말하는 것을 피한다.

측정 기준은 정량적이다. 15 dB SNR에서 Hamming VAD 벤치마크(benchmark) 기준 WER 8% 미만. 측정된 100건의 통화에서 첫 오디오 출력 p50 800ms 미만. 거짓 끊김 비율 3% 미만. TTS는 MOS 4.2 초과. 단일 g5.xlarge에서 50개의 동시 통화. 이 숫자들이 결과물이다.

## 아키텍처 (Architecture)

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 스택 (Stack)

- 전송: LiveKit Agents 1.0 (WebRTC)에 더해 Twilio PSTN 게이트웨이; 대안 프레임워크로 Pipecat 0.0.70
- ASR: Deepgram Nova-3(스트리밍, 첫 부분 전사 300ms 미만) 또는 자체 호스팅된 faster-whisper Whisper-v3-turbo
- VAD: Silero VAD v5에 더해 LiveKit 턴 감지기(부분 전사를 읽는 작은 트랜스포머(transformer))
- LLM: 긴밀한 통합을 위한 OpenAI GPT-4o-realtime, Gemini 2.5 Flash Live, 또는 캐스케이드된 Claude Haiku 4.5(스트리밍 완료, 별도 오디오 경로)
- TTS: Cartesia Sonic-2(가장 낮은 첫 바이트), ElevenLabs Flash v3, 또는 자체 호스팅용 오픈소스 Orpheus
- 도구: 날씨/캘린더/예약을 위한 FastMCP 사이드 채널; 도구가 300ms를 초과하면 에이전트는 필러(filler)를 미리 방출
- 관측성(observability): OpenTelemetry 음성 스팬(span), 오디오 재생이 가능한 Langfuse 음성 트레이스
- 배포: 자체 호스팅 Whisper + Orpheus를 위한 단일 g5.xlarge(24GB VRAM); 가장 낮은 지연 시간을 위한 호스팅 API

```figure
ce-voice-latency
```

## 직접 만들기 (Build It)

1. **WebRTC 세션.** LiveKit 룸과 마이크 오디오를 스트리밍하는 웹 클라이언트를 띄운다. 서버에서, 그 룸에 합류하는 에이전트 워커를 붙인다.

2. **ASR 스트리밍.** 20ms PCM 프레임을 Deepgram Nova-3(또는 GPU의 faster-whisper)에 공급한다. 부분 및 최종 전사를 구독한다. 부분 전사당 지연 시간을 로깅한다.

3. **VAD와 턴 감지기.** 프레임 스트림에 Silero VAD v5를 실행한다. 음성 종료 이벤트 시, 가장 최근의 부분 전사에 LiveKit 턴 감지기를 발동시킨다. VAD가 500ms 동안 침묵이라고 말하고 턴 감지기가 완료 점수 > 0.6을 매길 때만 "턴 완료"로 확정한다.

4. **LLM 스트림.** 턴 완료 시, 진행 중인 대화에 최종 전사를 더해 LLM 호출을 시작한다. 토큰을 스트리밍한다. 첫 토큰에서 TTS로 넘긴다.

5. **TTS 스트림.** Cartesia Sonic-2가 오디오 청크(chunk)를 스트리밍해 돌려준다. 첫 청크는 첫 LLM 토큰으로부터 200ms 이내에 서버를 떠나야 한다. LiveKit 룸으로 청크를 방출한다. 클라이언트는 WebRTC 지터 버퍼(jitter buffer)를 통해 재생한다.

6. **끼어들기.** TTS가 재생되는 동안 VAD가 새로운 사용자 음성을 감지하면, TTS 스트림을 즉시 취소하고, 남은 LLM 출력을 버리고, ASR을 재무장한다. `tts_canceled` 스팬을 발행한다.

7. **도구 사이드 채널.** 날씨와 캘린더를 함수 호출 도구로 등록한다. 호출되면, 호출을 동시에 발동시킨다. 300ms 이내에 해결되지 않으면, LLM이 "잠시만요, 확인해 볼게요"를 필러로 방출하게 한다. 도구가 반환되면 재개한다.

8. **평가 하네스(harness).** 100건의 통화를 녹음한다. WER(홀드아웃 전사 대비), 거짓 끊김 비율(사용자가 문장 중간일 때 TTS가 취소됨), 첫 오디오 출력 p50, TTS MOS(사람 또는 NISQA), 그리고 지터-손실 테스트(패킷의 3%를 드롭)를 계산한다.

9. **부하 테스트.** 합성 발신자(synthetic caller)로 단일 g5.xlarge에서 50개의 동시 통화를 구동한다. 지속되는 첫 오디오 출력 p95를 측정한다.

## 라이브러리로 써보기 (Use It)

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 산출물 (Ship It)

`outputs/skill-voice-agent.md`이 결과물이다. 도메인(고객 지원, 일정 관리, 또는 키오스크)이 주어지면, 측정 기준에 맞춰 튜닝된 ASR/VAD/LLM/TTS 파이프라인을 가진 LiveKit 에이전트를 띄운다. 평가 기준:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 종단 간 지연 시간 | 녹음된 100건의 통화에 걸쳐 첫 오디오 출력 p50 800ms 미만 |
| 20 | 턴 주고받기 품질 | Hamming VAD 벤치마크에서 거짓 끊김 비율 3% 미만 |
| 20 | 도구 사용 정확성 | 오디오를 멈칫하게 하지 않으면서 올바른 데이터를 반환하는 대화 중간 도구 호출 |
| 20 | 패킷 손실 하의 신뢰성 | 3% 패킷 드롭을 주입한 상태에서의 WER과 턴 주고받기 안정성 |
| 15 | 평가 하네스 완전성 | 공개된 설정으로 재현 가능한 측정 |
| **100** | | |

## 연습 문제 (Exercises)

1. Deepgram Nova-3을 g5.xlarge의 faster-whisper v3 turbo로 교체한다. 지연 시간과 WER 격차를 측정한다. CPU 대 GPU 결정이 어디서 중요한지 식별한다.

2. 중단 중재 정책을 추가한다. 도구 호출 중에 사용자가 끼어들면 에이전트는 무엇을 하는가? 세 가지 정책(하드 취소, 도구 완료 후 중지, 다음 턴 큐잉)을 비교한다.

3. 적대적 턴 감지기 테스트를 실행한다. 사용자에게 문장 중간에 긴 멈춤을 주게 한다. 900ms를 넘기지 않으면서 가장 낮은 거짓 끊김을 위해 VAD 침묵 임계값과 턴 감지기 점수 임계값을 튜닝한다.

4. Twilio를 통해 PSTN에 같은 에이전트를 배포한다. PSTN 첫 오디오 출력을 WebRTC와 비교한다. 지터 버퍼와 코덱(codec) 차이를 설명한다.

5. 비영어권 언어(일본어, 스페인어)에 대한 음성 활동 감지를 추가한다. Silero VAD v5의 거짓 발동 비율을 언어별 파인튜닝(fine-tune)과 비교 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 턴 감지(Turn detection) | "발화의 끝" | VAD 침묵과 부분 전사가 주어지면, 사용자가 말을 끝냈는지 결정하는 분류기(classifier) |
| 끼어들기(Barge-in) | "중단 처리" | VAD가 새로운 사용자 음성을 감지하면 재생 중인 TTS를 취소 |
| 첫 오디오 출력(First-audio-out) | "지연 시간" | 사용자가 말을 멈춘 시점부터 첫 오디오 패킷이 서버를 떠나는 시점까지의 시간 |
| VAD | "음성 게이트" | 오디오 프레임을 음성 대 침묵으로 분류하는 모델; Silero VAD v5가 2026 기본값 |
| 지터 버퍼(Jitter buffer) | "오디오 평활화" | 네트워크 변동을 흡수하기 위해 패킷을 잠시 보류하는 클라이언트 측 버퍼 |
| 필러(Filler) | "확인 토큰" | 도구가 느릴 때 침묵을 피하기 위해 에이전트가 방출하는 짧은 문구 |
| MOS | "평균 의견 점수" | 지각적 음성 품질 평가; NISQA가 자동화된 대리 지표 |

## 더 읽을거리 (Further Reading)

- [LiveKit Agents 1.0](https://github.com/livekit/agents): 레퍼런스 WebRTC 에이전트 프레임워크
- [Pipecat](https://github.com/pipecat-ai/pipecat): 대안 Python 우선 스트리밍 에이전트 프레임워크
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime): 통합 음성 모델 레퍼런스
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs): 스트리밍 ASR 레퍼런스
- [Silero VAD v5](https://github.com/snakers4/silero-vad): VAD 레퍼런스 모델
- [Cartesia Sonic-2](https://docs.cartesia.ai): 저지연 TTS 레퍼런스
- [Retell AI architecture](https://docs.retellai.com): 프로덕션 음성 에이전트 아키텍처
- [Vapi.ai production stack](https://docs.vapi.ai): 대안 프로덕션 레퍼런스
