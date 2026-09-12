# A2A: 에이전트 간(Agent-to-Agent) 프로토콜

> MCP는 에이전트-툴(agent-to-tool)이다. A2A(Agent2Agent)는 에이전트 간(agent-to-agent)이다. 서로 다른 프레임워크 위에 구축된 불투명한(opaque) 에이전트들이 협업하게 하는 오픈 프로토콜이다. 2025년 4월 Google이 공개했고, 2025년 6월 Linux Foundation에 기증되었으며, AWS, Cisco, Microsoft, Salesforce, SAP, ServiceNow를 포함한 150곳 이상의 지지자와 함께 2026년 4월 v1.0에 도달했다. IBM의 ACP를 흡수했고 AP2 결제 확장을 추가했다. 이 레슨은 에이전트 카드(Agent Card), 태스크(Task) 수명 주기, 그리고 두 가지 전송 바인딩(transport binding)을 살펴본다.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 에이전트-툴(MCP)과 에이전트 간(A2A) 사용 사례를 구별하기.
- 스킬(skill)과 엔드포인트 메타데이터를 담은 에이전트 카드를 `/.well-known/agent.json`에 게시하기.
- 태스크 수명 주기(submitted → working → input-required → completed / failed / canceled / rejected)를 따라가기.
- 파트(Part)(text, file, data)를 가진 메시지(Message)와 출력으로서의 아티팩트(Artifact)를 사용하기.

## 문제 (The Problem)

어느 고객 서비스 에이전트가 보고서 작성을 전문 작성 에이전트에 위임해야 한다. A2A 이전의 선택지:

- 커스텀 REST API. 작동하지만 모든 짝짓기가 일회성이다.
- 공유 코드베이스. 두 에이전트가 같은 프레임워크를 실행해야 한다.
- MCP. 맞지 않는다: MCP는 툴을 호출하기 위한 것이지, 두 에이전트가 각자의 불투명한 내부 추론을 보존하면서 협업하기 위한 것이 아니다.

A2A가 그 간극을 메운다. 한 에이전트가 다른 에이전트에 수명 주기, 메시지, 아티팩트를 갖춘 태스크를 보내는 식으로 상호작용을 모델링한다. 호출되는 에이전트의 내부 상태는 불투명하게 유지되고, 호출자는 태스크 상태 전이와 최종 출력만 본다.

A2A는 "프레임워크를 가로질러 에이전트들이 서로 대화하게 하는" 프로토콜이다. MCP를 대체하지 않으며; 둘은 상호 보완적이다.

## 개념 (The Concept)

### 에이전트 카드 (Agent Card)

모든 A2A 준수 에이전트는 `/.well-known/agent.json`에 카드를 게시한다:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

디스커버리는 URL 기반이다: 카드를 가져와 A2A 엔드포인트의 URL을 알아내고, 스킬을 열거한다.

### 서명된 에이전트 카드 (AP2)

AP2 확장(2025년 9월)은 에이전트 카드에 암호 서명을 추가한다. 게시자는 자신의 카드를 JWT로 서명하고, 소비자는 이를 검증한다. 사칭(impersonation)을 막는다.

### 태스크 수명 주기

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

클라이언트는 `tasks/send`로 시작한다. 호출되는 에이전트는 상태들을 거쳐 전이하고, 클라이언트는 SSE를 통해 상태 업데이트를 구독하거나 폴링한다.

### 메시지와 파트 (Messages and Parts)

메시지는 하나 이상의 파트를 운반한다:

- `text`: 일반 콘텐츠.
- `file`: mimeType을 가진 base64 블롭(blob).
- `data`: 타입이 지정된 JSON 페이로드(호출되는 에이전트를 위한 구조화된 입력).

예시:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 아티팩트 (Artifacts)

출력은 원시 문자열이 아니라 아티팩트다. 아티팩트는 명명되고 타입이 지정된 출력이다:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

아티팩트는 청크(chunk)로 스트리밍될 수 있다. 호출자가 누적한다.

### 두 가지 전송 바인딩

1. **HTTP 위의 JSON-RPC.** `/a2a` 엔드포인트, 요청에는 POST, 스트리밍에는 선택적 SSE. 기본 바인딩.
2. **gRPC.** gRPC가 네이티브인 엔터프라이즈 환경용.

두 바인딩 모두 동일한 논리적 메시지 형태를 운반한다.

### 불투명성 보존 (Opacity preservation)

핵심 설계 원칙: 호출되는 에이전트의 내부 상태는 불투명하다. 호출자는 태스크 상태와 아티팩트를 본다. 호출되는 에이전트의 사고 사슬(chain-of-thought), 그것의 툴 호출, 하위 에이전트 위임: 모두 보이지 않는다. 이는 툴 호출이 투명한 MCP와 다르다.

근거: A2A는 경쟁자들이 내부를 드러내지 않고 협업하게 한다. 호출자는 그 에이전트가 서비스를 어떻게 구현하는지 알지 못한 채 "이 고객 서비스 에이전트를 호출"하는 것일 수 있다.

### 타임라인

- **2025-04-09.** Google이 A2A를 발표한다.
- **2025-06-23.** Linux Foundation에 기증된다.
- **2025-08.** IBM의 ACP를 흡수한다.
- **2025-09.** AP2 확장(Agent Payments)이 출하된다.
- **2026-04.** 150곳 이상의 지지 조직과 함께 v1.0이 공개된다.

### MCP와의 관계

| 차원 | MCP | A2A |
|-----------|-----|-----|
| 사용 사례 | 에이전트-툴 | 에이전트 간 |
| 불투명성 | 투명한 툴 호출 | 불투명한 내부 추론 |
| 전형적 호출자 | 에이전트 런타임 | 또 다른 에이전트 |
| 상태 | 툴 호출 결과 | 수명 주기를 가진 태스크 |
| 인가 | OAuth 2.1 (Phase 13 · 16) | JWT 서명 에이전트 카드 (AP2) |
| 전송 | Stdio / Streamable HTTP | HTTP 위의 JSON-RPC / gRPC |

특정 툴을 호출하고 싶을 때 MCP를 사용하라. 전체 태스크를 다른 에이전트에 위임하고 싶을 때 A2A를 사용하라. 많은 프로덕션 시스템은 둘 다 사용한다: 에이전트가 툴 계층에는 MCP를, 협업 계층에는 A2A를 쓴다.

```figure
a2a-task-lifecycle
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 최소 A2A 하니스(harness)를 구현한다: 리서치 에이전트가 자신의 카드를 게시하고, 작성 에이전트가 PDF와 텍스트 지시를 포함한 파트와 함께 `tasks/send`를 받아, working → input_required → working → completed로 전이하고, 텍스트 아티팩트를 반환한다. 모두 stdlib이며, 메시지 형태에 집중하기 위해 인메모리 전송을 사용한다.

볼 것:

- 에이전트 카드 JSON 형태.
- 태스크 id 할당과 상태 전이.
- 혼합 타입 파트를 가진 메시지.
- 태스크 중간의 input-required 분기.
- 완료 시 아티팩트 반환.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-a2a-agent-spec.md`를 만든다. 다른 에이전트가 호출할 수 있어야 하는 새 에이전트가 주어지면, 이 스킬은 에이전트 카드 JSON, 스킬 스키마, 엔드포인트 청사진을 만들어낸다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 호출되는 에이전트가 명확화를 요청하는 input-required 일시 정지를 포함해 전체 태스크 수명 주기를 추적하라.

2. 서명된 에이전트 카드를 추가하라. 카드의 표준(canonical) JSON에 대해 HMAC으로 서명하라. 검증자를 작성하고 변형된 카드에서 실패하는지 확인하라.

3. 태스크 스트리밍을 구현하라: 작성 에이전트가 SSE를 통해 세 개의 점진적 아티팩트 청크를 방출하고 호출자가 그것들을 누적한다.

4. MCP 서버를 감싸는 A2A 에이전트를 설계하라. 각 MCP 툴을 A2A 스킬에 매핑하라. 트레이드오프(trade-off)를 기록하라. 어떤 불투명성이 사라지는가?

5. A2A v1.0 발표를 읽고 2026년 4월 기준 어떤 프레임워크에서도 아직 구현되지 않은 기능 하나를 식별하라. (힌트: 멀티홉(multi-hop) 태스크 위임과 관련 있다.)

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| A2A | "에이전트 간 프로토콜" | 불투명한 에이전트 협업을 위한 오픈 프로토콜 |
| 에이전트 카드(Agent Card) | "`.well-known/agent.json`" | 에이전트의 스킬과 엔드포인트를 기술하는 게시된 메타데이터 |
| 스킬(Skill) | "호출 가능한 단위" | 에이전트가 지원하는 명명된 연산(MCP 툴에 대응) |
| 태스크(Task) | "위임 단위" | 수명 주기와 최종 아티팩트를 가진 작업 항목 |
| 메시지(Message) | "태스크 입력" | 파트(text, file, data)를 운반한다 |
| 파트(Part) | "타입이 지정된 청크" | 메시지의 `text` / `file` / `data` 요소 |
| 아티팩트(Artifact) | "태스크 출력" | 완료 시 반환되는 명명되고 타입이 지정된 출력 |
| AP2 | "Agent Payments Protocol" | 신뢰와 결제를 위한 서명된 에이전트 카드 확장 |
| 불투명성(Opacity) | "블랙박스 협업" | 호출되는 에이전트의 내부가 호출자로부터 숨겨진다 |
| Input-required | "태스크 일시 정지" | 에이전트가 더 많은 정보를 필요로 할 때의 수명 주기 상태 |

## 더 읽을거리 (Further Reading)

- [a2a-protocol.org](https://a2a-protocol.org/latest/): 표준 A2A 명세
- [a2aproject/A2A: GitHub](https://github.com/a2aproject/A2A) 레퍼런스 구현과 SDK
- [Linux Foundation: A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) 2025년 6월 거버넌스 이전
- [Google Cloud: A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) 로드맵과 파트너 모멘텀
- [Google Dev: A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 릴리스 노트와 하위 호환성 지침
