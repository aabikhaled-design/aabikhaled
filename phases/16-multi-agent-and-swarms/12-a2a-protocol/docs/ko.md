# A2A(에이전트 간 프로토콜 (A2A) The Agent-to-Agent Protocol)

> 구글은 2025년 4월 A2A를 발표했다. 2026년 4월 기준 스펙은 https://a2a-protocol.org/latest/specification/ 에 있고 150개 이상의 조직이 이를 지지한다. A2A는 MCP(Lesson 13)에 대한 수평적 보완재다. MCP가 수직적(에이전트 ↔ 도구)인 반면, A2A는 피어 투 피어(peer-to-peer, 에이전트 ↔ 에이전트)다. A2A는 에이전트 카드(Agent Card, 발견), 아티팩트(artifact, 텍스트·구조화 데이터·비디오)를 동반한 작업, 불투명 작업 수명주기(opaque task lifecycle), 인증을 정의한다. 프로덕션(production) 시스템은 점점 더 MCP와 A2A를 짝지어 쓴다. 구글 클라우드는 2025~2026년에 걸쳐 A2A 지원을 Vertex AI Agent Builder에 통합했다.

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75분

## 문제 (Problem)

에이전트(agent)가 다른 시스템에 있는 또 다른 에이전트를 호출해야 한다. 어떻게? HTTP 엔드포인트를 노출하고, 맞춤 JSON 스키마를 정의하고, 상대편이 그것을 알아듣기를 바라는 방법이 있다. 그러면 모든 에이전트 쌍이 커스텀 통합이 된다.

A2A는 그 호출을 위한 범용 와이어 프로토콜(wire protocol)이다. 표준 발견, 표준 작업 모델, 표준 전송, 표준 아티팩트. HTTP+REST와 비슷하되 에이전트를 일급 시민(first-class citizen)으로 취급한다.

## 개념 (Concept)

### 네 가지 요소

**에이전트 카드(Agent Card).** `/.well-known/agent.json`에 있는 JSON 문서로 에이전트를 기술한다. 이름, 스킬, 엔드포인트, 지원 모달리티(modality), 인증 요구사항. 발견은 카드를 읽음으로써 이루어진다.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**작업(Task).** 작업의 단위다. 수명주기를 가진 비동기·상태 보존 객체다: `submitted → working → completed / failed / canceled`. 클라이언트가 작업을 보내고, 업데이트를 폴링(poll)하거나 구독한다.

**아티팩트(Artifact).** 작업이 생성하는 결과 타입이다. 텍스트, 구조화 JSON, 이미지, 비디오, 오디오. 아티팩트는 타입이 지정되어 있어 서로 다른 모달리티가 일급으로 취급된다.

**불투명 수명주기(Opaque lifecycle).** A2A는 원격 에이전트가 작업을 *어떻게* 푸는지 규정하지 않는다. 클라이언트는 상태 전이와 아티팩트를 본다. 구현은 어떤 프레임워크든 자유롭게 쓸 수 있다.

### MCP/A2A 분리

- **MCP**(Lesson 13): 에이전트 ↔ 도구. 에이전트는 JSON-RPC를 통해 도구 서버에 읽고 쓴다. 기본적으로 무상태(stateless)다.
- **A2A**: 에이전트 ↔ 에이전트. 피어 프로토콜. 양쪽 모두 자체 추론을 가진 에이전트다.

프로덕션 멀티 에이전트 시스템은 둘 다 쓴다. A2A 피어가 자기 쪽에서 MCP 도구를 호출한다. 이 분리는 두 관심사를 깔끔하게 유지한다.

### 발견 흐름

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

또는 스트리밍으로: 푸시 업데이트를 위한 `/tasks/{id}/events`에 대한 SSE 구독.

### 인증 (Auth)

A2A는 흔한 세 가지 패턴을 지원한다.

- **베어러 토큰(Bearer token)**: OAuth2 또는 불투명 토큰.
- **mTLS**: 상호 TLS. 조직들이 서로에게 신원을 증명한다.
- **서명된 요청(Signed requests)**: 페이로드에 대한 HMAC.

인증은 에이전트 카드에 선언된다. 클라이언트는 이를 발견하고 준수한다.

### 2026년 4월까지 150개 이상의 조직

엔터프라이즈 도입이 A2A의 규모를 견인했다. 그 결과 A2A는 엔터프라이즈 에이전트 시스템이 신뢰 경계(trust boundary)를 넘나드는 방식이 되었다. 구글 클라우드는 Vertex AI Agent Builder A2A 지원을 출시했고, Microsoft Agent Framework도 이를 지원하며, 대부분의 주요 프레임워크(LangGraph, CrewAI, AutoGen)가 A2A 어댑터를 제공한다.

### A2A가 이기는 곳

- **조직 간 호출.** A사의 에이전트가 B사의 에이전트를 호출한다. A2A가 없으면 모든 쌍이 맞춤 계약이다.
- **이질적 프레임워크.** LangGraph 에이전트가 CrewAI 에이전트를 호출하고, 그것이 다시 커스텀 파이썬 에이전트를 호출한다. A2A가 이를 정규화한다.
- **타입이 지정된 아티팩트.** 비디오 결과, 구조화 JSON, 오디오, 모두 일급이다.
- **장기 실행 작업.** 불투명 수명주기 + 폴링이 몇 시간짜리 작업을 간단하게 만든다.

### A2A가 어려워하는 곳

- **지연 시간(latency)에 민감한 마이크로 호출.** A2A의 수명주기는 비동기다. 밀리초 미만의 에이전트 간 호출에는 맞지 않는다. 직접 RPC를 쓰라.
- **긴밀하게 결합된 인프로세스(in-process) 에이전트.** 두 에이전트가 같은 파이썬 프로세스에서 실행되면 A2A의 HTTP 왕복은 과하다.
- **소규모 팀.** 스펙 오버헤드는 실재한다. 내부 전용 에이전트는 그 격식이 필요 없을 수 있다.

### A2A vs ACP, ANP, NLIP

2024~2026년에 여러 관련 스펙이 등장했다.

- **ACP**(IBM/Linux Foundation): A2A의 전신, 범위가 더 좁다.
- **ANP**(Agent Network Protocol): 피어 발견 중심, 탈중앙화 우선.
- **NLIP**(Ecma Natural Language Interaction Protocol, 2025년 12월 표준화): 자연어 콘텐츠 타입.

A2A는 2026년 4월 기준 가장 많이 채택된 피어 프로토콜이다. 비교는 arXiv:2505.02279(Liu et al., "A Survey of Agent Interoperability Protocols")를 보라.

```figure
sw-agent-card-discovery
```

## 직접 만들기 (Build It)

`code/main.py`는 `http.server`와 JSON을 사용해 A2A 최소 서버와 클라이언트를 구현한다. 서버는:

- `/.well-known/agent.json`을 노출하고,
- `POST /tasks`를 받고,
- 작업 상태를 관리하고,
- `GET /tasks/{id}`에서 아티팩트를 반환한다.

클라이언트는:

- 에이전트 카드를 가져오고,
- 작업을 제출하고,
- 완료될 때까지 폴링하고,
- 아티팩트를 읽는다.

실행:

```
python3 code/main.py
```

이 스크립트는 백그라운드 스레드에서 서버를 시작한 뒤 그 서버를 대상으로 클라이언트를 실행한다. 발견, 제출, 폴링, 아티팩트의 전체 흐름을 볼 수 있다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-a2a-integrator.md`는 A2A 통합을 설계한다. 에이전트 카드 내용, 작업 스키마, 인증 선택, 스트리밍 vs 폴링.

## 산출물 (Ship It)

체크리스트:

- **스펙 버전 고정.** A2A는 아직 진화 중이다. 에이전트 카드가 프로토콜 버전을 선언해야 한다.
- **멱등(idempotent) 작업 생성.** 중복 제출(네트워크 재시도)은 하나의 작업만 생성해야 한다.
- **아티팩트 스키마.** 에이전트가 반환하는 형태를 선언하라. 소비자는 검증해야 한다.
- **레이트 리밋 + 인증.** A2A는 외부에 노출된다. 표준 웹 보안을 적용하라.
- **실패 작업을 위한 데드 레터(dead-letter).** 반복되는 실패 유형을 위해 시간에 따른 패턴을 검사하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 클라이언트가 서버를 발견하고 올바른 아티팩트를 받는지 확인하라.
2. 서버에 두 번째 스킬(예: "summarize")을 추가하라. 에이전트 카드를 갱신하라. 작업 유형에 따라 스킬을 고르는 클라이언트를 작성하라.
3. SSE 스트리밍 엔드포인트를 구현하라. 상태 변화를 내보내는 `/tasks/{id}/events`다. 클라이언트는 무엇을 다르게 해야 하는가?
4. A2A 스펙(https://a2a-protocol.org/latest/specification/)을 읽어라. 이 데모가 구현하지 않은, 스펙이 의무화하는 세 가지를 식별하라.
5. A2A(에이전트 카드 발견)를 MCP(`listTools`를 통한 서버 측 능력 나열)와 비교하라. 자기 기술(self-describing) 에이전트와 능력 탐색(capability-probing) 사이의 트레이드오프(trade-off)는 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| A2A | "에이전트 간" | 에이전트가 시스템을 가로질러 다른 에이전트를 호출하기 위한 피어 프로토콜. 구글 2025. |
| 에이전트 카드(Agent Card) | "에이전트의 명함" | 스킬, 엔드포인트, 인증을 기술하는 `/.well-known/agent.json`의 JSON. |
| 작업(Task) | "작업의 단위" | 수명주기를 가진 비동기·상태 보존 객체. 완료 시 아티팩트가 생성된다. |
| 아티팩트(Artifact) | "결과물" | 타입이 지정된 출력: 텍스트, 구조화 JSON, 이미지, 비디오, 오디오. 일급 미디어. |
| 불투명 수명주기(Opaque lifecycle) | "어떻게 푸는지는 에이전트의 몫" | 클라이언트는 상태 전이를 본다. 서버는 프레임워크/도구를 자유롭게 고른다. |
| 발견(Discovery) | "에이전트 찾기" | `GET /.well-known/agent.json`이 카드를 반환한다. |
| MCP vs A2A | "도구 vs 피어" | MCP: 수직적 에이전트 ↔ 도구. A2A: 수평적 에이전트 ↔ 에이전트. |
| ACP / ANP / NLIP | "형제 프로토콜" | 인접 스펙. A2A가 2026년 기준 가장 많이 채택됨. |

## 더 읽을거리 (Further Reading)

- [A2A specification](https://a2a-protocol.org/latest/specification/): 정전(canonical) 스펙
- [Google Developers Blog: A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) 2025년 4월 출시 게시물
- [A2A GitHub repo](https://github.com/a2aproject/A2A): 레퍼런스 구현과 SDK
- [Liu et al.: A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) MCP, ACP, A2A, ANP 비교
