# FIPA-ACL과 발화 행위의 유산 (Heritage of FIPA-ACL and Speech Acts)

> MCP 이전, A2A 이전에 FIPA-ACL이 있었다. 2000년 IEEE 지능형 물리 에이전트 재단(Foundation for Intelligent Physical Agents)은 스무 개의 수행문(performative), 두 개의 콘텐츠 언어, 그리고 일련의 상호작용 프로토콜(contract net, subscribe/notify, request-when)을 갖춘 에이전트 통신 언어를 비준했다. 온톨로지(ontology) 오버헤드가 웹에 비해 너무 무거웠기에 산업계에서 사라졌지만, 멀티 에이전트(multi-agent) 시스템의 LLM 부흥은 형식 의미론(formal semantics) 없이 똑같은 아이디어를 조용히 재구현하고 있다: JSON 계약이 수행문을 대신하고, 자연어가 온톨로지를 대신한다. 이 레슨에서 FIPA-ACL을 진지하게 읽고 나면, 어떤 2026년 프로토콜 결정이 재발명이고, 어떤 것이 진짜 새로움이며, 현재의 물결이 2000년대가 이미 해결한 문제들을 어디서 다시 발견하게 될지가 보인다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60분

## 문제 (Problem)

2026년 에이전트 프로토콜 지형은 분주하다: 도구를 위한 MCP, 에이전트를 위한 A2A, 엔터프라이즈 감사를 위한 ACP, 탈중앙 신뢰를 위한 ANP, 자연어 콘텐츠를 위한 NLIP, 거기에 CA-MCP와 스물네 개의 연구 제안들. 각 명세는 스스로를 근본적인 것이라 선언한다.

솔직한 해석은, 그 대부분이 매우 구체적인 20년 된 결정 트리를 재발견하고 있다는 쪽이다. 오스틴(Austin, 1962)과 설(Searle, 1969)의 발화 행위 이론(speech-act theory)은 우리에게 "발화는 행위다"를 주었다. KQML(1993)은 이를 와이어 프로토콜(wire protocol)로 바꿨다. FIPA-ACL(2000년 비준)은 참조 표준화를 만들어냈다: 스무 개의 수행문, 콘텐츠 언어 SL0/SL1, contract-net과 subscribe-notify를 위한 상호작용 프로토콜. JADE와 JACK은 자바 참조 플랫폼이었다. 온톨로지 오버헤드가 너무 무거웠고 웹이 이기고 있었기에, 이 노력은 2010년경 사라졌다.

MCP의 `tools/call`, A2A의 작업 생명주기, 혹은 CA-MCP의 공유 컨텍스트 저장소를 볼 때 우리가 보는 것은 FIPA 결정들의 더 부드럽고 JSON 네이티브한 재탕이다. 이 유산을 알면 두 가지가 보인다. 어떤 새로운 "혁신"이 사실은 재발명인지, 그리고 새 명세들이 어떤 오래된 실패 양식을 다시 발견하게 될지다.

## 개념 (Concept)

### 발화 행위, 한 문단으로

오스틴은 어떤 문장들은 세상을 기술(describe)하지 않고 변화시킨다는 점에 주목했다. "약속합니다." "요청합니다." "선언합니다." 그는 이것들을 수행적 발화(performative utterance)라 불렀다. 설은 다섯 범주로 형식화했다: 단언(assertive), 지시(directive), 언약(commissive), 표현(expressive), 선언(declarative). KQML(Finin et al., 1993)은 이를 소프트웨어 에이전트를 위해 실용화했다: 메시지는 수행문(행위)에 콘텐츠(행위가 무엇에 관한 것인지)를 더한 것이다. FIPA-ACL은 KQML의 빈틈을 정리하고 약 스무 개의 수행문을 중심으로 표준화했다.

### 스무 개의 FIPA 수행문 (일부 목록)

| 수행문 | 의도 |
|---|---|
| `inform` | "나는 너에게 P가 참이라고 말한다" |
| `request` | "나는 너에게 X를 하라고 요청한다" |
| `query-if` | "P가 참인가?" |
| `query-ref` | "X의 값은 무엇인가?" |
| `propose` | "나는 우리가 X를 하자고 제안한다" |
| `accept-proposal` | "나는 그 제안을 수락한다" |
| `reject-proposal` | "나는 그 제안을 거절한다" |
| `agree` | "나는 X를 하기로 동의한다" |
| `refuse` | "나는 X를 하기를 거부한다" |
| `confirm` | "나는 P가 참임을 확인한다" |
| `disconfirm` | "나는 P를 부정한다" |
| `not-understood` | "너의 메시지가 파싱되지 않았다" |
| `cfp` | "X에 대한 제안 요청" |
| `subscribe` | "X가 변경되면 나에게 알려라" |
| `cancel` | "진행 중인 X를 취소하라" |
| `failure` | "나는 X를 시도했고 실패했다" |

전체 목록은 `fipa00037.pdf`(FIPA ACL Message Structure)에 있다. 요점은 이를 외우는 데 있지 않다. 이들 각각이 결국 LLM 프로토콜이 다시 추가하는 프리미티브(primitive)에 대응한다는 것이 요점이다.

### 정전(canonical) FIPA-ACL 메시지

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

일곱 개의 필드가 프로토콜 봉투(envelope)를 운반하고, 한 필드(`content`)가 페이로드(payload)를 운반한다. 나머지 필드들은 JSON 프로토콜에 재시도, 스레딩, 온톨로지를 덧붙일 때마다 다시 발명하게 되는 바로 그것이다.

### 두 개의 레거시 플랫폼

**JADE**(Java Agent DEvelopment framework, 1999–2020년대)는 가장 많이 쓰인 FIPA 호환 런타임이었다. 에이전트는 베이스 클래스를 확장하고, ACL 메시지를 주고받고, 컨테이너 안에서 실행되고, "behavior"를 사용해 조율했다. 상호작용 프로토콜 라이브러리에는 contract-net, subscribe-notify, request-when, propose-accept가 함께 제공됐다.

**JACK**(Agent Oriented Software, 상용)은 FIPA 메시지 위에 BDI(Belief-Desire-Intention) 추론을 강조했다. 더 형식적이지만 덜 채택됐다.

웹 스택이 멀티 에이전트 활용 사례를 집어삼키면서 둘 다 쇠퇴했다. MCP와 A2A는 2026년의 런타임 "컨테이너"다.

### FIPA가 사라진 이유

- **온톨로지 오버헤드.** FIPA는 `content`를 파싱하기 위해 공유 온톨로지를 요구했다. 온톨로지에 합의하는 것은 수년이 걸리는 표준화 과정이다. 웹은 그냥 HTTP + JSON을 썼다.
- **아무도 쓰지 않은 형식 의미론.** SL(Semantic Language)은 엄밀한 진리 조건을 주었지만, 대부분의 프로덕션 시스템은 자유 형식 콘텐츠를 쓰고 그 형식론을 무시했다.
- **툴링 종속(lock-in).** JADE는 자바 전용이었고, JACK은 상용이었다. 다중 언어(polyglot) 팀들은 둘 다 우회했다.
- **인터넷이 스택을 이겼다.** REST, 그다음 JSON-RPC, 그다음 gRPC가 ACL의 전송을 대체했다.

### LLM 부흥은 FIPA-라이트다

FIPA `request`를 MCP `tools/call`과 비교해 보라:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

같은 봉투, 다른 문법이다. 둘 다 누가, 누구에게, 의도, 페이로드, 상관 id(correlation id)를 운반한다. 어느 쪽도 다른 쪽에 대한 혁명이 아니라, 같은 설계를 두고 내린 다른 트레이드오프(trade-off)일 뿐이다.

Liu et al.의 2025년 서베이("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279)는 이 계보를 명시적으로 보여준다: MCP는 도구 사용 발화 행위에, A2A는 에이전트-피어 발화 행위에, ACP는 감사 추적(audit-trail) 발화 행위에, ANP는 탈중앙 신원 확장에 대응한다. 새 명세들은 JSON 문법과 더 느슨한 의미론을 가진 ACL의 후손이다.

### 트레이드오프, 단도직입적으로

**FIPA가 주었으나 현대 명세들이 버린 것:**

- 형식 의미론: `inform`이 발신자가 콘텐츠를 믿는다는 것을 함의함을 증명할 수 있다.
- 수행문의 정전 카탈로그: "`cancel`을 둬야 하는가?"를 다시 논쟁할 필요가 없다.
- 수십 년의 상호작용 프로토콜 패턴(contract-net, subscribe-notify, propose-accept) 알려진 정확성 속성과 함께.

**현대 명세들이 주지만 FIPA가 주지 않은 것:**

- 모든 현대 도구와 호환되는 JSON 네이티브 페이로드.
- 손으로 짠 온톨로지 없이 LLM이 해석할 수 있는 자연어 콘텐츠.
- 웹 스택 전송(HTTP, SSE, WebSocket).
- 자기 기술(self-describing) 문서를 통한 역량 발견(capability discovery)(MCP `listTools`, A2A Agent Card).

구현을 더 쉽게 하는 대신 의도 의미론을 더 느슨하게 푼 것, 이것이 바로 그 거래다.

### 이식할 가치가 있는 상호작용 프로토콜

FIPA는 약 15개의 상호작용 프로토콜을 제공했다. 그중 세 개는 LLM 멀티 에이전트 시스템으로 가져갈 가치가 있다.

1. **계약망 프로토콜(Contract Net Protocol, CNP).** 매니저가 `cfp`(제안 요청)를 발행하고, 입찰자들이 `propose`로 응답하고, 매니저가 수락/거절한다. 이것이 정전적 작업 시장(task-market) 패턴이다(Phase 16 · 16 Negotiation).
2. **구독/알림(Subscribe/Notify).** 구독자가 `subscribe`를 보내고, 발행자가 토픽이 변경될 때마다 `inform`을 보낸다. 이것이 2026년의 모든 이벤트 버스다.
3. **Request-When.** "조건 Y가 성립할 때 X를 하라." 사전 조건을 갖춘 지연 행동. 2026년의 유사물은 지속성 워크플로 엔진(durable workflow engine)의 지연 작업이다(Phase 16 · 22 Production Scaling).

각각은 현대의 메시지 큐, HTTP + 폴링, 혹은 SSE 스트리밍에 깔끔하게 매핑된다.

### 온톨로지를 버리면 무엇이 깨지는가

공유 온톨로지가 없으면, 에이전트는 자연어 콘텐츠로부터 의미를 추론한다. 문서화된 2026년 실패 양식은 **의미 표류(semantic drift)**다: 두 에이전트가 같은 단어(`"customer"`)를 미묘하게 다른 개념으로 사용하고, 수신 에이전트가 잘못된 해석에 따라 행동하며, 어떤 스키마 검증기도 이를 잡지 못한다. FIPA의 온톨로지 요구사항은 파싱 시점에 그 메시지를 거부했을 것이다.

완전한 온톨로지로 가지 않는 완화책:

- `content`에 대한 JSON Schema: 와이어에서 구조적 오류를 거부한다.
- 타입이 지정된 아티팩트(A2A): 잘못된 양식(modality)을 거부한다.
- 봉투 안의 명시적 수행문: 콘텐츠가 자연어일 때조차 의도를 명확하게 만든다.

### 2026년 명세들, 발화 행위 유산에 매핑하기

| 현대 명세 | FIPA 유사물 | 유지하는 것 | 버리는 것 |
|---|---|---|---|
| MCP `tools/call` | `request` | 명시적 의도, 상관 id | 형식 의미론, 온톨로지 |
| MCP `resources/read` | `query-ref` | 명시적 의도, 상관 id | 형식 의미론 |
| A2A Task 생명주기 | contract-net + request-when | 비동기 생명주기, 상태 전이 | 형식적 완전성 보장 |
| A2A 스트리밍 이벤트 | subscribe/notify | 비동기 푸시 | 타입 술어(typed-predicate) 구독 |
| CA-MCP 공유 컨텍스트 | 블랙보드(blackboard)(Hayes-Roth 1985) | 다중 작성자 공유 메모리 | 논리적 일관성 모델 |
| NLIP | 자연어 콘텐츠 | LLM 네이티브 | 스키마 |

표를 위에서 아래로 읽으면 패턴은 이렇다: 구조적 프리미티브는 유지하고, 형식론은 버리고, LLM이 그 모호함을 덮게 한다.

```figure
sw-contract-net
```

## 직접 만들기 (Build It)

`code/main.py`는 순수 stdlib FIPA-ACL 번역기를 구현한다. 정전적 ACL 봉투를 인코딩·디코딩하고, 모든 MCP / A2A 메시지 형태가 어떻게 동일한 일곱 개 필드로 환원되는지 보여준다. 데모는:

- 다섯 개의 MCP 스타일 및 A2A 스타일 메시지를 FIPA-ACL로 인코딩한다.
- FIPA-ACL을 현대 등가물로 다시 디코딩한다.
- `cfp`, `propose`, `accept-proposal`, `reject-proposal`을 사용해 한 매니저와 세 입찰자 사이의 장난감 계약망 협상을 실행한다.

실행:

```
python3 code/main.py
```

출력은 각 현대 메시지를 2026년 JSON 형태와 FIPA-ACL 형태 양쪽으로 보여주는 나란한 트레이스(trace)이며, 그다음 계약망 입찰의 왕복(round-trip)이다. 동일한 프로토콜 프리미티브가 왕복에서 살아남는다; 오직 문법만 다르다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-fipa-mapper.md`는 임의의 에이전트 프로토콜 명세를 읽어 FIPA-ACL 매핑을 만들어내는 스킬이다. 새 프로토콜을 채택하기 전에 이것으로 "이것이 진정 새로운가, 아니면 JSON 문법을 입은 `inform`인가?"에 답하라.

## 산출물 (Ship It)

FIPA-ACL을 다시 가져오지 마라. 그 체크리스트를 가져와라:

- 각 메시지의 의도 프리미티브(수행문)는 무엇인가?
- 요청-응답과 취소를 위한 상관 id가 있는가?
- 명시적 콘텐츠 언어(JSON-RPC, 평문, 구조화된 타입 아티팩트)가 있는가?
- 상호작용 프로토콜이 일급(first-class)인가, 아니면 contract-net을 밑바닥부터 재구현하고 있는가?
- 두 에이전트가 콘텐츠 의미에 대해 불일치할 때(의미 표류) 무슨 일이 일어나는가?

이 다섯 질문을, 어떤 새 프로토콜이든 프로덕션에 출하하기 전에 문서화하라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 왕복 인코딩을 관찰하라. `tools/call`, `resources/read`, A2A 작업 생성에 어떤 FIPA 수행문이 대응하는지 식별하라.
2. 매니저가 입찰 도중에 작업을 철회할 수 있게 하는 `cancel` 수행문으로 계약망 데모를 확장하라. `cancel`은 재시도만으로는 해결되지 않는 어떤 실패 사례를 해결하는가?
3. FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) 섹션 4.1–4.3을 읽어라. 이 레슨에서 다루지 않은 수행문 하나를 골라 그 현대 JSON-RPC 유사물을 기술하라.
4. Liu et al., arXiv:2505.02279을 읽어라. MCP, A2A, ACP, ANP 각각에 대해, 그들이 유지하고 버리는 FIPA 수행문 계열을 나열하라.
5. 직접 만든 시스템에서 `request` 수행문의 `content` 필드에 쓸 최소 JSON-Schema를 설계하라. 그 스키마가 순수 자연어가 주지 않는 무엇을 주며, 무엇을 치르게 하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 발화 행위 (Speech act) | "무언가를 하는 발화" | 오스틴/설: 행위로서의 발화. ACL의 이론적 모태. |
| FIPA | "그 오래된 XML 것" | IEEE 지능형 물리 에이전트 재단. 2000년에 ACL을 표준화했다. |
| ACL | "에이전트 통신 언어(Agent Communication Language)" | FIPA의 봉투 형식: 수행문 + 콘텐츠 + 메타데이터. |
| 수행문 (Performative) | "동사" | 메시지의 의도 클래스: `inform`, `request`, `propose`, `cfp` 등. |
| KQML | "FIPA의 전신" | Knowledge Query and Manipulation Language(1993). 더 단순하고 더 좁다. |
| 온톨로지 (Ontology) | "공유 어휘" | 콘텐츠 언어가 말하는 개념들에 대한 형식적 정의. |
| SL0 / SL1 | "FIPA 콘텐츠 언어" | Semantic Language 레벨 0과 1: 형식적 콘텐츠 언어 계열. |
| 계약망 (Contract Net) | "작업 시장" | 매니저가 cfp를 발행하고, 입찰자가 제안하고, 매니저가 수락한다. 정전적 상호작용 프로토콜. |
| 상호작용 프로토콜 (Interaction protocol) | "메시지의 패턴" | 알려진 정확성을 갖는 수행문의 시퀀스: request-when, subscribe-notify 등. |

## 더 읽을거리 (Further Reading)

- [Liu et al.: A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) 현대 명세들을 FIPA 유산에 연결하는 정전적 2025년 서베이
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/): 2000년에 비준된 봉투 형식
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/): 전체 수행문 카탈로그
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25): `request`/`query-ref`의 현대 도구 사용 등가물
- [A2A specification](https://a2a-protocol.org/latest/specification/): contract-net과 subscribe-notify의 현대 에이전트-피어 등가물
