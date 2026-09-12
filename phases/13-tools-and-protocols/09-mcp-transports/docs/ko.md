# MCP 전송: stdio와 무상태 Streamable HTTP (MCP Transports: stdio and Stateless Streamable HTTP)

> 전송은 MCP 메시지를 실어 나른다. 빠져 있는 프로토콜 상태를 대신 채워 주지는 않는다. `2026-07-28`에서는 로컬 stdio와 원격 Streamable HTTP 둘 다 자기 자신을 설명하는 요청을 실어 나른다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 minutes

## 학습 목표 (Learning Objectives)

- 로컬 자식 프로세스에는 stdio를, 네트워크 서비스에는 Streamable HTTP를 고른다.
- 단일 엔드포인트에 POST만 받는 현대 Streamable HTTP 계약을 구현한다.
- MCP 버전, 메서드, 이름 헤더를 JSON-RPC 본문과 대조해 복제하고 검증한다.
- 요청 범위 SSE와 오래 유지되는 `subscriptions/listen` 스트림을 각각 올바르게 전달한다.
- 세션 기반 배포와 레거시 HTTP+SSE 배포를, 레거시 동작을 현대 방식인 것처럼 내세우지 않으면서 이전한다.

## 문제 (The Problem)

이전 Streamable HTTP 개정판은 프로토콜 협상을 연결 동작과 세션 동작에 섞어 두었다. 서버는 `Mcp-Session-Id`를 발급하고, 독립 GET 스트림을 열어 두고, 세션 종료용 DELETE를 받고, `Last-Event-ID`로 SSE를 이어받을 수 있었다.

MCP `2026-07-28`은 그 장치들을 현대 전선에서 걷어냈다. 프로토콜 버전과 클라이언트 역량이 요청 본문에 실려 다니므로 모든 요청은 건강한 워커 아무 곳에나 떨어져도 된다. HTTP 헤더는 라우팅과 정책을 위해 선택된 필드를 복제하지만, 서버는 실행 전에 그 헤더를 본문과 대조해 검증한다.

그 결과 규모를 키우기도 쉬워지고 동작을 따져 보기도 쉬워진다. 동시에, 2025년 전송 방식을 현재 방식이라고 가르치는 서버는 잘못된 장애 모델과 보안 모델을 가르치고 있다는 뜻이기도 하다.

## 개념 (The Concept)

### stdio

stdio 바인딩은 클라이언트가 띄운 하위 프로세스를 위한 것이다.

- 클라이언트는 UTF-8 JSON-RPC 메시지 하나를 한 줄씩 stdin에 쓴다.
- 서버는 UTF-8 JSON-RPC 메시지 하나를 한 줄씩 stdout에 쓴다.
- 서버는 진단 정보를 stderr에 쓴다.
- 서버는 stdin이 EOF에 닿으면 곧바로 종료한다.
- 모든 현대 요청은 버전과 클라이언트 역량을 `params._meta`에 싣는다.

프로세스는 여러 호출에 걸쳐 살아 있을 수 있지만, 그것이 현대 프로토콜 세션은 아니다. 프로세스가 예기치 않게 종료되면 처리 중이던 요청은 유실된다. 프로세스를 재시작하고, 다시 탐색하고, 목록을 다시 받고, 구독을 다시 열고, 안전한 작업을 새 요청 id로 재시도하라.

### 2026-07-28의 Streamable HTTP (Streamable HTTP in 2026-07-28)

현대 서버는 `/mcp` 같은 MCP 엔드포인트를 하나만 노출하고 그곳에서 POST를 받는다.

JSON-RPC 요청이나 알림은 하나하나가 새 HTTP POST다. 본문에는 JSON-RPC 메시지 하나가 담긴다. 클라이언트는 서버에 JSON-RPC 응답을 보내지 않는다.

요청에 대해 서버는 둘 중 하나를 돌려준다.

- JSON-RPC 응답 하나를 담은 `Content-Type: application/json`, 또는
- 그 요청과 연관된 알림들에 이어 최종 JSON-RPC 응답이 오는 `Content-Type: text/event-stream`.

받아들인 알림에 대해서는 본문 없이 `202 Accepted`를 돌려준다.

클라이언트는 두 응답 형식을 모두 알린다.

```http
Accept: application/json, text/event-stream
```

### POST만 받는다는 말은 POST만 받는다는 뜻이다 (POST-only means POST-only)

현대 Streamable HTTP에는 독립 GET 스트림도, DELETE 세션 엔드포인트도 없다.

- `GET /mcp`는 `405 Method Not Allowed`를 돌려준다.
- `DELETE /mcp`는 `405 Method Not Allowed`를 돌려준다.
- `Mcp-Session-Id`는 무시되며 발급되지도 되돌려지지도 않는다.
- `Last-Event-ID`는 현대 스트림이 이어받을 수 없으므로 무시된다.

요청 범위 스트림이 최종 응답 전에 끊기면 클라이언트는 처리 중이던 그 요청을 잃은 것이다. 재시도가 안전하다면 새 JSON-RPC id로 새 요청을 낼 수 있다. 스트림을 이어받으려 해서는 안 된다.

### Origin 검증 (Origin validation)

서버는 DNS 리바인딩을 막기 위해 들어오는 연결의 `Origin`을 검증한다. 헤더가 있는데 명시적으로 허용된 값이 아니면 `403 Forbidden`을 돌려준다. 브라우저가 아닌 클라이언트는 `Origin`을 생략할 수 있고, 공식 전송 규칙도 이를 허용한다.

로컬 서버는 모든 인터페이스가 아니라 `127.0.0.1`에 바인딩해야 한다. 네트워크 서비스에는 여전히 요청마다 인증과 인가가 필요하다. Origin 검증은 인증이 아니다.

설정을 정규화한 뒤 origin을 정확히 일치 비교하라. `origin.startswith("https://trusted.example")` 같은 접두사 검사는 공격자가 조종하는 접미사를 받아들일 수 있어 안전하지 않다.

### 필수 HTTP 메타데이터 헤더 (Required HTTP metadata headers)

모든 현대 POST 요청에는 다음이 들어간다.

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

헤더 규칙은 이렇다.

- `MCP-Protocol-Version`은 필수이며 `params._meta.io.modelcontextprotocol/protocolVersion`과 같아야 한다.
- `Mcp-Method`는 필수이며 JSON-RPC `method`와 같아야 한다.
- `Mcp-Name`은 `tools/call`, `resources/read`, `prompts/get`에 필수다.
- `Mcp-Name`은 `params.name`과 같고, `resources/read`에서는 `params.uri`와 같다.
- 헤더 이름은 대소문자를 가리지 않지만 헤더 값은 대소문자를 가린다.

안전하지 않거나 ASCII가 아닌 `Mcp-Name` 값은 정확히 이 UTF-8 Base64 표식을 쓴다.

```text
=?base64?{Base64EncodedValue}?=
```

서버는 그 값을 디코딩한 뒤에 본문과 비교한다.

복제 헤더가 빠졌거나, 형식이 어긋났거나, 본문과 맞지 않으면 JSON-RPC 코드 `-32020`과 함께 HTTP `400`을 돌려준다. 헤더와 본문이 서로 같은 버전을 가리키는데 서버가 그 버전을 지원하지 않으면, `-32022`와 함께 HTTP `400`을 돌려주고 `{"supported":["2026-07-28"],"requested":"2027-01-01"}` 같은 정확한 오류 데이터를 싣는다.

알 수 없는 현대 메서드는 JSON-RPC `-32601`과 함께 HTTP `404`를 돌려준다. 두 시대를 다루는 클라이언트는 이 JSON-RPC 본문으로 현대 오류와 레거시 엔드포인트 부재를 구별하므로 본문이 중요하다.

### 요청 범위 SSE (Request-scoped SSE)

서버는 오래 걸리는 요청 하나에 대해 SSE를 고를 수 있다.

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

서버는 이 스트림에 독립적인 JSON-RPC 요청을 보내서는 안 된다. 샘플링, 유도, 루트 상호작용은 Multi Round-Trip Request 결과를 쓴다. 응답 스트림을 닫으면 그 요청은 취소된다.

재생을 위해 SSE 이벤트 id를 붙이지 마라. `Last-Event-ID` 이어받기는 현대 개정판에 없다.

### 오래 유지되는 변경 알림은 subscriptions/listen을 쓴다 (Long-lived changes use subscriptions/listen)

변경 알림은 독립 GET이 아니라 클라이언트가 여는 요청을 쓴다.

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

이 POST의 응답은 오래 유지되는 SSE 스트림이다. 첫 프로토콜 메시지는 `notifications/subscriptions/acknowledged`다. 확인 응답과 모든 변경 알림, 그리고 최종 결과는 `_meta`에 listen 요청 id와 같은 `io.modelcontextprotocol/subscriptionId`를 싣는다. 서버는 연결 유지를 위해 SSE 주석을 내보낼 수 있다. 스트림이 끊기면 클라이언트는 새 요청 id로 `subscriptions/listen`을 다시 내고 영향을 받은 데이터를 다시 가져온다.

`resources/subscribe`와 `resources/unsubscribe`는 레거시 시대에 속한다. 현대 연결에서는 쓰지 마라.

### 명시적 애플리케이션 상태 (Explicit application state)

프로토콜 세션을 없앴다고 해서 상태를 가진 작업 흐름이 금지되는 것은 아니다. 서버는 불투명한 상태 핸들을 발급해 평범한 도구 결과로 돌려줄 수 있다. 클라이언트는 이후 호출에서 그 핸들을 명시적인 인자로 넘긴다.

핸들을 인증된 주체에 묶고, 추측할 수 없게 만들고, 만료시키고, 쓸 때마다 인가하라. 이렇게 하면 상태가 전송 계층의 고정 배정 뒤에 숨지 않고 애플리케이션 계층에서 눈에 보인다.

복제본에 숨은 상태가 일으키는 장애는 기계적으로 이렇게 진행된다.

1. 요청 A가 복제본 1에 닿아 그 프로세스 메모리 안에 초안을 만든다.
2. 연결이 초안을 식별한다고 구현이 가정했기 때문에 응답은 초안 핸들을 돌려주지 않는다.
3. 요청 B는 새로운 POST라서 복제본 2에 닿는다.
4. 복제본 2에는 유효한 프로토콜 메타데이터가 있지만 그 초안을 지칭하거나 불러올 방법이 없어서, 작업 흐름이 실패하거나 엉뚱한 지역 객체를 읽는다.
5. 고정 라우팅이 증상을 고친 것처럼 보이다가, 재시작이나 배포, 재스케줄, 장애 조치가 다음 요청을 옮기는 순간 무너진다.

올바른 경계는 두 부분으로 나뉜다. 프로토콜 맥락은 요청마다 실려 다닌다. 오래 남아야 하는 애플리케이션 상태는 서버가 발급해 클라이언트에게 돌려준 핸들 아래에서 공유 저장소에 산다. 다음 호출이 그 핸들을 실어 보내면 어느 복제본이든 같은 레코드를 불러오고, 인가가 그 레코드를 인증된 주체와 테넌트에 묶는다. 복제본 메모리가 레코드를 캐시하는 것은 괜찮지만, 정확성을 위해 필요한 유일한 사본이 되어서는 안 된다.

상태 장치는 수명에 맞춰 고르라. 요청 지역 변수는 호출 하나를 감당한다. 짧은 MRTR 연속 처리에는 무결성이 보호되는 `requestState`를 쓸 수 있다. 초안이나 오래 남는 작업에는 명시적 핸들에 더해 공유 영속성, 만료, 동시성 제어, 멱등성이 필요하다. 이 중 어느 것도 MCP 프로토콜 세션이 아니다.

### HTTP의 두 시대 호환성 (HTTP dual-era compatibility)

현대 서버와 레거시 서버를 모두 지원하는 클라이언트는 현대 POST를 먼저 시도한다. HTTP `400`, `404`, `405`를 받으면 본문을 살펴본다.

- 인식 가능한 현대 JSON-RPC 오류는 서버가 현대 방식임을 증명한다. 요청을 고치거나 서버가 알린 버전으로 재시도하라. 등급을 낮추지 마라.
- 빈 본문이나 알 수 없는 응답은 레거시 HTTP+SSE 서버라는 뜻일 수 있다. 그때만 옛 GET 엔드포인트를 시도하고 레거시 `endpoint` 이벤트를 기대하라.

서버는 이전 기간 동안 두 시대를 함께 지원할 수 있다. 현대 메타데이터는 POST만 받는 현대 구현으로 보내고, 옛 클라이언트를 위해 레거시 엔드포인트를 따로 남겨 두면 된다. 레거시 GET, DELETE, 세션 id, 재생 동작을 `2026-07-28`의 일부인 것처럼 설명하지 마라.

```figure
tp-transport-handshake
```

## 직접 해 보기 (Use It)

`code/main.py`는 파이썬 표준 라이브러리만으로 유한하게 도는 현대 Streamable HTTP 서버를 구현한다. Origin과 복제 헤더를 검증하고, 사라진 세션 헤더를 무시하고, 평범한 호출에는 JSON을 돌려주며, 유한한 `subscriptions/listen` SSE 스트림을 보여 준다.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

탐침은 다음을 확인한다.

- 유효하지 않은 Origin이 거부된다,
- 세션 id 없이 탐색이 성공한다,
- `Mcp-Session-Id`와 `Last-Event-ID`가 무시된다,
- 헤더 불일치가 `-32020`을 돌려준다,
- 지원하지 않는 버전이 정확한 `supported`와 `requested` 데이터와 함께 `-32022`를 돌려준다,
- id가 없는 알림을 받아들이면 본문 없이 HTTP `202`를 돌려준다,
- GET과 DELETE가 `405`를 돌려준다,
- `subscriptions/listen`은 확인 응답과 알림, 최종 결과가 모두 구독 id를 싣는 POST 응답 스트림이다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-mcp-transport-migrator.md`를 만든다. 현대 프로토콜 세션을 걷어내고, 헤더와 본문을 대조하는 검증을 넣고, 독립 GET을 `subscriptions/listen`으로 바꾸며, 레거시 다리는 눈에 보이게 분리해 둔다.

## 연습 문제 (Exercises)

1. POST에서 `Mcp-Method`를 빼라. HTTP `400`과 오류 `-32020`이 나오는지 확인하라.
2. 헤더와 본문 버전을 모두 `2027-01-01`로 맞춰 보내라. HTTP `400`과 오류 `-32022`, 그리고 정확한 데이터 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`가 나오는지 확인하라.
3. ASCII가 아닌 리소스 URI에 대해 Base64 표식 `Mcp-Name`을 보내라. 디코딩한 값이 `params.uri`와 비교되는지 확인하라.
4. 유한한 listen 스트림을 최종 응답 전에 끊어라. 새 JSON-RPC id로 다시 내고 도구를 다시 가져오라.
5. ping 도구에 명시적 작업 흐름 핸들을 추가하라. 연결 고정 배정을 쓰지 않고 인가 주체에 묶어라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| stdio | 클라이언트가 띄운 하위 프로세스 위에서 줄 단위로 주고받는 JSON-RPC |
| Streamable HTTP | 현대 메시지 하나하나가 새 POST가 되는 단일 엔드포인트 |
| Request-scoped SSE | 연관된 알림과 최종 응답을 담은 POST 응답 스트림 |
| `subscriptions/listen` | 신청한 변경 알림을 받기 위해 오래 유지하는 POST 요청 |
| Header mismatch | 복제 헤더가 본문과 어긋날 때의 HTTP `400`과 JSON-RPC `-32020` |
| Origin validation | 들어오는 연결에 대한 DNS 리바인딩 방어이며 인증이 아니다 |
| Explicit state handle | 숨은 세션 상태 대신 평범한 인자로 넘기는 애플리케이션 토큰 |
| Legacy bridge | 호환성만을 위해 남겨 둔, 분리된 이전 시대 동작 |

## 더 읽을거리 (Further Reading)

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
