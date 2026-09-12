# 무상태 프로토콜 위의 MCP Apps (MCP Apps on the Stateless Protocol)

> 대화형 결과도 여전히 MCP의 도구 교환이자 리소스 교환이다. 2026-07-28 코어는 그 교환을 자기완결적으로 만들고, Apps 확장은 모래상자에 담긴 브라우저 표면을 더한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- `server/discover`와 요청마다 실리는 확장 역량으로 MCP Apps를 알린다.
- 도구가 호출되기 전에 그 도구에 `ui://` 리소스를 선언한다.
- 2026-07-28 무상태 전선 위에서 완결된 도구 결과와 리소스 결과를 돌려준다.
- Apps의 `ui/initialize` 다리 메시지를 사라진 MCP 코어 악수와 갈라놓는다.
- origin 검증, 모래상자, CSP, 최소 권한을 적용한다.

## 문제 (The Problem)

텍스트 결과는 타임라인을 설명할 수 있다. 하지만 사용자가 걸러 보고, 들여다보고, 손댈 수 있는 타임라인을 줄 수는 없다.

MCP Apps는 선택적 확장으로 이 표현의 문제를 푼다. 도구 정의가 `ui://` 리소스를 가리킨다. 호스트는 도구가 돌기 전에 그 리소스를 가져와 검토하고, 모래상자 iframe 안에 그리고, 앱의 모든 동작을 JSON-RPC 다리로 중재할 수 있다.

2026-07-28에서 코어 프로토콜이 바뀌었다. 앱을 옛 연결 생애주기로 감싸지 마라.

- 코어의 `initialize` 요청도 `notifications/initialized` 알림도 없다.
- `Mcp-Session-Id` 헤더가 없다.
- 모든 요청이 프로토콜 버전과 클라이언트 역량을 `params._meta`에 싣는다.
- 서버는 `server/discover`를 구현하므로 클라이언트가 버전, 코어 역량, 확장을 살펴볼 수 있다.
- 성공한 결과에는 모두 `resultType` 구분자가 붙는다.
- Streamable HTTP는 요청 하나에 POST 하나를 쓴다. 현대 방식에서 GET과 DELETE 진입점은 405를 돌려준다.

Apps 다리에는 여전히 `ui/initialize`라는 메서드가 있다. 그것은 iframe postMessage 방언에 속한다. 코어 MCP 세션을 되살리지 않는다.

## 개념 (The Concept)

### 프로토콜 둘, 기능 하나 (Two protocols, one feature)

계층을 뚜렷하게 나눠 두라.

1. MCP 코어가 `server/discover`, `tools/list`, `tools/call`, `resources/list`, `resources/read`를 실어 나른다.
2. MCP Apps 확장이 UI를 선언하고 iframe과 호스트 사이의 다리를 정의한다.
3. 브라우저 모래상자 규칙이 UI가 닿을 수 있는 범위를 제한한다.

확장 식별자는 `io.modelcontextprotocol/ui`다. 양쪽 모두 참여를 선언한다. 클라이언트는 요청마다 역량 객체 안에 확장 지원을 실어 보낸다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo`는 진단을 위해 넣어 두는 편이 좋다. 스스로 밝힌 데이터일 뿐 인가를 위한 신원은 아니다.

### 그리기 전에 탐색한다 (Discover before rendering)

서버의 탐색 결과가 확장을 알린다.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

서버는 탐색을 지원해야 한다. 동작마다 자기 역량을 싣고 다니므로 클라이언트가 매 동작 전에 탐색을 부를 의무는 없다.

### UI를 도구 정의에 선언한다 (Declare the UI on the tool definition)

현대 Apps 계약은 `tools/list`에서 UI를 도구에 묶는다.

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

이것은 의도적으로 호출 전에 놓인 메타데이터다. 결과가 화면에 띄워 달라고 하기 전에 호스트가 HTML을 미리 받고, 캐시하고, 보안 검토할 수 있다. 호환성 코드가 예전의 평평한 메타데이터 키를 받아 줄 수는 있지만, 새 서버는 중첩된 `_meta.ui.resourceUri` 형태를 내보내야 한다.

현행 코어에서 `tools/list`는 캐시할 수 있다. 결정적인 순서와 `ttlMs`, `cacheScope`를 담으라. 사용자나 토큰에 따라 보이는 도구가 달라진다면 `private`를 쓰라.

### 데이터를 돌려주고, 화면 연결은 호스트에 맡긴다 (Return data, then let the host bind the view)

도구 호출은 평범한 내용과 구조화된 데이터를 함께 돌려준다.

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

호스트는 그 도구에 어떤 화면이 딸려 있는지 이미 안다. URI를 되풀이하려고 새 내용 블록을 만들어 내지 마라.

### 앱을 리소스로 제공한다 (Serve the app as a resource)

서버가 탐색에서 `resources`를 알리므로 필수인 `resources/list` 작업도 구현한다. 결정적인 목록 항목에는 정식 URI, 안정적인 이름, 설명, MIME 타입이 들어간다. 목록 결과에는 결정적인 도구 목록과 마찬가지로 `resultType`, 서버 신원 메타데이터, `ttlMs`, `cacheScope`가 들어간다.

호스트는 `resources/read`를 보낸다. Streamable HTTP에서 그 요청은 이렇다.

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

헤더 값과 JSON-RPC 본문은 서로 맞아야 한다. 어긋나면 프로토콜 오류 `-32020`이다.

결과에는 HTML 리소스와 캐시 힌트가 담긴다.

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### UI 리소스는 실행되는 내용으로 캐시한다 (Cache UI resources as executable content)

앱 리소스는 평범한 산문과 바꿔 쓸 수 있는 것이 아니다. 그 캐시 항목은 다리 코드를 실행하고, 도구 데이터를 그리고, 호스트가 중재하는 동작을 요청할 수 있다. 정식 `ui://` URI, 받아들인 서버 신원과 버전, 리소스 내용 요약값, 그리고 `cacheScope`가 private일 때는 인가 맥락까지 키에 넣으라. URI가 같더라도 HTML이나 정책 메타데이터가 다를 수 있으므로, 비공개 앱 리소스를 주체끼리 돌려쓰지 마라.

`ttlMs`가 만료되거나, 도구의 `_meta.ui.resourceUri` 결속이 바뀌거나, 서버 버전이나 받아들인 서술자 고정 값이 바뀌거나, 확인된 리소스 변경 구독이 그 URI를 지목하면 항목을 무효화하라. 다시 붙이기 전에 리소스를 다시 가져오고 CSP와 권한 검토를 다시 적용하라. 새 리소스 버전이 아직 오지 않았다는 이유만으로 낡은 iframe이 더 넓은 권한을 유지해서는 안 된다.

### 기능 정책보다 먼저 전선의 모호함을 거부한다 (Reject wire ambiguity before feature policy)

검증에는 의도된 순서가 있다. 먼저 JSON-RPC 형태를 검증하고 프로토콜 메타데이터가 문자열인지, 클라이언트 역량 맵이 객체인지 요구한다. 그다음 라우팅 헤더를 본문과 비교한다. 그러고 나서야 맞춰진 프로토콜 버전을 지원하는지 판단한다. 이 순서가 프록시와 서버가 서로 다른 요청으로 해석하는 일을 막아 준다.

| 조건 | HTTP | JSON-RPC 오류 |
|-----------|------|----------------|
| 헤더와 본문의 버전, 메서드, 이름이 어긋난다 | 400 | `-32020` |
| 헤더와 본문이 지원하지 않는 버전에서 일치한다 | 400 | `-32022`, `data`는 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read`에 Apps 확장 역량이 없다 | 400 | `-32021`, `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` 포함 |
| 메서드를 알 수 없다 | 404 | `-32601` |

JSON-RPC 알림에는 `id`가 없으므로 서버는 그것에 JSON-RPC 응답을 절대 내보내지 않는다. 받아들인 HTTP 알림은 빈 본문과 함께 202를 돌려준다. 오류가 HTTP 상태를 바꿀 수는 있지만, 그래도 알림에 JSON-RPC 오류 본문을 만들어 줄 수는 없다.

### 모래상자는 경계이지 신뢰 판정이 아니다 (The sandbox is a boundary, not a trust verdict)

iframe은 호스트가 통제한다. 앱은 호스트의 쿠키, 로컬 저장소, 페이지 DOM을 직접 읽을 수 없다. 권한이 필요한 일은 모두 다리를 건너야 한다.

이런 기본값을 쓰라.

- CSP 도메인 목록을 모두 비워 둔 다음, 앱에 필요한 origin만 더한다. fetch, XHR, WebSocket에는 `connectDomains`를, 스크립트, 스타일, 이미지, 글꼴에는 `resourceDomains`를 쓴다.
- 가능하면 코드와 데이터를 함께 묶어 넣는다.
- 눈에 보이는 기능이 필요로 하지 않는 한 카메라, 마이크, 위치 권한을 요청하지 않는다.
- `postMessage`를 정확한 상대 origin에 고정하고 다른 origin에서 온 이벤트는 전부 거부한다.
- 도구 인자, 도구 결과, 리소스 텍스트, 다리 메시지를 모두 신뢰할 수 없는 입력으로 다룬다.
- 사용자 동의는 호스트가 쥔다. iframe이 결과가 무거운 자기 동작을 스스로 승인할 수 없다.

튜토리얼에 적힌 고정된 `sandbox` 속성을 모든 호스트에 그대로 복사하지 마라. 호스트는 앱의 origin 모형과 자신의 격리 설계에 맞춰 플래그를 골라야 한다.

허용된 도메인도 여전히 유출 경로다. `connectDomains: ["https://api.example.com"]`은 앱 안에서 실행되는 어떤 스크립트든 허용된 데이터를 그곳으로 보낼 수 있다는 뜻이다. origin을 정확히 일치 비교하면 목적지를 헷갈리는 일은 막지만, 실어 보내는 내용이 적절한지까지 판정해 주지는 않는다. connect 접근은 기본을 비워 두고, 소지자 토큰을 iframe에 두지 말고, 가능하면 좁은 작업만 호스트로 중계하고, 요청과 응답 크기를 제한하고, 나가는 요청마다 어떤 사용자 동작이 원인이었는지 감사하라. `resourceDomains`를 `connectDomains`와 별개로 다루라. 글꼴이나 스크립트를 불러올 권한이 임의의 데이터 업로드 권한이 되어서는 안 된다.

### Apps 다리에는 자기만의 생애주기가 있다 (The Apps bridge has its own lifecycle)

Apps 다리는 `postMessage` 위에 얹은 JSON-RPC 방언이다. `ui/initialize`와 `ui/*` 알림을 주고받을 수 있고, `tools/call`처럼 코어를 닮은 메서드를 중계할 수도 있다.

뷰는 `appInfo`와 `appCapabilities` 객체를 실어 `ui/initialize`를 보낸다. 호스트는 자신의 역량과 호스트 맥락을 돌려준다. 그 응답을 받은 뒤에야 뷰가 `ui/notifications/initialized`를 보낸다. 호스트는 뷰에 메시지를 보내기 전에 이 Apps 알림을 기다려야 한다.

그 지역적인 악수는 iframe 하나와 호스트 프레임 하나 사이에 다리를 놓는다. MCP 프로토콜 버전을 협상하지도, 서버 상태를 만들지도, 전송 세션을 발급하지도 않는다. 접두사를 정확히 보라. 코어의 `notifications/initialized`는 사라졌지만 Apps의 `ui/notifications/initialized`는 남아 있다. 다리를 거친 도구 호출이 만들어 내는 코어 요청은 새 JSON-RPC id와 온전한 요청 메타데이터를 갖춘, 새로운 자기완결적 요청이다.

### 호스트 맥락, 동작, 권한 회수 (Host context, actions, and revocation)

다리를 초기화한 뒤에도 권한은 호스트에 남는다. 뷰는 호스트가 알린 역량을 통해서만 도구 동작, 이동, 클립보드 사용, 그 밖의 권한 있는 효과를 요청할 수 있다. 호스트는 타입이 정해진 요청, 현재 사용자, 대상, 인자를 검증하고, 승인 정책을 적용하고, 거절할 수도 있다. 버튼 클릭과 유효한 다리 메시지는 의사를 나타낼 뿐, 어느 쪽도 권한을 주지 않는다.

테마, 크기, 접근성은 한 번 주고 마는 렌더링 입력이 아니라 변하는 호스트 맥락으로 다루라.

- 호스트가 준 색과 타이포그래피 토큰을 적용하고, 테마나 대비 선호가 바뀌면 반응하라.
- 뷰가 원하는 크기를 알리게 하되, iframe 크기를 정하고 상한을 거는 것은 호스트가 맡아 내용이 배치를 벗어나거나 사람을 속이는 덧칠을 만들지 못하게 하라.
- iframe 안에서 키보드 순서, 보이는 초점, 접근 가능한 이름, 화면 낭독기 상태, 충분한 대비, 확대, 동작 줄이기 설정을 지키라.
- 크기를 바꾸거나 다시 그린 뒤에 호스트 컨트롤과 뷰 컨트롤 사이의 초점 이동을 다시 시험하라.

앱이 열려 있는 동안에도 역량은 회수될 수 있다. 사용자가 계정을 바꾸거나, 정책이 바뀌거나, 서버가 격리되거나, 호스트가 동의 범위를 좁히기 때문이다. 역량과 인가는 `ui/initialize` 때만이 아니라 동작을 수행하는 시점에 확인하라. 회수되면 대기 중인 권한 있는 호출을 거절하고, 정책에 더는 맞지 않는 네트워크 활동을 멈추고, 민감한 화면 상태를 지우고, UI 리소스 자체가 더는 받아들여지지 않는다면 다시 붙이거나 텍스트로 물러서라. 뷰는 거절을 평범한 결과로 다뤄야 하며, 호스트가 꺾일 때까지 재시도해서는 안 된다.

### 대체 동작도 계약의 일부다 (Fallback is part of the contract)

Apps를 아는 서버도 UI 확장을 알리지 않는 호스트를 계속 상대할 수 있다.

- `tools/list`에서 `_meta.ui` 없이 같은 도구를 돌려준다.
- `tools/call`에 쓸 만한 텍스트 결과를 남겨 둔다.
- UI에 대한 `resources/read`는 역량 누락 오류로 거부한다.
- 도구가 끝났는지 판단할 때 iframe이 있다고 가정하지 않는다.

```figure
t3-ui-sandbox
```

## 만들어 보기 (Build It)

`code/main.py`는 SDK 없이 프로세스 안에서 도는 작은 프로토콜 모형을 만든다. 현행 요청 봉투와 Streamable HTTP 라우팅 값을 검증하고, `server/discover`로 Apps를 알리고, 도구와 리소스를 나열하고, 도구를 실행하고, 자기완결적인 HTML 리소스를 제공한다.

이 모형은 이미 파싱된 본문과 라우팅 헤더를 받는다. 완전한 HTTP 어댑터가 아니며 `Content-Type`이나 `Accept`를 파싱하지 않는다. `Content-Type: application/json`과 `application/json`, `text/event-stream`을 모두 담은 `Accept` 값을 요구하는 완전한 Streamable HTTP 어댑터는 레슨 09를 보라.

실행은 이렇게 한다.

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

출력에서 네 가지를 살펴보라.

1. 모든 호출이 독립적이다.
2. 모든 요청에 `_meta` 역량이 있다.
3. `resources/list`가 리소스를 읽기 전에 안정적인 서술자를 돌려준다.
4. 모든 결과에 `resultType`과 서버 신원 메타데이터가 있다.
5. 코어 세션 식별자가 어디에도 나타나지 않는다.

## 직접 해 보기 (Use It)

`server/discover`로 시작하라. 서버 확장 맵에 `io.modelcontextprotocol/ui`가 나타나는지 확인하라. 그다음 `tools/list`를 두 번 부르되, 한 번은 Apps 역량을 실어서, 한 번은 빼고 불러라. 첫 응답은 리소스를 선언한다. 두 번째는 텍스트만으로도 쓸 만한 도구로 남는다.

`ui://notes/timeline.html`을 읽어 보라. HTML에서 `hostOrigin`과 `event.origin` 방어 코드를 찾아보라. 그 두 줄이 다리가 와일드카드 대상을 쓰지 않는다는, 눈으로 볼 수 있는 최소한의 증거다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-mcp-apps-spec.md`를 만든다. 프레임워크 코드를 쓰기 전에 앱 계약을 검토하는 데 쓰라. 현행 코어 봉투, 확장 협상, 대체 동작, UI 리소스, 캐시 정책, CSP, 권한, 다리 메서드, 동의 경계를 저자가 반드시 밝히게 만든다.

## 연습 문제 (Exercises)

1. 클라이언트 역량을 빈 확장 맵으로 바꿔라. `tools/list`가 도구는 남기되 UI 결속은 없애는지 확인하라.
2. 타임라인을 읽는 본문과 함께 `Mcp-Name: ui://notes/other.html`을 보내라. 오류 `-32020`이 나오는지 확인하라.
3. 리소스를 `cacheScope: private`로 바꿔라. 그것을 정당화하는 사용자별 조건을 설명하라.
4. 스크립트를 `https://static.example.com/app.js`로 옮겨라. 그 origin을 `resourceDomains`에 넣고 새로 생긴 공급망 위험을 설명하라.
5. `notes_open` 도구를 추가하고 버튼 클릭을 호스트를 거쳐 라우팅하라. 사용자 승인은 호스트에 남겨 두라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| MCP Apps | MCP 호스트가 그리는 대화형 HTML을 위한 선택적 확장 |
| `io.modelcontextprotocol/ui` | 양쪽 상대가 함께 알리는 확장 식별자 |
| `ui://` | 앱의 UI 템플릿을 가리키는 리소스 스킴 |
| `text/html;profile=mcp-app` | MCP 앱 HTML의 MIME 타입 |
| `server/discover` | 프로토콜과 역량을 탐색하는 현행 RPC |
| `resources/list` | 서버가 리소스를 알릴 때 반드시 구현해야 하는 리소스 목록 메서드 |
| `resultType` | 성공한 현대 결과에 필수인 구분자 |
| `ui/initialize` | 사라진 코어 초기화와는 별개인, Apps 다리의 첫 요청 |
| `ui/notifications/initialized` | 호스트가 응답한 뒤 뷰가 보내는 Apps 준비 완료 알림 |
| CSP | 스크립트, 스타일, 이미지, 네트워크 origin을 제한하는 브라우저 정책 |
| Text fallback | Apps를 지원하지 않는 호스트를 위해 남겨 두는 도구 동작 |

## 더 읽을거리 (Further Reading)

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
