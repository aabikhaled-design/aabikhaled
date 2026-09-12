# MCP 인가: CIMD, 발급자 결속, PKCE, 단계 상향 (MCP Authorization: CIMD, Issuer Binding, PKCE, and Step-Up)

> 원격 MCP 요청은 무상태이지만 그 인가가 익명인 것은 아니다. 모든 자격 증명을 그것을 만든 발급자에 묶고, 모든 토큰을 그것을 받는 리소스에 묶으라.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- 보호된 리소스 메타데이터를 통해 인가 서버를 찾아낸다.
- 폐기 예정인 동적 클라이언트 등록보다 Client ID Metadata Document를 먼저 쓴다.
- DCR 호환 경로를 피할 수 없을 때 올바른 `application_type`을 선언한다.
- 인가 응답의 `iss`를 검증하고 자격 증명을 발급자별로 격리한다.
- PKCE, 리소스 지시자, 대상 검증, 점진적 스코프를 쓴다.
- 프로토콜 세션 없이 인가된 MCP 2026-07-28 요청을 보낸다.

## 문제 (The Problem)

원격 MCP 서버는 사적인 기록을 읽거나, 외부 시스템에 쓰거나, 비용이 큰 작업을 일으킬 수 있다. 인증은 누가 자격 증명을 내밀었는지 알려 준다. 인가는 여기에 더해 다음에도 답해야 한다.

- 그 자격 증명은 어느 인가 서버가 발급했는가?
- 그 토큰은 어느 MCP 리소스를 위한 것인가?
- 어느 클라이언트와 리디렉션 URI가 흐름을 마쳤는가?
- 사용자는 어떤 작업을 승인했는가?
- 바로 이 요청이 아직 그 승인에 들어맞는가?

2026-07-28 인가 프로필은 클라이언트 등록과 발급자 처리를 더 단단하게 만들었다. Client ID Metadata Document를 먼저 쓰고, 동적 클라이언트 등록은 권장하지 않고, DCR에는 올바른 `application_type`을 요구하고, RFC 9207 발급자 응답을 검증하며, 발급자를 넘나드는 자격 증명 재사용을 금지한다.

이 규칙들은 무상태 코어를 보완한다. 코어 악수나 `Mcp-Session-Id`를 되살리지는 않는다.

## 개념 (The Concept)

### 세 역할을 알아 두라 (Know the three roles)

- **MCP 클라이언트:** 리소스 소유자를 대신해 요청을 보낸다.
- **MCP 리소스 서버:** 접근 토큰을 받아들이고 MCP 엔드포인트를 제공한다.
- **인가 서버:** 리소스 소유자를 인증하고, 동의를 받고, 토큰을 발급한다.

리소스 서버와 인가 서버를 함께 운영할 수는 있지만, 식별자와 검증 책임은 분리해 두라.

### 인가는 HTTP에 적용된다 (Authorization applies to HTTP)

MCP 인가 명세는 HTTP 기반 전송에 적용된다. 로컬 stdio 서버는 프로세스와 운영체제의 신뢰 경계 아래에서 돈다. 대칭을 맞추겠다고 stdio에 가짜 브라우저 OAuth 흐름을 붙이지 마라.

원격 Streamable HTTP에서는 요청마다 `Authorization` 헤더에 소지자 토큰을 실어 보낸다. URL에는 절대 넣지 마라.

### 보호된 리소스 메타데이터에서 출발하라 (Start with protected-resource metadata)

리소스 서버는 RFC 9728 메타데이터를 게시한다.

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

클라이언트는 MCP 리소스 URL에서 출발해 이 문서를 가져오고, 거기 실린 인가 서버를 하나 고른 다음, 그 서버의 OAuth 또는 OpenID Connect 메타데이터를 가져온다.

RFC 9728의 잘 알려진 URL을 만들 때 리소스 경로를 보존하라. 리소스가 `https://notes.example.com/mcp`라면 이 레슨은 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`를 쓴다. `/mcp` 꼬리를 떼면 같은 origin의 다른 보호된 리소스의 메타데이터를 고를 수 있다.

호스트 이름을 보고 인가 서버를 짐작하지 마라. 검증되지 않은 오류 본문에서 발견한 발급자를 따라가지 마라. 클라이언트가 어떤 발급자를 신뢰할 의사가 있는지 정책으로 정해 두라.

### 인가 서버 메타데이터를 확인하라 (Verify authorization server metadata)

메타데이터는 엔드포인트와 지원하는 통제 수단을 드러내야 한다.

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

PKCE에는 S256을 요구하라. 발급자 문자열을 정확히 기록하라. 그 정확한 값이 등록과 토큰 저장의 키가 된다.

### 등록 우선순위를 따르라 (Follow the registration priority)

클라이언트가 고른 발급자와 이미 명시적인 관계를 맺고 있다면 미리 등록된 클라이언트 정보를 쓰라. 그렇지 않다면, 인가 서버가 지원을 알릴 때 Client ID Metadata Document를 먼저 쓰라. DCR은 폐기 예정인 호환 대비책으로만 쓰고, 이 중 어느 것도 쓸 수 없으면 클라이언트 정보를 사용자에게 물어보라.

### Client ID Metadata Document를 먼저 쓰라 (Prefer Client ID Metadata Documents)

Client ID Metadata Document는 인가 서버에게 클라이언트 식별자이면서 동시에 그 메타데이터의 위치이기도 한 HTTPS URL을 준다.

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

인가 서버가 그 문서를 가져와 검증한다. `client_id`는 경로가 있는 HTTPS URL이어야 하고, 문서 안의 값은 그 URL과 정확히 같아야 한다. 필수 문서 필드는 `client_id`, `client_name`, `redirect_uris`다. `application_type`은 이 예제에 나오지만 CIMD의 필수 항목은 아니다. 이 항목이 새로 필수가 된 곳은 DCR 경로다.

문서를 가져오는 일을 SSRF에 민감한 작업으로 다루라. 목적지를 해석하고 검증하고, 루프백과 사설, 링크 로컬, 그 밖에 허용되지 않는 주소를 거부하고, 리디렉션과 DNS 변경 뒤에 다시 확인하고, 리디렉션 횟수와 바이트와 시간을 제한하고, JSON을 요구하고, 검증된 HTTP 캐시 제어를 따를 때만 캐시하라. `client_name`을 비롯한 표시 필드는 신뢰할 수 없는 텍스트로 다루라.

CIMD는 처음 만날 때마다 새 동적 식별자를 찍어 낼 필요를 없애 준다. 리디렉션 URI 검증이나 발급자 정책, 사용자 동의를 없애 주지는 않는다.

### DCR은 호환 경로다 (DCR is a compatibility path)

동적 클라이언트 등록은 옛 인가 서버를 위해 남아 있지만, 새 MCP 구현에서는 권장되지 않는다.

DCR을 쓸 때는 `application_type`을 선언하라.

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 데스크톱, 모바일, 명령줄, 루프백 클라이언트는 `native`를 쓴다.
- 원격에 호스팅되는 브라우저 애플리케이션은 `web`과 원격 HTTPS 리디렉션을 쓴다.

이 필드를 빼면 OpenID Connect 등록 구현에서 `web`이 기본값이 될 수 있고, 정당한 루프백 리디렉션이 실패한다.

DCR 코드는 명시적인 대비책 판단 뒤에 두라. CIMD 검증이 아무 이유로든 실패했다고 조용히 물러서지 마라. 그러면 보안 실패가 더 약한 등록 경로로 바뀔 수 있다.

### 자격 증명을 발급자에 묶어라 (Bind credentials to the issuer)

발급자가 찍어 준 등록 자료는 정확한 발급자 아래에 저장하라.

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

보호된 리소스 탐색 결과가 `https://auth-one.example`에서 `https://auth-two.example`로 바뀌면 신뢰를 다시 따져 보라. 첫 발급자의 클라이언트 비밀, DCR 클라이언트 id, 등록 접근 토큰, 갱신 토큰, 접근 토큰을 두 번째 발급자에게 절대 보내지 마라. 미리 등록된 클라이언트와 DCR 클라이언트는 새 발급자용으로 발급된 자격 증명을 써야 한다.

CIMD 클라이언트 id는 다르다. 인가 서버가 찍어 낸 자격 증명이 아니라 스스로 호스팅하는 HTTPS URL이기 때문이다. 같은 CIMD URL은 옮겨 쓸 수 있다. 새로 신뢰하게 된 발급자가 DCR 재등록 없이 그 문서를 가져와 검증한다. 인가 응답과 토큰은 여전히 새 발급자 아래에서 검증되고 저장된다.

### PKCE를 곁들인 인가 코드 (Authorization code with PKCE)

대화형 흐름은 이렇다.

1. 엔트로피가 높은 `code_verifier`를 만든다.
2. S256 `code_challenge`를 유도한다.
3. 정확한 `client_id`, `redirect_uri`, `scope`, `code_challenge`, `resource`를 실어 인가 요청을 보낸다.
4. `code`와, 제공된다면 `iss`를 담은 인가 응답을 받는다.
5. 응답 필드를 하나라도 쓰기 전에 기록해 둔 발급자와 `iss`를 대조해 검증한다.
6. `code_verifier`, 같은 리디렉션 URI, 같은 `resource`를 실어 코드를 교환한다.
7. 받아 낸 토큰을 `(issuer, resource)` 아래에 저장한다.

RFC 8707의 `resource` 매개변수는 인가 요청과 토큰 요청 양쪽에 나타난다. 이것이 정식 MCP 서버 URI를 식별한다.

### `iss`를 정확히 검증하라 (Validate `iss` exactly)

RFC 9207은 한 발급자의 인가 응답이 다른 발급자의 응답과 헷갈리는 것을 막아 준다.

`iss`가 있으면 기록해 둔 발급자와 비교하되, 대소문자를 접거나, 끝의 슬래시를 바꾸거나, 기본 포트를 떼거나, 퍼센트 인코딩을 정규화하지 마라. 어긋나면 그 코드를 쓰지 말고, 그 응답에 담긴 공격자 조종 오류 내용을 화면에 보여 주지도 마라.

`iss`를 담아 주는 인가 서버는 `authorization_response_iss_parameter_supported: true`를 알린다. 현행 클라이언트는 그런 알림이 없더라도 들어 있는 `iss`를 검증한다.

### MCP 서버에서 대상을 검증하라 (Validate audience at the MCP server)

리소스 서버는 자기 자신을 위해 발급된 토큰만 받아들인다.

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

유효하지 않거나, 만료됐거나, 발급자가 다르거나, 대상이 다른 토큰은 401을 받는다. MCP 서버는 다른 서비스용 토큰을 받아들이거나 흘려보내서는 안 된다.

### 지금 필요한 가장 작은 스코프를 요청하라 (Request the smallest current scope)

지금 필요한 스코프에서 시작하라. 나중에 어떤 도구가 더 필요로 하면, 서버는 권위 있는 스코프 요구와 함께 403을 돌려준다.

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

클라이언트는 새 권한을 설명하고, 동의를 받고, 합친 스코프 집합으로 새 인가 흐름을 수행한 다음, 새 JSON-RPC id로 MCP 요청을 재시도한다.

요구된 스코프가 `scopes_supported`의 부분집합이라고 가정하지 마라. 지금 이 작업에 대해서는 그 요구가 권위를 가진다.

### 인가와 무상태 MCP 전선 (Authorization and the stateless MCP wire)

인가된 도구 호출도 현행 요청 봉투를 온전히 싣고 다닌다.

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

토큰은 주체를 인가한다. 요청 메타데이터는 프로토콜 동작을 협상한다. 어느 쪽도 다른 쪽을 대신하지 않는다.

전선은 정해진 순서로 검증하라. JSON-RPC와 메타데이터 타입, 헤더와 본문의 일치, 그다음 프로토콜 지원 여부다. 라우팅 헤더나 버전 헤더가 어긋나면 `-32020`과 함께 HTTP 400을 돌려준다. 헤더와 본문이 지원하지 않는 버전에서 일치하면 `-32022`와 함께 HTTP 400을 돌려주고 `data`는 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`로 채운다. 알 수 없는 메서드에는 `-32601`과 함께 HTTP 404를 돌려준다.

401 유효하지 않은 토큰과 403 스코프 부족을 포함해 모든 요청 오류는 원래 요청 `id`를 담은 JSON-RPC 오류 봉투다. 구조화된 복구 정보는 선택적인 오류 `data`에 담고, `WWW-Authenticate`는 HTTP 응답 헤더로 남겨 둔다. 알림에는 `id`가 없으므로 JSON-RPC 본문을 받지 않는다. 받아들인 HTTP 알림은 빈 본문과 함께 202를 돌려준다.

서버는 `server/discover`를 구현하고 도구를 알리므로 필수인 `tools/list` 메서드도 구현한다. 도구 서술자에는 안정적인 이름과 설명, 루트가 객체인 `inputSchema` 값이 들어간다. 목록은 결정적이며 `resultType`, 서버 신원 메타데이터, 상한이 있는 `ttlMs`, `cacheScope`를 돌려준다. 탐색과 사용자에 무관한 도구 목록은 인가 전에도 제공할 수 있다. 둘 중 하나가 주체에 따라 달라진다면 평소의 정책과 비공개 캐싱을 적용하라.

### 토큰을 그대로 넘기지 마라 (No token passthrough)

MCP 서버는 클라이언트의 MCP 접근 토큰을 하위 API로 그대로 넘겨서는 안 된다. 올바른 대상을 가진 별도의 하위 토큰을 얻거나 명시적인 토큰 교환 설계를 쓰라. 대상 검증은 서비스들이 남을 위해 찍힌 토큰을 거부할 때만 작동한다.

### 갱신 토큰 (Refresh tokens)

갱신 토큰은 선택 사항이다. 발급되면 비밀로 저장하고 발급자와 리소스를 키로 삼으라. 그것이 있으리라고 가정하지 마라. 인가 서버가 순환을 지원하면 순환시키고, 무효화된 값이 다시 쓰이는 것을 탐지하라.

```figure
t3-scope-stepup
```

## 만들어 보기 (Build It)

`code/main.py`는 프로세스 안에서 도는 프로토콜 및 인가 시뮬레이터다. 보호된 리소스 탐색, 인가 서버 메타데이터, CIMD 등록, 버전으로 걸러진 DCR 대비책, 애플리케이션 타입 검사, PKCE, 발급자 검증, 리소스에 묶인 토큰, 스코프 단계 상향, `server/discover`, `tools/list`, 그리고 무상태 도구 요청을 구현한다.

이 모형은 이미 파싱된 요청 본문과 라우팅 헤더를 받는다. 완전한 HTTP 어댑터가 아니며 `Content-Type`이나 `Accept`를 파싱하지 않는다. `Content-Type: application/json`과 `application/json`, `text/event-stream`을 모두 담은 `Accept` 값을 요구하는 레슨 09의 Streamable HTTP 어댑터에 붙여 쓰라.

실행은 이렇게 한다.

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

출력은 먼저 탐색을 보여 주고, CIMD 등록, 평범한 읽기, 서로 다른 스코프 단계 상향 두 번, 발급자를 키로 삼은 자격 증명 저장을 차례로 보여 준다.

## 직접 해 보기 (Use It)

시뮬레이터의 객체를 실제 운영 구성 요소에 대응시켜 보라.

- `ResourceServer.protected_resource_metadata`는 RFC 9728 엔드포인트가 된다.
- `AuthorizationServer.metadata`는 RFC 8414이나 OpenID Connect 탐색이 된다.
- `Client.enroll`은 CIMD 해석과 명시적인 DCR 호환 분기가 된다.
- 발급자가 찍어 준 클라이언트 자격 증명과 `tokens_by_issuer_resource`는 암호화된 레코드가 된다. CIMD URL은 옮겨 쓸 수 있는 채로 남아도, 그 인가 결과는 발급자에 묶인 채로 남는다.
- `ResourceServer.handle`은 디스패치 전에 현행 MCP 헤더와 토큰, 도구 스코프를 검증하면서 모든 요청 오류를 짝이 맞는 JSON-RPC 봉투에 담아 두는 미들웨어가 된다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-oauth-scope-planner.md`를 만든다. 이제 등록 우선순위, 발급자에 묶인 자격 증명 저장, 애플리케이션 타입, PKCE, 리소스 지시자, 스코프 요구, 현행 무상태 요청 경계까지 설계해 준다.

## 연습 문제 (Exercises)

1. 갱신 토큰 순환을 추가하고 이전 갱신 토큰의 재사용을 거부하라.
2. 발급자 허용 목록을 추가하라. 발급자가 바뀌면 옮겨 쓸 수 있는 CIMD URL만 다시 쓰고, 앞선 발급자가 찍어 준 자격 증명과 토큰은 모두 거부하라.
3. 인가 코드에 만료를 추가하고 늦은 교환이 실패하는지 확인하라.
4. 원격 HTTPS 리디렉션을 쓰는 웹 클라이언트 변형을 만들고, 그 DCR 메타데이터를 네이티브 클라이언트의 것과 비교하라.
5. 같은 발급자 아래에 리소스를 하나 더 추가하라. 그 접근 토큰을 첫 리소스에서는 쓸 수 없음을 확인하라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| Protected-resource metadata | 리소스와 인가 서버를 식별해 주는 RFC 9728 문서 |
| CIMD | URL 자체가 OAuth 클라이언트 식별자인 HTTPS 메타데이터 문서 |
| DCR | 호환성을 위해 남겨 둔, 폐기 예정인 동적 클라이언트 등록 |
| `application_type` | 리디렉션 URI 규칙을 검증하는 데 쓰는 `native` 또는 `web` |
| PKCE | 가로채인 인가 코드를 보호하는 검증자와 S256 도전 값 |
| `iss` | RFC 9207의 인가 응답 발급자 식별자 |
| Resource indicator | 토큰 요청을 MCP 리소스에 묶는 RFC 8707 매개변수 |
| Audience | 토큰이 유효한 리소스 |
| Step-up | 지금 이 작업에 필요한 추가 스코프를 위한 새 동의와 토큰 발급 |
| Issuer-bound credentials | 정확한 인가 서버 발급자별로 격리한 등록 및 토큰 레코드 |

## 더 읽을거리 (Further Reading)

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
