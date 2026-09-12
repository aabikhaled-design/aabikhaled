# 실제 운영에서의 MCP 인증: 발급자에 묶인 등록과 토큰 (MCP Auth in Production: Issuer-Bound Enrollment and Tokens)

> 레슨 16은 OAuth 2.1 상태 기계를 만들었다. 이 레슨은 그 운영 경계를 MCP 2026-07-28에 맞춰 단단하게 만든다. Client ID Metadata Document를 먼저 쓰고, 폐기 예정인 동적 등록은 호환용으로만 남기고, 인가 응답의 발급자를 검증하고, 클라이언트 자격 증명을 발급자별로 저장하고, JWKS를 갱신하며, 무상태 요청마다 대상에 고정된 토큰을 확인한다.
>
> **명세 참고 (2026-07-28):** 동적 클라이언트 등록은 Client ID Metadata Document에 자리를 내주고 폐기 예정이 됐다. DCR은 호환 수단으로 남는다. 그것을 쓸 때 클라이언트는 올바른 `application_type`을 선언한다. 클라이언트는 실려 온 RFC 9207 `iss` 값을 검증하고, 인가 서버 발급자를 넘나들며 자격 증명을 재사용하지 않는다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- RFC 8414 메타데이터로 인가 서버를 찾아내고 그 계약을 확인한다.
- Client ID Metadata Document로 등록하고, 폐기 예정인 DCR은 대비책으로 떼어 놓는다.
- RFC 9207 `iss`를 검증하고, 등록은 인가 서버 발급자별로, 리소스에 묶인 토큰은 발급자와 리소스 쌍으로 저장한다.
- JWKS 키를 캐시하고 일정에 맞춰 갱신해, 키가 바뀌어도 서명 검증이 살아남게 한다.
- RFC 8707 리소스 지시자로 토큰을 MCP 리소스 하나에 고정하고, 헷갈린 대리인 식의 재사용을 거부한다.
- JWT 검증과 토큰 내부 조회 중 하나를 고르고, 폐기 신선도를 정의하고, 신원 의존 서비스가 죽었을 때 안전하게 실패한다.
- 인가 서버, 리소스 서버, 클라이언트를 갈라놓아 각자 자기 몫의 검사만 강제하게 한다.
- 배포 점검표에 비춰 인가 서버를 감사하고, 안전하지 않은 등록이나 토큰 재사용을 거부한다.

## 문제 (The Problem)

레슨 16의 시뮬레이터는 OAuth 2.1을 메모리 안에서 돌린다. 실제 운영에는 메모리만 쓰는 시뮬레이터가 보지 못하는 운영상의 틈이 셋 있다.

첫째 틈은 등록과 자격 증명 격리다. 실제 조직은 MCP 서버 수백 개와 MCP 클라이언트 수천 개를 굴릴 수 있다. 2026-07-28 개정판은 **Client ID Metadata Document**를 먼저 쓴다. 클라이언트가 자기가 통제하는, 경로가 있는 HTTPS URL을 식별자로 삼고 인가 서버가 그 메타데이터를 가져온다. RFC 7591 동적 등록은 폐기 예정인 호환 경로로만 남는다. DCR을 피할 수 없다면 요청에 올바른 `application_type`을 선언한다. 클라이언트는 등록을 인가 서버 발급자 아래에, 접근 토큰을 `(issuer, resource)` 쌍 아래에 저장한다. 발급자가 바뀌면 새로 등록해야 하고, 리소스가 다르면 대상이 따로 묶인 토큰이어야 한다.

둘째 틈은 키 교체다. JWT 검증은 인가 서버의 서명 키에 기대며, 그 키는 JSON Web Key Set(JWKS)으로 게시된다. 인가 서버는 이것을 일정에 맞춰 교체한다(보통 매시간, 사고 대응 중에는 더 빠르게). 부팅할 때 JWKS를 한 번만 가져오는 MCP 서버는 교체 시점까지는 잘 검증하다가, 그 순간부터 재시작할 때까지 모든 요청이 실패한다. 실제 운영에서는 JWKS를 캐시 값으로 두고, 이전 키가 만료되기 전에 캐시를 덮어쓰는 갱신 작업을 붙이며, 캐시보다 새로운 키로 서명된 토큰이 도착하는 경우를 위해 캐시 미스 때 한 번 가져오는 대비책도 둔다.

셋째 틈은 대상 결속이다. 레슨 16은 RFC 8707 리소스 지시자를 소개했다. 실제 운영에서 그 지시자는 요청마다 강제되는 클레임 검사가 된다. MCP 서버는 `token.aud`를 자기 정식 리소스 URL과 비교하고, 어긋나면 HTTP 401로 거부한다. 이것이 같은 신뢰망 안의 상위 MCP 서버(또는 다른 서버용 토큰을 쥔 악의적 클라이언트)가 그 토큰을 다른 서버에 재생하는 것을 막는 유일한 방어다.

이 레슨은 그 틈 하나하나를 표면의 구체적인 조각에 대응시킨다. 메타데이터 문서는 HTTP 엔드포인트다. JWKS 캐시 갱신은 예약 작업에 키-값 캐시를 더한 것이다. JWT 검증은 리소스 서버가 어떤 도구를 부르기 전에 돌리는 루틴이다. 세 역할을 갈라 두면 각자 자기가 맡은 검사만 강제한다. 인가 서버는 발급하고 키를 교체하고, 리소스 서버는 캐시하고 검증하고, 클라이언트는 찾아내고 등록한다.

## 범위: 레슨 16 이후의 운영 강제 (Scope: Production Enforcement After Lesson 16)

[레슨 16: OAuth 2.1로 보는 MCP 보안](../../16-mcp-security-oauth-2-1/docs/ko.md)이 인가 코드 상태 기계, PKCE, 보호된 리소스 탐색, 리소스 지시자, 스코프 판단을 맡는다. 이 레슨은 두 번째 OAuth 흐름을 정의하지 않는다. 그 계약들이 이미 존재하는 지점에서 시작해, 배포된 리소스 서버가 키 교체와 불투명 토큰 검증, 폐기, 의존 서비스 장애, 배포, 사고 대응을 겪는 동안 그 계약을 어떻게 계속 강제하는지 묻는다.

운영 경계는 더 좁고 더 실무적이다.

- JWT 경로는 요청마다 고정된 발급자, 알고리즘, 서명 키, 대상, 시각 클레임, 스코프를 확인하면서 JWKS를 안전하게 갱신한다.
- 불투명 토큰 경로는 발급자의 인증된 내부 조회 엔드포인트를 부르고, 돌아온 활성 상태와 대상 또는 리소스, 만료, 주체, 스코프를 검증한다.
- 폐기 정책은 자격 증명이 얼마나 빨리 먹히지 않아야 하는지와, 어느 캐시가 그 사실을 늦출 수 있는지를 정한다.
- 실패 정책은 탐색이나 JWKS, 내부 조회, 폐기 인프라를 쓸 수 없을 때 무슨 일이 벌어지는지를 정한다.
- 증거는 어떤 발급자 메타데이터와 키 집합 또는 내부 조회 응답, 토큰 클레임, 정책 버전, 거부 이유가 그 결과를 낳았는지를 토큰 자체를 저장하지 않고 기록한다.

이렇게 나누면 두 레슨을 겹쳐 쓸 수 있다. 레슨 16은 흐름을 증명한다. 레슨 18은 토큰이 실제 MCP 요청 경로에 닿은 뒤에도 믿을 만한 상태로 남아 있거나, 아니면 거부된다는 것을 증명한다.

## 개념 (The Concept)

### RFC 8414: OAuth 인가 서버 메타데이터 (RFC 8414 — OAuth Authorization Server Metadata)

`/.well-known/oauth-authorization-server`에 있는 문서가 클라이언트에 필요한 모든 것을 설명한다.

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCP 리소스 URL을 받은 클라이언트는 탐색을 사슬처럼 잇는다. RFC 9728의 `oauth-protected-resource`(리소스 서버의 문서)가 발급자를 지목하고, 그다음 이 RFC의 `oauth-authorization-server`가 모든 엔드포인트를 알려 준다. 클라이언트는 인가 URL을 코드에 박아 넣지 않는다.

경로가 있는 리소스 식별자라면 그 경로 앞에 잘 알려진 조각을 끼워 넣는다. 예를 들어 `https://mcp.example.com/team/server`의 보호된 리소스 메타데이터는 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`에 있다. 리소스 경로 뒤에 `/.well-known/...`을 이어 붙이는 것은 틀렸다.

MCP를 위해 어떤 IdP를 믿기 전에 확인할 계약은 이렇다.

- `code_challenge_methods_supported`에 `S256`이 들어 있다(RFC 7636의 PKCE). 명세는 분명하다. 이 필드가 **없다면** 인가 서버는 PKCE를 지원하지 않는 것이고, 클라이언트는 진행을 **반드시** 거부해야 한다.
- `grant_types_supported`에 `authorization_code`가 있고 `password`와 `implicit`은 거부한다.
- 등록 경로가 적어도 하나는 있다. `client_id_metadata_document_supported: true`(CIMD, 권장), 미리 등록된 클라이언트, 또는 `registration_endpoint`(폐기 예정인 RFC 7591 호환)다.
- `authorization_response_iss_parameter_supported`가 참이면, 클라이언트는 돌아온 RFC 9207 `iss`를 요구하고 리디렉션 전에 기록해 둔 발급자와 정확히 비교한다.
- OAuth 2.1에서 `response_types_supported`는 정확히 `["code"]`다.

`S256`이 없으면 MCP 서버는 이 IdP를 상대로 배포하기를 거부한다. PKCE에는 성능을 낮춘 모드가 없다. 등록 경로가 *둘 다* 알려지지 않았는데 미리 받아 둔 `client_id`도 없다면 등록 자체가 불가능하다. 그때 잘못된 것은 코드가 아니라 배포 명세다.

### RFC 9728 (복습): 보호된 리소스 메타데이터 (RFC 9728 (recap) — Protected Resource Metadata)

레슨 16이 RFC 9728을 다뤘다. 실제 운영에서의 차이는 이렇다. 이 문서가 *이* MCP 서버가 신뢰하는 인가 서버를 찾을 수 있는 유일한 곳이다. MCP 서버 하나가 여러 IdP의 토큰을 받아들일 수 있다(직원용 하나, 파트너용 하나). RFC 9728이 그 집합을 선언하고, RFC 8414가 각 IdP가 무엇을 지원하는지 설명한다.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Document (권장 기본값) (Client ID Metadata Documents (the recommended default))

CIMD는 등록을 *밀어 넣기*에서 *끌어오기*로 뒤집는다. 인가 서버에게 `client_id`를 찍어 달라고 부탁하는 대신, 클라이언트가 자기가 통제하는 HTTPS URL을 `client_id`로 **삼는다**. 그 URL은 JSON 메타데이터 문서로 이어지고, 인가 서버가 OAuth 흐름 도중에 필요할 때 그것을 가져온다. 신뢰의 뿌리는 DNS다. 서버 운영자가 `app.example.com`을 믿는다면 `https://app.example.com/client.json`에서 제공되는 클라이언트를 믿는다. 등록을 위한 왕복도, 소진될 `client_id` 이름 공간도, 서버마다 맞춰 둘 상태도 없다.

클라이언트가 제공하는 메타데이터 문서는 이렇게 생겼다.

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

문서 안의 `client_id` 값은 그 문서가 제공되는 URL과 **반드시** 같아야 한다(인가 서버가 이것을 확인하고, 어긋나면 거부한다). 인가 서버는 RFC 8414 메타데이터에 `client_id_metadata_document_supported: true`로 지원을 알린다.

현행 CIMD 계약에서 필수는 `client_id`, `client_name`, 그리고 비어 있지 않은 `redirect_uris` 배열이다. 클라이언트 식별자는 경로가 있는 절대 HTTPS URL이다. `application_type`을 넣어도 되지만 CIMD의 필수 필드는 아니다. DCR의 `application_type` 요구사항을 권장 경로인 CIMD로 베껴 오지 마라.

명세가 딱 잘라 말하는 보안 사실이 둘 있다.

- **SSRF.** 인가 서버가 공격자가 건넨 URL을 가져온다. 서버 측 요청 위조를 반드시 막아야 한다(내부나 관리용 엔드포인트를 가져오지 않는다).
- **localhost 사칭.** CIMD만으로는 지역 공격자가 정당한 클라이언트의 메타데이터 URL을 사칭해 아무 `localhost` 리디렉션이나 묶는 것을 막을 수 없다. 인가 서버는 동의 화면에서 리디렉션 URI의 호스트 이름을 **반드시** 뚜렷하게 보여 줘야 하고, `localhost`로만 가는 리디렉션에는 경고를 **띄우는 것이 좋다**.

CIMD는 서버 쪽 상태가 필요 없으므로, DCR처럼 등록기를 세울 일이 없다. 클라이언트 쪽은 읽기 전용이다. 메타데이터 문서를 정적 HTTPS 엔드포인트에서 제공하고 인가 서버가 끌어가게 두면 된다.

인가 서버 운영자가 이미 클라이언트 식별자를 내어 줬다면, 자동 등록을 시도하기 전에 그 발급자 범위의 등록을 쓰라. 그렇지 않다면 CIMD를 먼저 쓰라. 발급자가 미리 등록도 CIMD도 쓸 수 없을 때만 폐기 예정인 DCR을 쓰라.

### RFC 7591: 폐기 예정인 호환용 등록 (RFC 7591: deprecated compatibility enrollment)

DCR은 2026-07-28 개정판에서 폐기 예정이다. CIMD를 소화하지 못하고 미리 등록도 현실적으로 어려운 인가 서버를 위해서만 남겨 두라. 호환용 클라이언트는 이렇게 보낸다.

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

서버는 `client_id`와, 나중에 갱신할 때 쓸 `registration_access_token`을 돌려준다.

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type`은 장식이 아니다. 루프백을 쓰는 데스크톱 클라이언트는 `native`를 선언하고, 서버에 호스팅되는 클라이언트는 `web`을 선언하며 HTTPS 리디렉션 URI를 쓴다. 공개 네이티브 클라이언트에는 `token_endpoint_auth_method: none`이 올바른 기본값이다. `client_id`만 받고, 소지 증명은 PKCE가 맡는다.

실제 운영에서 빠지기 쉬운 함정이 셋 있다.

- 등록 엔드포인트는 출발지 IP별로 속도를 제한해야 한다. 그러지 않으면 적대적인 상대가 가짜 등록을 수백만 개 만들어 `client_id` 이름 공간을 소진시킨다. 등록기가 요청을 처리하기 전에 속도 제한 검사를 돌리라.
- 일부 기업용 IdP는 `software_statement`(클라이언트를 보증하는 서명된 JWT)를 요구한다. 이 레슨의 모형은 그것을 건너뛴다. 실제 운영에서는 루프백 리디렉션 URI가 아닌 곳에서 온 서명 없는 등록을 거부하는 검증 단계를 붙인다.
- `registration_access_token`은 평문이 아니라 해시로 저장해야 한다. 이 토큰을 빼앗기면 공격자가 클라이언트의 리디렉션 URI를 고쳐 쓸 수 있다.

### RFC 8707 (복습): 리소스 지시자 (RFC 8707 (recap) — Resource Indicators)

레슨 16이 형태를 잡았다. 실제 운영의 규칙은 이렇다. 모든 토큰 요청에 `resource=<canonical-mcp-url>`을 싣고, MCP 서버는 호출마다 `token.aud`가 자기 리소스 URL과 맞는지 확인한다. 정식 URI는 그 서버를 가리키는 *가장 구체적인* 식별자다. 스킴과 호스트는 소문자, 프래그먼트는 없고, 관례상 끝의 슬래시도 없다. 경로는 규칙상 **떼어 내지 않는다**. 명세는 개별 MCP 서버를 식별하는 데 필요하면 경로를 남긴다. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, `https://mcp.example.com/server/mcp`는 모두 유효한 정식 URI다. 서버마다 하나를 골라 `aud`를 정확히 거기에 고정하라. (이 레슨의 모형은 간결하게 쓰려고 `https://notes.example.com` 같은 맨 호스트 대상을 쓴다. origin 하나에 MCP 서버를 여럿 얹는 배포라면 경로로 구별한다.)

### RFC 7636 (복습): PKCE (RFC 7636 (recap) — PKCE)

OAuth 2.1에서 PKCE는 필수다. 이 레슨의 인가 코드 흐름은 언제나 `code_challenge`와 `code_verifier`를 싣는다. 서버는 검증자가 없거나 저장된 도전 값으로 해시되지 않는 토큰 요청을 거부한다.

### MCP 2026-07-28 인가 프로필 (MCP 2026-07-28 authorization profile)

현행 MCP 개정판은 MCP 전송을 무상태로 만들면서도 OAuth 리소스 서버 경계는 그대로 둔다. 신원 판단을 캐시해 둘 프로토콜 세션이 없다. 그래서 인가 계층은 요청마다 독립적으로 검증한다.

- RFC 9728 보호된 리소스 메타데이터를 구현하고, 401에 실리는 `WWW-Authenticate: Bearer resource_metadata="..."` 헤더나 잘 알려진 URI `/.well-known/oauth-protected-resource` 중 **하나로** 그 위치를 알려라(SEP-985가 헤더를 선택으로 만들고 잘 알려진 URI를 대비책으로 두었다). 메타데이터의 `authorization_servers` 필드는 서버를 적어도 하나 **반드시** 지목해야 한다.
- 토큰은 **모든** 요청에서 `Authorization: Bearer ...`로만 받아들여라. 질의 문자열에 담지 말고, 세션 시작 때만 검증하지도 마라.
- 요청마다 `aud`, `iss`, `exp`, 필요한 스코프를 검증하라. 서버는 그 토큰이 자기 자신을 위해 발급됐는지(대상) **반드시** 검증해야 한다. `aud`가 없거나 어긋나면 거부하고, 절대 아무 값이나 되는 것으로 취급하지 마라.
- 401이나 403에는 `error=...`와, 메타데이터 문서의 URL을 담은 `resource_metadata="<PRM-URL>"` 매개변수(맨 리소스가 *아니다*), 그리고 `insufficient_scope`(403)일 때는 `scope="..."`를 실은 `WWW-Authenticate: Bearer`를 돌려주라. 참고로 그 매개변수는 탐색 포인터인 `resource_metadata`다. 이 도전 값에 `resource` 매개변수는 없다.
- 인가 서버 탐색은 RFC 8414 OAuth 메타데이터 **또는** OpenID Connect Discovery 1.0을 받아들인다. 클라이언트는 두 잘 알려진 접미사를 우선순위 순서로 시도해야 한다.
- **혼동 공격**은 서버가 아니라 클라이언트가 막는다. 리디렉션 전에 기대하는 `issuer`를 기록해 두고, 코드를 교환하기 전에 실제 인가 응답에 돌아온 `iss` 값(RFC 9207)을 검증한다. PKCE만으로는 혼동 공격을 막지 못한다. 클라이언트가 자기 `code_verifier`를 자신이 끌려간 그 토큰 엔드포인트에 건네주기 때문이다.
- 클라이언트 자격 증명은 인가 서버 발급자 하나에 속한다. 탐색이 다른 발급자로 이어지면, 클라이언트는 옛 `client_id`나 등록 토큰, 접근 토큰을 내밀지 말고 다시 등록한다.
- CIMD가 권장 등록 방식이다. DCR은 폐기 예정이며, 호환용 DCR 요청이라도 올바른 `application_type`을 선언한다.

OAuth 2.1 초안이 바탕이고, RFC 8414/7591/8707/9728/9207에 RFC 7636과 CIMD를 더한 것이 표면이며, MCP 명세가 그 프로필이다.

### 배포 역량 점검표 (Deployment capability checklist)

공급사의 기능 비교표는 금세 낡는다. 대신 실제로 배포할 인가 서버가 돌려주는 메타데이터를 들여다보라. 통과 기준은 기계적이다.

| 점검 항목 | 필요한 판단 |
|---|---|
| 탐색된 발급자 | 정책이 기대하는 정확한 HTTPS 발급자 |
| PKCE | `S256`이 알려져 있어야 하며, 아니면 중단 |
| 등록 | CIMD 우선, 미리 등록 허용, DCR은 폐기 예정 호환용으로만 |
| 인가 응답 | RFC 9207 `iss`가 실렸거나 알려졌으면 검증 |
| 리소스 결속 | 토큰 요청이 `resource`를 싣고, 리소스 서버가 맞는 `aud`를 요구 |
| 자격 증명 저장 | 클라이언트 ID와 등록 자격 증명은 발급자별로, 접근 토큰은 발급자와 리소스별로 |
| DCR 호환 | `native`나 `web`을 선언하고, 선언한 애플리케이션 타입에 맞지 않는 리디렉션 URI는 거부 |

제품 이름이나 요금제 등급을 보고 지원 여부를 추측하지 마라. 탐색한 문서를 배포 증거로 붙잡아 두고, 필수 필드가 없으면 막는 쪽으로 실패하라.

### JWKS 갱신 방식 (인가 서버가 교체하고 리소스 서버가 갱신한다) (JWKS refresh pattern (rotate at the AS, refresh at the resource server))

두 동사를 갈라 두라. 이 둘을 뭉개는 것이 실제 운영 버그다.

- **교체**는 *인가 서버*가 하는 일이다. 새 서명 키를 찍어 JWKS에 게시하고, 옛 키는 나중에 물린다. 리소스 서버는 여기에 관여하지 않고 할 수도 없다. IdP의 비밀 키를 쥐고 있지 않기 때문이다.
- **갱신**은 *리소스 서버*가 하는 일이다. 게시된 JWKS를 다시 `GET`해 자기 캐시에 넣는다. 리소스 서버가 JWKS에 대해 하는 일은 그것뿐이다.

실제 운영의 장애 양상은 낡은 캐시다. 예약 갱신 작업과 키-값 캐시로 풀라. 리소스 서버는 정해진 간격마다 `<issuer>/.well-known/jwks.json`을 가져와 `cache[issuer] = {keys, fetched_at}`를 덮어쓰는 작업(cron이든 타이머든 런타임이 주는 무엇이든)을 돌린다. 검증기는 그 캐시에서 읽는다. `kid`가 캐시에 없는 토큰은 동기적 갱신을 **한 번** 일으키고 다시 확인한다. 이것이 두 경우를 한꺼번에 해결한다. 예약 갱신과, 갓 만든 키로 서명된 토큰이 다음 예약 갱신보다 먼저 도착하는 키 겹침 구간이다.

이 대비책은 **반드시 다시 가져오기여야 하고 절대 교체여서는 안 된다**. 캐시 미스 경로를 교체 후 발급으로 연결하면 두 가지가 깨진다. (1) 새 키를 찍어 봐야 그 `kid`는 *여전히* 토큰과 맞지 않으므로 조회는 어차피 실패하고, (2) 무작위 `kid`를 실은 토큰을 뿌리는 공격자가 끝없는 키 생성을 강제해 스스로에게 서비스 거부를 일으킨다. 다시 가져오기는 멱등이므로 엉터리 `kid`는 헛된 요청 한 번의 비용만 낸다.

캐시의 모양은 이렇다.

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

키가 둘 있는 것이 정상 상태다. 인가 서버는 이전 키(`k_2026_03`)를 물리기 전에 다음 키(`k_2026_04`)를 들여오는 식으로 교체하므로, 옛 키로 발급된 토큰은 만료될 때까지 유효하다. 캐시는 그 합집합을 쥐고, 검증기는 `kid`로 고른다.

### 검증 루틴 (The validation routine)

MCP 서버는 어떤 도구를 부르기 전에 검증을 돌린다. `code/main.py`가 쓰는 형태는 이렇다.

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate`는 JWT를 디코딩하고, JWKS 캐시에서 서명 키를 찾고(미스면 한 번 갱신하고), 서명을 확인한 다음, `iss`를 허용 목록과, `aud`를 이 서버의 정식 리소스와 비교하고, `exp`와 필요한 스코프를 확인한다. 처음 실패하는 지점에서 `WWW-Authenticate` 도전 값을 돌려준다. 이것을 리소스 서버의 단일 루틴으로 두면 모든 진입점(모든 도구 호출, 모든 전송)이 같은 검사를 거친다. 검증을 건너뛰고 도구에 닿는 경로가 아예 없다.

### 불투명 토큰은 짐작이 아니라 내부 조회를 쓴다 (Opaque tokens use introspection, not guesswork)

모든 접근 토큰이 JWT인 것은 아니다. 발급자가 불투명 토큰을 쓴다고 문서화했다면, 리소스 서버는 그것을 믿을 만한 클레임으로 풀어낼 수 없다. 인증된 뒷길로 발급자의 RFC 7662 내부 조회 엔드포인트에 토큰을 보내고, `active: true`와 기대한 발급자 맥락, 정확한 MCP 대상 또는 리소스, 만료되지 않은 시각 클레임, 그리고 그 도구가 요구하는 스코프를 요구한다.

내부 조회 결과는 발급자와 토큰의 단방향 요약값, MCP 리소스를 키로 삼아 캐시하라. 평문 토큰을 로그나 캐시 이름표로 절대 쓰지 마라. 긍정 캐시 항목의 수명은 토큰 만료, 발급자의 캐시 안내, 배포의 폐기 신선도 목표 중 가장 이른 것에 맞추라. 부정 캐시는 갓 발급된 토큰이 잘못 비활성으로 남지 않을 만큼 짧게 두라. 불투명 토큰 문자열이 같더라도 한 리소스에 대한 결과가 다른 리소스를 인가해 주지는 않는다.

공격자가 조종하는 토큰 내용을 보고 검증 방식을 고르지 마라. JWT를 쓸지 내부 조회를 쓸지는 검증된 발급자 메타데이터와 배포 설정에 고정하라. JWT 경로에서는 받아들일 알고리즘과 신뢰하는 `jwks_uri`를 고정하고, 토큰 헤더만 보고 고른 키 URL이나 알고리즘을 절대 따라가지 마라.

### 폐기는 신선도 계약이다 (Revocation is a freshness contract)

RFC 7009은 클라이언트가 인가 서버에 토큰 폐기를 요청할 수 있게 한다. 그 요청이 이미 각 리소스 서버가 캐시해 둔 사본을 지워 주지는 않는다. 받아들일 수 있는 최대 폐기 지연을 정하고, 모든 캐시가 그것을 지키게 하라.

불투명 토큰 배포는 위험이 큰 호출마다 내부 조회를 하거나 긍정 캐시를 짧게 두어 더 촘촘한 폐기를 이룰 수 있다. 자기완결적인 JWT 배포는 보통 짧은 접근 토큰 수명에 갱신 토큰 폐기, 발급자 전체 사고에 대한 키 물리기, 그리고 비상시 지역 거부를 위한 선택적 주체나 세션, 토큰 id 차단 목록을 함께 쓴다. 서명된 JWT는 리소스 서버가 최신 외부 폐기 증거를 가지고 있지 않는 한 만료될 때까지 암호학적으로 유효하다.

로그아웃, 계정 비활성화, 동의 철회, 사고 대응은 서로 다른 계기지만, 측정 가능한 한 문장으로 모여야 한다. 선언한 폐기 구간이 지나면 모든 복제본이 그 자격 증명을 거부한다는 문장이다. 그 문장은 따뜻하게 돌아가는 프로세스 하나가 아니라 부하 분산기를 거쳐 시험하라.

### 의존 서비스 장애에는 선언된 판단이 필요하다 (Dependency failure needs a declared decision)

가용성 정책을 예외 처리기 안에서 즉흥으로 만들지 마라.

| 장애 | 안전한 운영 동작 |
|---|---|
| 예약 JWKS 갱신이 실패했고 알려진 `kid`가 아직 유효한 제한 캐시에 남아 있다 | 선언한 오류 시 유지 구간 안에서만 계속하고 성능 저하 상태를 건강 증거로 내보낸다 |
| 토큰의 `kid`를 모르는데 허용된 한 번의 갱신도 실패했다 | 거부한다. 확인할 수 없는 서명을 절대 받아들이지 않는다 |
| 내부 조회를 쓸 수 없다 | 보호된 호출은 막는 쪽으로 실패한다. 네트워크 장애를 `active: true`로 바꾸지 않는다 |
| 보호된 리소스나 발급자 메타데이터가 예기치 않게 바뀌었다 | 새 등록과 토큰 획득을 멈추고, 제한된 사고 정책 아래에서 명시적으로 고정한 만료 전 설정만 유지한다 |
| 폐기 엔드포인트를 쓸 수 없다 | 로그아웃이나 폐기를 미완으로 보고하고, 가능하면 자격 증명을 지역에서 쓸 수 없게 남기며, 전역 폐기가 성공했다고 주장하지 않는다 |
| 시계 원본이나 클레임 타입이 유효하지 않다 | 토큰이 통과할 때까지 허용 오차를 넓히지 말고 거부한다 |

장애를 유효하지 않은 자격 증명과 따로 분류하라. 의존 서비스 중단은 건강 상태와 재시도 정책이 붙는 운영 오류다. 잘못된 서명, 발급자, 대상, 만료, 스코프는 인가 거부다. 둘 다 도구 처리기에 닿지 않으며, 둘 다 토큰 내용을 감사 증거로 흘려서는 안 된다.

### 대상 재생 따라가 보기 (접근 토큰 권한 제한) (Audience-replay walkthrough (access-token privilege restriction))

서버 A(`notes.example.com`)와 서버 B(`tasks.example.com`)가 같은 인가 서버에 등록돼 있다. 서버 A가 침해당했다. 공격자가 사용자의 메모 토큰을 가져다 서버 B에 재생한다.

서버 B의 검증기는 이렇게 한다.

1. JWT를 디코딩하고, `kid`로 JWKS를 가져와, 서명을 확인한다.
2. 자기 보호된 리소스 메타데이터의 `authorization_servers`와 `iss`를 대조한다. (통과. 같은 IdP다.)
3. `aud == "https://tasks.example.com"`을 확인한다. (실패. 토큰의 `aud`는 `https://notes.example.com`이다.)
4. `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`와 함께 401을 돌려준다.

프로토콜 계층에서 이 공격을 막는 것은 대상 클레임뿐이다. 성능을 이유로 그것을 건너뛰는 것이 가장 흔한 운영 실수다. 검증기는 세션 시작 때만이 아니라 요청마다 돌아야 한다. 명세는 이것을 **접근 토큰 권한 제한**이라고 부른다. MCP 서버는 자기를 대상으로 지목하지 않은 토큰을 `MUST` 거부해야 한다.

> **이름에 관한 참고.** 명세는 *헷갈린 대리인*이라는 말을 비슷하지만 다른 문제에 쓴다. MCP 서버가 제3자 API에 대한 OAuth **프록시** 노릇을 하면서 고정된 클라이언트 ID를 쓰고, 클라이언트별 사용자 동의를 받지 않은 채 토큰을 넘겨주는 경우다. 위의 재생은 대상 결속이 막는다. 헷갈린 대리인은 클라이언트별 동의 **더하기** 들어온 토큰을 상위 API로 절대 넘기지 않는 것으로 막는다(MCP 서버는 상위용 토큰을 따로 `MUST` 받아야 한다).

### 혼동 공격 (서버가 대신해 줄 수 없는 클라이언트 쪽 방어) (Mix-up attacks (a client-side defense the server cannot provide))

클라이언트는 평생에 걸쳐 인가 서버 여럿과 이야기한다. 악의적인 인가 서버는 클라이언트가 정직한 인가 서버의 인가 코드를 공격자의 토큰 엔드포인트에서 교환하게 만들려 할 수 있다. 대상 결속은 여기서 도움이 되지 않는다. 이 공격은 토큰이 생기기 전에 일어나기 때문이다. 방어는 클라이언트에 있다(RFC 9207).

1. 리디렉션 전에, 클라이언트는 검증된 인가 서버 메타데이터에서 기대하는 `issuer`를 기록한다.
2. 인가 응답이 오면, 클라이언트는 돌아온 `iss` 매개변수를 그 기록해 둔 발급자와 비교한다(정규화 없이 단순 문자열 비교). 그 뒤에야 코드를 어딘가로 보낸다.
3. 어긋나거나, 인가 서버가 `authorization_response_iss_parameter_supported`를 알렸는데 `iss`가 없다면 거부하고, `error` 필드를 화면에 보여 주지도 않는다.

PKCE만으로는 혼동 공격을 막지 못한다. 클라이언트가 자기 `code_verifier`를 자신이 끌려간 그 토큰 엔드포인트에 건네주기 때문이다. 그래서 명세는 PKCE 검증자와 `state` 옆에 발급자를 요청마다 기록해 두게 한다.

### 실패 양상 (Failure modes)

- **낡은 JWKS.** 인가 서버가 키를 교체한 뒤 검증기가 멀쩡한 토큰을 거부한다. 해법은 위의 예약 갱신에 캐시 미스 시 다시 가져오기를 더한 방식이다. 갱신 작업 없이 JWKS를 캐시하지 마라.
- **대비책으로 쓴 교체.** 캐시 미스 경로를 다시 가져오기가 아니라 교체 후 발급으로 연결하는 것은 실재하는 버그다. 빠진 `kid`를 만들어 내지도 못하면서, 공격자가 조종하는 `kid` 값을 키 생성 서비스 거부로 바꿔 놓는다. 대비책은 멱등인 `refresh-jwks`여야 한다.
- **`aud` 클레임 누락.** 어떤 IdP는 토큰 요청에 `resource`가 없으면 기본적으로 `aud`를 뺀다. 검증기는 `aud`가 없는 토큰을 거부해야 하며, 없는 것을 아무 값이나 되는 것으로 취급해서는 안 된다.
- **`iss` 검사를 빠뜨린 혼동 공격.** 리디렉션 전에 기록한 발급자와 RFC 9207 `iss` 인가 응답 매개변수를 대조하지 않는 클라이언트는, 정직한 인가 서버의 코드를 공격자의 토큰 엔드포인트에서 교환하도록 끌려갈 수 있다. 이것은 클라이언트 쪽 실패이며 리소스 서버가 대신 막아 줄 수 없다.
- **스코프 상향 경합.** 같은 사용자에 대한 단계 상향 흐름 둘이 동시에 성공해 스코프가 다른 접근 토큰 둘을 만들 수 있다. 검증기는 "사용자의 현재 스코프"를 조회하지 말고 그 요청에 실려 온 토큰을 써야 한다. 조회는 확인과 사용 사이에 틈을 만든다.
- **등록 토큰 도난.** `registration_access_token`이 새어 나가면 공격자가 리디렉션 URI를 고쳐 쓸 수 있다. 저장할 때 해시하고, 갱신할 때마다 클라이언트가 평문을 내밀게 하고, 의심스러우면 교체하라.
- **`iss`를 고정하지 않음.** 아무 `iss`나 받아들이는 검증기는 공격자가 자기 인가 서버를 세우고, 대상 서버용 클라이언트를 등록하고, 토큰을 발급하게 내버려 둔다. 보호된 리소스 메타데이터의 `authorization_servers` 목록이 허용 목록이니 그것을 강제하라.
- **자격 증명이나 토큰 캐시 충돌.** 등록을 리소스만으로 키를 삼는 클라이언트는 한 인가 서버의 신원을 다른 인가 서버에 내밀 수 있다. 접근 토큰을 발급자만으로 키를 삼는 클라이언트는 엉뚱한 대상에 토큰을 재생할 수 있다. 등록은 검증된 발급자로, 접근 토큰은 `(issuer, resource)`로 키를 삼고, 발급자가 바뀌면 다시 등록하라.

```figure
t3-jwks-rotate
```

## 직접 해 보기 (Use It)

`code/main.py`는 표준 라이브러리 파이썬과 `AuthorizationServer`, `ResourceServer`, `Client` 세 역할로 운영 흐름 전체를 훑는다. 흐름은 이렇다.

저장소 루트에서 실행하라.

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

첫 명령은 발급자에 묶인 등록과 토큰 검증 기록을 찍는다. 두 번째는 열여덟
개의 검사가 통과했다고 보고한다. 어느 쪽도 네트워크 리스너를 열거나 자격
증명을 파일에 쓰지 않는다.

1. 인가 서버가 `/.well-known/oauth-authorization-server`에 RFC 8414 메타데이터를 게시한다.
2. MCP 클라이언트가 그 메타데이터 엔드포인트를 부르고 등록 선택지(CIMD를 위한 `client_id_metadata_document_supported`, DCR을 위한 `registration_endpoint`)와 `S256` PKCE 지원을 확인한다.
3. 클라이언트는 발급자 범위의 미리 등록이 있는지 보고, 없으면 자기 HTTPS Client ID Metadata Document로 등록한다. 폐기 예정인 DCR은 따로 시험할 수 있는 호환 수단으로 남는다.
4. 클라이언트는 검증된 발급자를 기록하고, S256 도전 값을 만들고, 일회용 인가 코드와 `iss`를 받고, 그 돌아온 발급자를 검증한 다음, 원래 검증자와 RFC 8707 `resource` 지시자를 실어 코드를 교환한다.
5. MCP 클라이언트가 `Authorization: Bearer ...`를 실어 MCP 서버의 도구를 부른다.
6. MCP 서버가 `validate`를 돌려 JWKS 캐시에서 서명 키를 찾는다.
7. IdP가 키를 교체하고, 예약 갱신이 JWKS를 캐시로 다시 끌어온다.
8. 다음 호출은 재시작 없이 갱신된 키로 검증되고, 겹침 구간 동안에는 이전 토큰도 여전히 검증된다.
9. 다른 MCP 리소스를 향한 대상 재생 시도는 `audience mismatch`와 `resource_metadata` 포인터가 실린 401을 받는다.

여기의 JWT는 공유 비밀을 쓰는 HS256이다(그래야 레슨이 표준 라이브러리만으로 돈다). 실제 운영은 위의 JWKS 방식과 함께 RS256이나 EdDSA를 쓰며, 검증 로직은 그 밖에는 똑같다. IdP와 리소스 서버가 한 프로세스 안에 살기 때문에 `refresh_jwks`는 인가 서버의 키 목록을 곧바로 읽는다. 전선 위에서라면 `jwks_uri`에 대한 HTTP `GET`이다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-mcp-auth.md`를 만든다. MCP 서버 설정과 IdP 역량 목록을 주면, 이 스킬은 세워야 할 인증 표면을 내놓는다. 보호된 리소스 메타데이터, 쓸 등록 경로(CIMD, 미리 등록, 또는 DCR 대비책), JWKS 갱신 일정, 스코프 대응 관계, 그리고 IdP가 RFC 프로필 전체를 지원하지 않을 때 적용할 거부 규칙이다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 흐름을 따라가 보라. 6단계에서 IdP가 키를 교체하고, 예약된 `refresh_jwks`가 게시된 집합을 다시 끌어오며, 겹침 구간의 옛 토큰과 새 토큰이 모두 재시작 없이 검증되는 것을 확인하라.

2. 보호된 리소스 메타데이터의 `authorization_servers` 목록에 IdP를 하나 더 넣으라. 새 IdP가 서명한 토큰을 발급해 검증기가 받아들이는지 확인하라. 목록에 없는 IdP가 서명한 토큰을 발급해 검증기가 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`로 거부하는지 확인하라.

3. 등록기가 요청을 받아들이기 전에 도는 속도 제한 검사를 `register_client`에 추가하라. IP를 키로 삼는 작은 딕셔너리에 출발지 IP별 토큰 버킷을 두면 된다.

4. RFC 7591을 읽고 이 레슨의 `/register` 처리기가 검증하지 않는 필드를 둘 찾아내라. 그 검증을 추가하라. (힌트: `software_statement`와 `redirect_uris`의 URI 스킴.)

5. 인가 서버를 하나 더 추가하라. 클라이언트가 발급자별로 별도의 등록을 저장하고, 첫 발급자의 토큰이나 `client_id`를 다시 쓰기를 거부하는지 확인하라.

6. 서비스 거부를 막는 수정이 제대로 됐는지 증명하라. 무작위 `kid`를 실은 토큰을 검증기에 보내 `refresh_jwks`가 많아야 한 번 돌고 인가 서버의 키 개수가 늘지 않는지 확인하라. 그다음 일부러 대비책을 교체 후 발급으로 바꿔 엉터리 토큰마다 키 개수가 올라가는 것을 보고, 확인한 뒤에는 다시 가져오기로 되돌려라.

7. 폐기 예정인 DCR을 `native` 클라이언트와 `web` 클라이언트 양쪽으로 돌려 보라. HTTP 리디렉션 URI를 쓰는 웹 클라이언트와, 정확한 루프백 리디렉션이 없는 네이티브 클라이언트가 거부되는지 확인하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제로 뜻하는 것 |
|------|----------------|------------------------|
| ASM | "OAuth 메타데이터 문서" | RFC 8414의 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "클라이언트 메타데이터 URL" | Client ID Metadata Document. `client_id`로 쓰는 HTTPS URL이며 인가 서버가 그 JSON을 끌어온다. MCP 2026-07-28의 권장 등록 방식 |
| DCR | "셀프 서비스 클라이언트 등록" | RFC 7591의 `POST /register`. 현행 MCP에서는 폐기 예정이며 호환용으로만 남는다 |
| JWKS | "JWT 검증용 공개 키" | `jwks_uri`에서 가져와 `kid`로 색인하는 JSON Web Key Set |
| Rotate vs refresh | "키 갱신" | *교체*는 인가 서버가 서명 키를 찍고 물리는 일, *갱신*은 리소스 서버가 게시된 집합을 다시 가져오는 일. 리소스 서버는 언제나 갱신만 한다 |
| Resource indicator | "대상 매개변수" | 토큰을 서버 하나에 고정하는 RFC 8707의 `resource` 매개변수 |
| `aud` claim | "대상" | 검증기가 정식 리소스 URL과 비교하는 JWT 클레임 |
| Audience replay | "토큰 재생" | 서버 A용으로 발급된 토큰을 서버 B에 내미는 일. 대상 검증으로 막는다(명세 용어로는 접근 토큰 권한 제한) |
| Confused deputy | "프록시 토큰 오용" | 고정된 클라이언트 ID를 쓰는 MCP 프록시가 클라이언트별 동의 없이 토큰을 넘기는 일. 대상 재생과는 다르다 |
| Mix-up attack | "엉뚱한 토큰 엔드포인트" | 정직한 인가 서버의 코드를 공격자의 엔드포인트에서 교환하도록 클라이언트를 끌고 가는 일. RFC 9207 `iss`로 클라이언트 쪽에서 막는다 |
| `iss` allow-list | "신뢰하는 인가 서버" | 보호된 리소스 메타데이터의 `authorization_servers`가 지목하는 집합 |
| `resource_metadata` | "PRM 문서 위치" | 401이나 403에서 RFC 9728 메타데이터 URL을 알려 주는 `WWW-Authenticate` 매개변수 |
| Public client | "네이티브나 브라우저 클라이언트" | `client_secret`이 없는 OAuth 클라이언트. PKCE가 그것을 대신한다 |
| `WWW-Authenticate` | "401/403 응답 헤더" | 클라이언트의 복구를 이끄는 `Bearer error=...` 지시를 싣는다 |

## 더 읽을거리 (Further Reading)

- [MCP authorization specification (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - 현행 MCP 인가 프로필
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD, 발급자 검증, DCR 폐기, 발급자별 자격 증명 변경
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) CIMD
- [RFC 8414: OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) 탐색 계약
- [RFC 7591: OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) DCR (대비책 경로)
- [RFC 7636: Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) 공개 클라이언트의 소지 증명
- [RFC 8707: Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) 대상 고정
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) 리소스 서버 탐색
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) 혼동 공격을 막는 `iss` 매개변수
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
