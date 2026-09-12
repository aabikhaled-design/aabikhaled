# 무상태 MCP 게이트웨이와 레지스트리 승인 (Stateless MCP Gateways and Registry Admission)

> 게이트웨이는 모든 경로를 명시적으로 만들어야 한다. 2026-07-28 프로토콜은 전송 세션 없이도 메서드, 이름, 버전, 역량, 신원, 캐시, 추적의 경계를 게이트웨이에 쥐여 준다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- 세션 고정 없이 여러 MCP 서버를 2026-07-28 엔드포인트 하나 뒤에 모은다.
- 정책을 적용하거나 전달하기 전에 요청마다 실린 메타데이터와 라우팅 헤더를 검증한다.
- 안정적인 이름 공간, 결정적인 순서, 서술자 고정, RBAC, 비공개 캐싱으로 도구를 병합한다.
- 레지스트리 기록을 승인 정책이 여전히 필요한 탐색 증거로 다룬다.
- 요청 범위 SSE, `subscriptions/listen`, MRTR 재시도, Tasks 확장 호출을 각각 올바르게 라우팅한다.
- 레거시 악수와 세션 지원을 현대 경로에서 분리한다.

## 문제 (The Problem)

클라이언트 하나를 서버 하나에 바로 잇는 것은 간단하다. 규모가 커지면 더 어려운 질문에 일관된 답이 필요해진다.

- 어떤 서버가 허용되는가?
- 어떤 주체가 각 도구를 보고 호출할 수 있는가?
- 백엔드 둘이 같은 이름을 내놓으면 어떻게 되는가?
- 서술자 변경은 어떻게 검토하는가?
- 속도 제한과 감사 이벤트는 어디에 적용하는가?
- 다음 요청을 아무 인스턴스나 처리할 수 있는가?

게이트웨이는 클라이언트와 백엔드 MCP 서버 사이에 앉는다. MCP 엔드포인트 하나를 내놓고, 전체에 걸친 정책을 적용하고, 승인된 요청을 넘겨준다.

옛 게이트웨이 설계는 클라이언트 세션 하나를 백엔드 세션 여럿으로 다중화하고 `Mcp-Session-Id`를 다시 쓰는 일이 많았다. 그것은 레거시 호환 설계다. 2026-07-28 코어에는 프로토콜 세션이 없다.

## 개념 (The Concept)

### 현대 게이트웨이 경로 (The modern gateway path)

요청마다 이렇게 한다.

1. 전송 인가에서 주체를 인증한다.
2. `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, `params._meta`를 검증한다.
3. 주체, 리소스, 메서드, 도구, 인자를 인가한다.
4. 서술자, 레지스트리, 속도, 데이터 정책을 적용한다.
5. 고른 백엔드를 위해 자기완결적인 새 요청을 만든다.
6. 백엔드 결과를 검증하고 게이트웨이 결과를 돌려준다.
7. 비밀 값을 남기지 않고 감사 이벤트를 기록한다.

어느 단계에도 감춰진 프로토콜 세션이 필요하지 않다. 애플리케이션 상태는 데이터베이스, 명시적 핸들, Tasks, 무결성이 보호되는 MRTR 상태에 여전히 존재할 수 있다.

### 실행 시점 정책이 게이트웨이의 주된 판단이다 (Runtime policy is the primary gateway decision)

승인은 어떤 백엔드 버전이 게이트웨이에 들어올 수 있는지를 정한다. 살아 있는 호출을 인가하지는 않는다. 게이트웨이는 요청마다 인증된 주체, 발급자와 리소스, 테넌트, 맞춰진 메서드와 이름, 정규화된 인자, 승인된 서술자 고정 값, 현재 백엔드 상태, 역량 교집합, 데이터 분류, 속도 상태, 동작에 묶인 승인을 모두 다시 계산해 정책을 만든다.

이 순서가 중요하다. 사용자의 역할이 회수됐는데도 레지스트리 기록은 활성 상태로 남아 있을 수 있다. 목적지 인자가 테넌트 경계를 넘는데도 서술자는 고정된 채로 남아 있을 수 있다. 사고 대응 정책이 상태를 바꾸는 호출을 격리하는데도 백엔드는 승인된 채로 남아 있을 수 있다. 그래서 실행 시점 정책이 허용과 거부를 정하는 주된 판단이 되고, 레지스트리와 서술자 증거는 그 입력이 된다.

허용 판단을 연결이나 사라진 세션 식별자 아래에 캐시하지 마라. 정책을 조회할 수 없다면 작업 종류별로 선언해 둔 실패 정책을 따르라. 상태 변경과 민감한 읽기는 막는 쪽으로 실패하는 것이 안전한 기본값이고, 명시적으로 승인된 공개 읽기 경로는 위험 모형이 허락할 때만 수명이 짧은 마지막 정책을 쓸 수 있다. 어떤 정책 버전과 어떤 실패 경로가 그 판단을 내렸는지 기록하고, 백엔드 결과를 돌려주기 전에 검증하라.

### POST 엔드포인트 하나 (One POST endpoint)

현대 Streamable HTTP는 JSON-RPC 메시지를 모두 POST로 보낸다.

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

게이트웨이는 그 POST에 JSON이나 요청 범위 SSE를 돌려줄 수 있다. 현대 요청에서 GET과 DELETE는 405를 돌려준다. `Mcp-Session-Id`와 `Last-Event-ID`는 권한도, 고정 배정도, 재생 동작도 만들지 않는다.

헤더 값과 본문 값은 서로 맞아야 한다. 백엔드를 찾기 전에 어긋난 요청을 `-32020`으로 거부하라. 그러면 부하 분산기와 게이트웨이, 속도 제한기가 본문을 전부 파싱하지 않고도 라우팅하면서 종단 간 무결성을 지킬 수 있다.

정확히 한 순서로 검증하라. JSON-RPC와 메타데이터 타입, 헤더와 본문의 일치, 그다음 맞춰진 버전의 지원 여부다. 어긋나면 `-32020`과 함께 HTTP 400을 돌려준다. 헤더와 본문이 지원하지 않는 버전에서 일치하면 `-32022`와 함께 HTTP 400을 돌려주고 `data`는 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`로 채운다. 알 수 없는 메서드에는 `-32601`과 함께 HTTP 404를 돌려준다.

`ProtocolError`는 선택적인 `data`를 싣고, 게이트웨이는 그것을 JSON-RPC 오류 객체로 직렬화한다. 알림에는 `id`가 없으므로 JSON-RPC 성공도 오류도 받지 않는다. 받아들인 HTTP 알림은 빈 본문과 함께 202를 돌려준다.

### 모든 계층에서 탐색을 구현하라 (Implement discovery at every layer)

게이트웨이는 클라이언트를 위해 `server/discover`를 구현한다. 프로토콜 버전과 역량, 확장을 알기 위해 각 백엔드도 탐색한다.

게이트웨이 결과의 예는 이렇다.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

게이트웨이가 종단 간으로 지킬 수 있는 역량 교집합만 알리라. 백엔드에 있는 기능이라고 해서 내놓아도 안전한 것은 아니다. 백엔드 경로가 없는 게이트웨이 기능을 알리는 것도 쓸모없다.

`serverInfo`는 스스로 밝힌 표시용 진단 데이터다. 레지스트리나 발행자의 증거로 쓰지 마라.

### 요청마다 실리는 클라이언트 역량 (Per-request client capabilities)

넘겨주는 요청마다 현재의 `_meta` 봉투가 필요하다.

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

바깥 클라이언트의 역량을 백엔드로 그대로 베껴 보내지 마라. 백엔드 입장에서 클라이언트는 게이트웨이다. 게이트웨이가 올바르게 중재할 기능만 알리라.

### 결정적인 이름 공간 (Deterministic namespacing)

백엔드 도구를 안정적인 공개 이름 아래로 병합하라.

```text
notes.search
notes.create
issues.list
issues.open
```

공개 이름에서 백엔드와 원래 도구 이름으로 가는 대응 관계를 유지하라. 충돌했을 때 첫 번째나 마지막을 고르는 식으로 정하지 마라. 공개 이름은 승인과 감사 계약의 일부이므로, 그것을 바꾸는 일은 이전 작업이 된다.

`tools/list`는 결정적이어야 한다. 주체에 따라 가시성이 달라지면 `cacheScope: private`를 돌려주라. 상한이 있는 `ttlMs`는 사용자별 목록이 인가 맥락을 넘어 새어 나가게 하지 않으면서 백엔드 탐색 부담을 줄여 준다.

밖으로 내놓는 모든 도구 서술자에는 안정적인 이름과 설명, 루트가 객체인 `inputSchema`가 들어간다. 이름 공간을 붙인다고 필수 서술자 필드를 뺄 수 있는 것은 아니다. 완결된 목록 결과에는 `resultType`, 서버 신원 메타데이터, 캐시 힌트도 들어간다.

### 승인된 서술자를 고정하라 (Pin approved descriptors)

승인 시점에 서술자 전체를 정규화해 그 요약값을 수식된 공개 이름 아래에 저장하라. 목록을 만들 때와 호출할 때, 살아 있는 서술자를 승인된 요약값과 비교하라.

바뀌었다면 이렇게 한다.

- `tools/list`에서 뺀다.
- 직접 호출을 거부한다.
- 감사 이벤트를 남긴다.
- 고정 값을 갱신하기 전에 정책이나 사람의 재승인을 요구한다.

게이트웨이는 쓸모 있는 중앙 집행 지점이지만, 처음 본 서술자를 안전한 것으로 만들어 주지는 않는다. 최초 검토는 여전히 필요하다.

### 레지스트리는 탐색을 돕지, 판단하지 않는다 (Registries help discover, not decide)

레지스트리의 `server.json`은 게시 메타데이터를 준다. 패키지에 기반한 기록은 이렇게 생겼다.

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

게시 메타데이터가 게이트웨이의 보안 판단을 담고 있지는 않다. 검증된 발행자와 출처 증거는 별도의 승인 상태에 두라.

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

게이트웨이는 `server.json`의 형태를 확인하고 그것을 바깥의 그 상태와 이어 붙인다. 그래도 게이트웨이에는 승인 정책이 따로 필요하다.

승인한 백엔드마다 다음을 기록하라.

- 정확한 레지스트리와 기록 식별자.
- 검증된 발행자 이름 공간이나 도메인 증거.
- 허용된 전송과 엔드포인트.
- 고정된 버전 또는 승인된 업그레이드 정책.
- 산출물이나 서술자 요약값.
- 인가 발급자와 리소스.
- 검토자, 승인 시각, 만료.

표시 이름이 낯익은 제품과 비슷하다는 이유로 서버를 받아들이지 마라. 레지스트리에 있다는 사실을 운영 보안 검토로 여기지 마라. 공개 레지스트리에 전혀 나타나지 않는 사설 서버도 같은 증거 스키마를 거쳐 승인할 수 있다.

이 레슨은 게이트웨이 이음매를 구현한다. 백엔드가 라우팅 대상이 되기 전에 게시 증거를 지역 승인에 이어 붙이는 부분이다. [레슨 30: MCP 레지스트리 공급망, 승인, 표류, 롤백](../../30-mcp-registry-supply-chain-and-drift/docs/ko.md)은 정확한 이름 공간 증명, 산출물 출처, 불변 고정, 살아 있는 서술자 표류, 레지스트리 상태 조정, 변조가 드러나는 승인 장부, 증거에 뒷받침되는 롤백까지 아우르는 제어 평면 전체를 세운다. 그 공급망 상태는 위의 요청별 실행 시점 판단과 분리해 두라.

### 자격 증명 중재 (Credential mediation)

게이트웨이는 자신의 호출자를 인증하고, 백엔드에는 따로 인증한다. 백엔드 자격 증명은 클라이언트에게 절대 가지 않는다.

이 결속을 뚜렷하게 두라.

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

바깥 게이트웨이 토큰을 백엔드에 넘기지 마라. 백엔드 토큰을 다른 발급자나 다른 리소스에서 다시 쓰지 마라. 도구가 최종 사용자를 대신해 동작한다면, 공유 서비스 자격 증명으로 사용자 행세를 하지 말고 설계된 교환이나 클레임 모형으로 그 위임을 보존하라.

### 세션 없는 속도 제한 (Rate limits without sessions)

인증된 주체, 발급자, 리소스, 공개 도구, 비용 등급, 시간 구간을 키로 삼아 제한하라. 세션 id는 존재하지 않고, 설령 있더라도 쉽게 갈아 낄 수 있다.

값비싼 작업을 소비하기 전에 값싼 검증을 먼저 하라. 거부된 호출을 남용 한도에 셀지, 사업 할당량에 셀지, 둘 다에 셀지 정하라.

### 판단의 사슬을 감사하라 (Audit the decision chain)

호출을 재구성할 수 있을 만큼 기록하라.

- 요청과 추적 식별자.
- 인증된 주체와 발급자.
- 공개 도구와 백엔드 경로.
- 서술자 고정 버전.
- 정책 판단과 그 이유.
- 지연 시간과 결과 등급.
- 해당한다면 MRTR 라운드나 태스크 식별자.

소지자 토큰, 인가 코드, 갱신 토큰, 날것의 비밀 값, 굳이 필요 없는 민감한 인자는 가리라.

### 요청 범위 SSE (Request-scoped SSE)

작업이 그 요청 하나 동안 흘러나온다면 평범한 POST가 요청 범위 SSE를 돌려줄 수 있다. 응답 스트림을 닫으면 처리 중인 그 현대 HTTP 요청이 취소된다.

별도의 GET 스트림을 만들지 말고 Last-Event-ID 재생을 약속하지 마라. 그것들은 옛 전송의 전제다.

### 오래 유지되는 변경 알림 (Long-lived change notifications)

목록 변경과 리소스 변경 알림을 받으려면, 현행 클라이언트는 POST로 `subscriptions/listen`을 보내고 SSE 응답을 받는다. 알림 필터는 평평한 필드 `toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`를 정확히 그 이름으로 쓴다.

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

첫 이벤트가 지원되는 부분집합을 확인해 준다. 그 구독 식별자는 스트림을 연 요청의 JSON-RPC id다.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

그다음 게이트웨이는 확인된 변경 종류만 넘겨준다. 그 스트림의 모든 알림은 `params._meta`에 같은 `io.modelcontextprotocol/subscriptionId`를 싣는다. 자동 재생도, 자동 재구독도 없다. 다시 연결하면 클라이언트가 구독을 새로 열고 자신이 기대는 목록을 새로 고친다. 서버가 정상적으로 닫을 때는 같은 구독 id를 단 최종 complete 결과를 돌려준다.

현대 경로는 `resources/subscribe`와 `resources/unsubscribe`, 그리고 요청 없이 열리는 독립 GET 스트리밍을 대체한다. 그것들은 버전으로 걸러진 옛 경로에만 남겨 두라.

### 게이트웨이를 거치는 MRTR (MRTR through a gateway)

백엔드가 `resultType: input_required`를 돌려주면, 게이트웨이는 바깥 클라이언트가 필요한 입력 요청을 지원할 때만 그 결과를 넘겨줄 수 있다. 게이트웨이가 상호작용을 의도적으로 끝내고 다시 내는 경우가 아니라면 `requestState`를 바이트 단위로 그대로 보존하라.

클라이언트는 새 JSON-RPC id와 `inputResponses`를 실어 원래 공개 도구를 재시도한다. 게이트웨이는 그 재시도를 다시 인가하고, 같은 공개 경로인지 확인한 다음, 새 백엔드 요청을 만들어 넘긴다. 앞선 라운드가 무제한 승인을 줬다고 가정해서는 안 된다.

### Tasks 확장 라우팅 (Tasks extension routing)

Tasks는 `io.modelcontextprotocol/tasks`로 식별되는 공식 확장이다. 코어 세션을 대신하는 물건이 아니다.

클라이언트는 요청마다 실리는 클라이언트 역량 안에서 확장을 선언하고, 게이트웨이는 생애주기를 종단 간으로 지킬 수 있을 때만 탐색에서 그것을 알린다. 지원되는 `tools/call`에 대해, 평범한 결과를 돌려줄지 `resultType: task`를 돌려줄지는 백엔드 혼자 정한다. 태스크 결과는 `taskId`, `status`, 시각, `ttlMs`, 그리고 선택적인 `pollIntervalMs`를 결과에 곧바로 싣는다. 그 결과를 보내기 전에 태스크는 이미 지속 저장소에서 읽을 수 있어야 한다.

게이트웨이는 불투명한 태스크 식별자에 대해 인증된 주체와 백엔드 경로를 기록한다. 이어지는 `tasks/get`, `tasks/update`, `tasks/cancel` 호출은 `params.taskId`를 `Mcp-Name`으로 쓰며, 그것이 중간 구성 요소에 라우팅 키가 되어 준다. `tasks/get`은 현재 태스크 상태와 함께 `resultType: complete`를 돌려주고, 종료 상태에서는 최종 결과나 프로토콜 오류를 그 안에 담아 준다. `tasks/update`는 남아 있는 태스크 입력에 키를 맞춘 `inputResponses`를 보내고 빈 complete 확인 응답을 받는다. `tasks/cancel`은 빈 complete 확인 응답을 받는 협조적 의사 표시이지, 작업이 멈춘다는 보장이 아니다.

`tasks/list`나 `tasks/result`를 새로 구현하지 마라. 그것들은 옛 실험 모형에 속한다. 입력이 필요한 태스크는 `tasks/get`으로 완결된 내장 요청을 보여 주고, 클라이언트는 원래 도구 호출을 재시도하는 대신 `tasks/update`로 답한다. 클라이언트는 여전히 권장 간격으로 상태를 묻고, 태스크 생성은 서버가 주도하는 채로 남는다.

지속되는 태스크 경로 상태는 태스크 핸들을 키로 삼는 애플리케이션 데이터이지 프로토콜 세션이 아니다.

### 호환성 경계 (Compatibility boundary)

게이트웨이가 옛 클라이언트나 옛 백엔드를 상대해야 한다면 이렇게 하라.

- 시대를 명시적으로 식별한다.
- 초기화, 전송 세션, GET 스트림, 리소스 구독, 옛 태스크 어휘를 레거시 어댑터 안에 가둔다.
- 레거시 세션 id가 현대 라우팅이나 인가로 새어 나가게 두지 않는다.
- 조용한 등급 낮추기 대신 제한된 탐색 탐지와 명시적인 대비책 정책을 쓴다.

```figure
t3-gateway-funnel
```

## 만들어 보기 (Build It)

`code/main.py`는 프로세스 안에서 도는 프로토콜 게이트웨이와 백엔드 서버 두 개를 구현한다. 각 백엔드는 현행 프로토콜 요청을 새로 받는다. 게이트웨이는 탐색, 사용자별로 걸러진 결정적 `tools/list`, 이름 공간 라우팅, 레지스트리 `server.json`과 외부 승인 상태, 서술자 고정, RBAC, 주체를 키로 삼은 속도 제한, 감사 판단, 그리고 모형화한 `subscriptions/listen` SSE 확인 응답을 제공한다.

이 모형은 이미 파싱된 요청 본문과 라우팅 헤더, 그리고 인증된 소지자 신원을 받는다. 완전한 HTTP 어댑터가 아니며 `Content-Type`이나 `Accept` 계약 전체를 파싱하지 않는다. `Content-Type: application/json`과 `application/json`, `text/event-stream`을 모두 담은 `Accept` 값을 요구하는 레슨 09의 Streamable HTTP 어댑터에 붙여 쓰라.

실행은 이렇게 한다.

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

예제는 바깥 요청 id와 새로 만든 백엔드 요청 id를 함께 찍어서 무상태 중계를 눈에 보이게 한다.

## 직접 해 보기 (Use It)

프로세스 안의 백엔드 객체를 진짜 현행 프로토콜 클라이언트로 바꾸라. 이음매는 그대로 두라.

- 연결 전에 승인 기록.
- 역량을 내놓기 전에 백엔드 탐색.
- 인가 전에 수식된 공개 이름.
- 목록이나 호출 전에 서술자 고정.
- 넘겨주기 전에 요청마다 새로 만든 메타데이터.
- 돌려주기 전에 결과 검증.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-gateway-bootstrap.md`를 만든다. 진입, 탐색, 승인, 이름 공간, 인가, 캐싱, 스트리밍, 구독, MRTR, Tasks, 관측 가능성, 레거시 격리를 아우르는 현대 게이트웨이 설계를 만들어 준다.

## 연습 문제 (Exercises)

1. 바깥 요청과 넘겨주는 요청의 메타데이터에 추적 맥락을 넣고 그 상관 관계를 감사 이벤트에 기록하라.
2. Tasks를 지원하는 백엔드를 추가하고 `Mcp-Name`의 태스크 id로 `tasks/get`을 라우팅하라.
3. 백엔드 서술자 하나를 바꾸고 탐색과 직접 호출이 둘 다 막히는지 증명하라.
4. 주체별로 달라지는 서버 역량을 추가하고, 탐색이 왜 비공개로 캐시돼야 하는지 설명하라.
5. 현대 `Gateway` 클래스에 레거시 상태를 하나도 더하지 않고 레거시 어댑터 인터페이스를 작성하라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| MCP gateway | 클라이언트와 백엔드 MCP 서버 사이에 놓인 정책 및 라우팅 서버 |
| Admission record | 백엔드 하나를 게이트웨이에 들이는 증거와 정책 판단 |
| Qualified tool name | `notes.search` 같은 안정적인 공개 경로 |
| Descriptor pin | 탐색과 디스패치 때 대조하는 승인된 요약값 |
| Private cache scope | 인가 맥락 하나로 제한된 캐시 결과 |
| Request-scoped SSE | POST 요청 하나에 붙은 스트리밍 응답 |
| `subscriptions/listen` | 고른 장기 변경 알림을 위해 클라이언트가 여는 SSE 스트림 |
| Task route | 불투명한 태스크 id에서 그 백엔드로 가는 애플리케이션 대응 관계 |
| Legacy adapter | 옛 악수와 세션 동작을 위해 버전으로 걸러 둔 명시적 경계 |

## 더 읽을거리 (Further Reading)

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
