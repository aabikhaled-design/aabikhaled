# MCP Tasks 확장: 무상태 코어 위의 지속되는 작업 (MCP Tasks Extension: Durable Work on a Stateless Core)

> 무상태 MCP라고 해서 모든 작업이 요청 하나 안에서 끝나야 한다는 뜻은 아니다. 공식 Tasks 확장은 오래 걸리는 작업에 명시적이고 지속되는 핸들을 준다. 서버는 `tools/call`에서 그 핸들을 돌려줄 수 있고, 어느 인스턴스든 `tasks/get`에 답할 수 있으며, 클라이언트 입력은 프로토콜 세션을 되살리지 않고 `tasks/update`로 도착한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## 학습 목표 (Learning Objectives)

- 무상태 프로토콜 전송과 지속되는 애플리케이션 작업 상태를 구별한다.
- 요청마다 실리는 역량과 `server/discover`에서 `io.modelcontextprotocol/tasks` 확장을 협상한다.
- 지속 저장이 끝난 뒤에만 서버가 주도하는 `resultType: "task"` `CreateTaskResult`를 돌려준다.
- `tasks/get`으로 상태를 물어보고, `tasks/update`로 작업 입력을 채우고, `tasks/cancel`로 협조적 취소를 요청한다.
- 옛 `tasks/status`, `tasks/result`, `tasks/list`에 대한 가정을 걷어낸다.
- POST 응답 SSE 스트림 위의 `subscriptions/listen`으로 선택적 작업 알림을 구독한다.
- 작업 만료, 재시작 복구, 입력 키 중복 제거, 실행 오류를 올바르게 모형화한다.

## Tasks가 확장인 이유 (Why Tasks Are an Extension)

Tasks는 2025-11-25에 실험적 코어 기능으로 처음 등장했다. 2026년 7월 재설계는 이것을 공식 `io.modelcontextprotocol/tasks` 확장으로 옮겨, 모두를 위한 코어 프로토콜을 부풀리지 않고도 클라이언트와 서버가 추가 생애주기를 선택해 쓸 수 있게 했다.

확장 명세는 현재 Tasks의 공식 거처이기는 하지만 여전히 초안 표면이다. SDK가 지원하는 확장 버전을 고정하고, 적합성 시나리오를 돌리고, 전선 어댑터를 워커 및 저장소 영역과 분리해 두라.

작업이 다음 성질을 하나 이상 가질 때 태스크를 쓰라.

- 평범한 요청 타임아웃보다 오래 갈 수 있다.
- 워커 큐나 외부 잡 시스템이 이미 실행을 맡고 있다.
- 클라이언트가 자기 자신이 재시작한 뒤에 복구해야 한다.
- 실행 도중 사용자나 모델의 입력을 받으려고 멈춘다.
- 취소와 지속되는 결과 조회가 제품 요구사항이다.

값싸고 결정적인 조회에 태스크를 만들지 마라. 핸들, 영속성, 상태 조회, 만료, 취소는 모두 실재하는 복잡성이다.

## 무상태 코어, 상태를 가진 애플리케이션 (Stateless Core, Stateful Application)

MCP 2026-07-28은 `initialize`, `notifications/initialized`, 프로토콜 세션, `Mcp-Session-Id`를 걷어냈다. 그렇다고 상태를 가진 제품이 금지되는 것은 아니다.

태스크 id는 명시적인 애플리케이션 상태다.

- 서버는 그것을 돌려주기 전에 저장한다.
- 클라이언트는 그것을 보관했다가 재시작한 뒤에 다시 물어볼 수 있다.
- 같은 지속 저장소를 바라보는 복제본이라면 어디로든 id를 라우팅할 수 있다.
- 태스크 메서드마다 인가를 확인한다.
- 만료와 삭제는 전송의 수명이 아니라 태스크 필드가 정의한다.

이것은 연결에 붙은 숨은 상태와는 운영 측면에서 다르다.

네 가지 수명을 따로 두라.

| 상태 | 수명 | 있어야 할 곳 |
|---|---|---|
| 프로토콜 메타데이터 | 요청 하나 | `params._meta`, 호출마다 다시 검증한다 |
| 전송 작업 | stdio 요청 하나 또는 HTTP 응답 하나 | 기한이 정해진 처리 중 조정자 |
| MRTR 연속 처리 | 재시도 흐름 하나 | 무결성이 보호되는 `requestState`, 필요하면 재생 방지 통제까지 |
| 지속되는 태스크 | 요청, 복제본, 재시작, 재연결을 넘어 | 인가된 `taskId`를 키로 쓰는 공유 애플리케이션 저장소 |

태스크 레코드를 프로세스 메모리로 옮긴다고 MCP가 상태를 갖게 되지는 않는다. 애플리케이션이 불안정해질 뿐이다. 프로토콜은 여전히 무상태지만, 나중에 다른 복제본으로 라우팅된 `tasks/get`은 그 레코드를 되찾을 수 없다. 핸들을 돌려주기 전에 저장하고, 모든 태스크 메서드가 테넌트와 주체 검사를 거쳐 같은 공유 레코드를 찾아가게 하라.

## 역량 협상 (Capability Negotiation)

클라이언트는 해당하는 모든 요청에서 지원을 알린다.

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

서버는 `server/discover`에서 정확한 `supportedVersions`, 역량, `ttlMs`, `cacheScope`를 돌려주고 역량 아래에 같은 확장을 담는다. 도구를 알리므로 필수인 `tools/list`도 구현한다. 그 결과는 결정적인 `generate_report` 서술자, 유효한 객체 `inputSchema`, `resultType: "complete"`, 서버 신원 메타데이터, 공개 캐시 힌트를 돌려준다.

확장을 선언하지 않은 클라이언트가 태스크 메서드를 부르면 `data.requiredCapabilities`를 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`로 채운 `-32021`, 즉 필수 클라이언트 역량 누락을 돌려준다. 지원하지 않는 프로토콜 문자열에는 정확한 `supported`, `requested` 데이터와 함께 `-32022`를 돌려주고, 버전이 없거나 문자열이 아니면 `-32602`를 돌려준다.

JSON-RPC `id`가 없는 봉투는 알림이다. 수신 측은 그것을 처리해도 되지만 JSON-RPC 결과도 오류도 내보내지 않는다. Streamable HTTP 어댑터는 받아들인 알림에 본문 없이 `202 Accepted`를 돌려준다.

현재는 `tools/call`만 태스크로 확장된 실행을 지원한다. 앞으로 다른 요청 종류가 생겨도 저장소를 다시 쓰지 않아도 되게 내부 추상화를 설계하라.

## 서버가 주도하는 태스크 생성 (Server-Directed Task Creation)

클라이언트가 쓰던 옛 플래그 `params._meta.task.required`는 사라졌다. 클라이언트는 확장 지원을 선언하고, 특정 `tools/call`을 태스크로 만들지는 서버가 정한다.

요청은 이렇다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

응답은 이렇다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

서버는 그 id로 `tasks/get`이 해결될 수 있게 되기 전에는 이 핸들을 돌려줘서는 안 된다. 최종적 일관성을 가진 저장소라면 읽기가 보이게 될 때까지 기다렸다가 답하라. 그러지 않으면 클라이언트가 멀쩡해 보이는 id를 받고도 곧바로 "찾을 수 없음"을 만난다.

태스크 응답은 클라이언트가 태스크 모드를 요청하지 않았다는 뜻에서 요청되지 않은 응답이다. 협상되지 않은 응답은 아니다. 그 요청 역시 확장을 알리고 있어야 한다.

## 태스크의 형태 (The Task Shape)

모든 태스크는 다음을 싣는다.

- `taskId`: 서버가 만든 안정적인 식별자,
- `status`: `working`, `input_required`, `completed`, `cancelled`, `failed` 중 하나,
- `createdAt`과 `lastUpdatedAt`: ISO 8601 시각,
- `ttlMs`: 생성 시점부터의 만료 기간, 알릴 한계가 없으면 `null`,
- 선택적인 `pollIntervalMs`: 서버가 지금 권하는 최소 조회 간격,
- 선택적인 `statusMessage`: 사용자나 모델에게 보여 줄 맥락.

상태별 필드는 해당할 때만 나타난다.

- `input_required`에는 `inputRequests`가 들어간다.
- `completed`에는 원래 요청의 `result` 형태가 들어간다.
- `failed`에는 JSON-RPC `error` 객체가 들어간다.

클라이언트는 `pollIntervalMs`를 존중해야 한다. 서버는 더 공격적인 조회에 속도 제한을 걸 수 있고, 태스크가 사는 동안 간격을 바꿀 수도 있다.

## `tasks/get`으로 상태 물어보기 (Poll with `tasks/get`)

클라이언트는 현재 스냅샷을 요청한다.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 자체는 끝났으므로 그 결과는 언제나 `resultType: "complete"`다. 안에 담긴 태스크는 여전히 `status: "working"`이나 `status: "input_required"`일 수 있다.

이 구분이 흔한 파서 버그를 막아 준다.

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

`tasks/result` 호출은 없다. 태스크가 끝나면 다음 `tasks/get` 응답이 원래 `CallToolResult`를 `result` 아래에 그대로 담아 준다.

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

바깥 `resultType`은 `tasks/get` RPC가 끝났다고 말한다. 안쪽 `result.resultType`은 원래 도구 호출이 끝났다고 말한다. 이 안쪽 구분자는 필수다. 안쪽 `CallToolResult`도 자기 자신의 `io.modelcontextprotocol/serverInfo`를 싣는 편이 좋다. 이 레슨은 타입 없는 페이로드를 저장하는 대신 그것을 넣어 둔다.

`tasks/list`는 없다. 세션이 없는 서버는 어떤 태스크가 한 연결 범위의 목록에 들어가야 하는지 안전하게 추론할 수 없다. 이력이 필요한 애플리케이션은 명시적인 필터와 소유권 규칙을 갖춘 인가된 도메인 도구를 따로 내놓아야 한다.

## 태스크 실행 도중의 입력 (Input During Task Execution)

태스크 입력과 코어 MRTR은 닮아 보이지만 서로 다른 연속 처리를 쓴다.

### 태스크 생성 전에 입력이 필요할 때 (Input needed before task creation)

원래 `tools/call`에서 코어의 `resultType: "input_required"`를 돌려준다. 클라이언트가 그것을 채우고 그 원래 호출을 재시도한다. 동기적인 MRTR 라운드가 끝난 뒤에야 태스크를 만든다.

### 태스크 생성 후에 입력이 필요할 때 (Input needed after task creation)

태스크를 `input_required`로 바꾼다. `tasks/get`이 남아 있는 `inputRequests`를 보여 주고, 클라이언트는 `tasks/update`로 응답을 보낸다. 클라이언트는 원래 `tools/call`을 재시도하지 않는다.

스냅샷은 이렇다.

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

갱신은 이렇다.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

성공 응답은 빈 확인 응답에 `resultType: "complete"`를 붙인 것이다. 상태 변화는 최종적으로만 일관될 수 있으므로 클라이언트는 계속 상태를 묻거나 구독을 유지한다.

`inputRequests`의 각 키는 태스크가 사는 내내 유일해야 한다. `tasks/get` 스냅샷을 거듭 받으면 같은 미해결 키가 계속 보일 수 있다. 클라이언트는 UI에서 중복을 걸러내고, 서버는 알 수 없거나 대체됐거나 이미 채워진 키에 대한 응답을 무시한다. 일부만 갱신하면 필수 키가 전부 채워질 때까지 태스크가 `input_required`에 남을 수 있다.

## 취소는 협조적이다 (Cancellation Is Cooperative)

`tasks/cancel`은 의사를 전달하고 빈 complete 확인 응답을 돌려준다. 그 확인 응답이 워커가 멈췄음을 보장하지는 않는다. 작업이 먼저 끝날 수도, 취소를 무시할 수도, 나중에 상태가 바뀔 수도 있다.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

세 태스크 메서드 모두에서 `Mcp-Name`은 `params.taskId`를 복제한다. JSON-RPC 메서드 이름을 되풀이하지 않는다. `code/main.py`는 이 규칙을 `make_http_request` 한 곳에 모아 둔다.

이 레슨의 워커는 취소를 곧바로 따르므로 같은 호출을 거듭해도 결과가 같다. 실제 운영 클라이언트는 그래도 취소를 협조적인 것으로 다뤄야 하고, 확인 응답으로부터 최종 태스크 상태를 추론해서는 안 된다.

태스크를 취소하는 데 `notifications/cancelled`를 쓰지 마라. 그 알림은 지속되는 Tasks가 아니라 요청 취소에 속한다.

이 구분은 라우팅 경계에서 중요해진다. 요청 취소는 처리 중인 JSON-RPC 작업 하나나 그 요청 범위의 HTTP 응답을 겨냥한다. `tools/call`이 이미 `resultType: "task"`를 돌려준 뒤라면 그 요청은 끝난 것이고, 전송을 닫아도 지속되는 잡을 지칭하거나 멈출 수 없다. `tasks/cancel`은 새로 인가되는 RPC다. `params.taskId`를 싣고, 그 id를 `Mcp-Name`에 복제하고, 태스크를 소유한 백엔드를 찾아가고, 협조적 취소 의사를 기록하고, 워커가 멈췄다고 주장하지 않은 채 확인 응답을 돌려준다.

그래서 게이트웨이는 요청 조정자와 태스크 경로를 서로 다른 표에 두어야 한다. 요청 표는 응답이 끝나면 사라져도 된다. 태스크 경로는 종료 상태와 보존 만료 때까지 살아 있어야 한다. [레슨 29: MCP 신뢰성, 취소, 흐름 제어](../../29-mcp-reliability-cancellation-and-flow-control/docs/ko.md)가 두 경로 모두에 대해 경합, 타임아웃, 멱등성, 배압, 재시도 규칙을 세운다.

## 선택적 알림 (Optional Notifications)

상태를 주기적으로 묻는 것이 기본이다. 밀어 주는 갱신을 원하는 클라이언트는 태스크 id를 실어 `subscriptions/listen`을 보낸다. Streamable HTTP에서는 응답이 요청 범위 SSE 스트림이 되는 POST다. 독립 GET 이벤트 스트림도, 살려 둘 프로토콜 세션도 없다.

서버는 받아들인 id들을 `notifications/subscriptions/acknowledged`로 확인해 주고, 그 뒤에 `notifications/tasks`로 전체 스냅샷을 보낼 수 있다. 확인 응답과 모든 태스크 알림은 `_meta`에 `subscriptions/listen` 요청 id와 같은 `io.modelcontextprotocol/subscriptionId`를 싣는다. 그 밖에는 각 태스크 알림이 그 순간 `tasks/get`이 돌려줬을 내용과 같다.

클라이언트는 여전히 Tasks 확장을 선언해야 한다. 이벤트 재생이나 `Last-Event-ID`에 기대지 말고, 다시 연결해 지속되는 태스크 id에서 이어 가야 한다.

## 실패의 의미 (Failure Semantics)

두 오류 계층을 올바르게 쓰라.

### 프로토콜 오류 (Protocol error)

잘못된 메서드 매개변수나 알 수 없는 태스크 id는 JSON-RPC 오류를 돌려주며, 보통 `-32602`다. 확장 지원이 없으면 필요한 역량 객체와 함께 `-32021`을 돌려준다.

### 태스크 실행 결과 (Task execution outcome)

- `isError: true`인 평범한 도구 결과도 `completed` 태스크다. 도구 호출이 정의된 결과를 냈기 때문이다.
- 지연 실행 도중의 JSON-RPC 오류는 태스크를 `failed`로 만들고 그 JSON-RPC 오류를 `error`에 저장한다.
- 사용자의 거절은 `cancelled`가 될 수도, 완결된 거절 결과가 될 수도, 도메인에 맞는 다른 안전한 결과가 될 수도 있다. 무엇을 골랐는지 문서로 남기라.

## 지속성, 만료, 소유권 (Durability, Expiry, and Ownership)

적어도 태스크 id, 상태, 시각, ttl, 조회 간격, 원래 작업의 소유권, 결과 또는 오류, 남아 있는 입력 요청, 지금까지 발급한 모든 입력 키를 저장하라.

저장소 키에는 권위 있는 테넌트와 주체가 들어 있거나 그것을 찾아갈 수 있어야 한다. 태스크 id를 안다는 사실이 접근 권한이 되어서는 안 된다. `tasks/get`, `tasks/update`, `tasks/cancel`, 구독마다 소유권을 확인하라.

`ttlMs`는 생성 시점부터 재며 바뀔 수 있다. 태스크가 관측 가능한 갱신을 더는 내놓지 않을 때 클라이언트는 이것을 마지막 보루로 삼을 수 있다. 서버는 만료된 태스크를 실패로 처리하고 나중에 지울 수 있다. 완료된 결과를 완료 후 그만큼의 밀리초 동안 보관하겠다는 약속으로 설명하지 마라.

원자적 쓰기나 트랜잭션을 쓰라. 이 레슨은 임시 파일을 쓴 다음 원자적으로 이름을 바꾼다. 복제본이 여럿인 서비스라면 공유 지속 저장소와 워커 임대, 또는 그에 준하는 동시성 제어를 써야 한다.

```figure
tp-task-lifecycle
```

## 만들어 보기 (Build It)

`code/main.py`는 결정적인 태스크 서비스를 구현한다.

- `server/discover`가 `supportedVersions`, 캐시 힌트, Tasks 확장을 돌려준다.
- `tools/list`가 유효한 입력 스키마를 가진, 결정적이고 캐시 가능한 `generate_report` 서술자를 돌려준다.
- `tools/call`이 `resultType: "task"`를 돌려주기 전에 태스크를 만들어 저장한다.
- 새 서비스 인스턴스가 같은 태스크를 다시 읽어 들여 재시작 복구를 보여 준다.
- `tasks/get`이 완결된 태스크 스냅샷을 돌려준다.
- 워커가 `working`에서 `input_required`로 넘어간다.
- `tasks/update`가 폼 응답을 받아들이고 빈 complete 확인 응답을 돌려준다.
- 워커가 자기 `resultType`과 서버 신원을 가진 중첩 `CallToolResult`를 저장한 다음 `completed`로 넘어간다.
- 이 구현에서 `tasks/cancel`은 몇 번을 불러도 결과가 같다.
- HTTP 빌더가 `tasks/get`, `tasks/update`, `tasks/cancel`의 `Mcp-Name`을 `params.taskId`로 맞춘다.
- 알림 도우미는 `notifications/subscriptions/acknowledged`와 `notifications/tasks`를 쓰며, 둘 다 listen 요청 id를 달고 있다.
- id 없는 알림은 JSON-RPC 응답을 만들지 않는다.

워커는 배경 스레드에서 잠드는 대신 명시적으로 단계를 넘어간다. 그 덕분에 모든 상태 전이가 결정적이 되고, 프로토콜 예제가 큐 기계 장치와 분리된 채로 남는다.

## 직접 해 보기 (Use It)

저장소 루트에서 시작한다.

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

기대하는 결과 순서는 이렇다.

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

현대 서비스에서 `tasks/status`, `tasks/result`, `tasks/list`가 메서드 없음을 돌려주는지도 확인하라.
`tools/list`가 결정적인지, 그리고 현재의 모든 HTTP 태스크 메서드가 `Mcp-Name`으로 태스크 id를 복제하는지 확인하라.

## 결과물 (Ship It)

`outputs/skill-task-store-designer.md`는 이제 확장을 아는 설계를 만들어 낸다. 역량 협상, 돌려주기 전에 저장하는 생성, 현행 메서드, 입력 갱신 흐름, 소유권, 만료, 취소, 구독, 그리고 사라진 실험적 메서드로부터의 이전까지 다룬다.

## 연습 문제 (Exercises)

1. 남아 있는 입력 키를 하나 더 추가하라. `tasks/update`를 일부만 보내고, 두 키가 모두 채워질 때까지 태스크가 `input_required`에 남는지 증명하라.
2. 저장소에 테넌트 소유권을 추가하고, 엉뚱한 인증 주체가 내민 유효한 태스크 id를 거부하라.
3. 만료가 있는 워커 임대를 추가하라. 서비스 인스턴스 두 개가 같은 태스크를 동시에 끝낼 수 없음을 보여라.
4. `subscriptions/listen`을 위한 POST 응답 SSE 어댑터를 구현하라. GET, `Last-Event-ID`, 세션 헤더를 추가하지 마라.
5. 만료 정리를 추가하라. 테넌트 사이의 존재 여부를 흘리지 않으면서 만료된 태스크와 형식이 어긋난 태스크 id를 구별하라.

## 핵심 용어 (Key Terms)

| 용어 | 현행 확장에서의 뜻 |
|------|----------------------------------|
| Tasks extension | 지속되는 비동기 작업을 위한 선택적 `io.modelcontextprotocol/tasks` 역량 |
| `CreateTaskResult` | 해당하는 요청에 서버가 주도해 돌려주는 `resultType: "task"` 응답 |
| `tasks/get` | 종료 결과나 대기 중 입력까지 담은 현재 태스크 스냅샷 전체를 물어보는 호출 |
| `tasks/update` | 태스크에 남아 있는 `inputRequests`에 응답을 제출하는 호출 |
| `tasks/cancel` | 협조적 취소 의사를 확인해 주는 호출 |
| `input_required` | 클라이언트 입력이 남아 있음을 가리키는 태스크 상태 |
| `pollIntervalMs` | 다음 조회까지 서버가 권하는 최소 지연 |
| `ttlMs` | 태스크 생성 시점부터 재는 만료 기간 |
| Durable-before-return | 핸들을 보내기 전에 태스크 id가 해결돼야 한다는 규칙 |
| `notifications/tasks` | 구독된 SSE 응답으로 전달되는 선택적 태스크 스냅샷 전체 |

## 레거시 호환성 (Legacy Compatibility)

2025-11-25의 실험적 표면은 클라이언트가 요청하는 태스크 확장, `tasks/status`, `tasks/result`, 선택적 `tasks/list`를 썼다. 그 이름들은 버전이 고정된 레거시 어댑터 안에만 두라. 현행 클라이언트는 확장 역량을 쓰고, 서버가 주도해 건네는 핸들을 받아들이고, `tasks/get`으로 상태를 묻고, `tasks/update`로 입력을 공급하고, 태스크 스냅샷에서 최종 결과를 읽는다.

## 더 읽을거리 (Further Reading)

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
