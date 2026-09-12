# 명시적 범위와 무상태 유도 (Explicit Scope and Stateless Elicitation)

> 루트는 MCP 2026-07-28에서 폐기 예정이며 애초에 보안 모래상자였던 적이 없다. 범위는 눈에 보이는 도구 인자나 리소스 URI에 담고, 서버에서 인가하고, 도구가 정말로 사용자 입력이 필요할 때는 MRTR을 쓰라. 사용자는 결정을 보고, 모델은 핸들을 보며, 어느 서버 인스턴스든 재시도를 처리할 수 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- 폐기 예정인 루트를 명시적인 작업 공간 매개변수, 리소스 URI, 서버 설정으로 바꾼다.
- 범위 힌트를 인가, 경로 봉쇄, 운영체제 모래상자와 갈라놓는다.
- 폼 모드 `elicitation/create`를 MRTR의 `input_required` 결과로 전달한다.
- 요청마다 실리는 클라이언트 역량으로 유도 지원을 알리고, 지원하지 않는 모드는 거부한다.
- `accept`, `decline`, `cancel`을 서로 다른 결과로 검증한다.
- 파괴적 작업의 확인을 인증된 주체, 원래 인자, 후보 집합, 만료에 묶는다.

## 닮아 보이는 두 문제 (Two Problems That Look Similar)

메모 도구가 이런 요청을 받는다. "예전 TPS 보고서를 지워 줘."

서버는 서로 다른 두 질문에 답해야 한다.

1. 이 작업이 건드려도 되는 작업 공간은 어디인가?
2. 일치하는 메모 셋 중에 사용자가 가리킨 것은 어느 것인가?

첫째는 범위와 인가의 문제다. 둘째는 대화로 풀어야 하는 지목의 문제다. 이 둘을 섞으면 위험한 설계가 나온다. 클라이언트가 건넨 폴더를, 호출자가 그 안의 모든 것을 지워도 된다는 증거로 취급하는 식이다.

## 루트는 이전 대상 표면이다 (Roots Are a Migration Surface)

이전 MCP 개정판에서는 클라이언트가 루트를 알리고 목록이 바뀌면 서버에 알려 줄 수 있었다. 루트는 참고용 안내였다. 서버 프로세스가 무엇을 읽을 수 있는지 제약하지 않았고, 호출자를 인가하지도 않았고, 운영체제 모래상자를 만들지도 않았다.

MCP 2026-07-28은 새 설계에서 `roots/list`와 `notifications/roots/list_changed`를 권장하지 않는다. 다음 명시적 대안 중 하나를 쓰라.

- 호출마다 범위가 달라진다면 `workspaceUri`나 `directory` 도구 인자.
- 작업이 이미 리소스를 겨냥한다면 리소스 URI.
- 배포 하나가 고정된 작업 공간 하나를 맡는다면 서버 설정.
- 코드가 기술적으로 빠져나갈 수 없어야 한다면 프로세스 모래상자나 가둔 파일 시스템.

기존 2026-07-28 통합이 폐기 예고 기간 동안 여전히 `roots/list`를 필요로 한다면, 서버는 그것을 MRTR의 `inputRequests`에 담는다. 살아 있는 역방향 요청을 보내서는 안 된다. 그것은 이전용 어댑터이며, 새 처리기는 대신 명시적 범위를 받아야 한다.

모델은 명시적 핸들을 보고 되풀이할 수 있다. 전송 세션에 숨은 범위는 살펴보기도, 재현하기도, 감사하기도, 라우팅하기도 더 어렵다.

### 세 겹 규칙 (The three-layer rule)

명시적 URI라고 해서 스스로를 인가하지는 않는다. 세 겹을 모두 강제하라.

1. **인가:** 이 인증된 주체가 이 작업 공간을 써도 되는가?
2. **봉쇄:** 정규화한 대상 URI가 인가된 작업 공간 경계 안에 머무는가?
3. **모래상자:** 서버가 침해당하더라도 운영체제가 탈출을 막을 수 있는가?

실행 가능한 서버는 인가된 작업 공간 URI의 허용 목록을 유지하고, 퍼센트 인코딩된 경로를 정규화하고, 진짜 경로 구성 요소 경계를 확인하며, 삭제 직전에 봉쇄를 다시 확인한다.

문자열 접두사만 보는 순진한 검사는 틀렸다.

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

두 적대적 경로 모두 사람을 속이는 문자열로 시작한다. 먼저 정규화한 다음 경로 구성 요소를 비교하라. 실제 운영 파일 시스템 서버는 심볼릭 링크 경합과 플랫폼마다 다른 경로 의미까지 막아야 한다.

## 유도는 남아 있지만 전달 방식이 바뀌었다 (Elicitation Still Exists, but Delivery Changed)

유도는 `tools/call`, `prompts/get`, `resources/read` 도중에 사용자 입력을 모으는 현행 클라이언트 기능이다. 메서드 이름은 여전히 `elicitation/create`다. 바뀐 것은 전선 위 흐름의 방향이다.

2026-07-28 서버는 역방향 JSON-RPC 요청을 보내지 않는다. 대신 `InputRequiredResult`를 돌려준다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

호스트가 폼을 그린다. 사용자는 수락하거나, 명시적으로 거절하거나, 창을 닫아 버릴 수 있다. 그다음 클라이언트는 새 id로 원래 `tools/call`을 재시도한다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

두 호출 사이에 프로토콜 세션은 없다. 서버는 되돌아온 상태를 검증하고, 응답을 기대한 스키마와 대조해 확인하고, 선택된 메모가 서명된 후보 집합에 있었는지 확인하고, 작업 공간을 다시 인가하고, 봉쇄를 다시 확인한 다음 삭제한다.

## 역량 협상은 요청마다 한다 (Capability Negotiation Is Per Request)

폼 모드 유도를 지원하는 클라이언트는 이렇게 선언한다.

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

빈 유도 역량인 `"elicitation": {}`은 호환성을 위해 폼만 지원하는 것과 같은 뜻으로 남아 있다. 명시적인 `"elicitation": {"form": {}}`도 폼 모드를 지원한다. URL만 선언한 `"elicitation": {"url": {}}`은 지원하지 않는다. 서버는 앞선 요청이 알렸더라도 현재 요청의 역량에 없는 모드를 담아서는 안 된다.

모든 요청은 `io.modelcontextprotocol/protocolVersion`도 싣는다. 버전이 없거나 문자열이 아니면 `-32602`를 돌려준다. 지원하지 않는 문자열에는 정확한 `supported`, `requested` 데이터와 함께 `-32022`를 돌려준다. 유도 지원이 없거나 URL만 지원하면 `data.requiredCapabilities`를 `{"elicitation":{"form":{}}}`로 채워 `-32021`을 돌려준다.

JSON-RPC `id`가 없는 봉투는 알림이다. JSON-RPC 성공 응답도 오류 응답도 내보내지 말고 처리하라. Streamable HTTP에서 받아들인 알림은 본문 없이 `202 Accepted`를 받는다.

`clientInfo`는 진단을 위해 넣어 두는 것이 좋지만 스스로 밝힌 값이므로 인가를 위해 사용자를 식별하지는 못한다.

서버는 `server/discover`를 구현하고 `supportedVersions`, 역량, `ttlMs`, `cacheScope`를 `resultType: "complete"`와 함께 돌려준다. 이 현대적 설계에서는 루트를 알리지 않는다. 도구를 알리므로 필수인 `tools/list`도 구현한다. 그 결과는 결정적인 `notes_delete` 서술자, 유효한 객체 `inputSchema`, 서버 신원 메타데이터, 공개 캐시 힌트를 돌려준다.

## 폼 모드 (Form Mode)

폼 모드는 쓸 만한 대화 상자를 위해 설계된 제한적 JSON 스키마를 쓴다. 루트는 객체이고, 그 속성은 평평한 원시 필드이거나 지원되는 열거형 배열이다. 깊이 중첩된 객체나 범용 문서 스키마는 확인 대화 상자에 어울리지 않는다.

폼 모드는 이런 데 쓰라.

- 여러 후보 중 하나를 고를 때,
- 파괴적인 작업을 확인할 때,
- 민감하지 않은 선호를 모을 때,
- 모델이 아니라 사용자가 정해야 하는 값 몇 개를 받을 때.

암호, API 키, 접근 토큰, 결제 자격 증명에는 폼 모드를 쓰지 마라. 그런 비밀 값은 MCP 클라이언트를 거쳐 가며 로그나 모델 맥락에 닿을 수 있다.

서버는 돌아온 내용을 다시 검증한다. 클라이언트 쪽 폼 검증은 사용 경험을 좋게 하지만 신뢰를 만들어 주지는 않는다.

## URL 모드 (URL Mode)

URL 모드는 대역 밖 상호작용을 위해 안전한 웹 URL을 보낸다.

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

제3자 인가처럼 민감한 정보가 서버가 관리하는 웹 흐름으로 곧장 가야 할 때 쓰라. 클라이언트는 목적지를 전부 보여 주고 동의를 받은 뒤에 연다. URL을 미리 가져와서는 안 된다.

`accept` 응답은 사용자가 URL을 여는 데 동의했다는 뜻이다. 외부 흐름이 끝났다는 증명은 아니다. 재시도가 오면 서버는 자신의 상태를 확인해 작업을 마치거나 또 다른 `input_required` 결과를 돌려준다.

URL 유도는 MCP 클라이언트와 MCP 서버 사이의 인가를 대신하지 않는다. MCP 서버가 사용자를 대신해 수행해야 하는 외부 상호작용을 위한 것이다. 서버는 브라우저 앞의 사용자를 MCP 작업을 시작한 그 인증된 주체와 같은 사람으로 묶어야 한다.

## 응답 분기 (Response Branches)

이 동작들을 별칭이 아니라 제품 차원의 결정으로 다루라.

| 동작 | 뜻 | 안전한 서버 동작 |
|--------|---------|----------------------|
| `accept` | 사용자가 상호작용을 제출했다 | 내용을 검증하고 계속 진행한다 |
| `decline` | 사용자가 명시적으로 거절했다 | 오류가 아닌 완결된 거절 결과를 돌려준다 |
| `cancel` | 사용자가 창을 닫았거나 끝내지 못했다 | 안전하게 멈추고 나중에 재시도할 수 있게 둔다 |

내용이 없는 것을 동의로 해석하지 마라. 거절을 프롬프트가 되풀이되는 루프로 바꾸지 마라.

## 파괴적 MRTR 상태 보호하기 (Protecting Destructive MRTR State)

후보 목록이 프롬프트나 서명 없는 Base64 값 안에만 있어서는 안 된다. 클라이언트는 자신이 되돌려 보내는 모든 것을 조종한다.

이 레슨은 다음을 담은 상태 페이로드에 서명한다.

- 인증된 주체,
- 요청이 출발한 메서드,
- `workspaceUri`와 `title`의 요약값,
- 폼에 보여 준 허용 메모 id들,
- 작업 단계,
- 짧은 만료.

변경 직전에 서버는 살아 있는 메모 레코드도 확인한다. 이렇게 하면 삭제 경합과, 폼을 보여 준 뒤 대상이 작업 공간 밖으로 옮겨진 경우를 잡아낸다.

한 번만 일어나야 하는 금전적 작업이나 되돌릴 수 없는 작업이라면, HMAC만으로는 유효한 상태가 만료 전에 재생되는 것을 막지 못한다. 모든 처리기 인스턴스가 함께 쓰는 재생 방지 저장소에 난스를 저장하고 정확히 한 번만 소비하라. 이 레슨은 크기와 TTL이 정리되는 저장소를 주입하고, 메모리 안 삭제를 수행하는 동안 그 원자적 점유를 붙들고 있는다. 실제 운영 데이터베이스라면 난스 점유와 변경을 한 트랜잭션이나 그에 준하는 조건부 쓰기 경계로 묶어야 한다.

난스를 점유하기 전에 상호작용을 검증하라. 형식이 어긋난 응답이나 `cancel`은 아무것도 바꾸지 않고, 상태는 만료될 때까지 재시도 가능한 채로 남는다. 명시적인 `decline`은 종결이므로, 이 레슨은 아무것도 지우지 않고 난스를 소비한다.

```figure
t3-roots-boundary
```

## 만들어 보기 (Build It)

`code/main.py`는 현대적인 `notes_delete` 도구를 보여 준다.

- `tools/list`는 필수인 작업 공간과 제목 스키마를 담은, 결정적이고 캐시 가능한 서술자를 돌려준다.
- 범위는 명시적인 `workspaceUri` 인자다.
- 서버 설정이 그 작업 공간을 레슨용 주체에게 인가한다.
- URI 정규화가 접두사 혼동과 인코딩된 경로 탈출을 거부한다.
- 파괴적인 삭제에는 모두 폼 모드 유도가 필요하다.
- 유도는 `resultType: "input_required"` 안에 실려 간다.
- 서명된 `requestState`가 정확한 후보 목록과 원래 인자를 묶는다.
- 주입된 재생 방지 저장소가 서버 인스턴스를 넘나드는 같은 수락 상태나 거절 상태를 거부한다.
- 재시도는 새 요청 id를 쓰고 `resultType: "complete"`를 돌려준다.

데이터 저장소는 프로토콜 동작을 살펴보기 쉽도록 메모리에 둔다. 데이터베이스를 쓰더라도 보안 규칙은 그대로다.

## 직접 해 보기 (Use It)

저장소 루트에서 시작한다.

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

확인할 지점은 이렇다.

- 탐색이 루트 없이 도구만 알린다.
- 도구 탐색이 `resultType`, 서버 신원, 캐시 힌트와 함께 `notes_delete`를 돌려준다.
- 요청 id `1`이 `inputRequests.delete_choice`에 폼을 담아 돌려준다.
- 요청 id `2`가 서명된 상태를 되돌려 보내고 삭제를 끝낸다.
- 접두사 경로와 인코딩된 탈출 경로가 둘 다 봉쇄에 걸린다.
- 제목이 바뀌면 원래 확인 상태를 다시 쓸 수 없다.
- 거절하면 메모가 그대로 남는다.
- 메모와 재생 방지 상태를 공유하는 서버 객체 두 개가 같은 확인을 둘 다 실행하지는 못한다.
- 빈 폼 선언과 명시적 폼 선언은 동작하고, URL만 지원하면 정확한 `-32021` 폼 요구사항이 돌아온다.
- 지원하지 않는 버전의 실패가 정확한 `-32022` 데이터 형태를 쓴다.
- id 없는 알림은 JSON-RPC 응답을 만들지 않는다.

## 결과물 (Ship It)

`outputs/skill-elicitation-form-designer.md`는 명시적 범위, 인가 검사, MRTR 폼, 응답 분기, 상태 결속을 설계해 준다. 폐기 예정인 루트를 모래상자로 취급하거나 폼 모드로 비밀 값을 모으는 것은 거부한다.

## 연습 문제 (Exercises)

1. 메모리 재생 방지 저장소를 SQLite로 바꿔라. 트랜잭션 하나로 난스를 점유하고 메모를 지운 다음, 프로세스 두 개가 둘 다 커밋할 수는 없음을 증명하라.
2. `url` 역량 협상과 대역 밖 설정 흐름을 추가하라. 제3자 자격 증명이 `inputResponses`에 들어가지 않게 하라.
3. 메모리 메모 맵을 임시 SQLite 데이터베이스로 바꿔라. 변경 트랜잭션 안에서 인가와 봉쇄를 다시 확인하라.
4. 실제 파일 시스템 구현을 위한 심볼릭 링크 정책을 추가하라. URI를 글자 그대로 봉쇄하는 것만으로 심링크 탈출을 막을 수 없는 이유를 설명하라.
5. 현대 MRTR 처리기의 출력을 서버가 먼저 보내는 레거시 유도로 옮겨 주는 2025-11-25 어댑터를 설계하라. 현재 처리기와는 분리된 채로 두라.

## 핵심 용어 (Key Terms)

| 용어 | 2026-07-28에서의 뜻 |
|------|------------------------|
| Roots | 폐기 예정인 참고용 작업 공간 힌트이며, 인가도 모래상자도 아니다 |
| Explicit scope | 요청 인자에 드러나는 작업 공간, 디렉터리, 리소스 핸들 |
| Containment | 대상을 경계 안에 붙들어 두는, 정규화된 경로 구성 요소 검사 |
| Elicitation | MCP 작업 도중 사용자 입력을 받는 클라이언트 기능 |
| Form mode | 제한된 평평한 스키마로 대역 안에서 받는 구조화된 사용자 입력 |
| URL mode | 민감하거나 외부에 있는 작업 흐름을 위한 대역 밖 상호작용 |
| MRTR | 입력 필요 결과를 돌려준 뒤 새 요청으로 재시도하는 무상태 방식 |
| `requestState` | 그대로 되돌아오고 서버가 무결성을 검사하는 불투명한 상태 |
| Decline | 사용자의 명시적 거절 |
| Cancel | 승인 없이 창을 닫았거나 끝내지 못한 상호작용 |

## 레거시 호환성 (Legacy Compatibility)

2025-11-25에 고정된 상대에게는 `roots/list`, `notifications/roots/list_changed`, 그리고 서버가 먼저 보내는 살아 있는 `elicitation/create`가 아직 있을 수 있다. 그 어댑터에는 레거시라는 이름표를 붙이라. 레거시 루트 목록이 서버 인가를 건너뛰게 두지 말고, 프로토콜 세션을 전제한 가정을 현대 처리기로 끌어오지 마라.

## 더 읽을거리 (Further Reading)

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
