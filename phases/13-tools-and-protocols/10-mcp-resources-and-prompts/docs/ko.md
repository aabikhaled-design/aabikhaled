# MCP 리소스와 프롬프트: 무상태 서버를 위한 주소 지정 가능한 맥락 (MCP Resources and Prompts: Addressable Context for Stateless Servers)

> 도구는 작업을 수행한다. 리소스는 주소로 지정할 수 있는 내용을 내놓는다. 프롬프트는 사용자가 고르는 메시지 템플릿을 묶어 준다. 좋은 MCP 서버는 이 계약들을 서로 분리하고 예측 가능하게 유지한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- 소비자의 의도에서 출발해 도구, 리소스, 프롬프트 중 무엇을 쓸지 고른다.
- 필수 메서드인 `server/discover`로 리소스와 프롬프트의 표면을 알린다.
- 결정적인 `resources/list`와 `prompts/list` 결과를 만든다.
- 사용자별 데이터를 새어 나가게 하지 않으면서 `ttlMs`와 `cacheScope`를 적용한다.
- 유효하지 않거나 알 수 없는 리소스 URI에 JSON-RPC 오류 `-32602`를 돌려준다.
- `subscriptions/listen` POST 응답 스트림을 열고 모든 이벤트를 구독 ID로 맞춰 본다.
- 리소스 내용과 프롬프트 템플릿을 신뢰할 수 없는 서버 출력으로 취급한다.

## 소비자에서 출발하라 (Start With the Consumer)

MCP를 잘못 쓰는 가장 쉬운 길은 구현 코드에서 출발하는 것이다. 함수가 익숙하다는 이유로 데이터베이스 질의가 도구가 된다. 파일에 저장돼 있다는 이유로 재사용 가능한 작업 흐름이 리소스가 된다. 호스트가 밀어 넣을 수 있다는 이유로 프롬프트가 감춰진 정책이 된다.

누가 고르고 무엇을 기대하는지에서 시작하라.

| 기본 요소 | 주된 의도 | 선택 주체 | 일반적인 결과 |
|---|---|---|---|
| Tool | 작업을 수행한다 | 모델 또는 애플리케이션 | 구조화된 동작 결과 |
| Resource | URI에 있는 내용을 읽는다 | 호스트, 애플리케이션, 또는 사용자 | 텍스트 또는 이진 내용 |
| Prompt | 재사용 가능한 메시지 작업 흐름을 시작한다 | 호스트 UI를 통한 사용자 | 하나 이상의 프롬프트 메시지 |

`notes://note-1`에 있는 메모는 주소로 지정할 수 있는 내용이므로 리소스다. `delete_note`는 상태를 바꾸므로 도구다. `review_note`는 사용자가 준비된 검토 작업 흐름을 고르는 것이므로 프롬프트다.

완성도가 있어 보이려고 작업 하나를 셋 모두로 노출하지 마라. 표면이 하나 늘 때마다 탐색, 인가, 캐싱, 오류 처리, 테스트, 문서가 따라붙는다.

## 2026-07-28의 무상태 봉투 (The 2026-07-28 Stateless Envelope)

이 레슨은 MCP 프로토콜 개정판 `2026-07-28`을 대상으로 한다. 이 프로필에는 초기화 악수도 프로토콜 세션도 없다. 모든 요청이 예약된 `_meta` 키에 프로토콜 버전과 클라이언트 역량을 싣는다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

서버는 `server/discover`를 반드시 구현해야 한다. 그 결과는 지원 버전,
리소스와 프롬프트 역량, 구현 신원, 캐시 힌트를 알린다. 클라이언트가 다른
메서드를 바로 호출해도 되지만, 탐색은 UI를 만들기 전에 안정적인 스냅샷을
하나 쥐여 준다.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

정상 결과는 `"resultType": "complete"`를 선언한다. 응답의 `_meta`는 `io.modelcontextprotocol/serverInfo`로 응답을 처리한 구현을 식별해 준다. 이 정보는 진단에 쓸모가 있다. 인증된 신원은 아니다. 지원하지 않는 개정판을 실은 요청에는 요청된 개정판과 서버가 지원하는 개정판을 함께 담아 `-32022`를 돌려준다.

무상태 계약은 설계 감각을 바꾼다. 목록이 한 연결 위의 이전 호출에 기대서는 안 된다. 자격 증명은 요청 입력이므로 인가에 따라 보이는 집합이 달라질 수는 있지만, 연결 이력이 그것을 바꿔서는 안 된다.

## 리소스는 안정적인 URI 계약이다 (Resources Are Stable URI Contracts)

리소스는 URI로 식별되는 내용이다. 처리기보다 URI를 먼저 설계하라.

좋은 URI의 성질은 이렇다.

- 북마크하거나 요청 사이에 넘길 만큼 안정적이다.
- 서버 도메인에 맞춰 이름 공간이 붙어 있다.
- 프로세스 ID나 연결과 무관하다.
- 저장소에 접근하기 전에 검증된다.
- 읽을 때마다 인가된다.

`notes://note-1`은 이름 공간이 명시적이므로 `note-1`보다 낫다. 파일 서버는 `file://` URI를 쓸 수 있지만, 심볼릭 링크와 상대 경로 조각을 풀어낸 뒤에도 설정된 디렉터리 경계를 반드시 확인해야 한다.

`resources/list`는 호출자에게 현재 보이는 리소스를 돌려준다. URI 같은 안정적인 키로 정렬하라. 결정적인 순서는 시끄러운 캐시 미스, 매번 달라지는 스냅샷, 새로고침마다 튀는 호스트 UI를 막아 준다.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read`는 내용 항목을 하나 이상 돌려준다. 알 수 없는 URI는 성공적인 빈 읽기가 아니다. 현행 리소스 명세는 유효하지 않거나 알 수 없는 리소스 URI를 JSON-RPC의 잘못된 매개변수, 즉 코드 `-32602`로 배정한다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

이 구분 덕분에 클라이언트는 부재와 유효한 빈 문서를 갈라낼 수 있다. 더 넓은 범위를 다시 뒤지는 실수도 막아 준다.

### 리소스 템플릿 (Resource templates)

리소스 템플릿은 매개변수를 가진 URI 묶음을 설명한다. 구체적인 항목을 전부 나열하면 비용이 크거나 끝이 없을 때 쓴다. 예를 들어 `notes://projects/{project}/decisions/{decision}`은 모든 결정을 돌려주지 않고도 유효한 주소를 만드는 법을 클라이언트에게 알려 준다.

템플릿이 검증을 느슨하게 만들지는 않는다. 변수를 파싱하고, 인가를 적용하고, 길이와 문자 제한을 강제하고, 저장소 질의는 타입이 붙은 매개변수로 만들어라. 임의의 URI 꼬리를 파일 시스템 경로나 데이터베이스 문장에 이어 붙이지 마라.

### 내용은 신뢰할 수 있는 지시가 아니다 (Content is not trusted instruction)

리소스 텍스트에는 프롬프트 주입, 비밀 값, 오도하는 명령, 망가진 마크업이 들어 있을 수 있다. 호스트는 출처를 보존하고 리소스 내용을 데이터로 취급해야 한다. 서버는 내용 크기를 제한하고, 정확한 MIME 타입을 돌려주고, 호출자가 접근할 수 없는 필드를 가리고, 관계없는 레코드를 돌려주지 않아야 한다.

## 프롬프트는 사용자가 조종하는 템플릿이다 (Prompts Are User-Controlled Templates)

MCP 프롬프트는 사용자가 명시적으로 고르도록 설계돼 있다. 호스트는 이것을 슬래시 명령, 메뉴 항목, 작업 흐름 버튼으로 그릴 수 있다. 프로토콜이 특정 UI를 요구하지는 않는다.

`prompts/list`는 같은 요청 인가에 대해 결정적이어야 한다. 프롬프트마다 안정적인 이름, 쓸모 있는 설명, 그리고 호스트가 `prompts/get` 전에 입력을 모을 수 있게 해 주는 인자 선언이 필요하다.

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get`은 인자를 메시지로 풀어낸다. 호스트의 시스템 지시를 대체하지는 않는다. 돌려받은 메시지가 모델 맥락에 어떻게 들어갈지는 호스트가 정하며, 호스트는 자신이 신뢰하는 정책을 더 높은 우선순위로 유지한다.

프롬프트 인자는 서버 경계에서 검증하라. 프롬프트에 담긴 URI는 리소스를 직접 읽을 때와 똑같은 인가 검사를 통과해야 한다. 프롬프트를 리소스 접근을 우회하는 샛길로 만들지 마라.

## 캐시 힌트는 정확성의 일부다 (Cache Hints Are Part of Correctness)

`ttlMs`는 결과를 얼마나 오래 다시 써도 되는지 클라이언트에게 알려 준다. `cacheScope`는 그 캐시 값을 누구와 공유해도 되는지 설명한다.

| 범위 | 뜻 | 일반적인 용도 |
|---|---|---|
| `public` | 인가가 허락하는 한 사용자 간에 재사용해도 된다 | 공개 프롬프트 카탈로그 |
| `private` | 요청한 사용자나 자격 증명 맥락에 묶인다 | 사용자 소유 메모 내용 |

TTL은 데이터가 바뀌는 속도와 낡은 값이 일으키는 피해를 보고 고르라. 공개 프롬프트 카탈로그에는 5분이 어울릴 수 있다. 비공개 메모 읽기에는 1분을 쓸 수 있다.

MCP는 `cacheScope` 값으로 `public`과 `private`만 정의한다. 비밀 값을 담았거나 빠르게 바뀌는 결과에는 `cacheScope: "private"`에 `ttlMs: 0`을 붙여 돌려주고, 더 엄격한 저장 금지 규칙은 호스트 캐시 정책에서 적용하라. `no-store` 자체는 MCP의 `cacheScope` 값이 아니다.

캐시 힌트는 결코 인가를 대신하지 않는다. 캐시 키에는 가시성을 바꾸는 요청 차원이 전부 들어가야 한다. 테넌트, 사용자, 스코프, 로케일, 페이지 커서까지 포함된다. 공유 캐시가 그 차원들을 안전하게 표현하지 못한다면, TTL이 0인 `private`와 호스트 수준의 저장 금지 정책을 쓰라.

## 구독은 클라이언트가 연 응답 스트림을 쓴다 (Subscriptions Use a Client-Opened Response Stream)

현대 구독 방식은 예전의 `resources/subscribe` RPC와 옛 HTTP GET 이벤트 엔드포인트를 대체한다.

클라이언트는 `subscriptions/listen`을 평범한 JSON-RPC 요청으로 보낸다. Streamable HTTP에서는 응답이 SSE 스트림으로 열린 채 남는 POST가 된다. `notifications` 객체는 허용 목록이다. 서버는 요청되지 않은 알림 종류를 전달해서는 안 된다.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

요청 ID가 곧 구독 ID다. 요청된 이벤트를 하나라도 보내기 전에 서버는 `notifications/subscriptions/acknowledged`를 보낸다. 거기 담긴 필터에는 서버가 받아들인 부분만 들어 있다.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

그 스트림에 이어지는 모든 이벤트는 같은 메타데이터를 싣는다.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

알림은 리소스가 바뀌었다는 사실만 말한다. 클라이언트는 현재 인가를 따라 `resources/read`로 그것을 다시 읽는다. 이벤트 안에 새 문서가 들어 있다고 가정하지 않는다.

구독 여러 개가 stdio 채널 하나를 함께 쓸 수 있다. 구독 ID가 있어서 클라이언트는 그것들을 갈라낼 수 있다. HTTP에서는 응답 스트림을 닫으면 구독이 취소된다. 스트림을 정상적으로 끝내는 서버는 원래 요청과 짝이 맞는 `resultType: "complete"` 응답을 마지막에 돌려준다.

구독 스트림을 프로토콜 세션처럼 쓰지 마라. 이후의 읽기도 여전히 완결된 요청이므로 건강한 서버 인스턴스 아무 곳에나 닿을 수 있다.

```figure
t3-primitive-sort
```

## 대화형 실습 (Interactive Lab)

그림을 보고 프로젝트 추적기의 기능 다섯 가지를 분류하라. 이슈 상세, 이슈 생성, 스프린트 회고 템플릿, 프로젝트 정책, 이슈 종료다. 그다음 어떤 목록을 공개 캐시에 담아도 되는지, 어떤 읽기가 비공개로 남아야 하는지, 어떤 리소스가 갱신 알림을 받을 만한지 정하라.

분류할 때마다 고르는 주체가 누구인지 말하라. 모델이 동작을 수행한다면 도구를 쓴다. 호스트가 URI로 지정된 내용을 읽는다면 리소스를 쓴다. 사용자가 준비된 메시지 작업 흐름을 시작한다면 프롬프트를 쓴다.

## 실습 (Practice Lab)

저장소 루트에서 시뮬레이터를 실행하라.

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

기록을 이 순서로 살펴보라.

1. `server/discover`가 현재 개정판과 두 역량을 모두 알리는지 확인한다.
2. 두 목록 결과가 정렬돼 있고 `resultType: "complete"`를 쓰는지 확인한다.
3. 목록과 읽기 결과가 의도한 캐시 힌트를 싣는지 확인한다.
4. 읽기 URI를 `notes://missing`으로 바꿔 `-32602`가 나오는지 본다.
5. 구독 확인 응답이 리소스 이벤트보다 먼저 오는지 확인한다.
6. 이벤트와 정상 종료가 둘 다 구독 ID `5`를 싣는지 확인한다.

이 파이썬 모형은 실제 HTTP 연결을 열지 않는다. SDK가 요청 범위 응답 스트림에 올려야 할 메시지를 나타낼 뿐이다. 실제 운영에서는 공식 SDK로 프레이밍과 전송을 처리하라.

## 산출물 (Shipped Artifact)

`outputs/skill-primitive-splitter.md`는 MCP 기본 요소 선택을 위한 재사용 가능한 설계 검토서다. 이제 결정적 탐색, 캐시 범위, 유효하지 않은 URI 동작, 현대 구독 필터까지 점검한다.

이 레슨은 `assets/primitive-split.svg`도 함께 내놓는다. 기본 요소와 구독 경계를 정적으로 그린 그림이라 연결 없이도 공부할 수 있다.

## 확인하기 (Verify It)

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

기대 결과는 이렇다. 메인 프로그램이 JSON 기록을 출력하고, 테스트 명령이 최소 열두 개의 테스트 통과를 보고한다.

## 캡스톤 연결 (Capstone Connection)

캡스톤 서버가 동작 옆에 주소 지정 가능한 지식을 함께 내놓는다면 이 계약을 쓰라. 결정적인 카탈로그 스냅샷 하나, 인가된 리소스 읽기 하나, 프롬프트 해석 하나, 유효하지 않은 URI 사례 하나, 구독 기록 하나를 포함하라.

제출하는 증거는 어떤 목록도 연결 이력에 기대지 않는다는 점과, 구독 이벤트가 바탕 리소스에 대한 접근 권한을 주지 않는다는 점을 보여 줘야 한다.

## 연습 문제 (Exercises)

1. `notes://projects/{project}/notes/{id}` 리소스 템플릿을 추가하고 두 변수를 모두 검증하라.
2. 결정적인 순서를 유지하면서 `resources/list`에 페이지 나누기를 추가하라.
3. 리소스 하나를 `ttlMs: 0`인 `cacheScope: "private"`로 바꾸고, 호스트 수준 저장 금지 정책을 더한 뒤, 두 통제가 왜 필요한지 위협으로 설명하라.
4. 프롬프트 목록 변경 구독을 추가하고, 필터에 `promptsListChanged`가 없으면 이벤트가 전혀 가지 않음을 증명하라.
5. 구독 두 개를 동시에 만들고 각 이벤트가 올바른 요청 ID를 싣는지 증명하라.
6. 읽기 처리기에 인가 주체를 추가하고, 캐시 항목이 주체를 넘나들 수 없음을 증명하라.

## 핵심 용어 (Key Terms)

- **Resource:** MCP 서버가 내놓는, URI로 주소를 지정하는 내용.
- **Prompt:** MCP 서버가 내놓는, 사용자가 조종하는 메시지 템플릿.
- **Deterministic list:** 같은 요청 입력에 대해 구성원과 순서가 안정적인 탐색 결과.
- **`ttlMs`:** 밀리초 단위의 캐시 신선도 기간.
- **`cacheScope`:** 캐시된 결과를 공유해도 되는 경계.
- **`subscriptions/listen`:** 응답 스트림으로 명시적으로 걸러진 알림을 전달하는, 오래 유지되는 요청.
- **Subscription ID:** 알림 메타데이터에 되풀이되는, 원래 listen 요청의 ID.
- **Invalid parameters:** 유효하지 않거나 알 수 없는 리소스 URI에 쓰는 JSON-RPC 오류 `-32602`.
- **Unsupported protocol version:** `supported`와 `requested` 개정판을 담는 JSON-RPC 오류 `-32022`.
- **`server/discover`:** 지원 개정판, 역량, 신원, 선택적 캐시 힌트를 돌려주는 필수 서버 메서드.

## 더 읽을거리 (Further Reading)

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/basic/utilities/caching)
