# MCP 기초: 무상태 요청과 JSON-RPC (MCP Fundamentals: Stateless Requests and JSON-RPC)

> 요즘의 MCP에는 핸드셰이크도 없고 프로토콜 세션도 없다. 요청 하나하나가 스스로 해석되고 인가되고 라우팅되고 재시도될 수 있을 만큼의 메타데이터를 담고 있어야 한다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## 학습 목표 (Learning Objectives)

- MCP의 서버 기본 요소와 클라이언트 쪽 기능을 구분한다.
- MCP `2026-07-28`에 맞는 유효한 JSON-RPC 2.0 요청과 응답을 만든다.
- 모든 요청에 프로토콜 판본과 클라이언트 역량, 클라이언트 신원을 붙인다.
- 핸드셰이크 없이 `server/discover`를 쓰고 `UnsupportedProtocolVersionError`를 처리한다.
- 독립적인 요청 하나가 검증에서 완료 결과까지 가는 길을 따라가 본다.

## 문제 (The Problem)

MCP 서버는 같은 프로세스나 HTTP 워커에서, 서로 다른 클라이언트가 서로 다른 역량을 선언한 요청 두 개를 연달아 받을 수 있다. 서버가 직전 요청이 선언한 내용을 기억하고 있으면, 엉뚱한 권한을 적용하거나 엉뚱한 형식으로 응답하게 된다.

MCP `2026-07-28`은 그 모호함을 없앤다. 프로토콜의 핵심이 무상태다. 서버는 지금 요청을 어떻게 처리할지를 연결의 이력이 아니라 지금 요청만 보고 정해야 한다.

이것이 머릿속 모델을 바꾼다. 예전 순서는 연결을 먼저 맺고, 핸드셰이크를 하고, 그다음에 연산을 하는 것이었다. 요즘 순서는 더 단순하다.

1. 클라이언트가 스스로를 설명하는 요청을 보낸다.
2. 서버가 그 요청의 판본과 역량을 검증한다.
3. 서버가 그 메서드를 처리한다.
4. 서버가 타입이 정해진 결과나 JSON-RPC 오류를 돌려준다.

다음 요청은 같은 과정을 처음부터 되풀이한다.

## 개념 (The Concept)

### 서버의 기본 요소

MCP 서버는 주요 기본 요소 세 가지를 노출한다.

1. **도구(Tools)**는 모델이 통제하는 행동이며, `tools/list`로 찾고 `tools/call`로 호출한다.
2. **자원(Resources)**은 URI로 주소를 매긴 데이터이며, `resources/list`로 찾고 `resources/read`로 가져온다.
3. **프롬프트(Prompts)**는 다시 쓸 수 있는 템플릿이며, `prompts/list`로 찾고 `prompts/get`으로 채워 낸다.

루트와 샘플링, 로깅은 호환을 위해 `2026-07-28` 스키마에 남아 있지만 폐기 예정이다. 새로 만드는 구현이라면 루트는 명시적인 도구나 자원 입력으로, 샘플링은 모델 공급자의 API를 직접 불러서, 로깅은 stderr나 OpenTelemetry로 처리해야 한다. 사용자에게 되묻는 일은 여러 번 왕복하는 요청(Multi Round-Trip Request)으로 여전히 할 수 있다. 서버가 입력 요청을 돌려주면 클라이언트가 원래 연산을 다시 호출하는 방식이다. 요즘의 서버는 자기가 먼저 독립적인 JSON-RPC 요청을 시작하지 않는다.

### JSON-RPC 봉투

MCP는 JSON-RPC 2.0을 쓴다.

- 요청: `{jsonrpc, id, method, params}`
- 응답: `{jsonrpc, id, result}` 또는 `{jsonrpc, id, error}`
- 알림: `id`가 없는 `{jsonrpc, method, params}`

요청의 `id`는 응답 하나를 짝지어 줄 뿐이다. 프로토콜 세션을 만들지 않는다.

### 요청에 반드시 필요한 메타데이터

요즘의 모든 요청은 `params` 안에 `_meta` 객체를 담고 다닌다.

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

프로토콜 판본과 클라이언트 역량은 필수다. 클라이언트 신원은 권장 사항이다. 그것은 표시와 디버깅을 위해 스스로 밝힌 정보이지 보안 자격 증명이 아니다.

서버는 이 값들을 앞선 요청이나 stdio 프로세스, HTTP 연결, 전송 계층 헤더만 보고 짐작해서는 안 된다.

### 완료 결과와 서버 신원

요즘 방식의 성공 결과에는 모두 `resultType`이 들어 있다. 평범한 최종 결과는 `"complete"`를 쓴다. 서버는 결과 메타데이터에 자기 신원도 밝혀야 한다.

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`와 `resources/list`, `prompts/list`, `resources/templates/list`, `resources/read`, `server/discover`는 캐시할 수 있는 결과다. 그래서 `ttlMs`와 `cacheScope`를 함께 담는다. 안전한 기본값은 `ttlMs: 0`과 `cacheScope: "private"`이다. 목록의 항목은 순서가 일정해야 한다. 그래야 같은 응답이 같은 캐시 키를 만들고, 모델이 보는 맥락도 흔들리지 않는다.

### 핸드셰이크 없는 탐색

요즘의 서버는 모두 `server/discover`를 구현해야 한다. 클라이언트는 다른 메서드를 부르기 전에 이것을 불러 다음을 받아 올 수 있다.

- `supportedVersions`
- 서버의 `capabilities`
- 선택적인 사용 안내 `instructions`
- 결과 `_meta` 안의 서버 신원
- 캐시 힌트

탐색은 쓸모가 있지만 반드시 거쳐야 하는 관문은 아니다. `tools/list`를 먼저 보내도 된다. 그 요청 자체가 이미 프로토콜 판본과 역량을 담고 있기 때문이다.

요청한 판본을 지원하지 않으면 서버는 JSON-RPC 코드 `-32022`와 함께 다음을 돌려준다.

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

클라이언트는 서로 지원하는 최신 판본을 골라, 새 JSON-RPC 요청 식별자로 다시 시도한다.

### 요청 하나의 수명 주기

요즘의 요청은 다음 순서로 따라가면 된다.

1. JSON-RPC 봉투 하나를 파싱한다.
2. `jsonrpc`가 `"2.0"`인지, `id`가 있는지, `method`가 문자열인지, `params`가 객체인지 확인한다.
3. `params._meta`에 판본 문자열과 역량 객체가 있어야 한다. 메타데이터가 없거나 형식이 잘못되었으면 `-32602`다.
4. HTTP 경계에서는 판본과 메서드, 해당되는 이름 헤더를 본문과 비교한다. 어긋나면 두 판본 값 가운데 하나가 지원되지 않는 것이더라도 `-32020`이다.
5. 서로 같다는 것이 확인된 뒤에, 맞춰진 판본이 지원되지 않으면 `-32022`로 물리친다.
6. 필요한 역량을 확인하고, `method`로 경로를 정하고, 메서드별 인자를 검증한다.
7. 핸들러가 돌기 전에 그 구체적인 연산에 대해 인증하고 인가한다.
8. 서버 신원을 담은 완료 결과를 돌려준다.
9. 요청 단위의 프로토콜 메타데이터를 잊는다.

이 순서를 지켜야 두 구성 요소가 서로 다른 호출을 해석하는 일이 생기지 않는다. 게이트웨이가 `Mcp-Name: notes.read`를 허가했는데 원본 서버가 `params.name: notes.delete`를 실행해서는 안 된다. 그리고 형식이 잘못된 입력과 헤더 혼선, 판본 협상, 역량 부족, 인가, 핸들러 실패가 각각 다른 근거로 남는다.

stdin을 닫거나 HTTP 응답을 끝내면 전송 계층의 활동이 끝난다. 그렇다고 프로토콜 세션이 끝나는 것은 아니다. 요즘의 MCP에는 프로토콜 세션이 없기 때문이다.

### 구판 호환은 명시적으로

`2025-11-25`까지의 판본은 `initialize`와 `notifications/initialized`, 연결 단위 역량을 쓰고, 예전 Streamable HTTP에서는 선택적인 프로토콜 세션도 썼다. 두 시대를 모두 다루는 클라이언트가 옛 서버와 이야기할 때는 그 동작이 여전히 필요하다.

두 시대를 갈라 두어라. 요즘 요청은 요청마다 실린 필수 메타데이터로 알아본다. 구판 연결은 문서로 정해 둔 대체 경로를 거쳤을 때만 고른다. `2026-07-28` 서버에 `initialize`를 기본값으로 보내지 마라.

그래서 "무상태"라는 말도 시대마다 뜻이 다르다. `2026-07-28`에서는 프로토콜의 불변 조건이다. 평범한 요청은 모두 스스로 해석될 수 있고 MCP 세션은 존재하지 않는다. `2025-11-25`까지의 판본에서는 초기화와 합의된 역량이 연결에 속하므로, 호환 어댑터가 그 구판 연결 상태를 들고 있을 수 있다. 두 시대를 함께 다루는 구현은 느슨한 상태 기계 하나가 아니다. 무상태인 최신 핵심과, 그 옆에 격리된 구판 어댑터가 있고, 어느 파서를 돌릴지 정하는 명시적인 판단이 그 앞에 있다.

두 뜻 어느 쪽도 오래 남는 애플리케이션 상태를 금지하지 않는다. 워크플로나 태스크, 초안은 공유 저장소에서 속을 들여다볼 수 없는 핸들 뒤에 살 수 있다. 클라이언트는 그 핸들을 평범한 입력으로 보내고, 모든 복제본이 그 사용을 인증하고 인가한다. 다만 사라진 세션을 대신하겠다고 프로토콜 맥락을 그 저장소에 흘려 넣어서는 안 된다.

```figure
mcp-tool-call
```

## 실제로 써 보기 (Use It)

`code/main.py`는 프레임워크 없이 요즘 방식의 MCP 메시지를 만들고, 검증하고, 따라가고, 처리한다. 다음으로 실행한다.

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

출력에서 불변 조건 세 가지를 확인하라.

- 모든 요청이 자기 `_meta` 필드를 되풀이해 담는다.
- 성공한 결과는 모두 `resultType: "complete"`이고 서버 신원을 담고 있다.
- 목록 결과의 순서가 일정하고 캐시 힌트가 명시되어 있다.

## 결과물로 남기기 (Ship It)

이 레슨은 `outputs/skill-mcp-handshake-tracer.md`를 남긴다. 파일 이름은 예전 그대로 두었지만, 그 내용은 이제 무상태 요청 추적기다. 메시지를 하나씩 따로 살펴보고, 구판 핸드셰이크 통신은 실제로 있을 때만 그렇게 표시한다.

## 연습 문제 (Exercises)

1. 요청 하나의 프로토콜 판본을 `2027-01-01`로 바꿔라. 오류 코드가 `-32022`이고 데이터가 지원되는 판본을 알려 주는지 확인하라.
2. 두 번째 요청에서 `io.modelcontextprotocol/clientCapabilities`를 지워라. 서버가 첫 번째 요청의 역량을 다시 쓰지 않는지 확인하라.
3. 메모리 안의 도구 목록 순서를 뒤집어라. `tools/list`가 여전히 같은 순서를 돌려주는지 확인하라.
4. `cacheScope`를 `public`에서 `private`으로 바꿔라. 각 경우에 어떤 인가 맥락이 그 응답을 다시 쓸 수 있는지 설명하라.
5. `clientInfo`를 빼는 시험을 추가하라. 클라이언트 신원은 필수가 아니라 권장이므로 그 요청은 여전히 유효해야 한다.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| Stateless protocol | 요청마다 그것을 해석하는 데 필요한 메타데이터를 스스로 담는 방식 |
| Request metadata | `params._meta`에 담는 판본과 클라이언트 역량, 그리고 권장되는 클라이언트 신원 |
| `server/discover` | 판본과 역량, 안내, 신원을 알려 주는 필수 서버 메서드 |
| `resultType` | 요즘 방식의 성공 결과마다 붙는 갈래 값 |
| Cacheable result | 필수 항목인 `ttlMs`와 `cacheScope` 힌트를 담은 결과 |
| Protocol era | 요청마다 메타데이터를 싣는 최신 방식인지, 연결 단위로 초기화하는 구판 방식인지 |
| Transport lifetime | 프로토콜 세션 상태가 아니라 프로세스와 연결, 응답 스트림의 수명 |
| `-32022` | 요청한 판본과 지원되는 판본을 함께 알려 주는, 지원하지 않는 프로토콜 판본 오류 |

## 더 읽을거리 (Further Reading)

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
