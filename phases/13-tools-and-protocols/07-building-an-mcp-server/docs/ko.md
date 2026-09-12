# MCP 서버 만들기: 무상태 Python과 TypeScript (Building an MCP Server: Stateless Python and TypeScript)

> 요즘의 MCP 서버는 핸드셰이크를 기억하지 않는다. 요청마다 메타데이터를 검증하고, 핸들러 하나를 돌리고, 타입이 정해진 결과 하나를 돌려준다.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## 학습 목표 (Learning Objectives)

- MCP `2026-07-28`이 요구하는 `server/discover`를 구현한다.
- 요청마다 프로토콜 판본과 클라이언트 역량을 검증한다.
- 도구와 자원, 프롬프트를 순서가 일정한 목록으로 노출한다.
- 알맞은 결과에 `resultType`과 서버 신원, 캐시 힌트를 담아 돌려준다.
- 같은 무상태 계약을, 줄 단위로 구분되는 stdio 위에서 Python과 TypeScript로 각각 제공한다.

## 문제 (The Problem)

첫 메시지를 받고 클라이언트 역량을 저장해 두는 서버는 만들기는 쉽고 운영하기는 어렵다. 같은 프로세스가 여러 클라이언트를 차례로 받을 수 있다. 원격 요청은 다른 워커에 떨어질 수 있다. 낡은 역량 선언 하나가 인가 경계를 넘어 동작을 새어 나가게 만들 수 있다.

MCP `2026-07-28`은 요청 하나하나가 스스로를 설명하게 만들어서 그 문제의 프로토콜 쪽을 푼다. 애플리케이션은 여전히 오래 남는 메모나 작업, 명시적인 상태 핸들을 가질 수 있다. 가질 수 없는 것은, 이후 요청을 해석하는 방식을 바꾸는 숨은 프로토콜 상태다.

이 레슨에서는 메모 서버를 두 번 만든다. Python 판본과 TypeScript 판본 모두 프로토콜 핵심에는 표준 라이브러리만 쓴다. 둘은 같은 메서드를 노출하고 같은 통신 계약을 강제한다.

## 개념 (The Concept)

### 요즘의 처리 루프

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

stdio에 대한 세 가지 규칙은 여전히 중요하다.

- 표준 출력에는 JSON-RPC 메시지만 쓴다. 진단 정보는 stderr로 보낸다.
- 메시지는 줄바꿈으로 구분하고 응답마다 버퍼를 비운다.
- stdin이 끝에 닿으면 곧바로 종료한다.

프로세스의 수명은 전송 계층의 수명이다. 요즘 MCP의 세션이 아니다.

### 요청 검증

모든 요청에는 다음이 있어야 한다.

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

앞의 두 필드는 필수다. `clientInfo`는 권장 사항이다. 신원이 들어 있다면 그 형태는 검증하되, 그것을 인증으로 다루지는 마라.

판본을 지원하지 않으면 `requested`와 `supported`를 담아 코드 `-32022`를 돌려준다. 요청 메타데이터가 빠져 있으면 잘못된 파라미터이므로 코드 `-32602`다. 빠진 필드를 앞선 호출에서 가져다 채우지 마라.

### 탐색은 필수다

요즘의 서버는 `server/discover`를 반드시 구현해야 한다. 완결된 탐색 결과에는 지원하는 최신 판본과 역량, 선택적인 안내, 캐시 힌트, 그리고 결과 `_meta` 안의 서버 신원이 들어간다.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

탐색이 서버를 열어 주는 열쇠는 아니다. 클라이언트는 탐색을 부르지 않고도 `tools/list`를 부를 수 있다. `tools/list`가 이미 같은 요청 메타데이터를 담고 있기 때문이다.

### 도구

`tools/list`는 순서가 일정한 도구 기술자 목록을 돌려준다. 순서가 안정되어 있어야 응답 캐싱이 잘 되고 모델이 보는 맥락도 흔들리지 않는다. 이 결과에는 `ttlMs`와 `cacheScope`도 반드시 들어간다.

`tools/call`은 콘텐츠 블록과 `isError`를 돌려준다. JSON-RPC 봉투나 메서드 파라미터가 잘못되었을 때는 JSON-RPC 오류를 쓴다. 유효한 도구 호출이 실행되었는데 도구 자체가 실패했을 때는 `isError: true`를 쓴다.

도구 주석은 여전히 힌트일 뿐 강제가 아니다.

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

호스트는 이것을 확인 절차와 화면 표시에 쓰면 된다. 실제 인가는 서버가 강제해야 한다.

### 자원

`resources/list`는 안정된 URI 기술자를 돌려준다. `resources/read`는 타입이 정해진 내용을 돌려준다. `2026-07-28`에서는 둘 다 캐시할 수 있으므로 둘 다 `ttlMs`와 `cacheScope`를 담는다.

사용자별 메모 데이터에는 `cacheScope: "private"`을 쓴다. 공유 캐시가 인가 맥락을 넘어 비공개 응답을 다시 써서는 안 된다.

요즘의 변경 알림은 `resources/subscribe`를 쓰지 않는다. 클라이언트가 `subscriptions/listen`을 열고 `resourceSubscriptions`나 목록 변경 범주를 요청한다. 그 흐름은 10번 레슨에서 만든다.

### 프롬프트

`prompts/list`는 캐시할 수 있고 순서가 일정하다. `prompts/get`은 이름이 붙은 프롬프트를 인자와 함께 채워 낸다. 채워진 프롬프트 결과는 완료 결과이지만, 캐시 힌트를 요구하는 목록이나 읽기 결과에는 속하지 않는다.

### 성공 결과에는 모두 타입이 붙는다

예제는 모든 성공에 감싸개 하나를 쓴다.

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

목록과 읽기, 탐색 핸들러는 여기에 `ttlMs`와 `cacheScope`를 더한다. 감싸개를 한곳에 모아 두면 어떤 핸들러가 요즘 방식의 결과 필드를 조용히 빠뜨리는 일이 생기지 않는다.

### 서버가 먼저 요청을 시작하지 않는다

요즘의 서버는 클라이언트 요청과 관련된 알림이나, 클라이언트가 연 `subscriptions/listen` 스트림 위의 알림을 보낼 수 있다. 그러나 자기 JSON-RPC 요청을 먼저 보내서는 안 된다.

핸들러에 샘플링이나 사용자 되묻기, 루트 입력이 필요하면 `input_required` 결과를 돌려준다. 클라이언트가 거기 담긴 입력 요청을 채워서, 새 요청 식별자로 원래 메서드를 다시 부른다. 여러 번 왕복하는 그 패턴은 11번 레슨에서 다룬다.

### 구판 호환은 명시적으로

두 시대를 함께 다루는 서버라면 `2025-11-25` 핸드셰이크를 분명히 분리된 구판 갈래에 구현할 수도 있다. 필수 최신 `_meta` 필드가 있으면 최신 동작을 고르고, `initialize`를 받으면 구판 동작을 고른다.

`2026-07-28` 요청을 구판 핸드셰이크 경로에 넣지 마라. 구판 초기화 결과에 최신 `resultType` 필드를 찍어 붙이지도 마라. 이 레슨의 코드는 불변 조건이 눈에 보이도록 일부러 최신 방식만 다룬다.

```figure
t3-dispatch-loop
```

## 실제로 써 보기 (Use It)

Python 서버의 정해진 만큼만 도는 데모와 테스트를 실행한다.

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

TypeScript 판본은 TypeScript 실행기로 돌린다.

```bash
npx tsx main.ts --demo
```

데모는 `server/discover`를 보내고, 기본 요소를 각각 나열하고, 도구를 호출하고, 지원하지 않는 판본 오류를 보여 준다. 요즘 방식의 요청은 모두 메타데이터를 되풀이해 담는다. 성공한 결과는 모두 서버 신원을 담는다.

## 결과물로 남기기 (Ship It)

이 레슨은 `outputs/skill-mcp-server-scaffolder.md`를 남긴다. 탐색 계약과 요청별 검증, 순서가 일정한 캐시 가능 목록, 그리고 선택적으로 격리된 구판 어댑터를 갖춘 최신 서버 설계를 만들어 낸다.

## 연습 문제 (Exercises)

1. 요청 하나에서 역량을 지우고, 서버가 직전 요청의 선언을 다시 쓰지 않는다는 것을 증명하라.
2. `TOOLS`와 `PROMPTS`, 메모를 넣는 순서를 뒤집어라. 모든 목록 결과가 그대로 안정되어 있는지 확인하라.
3. 파괴적인 `notes_delete` 도구를 추가하고, 실행기 안에서 인가를 확인하도록 요구하라. `destructiveHint`는 사용자 경험을 위한 힌트로만 두어라.
4. `resources/templates/list`를 `ttlMs`와 `cacheScope`, 일정한 순서와 함께 추가하라.
5. `2025-11-25`용 구판 어댑터를 따로 만들어라. 그리고 최신 요청이 그쪽으로 들어가지 않는다는 것을 테스트로 증명하라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| Stateless server | 프로토콜 세션 기억 없이, 요청마다 그 자신의 메타데이터만으로 처리하는 서버 |
| `server/discover` | 판본과 역량을 알리는, 요즘 방식의 필수 메서드 |
| Complete result | `resultType: "complete"`인 요즘 방식의 성공 결과 |
| Cacheable result | `ttlMs`와 `cacheScope`가 붙은 탐색·목록·자원 읽기 결과 |
| Deterministic list | 같은 논리적 목록이 언제나 같은 항목 순서를 내놓는 것 |
| Server identity | 결과 `_meta`에 담는, 권장되는 `io.modelcontextprotocol/serverInfo` |
| Tool error | 유효한 도구 호출이 `isError: true`인 콘텐츠를 돌려주는 경우 |
| Protocol error | 잘못된 JSON-RPC나 MCP 요청을 `error`로 돌려주는 경우 |

## 더 읽을거리 (Further Reading)

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
