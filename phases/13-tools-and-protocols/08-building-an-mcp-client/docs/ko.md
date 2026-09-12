# MCP 클라이언트 만들기: 탐색, 라우팅, 두 시대 폴백 (Building an MCP Client: Discovery, Routing, and Dual-Era Fallback)

> 현대 MCP 클라이언트는 요청마다 자신의 계약을 반복해서 보낸다. 가장 어려운 호환성 판단은 오래된 서버가 정말로 오래된 것인지, 아니면 현대 서버가 고칠 수 있는 오류를 보고한 것인지 구별하는 일이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## 학습 목표 (Learning Objectives)

- 모든 MCP `2026-07-28` 요청을 현재 메타데이터와 함께 구성한다.
- `server/discover`로 stdio 서버를 탐지하고 양쪽이 지원하는 버전을 고른다.
- 명시적으로 허용 목록에 올린 상대에게만 제한된 레거시 탐지를 허가한다.
- 지원 개정판에 대한 긍정적 `initialize` 결과를 검증한 뒤에만 레거시 시대를 받아들인다.
- 이름 충돌을 조용히 덮어쓰지 않으면서 결정적인 도구 목록을 병합한다.
- 프로토콜 세션을 지어내지 않고 각 도구를 소유한 상대에게 호출을 라우팅한다.

## 문제 (The Problem)

에이전트 호스트는 보통 MCP 서버를 여러 개 상대한다. 각 서버를 탐색하고, 도구 카탈로그를 병합하고, 중복된 이름을 해결하고, 호출을 라우팅하고, 전송 장애에서 복구해야 한다.

`2026-07-28` 개정판은 요청마다 자기완결적이기 때문에 정상 운영 상태는 더 단순해진다. 복잡해지는 쪽은 시작 시점의 호환성이다. 클라이언트가 마주칠 수 있는 경우는 이렇다.

- 선호 버전을 지원하는 현대 서버,
- 인식 가능한 버전 오류나 헤더 오류를 돌려주는 현대 서버,
- `server/discover`를 한 번도 들어본 적 없는 레거시 서버,
- `initialize`를 받을 때까지 침묵하는 레거시 서버.

탐지 오류를 전부 레거시로 취급하는 것은 위험하다. 잘못 만든 현대 요청, 과부하 상태인 서버, 죽은 프로세스, 오래된 서버가 모두 똑같은 타임아웃이나 연결 종료를 만들어 낸다. 이 신호들은 애매하다. 클라이언트는 명시적인 운영자 의도와 긍정적인 프로토콜 증거를 함께 확보한 뒤에야 레거시 시대를 선택해야 한다.

## 개념 (The Concept)

### 프로토콜 세션이 아니라 상대 (A peer, not a protocol session)

서버 프로세스나 엔드포인트마다 전송 상대 레코드를 하나씩 유지한다.

- 전송 핸들 또는 전송 함수,
- 선택한 프로토콜 시대와 버전,
- 마지막으로 탐색한 서버 역량,
- 마지막 결정적 도구 목록,
- 상관 관계를 맞추기 위한 대기 중 요청 id,
- 전송 상태.

이것은 클라이언트의 장부일 뿐이다. 프로토콜 세션 상태가 아니다. 현대 MCP에서 서버는 여전히 요청마다 현재 버전과 역량을 받는다.

### 모든 현대 요청을 처음부터 구성한다 (Build every modern request from scratch)

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

메타데이터를 연결 객체에 한 번 붙여 놓고 그것이 전선까지 갔으리라고 가정하지 마라. 최종 직렬화된 요청에 도장을 찍고 그 결과를 확인하라.

### 현대 방식 탐색 (Modern discovery)

`server/discover`는 지원 버전, 서버 역량, 사용 지침, 캐시 힌트, 권장 서버 신원을 돌려준다. 클라이언트는 양쪽이 지원하는 현대 버전 중 가장 높은 것을 고른다.

현대 방식만 쓰는 클라이언트에게 탐색은 선택 사항이지만 stdio에서는 권장된다. 일부 레거시 서버는 초기화 전에도 작업을 받아 주기 때문에 `tools/list`를 먼저 보내면 애매한 성공이 나올 수 있다. `server/discover`는 시대 경계를 깔끔하게 그어 준다.

### stdio 호환성 탐지 (The stdio compatibility probe)

두 시대를 다루는 stdio 클라이언트는 다른 어떤 요청보다 먼저 선호하는 현대 메타데이터를 실어 `server/discover`를 보낸다. 결과는 세 종류로 나뉜다.

1. **DiscoverResult.** 서버가 현대 방식이다. 양쪽이 지원하는 버전을 고르고 요청마다 메타데이터를 실어 계속 진행한다.
2. **인식 가능한 현대 오류.** 서버가 현대 방식이다. `-32022`라면 `data.supported`에서 골라 새 요청 id로 재시도한다. 헤더 오류나 역량 오류라면 요청을 고친다. `initialize`를 보내지 마라.
3. **애매한 신호.** 인식하지 못하는 JSON-RPC 오류, 타임아웃, 연결 종료, 빈 응답은 시대를 식별해 주지 않는다. 바로 그 상대가 레거시 호환용으로 설정돼 있지 않다면 막는 쪽으로 실패하라.

인식 가능한 현대 프로토콜 오류는 다음과 같다.

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

인식 가능한 현대 오류는 상대가 레거시 허용 목록에 있더라도 여전히 현대다. 서버가 현대 오류 어휘를 이해한다는 사실을 증명한 이상, `initialize`를 보내는 것은 등급을 낮추는 행위가 된다.

`-32601`을 긍정적인 레거시 증거로 취급하지 마라. 그것은 명시적으로 허용 목록에 오른 상대에게 레거시 탐지 한 번을 받을 자격을 줄 뿐이다. 타임아웃, 연결 종료, 빈 응답에도 같은 규칙이 적용된다.

### 허용 목록은 증거가 아니라 운영자 의도다 (Allowlisting is operator intent, not evidence)

레거시 호환성은 고정된 상대 설정 하나에 명시적으로 붙는 속성이어야 한다.

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

그 선택을 설정된 명령이나 엔드포인트에 묶어라. 임의의 서버가 스스로 약한 의미 체계를 골라 들어오게 하는 와일드카드를 쓰지 마라. `allow_legacy=True`가 없는 상대는 탐색 결과가 애매하면 실패하고 `initialize`를 절대 받지 않는다.

허용 목록은 탐지할 권한을 줄 뿐 시대를 고르지는 않는다. 클라이언트는 전송 계층이 강제하는 기한 안에서 `initialize`를 한 번 보내고, 다음을 전부 요구한다.

- 요청 id가 일치하는 JSON-RPC `2.0` 응답,
- `error` 없이 정확히 하나의 `result`,
- 클라이언트가 설정한 레거시 개정판 집합에 속하는 `protocolVersion`,
- 객체 값을 가진 `capabilities` 필드,
- 비어 있지 않은 문자열 `name`과 `version` 필드를 가진 `serverInfo` 객체.

타임아웃, 연결 종료, 오류 응답, 형식이 어긋난 결과, id 불일치, 지원하지 않는 개정판은 모두 막는 쪽으로 실패한다. 구조가 유효한 긍정 결과만이 레거시 시대를 선택한다. 코드는 `legacy_probe_timeout_ms`를 전송 어댑터에 넘기며, 실제 stdio나 HTTP 어댑터는 그 기한을 기록만 하지 말고 강제해야 한다.

선택한 시대는 전송 상대별로 캐시한다. 호출할 때마다 다시 탐지하지 마라.

### 레거시는 호환성 분기다 (Legacy is a compatibility branch)

제한된 탐지가 유효한 긍정적 레거시 증거를 돌려주면, 클라이언트는 선택한 레거시 버전을 그 개정판이 정의한 대로 정확히 사용한다.

1. 응답 봉투와 상관 id를 검증한다.
2. 협상된 개정판이 설정된 레거시 집합에 있는지 검증한다.
3. 검증된 역량과 서버 신원을 기록한다.
4. 모든 검사를 통과한 뒤에만 `notifications/initialized`를 보낸다.
5. 그 전송이 살아 있는 동안 레거시 요청 형태를 사용한다.

이 분기는 알려진 상대와의 상호 운용을 위해 존재한다. 새 서버나 새 요청의 기본 설계가 아니다. 전송이 재시작되거나 엔드포인트가 바뀌면 상대별 시대 캐시를 버리고 다시 협상한다.

### 도구 탐색과 캐싱 (Discovering and caching tools)

활성 상대마다 `tools/list`를 호출한다. 현대 결과에는 `resultType`, `ttlMs`, `cacheScope`가 들어 있다. 올바른 인가 맥락 안에서 신선도 힌트를 존중하라. 만료된 뒤나 구독한 목록 변경 이벤트를 받은 뒤에 다시 가져온다.

클라이언트는 레거시 서버가 `resultType`을 빼먹었을 때 그것을 `"complete"`로 취급해야 한다. 앞선 시대로 협상된 응답에 현대 캐시 필드를 요구하지 마라.

서버는 결정적인 순서로 돌려줘야 한다. 클라이언트도 병합 전에 정렬해서 지역 레지스트리 순서가 프로세스 시작 타이밍에 좌우되지 않게 해야 한다.

### 충돌에 안전한 네임스페이스 병합 (Collision-safe namespace merge)

서버 두 개가 모두 `search`를 노출할 수 있다. 선언된 정책 하나를 고르라.

1. **충돌 시 접두사.** 첫 번째 정식 이름을 유지하고 이후 충돌은 `<server>/<tool>`로 노출한다.
2. **충돌 시 거부.** 중복된 것을 적재하지 않고 설정 오류를 분명하게 드러낸다.
3. **조용한 덮어쓰기.** 절대 쓰지 마라. 모델이 고른 동작을 어느 서버가 받는지 감춰 버린다.

정식 이름과 지역 이름을 둘 다 저장한다. 모델은 정식 이름을 본다. 나가는 `tools/call`은 소유 서버가 선언한 지역 이름을 쓴다.

### 호출 라우팅 (Routing a call)

라우팅은 순수한 조회다.

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

소유 전송이 사용할 수 없는 상태라면 호출을 보내지 마라. 전송을 재연결하거나 재시작한 뒤 탐색과 `tools/list`를 다시 실행한다. 끊어진 전송에서 유실된 현대 요청은 그 작업의 안전 정책이 허락하는 한 새 JSON-RPC id로 재시도할 수 있다.

### 알림과 구독 (Notifications and subscriptions)

현대 방식에서 목록 변경과 리소스 변경은 클라이언트가 연 `subscriptions/listen` 스트림으로만 도착한다. 클라이언트는 알림 필터를 보내고 `notifications/subscriptions/acknowledged`를 기다린 다음, 알림 메타데이터에 담긴 listen 요청 id로 이벤트를 맞춰 본다.

연결이 끊기면 새 listen 요청을 열고 관련 목록이나 리소스를 다시 가져온다. 현대 스트림은 `Last-Event-ID`로 이어지지 않는다.

### 서버가 먼저 요청하지 않는다 (No server-initiated requests)

현대 서버는 샘플링, 유도, 루트를 위해 독립적인 JSON-RPC 요청으로 클라이언트를 호출하지 않는다. 서버는 `input_required`를 돌려주고, 클라이언트는 안에 담긴 입력 요청을 채운 뒤 원래 요청을 재시도한다.

입력을 채우는 동안 상대의 응답 판독기를 막지 마라. 상관 관계를 유지하고 재시도에는 새 JSON-RPC id를 만들어라.

```figure
tp-client-merge
```

## 직접 해 보기 (Use It)

`code/main.py`는 프로토콜 판단이 눈에 보이도록 프로세스 안에서 도는 상대 함수를 쓴다. 현대 상대 둘과 의도적으로 허용 목록에 올린 레거시 상대 하나에 연결한 다음, 그들의 도구를 병합하고 라우팅한다. 전송 호출 가능 객체는 타임아웃 예산을 받으므로 호환성 분기가 기한 없는 탐지를 숨길 수 없다.

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

테스트는 보통의 예제가 놓치는 경계를 증명한다.

- 현대 요청은 메타데이터를 반복해서 싣는다,
- `-32022`는 초기화 없이 현대 탐색을 재시도한다,
- 인식 가능한 현대 오류는 허용 목록에 오른 상대라도 등급을 낮추지 않는다,
- 타임아웃, 연결 종료, 빈 응답, 인식하지 못하는 오류는 허용 목록 없이 `initialize`를 부르지 않는다,
- 허용 목록에 오른 상대는 유효하고 지원되는 `initialize` 결과가 나온 뒤에야 레거시가 된다,
- 형식이 어긋나거나 지원하지 않는 레거시 결과는 상대를 사용 불가 상태로 남긴다,
- 성공적으로 선택한 시대는 전송이 살아 있는 동안 캐시된다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-mcp-client-harness.md`를 만든다. 현대 요청 도장 찍기, stdio 시대 협상, 결정적 네임스페이스 병합, 라우팅, 막는 쪽으로 실패하는 레거시 호환성 분기의 뼈대를 잡아 준다.

## 연습 문제 (Exercises)

1. 가짜 서버가 양쪽이 지원하는 버전 없이 `-32022`를 돌려주게 하라. 클라이언트가 `initialize`를 보내는 대신 실패하는지 확인하라.
2. 가짜 레거시 서버를 허용 목록에 올리고 제한된 `initialize` 탐지를 타임아웃시켜서, 그 상대가 `unknown`이자 사용 불가 상태로 남는지 증명하라.
3. 인가 맥락 두 개에 대해 `cacheScope: "private"` 도구 목록을 추가하라. 한쪽 맥락의 캐시 결과를 다른 쪽과 절대 공유하지 않는지 확인하라.
4. 충돌 정책을 거부로 바꾸고, 오류 메시지에 상대 이름이 둘 다 담긴 채로 시작이 실패하게 하라.
5. 유한한 `subscriptions/listen` 시뮬레이터를 추가하라. 스트림이 끊기면 새 요청 id로 다시 listen하고 도구를 다시 가져오라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| Peer | 서버 전송 하나와 그로부터 탐색한 데이터를 담는 클라이언트 쪽 레코드 |
| Protocol era | 요청마다 메타데이터를 싣는 현대 방식 또는 초기화 의미 체계를 쓰는 레거시 방식 |
| Discovery probe | stdio 시대를 식별하는 데 쓰는 최초의 `server/discover` |
| Recognized modern error | 현대 동작을 증명하며 레거시 폴백을 금지하는 오류 |
| Legacy allowlist | 고정된 상대에게 제한된 호환성 탐지 한 번을 허가하는 운영자 설정 |
| Positive legacy evidence | 명시적으로 지원하는 레거시 개정판에 대한, 유효하고 상관이 맞는 `initialize` 결과 |
| Merged namespace | 활성 상대 전체에 걸친 정식 도구 이름 |
| Collision policy | 중복된 도구 이름에 적용하는 접두사 또는 거부 규칙 |
| Era cache | 전송 상대 하나에 대해 저장해 둔 현대 또는 레거시 동작 선택 |
| Transport recovery | 재시작하거나 재연결하고, 다시 탐색하고, 다시 목록을 받고, 새 id로 안전하게 재시도하는 일 |

## 더 읽을거리 (Further Reading)

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
