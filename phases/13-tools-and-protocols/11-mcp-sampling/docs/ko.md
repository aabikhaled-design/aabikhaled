# MCP 모델 입력: 샘플링 이전과 무상태 MRTR (MCP Model Input: Sampling Migration and Stateless MRTR)

> MCP 2026-07-28은 새 설계에서 샘플링을 권장하지 않기로 하고, 서버가 클라이언트를 향해 요청을 보내던 경로를 없앴다. 기존 작업 흐름이 여전히 클라이언트의 모델을 필요로 한다면, 서버는 `input_required` 결과를 돌려주고 클라이언트가 모델 출력을 실어 원래 요청을 재시도한다. 추론 루프는 프로토콜 계층에서 명시적이고, 한계가 정해져 있고, 무상태가 된다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 학습 목표 (Learning Objectives)

- MCP 2026-07-28에서 샘플링이 왜 권장되지 않는지 설명하고, 새 서버에는 모델을 직접 붙이는 기본값을 고른다.
- `sampling/createMessage`를 Multi Round-Trip Request(MRTR)로 실어 나르는 호환성 작업 흐름을 구현한다.
- 프로토콜 개정판과 클라이언트 역량을 모든 요청의 `_meta` 객체에 넣는다.
- `resultType: "input_required"`를 돌려주고 새 JSON-RPC id로 원래 메서드를 재시도한다.
- `requestState`의 무결성을 보호하고 주체, 메서드, 인자, 만료에 묶는다.
- 역량 검사, 승인, 응답 검증, 라운드 상한으로 모델을 쓰는 루프에 한계를 건다.

## 프로토콜보다 먼저 해야 할 결정 (The Decision Before the Protocol)

`summarize_repo` 같은 도구에는 두 종류의 일이 필요하다.

1. 결정적인 일: 파일을 나열하고, 허용된 파일을 읽고, 경로를 검증하고, 내용을 모은다.
2. 모델의 일: 대표가 될 파일을 고르고 요약을 지어낸다.

이제 유효한 구조가 두 가지다.

### 새 서버: 모델 제공자와 직접 통합한다 (New server: integrate with a model provider directly)

이것이 현재의 기본값이다. 서버가 모델 선택, 자격 증명, 예산, 재시도, 관측 가능성을 직접 쥔다. MCP 클라이언트에는 평범한 `tools/call` 결과 하나를 돌려준다.

서버가 이미 호스팅되는 서비스이거나, 호스트의 모델을 쓰는 것보다 예측 가능한 모델 동작이 더 중요할 때 이쪽을 고르라.

### 기존 샘플링 작업 흐름: MRTR로 옮긴다 (Existing Sampling workflow: migrate it to MRTR)

샘플링은 폐기 예고 기간 동안 아직 남아 있다. 2026-07-28을 대상으로 하는 서버는 클라이언트에게 살아 있는 `sampling/createMessage` 요청을 되돌려 보낼 수 없다. 대신 그 요청을 `InputRequiredResult` 안에 담는다.

클라이언트의 모델과 자격 증명을 쓰는 것이 진짜 제품 요구사항일 때만 이 호환 경로를 고르라. 새 구현은 폐기 예정인 샘플링을 채택하면 안 되므로 제거 계획을 함께 기록해 두라.

## 무상태 계약 (The Stateless Contract)

2026년 7월 프로토콜에는 `initialize` 교환도, `notifications/initialized`도, `Mcp-Session-Id`도 없다. 악수에 담겨 있던 정보를 이제 모든 요청이 싣고 다닌다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

서버는 요청마다 개정판을 검증한다. 버전이 없거나 문자열이 아니면 잘못된 매개변수, 즉 `-32602`다. 지원하지 않는 문자열에는 정확한 데이터 `{"supported":["2026-07-28"],"requested":"<client version>"}`와 함께 `-32022`를 돌려준다. 샘플링 역량이 빠져 있으면 `data.requiredCapabilities`를 `{"sampling":{}}`로 채워 `-32021`을 돌려준다.

JSON-RPC `id`가 없는 봉투는 알림이다. 수신 측은 그것을 처리해도 되지만, 성공 응답도 오류 응답도 내보내지 않는다. Streamable HTTP 어댑터는 받아들인 알림에 본문 없이 `202 Accepted`를 돌려준다.

서버는 `server/discover`도 구현하며, 클라이언트가 도구를 호출하기 전에 서버 계약을 알고 캐시할 수 있도록 `supportedVersions` 키와 역량, `ttlMs`, `cacheScope`를 정확히 담아 돌려준다. 탐색이 `tools`를 알리므로 서버는 필수인 `tools/list`도 구현한다. 결정적인 `summarize_repo` 서술자에는 유효한 객체 `inputSchema`, `resultType: "complete"`, 서버 신원 메타데이터, 공개 캐시 힌트가 들어간다.

성공한 현대 결과에는 모두 구분자가 붙는다.

- `resultType: "complete"`는 작업이 끝났다는 뜻이다.
- `resultType: "input_required"`는 클라이언트가 안에 담긴 요청을 채우고 재시도해야 한다는 뜻이다.
- 확장이 결과 타입을 더 정의할 수 있다. Tasks 확장은 레슨 13에서 `"task"`를 더한다.

## MRTR 한 라운드 (One MRTR Round)

서버는 요청을 처리하는 도중에 클라이언트를 호출할 수 없다. 대신 이런 결과를 돌려준다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

클라이언트는 자신이 샘플링을 지원하는지 확인하고, 자신의 승인 정책과 모델 정책을 적용해 모델 응답을 얻는다. 그런 다음 다른 JSON-RPC id로 새 요청을 보낸다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

재시도는 프로토콜 세션을 이어 가는 것이 아니다. 원래 메서드와 인자를 그대로 되풀이하고, 이번 라운드의 `inputResponses`만 더하고, `requestState`를 바이트 단위로 그대로 되돌려 보내는 새 요청이다.

MRTR은 `tools/call`, `prompts/get`, `resources/read`에서만 허용된다. 서버는 관계없는 메서드에서 `input_required`를 돌려줘서는 안 된다.

## 여러 라운드에 걸친 상태 (Multi-Round State)

이 레슨에는 모델 호출이 두 번 필요하다.

1. `pick_files`는 JSON 배열을 돌려준다.
2. `summary`는 최종 산문을 돌려준다.

재시도는 그 라운드의 응답만 싣는다. 그래서 서버는 단계와 검증된 중간 데이터를 다음 `requestState`에 넣는다.

그 값은 공격자가 조종할 수 있다고 보라. 단계 이름만 날것으로 서명해서는 부족하다. 상태를 다음에 묶어라.

- 스스로 밝힌 `clientInfo`가 아니라 인증된 주체,
- 요청이 출발한 메서드,
- 원래 인자의 요약값,
- 짧은 만료,
- 현재 단계와 검증된 중간 값.

기밀성이 필요 없으면 HMAC을 쓰라. 클라이언트가 상태를 읽어서는 안 된다면 인증된 암호화를 쓰라. 서명이 틀렸거나, 값이 만료됐거나, 주체가 바뀌었거나, 인자가 바뀌었으면 `-32602`로 거부하라.

클라이언트는 `requestState`를 파싱하거나 고쳐서는 안 된다. 재시도에 정확히 같은 문자열을 되돌려 보내는 것이 유일한 역할이다.

## 모델 선호도는 힌트다 (Model Preferences Are Hints)

`costPriority`, `speedPriority`, `intelligencePriority`는 서로 독립적인 선호도다. 확률 분포가 아니므로 합이 1이 될 필요가 없다. 모델 정책은 클라이언트의 것이므로 클라이언트가 이 값들을 무시해도 된다.

레거시 샘플링 흐름을 유지한다면 `includeContext`를 `"none"`으로 두라. 다른 맥락 모드는 유출 위험을 키우고, 그 자체로도 폐기 예정이다. 요청에는 명시적인 최소 맥락만 실어라.

## 안전 불변식 (Safety Invariants)

안에 담긴 샘플링 요청에 대한 신뢰 경계는 클라이언트다.

- 정책이 승인을 요구하면, 서버가 모델에게 무엇을 시키려 하는지 사용자에게 보여 준다.
- MRTR 라운드에 상한을 걸어라. 그러지 않으면 악의적인 서버가 모델 비용을 태우는 루프를 만들 수 있다.
- 샘플링 응답을 파일 이름, URL, 도구 입력으로 쓰기 전에 반드시 검증하라.
- 라운드마다 바이트와 토큰을 제한하라.
- 현재 클라이언트 역량에 선언되지 않은 입력 요청은 거부하라.
- 모델 출력을 인가 판단에 끌어들이지 마라.
- 민감한 프롬프트 내용은 남기지 말고, 요청이 출발한 메서드와 입력 요청 키만 기록하라.

`clientInfo`와 `serverInfo`는 표시와 진단을 위한 메타데이터다. 둘 중 어느 것도 인증된 신원으로 쓰지 마라.

```figure
t3-sampling-flip
```

## 만들어 보기 (Build It)

`code/main.py`는 서드파티 패키지 없이 두 라운드 흐름 전체를 구현한다.

- `server/discover`는 `supportedVersions`를 돌려주고, 도구 지원을 알리고, 캐시 힌트를 싣는다.
- `tools/list`는 객체 입력 스키마를 가진, 결정적이고 캐시 가능한 `summarize_repo` 서술자를 돌려준다.
- `tools/call`은 요청마다 실린 메타데이터를 검증한다.
- 첫 결과는 파일 선택을 위한 `sampling/createMessage`를 담는다.
- 첫 재시도는 모델 결과를 검증하고 두 번째 요청을 담는다.
- HMAC으로 보호한 `requestState`가 독립적인 요청들 사이로 단계를 실어 나른다.
- 최종 결과는 `resultType: "complete"`를 쓴다.

가짜 호스트 모델이 예제를 결정적으로 만들어 준다. 실제 호스트에 연결할 때는 `fake_host_model`만 바꾸면 된다. 서버 쪽 상태 기계는 결정적이고 테스트 가능한 상태로 두어야 한다.

## 직접 해 보기 (Use It)

저장소 루트에서 시작한다.

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

확인할 지점은 이렇다.

- 탐색이 `ttlMs`와 `cacheScope`를 담은 complete 결과를 돌려준다.
- 도구 탐색이 `resultType`, 서버 신원, 캐시 힌트를 담아 정렬된 같은 서술자를 돌려준다.
- 빠진 역량과 지원하지 않는 버전이 정확한 `-32021`, `-32022` 오류 데이터를 쓴다.
- id 없는 알림은 JSON-RPC 응답을 만들지 않는다.
- 요청 id가 `[1, 2, 3]`이라서 MRTR 라운드마다 독립적임이 드러난다.
- 처음 두 결과는 `input_required`다.
- 최종 결과는 `complete`이며 선택된 파일과 요약을 담는다.
- 재시도에서 원래 인자를 바꾸면 요청 상태 검사에 걸린다.

## 결과물 (Ship It)

`outputs/skill-sampling-loop-designer.md`는 이제 이전 계획 도구다. 먼저 샘플링을 걷어내고 모델을 직접 붙일지부터 판단한다. 호환성이 필요하다면 MRTR 라운드, 상태 결속, 역량 관문, 예산, 검증, 제거 계획을 만들어 준다.

## 연습 문제 (Exercises)

1. 파일 선택 응답을 잘못된 JSON으로 바꿔라. 서버가 모델 출력을 믿는 대신 `-32602`를 돌려주는지 확인하라.
2. 첫 호출과 재시도 사이에 `audience`를 바꿔라. 봉인된 상태가 왜 요청을 넘나드는 재사용을 막는지 설명하라.
3. 호스트에게 요약을 비평해 달라고 하는 세 번째 라운드를 추가하라. 앞선 요약을 서명된 상태 안에 실어 나르고 전체 흐름을 세 라운드로 제한하라.
4. 가짜 호스트 콜백을 서버가 소유한 모델 어댑터로 바꿔 샘플링을 제거하라. 승인, 과금, 관측 가능성 책임 중 무엇이 서버로 넘어오는지 나열하라.
5. 기한을 1초 넘긴 상태 값으로 만료 테스트를 추가하라.

## 핵심 용어 (Key Terms)

| 용어 | 2026-07-28에서의 뜻 |
|------|------------------------|
| Sampling | 클라이언트의 모델에 완성을 요청하는, 폐기 예정 기능 |
| MRTR | 요청 처리 중 클라이언트 입력이 필요할 때 쓰는 무상태 재시도 방식 |
| `InputRequiredResult` | `resultType: "input_required"`를 가진 결과 |
| `inputRequests` | 서버가 키를 붙여 담은 유도, 샘플링, 루트 요청의 묶음 |
| `inputResponses` | `inputRequests`와 같은 키로 담는 이번 라운드의 클라이언트 결과 |
| `requestState` | 클라이언트가 그대로 되돌려 보내고 서버가 검증하는 불투명한 서버 상태 |
| `resultType` | 현대 MCP 결과에 필수인 구분자 |
| Direct model integration | 모델 추론이 필요한 새 서버에 권장되는 대체 방식 |
| Capability gate | 클라이언트가 알리지 않은 요청을 담아 보내지 못하게 막는 규칙 |
| Loop budget | 그 작업에 허용되는 최대 라운드, 토큰, 바이트, 시간, 비용 |

## 레거시 호환성 (Legacy Compatibility)

2025-11-25에 고정된 클라이언트는 살아 있는 연결 위에서 서버가 먼저 보내는 옛 `sampling/createMessage` 흐름을 아직 쓸 수 있다. 그 동작은 버전별 어댑터 안에만 두라. 세션에 기대는 그 경로를 2026-07-28 서버의 구조로 삼지 마라.

공식 SDK는 현대의 `input_required` 처리기를 옛 상대용으로 번역해 줄 수 있다. 그 완충 계층은 호환성 경계일 뿐, 세션에 의존하는 새 로직을 더해도 된다는 허락이 아니다.

## 더 읽을거리 (Further Reading)

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
