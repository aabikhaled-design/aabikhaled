# MCP 보안: 오염된 메타데이터, 라우팅, MRTR 상태 (MCP Security: Poisoned Metadata, Routing, and MRTR State)

> 무상태라는 말이 신뢰가 필요 없다는 뜻은 아니다. 서버와 게이트웨이가 호출을 각자 검증하는 데 필요한 증거를 요청마다 드러낸다는 뜻이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 minutes

## 학습 목표 (Learning Objectives)

- 도구 설명, 주석, 클라이언트 정보, 서버 정보를 신뢰할 수 없는 데이터로 다룬다.
- 메타데이터 오염, 서술자 변경, 서버 간 이름 충돌을 잡아낸다.
- 2026-07-28의 요청 메타데이터와 Streamable HTTP 라우팅 헤더를 검증한다.
- MRTR의 `requestState`를 변조로부터 보호하고 확인을 정확한 인자에 묶는다.
- 사라진 프로토콜 세션이 아니라 주체에 인가와 속도 제한을 적용한다.

## 문제 (The Problem)

모델은 무엇을 호출할지 정하려고 도구 설명을 읽는다. 라우터는 요청을 어디로 보낼지 정하려고 도구 이름을 읽는다. 사용자는 무엇을 승인할지 정하려고 이름표를 읽는다. 악의적인 서술자 하나가 이 셋 모두를 노릴 수 있다.

공식 MCP 보안 지침은 단도직입적이다. 신뢰할 수 있는 서버에서 온 것이 아니라면 설명과 주석은 신뢰할 수 없는 것으로 다뤄야 한다. 신뢰할 수 있는 서버라도 배포 시점의 신뢰는 바뀔 수 있다. 서버 갱신, 침해된 패키지, 레지스트리의 실수, 게이트웨이 병합이 모델이 보는 것을 바꿔 놓을 수 있다.

현행 프로토콜은 보안 경계도 바꿔 놓았다. 2026-07-28에는 코어 악수도 전송 세션도 없다. 승인이나 속도 제한, 감사 이력을 `Mcp-Session-Id`만으로 묶는 보안 설계는 현행 설계가 아니다.

## 개념 (The Concept)

### 확인할 가치가 있는 공격 표면 일곱 가지 (Seven attack surfaces worth checking)

조심하라는 막연한 지시 대신 구체적인 목록을 쓰라.

1. **메타데이터 오염.** 설명에 선언된 도구 동작과 무관한 지시가 들어 있다.
2. **서술자 교체 사기.** 이미 승인된 이름, 설명, 스키마, 주석이 바뀐다.
3. **서버 간 가리기.** 백엔드 둘이 수식 없는 같은 도구 이름을 내놓고 라우팅이 조용히 하나를 고른다.
4. **헤더와 본문 혼동.** `Mcp-Method`나 `Mcp-Name`이 JSON-RPC 요청과 어긋난다.
5. **역량 상승.** 상대가 확장이나 클라이언트 기능을 주장하는데 서버가 그 선언을 인가로 착각한다.
6. **MRTR 상태 변조.** 클라이언트가 `requestState`를 바꾸거나, 다른 질문에 답하거나, 인자를 바꿔 확인을 재사용한다.
7. **공급망 신원 혼동.** 낯익은 표시 이름을 발행자나 서버 신원의 증거로 취급한다.

이 표면들은 서로 겹친다. 해시 고정은 서술자 변경에 도움이 되지만 최초 서술자가 안전했음을 증명하지는 못한다. 정적 스캔은 뻔한 문구를 잡아내지만 은근한 지시는 놓친다. 이름 공간은 충돌 한 종류를 막지만 악의적인 이름 공간 서버는 막지 못한다. 통제를 겹쳐 쌓으라.

### 현행 요청 봉투는 증거이지 신원이 아니다 (The current request envelope is evidence, not identity)

2026-07-28의 모든 요청에는 이런 것이 담긴다.

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

요청마다 버전과 역량의 형태를 검증하라. 역량은 호환되는 응답 형태를 고르는 데 쓰라. `clientInfo`를 인증된 주체로 쓰지 마라. 스스로 밝힌 값이다.

결과 메타데이터의 `io.modelcontextprotocol/serverInfo`에도 같은 경고가 적용된다. 로그와 디버깅에는 쓸모가 있다. 인증서도, 레지스트리 증명도, 인가 판단도 아니다.

### 정책보다 먼저 라우팅을 검증하라 (Validate routing before policy)

`tools/call`이라면 Streamable HTTP에 이런 것이 들어간다.

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

헤더 메서드는 본문 메서드와 같아야 한다. 헤더 이름은 `params.name`과 같아야 한다. 백엔드를 고르거나, RBAC을 적용하거나, 속도 제한 토큰을 쓰기 전에 어긋난 요청을 `-32020`으로 거부하라.

이 순서가 흔한 모호함을 닫는다. 한 구성 요소는 본문으로 인가하고 다른 구성 요소는 헤더로 라우팅하는 상황 말이다.

전선 검증은 정확히 한 순서를 따른다. JSON-RPC와 메타데이터 타입을 검증하고, 헤더 값을 본문과 비교하고, 그다음에 맞춰진 버전을 지원하는지 확인한다. 어긋난 헤더에는 `-32020`과 함께 HTTP 400을 돌려준다. 헤더와 본문이 지원하지 않는 버전에서 일치하면 `-32022`와 함께 HTTP 400을 돌려주고 `data`는 정확히 `{"supported":["2026-07-28"],"requested":"<actual>"}`로 채운다. 알 수 없는 메서드에는 `-32601`과 함께 HTTP 404를 돌려준다.

계약이 구조화된 복구 정보를 필요로 하면 모든 오류 객체가 선택적인 `data`를 담는다. 알림에는 `id`가 없으므로 JSON-RPC 성공 응답도 오류 응답도 받지 않는다. 받아들인 HTTP 알림은 빈 본문과 함께 202를 돌려준다.

### 서술자 전체를 고정하라 (Pin the whole descriptor)

설명만 해시하면 스키마와 주석 변경을 놓친다. 사용자가 승인한 서술자 필드를 정규화해서 해시하라.

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

이 장난감 예제 바깥에서는 요약값을 `notes.export` 같은 수식된 키 아래에 발행자 증거, 승인 시각과 함께 저장하라.

새로 고칠 때마다 이렇게 한다.

- 알 수 없는 키: 검토할 때까지 격리한다.
- 같은 키인데 요약값이 다름: 다시 승인할 때까지 교체 사기로 보고 격리한다.
- 수식 없는 이름이 중복됨: 결정적인 이름 공간을 요구한다.
- 스캐너 적중: 막아 두고 서술자 전체를 검토한다.

해시가 같다는 것은 안정성을 증명할 뿐 안전을 증명하지 않는다. 오염된 서술자는 완벽하게 고정해 두어도 오염된 채다.

### 정적 스캔은 걸림줄이다 (Static scanning is a tripwire)

간단한 패턴만으로도 역할 태그, 지시 덮어쓰기, 은폐, 비밀 값 접근, 가려진 네트워크 목적지를 표시할 수 있다. 설치 시점과 CI에 돌리기에 충분히 값싸다.

하지만 의미론적 증명은 아니다. 안전한 설명도 정당한 경고문 안에 표시 대상 문구를 담을 수 있다. 악의적인 설명은 모든 문구를 피해 갈 수 있다. 스캐너 출력은 자동으로 매겨지는 결백 점수가 아니라 검토용 증거로 다루라.

### 병합하기 전에 이름 공간을 붙여라 (Namespace before merging)

서버 둘이 모두 `search`를 내놓는다고 하자. 탐색 순서가 승자를 정하게 두지 마라.

```text
notes.search
issues.search
```

수식된 이름이 게이트웨이의 공개 이름이다. 백엔드 대응 관계는 따로 기록하라. 이름이 안정적이어야 승인, 감사, 해시 고정, `Mcp-Name` 라우팅이 모두 같은 대상을 가리킨다.

### 역량은 호환성 선언이다 (Capabilities are compatibility declarations)

요청마다 실리는 `clientCapabilities`는 클라이언트가 어떤 프로토콜 기능을 처리할 수 있는지 서버에게 알려 준다. 클라이언트에게 도구나 데이터, 동작에 대한 접근 권한을 주지는 않는다.

인가는 여전히 인증된 주체와 리소스 정책에서 나온다. 순서는 이렇다.

1. 전송 자격 증명을 인증한다.
2. 버전, 헤더, 요청 형태를 검증한다.
3. 역량 호환성을 확인한다.
4. 주체, 도구, 리소스, 인자를 인가한다.
5. 실행하거나 사용자 입력을 요청한다.

### 무상태 MRTR 확인을 보호하라 (Protect stateless MRTR confirmation)

결과가 무거운 도구는 사용자 확인을 필요로 할 수 있다. 현행 MCP는 서버가 클라이언트를 되부르는 대신 Multi Round-Trip Request를 쓴다.

첫 응답은 이렇다.

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

클라이언트는 입력을 받아 새 JSON-RPC id로 원래 메서드를 재시도한다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

`inputRequests`의 각 값은 `method`와 `params`를 갖춘 완결된 내장 요청이다. 그 키는 `inputResponses`의 해당 항목과 맞아야 한다. 폼 유도는 루트가 객체인 `requestedSchema`를 쓰고, 서버가 요청하기 전에 클라이언트가 폼 유도 역량을 선언해 두었어야 한다.

현행 역량에는 유효한 폼 선언이 두 가지 있다. `{"elicitation":{}}`은 폼 유도를 암묵적으로 지원하고, `{"elicitation":{"form":{}}}`은 명시적으로 밝힌다. `{"elicitation":{"url":{}}}`처럼 URL만 선언한 것은 폼 요청을 지원하지 않는다. 서버는 `-32021`과 `{"elicitation":{"form":{}}}`인 `data.requiredCapabilities`를 실어 HTTP 400을 돌려준다.

`requestState`를 적대적인 입력으로 다루라. 서명하거나 암호화하고, 검증하고, 메서드, 도구, 정확한 인자, 목적, 만료, 주체에 묶으며, 재생이 문제가 된다면 일회용 난스에도 묶으라. 이 레슨의 코드는 경계가 눈에 보이도록 HMAC과 정확한 인자 일치 비교를 쓴다.

난스 장부가 게이트웨이 객체 하나 안에 있어서는 안 된다. 실행 가능한 모형은 게이트웨이 인스턴스 여럿이 함께 쓸 수 있는, 크기와 TTL이 정리되는 재생 방지 저장소를 주입한다. 그 원자적 점유가 실행 경계다. 검증된 수락이나 명시적인 종결 거절만이 상태를 소비한다. 형식이 어긋난 응답이나 `cancel`은 아무것도 실행하지 않고, 만료될 때까지 재시도 가능한 채로 남는다. 실제 운영 대수에도 공유 지속 저장소 안에 같은 조건부 점유가 필요하다.

감춰진 확인 맥락을 프로토콜 세션에 저장하지 마라. 어느 서버 인스턴스든 재시도를 검증할 수 있어야 한다.

### 위험이 큰 호출에는 둘의 규칙 (Rule of two for high-risk calls)

호출을 세 축으로 분류하라.

- 신뢰할 수 없는 입력을 소비한다.
- 민감한 데이터에 접근할 수 있다.
- 결과가 무거운 외부 동작을 일으킨다.

자동으로 도는 한 단계가 이 셋을 모두 겸해서는 안 된다. 단계를 쪼개거나, 권한을 줄이거나, MRTR로 사용자 입력을 명시적으로 요청하라. 이것은 설계 어림법이지 프로토콜 역량이 아니다.

### 실행 전에 권한을 줄여라 (Reduce authority before execution)

무상태라는 것만으로는 안전하지 않다. 감춰진 프로토콜 이력을 없앨 뿐, 자기완결적인 요청도 힘이 지나친 처리기에게 데이터를 흘리거나 되돌릴 수 없는 변경을 하라고 시킬 수 있다. 안전은 경계마다 권한을 줄이는 데서 나온다.

1. **타입이 정해진 동사.** `archive_note`처럼 범위가 한정된 작업 하나를 내놓고, 무관한 힘까지 표현할 수 있는 범용 `run`이나 `request` 도구를 두지 마라.
2. **검증된 인자.** 가능하면 닫힌 스키마를 쓰고, 모르는 필드를 거부하고, 식별자를 한 번 정규화하고, 크기에 상한을 걸고, 정책을 평가하기 전에 목적지와 테넌트와 리소스 소유권을 검증하라.
3. **현재의 인가.** 인증된 주체를 정확한 동사, 리소스, 환경, 정규화된 인자에 묶으라. 도구 주석과 클라이언트 역량은 이 권한을 주지 않는다.
4. **동작에 묶인 승인.** 결과가 무거운 호출이라면 승인을 타입이 정해진 동사와 정규화된 인자의 요약값에, 그리고 주체, 만료, 일회성 정책에 묶으라. 필드가 하나라도 바뀌면 새로 판단해야 한다.
5. **일급 시민으로서의 거절.** 정책 거부, 만료된 승인, 사용자 거절, 안전하지 않은 목적지를 부수 효과가 전혀 없는 평범한 결과로 모형화하라. 거절을 더 약한 대체 도구로 번역하지 마라.
6. **가린 감사 증거.** 누가 요청했는지, 어떤 승인된 서술자와 정책 버전을 썼는지, 어떤 정규화된 대상이 인가됐는지, 왜 허용하거나 거절했는지, 실행이 시작됐는지를 기록하라. 비밀 값 대신 요약값이나 가린 값을 저장하라.

각 단계는 다음 구성 요소가 할 수 있는 일을 좁힌다. 최종 처리기는 날것의 모델 텍스트와 넓은 자격 증명이 아니라 이미 검증된 도메인 명령을 받아야 한다. MRTR 재시도, 태스크 갱신, 게이트웨이가 넘겨준 호출에서도 이 사슬 전체를 되풀이하라. 앞선 승인이 이후 요청을 신뢰받는 세션 트래픽으로 바꿔 주지는 않는다.

### 현행 상호작용 경로와 레거시 경로 (Current and legacy interaction paths)

루트, 샘플링, 로깅은 새 2026-07-28 구현에서 권장되지 않는다. 게이트웨이는 예전 요청 채널 코드를 버전으로 걸러진 호환 경로로만 남겨 둘 수 있다.

세션별 샘플링 제한기 위에 새 방어를 쌓지 마라. 인증된 주체, 발급자, 리소스, 도구, 시간 구간에 할당량을 적용하라. 지금의 대화형 작업은 MRTR의 입력 요청과 응답을 들여다보면 된다.

### 무상태 전송 점검 (Stateless transport checks)

- 현대 MCP 메시지를 단일 POST 엔드포인트에서 받는다.
- 현대 방식의 GET과 DELETE에는 405를 돌려준다.
- `Mcp-Session-Id`를 발급하지도, 그것에 기대지도 않는다.
- 레거시 세션 헤더와 재생 헤더를 권한 입력으로 삼지 않는다.
- 그 POST에 JSON이나 요청 범위 SSE를 돌려준다.
- `subscriptions/listen`은 신청한 장기 변경 알림에만 쓴다.

```figure
tp-tool-poisoning
```

## 만들어 보기 (Build It)

`code/main.py`는 프로세스 안에서 도는 작은 보안 게이트웨이 모형을 구현한다. 도구 서술자 전체를 정규화해 고정하고, 메타데이터 오염과 가리기를 보고하고, 현대 요청 봉투와 라우팅 값을 검증하고, 서명된 `requestState`와 주입된 공유 재생 방지 저장소로 두 라운드에 걸친 확인된 내보내기를 수행한다.

이 모형은 HTTP 어댑터가 JSON 본문과 라우팅 헤더를 이미 파싱한 다음부터 시작한다. `Content-Type`이나 `Accept`는 검증하지 않는다. 같은 디스패처를 레슨 09의 완전한 Streamable HTTP 어댑터에 붙이라. 그 어댑터는 `Content-Type: application/json`과 `application/json`, `text/event-stream`을 모두 담은 `Accept` 값을 요구한다.

실행은 이렇게 한다.

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

예제는 일부러 서술자를 바꿔 놓는다. 스캐너와 요약값 비교가 서로 독립적인 발견을 내놓는다. 그다음 내보내기가 `input_required` 응답과 무상태 재시도를 보여 준다.

## 직접 해 보기 (Use It)

`SAFE_TOOLS`를 여러분이 승인한 서버에서 뜬 정규화된 스냅샷으로 바꾸라. 자격 증명과 비밀 값은 스냅샷에 넣지 마라. 새로 생기거나 바뀐 서술자는 요약값을 갱신하기 전에 모두 검토하라.

게이트웨이에서는 탐색 때 같은 검사를 돌리고 디스패치 직전에 한 번 더 돌려라. 캐시가 탐색 부담을 줄여 줄 수는 있지만, 캐시된 승인은 서술자가 바뀌면 만료되거나 무효화돼야 한다.

## 결과물 (Ship It)

이 레슨은 `outputs/skill-mcp-threat-model.md`를 만든다. 메타데이터, 라우팅, 역량, 인가, MRTR, 캐싱, 레지스트리, 호환성 경계를 아우르는 현행 프로토콜 위협 모형을 만들어 준다.

## 연습 문제 (Exercises)

1. 인증된 주체와 현재의 인가 판단을 봉인된 MRTR 상태에 묶은 다음, 다른 주체가 보낸 재시도를 거부하라.
2. 메모리 재생 방지 저장소를 지속되는 조건부 삽입으로 바꾸고, 프로세스 둘이 난스 하나를 둘 다 점유할 수 없음을 증명하라.
3. 재생 점유 뒤 모의 내보내기 전에 실패를 주입하라. 복구를 안전하게 만드는 트랜잭션 규칙이나 멱등성 규칙을 정의하고 시험하라.
4. 도구의 설명은 그대로 두고 `inputSchema`만 바꿔라. 서술자 전체 고정이 그것을 잡아내는지 확인하라.
5. `tools/list`가 주체마다 달라지면 공개 캐싱을 거부하는 정책을 추가하라.
6. 게이트웨이 뒤에 있는 옛 서버를 모형화하라. 악수와 세션 동작을 전부 명시적인 `2025-11-25` 호환 분기 안에 넣으라.

## 핵심 용어 (Key Terms)

| 용어 | 뜻 |
|------|---------|
| Metadata poisoning | 도구 서술자에 심어 둔 지시나 사람을 속이는 주장 |
| Rug pull | 이미 승인된 서술자에 가해진 변경 |
| Tool shadowing | 수식 없는 이름이 겹쳐서 생기는 모호한 라우팅 |
| Header mismatch | 라우팅 헤더와 JSON-RPC 본문의 불일치, 오류 `-32020` |
| Hash pin | 승인된 서술자 전체의 요약값 |
| MRTR | 서버가 요청한 입력을 위한 무상태 응답과 재시도 방식 |
| `requestState` | 신뢰할 수 없는 입력으로 다뤄야 하는, 오가는 불투명한 값 |
| Capability declaration | 인가가 아니라 프로토콜 호환성에 대한 진술 |
| Implicit form support | 폼 지원과 같은 뜻인 빈 `elicitation` 역량 객체 |
| Qualified tool name | `notes.search` 같은 안정적인 게이트웨이 이름 |

## 더 읽을거리 (Further Reading)

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
