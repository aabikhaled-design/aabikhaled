# 사람 개입(Human-in-the-Loop): 제안-후-커밋(Propose-Then-Commit)

> HITL(human-in-the-loop)에 대한 2026년 합의는 구체적이다. 그것은 "에이전트(agent)가 묻고, 사용자가 승인을 클릭한다"가 아니다. 제안-후-커밋(propose-then-commit)이다: 제안된 액션이 멱등성 키(idempotency key)와 함께 지속 저장소(durable store)에 지속되고; 의도(intent), 데이터 계보(data lineage), 건드린 권한, 폭발 반경(blast radius), 롤백 계획과 함께 검토자에게 표출되며; 긍정적 확인 후에만 커밋되고; 실행 후 부작용이 실제로 일어났는지 검증된다. LangGraph의 `interrupt()` + PostgreSQL 체크포인팅, Microsoft Agent Framework의 `RequestInfoEvent`, 그리고 Cloudflare의 `waitForApproval()`은 모두 같은 형태를 구현한다. 정형적 실패 양상은 고무 도장(rubber-stamp) 승인이다: 검토 없이 "Approve?"가 클릭된다. 문서화된 완화책은 명시적 체크리스트를 갖춘 챌린지-앤-리스폰스(challenge-and-response)다.

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60분

## 문제 (The Problem)

에이전트가 액션을 취한다. 사용자는 승인할지 말지 결정해야 한다. 결정이 즉각적이면 검토라고 보기 어렵다. 결정이 구조화되어 있으면 느리지만 신뢰할 만하다. 엔지니어링 질문은 구조화된 검토를 어떻게 최소 저항 경로로 만드느냐다.

2023년 시대의 HITL 패턴은 동기식 프롬프트(prompt)였다: "에이전트가 본문 Y로 X에게 이메일을 보내려 한다. 승인?" 사용자가 승인을 클릭한다. 모두가 시스템이 안전하다고 느낀다. 하지만 이 표면은 심하게 고무 도장된다. 사용자는 빠르게 승인하고, 승인은 거의 예측력이 없으며, 에이전트가 잘못되면 감사 추적(audit trail)은 사용자가 기억하지 못하는 긴 승인 이력만 보여준다.

2026년 패턴(제안-후-커밋)은 HITL을 지속 기반(durable substrate) 위로 옮기고, 구조화된 메타데이터를 첨부하며, 긍정적 커밋을 요구한다. 모든 관리형 에이전트 SDK가 한 버전을 출하한다: LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. API 이름은 다르지만, 형태는 다르지 않다.

## 개념 (The Concept)

### 제안-후-커밋 상태 기계

1. **제안(Propose).** 에이전트가 제안된 액션을 생성한다. 지속 저장소(PostgreSQL, Redis, Durable Object)에 지속된다. 포함 사항:
   - 의도(에이전트가 왜 이것을 하는가)
   - 데이터 계보(어떤 출처가 이 제안으로 이어졌는가)
   - 건드린 권한(어떤 범위/파일/엔드포인트)
   - 폭발 반경(최악의 경우는 무엇인가)
   - 롤백 계획(커밋되면 어떻게 되돌리는가)
   - 멱등성 키(제안마다 고유함; 재제출 시 같은 레코드를 반환)
2. **표출(Surface).** 검토자가 모든 메타데이터와 함께 제안을 본다. 검토자는 사람이다(에이전트가 자기 자신을 검토하는 것이 아님).
3. **커밋(Commit).** 긍정적 확인. 액션이 실행된다.
4. **검증(Verify).** 실행 후, 부작용이 다시 읽혀 확인된다. 검증 단계가 실패하면, 시스템은 알려진 나쁜 상태에 있고 경보가 작동한다.

### 멱등성 키

멱등성 키가 없으면 일시적 실패 후 재시도가 승인된 액션을 이중 실행한다. 구체적 예를 보자. 사용자가 "A에서 B로 100달러 이체"를 승인한다. 네트워크가 깜빡인다. 워크플로가 재시도한다. 사용자는 한 번 승인했지만 이체가 두 번 실행된다. 멱등성 키는 승인을 단일하고 고유한 부작용에 묶는다. 두 번째 실행은 무연산(no-op)이다.

이것은 Stripe와 AWS API가 쓰는 것과 같은 멱등성 패턴이다. 이를 에이전트 승인에 재사용하는 것이 Microsoft Agent Framework 문서에 명시되어 있다.

### 지속성: 승인이 프로세스보다 오래 사는 이유

승인 대기실은 에이전트가 소유하지 않는 상태 조각이다. 워크플로가 일시 정지된다(Lesson 12). 승인이 도착하면, 워크플로가 정확히 그 지점에서 재개한다. 이것이 LangGraph가 `interrupt()`를 인메모리 상태가 아니라 PostgreSQL 체크포인팅과 짝짓는 이유다. 이틀 뒤의 승인도 워크플로를 온전히 찾아낸다.

### 고무 도장 승인과 챌린지-앤-리스폰스 완화책

HITL의 기본 UI("Approve" / "Reject" 버튼)는 진정한 검토 없는 빠른 승인을 낳는다. 문서화된 완화책은 챌린지-앤-리스폰스 체크리스트다. Approve 버튼이 활성화되기 전에 특정 질문에 대한 긍정적 답변을 요구한다. 구체적 형태는 이렇다.

- "이것이 어떤 리소스를 건드리는지 이해하는가? [ ]"
- "폭발 반경이 받아들일 만한지 검증했는가? [ ]"
- "이것이 실패하면 롤백 계획이 있는가? [ ]"

관료주의 그 자체가 아니라 강제 함수(forcing function)다. 박스를 체크할 수 없는 검토자는 명확화를 요청하거나(격상) 거절한다(안전한 기본값). Anthropic 에이전트 안전 연구는 고무 도장 승인 패턴에 대한 완화책으로 체크리스트 기반 HITL을 명시적으로 인용한다.

### 무엇이 중대한(consequential) 것으로 간주되는가

모든 액션이 제안-후-커밋을 필요로 하지는 않는다. 2026년 지침:

- **중대한 액션**(항상 HITL): 되돌릴 수 없는 쓰기, 금융 거래, 외향 통신, 프로덕션(production) 데이터베이스 변경, 파괴적 파일 시스템 연산.
- **되돌릴 수 있는 액션**(때때로 HITL): 로컬 파일 편집, 스테이징 환경 변경, 명확한 롤백을 갖춘 되돌릴 수 있는 쓰기.
- **읽기와 검사**(결코 HITL 아님): 파일 읽기, 리소스 나열, 읽기 전용 API 호출.

### 액션 후 검증

"커밋이 실행되었다"는 "부작용이 일어났다"와 같지 않다. 네트워크 분할(network-partition)과 경쟁 조건(race condition)이 일어나면, 백엔드가 지속하지 않았는데도 워크플로가 성공했다고 여길 수 있다. 검증 단계는 커밋 후 대상 리소스를 다시 읽어 확인한다. 이것은 `RETURNING` 절을 갖춘 데이터베이스 트랜잭션이나 `PutObject` 후 AWS `GetObject`와 같은 패턴이다.

### EU AI Act 제14조

제14조는 EU의 고위험 AI 시스템에 대해 효과적인 사람의 감독을 의무화한다. "효과적"은 장식이 아니다. 규제 언어는 고무 도장 패턴을 특정하여 배제한다. 챌린지-앤-리스폰스를 갖춘 제안-후-커밋은 Microsoft Agent Governance Toolkit 컴플라이언스 문서에서 제14조 정밀 조사를 견뎌내는 형태다.

```figure
mx-propose-then-commit
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 stdlib Python으로 제안-후-커밋 상태 기계를 구현한다. 지속 저장소는 JSON 파일이다. 멱등성 키는 (thread_id, action_signature)의 해시다. 드라이버는 세 경우를 시뮬레이션한다: 깔끔한 승인 흐름, 일시적 실패 후 재시도(이중 실행되어서는 안 됨), 그리고 고무 도장 기본값 대 챌린지-앤-리스폰스 흐름.

## 산출물 (Ship It)

`outputs/skill-hitl-design.md`는 제안된 HITL 워크플로가 제안-후-커밋 형태인지 검토하고 누락된 메타데이터, 멱등성, 검증, 또는 챌린지-앤-리스폰스 층을 표시한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 승인된 제안의 재시도가 지속 레코드를 사용하고 재실행하지 않음을 확인하라. 이제 멱등성 키에 타임스탬프를 포함하도록 바꾸고 재시도가 이중 실행됨을 보여라.

2. 제안 레코드를 `rollback` 필드로 확장하라. 검증 단계가 실패하는 실행을 시뮬레이션하라. 롤백이 자동으로 발사됨을 보여라.

3. Microsoft Agent Framework의 `RequestInfoEvent` 문서를 읽어라. API가 포함하지만 장난감 엔진에는 없는 메타데이터 필드 하나를 식별하라. 그것을 추가하고 그것이 무엇을 방어하는지 설명하라.

4. 특정 액션(예: "공개 Twitter 계정에 게시")을 위한 챌린지-앤-리스폰스 체크리스트를 설계하라. 검토자가 답해야 할 세 질문은 무엇인가? 왜 그 세 가지인가?

5. 동기식 "Approve?" 프롬프트로 충분한(지속 저장소가 필요 없는) 경우 하나를 골라라. 왜 그런지 설명하고, 당신이 받아들이는 위험 부류를 명명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 제안-후-커밋 (Propose-then-commit) | "2단계 승인" | 지속된 제안 + 긍정적 커밋 + 검증 |
| 멱등성 키 (Idempotency key) | "재시도 안전 토큰" | 제안마다 고유함; 두 번째 실행은 무연산 |
| 데이터 계보 (Data lineage) | "어디서 왔는가" | 제안으로 이어진 특정 출처 콘텐츠 |
| 폭발 반경 (Blast radius) | "최악의 경우" | 액션이 잘못될 경우의 영향 범위 |
| 고무 도장 (Rubber-stamp) | "빠른 승인" | 진정한 검토 없이 클릭된 "Approve" |
| 챌린지-앤-리스폰스 (Challenge-and-response) | "강제 체크리스트" | 검토자가 특정 질문을 긍정적으로 확인해야 함 |
| RequestInfoEvent | "MS Agent Framework 기본 요소" | 구조화된 메타데이터를 갖춘 지속 HITL 요청 |
| `interrupt()` / `waitForApproval()` | "프레임워크 기본 요소" | 같은 형태의 LangGraph / Cloudflare 등가물 |

## 더 읽을거리 (Further Reading)

- [Microsoft Agent Framework: Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) `RequestInfoEvent`, 지속 승인.
- [Cloudflare Agents: Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) `waitForApproval()`과 Durable Objects.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평 위험에 대한 완화책으로서의 HITL.
- [EU AI Act: Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) 고위험 시스템에 대한 규제 베이스라인.
- [Anthropic: Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) 감독을 둘러싼 헌법적 프레이밍.
