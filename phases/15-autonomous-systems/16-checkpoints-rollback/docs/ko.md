# 체크포인트와 롤백 (Checkpoints and Rollback)

> 모든 그래프 상태 전이(graph-state transition)는 영속화된다. 워커(worker)가 죽으면 그 리스(lease)가 만료되고 다른 워커가 가장 최근 체크포인트(checkpoint)에서 작업을 이어받는다. Cloudflare Durable Objects는 몇 시간에서 몇 주에 걸쳐 상태를 유지한다. 제안 후 커밋(propose-then-commit, Lesson 15)은 각 액션마다 롤백 계획을 정의한다. 액션 사후 검증(post-action verification)이 그 루프를 닫는다. EU AI Act 제14조는 고위험 시스템에 대해 효과적인 인간 감독(human oversight)을 의무화한다. 실무적으로 이는 체크포인트가 조회 가능해야 하고, 롤백이 리허설되어야 하며, 감사 추적(audit trail)이 배포(deploy)를 견뎌내야 한다는 뜻이다. 실패 모드는 날카롭다. 멱등성 키(idempotency key)와 사전 조건 검사(precondition check)가 없으면, 일시적 실패 이후의 재시도(retry)가 이미 승인된 액션을 이중 실행한다. 액션 사후 검증이 바로 이를 잡아낸다.

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60분

## 문제 (The Problem)

지속 실행(durable execution, Lesson 12)은 죽은 에이전트(agent)를 재개 가능하게 만든다. 제안 후 커밋(propose-then-commit, Lesson 15)은 승인된 액션을 감사 가능하게 만든다. 이 레슨은 둘을 결합한다. 승인된 액션이 부분적으로 실행되다가 죽고 다시 재개되면 무슨 일이 일어나는가? 롤백은 언제, 어떤 상태에 대해 실행되는가?

실제 시스템들은 이를 서로 다르게 연결한다.

- **LangGraph**는 모든 그래프 상태 전이를 PostgreSQL에 체크포인트로 저장한다. 워커가 죽으면 리스가 해제되고 다른 워커가 가장 최근 체크포인트에서 재개한다. 워크플로(workflow)는 `interrupt()`에서 멈추며, 이 멈춤 자체도 영속화된다.
- **Cloudflare Durable Objects**는 키별 상태를 몇 시간에서 몇 주에 걸쳐 유지한다. 승인된 액션에 대해 연산을 스토리지와 같은 위치에 둔다(co-locate).
- **Microsoft Agent Framework**는 워크플로 API에서 `Checkpoint` 프리미티브를 노출한다. 재생(replay)과 멱등성이 재시도를 처리한다.

어느 경우든 실제로 작동하는 조합은 같다. 멱등성 키(이중 실행을 방지) + 사전 조건 검사(상태가 여전히 우리가 승인 당시 기준이던 그대로인지 확인) + 액션 사후 검증(부작용이 실제로 일어났는지 확인) + 검증 실패 시 롤백.

## 개념 (The Concept)

### 모든 전이는 영속화된다

그래프 상태 전이는 워크플로를 하나의 명명된 상태에서 다른 상태로 옮기는 모든 단계다. 순진한 구현은 특정 커밋 지점에서만 영속화하지만, 프로덕션 구현은 모든 전이를 영속화한다. 그 비용(약간의 추가 쓰기 작업)은 얻는 신뢰성(재생이 어디에서든 안착하고, 리스 복구가 정밀해진다)에 비하면 작다.

### 리스 복구 (Lease recovery)

워커가 죽어도 워크플로는 잃어버리지 않는다. 리스(lease, 이 워커가 이 실행을 수행 중이라는 단명(短命) 점유권)가 그저 만료될 뿐이다. 다른 워커가 가장 최근 체크포인트를 집어 들고 재개한다. 프로덕션 시스템이 진행 중인 작업을 잃지 않고 롤링 배포(rolling deploy)를 견뎌내는 것은 바로 이 리스 메커니즘 덕분이다.

### 멱등성과 사전 조건

멱등성만으로는 충분하지 않다. 다음을 생각해보자. 어떤 워크플로가 "잔액 > $1000일 때 A에서 B로 $100 송금"을 승인받았다. 이 워크플로가 커밋되고, 실행 도중에 죽고, 재개된다. 멱등성 키만 검사하고 실행을 재개한다면, 송금은 한 번만 실행된다(올바름). 그러나 죽음과 재개 사이에 A의 잔액이 다른 워크플로를 통해 $500로 떨어졌다고 하자. 멱등성 검사는 여전히 통과한다. 하지만 사전 조건은 통과하지 못한다. 사전 조건 검사가 없으면 우리는 초과 인출(overdraft)을 출고하게 된다.

결과를 초래하는 모든 액션에는 두 가지가 모두 필요하다.

- **멱등성 키(Idempotency key)**: 이중 실행을 방지한다.
- **사전 조건 검사(Precondition check)**: 상태가 여전히 승인된 내용과 일관적인지 확인한다.

### 액션 사후 검증 (Post-action verification)

"도구가 200을 반환했다"는 검증이 아니다. 진짜 검증은 대상 상태를 다시 읽어서 부작용이 실제로 일어났는지 확인한다. 패턴들:

- 데이터베이스 갱신: `UPDATE ... RETURNING *` 후 반환된 행이 의도한 상태와 일치하는지 단언(assert)한다.
- 이메일 전송: 제출 후 보낸 편지함에서 메시지 ID를 확인한다.
- 파일 쓰기: 파일을 다시 읽어서 해시(hash)한다.
- API 호출: 대상 리소스에 대한 후속 `GET`을 수행한다.

검증이 실패하면 워크플로는 알려진 불량(known-bad) 상태에 있는 것이다. 롤백이 작동한다.

### 롤백 계획 (Rollback plans)

제안 후 커밋(propose-then-commit, Lesson 15)에서 결과를 초래하는 모든 액션은 롤백 계획을 지닌다. 종류:

- **인밴드 롤백(In-band rollback)**: 부작용을 직접 되돌린다(`INSERT` 후 `DELETE`, 전송 후 `Send-correction-email`).
- **보상 트랜잭션(Compensating transaction)**: 원래 액션을 중화하는 새로운 액션(표준 SAGA 패턴).
- **아웃오브밴드 롤백(Out-of-band rollback)**: 인간에게 알리고, 워크플로를 멈추며, 조사를 위해 불량 상태를 그대로 남겨둔다.

무동작 롤백("이것은 되돌릴 수 없다")은 제안 안에서 반드시 명시되어야 한다. 롤백이 없는 액션은 커밋 시점에 더 강한 HITL을 요구한다(Lesson 15의 챌린지 앤 리스폰스(challenge-and-response)).

### EU AI Act 제14조의 운영적 해석

제14조는 고위험 시스템에 대해 "효과적인 인간 감독"을 요구한다. 운영적 관점에서 구현자들은 이를 다음과 같이 읽는다.

- 체크포인트는 감사인이 조회할 수 있다.
- 롤백은 리허설된다(적어도 한 번은 종단 간(end-to-end)으로 테스트된다).
- 감사 추적은 배포를 견뎌낸다(체크포인트 백엔드가 일시적(ephemeral)이지 않다).
- 실패한 검증은 조용히 로깅되는 것이 아니라 알림이 발생한다.

커밋 도중에 죽어서 재개된 뒤, 검증 + 롤백 경로 없이 부작용을 완료하는 워크플로는 제14조 테스트를 통과하지 못한다.

### 날카로운 실패 모드: 이중 실행

이 영역에서 가장 흔한 프로덕션 사고:

1. 액션 승인, 멱등성 키 k.
2. 커밋 시작, 실행, 200 반환.
3. "커밋됨" 상태를 영속화하기 전에 워크플로가 죽는다.
4. 워크플로 재개. "승인됨이지만 커밋되지 않음"을 보고 재실행한다.
5. 부작용이 두 번 발생한다.

완화책: 실행 전에 "진행 중(in-flight)" 의도를 영속화하고, 멱등성 키로 실행하며, 액션 사후 검증이 성공한 뒤에야 "커밋됨"으로 표시한다. 액션은 발생했는데 상태 쓰기가 실패하면, 검증하고 (필요하면) 재발생시켜야 함을 알 수 있다. 상태 쓰기는 성공했는데 액션이 실패하면, 복구 경로를 통해 검증하고 정확히 한 번 발생시킨다.

```figure
checkpoint-replay
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 멱등성, 사전 조건, 검증, 롤백을 갖춘 체크포인트 기반 워크플로를 구현한다. 드라이버는 네 가지 시나리오를 시뮬레이션한다. 정상 실행, 죽음 이후 재시도(멱등성이 잡아냄), 사전 조건 실패(워크플로가 발생 없이 중단됨), 검증 실패(롤백 발생).

## 산출물 (Ship It)

`outputs/skill-rollback-rehearsal.md`는 제안된 워크플로에 대한 롤백 리허설 테스트를 설계하고, 감사 추적 영속성을 위해 체크포인트 백엔드를 감사한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 네 가지 시나리오를 검증하라. 커밋 도중 죽는 경우에 대해, 재시도 전반에 걸쳐 액션이 정확히 한 번 발생하는지 확인하라.

2. "먼저 완료로 표시한 다음 실행" 패턴을 수정하여 상태 쓰기가 액션 이후에 발생하도록 하라. 죽음 시나리오를 다시 실행하라. 얼마나 많은 중복 액션이 발생하는지 측정하라.

3. 특정 프로덕션 액션(예: "Slack 채널에 게시")에 대한 롤백 계획을 설계하라. 인밴드, 보상, 아웃오브밴드 중 하나로 분류하라. 그 선택을 정당화하라.

4. 당신이 아는 워크플로 하나를 골라라. 모든 상태 전이를 식별하라. 각각에 지속성 요구사항(영속화 / 영속화 안 함)을 표시하라. 현재 영속화하고 있지 않은 것들의 개수를 세어라.

5. 리허설된 롤백 테스트: 실제 워크플로를 실행하고, 그것을 죽이고, 롤백 경로가 발생하는지 확인하는 종단 간 테스트를 설계하라. 그 테스트는 무엇을 단언하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 체크포인트 (Checkpoint) | "세이브 포인트" | 모든 그래프 상태 전이가 지속성 저장소에 영속화된다 |
| 리스 (Lease) | "워커 점유권" | 워커가 어떤 실행을 수행 중이라는 단명 점유권. 죽으면 만료된다 |
| 사전 조건 (Precondition) | "상태 게이트" | 상태가 여전히 승인된 액션과 일관적이라는 단언 |
| 액션 사후 검증 (Post-action verify) | "재읽기 검사" | 대상 시스템에서 부작용이 실제로 일어났는지 확인 |
| 인밴드 롤백 (In-band rollback) | "직접 되돌리기" | 역연산으로 부작용을 되돌린다 |
| 보상 트랜잭션 (Compensating transaction) | "SAGA 되돌리기" | 원래 액션을 중화하는 새로운 액션 |
| 먼저 완료 표시 (Mark-as-done-first) | "상태 쓰기 순서" | 커밋에서 반환하기 전에 커밋됨 상태를 영속화한다 |
| 제14조 (Article 14) | "EU AI Act 인간 감독" | 운영적으로: 조회 가능한 체크포인트, 리허설된 롤백, 감사 가능한 추적 |

## 더 읽을거리 (Further Reading)

- [Microsoft Agent Framework: Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) 체크포인트 프리미티브와 리스 복구.
- [Cloudflare Agents: Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) 상태 기질(substrate)로서의 Durable Objects.
- [EU AI Act: Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) 규제 베이스라인(baseline).
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평(long-horizon) 워크플로에 대한 신뢰성 프레이밍.
- [Anthropic: Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) Claude Code Routines를 위한 워크플로 형태.
