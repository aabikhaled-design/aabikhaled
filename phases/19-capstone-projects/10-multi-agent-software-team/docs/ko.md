# Capstone 10: 멀티 에이전트 소프트웨어 엔지니어링 팀

> SWE-AF의 팩토리(factory) 아키텍처, MetaGPT의 역할 기반 프롬프팅, AutoGen 0.4의 타입 지정 액터 그래프, Cognition의 Devin, Factory의 Droids는 모두 같은 2026년 형태로 수렴했다. 아키텍트(architect)가 계획하고, N명의 코더(coder)가 병렬 worktree에서 작업하고, 리뷰어(reviewer)가 게이팅하고, 테스터(tester)가 검증한다. 병렬 worktree는 벽시계 시간을 처리량(throughput)으로 바꾼다. 공유 상태와 핸드오프(handoff) 프로토콜이 실패 표면이 된다. 캡스톤(capstone)은 그 팀을 만들고, SWE-bench Pro에서 평가하고, 어떤 핸드오프가 얼마나 자주 깨지는지 보고하는 것이다.

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 hours

## 문제 (Problem)

단일 에이전트 코딩 하네스(harness)는 큰 작업에서 천장에 부딪힌다. 개별 에이전트가 약해서가 아니라, 200k 토큰(token) 컨텍스트가 아키텍처 계획에 더해 네 개의 병렬 코드베이스 조각에 더해 리뷰어 논평에 더해 테스트 출력을 담을 수 없기 때문이다. 멀티 에이전트 팩토리는 문제를 쪼갠다. 아키텍트가 계획을 소유하고, 코더들이 병렬 worktree에서 구현을 소유하고, 리뷰어가 게이팅하고, 테스터가 검증한다. SWE-AF의 "팩토리" 아키텍처, MetaGPT의 역할, AutoGen의 타입 지정 액터 그래프. 세 프레이밍 모두 같은 형태를 기술한다.

실패 표면은 핸드오프다. 아키텍트가 코더들이 구현할 수 없는 무언가를 계획한다. 코더들이 충돌하는 diff를 만든다. 리뷰어가 환각된(hallucinated) 수정을 승인한다. 테스터가 아직 작성 중인 코더와 경쟁(race)한다. 이번 캡스톤에서는 이런 팀 하나를 만들어 50개의 SWE-bench Pro 이슈에서 실행하고, 모든 핸드오프를 추적하고, 사후 분석(post-mortem)을 발행한다.

## 개념 (Concept)

역할은 타입 지정 에이전트다. **아키텍트**(Claude Opus 4.7)는 이슈를 읽고, 계획을 쓰고, 명시적 인터페이스와 함께 하위 작업(subtask)으로 쪼갠다. **코더들**(Claude Sonnet 4.7, N개의 병렬 인스턴스, 각각 `git worktree` + Daytona 샌드박스(sandbox) 안에 있음)이 하위 작업을 독립적으로 구현한다. **리뷰어**(GPT-5.4)는 병합된 diff를 읽고 승인하거나 구체적인 변경을 요청한다. **테스터**(Gemini 2.5 Pro)는 격리된 상태로 테스트 스위트를 실행하고 아티팩트(artifact)와 함께 통과/실패를 보고한다.

통신은 공유 작업 보드(task board, 파일 기반 또는 Redis)를 통한다. 각 역할은 자신이 처리하도록 허용된 작업을 소비한다. 핸드오프는 A2A 프로토콜로 타입이 지정된 메시지다. 조율 관심사: 병합 충돌(merge-conflict) 해결(코디네이터 역할 또는 자동 3방향 병합), 공유 상태 동기화(코더가 시작하면 계획은 동결됨; 재계획(replan)은 별도 이벤트), 그리고 리뷰어 게이트키핑(리뷰어는 자신의 변경이나 자신이 제안한 변경을 승인할 수 없음).

토큰 증폭(token amplification)은 숨은 비용이다. 모든 역할 경계는 요약 프롬프트와 핸드오프 컨텍스트를 추가한다. 40턴(turn) 단일 에이전트 실행이 네 역할에 걸쳐 총 160턴이 된다. 평가 기준은 "멀티 에이전트가 작동하는가"가 아니라 "달러당 이기는가"다. 그래서 단일 에이전트 베이스라인(baseline) 대비 토큰 효율성을 특별히 따진다.

## 아키텍처 (Architecture)

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 스택 (Stack)

- 오케스트레이션: 공유 상태 + 에이전트별 서브 그래프를 갖춘 LangGraph
- 메시징: 타입 지정 에이전트 간 메시지를 위한 A2A 프로토콜(Google 2025)
- 모델: Opus 4.7(아키텍트), Sonnet 4.7(코더들), GPT-5.4(리뷰어), Gemini 2.5 Pro(테스터)
- worktree 격리: 코더별 `git worktree add` + Daytona 샌드박스
- 병합 코디네이터: 커스텀 3방향 병합 + LLM 매개 충돌 해결
- 평가: SWE-bench Pro(50개 이슈), SWE-AF 시나리오, 단위 테스트를 위한 HumanEval++
- 관측성(observability): 역할 태그가 달린 스팬(span)을 갖춘 Langfuse, 에이전트별 토큰 회계
- 배포: 각 역할을 별도 Deployment + 백로그(backlog)에 대한 HPA로 갖춘 K8s

```figure
ce-team-handoff
```

## 직접 만들기 (Build It)

1. **작업 보드.** 타입 지정 메시지를 갖춘 파일 기반 JSONL: `plan_request`, `subtask`, `diff_ready`, `review_needed`, `test_needed`, `approved`, `rejected`, `replan_needed`. 에이전트는 태그를 구독한다.

2. **아키텍트.** GitHub 이슈를 읽고, 명시적 하위 작업 인터페이스(건드린 파일, 공개 함수, 테스트 영향)를 요구하는 계획 템플릿으로 Opus 4.7을 실행한다. 하위 작업의 DAG를 가진 하나의 `plan_request`를 방출한다.

3. **코더들.** N개의 병렬 워커, 각각 보드에서 하나의 하위 작업을 가져온다. 각각 새 `git worktree add` 브랜치와 Daytona 샌드박스를 띄운다. 하위 작업을 구현한다. 패치 + 테스트 차이와 함께 `diff_ready`를 방출한다.

4. **병합 코디네이터.** 모든 코더가 완료되면, N개 브랜치를 스테이징 브랜치로 3방향 병합한다. 파일 수준 겹침이 존재할 때만 LLM 매개 충돌 해결.

5. **리뷰어.** GPT-5.4가 병합된 diff를 읽는다. 자신이 작성한 diff는 승인할 수 없다. `approved`(무동작) 또는 해당 코더에게 다시 라우팅되는 구체적 변경 요청과 함께 `review_feedback`을 방출한다.

6. **테스터.** Gemini 2.5 Pro가 깨끗한 샌드박스에서 테스트 스위트를 실행한다. 아티팩트를 포착한다. `test_passed` 또는 스택트레이스와 함께 `test_failed`를 방출한다. 실패한 테스트는 실패한 하위 작업을 소유한 코더에게 다시 루프백된다.

7. **핸드오프 회계.** 역할 경계를 가로지르는 모든 메시지는 페이로드 크기와 사용된 모델과 함께 Langfuse에서 스팬을 얻는다. 하위 작업별 토큰 증폭(coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens)을 계산한다.

8. **평가.** 50개 SWE-bench Pro 이슈에서 실행한다. 단일 에이전트 베이스라인(단일 worktree의 Sonnet 4.7 하나) 대비 pass@1과 해결된 이슈당 $를 비교한다.

9. **사후 분석.** 실패한 각 이슈에 대해, 깨진 핸드오프(계획이 너무 모호함, 병합 충돌, 리뷰어 거짓 승인, 테스터 플레이크(flake))를 식별한다. 핸드오프 실패 히스토그램을 만든다.

## 라이브러리로 써보기 (Use It)

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 산출물 (Ship It)

`outputs/skill-multi-agent-team.md`이 결과물이다. 이슈 URL과 병렬성 수준이 주어지면, 팀은 역할별 토큰 회계와 함께 병합 준비된 PR을 만들어 낸다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 매칭된 50개 이슈 서브셋, pass@1 |
| 20 | 병렬 속도 향상 | 단일 에이전트 베이스라인 대비 벽시계 시간 |
| 20 | 리뷰 품질 | 주입된 버그 프로브에서의 거짓 승인 비율 |
| 20 | 토큰 효율성 | 단일 에이전트 대비 해결된 이슈당 총 토큰 |
| 15 | 조율 엔지니어링 | 병합 충돌 해결, 핸드오프 실패 히스토그램 |
| **100** | | |

## 연습 문제 (Exercises)

1. 실행 중간에 diff에 명백한 버그(메인 본문 앞에 여분의 `return None`)를 주입한다. 리뷰어의 거짓 승인 비율을 측정한다. 거짓 승인이 5% 미만이 될 때까지 리뷰어 프롬프트를 튜닝한다.

2. 두 코더로 줄인다(아키텍트 + 코더 + 리뷰어 + 테스터, 코더가 두 하위 작업을 순차적으로 실행). 벽시계 시간과 통과율을 비교한다.

3. 병합 코디네이터를 단일 작성자(single-writer) 제약(하위 작업이 서로소(disjoint) 파일 집합을 건드림)으로 교체한다. 아키텍트에게 가해지는 계획 부담을 측정한다.

4. 리뷰어를 GPT-5.4에서 Claude Opus 4.7로 교체한다. 거짓 승인 비율과 토큰 비용 차이를 측정한다.

5. 다섯 번째 역할을 추가한다. 문서 작성자(documenter, Haiku 4.5). 리뷰 후, 변경 로그(changelog) 항목을 만든다. 문서 품질이 추가 토큰 지출을 정당화하는지 측정한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 병렬 worktree(Parallel worktree) | "격리된 브랜치" | 코더별로 새 작업 트리를 만드는 `git worktree add` |
| 작업 보드(Task board) | "공유 메시지 버스" | 에이전트가 구독하는 타입 지정 메시지의 파일 또는 Redis 저장소 |
| 핸드오프(Handoff) | "역할 경계" | 한 역할의 컨텍스트에서 다른 역할의 컨텍스트로 넘어가는 모든 메시지 |
| 토큰 증폭(Token amplification) | "멀티 에이전트 오버헤드" | 같은 작업에 대한 역할 전반의 총 토큰 / 단일 에이전트 토큰 |
| A2A 프로토콜(A2A protocol) | "에이전트 간(agent-to-agent)" | 타입 지정 에이전트 간 메시지를 위한 Google의 2025 스펙 |
| 병합 코디네이터(Merge coordinator) | "통합기(integrator)" | 3방향 병합을 실행하고 충돌을 매개하는 컴포넌트 |
| 거짓 승인(False approval) | "리뷰어 환각" | 리뷰어가 알려진 버그가 있는 diff를 승인 |

## 더 읽을거리 (Further Reading)

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF): 레퍼런스 2026 멀티 에이전트 팩토리
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT): 역할 기반 멀티 에이전트 프레임워크
- [AutoGen v0.4](https://github.com/microsoft/autogen): Microsoft의 타입 지정 액터 프레임워크
- [Cognition AI (Devin)](https://cognition.ai): 레퍼런스 제품
- [Factory Droids](https://www.factory.ai): 대안 레퍼런스 제품
- [Google A2A protocol](https://developers.google.com/agent-to-agent): 에이전트 간 메시징 스펙
- [git worktree documentation](https://git-scm.com/docs/git-worktree): 격리 기반
- [SWE-bench Pro](https://www.swebench.com): 평가 대상
