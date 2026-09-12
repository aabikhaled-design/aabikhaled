# Capstone 09: 코드 마이그레이션 에이전트 (Repo-Level Language / Runtime Upgrade)

> Amazon의 MigrationBench(Java 8 to 17)와 Google의 App Engine Py2-to-Py3 마이그레이터는 2026년의 기준을 세웠다. Moderne의 OpenRewrite는 대규모로 결정론적(deterministic) AST 재작성을 수행한다. Grit은 codemod 스타일 DSL로 같은 문제를 겨냥한다. 프로덕션(production) 패턴은 둘을 결합한다. 안전한 재작성을 위한 결정론적 기반(substrate)에 더해 모호한 경우를 위한 에이전트 계층, 브랜치별 빌드를 위한 샌드박스(sandbox), 그리고 PR이 열리기 전에 그린(green)으로 바뀌는 테스트 하네스(harness). 캡스톤(capstone)은 실제 저장소 50개를 마이그레이션하고 실패 분류 체계(taxonomy)와 함께 통과율을 발행하는 것이다.

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 hours

## 문제 (Problem)

대규모 코드 마이그레이션은 2026년 코딩 에이전트(coding agent)의 가장 깔끔한 프로덕션 응용 중 하나다. 정답 기준(ground truth)이 명백하고(마이그레이션 후 테스트 스위트가 통과하는가?), 보상이 실재하며(Java-8 플릿 마이그레이션은 인원 규모의 프로젝트다), 벤치마크(benchmark)가 공개되어 있다(MigrationBench 50-저장소 서브셋). Moderne의 OpenRewrite는 결정론적 측면을 처리한다. 에이전트 계층은 OpenRewrite 레시피가 할 수 없는 모든 것을 처리한다. 모호한 재작성, 빌드 시스템 드리프트(drift), 롱테일(long-tail) 구문, 전이적 의존성 파손.

Java 8 저장소(또는 Python 2 저장소)를 받아 그린-CI 마이그레이션된 브랜치를 만드는 에이전트를 만든다. 통과율, 테스트 커버리지 보존, 저장소당 비용을 측정하고, 실패 분류 체계를 만든다. 결정론 전용 베이스라인(baseline)과 나란히 비교하면 에이전트의 가치가 실제로 어디에 있는지가 드러난다.

## 개념 (Concept)

파이프라인에는 두 계층이 있다. **결정론적 기반**(Java용 OpenRewrite, Python용 libcst)은 기계적 재작성의 대부분을 안전하게 실행한다. imports, 메서드 시그니처, 널 안전(null-safety) 편집, try-with-resources, 폐기된(deprecated) API 교체. 빠르고 감사 가능한(auditable) diff를 만든다. **에이전트 계층**(Claude Opus 4.7과 GPT-5.4-Codex 위의 OpenAI Agents SDK 또는 LangGraph)은 레시피가 할 수 없는 경우를 처리한다. 빌드 파일 업그레이드(Maven/Gradle/pyproject), 전이적 의존성 충돌, 테스트 플레이크(flake), 커스텀 어노테이션.

저장소마다 타깃 런타임이 사전 설치된 Daytona 샌드박스를 하나씩 배정받는다. 에이전트는 빌드 실행, 실패 분류, 수정 적용, 재실행을 반복한다. 하드 한도: 저장소당 30분, 저장소당 $8, 에이전트 턴(turn) 20회. 모든 테스트가 통과하고 커버리지 차이가 음수가 아니면, 브랜치가 PR을 연다. 그렇지 않으면, 저장소는 증거와 함께 실패 클래스로 분류된다.

실패 분류 체계가 결과물이다. 50개 저장소에 걸쳐, 무엇이 깨졌는가? 전이적 의존성? 커스텀 어노테이션? 빌드 도구 버전? 마이그레이션과 무관한 테스트 플레이크? 각 클래스마다 개수와 예시 diff를 붙인다. 이후 레시피 작성자는 상위 세 가지를 겨냥하면 된다.

## 아키텍처 (Architecture)

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 스택 (Stack)

- 결정론적 기반: OpenRewrite(Java) 또는 libcst(Python)
- 에이전트: Claude Opus 4.7 + GPT-5.4-Codex 위의 OpenAI Agents SDK 또는 LangGraph
- 샌드박스: 브랜치별 Daytona devcontainer, 사전 설치된 타깃 런타임(Java 17 / Python 3.12)
- 빌드 시스템: Maven, Gradle, uv(Python)
- 벤치마크: Amazon MigrationBench 50-저장소 서브셋(Java 8 to 17), Google App Engine Py2-to-Py3 저장소
- 테스트 하네스: 병렬 러너, Jacoco(Java) 또는 coverage.py(Python)를 통한 커버리지
- 관측성(observability): Langfuse + 모든 diff 청크(chunk)가 담긴 저장소별 트레이스 번들
- 대시보드: 클래스별 개수와 예시 diff를 갖춘 실패 분류 체계 대시보드

```figure
ce-migration-funnel
```

## 직접 만들기 (Build It)

1. **레시피 패스.** OpenRewrite(Java) 또는 libcst(Python) 레시피를 먼저 실행한다. 기계적인 마이그레이션의 70-80%를 잡는다. "recipe" 커밋으로 커밋한다.

2. **빌드 시도.** Daytona 샌드박스: 타깃 런타임을 설치하고 빌드를 실행한다. 그린이면 테스트로 건너뛴다. 레드(red)이면 에이전트로 넘긴다.

3. **에이전트 루프.** 도구를 갖춘 LangGraph: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`. 에이전트가 실패를 분류(의존성, 구문, 테스트, 빌드 도구)하고 겨냥된 수정을 적용한다. 재실행한다.

4. **예산 상한.** 저장소당 벽시계 30분, 비용 $8, 에이전트 턴 20회. 어떤 위반이든 중단하고 현재 diff와 함께 "budget_exhausted"로 분류한다.

5. **테스트 + 커버리지 게이트.** 빌드가 그린이 된 후 테스트 스위트를 실행한다. 커버리지를 베이스 저장소와 비교한다. 커버리지가 2% 넘게 떨어졌으면 "coverage_regression"으로 분류한다.

6. **PR 열기.** 성공 시 브랜치를 푸시하고, diff와 함께 어떤 레시피가 적용되었고 에이전트가 어떤 커밋을 작성했는지 요약을 담아 PR을 연다.

7. **실패 분류 체계.** 실패한 저장소마다 클래스로 태그한다. `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. 대시보드를 만든다.

8. **50-저장소 실행.** MigrationBench 서브셋에 걸쳐 실행한다. 클래스별 통과율, 저장소당 비용, 커버리지 보존, 그리고 결정론 전용 베이스라인과의 비교를 보고한다.

## 라이브러리로 써보기 (Use It)

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 산출물 (Ship It)

`outputs/skill-migration-agent.md`이 결과물이다. 저장소가 주어지면 결정론적 레시피를 실행한 다음 에이전트 루프를 돌려 그린 마이그레이션된 브랜치를 만들거나, 저장소를 분류 체계 클래스로 분류한다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | MigrationBench 통과율 | 50-저장소 서브셋 pass@1 |
| 20 | 테스트 커버리지 보존 | 베이스 대비 평균 커버리지 차이 |
| 20 | 마이그레이션된 저장소당 비용 | 통과한 실행에서 저장소당 $ |
| 20 | 에이전트 / 결정론적 도구 통합 | OpenRewrite가 처리한 수정 대 에이전트가 작성한 수정의 비율 |
| 15 | 실패 분석 작성 | 예시를 갖춘 분류 체계 완전성 |
| **100** | | |

## 연습 문제 (Exercises)

1. OpenRewrite만으로(에이전트 없이) 마이그레이션 파이프라인을 실행한다. 통과율을 전체 파이프라인과 비교한다. 에이전트 단독이 차이를 만드는 경우를 식별한다.

2. "lint 클린" 검사를 구현한다. 마이그레이션 후 스타일 린터(Java용 spotless, Python용 ruff)를 실행한다. 새 lint 오류가 나타나면 PR을 실패시킨다. 커버리지는 보존되었지만 스타일이 퇴행한(coverage-preserved-but-style-regressed) 비율을 측정한다.

3. "최소 diff" 옵티마이저를 추가한다. 에이전트의 브랜치가 테스트를 통과한 후 두 번째 패스로 불필요한 변경을 잘라낸다. diff 크기 감소를 보고한다.

4. 세 번째 마이그레이션으로 확장한다. Node 18 to Node 22. 샌드박스 래핑을 재사용한다. 레시피 계층을 커스텀 codemod로 교체한다.

5. UX 지표로 첫 그린 빌드까지의 시간(TTFGB, time-to-first-green-build)을 측정한다. 목표: p50 10분 미만.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 결정론적 기반(Deterministic substrate) | "레시피 엔진" | OpenRewrite / libcst: 안전성 보장을 갖춘 선언적 AST 재작성 |
| Codemod | "코드 수정 프로그램" | 소스 코드를 기계적으로 변경하는 재작성 규칙 |
| 빌드 드리프트(Build drift) | "도구 버전 편차" | 메이저 버전 간의 미묘한 Maven / Gradle / uv 동작 변화 |
| 실패 클래스(Failure class) | "분류 체계 버킷" | 저장소가 마이그레이션되지 않은 레이블링된 이유: 의존성, 구문, 테스트, 빌드 도구, 예산 |
| 커버리지 차이(Coverage delta) | "커버리지 보존" | 베이스에서 마이그레이션된 브랜치로의 테스트 커버리지 % 변화 |
| 에이전트 턴(Agent turn) | "도구 호출 라운드" | 에이전트 루프에서의 한 번의 계획 -> 행동 -> 관찰 사이클 |
| 예산 소진(Budget exhaustion) | "상한에 도달" | 저장소가 통과하지 못한 채 30분 / $8 / 20턴 한도를 소비함 |

## 더 읽을거리 (Further Reading)

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/): 정전(canonical)에 해당하는 2026 벤치마크
- [Moderne.io OpenRewrite platform](https://www.moderne.io): 결정론적 기반 레퍼런스
- [OpenRewrite documentation](https://docs.openrewrite.org): 레시피 작성
- [Grit.io](https://www.grit.io): 대안 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent): Agents SDK 레퍼런스
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine): 대안 마이그레이션 벤치마크
- [libcst](https://github.com/Instagram/LibCST): Python 결정론적 기반
- [Daytona sandboxes](https://daytona.io): 레퍼런스 브랜치별 샌드박스
