# Capstone 16: GitHub 이슈-투-PR 자율 에이전트 (GitHub Issue-to-PR Autonomous Agent)

> AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, 그리고 Google Jules는 모두 동일한 2026 제품 형태를 출하한다: 이슈에 레이블을 붙이면 PR을 받는다. 클라우드 샌드박스(sandbox)에서 에이전트(agent)를 실행하고, 테스트가 통과하는지 검증하며, 근거(rationale)와 함께 리뷰 준비가 된 PR을 게시한다. 어려운 부분은 레포의 빌드 환경을 자동으로 재현하기, 자격 증명(credential) 유출 방지, 레포별 예산(budget) 시행, 그리고 에이전트가 force-push할 수 없게 하기다. 이 캡스톤(capstone)은 셀프 호스팅(self-hosted) 버전을 만들고 비용과 통과율(pass rate)에서 호스팅형 대안과 비교한다.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30시간

## 문제 (Problem)

비동기 클라우드 코딩 에이전트는 인터랙티브 코딩 에이전트(캡스톤 01)와는 별도의 제품 범주다. UX는 GitHub 레이블이다. 이슈에 `@agent fix this` 레이블을 붙이면, 워커(worker)가 클라우드 샌드박스에서 기동하고, 레포를 클론하며, 테스트를 실행하고, 파일을 편집하고, 검증한 뒤, 본문에 에이전트의 근거를 담은 PR을 연다. 인터랙티브 루프(loop)도, 터미널도 없다. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules, 그리고 Factory Droids가 모두 이것에 수렴한다.

엔지니어링 과제는 구체적이다: 환경 재현(에이전트가 캐시된 dev 이미지 없이 레포를 밑바닥부터 빌드해야 함), 불안정한(flaky) 테스트(재실행하거나 격리해야 함), 자격 증명 스코핑(최소한의 세분화된 권한을 갖는 GitHub App), 레포당 일당 예산 시행, 그리고 no-force-push 정책. 이 캡스톤은 호스팅형 대안 대비 통과율, 비용, 안전성을 측정한다.

## 개념 (Concept)

트리거는 GitHub 웹훅(이슈 레이블 또는 PR 코멘트)이다. 디스패처(dispatcher)가 ECS Fargate 또는 Lambda로 작업을 큐에 넣는다. 워커는 레포에서 추론한(언어, 프레임워크) 범용 Dockerfile로 레포를 Daytona 또는 E2B 샌드박스에 올린다. 에이전트는 Claude Opus 4.7 또는 GPT-5.4-Codex에 대해 mini-swe-agent 또는 SWE-agent v2 루프를 실행한다. 이렇게 반복한다: 코드 읽기, 수정 제안, 패치 적용, 테스트 실행.

검증은 게이팅 단계다. PR이 열리기 전에 샌드박스에서 전체 CI가 통과해야 한다. 커버리지(coverage) 차이가 계산된다. 임계값을 넘어 음수이면, PR은 열리지만 `needs-review` 레이블이 붙는다. 에이전트는 근거를 PR 설명으로 게시하고 리뷰어가 후속 작업을 위해 핑(ping)할 수 있는 `@agent` 스레드를 추가한다.

안전성은 서로 다른 두 GitHub 표면에서 스코핑된다: App은 `workflows: read`와 좁은 레포 컨텐츠/PR 스코프를 갖는 단수명(short-lived) 설치 토큰을 제공한다. 브랜치 보호(앱 권한이 아님)가 "`main`에 직접 쓰기 금지"와 "force-push 금지"를 시행한다. 앱은 결코 우회(bypass) 목록에 추가되지 않는다. `.github/workflows`로 경로를 좁힌 읽기 전용 접근은 실제 GitHub App 원시 기능이 아니므로, 파일 편집을 제한하는 에이전트의 허용 목록(allow-list)이 워커에서 이를 시행해야 한다. 레포당 일당 예산 상한은 디스패처에서 시행된다(예: 레포당 일당 최대 5 PR, PR당 $20).

## 아키텍처 (Architecture)

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 스택 (Stack)

- 트리거: 세분화된 토큰을 갖는 GitHub App; Lambda 또는 Fly.io를 통한 웹훅 리시버
- 워커: ECS Fargate 태스크(또는 GitHub Actions 셀프 호스팅 러너)
- 샌드박스: 태스크별 Daytona devcontainer 또는 E2B 샌드박스
- 에이전트 루프: Claude Opus 4.7 / GPT-5.4-Codex 위의 mini-swe-agent 베이스라인 또는 SWE-agent v2
- 검색(Retrieval): tree-sitter 레포 맵 + ripgrep
- 검증: 샌드박스 내 전체 CI + 커버리지 차이 게이트
- 관측성: PR 본문에서 링크된 PR별 트레이스 아카이브를 갖는 Langfuse
- 예산: 레포별 일당 달러 상한; 레포당 일당 최대 PR 수

```figure
cf-issue-to-pr
```

## 직접 만들기 (Build It)

1. **GitHub App.** 세분화된 설치 토큰: issues 읽기+쓰기, pull_requests 쓰기, contents 읽기+쓰기, workflows 읽기. 브랜치 보호(이것을 할 수 있는 유일한 표면)가 "`main`에 직접 푸시 금지"와 "force-push 금지"를 시행한다. 앱은 우회 목록에 없다. GitHub App 권한은 경로 스코프가 아니므로, 제안된 diff를 허용 목록으로 검사해 "`.github/workflows` 아래 쓰기 금지"를 워커가 시행한다.

2. **웹훅 리시버.** Lambda 함수가 이슈 레이블 / PR 코멘트 웹훅을 받는다. `@agent fix this` 레이블로 필터링한다. SQS로 큐에 넣는다.

3. **디스패처.** SQS에서 작업을 꺼낸다. 레포별 일당 예산을 시행한다. 레포 URL, 이슈 본문, 새 Daytona 샌드박스와 함께 ECS Fargate 태스크를 기동한다.

4. **환경 추론.** 언어(Python, Node, Go, Rust)와 패키지 매니저(uv, pnpm, go mod, cargo)를 탐지한다. 없으면 즉석에서 Dockerfile을 생성한다.

5. **에이전트 루프.** Claude Opus 4.7을 갖춘 mini-swe-agent 또는 SWE-agent v2. 도구: ripgrep, tree-sitter 레포 맵, read_file, edit_file, run_tests, git. 하드 리밋: $20 비용, 30분 실시간, 30 에이전트 턴.

6. **검증.** 루프가 종료된 후, 샌드박스에서 전체 테스트 스위트를 실행한다. jacoco / coverage.py를 통해 커버리지 차이를 계산한다. CI가 적색이면: 중단하고, PR을 열지 않는다. 커버리지가 2% 넘게 떨어지면: `needs-review` 레이블과 함께 PR을 연다.

7. **PR 게시.** 에이전트 브랜치를 푸시한다. GitHub API를 통해 다음과 함께 PR을 연다: 제목, 근거, diff 요약, 트레이스 URL, 비용, 턴 수.

8. **자격 증명 위생.** 워커는 단수명 GitHub App 설치 토큰으로 실행된다. 로그는 아카이브 전에 시크릿이 스크럽된다.

9. **평가.** 다양한 난이도의 시드된 내부 이슈 30개. 통과율, PR 품질(diff 크기, 스타일, 커버리지), 비용, 지연 시간을 측정한다. 동일한 이슈에서 Cursor Background Agents 및 AWS Remote SWE Agents와 비교한다.

## 라이브러리로 써보기 (Use It)

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 산출물 (Ship It)

`outputs/skill-issue-to-pr.md`가 결과물이다. 제한된 비용과 스코핑된 자격 증명으로 레이블된 이슈를 리뷰 준비가 된 PR로 바꾸는 GitHub App + 비동기 클라우드 워커.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 30개 이슈에서의 통과율 | 종단 간 성공 (CI 녹색 + 커버리지 OK) |
| 20 | PR 품질 | diff 크기, 커버리지 차이, 스타일 준수 |
| 20 | 해결된 이슈당 비용과 지연 시간 | PR당 $ 및 실시간 |
| 20 | 안전성 | 스코핑된 토큰, 레포별 예산, force-push 금지, 자격 증명 위생 |
| 15 | 운영자 UX | 근거 코멘트, 재시도 어포던스, @-멘션 후속 작업 |
| **100** | | |

## 연습 문제 (Exercises)

1. "불안정 테스트 수정(fix flaky test)" 모드를 추가한다: `@agent stabilize-flake TestX` 레이블이 샌드박스에서 테스트를 50번 실행하고 그것을 안정화하는 최소 변경을 제안한다.

2. 세 개의 공유 이슈에서 Cursor Background Agents 대비 비용을 비교한다. 어떤 도구가 어디서 이기는지 보고한다.

3. 예산 대시보드를 구현한다: 레포별 일당 비용, 사용자별 비용. 이상에 경보한다.

4. CI를 실행하지 않고 드래프트 PR을 여는 "드라이 런(dry-run)" 모드를 만들어, 리뷰어가 저렴하게 계획을 검토할 수 있게 한다.

5. 보존 정책을 추가한다: 머지 없이 7일보다 오래된 PR 브랜치는 자동으로 삭제된다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| GitHub App | "스코핑된 봇 아이덴티티" | 세분화된 권한 + 단수명 설치 토큰을 갖는 앱 |
| Async cloud agent | "백그라운드 에이전트" | 터미널이 아니라 클라우드 샌드박스에서 실행되는 비인터랙티브 워커 |
| Environment inference | "Dockerfile 합성" | 언어 + 패키지 매니저 탐지, 없으면 Dockerfile 생성 |
| Verification | "샌드박스 내 CI" | PR을 열기 전에 워커 내부에서 전체 테스트 스위트 실행 |
| Coverage delta | "커버리지 보존" | 베이스에서 에이전트 브랜치까지 테스트 커버리지 % 변화 |
| Per-repo budget | "일당 상한" | 디스패처에서 시행되는 달러 및 PR 수 상한 |
| Rationale | "PR 본문 설명" | 무엇이 왜 바뀌었는지에 대한 에이전트의 요약; PR 본문에 필수 |

## 더 읽을거리 (Further Reading)

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents): 표준적인 비동기 클라우드 에이전트 레퍼런스
- [SWE-agent](https://github.com/SWE-agent/SWE-agent): CLI 레퍼런스
- [Cursor Background Agents](https://docs.cursor.com/background-agent): 상용 대안
- [OpenAI Codex (cloud)](https://openai.com/codex): 호스팅형 경쟁자
- [Google Jules](https://jules.google): Google의 호스팅형 버전
- [Factory Droids](https://www.factory.ai): 대안 상용 레퍼런스
- [GitHub App documentation](https://docs.github.com/en/apps): 스코핑된 봇 아이덴티티
- [Daytona cloud sandboxes](https://daytona.io): 레퍼런스 샌드박스
