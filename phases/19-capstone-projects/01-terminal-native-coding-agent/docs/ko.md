# Capstone 01: 터미널 네이티브 코딩 에이전트 (Terminal-Native Coding Agent)

> 2026년이면 코딩 에이전트(coding agent)의 형태는 이미 정해져 있다. TUI 하네스(harness), 상태를 가진 계획(plan), 샌드박스(sandbox)로 격리된 도구 표면(tool surface), 계획하고·행동하고·관찰하고·복구하는 루프. Claude Code, Cursor 3, OpenCode는 50피트 밖에서 보면 모두 똑같이 생겼다. 이 캡스톤(capstone)은 그런 에이전트를 처음부터 끝까지 하나 만드는 것을 요구한다. CLI로 들어가서 풀 리퀘스트(pull request)로 나온다. 그리고 SWE-bench Pro에서 mini-swe-agent와 Live-SWE-agent를 기준으로 측정한다. 어려운 부분은 모델 호출이 아니라 도구 루프(tool loop), 샌드박스, 50턴(turn) 실행에 걸리는 비용 상한임을 배우게 된다.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## 문제 (Problem)

코딩 에이전트는 2026년에 지배적인 AI 애플리케이션 범주가 되었다. Claude Code(Anthropic), Composer 2 및 Agent Tabs를 탑재한 Cursor 3(Cursor), Amp(Sourcegraph), OpenCode(스타 11.2만 개), Factory Droids, Google Jules는 모두 동일한 아키텍처의 변형을 출시한다. 터미널 하네스, 권한이 부여된 도구 표면, 샌드박스, 프런티어 모델(frontier model)을 중심으로 구축된 계획-행동-관찰 루프다. 프런티어는 좁다. Live-SWE-agent는 Opus 4.5로 SWE-bench Verified에서 79.2%에 도달했다. 하지만 엔지니어링 기교는 넓다. 대부분의 실패 모드는 모델의 실수가 아니다. 도구 루프 불안정성, 컨텍스트 오염(context poisoning), 폭주하는 토큰(token) 비용, 파괴적인 파일시스템 연산이다.

이런 에이전트를 바깥에서 추론할 수는 없다. 직접 하나 만들어 보고, ripgrep이 8MB짜리 매치를 반환할 때 47번째 턴에서 루프가 무너지는 것을 지켜본 다음, 절단(truncation) 계층을 다시 만들어야 한다. 그것이 이 캡스톤의 핵심이다.

## 개념 (Concept)

하네스에는 네 개의 표면이 있다. **계획(Plan)**은 모델이 매 턴 새로 쓰는 TodoWrite 스타일의 상태 객체를 유지한다. **행동(Act)**은 도구 호출(읽기, 편집, 실행, 검색, git)을 디스패치(dispatch)한다. **관찰(Observe)**은 stdout / stderr / 종료 코드를 포착하고, 절단하고, 요약을 다시 피드백한다. **복구(Recover)**는 컨텍스트 윈도우(context window)를 날려버리거나 영원히 루프에 빠지지 않으면서 도구 오류를 처리한다. 2026년의 형태는 한 가지를 더 추가한다. **훅(hooks)**이다. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, `PreCompact`: 운영자가 정책, 텔레메트리(telemetry), 가드레일(guardrail)을 주입하는 설정 가능한 확장 지점이다.

샌드박스는 E2B 또는 Daytona다. 각 작업은 git worktree가 읽기-쓰기로 마운트된 새 devcontainer에서 실행된다. 하네스는 호스트 파일시스템을 절대 건드리지 않는다. worktree는 성공하든 실패하든 해체된다. 비용 통제는 세 계층에서 강제된다. 턴당 토큰 상한, 세션당 달러 예산, 하드 턴 제한(보통 50)이다. 관측성(observability) 계층은 GenAI 시맨틱 컨벤션(semantic convention)을 따르는 OpenTelemetry 스팬(span)이며, 자체 호스팅한 Langfuse로 전송된다.

## 아키텍처 (Architecture)

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 스택 (Stack)

- 하네스 런타임: Bun 1.2 + Ink 5 (터미널 안의 React)
- 모델 접근: Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5(가장 어려운 작업용)를 갖춘 OpenRouter 통합 API
- 도구 전송: Model Context Protocol StreamableHTTP (MCP 2026 개정판)
- 샌드박스: E2B 샌드박스(JS SDK) 또는 Daytona devcontainer
- 코드 검색: ripgrep 서브프로세스, 17개 언어용 tree-sitter 파서(사전 컴파일됨)
- 격리: 작업마다 `git worktree add`, 성공 / 실패 시 정리
- 평가 하네스: SWE-bench Pro(검증된 서브셋) + Terminal-Bench 2.0 + 직접 만든 30개 작업 홀드아웃(holdout)
- 관측성: `gen_ai.*` semconv를 따르는 OpenTelemetry SDK → 자체 호스팅 Langfuse
- PR 게시: 세분화된 토큰을 가진 GitHub App, 범위는 대상 저장소로 제한

```figure
ce-agent-loop
```

## 직접 만들기 (Build It)

1. **TUI와 명령 루프.** Ink를 사용해 Bun 프로젝트를 스캐폴딩(scaffold)한다. `agent run <repo> "<task>"`를 받는다. 분할 뷰를 출력한다. 계획 패널(위), 도구 호출 스트림(가운데), 토큰 예산(아래). Ctrl-C에 취소를 추가하되, 종료 전에 `SessionEnd` 훅을 발동시킨다.

2. **계획 상태.** 타입이 지정된 TodoWrite 스키마(메모가 달린 pending / in_progress / done 항목)를 정의한다. 모델은 매 턴 도구 호출로 전체 상태를 새로 쓴다. 점진적으로 변경(mutate)하게 두지 말 것. 크래시 후 재개할 수 있도록 계획을 `.agent/state.json`에 영속화한다.

3. **도구 표면.** 여섯 개의 도구를 정의한다. `read_file`, `edit_file`(diff 미리보기 포함), `ripgrep`, `tree_sitter_symbols`, `run_shell`(타임아웃 포함), `git`(status / diff / commit / push). MCP StreamableHTTP로 노출해 하네스를 전송 방식과 무관하게(transport-agnostic) 만든다. 모든 도구는 절단된 출력을 반환한다(호출당 4k 토큰으로 상한).

4. **샌드박스 래핑.** 각 작업은 E2B 샌드박스를 띄운다. `git worktree add -b agent/$TASK_ID`로 새 브랜치를 만든다. 모든 도구 호출은 샌드박스 안에서 실행된다. 호스트 파일시스템은 접근 불가능하다.

5. **훅.** 2026년의 여덟 가지 훅 타입을 모두 구현한다. 사용자가 작성한 훅을 최소 네 개 연결한다. (a) worktree 바깥에서의 `rm -rf`를 차단하는 `PreToolUse` 파괴적 명령 가드, (b) `PostToolUse` 토큰 회계, (c) `SessionStart` 예산 초기화, (d) 최종 트레이스 번들(trace bundle)을 기록하는 `Stop`.

6. **평가 루프.** SWE-bench Pro Python의 30개 이슈 서브셋을 클론(clone)한다. 각각에 대해 하네스를 실행한다. pass@1, 작업당 턴 수, 작업당 달러를 기준으로 mini-swe-agent(최소 베이스라인(baseline))와 비교한다. 결과를 `eval/results.jsonl`에 기록한다.

7. **비용 통제.** 하드 컷오프: 50턴, 200k 컨텍스트, 작업당 $5. `PreCompact` 훅은 150k 지점에서 오래된 턴들을 사전 상태 블록으로 요약하여, 계획을 잃지 않으면서 새 관찰을 위한 공간을 확보한다.

8. **PR 게시.** 성공 시, 마지막 단계는 `git push`와, 본문에 계획과 diff 요약을 담은 PR을 여는 GitHub API 호출이다.

## 라이브러리로 써보기 (Use It)

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 산출물 (Ship It)

결과물 스킬은 `outputs/skill-terminal-coding-agent.md`에 위치한다. 저장소 경로와 작업 설명이 주어지면, 샌드박스 안에서 전체 계획-행동-관찰 루프를 실행하고 PR URL과 트레이스 번들을 반환한다. 이 캡스톤의 평가 기준:

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 대 베이스라인 | 30개의 매칭된 Python 작업에서 당신의 하네스 대 mini-swe-agent |
| 20 | 아키텍처 명료성 | 계획/행동/관찰 분리, 훅 표면, 도구 스키마: Live-SWE-agent 레이아웃과 대조하여 검토 |
| 20 | 안전성 | 샌드박스 탈출 테스트, 권한 프롬프트, 파괴적 명령 가드가 레드팀(red-team)을 통과 |
| 20 | 관측성 | 트레이스 완전성(도구 호출의 100%가 스팬으로 기록됨), 턴당 토큰 회계 |
| 15 | 개발자 UX | 콜드 스타트(cold-start) < 2초, 크래시 복구가 계획을 재개, Ctrl-C가 도구 실행 중간에 깔끔하게 취소 |
| **100** | | |

## 연습 문제 (Exercises)

1. 백엔드 모델을 Claude Sonnet 4.7에서 vLLM으로 서빙되는 Qwen3-Coder-30B로 교체한다. pass@1과 작업당 달러를 비교한다. 오픈 모델이 어디서 성능이 떨어지는지 보고한다.

2. PR 게시 전에 diff를 읽고 수정 루프를 요청할 수 있는 `reviewer` 서브 에이전트(sub-agent)를 추가한다. 거짓 양성(false-positive) 리뷰가 SWE-bench 통과율을 단일 에이전트 베이스라인 아래로 떨어뜨리는지 측정한다(힌트: 보통은 그렇다).

3. 샌드박스를 스트레스 테스트한다. 외부 URL을 `curl`하려는 작업과 worktree 바깥에 쓰려는 작업을 작성한다. 둘 다 PreToolUse 훅에 의해 차단되는지 확인한다. 시도를 로깅한다.

4. 더 작은 모델(Haiku 4.5)로 `PreCompact` 요약을 구현한다. 3배 압축에서 계획 충실도(fidelity)가 얼마나 손실되는지 측정한다.

5. MCP StreamableHTTP 전송을 stdio로 교체한다. 콜드 스타트와 호출당 지연 시간(latency)을 벤치마크(benchmark)한다. 로컬 전용 사용에 대한 승자를 고른다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 하네스(Harness) | "에이전트 루프" | 도구를 디스패치하고, 계획 상태를 유지하고, 예산을 강제하는, 모델을 둘러싼 코드 |
| 훅(Hook) | "에이전트 이벤트 리스너" | 하네스가 여덟 가지 라이프사이클 이벤트 중 하나에서 실행하는, 사용자가 작성한 스크립트 |
| Worktree | "Git 샌드박스" | 별도 경로에 연결된 git 체크아웃; 메인 클론을 건드리지 않고 폐기 가능 |
| TodoWrite | "계획 상태" | 모델이 매 턴 새로 쓰는, pending/in-progress/done 항목의 타입 지정 리스트 |
| StreamableHTTP | "MCP 전송" | 2026 MCP 개정판: 양방향 스트리밍을 갖춘 장수명 HTTP 연결; SSE를 대체 |
| 토큰 상한(Token ceiling) | "컨텍스트 예산" | 입력+출력 토큰에 대한 턴당 또는 세션당 상한; 압축 또는 종료를 트리거 |
| pass@1 | "단일 시도 통과율" | 재시도나 테스트셋 엿보기 없이 첫 실행에서 풀린 SWE-bench 작업의 비율 |

## 더 읽을거리 (Further Reading)

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code): Anthropic의 레퍼런스 하네스
- [Cursor 3 changelog](https://cursor.com/changelog): Agent Tabs 및 Composer 2 제품 노트
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent): SWE-bench 하네스 비교용 최소 베이스라인
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent): Opus 4.5로 SWE-bench Verified 79.2%
- [OpenCode](https://opencode.ai): 오픈 하네스, 스타 11.2만 개
- [SWE-bench Pro leaderboard](https://www.swebench.com): 이 캡스톤이 목표로 하는 평가
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/): StreamableHTTP, 능력 메타데이터
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/): 도구 호출과 토큰 사용량을 위한 스팬 스키마
