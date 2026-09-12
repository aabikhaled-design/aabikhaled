# Claude Agent SDK: 서브에이전트와 세션 저장소

> Claude Agent SDK는 Claude Code 하니스(harness)의 라이브러리 형태다. 내장 도구, 컨텍스트 격리(context isolation)를 위한 서브에이전트(subagent), 후크(hook), W3C 트레이스 전파(trace propagation), 세션 저장소 동등성(parity). Claude Managed Agents는 장기 실행 비동기 작업을 위한 호스팅 대안이다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- Anthropic Client SDK(원시 API)와 Claude Agent SDK(하니스 형태)의 차이를 설명하기.
- 서브에이전트(병렬화(parallelization)와 컨텍스트 격리)를 기술하고, 언제 쓰면 좋을지 설명하기.
- Python SDK의 세션 저장소 표면(`append`, `load`, `list_sessions`, `delete`, `list_subkeys`)의 이름을 대고 `--session-mirror`의 역할을 설명하기.
- 내장 도구, 격리된 컨텍스트를 가진 서브에이전트 생성, 생명주기 후크(lifecycle hook), 세션 저장소를 갖춘 stdlib 하니스를 구현하기.

## 문제 (The Problem)

원시 LLM API는 한 번의 왕복(round-trip)을 준다. 프로덕션 에이전트(agent)는 도구 실행, MCP 서버, 생명주기 후크, 서브에이전트 생성, 세션 영속성(persistence), 트레이스 전파가 필요하다. Claude Agent SDK는 이 형태를 라이브러리로 제공한다. Claude Code가 사용하는 바로 그 하니스를, 커스텀 에이전트를 위해 노출한 것.

## 개념 (The Concept)

### Client SDK 대 Agent SDK

- **Client SDK (`anthropic`).** 원시 Messages API. 루프, 도구, 상태를 직접 관리한다.
- **Agent SDK (`claude-agent-sdk`).** 내장 도구 실행, MCP 연결, 후크, 서브에이전트 생성, 세션 저장소. 라이브러리로서의 Claude Code 루프.

### 내장 도구

SDK는 10개 이상의 도구를 기본 제공한다: 파일 읽기/쓰기, 셸(shell), grep, glob, 웹 페치(web fetch), 그 외 더. 커스텀 도구는 표준 도구 스키마 인터페이스를 통해 등록한다.

### 서브에이전트

Anthropic이 문서화한 두 가지 목적:

1. **병렬화.** 독립적인 작업을 동시에 실행한다. "이 20개 모듈 각각에 대한 테스트 파일을 찾아라"는 20개의 병렬 서브에이전트 작업이다.
2. **컨텍스트 격리.** 서브에이전트는 자신의 컨텍스트 윈도우(context window)를 사용하고, 결과만 오케스트레이터(orchestrator)로 반환된다. 오케스트레이터의 예산이 보존된다.

Python SDK 최근 추가: 서브에이전트 트랜스크립트(transcript)를 읽기 위한 `list_subagents()`, `get_subagent_messages()`.

### 세션 저장소

TypeScript와의 프로토콜 동등성:

- `append(session_id, message)`: 턴을 추가한다.
- `load(session_id)`: 대화를 복원한다.
- `list_sessions()`: 열거한다.
- `delete(session_id)`: 서브에이전트 세션으로 연쇄(cascade)된다.
- `list_subkeys(session_id)`: 서브에이전트 키를 나열한다.

`--session-mirror`(CLI 플래그)는 디버깅을 위해 트랜스크립트가 스트리밍되는 대로 외부 파일에 미러링한다.

### 후크 (Hooks)

등록할 수 있는 생명주기 후크:

- `PreToolUse`, `PostToolUse`: 도구 호출을 게이트(gate)하거나 감사한다.
- `SessionStart`, `SessionEnd`: 설정하고 해체한다.
- `UserPromptSubmit`: 모델이 보기 전에 사용자 입력에 대해 행동한다.
- `PreCompact`: 컨텍스트 압축(compaction) 전에 실행한다.
- `Stop`: 에이전트 종료 시 정리한다.
- `Notification`: 사이드 채널(side-channel) 알림.

후크는 pro-workflow(Phase 14 커리큘럼 참조)와 유사 시스템이 횡단(cross-cutting) 동작을 추가하는 방법이다.

### W3C 트레이스 컨텍스트

호출자에서 활성인 OTel 스팬(span)은 W3C 트레이스 컨텍스트 헤더를 통해 CLI 하위 프로세스(subprocess)로 전파된다. 전체 멀티 프로세스 트레이스가 백엔드에서 하나의 트레이스로 나타난다.

### Claude Managed Agents

호스팅 대안(베타 헤더 `managed-agents-2026-04-01`). 장기 실행 비동기 작업, 내장 프롬프트 캐싱(prompt caching), 내장 압축. 관리형 인프라를 얻는 대신 제어권을 내준다.

### 이 패턴이 잘못되는 지점

- **서브에이전트 과잉 생성.** 100개의 작은 작업을 위해 100개의 서브에이전트를 생성한다. 오버헤드가 지배한다. 대신 배치(batch)하라.
- **후크 증식(creep).** 모든 팀이 후크를 추가하고, 시작 시간이 부풀어 오른다. 후크를 분기마다 검토하라.
- **세션 비대화.** 세션이 누적되고, 크기가 자란다. `list_sessions` + 만료(expiry) 정책을 사용하라.

```figure
ae-subagent-isolation
```

## 직접 만들기 (Build It)

`code/main.py`는 SDK 형태를 stdlib로 구현한다.

- 내장 `read_file`, `write_file`, `list_dir`을 가진 `Tool`, `ToolRegistry`.
- `Subagent`: 사설 컨텍스트, 격리된 실행, 반환되는 결과.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- 데모: 메인 에이전트가 3개의 서브에이전트를 병렬로(각각 격리) 생성하고, 결과를 집계하며, 세션을 영속한다.

실행:

```
python3 code/main.py
```

트레이스는 서브에이전트 컨텍스트 격리(오케스트레이터 컨텍스트 크기가 유한하게 유지됨), 후크 실행, 세션 영속성을 보여준다.

## 라이브러리로 써보기 (Use It)

- **Claude Agent SDK**: Claude Code 하니스 형태를 원하는 Claude 우선 제품용.
- **Claude Managed Agents**: 호스팅 장기 실행 비동기 작업용.
- **OpenAI Agents SDK**(Lesson 16): OpenAI 우선 대응물용.
- **LangGraph + 커스텀 도구**: 대신 그래프 형태의 상태 기계를 원할 때.

## 산출물 (Ship It)

`outputs/skill-claude-agent-scaffold.md`는 서브에이전트, 후크, 세션 저장소, MCP 서버 연결, W3C 트레이스 전파를 갖춘 Claude Agent SDK 앱을 스캐폴딩한다.

## 연습 문제 (Exercises)

1. 20개 작업을 5개 병렬 서브에이전트 그룹으로 배치하는 서브에이전트 생성기를 추가하라. 작업당 하나 대비 오케스트레이터 컨텍스트 크기를 측정하라.
2. `write_file` 호출을 (세션당 분당 5회로) 레이트 리밋(rate-limit)하는 `PreToolUse` 후크를 구현하라. 동작을 추적하라.
3. `list_subkeys`를 배선하여 서브에이전트 트리를 렌더링하라. 깊은 중첩(nesting)은 어떻게 보이는가?
4. 토이를 진짜 `claude-agent-sdk` Python 패키지로 포팅하라. 도구 등록에 대해 무엇이 바뀌는가?
5. Claude Managed Agents 문서를 읽어라. 언제 자체 호스팅에서 관리형으로 전환하겠는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Agent SDK | "라이브러리로서의 Claude Code" | 하니스 형태: 도구, MCP, 후크, 서브에이전트, 세션 저장소 |
| 서브에이전트(Subagent) | "자식 에이전트" | 별도 컨텍스트, 자체 예산; 결과가 위로 올라옴 |
| 세션 저장소(Session store) | "대화 DB" | 서브에이전트 연쇄와 함께 턴을 영속, 로드, 나열, 삭제 |
| 후크(Hook) | "생명주기 콜백" | 도구 전/후, 세션, 프롬프트 제출, 압축, 중지 |
| W3C 트레이스 컨텍스트(W3C trace context) | "프로세스 간 트레이스" | 부모 스팬이 CLI 하위 프로세스로 전파됨 |
| Managed Agents | "호스팅 하니스" | Anthropic 호스팅 장기 실행 비동기 작업 |
| `--session-mirror` | "트랜스크립트 미러" | 세션 턴을 스트리밍되는 대로 외부 파일에 씀 |
| MCP 서버 | "도구 표면" | 에이전트에 연결된 외부 도구/리소스 소스 |

## 더 읽을거리 (Further Reading)

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview): Claude Code의 라이브러리 형태
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk): 프로덕션 패턴
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview): 호스팅 대안
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/): 대응물
