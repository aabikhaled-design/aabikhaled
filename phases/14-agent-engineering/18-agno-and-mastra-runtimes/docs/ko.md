# Agno와 Mastra: 프로덕션 런타임

> Agno(Python)와 Mastra(TypeScript)는 2026년 프로덕션 런타임(runtime) 짝이다. Agno는 마이크로초(microsecond) 단위 에이전트(agent) 인스턴스화와 무상태(stateless) FastAPI 백엔드를 목표로 한다. Mastra는 Vercel AI SDK 기반(substrate) 위에 에이전트, 도구, 워크플로(workflow), 통합 모델 라우팅(unified model routing), 복합 스토리지(composite storage)를 제공한다.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45분

## 학습 목표 (Learning Objectives)

- Agno의 성능 목표와 그것이 언제 중요한지 식별하기.
- Mastra의 세 가지 원시 요소(primitive)(Agents, Tools, Workflows)와 지원되는 서버 어댑터의 이름을 대기.
- 무상태 세션 한정(session-scoped) FastAPI 백엔드가 왜 권장되는 Agno 프로덕션 경로인지 설명하기.
- 주어진 스택(stack)에 대해 Agno 대 Mastra를 고르기(Python 우선 대 TypeScript 우선).

## 문제 (The Problem)

LangGraph, AutoGen, CrewAI는 프레임워크가 무겁다. "그냥 에이전트 루프, 빠르게, 내 런타임 안에서"를 원하는 팀들은 Agno(Python)나 Mastra(TypeScript)에 손을 뻗는다. 둘 다 프레임워크가 떠안던 원시 요소 일부를 포기하는 대신 원시 속도(raw speed)와 주변 스택에 더 잘 맞는 적합(fit)을 얻는다.

## 개념 (The Concept)

### Agno

- Python 런타임, 이전 명칭 Phi-data.
- "그래프도, 체인도, 복잡한 패턴도 없다. 그저 순수한 python."
- 문서상 성능 목표: ~2μs 에이전트 인스턴스화, 에이전트당 ~3.75 KiB 메모리, ~23개 모델 제공자.
- 프로덕션 경로: 무상태 세션 한정 FastAPI 백엔드. 각 요청이 새 에이전트를 시작하고, 세션 상태는 DB에 산다.
- 네이티브 멀티모달(텍스트, 이미지, 오디오, 비디오, 파일)과 에이전틱(agentic) RAG.

속도 목표는 초당 수천 개의 단명(short-lived) 에이전트가 있을 때(채팅 팬인(fan-in), 평가 파이프라인) 중요하다. 한 에이전트가 10분간 실행될 때는 덜 중요하다.

### Mastra

- TypeScript, Vercel AI SDK 위에 구축됨.
- 세 가지 원시 요소: **Agents**, **Tools**(Zod 타입), **Workflows**.
- 통합 모델 라우터(Unified Model Router): 94개 제공자에 걸친 3,300개 이상의 모델(2026년 3월).
- 복합 스토리지: 메모리, 워크플로, 관측 가능성(observability)을 서로 다른 백엔드로; 대규모 관측 가능성에는 ClickHouse 권장.
- Apache 2.0이며, `ee/` 디렉터리는 소스 공개(source-available) 엔터프라이즈 라이선스 하에 있다.
- Express, Hono, Fastify, Koa를 위한 서버 어댑터; 일급(first-class) Next.js와 Astro 통합.
- 디버깅을 위한 Mastra Studio(localhost:4111) 제공.
- 22k+ GitHub 스타, 1.0(2026년 1월) 시점 주당 300k+ npm 다운로드.

### 포지셔닝

둘 다 LangGraph가 되려고 하지 않는다. 경쟁이 갈리는 지점은 다음과 같다.

- **언어 적합.** Python 우선 팀에는 Agno; TypeScript 우선 팀에는 Mastra.
- **런타임 사용성(ergonomics).** Agno = 거의 0에 가까운 오버헤드; Mastra = Vercel 생태계와의 통합.
- **관측 가능성.** 둘 다 Langfuse/Phoenix/Opik(Lesson 24)와 통합되지만 Mastra Studio는 일급이다.

### 각각을 언제 고를 것인가

- **Agno**: Python 백엔드, 다수의 단명 에이전트, 강한 성능 요구, FastAPI 매장(shop).
- **Mastra**: TypeScript 백엔드, Next.js / Vercel 배포, 통합 멀티 프로바이더 모델 라우팅, Zod 타입 도구.
- **LangGraph**(Lesson 13): 내구성 있는 상태와 명시적 그래프 추론이 원시 속도보다 더 중요할 때.
- **OpenAI / Claude Agent SDK**: 제공자의 제품화된 형태를 원할 때(Lesson 16-17).

### 이 패턴이 잘못되는 지점

- **성능을 위한 성능.** 워크로드가 요청당 한 번의 느린 에이전트 호출일 때 "2μs"가 좋게 들린다는 이유로 Agno를 고름. 오버헤드는 병목이 아니다.
- **생태계 종속(lock-in).** Mastra의 Vercel 풍 통합은 Vercel에서는 장점, 다른 곳에서는 단점이다.
- **엔터프라이즈 라이선스 혼동.** Mastra의 `ee/` 디렉터리는 Apache 2.0이 아니라 소스 공개다. 포크(fork)를 계획한다면 라이선스를 읽어라.

```figure
wb-runtime-spawn
```

## 직접 만들기 (Build It)

이 레슨은 주로 비교(comparative)에 초점을 둔다. 단일 코드 산출물 하나로는 두 프레임워크를 제대로 담아내기 어렵다. 나란히 놓인 토이는 `code/main.py`를 보라: 최소한의 "에이전트를 실행하고, 출력을 스트리밍하며, 세션을 영속한다" 흐름을 두 번 구현한 것(한 번은 Agno 형태로, 한 번은 Mastra 형태로).

실행:

```
python3 code/main.py
```

구조적으로는 다르지만 기능적으로는 동등한 두 개의 트레이스.

## 라이브러리로 써보기 (Use It)

- **Agno**: 속도와 FastAPI 형태가 필요한 Python 백엔드.
- **Mastra**: 다수의 제공자와 워크플로 원시 요소를 가진 TypeScript 백엔드.
- 둘 다 일급 관측 가능성 후크를 제공한다. 둘 다 Langfuse와 통합된다.

## 산출물 (Ship It)

`outputs/skill-runtime-picker.md`는 스택, 지연 시간 예산, 운영 형태를 바탕으로 Agno, Mastra, LangGraph, 또는 제공자 SDK를 고른다.

## 연습 문제 (Exercises)

1. Agno의 문서를 읽어라. stdlib ReAct 루프(Lesson 01)를 Agno로 포팅하라. 무엇이 사라졌는가? 무엇이 남았는가?
2. Mastra의 문서를 읽어라. 같은 루프를 Mastra로 포팅하라. 도구 타이핑(Zod 대 아무것도 없음)에서 무엇이 바뀌었는가?
3. 벤치마크: 당신의 스택에서 에이전트 인스턴스화 지연 시간을 측정하라. Agno의 2μs가 당신의 워크로드에 중요한가?
4. 마이그레이션을 설계하라: Python에서 CrewAI를 운영해 왔다면, Agno로 옮길 때 무엇이 깨지는가?
5. Mastra의 `ee/` 라이선스 조항을 읽어라. 오픈소스 포크에 영향을 줄 제한은 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| Agno | "빠른 Python 에이전트" | 무상태 세션 한정 에이전트 런타임 |
| Mastra | "Vercel AI SDK 위의 TypeScript 에이전트" | Agents + Tools + Workflows + Model Router |
| 통합 모델 라우터(Unified Model Router) | "멀티 프로바이더 접근" | 94개 제공자에 걸친 3,300개 이상의 모델을 위한 단일 클라이언트 |
| 복합 스토리지(Composite storage) | "다중 백엔드" | 메모리/워크플로/관측 가능성을 각각 다른 저장소로 |
| Mastra Studio | "로컬 디버거" | 에이전트를 내성(introspect)하기 위한 localhost:4111 UI |
| 소스 공개(Source-available) | "OSS 아님" | 라이선스가 소스 읽기는 허용하나 상업적 사용은 제한 |

## 더 읽을거리 (Further Reading)

- [Agno Agent Framework docs](https://www.agno.com/agent-framework): 성능 목표, FastAPI 통합
- [Mastra docs](https://mastra.ai/docs): 원시 요소, 서버 어댑터, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 상태 보존 그래프 대안
- [Comet Opik](https://www.comet.com/site/products/opik/): Mastra 통합이 인용하는 관측 가능성 비교
