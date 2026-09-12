# 브라우저 에이전트와 장기 지평 웹 과제

> ChatGPT agent(2025년 7월)는 Operator와 deep research를 하나의 브라우저/터미널 에이전트(agent)로 합치고, BrowseComp에서 68.9%로 SOTA를 세웠다. OpenAI는 2025년 8월 31일 Operator를 종료했다. 제품 층에서 통합한 것이다. Anthropic의 Vercept 인수는 OSWorld에서 Claude Sonnet을 15% 미만에서 72.5%로 끌어올렸다. WebArena-Verified(ServiceNow, ICLR 2026)는 원본 WebArena의 거짓 음성률(false-negative rate) 11.3퍼센트포인트를 고치고, 258개 과제의 Hard 부분집합을 출시했다. 수치는 진짜다. 공격 표면(attack surface)도 진짜다. OpenAI의 대비(preparedness) 책임자는 브라우저 에이전트로의 간접 프롬프트 주입(indirect prompt injection)이 "완전히 패치할 수 있는 버그가 아니다"라고 공개적으로 밝혔다. 문서화된 2025~2026년 공격들: Tainted Memories(Atlas CSRF), HashJack(Cato Networks), 그리고 Perplexity Comet의 원클릭 하이재킹.

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45분

## 문제 (The Problem)

브라우저 에이전트는 신뢰할 수 없는 콘텐츠를 읽고 중대한 액션을 취하는 장기 지평(long-horizon) 에이전트다. 에이전트가 방문하는 모든 페이지는 사용자가 작성하지 않은 입력이다. 모든 페이지의 모든 폼(form)은 잠재적 명령 채널이다. 2025~2026년 공격 말뭉치는 이것이 가설이 아님을 보여준다. Tainted Memories는 조작된 페이지를 통해 공격자가 악성 명령을 에이전트의 메모리에 묶게 한다. HashJack은 에이전트가 방문하는 URL 프래그먼트(fragment)에 명령을 숨긴다. Perplexity Comet 하이재킹은 단 한 번의 클릭으로 적중했다.

방어 그림은 불편하다. OpenAI의 대비 책임자는 조용히 묻혀 있던 부분을 크게 말했다. 간접 프롬프트 주입은 "완전히 패치할 수 있는 버그가 아니다." 이는 공격이 에이전트의 읽기-대-행동 경계에 살기 때문인데, 그 경계는 아키텍처적으로 흐릿하다. 모델이 읽는 모든 토큰(token)은 원리상 명령으로 읽힐 수 있다.

이 레슨은 공격 표면을 명명하고, 벤치마크 지형(BrowseComp, OSWorld, WebArena-Verified)을 명명하며, 최소한의 간접 프롬프트 주입 시나리오를 모델링하여, Lesson 14와 18에서 실제 방어를 추론할 수 있게 한다.

## 개념 (The Concept)

### 2026 지형, 시스템당 한 문단으로

**ChatGPT agent (OpenAI).** 2025년 7월 출시. Operator(브라우징)와 Deep Research(수 시간짜리 리서치)를 통합. 2025년 8월 31일 독립형 Operator 종료. BrowseComp에서 68.9%로 SOTA; OSWorld와 WebArena-Verified에서 강한 수치.

**Claude Sonnet + Vercept (Anthropic).** Anthropic의 Vercept 인수는 컴퓨터 사용(computer-use) 능력에 초점을 맞췄다. OSWorld에서 Claude Sonnet을 15% 미만에서 72.5%로 이동시켰다. Claude Computer Use는 도구 API로 출시된다.

**Gemini 3 Pro with Browser Use (DeepMind).** Browser Use 통합은 컴퓨터 사용 제어를 제공한다. FSF v3(2026년 4월, Lesson 20)는 특히 ML R&D 영역에서의 자율성을 추적한다.

**WebArena-Verified (ServiceNow, ICLR 2026).** 잘 문서화된 문제를 고친다. 원본 WebArena는 약 11.3%의 거짓 음성률을 가졌다(실제로는 해결된 과제가 실패로 표시됨). Verified 릴리스는 사람이 큐레이션한 성공 기준으로 재채점하고 258개 과제의 Hard 부분집합을 추가한다(ICLR 2026 논문, openreview.net/forum?id=94tlGxmqkN).

### BrowseComp vs OSWorld vs WebArena

| 벤치마크 | 측정 대상 | 지평 |
|---|---|---|
| BrowseComp | 시간 압박 하에서 열린 웹에서 특정 사실 찾기 | 분 |
| OSWorld | 전체 데스크톱을 조작하는 에이전트(마우스, 키보드, 셸) | 수십 분 |
| WebArena-Verified | 시뮬레이션된 사이트에서의 거래성 웹 과제 | 분 |
| Hard 부분집합 | 다중 페이지 상태 전이가 있는 WebArena-Verified 과제 | 수십 분 |

서로 다른 축이다. 높은 BrowseComp 점수는 에이전트가 사실을 찾는다는 뜻이지, 항공편을 예약할 수 있다는 뜻은 아니다. OSWorld 점수는 "내 데스크톱에서 작동하는가"에 더 가깝다. WebArena-Verified는 "흐름을 완료할 수 있는가"에 더 가깝다. 어떤 프로덕션 결정이든 과제 분포와 일치하는 벤치마크가 필요하다.

### 공격 표면, 명명하기

1. **간접 프롬프트 주입.** 신뢰할 수 없는 페이지 콘텐츠가 명령을 담는다. 에이전트가 그것을 읽는다. 에이전트가 그것을 실행한다. 공개 사례: 2024년 Kai Greshake et al., 2025년 Tainted Memories 논문, 2026년 HashJack(Cato Networks).
2. **URL 프래그먼트 / 쿼리 주입.** 크롤링된 URL의 `#fragment`나 쿼리 문자열이 명령을 담는다. 결코 가시적으로 렌더링되지 않지만 여전히 에이전트의 컨텍스트 안에 있다.
3. **메모리 바인딩 공격.** 페이지가 에이전트에게 지속(persistent) 메모리를 쓰도록 지시한다(Lesson 12가 지속 상태를 다룬다). 다음 세션에서 메모리가 가시적 트리거 없이 페이로드를 발사한다.
4. **인증된 세션에 대한 CSRF 형태 공격.** Tainted Memories 부류: 에이전트가 어딘가에 로그인되어 있고; 공격자의 페이지가 사용자의 쿠키로 에이전트가 실행하는 상태 변경 요청을 발행한다.
5. **원클릭 하이재킹.** 시각적으로 무해한 버튼이 에이전트가 따르는 페이로드를 태우고 간다. Comet 부류.
6. **에이전트 호스트 표면의 Content-Security-Policy 구멍.** 렌더링 및 도구 층 자체가 공격 벡터가 될 수 있다. 브라우저-안의-브라우저-에이전트 스택은 넓다.

### "완전히 패치할 수 없는" 이유

공격은 에이전트의 능력과 동형(isomorphic)이다. 에이전트는 자기 일을 하려면 신뢰할 수 없는 콘텐츠를 읽어야 한다. 에이전트가 읽는 모든 콘텐츠는 명령을 담을 수 있다. 에이전트가 따르는 모든 명령은 사용자의 실제 요청과 정렬에서 벗어날 수 있다. 방어(신뢰 경계, 분류기(classifier), 도구 허용 목록(allowlist), 중대한 액션에 대한 HITL)는 공격의 비용을 올리고 폭발 반경(blast radius)을 줄인다. 그것들이 이 부류를 닫지는 못한다.

이것은 Löb의 정리(Lesson 8)와 같은 추론 패턴이다. 에이전트는 다음 토큰이 안전하다고 증명할 수 없다. 안전하지 않은 토큰이 더 탐지 가능한 시스템을 세울 수 있을 뿐이다.

### 실제로 출하되는 방어 태세

- **읽기 / 쓰기 경계.** 읽기는 결코 중대하지 않다. 쓰기(폼 제출, 콘텐츠 게시, 부작용이 있는 도구 호출)는 개시 콘텐츠가 신뢰 경계 밖에서 왔다면 새로운 사람 승인을 요구한다.
- **과제별 도구 허용 목록.** 에이전트는 브라우징할 수 있다. 그 도구가 과제를 위해 명시적으로 활성화되지 않았다면 송금을 개시할 수 없다. Lesson 13이 예산을 다룬다.
- **세션 격리.** 브라우저 에이전트 세션은 범위가 한정된 자격 증명(credentials)으로만 돈다. 프로덕션 인증 없음, 개인 이메일 없음. 모든 HTTP 요청의 로그를 감사를 위해 보존한다.
- **콘텐츠 살균기(sanitizer).** 가져온 HTML은 모델 컨텍스트에 이어 붙이기 전에 알려진-나쁜 패턴이 제거된다. (쉬운 공격을 줄인다. 정교한 페이로드를 막지는 못한다.)
- **중대한 액션에 대한 HITL.** 제안-후-커밋(propose-then-commit) 패턴(Lesson 15).
- **메모리에 대한 카나리아 토큰(canary token).** 메모리 항목이 발사되면 사용자가 그것을 본다(Lesson 14).

```figure
injection-boundary
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 세 개의 합성 페이지에 대해 작은 브라우저 에이전트 실행을 모델링한다. 한 페이지는 양성(benign)이고, 한 페이지는 가시적 텍스트에 직접 프롬프트 주입 덩어리가 있으며, 한 페이지는 URL 프래그먼트 주입(가시적이지 않지만 에이전트의 컨텍스트 안에 있음)이 있다. 스크립트는 (a) 순진한 에이전트가 무엇을 할지, (b) 읽기/쓰기 경계가 무엇을 잡는지, (c) 살균기가 무엇을 잡는지, (d) 둘 다 잡지 못하는 것이 무엇인지를 보여준다.

## 산출물 (Ship It)

`outputs/skill-browser-agent-trust-boundary.md`는 제안된 브라우저 에이전트 배포(deployment)의 범위를 정한다: 어떤 신뢰 구역을 건드리는지, 무엇을 쓰도록 인가되었는지, 첫 실행 전에 어떤 방어가 갖춰져야 하는지.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 살균기는 잡지만 읽기/쓰기 경계는 잡지 못하는 공격, 그리고 읽기/쓰기 경계만 잡는 공격을 식별하라.

2. 살균기를 확장하여 HashJack 스타일 URL 프래그먼트 주입의 한 부류를 탐지하라. 합법적 프래그먼트가 있는 양성 URL에서 거짓 양성률(false-positive rate)을 측정하라.

3. 알고 있는 실제 브라우저 에이전트 워크플로 하나를 골라라(예: "항공편 예약"). 모든 읽기와 모든 쓰기를 나열하라. 어느 쓰기가 HITL이 필요하며 왜 그런지 표시하라.

4. WebArena-Verified ICLR 2026 논문을 읽어라. 원본 WebArena의 채점이 신뢰할 수 없었던 과제 범주 하나를 식별하고, Verified 부분집합이 그것을 어떻게 해결하는지 설명하라.

5. 브라우저 에이전트 환경을 위한 메모리 카나리아를 설계하라. 무엇을, 어디에 저장하고, 무엇이 경보를 촉발하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 간접 프롬프트 주입 (Indirect prompt injection) | "나쁜 페이지 텍스트" | 에이전트가 읽는 페이지의 신뢰할 수 없는 콘텐츠가 에이전트가 실행하는 명령을 담음 |
| Tainted Memories | "메모리 공격" | 에이전트가 공격자가 제공한 명령을 지속 메모리에 씀; 다음 세션에 촉발됨 |
| HashJack | "URL 프래그먼트 공격" | URL 프래그먼트/쿼리 문자열에 숨겨진 페이로드가 에이전트의 컨텍스트에는 있지만 가시적으로 렌더링되지 않음 |
| 원클릭 하이재킹 (One-click hijack) | "나쁜 버튼" | 가시적 어포던스(affordance)가 에이전트가 실행하는 후속 페이로드를 태우고 감 |
| BrowseComp | "웹 검색 벤치마크" | 열린 웹에서 특정 사실 찾기; 분 단위 지평 |
| OSWorld | "데스크톱 벤치마크" | 전체 OS 제어; 다단계 GUI 과제 |
| WebArena-Verified | "고친 웹 과제 벤치마크" | Hard 부분집합이 있는 ServiceNow의 재채점 WebArena |
| 읽기/쓰기 경계 (Read/write boundary) | "부작용 게이트" | 읽기는 결코 중대하지 않음; 콘텐츠가 신뢰 밖이면 쓰기는 새로운 승인을 요구 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) Operator와 deep research의 병합; BrowseComp SOTA.
- [OpenAI: Computer-Using Agent](https://openai.com/index/computer-using-agent/) Operator 계보와 ChatGPT agent가 된 아키텍처.
- [Zhou et al.: WebArena](https://webarena.dev/) 원본 벤치마크.
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN): ICLR 2026 고친 부분집합 논문.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 컴퓨터 사용 에이전트에 대한 공격 표면 논의 포함.
