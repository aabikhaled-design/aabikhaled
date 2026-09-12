# 모더레이션 시스템: OpenAI, Perspective, Llama Guard

> 프로덕션 모더레이션(moderation) 시스템은 레슨 12-16에서 정의된 안전 정책을 실제로 작동시킨다. OpenAI Moderation API: GPT-4o를 기반으로 한 `omni-moderation-latest`(2024)는 텍스트 + 이미지를 한 번의 호출로 분류한다. 이전 버전보다 다국어 테스트 세트에서 42% 더 우수하다. 응답 스키마(schema)는 13개의 범주 불리언(boolean)을 반환한다. harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic. 대부분의 개발자에게 무료다. 계층화된 패턴: 입력 모더레이션(생성 전), 출력 모더레이션(생성 후), 맞춤 모더레이션(도메인 규칙). 비동기 병렬 호출이 지연 시간(latency)을 숨긴다. 플래그(flag) 시 자리표시자 응답. Llama Guard 3/4(레슨 16): 14개 MLCommons 위해(hazard), Code Interpreter Abuse, 8개 언어(v3), 다중 이미지(v4). Perspective API(Google Jigsaw): LLM-as-moderator 물결에 앞선 유해성(toxicity) 점수화. 주로 단일 차원 유해성에 severe-toxicity/insult/profanity 변형. 콘텐츠 모더레이션 연구의 베이스라인(baseline). 폐기 예정: Azure Content Moderator는 2024년 2월 폐기 예정 처리, 2027년 2월 퇴역, Azure AI Content Safety로 대체.

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- OpenAI Moderation API의 범주 분류 체계(taxonomy)와 그것이 Llama Guard 3의 MLCommons 집합과 어떻게 다른지 기술하기.
- 세 모더레이션 계층 패턴(입력, 출력, 맞춤)을 기술하고 각각의 실패 모드 하나씩을 명명하기.
- LLM 이전 시대의 베이스라인으로서의 Perspective API의 위치와, 그것이 연구에서 여전히 쓰이는 이유를 기술하기.
- Azure 폐기 일정을 진술하기.

## 문제 (The Problem)

레슨 12-16은 공격과 방어 도구를 기술한다. 레슨 29는 사용자가 제품과 접하는 표면에서 방어를 실제로 작동시키는, 배포된 모더레이션 시스템을 다룬다. 세 계층 패턴이 2026년의 기본 구성이다.

## 개념 (The Concept)

### OpenAI Moderation API

`omni-moderation-latest`(2024). GPT-4o 기반. 텍스트 + 이미지를 한 번의 호출로 분류한다. 대부분의 개발자에게 무료다.

범주(응답 스키마의 13개 불리언):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

멀티모달(multimodal) 지원은 `violence`, `self-harm`, `sexual`에 적용되지만 `sexual/minors`에는 적용되지 않는다. 나머지는 텍스트 전용이다.

`code/main.py`의 코드 하니스(harness)에서는 교육적 단순성을 위해 `/threatening`, `/intent`, `/instructions`, `/graphic` 하위 범주를 각각의 최상위 부모로 통합한다. 프로덕션 코드는 전체 13개 범주 스키마를 사용해야 한다.

이전 세대 모더레이션 엔드포인트(endpoint)보다 다국어 테스트 세트에서 42% 더 우수하다. 범주별 점수. 애플리케이션이 임계값을 설정한다.

### Llama Guard 3/4

레슨 16에서 다룸. 14개 MLCommons 위해 범주(OpenAI의 13개 응답 스키마 불리언과 다르게 조직됨). 8개 언어 지원(v3). Llama Guard 4(2025년 4월)는 본질적으로 멀티모달이며 12B다.

OpenAI와 Llama Guard 분류 체계는 겹치지만 발산한다. OpenAI는 "illicit"을 광범위한 범주로 가진다. Llama Guard는 "violent crimes"와 "non-violent crimes"를 별도로 가진다. 배포 환경마다 정책 분류 체계에 얼마나 들어맞는지를 보고 선택한다.

### Perspective API(Google Jigsaw)

LLM-as-moderator 물결에 앞선(2020년 이전) 유해성 점수화 시스템. 범주: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. 하위 차원 변형을 가진 단일 차원 주 점수(TOXICITY).

API가 안정적이고 문서화도 잘 되어 있으며 수년간의 보정(calibration) 데이터가 쌓여 있어 콘텐츠 모더레이션 연구 베이스라인으로 널리 쓰인다. 현대 LLM 인접 사용 사례에는 일반적으로 Llama Guard나 OpenAI Moderation이 더 적합하다.

### 세 계층 패턴

1. **입력 모더레이션.** 생성 전에 사용자의 프롬프트를 분류한다. 플래그되면 거부한다. 지연 시간: 분류기 호출 한 번.
2. **출력 모더레이션.** 전달 전에 모델의 출력을 분류한다. 플래그되면 거부 응답으로 교체한다. 지연 시간: 생성 후 분류기 호출 한 번.
3. **맞춤 모더레이션.** 도메인 특정 규칙(정규식, 허용 목록, 비즈니스 정책). 입력 또는 출력 중 어느 쪽에서든 실행된다.

세 계층은 설계상 순차적이다: 입력 모더레이션이 생성 전에 완료되어야 하고, 출력 모더레이션은 생성 후에 실행된다. 병렬성은 한 계층 내에서 적용된다. 동일한 텍스트에 대해 여러 분류기(예: OpenAI Moderation + Llama Guard + Perspective)를 동시에 실행하면 분류기별 지연 시간을 숨긴다. 선택적 최적화로, 입력 모더레이션이 완료되고 토큰-1 스트리밍(streaming)이 연기되는 동안 자리표시자 응답("잠시만요, 확인 중...")을 보여줄 수 있다. 플래그 동작은 구성 가능하다: 거부, 정화(sanitize), 사람 검토로 격상.

### 실패 모드

- **입력 전용.** 출력 환각(hallucination)을 잡지 못한다(레슨 12-14의 인코딩 공격이 입력 분류기를 우회한다).
- **출력 전용.** 모든 입력이 모델에 도달하도록 허용한다. 비용을 늘린다. 내부 추론을 공격자에게 노출한다.
- **맞춤 전용.** 범주 전반에 걸쳐 견고하지 않다. 정규식은 깨지기 쉽다.

계층화가 기본이다. 이중 안전장치(belt-and-suspenders).

### Azure 폐기

Azure Content Moderator: 2024년 2월 폐기 예정 처리, 2027년 2월 퇴역. Azure AI Content Safety로 대체되며, 이는 LLM 기반이고 Azure OpenAI와 통합된다. 마이그레이션(migration)은 Azure 배포에서 2024-2027년에 걸쳐 진행하는 현장 수준 프로젝트다.

### Phase 18에서 이 레슨의 위치

레슨 16은 레드팀(red-team) 맥락에서 모더레이션 도구를 다룬다. 레슨 29는 배포된 모더레이션을 다룬다. 레슨 30은 현재의 이중 용도(dual-use) 역량 증거로 마무리한다.

```figure
an-moderation-layers
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 세 계층 모더레이션 하니스를 구성한다: 입력 모더레이터(키워드 + 범주 점수), 출력 모더레이터(출력에 대한 동일 분류기), 맞춤 모더레이터(도메인 규칙). 입력을 통과시키며 어느 계층이 무엇을 잡는지 관찰할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-moderation-stack.md`를 생성한다. 배포가 주어지면, 모더레이션 스택(stack) 구성을 권고한다: 입력에 어떤 분류기, 출력에 어떤 분류기, 어떤 맞춤 규칙, 그리고 경계 사례(edge case)에 대한 어떤 판정자(judge).

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 무해한, 경계선의, 유해한 입력을 세 계층 모두를 통과시켜라. 각각에 대해 어느 계층이 발화하는지 보고하라.

2. 특정 범주에 대한 Perspective-API 방식의 유해성 점수화로 하니스를 확장하라. 그것의 임계값 동작을 범주 점수와 비교하라.

3. OpenAI Moderation API 문서와 Llama Guard 3 범주 목록을 읽어라. 각 OpenAI 범주를 가장 가까운 Llama Guard 범주에 대응시켜라. 깔끔하게 대응되지 않는 범주 셋을 식별하라.

4. 코드 어시스턴트 배포(예: GitHub Copilot)를 위한 모더레이션 스택을 설계하라. 가장 관련 있는 범주와 가장 관련 없는 범주를 식별하고 맞춤 규칙을 제안하라.

5. Azure Content Moderator는 2027년 2월 퇴역한다. Azure AI Content Safety로의 마이그레이션을 계획하라. 마이그레이션에서 가장 위험이 높은 요소를 식별하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | 부분적 멀티모달 지원을 가진 GPT-4o 기반 13개 범주(텍스트) 분류기 |
| Perspective API | "Google Jigsaw 유해성" | LLM 이전 시대의 유해성 점수화 베이스라인 |
| Llama Guard | "MLCommons 14개 범주" | Meta의 위해 분류기(v3: 8B 텍스트, 8개 언어; v4: 12B 멀티모달) |
| 입력 모더레이션(Input moderation) | "생성 전 필터" | 모델 호출 전 사용자 프롬프트에 대한 분류기 |
| 출력 모더레이션(Output moderation) | "생성 후 필터" | 전달 전 모델 출력에 대한 분류기 |
| 맞춤 모더레이션(Custom moderation) | "도메인 규칙" | 배포 특정 규칙(정규식, 허용 목록, 정책) |
| 계층화된 모더레이션(Layered moderation) | "세 계층 모두" | 표준 프로덕션 배포 패턴 |

## 더 읽을거리 (Further Reading)

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations): omni-moderation 엔드포인트
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama): Llama Guard 저장소
- [Google Jigsaw Perspective API](https://perspectiveapi.com/): 유해성 점수화
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/): Azure 대체물
