# Capstone 17: 개인 AI 튜터 (Personal AI Tutor: Adaptive, Multimodal, with Memory)

> Khanmigo(Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat, 그리고 Synthesis Tutor는 모두 2026년에 적응형(adaptive) 멀티모달(multimodal) 튜터링을 대규모로 출하했다. 공통 형태는 소크라테스식(Socratic) 정책(절대 답을 그냥 던지지 않음), 모든 상호작용 후 갱신되는 학습자 모델(learner model)(베이지안 지식 추적(Bayesian knowledge tracing) 스타일), 음성 + 텍스트 + 사진 수학(photo-math) 입력, 커리큘럼 그래프(curriculum graph) 검색, 간격 반복(spaced-repetition) 스케줄링, 연령 적합 콘텐츠를 위한 강한 안전 필터다. 이 캡스톤(capstone)은 과목 특화 튜터(K-12 대수 또는 입문 Python)를 출하하고, 학습자 10명과 2주간의 효능 연구(efficacy study)를 실행하며, 콘텐츠 안전 감사(content-safety audit)를 통과하는 것이다.

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30시간

## 문제 (Problem)

적응형 튜터링은 한때 에듀테크 연구의 틈새였다. 2026년에는 소비자 제품이다. Khanmigo는 대부분의 미국 학군에 배포되어 있다. Duolingo Max는 수천만 MAU에 도달했다. Google의 LearnLM / Gemini for Education은 Google Classroom에서 튜터링을 구동한다. Quizlet Q-Chat은 플래시카드 옆에 자리한다. Synthesis Tutor는 호기심 많은 아이들을 위한 튜터로 바이럴이 되었다. 공통 요소는 멀티모달 입력(타이핑, 말하기, 방정식 촬영), 소크라테스식 교수법(먼저 묻고 나중에 설명), 각 상호작용 후 갱신되는 학습자 모델, 엄격한 연령 적합 안전성이다.

이번에는 특정 코호트(cohort)를 위해 이런 튜터 하나를 직접 만든다. 측정 기준은 실제 효능 연구다: 학습자 10명과 함께 2주에 걸친 사전 시험과 사후 시험 점수. 음성 루프(voice loop)는 자연스럽게 느껴져야 한다(캡스톤 03 서브 스택). 메모리(memory)는 프라이버시를 존중해야 한다. 안전 필터는 K-12를 위한 COPPA 인지(COPPA-aware) 레드팀을 통과해야 한다.

## 개념 (Concept)

네 가지 구성 요소. **튜터 정책(Tutor policy)**은 소크라테스식 루프다: 학습자가 답을 요청하면, 정책은 유도 질문을 던진다. 학습자가 맞히면, 다음 개념으로 넘어간다. 막히면, 비계화된(scaffolded) 힌트를 제공한다. **학습자 모델(Learner model)**은 각 상호작용 후 커리큘럼 노드별 숙달(mastery) 확률을 갱신하는 베이지안 지식 추적(또는 간단한 변형)이다. **커리큘럼 그래프(Curriculum graph)**는 선수(prerequisite) 간선을 갖는 개념의 Neo4j다. 정책은 그래프를 순회하여 다음 개념을 고른다. **메모리(Memory)**는 과거 상호작용, 실수, 선호를 담는 에피소드(episodic) + 시맨틱(semantic) 저장소(agentmemory 스타일)다.

UX는 멀티모달이다. 타이핑한 답변은 텍스트로 입력한다. 음성 입력은 LiveKit + Whisper로 받는다(캡스톤 03 재사용). 수학 문제 사진은 dots.ocr 또는 PaliGemma 2로 처리한다. 음성 출력은 Cartesia Sonic-2가 맡는다. 안전성은 Llama Guard 4에 더해 연령 적합 필터(성인 콘텐츠, 폭력, 자해 차단)와 COPPA 인지 메모리 보존 정책을 사용한다.

효능 연구가 결과물이다. 학습자 10명, 사전 시험과 사후 시험, 2주. 학습 향상 차이와 신뢰 구간(confidence interval)을 보고한다. 비적응형 베이스라인(튜터 정책 없이 동일 콘텐츠를 선형으로 전달)과 비교한다.

## 아키텍처 (Architecture)

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 스택 (Stack)

- 과목 선택: K-12 대수 또는 입문 Python (깊이를 위해 하나 선택)
- 튜터 정책: Claude Sonnet 4.7 위의 LangGraph (프롬프트 캐싱 포함)
- 학습자 모델: 베이지안 지식 추적(고전) 또는 간격을 위한 FSRS
- 커리큘럼 그래프: 개념 + 선수 간선 + OER 콘텐츠의 Neo4j
- 메모리: agentmemory 스타일 영속 벡터 + 에피소드 + 시맨틱 저장소
- 음성: LiveKit Agents 1.0 + Cartesia Sonic-2 (캡스톤 03 서브 스택 재사용)
- 사진 수학: 방정식 인식을 위한 dots.ocr 또는 PaliGemma 2
- 안전: Llama Guard 4 + 커스텀 연령 적합 필터
- 평가: Bloom 수준 문제 생성, 사전/사후 시험 하네스, 효능 연구 도구

```figure
cf-tutor-loop
```

## 직접 만들기 (Build It)

1. **커리큘럼 그래프.** 선수 간선을 갖는 50-150개 개념 노드(예: "수직선(number line)"부터 "근의 공식(quadratic formula)"까지의 K-12 대수)의 Neo4j를 만든다. 노드별로 OER 콘텐츠(Open Textbook, OpenStax)를 붙인다.

2. **학습자 모델.** 사전 확률(prior)로 베이지안 지식 추적을 초기화한다: guess, slip, learn-rate. 각 상호작용 후 개념별 숙달을 갱신한다. 학습자별로 영속화한다.

3. **튜터 정책.** 다음 노드를 갖는 LangGraph: `read_signal`(학습자의 답이 정답 / 부분 정답 / 막힘이었나?), `select_concept`(최우선 개념을 고르며 커리큘럼 그래프 순회), `scaffold`(소크라테스식 프롬프트), `update_mastery`.

4. **메모리.** 모든 상호작용이 에피소드 저장소에 기록된다. 실수와 선호는 시맨틱 메모리로 승격된다. COPPA 인지 보존 정책: 1년 후 자동 삭제, 부모 접근 가능.

5. **음성 경로.** 튜터 정책에 부착된 LiveKit Agents 워커. ASR은 Whisper-v3-turbo, TTS는 Cartesia Sonic-2로 처리한다. 바지인(barge-in) 지원(캡스톤 03 메커니즘 재사용).

6. **사진 수학 경로.** 이미지를 업로드하거나 촬영한다. dots.ocr 또는 PaliGemma 2를 실행해 방정식을 인식한다. 구조화된 입력으로 튜터에 공급한다.

7. **안전.** 모든 모델 출력은 Llama Guard 4 + 연령 적합 필터(자해, 성인 콘텐츠, 폭력 차단)를 통과한다. 메모리 접근은 학습자 ID로 스코핑된다. 삭제를 위한 부모 접근 표면.

8. **효능 연구.** 학습자 10명, 사전 시험(표준화된 30문항 베이스라인), 2주간의 튜터 상호작용(주 3회), 사후 시험. 동일 콘텐츠에 대해 비적응형 베이스라인 코호트 10명과 비교한다.

9. **주간 진척 보고서.** 학습자별로 탐구한 주제, 숙달 궤적, 권장 다음 단계의 PDF 요약을 자동 생성한다.

## 라이브러리로 써보기 (Use It)

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 산출물 (Ship It)

`outputs/skill-ai-tutor.md`가 결과물이다. 멀티모달 입력, 학습자 모델, 메모리, 안전성, 측정된 효능까지 갖춘 과목 특화 적응형 튜터.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | 학습 향상 차이 | 10명 학습자 2주 연구에서의 사전/사후 시험 차이 |
| 20 | 소크라테스식 충실도 | 트랜스크립트 샘플에 대한 루브릭 점수 |
| 20 | 멀티모달 UX | 음성 + 사진 + 텍스트의 종단 간 일관성 |
| 20 | 안전 + 프라이버시 태세 | Llama Guard 4 통과율 + COPPA 인지 보존 |
| 15 | 커리큘럼 폭과 그래프 품질 | 개념 커버리지 + 선수 그래프 일관성 |
| **100** | | |

## 연습 문제 (Exercises)

1. 적응형 학습자 모델 유무(무작위 개념 순서)에 따라 효능 연구를 실행한다. 차이를 보고한다. 적응형이 이길 것으로 예상되지만, 그 크기가 흥미로운 숫자다.

2. 멀티모달 프로브를 추가한다: 동일 개념 질문을 텍스트, 음성, 사진으로 전달한다. 학습자가 선호하는 양식(modality)으로 더 빨리 수렴하는지 측정한다.

3. 부모 대시보드를 만든다: 연습한 주제, 숙달 궤적, 다가올 개념, 안전 이벤트(가드레일 적중 여부). COPPA 정렬.

4. 언어 전환 모드를 추가한다: 튜터가 스페인어 입력을 받고 스페인어로 가르친다. X-Guard 커버리지를 측정한다.

5. 메모리 프라이버시를 압박한다: 학습자 A가 음성 클립 재수집(re-ingest) 공격을 통해서도 학습자 B의 데이터를 볼 수 없는지 검증한다. 시도된 접근을 로깅하고 경보한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| Socratic policy | "던지지 말고 물어라" | 튜터가 답을 주는 대신 유도 질문을 함 |
| Bayesian knowledge tracing | "BKT" | 개념별 숙달 확률을 위한 고전적 학습자 모델 방정식 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 간격 반복 스케줄러, SM-2보다 우수 |
| Curriculum graph | "개념 DAG" | 선수 간선을 갖는 개념의 Neo4j |
| Episodic memory | "상호작용별 로그" | 나중 검색을 위해 저장된 모든 상호작용 |
| Semantic memory | "학습된 패턴 저장소" | 에피소드에서 승격된 압축된 실수와 선호 |
| COPPA | "아동 프라이버시 법" | 13세 미만 아동의 데이터 수집을 제한하는 미국 법 |

## 더 읽을거리 (Further Reading)

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai): 레퍼런스 소비자 K-12 튜터
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/): 레퍼런스 언어 학습 튜터
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm): 호스팅형 레퍼런스 모델
- [Quizlet Q-Chat](https://quizlet.com): 대안 레퍼런스
- [Synthesis Tutor](https://www.synthesis.com): 스타트업 레퍼런스
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki): 간격 반복 스케줄러
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing): 학습자 모델 고전
- [LiveKit Agents](https://github.com/livekit/agents): 음성 스택
