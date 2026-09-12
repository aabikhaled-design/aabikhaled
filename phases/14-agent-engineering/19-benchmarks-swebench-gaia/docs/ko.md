# 벤치마크: SWE-bench, GAIA, AgentBench

> 세 가지 벤치마크가 2026년의 에이전트(agent) 평가를 떠받친다. SWE-bench는 코드 패치(patch)를 테스트한다. GAIA는 범용 도구 사용(tool use)을 테스트한다. AgentBench는 다중 환경 추론을 테스트한다. 이들의 구성, 오염(contamination) 이력, 그리고 이들이 측정하지 못하는 것을 알아야 한다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- SWE-bench의 테스트 하니스(test harness, FAIL_TO_PASS)를 거명하고, 왜 이것이 단위 테스트(unit test)를 게이트로 삼는지 설명하기.
- SWE-bench Verified(OpenAI, 500개 과제)가 존재하는 이유와 그것이 무엇을 제거하는지 설명하기.
- GAIA의 설계를 기술하기: 인간에게는 쉽고 AI에게는 어렵다; 세 가지 난이도 단계.
- AgentBench의 여덟 가지 환경과 오픈소스 LLM의 주된 걸림돌을 거명하기.
- SWE-bench+의 오염 발견 결과와 그 함의를 요약하기.

## 문제 (The Problem)

리더보드(leaderboard)는 하나의 벤치마크에서 어떤 모델(model)이 이기는지를 알려준다. 그러나 다음은 알려주지 않는다:

- 벤치마크가 오염되었는지(학습 데이터에 정답이 들어 있거나, 테스트가 누출되었는지) 여부.
- 벤치마크가 정작 중요한 것을 측정하는지(코드 대 브라우징 대 범용) 여부.
- 평가자(evaluator)가 견고한지(AST 매칭, 상태 검사, 인간 리뷰) 여부.

숫자를 인용하기 전에, 세 가지 기준점이 되는 벤치마크와 그들의 실패 양상(failure mode)을 알아야 한다.

## 개념 (The Concept)

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 인기 있는 12개 Python 저장소에서 가져온 실제 GitHub 이슈 2,294개.
- 에이전트가 받는 것: 수정 전(pre-fix) 커밋 시점의 코드베이스 + 자연어 이슈 설명.
- 에이전트가 만드는 것: 패치 하나.
- 평가자: 패치를 적용하고, 저장소의 테스트 스위트(test suite)를 실행한다. 패치는 PASS_TO_PASS 테스트를 깨뜨리지 않으면서 FAIL_TO_PASS 테스트(이전에는 실패했으나 이제는 통과해야 하는 테스트)를 뒤집어야 한다.

SWE-agent(Yang et al., 2024)는 출시 시점에 12.5%를 기록했는데, 에이전트-컴퓨터 인터페이스(모델이 이해하는 파일 편집기 명령, 검색 문법)를 강조한 결과였다.

### SWE-bench Verified

OpenAI, 2024년 8월. 인간이 선별한 500개 과제 부분집합. 모호한 이슈, 신뢰할 수 없는 테스트, 수정 방향이 불분명한 과제를 제거한다. "당신의 에이전트가 실제 패치를 출하하는가?"에 대한 주된 벤치마크다.

### 오염 (Contamination)

- SWE-bench 이슈의 94% 이상이 대부분의 모델 컷오프(cutoff) 이전에 만들어졌다.
- **SWE-bench+**는 성공한 패치의 32.67%가 이슈 텍스트에 정답을 누출했고(모델이 설명에서 수정 내용을 봄), 31.08%가 약한 테스트 커버리지 때문에 의심스럽다는 것을 발견했다.
- Verified는 더 깨끗하지만 오염이 전혀 없는 것은 아니다.

실용적 함의: SWE-bench에서 50%를 기록하는 모델이 SWE-bench+에서는 35%를 기록할 수 있다. SWE-bench 성능을 주장한다면 항상 두 가지를 모두 보고해야 한다.

### GAIA (Mialon et al., 2023년 11월)

- 466개 질문; huggingface.co/gaia-benchmark의 비공개 리더보드를 위해 300개를 남겨둠.
- 설계 철학: "인간에게는 개념적으로 단순하지만(92%) AI에게는 어렵다(플러그인을 갖춘 GPT-4: 15%)."
- 추론, 멀티모달리티(multi-modality), 웹, 도구 사용을 테스트한다.
- 세 가지 난이도 단계; 레벨 3은 여러 모달리티를 가로지르는 긴 도구 체인(tool chain)을 요구한다.

GAIA는 "범용 능력(generalist capability)"을 측정하기 위해 실행하는 것이다. 코드 특화 벤치마크와 혼동하지 말 것.

### AgentBench (Liu et al., ICLR 2024)

- 코드(Bash, DB, KG), 게임(Alfworld, LTP), 웹(WebShop, Mind2Web), 개방형 생성에 걸친 8개 환경.
- 멀티턴(multi-turn), 분할(split)당 약 4천~1만 3천 턴.
- 주된 발견: 장기 추론, 의사결정, 지시 따르기가 오픈소스(OSS) LLM이 상업용을 따라잡는 데 걸림돌이다.

### 이것들이 측정하지 못하는 것

- 실제 운영 비용(토큰, 벽시계 시간).
- 적대적 조건에서의 안전 행동.
- 자신의 도메인에서 내는 성능(직접 만든 평가를 쓰라, Lesson 30).
- 꼬리 실패(tail failure)(벤치마크는 평균을 낸다; 프로덕션 운영자는 최악의 1%를 신경 쓴다).

### 벤치마킹이 잘못되는 지점

- **단일 숫자 집착.** SWE-bench 50%는 P50/P75/P95 비용 + 스텝 분포보다 알려주는 것이 적다.
- **오염된 주장.** Verified나 SWE-bench+를 언급하지 않고 SWE-bench를 보고하는 것은 오해를 부른다.
- **개발 목표로서의 벤치마크.** 벤치마크를 위해 최적화하면 프로덕션 유용성에서 멀어진다.

```figure
ae-swebench-gate
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 수준의 SWE-bench 유사 하니스를 구현한다:

- 합성 버그 수정 과제(3개 과제).
- 패치를 제안하는 스크립트화된 "에이전트".
- FAIL_TO_PASS(버그가 이제 고쳐짐)와 PASS_TO_PASS(아무것도 깨지지 않음)를 검사하는 테스트 러너.
- 질문 분해 깊이에 기반한 GAIA 스타일 난이도 분류기.

실행:

```
python3 code/main.py
```

출력은 과제별 + 난이도별 해결률(resolution rate)을 보여주고, 평가자 규칙을 구체화한다.

## 라이브러리로 써보기 (Use It)

- 코드 에이전트에는 **SWE-bench Verified**. 항상 Verified 점수를 보고하라.
- 범용 에이전트에는 **GAIA**. 비공개 리더보드 분할을 사용하라.
- 다중 환경 비교에는 **AgentBench**.
- 자기 제품의 실제 형태에는 **맞춤 평가(custom eval)**(Lesson 30).

## 산출물 (Ship It)

`outputs/skill-benchmark-harness.md`는 임의의 코드베이스-과제 쌍에 대해 FAIL_TO_PASS / PASS_TO_PASS 게이팅을 갖춘 SWE-bench 스타일 하니스를 구축한다.

## 연습 문제 (Exercises)

1. 장난감 하니스를 실제 저장소에서 실행하도록 이식하라(자신의 저장소 중 하나를 골라라). 알려진 버그에 3개의 FAIL_TO_PASS 테스트를 작성하라.
2. 스텝 수(step-count) 지표를 추가하라. 앞서 만든 3개 과제에서 해결당 에이전트 스텝은 몇 개인가?
3. SWE-bench+ 논문을 읽어라. 정답 누출 검사(이슈 텍스트를 diff와 패턴 매칭)를 구현하라.
4. 공개 분할에서 GAIA 질문 하나를 내려받아라. GPT-4급 에이전트가 무엇을 할지 추적하라. 어떤 도구가 필요한가?
5. AgentBench의 환경별 분석을 읽어라. 어떤 환경이 자기 제품의 표면을 반영하는가? 거기서 "SOTA"는 어떤 모습인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| SWE-bench | "코드 에이전트 벤치마크" | GitHub 이슈 2,294개; 패치가 FAIL_TO_PASS 테스트를 뒤집어야 함 |
| SWE-bench Verified | "깨끗한 SWE-bench" | 인간이 선별한 500개 과제, OpenAI |
| FAIL_TO_PASS | "수정 게이트" | 이전에 실패했고 패치 후 통과해야 하는 테스트 |
| PASS_TO_PASS | "무회귀 게이트" | 통과하고 있었고 여전히 통과해야 하는 테스트 |
| GAIA | "범용 벤치마크" | 인간에게 쉽고 AI에게 어려운 다중 도구 질문 466개 |
| AgentBench | "다중 환경 벤치마크" | 8개 환경; 장기 멀티턴 |
| Contamination | "학습셋 누출" | 모델 학습에 들어 있는 벤치마크 과제 |
| SWE-bench+ | "오염 감사" | 성공한 SWE-bench 패치에서 32.67% 정답 누출 발견 |

## 더 읽을거리 (Further Reading)

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770): 원본 벤치마크
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/): 선별된 부분집합
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983): 범용 벤치마크
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688): 다중 환경 스위트
