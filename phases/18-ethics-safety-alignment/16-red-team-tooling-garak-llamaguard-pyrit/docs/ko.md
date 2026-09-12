# 레드팀 도구(Garak, Llama Guard, PyRIT (Red-Team Tooling) Garak, Llama Guard, PyRIT)

> 세 가지 프로덕션 도구가 2026년 레드팀 스택을 구성한다. Llama Guard(Meta): 14개 MLCommons 위험 범주를 파인튜닝(fine-tuning)한 Llama-3.1-8B 분류기; 2025년 Llama Guard 4는 Llama 4 Scout를 가지치기(pruning)한 12B 네이티브 멀티모달 분류기다. Garak(NVIDIA): 환각(hallucination), 데이터 유출, 프롬프트 주입, 유해성(toxicity), 탈옥(jailbreak)에 정적·동적·적응형 프로브(probe)를 갖춘 오픈소스 LLM 취약점 스캐너. PyRIT(Microsoft): Crescendo, TAP, 깊은 익스플로잇을 위한 커스텀 변환기 체인을 갖춘 다중 턴(multi-turn) 레드팀 캠페인. Llama Guard 3은 Meta의 "Llama 3 Herd of Models"(arXiv:2407.21783)에 문서화돼 있고, Llama Guard 3-1B-INT4는 arXiv:2411.17713에, Garak의 프로브 아키텍처는 github.com/NVIDIA/garak에 있다. 이 도구들은 레드팀 연구(Lesson 12-15)와 배포(Lesson 17+) 사이의 2026년 프로덕션 인터페이스다.

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- 안전 스택에서 Llama Guard 3/4의 위치를 기술하기: 입력 분류기, 출력 분류기, 또는 둘 다.
- 14개 MLCommons 위험 범주를 제시하고 직관적이지 않은 것 하나(Code Interpreter Abuse)를 진술하기.
- Garak의 프로브 아키텍처를 기술하기: 프로브, 디텍터(detector), 하니스(harness).
- PyRIT의 다중 턴 캠페인 구조와 그것이 Garak 프로브와 어떻게 합성되는지 기술하기.

## 문제 (The Problem)

Lesson 12-15는 공격 표면을 제시한다. 프로덕션 배포에는 반복 가능하고 확장 가능한 평가가 필요하다. 2026년에는 세 도구가 지배한다: Llama Guard(방어 분류기), Garak(스캐너), PyRIT(캠페인 오케스트레이터). 각각은 레드팀 라이프사이클의 다른 계층을 겨냥한다.

## 개념 (The Concept)

### Llama Guard (Meta)

Llama Guard 3은 MLCommons AILuminate 14개 범주에 대한 입력/출력 분류용으로 파인튜닝된 Llama-3.1-8B 모델이다:
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Specialized advice, privacy, IP, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code-interpreter abuse

8개 언어를 지원한다. 사용법: LLM 앞에 두거나(입력 조정), LLM 뒤에 두거나(출력 조정), 둘 다. 두 용법은 서로 다른 학습 분포를 만든다. Llama Guard 3은 둘 다 처리하는 단일 모델로 출시된다.

Llama Guard 3-1B-INT4(arXiv:2411.17713, 440MB, 모바일 CPU에서 초당 약 30 토큰)는 양자화된 엣지(edge) 변형이다.

Llama Guard 4(2025년 4월)는 12B이며 네이티브 멀티모달이고, Llama 4 Scout를 가지치기한 모델이다. 8B 텍스트와 11B 비전 전임 모델 둘 다를 텍스트 + 이미지를 받아들이는 하나의 분류기로 대체한다.

### Garak (NVIDIA)

오픈소스 취약점 스캐너. 아키텍처:
- **프로브(Probes).** 환각, 데이터 유출, 프롬프트 주입, 유해성, 탈옥에 대한 공격 생성기. 정적(고정 프롬프트), 동적(생성된 프롬프트), 적응형(대상 출력에 반응).
- **디텍터(Detectors).** 예상된 실패 모드(유해함, 유출됨, 탈옥됨)에 대해 출력에 점수를 매긴다.
- **하니스(Harnesses).** 프로브-디텍터 쌍을 관리하고, 캠페인을 실행하고, 보고서를 생성한다.

TrustyAI는 Garak를 Llama-Stack 실드(Prompt-Guard-86M 입력 분류기, Llama-Guard-3-8B 출력 분류기)와 통합하여 종단 간(end-to-end) 실드된 대상 평가를 수행한다. 티어 기반 점수(TBSA)는 이진 합격/불합격을 대체한다. 한 모델이 같은 프로브에서 심각도 티어 3은 통과하고 심각도 티어 5는 실패할 수 있다.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. 다중 턴 레드팀 캠페인. 다음을 중심으로 구축됨:
- **변환기(Converters).** 시드(seed) 프롬프트를 변형한다. 패러프레이즈, 인코딩, 번역, 역할극.
- **오케스트레이터(Orchestrators).** 캠페인을 실행한다: Crescendo(점증), TAP(분기), RedTeaming(커스텀 루프).
- **점수(Scoring).** LLM-as-judge 또는 classifier-as-judge.

PyRIT는 Garak의 더 무거운 사촌이다. Garak는 수천 개의 단일 턴 프로브를 실행하고; PyRIT는 특정 실패 모드를 깨뜨리도록 설계된 깊은 다중 턴 캠페인을 실행한다.

### 스택

Llama Guard를 모델 양쪽에 둔다. 회귀(regression)를 위해 Garak를 야간에 실행한다. 출시 전 캠페인을 위해 PyRIT를 실행한다. 이것이 대부분의 프로덕션 배포를 위한 2026년 기본 구성이다.

### 평가 함정

- **심판 정체성(Judge identity).** 세 도구 모두 LLM 심판을 사용할 수 있다; 심판 보정(calibration)이 보고된 ASR을 좌우한다(Lesson 12). 도구와 함께 심판을 명시하라.
- **프로브 노후화(Probe staleness).** Garak 프로브는 모델이 그에 맞춰 패치됨에 따라 노후화한다. 적응형 프로브(PAIR 형태)는 정적 프로브보다 더 느리게 노후화한다.
- **양성 콘텐츠에 대한 Llama Guard FPR.** 초기 Llama Guard 버전은 정치적·LGBTQ+ 콘텐츠를 과도하게 표시했다; Llama Guard 3/4 보정은 개선되었지만 배포별로 보정되어 있지는 않다.

### Phase 18에서의 위치

Lesson 12-15는 공격 계열이다. Lesson 16은 프로덕션 도구다. Lesson 17(WMDP)은 이중 용도(dual-use) 역량에 대한 평가다. Lesson 18은 이 도구들을 정책 구조로 감싸는 프런티어(frontier) 안전 프레임워크다.

```figure
al-guard-stack
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 Llama Guard 방식 분류기(14개 범주에 대한 키워드 + 의미 특성), 장난감 Garak 하니스(프로브-디텍터 루프), PyRIT 방식 다중 턴 변환기 체인을 만든다. 세 도구를 모의 대상에 대해 실행하고 서로 다른 커버리지 시그니처를 관찰할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-red-team-stack.md`를 만든다. 배포 설명이 주어지면, 세 도구 중 어느 것이 적절한지, 각각에서 무엇을 구성할지, 어떤 회귀 주기로 실행할지를 명명한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 단일 턴 공격 대 다중 턴 공격에서 Llama Guard 방식 분류기의 탐지율을 비교하라.

2. 새로운 Garak 프로브를 구현하라: base64로 인코딩된 유해 요청. Llama Guard 방식 분류기에 의한 탐지를 측정하라.

3. PyRIT 방식 변환기 체인을 "프랑스어로 번역한 다음 패러프레이즈" 변환기로 확장하라. 공격 성공을 다시 측정하라.

4. Llama Guard 3의 위험 범주 목록을 읽어라. 정당한 개발자 콘텐츠에 대해 현실적으로 높은 거짓 양성 비율을 만들 두 범주를 식별하라.

5. Garak와 PyRIT의 설계 원칙을 비교하라. 각각이 올바른 도구가 되는 배포를 논증하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Llama Guard | "그 분류기" | 14개 위험 범주를 가진 파인튜닝된 Llama-3.1-8B/4-12B 안전 분류기 |
| Garak | "그 스캐너" | NVIDIA 오픈소스 취약점 스캐너; 프로브, 디텍터, 하니스 |
| PyRIT | "그 캠페인 도구" | Microsoft 다중 턴 레드팀 오케스트레이터; 변환기, 오케스트레이터, 점수 |
| Prompt-Guard | "그 작은 분류기" | Llama Guard와 짝을 이루는 Meta의 86M 프롬프트 주입 분류기 |
| TBSA | "티어 기반 점수" | 이진 결과를 대체하는 Garak의 티어 기반 합격/불합격 |
| 변환기 체인 (Converter chain) | "패러프레이즈 + 인코딩 + ..." | 다단계 공격을 구축하는 PyRIT 합성 프리미티브 |
| MLCommons 위험 범주 | "14개 분류 체계" | Llama Guard가 겨냥하는 업계 표준 분류 체계 |

## 더 읽을거리 (Further Reading)

- [Meta: Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) 8B 분류기
- [Meta: Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) 양자화 모바일 분류기
- [NVIDIA Garak: GitHub](https://github.com/NVIDIA/garak) 스캐너 저장소와 문서
- [Microsoft PyRIT: GitHub](https://github.com/Azure/PyRIT) 캠페인 툴킷
