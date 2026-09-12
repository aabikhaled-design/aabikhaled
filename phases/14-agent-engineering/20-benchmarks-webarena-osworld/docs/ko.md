# 벤치마크: WebArena와 OSWorld

> WebArena는 네 개의 자체 호스팅(self-hosted) 앱에 걸쳐 웹 에이전트(web-agent) 능력을 테스트한다. OSWorld는 Ubuntu, Windows, macOS에 걸쳐 데스크톱 에이전트(desktop-agent) 능력을 테스트한다. 출시 당시(2023~2024) 두 벤치마크 모두 최고 수준 에이전트와 인간 사이의 큰 격차를 보여주었다. 그 격차는 좁혀지고 있지만, 실패 양상(failure mode)은 바뀌지 않았다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- WebArena의 네 개 자체 호스팅 앱과 실행 기반(execution-based) 평가가 중요한 이유를 기술하기.
- OSWorld가 접근성 API 대신 실제 OS 스크린샷을 사용하는 이유를 설명하기.
- OSWorld의 두 가지 주된 실패 양상을 거명하기: GUI 그라운딩(GUI grounding)과 운영 지식(operational knowledge).
- OSWorld-G와 OSWorld-Human이 기본 벤치마크 위에 무엇을 더하는지 요약하기.

## 문제 (The Problem)

범용 에이전트는 도구를 호출한다. 그렇다면 쇼핑 결제를 끝내려고 20번 클릭하며 브라우저를 끌고 갈 수 있는가? 키보드와 마우스만으로 리눅스 머신을 설정할 수 있는가? 이것이 WebArena와 OSWorld가 답하는 질문이다.

## 개념 (The Concept)

### WebArena (Zhou et al., ICLR 2024)

- 네 개의 자체 호스팅 웹 앱(쇼핑 사이트, 포럼, GitLab 유사 개발 도구, 비즈니스 CMS)에 걸친 812개의 장기(long-horizon) 과제.
- 더해서 유틸리티: 지도, 계산기, 스크래치패드.
- 평가는 gym API를 통한 실행 기반이다. 주문이 접수되었는가, 이슈가 닫혔는가, CMS 페이지가 갱신되었는가?
- 출시 당시: 최고의 GPT-4 에이전트가 14.41% 성공률을 기록한 반면 인간은 78.24%였다.

자체 호스팅 구도가 중요하다. 대상 앱이 고정(pinned)되어 있고 재현 가능하므로 벤치마크가 불안정하지 않다.

### 확장판 (Extensions)

- **VisualWebArena**: 성공이 이미지 해석에 달려 있는 시각적으로 그라운딩된(visually grounded) 과제(스크린샷을 일급 관찰값으로 사용).
- **TheAgentCompany** (2024년 12월): 터미널 + 코딩을 추가; 실제 원격 근무 환경에 더 가깝다.

### OSWorld (Xie et al., NeurIPS 2024)

- Ubuntu, Windows, macOS에 걸친 369개의 실제 컴퓨터 과제.
- 실제 애플리케이션에 대한 자유 형식 키보드·마우스 제어.
- 관찰값으로 1920×1080 스크린샷.
- 출시 당시: 최고 모델 12.24% 대 인간 72.36%.

### 주된 실패 양상

1. **GUI 그라운딩(GUI grounding).** 픽셀 → 요소 매핑. 모델은 1920×1080에서 UI 요소를 안정적으로 위치 지정하는 데 어려움을 겪는다.
2. **운영 지식(operational knowledge).** 어느 메뉴에 설정이 있는지, 어느 키보드 단축키인지, 어느 환경설정 창인지. 인간이 수년에 걸쳐 쌓는 지식의 꼬리(tail).

### 후속작 (Follow-ups)

- **OSWorld-G**: 564개 샘플의 그라운딩 스위트 + Jedi 학습 세트. 그라운딩을 계획(planning)에서 분해하여 따로 측정할 수 있게 한다.
- **OSWorld-Human**: 수작업으로 선별한 골드(gold) 행동 궤적(trajectory). 최상위 에이전트가 필요한 것보다 1.4~2.7배 많은 스텝을 사용함을 보여준다(궤적 효율성 격차).

### 이것이 중요한 이유

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use(Lesson 21)는 모두 WebArena와 OSWorld가 형성한 워크로드로 학습한다. 벤치마크가 목표라면, 프로덕션 모델은 그에 맞춰 출하한 답이다.

### 벤치마킹이 잘못되는 지점

- **스크린샷 전용 평가.** OSWorld는 스크린샷 기반이다. DOM이나 접근성 API를 사용하는 에이전트를 OSWorld에서 평가하면 그라운딩 도전 과제를 놓친다.
- **궤적 길이 무시.** 성공률만 채점하면 OSWorld-Human이 드러내는 1.4~2.7배의 스텝 비효율을 놓친다.
- **오래된 자체 호스팅 앱.** WebArena의 앱은 특정 버전을 고정한다. 재선별 없이 업데이트하면 비교 가능성이 깨진다.

```figure
ae-agent-human-gap
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 수준의 웹 에이전트 하니스(harness)를 구현한다:

- 최소한의 "쇼핑 앱" 상태 기계(state machine): list_items, add_to_cart, checkout.
- 3개 과제에 대한 골드 궤적.
- 각 과제를 시도하는 스크립트화된 에이전트.
- 실행 기반 평가자(상태 검사)와 궤적 효율성 지표(스텝 대 골드).

실행:

```
python3 code/main.py
```

출력: OSWorld-Human의 방법론을 반영한 과제별 성공률과 궤적 효율성.

## 라이브러리로 써보기 (Use It)

- 지속적 평가를 위해 내부 클러스터에 자체 호스팅한 **WebArena Verified**.
- 데스크톱 에이전트를 위한 VM 플릿(fleet)의 **OSWorld**.
- **컴퓨터 사용 에이전트(computer-use agent)**(Lesson 21)(Claude, OpenAI CUA, Gemini) 모두 이와 같은 워크로드로 학습됨.
- **자기 제품의 흐름**: 상위 20개 과제의 골드 궤적을 포착하라. 매주 그 과제로 에이전트를 돌려라.

## 산출물 (Ship It)

`outputs/skill-web-desktop-harness.md`는 실행 기반 평가와 궤적 효율성 지표를 갖춘 웹/데스크톱 에이전트 하니스를 구축한다.

## 연습 문제 (Exercises)

1. 장난감 하니스를 두 번째 앱(포럼)으로 확장하라. 3개 과제와 골드 궤적을 작성하라.
2. 과제별 궤적 효율성 보고를 추가하라. 이 장난감에서 에이전트는 골드 대비 1배, 2배, 3배인가?
3. "방해(distractor)" 도구(골드 궤적이 결코 사용하지 않는 도구)를 구현하라. 스크립트화된 에이전트가 유혹에 빠지는가?
4. OSWorld-G를 읽어라. 직접 만든 평가에서 그라운딩 실패와 계획 실패를 어떻게 분리하겠는가?
5. WebArena의 앱 README를 읽어라. 고정된 앱 버전 중 하나를 업그레이드하면 무엇이 깨지는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| WebArena | "웹 에이전트 벤치마크" | 4개 자체 호스팅 앱에 걸친 812개 과제; gym 스타일 평가 |
| VisualWebArena | "비주얼 WebArena" | 시각적으로 그라운딩된 WebArena; 스크린샷이 관찰값 |
| OSWorld | "데스크톱 에이전트 벤치마크" | 실제 Ubuntu/Windows/macOS에서 369개 과제 |
| GUI grounding | "픽셀-요소 매핑" | 1920x1080에서 모델이 UI 요소를 위치 지정 |
| Operational knowledge | "OS 노하우" | 어느 메뉴, 어느 단축키, 어느 환경설정 창 |
| OSWorld-G | "그라운딩 스위트" | 그라운딩 전용 샘플 564개 + 학습 세트 |
| OSWorld-Human | "골드 궤적" | 효율성 측정을 위한 수작업 전문가 행동 시퀀스 |
| Trajectory efficiency | "골드 대비 스텝" | 에이전트 스텝 수를 인간 최소치로 나눈 값 |

## 더 읽을거리 (Further Reading)

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854): 4개 앱 웹 벤치마크
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972): 크로스 OS 데스크톱 벤치마크
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use): Claude의 벤치마크 형태의 능력
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/): OSWorld와 WebArena 수치
