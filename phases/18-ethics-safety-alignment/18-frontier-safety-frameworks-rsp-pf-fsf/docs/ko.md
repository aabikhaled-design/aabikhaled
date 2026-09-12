# 프런티어 안전 프레임워크(RSP, PF, FSF (Frontier Safety Frameworks) RSP, PF, FSF)

> 세 개의 주요 연구소 프레임워크가 프런티어(frontier) 역량에 대한 2026년 산업 거버넌스를 정의한다. Anthropic Responsible Scaling Policy v3.0(2026년 2월)은 생물안전 등급을 본떠 만든 계층화된 AI 안전 등급(AI Safety Level, ASL-1부터 ASL-5+까지)을 도입하며, CBRN 관련 모델에 대해 2025년 5월 ASL-3을 발동했다. OpenAI Preparedness Framework v2(2025년 4월)는 추적 역량에 대한 다섯 가지 기준을 정의하고 역량 보고서(Capabilities Reports)와 안전장치 보고서(Safeguards Reports)를 분리한다. DeepMind Frontier Safety Framework v3.0(2025년 9월)은 새로운 유해 조작(Harmful Manipulation) CCL을 포함한 핵심 역량 수준(Critical Capability Levels)을 도입한다. 이제 세 프레임워크 모두 동료 연구소가 비교 가능한 안전장치 없이 출시할 경우 연기를 허용하는 경쟁자 조정 조항(competitor-adjustment clauses)을 포함한다. 연구소 간 정렬은 용어가 아니라 구조적으로 유지된다: "Capability Thresholds", "High Capability thresholds", "Critical Capability Levels"는 유사한 구성물을 가리킨다.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- Anthropic의 ASL 티어 구조와 ASL-3을 발동시킨 것을 기술하기.
- OpenAI Preparedness Framework v2의 추적 역량에 대한 다섯 가지 기준을 명명하기.
- DeepMind의 핵심 역량 수준 구조와 유해 조작 CCL을 기술하기.
- 경쟁자 조정 조항과 그것이 경쟁 역학(race dynamics)에 중요한 이유를 설명하기.
- 안전 사례(safety case)를 정의하고 세 기둥 구조(모니터링, 비가독성, 무능력)를 기술하기.

## 문제 (The Problem)

Lesson 7-17은 기만이 가능하고, 이중 용도(dual-use) 역량이 존재하며, 평가에 한계가 있음을 확립한다. 프런티어 역량 모델을 가진 연구소에는 다음을 하는 내부 거버넌스 구조가 필요하다:
- 새로운 안전장치가 필요한 시점에 대한 임계값을 정의한다.
- 확장 전에 요구되는 평가를 정의한다.
- 안전 사례가 어떤 모습인지 기술한다.
- 경쟁 역학 문제를 다룬다(경쟁자가 안전장치 없이 출시하면, 어떻게 할 것인가?).

세 개의 2025-2026년 프레임워크는 최신 기술이다. 불완전하고 진화 중이지만 연구소 간에 충분히 정렬되어 있어, 이제 거버넌스 질문은 프레임워크가 존재하는지가 아니라 적절한지다.

## 개념 (The Concept)

### Anthropic Responsible Scaling Policy v3.0 (2026년 2월)

ASL 구조:
- ASL-1: 프런티어 모델이 아님(프런티어보다 약한 베이스라인에 포섭됨).
- ASL-2: 현재 프런티어 베이스라인; 통상적 안전장치로 배포됨.
- ASL-3: 재앙적 오용의 실질적으로 더 높은 위험; CBRN 관련 역량. 2025년 5월 발동됨.
- ASL-4: AI R&D-2 통과 임계값; 입문 수준 AI 연구를 자동화할 수 있는 모델.
- ASL-5+: 고급 AI R&D; 유효 확장을 극적으로 가속하는 모델.

v3.0의 새로운 점:
- 프런티어 안전 로드맵(편집된 형태로 공개).
- 위험 보고서(분기별, 일부 외부 검토).
- AI R&D가 AI R&D-2와 AI R&D-4로 분리됨.
- AI R&D-4를 일단 넘으면, 잘못 정렬된 목표를 추구하는 모델로부터의 오정렬 위험을 식별하는 긍정적 안전 사례(affirmative safety case)가 요구됨.

### OpenAI Preparedness Framework v2 (2025년 4월 15일)

추적 역량에 대한 다섯 가지 기준:
- **타당함(Plausible).** 합당한 위협 모델이 존재한다.
- **측정 가능함(Measurable).** 경험적 평가가 가능하다.
- **심각함(Severe).** 피해가 크다.
- **순수 신규(Net-new).** 기존 위험을 확장한 것이 아니다.
- **즉각적이거나 회복 불가(Instantaneous-or-irremediable).** 피해가 빠르게 발생하거나 되돌릴 수 없다.

다섯 가지 모두를 충족하는 역량이 추적된다. 그 외는 추적되지 않는다.

기타 PF v2 구조:
- 역량 보고서(모델이 무엇을 할 수 있는지)를 안전장치 보고서(어떤 통제가 존재하는지)와 분리.
- 안전 자문 그룹(Safety Advisory Group) 검토.
- 리더십이 승인하고; 이사회의 안전·보안 위원회(Safety & Security Committee)가 감독.
- "조정 조항(Adjustment clause)": 다른 연구소가 비교 가능한 안전장치 없이 출시하면 OpenAI는 요구사항을 줄일 수 있다.

### DeepMind Frontier Safety Framework v3.0 (2025년 9월)

도메인별 핵심 역량 수준(Critical Capability Levels, CCLs):
- 생물무기 상승(Bioweapon Uplift)
- 사이버 상승(Cyber Uplift)
- ML R&D 가속(ML R&D Acceleration)
- 유해 조작(Harmful Manipulation, v3.0의 새 항목): 고위험 맥락에서 신념/행동을 실질적으로 바꿀 수 있는 모델.

v2.0(2025년 2월)은 기만적 정렬(Deceptive Alignment) 절과 ML R&D CCL에 대한 더 높은 보안 수준을 추가했다.

### 연구소 간 정렬

- Anthropic "Capability Thresholds".
- DeepMind "Critical Capability Levels".
- OpenAI "High Capability thresholds".

업계 표준 용어는 없다. 구조적으로 정렬됨: 발표된 평가 기준을 가진 세 단계의 프런티어 역량. 세 프레임워크 모두 경쟁자 조정 조항(2025년 이후)을 포함한다.

### 안전 사례

안전 사례는 어떤 배포가 최악의 가정 하에서 수용 가능할 만큼 안전하다는 서면 논증이다. 표준 구조는 세 기둥을 겨냥한다:

- **모니터링(Monitoring).** 나쁜 행동이 발생하면 탐지할 수 있는가?
- **비가독성(Illegibility).** 모델이 피해를 유발할 일관된 계획을 실행할 능력이 부족한가?
- **무능력(Incapability).** 모델이 해당 피해를 유발할 역량이 부족한가?

서로 다른 안전 사례는 서로 다른 기둥을 겨냥한다. ASL-3 CBRN 사례의 경우 (언러닝을 통한) 무능력이 주요 목표다. 기만적 정렬의 경우 모니터링과 비가독성이 목표다. 사이버 상승의 경우 셋 다 관련이 있다.

### 경쟁 역학 문제

경쟁자 조정 조항은 논쟁적이다. 비판자들은 그것이 바닥으로의 경쟁(race to the bottom)을 만든다고 주장한다: 세 연구소 모두 경쟁자가 이탈할 때 요구사항을 줄인다면, 균형이 이탈 쪽으로 이동한다. 옹호자들은 대안(일방적 안전장치)이 이탈하는 연구소가 덜 안전 의식적일 경우 더 나쁜 결과를 낳는다고 주장한다.

UK AISI, US CAISI, EU AI Office(Lesson 24)가 외부 거버넌스 대응물이다. 연구소 프레임워크는 자발적이고, 규제 프레임워크는 아직 부상 중이다.

### Phase 18에서의 위치

Lesson 17-18은 기만 및 레드팀 분석 위에 놓인 측정-및-거버넌스 계층이다. Lesson 19-24는 복지, 편향, 프라이버시, 워터마킹, 규제 구조를 다룬다. Lesson 28은 평가를 조작화하는 연구 생태계(MATS, Redwood, Apollo, METR)를 지도화한다.

```figure
al-asl-ladder
```

## 라이브러리로 써보기 (Use It)

이 레슨에는 코드가 없다. 세 개의 1차 자료를 읽어라: RSP v3.0, PF v2, FSF v3.0. 각 연구소의 티어 구조를 다른 연구소들에 매핑하고 각 연구소가 정의하지만 다른 곳은 정의하지 않는 임계값 하나씩을 식별하라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-framework-diff.md`를 만든다. 안전 프레임워크나 릴리스 노트가 주어지면, 그 프레임워크의 임계값 정의, 요구되는 평가, 안전 사례 구조를 RSP v3.0, PF v2, FSF v3.0과 비교하고 연구소 간 격차를 표시한다.

## 연습 문제 (Exercises)

1. RSP v3.0, PF v2, FSF v3.0을 읽어라. 각 연구소의 CBRN 임계값, 각각의 AI R&D 임계값, 각각의 요구되는 배포 전 평가를 표로 정리하라.

2. 경쟁자 조정 조항은 세 프레임워크 모두(2025년 이후)에 있다. 그것에 찬성하는 한 문단을 쓰고; 반대하는 한 문단을 써라. 각 입장이 의존하는 가정을 식별하라.

3. Anthropic의 AI R&D-4 임계값을 넘는 모델에 대한 안전 사례를 설계하라. 세 기둥(모니터링, 비가독성, 무능력) 각각이 요구하는 증거를 명명하라.

4. DeepMind의 FSF v3.0은 유해 조작 CCL을 도입한다. 모델이 이 임계값을 넘었음을 나타낼 세 가지 경험적 측정을 제안하라.

5. METR의 "Common Elements of Frontier AI Safety Policies"(2025)를 읽어라. 가장 강한 연구소 간 수렴 세 가지와 가장 큰 차이 두 가지를 명명하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|------------------------|
| RSP | "Anthropic의 프레임워크" | Responsible Scaling Policy; ASL 티어; v3.0 2026년 2월 |
| PF | "OpenAI의 프레임워크" | Preparedness Framework; 다섯 기준; v2 2025년 4월 |
| FSF | "DeepMind의 프레임워크" | Frontier Safety Framework; CCL; v3.0 2025년 9월 |
| ASL-3 | "생물안전 등급 3 유사물" | CBRN 관련 역량에 대한 Anthropic 티어; 2025년 5월 발동 |
| CCL | "핵심 역량 수준" | DeepMind의 임계값 구성물; 도메인별 |
| 안전 사례 (Safety case) | "공식 논증" | 최악의 U 하에서 배포가 수용 가능할 만큼 안전하다는 서면 논증 |
| 조정 조항 (Adjustment clause) | "경쟁자 이탈 허용" | 경쟁자가 비교 가능한 안전장치 없이 출시할 경우 요구사항을 줄이는 프레임워크 조항 |

## 더 읽을거리 (Further Reading)

- [Anthropic: Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) ASL 티어, 로드맵, AI R&D 분리
- [OpenAI: Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) 다섯 기준, 조정 조항
- [DeepMind: Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) CCL v3.0, 유해 조작
- [METR: Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) 연구소 간 비교
