# 규제 프레임워크: EU, 미국, 영국, 한국

> 네 가지 주요 규제 체제가 2026년 AI 거버넌스(governance) 지형을 정의한다. EU AI Act(2024년 8월 1일 발효): 금지된 관행과 AI 리터러시(literacy)는 2025년 2월 2일부터, GPAI 의무는 2025년 8월 2일부터, 완전한 적용 및 제50조 투명성은 2026년 8월 2일부터, 레거시 GPAI 및 내장된 고위험(high-risk) 시스템은 2027년 8월 2일부터. 벌금은 최대 1,500만 EUR 또는 전 세계 매출의 3%. GPAI 실천 강령(GPAI Code of Practice)(2025년 7월 10일): 세 개 장: 투명성(Transparency), 저작권(Copyright), 안전 및 보안(Safety and Security): 12개 약속. 집행은 2026년 8월 시작. 영국 AISI -> AI Security Institute(2025년 2월): 명칭 변경이 더 좁은 범위를 시사한다. 미국 AISI -> CAISI(2025년 6월): NIST 산하 Center for AI Standards and Innovation. 성장 친화적 입장으로의 전환. 한국 AI 기본법(2024년 12월 통과, 2026년 1월 시행): 제12조가 MSIT 산하에 AISI를 설립한다. 외국 AI 기업에 대한 국내 대리인(local representative), 위험 평가(risk assessment), 고영향(high-impact) 및 생성형 AI에 대한 안전 조치를 의무화한다.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75분

## 학습 목표 (Learning Objectives)

- EU AI Act의 위험 등급(prohibited, high-risk, general-purpose, limited-risk)과 2025년 8월 / 2026년 8월 / 2027년 8월 일정을 기술하기.
- GPAI 실천 강령의 세 개 장과 각각이 어떤 제공자를 구속하는지 기술하기.
- 2025년 명칭 변경을 기술하기: 영국 AISI -> AI Security Institute, 미국 AISI -> CAISI. 각 명칭 변경이 정책 방향에 대해 무엇을 함의하는지.
- 한국 AI 기본법의 핵심 조항을 진술하기.

## 문제 (The Problem)

연구소(lab) 프레임워크(레슨 18)는 자발적이다. 규제 프레임워크는 강제적이다. 2024-2026년 기간에는 포괄적인 AI 규제의 첫 물결이 발효되었다. 배포자(deployer)는 기술적 통제를 규제 의무에 대응시켜야 한다. 그 대응은 관할권(jurisdiction)에 따라 다르다.

## 개념 (The Concept)

### EU AI Act

**2024년 8월 1일 발효.** 위험 등급 구조:

- **금지된 관행**(제5조). 사회적 점수화(social scoring), 공공장소에서의 실시간 원격 생체 인식(law-enforcement 예외 포함), 취약 집단에 대한 착취적 조작. 2025년 2월 2일 적용.
- **고위험 시스템**(부속서 III). 고용, 교육, 신용, 법 집행, 사법, 이주. 적합성 평가(conformity assessment), 위험 관리, 로깅(logging), 투명성을 요구한다.
- **범용 AI(General-Purpose AI, GPAI) 모델.** 2025년 8월 2일 적용. 모든 GPAI 제공자에게 의무가 있다. 시스템적 위험(systemic-risk) GPAI(학습 계산량 >1e25 FLOP)에는 추가 의무가 있다.
- **제한적 위험 시스템.** 제50조 하의 투명성 의무(AI 생성 콘텐츠 라벨링). 2026년 8월 2일 적용.

일정:
- 2025년 2월 2일: 금지된 관행 + AI 리터러시.
- 2025년 8월 2일: GPAI + 거버넌스.
- 2026년 8월 2일: 완전한 적용 + 제50조 투명성 + 최대 1,500만 EUR / 전 세계 매출 3% 벌금.
- 2027년 8월 2일: 레거시 GPAI + 내장된 고위험.

위원회(Commission)는 2025년 말에 고위험 일정을 16개월로 조정할 것을 제안했다.

### GPAI 실천 강령

2025년 7월 10일 발표. 세 개 장:

- **투명성.** 모든 GPAI 제공자.
- **저작권.** 모든 GPAI 제공자.
- **안전 및 보안.** 시스템적 위험 GPAI 제공자(추정 5-15개 기업).

총 12개 약속. AI Office가 의장을 맡는 서명자 태스크포스(Signatory Taskforce)가 이행을 관리한다. 집행은 2026년 8월 2일 시작된다. 그때까지는 선의의(good-faith) 준수가 인정된다.

### 제50조를 위한 투명성 강령

첫 초안 2025년 12월 17일. 두 번째 초안 2026년 3월. 최종본 2026년 6월. 딥페이크(deepfake)를 포함한 AI 생성 콘텐츠 라벨링을 다룬다. 레슨 23의 워터마킹 기술을 요구하는 규제 계층이다.

### 영국 AI Security Institute(2025년 2월)

AI Safety Institute에서 명칭을 변경했다. 이 명칭 변경은 범위를 좁힌다: 알고리즘 편향과 표현의 자유 관점을 제외하고, 프런티어(frontier) 역량 보안에 집중한다. Inspect 평가 도구를 오픈소스화했다(2024년 5월). 통제(control) 안전 사례에 대해 Redwood(레슨 10)와 협력한다.

### 미국 CAISI(2025년 6월)

트럼프 행정부가 NIST의 AI Safety Institute를 Center for AI Standards and Innovation으로 전환한다. 밴스(Vance) 부통령의 Paris AI Action Summit 발언에 따르면 "성장 친화적 AI 정책"으로 전환한다. 배포 전(pre-deployment) 평가 강조를 줄이고, 표준과 혁신 지원을 강조한다. EU AI Act의 규제 입장에 대한 국내적 균형추다.

### 한국 AI 기본법

2024년 12월 통과. 2025년 1월 제정. 2026년 1월 시행. 19개의 개별 AI 법안을 통합한다.

제12조는 과학기술정보통신부(MSIT) 산하에 AISI를 설립한다. 의무화하는 것:
- 한국에서 운영하는 외국 AI 기업에 대한 국내 대리인.
- "고영향" AI 시스템에 대한 위험 평가.
- 생성형 AI 및 고영향 AI에 대한 안전 조치.

포괄적 수평(horizontal) AI 규제를 가진 최초의 아시아 관할권이다.

### 관할권 간 동역학

- EU: 엄격하고, 위험 등급화되어 있으며, 무거운 벌금. 프라이버시 인접 규제의 벤치마크(benchmark).
- 미국: 혁신 우호적, 분산적, 주(예: California AB 2013: 레슨 27)가 연방 공백을 채운다.
- 영국: 좁은 보안 초점, 강력한 평가 인프라.
- 한국: MSIT 주도, 외국 제공자 중심.

경쟁하는 규제 철학들. 다수 관할권의 배포자는 가장 엄격한 규제를 준수해야 하며, 2026년에 그것은 일반적으로 EU AI Act이다.

### Phase 18에서 이 레슨의 위치

레슨 18은 연구소 자발적 거버넌스다. 레슨 24는 규제적이다. 레슨 25는 AI 시스템에 대한 새로 떠오르는 CVE 부류다. 레슨 26-27은 문서화(카드)와 학습 데이터 거버넌스를 다룬다.

```figure
an-eu-act-timeline
```

## 라이브러리로 써보기 (Use It)

코드 없음. EU AI Act 원문 자료를 읽어라: 규정 텍스트, GPAI 실천 강령, 영국 AISI Inspect 프레임워크. 각 관할권마다 적용 가능한 의무에 배포를 대응시켜라.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-regulatory-map.md`를 생성한다. 배포 설명이 주어지면, 적용 가능한 관할권, 각 관할권에서의 등급 분류, 관할권별 의무, 그리고 기한 구조를 대응시킨다.

## 연습 문제 (Exercises)

1. EU AI Act(규정 2024/1689)와 GPAI 실천 강령(2025년 7월 10일)을 읽어라. 모든 GPAI 제공자에게 적용되는 의무 셋과 시스템적 위험 GPAI에만 적용되는 의무 셋을 식별하라.

2. 어떤 배포를 미국 기업이 만들고, EU 인프라에서 실행하며, 한국 사용자에게 서비스한다고 하자. 세 관할권 중 어느 규칙이 적용되며, 각 실질적 문제에서 어느 규칙이 구속하는가?

3. 영국 AI Security Institute의 명칭 변경은 범위를 좁힌다. 더 좁은 관점에 대해 찬성과 반대를 논하라. 각 입장이 의존하는 정책 가정을 식별하라.

4. CAISI의 "성장 친화적" 관점은 2022-2024년 AI 안전 연구소 모델로부터의 이탈이다. 이 관점에서 따라 나올 측정 가능한 정책 전환 둘을 식별하라.

5. 한국 AI 기본법은 외국 제공자에게 국내 대리인을 요구한다. 한국 사용자에게 서비스하는 베이 에어리어(Bay Area) 기업에 대한 운영적 함의를 기술하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| EU AI Act | "그 규정" | 위험 등급 기반 수평 AI 규제. 2024년 8월 발효 |
| GPAI | "범용 AI" | 대규모 파운데이션 모델. 시스템적 위험 하위 집합에 추가 의무 |
| 제50조(Article 50) | "투명성 의무" | AI 생성 콘텐츠 라벨링. 2026년 8월 적용 |
| 영국 AISI(UK AISI) | "AI Security Institute" | 2025년 2월 명칭 변경. 더 좁은 프런티어 보안 초점 |
| CAISI | "미국 AI 표준 센터" | 2025년 6월 AI Safety Institute에서 명칭 변경. 성장 친화적 입장 |
| 한국 AI 기본법(Korean AI Framework Act) | "MSIT 수평 규제" | 최초의 아시아 포괄 AI 법. 2026년 1월 시행 |
| 시스템적 위험 GPAI(Systemic-risk GPAI) | "1e25 FLOP 임계값" | 추가 의무 등급. 추정 5-15개 기업 구속 |

## 더 읽을거리 (Further Reading)

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai): 규정과 일정
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice): 세 개 장 강령
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute): 공식 페이지
- [CSET: South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) 한국 프레임워크 분석
