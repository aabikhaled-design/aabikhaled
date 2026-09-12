# 모델, 시스템, 데이터셋 카드

> 세 가지 문서화 형식이 AI 투명성을 구조화한다. 모델 카드(Model Cards)(Mitchell et al. 2019): 모델을 위한 영양 표시(nutrition label): 학습 데이터, 정량적으로 분해된(disaggregated) 분석, 윤리적 고려 사항, 주의 사항. Hugging Face 모델 카드의 0.3%만이 윤리적 고려 사항을 문서화한다(Oreamuno et al. 2023). 데이터셋을 위한 데이터시트(Datasheets for Datasets)(Gebru et al. 2018, CACM): 동기, 구성, 수집 과정, 라벨링, 배포, 유지보수. 전자 부품 데이터시트 비유. 데이터 카드(Data Cards)(Pushkarna et al., Google 2022): 다양한 독자를 위한 경계 객체(boundary object)로서의 모듈식 계층적 상세도(망원경적/잠망경적/현미경적). 2024-2025년 발전: LLM을 통한 자동 생성(CardGen, Liu et al. 2024). 모델 카드의 상세도는 HF에서 최대 29%의 다운로드 증가와 상관관계를 가진다(Liang et al. 2024). 검증 가능한 증명(verifiable attestation)(Laminator, Duddu et al. 2024). 탄소/물에 대한 지속가능성 보고 추가(Jouneaux et al. 2025년 7월). EU/ISO 규제 카드 출현. 시스템 카드(System Cards)(Sidhpurwala 2024; Meta 시스템 수준 투명성; "Blueprints of Trust" arXiv:2509.20394): 보안 역량, 프롬프트 인젝션(prompt-injection) 보호, 데이터 유출(data-exfiltration) 탐지, 인간 가치와의 정렬(alignment)을 포괄하는 종단 간(end-to-end) AI 시스템 문서화.

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 원본 Mitchell et al. 2019 모델 카드와 Gebru et al. 2018 데이터시트를 기술하기.
- 데이터 카드의 망원경적/잠망경적/현미경적 계층화를 기술하기.
- 시스템 카드와 그 종단 간 커버리지를 기술하기.
- 2024-2025년 발전 셋(자동 생성, 검증 가능한 증명, 지속가능성 보고)을 진술하기.

## 문제 (The Problem)

규제 프레임워크(레슨 24)와 연구소(lab) 안전 정책(레슨 18)은 둘 다 문서화를 요구한다. 문서화 형식은 모델 특정적(model-specific)(모델 카드)에서 데이터셋 특정적(데이터시트)으로, 다시 시스템 특정적(시스템 카드)으로 진화했다. 각각은 서로 다른 투명성 범위를 다룬다. 2024-2025년의 자동화 및 검증 가능 증명 작업은 오랫동안 지속된 채택(adoption) 문제를 다룬다.

## 개념 (The Concept)

### 모델 카드(Mitchell et al. 2019)

섹션:
- 모델 세부 사항.
- 의도된 사용.
- 요인(평가와 관련된 인구통계학적 또는 환경적 요인).
- 척도.
- 평가 데이터.
- 학습 데이터.
- 정량적 분석(요인별로 분해됨).
- 윤리적 고려 사항.
- 주의 사항과 권고.

채택 문제: Oreamuno et al. 2023의 Hugging Face 모델 카드 감사는 0.3%만이 윤리적 고려 사항을 문서화함을 발견했다.

### 데이터셋을 위한 데이터시트(Gebru et al. 2018)

전자 부품 데이터시트 비유. 섹션:
- 동기(데이터셋이 왜 만들어졌는가).
- 구성(그 안에 무엇이 있는가).
- 수집 과정(어떻게 모아졌는가).
- 라벨링(해당되는 경우).
- 용도(의도된 것, 금지된 것, 위험).
- 배포.
- 유지보수.

CACM 2021에 발표되었다. 데이터시트는 상류(upstream) 문서다. 모델 카드는 데이터시트가 정확하다는 데 의존한다.

### 데이터 카드(Pushkarna et al., Google 2022)

모듈식 계층적 상세도. 세 가지 확대 수준:
- **망원경적(Telescopic).** 비전문가를 위한 고수준 요약.
- **잠망경적(Periscopic).** ML 실무자를 위한 중간 수준 개요.
- **현미경적(Microscopic).** 감사자를 위한 상세한 특성 수준 문서화.

경계 객체 관점: 서로 다른 독자가 동일한 문서에서 서로 다른 정보를 추출한다.

### 시스템 카드

범위: 모델 + 안전 스택(safety stack) + 배포 맥락을 포함하는 종단 간 AI 시스템. 일반적으로 포함되는 섹션:
- 보안 역량.
- 프롬프트 인젝션 보호.
- 데이터 유출 탐지.
- 명시된 인간 가치와의 정렬.
- 사고 대응(incident response).

Sidhpurwala 2024와 Meta의 시스템 수준 투명성 작업. "Blueprints of Trust"(arXiv:2509.20394)는 시스템 카드를 모델 카드에 대한 배포 계층 보완물로 형식화한다.

### 2024-2025년 발전

- **CardGen(Liu et al. 2024).** LLM을 통한 자동 모델 카드 생성. 표준화된 Mitchell 2019 필드에서 많은 사람이 작성한 카드보다 높은 객관성을 보고한다.
- **다운로드 상관관계(Liang et al. 2024).** 상세한 모델 카드는 HF에서 최대 29% 높은 다운로드율과 상관관계를 가진다. 채택 압력은 이제 준수 주도가 아니라 시장 주도다.
- **Laminator(Duddu et al. 2024).** 하드웨어 TEE / 암호학적 서명을 통한 검증 가능한 증명: 모델 카드가 단순한 주장이 아니라 주장의 증명(proof-of-claim)을 담게 한다.
- **지속가능성(Jouneaux et al. 2025년 7월).** 탄소, 물, 계산 에너지 발자국에 대한 추가. 출현하는 ISO 표준.
- **규제 카드.** EU AI Act(레슨 24) GPAI 실천 강령의 투명성 장은 모델 카드를 준수 산출물로 요구한다.

### Phase 18에서 이 레슨의 위치

레슨 24-25는 규제 및 CVE 계층이다. 레슨 26은 문서화 계층이다. 레슨 27은 데이터시트의 상류인 학습 데이터 거버넌스(governance)다. 레슨 28은 카드에서 참조되는 평가를 생산하는 연구 생태계다.

```figure
an-card-scopes
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 수준의 배포를 위한 최소한의 모델 카드, 데이터시트, 시스템 카드를 생성한다. 각각은 표준 섹션 구조를 따른다. 형식을 살펴보고 세 가지 범위를 비교할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-card-audit.md`를 생성한다. 모델 카드, 데이터시트, 또는 시스템 카드가 주어지면, 섹션 커버리지, 수치적 분해, 그리고 검증 가능한 증명이 존재하는지를 감사한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 생성된 카드를 살펴보라. 약한(자리표시자뿐인) 섹션을 식별하고, 어떤 증거가 그것들을 강화할지 명시하라.

2. 두 인구통계학적 집단(레슨 20)에 걸친 정량적 분해 분석으로 모델 카드를 확장하라.

3. 0.3% 채택률에 관한 Oreamuno et al. 2023을 읽어라. 윤리적 고려 사항 채택을 늘릴 모델 카드 명세에 대한 구조적 변경 하나를 제안하라.

4. Laminator(Duddu et al. 2024)는 검증 가능한 증명에 TEE를 사용한다. 평가 결과의 암호학적 증명을 담는 모델 카드 필드를 설계하고, 검증자의 역할을 기술하라.

5. 자신의 과거 프로젝트 하나나 가상의 배포에 대한 시스템 카드(모델 카드가 아니라 시스템 카드)를 작성하라. 제3자 감사자에게 가장 가치 있는 섹션을 식별하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 모델 카드(Model Card) | "Mitchell 카드" | ML 모델을 위한 Mitchell et al. 2019 표준 문서 |
| 데이터시트(Datasheet) | "Gebru 데이터시트" | 데이터셋을 위한 Gebru et al. 2018 표준 문서 |
| 데이터 카드(Data Card) | "Pushkarna 카드" | Google 2022 모듈식 계층적 데이터 문서 |
| 시스템 카드(System Card) | "배포 카드" | 안전 스택을 포함한 종단 간 AI 시스템 문서 |
| 경계 객체(Boundary object) | "서로 다른 독자, 하나의 문서" | 데이터 카드 관점: 동일한 문서가 다양한 청중을 섬김 |
| 검증 가능한 증명(Verifiable attestation) | "Laminator 증명" | 문서 주장에 첨부된 암호학적 또는 TEE 증명 |
| 지속가능성 필드(Sustainability field) | "탄소 / 물 발자국" | 환경 회계를 위해 출현하는 2025년 추가 |

## 더 읽을거리 (Further Reading)

- [Mitchell et al.: Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) 표준적인 모델 카드
- [Gebru et al.: Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) 데이터시트 논문
- [Pushkarna et al.: Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) 계층적 데이터 문서
- [Sidhpurwala et al.: Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) 시스템 카드 형식화
