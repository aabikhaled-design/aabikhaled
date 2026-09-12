# 데이터 출처와 학습 데이터 거버넌스

> EU AI Act은 2025년 8월까지 GPAI를 위한 기계 판독 가능(machine-readable) 옵트아웃(opt-out) 표준을 요구한다(EU 저작권 지침 TDM 예외를 통해). California AB 2013(2024년 서명): 생성형 AI 학습 데이터 투명성은 개발자가 12개의 의무 필드(field)를 갖춘 데이터셋 요약을 게시할 것을 요구한다. 적법한 이익(legitimate interest)에 대한 2025년 DPA 정렬(alignment): 아일랜드 DPC(2025년 5월 21일)는 EDPB 의견 이후 안전장치를 갖춘, Meta의 1자(first-party) 공개 EU/EEA 성인 콘텐츠에 대한 LLM 학습을 수용한다. 쾰른 고등지방법원(2025년 5월 23일)은 가처분을 기각한다. 함부르크 DPA는 긴급성 절차를 철회한다. 영국 ICO(2025년 9월 23일)는 LinkedIn의 AI 학습 안전장치(투명성, 간소화된 옵트아웃, 연장된 이의제기 기간)에 대해 긍정적인 규제 응답을 내리고 모니터링을 계속한다. 공식 승인(clearance)은 아니다. 브라질 ANPD(2024년 7월 2일)는 정보 투명성 불충분을 이유로 Meta의 처리를 중단시켰다. 예방 조치는 Meta가 준수 계획을 제출한 후 2024년 8월 30일에 해제되었다. 핵심 비가역성(irreversibility) 문제: 쿠키 동의 프레임워크는 실시간이고 가역적인 추적을 위해 설계되었다. 일단 데이터가 모델 가중치(model weights)에 들어가면 외과적 삭제는 불가능하다. 학습된 신경망(neural network)에 대한 실용적인 GDPR 삭제권(right-to-erasure)은 없다. 준수 창은 수집 시점에 있다. 데이터 출처 이니셔티브(Data Provenance Initiative)(dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", 2024년 7월): 대규모 감사가 출판사들이 robots.txt 제한을 추가함에 따라 AI 데이터 공유지(data commons)가 급속히 쇠퇴함을 보인다.

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- 생성형 AI 학습 데이터 투명성을 위한 California AB 2013의 12개 의무 필드를 기술하기.
- 적법한 이익 기반 LLM 학습에 대한 2025년 DPA 입장(아일랜드 DPC, 영국 ICO, 함부르크, 쾰른)을 진술하기.
- 비가역성 문제를 기술하기: 왜 GDPR 삭제권이 학습된 신경망에 대해 실용적 등가물을 갖지 못하는가.
- 데이터 출처 이니셔티브의 "Consent in Crisis" 발견을 진술하기.

## 문제 (The Problem)

학습 데이터 거버넌스는 모든 모델 카드(레슨 26)와 규제 의무(레슨 24)의 상류(upstream)다. 2024-2025년에 규제 지형은 세 가지 원칙으로 통합되었다: 옵트아웃 인프라, 데이터셋별 공개, 공개적으로 이용 가능한 데이터에 대한 적법한 이익 수용. 수집 시점에 준수하지 않는 제공자는 하류(downstream)에서 시정할 수 없다.

## 개념 (The Concept)

### California AB 2013

2024년 서명. 2022년 1월 1일 이후 출시된 시스템에 대해 문서는 2026년 1월 1일까지 게시되어야 한다. 제3111조(a)는 개발자가 학습에 사용된 데이터셋의 고수준 요약을 12개의 법정 항목과 함께 게시할 것을 요구한다:
1. 데이터셋의 출처 또는 소유자.
2. 데이터셋이 AI 시스템의 의도된 목적을 어떻게 증진하는지에 대한 설명.
3. 데이터셋의 데이터 포인트 수(일반적인 범위 허용; 동적 데이터셋에 대한 추정치).
4. 데이터 포인트 유형에 대한 설명(라벨링된 데이터셋의 경우 레이블 유형; 라벨링되지 않은 경우 일반적 특성).
5. 데이터셋이 저작권, 상표, 또는 특허로 보호되는 데이터를 포함하는지, 또는 전적으로 퍼블릭 도메인에 있는지.
6. 데이터셋이 구매되었거나 라이선스되었는지.
7. 데이터셋이 개인정보를 포함하는지(Cal. Civ. Code §1798.140(v)에 따라).
8. 데이터셋이 집계 소비자 정보를 포함하는지(Cal. Civ. Code §1798.140(b)에 따라).
9. 개발자에 의한 정제, 처리, 또는 기타 수정, 의도된 목적과 함께.
10. 데이터가 수집된 기간, 수집이 진행 중이라면 그 고지와 함께.
11. 개발 중 데이터셋이 처음 사용된 날짜.
12. 시스템이 합성 데이터(synthetic data) 생성을 사용하거나 지속적으로 사용하는지.

항목 12(합성 데이터)는 Gebru et al. 2018 데이터시트 대비 새롭다. 항목 7(개인정보)은 프라이버시권리법(CPRA) 의무를 유발한다. 이 법령은 보안/무결성, 항공기 운항, 연방 전용 국가 안보 시스템을 면제한다(제3111조(b)).

### EU AI Act(레슨 24)과 TDM 옵트아웃

EU 저작권 지침의 텍스트·데이터 마이닝(text-and-data-mining) 예외는, 권리자가 옵트아웃하지 않는 한 공개적으로 이용 가능한 콘텐츠 학습을 허용한다. EU AI Act GPAI 실천 강령의 저작권 장은 GPAI 제공자가 기계 판독 가능한 옵트아웃 신호(robots.txt, C2PA "No AI Training" 주장 등)를 존중할 것을 요구한다.

### 적법한 이익에 대한 2025년 DPA 수렴

아일랜드 DPC(2025년 5월 21일): EDPB 의견 이후 안전장치를 갖춰, 1자 공개 EU/EEA 성인 사용자 콘텐츠에 대한 학습이라는 Meta의 계획이 수용됨. 쾰른 고등지방법원(2025년 5월 23일)은 Meta에 대한 가처분을 기각한다: 옵트아웃으로 충분하다. 함부르크 DPA는 EU 전역의 일관성을 위해 긴급성 절차를 철회한다. 영국 ICO(2025년 9월 23일)는 유사한 안전장치와 지속적인 모니터링을 동반한 LinkedIn의 AI 학습 재개에 대해 긍정적인 규제 응답을 내렸다. 공식 승인은 아니다.

수렴하는 원칙: 적법한 이익은 옵트아웃을 갖춘 공개적으로 이용 가능한 1자 콘텐츠에 대한 학습을 정당화할 수 있다. 동의(consent)는 필요하지 않다.

### 브라질 ANPD(2024년 6월)

정보 투명성 불충분을 이유로 AI 학습을 위한 브라질 사용자 데이터에 대한 Meta의 처리를 중단시켰다. EU DPA와 다른 결과: ANPD는 적법한 이익 허용성보다 투명성을 우선시했다.

### 비가역성 문제

쿠키 동의는 실시간이고 가역적인 추적을 위해 설계되었다. 학습 데이터는 다르다: 일단 데이터가 모델 가중치에 들어가면 외과적 삭제는 불가능하다. 밑바닥부터 재학습하는 것이 유일한 완전한 시정책이며, 비용이 감당하기 어렵다.

부분적 시정책:
- **언러닝(Unlearning).** 근사적 제거. MIA(레슨 22)로 측정됨.
- **영향 함수 기반 국소화(Influence function-based localization).** 데이터에 가장 큰 영향을 받은 가중치를 식별하고, 선택적으로 갱신함.
- **파인튜닝 억제(Fine-tune-suppression).** 데이터에서 파생된 출력을 거부하도록 모델을 학습시킴.

어느 것도 문제를 완전히 해결하지 못한다. 준수 창은 수집 시점에 있다.

### 데이터 출처 이니셔티브

dataprovenance.org. Longpre, Mahari, Lee et al. "Consent in Crisis"(2024년 7월): AI 학습 데이터 공유지에 대한 대규모 감사. 발견: 출판사들이 가속하는 속도로 robots.txt 제한을 추가하고 있다. 공개적으로 학습 가능한 공유지가 급속히 축소되고 있다. 2023년 -> 2024년에 상위 학습 출처의 약 25%가 어떤 제한을 추가했다. 함의: 미래의 학습 데이터 가용성은 새로운 획득 패러다임(라이선싱, 합성 생성, 유인된 참여)에 의존한다.

### Phase 18에서 이 레슨의 위치

레슨 26은 모델 수준 문서화다. 레슨 27은 데이터셋 수준 거버넌스다. 둘이 함께 투명성 계층을 정의한다. 레슨 28은 이러한 질문을 다루는 연구 생태계를 대응시킨다.

```figure
an-provenance-oneway
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 수준의 데이터셋에 대해 California AB 2013을 준수하는 12개 필드 데이터셋 요약 골격(scaffold)을 생성한다. 필드를 채우고, 어느 것이 프라이버시 또는 저작권 후속 의무를 유발하는지 관찰할 수 있다.

## 산출물 (Ship It)

이 레슨은 `outputs/skill-provenance-check.md`를 생성한다. 학습에 사용된 데이터셋이 주어지면, AB 2013 12개 필드 커버리지, 옵트아웃 인프라 준수, DPA 정렬, 비가역성 위험 평가를 확인한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 장난감 수준의 데이터셋에 대한 12개 필드 요약을 생성하고, 어느 필드가 부족하게 명세되어 있는지 식별하라.

2. EU 저작권 지침 TDM 옵트아웃은 기계 판독 가능하다. 옵트아웃 신호를 위한 표준 형식을 제안하고, 그것을 robots.txt 및 C2PA "No AI Training"과 비교하라.

3. 데이터 출처 이니셔티브의 "Consent in Crisis"(2024년 7월)를 읽어라. 가장 빠르게 제한되는 콘텐츠 범주 셋을 기술하고, 하나의 경제적 결과를 논하라.

4. 2025년 DPA 정렬은 공개 콘텐츠 학습에 대한 적법한 이익을 수용한다. 적법한 이익이 충분하지 않을 시나리오를 구성하고, 제공자가 대신 필요로 할 법적 근거를 식별하라.

5. AB 2013 필드와 각 데이터셋에 대한 C2PA 서명 출처 체인과 결합되는 학습 데이터 출처 매니페스트(manifest)를 개략적으로 그려라. 하나의 기술적 장벽과 하나의 법적 장벽을 식별하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| AB 2013 | "그 캘리포니아 법" | 생성형 AI 학습 데이터 투명성. 12개 의무 필드 |
| TDM 예외(TDM exception) | "텍스트·데이터 마이닝" | 옵트아웃을 갖춘 EU 저작권 지침 학습 데이터 예외 |
| 적법한 이익(Legitimate interest) | "그 EU 근거" | 공개 콘텐츠 학습을 정당화할 수 있는 GDPR 제6조 근거 |
| 옵트아웃 신호(Opt-out signal) | "기계 판독 가능 학습 금지" | robots.txt, C2PA "No AI Training," TDM.Reservation |
| 비가역성(Irreversibility) | "학습 취소 불가" | 모델 가중치의 데이터는 외과적으로 제거 불가 |
| 언러닝(Unlearning) | "근사적 제거" | 특정 데이터에 대한 모델 의존도를 줄이는 학습 후 개입 |
| Consent in Crisis | "그 DPI 감사" | 가속하는 robots.txt 제한에 대한 2024년 7월 발견 |

## 더 읽을거리 (Further Reading)

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013): 생성형 AI 학습 데이터 투명성 법
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai): 저작권 장
- [Longpre, Mahari, Lee et al.: Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) DPI 감사
- [IAPP: EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) 규제 맥락
