# Anthropic 책임 있는 스케일링 정책 v3.0 (Anthropic Responsible Scaling Policy v3.0)

> RSP v3.0은 2026년 2월 24일에 발효되어 2023년 정책을 대체했다. 2단계 완화책: Anthropic이 일방적으로 할 것 대(對) 업계 전반의 권고로 프레이밍된 것(RAND SL-4 보안 표준 포함). 일회성 산출물이 아니라 상설 문서로서 Frontier Safety Roadmap과 Risk Report를 추가한다. 2023년 일시 중지(pause) 약속을 폐기한다. AI R&D-4 임계값(threshold)을 도입한다. 일단 넘어서면 Anthropic은 정렬 오류(misalignment) 위험과 완화책을 식별하는 적극적 입증(affirmative case)을 발표해야 한다. Claude Opus 4.6은 이를 넘지 않는다. Anthropic은 v3.0 발표에서 "이것을 자신 있게 배제하기가 점점 어려워지고 있다"고 진술한다. SaferAI는 2023년 RSP를 2.2로 평가했다. 그들은 v3.0을 1.9로 강등하여 Anthropic을 OpenAI 및 DeepMind와 함께 "약함(weak)" RSP 범주에 넣었다. 정성적 임계값이 2023년 정량적 약속을 대체했다. 일시 중지 조항을 제거한 것이 가장 날카로운 퇴보다.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45분

## 문제 (The Problem)

프런티어 연구소(frontier lab)들은 부분적으로는 기술 문서, 부분적으로는 거버넌스 문서, 부분적으로는 규제 당국을 향한 신호인 스케일링 정책을 발표한다. RSP v3.0은 현재의 Anthropic 문서다. 이를 면밀히 읽는 것이 중요한 이유는 그 준수가 구속력이 있어서가 아니라(그렇지 않다), 그 프레이밍이 연구소가 파국적 위험(catastrophic risk)을 어떻게 개념화하고 트레이드오프(trade-off)를 대중에게 어떻게 전달하는지를 형성하기 때문이다.

v3.0 대 v2.0의 차이가 유용한 단위다. 추가된 것: Frontier Safety Roadmap, Risk Report, AI R&D-4 임계값. 제거된 것: 2023년 일시 중지 약속. 재프레이밍된 것: Anthropic 일방적인 것과 업계 권고 사이로 나뉜 2단계 완화 일정. 외부 검토(SaferAI)가 점수를 2.2(v2)에서 1.9(v3.0)로 강등했다. 이렇게 스케일링 정책은 더 다듬어 보이면서도 덜 엄격해질 수 있다.

## 개념 (The Concept)

### 2단계 완화 일정

- **Anthropic 일방적 조치**: 다른 연구소들이 무엇을 하든 상관없이 Anthropic이 할 것. 임계값 위에서의 학습 중단, 특정 보안 조치, 특정 배포 게이트.
- **업계 전반 권고**: Anthropic이 업계가 집단적으로 해야 한다고 생각하는 것. RAND SL-4 보안 표준을 포함한다. 이것들은 Anthropic 측의 약속이 아니다. 정책 옹호다.

2단계 구조는 v2에 없었다. 그래서 독자는 각 약속이 어느 열에 속하는지 직접 따져 봐야 한다. "업계 전반 권고" 열에 있는 보안 조치는 Anthropic의 약속이 아니다. Anthropic의 희망이다.

### AI R&D-4 임계값

이것은 RSP v3.0이 중요한 다음 임계값으로 명명하는 능력 수준이다. 구체적으로: 경쟁력 있는 비용으로 AI 연구의 상당 부분을 자동화할 수 있는 모델. 일단 Anthropic이 어떤 모델이 이를 넘는다고 믿으면, 계속된 스케일링 전에 정렬 오류 위험과 완화책을 식별하는 적극적 입증을 발표해야 한다.

v3.0 발표에 따르면 Claude Opus 4.6은 이를 넘지 않는다. 문서는 덧붙인다. "이것을 자신 있게 배제하기가 점점 어려워지고 있다." 그 표현이 중요하다. 임계값이 사변적 한계가 아니라 살아 있는 우려가 될 만큼 가깝다는 것을 인정하기 때문이다.

Lesson 6(Automated Alignment Research)과 Lesson 7(Recursive Self-Improvement)이 이 임계값에 직접 공급된다. 연구 품질 기준을 넘는 자동 정렬 연구자(automated alignment researcher)는 AI R&D-4 임계값이 다가오고 있다는 증거다.

### Frontier Safety Roadmap과 Risk Report

v3.0은 두 가지 산출물 유형을 상설 문서로 격상한다.

- **Frontier Safety Roadmap**: 계획된 안전 작업, 능력 기대치, 완화 연구를 기술하는 미래 지향적 문서.
- **Risk Report**: 출시 후 특정 모델에 대한 회고적 문서로, 관찰된 능력과 잔여 위험을 기술한다.

둘 다 공개된다. 둘 다 선언된 주기로 갱신된다. 그 유용성은: 독자가 Anthropic이 Roadmap에서 하겠다고 말한 것이 Risk Report에서 보고하는 것과 어떻게 비교되는지 추적할 수 있다는 점이다.

### 일시 중지 조항 제거

2023년 RSP는 명시적 일시 중지 약속을 포함했다. 모델이 특정 능력 임계값을 넘으면, 완화책이 갖춰질 때까지 학습이 중단된다는 것이다. v3.0은 명시적 일시 중지를 더 부드러운 정식화(적극적 입증을 발표하고, 완화책이 적절하면 진행)로 대체한다. SaferAI와 다른 분석가들은 이를 새 문서에서 가장 강력한 퇴보로 직접 지목했다.

이 변경에 대한 정책적 논거: 2023년의 정량적 임계값이 2026년 시대 능력 벤치마크(benchmark)로는 도달 불가능한 것으로 드러났는데, 벤치마크 자체가 재척도화되었기 때문이다. 반론: 스케일링 정책의 일시 중지 조항은 약속 장치다. 이를 제거하면 정책의 신뢰성이 사라진다.

### SaferAI의 강등

SaferAI는 RSP 형식의 문서를 평가하는 독립 조직이다. 그들의 공개 평가: 2023년 Anthropic RSP는 2.2점을 받았다(4.0이 현존 최고 RSP, 1.0이 명목상인 척도에서). v3.0은 1.9점을 받았다. 이는 Anthropic을 "보통(moderate)"에서 "약함"으로 이동시켜, OpenAI 및 DeepMind와 함께 약함 범주에 합류시켰다.

SaferAI에 따른 강등 요인:
- 정성적 임계값이 정량적 임계값을 대체했다.
- 일시 중지 약속이 제거되었다.
- AI R&D-4 임계값 완화책이 구체적 조치가 아니라 "적극적 입증"으로 기술되었다.
- 검토 메커니즘이 Anthropic의 Safety Advisory Group에 의존하며, 독립적 감독이 제한적이다.

### 이 레슨이 아닌 것

이것은 컴플라이언스 레슨이 아니다. RSP v3.0은 규제가 아니다. 그 무엇도 Anthropic이 이를 따르도록 강제하지 않는다. 이 레슨은 그 문서를 마땅히 받아야 할 구체성과 회의로 읽는 데 있다. 스케일링 정책은 프런티어 연구소가 파국적 위험 태세에 대해 내보내는 주된 공개 신호다. 그것을 잘 읽는 일은 자신의 작업이 프런티어 능력에 의존하는 모든 이에게 실용적 기술이다.

```figure
a5-rsp-ladder
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 RSP 임계값 평가 형태를 반영하는 작은 결정 엔진을 구현한다. 후보 모델과 능력 측정 집합이 주어지면, AI R&D-4 임계값이 넘어섰는지, 요구되는 적극적 입증 섹션들, 그리고 배포가 진행될 수 있는지를 반환한다. 의도적으로 단순하게 만들었다. 요점은 문서의 논리를 명시적으로 드러내는 것이다.

## 산출물 (Ship It)

`outputs/skill-scaling-policy-review.md`는 스케일링 정책(Anthropic, OpenAI, DeepMind, 또는 내부)을 v3.0 참조 기준으로 검토한다. 2단계 구조, 임계값, 일시 중지 약속, 독립적 검토.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 서로 다른 능력 수준의 합성 모델 세 개를 입력하라. 임계값 평가기가 예상대로 동작하고 올바른 적극적 입증 템플릿을 생성하는지 확인하라.

2. RSP v3.0을 전부 읽어라(32쪽). "업계 전반 권고" 단계에 속하는 모든 약속을 식별하라. 그중 어떤 약속이 v2에서는 "Anthropic 일방적"이었겠는가?

3. SaferAI의 RSP 채점 방법론을 읽어라. 그들의 루브릭을 문서에 적용하여 v3.0에 대한 1.9 점수를 재현하라. 어느 루브릭 행이 강등을 가장 크게 견인했는가?

4. 2023년 일시 중지 약속이 제거되었다. 2026년 벤치마크 재척도화 문제를 인정하면서도 정책의 신뢰성을 보존하는 대체 약속을 제안하라.

5. RSP v3.0을 OpenAI Preparedness Framework v2(Lesson 20)와 비교하라. v3.0이 더 강한 영역 하나를 골라라. Preparedness Framework가 더 강한 영역 하나를 골라라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| RSP | "Anthropic의 스케일링 정책" | Responsible Scaling Policy; v3.0은 2026년 2월 24일 발효 |
| AI R&D-4 | "연구 자동화 임계값" | 경쟁력 있는 비용으로 상당한 AI 연구를 자동화하는 능력 |
| 적극적 입증 (Affirmative case) | "안전 정당화" | 위험이 식별되고 완화책이 적절하다는 발표된 논증 |
| Frontier Safety Roadmap | "미래 계획" | 계획된 안전 작업과 기대 능력에 대한 상설 문서 |
| Risk Report | "모델에 대한 회고" | 출시 후 관찰된 능력과 잔여 위험에 대한 상설 문서 |
| 2단계 완화 (Two-tier mitigation) | "일방적 대 업계" | Anthropic 약속 대 업계 권고, 분리됨 |
| 일시 중지 약속 (Pause commitment) | "2023년 조항" | 학습을 중단하겠다는 명시적 약속; v3.0에서 제거됨 |
| SaferAI 평가 (SaferAI rating) | "독립적 RSP 등급" | 제3자 루브릭; v3.0은 1.9점(v2는 2.2점) |

## 더 읽을거리 (Further Reading)

- [Anthropic: Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) 전체 32쪽 정책.
- [Anthropic: RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) v2로부터의 변경 사항 요약.
- [Anthropic: Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) RSP v3.0에서 연결된 상설 문서.
- [Anthropic: Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) 현재 프런티어 모델에 대한 회고.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) AI R&D-4를 측정된 자율성과 연결한다.
