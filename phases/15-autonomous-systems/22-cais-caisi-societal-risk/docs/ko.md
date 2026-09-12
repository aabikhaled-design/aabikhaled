# CAIS, CAISI, 그리고 사회적 규모 위험 (CAIS, CAISI, and Societal-Scale Risk)

> Center for AI Safety(CAIS, 샌프란시스코, 2022년 Hendrycks와 Zhang이 설립)는 네 가지 위험 프레임워크: 악의적 사용(malicious use), AI 경쟁(AI races), 조직적 위험(organizational risks), 악성 AI(rogue AIs)와, 수백 명의 교수와 기업 지도자가 서명한 2023년 5월 멸종 위험 성명을 발표한다. CAIS의 2026년 발표물: 프런티어 모델(frontier model) 평가를 위한 AI Dashboard, Remote Labor Index(Scale AI와 공동), Superintelligence Strategy Paper, AI Frontiers 뉴스레터. 별개의 실체: NIST Center for AI Standards and Innovation(CAISI): 미국 정부 대상 자발적 협약과, 사이버·생물·화학 무기 위험에 초점을 맞춘 비분류(unclassified) 능력 평가. CAIS는 조직적 위험을 네 가지 최상위 위험 중 하나로 표시한다. 안전 문화, 엄격한 감사, 다층 방어, 정보 보안이 기초적이지만 일상적으로 배포 속도와 트레이드오프(trade-off)된다. California SB-53는 서명되면 미국 최초의 주(州) 수준 파국적 위험(catastrophic-risk) 규제가 될 것이다.

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45분

## 문제 (The Problem)

Lessons 19와 20은 연구소 내부 스케일링 정책을 다뤘다. Lesson 21은 독립적 능력 평가를 다뤘다. 이 레슨은 세 번째 관점을 다룬다. 파국적 AI 위험을 둘러싼 공적 논의와 규제 베이스라인(baseline)을 형성하는 시민 사회 및 정부 조직이다.

두 개의 별개 실체가 중요하다. CAIS는 AI 위험을 사고하기 위한 프레임워크를 발표하고 공개 성명을 조율하는 비영리 연구 조직이다. CAISI는 NIST 내의 미국 정부 센터로, 연구소들과의 자발적 협약과 비분류 능력 평가를 운영한다. 이름은 운(韻)이 맞지만, 사명은 겹치지 않는다. 실무자는 둘 다 알아야 한다.

실무적 내용: CAIS의 네 가지 위험 프레임워크는 문헌에서 가장 널리 인용되는 사회적 규모 위험 분류 체계(taxonomy)다. 안전 문화와 조직적 위험이 그 넷 중 하나이며, 이것이 실무자의 통제 하에 가장 직접적으로 놓인 것이다. SB-53(캘리포니아)는 서명되면 미국 최초의 주 수준 파국적 위험 규제가 될 것이다. 이 법안의 프레이밍이 중요한 이유는, 미국 기술 정책에서 주 수준 규제가 역사적으로 연방 조치를 이끌어왔기 때문이다.

## 개념 (The Concept)

### CAIS — Center for AI Safety

- 설립: 2022년 샌프란시스코, Dan Hendrycks와 동료들에 의해("Zhang"이라는 이름은 현재 공동 설립자가 아니라 초기 협력자를 가리킨다. 현재 리더십은 CAIS 웹사이트 참조).
- 지위: 501(c)(3) 비영리.
- 주목할 2023년 산출물: 수백 명의 연구자와 CEO가 공동 서명한 멸종 위험 성명. 진술: "AI로 인한 멸종 위험을 완화하는 것은 팬데믹과 핵전쟁 같은 다른 사회적 규모 위험과 함께 전 지구적 우선순위가 되어야 한다."
- 2026년 산출물: 프런티어 모델 평가를 위한 AI Dashboard, Remote Labor Index(Scale AI와 공동), Superintelligence Strategy Paper, AI Frontiers 뉴스레터.

### 네 가지 위험 프레임워크

CAIS의 프레임워크는 파국적 AI 위험을 네 가지 최상위 범주로 묶는다.

1. **악의적 사용(Malicious use)**: 악의적 행위자가 AI를 사용해 해를 끼친다(생물무기 합성, 허위정보, 사이버 공격).
2. **AI 경쟁(AI races)**: 연구소, 기업, 또는 국가 간의 경쟁 압력이 배포를 안전한 지점 너머로 밀어붙인다.
3. **조직적 위험(Organizational risks)**: 내부 연구소 역학(안전 문화 실패, 불충분한 감사, 자원이 부족한 보안)이 나쁜 배포를 낳는다.
4. **악성 AI(Rogue AIs)**: 충분히 유능한 AI가 인간 복지와 충돌하는 목표를 추구한다.

이것이 유일한 분류 체계는 아니다. 가장 많이 인용되는 것이다. 범주들은 상호 배타적이지 않다. 경쟁 속에서 속도를 위해 감사를 거래한 조직이 만들어낸 악성 AI는 네 가지 모두에 해당한다.

### 조직적 위험이 존재하는 곳

네 범주 중 조직적 위험이 실무자에게 가장 실행 가능하다. 연구소의 안전 문화, 감사 엄격성, 방어 계층화, 정보 보안이 곧 모델이 Lessons 10–18의 통제를 실제로 갖춘 채 출고되는지, 아니면 그 통제가 아무도 검증하지 않은 체크리스트 항목인지를 결정한다.

구체적인 조직적 위험 레버:

- **안전 문화(Safety culture)**: 팀원들이 경력 비용 없이 우려를 에스컬레이션할 수 있다고 느끼는가? CAIS 설문은 이것이 다른 레버들의 강력한 예측 변수임을 발견한다.
- **엄격한 감사(Rigorous audits)**: 외부와 내부. 내부 전용 감사는 낙관적 보고서를 낳는다.
- **다층 방어(Multi-layered defenses)**: 어느 단일 계층(layer)도 충분하지 않다(Phase 15의 일관된 주제).
- **정보 보안(Information security)**: 모델 가중치(weight) 유출, 평가 데이터 유출, 모니터 우회 기법 유출. Lesson 19의 RAND SL-4가 구체적 표준이다.

### CAISI — Center for AI Standards and Innovation

- NIST 내에서 운영된다.
- 프런티어 연구소들과의 자발적 협약을 운영한다.
- 사이버, 생물, 화학 무기 위험에 초점을 맞춘 비분류 능력 평가를 발표한다.
- CAIS와는 별개다. 약어가 충돌한다. 어느 것을 읽고 있는지 확인하려면 URL(nist.gov)을 확인하라.

CAISI의 역할은 METR의 비공개 연구소 참여(Lesson 21)에 대한 공개적, 정부 대상 대응물이다. CAISI 보고서는 비분류다. METR 보고서는 종종 NDA로 게이트(gate)된다. 둘 다 읽는 실무자는 더 완전한 그림을 얻는다.

### California SB-53

캘리포니아 상원 법안(2025–2026년 회기)은 프런티어 모델로부터의 파국적 위험을 다룬다. 초안 기준 핵심 조항:

- 주 수준 의무를 발동하는 특정 능력 임계값(threshold).
- AI 연구소 직원을 위한 내부 고발자(whistleblower) 보호.
- 파국적 실패에 대한 사고 보고 요건.

서명되면, 미국 최초의 주 수준 파국적 위험 규제가 될 것이다. 서명 여부와 무관하게, 이 법안의 프레이밍은 다른 주 의회들이 이 문제에 접근하는 방식을 형성한다. 캘리포니아의 실무자는 이 법안의 상태를 추적해야 한다. 다른 곳의 실무자는 미국 주 수준 규제가 어떤 모습일 가능성이 높은지 이해하기 위해 이를 읽어야 한다.

### 사회적 규모 위험은 단일 계층 문제가 아니다

Phase 15의 일관된 주제(심층 방어(defense in depth))는 사회적 계층에서도 적용된다. 어느 단일 조직, 규제, 또는 프레임워크도 파국적 위험을 닫지 못한다. 생태계는 다음과 같을 때만 기능한다.

- 연구소가 스케일링 정책을 출고한다(Lessons 19, 20).
- 외부 평가자가 측정을 생성한다(Lesson 21).
- 시민 사회가 추적하고 공론화한다(CAIS).
- 정부가 자발적 프로그램과 베이스라인 규제를 운영한다(CAISI, SB-53).
- 실무자가 다층 통제를 구축한다(Lessons 10–18).

이것이 이 단계(phase)의 최종 종합이다. 모든 이전 레슨은 어느 단일 계층의 강도보다 그 완전성이 더 중요한 스택 안의 한 계층이다.

```figure
a5-four-risks
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 작은 위험 인벤토리 도구를 구현한다. 제안된 배포(deployment)가 주어지면, 그 배포를 네 가지 위험 범주에 대조하여 태그하고 완화 체크리스트를 반환한다. 이것은 프레임워크를 위한 읽기 보조 도구이지 인간 판단의 대체물이 아니다.

## 산출물 (Ship It)

`outputs/skill-societal-risk-review.md`는 배포를 사회적 규모 위험 태세에 대해 검토한다. 네 범주 중 어느 것을 건드리는지, 어떤 완화책이 갖춰져 있는지, 조직적 위험 노출이 무엇인지.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 서로 다른 규모의 합성 배포 세 개를 입력하라. 네 가지 위험 태그가 예상과 일치하는지 확인하라. 도구가 과소 태그하거나 과대 태그하는 사례 하나를 식별하라.

2. CAIS 네 가지 위험 논문을 전부 읽어라. 한 위험 범주를 골라, 그 범주에서 가장 중요하다고 믿는 2026년 발전에 대해 두 문단을 작성하라.

3. California SB-53의 현재 초안을 읽어라. 파국적 위험 태세를 강화한다고 믿는 조항 하나와 약화한다고 믿는 조항 하나를 식별하라. 둘 다 정당화하라.

4. 알고 있는 프로덕션 AI 배포(직접 다룬 것이거나 발표된 것)를 골라라. 조직적 위험 하위 레버(안전 문화, 감사 엄격성, 다층 방어, 정보 보안)에 점수를 매겨라. 어느 것이 가장 약한가? 그것을 동등 수준으로 끌어올리는 데 무엇이 들겠는가?

5. 추가된 1년의 능력과 추가된 1년의 배포 경험을 반영하는 네 가지 위험 프레임워크의 2028년 버전을 스케치하라. 무엇을 추가, 제거, 또는 재편하겠는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| CAIS | "Center for AI Safety" | 비영리; 네 가지 위험 프레임워크; 2023년 멸종 성명 |
| CAISI | "미국 정부 AI 안전" | NIST 센터; 자발적 협약; 비분류 평가 |
| 네 가지 위험 프레임워크 (Four-risk framework) | "CAIS의 분류 체계" | 악의적 사용, AI 경쟁, 조직적 위험, 악성 AI |
| 악의적 사용 (Malicious use) | "악의적 행위자가 AI를 사용" | 생물무기, 허위정보, 사이버 공격 |
| AI 경쟁 (AI races) | "경쟁 압력" | 연구소/기업/국가가 배포를 안전 너머로 밀어붙인다 |
| 조직적 위험 (Organizational risk) | "연구소 내부 실패" | 안전 문화, 감사, 방어, 정보 보안 |
| 악성 AI (Rogue AI) | "정렬되지 않은 에이전트" | 인간 복지와 충돌하는 목표를 추구하는 유능한 AI |
| California SB-53 | "주 수준 규제" | 2025–2026년 법안; 서명되면 미국 최초 주 파국적 위험 규제 |

## 더 읽을거리 (Further Reading)

- [Center for AI Safety](https://safe.ai/): 네 가지 위험 프레임워크의 기관 본거지.
- [CAIS: AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) 네 가지 위험 논문.
- [CAIS: May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) 짧은 공동 성명.
- [NIST CAISI](https://www.nist.gov/caisi): 정부 대상 AI 표준 및 혁신 센터.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 연구소 수준 약속을 사회적 규모 프레이밍과 연결한다.
