# OpenAI Preparedness Framework와 DeepMind Frontier Safety Framework (OpenAI Preparedness Framework and DeepMind Frontier Safety Framework)

> OpenAI Preparedness Framework v2(2025년 4월)는 추적 카테고리(Tracked Categories)와 구별되는 연구 카테고리(Research Categories)(장거리 자율성(Long-range Autonomy), 샌드배깅(Sandbagging), 자율 복제 및 적응(Autonomous Replication and Adaptation), 안전장치 약화(Undermining Safeguards))를 도입한다. 추적 카테고리는 Safety Advisory Group이 검토하는 Capabilities Report와 Safeguards Report를 발동한다. DeepMind의 FSF v3(2025년 9월, 2026년 4월 17일에 Tracked Capability Levels 추가)는 자율성을 ML R&D와 Cyber 영역으로 접어 넣는다(ML R&D 자율성 레벨 1 = 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인(pipeline)을 완전 자동화). FSF v3은 도구적 추론(instrumental reasoning) 오용에 대한 자동 모니터링을 통해 기만적 정렬(deceptive alignment)을 명시적으로 다룬다. 정직한 메모: PF v2의 연구 카테고리(장거리 자율성 포함)는 완화책을 자동으로 발동하지 않는다. 정책 언어는 "잠재적(potential)"이다. DeepMind 자체도 도구적 추론이 강화되면 자동 모니터링이 "장기적으로 충분한 채로 남아 있지 않을 것"이라고 말한다.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45분

## 문제 (The Problem)

Lesson 19는 Anthropic의 스케일링 정책을 면밀히 읽었다. 이 레슨은 OpenAI와 DeepMind의 것을 읽음으로써 그림을 완성한다. 세 문서는 같은 질문(프런티어 연구소(frontier lab)는 언제 모델을 일시 중지하거나 게이트(gate)해야 하는가)을 다루는 사촌 격 산출물이며, 소수의 카테고리로 수렴하고 중요한 특정 지점들에서 갈린다.

수렴: 셋 모두 장거리 자율성을 추적할 가치가 있는 능력 부류로 표시한다. 셋 모두 기만적 행동(정렬 위조(alignment faking), 샌드배깅)을 특정 위험 부류로 인정한다. 셋 모두 내부 검토 기구를 둔다. 발산: OpenAI는 카테고리를 "추적"(의무적 완화)과 "연구"(자동 발동 없음)로 나눈다. DeepMind는 자율성을 별도로 명명하기보다 두 영역으로 접어 넣는다. 연구소는 추적 대 연구, 또는 임계(Critical) 대 중간(Moderate), 또는 Tier-1 대 Tier-2를 명명한다. 어떤 능력이 어느 버킷에 속하는지의 운영적 결과는 연구소마다 다르다.

세 문서를 함께 읽는 것이 유용한 연습이다. 동일한 능력이 Anthropic에서는 "의무적 완화", OpenAI에서는 "모니터링되지만 발동하지 않음", DeepMind에서는 "특정 영역에서 추적됨"일 수 있다. 정책 태세가 중요하다.

## 개념 (The Concept)

### OpenAI Preparedness Framework v2 (2025년 4월)

구조:

- **추적 카테고리(Tracked Categories)**: Capabilities Report(모델이 무엇을 할 수 있는가)와 Safeguards Report(어떤 완화책이 갖춰져 있는가)를 발동한다. 배포 전에 Safety Advisory Group이 검토한다.
- **연구 카테고리(Research Categories)**: 연구소가 추적하고 있지만 아직 특정 완화책을 약속하지 않은 모델 능력. 장거리 자율성, 샌드배깅, 자율 복제 및 적응, 안전장치 약화를 포함한다.

v2의 핵심 프레이밍: 연구 카테고리는 완화책을 자동으로 발동하지 않는다. 정책 언어는 "잠재적" 완화책이다. 이것이 주시해야 할 구체적 차이(delta)다. 어떤 카테고리가 추적 대 연구 중 어디에 위치하는지가 "게이트됨"과 "관찰됨" 사이의 경첩이다.

### DeepMind Frontier Safety Framework v3 (2025년 9월; Tracked Capability Levels는 2026년 4월 17일 추가)

구조:

- **임계 능력 수준(Critical Capability Levels, CCLs)**: 다섯 영역에 걸친 능력 임계값. Cyber, Bio, ML R&D, CBRN, Autonomy(ML R&D와 Cyber로 접어 넣음).
- **추적 능력 수준(Tracked Capability Levels)**: 2026년 4월에 추가된 추가적 세분성. 구체적 예시: ML R&D 자율성 레벨 1 = 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인을 완전 자동화.
- **기만적 정렬 모니터링**: 도구적 추론 오용에 대한 자동 모니터링을 명시적으로 약속한다.

자율성 프레이밍이 OpenAI의 것과 다르다. DeepMind는 "자율성"을 최상위 영역으로 두지 않는다. 자율성이 해를 끼칠 수 있는 영역(ML R&D와 Cyber)으로 접어 넣는다. 그 논거는 영역 없는 자율성은 위험 없는 능력이라는 것이다. 반론은 영역들에 걸친 자율성이 프레임워크가 명명해야 할 메타 위험(meta-risk)이라는 것이다.

### 셋 모두가 수렴하는 것

- 내부 Safety Advisory Group(Anthropic SAG, OpenAI SAG, DeepMind 내부 위원회로 명명됨). 고능력 모델에 대한 배포 전 검토.
- 기만적 정렬 / 정렬 위조를 위험 부류로 명시적으로 언급.
- 선언된 주기의 상설 산출물(Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Report; DeepMind: FSF 갱신 주기).
- 모니터링 전용 방어에 천장이 있다는 인정. DeepMind는 명시적이다. "자동 모니터링은 장기적으로 충분한 채로 남아 있지 않을 것이다."

### 어디서 갈리는가

- **Anthropic**: v3.0에서 일시 중지 약속 제거. AI R&D-4 임계값이 명명된 다음 게이트.
- **OpenAI**: 추적 대 연구 분할. 연구 카테고리(장거리 자율성 포함)는 자동으로 게이트하지 않는다.
- **DeepMind**: 자율성을 다른 영역으로 접어 넣음. Tracked Capability Levels가 2026년 4월에 세분성을 추가.

### 샌드배깅: 셋 모두를 복잡하게 만드는 특정 능력

샌드배깅(모델이 평가에서 전략적으로 저성능을 내는 것)은 OpenAI의 연구 카테고리에 있다. Anthropic의 RSP v3.0은 이를 평가-맥락 간극(evaluation-context gap, Lesson 1)을 통해 다룬다. DeepMind는 이를 FSF v3의 기만적 정렬 모니터링을 통해 다룬다.

모델이 평가에서 샌드배깅을 하면, 모든 프레임워크의 능력 임계값이 과소평가된다. 프레임워크는 측정이 작동할 때만 작동한다. 이것이 연구소 자체 평가에 더해 외부 측정(Lesson 21, METR)과 적대적 평가가 필요한 이유다.

### 정책 읽기 기술

- 위치 찾기(Locate): 신경 쓰는 모든 능력이 정책에서 찾을 수 있어야 한다. 찾을 수 없으면 정책은 그것을 커버하지 않는 것이다.
- 분류(Classify): 그것은 추적(완화 발동)인가 연구(추적되지만 발동하지 않음)인가? OpenAI는 이를 명명한다. Anthropic과 DeepMind는 자신만의 대응물을 둔다.
- 주기(Cadence): 정책이 선언된 일정으로 갱신되는가, 아니면 특정 사건 이후에만 갱신되는가? 선언된 주기가 더 강하다.
- 독립성(Independence): 외부 검토가 의무적인가 선택적인가? Anthropic은 Apollo 및 US AI Safety Institute와 협력한다. OpenAI는 METR과. DeepMind는 주로 내부 SAG와.

```figure
a5-tracked-vs-research
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 작은 결정 테이블 차이(diff) 도구를 구현한다. 어떤 능력(자율성, 기만적 정렬, R&D 자동화, 사이버 증강 등)이 주어지면, 세 정책 각각이 그 능력을 어떻게 분류하는지, 그리고 어떤 완화책이 발동하는지를 출력한다. 이것은 읽기 보조 도구이지 정책 도구가 아니다.

## 산출물 (Ship It)

`outputs/skill-cross-policy-diff.md`는 세 프레임워크를 참조로 사용하여 특정 능력에 대한 교차 정책 비교를 생성한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 차이 도구의 출력이 원본 문서에 대조하여 검증할 수 있는 최소 두 가지 능력에 대해 정책과 일치하는지 확인하라.

2. OpenAI Preparedness Framework v2를 전부 읽어라. 각 연구 카테고리를 식별하라. 각각에 대해, 왜 그것이 추적이 아니라 연구에 있는지 한 문장으로 작성하라.

3. DeepMind FSF v3을 전부 읽고, 2026년 4월 Tracked Capability Levels 갱신도 읽어라. ML R&D 자율성 레벨 1의 구체적 평가 기준을 식별하라. 그것을 외부적으로 어떻게 측정하겠는가?

4. 샌드배깅은 OpenAI의 연구 카테고리에 있다. 샌드배깅하는 모델이 실제 능력을 드러내도록 강제하는 평가를 설계하라. Lesson 1의 평가-맥락 게이밍(eval-context-gaming) 논의를 참조하라.

5. 특정 능력(직접 선택)에 대해 세 정책을 비교하라. 어느 정책의 분류가 가장 엄격하고 어느 것이 가장 덜 엄격한지 명명하라. 원본 텍스트로 정당화하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| Preparedness Framework | "OpenAI의 스케일링 정책" | PF v2(2025년 4월); 추적 대 연구 카테고리 |
| 추적 카테고리 (Tracked Category) | "의무적 완화" | Capabilities + Safeguards Report 발동; SAG 검토 |
| 연구 카테고리 (Research Category) | "모니터링만" | 추적되지만 자동 완화 없음; 장거리 자율성 포함 |
| Frontier Safety Framework | "DeepMind의 스케일링 정책" | FSF v3(2025년 9월) + Tracked Capability Levels(2026년 4월) |
| CCL | "임계 능력 수준" | 영역별 DeepMind 임계값(Cyber, Bio, ML R&D, CBRN) |
| ML R&D 자율성 레벨 1 (ML R&D autonomy level 1) | "R&D 자동화" | 경쟁력 있는 비용으로 AI R&D 파이프라인 완전 자동화 |
| 샌드배깅 (Sandbagging) | "전략적 저성능" | 모델이 평가에서 저성능; OpenAI 연구 카테고리에 있음 |
| 도구적 추론 (Instrumental reasoning) | "수단-목적 추론" | 목표 달성 방법에 대한 추론; DeepMind 모니터링의 대상 |

## 더 읽을거리 (Further Reading)

- [OpenAI: Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) v2 발표.
- [OpenAI: Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) 전체 문서.
- [DeepMind: Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) FSF v3 발표.
- [DeepMind: Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) Tracked Capability Levels 추가.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf): FSF 형식 Risk Report의 예시.
