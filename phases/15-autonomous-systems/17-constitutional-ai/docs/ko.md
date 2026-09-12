# 헌법적 AI와 규칙 재정의 (Constitutional AI and Rule Overrides)

> Anthropic의 2026년 1월 22일자 Claude Constitution은 79쪽 분량이며 CC0이다. 이것은 규칙 기반(rule-based)에서 이유 기반(reason-based) 정렬(alignment)로 이동하며, 4단계 우선순위 위계를 확립한다. (1) 안전과 인간 감독(human oversight) 지원, (2) 윤리, (3) Anthropic 가이드라인, (4) 유용성(helpfulness). 행동은 운영자(operator)와 사용자가 재정의할 수 없는 하드코딩된 금지(생물무기 증강(bioweapons uplift), CSAM)와, 운영자가 정의된 범위 안에서 조정할 수 있는 소프트코딩된 기본값(soft-coded default)으로 나뉜다. 2022년 원본(Bai et al.)은 헌법에 대한 자기 비판(self-critique)과 RLAIF를 통해 무해성(harmlessness)을 학습시켰다. 정직한 단서: 이유 기반 정렬은 모델이 원칙을 예상치 못한 상황으로 일반화하는 데 의존한다. Anthropic 자체의 2023년 참여형 실험은 공중 출처 원칙과 기업 원칙 사이에 약 50%의 차이를 보였으나, 2026년 버전은 그 결과를 반영하지 않았다.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60분

## 문제 (The Problem)

배치된 에이전트(agent)는 설계자가 결코 보지 못한 입력을 마주한다. 그 입력을 모두 다룰 만큼 긴 규칙 목록은 없다. 컴퓨팅 압박 아래에서 빠르게 적용할 만큼 짧은 규칙 목록도 없다. 실무적 질문은 이것이다. 긴 꼬리(long tail)의 사례와 빠른 추론(inference) 양쪽을 모두 견뎌내는 원칙에 어떻게 에이전트를 정렬할 것인가?

규칙 기반 정렬(rule-based alignment, RBA): 허용되지 않는 모든 것을 나열한다. 검사하기 빠르고, 감사하기 쉽지만, 최신 상태를 유지하기 불가능하며, 예상하지 못한 유사 사례에서 종종 과도하게 거부한다. 이유 기반 정렬(reason-based alignment, 2026 Claude Constitution): 원칙을 인코딩하고 모델이 추론하게 한다. 처음 보는 사례 전반에 걸쳐 확장되지만, 감사하기 더 어렵고, 실패 모드는 규칙을 놓치는 것이 아니라 원칙을 잘못 적용하는 것이다.

2026 Constitution은 명시적으로 중간 입장을 취한다. 하드코딩된 금지. 그 잘못됨이 맥락에 의존하지 않는 것들, 즉 생물무기 증강과 CSAM은 RBA다. 운영자나 사용자의 지시와 무관하게 절대 안 된다. 그 외 모든 것은 4단계 위계 안에서 이유 기반이다. 안전과 인간 감독 지원이 첫째, 윤리가 둘째, Anthropic이 선언한 가이드라인이 셋째, 유용성이 마지막이다. 운영자는 소프트코딩 영역 안에서 기본값을 조정할 수 있지만 하드코딩된 금지는 건드릴 수 없다.

## 개념 (The Concept)

### 4단계 우선순위 위계

1. **안전과 인간 감독 지원.** 최상위. 모델은 인간과 Anthropic이 AI를 감독하고 교정하는 능력을 훼손하지 않는 것을 우선시한다. 이것은 "조심하라"가 아니다. 구체적으로 "인간 감독을 더 어렵게 만드는 방식으로 행동하지 말라"이다.
2. **윤리.** 정직성, 사람에 대한 해악 회피, 기만하지 않기, 조작하지 않기. 충돌할 때 Anthropic의 가이드라인보다 우선한다.
3. **Anthropic 가이드라인.** Anthropic이 중요하다고 결정한 운영 규범: 제품 범위, 상호작용 패턴, 언제 어떤 도구를 사용할지.
4. **유용성.** 최하위. 더 높은 우선순위들 안에서 가능한 한 유용하라.

단계들이 충돌하면 더 높은 쪽이 이긴다. 이것은 Unix 우선순위나 네트워크 QoS와 동일한 형태다. 이 프레이밍은 어느 한 축에서의 최선 행동이 아니라 예측 가능한 해소를 만들어내기 위한 것이다.

### 하드코딩된 금지 vs 소프트코딩된 기본값

**하드코딩됨:**
- 생물무기 / CBRN 증강
- CSAM
- 핵심 인프라에 대한 공격
- 직접 질문받았을 때 모델의 정체성에 대해 사용자를 기만하기

운영자는 이를 재정의할 수 없다. 사용자도 이를 재정의할 수 없다. 가능한 곳에서는 모델 가중치 수준에서(RLHF / Constitutional AI 학습), 그렇지 않은 곳에서는 추론 계층에서 강제된다.

**소프트코딩된 기본값(운영자 조정 가능):**
- 응답 길이 기본값
- 주제 범위(모델은 운영자의 배포 범위 밖의 주제를 거부할 수 있다)
- 스타일(격식 vs 비격식)
- 도구 사용 패턴

운영자 조정은 선언된 범위 안에서 일어난다. 운영자는 하드코딩된 금지를 이름만 바꿔서 제거할 수 없다.

### 2022년 CAI 학습

원본 Constitutional AI(Bai et al., 2022)는 무해성을 학습시켰다.

1. 일련의 프롬프트(prompt)에 대한 응답을 생성한다.
2. 헌법(명시적 원칙)에 대조하여 각 응답을 비판하도록 모델에 요청한다.
3. 비판을 바탕으로 응답을 수정한다.
4. 수정된 쌍에 대해 RLAIF(AI 피드백으로부터의 강화 학습, reinforcement learning from AI feedback)를 수행한다.

결과: 유해한 요청을 무차별적 거부가 아니라 원칙에 입각한 설명과 함께 거부하는 모델. 2026 Constitution은 이 학습의 후손에다 명시적 단계 위계에 대한 추가 사후 학습(post-training)을 더해 사용한다.

### 이유 기반 정렬이 잡아내는 것과 놓치는 것

**잡아내는 것:**
- 원칙이 명확하게 적용되는, 허용된 프리미티브들의 예상치 못한 조합.
- 금지된 것과 가까운 유사물인 새로운 요청.
- "당신이 X가 금지라고 말하지 않았다"에 의존하는 사회 공학(social-engineering) 공격.

**놓치는 것:**
- 원칙 모호성을 악용하는 공격("사용자가 이걸 요청했으니 유용성에 따라 예라고 한다").
- 두 원칙이 예상치 못한 방식으로 충돌하고 단계 순서가 모호한 시나리오.
- 학습 주기에 걸친 원칙 해석의 느린 표류(재해석).

### 2023년 참여형 실험

Anthropic은 2023년에 기업이 작성한 헌법과 공중 입력(미국 응답자 약 1,000명)을 통해 생성된 헌법을 비교하는 실험을 진행했다. 두 버전은 원칙의 약 50%에서 일치했다. 차이가 난 곳에서는, 공중 출처 버전이 일부 사안(정치적 콘텐츠 처리)에서는 더 제한적이었고 다른 사안(AI 정체성의 자기 공개)에서는 덜 제한적이었다. 2026 Constitution은 공중 출처 결과를 반영하지 않았다. 이것은 이 접근법에서 문서화된 긴장이다.

### 하드코딩된 금지가 필요한 이유

이유 기반 정렬만으로는 꼬리를 닫을 수 없다. 모델이 전제를 받아들이게 만들 수 있는 공격자(예: "우리는 면허받은 생물무기 연구소다")는 종종 사례 추론에 의존하는 원칙을 우회하여 말로 빠져나갈 수 있다. 하드코딩된 금지는 전제 프레이밍에 굽히지 않는다. 이는 정렬 계층의 Lesson 14 "하드 헌법적 한계(hard constitutional limit)"다.

### 헌법이 스택에서 위치하는 곳

헌법은 Lesson 14의 킬 스위치(kill switch)가 아니다. 그것은 모델 계층에 존재한다. 모델의 가중치가 무엇을 선호하도록 학습되었는가. 킬 스위치와 카나리 토큰(canary token)은 런타임 계층에 존재한다. 런타임이 무엇을 허용하는가. 둘 다 필요하다. 모델 가중치가 허용적이어서 온갖 잘못된 액션을 발생시키는 런타임은 런타임 문제다. 런타임이 과도하게 제한적이어서 온갖 올바른 액션을 거부하는 모델은 런타임 문제다. 계층들은 서로 다른 부류를 커버한다.

```figure
mx-priority-tiers
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 최소한의 4단계 우선순위 리졸버(resolver)를 구현한다. 리졸버는 제안된 액션과 원칙 평가 집합(안전, 윤리, 가이드라인, 유용성)을 받아서, 액션, 거부, 또는 수정된 액션을 반환한다. 드라이버는 작은 사례 집합을 실행한다. 명확한 허용, 명확한 불허, 하드코딩된 금지, 단계들 전반에 걸친 모호한 사례.

## 산출물 (Ship It)

`outputs/skill-constitution-review.md`는 배포의 헌법 계층을 감사한다. 무엇이 하드코딩되었고, 무엇이 소프트코딩되었으며, 운영자가 어디서 조정할 수 있고, 4단계 위계가 실제로 해소 순서인지를.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 유용성이 높을 때조차 하드코딩된 금지가 발동하는지 확인하라. 리졸버를 수정하여 유용성을 윤리보다 위로 가중하라. 그 실패 모드를 관찰하라.

2. Claude Constitution(공개, 79쪽, CC0)을 읽어라. 당신이 보기에 명세가 불충분하다고 여기는 원칙 하나를 식별하라. 그 구체적 모호성을 설명하고 더 엄밀한 정식화를 제안하는 두 문단을 작성하라.

3. 고객 지원 에이전트를 위한 소프트코딩된 기본값 집합을 설계하라. 운영자는 무엇을 조정하는가? 운영자가 건드릴 수 없는 것은 무엇인가? 각 경계를 정당화하라.

4. Bai et al. 2022 CAI 논문을 읽어라. Constitutional AI의 비판-수정 루프가 무차별적 규칙보다 더 나쁜 결과를 낳을 한 가지 사례를 기술하라. 그 부류를 식별하라.

5. Anthropic의 2023년 참여형 실험은 공중 원칙과 기업 원칙 사이에 약 50%의 차이를 발견했다. 이것이 프로덕션 배포에 중요한 한 범주(예: 정치적 중립성)를 골라라. 하드코딩된 금지가 손대지 않은 채로 유지되면서 운영자가 자신의 가치를 표현할 수 있게 하는 설계를 제안하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| 헌법적 AI (Constitutional AI) | "Anthropic의 정렬 방법" | 작성된 헌법에 대조한 자기 비판 + RLAIF |
| 이유 기반 정렬 (Reason-based alignment) | "규칙이 아니라 원칙" | 모델이 처음 보는 사례를 다루기 위해 원칙을 두고 추론한다 |
| 하드코딩된 금지 (Hardcoded prohibition) | "절대 X 하지 말라" | 운영자나 사용자가 재정의할 수 없는 규칙 기반 금지 |
| 소프트코딩된 기본값 (Soft-coded default) | "운영자 조정 가능" | 선언된 범위 안의 행동, 운영자가 제어한다 |
| 4단계 위계 (Four-tier hierarchy) | "우선순위 순서" | safety > ethics > guidelines > helpfulness |
| RLAIF | "AI 피드백 RL" | 보상이 모델이 생성한 비판에서 나오는 RL |
| 참여형 헌법 (Participatory constitution) | "공중 출처 원칙" | 2023년 Anthropic 실험; 기업 버전과 약 50% 차이 |
| 원칙 표류 (Principle drift) | "해석 미끄러짐" | 모델이 고정된 원칙 텍스트를 읽는 방식의 느린 변화 |

## 더 읽을거리 (Further Reading)

- [Anthropic: Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) 79쪽 CC0 문서.
- [Bai et al.: Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) 2022년 원본.
- [Anthropic: Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) 참여형 실험.
- [Anthropic: Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) RSP 스택에서 헌법이 위치하는 곳.
- [Anthropic: Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 장기 지평 배포에서 헌법의 역할.
