# 협상과 흥정 (Negotiation and Bargaining)

> 에이전트(agent)는 자원, 가격, 작업 할당, 조건을 협상한다. 2026 벤치마크(benchmark) 모음은 명확하다. NegotiationArena(arXiv:2402.05863)는 LLM이 페르소나(persona) 조작("절박함")으로 보수(payoff)를 ~20% 개선할 수 있음을 보여준다. "Measuring Bargaining Abilities"(arXiv:2402.15813)는 구매자가 판매자보다 어렵고 규모가 도움이 안 됨을 보여준다. 이 연구의 **OG-Narrator**(결정론적 제안 생성기 + LLM 내레이터)는 거래 성사율을 26.67%에서 88.88%로 끌어올렸다. Large-Scale Autonomous Negotiation Competition(arXiv:2503.06416)은 ~18만 건의 협상을 돌렸고 **사고 사슬 은폐(chain-of-thought-concealing)** 에이전트가 상대로부터 추론을 숨겨 이긴다는 것을 발견했다. Bhattacharya et al. 2025는 Harvard Negotiation Project 지표에서 Llama-3을 가장 효과적, Claude-3을 공격적, GPT-4를 가장 공정한 것으로 순위 매겼다. 이 레슨은 Contract Net Protocol(FIPA 조상, Lesson 02)을 구현하고, LLM 스타일 구매자/판매자를 배선하고, OG-Narrator 스타일 분해를 돌리고, 각 구조적 선택에 따라 거래 성사율이 어떻게 바뀌는지 측정한다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75분

## 문제 (Problem)

두 에이전트가 가격에 합의해야 한다. 순수 언어 프롬프트(prompt)만으로 방치하면, 2024~2026 LLM은 놀랍도록 낮은 비율로 거래를 성사시킨다(arXiv:2402.15813에서 빡빡하게 파라미터화된 흥정에서 ~27%). 규모는 이를 고치지 못한다. GPT-4는 흥정에서 GPT-3.5보다 구조적으로 더 낫지 않다. 흥정의 *언어*에 더 능할 뿐이다.

근본 문제는 LLM이 두 가지 일을 혼동한다는 것이다. 제안을 결정하는 것과 제안을 서술(narrate)하는 것이다. OG-Narrator는 이를 분리했다. 결정론적 제안 생성기가 수치적 수를 계산하고, LLM은 서술만 한다. 거래 성사율이 ~89%로 뛴다.

여기엔 고전적인 멀티 에이전트 교훈이 담겨 있다. 메커니즘을 통신 층에서 분리하는 쪽이 이긴다. Contract Net Protocol(FIPA, 1996; Smith, 1980)은 레퍼런스 작업 시장(task-market) 메커니즘이다. 서술 슬롯에 LLM을 꽂으면 현대적 LLM 기반 작업 시장이 된다.

## 개념 (Concept)

### Contract Net, 한 문단으로

Smith의 1980년 Contract Net Protocol: **매니저(manager)**가 **제안 요청(call for proposals, cfp)**을 브로드캐스트한다. **입찰자(bidders)**가 자신의 제안을 담은 **propose** 메시지로 응답한다. 매니저가 승자를 고르고 승자에게 **accept-proposal**을, 패자에게 **reject-proposal**을 보낸다. 승자가 작업을 수행한다. 선택적 메시지: **refuse**(입찰자가 제안을 거절). FIPA는 이를 `fipa-contract-net` 상호작용 프로토콜로 코드화했다.

### 왜 OG-Narrator가 이기는가

"Measuring Bargaining Abilities of Language Models"(arXiv:2402.15813)는 다음을 관찰했다.

- LLM은 흥정 규칙을 자주 어긴다(말이 안 되는 가격으로 제안, 상대의 ZOPA 무시).
- 앵커링(anchoring)을 못 한다(나쁜 첫 제안을 수락, 전략적이 아니라 상징적인 금액으로 역제안).
- 규모만으로는 이를 고치지 못한다. 더 큰 모델은 비슷한 전략적 오류를 가진 채 더 그럴듯한 언어를 만든다.

OG-Narrator 분해:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

제안 생성기는 고전적 협상 전략이다. 루빈스타인 흥정 모델(Rubinstein bargaining model), 제우텐 전략(Zeuthen strategy), 또는 가격에 대한 단순 팃포탯(tit-for-tat)이다. LLM이 서술한다. 메시지는 결정론적 가격과 자연어 프레이밍을 담는다.

거래 성사율이 뛰는 이유는:
- 가격이 흥정 구역(bargaining zone) 안에 머문다.
- 앵커가 감정적이 아니라 전략적이다.
- LLM이 잘하는 것을 한다: 글쓰기.

### NegotiationArena 발견

arXiv:2402.05863은 정전(canonical) 벤치마크를 제공한다. 핵심 발견:

- LLM은 페르소나를 채택해("나는 금요일까지 이것을 팔아야 해서 절박하다") 보수를 ~20% 개선할 수 있다. 페르소나 조작은 실재하는 전술이다.
- 공정/협력적 에이전트는 적대적 에이전트에게 착취당한다. 방어에는 명시적 맞대응 자세(counter-posturing)가 필요하다.
- 대칭적 짝짓기가 벤치마크 시나리오의 약 40%에서 불공평한 결과로 수렴한다.

이는 "LLM이 나쁜 협상가"라는 것이 아니다. "LLM이 인간과 너무 비슷하게 협상한다. 착취당할 수 있는 부분까지 포함해서"라는 것이다.

### 사고 사슬 은폐

Large-Scale Autonomous Negotiation Competition(arXiv:2503.06416)은 많은 LLM 전략에 걸쳐 ~18만 건의 협상을 돌렸다. 승자들은 상대로부터 추론을 은폐했다.

- 에이전트가 공개적으로 보이는 스크래치패드(scratchpad)에 "나는 $75까지만 갈 것이다. 내 유보 가격(reservation price)은 $70이다"라고 출력하면, 상대가 그것을 읽는다.
- 승자들은 전략을 비공개로 계산한다. 출력 채널에는 제안과 최소한의 필수 서술만 담긴다.

고전적 게임 이론(Aumann 1976, 합리성과 정보)이 2026년에 다시 울리는 셈이다. 자신의 사적 평가(private valuation)를 드러내면 보수가 든다. LLM은 이를 직관하지 못하고, 상대에게 보이게 되는 추론 추적에 자신의 유보값을 즐겁게 타이핑한다.

엔지니어링 요점: 비공개 스크래치패드 맥락(context)을 공개 메시지 맥락에서 분리하라. 선택 사항이 아니다.

### Bhattacharya et al. 2025: 모델 순위

Harvard Negotiation Project 지표(원칙 협상, BATNA 존중, 이해관계 호혜성)에서:

- **Llama-3**가 흥정 성사에 가장 효과적이었다(거래 성사율 + 보수).
- **Claude-3**가 가장 공격적인 협상가였다(높은 앵커, 늦은 양보).
- **GPT-4**가 가장 공정했다(짝짓기에 걸친 보수 분산이 가장 작음).

이는 2025년 스냅샷이다. 요점은 2026년 4월에 어느 모델이 이기느냐가 아니다. 서로 다른 베이스 모델이 지속적인 협상 스타일을 가진다는 것이다. 이질적 앙상블(Lesson 15)은 이를 다양성 소스로 포함한다.

### Contract Net + LLM을 통한 작업 할당

LLM 멀티 에이전트를 위한 Contract Net의 현대적 재사용:

1. 매니저 에이전트가 작업을 단위로 분해한다.
2. 작업 설명을 담은 `cfp`를 워커 에이전트에게 브로드캐스트한다.
3. 각 워커가 제안을 반환한다: `(price, eta, confidence)`. 여기서 price는 토큰, 컴퓨트 단위, 또는 달러일 수 있다.
4. 매니저가 승자(작업에 따라 단일 또는 다수)를 고르고 수여한다.
5. 거부된 워커는 다른 작업에 자유롭게 입찰한다.

조정이 동기적 채팅이 아니라 브로드캐스트-앤-응답이기 때문에 워커 100개를 훨씬 넘어서까지 잘 확장된다. 프로덕션(production)에서 사용됨: Microsoft Agent Framework의 오케스트레이션 패턴, 일부 LangGraph 구현.

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024(https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf)는 **비밀 점수(secret scores)**와 **최소 수락 임계값(minimum-acceptance thresholds)**을 가진 다자 채점 게임을 도입한다. 각 이해관계자가 사적 효용(private utility)을 가진다. LLM은 메시지로부터 이를 추론해야 한다. 이는 양자 흥정을 N자 연합 형성(coalition formation)으로 일반화한 것이다. 이질적 워커 능력을 가진 프로덕션 작업 시장에 관련 있다.

### 서술 vs 메커니즘 규칙

모든 2024~2026 협상 벤치마크에 걸쳐, 일관된 엔지니어링 규칙은 다음과 같다.

> LLM이 서술하게 하라. LLM이 제안을 계산하게 하지 마라.

제안이 숫자여야 한다면(가격, ETA, 수량), 협상 상태에서 결정론적으로 생성하고 LLM이 프레이밍을 만들게 하라. 제안이 제안 구조여야 한다면(작업 분해, 역할 배정), LLM이 초안을 작성하게 하되, 보내기 전에 스키마에 대해 검증하고 제약 검사하라.

```figure
a5-og-narrator
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다.

- `ContractNetManager`, `ContractNetTask`, `Bid`: 매니저 + 입찰자, cfp 브로드캐스트, 제안 수집, 수여.
- `og_narrator_bargain(state, rng)`: OG-Narrator 구매자: 중간점을 향한 결정론적 제우텐 스타일 양보.
- `seller_response(state, rng)`: 결정론적 판매자 역제안 정책(두 스타일 모두의 구조적 정답).
- `naive_llm_bargain(state, rng)`: 전부 LLM인 흥정가를 시뮬레이션한다: 높은 분산으로 가격을 고르며, 종종 ZOPA 밖이다.
- 측정: 시도마다 새로 샘플링한 유보 가격으로 1000회 시도에 걸친 거래 성사율.

실행:

```
python3 code/main.py
```

예상 출력: 순진한-LLM 거래 성사율 ~65~75%, OG-Narrator 거래 성사율 ~85~95%. 15~25점 격차가 제안 생성을 서술에서 분해한 구조적 이점이다. 더불어 입찰자 셋과 작업 하나를 동반한 Contract Net 작업 시장 할당 예제.

## 라이브러리로 써보기 (Use It)

`outputs/skill-bargainer-designer.md`는 흥정 프로토콜을 설계한다. 누가 제안을 생성하는지(결정론적 또는 LLM), 누가 서술하는지, 비공개 스크래치패드가 공개 메시지에서 어떻게 분리되는지, 거래 성사율이 어떻게 모니터링되는지.

## 산출물 (Ship It)

프로덕션 흥정 체크리스트:

- **별도 스크래치패드.** 비공개 상태가 절대 상대의 맥락에 닿지 않는다. 이는 타협 불가다.
- **결정론적 제안 생성.** 가격, 수량, ETA: 계산하라, 프롬프트하지 마라.
- **모든 수신 제안을 스키마에 대해 검증하라.** 프로토콜 경계에서 ZOPA 밖 제안을 거부하라.
- **라운드를 한정하라.** 최대 3~5라운드. 교착(deadlock) 시 중재자(mediator)로 에스컬레이션(escalation)하라.
- **거래 성사율과 보수 분산을 지속적으로 측정하라.** 떨어지는 거래 성사율은 증상이다. 종종 프롬프트 드리프트(drift)나 상대 측 공격이다.
- **모든 거부된 제안을 결정론적 근거와 함께 기록하라.** Contract Net 매니저의 경우, 패한 입찰자가 이유를 이해할 필요가 있다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. OG-Narrator가 거래 성사율에서 순진한-LLM을 이기는지 확인하라. 얼마나 차이가 나는가?
2. **페르소나 기반 보수 개선**(arXiv:2402.05863)을 구현하라. 구매자가 서술에서만 "이번 주에 이것을 사야 해서 절박하다" 페르소나를 채택한다. 제안 생성기는 변경하지 않는다. 거래 성사율이나 보수가 바뀌는가?
3. 사고 사슬 **은폐**를 구현하라. 상대에게 전달되지 않는 비공개 스크래치패드 문자열을 유지한다. 실수로 이를 유출하면(채널을 바꿔서 시뮬레이션) 무슨 일이 일어나는가?
4. Contract Net를 유보 가격(reserve price)을 가진 N-입찰자 경매로 확장하라. 입찰이 모두 유보가를 넘으면, 매니저는 최저가와 최고 품질 사이에서 어떻게 결정하는가? 어떤 수여 규칙을 고르며 그 이유는?
5. Harvard Negotiation Project 지표에 관한 Bhattacharya et al. 2025를 읽어라. 서로 다른 스타일(공격적 vs 공정)을 가진 두 흥정가를 구현하라. 대칭 및 비대칭 짝짓기 하에서 보수 분산을 측정하라.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Contract Net | "작업 시장" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. 정전적 작업 시장. |
| ZOPA | "합의 가능 구역" | 구매자 최대와 판매자 최소 사이의 겹침. 그 밖의 제안은 성사될 수 없다. |
| BATNA | "협상된 합의에 대한 최선의 대안" | 이 거래가 실패할 때의 폴백(fallback). 유보 가격을 정하는 기준이 된다. |
| OG-Narrator | "제안 생성기 + 내레이터" | 분해: 결정론적 제안, LLM 서술. |
| 제우텐 전략(Zeuthen strategy) | "위험 최소화 양보" | 위험 한계에 기반해 양보하는 고전적 제안 생성기. |
| 루빈스타인 흥정(Rubinstein bargaining) | "교대 제안 균형" | 할인을 동반한 무한 지평 흥정에 대한 게임 이론 모델. |
| CoT 은폐 | "추론을 숨겨라" | arXiv:2503.06416의 승자들은 비공개 스크래치패드를 유지했다. 공개 채널은 제안만 보인다. |
| 페르소나 조작(Persona manipulation) | "감정적 자세 취하기" | arXiv:2402.05863: 절박함/긴급함 페르소나로 ~20% 보수 향상. |

## 더 읽을거리 (Further Reading)

- [NegotiationArena](https://arxiv.org/abs/2402.05863): 벤치마크. 페르소나 조작과 착취 발견
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813): OG-Narrator와 구매자가 판매자보다 어렵다는 결과
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416): ~18만 건의 협상. 사고 사슬 은폐가 이긴다
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf): 비밀 효용을 가진 다자 채점 게임
- [Smith 1980: The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) 고전적 메커니즘, IEEE Transactions on Computers
