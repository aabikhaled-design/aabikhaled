# 에이전트를 위한 합의와 비잔틴 결함 허용 (Consensus and Byzantine Fault Tolerance for Agents)

> 고전적 분산 시스템 BFT가 확률적 LLM을 만난다. 2025~2026년에 세 가지 연구 방향이 등장했다. **CP-WBFT**(arXiv:2511.10400)는 신뢰도 프로브(confidence probe)로 각 투표에 가중치를 둔다. **DecentLLMs**(arXiv:2507.14928)는 병렬 워커 제안과 기하 중앙값(geometric-median) 집계로 리더 없이(leaderless) 간다. **WBFT**(arXiv:2505.05103)는 가중 투표를 계층 구조 클러스터링(Hierarchical Structure Clustering)과 결합해 Core 노드와 Edge 노드를 나눈다. "Can AI Agents Agree?"(arXiv:2603.01213)의 정직한 실증 결과는, 오늘날 스칼라 합의(scalar agreement)조차 취약하다는 것이다. 단 하나의 기만적(deceptive) 에이전트(agent)가 Mixture-of-Agents를 무너뜨릴 수 있다. BFT는 필요하지만 충분하지는 않다. 이 레슨은 최소 BFT 프로토콜을 만들고, 에이전트 고유의 세 가지 공격(비잔틴 거짓말, 아첨적 동조, 상관 오류 단일 문화)을 주입하고, 각 합의 변형이 어떻게 대처하는지 측정한다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75분

## 문제 (Problem)

각각 답을 내는 N개의 LLM 에이전트가 있다. 그들은 의견이 다르다. 두 에이전트가 상관되어 있기(같은 베이스 모델, 같은 학습(training) 데이터, 같은 실패 모드) 때문에 다수결이 틀린 답을 고른다. 세 번째 에이전트는 우연히 새로운 방식으로 틀린다. 그래서 다수는 거짓 다수다.

이제 기만적 에이전트를 더하자. 고의로 거짓말을 한다. 또는 아첨적(sycophantic) 에이전트: 마지막으로 말한 쪽에 동조한다. 고전적 BFT에서는 비잔틴(Byzantine) 노드가 일부 `f < n/3`이며 임의로 행동한다고 가정한다. 2026년 현실은, LLM 노드가 정직할 때조차 확률적이고 모델 간에 상관되어 있으며 서로의 출력에 영향을 받는다는 것이다. 이들을 독립적인 베르누이(Bernoulli) 투표자로 다룰 수 없다.

고전적 BFT(PBFT, 1999)는 틀리지 않았다. 불완전할 뿐이다. 임의의 비트 뒤집기(bit-flipping)는 다룬다. "정직한 세 에이전트가 학습 데이터를 공유하기 때문에 같은 환각(hallucination)을 공유한다"는 다루지 못한다. 이 레슨은 PBFT의 토대에서 시작해 2025~2026년의 세 가지 적응을 그 위에 쌓는다.

## 개념 (Concept)

### 고전적 BFT가 주는 것

실용적 비잔틴 결함 허용(Practical Byzantine Fault Tolerance, Castro & Liskov, OSDI 1999)은 `f < n/3`개의 비잔틴 노드를 허용한다. 프로토콜은 세 단계(pre-prepare, prepare, commit)와 두 프리미티브(서명된 메시지, 정족수 증명서(quorum certificate))를 가진다. `n >= 3f + 1`개의 정직하거나 악의적인 노드 사이에서 단일 값에 대한 합의를 이룬다.

보장은 강력하지만 다음을 가정한다.

1. **독립적 결함.** 비잔틴들이 협력하지 않는다.
2. **정직한 노드는 진정 정직하다.** 정직한 출력의 정확성은 문제가 아니다. 프로토콜은 불일치만 정렬한다.
3. **질문에 정답(ground-truth)이 있다.** 틀린 사실에 대한 합의도 여전히 합의다.

LLM 에이전트는 이 셋을 모두 위반한다. 같은 베이스 모델을 돌리는 두 에이전트는 결함을 공유한다. "정직한" LLM도 여전히 환각한다. 모호한 질문에서는 "진실"을 에이전트들이 결정한다. 외부 오라클(oracle)이 없다.

### LLM 고유의 세 가지 공격

**비잔틴 거짓말(Byzantine lie).** 한 에이전트가 고의로 틀린 답을 출력한다. `f < n/3`이면 고전적 BFT가 이를 다룬다.

**아첨적 동조(Sycophantic conformity).** 한 에이전트가 투표 전에 다른 답들을 읽고 마지막으로 말한 쪽에 정렬한다. 악의적이지는 않지만 가장 큰 목소리와 상관된다. 그 에이전트가 모든 서명 검사를 통과하기 때문에 고전적 BFT는 이를 막지 못한다.

**상관 오류 단일 문화(Correlated-error monoculture).** 세 에이전트가 베이스 모델을 공유한다. 같은 틀린 답을 환각한다. 다수가 틀렸다. 세 에이전트 모두 "정직하게" 동의하기 때문에 고전적 BFT는 도움이 안 된다.

### 2025~2026년의 대응

**CP-WBFT**(arXiv:2511.10400): 신뢰도 프로브 가중 BFT(Confidence-Probed Weighted BFT). 각 투표자가 자신의 답에 신뢰도 프로브(자기 보고 확률, 또는 별도 보정(calibration) 모델의 예측)를 붙인다. 투표 가중치가 신뢰도에 비례한다. 완전 그래프(complete graph)에서 +85.71% BFT 개선을 보고했다. 완화 대상: 아첨적 동조(동조하는 에이전트는 자청한 입장에 낮은 신뢰도를 보이는 경향이 있다).

**DecentLLMs**(arXiv:2507.14928): 리더 없음. 워커 에이전트가 병렬로 제안하고, 평가자(evaluator) 에이전트가 제안에 점수를 매기며, 최종 답은 점수 매겨진 입장들의 기하 중앙값이다. `f < n/2`일 때 견고하다. 완화 대상: 비잔틴 거짓말과 상관 오류(기하 중앙값은 이상치에 견고하며 모델 편향된 평균이 아니라 조밀한 클러스터 쪽으로 당겨진다).

**WBFT**(arXiv:2505.05103): 계층 구조 클러스터링을 동반한 가중 BFT. 투표 가중치는 응답 품질에 이력에서 학습한 신뢰 점수(trust score)를 더해 할당된다. 에이전트를 Core와 Edge로 클러스터링한다. Core 에이전트가 먼저 합의를 이루어야 하고, Edge 에이전트가 따른다. 완화 대상: 확장성(Core 합의가 작고 빠르다)과 단일 문화에 대한 부분적 완화(Core를 다양성을 위해 고를 수 있다).

### 실증: "Can AI Agents Agree?" (arXiv:2603.01213)

이 논문은 여러 프런티어(frontier) 모델에 걸쳐 스칼라 합의(LLM 에이전트가 단일 숫자 값에 동의하는 것)를 측정한다. 발견은 불편하다.

- 적대자가 없을 때조차, LLM 에이전트는 많은 벤치마크(benchmark)에서 30%를 넘는 비율로 스칼라 질문에서 의견이 갈린다.
- 기만적 페르소나(deceptive persona)를 채택한 단 하나의 에이전트가 Mixture-of-Agents 합의를 정직한 베이스라인(baseline)에서 40퍼센트포인트 이상 끌어낼 수 있다.
- 불일치율은 모델 다양성과 상관된다. 이질적 앙상블(ensemble)은 동질적 앙상블보다 더 많이 갈리지만(좋음: 상관되지 않은 오류) 또한 더 느리게 드리프트(drift)한다(나쁨: 합의까지의 시간이 더 길다).

요점: BFT는 출력을 정렬할 기계장치를 주지만, 정렬된 출력이 옳은지는 말해주지 않는다. 검증(Phase 16 · 08 역할 특화), 다양성(Phase 16 · 15 토론 변형), 평가자 에이전트(Phase 16 · 24 벤치마크)와 결합하라.

### 핵심 프로토콜, 군더더기를 뺀 형태

LLM 에이전트를 위한 최소 BFT 라운드:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

의미론적 클러스터링(semantic clustering) 단계가 LLM 고유의 묘수다. "이 연구는 4.2%를 보고한다"와 "4.2% 향상"이라는 두 답은 같은 클러스터다. 순진한 문자열 동등성 검사는 이를 놓친다. 프로덕션(production)에서는 저렴한 임베딩(embedding) 모델이나 명시적 정규화(canonicalization)를 쓰라.

### 임계값 튜닝

`threshold` 파라미터(parameter)는 언제 수락하고 언제 재시도할지 결정한다. 너무 낮으면 약한 다수를 수락한다. 너무 높으면 아무것도 수락하지 못한다. 실증 범위: `n=5-7` 에이전트에서 0.5~0.67, 더 작은 `n`에서는 더 높게. 임계값 아래에서는 사람이나 다른 에이전트 앙상블로 에스컬레이션(escalation)한다.

### 합의가 도움이 안 되는 곳

- **모호한 질문.** 질문에 정답이 없으면 합의는 의견이다. 그렇게 부르라.
- **복합 질문.** "코드를 쓰고 설명하라". 두 개의 답이다. 각각 독립적으로 투표하라.
- **적대적 다중 라운드.** 에이전트가 이전 라운드를 관찰하고 모방할 수 있으면(Du 2023 토론), 진실과 무관하게 서로 동의하기 시작한다. 라운드를 한정하라(보통 2~3).

```figure
swarm-consensus-wave
```

## 직접 만들기 (Build It)

`code/main.py`는 다음을 구현한다.

- `AgentVoter`: (answer, confidence)를 가진 스크립트된 정책.
- `MajorityVote`: 고전적 최다 득표(plurality).
- `CPWBFT`: 의미론적 클러스터링을 동반한 신뢰도 가중 투표.
- `DecentLLMs`: 점수 매겨진 제안에 대한 기하 중앙값 집계.
- `Scenario`: 세 가지 공격 패턴 하에서 각 집계기를 실행한다.

구현된 공격 패턴:

1. `byzantine`: 한 에이전트가 높은 신뢰도로 거짓말한다.
2. `sycophancy`: 한 에이전트가 처음 본 답을, 일치하는 신뢰도로 복사한다.
3. `monoculture`: 세 에이전트가 보통 신뢰도로 틀린 답(상관 오류)을 공유한다.

실행:

```
python3 code/main.py
```

예상 출력: (attack, aggregator) -> final answer의 표로, 정답이 강조된다. 최다 득표는 단일 문화 경우에서 실패한다. CPWBFT의 신뢰도 가중은 아첨을 완화한다. DecentLLMs의 기하 중앙값은 단일 문화가 인구의 절반 미만일 때 정직한 클러스터 쪽으로 당긴다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-consensus-designer.md`는 멀티 에이전트 앙상블을 위한 합의 프로토콜을 설계한다. 클러스터링 방법, 가중치, 임계값, 그리고 임계값 미만 라운드에 대한 에스컬레이션 정책.

## 산출물 (Ship It)

어떤 합의 메커니즘이든 출시하기 전에:

- 위의 **세 패턴으로 최소한 공격 테스트하라.** 프로토콜은 조용히가 아니라 예측 가능하게 실패해야 한다.
- **모든 소수 클러스터를 출처와 함께 기록하라.** 소수 클러스터는 상관 오류에 대한 조기 경보 시스템이다.
- **한정된 라운드를 강제하라.** "동의할 때까지 계속 토론"은 금지다. 그것은 아첨을 보상한다.
- **합의와 정확성을 분리하라.** 합의 출력은 검증자(verifier)로 간다. 검증자는 앙상블과 독립적이다.
- **합의율을 모니터링하라.** 급격한 상승은 동조 편향을, 급격한 하락은 모델 드리프트를 뜻한다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 최다 득표가 단일 문화 공격에서 실패하지만, 단일 문화 신뢰도가 0.7 미만일 때 CPWBFT가 이를 부분적으로 완화하는지 확인하라.
2. 네 번째 공격 패턴을 추가하라. **무응답(silent abstention)**: 한 에이전트가 답하기를 거부한다("모르겠다"). 각 집계기는 무응답을 어떻게 다루어야 하는가? 당신의 선택을 구현하라.
3. 의미론적 클러스터링을 문자열 정규화에서 임베딩 유사도로 바꿔라(어떤 오픈소스 임베딩 모델이든 사용). 아첨 공격에 무슨 일이 일어나는가?
4. CP-WBFT(arXiv:2511.10400)를 읽어라. 신뢰도 프로브 보정 단계(별도 보정 모델이 각 에이전트의 자기 보고 신뢰도를 검사)를 구현하라. 단일 문화 시나리오에서의 정확도 향상을 측정하라.
5. "Can AI Agents Agree?"(arXiv:2603.01213)를 읽어라. 단순화된 스칼라 합의 실험을 재현하라. 세 에이전트, 하나의 스칼라 질문, 기만적 페르소나 프롬프트(prompt). CPWBFT나 DecentLLMs가 이를 잡아내는가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| BFT | "비잔틴 결함 허용" | `f < n/3`개의 임의 결함에 대한 합의를 위한 Castro-Liskov 1999 프로토콜. |
| 비잔틴(Byzantine) | "임의의 나쁜 행동" | 거짓말, 메시지 누락, 조용한 실패 등 안전한 충돌만 빼고 무엇이든 할 수 있는 노드. |
| 신뢰도 프로브(Confidence probe) | "얼마나 확신하나?" | 투표에 붙는 자기 보고 또는 보정기 예측 확률. |
| 의미론적 클러스터링(Semantic clustering) | "같은 답, 다른 단어" | 투표를 세기 전에 동등한 답을 묶는 것. |
| 기하 중앙값(Geometric median) | "견고한 중심" | 표본 점들까지의 거리 합을 최소화하는 점. 평균과 달리 이상치에 견고하다. |
| 단일 문화(Monoculture) | "같은 모델, 같은 실패" | 에이전트가 학습 데이터나 베이스 모델을 공유할 때의 상관 오류. |
| 아첨적 동조(Sycophantic conformity) | "큰 목소리에 동의" | 에이전트의 투표가 먼저/가장 크게 말한 쪽으로 편향됨. |
| Core/Edge | "계층적 BFT" | WBFT 분할: 작은 Core 합의 먼저, Edge 노드가 따름. 지연 시간(latency)을 한정한다. |

## 더 읽을거리 (Further Reading)

- [Castro & Liskov: Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) 토대
- [CP-WBFT: Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) 신뢰도에 의한 투표 가중
- [DecentLLMs: leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) 기하 중앙값 집계
- [WBFT: Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) 한정된 지연 시간을 위한 Core/Edge 분할
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213): 스칼라 합의의 취약성과 기만적 페르소나 공격
