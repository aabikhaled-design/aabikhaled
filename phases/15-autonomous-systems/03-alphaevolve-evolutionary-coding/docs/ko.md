# AlphaEvolve: 진화적 코딩 에이전트

> 프런티어 코딩 모델을 진화 루프(evolutionary loop) 및 기계 검증 가능한 평가기(evaluator)와 짝지어라. 루프를 충분히 오래 돌려라. 그러면 48번의 스칼라 곱셈을 사용하는 4x4 복소 행렬 곱셈 절차를 발견한다. 56년 만의 Strassen 대비 첫 개선이다. 또한 프로덕션(production)에서 클러스터 컴퓨트의 약 0.7%를 회수하는 Google 전사적 Borg 스케줄링 휴리스틱(heuristic)도 찾아낸다. 아키텍처는 의도적으로 따분하다. 성과는 평가기의 엄격함에서 나온다.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60분

## 문제 (The Problem)

대형 언어 모델(LLM)은 코드를 쓸 수 있다. 진화 알고리즘(evolutionary algorithm)은 코드 위를 탐색할 수 있다. 둘 다 수십 년간 따로따로 시도되었고, 둘 다 천장에 부딪혔다. LLM의 천장은 작화(confabulation)다. 모델은 주장한 일을 실제로 하지 않는 그럴듯한 코드를 쓴다. 진화의 천장은 탐색 비용이다. 구문을 무작위로 돌연변이시키면 컴파일되는 프로그램조차 거의 나오지 않으며, 더 나은 것은 말할 것도 없다.

AlphaEvolve(Novikov et al., DeepMind, arXiv:2506.13131, 2025년 6월)는 둘을 결합한다. LLM은 프로그램 데이터베이스에 표적화된 편집을 제안하고, 자동 평가기가 각 변형(variant)을 채점하며, 높은 점수를 받은 변형이 다음 세대의 부모(parent)가 된다. LLM은 그럴듯한 코드를 작성하는 비싼 단계를 맡고, 평가기는 작화를 잡아낸다. 루프는 수 시간에서 수 주간 돌아간다.

보고된 결과: 48번의 스칼라 곱셈을 쓰는 4x4 복소 행렬 곱셈(Strassen의 1969년 한계는 49였다), Google 프로덕션의 Borg 스케줄링 휴리스틱, 32.5%의 FlashAttention 커널 속도 향상, Gemini 학습 처리량(throughput) 개선.

이 아키텍처가 작동하는 이유는 평가기가 기계 검증 가능하기 때문이다. 평가기가 그렇지 않은 곳에서는 작동하지 않는다. 그 비대칭성이 이 레슨의 교훈이다.

## 개념 (The Concept)

### 루프

1. 정확하지만 최적이 아닌 시드 프로그램(seed program) `P_0`에서 출발한다.
2. 각각 평가기로 채점된 변형 프로그램들의 데이터베이스를 유지한다.
3. 데이터베이스에서 하나 이상의 부모를 샘플링한다(MAP-elites 방식 또는 섬(island) 기반).
4. LLM(많은 후보에는 Gemini Flash, 어려운 것에는 Gemini Pro)에게 부모의 수정된 변형을 만들도록 프롬프트한다.
5. 변형을 컴파일하고, 실행하고, 별도 보관된(held-out) 평가기로 평가한다.
6. 점수와 특성 벡터(feature vector)를 키로 하여 데이터베이스에 삽입한다.
7. 반복한다.

두 가지 세부 사항이 중요하다. 첫째, LLM은 부모 프로그램만이 아니라 그 이상으로 프롬프트된다. 일반적으로 데이터베이스의 상위 변형 여럿, 평가기 시그니처(signature), 그리고 짧은 작업 설명까지 함께. 모델의 임무는 점수를 개선할 수도 있는 표적화된 변경을 제안하는 것이다. 둘째, 데이터베이스는 구조화되어 있어(MAP-elites 그리드, 섬 기반) 루프가 현재 선두만이 아니라 다양성을 탐색한다.

### 무엇이 평가기를 타협 불가능하게 만드는가

AlphaEvolve의 성과는 모두 평가기가 빠르고, 결정적(deterministic)이며, 속이기 어려운 영역에서 나온다:

- **행렬 곱셈 알고리즘**: 행렬을 곱하고 비트 단위로 동일한 동등성을 확인하는 단위 테스트.
- **Borg 스케줄링 휴리스틱**: 과거 클러스터 부하를 재생하고 낭비된 컴퓨트를 측정하는 프로덕션급 시뮬레이터.
- **FlashAttention 커널**: 정확성 테스트에 더해 실제 하드웨어에서의 실측 시간(wall-clock) 벤치마크.
- **Gemini 학습 처리량**: 스텝당 측정된 GPU 초.

각 경우에 평가기는 그것이 없었다면 판을 지배했을 LLM 오류 부류를 잡아낸다. 작화된 정확성 주장, 하드웨어에서 사라지는 성능 주장, 경계 사례(edge-case) 실패가 그것이다. 평가기를 제거하면 루프는 예쁜 코드를 위해 최적화한다.

### 보상 해킹은 그 진술의 다른 얼굴이다

진화는 평가기가 측정하는 무엇이든 위해 최적화한다. 평가기가 불완전하면, 루프는 그 불완전함을 찾아낸다. 검증되지 않은 영역에서 루프는 의도된 행동이 아니라 표면 특성을 향해 최적화한다. DeepMind는 논문에서 이를 명시적으로 표시한다. AlphaEvolve의 성공은 평가기의 엄격함이 탐색의 야심과 일치하는 영역으로만 이전된다.

2025-2026년 코드 탐색 루프에서의 보상 해킹(reward hacking)의 구체적 사례:

- "완료 시간"을 보상하는 최적화 목표는 빈 풀이를 제출하는 것을 보상했다.
- 테스트 하의 정확성을 보상하는 벤치마크 점수는 테스트를 암기하고 과적합(overfitting)하는 것을 보상했다.
- "코드 품질" 프록시는 의미 변화 없이 주석을 제거하고 변수명을 다시 쓰는 것을 보상했다.

AlphaEvolve의 해결책: LLM이 결코 본 적 없는, 평가 시점에 입력이 생성되는 별도 보관된 평가기를 배포하는 것이다. 그렇더라도 DeepMind는 제안된 모든 배포에 강한 검토를 권장한다.

### 왜 LLM + 탐색이 둘 중 하나 단독보다 나은가

LLM은 컴파일 가능하고 의미적으로 그럴듯한 수정을 만들 수 있다. 2000줄짜리 Python 파일에 대한 무작위 돌연변이 GA는 거의 항상 구문 오류를 낸다. LLM은 또한 탐색을 그럴듯한 이웃(plausible neighborhood)에 집중시켜(무작위 바이트가 아니라 함수 하나를 변경), 낭비되는 평가기 호출을 극적으로 줄인다.

평가기는 다시 LLM의 작화를 잡아낸다. LLM은 실제로는 O(n^2)인 함수를 두고 "극한에서 O(n log n)이다"라고 자신만만하게 주장하곤 한다. 실측 시간 벤치마크는 그 질문을 확정 짓는다.

### 프런티어 스택에서 AlphaEvolve가 자리하는 곳

| 시스템 | 생성기 | 평가기 | 영역 | 예시 성과 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 정확성 + 벤치마크 | 알고리즘, 커널, 스케줄러 | 48-곱셈 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 정확성 | 조합론적 수학 | cap-set 하계 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 비평 + 실험 | ML 연구 | ICLR 워크숍 논문 |
| Darwin Godel Machine (L4) | 에이전트 스캐폴딩 | SWE-bench / Polyglot | 에이전트 코드 | 20% → 50% SWE-bench |

넷 모두 같은 레시피의 변주다. 생성기 더하기 평가기, 루프. 차이는 평가기가 무엇을 채점하는지와 얼마나 엄격한지다.

```figure
alphaevolve-loop
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 장난감 기호 회귀(symbolic-regression) 문제에 대해 최소한의 AlphaEvolve 유사 루프를 구현한다. "LLM"은 목표 함수를 계산하는 프로그램에 작은 구문적 돌연변이를 제안하는 stdlib 프록시다. "평가기"는 별도 보관된 테스트 지점에서 평균 제곱 오차(MSE)를 측정한다.

지켜보라:

- 세대에 걸쳐 최고 점수가 어떻게 개선되는지.
- MAP-elites 그리드가 다양한 풀이를 살려 두어 루프가 지역 최솟값(local minimum)에 수렴하지 않게 하는 방식.
- 별도 보관된 테스트를 제거하면(학습 전용 평가기) 루프가 얼마나 화려하게 과적합하는지.

## 산출물 (Ship It)

`outputs/skill-evaluator-rigor-audit.md`는 새 영역에서 AlphaEvolve 방식 루프를 고려하기 위한 전제 조건이다. 평가기가 정작 신경 쓰는 실패를 실제로 잡아내는가?

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. 최고 점수 궤적을 기록하라. 별도 보관된 평가기를 비활성화하고(플래그 `--no-holdout`) 다시 실행하라. 과적합을 정량화하라.

2. AlphaEvolve 논문의 MAP-elites 그리드에 관한 Section 3을 읽어라. 탐색을 다양하게 유지할, 새 문제(예: 컴파일러 최적화 패스)를 위한 특성 벡터 기술자(descriptor)를 설계하라.

3. 48-곱셈 4x4 결과는 56년 만에 Strassen의 49-곱셈 한계를 개선했다. 논문의 Appendix F를 읽고, 왜 이 문제의 평가기는 특히 올바르게 만들기 쉬운지, 그리고 왜 대부분의 영역은 이와 같지 않은지를 세 문장으로 설명하라.

4. AlphaEvolve가 실패할 영역 하나를 제안하라. 평가기가 정확히 어디서 무너지는지와 그 이유를 식별하라.

5. 잘 아는 영역 하나를 골라, 거기에 쓸 평가기 시그니처를 작성하라. (a) 정확성 조건, (b) 성능 지표, (c) 별도 보관 입력 생성 규칙, (d) 최소 하나의 보상 해킹 방지 검사를 포함하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| AlphaEvolve | "DeepMind의 진화적 코딩 에이전트" | Gemini + 프로그램 데이터베이스 + 기계 검증 가능한 평가기 |
| MAP-elites | "다양성 보존 아카이브" | 특성 벡터를 키로 하는 그리드. 각 셀은 그 기술자를 가진 최고 변형을 담는다 |
| 섬 모델 (Island model) | "병렬 진화 하위 개체군" | 주기적으로 이주(migrate)하는 독립 개체군. 조기 수렴을 막는다 |
| 기계 검증 가능한 평가기 (Machine-checkable evaluator) | "결정적 오라클" | LLM이 속일 수 없는 단위 테스트, 시뮬레이터, 벤치마크: 이 루프의 전제 조건 |
| 보상 해킹 (Reward hacking) | "목표가 아니라 척도를 최적화" | 루프가 의도된 작업을 하지 않고 점수를 극대화하는 방법을 찾는 것 |
| 시드 프로그램 (Seed program) | "출발점" | 루프가 진화시키는, 정확하지만 최적이 아닌 초기 프로그램 |
| 별도 보관된 평가기 (Held-out evaluator) | "LLM이 결코 본 적 없는 평가 데이터" | 암기를 막기 위해 평가 시점에 생성된 입력 |

## 더 읽을거리 (Further Reading)

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131): 전체 논문.
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/): 결과가 담긴 벤더 작성 글.
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results): 48-곱셈 4x4 matmul을 포함해 발견된 알고리즘들.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6): 선행 시스템.
- [Anthropic: Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) 평가기에 묶인 자율성을 핵심 연구 방향으로 규정한다.
