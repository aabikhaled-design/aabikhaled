# Reflexion: 언어적 강화 학습(Verbal Reinforcement Learning)

> 그래디언트 기반 강화 학습(reinforcement learning)은 하나의 실패 양상을 고치는 데 수천 번의 시행과 GPU 클러스터가 필요하다. Reflexion(Shinn et al., NeurIPS 2023)은 이를 자연어로 한다: 각 실패한 시행 이후, 에이전트는 성찰(reflection)을 쓰고, 그것을 일화적 메모리(episodic memory)에 저장하며, 다음 시행을 그 메모리에 조건화한다. 이것이 Letta의 수면 시간 연산(sleep-time compute), Claude Code의 CLAUDE.md 학습 내용, pro-workflow의 learn-rule 뒤에 있는 패턴이다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60분

## 학습 목표 (Learning Objectives)

- Reflexion의 세 구성 요소(행위자 Actor, 평가자 Evaluator, 자기 성찰자 Self-Reflector)와 일화적 메모리의 역할을 명명하기.
- 이진 평가자, 성찰 버퍼, 새 재시도를 갖춘 stdlib 기반 Reflexion 루프를 구현하기.
- 주어진 작업에 대해 스칼라, 휴리스틱, 자기 평가 피드백 소스 중에서 선택하기.
- 그래디언트 기반 강화 학습이 고치는 데 수천 번의 시행이 필요할 오류를 언어적 강화가 왜 잡아내는지 설명하기.

## 문제 (The Problem)

에이전트가 작업에 실패한다. 표준 강화 학습에서는 수천 번의 시행을 더 돌리고, 그래디언트(gradient)를 계산하고, 가중치(weight)를 갱신할 것이다. 비싸고 느리며 대부분의 프로덕션(production) 에이전트는 모든 실패마다 학습 예산이 있지 않다.

Reflexion(Shinn et al., arXiv:2303.11366)은 다른 질문을 던진다: 에이전트가 그저 왜 실패했는지 생각해보고 그 생각을 프롬프트(prompt)에 담아 다시 시도하면 어떨까? 가중치 갱신 없음. 그래디언트 없음. 시행 사이에 저장된 자연어뿐.

결과: ALFWorld에서 ReAct와 다른 비파인튜닝 베이스라인(baseline)을 이긴다. HotpotQA에서 ReAct보다 개선된다. 코드 생성(HumanEval/MBPP)에서는 당시 최첨단(state of the art)을 세운다. 단 한 번의 그래디언트 스텝도 없이.

## 개념 (The Concept)

### 세 구성 요소

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

여기에 자료 구조 하나가 더해진다:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

한 번의 시행이 행위자를 돌린다. 평가자가 채점한다. 점수가 낮으면 자기 성찰자가 성찰을 생성한다("나는 질문을 Y에 관한 것인데 X에 관한 것으로 잘못 읽어서 틀린 도구를 골랐다"). 성찰은 일화적 메모리로 들어간다. 다음 시행은 새로 시작하지만 그 성찰을 본다.

### 세 가지 평가자 유형

1. **스칼라(Scalar)**: 외부 이진 신호. ALFWorld는 성공하거나 실패한다. HumanEval 테스트는 통과하거나 실패한다. 가장 단순하고, 신호가 가장 강하다.
2. **휴리스틱(Heuristic)**: 미리 정의된 실패 시그니처. "에이전트가 같은 행동을 연속 두 번 했으면 막힘으로 표시." "트래젝토리가 50단계를 넘으면 비효율로 표시."
3. **자기 평가(Self-evaluated)**(LLM이 자신의 트래젝토리를 채점한다. 정답(ground truth)이 없을 때 필요하다. 신호가 약하다. 도구에 근거한 검증(Lesson 05) CRITIC)과 잘 짝지어진다.

2026년 기본값은 혼합이다: 가능하면 스칼라, 없으면 자기 평가, 안전 레일로서 휴리스틱.

### 왜 이것이 일반화되는가

Reflexion은 새로운 알고리즘이라기보다 이름 붙은 패턴이다. 거의 모든 프로덕션 "자가 치유(self-healing)" 에이전트가 어떤 변형을 돌린다:

- Letta의 수면 시간 연산(Lesson 08): 별도의 에이전트가 과거 대화를 성찰하고 메모리 블록(memory block)에 쓴다.
- Claude Code의 `CLAUDE.md` / "메모리 저장" 패턴: 성찰이 학습 내용으로 포착되어 미래 세션 앞에 붙는다.
- pro-workflow의 `/learn-rule` 명령: 교정이 명시적 규칙으로 포착된다.
- LangGraph의 성찰 노드: 출력을 채점하고 필요하면 정제로 라우팅하는 노드.

모두 같은 통찰에서 파생된다: 자연어는 "실패에서 배운 것"을 실행 사이에 운반하기에 충분히 풍부한 매체다.

### 언제 작동하고 언제 안 하는가

Reflexion이 작동할 때:

- 명확한 실패 신호가 있을 때(테스트 실패, 도구 에러, 틀린 답).
- 작업 종류가 재현 가능할 때(같은 유형의 질문을 다시 물을 수 있음).
- 성찰이 트래젝토리를 개선할 여지가 있을 때(충분한 행동 예산).

Reflexion이 도움이 안 될 때:

- 에이전트가 이미 첫 시도에 성공할 때.
- 실패가 외부적일 때(네트워크 다운, 도구 고장): "네트워크가 다운됐다"에 대한 성찰은 미래 실행에 도움이 안 된다.
- 성찰이 미신으로 변할 때: 일회성의 불안정한 실행에 대한 서사를 저장하는 것.

2026년 함정: 메모리 부패(memory rot). 성찰이 쌓이고 일부는 낡거나 틀렸으며 일화적 버퍼가 커지면서 재실행이 느려진다. 완화책: 주기적 압축(compaction)(Lesson 06), 성찰에 대한 TTL, 또는 별도의 수면 시간 정리 에이전트(Letta).

```figure
react-trace
```

## 직접 만들기 (Build It)

`code/main.py`는 장난감 퍼즐에 Reflexion을 구현한다: 목표값으로 합산되는 3원소 리스트를 만들기. 행위자는 후보 리스트를 방출하고, 평가자는 합을 확인하며, 자기 성찰자는 무엇이 잘못됐는지에 대한 한 줄을 쓴다. 성찰은 다음 시행을 위해 일화적 메모리로 들어간다.

구성 요소:

- `Actor`: 성찰을 보면 개선되는 스크립트된 정책.
- `Evaluator.binary()`: 목표 합에 대한 통과/실패.
- `SelfReflector`: 실패에 대한 한 줄 진단을 생성.
- `EpisodicMemory`: TTL 의미론을 가진 한도가 있는 리스트.

실행:

```
python3 code/main.py
```

트레이스는 세 번의 시행을 보여준다. 시행 1은 실패하고, 성찰이 저장되며, 시행 2는 성찰을 보고 개선되지만 여전히 실패하고, 시행 3은 성공한다. 베이스라인 실행(성찰 없음)과 비교하라. 그것은 시행 1의 답에 막혀 머무른다.

## 라이브러리로 써보기 (Use It)

LangGraph는 성찰을 노드 패턴으로 제공한다. Claude Code의 `/memory` 명령과 pro-workflow의 `/learn-rule`은 일화적 버퍼를 마크다운(markdown) 파일로 외부화한다. Letta의 수면 시간 연산은 자기 성찰자를 유휴 시간에 돌려 주 에이전트가 지연(latency)에 묶이지 않게 한다. OpenAI Agents SDK는 Reflexion을 직접 제공하지 않는다. 점수로 트래젝토리를 거부하는 커스텀 Guardrail과 실행 간에 살아남는 메모리 `Session`으로 그것을 만든다.

## 산출물 (Ship It)

`outputs/skill-reflexion-buffer.md`는 성찰 포착, TTL, 중복 제거를 갖춘 일화적 버퍼를 생성하고 유지한다. 작업 종류와 실패가 주어지면, 다음 시행에 실제로 도움이 되는(일반적인 "더 조심하라"가 아닌) 성찰을 방출한다.

## 연습 문제 (Exercises)

1. 이진 평가자에서 거리 척도(목표에서 얼마나 먼지)를 반환하는 스칼라 평가자로 전환하라. 더 빨리 수렴하는가?
2. 성찰에 10 시행의 TTL을 추가하라. 그 시점 이후 오래된 성찰은 해가 되는가 도움이 되는가?
3. 휴리스틱 평가자를 구현하라: 같은 행동이 반복되면 시행을 막힘으로 표시. 이것이 자기 성찰자와 어떻게 상호작용하는가?
4. 성찰을 무시하는 적대적 행위자로 Reflexion을 돌려라. 행위자가 성찰을 알아채도록 강제하는 최소한의 성찰 프롬프트 엔지니어링은 무엇인가?
5. ALFWorld에 관한 Reflexion 논문의 4절을 읽어라. 130% 성공률 개선을 개념적으로 재현하라: 바닐라 ReAct 대비 핵심 차이는 무엇인가?

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Reflexion | "자기 교정" | Shinn et al. 2023: 행위자, 평가자, 자기 성찰자와 일화적 메모리 |
| 언어적 강화(Verbal reinforcement) | "그래디언트 없는 학습" | 다음 시행의 프롬프트 앞에 붙는 자연어 성찰 |
| 일화적 메모리(Episodic memory) | "작업별 성찰" | 한 작업 종류에 대한 이전 성찰의 한도 있는 버퍼 |
| 스칼라 평가자(Scalar evaluator) | "이진 성공 신호" | 정답에서 나온 통과/실패 또는 수치 점수 |
| 휴리스틱 평가자(Heuristic evaluator) | "패턴 기반 탐지기" | 미리 정의된 실패 시그니처(예: 막힘 루프, 너무 많은 단계) |
| 자기 평가자(Self-evaluator) | "자신의 트레이스에 대한 LLM 심판" | 정답이 없을 때의 신호 약한 대체책: 도구 근거 검증과 짝지을 것 |
| 메모리 부패(Memory rot) | "낡은 성찰" | 일화적 버퍼가 낡은 항목으로 가득 참; 압축/TTL로 해결 |
| 수면 시간 성찰(Sleep-time reflection) | "비동기 자기 성찰" | 주 에이전트가 빠르게 유지되도록 핫 패스 밖에서 자기 성찰자 실행 |

## 더 읽을거리 (Further Reading)

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366): 표준 논문
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute): 프로덕션에서의 비동기 성찰
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): 일화적 버퍼를 컨텍스트의 일부로 관리하기
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 성찰 노드 패턴
