# 자동화된 정렬 연구 (Anthropic AAR)

> Anthropic은 Claude Opus 4.6 자율 정렬 연구자(Autonomous Alignment Researcher) 팀들을 독립적인 샌드박스(sandbox)에서 병렬로 실행했으며, 이들은 어떤 샌드박스 바깥에 로그가 존재하는 공유 포럼(forum)을 통해 협력한다(따라서 에이전트는 자기 자신의 기록을 삭제할 수 없다). 약-대-강(weak-to-strong) 학습 문제에서, AAR은 인간 연구자를 능가했다. Anthropic 자신의 요약은 규정된 워크플로(prescribed workflow)가 종종 AAR의 유연성을 제약하고 성능을 떨어뜨린다는 점을 표시한다. 정렬 연구를 자동화하는 것은, 타임라인을 RSP가 탐지하려는 바로 그 오정렬(misalignment) 위험 쪽으로 압축하는 단계다.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60분

## 문제 (The Problem)

정렬 연구(alignment research)는 인간 연구자의 시간을 많이 잡아먹는다. 확장 가능한 감독(scalable oversight), 보상 명세(reward specification), 또는 약-대-강 학습 같은 문제는 반복(iteration)당 수 주가 걸리는 실험을 요구한다. 프런티어 능력이 발전하면서, 정렬 작업량은 자격 있는 연구자의 공급보다 빠르게 늘어난다.

자동화된 정렬 연구(Automated Alignment Research, AAR)는, 능력이 정렬을 앞지르고 있는 바로 그 프런티어 모델이 그 간극을 좁히는 데 기여할 수 있는지 묻는다. AAR이 수행한 약-대-강 학습 연구에 대한 Anthropic의 2026년 보고서(alignment.anthropic.com/2026/automated-w2s-researcher/)는 이 부류의 배포된 시스템에서 나온 최초의 공개 결과 중 하나다.

그 결과는 진정으로 긍정적이다. AAR은 인간 베이스라인보다 연구 문제를 더 잘 풀었다. 그 결과는 또한 이 단계가 발전시켜 온 모든 우려를 하나의 시스템에 집약한다. 정렬 연구가 자동화될 수 있다면, 안전장치를 훼손하는 부분도 자동화될 수 있다. AI R&D 능력에 대한 RSP의 임계값은 이 루프를 염두에 두고 작성되었다.

## 개념 (The Concept)

### 아키텍처 (공개된 설명대로)

- **병렬 에이전트.** 여러 Claude Opus 4.6 인스턴스가 각각 독립적인 샌드박스에서 AAR로 실행된다. 샌드박스는 파일 시스템, 네트워크, 프로세스 표면을 제한한다.
- **공유 포럼.** 발견은, 저장 백엔드가 모든 샌드박스 바깥에 자리한 공유 포럼에 게시된다. 에이전트는 포럼을 읽을 수 있지만, 자기 자신의 샌드박스 안에서 포럼 기록을 삭제하거나 편집할 수 없다.
- **작업 할당.** 인간이 설계한 작업 큐(task queue)가 연구 문제를 AAR에 배정한다. AAR은 분해(decomposition)를 제안하고, 실험을 실행하고, 발견을 작성한다.
- **약-대-강 학습 초점.** 공개된 연구는 약-대-강 학습을 다루었다. 더 약한 감독자(supervisor) 모델을 사용해 더 강한 학생(student) 모델을 학습시키는 것. AAR은 이 작업에서 인간 연구자를 능가했다.

### 샌드박스 바깥 로그가 중요한 이유

로그 무결성(log-integrity) 속성은 통제된 실험과 잠재적으로 훼손된 실험을 가르는 차이다. AAR이 자기 자신의 로그를 편집할 수 있다면, 실패했거나 미심쩍은 실험을 은폐할 수 있다. Anthropic은 이를 직접 표시한다. 그 속성은 연구 산출물을 신뢰하기 위한 필요조건이며, 검증하기가 간단치 않다.

인프라 용어로의 대응물은, 외부 저장소(예: S3 object-lock, 불변 Datadog 로그)로의 라이트스루(write-through)를 동반한 추가 전용(append-only) 로그다. 에이전트는 과거 기록을 수정할 자격 증명(credential)이 없다. 이는 2026년 기업 보안 팀이 랜섬웨어 저항성 감사 추적(audit trail)을 만드는 데 사용하는 것과 같은 패턴이다.

### 규정된 워크플로 트레이드오프

Anthropic의 보고서는 규정된 워크플로가 종종 AAR의 유연성을 제약하고 성능을 떨어뜨린다고 언급한다. 다시 말해, 인간 설계자가 연구 계획을 단계별로 작성하면, AAR은 고수준 목표와 분해의 자유를 받았을 때보다 성능이 떨어진다. 이는 에이전트형 연구 전반의 결과(LangGraph, Microsoft의 Agent Framework)와 일관된다. 과도하게 명세된 스캐폴드는 유능한 에이전트를 해친다.

안전 함의는 미묘하다. AAR 성능을 향상시키는 그 유연성이, 또한 승인되지 않은 분해, 새로운 도구 사용, 목표 드리프트(goal-drift)를 허용하는 유연성이기도 하다. 이 트레이드오프는 설계 공간의 결정이다. 얼마만큼의 유연성이, 에이전트에게 목표를 분포 밖(off-distribution)으로 일반화할 여지를 주지 않으면서, 인간을 이길 만큼 충분한 능력을 사 오는가?

### 압축 위험

RSP v3.0(Lesson 19)은 AI R&D 능력 임계값을 도입한다. 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인을 완전히 자동화하는 능력. DeepMind의 FSF v3는 유사한 ML R&D 자율성 수준을 포함한다. 두 프레임워크 모두 이 임계값을 강화된 통제의 방아쇠로 취급한다.

AAR은 그 임계값에서 한 걸음 모자라다. 파이프라인의 일부(구체적이고 잘 범위가 정해진 작업에 대한 정렬 연구)는 자동화하지만, 끝에서 끝까지의 능력 개발 루프는 자동화하지 않는다. 타임라인 질문은 그 간극이 얼마나 빨리 닫히느냐다.

압축된 타임라인은 복리적 실패(compounding-failure) 우려다. 정렬 연구와 능력 연구가 비슷한 속도로 복리화된다면, 오정렬 위험 표면은 적어도 능력만큼 빠르게 커진다. 능력이 더 빠르게 복리화된다면(역사적 추세), 간극은 벌어진다. 이것이 AAR이 조건부 선(qualified good)이라는 논거다. 각 추가 정렬 결과는, 오직 연구 과정이 신뢰할 만할 때에만 간극을 줄인다.

### AAR이 대체하지 않는 것

인간 연구자는 작업 큐를 설정하고, 결과를 검토하며, 헌법적 권한(constitutional authority)을 가진다. AAR은 파이프라인의 끝이 아니라 중간을 가속한다. Anthropic의 공개 산출물은 AAR의 기여와, 무엇을 게시하고 무엇을 철회하며 무엇을 다듬을지에 대한 인간 연구자의 판단 모두를 포함한다.

이는 Lesson 15의 제안-후-확정(propose-then-commit) 패턴을 연구 자체에 적용한 것과 일치한다. AAR이 제안하고, 인간이 확정한다.

```figure
aar-forum
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 다중 AAR 연구 포럼을 시뮬레이션한다. 세 개의 독립적인 "연구 에이전트"가 병렬로 실행되며, 선택적 변조(tamper) 시도와 함께 추가 전용 포럼에 발견을 게시한다.

지켜보라:

- 샌드박스 바깥 로깅이 로그 변조 시도를 어떻게 잡아내는지.
- 자유 분해를 동반한 작업 할당이 더 어려운 문제에서 고정 워크플로를 어떻게 능가하는지.
- 로그가 변조 증거성(tamper-evident)을 갖지 않을 때, 단 하나의 훼손된 AAR이 포럼의 신호를 어떻게 오염시킬 수 있는지.

## 산출물 (Ship It)

`outputs/skill-aar-deployment-review.md`는 자동화된 정렬 연구 파이프라인에 대한 배포 전 검토다.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하라. "고정 워크플로(fixed-workflow)" 설정과 "자유 분해(free-decomposition)" 설정을 비교하라. 자유 분해가 항상 이기는가, 아니면 고정 워크플로가 경쟁력 있는 문제 부류가 있는가?

2. 한 에이전트가 로그 변조를 시도하도록 시뮬레이터를 수정하라. 추가 전용 로그가 그것을 탐지하는지 확인하라. 로그에서 탐지 시그니처가 정확히 어떻게 보이는지 설명하는 한 문단을 작성하라.

3. Anthropic의 약-대-강 AAR 보고서를 읽어라. AAR이 인간 연구자를 이긴 구체적 하위 작업을 식별하라. 무엇이 그것을 자동화에 적합하게 만들었는가?

4. AAR 유연성(더 나은 결과)과 규정된 워크플로 제약(더 쉬운 감사) 사이에서 균형을 잡는 작업 큐 할당 정책을 설계하라. 둘을 어떻게 A/B 테스트할지 기술하라.

5. RSP v3.0의 AI R&D-4 임계값을 읽어라. 한 문단으로, AAR이 현재 넘지 못하는, 그것을 넘게 만들 것이 무엇이라고 생각하는지 기술하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| AAR | "자동화된 정렬 연구자" | 정렬 문제에 자율적으로 운영된 Claude Opus 4.6 인스턴스 |
| 약-대-강 학습 (Weak-to-strong training) | "더 약한 감독자로 더 강한 모델을 학습" | AAR이 인간을 능가한 고전적 확장 가능 감독 벤치마크 |
| 공유 포럼 (Shared forum) | "에이전트가 발견을 게시하는 곳" | 추가 전용, 샌드박스 바깥 저장소 |
| 샌드박스 바깥 로그 (Out-of-sandbox log) | "에이전트가 자기 기록을 편집할 수 없음" | 외부 저장소로의 변조 증거성 라이트스루 |
| 규정된 워크플로 (Prescribed workflow) | "인간 설계자가 준 단계별 계획" | AAR을 제약함. 종종 자유 분해 대비 성능을 떨어뜨림 |
| 자유 분해 (Free decomposition) | "에이전트가 작업을 어떻게 쪼갤지 결정" | 더 유능하지만 감사하기 더 어려움 |
| AI R&D 임계값 (AI R&D threshold) | "RSP/FSF 능력 수준" | 경쟁력 있는 비용으로 R&D 파이프라인을 완전 자동화 |
| 압축된 타임라인 (Compressed timeline) | "정렬 대 능력 경주" | 능력이 정렬보다 빠르게 복리화되면, 오정렬 위험이 커진다 |

## 더 읽을거리 (Further Reading)

- [Anthropic: Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) 1차 자료.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0): AI R&D 임계값 규정.
- [Anthropic: Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) 더 넓은 에이전트 자율성 규정.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/): RSP와 병행하는 ML R&D 자율성 수준.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/): AAR이 공략한 근본 문제.
