# AI Scientist v2: 워크숍 수준의 자율 연구

> Sakana의 AI Scientist v2(Yamada et al., arXiv:2504.08066)는 전체 연구 루프를 실행한다. 가설, 코드, 실험, 그림, 작성, 제출. 이것은 생성된 논문이 ICLR 2025 워크숍에서 동료 심사(peer review)를 통과한 최초의 시스템이다. 독립 평가(Beel et al.)는 실험의 42%가 코딩 오류로 실패했고 문헌 검토(literature review)가 정립된 개념을 자주 새로운 것으로 잘못 분류했음을 발견했다. Sakana 자신의 문서는 코드베이스가 LLM이 작성한 코드를 실행한다고 경고하며 Docker 격리를 권장한다. 그 그림의 두 절반 모두가 핵심이다.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60분

## 문제 (The Problem)

연구는 개방형(open-ended) 작업이다. AlphaEvolve의 알고리즘 탐색이나 DGM의 벤치마크에 묶인 자기 수정과 달리, 연구 결과에는 기계 검증 가능한 정확성 기준이 없다. 논문은 단위 테스트가 아니라 심사위원(reviewer)이 판단한다. 그래서 루프를 닫기가 더 어렵다. 그러나 닫는다면 더 가치 있는데, 연구야말로 복리적(compounding) 진보가 자리하는 곳이기 때문이다.

AI Scientist v1(Sakana, 2024)은 인간이 작성한 템플릿에서 출발함으로써 루프를 닫았다. LLM은 고정된 스캐폴딩 안에서 실험을 채워 넣었다. AI Scientist v2(Yamada et al., 2025)는 비전-언어 모델(vision-language model) 비평 루프를 동반한 에이전트형 트리 탐색(agentic tree search)을 사용해 템플릿 요구 사항을 제거한다. 시스템은 아이디어를 생성하고, 실험을 구현하고, 그림을 만들고, 논문을 쓰고, 심사위원 피드백에 따라 반복한다.

동료 심사 평결: v2가 생성한 논문 하나가 ICLR 2025 워크숍에서 (출처를 공개한 채) 채택되었다. 독립 평가 평결: 시스템은 신뢰할 만함과는 거리가 멀다. 둘 다 참이다.

## 개념 (The Concept)

### 아키텍처

1. **아이디어 생성.** LLM은 주제와 선행 문헌에 조건화된 연구 아이디어를 제안한다. v1은 템플릿을 썼고, v2는 가설 공간 위의 에이전트형 탐색을 쓴다.
2. **새로움 점검.** 문헌 검색 단계가 아이디어가 이미 발표되었는지 확인한다. 이것이 Beel et al.의 평가에서 잘못 분류가 발견된 단계로, 정립된 방법이 자주 새로운 것으로 분류되었다.
3. **실험 계획.** 에이전트가 실험 프로토콜을 초안하고 코드를 작성한다.
4. **실행.** 코드가 샌드박스(sandbox)에서 실행된다. 실패는 재시도 루프로 되먹임된다. Beel et al.의 측정에서, 이 단계에서 실험의 42%가 코딩 오류로 실패했다.
5. **그림 생성.** 비전-언어 모델이 생성된 그림을 읽고 명료함을 위해 다시 쓴다. 이것이 v2의 핵심 기술적 추가였다.
6. **작성.** LLM이 논문을 초안하고, 내부 심사위원과 함께 반복한다.
7. **선택 사항: 제출.** 논문이 학회에 제출된다.

### 워크숍 채택 결과가 의미하는 것

v2가 생성한 논문 하나가 ICLR 2025 워크숍에서 동료 심사를 통과했다. 저자들은 그 논문의 출처를 프로그램 위원회에 공개했다. 그 채택은 하나의 데이터 포인트다. 시스템이 "연구를 한다"고 주장할 면허는 아니다.

중요한 맥락: 워크숍 논문은 본 학회 논문보다 낮은 기준이다. 동료 심사는 노이즈가 많다. 제출물의 작은 일부가 어느 날에든 채택된다. 한 번의 성공은 개념 증명(proof of concept)이지 신뢰도 주장이 아니다. Nature 2026 논문은 끝에서 끝까지(end-to-end) 루프를 기록했으며 그 자체로 인간 연구자들이 공동 저술했다. 이것은 "시스템이 Nature 논문을 썼다"는 것이 아니다.

### 독립 평가가 발견한 것

Beel et al.(arXiv:2502.14297)은 외부 평가를 수행했다. 주요 발견:

- **실험 실패.** 실험의 42%가 코딩 오류(잘못된 임포트, 형태(shape) 불일치, 정의되지 않은 변수)로 실패했다. 재시도 루프는 일부는 잡았으나 전부는 아니었다.
- **새로움 잘못 분류.** 문헌 검색 단계가 정립된 개념을 자주 새로운 것으로 표시했다. 이는 연구판 환각에 해당한다.
- **표현 품질 격차.** 비전-언어 그림 비평이 출판급 시각 자료를 만들어내, 근본적인 실험적 약점을 가렸다.

마지막 발견이 이 단계에서 중요하다. 설득력 있는 연구를 하지 않으면서 설득력 있는 산출물을 만드는 시스템은, 명백히 실패하는 시스템보다 더 안전한 것이 아니라 더 위험하다. 평가는 그림에서 멈추지 않고 근본적인 주장까지 도달해야 한다.

### 샌드박스 탈출 우려

Sakana 자신의 저장소 README는 경고한다:

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

이것이 검증되지 않은 영역에서 자율성이 작동하는 운영적 형태다. LLM이 코드를 작성하고, 코드가 실행되며, 코드는 프로세스가 허용된 무엇이든 할 수 있다. 파일 시스템, 네트워크, 프로세스 동작을 강하게 제한하는 샌드박스가 없으면, 어떤 자기 주도적 연구 에이전트든 데이터를 유출하거나, 컴퓨트를 태우거나, 자기 자신을 다시 쓸 수 있다.

AlphaEvolve의 샌드박스 서사는 그 평가기가 빡빡하기 때문에 더 쉽다. AI Scientist v2의 루프는 개방형 목표로 개방형 코드를 실행한다. 그래서 더 강한 격리(최소 Docker, seccomp / gVisor 선호)와, 시스템을 떠나기 전 모든 제출물에 대한 수동 검토가 필요하다.

### v2가 프런티어 스택에서 자리하는 곳

| 시스템 | 대상 | 출력 종류 | 평가기 | 알려진 실패 |
|---|---|---|---|---|
| AlphaEvolve | 알고리즘 | 코드 | 단위 + 벤치마크 | 평가기 엄격함에 묶임 |
| DGM | 에이전트 스캐폴딩 | 코드 | SWE-bench | 보상 해킹 |
| AI Scientist v2 | 연구 논문 | 텍스트 + 코드 + 그림 | 동료 심사(약함) | 실험 실패, 잘못 분류, 광택이 약점을 가림 |

v2는 셋 중 가장 약한 자동 평가기, 가장 넓은 출력 표면, 그리고 공개 산출물까지의 가장 짧은 경로를 가진다. 운영적 통제 수단(샌드박스, 검토, 공개)이 안전 작업의 대부분을 수행하고 있다.

```figure
mx-research-loop
```

## 라이브러리로 써보기 (Use It)

`code/main.py`는 v2 루프를 상태 기계(state machine)로 시뮬레이션한다. 아이디어 → 새로움 점검 → 실험 → 그림 → 작성 → 심사 → 채택-또는-반복. 각 상태는 Beel et al.의 발견에서 가져온, 설정 가능한 실패 확률을 가진다. 시뮬레이터를 N번의 루프 동안 실행하고 세어 보라:

- 몇 개의 아이디어가 제출에 도달하는지.
- 몇 개의 제출물이, 광택 낸 논문이 숨기는 치명적 실험 결함을 가지는지.
- 재시도 예산이 품질 대 수율(yield)을 어떻게 절충하는지.

## 산출물 (Ship It)

`outputs/skill-ai-scientist-sandbox-review.md`는 연구 루프 에이전트가 만든 무엇이든 샌드박스를 떠나기 전에 거치는 두 관문(two-gate) 검토 체크리스트다.

## 연습 문제 (Exercises)

1. 기본 파라미터로 `code/main.py`를 실행하라. 루프 실행의 몇 비율이 "깨끗한" 논문을 만드는가? 몇 비율이 그림 비평이 광택을 입힌 실험 실패 결함을 가진 논문을 만드는가?

2. 기본값은 이미 Beel et al.의 42% / 25%를 사용한다. `--experiment-failure 0.20 --novelty-mislabel 0.10`으로, 그다음 `--experiment-failure 0.60 --novelty-mislabel 0.40`으로 다시 실행하라. 광택은 났지만 결함 있는 비율이 두 실행 사이에서 어떻게 이동하는가?

3. 샌드박스 요구 사항에 관한 Sakana의 AI Scientist v2 저장소 README를 읽어라. 며칠짜리 자율 실행에 대해 당신이 적용할 (Docker를 넘어선) 추가 제한 두 가지를 들어라.

4. 표현 품질 격차에 관한 Beel et al. Section 4를 읽어라. 광택은 났지만 실험적으로 결함 있는 논문을 잡아낼 추가 평가기 하나를 설계하라.

5. "박사 한 명이 모든 논문을 읽는다"보다 더 잘 확장되는, 연구 에이전트 출력을 위한 인간 검토 프로토콜을 제안하라. 병목을 식별하고 그것을 중심으로 설계하라.

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|---|---|---|
| AI Scientist v1 | "Sakana의 템플릿화된 연구 에이전트" | 고정된 스캐폴드에 실험을 채워 넣었음 |
| AI Scientist v2 | "템플릿 없는 연구 에이전트" | VLM 그림 비평을 동반한 에이전트형 트리 탐색 |
| 에이전트형 트리 탐색 (Agentic tree search) | "분기하는 연구 에이전트" | 여러 실험 계획을 병렬로 확장하고, 내부 비평가로 가지치기 |
| 비전-언어 비평 (Vision-language critique) | "그림에 대한 VLM 광택" | 멀티모달 모델이 그림을 읽고 명료함을 위해 다시 씀 |
| 문헌 검색 (Literature retrieval) | "새로움 점검" | 아이디어의 새로움을 확인하기 위해 선행 연구를 검색: 잘못 분류한다고 기록됨 |
| 광택 가림 (Polish masking) | "예쁜 논문, 망가진 연구" | 표현 품질이 실험 품질을 능가함. 약점을 숨김 |
| 샌드박스 탈출 (Sandbox escape) | "LLM 코드가 빠져나감" | 에이전트가 실행한 코드가 루프 설계자가 의도하지 않은 일을 함 |

## 더 읽을거리 (Further Reading)

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066): 논문.
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/): 동료 심사 맥락을 담은 벤더 요약.
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297): 외부 평가 수치.
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292): 템플릿화된 선행 시스템.
- [Anthropic: Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) 개방형 연구 에이전트에 대한 더 넓은 규정.
