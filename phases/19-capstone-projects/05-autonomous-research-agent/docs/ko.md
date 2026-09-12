# Capstone 05: 자율 연구 에이전트 (AI-Scientist Class)

> Sakana의 AI-Scientist-v2는 완성된 논문을 발표했다. Agent Laboratory는 실험을 실행했다. Allen AI는 트레이스(trace)를 공유했다. 2026년의 형태는 실험을 다루는 계획-실행-검증(plan-execute-verify) 트리 탐색(tree search), 예산이 정해진 비용, 샌드박스(sandbox)로 격리된 코드 실행, 비전 피드백 LaTeX 작성기, 자동화된 NeurIPS 스타일 리뷰어 앙상블(reviewer ensemble)이다. 캡스톤(capstone)은 그런 것을 하나 만들어서, 논문당 $30 이내에서 처음부터 끝까지 실행하고, Sakana가 문서화한 샌드박스 탈출 레드팀(red team)에서 살아남는 것이다.

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 hours

## 문제 (Problem)

자율 연구 에이전트(autonomous research agent)는 2026년에 문턱을 넘었다. Sakana AI의 AI-Scientist-v2는 워크숍 동료 심사를 통과한 생성된 논문과 함께 Nature에 발표되었다. ShinkaEvolve(ICLR 2026)는 그 계보를 진화하는 가설로 확장했다. AMD의 Agent Laboratory는 재현 가능한 트레이스를 출시했다. 이 에이전트들은 마법이 아니다. 비용 상한, 시드(seed)에 묶인 샌드박스, 자동화된 리뷰를 갖추고 후보 실험의 트리 위에서 실행되는 계획-실행-검증 루프다. 기교는 루프와 예산, 안전성 서사에 있다.

좁은 도메인의 시드 아이디어(예: 1억 파라미터(parameter) 트랜스포머(transformer)에서의 어텐션 희소성(attention-sparsity) 절제(ablation))를 하나 구현하면서 루프를 배운다. 가치는 첫 실행에서 무언가 새로운 것을 발견하는 데 있지 않다. 가치는 인프라에 있다. 트리 탐색, 실험 샌드박스, 작성기-리뷰어 루프, 레드팀 보고서. Sakana 팀은 샌드박스 탈출 실패를 문서화했다. 우리가 만드는 에이전트도 같은 레드팀을 통과해야 한다.

## 개념 (Concept)

에이전트는 최선 우선(best-first) 트리 탐색이다. 노드(node)는 실험 명세다. (가설, 설정, 코드, 예상 결과). 확장(expand) 단계는 작은 편집(옵티마이저(optimizer) 교체, 배치(batch) 크기 이동, 컴포넌트 절제)으로 자식을 제안한다. 각 자식은 하드 리소스 상한이 걸린 새 샌드박스에서 실행된다. 결과는 노드를 (참신성 × 품질 × 남은 예산)으로 순위 매기는 채점 함수로 다시 피드백된다. 트리는 예산이 소진될 때까지 자라고, 그다음 최선의 분기가 작성된다.

작성기는 멀티모달(multimodal)이다. LaTeX 초안을 생성하고, 컴파일하고, 그림을 렌더링하고, 렌더링된 PDF를 Claude Opus 4.7의 비전 모드에 다시 공급하여 레이아웃, 그림 가독성, 주장-증거 정렬을 비평받는다. 다섯 개 LLM 심판의 리뷰어 앙상블은 NeurIPS 스타일 점수(참신성, 엄밀성, 명료성, 재현성, 영향력)를 방출한다. 평균이 임계값 아래로 떨어지면 논문은 비평과 함께 작성기로 돌아간다.

안전성은 부하를 견디는(load-bearing) 부분이다. 모든 실험은 네트워크 이그레스(egress)가 없고, 벽시계 시간이 제한되고, 리소스 한도가 고정된 E2B 또는 Daytona 샌드박스에서 실행된다. 에이전트의 코드 생성 단계는 샌드박스를 탈출하는 시스템 호출(syscall)을 차단하는 정책 계층을 통과한다. 레드팀 보고서는 Sakana가 문서화한 공격 표면(fork bomb, 파일시스템 탈출, LLM이 작성한 네트워크 호출)을 재현한다.

## 아키텍처 (Architecture)

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 스택 (Stack)

- 오케스트레이션: 체크포인팅(checkpointing)과 사람 승인 게이트를 갖춘 LangGraph
- 트리 탐색: 실험 노드에 대한 커스텀 최선 우선(Sakana v2의 AB-MCTS 스타일)
- 샌드박스: 실험마다 E2B, Docker-in-Docker 폴백; cgroups를 통한 리소스 상한
- 문헌: Semantic Scholar Graph API + OpenAlex + 초록의 로컬 FAISS 캐시
- 작성기: LaTeX 템플릿 + 그림 비평 및 레이아웃을 위한 Claude Opus 4.7(비전 모드)
- 리뷰어: 가중 집계를 갖춘 5개 심판 앙상블(Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max)
- 실험 프레임워크: 물리적 실험을 위한 PyTorch 2.5, 로깅을 위한 W&B
- 관측성(observability): 에이전트 트레이스를 위한 Langfuse, 논문당 $30 하드 예산

```figure
ce-experiment-tree
```

## 직접 만들기 (Build It)

1. **시드와 도메인 범위 설정.** 시드 아이디어(예: "1B 미만 트랜스포머의 어텐션 맵에서 희소성 패턴 조사")를 받는다. 탐색 공간을 정의한다. 모델, 데이터셋(dataset), 연산 예산.

2. **문헌 패스.** Semantic Scholar + OpenAlex에 가장 많이 인용된 관련 논문 50편을 질의한다. 초록을 로컬에 캐싱한다. 1페이지 도메인 다이제스트를 생성한다.

3. **트리 스캐폴딩.** 시드 가설로 루트를 초기화한다. 작은 편집 제안(자식당 하나의 설정 변경)으로 `expand(node) -> children`을 구현한다. 가중 참신성 × 품질 × 예산 항으로 `score(node)`를 구현한다.

4. **샌드박스 래핑.** 모든 실험은 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`(또는 동등한 E2B 정책)를 실행한다. 시드는 샌드박스에 쓰인다. 출력은 읽기 전용으로 다시 마운트되어 나온다.

5. **계획-실행-검증 루프.** `plan`은 자식을 제안한다. `execute`는 샌드박스를 실행하고 로그와 지표를 포착한다. `verify`는 지표에 대해 단위 검사를 실행한다(손실(loss)이 감소했는가? 절제가 효과를 분리했는가?). 실패한 노드는 실패 사유가 트리에 저장된다.

6. **작성기.** 예산 소진 후, 최선의 분기를 선택한다. matplotlib로 그림을 렌더링한다. 분기 트레이스를 컨텍스트에 담아 Claude Opus 4.7로 LaTeX 초안을 생성한다. 컴파일한다. 컴파일된 PDF를 Opus 4.7 비전에 다시 공급하여 비평을 받는다. 반복한다.

7. **리뷰어 앙상블.** 다섯 심판이 NeurIPS 스타일 평가 기준으로 초안을 (참신성, 엄밀성, 명료성, 재현성, 영향력) 항목으로 채점한다. 평균 < 4.0/5이면 비평과 함께 작성기로 돌아간다. 재작성 3회 후 하드 중지.

8. **레드팀.** 샌드박스를 겨냥하는 적대적 작업 집합을 만들거나 통합한다. fork bomb, 네트워크 유출(exfiltration) 시도, 파일시스템 탈출, LLM이 작성한 셸 메타문자(metacharacter). 모두 차단되는지 확인한다. 발견 사항을 작성한다.

9. **재현성.** 모든 논문은 트리 탐색 트레이스 JSON, 시드, W&B 실행 링크, 샌드박스 설정, 처음부터 끝까지 재현하는 README와 함께 출시된다.

## 라이브러리로 써보기 (Use It)

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 산출물 (Ship It)

`outputs/skill-ai-scientist.md`이 결과물이다. 시드 아이디어 + 도메인 + $30 예산이 주어지면 전체 파이프라인을 실행하고 심사 가능한 논문과 재현성 번들을 방출한다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 논문 품질 | 발표된 워크숍 논문 대비 블라인드 평가 기준 심사 |
| 20 | 실험적 엄밀성 | 베이스라인(baseline), 시드, 절제; 모든 주장이 결과 표의 셀로 뒷받침됨 |
| 20 | 비용 및 연산 규율 | 논문당 $30 상한 강제, Langfuse로 트레이스됨 |
| 20 | 안전성 | 샌드박스 레드팀 통과; 네트워크 정책과 킬 스위치(kill-switch) 검증됨 |
| 15 | 재현성 | 동일한 시드로 한 명령 재실행 시 논문이 재현됨 |
| **100** | | |

## 연습 문제 (Exercises)

1. 같은 도메인에서 세 가지 다른 시드 아이디어로 파이프라인을 실행한다. 트리 탐색의 어떤 부분이 겹치는지 비교한다. 중복된 낭비 연산을 식별한다.

2. $5 이상으로 추정되는 노드에 대해 실험 실행 전에 사람이 개입하는(human-in-the-loop) 게이트를 추가한다. 총비용이 얼마나 떨어지는지 측정한다.

3. 리뷰어 앙상블을 단일 심판으로 교체한다. 알려진 나쁜 논문의 홀드아웃 셋에서 거짓 수락 비율(false-accept rate)을 측정한다.

4. 네트워크 유출 레드팀 테스트를 도입한다. 에이전트가 외부 주소를 `curl`하려는 코드를 작성한다. `--network=none` 정책이 그것을 차단하는지 확인한다. 시도를 로깅한다.

5. 구현한 트리 탐색을 평탄한 무작위 베이스라인(같은 예산, 확장 전략 없음)과 비교한다. 참신성 × 품질 이득을 보고한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| 트리 탐색(Tree search) | "AB-MCTS 스타일 확장" | 참신성×품질×예산 점수로 실험 노드에 대한 최선 우선 탐색 |
| 샌드박스(Sandbox) | "실험 격리" | 네트워크가 없고, CPU/메모리가 제한되고, 시드가 고정되고, 입력이 읽기 전용인 컨테이너 |
| 비전 비평(Vision critique) | "렌더링 후 읽기" | 논문을 PDF로 컴파일하고, 그 PDF를 VLM에 다시 공급하여 레이아웃과 주장-증거 비평 |
| 리뷰어 앙상블(Reviewer ensemble) | "자동화된 동료 심사" | NeurIPS 평가 기준으로 논문을 채점하는 여러 LLM 심판; 가중 집계가 파이프라인을 게이팅 |
| 참신성 점수(Novelty score) | "이것이 새로운가?" | 50편 문헌 캐시에 대한 근접성을 벌점하는 휴리스틱 |
| 비용 상한(Cost ceiling) | "$ 예산" | 논문당 총지출에 대한 하드 상한; Langfuse 카운터 + 사전 실행 추정 |
| 레드팀(Red team) | "샌드박스 탈출 감사" | 정책이 잘못되면 샌드박스를 탈출할 적대적 작업 |

## 더 읽을거리 (Further Reading)

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2): 레퍼런스 프로덕션 연구 에이전트
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292): 원래 방법론
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai): 진화적 확장
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory): 다중 역할 연구실 프레임워크
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/): 레퍼런스 오케스트레이션 계층
- [Semantic Scholar Graph API](https://api.semanticscholar.org/): 문헌 검색
- [E2B sandboxes](https://e2b.dev): 레퍼런스 실험 격리
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines): 리뷰어 앙상블이 인코딩하는 평가 기준
