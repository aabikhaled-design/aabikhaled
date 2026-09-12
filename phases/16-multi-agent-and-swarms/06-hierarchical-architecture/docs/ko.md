# 계층형 아키텍처와 그 실패 양식 (Hierarchical Architecture and Its Failure Mode)

> 계층형(hierarchical)은 슈퍼바이저(supervisor)가 중첩된 것이다. 워커(worker) 위의 서브 매니저(sub-manager) 위의 매니저(manager) 에이전트(agent). CrewAI `Process.hierarchical`이 교과서적 버전이다: `manager_llm`이 작업을 동적으로 위임하고 출력을 검증한다. LangGraph 등가물은 `create_supervisor(create_supervisor(...))`이다. 작업이 실제 조직도일 때 이는 자연스러운 패턴이다. 또한 관리적 루핑(managerial looping)으로 무너질 가능성이 가장 높은 패턴이기도 하다. 매니저 에이전트가 작업을 잘못 할당하거나, 하위 출력을 오해하거나, 합의에 이르지 못한다. 순차형(sequential)이 종종 이를 이긴다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60분

## 문제 (Problem)

슈퍼바이저 패턴이 이해되고 나면, 자연스러운 다음 단계는 "워커들 자신이 슈퍼바이저라면 어떨까?"이다. 팀에는 하위 팀이 있고; 회사에는 부서의 부서가 있다. 계층형 아키텍처는 그것을 반영한다.

문제: LLM 매니저는 인간 매니저와 같지 않다. 인간 매니저는 자기 부하 직원이 무엇을 아는지 안정적인 사전 지식(prior)으로 알고 있다. LLM 매니저는 매 턴마다 자기 컨텍스트에 있는 것을 바탕으로 조직을 다시 추론한다. 그 컨텍스트가 조금만 표류(drift)해도 트리 전체가 작업을 잘못 배분한다.

## 개념 (Concept)

### 형태 (The shape)

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

모든 내부 노드는 계획하고, 위임하고, 종합한다. 오직 리프(leaf)만 일을 한다.

### 빛나는 곳

- **명확한 조직 매핑.** 실제 작업이 부서적("법무가 문서를 검토하고, 재무가 문서를 검토하고, 엔지니어링이 문서를 검토한 다음, 임원을 위해 요약")이라면, 계층이 명시적이다.
- **국소 요약.** 각 서브 매니저는 최상위 매니저가 보기 전에 자기 팀의 출력을 종합한다. 최상위 매니저는 열다섯 개의 워커 출력이 아니라 세 개의 서브 매니저 요약을 본다.

### 깨지는 곳

2026년 사후 분석(post-mortem)이 계속 발견하는 세 가지 실패 양식:

1. **작업 할당 오류.** 매니저가 목표를 읽고 분해를 환각(hallucinate)해서 잘못된 서브 매니저에게 위임한다. 서브 매니저는 받은 것을 순순히 작업하므로, 오류는 최상위 종합에 이르러서야 드러난다. 인간이 잡을 수 있었던 곳에서 한 단계 떨어진 지점이다.
2. **출력 오해.** 서브 매니저가 "주장 X를 검증할 수 없음"을 반환한다. 최상위 매니저는 이를 "주장 X 미확인"으로 요약한다. 의미가 매 단계마다 표류한다.
3. **합의 루프.** 두 서브 매니저가 불일치한다; 최상위 매니저가 그들에게 조정을 요청한다; 그들이 아래로 다시 위임한다; 워커가 다시 실행한다; 서브 매니저가 약간 다른 답을 반환한다; 루프. CrewAI의 `Process.hierarchical`은 스텝 제한으로 이를 막지만, 그 제한 자체가 이제 하이퍼파라미터(hyperparameter)다.

### 결정적 질문

순차형(선형 파이프라인) 대 계층형: 작업에 실제로 독립적인 하위 팀이 있는가, 아니면 트리인 척하는 하나의 선형 흐름인가? 후자라면 순차형을 쓰고, 전자라면 계층형을 쓰되 명시적 조정 규칙에 예산을 두어라.

### CrewAI의 구현

`Process.hierarchical`은 전문가 크루(crew)들 위에 매니저 LLM을 엮는다. 매니저는:

- 최상위 작업을 받고,
- 하위 작업을 크루에게 할당하고,
- 크루 출력을 평가하고,
- 수락할지, 다시 위임할지, 반복할지 결정한다.

문서: https://docs.crewai.com/en/introduction (Core Concepts 아래 "Hierarchical Process"를 찾아라).

### LangGraph의 구현

LangGraph는 중첩된 `create_supervisor` 호출을 사용한다. 내부 슈퍼바이저는 자신의 그래프를 가진다; 외부 슈퍼바이저는 내부 그래프를 불투명한 노드로 취급한다. 이는 디버깅에서 CrewAI보다 깔끔하지만(각 그래프를 별도로 단계별 실행할 수 있다) 트리의 동적 재구성을 표현하기는 더 어렵다.

레퍼런스: https://reference.langchain.com/python/langgraph-supervisor.

```figure
swarm-hierarchy-token
```

## 직접 만들기 (Build It)

`code/main.py`는 3단계 계층을 실행한다.

- 최상위 매니저: 작업을 "engineering"과 "legal" 분기로 나눈다,
- engineering 서브 매니저: "frontend"와 "backend" 워커로 나눈다,
- legal 서브 매니저: 워커 하나.

데모는 정상 경로(모두가 동의)와, 최상위 매니저의 분해가 "legal"을 "finance"로 잘못 레이블링하는 **교란된 경로(perturbed path)**를 대조하며 오류가 연쇄되는 과정을 지켜본다. 서브 매니저는 순순히 재무 작업을 하고, 최상위 종합자는 재무 발견을 보고하며, 원래의 법무 질문은 답을 얻지 못한다.

실행:

```
python3 code/main.py
```

출력은 "무엇이 요청되었는지" 대 "무엇이 전달되었는지"를 명확히 나란히 보여주며 두 경로를 보여준다.

## 라이브러리로 써보기 (Use It)

`outputs/skill-hierarchy-fitness.md`는 주어진 작업이 계층형, 순차형, 또는 평평한 슈퍼바이저를 써야 하는지 평가한다. 입력: 작업 설명, 조직 구조, 조정 예산. 출력: 막아야 할 구체적 실패 양식과 함께 하는 패턴 권장 사항.

## 산출물 (Ship It)

계층형을 출하한다면:

- **트리 깊이를 2로 상한.** 세 단계만 되어도 이미 대부분의 오류가 관측 가능성(observability) 밖으로 숨는다.
- **명시적 조정 예산.** 최상위 매니저가 확정해야 하기 전 최대 라운드를 설정하라. 보통 2.
- **모든 종합에 출처(provenance).** 각 노드의 요약은 어떤 리프 출력이 그것을 생성했는지 인용해야 한다.
- **분해 표류에 대한 경보.** 매니저의 분해를 스텝마다 로깅하라; 사용자 질의와 비교하라. 분해가 더 이상 질의를 포괄하지 않으면, 경보를 발생시켜라.

## 연습 문제 (Exercises)

1. `code/main.py`를 실행하고 정상 대 교란을 비교하라. 최상위 출력이 사용자의 질문에서 완전히 발산하기까지 매니저 핸드오프가 몇 단계 걸리는가?
2. 세 번째 단계를 추가하라(top → sub → sub-sub → worker). 깊이가 커질수록 교란된 경로가 스스로 교정하는 빈도 대 완전히 발산하는 빈도를 측정하라.
3. 각 서브 매니저에 항상 원래 사용자 질문을 변경 없이 받는 "카나리(canary)" 워커를 구현하라. 카나리 답을 사용해 분해 표류를 탐지하라. 카나리가 종합된 답과 불일치할 때 매니저는 어떻게 반응해야 하는가?
4. CrewAI의 `Process.hierarchical` 문서를 읽어라. CrewAI가 적용하는 구체적 가드레일 하나(스텝 제한, manager_llm 제약)를 식별하고 그것이 어떤 실패 양식을 겨냥하는지 기술하라.
5. 중첩된 LangGraph 슈퍼바이저를 CrewAI 계층형과 비교하라. 어느 것이 조정 루프를 더 저렴하게 탐지하게 하는가?

## 핵심 용어 (Key Terms)

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|------------------------|
| 계층형 (Hierarchical) | "조직도 패턴" | 슈퍼바이저 위의 슈퍼바이저; 오직 리프만 일을 한다. |
| 매니저 LLM (Manager LLM) | "보스" | 내부 노드에서 분해하고, 할당하고, 검증하는 LLM. |
| 분해 표류 (Decomposition drift) | "보스가 핵심을 잃었다" | 최상위 매니저의 분할이 더 이상 원래 질문을 포괄하지 않는다. |
| 조정 루프 (Reconciliation loop) | "끝없는 회의" | 서브 매니저들이 불일치한다; 최상위가 다시 위임한다; 워커가 다시 실행한다; 예산이 소진될 때까지 루프. |
| 깊이-2 상한 (Depth-2 ceiling) | "2단계보다 깊이 가지 마라" | 경험적 가드레일: 3단계 이상은 관측 가능성을 무너뜨린다. |
| 카나리 질문 (Canary question) | "모든 단계의 정답(ground truth)" | 표류를 탐지하기 위해 항상 원래 질의를 변경 없이 받는 워커. |
| 출처 사슬 (Provenance chain) | "누가 무엇을 말했는가" | 각 종합으로부터 그것을 생성한 리프 출력까지의 추적. |

## 더 읽을거리 (Further Reading)

- [CrewAI introduction: Process.hierarchical](https://docs.crewai.com/en/introduction) 매니저 LLM을 가진 교과서적 계층형
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor): `create_supervisor`를 통한 중첩 슈퍼바이저
- [Anthropic engineering: Research system](https://www.anthropic.com/engineering/multi-agent-research-system) Anthropic이 계층형 대신 평평한 슈퍼바이저를 의도적으로 선택한 이유
- [Cemri et al.: Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) MAST 분류 체계; 조율 실패에 관한 절이 분해 표류를 문서화한다
