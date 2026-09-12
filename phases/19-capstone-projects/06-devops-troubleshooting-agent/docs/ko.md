# Capstone 06: 쿠버네티스를 위한 DevOps 트러블슈팅 에이전트

> AWS의 DevOps Agent가 정식 출시(GA)되었고, Resolve AI는 K8s 플레이북을 발표했고 NeuBird는 시맨틱 모니터링을 시연했고 Metoro는 AI SRE를 서비스별 SLO에 묶었다. 프로덕션(production) 형태는 정해졌다. 알림 웹훅(webhook)이 발동하면, 에이전트가 텔레메트리(telemetry)를 읽고 K8s 객체의 그래프를 걷고 근본 원인 가설의 순위를 매기고 승인 버튼이 달린 Slack 브리핑을 게시한다. 기본은 읽기 전용. 모든 조치(remediation)는 사람에 의해 게이팅된다. 이 캡스톤(capstone)은 그런 에이전트로, 20개의 합성 인시던트(incident)에서 평가되고 세 개의 공유된 케이스에서 AWS의 Agent와 비교된다.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 hours

## 문제 (Problem)

2025-2026년의 SRE 서사는 이렇게 되었다. "AI 에이전트가 인시던트를 분류(triage)하고, 사람이 조치를 승인한다." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps는 모두 이 형태를 프로덕션에서 출시한다. 에이전트는 Prometheus 지표, Loki 로그, Tempo 트레이스(trace), kube-state-metrics, 그리고 K8s 객체의 지식 그래프(knowledge graph)를 읽는다. 텔레메트리 인용(citation)이 달린 순위가 매겨진 근본 원인 가설을 5분 이내에 만들어 낸다. Slack을 통한 명시적인 사람 승인 없이는 절대 파괴적 명령을 실행하지 않는다.

어려운 작업의 대부분은 추론이 아니라 범위 설정과 안전성이다. 에이전트는 기본적으로 읽기 전용인 RBAC 표면, 강화된 MCP 도구 서버, 그리고 고려된 명령 대 실행된 명령의 감사 로그(audit log)가 필요하다. 자신의 능력 밖일 때를 알고 에스컬레이션(escalate)할 줄 알아야 한다. 그리고 OOM-kill 연쇄가 $5천짜리 에이전트 청구서를 생성하지 않을 만큼 충분히 저렴하게 실행되어야 한다.

## 개념 (Concept)

에이전트는 지식 그래프 위에서 동작한다. 노드(node)는 K8s 객체(Pod, Deployment, Service, Node, HPA, PVC)에 텔레메트리 소스(Prometheus 시계열, Loki 스트림, Tempo 트레이스)를 더한 것이다. 간선(edge)은 소유권(Pod -> ReplicaSet -> Deployment), 스케줄링(Pod -> Node), 관찰(Pod -> Prometheus 시계열)을 인코딩한다. 그래프는 kube-state-metrics 동기화로 신선하게 유지되고 매 알림마다 다시 샘플링된다.

알림이 발동하면, 에이전트는 영향받은 객체로부터 근본 원인을 분석한다. 간선을 걷고 관련 텔레메트리 조각(최근 15분)을 끌어오고 가설의 초안을 작성한다. 가설은 증거로 순위가 매겨진다. 얼마나 많은 텔레메트리 인용이 그것을 뒷받침하는지, 얼마나 최근인지, 얼마나 구체적인지. 상위 3개 가설은 그래프 경로 시각화와 조치 액션을 위한 승인 버튼과 함께 Slack으로 간다.

조치는 게이팅된다. 허용되는 기본 액션은 읽기 전용이다. 파괴적 액션(스케일 다운, 롤백, Pod 삭제)은 Slack 승인을 요구한다. ArgoCD 롤백 훅은 에이전트가 결코 보유하지 않는 인증 토큰을 요구한다. 감사 로그는 에이전트가 *고려한* 모든 명령을(실행한 것만이 아니라) 기록하므로, 검토 과정이 아차사고(near-miss)를 잡아낸다.

## 아키텍처 (Architecture)

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 스택 (Stack)

- 관측성(observability) 소스: Prometheus, Loki, Tempo, kube-state-metrics
- 지식 그래프: K8s 객체 + 텔레메트리 간선의 Neo4j(매니지드) 또는 kuzu(임베디드)
- 에이전트: 도구별 허용 목록(allow-list)을 갖춘, 기본 읽기 전용 LangGraph
- 도구 전송: StreamableHTTP 위의 FastMCP; 승인 게이트 뒤의 파괴적 도구를 위한 별도 서버
- 모델: 근본 원인 추론을 위한 Claude Sonnet 4.7, 로그 요약을 위한 Gemini 2.5 Flash
- 조치: ArgoCD 롤백 웹훅, PagerDuty 에스컬레이트, Slack 승인 카드
- 감사: 추가 전용(append-only) 구조화된 로그(고려됨, 실행됨, 승인됨, 결과)
- 배포: 자체의 좁은 RBAC 역할을 가진 K8s 배포; 별도 네임스페이스

```figure
ce-rootcause-walk
```

## 직접 만들기 (Build It)

1. **그래프 수집.** kube-state-metrics를 30초마다 Neo4j/kuzu로 동기화한다. 노드: Pod, Deployment, Node, Service, PVC, HPA. 간선: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. 텔레메트리 오버레이 간선: OBSERVED_BY(Pod이 Prometheus 시계열에 의해 관찰됨).

2. **알림 수신기.** PagerDuty 또는 Alertmanager 웹훅을 받는 FastAPI 엔드포인트. 영향받은 객체와 SLO 위반을 추출한다.

3. **읽기 전용 도구 표면.** kubectl, Prometheus 쿼리, Loki logql, Tempo traceql을 FastMCP로 감싼다. 모든 도구는 좁은 RBAC 동사("get", "list", "describe")를 가진다. 기본 서버에는 "delete", "exec", "scale"이 없다.

4. **근본 원인 에이전트.** 세 개의 노드를 가진 LangGraph. `sample`은 최근 15분 텔레메트리 조각을 끌어온다. `walk`는 이웃 객체를 위해 그래프를 질의한다. `hypothesize`는 텔레메트리 인용과 함께 순위가 매겨진 근본 원인 후보의 초안을 작성한다.

5. **증거 채점.** 각 가설은 점수 = 최근성 * 구체성 * 그래프 경로 길이 역수 * 인용 수를 가진다. 상위 3개를 반환한다.

6. **Slack 브리핑.** 가설, 그래프 경로 시각화(서버 측에서 렌더링된 서브그래프 이미지), 그리고 최대 하나의 조치 액션을 위한 승인 버튼이 담긴 첨부를 게시한다.

7. **조치 게이트.** 파괴적 도구(스케일 다운, 롤백, 삭제)는 승인 토큰 뒤의 두 번째 MCP 서버에 위치한다. 에이전트는 Slack 카드가 사람에 의해 승인된 후에만 그것들을 호출할 수 있다.

8. **감사 로그.** 추가 전용 JSONL: 모든 후보 명령에 대해, 그것이 고려되었는지, 실행되었는지, 누가 승인했는지 로깅한다. 매일 S3로 전송한다.

9. **합성 인시던트 스위트.** 20개 시나리오를 만든다. OOMKill 연쇄, DNS 플랩, HPA 스래시(thrash), PVC 가득 참, 시끄러운 이웃(noisy neighbor), 결함 있는 사이드카(sidecar), 잘못된 ConfigMap 롤아웃, 인증서 교체, 이미지 풀(image-pull) 백오프 등. 근본 원인 정확도와 가설까지의 시간으로 에이전트를 채점한다.

## 라이브러리로 써보기 (Use It)

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 산출물 (Ship It)

`outputs/skill-devops-agent.md`이 결과물이다. K8s 클러스터와 알림 소스가 주어지면, 에이전트는 순위가 매겨진 근본 원인 가설과 Slack으로 게이팅된 조치 흐름을 만들어 낸다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 시나리오 스위트에서의 RCA 정확도 | 20개 합성 인시던트에 걸쳐 ≥80% 올바른 근본 원인 |
| 20 | 안전성 | 감사 로그에서 파괴적 액션 가드가 Slack 승인 없이 절대 발동하지 않음 |
| 20 | 가설까지의 시간 | 알림으로부터 Slack 브리핑까지 p50 5분 미만 |
| 20 | 설명 가능성 | 모든 가설이 그래프 경로와 텔레메트리 인용을 가짐 |
| 15 | 통합 완전성 | PagerDuty, Slack, ArgoCD, Prometheus 종단 간 작동 |
| **100** | | |

## 연습 문제 (Exercises)

1. AWS의 DevOps Agent가 시연된 것과 같은 세 인시던트에 당신의 에이전트를 실행한다. 나란히 비교를 발행한다. 에이전트가 어디서 갈라지는지 보고한다.

2. 에이전트가 *고려한* 명령 중 승인 없이 파괴적이었을 명령을 표시하는 "아차사고" 감사를 추가한다. 일주일에 걸친 아차사고 비율을 측정한다.

3. 가설 모델을 Claude Sonnet 4.7에서 자체 호스팅된 Llama 3.3 70B로 교체한다. RCA 정확도 차이와 인시던트당 달러를 측정한다.

4. 인과 필터를 만든다. 상관된 텔레메트리 급증을 진짜 근본 원인과 구별한다. 20개 시나리오 레이블로 작은 분류기(classifier)를 학습시킨다.

5. 롤백 드라이런(dry-run)을 추가한다. 같은 매니페스트로 스테이징 클러스터에 대해 ArgoCD 롤백. Slack 승인 버튼 전에 실제 클러스터에서 롤백 계획을 검증한다.

## 핵심 용어 (Key Terms)

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|------------------------|
| K8s 지식 그래프(K8s knowledge graph) | "클러스터 그래프" | 노드 = K8s 객체 + 텔레메트리 시계열; 간선 = 소유권, 스케줄링, 관찰 |
| 기본 읽기 전용(Read-only-by-default) | "범위가 지정된 RBAC" | 에이전트의 서비스 계정은 get/list/describe 동사만 가짐; 파괴적 동사는 승인 뒤 별도 서버에 위치 |
| 감사 로그(Audit log) | "고려됨 대 실행됨" | 모든 후보 명령, 실행 여부, 승인자의 추가 전용 기록 |
| 가설 순위(Hypothesis ranking) | "증거 점수" | 최근성 × 구체성 × 그래프 경로 길이 역수 × 인용 수 |
| Slack 승인 카드(Slack approval card) | "HITL 게이트" | 조치 버튼이 달린 대화형 Slack 메시지; 사람이 클릭할 때까지 에이전트가 진행할 수 없음 |
| 텔레메트리 인용(Telemetry citation) | "증거 포인터" | 주장을 뒷받침하는 Prometheus 쿼리, Loki 셀렉터, 또는 Tempo 트레이스 URL |
| MTTR | "해결까지의 시간" | 알림 발동으로부터 SLO 회복까지의 실제 경과 시간 |

## 더 읽을거리 (Further Reading)

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/): 정전(canonical)에 해당하는 2026 레퍼런스
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai): 경쟁사 레퍼런스
- [NeuBird semantic monitoring](https://www.neubird.ai): 시맨틱 그래프 접근법
- [Metoro AI SRE](https://metoro.io): SLO 우선 프로덕션 프레이밍
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics): 클러스터 상태 소스
- [LangGraph](https://langchain-ai.github.io/langgraph/): 레퍼런스 에이전트 오케스트레이터
- [FastMCP](https://github.com/jlowin/fastmcp): Python MCP 서버 프레임워크
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/): 게이팅된 조치 대상
